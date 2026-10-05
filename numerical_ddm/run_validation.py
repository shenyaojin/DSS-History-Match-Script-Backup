"""Run real MOOSE snapshots and compare to the untouched DDMPY source."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'fibeRIS/src'),str(ROOT),str(ROOT/'scripts/tensile_fault/data_transfer/from_pc')]
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fiberis.moose.numerical_ddm import Element,Well,FractureZ,NumericalConfig,write_input,axial_response
import importlib.util
spec=importlib.util.spec_from_file_location('original_ddmpy_element',ROOT/'scripts/tensile_fault/data_transfer/from_pc/DDMpy_log/Element.py')
ref_module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref_module)
ReferenceElement=ref_module.Element


def metrics(value,ref):
    delta=np.asarray(value)-ref
    return dict(relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(ref)) if np.linalg.norm(ref)>0 else None,
                peak_normalized_max=float(np.max(abs(delta))/np.max(abs(ref))) if np.max(abs(ref))>0 else None,
                max_absolute=float(np.max(abs(delta))))


def reference(elements,points):
    u=np.zeros_like(points)
    for e in elements:
        r=ReferenceElement(e.length,e.height,-e.width,-e.S1,-e.S2)
        r.mu=e.mu
        r.set_coors(*(points-e.center).T.copy())
        u+=np.column_stack([r.u1,r.u2,r.u3])
    return u


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--suite',action='store_true')
    ap.add_argument('--name')
    ap.add_argument('--h',type=float,default=20)
    ap.add_argument('--extent',type=float,default=800)
    ap.add_argument('--thickness',type=float,default=2)
    ap.add_argument('--band-cells',type=int,default=2)
    ap.add_argument('--order',default='SECOND')
    ap.add_argument('--quadrature',default='FIFTH')
    ap.add_argument('--mode',choices=['opening','s1','s2','mixed','zero','double'],default='opening')
    ap.add_argument('--generate-only',action='store_true')
    ap.add_argument('--analyze-only',action='store_true')
    ap.add_argument('--np',type=int,default=1)
    args=ap.parse_args()
    if args.suite:
        from run_suite import execute
        execute(args.np)
        return
    jumps={'opening':(.001,0,0),'s1':(0,.001,0),'s2':(0,0,.001),
           'mixed':(.001,.0004,-.0006),'zero':(0,0,0),'double':(.002,0,0)}
    e=Element(100,60,*jumps[args.mode])
    # Fixed a priori. All 101 receivers, including endpoints, enter the metrics.
    well=Well().set_well_path(np.linspace(-100,100,101),17.,40.)
    name=args.name or f'{args.mode}_h{args.h:g}_L{args.extent:g}_t{args.thickness:g}'.replace('.','p')
    if args.name is None:
        if args.band_cells!=2:name+=f'_b{args.band_cells}'
        if args.quadrature!='FIFTH':name+='_'+args.quadrature.lower()
        if args.order!='SECOND':name+='_'+args.order.lower()
    directory=ROOT/'output/numerical_ddm'/name
    c=NumericalConfig(str(ROOT/'moose_env/moose/modules/combined/combined-opt'),str(directory),
                     mesh_size=args.h,half_extent=args.extent,band_thickness=args.thickness,
                     band_cells=args.band_cells,order=args.order,quadrature=args.quadrature,
                     processors=args.np,mpiexec=str(Path.home()/'miniforge/envs/moose/bin/mpiexec'))
    frac=FractureZ().set_fracture_elements(e).set_monitor_wells(well).configure(c)
    if args.generate_only:
        print(write_input([e],well.points,c,directory))
        return
    if args.analyze_only:
        if not json.loads((directory/'run.json').read_text()).get('success'):
            raise RuntimeError('Cannot analyze an incomplete or failed solve')
        model=json.loads((directory/'model.json').read_text())
        stored=model['elements'][0]
        e=Element(stored['length'],stored['height'],-stored['width'],-stored['S1'],-stored['S2'])
        e.mu=stored['mu']
        e.set_global_coors(*[stored[k] for k in ['abs_x1','abs_x2','abs_x3']])
        data=np.load(directory/'displacement.npz')
        well=Well().set_well_path(*data['points'].T)
        u=data['displacement']
    else:
        frac.calculate()
        w=frac.monitor_wells[0]
        u=np.column_stack([w.u1,w.u2,w.u3])
    ref=reference([e],well.points)
    s,_,strain=axial_response(well.points,u)
    _,_,refstrain=axial_response(well.points,ref)
    result=dict(displacement=metrics(u,ref),strain=metrics(strain,refstrain),
                components={v:metrics(u[:,i],ref[:,i]) for i,v in enumerate(['ux','uy','uz'])},
                reference='untouched DDMpy_log.Element; corrected shared axial_response',
                receiver_count=len(s))
    (directory/'metrics.json').write_text(json.dumps(result,indent=2))
    np.savez(directory/'comparison.npz',points=well.points,u=u,reference=ref,strain=strain,reference_strain=refstrain,md=s)
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for i,ax in enumerate(axs.flat[:3]):
        ax.plot(s,ref[:,i]*1e6,label='DDMPY original field')
        ax.plot(s,u[:,i]*1e6,'--',label='MOOSE FE')
        ax.set(ylabel=f'u{i+1} (micrometre)',xlabel='MD (m)')
        ax.legend()
    axs.flat[3].plot(s,refstrain*1e6,label='DDMPY + common operator')
    axs.flat[3].plot(s,strain*1e6,'--',label='MOOSE + common operator')
    axs.flat[3].set(ylabel='Axial microstrain',xlabel='MD (m)')
    axs.flat[3].legend()
    fig.tight_layout()
    fig.savefig(directory/'comparison.png',dpi=170)
    fig.savefig(directory/'comparison.pdf')
    plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
