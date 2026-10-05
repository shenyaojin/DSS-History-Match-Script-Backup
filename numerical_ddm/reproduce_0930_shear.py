"""Reproduce the saved 0930 NonTouching notebook's F2 unit-response experiment.

This is explicitly the rectangular-shear forward component, not a re-fit of
its unavailable observation cache or a replacement of its 801-strip F1.
Original notebook code is extracted as AST and run against intact DDMpy_0930.
Run using base Python (JIN_pylib notebook dependencies are installed there).
"""
import argparse,ast,copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'fibeRIS/src')]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.tensile_fault.data_transfer.from_pc import DDMpy_0930 as ddm
from fiberis.moose.numerical_ddm import NumericalConfig,from_ddmpy,axial_response
from run_validation import metrics

NOTEBOOK=ROOT/'scripts/tensile_fault/data_transfer/from_pc/0930_transfer/OneDrive_4_9-30-2026/01_NonTouching_Baseline.ipynb'
FT=.3048
# Exact gap and crossing MD printed in the supplied, executed notebook.
GAP_FT=34.668574514637385
CROSSING_MD_FT=10317.837464
# This orientation is also in the original transferred modeling script.
STRIKE,DIP=.6,90.
OUT=ROOT/'output/numerical_ddm/reproduce_0930_non_touching'


def prepare():
    OUT.mkdir(exist_ok=True)
    survey_file=ROOT/'data_fervo/legacy/Gold_4_PB_Well_Geometry.csv'
    survey=pd.read_csv(survey_file)
    original=ddm.Well.set_well_by_points(survey[['x_gold','y_gold','z_gold']].to_numpy(float),N=len(survey)*10,smooth=31)
    original.gauge_length=10/FT
    md=np.arange(10120.,10580.+.125/2,.125)
    points=np.column_stack([np.interp(md,original.mds,getattr(original,k)) for k in ['x','y','z']])
    monitor=ddm.Well.Well().set_well_path(*points.T)
    monitor.mds=md
    monitor.gauge_length=10/FT
    hit=np.array([np.interp(CROSSING_MD_FT,original.mds,getattr(original,k)) for k in ['x','y','z']])
    rectangles={'non_touching':[(GAP_FT,300+GAP_FT,-100.,100.)]}
    nb=json.loads(NOTEBOOK.read_text())
    source='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rectangle_response')
    function=ast.Module(body=[node],type_ignores=[])
    ns=dict(np=np,ddm=ddm,rectangles=rectangles,
            original_shear=dict(strike_deg=STRIKE,dip_deg=DIP),hit={'global_ft':hit})
    exec(compile(function,str(NOTEBOOK),'exec'),ns)
    # Original function computes positive one-foot slip. Preserve it, then
    # scale its genuinely computed result to the notebook's +1 mm display.
    md_original,eps_per_ft=ns['rectangle_response']('non_touching',monitor)
    eps_original=eps_per_ft/(FT*1000)
    # Build exactly the same rectangle for explicit SI numerical conversion.
    e=ddm.Element.Element(300.,200.,0.,1/(FT*1000),0.)
    e.set_global_coors(GAP_FT+150,0,0)
    f=ddm.Fracture.fractureZ_to_global(ddm.Fracture.FractureZ().set_fracture_elements(e),STRIKE,DIP,*hit)
    f.set_monitor_wells(monitor)
    f.calculate()
    local=f.monitor_wells[0]
    R=ddm.Fracture.coor_global_to_local(*np.eye(3),STRIKE,DIP)
    u_local=np.column_stack([local.u1,local.u2,local.u3])
    u_global=u_local@R*FT
    np.testing.assert_allclose(local.strain,eps_original,rtol=2e-12,atol=1e-16)
    # Recenter the numerical frame at the rectangle centre. Algebraically
    # identical geometry, allowing graded mesh sizes relative to the source.
    center=hit+np.linalg.solve(R,np.array([GAP_FT+150,0,0]))
    f.fracx,f.fracy,f.fracz=center
    e.set_global_coors(0,0,0)
    f.set_monitor_wells(monitor)
    cfg=NumericalConfig(str(ROOT/'moose_env/moose/modules/combined/combined-opt'),str(OUT/'moose'),
        mesh_size=5,half_extent=800,core_extent=120,core_extents=(75,40,90),band_thickness=1,
        processors=8,mpiexec=str(Path.home()/'miniforge/envs/moose/bin/mpiexec'))
    numerical=from_ddmpy(f,cfg,length_scale=FT)
    np.savez(OUT/'original_0930.npz',md_ft=md_original,points_m=points*FT,
             u_m=u_global,strain=eps_original,strain_per_ft_slip=eps_per_ft)
    metadata=dict(notebook=str(NOTEBOOK),notebook_sha256=hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        survey_sha256=hashlib.sha256(survey_file.read_bytes()).hexdigest(),
        original_function='rectangle_response (verbatim AST from notebook)',
        ddmpy_path=ddm.__file__,source_slip_m=.001,gap_ft=GAP_FT,
        original_crossing_md_ft=CROSSING_MD_FT,strike_deg=STRIKE,dip_deg=DIP,
        global_origin_ft=hit.tolist(),rectangle_center_ft=center.tolist(),
        sampling_step_ft=.125,gauge_length_m=10,display_md_ft=[10200,10500],
        limitations=['F2 forward-component reproduction; full F1+F2 fitted notebook awaits missing local helper modules/observation cache.',
        'Crossing MD recovered from saved notebook output to 6 decimal ft; gap has full printed precision.',
        'Strike/dip taken from earlier transferred original model; confirm against missing uniform_fracture_fit.py before claiming full notebook equivalence.'])
    (OUT/'provenance.json').write_text(json.dumps(metadata,indent=2))
    return numerical,md,points*FT,u_global,eps_original


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference-only',action='store_true')
    parser.add_argument('--analyze-only',action='store_true')
    parser.add_argument('--h',type=float,default=5)
    args=parser.parse_args()
    f,md,p,ref_u,ref_eps=prepare()
    if args.reference_only:
        print('Original notebook F2 function executed with unmodified 0930 package:',OUT)
        return
    f.config.mesh_size=args.h
    f.config.output_directory=str(OUT/f'moose_h{args.h:g}')
    if args.analyze_only:
        u=np.load(Path(f.config.output_directory)/'displacement.npz')['displacement']
        w=f.monitor_wells[0];w.u1,w.u2,w.u3=u.T;w.calculate_strain()
    else:
        f.calculate()
        w=f.monitor_wells[0];u=np.column_stack([w.u1,w.u2,w.u3])
    shown=(md>=10200)&(md<=10500)
    result=dict(displacement=metrics(u[shown],ref_u[shown]),strain=metrics(w.strain[shown],ref_eps[shown]),
                all_padded_displacement=metrics(u,ref_u),all_padded_strain=metrics(w.strain,ref_eps),
                reference='Unmodified DDMpy_0930; original sample-mean 10 m gauge; shared original MD',
                status='partial notebook: F2 forward response; no observation refit or F1 claim')
    target=Path(f.config.output_directory)
    (target/'metrics.json').write_text(json.dumps(result,indent=2))
    np.savez(target/'comparison.npz',points=p,md_ft=md,u=u,reference=ref_u,
             strain=w.strain,reference_strain=ref_eps)
    fig,axes=plt.subplots(1,2,figsize=(12,7),sharey=True)
    for k in range(3):
        axes[0].plot(ref_u[shown,k]*1e6,md[shown],label=f'0930 DDMPY u{k+1}')
        axes[0].plot(u[shown,k]*1e6,md[shown],'--')
    axes[0].set(xlabel='Displacement (micrometre)',ylabel='MD (ft)');axes[0].legend()
    axes[1].plot(ref_eps[shown]*1e6,md[shown],label='Original 0930 DDMPY')
    axes[1].plot(w.strain[shown]*1e6,md[shown],'--',label='MOOSE FE')
    axes[1].set(xlabel='Axial microstrain (+1 mm S1, 10 m gauge)');axes[1].legend()
    axes[0].set_ylim(10500,10200)
    fig.suptitle('0930 NonTouching notebook: F2 rectangular shear forward component')
    fig.tight_layout();fig.savefig(target/'comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
