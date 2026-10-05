"""Complete forward reproduction of a self-contained original DDMPY notebook.

Runs the original numerical code unchanged against DDMpy_0930, then runs MOOSE
at each of the same 30 physical load times. No DDMPY field enters the FE solve.
Base Python is used for the original JIN_pylib dependency; MOOSE is external.
"""
import ast,copy,hashlib,json,platform,subprocess,sys,argparse,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'fibeRIS/src')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.tensile_fault.data_transfer.from_pc import DDMpy_0930 as ddm
from fiberis.moose.numerical_ddm import NumericalConfig,from_ddmpy,axial_response,time_derivative,write_input
from run_validation import metrics

SOURCE=Path(ddm.__file__).parent/'Examples/Horizontal well existing tensile and shear.ipynb'
OUT=ROOT/'output/numerical_ddm/notebook_existing_tensile_shear'


def original():
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=OUT/'frozen_reference'
    frozen.mkdir(exist_ok=True)
    sources=[SOURCE,*Path(ddm.__file__).parent.glob('*.py'),ROOT/'JIN_pylib/Data2D_XT.py']
    manifest={}
    for path in sources:
        name=path.name if path.parent.name!='JIN_pylib' else 'JIN_'+path.name
        target=frozen/name
        if target.exists() and target.read_bytes()!=path.read_bytes():
            raise RuntimeError('Original reference changed after freezing: '+str(path))
        if not target.exists():shutil.copy2(path,target)
        manifest[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'reference_sha256.json').write_text(json.dumps(manifest,indent=2))
    notebook=json.loads(SOURCE.read_text())
    cell=next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code'
              and 'frac.define_LHW' in ''.join(c['source']))
    (OUT/'original_model_cell.py').write_text(cell)
    ns=dict(np=np,Well=ddm.Well,DynamicFracture=ddm.DynamicFracture)
    strain=None
    for node in ast.parse(cell).body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
            call=node.value
            if isinstance(call.func,ast.Attribute) and call.func.attr=='take_time_diff':
                strain=ns['data'].data.copy()
            if isinstance(call.func,ast.Attribute) and call.func.attr=='plot_waterfall':
                break
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
    frac=ns['frac'];times=ns['taxis']
    assert strain is not None and strain.shape==(300,30)
    rate_native=ns['data'].data.copy()
    points=np.column_stack([ns['well'].x,ns['well'].y,ns['well'].z])
    R=ddm.Fracture.coor_global_to_local(*np.eye(3),frac.strike,frac.dip)
    u=[]
    for frame in frac.fractures:
        w=frame.monitor_wells[0]
        u.append(np.column_stack([w.u1,w.u2,w.u3])@R)
    u=np.array(u)
    # The notebook labels its raw 0..60 axis as minutes. Convert consistently
    # to seconds for MOOSE and strain-rate SI, preserving native arrays too.
    seconds=times*60.
    md=ns['well'].mds.copy()
    np.savez(OUT/'original_0930.npz',points=points,md=md,times_native_minutes=times,
             times_seconds=seconds,u=u,strain=strain,rate_per_minute=rate_native,
             rate_per_second=rate_native/60.)
    provenance=dict(notebook=str(SOURCE),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        package=str(Path(ddm.__file__).parent),
        commit=subprocess.check_output(['git','-C',str(Path(ddm.__file__).parent),'rev-parse','HEAD'],text=True).strip(),
        original_code_execution='AST statements from source cell executed unchanged up to plotting; original take_time_diff executed',
        physical=dict(length_m=300,height_m=300,center_m=[150,130,-30],strike_deg=0,dip_deg=90,
                      S1_final_m=.001,width_final_m=.0001,well_points=300,gauge_m=10),
        time_interpretation='0..60 raw notebook values interpreted as minutes as labeled; SI times = 60 * raw values; native rates retained and divided by 60 for per-second values',
        reference_postprocessing='Original 0930 Well projection and sample-mean gauge; original JIN_pylib backward difference with zero first column',
        python=sys.version,numpy=np.__version__,platform=platform.platform(),
        original_source_changes='none')
    (OUT/'reference_provenance.json').write_text(json.dumps(provenance,indent=2))
    return frac,points,seconds,md,u,strain,rate_native/60


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generate-only',action='store_true')
    parser.add_argument('--reference-only',action='store_true')
    parser.add_argument('--analyze-only',action='store_true')
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--refine',action='store_true')
    parser.add_argument('--recycle',action='store_true')
    parser.add_argument('--h',type=float,default=20)
    parser.add_argument('--thickness',type=float,default=.2)
    parser.add_argument('--extent',type=float,default=2400)
    parser.add_argument('--np',type=int,default=8)
    args=parser.parse_args()
    frac,p,t,md,reference_u,reference_strain,reference_rate=original()
    if args.reference_only:
        print('Original notebook completed all 30 frames with intact DDMpy_0930:',OUT)
        return
    indices=np.array([0,14,29]) if args.pilot else np.arange(30)
    times=t[indices];factors=indices/29.
    directory=OUT/f"{'pilot' if args.pilot else 'full'}_h{args.h:g}_t{args.thickness:g}_L{args.extent:g}{'_refined' if args.refine else ''}"
    if args.recycle:directory=directory.with_name(directory.name+'_recycled')
    cfg=NumericalConfig(str(ROOT/'moose_env/moose/modules/combined/combined-opt'),str(directory),
        mesh_size=args.h,half_extent=args.extent,core_extent=180,band_thickness=args.thickness,
        processors=args.np,mpiexec=str(Path.home()/'miniforge/envs/moose/bin/mpiexec'))
    cfg.reuse_constant_jacobian=True
    cfg.recycle_krylov_guess=args.recycle
    if args.refine:
        cfg.local_refinements=((0,-160.,-100.,5.),(1,-30.,30.,10.),(2,-40.,40.,2.5))
    numerical=from_ddmpy(frac.fractures[-1],cfg)
    if args.generate_only:
        R,origin=numerical.frame()
        print(write_input(numerical.elements,(p-origin)@R.T,cfg,directory,(times,factors)))
        return
    if args.analyze_only:
        u=np.load(directory/'displacement_history.npz')['displacement']
    else:
        u=numerical.sample_history(p,times,factors)
    strain=[]
    w=numerical.monitor_wells[0]
    for ut in u:
        w.u1,w.u2,w.u3=ut.T
        w.calculate_strain()
        strain.append(w.strain.copy())
    strain=np.column_stack(strain)
    # Exact legacy observation operator; report the physical point derivative
    # separately because the legacy first rate column is artificially zero.
    rate=np.column_stack([np.zeros(len(md)),np.diff(strain,axis=1)/np.diff(times)])
    refstrain=reference_strain[:,indices]
    refrate=np.column_stack([np.zeros(len(md)),np.diff(refstrain,axis=1)/np.diff(times)])
    if not args.pilot:
        np.testing.assert_allclose(refrate,reference_rate,rtol=1e-10,atol=1e-20)
        refrate=reference_rate  # Actual original take_time_diff output, in SI.
    point_rate=time_derivative(strain,times)
    reference_point_rate=time_derivative(refstrain,times)
    result=dict(displacement_all_times=metrics(u,reference_u[indices]),
                displacement_final=metrics(u[-1],reference_u[-1]),
                components_final={k:metrics(u[-1,:,i],reference_u[-1,:,i]) for i,k in enumerate(['ux','uy','uz'])},
                strain=metrics(strain,refstrain),legacy_strain_rate=metrics(rate,refrate),
                corrected_point_strain_rate=metrics(point_rate,reference_point_rate),
                zero_loading_max_displacement=float(abs(u[0]).max()),
                actual_fe_load_times=len(indices)-1,initial_state='zero load, zero displacement: equilibrium satisfied exactly',
                receiver_count=len(p),native_notebook_frames=30,completed_notebook_frames=len(indices),
                extent_m=args.extent,thickness_m=args.thickness,mesh_size_m=args.h)
    np.savez(directory/'comparison.npz',points=p,md=md,times_seconds=times,u=u,reference_u=reference_u[indices],
             strain=strain,reference_strain=refstrain,rate=rate,reference_rate=refrate,
             point_rate=point_rate,reference_point_rate=reference_point_rate)
    (directory/'metrics.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for k in range(3):
        line,=axes[0].plot(md,reference_u[-1,:,k]*1e6,label=f'0930 DDMPY u{k+1}')
        axes[0].plot(md,u[-1,:,k]*1e6,'--',color=line.get_color(),label=f'MOOSE u{k+1}')
    axes[0].set(xlabel='MD (m)',ylabel='Final displacement (micrometre)');axes[0].legend(fontsize=8)
    axes[1].plot(md,refstrain[:,-1]*1e6,label='Original 0930 DDMPY')
    axes[1].plot(md,strain[:,-1]*1e6,'--',label='MOOSE FE')
    axes[1].set(xlabel='MD (m)',ylabel='Final axial microstrain');axes[1].legend()
    axes[2].plot(md,refrate[:,-1]*1e9,label='Original 0930 DDMPY')
    axes[2].plot(md,rate[:,-1]*1e9,'--',label='MOOSE FE')
    axes[2].set(xlabel='MD (m)',ylabel='Final nanostrain / s');axes[2].legend()
    fig.suptitle('Original existing tensile/shear notebook: same geometry, jumps, well and gauge')
    fig.tight_layout();fig.savefig(directory/'profiles.png',dpi=180);fig.savefig(directory/'profiles.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
    for row,(num,ref,label,scale) in enumerate([(strain,refstrain,'Microstrain',1e6),(rate,refrate,'Nanostrain/s',1e9)]):
        lim=max(abs(ref).max()*scale,1e-15)
        for ax,value,title in zip(axes[row],[ref,num,num-ref],['0930 DDMPY','MOOSE FE','FE minus DDMPY']):
            limit=max(abs(value).max()*scale,1e-15) if title=='FE minus DDMPY' else lim
            im=ax.pcolormesh(times/60,md,value*scale,shading='auto',cmap='seismic',vmin=-limit,vmax=limit)
            ax.set(title=title,ylabel='MD (m)',xlabel='Time (min)')
            fig.colorbar(im,ax=ax,label=label)
    fig.tight_layout();fig.savefig(directory/'waterfalls.png',dpi=170);fig.savefig(directory/'waterfalls.pdf');plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
