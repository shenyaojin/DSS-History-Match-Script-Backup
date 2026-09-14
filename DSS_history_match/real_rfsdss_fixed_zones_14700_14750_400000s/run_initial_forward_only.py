from pathlib import Path
import argparse,json,re,shutil,sys
import numpy as np
import pandas as pd
from fiberis.moose.runner import MooseRunner

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
ROOT=HERE/'initial_forward_only'
TEMPLATE=HERE.parent/'optimizer_input_file_test/perm5layer_100_v6strain_zonalgrid/inv/_template/forward_and_adjoint.i'

def solver_sampling(run,cfg):
    template=TEMPLATE.read_text()
    times=np.fromstring(re.search(r"time_sequence = '([^']*)'",template)[1],sep=' ')
    times[-1]=cfg['end_time']
    assert times[0]==0 and np.all(np.diff(times)>0) and len(times)-1<=240
    # OptimizationData samples only at exact solver times. Give it matching
    # observations; keep the native 3318 samples separately for visualization.
    with np.load(HERE/'unrectified/observations.npz',allow_pickle=True) as d:
        obs=np.array([np.interp(times,d['taxis'],row) for row in d['data']])
        md=d['daxis'];native_count=len(d['taxis'])
    if cfg['rectified']:obs=np.maximum(obs,0)
    y=50+(md.astype(float)-14725)*.3048
    frame=pd.DataFrame(dict(measurement_time=np.repeat(times,len(md)),
        measurement_values=obs.T.ravel(),measurement_xcoord=cfg['receiver_x_m'],
        measurement_ycoord=np.tile(y,len(times)),measurement_zcoord=0.,
        misfit_values=0.,simulation_values=0.))
    frame.to_csv(run/'measurement_data.csv',index=False)
    edges=np.asarray(cfg['mesh_edges_m']);layer=np.searchsorted(edges,y,side='right')-1
    geometry=[np.tile(edges[layer+1],len(times)),np.tile(edges[layer],len(times)),
        np.tile(1/np.diff(edges)[layer],len(times)),
        np.full(obs.size,(np.floor(cfg['receiver_x_m']/.5)+.5)*.5)]
    cfg.update(n_time=len(times),native_n_time=native_count,n_solver_steps=len(times)-1,
        max_solver_steps=240,time_sequence_source=str(TEMPLATE),
        observation_sampling='Linear interpolation of signed observations to template times; rectify afterwards if requested',
        nl_abs_tol=.001,nl_rel_tol=.001,l_tol=.001)
    np.savetxt(run/'solver_times.csv',times,header='time_s',comments='')
    return times,geometry

def prepare(case):
    source=HERE/case;run=ROOT/case;run.mkdir(parents=True,exist_ok=True)
    if (run/'INITIAL_FORWARD_COMPLETE').exists():raise RuntimeError('Initial forward result already complete; refusing to overwrite')
    cfg=json.loads((source/'config.json').read_text())
    times,geometry=solver_sampling(run,cfg)
    alpha=np.full(200,cfg['background_alpha'])
    for value,indices in zip(cfg['initial_theta'],cfg['zone_layer_indices']):alpha[indices]=value
    assert np.array_equal(alpha,np.loadtxt(source/'initial_alpha.txt'))
    text=(source/'forward_and_adjoint.i').read_text()
    text=text.replace("type = 'TransientAndAdjoint'","type = Transient")
    text=text.replace("  forward_system = 'nl0'","  system_names = 'nl0'\n  solve_type = NEWTON")
    text=text.replace("  adjoint_system = 'adjoint'\n",'')
    fmt=lambda a:' '.join(f'{x:.16g}' for x in a)
    text,n=re.subn(r"time_sequence = '[^']*'",lambda m:"time_sequence = '"+fmt(times)+"'",text,count=1);assert n==1
    for key in ['nl_abs_tol','nl_rel_tol']:
        text,n=re.subn(rf'{key} = [^\n]+',f'{key} = 0.001',text,count=1);assert n==1
    text=text.replace('  end_time = ', '  num_steps = 240\n  abort_on_solve_fail = true\n  end_time = ',1)
    text=text.replace('console = false','console = true')
    pattern=r"(  \[sampling_geometry\].*?real_vector_values = ')[^']*(')"
    text,n=re.subn(pattern,lambda m:m[1]+'; '.join(fmt(a) for a in geometry)+m[2],text,count=1,flags=re.S);assert n==1
    # Supply the fixed initial parameters directly; no optimizer or MultiApp.
    pattern=r"(  \[params\].*?real_vector_values = ')[^']*(')"
    text,n=re.subn(pattern,lambda m:m[1]+'; '.join(f'{x:.16g}' for x in alpha)+m[2],text,count=1,flags=re.S);assert n==1
    data_input=f"""  [data]
    measurement_file = '{run/'measurement_data.csv'}'
    file_xcoord = measurement_xcoord
    file_ycoord = measurement_ycoord
    file_zcoord = measurement_zcoord
    file_time = measurement_time
    file_value = measurement_values
"""
    text,n=re.subn(r'  \[data\]\n',lambda m:data_input,text,count=1);assert n==1
    assert 'TransientAndAdjoint' not in text
    (run/'initial_forward.i').write_text(text)
    (run/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    np.savetxt(run/'initial_theta.txt',cfg['initial_theta'])
    np.savetxt(run/'initial_alpha.txt',alpha)
    return run,cfg

def plot_case(run,cfg,frame):
    import matplotlib.pyplot as plt
    times=np.sort(frame.measurement_time.unique());ys=np.sort(frame.measurement_ycoord.unique());md=14725+(ys-50)/.3048
    grid=lambda col:frame.pivot(index='measurement_ycoord',columns='measurement_time',values=col).reindex(index=ys,columns=times).to_numpy()*1e6
    sim=grid('simulation_values')
    with np.load(HERE/run.name/'observations.npz',allow_pickle=True) as d:
        obs=d['data']*1e6;obs_times=d['taxis']
        assert np.allclose(md,d['daxis'])
    fig,axs=plt.subplots(1,2,figsize=(14,7),sharex=True,sharey=True,constrained_layout=True)
    for ax,t,val,limit,title in zip(axs,[obs_times,times],[obs,sim],[.5,30.],['Observed DSS (native samples)',f'Simulation at ORIGINAL initial α ({len(times)-1} steps)']):
        im=ax.pcolormesh(t,md,val,cmap='bwr',vmin=-limit,vmax=limit,shading='auto',rasterized=True)
        fig.colorbar(im,ax=ax,label='Strain change [με]',extend='both',shrink=.8)
        ax.axvspan(times[-1],400000,color='0.8',hatch='//',alpha=.5)
        ax.set(title=title,xlabel='Time [s]',xlim=(0,400000),ylim=(14750,14700))
    axs[0].set_ylabel('Measured depth [ft]')
    fig.suptitle(f"POW-S | {run.name} | Forward only; no optimization\nInitial α: [-15.5, -15.5, -15.5, -16.5, -16.5]; background −18",fontsize=14)
    fig.savefig(run/'observed_vs_initial_simulation.png',dpi=160)
    fig.savefig(run/'observed_vs_initial_simulation.pdf');plt.close(fig)
    np.savez_compressed(run/'comparison_arrays.npz',time=times,observed_time=obs_times,md=md,observed_microstrain=obs,simulated_microstrain=sim)

def combine():
    names=['unrectified','half_wave_positive']
    if not all((ROOT/n/'INITIAL_FORWARD_COMPLETE').exists() for n in names):return
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(14,12),sharex=True,sharey=True,constrained_layout=True)
    for row,name in enumerate(names):
        with np.load(ROOT/name/'comparison_arrays.npz') as d:
            t=d['time'];md=d['md']
            for col,key in enumerate(['observed_microstrain','simulated_microstrain']):
                plot_t=d['observed_time'] if col==0 else t
                limit=.5 if col==0 else 30.
                ax=axs[row,col];im=ax.pcolormesh(plot_t,md,d[key],shading='auto',cmap='bwr',vmin=-limit,vmax=limit,rasterized=True)
                if row==0:
                    fig.colorbar(im,ax=axs[:,col].tolist(),label=f"{['Observed','Simulated'][col]} strain change [με]",extend='both',shrink=.75)
                ax.axvspan(t[-1],400000,color='0.8',hatch='//',alpha=.5)
                ax.set(xlim=(0,400000),ylim=(14750,14700),xlabel='Time [s]',title=f"{name} | {['Observed DSS','Original initial α simulation'][col]}")
            axs[row,0].set_ylabel('Measured depth [ft]')
    fig.suptitle('0–400,000 s | Original initial-parameter forward comparison\nNo parameter optimization; same fixed initial model in both cases',fontsize=15)
    fig.savefig(ROOT/'observed_vs_initial_simulation_both.png',dpi=160)
    fig.savefig(ROOT/'observed_vs_initial_simulation_both.pdf');plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['unrectified','half_wave_positive'],required=True)
    p.add_argument('--np',type=int,default=20);p.add_argument('--check-input',action='store_true')
    args=p.parse_args();run,cfg=prepare(args.case)
    runner=MooseRunner(str(REPO/'moose_env/moose/modules/combined/combined-opt'),'/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec')
    output=run/('input_check' if args.check_input else 'solver_output')
    ok,stdout,stderr=runner.run(str(run/'initial_forward.i'),str(output),num_processors=args.np,
          additional_args=['--check-input'] if args.check_input else None,
          log_file_name='initial_forward.log',stream_output=True,clean_output_dir=False)
    (run/'stderr.log').write_text(stderr)
    if not ok:raise RuntimeError('Initial forward failed; see solver log')
    if args.check_input:return
    files=list(output.glob('*_data_*.csv'))
    if not files:raise RuntimeError('Missing forward observation output')
    report=max(files,key=lambda q:int(re.search(r'_data_(\d+)\.csv$',q.name)[1]))
    frame=pd.read_csv(report)
    assert len(frame)==cfg['n_channels']*cfg['n_time']
    assert np.isclose(frame.measurement_time.max(),cfg['end_time'])
    assert np.all(np.isfinite(frame.simulation_values))
    assert np.any(frame.simulation_values!=0)
    active=frame.measurement_time>0
    assert np.allclose((frame.simulation_values-frame.measurement_values)[active],frame.misfit_values[active],rtol=1e-8,atol=1e-18)
    if cfg['rectified']:assert np.all(frame.simulation_values>=0)
    shutil.copy2(report,run/'strain_initial.csv')
    plot_case(run,cfg,frame)
    stats=dict(theta=cfg['initial_theta'],optimized=False,forward_only=True,n_time=cfg['n_time'],
               n_solver_steps=cfg['n_solver_steps'],native_n_time=cfg['native_n_time'],
               simulation_min=float(frame.simulation_values.min()),simulation_max=float(frame.simulation_values.max()),
               data_misfit=float(.5*np.sum(frame.misfit_values**2)))
    (run/'initial_forward_summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    (run/'INITIAL_FORWARD_COMPLETE').write_text('Original initial-alpha forward run and comparison figure complete; no optimization.\n')
    if args.case=='unrectified':
        # Both variants have identical PDEs and initial parameters. Rectification
        # is only an observation operator, so one physical forward solve suffices.
        rect_run,rect_cfg=prepare('half_wave_positive')
        assert rect_cfg['initial_theta']==cfg['initial_theta']
        assert rect_cfg['zone_layer_indices']==cfg['zone_layer_indices']
        rect=frame.copy()
        rect['simulation_values']=np.maximum(rect.simulation_values,0)
        rect['measurement_values']=np.maximum(rect.measurement_values,0)
        rect['misfit_values']=np.where(rect.measurement_time>0,rect.simulation_values-rect.measurement_values,0.)
        supplied=pd.read_csv(rect_run/'measurement_data.csv')
        assert np.allclose(rect.measurement_values,supplied.measurement_values,rtol=1e-7,atol=1e-18)
        rect.to_csv(rect_run/'strain_initial.csv',index=False)
        plot_case(rect_run,rect_cfg,rect)
        rect_stats=dict(stats)
        rect_stats.update(source='Half-wave transform of the identical unrectified initial-model forward solution',
                          simulation_min=float(rect.simulation_values.min()),simulation_max=float(rect.simulation_values.max()),
                          data_misfit=float(.5*np.sum(rect.misfit_values**2)))
        (rect_run/'initial_forward_summary.json').write_text(json.dumps(rect_stats,indent=2)+'\n')
        (rect_run/'INITIAL_FORWARD_COMPLETE').write_text('Same initial-alpha physical forward solution, half-wave observation operator applied; no optimization.\n')
    combine()
    print('INITIAL_FORWARD_COMPLETE',args.case,flush=True)
if __name__=='__main__':main()
