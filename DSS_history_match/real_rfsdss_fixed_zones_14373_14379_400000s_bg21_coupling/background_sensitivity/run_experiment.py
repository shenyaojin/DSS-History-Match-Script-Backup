"""Fixed-best-model forward pressure experiment: background -21, -19, -18.

No optimization, no retuning C, no change to zone tensor or pressure source.
The -21 control also checks that forward-only execution reproduces the saved fit.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'inversion'
ROOT = HERE.parents[3]
PYTHON_BIN = Path('/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin')
sys.path.insert(0, str(HERE.parents[1] / 'real_rfsdss_fixed_zones_14710_14725_400000s_bg21'))
from prepare_xy_comparison import replace_block, set_value, add_objects

BACKGROUNDS = [-21, -19, -18]


def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def status(path, stage, **kwargs):
    save(path, dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), **kwargs))


def case_dir(alpha):
    return HERE / f'bg_{abs(alpha)}'


def section(text, name):
    return re.search(r'^\[' + re.escape(name) + r'\].*?^\[\]', text, re.M | re.S)[0]


def prepare():
    if (HERE / 'manifest.json').exists() or any(case_dir(a).exists() for a in BACKGROUNDS):
        raise RuntimeError('Refusing to overwrite an existing experiment')
    cfg = json.loads((SOURCE / 'config.json').read_text())
    theta = np.loadtxt(SOURCE / 'theta_final.txt')
    original = (SOURCE / 'forward_and_adjoint.i').read_text()
    indices = cfg['zone_layer_indices'][0]
    zone_blocks = ' '.join(f'layer_{i+1}' for i in indices)
    other_blocks = ' '.join(f'layer_{i+1}' for i in range(200) if i not in indices)
    manifest = dict(source=str(SOURCE), fixed_theta=theta.tolist(), background_alphas=BACKGROUNDS,
        change_scope='Only non-zone kxx, kyy, kzz; zone tensor and C fixed at best coupled inversion',
        zone=cfg['zones'][0], zone_k_m2=[10**theta[0], 10**theta[1], 1e-21],
        n_solver_steps=129, max_steps=240, mesh=[200,216], mpi_ranks_per_case=20,
        max_concurrent_cases=2, observations='Existing positive native/interpolated data; exclude t=0',
        pressure_definition='pp is pressure change from initial zero; Pa in fields, MPa in plots',
        coverage_definition='Area-weighted cell-center pressure >= 1%, 5%, 10% of peak applied source pressure',
        coverage_caution='Thresholds describe response extent, not a sharp diffusion front or evidence of equilibrium',
        source_sha256={n: hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in
            ['forward_and_adjoint.i','theta_final.txt','measurement_data.csv','solver_times.csv','config.json']},
        cases={})
    for alpha in BACKGROUNDS:
        folder = case_dir(alpha); folder.mkdir()
        full = np.full(201, float(alpha)); full[indices] = theta[0]; full[-1] = theta[1]
        forward = replace_block(original, 'params', lambda b: set_value(b, 'real_vector_values',
            '; '.join(f'{v:.16g}' for v in full)))
        forward = replace_block(forward, 'func_kyy', lambda b: set_value(b, 'expression', f'1e{alpha}'))
        # Preserve zone kzz=1e-21 and assign background kzz only outside the zone.
        forward = replace_block(forward, 'perm_zz_aux', lambda b: b.replace(
            "    function = 'func_zero'", "    function = 'func_zero'\n    block = '"+zone_blocks+"'"))
        forward = add_objects(forward, 'AuxKernels', f"""  [background_zz]
    type = FunctionAux
    variable = 'perm_zz_var'
    function = 'func_kyy'
    block = '{other_blocks}'
    execute_on = 'INITIAL TIMESTEP_BEGIN TIMESTEP_END'
  []
""")
        forward = forward.replace('solver_measurement_data.csv', 'measurement_data.csv')
        # The standalone forward needs the observations formerly transferred by master.
        forward = replace_block(forward, 'data', lambda b: b.replace(
            "    variable = 'strain_positive'", """    variable = 'strain_positive'
    measurement_file = 'measurement_data.csv'
    file_xcoord = 'measurement_xcoord'
    file_ycoord = 'measurement_ycoord'
    file_zcoord = 'measurement_zcoord'
    file_time = 'measurement_time'
    file_value = 'measurement_values'"""))
        # TAA's forward solve is NEWTON + implicit Euler on nl0. Keep exactly that,
        # but do not invoke its backward adjoint loop or an optimization master.
        before = section(forward, 'Executioner')
        after = before.replace("type = 'TransientAndAdjoint'", "type = 'Transient'\n  solve_type = NEWTON\n  scheme = implicit-euler")
        after = after.replace("forward_system = 'nl0'", "system_names = 'nl0'")
        after = after.replace("  adjoint_system = 'adjoint'\n", '')
        forward = forward.replace(before, after)
        # The Sep-30 executable omits the previous custom adjoint-gradient VPP.
        # All 201 objects here are gradient-only and unused in a forward solve.
        vpp = section(forward, 'VectorPostprocessors')
        assert set(re.findall(r'type\s*=\s*(\S+)', vpp)) == {'PorousFlowOptimizationAnisotropicDiffusionInnerProduct'}
        forward = forward.replace(vpp, '')
        forward = add_objects(forward, 'Outputs', """  [pressure_fields]
    type = Exodus
    file_base = pressure
    show = 'pp strain_yy strain_positive'
    execute_on = 'INITIAL TIMESTEP_END'
  []
""")
        for name in ['Mesh','Kernels','Materials','BCs','Preconditioning','FluidProperties','Variables']:
            assert section(forward,name) == section(original,name), name
        assert 'num_steps = 240' in after and 'nl_abs_tol = 0.001' in after
        assert 'nl_rel_tol = 0.001' in after and 'l_tol = 0.001' in after
        assert 'time_sequence' in after
        (folder/'forward.i').write_text(forward)
        for name in ['measurement_data.csv','solver_times.csv']:
            shutil.copy2(SOURCE/name,folder/name)
        save(folder/'parameters.json', dict(background_alpha=alpha, background_k_m2=[10.**alpha]*3,
            zone_alpha_xx=float(theta[0]),zone_alpha_yy=float(theta[1]),zone_kzz_m2=1e-21,
            coupling_C=float(theta[2]), fixed_zone_blocks=zone_blocks))
        manifest['cases'][str(alpha)] = dict(input_sha256=hashlib.sha256(forward.encode()).hexdigest())
    save(HERE/'manifest.json', manifest)
    status(HERE/'status.json','prepared')
    print('Prepared: fixed zone/C; 3 backgrounds; 129 forward steps each', flush=True)


def solve(alpha, check=False):
    folder=case_dir(alpha)
    manifest=json.loads((HERE/'manifest.json').read_text())
    assert hashlib.sha256((folder/'forward.i').read_bytes()).hexdigest()==manifest['cases'][str(alpha)]['input_sha256']
    if not check and ((folder/'STARTED').exists() or (folder/'COMPLETE').exists()):
        raise RuntimeError(f'Refusing duplicate run {folder}')
    if not check:
        if not (folder/'INPUT_CHECK_PASSED').exists(): raise RuntimeError('Input check required')
        (folder/'STARTED').write_text(datetime.now(timezone.utc).isoformat())
    env=os.environ.copy();env['PATH']=str(PYTHON_BIN)+':'+env['PATH']
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TBB_NUM_THREADS']:
        env[name]='1'
    command=[str(PYTHON_BIN/'mpiexec'),'-n','20',str(ROOT/'moose_env/moose/modules/combined/combined-opt'),
             '-i','forward.i','--n-threads=1']
    if check: command.append('--check-input')
    status(folder/'status.json','checking_input' if check else 'running',command=command)
    start=time.monotonic()
    with (folder/('check.log' if check else 'forward.log')).open('w') as log:
        result=subprocess.run(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        status(folder/'status.json','failed',returncode=result.returncode)
        raise RuntimeError(f'Background {alpha}: MOOSE failed; see log')
    marker='INPUT_CHECK_PASSED' if check else 'COMPLETE'
    (folder/marker).write_text('Passed\n')
    status(folder/'status.json','input_checked' if check else 'complete',wall_seconds=time.monotonic()-start)
    print(f'Background {alpha}: {marker}',flush=True)


def strain(alpha):
    folder=case_dir(alpha)
    files=[p for p in folder.glob('forward*data_*.csv') if '_signed_data_' not in p.name]
    path=max(files,key=lambda p:int(re.search(r'_(\d+)\.csv$',p.name)[1]))
    df=pd.read_csv(path)
    canonical=pd.read_csv(SOURCE/'measurement_data.csv')
    # The observation cache also has placeholder simulation/misfit columns;
    # compare immutable observation fields, not those solver output columns.
    keys=['measurement_time','measurement_values','measurement_xcoord','measurement_ycoord','measurement_zcoord']
    np.testing.assert_allclose(df[keys],canonical[keys],rtol=1e-10,atol=1e-18)
    assert len(df)==18*130
    coupling=json.loads((HERE/'manifest.json').read_text())['fixed_theta'][2]
    df['physical_simulation_values']=df.simulation_values
    df['simulation_values']*=coupling
    df['misfit_values']=np.where(df.measurement_time>0,df.simulation_values-df.measurement_values,0)
    df.to_csv(folder/'strain_prediction.csv',index=False)
    return df


def baseline_check():
    df=strain(-21);previous=pd.read_csv(SOURCE/'strain_final.csv')
    use=df.measurement_time>0
    difference=df.loc[use,'simulation_values'].to_numpy()-previous.loc[use,'simulation_values'].to_numpy()
    delta=float(np.sqrt(np.mean(difference**2)))
    scale=float(np.sqrt(np.mean(previous.loc[use,'simulation_values']**2)))
    result=dict(rms_difference_strain=delta,relative_rms_difference=delta/scale,
                passed=bool(delta<=max(1e-12,1e-3*scale)))
    save(HERE/'baseline_check.json',result)
    if not result['passed']: raise RuntimeError('Forward-only control does not reproduce previous best fit; inspect before new cases')


def pressure(alpha):
    from scipy.io import netcdf_file
    # libMesh writes classic NetCDF here; scipy avoids the environment's broken
    # optional netCDF4/cftime import without changing any installed packages.
    with netcdf_file(case_dir(alpha)/'pressure.e','r',mmap=False) as dataset:
        ds=dataset.variables
        names=[row.tobytes().decode().strip('\x00 ') for row in ds['name_nod_var'][:]]
        idx=names.index('pp')+1
        pp=np.asarray(ds[f'vals_nod_var{idx}'][:],dtype=float)
        tt=np.asarray(ds['time_whole'][:],dtype=float)
        x=np.asarray(ds['coordx'][:]);y=np.asarray(ds['coordy'][:])
    xx=np.unique(x);yy=np.unique(y)
    assert len(xx)==201 and len(yy)==217 and len(tt)==130
    order=np.lexsort((x,y))
    np.testing.assert_allclose(x[order].reshape(len(yy),len(xx)),np.broadcast_to(xx,(len(yy),len(xx))))
    np.testing.assert_allclose(y[order].reshape(len(yy),len(xx)),np.broadcast_to(yy[:,None],(len(yy),len(xx))))
    p=pp[:,order].reshape(len(tt),len(yy),len(xx))
    expected=np.sort(pd.read_csv(SOURCE/'measurement_data.csv').measurement_time.unique())
    np.testing.assert_allclose(tt,expected,rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(p[0],0,atol=1e-8)
    assert np.isfinite(p).all()
    return tt,xx,yy,p


def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    manifest=json.loads((HERE/'manifest.json').read_text())
    fields={a:pressure(a) for a in BACKGROUNDS}
    tt,x,y,_=fields[-21]
    source_index=int(np.argmin(abs(x-45)))
    applied=fields[-21][3][:,0,source_index]
    peak=float(applied.max())
    assert peak>0
    # Nonuniform y cells require area weights, not sample/node counts.
    area=np.diff(y)[:,None]*np.diff(x)[None,:]
    yc=(y[1:]+y[:-1])/2;xc=(x[1:]+x[:-1])/2
    zone=manifest['zone'];zone_lo=50+(zone['lo_ft']-14376)*.3048;zone_hi=50+(zone['hi_ft']-14376)*.3048
    snapshots=[int(np.argmin(abs(tt-target))) for target in [100000,200000,300000,tt[-1]]]
    rows=[];all_metrics=[]
    metrics_by_alpha={}
    for alpha,(_,_,_,p) in fields.items():
        np.testing.assert_allclose(p[:,:,source_index],np.broadcast_to(applied[:,None],p[:,:,source_index].shape),rtol=1e-8,atol=1e-4)
        pc=(p[:,:-1,:-1]+p[:,1:,:-1]+p[:,:-1,1:]+p[:,1:,1:])/4
        frame=pd.DataFrame(dict(time_s=tt,source_pressure_MPa=applied/1e6,
            left_max_MPa=p[:,:,0].max(axis=1)/1e6,right_max_MPa=p[:,:,-1].max(axis=1)/1e6,
            corner_max_MPa=p[:,[0,0,-1,-1],[0,-1,0,-1]].max(axis=1)/1e6,
            left_mid_MPa=p[:,np.argmin(abs(y-50)),0]/1e6,
            right_mid_MPa=p[:,np.argmin(abs(y-50)),-1]/1e6,
            left_far_MPa=p[:,0,0]/1e6,right_far_MPa=p[:,0,-1]/1e6,
            domain_min_MPa=p.min(axis=(1,2))/1e6,domain_max_MPa=p.max(axis=(1,2))/1e6))
        for fraction in [.01,.05,.1]:
            frame[f'area_ge_{int(fraction*100)}pct_source_peak']=np.sum((pc>=fraction*peak)*area,axis=(1,2))/area.sum()
        frame.to_csv(case_dir(alpha)/'pressure_metrics.csv',index=False)
        metrics_by_alpha[alpha]=frame
        df=strain(alpha);use=df.measurement_time>0
        j=float(.5*np.sum(df.loc[use,'misfit_values']**2))
        rms=float(np.sqrt(np.mean(df.loc[use,'misfit_values']**2))*1e6)
        final=frame.iloc[-1].to_dict()
        final.update(background_alpha=alpha,data_objective=j,rms_microstrain=rms,
            maximum_area_ge_1pct=float(frame.area_ge_1pct_source_peak.max()),
            peak_corner_pressure_MPa=float(frame.corner_max_MPa.max()))
        rows.append(final)
        frame['background_alpha']=alpha;all_metrics.append(frame)
    pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False)
    pd.concat(all_metrics).to_csv(HERE/'pressure_history.csv',index=False)
    save(HERE/'summary.json',dict(source_peak_MPa=peak/1e6,final_time_s=float(tt[-1]),cases=rows,
        baseline_check=json.loads((HERE/'baseline_check.json').read_text()),
        fixed_zone_tensor_m2=manifest['zone_k_m2'],coupling_C=manifest['fixed_theta'][2],
        interpretation='Finite-domain forward sensitivity at fixed fitted zone/C; no claim that new backgrounds are reoptimized'))
    for zoom,name in [(False,'pressure_full_domain'),(True,'pressure_zone_zoom')]:
        fig,axes=plt.subplots(3,4,figsize=(17,11),layout='constrained',sharex=True,sharey=True)
        for i,alpha in enumerate(BACKGROUNDS):
            p=fields[alpha][3]
            for j,k in enumerate(snapshots):
                ax=axes[i,j]
                im=ax.pcolormesh(x,y,p[k]/1e6,cmap='viridis',norm=PowerNorm(.5,vmin=0,vmax=peak/1e6),shading='gouraud',rasterized=True)
                if p[k].min()<.01*peak<p[k].max():ax.contour(x,y,p[k],levels=[.01*peak],colors=['white'],linewidths=.65)
                ax.axvline(45,color='red',lw=.7,ls='--');ax.axvline(69.384,color='cyan',lw=.7,ls=':')
                for edge in [zone_lo,zone_hi]:ax.axhline(edge,color='orange',lw=.45)
                ax.set_xlim(0,100);ax.set_ylim((45,55) if zoom else (0,100))
                if i==0:ax.set_title(f't = {tt[k]/1000:.1f} ks')
                if j==0:ax.set_ylabel(f'Background alpha = {alpha}\ny [m]')
                if i==2:ax.set_xlabel('x [m]')
        fig.colorbar(im,ax=axes,label='Pressure change [MPa]; common sqrt color scale',shrink=.85)
        fig.suptitle('Fixed best-fit zone tensor and C | Red: source; cyan: receiver; white: 1% of source peak')
        fig.savefig(HERE/(name+'.png'),dpi=180);fig.savefig(HERE/(name+'.pdf'));plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for alpha in BACKGROUNDS:
        f=metrics_by_alpha[alpha];label=f'Background {alpha}'
        axes[0,0].plot(tt/1000,f.left_mid_MPa,label=label)
        axes[0,1].plot(tt/1000,f.right_mid_MPa,label=label)
        axes[1,0].plot(tt/1000,f.corner_max_MPa,label=label)
        axes[1,1].plot(tt/1000,100*f.area_ge_1pct_source_peak,label=label)
    for ax,title in zip(axes.flat,['Left boundary at zone center','Right boundary at zone center','Maximum over four corners','Area above 1% of peak source pressure']):
        ax.set(title=title,xlabel='Time [1000 s]',ylabel='Pressure change [MPa]');ax.grid(alpha=.2);ax.legend()
    axes[1,1].set_ylabel('Domain area [%]');axes[1,1].set_ylim(0,100)
    for ax in axes[0]:ax.plot(tt/1000,applied/1e6,color='k',ls='--',lw=.8,label='Source');ax.legend()
    fig.savefig(HERE/'boundary_and_coverage.png',dpi=180);plt.close(fig)
    # Same spatial/temporal scale in the three DSS predictions; observed native waterfall at left.
    native=np.load(HERE.parent/'data/half_wave_positive_native.npz')
    dfs={a:strain(a) for a in BACKGROUNDS}
    md=14376+(np.sort(dfs[-21].measurement_ycoord.unique())-50)/.3048
    def matrix(df,col):return df.pivot(index='measurement_ycoord',columns='measurement_time',values=col).to_numpy()*1e6
    predictions={a:matrix(df,'simulation_values') for a,df in dfs.items()}
    obs=matrix(dfs[-21],'measurement_values')
    vmax=max(3.,max(float(v.max()) for v in predictions.values()))
    fig=plt.figure(figsize=(22,10),layout='constrained');grid=fig.add_gridspec(6,5,width_ratios=[1.25]*4+[2.4])
    for j,(label,values,times) in enumerate([('Observed',native['data']*1e6,native['taxis'])]+[(f'Background {a}',predictions[a],tt) for a in BACKGROUNDS]):
        ax=fig.add_subplot(grid[:,j]);im=ax.pcolormesh(times/1000,md,values,shading='nearest',cmap='Reds',vmin=0,vmax=vmax)
        ax.set(title=label,xlabel='Time [1000 s]',ylim=(14379,14373),xlim=(0,400))
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        if j==0:ax.set_ylabel('Measured depth [ft]')
        else:ax.tick_params(labelleft=False)
    core=np.flatnonzero((md>=zone['lo_ft'])&(md<zone['hi_ft']));shown=np.r_[core[0]-1,core,core[-1]+1]
    for i,ch in enumerate(shown):
        ax=fig.add_subplot(grid[i,4]);ax.plot(tt/1000,obs[ch],color='k',label='Observation')
        for a in BACKGROUNDS:ax.plot(tt/1000,predictions[a][ch],label=f'Bg {a}')
        ax.set(title=f'{md[ch]:.3f} ft',xlim=(0,400),ylabel='microstrain');ax.grid(alpha=.2)
        if i==0:ax.legend(fontsize=8,ncol=2)
        if i==5:ax.set_xlabel('Time [1000 s]')
    fig.suptitle(f'Forward sensitivity only: zone tensor fixed; C={manifest["fixed_theta"][2]:.6f} fixed | Common waterfall scale 0–{vmax:.2f} microstrain')
    fig.savefig(HERE/'dss_comparison.png',dpi=160);plt.close(fig)
    status(HERE/'status.json','complete',cases=BACKGROUNDS)
    (HERE/'COMPLETE').write_text('Three forward cases and comparisons completed\n')
    (HERE/'FAILED').unlink(missing_ok=True)
    print(json.dumps(rows,indent=2),flush=True)


def run(reuse_control=False):
    if reuse_control:
        if not (HERE/'FAILED').exists() or not (case_dir(-21)/'COMPLETE').exists():
            raise RuntimeError('Resume requires a stopped controller and completed control solve')
        if any((case_dir(a)/'STARTED').exists() for a in [-19,-18]):
            raise RuntimeError('Resume refuses to duplicate new background solves')
    else:
        if (HERE/'STARTED').exists(): raise RuntimeError('Experiment already started')
        (HERE/'STARTED').write_text(datetime.now(timezone.utc).isoformat())
    try:
        status(HERE/'status.json','baseline_verification' if reuse_control else 'baseline_running')
        if not reuse_control:solve(-21)
        baseline_check()
        status(HERE/'status.json','new_backgrounds_running')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(solve,a) for a in [-19,-18]]
            for future in futures:future.result()
        summarize()
    except Exception as exc:
        status(HERE/'status.json','failed',reason=str(exc));(HERE/'FAILED').write_text(str(exc)+'\n');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','check','run','resume','summarize'])
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='check':
        for alpha in BACKGROUNDS:solve(alpha,check=True)
    elif args.action=='run':run()
    elif args.action=='resume':run(reuse_control=True)
    else:summarize()
