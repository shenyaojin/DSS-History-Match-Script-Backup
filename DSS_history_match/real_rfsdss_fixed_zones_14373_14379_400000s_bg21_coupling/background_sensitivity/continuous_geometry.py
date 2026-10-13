"""Continuous-depth geometry trials on locally refined, boundary-fitted meshes.

Observation coordinates and acquisition-to-solver interpolation remain immutable.
Each trial performs 129 physical solves followed by an exact weighted C refit.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

import numpy as np
import pandas as pd

import run_experiment as base
import plot_weighted_refit_separate as qc

HERE = Path(__file__).resolve().parent
OUT = HERE/'continuous_geometry_bg18'
SOURCE = HERE/'bg_18'
PRIOR = HERE/'fracture_geometry_bg18/deeper_edge_one_channel'
REFERENCE = HERE/'coupling_refit_weighted3_unbounded'
KEYS = ['measurement_time','measurement_values','measurement_xcoord','measurement_ycoord','measurement_zcoord']
PLOT_LOCK = threading.Lock()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mesh_for(center, width, h):
    cfg = json.loads((HERE.parent/'inversion/config.json').read_text())
    materials = np.array(cfg['mesh_edges_m'])
    lo, hi = center-width/2, center+width/2
    assert 14373 < lo < center < hi < 14379
    materials[[99,100,101]] = 50 + (np.array([lo,center,hi])-14376)*.3048
    obs = pd.read_csv(SOURCE/'measurement_data.csv')
    ys = np.sort(obs.measurement_ycoord.unique())
    edges = [0.]
    for left, right in zip(materials[:-1], materials[1:]):
        target = h if left >= 49-1e-10 and right <= 51+1e-10 else .5
        n = max(1, int(np.ceil((right-left)/target-1e-10)))
        # Avoid an observation exactly on an internal element edge, where a
        # discontinuous constant auxiliary field has an ambiguous point value.
        for attempt in range(10):
            part = np.linspace(left,right,n+1)
            if len(part)<=2 or np.min(abs(part[1:-1,None]-ys[None,:]))>1e-7:
                break
            n += 1
        edges.extend(part[1:])
    edges = np.array(edges)
    assert np.min(abs(materials[:,None]-ys[None,:]))>1e-7
    assert np.all(np.diff(edges)>0) and np.all(np.isin(materials,edges))
    assert len(np.unique(np.searchsorted(edges,ys)-1)) == 18
    mid=(edges[:-1]+edges[1:])/2
    local=(mid>=49)&(mid<=51)
    assert np.max(np.diff(edges)[local])<=h+1e-10
    blocks=np.searchsorted(materials,mid,side='right')-1
    np.testing.assert_allclose(np.sum(np.diff(edges)[np.isin(blocks,[99,100])]),width*.3048,rtol=0,atol=1e-10)
    return edges, materials, obs


def prepare_case(name, center, width, h, theta=None):
    folder = OUT/name
    if folder.exists():
        m = json.loads((folder/'manifest.json').read_text())
        np.testing.assert_allclose([center,width,h], [m['center_ft'],m['width_ft'],m['h_y_max_m']], rtol=0,atol=1e-9)
        if theta is not None:
            np.testing.assert_allclose(theta,m['theta'])
        return folder
    edges, materials, obs = mesh_for(center,width,h)
    original = (SOURCE/'forward.i').read_text()
    text = base.replace_block(original,'base_mesh',lambda b:base.set_value(b,'dy',' '.join(f'{d:.16g}' for d in np.diff(edges))))
    for i in range(98,102):
        text = base.replace_block(text,f'layer_{i+1}_bbox',lambda b,i=i:base.set_value(
            base.set_value(b,'bottom_left',f'0 {materials[i]:.16g} 0'),'top_right',f'100 {materials[i+1]:.16g} 0'))
    cells = np.searchsorted(edges,obs.measurement_ycoord.to_numpy(),side='right')-1
    sampling = [edges[cells+1],edges[cells],1/np.diff(edges)[cells],
                (np.floor(obs.measurement_xcoord.to_numpy()/.5)+.5)*.5]
    text = base.replace_block(text,'sampling_geometry',lambda b:base.set_value(b,'real_vector_values',
        '; '.join(' '.join(f'{v:.16g}' for v in values) for values in sampling)))
    prior_theta = json.loads((HERE/'manifest.json').read_text())['fixed_theta'][:2]
    theta = prior_theta if theta is None else list(theta)
    if theta != prior_theta:
        full = np.full(201,-18.);full[[99,100]]=theta[0];full[-1]=theta[1]
        text = base.replace_block(text,'params',lambda b:base.set_value(b,'real_vector_values','; '.join(f'{v:.16g}' for v in full)))
    for section in ['Kernels','Materials','AuxKernels','Functions','BCs','Executioner','Variables','FluidProperties','Preconditioning']:
        assert base.section(text,section)==base.section(original,section), section
    folder.mkdir(parents=True)
    (folder/'forward.i').write_text(text)
    for filename in ['measurement_data.csv','solver_times.csv']:
        shutil.copy2(SOURCE/filename,folder/filename)
    (folder/'.gitignore').write_text('pressure.e*\nforward_csv*\n*.log\n')
    core = json.loads((REFERENCE/'summary.json').read_text())['central_channel_depths_ft']
    base.save(folder/'manifest.json',dict(center_ft=float(center),width_ft=float(width),
        lo_ft=float(center-width/2),hi_ft=float(center+width/2),h_y_max_m=float(h),
        mesh=[200,len(edges)-1],element_edges_y_m=edges.tolist(),material_edges_y_m=materials.tolist(),
        theta=theta,background_alpha=-18.,zone_kzz_m2=1e-21,n_steps=129,max_steps=240,
        central_channel_depths_ft=core,core_weight=3.,coupling_bounds=None,
        source_sha256={str(SOURCE/f):digest(SOURCE/f) for f in ['forward.i','measurement_data.csv','solver_times.csv']},
        input_sha256=digest(folder/'forward.i'),measurement_sha256=digest(folder/'measurement_data.csv')))
    base.status(folder/'status.json','prepared')
    return folder


def summarize(folder, make_plot=True):
    manifest=json.loads((folder/'manifest.json').read_text())
    log=(folder/'forward.log').read_text()
    steps=list(map(int,re.findall(r'^Time Step (\d+),',log,re.M)))
    assert steps==list(range(130)) and 'Finished Executing' in log
    reports=[p for p in folder.glob('forward*data_*.csv') if '_signed_data_' not in p.name]
    latest=max(reports,key=lambda p:int(re.search(r'_(\d+)\.csv$',p.name)[1]))
    df=pd.read_csv(latest);canonical=pd.read_csv(SOURCE/'measurement_data.csv')
    np.testing.assert_allclose(df[KEYS],canonical[KEYS],rtol=1e-10,atol=1e-18)
    assert len(df)==18*130 and df.measurement_time.nunique()==130
    md=14376+(df.measurement_ycoord.to_numpy()-50)/.3048
    core=np.isclose(md[:,None],manifest['central_channel_depths_ft'],rtol=0,atol=1e-6).any(axis=1)
    w=np.where(core,3.,1.);use=df.measurement_time.to_numpy()>0
    assert w[use].sum()==26*129
    s=df.simulation_values.to_numpy().copy();d=df.measurement_values.to_numpy()
    assert np.isfinite([s,d]).all() and (s>=0).all() and (d>=0).all()
    ss=np.sum(w[use]*s[use]**2);sd=np.sum(w[use]*s[use]*d[use]);assert ss>0
    c=float(sd/ss);res=c*s-d
    assert abs(np.sum(w[use]*s[use]*res[use]))<1e-12*abs(sd)
    row=dict(background_alpha=-18,fitted_C=c,
        weighted_objective_after=float(.5*np.sum(w[use]*res[use]**2)),
        weighted_rms_after_microstrain=float(np.sqrt(np.sum(w[use]*res[use]**2)/w[use].sum())*1e6),
        unweighted_rms_after_microstrain=float(np.sqrt(np.mean(res[use]**2))*1e6),
        core_rms_after_microstrain=float(np.sqrt(np.mean(res[use & core]**2))*1e6))
    df['physical_simulation_values']=s;df['simulation_values']=c*s
    df['misfit_values']=np.where(use,res,0.);df['objective_weight']=w
    df.to_csv(folder/'bg_18_strain.csv',index=False)
    pressure=pressure_diagnostic(folder)
    summary=dict(method='Exact unbounded C=sum(w*s*d)/sum(w*s*s)',
        central_channel_depths_ft=manifest['central_channel_depths_ft'],core_weight=3.,
        coupling_bounds=None,geometry=manifest,results=[row],pressure_diagnostic=pressure,
        interpretation_limit='Geometry/mesh trial; pressure overshoot monitored, not silently corrected')
    base.save(folder/'summary.json',summary)
    if make_plot:
        with PLOT_LOCK:
            qc.main(out=folder,alphas=(-18,),geometry_note=f"Zone: {manifest['lo_ft']:.3f}-{manifest['hi_ft']:.3f} ft | local dy <= {manifest['h_y_max_m']:g} m")
    for p,expected in manifest['source_sha256'].items():assert digest(Path(p))==expected
    (folder/'COMPLETE').write_text('129 solves; immutable data; exact weighted C; QC complete\n')
    base.status(folder/'status.json','complete',**row)
    print(folder.name,json.dumps(row),flush=True)
    return summary


def pressure_diagnostic(folder):
    from scipy.io import netcdf_file
    text=(folder/'forward.i').read_text()
    block=re.search(r'^  \[injection_pressure_func\].*?^  \[\]',text,re.M|re.S)[0]
    sx=np.fromstring(re.search(r"^\s*x = '([^']*)'",block,re.M)[1],sep=' ')
    sy=np.fromstring(re.search(r"^\s*y = '([^']*)'",block,re.M)[1],sep=' ')
    with netcdf_file(folder/'pressure.e','r',mmap=False) as ds:
        names=[r.tobytes().decode().strip('\x00 ') for r in ds.variables['name_nod_var'][:]]
        times=np.asarray(ds.variables['time_whole'][:]);assert len(times)==130
        i=int(np.argmin(abs(times-315033)))
        maximum=float(ds.variables[f'vals_nod_var{names.index("pp")+1}'][i].max())
    source_max=max(float(sy[sx<=times[i]].max()),float(np.interp(times[i],sx,sy)))
    return dict(time_s=float(times[i]),max_pressure_MPa=maximum/1e6,source_history_max_MPa=source_max/1e6,
                overshoot_percent=100*(maximum/source_max-1))


def run_case(folder):
    if (folder/'COMPLETE').exists():return json.loads((folder/'summary.json').read_text())
    if (folder/'STARTED').exists():
        # Postprocess only a finished forward; never start a duplicate solve.
        return summarize(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    assert digest(folder/'forward.i')==manifest['input_sha256']
    env=os.environ.copy();env['PATH']=str(base.PYTHON_BIN)+':'+env['PATH']
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TBB_NUM_THREADS']:env[key]='1'
    cmd=[str(base.PYTHON_BIN/'mpiexec'),'-n','20',str(base.ROOT/'moose_env/moose/modules/combined/combined-opt'),'-i','forward.i','--n-threads=1']
    for checking in [True,False]:
        base.status(folder/'status.json','checking_input' if checking else 'running')
        if not checking:(folder/'STARTED').write_text(f'controller_pid={os.getpid()}\n')
        start=time.monotonic()
        with (folder/('check.log' if checking else 'forward.log')).open('w') as log:
            result=subprocess.run(cmd+(['--check-input'] if checking else []),cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            base.status(folder/'status.json','failed',returncode=result.returncode)
            raise RuntimeError(f'MOOSE failed in {folder}')
        if checking:(folder/'INPUT_CHECK_PASSED').write_text('Passed\n')
        else:base.status(folder/'status.json','forward_complete',wall_seconds=time.monotonic()-start)
    return summarize(folder)


def controls():
    prior=json.loads((PRIOR/'summary.json').read_text())['geometry']
    center=prior['new_center_depth_ft'];width=prior['new_width_ft']
    cases=[prepare_case('mesh_h050',center,width,.05),prepare_case('mesh_h025',center,width,.025)]
    base.status(OUT/'status.json','mesh_controls_running')
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run_case,cases))
    base.status(OUT/'status.json','mesh_controls_complete')
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['controls'])
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    controls()
