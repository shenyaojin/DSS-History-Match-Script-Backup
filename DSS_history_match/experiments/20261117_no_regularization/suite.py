"""Real MOOSE three-Kxx experiments and isolated zero-penalty legacy controls.

All meshes/time sequences and physical laws are inherited from saved inputs.
No source run is edited. Commands: prepare, run-core, run-legacy, status.
Failures stop a case; they never return a fake objective or zero gradient.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
PROJECT=ROOT/'scripts/DSS_history_match'
OLD=PROJECT/'optimizer_input_file_test'
TEMPLATE=OLD/'perm5layer_100_v2strain/fwd/output_gt/casing_model_test.i'
EXE=ROOT/'moose_env/moose/modules/combined/combined-opt'
MPI=Path('/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec')
RANKS=10


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def status(path,stage,**extra):
    save(path,dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**extra))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def section(text,name):
    match=re.search(r'^\['+re.escape(name)+r'\].*?^\[\]',text,re.M|re.S)
    if match is None:raise ValueError('Missing MOOSE section '+name)
    return match[0]


def replace(text,name,value):return text.replace(section(text,name),value)


def children(text,name):
    return [(m[1],m[0]) for m in re.finditer(r'^  \[([^\]]+)\].*?^  \[\]',section(text,name),re.M|re.S)]


def setvalue(text,name,value):
    result,count=re.subn(r'(^\s*'+re.escape(name)+r'\s*=).*$',lambda m:m[1]+' '+value,text,count=1,flags=re.M)
    if count!=1:raise ValueError('Cannot set '+name)
    return result


def three_layer_input(theta,shift=0.):
    """Three physical material blocks; only their Kxx components vary."""
    original=TEMPLATE.read_text()
    lo,hi=-3.+shift,3.+shift
    mesh='''[Mesh]
  [base_mesh]
    type = GeneratedMeshGenerator
    dim = 2
    nx = 200
    ny = 200
    xmin = 0
    xmax = 100
    ymin = -50
    ymax = 50
  []
'''
    previous='base_mesh'
    for i,(name,a,b) in enumerate([('lower',-50.,lo),('fracture',lo,hi),('upper',hi,50.)],1):
        mesh+=f'''  [{name}_bbox]
    type = SubdomainBoundingBoxGenerator
    input = '{previous}'
    block_id = {i}
    bottom_left = '0 {a:.16g} 0'
    top_right = '100 {b:.16g} 0'
  []
''';previous=name+'_bbox'
    mesh+=f'''  [injection_well_nodes]
    type = BoundingBoxNodeSetGenerator
    input = '{previous}'
    new_boundary = 'injection_well'
    bottom_left = '45 -50 0'
    top_right = '45 50 0'
  []
  [names]
    type = RenameBlockGenerator
    input = 'injection_well_nodes'
    old_block = '1 2 3'
    new_block = 'lower fracture upper'
  []
[]'''
    material_blocks=dict(children(original,'Materials'))
    mats='[Materials]\n'
    for name,alpha in zip(['lower','fracture','upper'],theta):
        for role in ['porosity','permeability','elasticity_tensor']:
            block=material_blocks[role+'_caprock'].replace('caprock',name)
            if role=='permeability':
                block=setvalue(block,'permeability',f"'{10.**float(alpha):.16e} 0 0 0 1e-18 0 0 0 1e-18'")
            mats+=block+'\n'
    for name,block in material_blocks.items():
        if not name.startswith(('porosity_','permeability_','elasticity_tensor_')):
            mats+=block.replace('caprock low_perm_srv caprock2 high_perm_srv caprock3','lower fracture upper')+'\n'
    mats+='[]'
    text=replace(replace(original,'Mesh',mesh),'Materials',mats)
    text=replace(text,'VectorPostprocessors','''[VectorPostprocessors]
  [observation_strain]
    type = LineValueSampler
    variable = 'strain_yy'
    sort_by = 'y'
    start_point = '60 -25 0'
    end_point = '60 25 0'
    num_points = 500
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]''')
    text=replace(text,'Outputs',"[Outputs]\n  file_base = forward\n  console = true\n  [csv]\n    type = CSV\n  []\n[]")
    # Preserve the original physics, source history, integration and tolerances.
    for name in ['Kernels','BCs','Functions','Executioner','Preconditioning','Variables','FluidProperties']:
        assert section(text,name)==section(original,name),name
    assert len(re.findall(r'type = PorousFlowPermeabilityConst\n',text))==3
    return text


def legacy_input(source,alpha):
    """Forward-only form of the historical two-zone input; physics untouched."""
    original=source.read_text();text=original
    block=dict(children(text,'Reporters'))['params']
    block=setvalue(block,'real_vector_values',"'"+'; '.join(f'{v:.16g}' for v in alpha)+"'")
    text=replace(text,'Reporters','[Reporters]\n'+block+'\n[]')
    for name in ['VectorPostprocessors','DiracKernels']:
        if re.search(r'^\['+name+r'\]',text,re.M):text=text.replace(section(text,name),'')
    ex=section(text,'Executioner').replace("type = 'TransientAndAdjoint'","type = 'Transient'\n  solve_type = NEWTON\n  scheme = implicit-euler")
    ex=ex.replace("forward_system = 'nl0'","system_names = 'nl0'").replace("  adjoint_system = 'adjoint'\n",'')
    text=replace(text,'Executioner',ex)
    text+='\n'+'''[VectorPostprocessors]
  [observation_strain]
    type = LineValueSampler
    variable = 'strain_yy'
    sort_by = 'y'
    start_point = '60 -25 0'
    end_point = '60 25 0'
    num_points = 500
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]
'''
    text=replace(text,'Outputs',"[Outputs]\n  file_base = forward\n  console = true\n  [csv]\n    type = CSV\n  []\n[]")
    for name in ['Mesh','Kernels','Materials','BCs','Variables','Preconditioning','FluidProperties']:
        assert section(text,name)==section(original,name),name
    return text


def execute(folder,text):
    """Atomic cached evaluation; completion requires a finished solver and all samples."""
    folder.mkdir(parents=True,exist_ok=True)
    packed=folder/'prediction.npz'
    if packed.exists() and (folder/'COMPLETE').exists():
        assert (folder/'forward.i').read_text()==text,'Cached input mismatch'
        with np.load(packed) as f:return f['time'],f['y'],f['strain']
    if (folder/'forward.i').exists() and (folder/'forward.log').exists():
        assert (folder/'forward.i').read_text()==text,'Existing input differs; refusing overwrite'
    (folder/'forward.i').write_text(text)
    env=os.environ.copy();env['PATH']=str(MPI.parent)+':'+env.get('PATH','')
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TBB_NUM_THREADS']:env[key]='1'
    cmd=[str(MPI),'-n',str(RANKS),str(EXE),'-i','forward.i','--n-threads=1']
    start=time.monotonic()
    finished=(folder/'forward.log').exists() and 'Finished Executing' in (folder/'forward.log').read_text()
    if not finished:
        status(folder/'status.json','input_check',command=cmd)
        with (folder/'check.log').open('w') as log:
            check=subprocess.run(cmd+['--check-input'],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
        if check.returncode:raise RuntimeError('Input check failed: '+str(folder/'check.log'))
        status(folder/'status.json','running',command=cmd,input_sha256=digest(folder/'forward.i'))
        with (folder/'forward.log').open('w') as log:
            result=subprocess.run(cmd,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status(folder/'status.json','failed',returncode=result.returncode)
            raise RuntimeError('MOOSE failed: '+str(folder/'forward.log'))
    log=(folder/'forward.log').read_text()
    if 'Finished Executing' not in log:raise RuntimeError('Solver did not report completion')
    expected=np.fromstring(re.search(r"time_sequence = '([^']+)'",text)[1],sep=' ')
    if len(expected)!=130:raise RuntimeError('Unexpected time sequence')
    files=sorted(folder.glob('forward_observation_strain_*.csv'))
    if len(files)!=len(expected):raise RuntimeError(f'Incomplete observations: {len(files)} of {len(expected)}')
    vectors=[];ys=None
    for i,path in enumerate(files):
        if int(path.stem.rsplit('_',1)[1])!=i:raise RuntimeError('Missing time-index output')
        data=pd.read_csv(path)
        if data.empty and i==0:
            vectors.append(np.zeros(500));continue
        if len(data)!=500 or 'strain_yy' not in data:raise RuntimeError('Malformed strain output '+str(path))
        if ys is None:ys=data['y'].to_numpy()
        else:np.testing.assert_allclose(ys,data['y'],rtol=0,atol=1e-10)
        vectors.append(data.strain_yy.to_numpy())
    values=np.array(vectors)
    if not np.all(np.isfinite(values)):raise RuntimeError('Nonfinite physical prediction')
    np.savez_compressed(packed,time=expected,y=ys,strain=values)
    status(folder/'status.json','complete',wall_seconds=time.monotonic()-start,shape=list(values.shape))
    (folder/'COMPLETE').write_text('130 native strain profiles verified\n')
    return expected,ys,values


def prepare():
    OUT.joinpath('runs').mkdir(exist_ok=True)
    core=[]
    for shift in [0.,1.,2.]:
        for beta in [0.,2e-11]:
            name=('clean' if shift==0 else f'shift_{shift:g}m')+('_no_reg' if beta==0 else '_L1')
            cfg=dict(name=name,kind='three_layer',shift_m=shift,beta=beta,initial=[-17.,-16.,-17.],
                truth=[-18.,float(np.log10(3e-15)),-18.],bounds=[[-21.,-12.]]*3,
                parameter_names=['log10_Kxx_lower','log10_Kxx_fracture','log10_Kxx_upper'],
                Kyy_m2=1e-18,Kzz_m2=1e-18,truth_band_m=[-3,3],assumed_band_m=[-3+shift,3+shift],
                mesh=[200,200],n_times=130,n_receivers=500,scale=1e6,
                ftol=1e-7,gtol=1e-8,maxiter=60,fd_step=.02,delta=.05,reference=-18.,
                regularization_definition='beta sum over exactly three parameters sqrt((theta-reference)^2+delta^2)-delta',
                observation='same clean MOOSE truth for all shifted-geometry cases; no C, no rectification, no free layers')
            core.append(cfg)
    # Noise replacement uses archived realizations, rescaled to the new signal anchor.
    for level in ['1','2','5','10']:
        core.append(dict(core[0],name=f'rfsdss_{level}pct_no_reg',noise_case='rfsdss_'+level+'pct'))
    for cfg in core:
        path=OUT/'runs/three_layer'/cfg['name']/'config.json'
        if not path.exists():save(path,cfg)
    legacy=[]
    for family in ['perm5layer_100_v5strain_zonal2p','perm5layer_100_v6strain_zonalgrid']:
        for result in sorted((OLD/family/'inv').rglob('optimized_theta_zonal.txt')):
            source=result.parent;name=source.name
            shift=0
            if name.startswith('pert_'):shift=2
            elif re.search(r'_s(\d+)$',name):shift=int(re.search(r'_s(\d+)$',name)[1])
            cfg=dict(name=name,kind='legacy_two_zone',source=str(source),beta=0.,shift_layers=shift,
                initial=[-16.,-16.],bounds=[[-25.,-10.]]*2,scale=1e6,ftol=1e-7,gtol=1e-10,maxiter=60,
                fd_step=.02,delta=.05,reference=-18.,
                comparison_limit='Same physical forward/data/initial/bounds; finite-difference derivatives replace unavailable historical custom adjoint. Method change is disclosed.')
            path=OUT/'runs/legacy_two_zone'/family/name/'config.json'
            if not path.exists():save(path,cfg)
            legacy.append(cfg)
    manifest=dict(prepared_utc=datetime.now(timezone.utc).isoformat(),core_cases=[c['name'] for c in core],
        legacy_cases=[c['source'] for c in legacy],ranks_per_forward=RANKS,max_parallel_forwards=2,
        executable_sha256=digest(EXE),template_sha256=digest(TEMPLATE),
        status='prepared; no completion implied',source_template=str(TEMPLATE),
        exclusions='No new layer-by-layer inversion; retained material grid is a forward discretization only.')
    save(OUT/'suite_manifest.json',manifest)
    # A concrete preflight input is available without running an inversion.
    preview=OUT/'preflight';preview.mkdir(exist_ok=True)
    (preview/'forward.i').write_text(three_layer_input(core[0]['truth']))
    print(json.dumps(dict(core=len(core),legacy=len(legacy),preflight=str(preview))))


def load_observation(cfg):
    if cfg['kind']=='legacy_two_zone':
        d=pd.read_csv(Path(cfg['source'])/'measurement_data.csv')
        pivot=d.pivot(index='measurement_time',columns='measurement_ycoord',values='measurement_values').sort_index().sort_index(axis=1)
        return pivot.index.to_numpy(),pivot.columns.to_numpy(),pivot.to_numpy()
    truth=OUT/'shared_truth'
    t,y,s=execute(truth,three_layer_input(cfg['truth']))
    if 'noise_case' in cfg:
        noise=OLD/'perm5layer_100_v5strain_zonal2p/noise_data'
        clean=pd.read_csv(noise/'measurement_data_clean.csv')
        noisy=pd.read_csv(noise/f"measurement_data_{cfg['noise_case']}.csv")
        keys=['measurement_time','measurement_ycoord','measurement_xcoord','measurement_zcoord']
        np.testing.assert_array_equal(clean[keys],noisy[keys])
        clean_pivot=clean.pivot(index='measurement_time',columns='measurement_ycoord',values='measurement_values').sort_index().sort_index(axis=1)
        noisy_pivot=noisy.pivot(index='measurement_time',columns='measurement_ycoord',values='measurement_values').sort_index().sort_index(axis=1)
        np.testing.assert_allclose(t,clean_pivot.index,rtol=0,atol=1e-8)
        np.testing.assert_allclose(y,clean_pivot.columns,rtol=0,atol=1e-8)
        old=clean_pivot.to_numpy();noise_values=noisy_pivot.to_numpy()-old
        factor=float(np.median(np.max(abs(s[1:]),axis=0))/np.median(np.max(abs(old[1:]),axis=0)))
        s=s+factor*noise_values
    return t,y,s


class Experiment:
    def __init__(self,path):
        self.path=path;self.cfg=json.loads((path/'config.json').read_text())
        self.t,self.y,self.obs=load_observation(self.cfg)
        self.cache={};self.gradients={};self.calls=0;self.accepted=0
        np.savez_compressed(path/'observations.npz',time=self.t,y=self.y,strain=self.obs)

    def prediction(self,theta):
        cfg=self.cfg
        if cfg['kind']=='three_layer':return three_layer_input(theta,cfg['shift_m'])
        alpha=np.full(200,-18.);yb=-50+np.arange(200)*.5;yt=yb+.5
        shift=cfg['shift_layers']*.5
        alpha[(yb>=14+shift)&(yt<=20+shift)]=theta[0]
        alpha[(yb>=-20+shift)&(yt<=-16+shift)]=theta[1]
        return legacy_input(Path(cfg['source'])/'forward_and_adjoint.i',alpha)

    def get(self,theta):
        key=tuple(float(v) for v in theta)
        if key in self.cache:return self.cache[key]
        name=hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
        folder=self.path/'evaluations'/name
        status(self.path/'status.json','forward',theta=list(key),accepted=self.accepted,evaluation=str(folder))
        t,y,s=execute(folder,self.prediction(theta))
        np.testing.assert_allclose(t,self.t,rtol=0,atol=1e-8)
        np.testing.assert_allclose(y,self.y,rtol=0,atol=1e-8)
        residual=s[1:]-self.obs[1:]
        objective=float(.5*np.sum(residual*residual))
        d=np.asarray(theta)-self.cfg['reference'];den=np.sqrt(d*d+self.cfg['delta']**2)
        penalty=float(self.cfg['beta']*np.sum(den-self.cfg['delta']))
        reg_grad=self.cfg['beta']*d/den
        value=dict(data=objective,reg=penalty,value=(objective+penalty)*self.cfg['scale'],
            reg_gradient=reg_grad*self.cfg['scale'],rms_microstrain=float(np.sqrt(np.mean(residual**2))*1e6),folder=str(folder))
        save(folder/'objective.json',{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in value.items()})
        self.cache[key]=value;return value

    def gradient(self,theta,h):
        stencils=[];points={}
        for j,(lo,hi) in enumerate(self.cfg['bounds']):
            if theta[j]-h<lo:offsets=[0,h,2*h];weights=np.array([-3.,4.,-1.])/(2*h)
            elif theta[j]+h>hi:offsets=[0,-h,-2*h];weights=np.array([3.,-4.,1.])/(2*h)
            else:offsets=[-h,h];weights=np.array([-1.,1.])/(2*h)
            keys=[]
            for off in offsets:
                p=np.array(theta,dtype=float);p[j]+=off;key=tuple(float(v) for v in p)
                points[key]=p;keys.append(key)
            stencils.append((keys,weights))
        missing=[p for key,p in points.items() if key not in self.cache]
        if missing:
            with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(self.get,missing))
        return np.array([np.dot(weights,[self.cache[key]['data'] for key in keys])*self.cfg['scale'] for keys,weights in stencils])

    def evaluate(self,theta):
        key=tuple(float(v) for v in theta);value=self.get(theta)
        if key not in self.gradients:self.gradients[key]=self.gradient(theta,self.cfg['fd_step'])+value['reg_gradient']
        self.calls+=1
        with (self.path/'evaluations.jsonl').open('a') as f:
            f.write(json.dumps(dict(theta=list(key),data=value['data'],regularization=value['reg'],scaled_objective=value['value'],gradient=self.gradients[key].tolist()))+'\n')
        return value['value'],self.gradients[key]

    def record(self,theta):
        self.accepted+=1;value=self.get(theta)
        save(self.path/'checkpoint.json',dict(iteration=self.accepted,theta=list(theta),data=value['data'],regularization=value['reg']))
        print(self.path.name,'ACCEPTED',self.accepted,list(theta),value['data'],flush=True)

    def run(self):
        x0=np.array(self.cfg['initial'])
        initial=self.get(x0)
        if self.cfg['kind']=='legacy_two_zone':
            hist=pd.read_csv(Path(self.cfg['source'])/'objective_history_zonal.csv')
            old=float(hist.obj_raw.iloc[0]);diff=abs(initial['data']-old)/max(abs(old),1e-30)
            save(self.path/'historical_forward_check.json',dict(old_initial_data_objective=old,new_initial_data_objective=initial['data'],relative_difference=diff,passed=diff<=.01))
            if diff>.01:raise RuntimeError('Historical forward reproduction differs >1%; comparison not started')
        h=self.cfg['fd_step'];a=self.gradient(x0,h);b=self.gradient(x0,h/2)
        checks=[dict(parameter=j,h=h,half_h=h/2,derivative_h=float(a[j]),derivative_half_h=float(b[j]),passed=bool(abs(a[j]-b[j])<=max(1e-7,.05*max(abs(a[j]),abs(b[j]))))) for j in range(len(x0))]
        save(self.path/'gradient_check.json',checks)
        if not all(v['passed'] for v in checks):raise RuntimeError('FD step-consistency check failed')
        self.gradients[tuple(x0)]=a+initial['reg_gradient']
        result=minimize(self.evaluate,x0,method='L-BFGS-B',jac=True,bounds=self.cfg['bounds'],callback=self.record,
            options=dict(maxiter=self.cfg['maxiter'],ftol=self.cfg['ftol'],gtol=self.cfg['gtol'],maxls=20))
        value=self.get(result.x)
        record=dict(success=bool(result.success),termination=str(result.message),iterations=int(result.nit),
            n_function_evaluations=int(result.nfev),theta_log10_Kxx=result.x.tolist(),Kxx_m2=(10.**result.x).tolist(),
            data_objective=value['data'],regularization=value['reg'],scaled_objective=value['value'],
            rms_microstrain=value['rms_microstrain'],gradient_scaled=result.jac.tolist(),final_forward=value['folder'],
            beta=self.cfg['beta'],n_optimized=len(result.x),bound_hits=[bool(abs(v-l)<1e-7 or abs(v-u)<1e-7) for v,(l,u) in zip(result.x,self.cfg['bounds'])])
        save(self.path/'result.json',record)
        if not result.success:raise RuntimeError('Optimizer did not converge: '+str(result.message))
        shutil.copy2(Path(value['folder'])/'prediction.npz',self.path/'prediction_final.npz')
        (self.path/'COMPLETE').write_text(str(result.message)+'\n')
        status(self.path/'status.json','complete',**record)


def run_group(group,only=None):
    paths=sorted((OUT/'runs'/group).rglob('config.json'))
    if only:paths=[p for p in paths if p.parent.name in only]
    if group=='three_layer':paths.sort(key=lambda p:(p.parent.name!='clean_no_reg',p.parent.name!='shift_1m_no_reg',p.parent.name!='shift_2m_no_reg',p.parent.name))
    failures=[]
    for config in paths:
        path=config.parent
        if (path/'COMPLETE').exists():continue
        try:
            Experiment(path).run()
        except Exception as exc:
            failures.append(dict(case=str(path),error=str(exc)))
            status(path/'status.json','failed',reason=f'{type(exc).__name__}: {exc}')
            # Do not burn an entire queue on an infrastructure or solver failure.
            status(OUT/'batch_status.json','needs_attention',group=group,failures=failures)
            raise
        status(OUT/'batch_status.json','running',group=group,last_completed=str(path))
    status(OUT/'batch_status.json','complete',group=group)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run-core','run-legacy','status'])
    parser.add_argument('--only',nargs='+',help='Optional explicit case names; existing completed cases are skipped')
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command.startswith('run-'):run_group('three_layer' if args.command=='run-core' else 'legacy_two_zone',args.only)
    else:
        for path in sorted((OUT/'runs').rglob('config.json')):
            record=path.parent/'status.json'
            print(path.parent.relative_to(OUT),json.loads(record.read_text()).get('stage') if record.exists() else 'prepared')


if __name__=='__main__':main()
