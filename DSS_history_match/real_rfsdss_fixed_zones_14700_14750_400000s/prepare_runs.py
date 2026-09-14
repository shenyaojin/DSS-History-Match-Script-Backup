from pathlib import Path
import json, re, shutil
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
SOURCE=REPO/'figs/rfsdss_14700_14750_0_40000s/initial_guess_comparison'
TEMPLATE=HERE.parent/'optimizer_input_file_test/perm5layer_100_v6strain_zonalgrid/inv/_template'
proposal=json.loads((SOURCE/'initial_guess.json').read_text())
zones=proposal['zones']
# Preserve the 100 m model span and 200 layers, move nearest mesh edges to
# the ten fixed physical zone boundaries, without moving any selected zone.
edges=np.linspace(0,100,201)
for z in zones:
    for key in ['lo_ft','hi_ft']:
        y=50+(z[key]-14725)*.3048
        edges[np.argmin(abs(edges-y))]=y
assert np.all(np.diff(edges)>0)
centers=(edges[:-1]+edges[1:])/2
masks=[]
for z in zones:
    lo=50+(z['lo_ft']-14725)*.3048
    hi=50+(z['hi_ft']-14725)*.3048
    masks.append(np.flatnonzero((centers>lo)&(centers<hi)).tolist())
assert all(masks) and sum(map(len,masks))==len(set(sum(masks,[])))
# Re-extract the corrected 400,000 s window directly from read-only DSS.
DSS_SOURCE=REPO/'data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz'
with np.load(DSS_SOURCE,allow_pickle=True) as f:
    depth=f['daxis'];full_times=f['taxis'];start=f['start_time'].item()
    selected_time=(full_times>=0)&(full_times<=400000)
    times=full_times[selected_time]
    raw=f['data']
    selected_depth=(depth>=14700)&(depth<=14750)
    md=depth[selected_depth]
    obs=raw[selected_depth][:,selected_time].copy()
    reference=(depth>7500)&(depth<15000)
    indices=np.flatnonzero(selected_time)
    for first in range(0,len(indices),100):
        idx=indices[first:first+100]
        obs[:,first:first+len(idx)]-=np.median(raw[np.ix_(reference,idx)],axis=0)[None,:]
    del raw
obs*=1e-6
(HERE/'data').mkdir(exist_ok=True)
np.savez_compressed(HERE/'data/unrectified_full.npz',data=obs,daxis=md,taxis=times,start_time=start,units='strain')
np.savez_compressed(HERE/'data/half_wave_positive_full.npz',data=np.maximum(obs,0),daxis=md,taxis=times,start_time=start,units='strain')
with np.load(REPO/'data/fiberis_format/prod/gauges/pressure_g1.npz',allow_pickle=True) as f:
    gt=f['taxis']+(f['start_time'].item()-start).total_seconds(); gp=f['data']*6894.76
assert gt.min()<=0 and gt.max()>=times[-1] and np.all(np.diff(gt)>0)
# Retain every native gauge sample in this window, interpolate only endpoints.
p_times=np.r_[0,gt[(gt>0)&(gt<times[-1])],times[-1]]
p_values=np.interp(p_times,gt,gp)-np.interp(0,gt,gp)
np.savetxt(HERE/'pressure_boundary.csv',np.c_[p_times,p_values],delimiter=',',header='time_s,delta_pressure_Pa',comments='')
np.savetxt(HERE/'mesh_layers.csv',np.c_[np.arange(1,201),edges[:-1],edges[1:],14725+(centers-50)/.3048],delimiter=',',header='layer,y_bottom_m,y_top_m,md_center_ft',comments='')
y=50+(md.astype(float)-14725)*.3048
layer=np.searchsorted(edges,y,side='right')-1
assert np.all((layer>=0)&(layer<200))
# Receiver separation follows the existing real-data baseline's 80 ft default.
receiver_x=45+80*.3048
x_center=(np.floor(receiver_x/.5)+.5)*.5
sample_geom=[np.tile(edges[layer+1],len(times)),np.tile(edges[layer],len(times)),
             np.tile(1/np.diff(edges)[layer],len(times)),np.full(obs.size,x_center)]
fmt=lambda arr:' '.join(f'{v:.16g}' for v in arr)
for name in ['unrectified','half_wave_positive']:
    run=HERE/name
    run.mkdir(exist_ok=True)
    if (run/'objective_history.csv').exists():raise RuntimeError(f'Refusing to overwrite an existing run: {run}')
    data=obs if name=='unrectified' else np.maximum(obs,0)
    cfg=dict(zones=zones,zone_layer_indices=masks,initial_theta=[z['alpha'] for z in zones],
             background_alpha=-18.,theta_bounds=[-18.,-12.],n_parameters=5,
             start_time=str(start),end_time=float(times[-1]),requested_end_time=400000.,
             md_center_ft=14725.,y_center_m=50.,receiver_x_m=receiver_x,injection_x_m=45.,
             receiver_separation_ft=80.,receiver_geometry_source='baseline_model_generator.py default, not a surveyed distance',
             n_time=len(times),n_channels=len(md),rectified=name=='half_wave_positive',
             units='strain',measurement_scale=1e-6,extra_calibration_factor=1.,
             first_frame='retained as supplied',pressure='prod pressure_g1; aligned by timestamps; delta from DSS t=0',
             mesh_edges_m=edges.tolist(),fixed_positions=True)
    # Preserve relative L1 strength from synthetic template, scaled by data energy.
    clean=pd.read_csv(TEMPLATE.parent.parent/'data/obs_strain_yy.csv')['measurement_values'].to_numpy()
    cfg['beta_l1']=float(2e-11*np.sum(obs**2)/np.sum(clean**2))
    cfg['objective_scale']=float(1/max(np.sum(obs**2),1e-30))
    (run/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    np.savez_compressed(run/'observations.npz',data=data,daxis=md,taxis=times,start_time=start,units='strain')
    df=pd.DataFrame(dict(measurement_time=np.repeat(times,len(md)),measurement_values=data.T.ravel(),
                         measurement_xcoord=receiver_x,measurement_ycoord=np.tile(y,len(times)),measurement_zcoord=0.,
                         misfit_values=0.,simulation_values=0.))
    df.to_csv(run/'measurement_data.csv',index=False)
    fwd=(TEMPLATE/'forward_and_adjoint.i').read_text()
    base="  [base_mesh]\n    type = CartesianMeshGenerator\n    dim = 2\n    dx = '100'\n    ix = '200'\n    dy = '"+fmt(np.diff(edges))+"'\n  []"
    fwd,n=re.subn(r'  \[base_mesh\].*?  \[\]',lambda m:base,fwd,count=1,flags=re.S);assert n==1
    for i in range(200):
        pattern=rf'(  \[layer_{i+1}_bbox\].*?bottom_left = )[^\n]+(\n    top_right = )[^\n]+'
        fwd,n=re.subn(pattern,lambda m:f"{m[1]}'0 {edges[i]:.16g} 0'{m[2]}'100 {edges[i+1]:.16g} 0'",fwd,count=1,flags=re.S);assert n==1
    # Only the two injection node-set boxes still reference the old y limits.
    fwd=fwd.replace("'44.999999 -50.0 0'","'44.999999 0 0'").replace("'45.000001 50.0 0'","'45.000001 100 0'")
    fwd=fwd.replace("'45.0 -50.0 0'","'45.0 0 0'").replace("'45.0 50.0 0'","'45.0 100 0'")
    func="  [injection_pressure_func]\n    type = PiecewiseLinear\n    x = '"+fmt(p_times)+"'\n    y = '"+fmt(p_values)+"'\n  []"
    fwd,n=re.subn(r'  \[injection_pressure_func\].*?  \[\]',lambda m:func,fwd,count=1,flags=re.S);assert n==1
    fwd=re.sub(r'  end_time = [^\n]+',f'  end_time = {times[-1]:.16g}',fwd)
    fwd=re.sub(r"    time_sequence = '[^']*'",lambda m:"    time_sequence = '"+fmt(times)+"'",fwd)
    # RankTwoAux strain_yy is element-constant. Its exact Q1 rectangular-element
    # derivative is a pair of midpoint loads on the upper and lower edges.
    # This replaces the synthetic uniform-grid +/-0.25 m approximation.
    fwd=fwd.replace("'data/measurement_xcoord'","'sampling_geometry/x_center'")
    fwd=fwd.replace("'y_up/values'","'sampling_geometry/y_up'").replace("'y_down/values'","'sampling_geometry/y_down'")
    for block in ['y_up','y_down']:
        fwd=re.sub(rf'  \[{block}\].*?  \[\]\n','',fwd,count=1,flags=re.S)
    for block,sign in [('misfit_up',''),('misfit_down','-')]:
        names='data/misfit_values sampling_geometry/inv_dy'
        symbols='m ih';expr=f'{sign}m * ih'
        if cfg['rectified']:
            names+=' data/simulation_values'; symbols+=' s';expr+=' * if(s>0,1,0)'
        text=f"  [{block}]\n    type = ParsedVectorReporter\n    name = values\n    vector_reporter_names = '{names}'\n    vector_reporter_symbols = '{symbols}'\n    expression = '{expr}'\n    outputs = 'none'\n  []"
        fwd,n=re.subn(rf'  \[{block}\].*?  \[\]',lambda m:text,fwd,count=1,flags=re.S);assert n==1
    geometry="  [sampling_geometry]\n    type = ConstantReporter\n    real_vector_names = 'y_up y_down inv_dy x_center'\n    real_vector_values = '"+'; '.join(fmt(a) for a in sample_geom)+"'\n    outputs = 'none'\n  []\n"
    fwd=fwd.replace('[Reporters]\n','[Reporters]\n'+geometry)
    if cfg['rectified']:
        fwd=fwd.replace('[AuxVariables]\n',"[AuxVariables]\n  [strain_positive]\n    order = CONSTANT\n    family = MONOMIAL\n  []\n")
        fwd=fwd.replace('[AuxKernels]\n',"[AuxKernels]\n  [strain_positive]\n    type = ParsedAux\n    variable = strain_positive\n    coupled_variables = strain_yy\n    expression = 'max(strain_yy,0)'\n    execute_on = 'INITIAL TIMESTEP_END'\n  []\n")
        fwd=fwd.replace("    variable = 'strain_yy'\n    outputs = 'none'","    variable = 'strain_positive'\n    outputs = 'none'")
    # Keep observation output for final/initial QC, drop large Exodus dumps.
    fwd=fwd.replace("    outputs = 'none'\n  []\n  # --- Dipole", "    outputs = 'csv'\n  []\n  # --- Dipole")
    fwd=re.sub(r'  \[exodus\]\s*type = Exodus\s*\[\]\s*','',fwd,count=1)
    fwd=fwd.replace('verbose = true','verbose = false').replace('console = true','console = false')
    # Materially tighter solves for gradient validation at real strain amplitudes.
    fwd=fwd.replace('nl_abs_tol = 0.001','nl_abs_tol = 1e-10').replace('nl_rel_tol = 0.001','nl_rel_tol = 1e-8')
    fwd,n=re.subn(r'(?m)^\s*\[csv\]\s*\n\s*type = CSV', '  [csv]\n    type = CSV\n    execute_on = FINAL\n    precision = 16',fwd,count=1);assert n==1
    (run/'forward_and_adjoint.i').write_text(fwd)
    master=(TEMPLATE/'optimize.i').read_text()
    alpha=np.full(200,-18.)
    for z,indices in zip(zones,masks):alpha[indices]=z['alpha']
    master=re.sub(r"(initial_condition\s*=\s*')[^']*(')",lambda m:m[1]+'; '.join(map(str,alpha))+m[2],master,count=1)
    (run/'optimize.i').write_text(master)
    np.savetxt(run/'initial_alpha.txt',alpha)
    print(name,'samples',len(df),'parameters',len(masks),'zone layer counts',list(map(len,masks)))
print('Prepared fixed-position runs; no solver started.')
