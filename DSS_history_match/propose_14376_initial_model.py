"""Review-only preparation for the selected 14376 ft example; no MOOSE calls."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'real_rfsdss_fixed_zones_14373_14379_400000s_bg21'
REFERENCE = HERE/'real_rfsdss_fixed_zones_14710_14725_400000s_bg21/comparison_xx_yy_60/fixed_yy'
SOURCE = ROOT/'data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz'


def main():
    if OUT.exists():
        raise RuntimeError('Proposal directory exists; refusing to overwrite')
    data = DSS2D()
    data.load_npz(str(SOURCE))
    ref = (data.daxis > 7500) & (data.daxis < 15000)
    data.data -= np.median(data.data[ref],axis=0)[None,:]
    data.select_time(0,400000)
    data.select_depth(14373,14379)
    md, t, signed = data.daxis.astype(float), data.taxis, data.data.astype(float)*1e-6
    assert signed.shape == (18,3318)
    core = np.where((md >= 14375.5) & (md <= 14376.7))[0]
    assert len(core) == 4 and np.all(np.diff(core) == 1)
    md_edges = np.r_[md[0]-(md[1]-md[0])/2, (md[:-1]+md[1:])/2,
                     md[-1]+(md[-1]-md[-2])/2]
    zone_lo, zone_hi = md_edges[core[0]], md_edges[core[-1]+1]
    yy = 50+(md-14376)*.3048
    local_edges = 50+(md_edges-14376)*.3048
    # Reuse the layered representation: 200 material blocks, one shared zone
    # parameter across its two constituent blocks, four native-scale elements.
    material_edges = np.linspace(0,100,201)
    material_edges[99] = local_edges[core[0]]
    material_edges[100] = local_edges[core[0]+2]
    material_edges[101] = local_edges[core[-1]+1]
    element_edges = np.unique(np.r_[material_edges[(material_edges < local_edges[0]) |
                                                   (material_edges > local_edges[-1])],local_edges])
    assert np.all(np.diff(material_edges)>0) and np.all(np.isin(material_edges,element_edges))
    cells = np.searchsorted(element_edges,yy,side='right')-1
    assert len(np.unique(cells)) == 18
    np.testing.assert_allclose(element_edges[cells],local_edges[:-1],rtol=0,atol=1e-12)
    positive = np.maximum(signed,0)
    solver_times = np.loadtxt(REFERENCE/'solver_times.csv',skiprows=1)
    assert len(solver_times)==130 and solver_times[-1]==t[-1]
    sampled = np.array([np.interp(solver_times,t,row) for row in positive])
    baseline = json.loads((REFERENCE/'config.json').read_text())
    gauge_path = ROOT/'data/fiberis_format/prod/gauges/pressure_g1.npz'
    gauge = np.load(gauge_path,allow_pickle=True)
    gt = gauge['taxis']+(gauge['start_time'].item()-data.start_time).total_seconds()
    gp = gauge['data']*.00689476
    gp -= np.interp(0,gt,gp)
    gauge_mask = (gt>=0)&(gt<=t[-1])
    OUT.mkdir(); (OUT/'data').mkdir()
    for name,values in [('unrectified_native',signed),('half_wave_positive_native',positive)]:
        np.savez_compressed(OUT/'data'/f'{name}.npz',data=values,daxis=md,taxis=t,
                            start_time=data.start_time,units='strain')
    np.savez_compressed(OUT/'data/half_wave_positive_solver.npz',data=sampled,daxis=md,
                        taxis=solver_times,start_time=data.start_time,units='strain')
    np.savetxt(OUT/'data/solver_times.csv',solver_times,header='time_s',comments='')
    pd.DataFrame(dict(measurement_time=np.repeat(solver_times,len(md)),
        measurement_values=sampled.T.ravel(),measurement_xcoord=69.384,
        measurement_ycoord=np.tile(yy,len(solver_times)),measurement_zcoord=0.,
        simulation_values=0.,misfit_values=0.)).to_csv(OUT/'data/measurement_data.csv',index=False)
    profile_window = (t>=140000)&(t<=320000)
    table = pd.DataFrame(dict(md_ft=md,core_zone=np.isin(np.arange(len(md)),core),
        median_signed_140_320ks_microstrain=np.median(signed[:,profile_window],axis=1)*1e6,
        median_positive_140_320ks_microstrain=np.median(positive[:,profile_window],axis=1)*1e6,
        native_peak_microstrain=signed.max(axis=1)*1e6,
        element_lower_ft=md_edges[:-1],element_upper_ft=md_edges[1:],
        element_height_m=np.diff(local_edges)))
    table.to_csv(OUT/'channel_profile_and_mesh.csv',index=False)
    proposal = dict(status='PROPOSAL_ONLY; user QC before any forward solve or inversion',
        source=str(SOURCE),pressure_source=str(gauge_path),start_time=str(data.start_time),
        requested_depth_ft=[14373,14379],actual_channel_depth_ft=[float(md[0]),float(md[-1])],
        nominal_time_s=[0,400000],actual_time_s=[float(t[0]),float(t[-1])],
        n_channels=18,n_native_times=3318,n_solver_times=130,n_solver_steps=129,max_solver_steps=240,
        negative_fraction=float(np.mean(signed<0)),
        observation_transform='Existing median drift subtraction; max(strain,0) at native times, then interpolation to unchanged solver times; exclude t=0 from misfit',
        zones=[dict(label='A',lo_ft=float(zone_lo),hi_ft=float(zone_hi),
                    width_m=float((zone_hi-zone_lo)*.3048),initial_alpha_xx=-17.5,initial_alpha_yy=-17.5,
                    kxx_m2=10**-17.5,kyy_m2=10**-17.5,kzz_m2=1e-21,
                    optimized_components=['xx','yy'],fixed_geometry=True,
                    layer_indices_zero_based=[99,100],n_elements_in_y=4)],
        parameter_order=['alpha_xx_A','alpha_yy_A'],n_parameters=2,theta_bounds=[-21,-12],
        background_tensor_m2=[1e-21]*3,kzz_m2=1e-21,
        initial_rationale='Heuristic permeability scale close to previous zone-B solution; not inferred from strain amplitude or pressure/strain correlation',
        mesh=dict(nx=200,ny=len(element_edges)-1,nelements=200*(len(element_edges)-1),
                  x_spacing_m=.5,observation_dy_m=[float(np.diff(local_edges).min()),float(np.diff(local_edges).max())],
                  local_element_edges_ft=md_edges.tolist(),element_edges_y_m=element_edges.tolist(),
                  material_edges_y_m=material_edges.tolist(),material_blocks=200,
                  observation_rule='One channel per native-depth cell; edges at adjacent channel midpoints',
                  refinement_scope='DSS depth direction only; retain existing x resolution'),
        physics=dict(domain_m=[100,100],youngs_modulus_Pa=35e9,poisson_ratio=.25,biot_coefficient=.8,
                     porosity=.01,fluid_bulk_modulus_Pa=2.2e9,viscosity_Pa_s=.001,
                     injection_x_m=45,receiver_x_m=69.384,well_separation_ft=80,
                     md_center_ft=14376,y_center_m=50,
                     boundary_and_storage='Retain existing validated framework; no HM or mechanical-BC changes in this proposal'),
        regularization=dict(form='beta*sum(sqrt((alpha+21)^2+delta^2)-delta)',
                            reference_alpha=-21,delta=.05,beta=baseline['beta_l1'],
                            scope='One term each for alpha_xx_A and alpha_yy_A, not per mesh element',
                            coefficient_policy='Retain existing beta for initial comparison; no crop-based recalibration'),
        objective_scale=baseline['objective_scale'],
        solver=dict(integration='ImplicitEuler',l_tol=.001,nl_abs_tol=.001,nl_rel_tol=.001,
                    l_max_its=2000,nl_max_its=200,linear_solver='LU/MUMPS',mpi_ranks=20),
        optimizer=dict(method='L-BFGS-B',maxiter=60,ftol=1e-7,gtol=1e-8,maxls=20),
        validation='After user QC: initial response comparison and two directional data-gradient checks on the new two-parameter model before optimization',
        old_zones='Previous 147xx-ft zones are outside this recentered local domain and are not copied or relocated into this case')
    (OUT/'PROPOSED_PARAMETERS.json').write_text(json.dumps(proposal,indent=2)+'\n')
    np.savez_compressed(OUT/'data/pressure_qc.npz',taxis=gt[gauge_mask],data=gp[gauge_mask],units='MPa')
    plot(proposal,md,t,signed,positive,gt,gp)
    print(json.dumps({k:proposal[k] for k in ['status','n_channels','negative_fraction','zones']},indent=2))
    print('Mesh:',proposal['mesh']['nx'],proposal['mesh']['ny'],proposal['mesh']['observation_dy_m'])
    print(table.to_string(index=False))


def plot(proposal,md,t,signed,positive,gt,gp):
    zone_lo=proposal['zones'][0]['lo_ft'];zone_hi=proposal['zones'][0]['hi_ft']
    core=np.where((md>=zone_lo)&(md<zone_hi))[0]
    gauge_mask=(gt>=0)&(gt<=t[-1])
    fig = plt.figure(figsize=(18,10),layout='constrained')
    grid=fig.add_gridspec(7,3,width_ratios=[2.5,1.05,2.6],height_ratios=[1]*6+[.25])
    ax=fig.add_subplot(grid[:6,0]);perm=fig.add_subplot(grid[:6,1],sharey=ax)
    cmap=LinearSegmentedColormap.from_list('positive_dss',['white','#ffaaaa','#e53935','#8b0000'])
    im=ax.pcolormesh(t/1000,md,positive*1e6,shading='nearest',cmap=cmap,vmin=0,vmax=3,rasterized=True)
    ax.set(title='Positive DSS | 18 native channels',xlabel='Time [1000 s]',ylabel='Measured depth [ft]',
           xlim=(0,400),ylim=(14379,14373))
    ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    ax.axvspan(t[-1]/1000,400,color='.88',hatch='///')
    for a in [ax,perm]:
        a.axhline(zone_lo,color='#bd7200',linestyle='--',linewidth=1.2)
        a.axhline(zone_hi,color='#bd7200',linestyle='--',linewidth=1.2)
    depths=[14373,zone_lo,zone_lo,zone_hi,zone_hi,14379]
    alphas=[-21,-21,-17.5,-17.5,-21,-21]
    perm.plot(alphas,depths,color='#dc8400',lw=3,label='kxx initial')
    perm.plot(alphas,depths,color='#155f9b',lw=1.5,ls='--',label='kyy initial')
    perm.axhspan(zone_lo,zone_hi,color='#ffdf9c',alpha=.25)
    perm.set(title='Proposed permeability',xlabel='log10(k / m²)',xlim=(-21.6,-16.6),xticks=[-21,-19,-17.5])
    perm.tick_params(labelleft=False);perm.grid(axis='x',alpha=.2)
    perm.text(-20.6,14374,'Background\nxx = yy = zz\n1e-21 m²',fontsize=9)
    perm.text(-20.6,14378.1,'Zone A:\nxx = yy = 10^-17.5\nzz = 1e-21 m²',fontsize=9)
    selected=np.r_[core[0]-1,core,core[-1]+1]
    first=None
    for row,channel in enumerate(selected):
        curve=fig.add_subplot(grid[row,2],sharex=first)
        if first is None: first=curve
        is_core=channel in core
        curve.set_facecolor('#fff8e5' if is_core else 'white')
        line,=curve.plot(t/1000,signed[channel]*1e6,color='#1765ad',lw=.65,label='Signed native DSS')
        curve.axhline(0,color='.6',lw=.5)
        curve.set(ylim=(-.6,3),xlim=(0,400),yticks=[0,1,2])
        curve.text(.015,.90,f'{md[channel]:.3f} ft | '+('zone A' if is_core else 'outside A'),
                   transform=curve.transAxes,va='top',fontsize=9)
        curve.grid(axis='x',alpha=.15)
        pa=curve.twinx();pressure_line,=pa.plot(gt[gauge_mask]/1000,gp[gauge_mask],color='#c43d35',lw=.9,label='Pressure change')
        pa.set(ylim=(-.15,2.45),yticks=[0,1,2]);pa.tick_params(labelcolor='#c43d35',labelsize=8)
        if row==2:
            curve.set_ylabel('Signed strain [microstrain]');pa.set_ylabel('Pressure change [MPa]',color='#c43d35')
        if row < 5: curve.tick_params(labelbottom=False)
        else: curve.set_xlabel('Time [1000 s]')
    cb=fig.add_subplot(grid[6,0]);fig.colorbar(im,cax=cb,orientation='horizontal',label='Positive strain [microstrain]')
    leg=fig.add_subplot(grid[6,1]);leg.axis('off');leg.legend(*perm.get_legend_handles_labels(),loc='center',fontsize=8)
    leg=fig.add_subplot(grid[6,2]);leg.axis('off');leg.legend(handles=[line,pressure_line],loc='center',ncol=2,fontsize=8)
    fig.suptitle(f'S-well 14376 ft | INITIAL MODEL PROPOSAL ONLY\n'
        f'Fixed zone A: {zone_lo:.3f}–{zone_hi:.3f} ft | 2 parameters: alpha_xx, alpha_yy | native-depth mesh\n'
        'Left: positive observation; right: native signed channels and pressure, no time shift. No simulated response yet.',fontsize=13)
    fig.savefig(OUT/'proposed_initial_qc.png',dpi=180);fig.savefig(OUT/'proposed_initial_qc.pdf');plt.close(fig)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    if args.plot_only:
        proposal=json.loads((OUT/'PROPOSED_PARAMETERS.json').read_text())
        data=np.load(OUT/'data/unrectified_native.npz')
        pressure=np.load(OUT/'data/pressure_qc.npz')
        plot(proposal,data['daxis'],data['taxis'],data['data'],np.maximum(data['data'],0),pressure['taxis'],pressure['data'])
    else:
        main()
