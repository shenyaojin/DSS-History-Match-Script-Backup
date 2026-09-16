"""Prepare observations and a reviewable proposal only. Never starts MOOSE."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'real_rfsdss_fixed_zones_14700_14750_400000s'
REPO = HERE.parents[2]
V6 = HERE.parent / 'optimizer_input_file_test/perm5layer_100_v6strain_zonalgrid'


def main():
    cfg = json.loads((PREVIOUS / 'initial_forward_only/unrectified/config.json').read_text())
    with np.load(PREVIOUS / 'data/unrectified_full.npz', allow_pickle=True) as src:
        select = (src['daxis'] >= 14710) & (src['daxis'] <= 14725)
        md, times = src['daxis'][select], src['taxis']
        signed = src['data'][select].copy()
        start = src['start_time'].item()
    positive = np.maximum(signed, 0.)
    assert np.isfinite(signed).all() and np.all(positive >= 0)
    assert np.array_equal(positive[signed >= 0], signed[signed >= 0])
    data_dir = HERE / 'data'
    data_dir.mkdir(exist_ok=True)
    for name, arr in [('unrectified', signed), ('half_wave_positive', positive)]:
        np.savez_compressed(data_dir / f'{name}_native.npz', data=arr, daxis=md,
                            taxis=times, start_time=start, units='strain')
    solver_times = np.loadtxt(PREVIOUS / 'initial_forward_only/unrectified/solver_times.csv', skiprows=1)
    assert len(solver_times) - 1 <= 240 and solver_times[-1] == times[-1]
    # Rectify at native sampling first, then interpolate the already approved
    # positive-only data to the solver times. Do not remove rows or close gaps.
    sampled = np.array([np.interp(solver_times, times, row) for row in positive])
    sampled_signed = np.array([np.interp(solver_times, times, row) for row in signed])
    np.savez_compressed(data_dir / 'half_wave_positive_solver.npz', data=sampled,
                        daxis=md, taxis=solver_times, start_time=start, units='strain')
    y = 50 + (md.astype(float) - 14725) * .3048
    pd.DataFrame(dict(measurement_time=np.repeat(solver_times, len(md)),
                      measurement_values=sampled.T.ravel(), measurement_xcoord=cfg['receiver_x_m'],
                      measurement_ycoord=np.tile(y, len(solver_times)), measurement_zcoord=0.,
                      simulation_values=0., misfit_values=0.)).to_csv(data_dir / 'measurement_data.csv', index=False)
    np.savetxt(data_dir / 'solver_times.csv', solver_times, header='time_s', comments='')

    active_ids = [i for i, z in enumerate(cfg['zones']) if z['hi_ft'] > 14710 and z['lo_ft'] < 14725]
    assert active_ids == [1, 3, 4]
    clean = pd.read_csv(V6 / 'data/obs_strain_yy.csv').measurement_values.to_numpy()
    energy = float(np.sum(sampled_signed ** 2))
    beta = float(2e-11 * energy / np.sum(clean ** 2))
    scale = 1. / energy
    zones = []
    for i, z in enumerate(cfg['zones']):
        zones.append(dict(z, original_zone_id=i + 1, optimized=i in active_ids,
                          initial_kxx_m2=10.**z['alpha'], layer_indices_zero_based=cfg['zone_layer_indices'][i]))
    source_text = (PREVIOUS / 'unrectified/forward_and_adjoint.i').read_text()
    functions = {}
    for name in ['func_kyy', 'func_zero']:
        functions[name] = re.search(r'  \[' + name + r'\].*?  \[\]', source_text, re.S)[0]
    plan = dict(status='AWAITING_USER_QC_APPROVAL; no inversion started',
                depth_requested_ft=[14710,14725], depth_actual_ft=[float(md[0]),float(md[-1])],
                time_requested_s=[0,400000], time_actual_s=[float(times[0]),float(times[-1])],
                start_time=str(start), n_channels=len(md), native_n_time=len(times),
                n_solver_times=len(solver_times), n_solver_steps=len(solver_times)-1,
                max_solver_steps=240, abort_on_solve_fail=True,
                negative_fraction=float(np.mean(signed < 0)), transform='max(strain,0); native rectification then linear time interpolation',
                n_parameters=3, active_original_zone_ids=[i+1 for i in active_ids],
                initial_theta=[cfg['initial_theta'][i] for i in active_ids],
                theta_bounds=[-18,-12], background_alpha=-18, zones=zones,
                beta_l1=beta, objective_scale=scale, regularization_delta=.05,
                regularization_reference_alpha=-18, active_layer_weights=[len(cfg['zone_layer_indices'][i]) for i in active_ids],
                solver=dict(integrator='ImplicitEuler',nl_abs_tol=.001,nl_rel_tol=.001,l_tol=.001,
                            nl_max_its=200,l_max_its=2000,preconditioner='LU/MUMPS',mpi_ranks=20),
                optimizer=dict(method='L-BFGS-B',maxiter=60,ftol=1e-7,gtol=1e-8,maxls=20),
                physics=dict(model_size_m=[100,100],mesh_cells=[200,200],youngs_modulus_Pa=35e9,
                             poissons_ratio=.25,porosity=.01,biot_coefficient=.8,
                             fluid_bulk_modulus_Pa=2.2e9,viscosity_Pa_s=.001,density_kg_m3=1000,
                             injection_x_m=45,receiver_x_m=69.384,separation_ft=80,
                             permeability_fixed_functions=functions),
                original_source=str(REPO/'data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz'),
                preprocessed_source=str(PREVIOUS/'data/unrectified_full.npz'),
                pressure_source=str(REPO/'data/fiberis_format/prod/gauges/pressure_g1.npz'))
    (HERE/'PROPOSED_PARAMETERS.json').write_text(json.dumps(plan,indent=2)+'\n')
    (HERE/'AWAITING_QC_APPROVAL').write_text('No inversion authorized until user approves this crop, transform and proposal.\n')

    positive_cmap = LinearSegmentedColormap.from_list('positive_strain', ['white','#ffaaaa','#ff3333','#a50000'])
    fig, axes = plt.subplots(3,1,figsize=(14,11),sharex=True,sharey=True,constrained_layout=True)
    panels = [(times,signed*1e6,'(a) Cropped DSS before rectification | 45 channels, native sampling','bwr',-5,5),
              (times,positive*1e6,'(b) Positive half-wave | negative samples set to zero',positive_cmap,0,3.5),
              (solver_times,sampled*1e6,'(c) Proposed inversion observations | 130 times / 129 solver steps',positive_cmap,0,3.5)]
    for ax,(t,a,title,cmap,vmin,vmax) in zip(axes,panels):
        im=ax.pcolormesh(t,md,a,shading='nearest',cmap=cmap,vmin=vmin,vmax=vmax,rasterized=True)
        ax.axvspan(times[-1],400000,color='.85',hatch='///',label='No recorded data')
        ax.set(title=title,ylabel='Measured depth [ft]',ylim=(14725,14710),xlim=(0,400000))
        ax.set_yticks([14710,14715,14720,14725])
        fig.colorbar(im,ax=ax,label='Strain change [με]',pad=.015)
    axes[-1].plot(solver_times,np.full(len(solver_times),14724.9),'|',color='.25',markersize=4)
    axes[-1].set_xlabel('Time since DSS start [s]')
    fig.suptitle('POW-S RFS-DSS | 14710–14725 ft | QC before inversion\n'
                 f'{np.mean(signed<0):.1%} of native samples set to zero; no rows removed; no amplitude normalization',fontsize=15)
    fig.savefig(HERE/'qc_crop_positive.png',dpi=160)
    fig.savefig(HERE/'qc_crop_positive.pdf')
    plt.close(fig)

    lines = ['# Proposed inversion parameters — awaiting QC approval','',
        'No inversion or new forward solve has been started. This directory contains only data preparation, QC figures and a proposal.', '',
        '## Observations','',
        f'- Requested depth: 14710–14725 ft. Actual native channel centers: {md[0]:.3f}–{md[-1]:.3f} ft; {len(md)} channels.',
        f'- Requested time: 0–400000 s. Actual data: 0–{times[-1]:.6f} s; {len(times)} native times. The final unrecorded interval is hatched, not extrapolated.',
        f'- DSS start: {start}. Source: `{plan["original_source"]}`.',
        '- Reuses the verified cache of that source: per-time median over 7500 < MD < 15000 ft was subtracted, then microstrain converted to strain (×1e-6). No further detrending, scaling or normalization.',
        f'- Positive half-wave: max(strain,0). {np.mean(signed<0):.4%} of native samples become zero. These remain zero-valued observations, not missing or masked observations.',
        '- Rectify native data first, then linearly interpolate to the template solver times. This ordering is explicit; it differs from the earlier full-window preview, which interpolated signed samples before rectifying.',
        f'- Inversion array: {len(md)} channels × {len(solver_times)} times = {sampled.size} records; t=0 shown for QC but excluded from transient data misfit ({(len(solver_times)-1)*len(md)} active records). Equal sample weights, no dt weighting.',
        '- QC color limits: signed −5 to +5 με; positive 0 to 3.5 με. All cropped native values fit within these limits.', '',
        '## Permeability proposal','',
        'Crop the observations only. Keep all original zone boundaries and the full physical model. Optimize the three overlapping zones; fix the two outside zones at the confirmed initial values. Boundary-crossing zones retain their full widths.', '',
        '| Original zone | Full model depth [ft] | Visible overlap [ft] | Initial α = log10(kxx / m²) | Initial kxx [m²] | Action |',
        '|---|---|---|---|---|---|']
    for z in zones:
        overlap=f'{max(14710,z["lo_ft"]):g}–{min(14725,z["hi_ft"]):g}' if z['optimized'] else 'None'
        lines.append(f'| {z["original_zone_id"]} | {z["lo_ft"]:g}–{z["hi_ft"]:g} | {overlap} | {z["alpha"]} | {z["initial_kxx_m2"]:.8e} | {"Optimize" if z["optimized"] else "Fixed"} |')
    lines += ['', '- Active vector (depth order): θ = [−15.5, −16.5, −16.5]. Bounds: −18 ≤ θj ≤ −12, or 1e−18 ≤ kxx ≤ 1e−12 m².',
        '- Background kxx = 1e−18 m². Positions and widths are fixed; this is not isotropic-permeability fitting. kyy and kzz retain the existing functions:',
        '```text',functions['func_kyy'],functions['func_zero'],'```','',
        '## Physics and source','',
        '- Domain 100 × 100 m; 200 × 200 elements, existing y edges fitted to original zone boundaries. y = 50 + (MD − 14725) × 0.3048 m.',
        '- Injection x = 45 m; observation x = 69.384 m. Separation 80 ft is inherited from the old model, not a surveyed distance.',
        '- E = 35 GPa; Poisson ratio 0.25; Biot coefficient 0.8; porosity 0.01.',
        '- Fluid bulk modulus 2.2 GPa; viscosity 0.001 Pa·s; density 1000 kg/m³; gravity zero.',
        '- Initial pressure perturbation and displacements zero. Pressure boundary uses timestamp-aligned production gauge pressure_g1.npz, minus its value at DSS t=0; same full pressure history as the successful 129-step forward solve.',
        '- Existing boundary conditions retained: injection pressure Dirichlet; zero specified x-traction on left/right and y-traction on top/bottom; natural flow boundaries elsewhere.', '',
        '## Solver and optimization','',
        '- V6 TimeSequenceStepper: 130 timestamps, 129 forward steps; last time adjusted to 384301.087267 s. Hard cap 240 forward steps; abort on solve failure. Adjoint replays the forward schedule.',
        '- Implicit Euler; Newton; LU/MUMPS; l_tol = nl_abs_tol = nl_rel_tol = 0.001; l_max_its = 2000; nl_max_its = 200. 20 MPI ranks, one thread per rank.',
        '- L-BFGS-B with adjoint gradient; maxiter=60, ftol=1e−7, gtol=1e−8, maxls=20. Record objective evaluations and accepted iterations separately.',
        '- Before optimization: compare the three-parameter adjoint directional derivative with central differences (h=0.02 in log10 permeability; two directions, 5% relative tolerance). Stop if validation fails; no automatic tolerance or time-step changes.', '',
        '## Misfit and L1 regularization','',
        'Use the same positive-half-wave observation operator for the simulation: r = max(simulated strain_yy,0) − rectified observation. The gradient has gate 1[simulated strain_yy > 0] (zero at the kink).', '',
        '`J = S * (0.5 * sum(r²) + beta * sum_layers(sqrt((alpha + 18)² + 0.05²) - 0.05))`', '',
        '- This is squared data misfit plus smoothed L1 regularization of log-permeability relative to −18; not an L1 data residual.',
        f'- beta = {beta:.12e}; S = {scale:.12e}. Recomputed for this cropped 130-time dataset using the existing rule: beta = 2e−11 × sum(cropped signed sampled data²) / sum(V6 clean data²); S = 1 / sum(cropped signed sampled data²).',
        '- L1 smoothing δ=0.05; active zone layer weights [1,1,3]. The two outside zones retain fixed initial α and contribute only constant regularization. Background deviation is zero.', '',
        '## Pending approval','',
        'Approve the positive-only crop and this proposed three-parameter setup before launching inversion. No solver inputs or launchers for the new inversion have been generated yet. Existing full-window simulations and old inversions remain unchanged.']
    (HERE/'PROPOSED_PARAMETERS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:plan[k] for k in ['n_channels','negative_fraction','n_solver_steps','beta_l1','objective_scale','initial_theta']},indent=2))
    print(HERE/'qc_crop_positive.png')


if __name__ == '__main__':
    main()
