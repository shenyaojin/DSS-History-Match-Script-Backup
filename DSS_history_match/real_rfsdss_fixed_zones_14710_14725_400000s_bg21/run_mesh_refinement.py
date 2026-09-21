"""Local channel-scale y refinement, one evaluation at the coarse final tensor.

Preserves the 200 material blocks and their parameters. CartesianMeshGenerator
iy subdivides only blocks intersecting the observation crop; no optimization.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil

import numpy as np
import pandas as pd

from prepare_xy_comparison import add_objects, replace_block, set_value
from run_xy_comparison import Comparison, save_json

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'comparison_xx_yy_60/fixed_yy'
RUN = HERE / 'mesh_refinement_channel_scale'


def prepare():
    if RUN.exists():
        raise RuntimeError('Refusing to overwrite an existing refinement case')
    cfg = json.loads((SOURCE/'config.json').read_text())
    theta = np.loadtxt(SOURCE/'theta_final.txt')
    data = pd.read_csv(SOURCE/'measurement_data.csv')
    y = np.sort(data.measurement_ycoord.unique())
    target = float(np.diff(y).min())
    edges = np.array(cfg['mesh_edges_m'])
    widths = np.diff(edges)
    lo, hi = 50+(np.array([14710., 14725.])-14725)*.3048
    selected = (edges[:-1] < hi-1e-10) & (edges[1:] > lo+1e-10)
    subdivisions = np.ones(200, dtype=int)
    subdivisions[selected] = np.ceil(widths[selected]/target).astype(int)
    fine_edges = np.r_[edges[0], np.concatenate([
        np.linspace(a, b, n+1)[1:] for a, b, n in zip(edges[:-1], edges[1:], subdivisions)])]
    cells = np.searchsorted(fine_edges, y, side='right')-1
    assert len(np.unique(cells)) == len(y) == 45
    assert np.all(np.isin(edges, fine_edges))
    assert np.all(widths[selected]/subdivisions[selected] <= target+1e-12)
    alpha = np.full(200, cfg['background_alpha'])
    for zone in cfg['zones']:
        alpha[zone['layer_indices_zero_based']] = zone['alpha']
    for value, indices in zip(theta, cfg['zone_layer_indices']):
        alpha[indices] = value
    full = np.r_[alpha, cfg['initial_kyy_theta']]
    values = '; '.join(f'{v:.16g}' for v in full)
    original = (SOURCE/'forward_and_adjoint.i').read_text()
    forward = replace_block(original, 'base_mesh', lambda b: b.replace(
        '    dim = 2', "    iy = '"+' '.join(map(str, subdivisions))+"'\n    dim = 2"))
    idx = np.searchsorted(fine_edges, data.measurement_ycoord.to_numpy(), side='right')-1
    lower, upper = fine_edges[idx], fine_edges[idx+1]
    arrays = [upper, lower, 1/(upper-lower), np.full(len(data), 69.25)]
    sampling = '; '.join(' '.join(f'{v:.16g}' for v in arr) for arr in arrays)
    forward = replace_block(forward, 'sampling_geometry', lambda b: set_value(b, 'real_vector_values', sampling))
    forward = replace_block(forward, 'params', lambda b: set_value(b, 'real_vector_values', values))
    forward = add_objects(forward, 'Reporters', """  [signed_data]
    type = OptimizationData
    variable = 'strain_yy'
    measurement_file = 'measurement_data.csv'
    file_xcoord = 'measurement_xcoord'
    file_ycoord = 'measurement_ycoord'
    file_zcoord = 'measurement_zcoord'
    file_time = 'measurement_time'
    file_value = 'measurement_values'
    outputs = 'csv'
  []
""")
    # Verify all original input objects except these three are byte-identical.
    def strip_allowed(text):
        for name in ['base_mesh', 'sampling_geometry', 'params', 'signed_data']:
            text = re.sub(r'^  \['+name+r'\].*?^  \[\]\n', '', text, flags=re.M | re.S)
        return text
    assert strip_allowed(original) == strip_allowed(forward)
    master = set_value((SOURCE/'optimize.i').read_text(), 'initial_condition', values)
    cfg.update(initial_theta=theta.tolist(), case_stage='Single fixed-tensor mesh comparison; no optimization',
               mesh_element_edges_m=fine_edges.tolist(), mesh_y_subdivisions=subdivisions.tolist(),
               mesh_note='mesh_edges_m and layer_indices retain original MATERIAL block meanings',
               reference_case=str(SOURCE), target_y_spacing_m=target)
    RUN.mkdir()
    for name in ['measurement_data.csv', 'solver_times.csv', 'permeability_final.csv']:
        shutil.copy2(SOURCE/name, RUN/name)
    (RUN/'forward_and_adjoint.i').write_text(forward)
    (RUN/'optimize.i').write_text(master)
    save_json(RUN/'config.json', cfg)
    np.savetxt(RUN/'initial_alpha.txt', full)
    pd.DataFrame(dict(md_ft=14725+(y-50)/.3048, y_m=y, element_index=cells,
                      element_lower_m=fine_edges[cells], element_upper_m=fine_edges[cells+1])) .to_csv(
        RUN/'channel_element_mapping.csv', index=False)
    names = ['config.json', 'forward_and_adjoint.i', 'optimize.i', 'measurement_data.csv',
             'solver_times.csv', 'initial_alpha.txt']
    audit = dict(input_sha256={n: hashlib.sha256((RUN/n).read_bytes()).hexdigest() for n in names},
                 source_input_sha256=hashlib.sha256(original.encode()).hexdigest(),
                 target_y_spacing_m=target, nx=200, ny=int(subdivisions.sum()),
                 old_elements=40000, new_elements=200*int(subdivisions.sum()),
                 distinct_observation_elements=45, material_blocks=200,
                 refined_material_blocks=(np.where(selected)[0]+1).tolist(),
                 crop_dy_range_m=[float((widths/subdivisions)[selected].min()),
                                  float((widths/subdivisions)[selected].max())],
                 n_solver_steps=129, max_solver_steps=240, mpi_ranks=20,
                 fixed_xx_alpha=theta.tolist(), fixed_yy_alpha=cfg['initial_kyy_theta'],
                 gradient_status='Sampling updated; no finite-difference validation or optimization in this case')
    save_json(RUN/'preflight.json', audit)
    print(json.dumps(audit, indent=2), flush=True)


def summarize():
    import matplotlib.pyplot as plt
    old = pd.read_csv(SOURCE/'strain_final.csv')
    new = pd.read_csv(RUN/'strain_refined.csv')
    signed = pd.read_csv(RUN/'strain_signed.csv')
    keys = ['measurement_time', 'measurement_ycoord', 'measurement_values']
    np.testing.assert_allclose(old[keys], new[keys], rtol=1e-10, atol=1e-18)
    np.testing.assert_allclose(new[keys], signed[keys], rtol=1e-10, atol=1e-18)
    np.testing.assert_allclose(new.simulation_values, np.maximum(signed.simulation_values, 0), atol=1e-16, rtol=1e-9)
    times = np.sort(new.measurement_time.unique())
    yy = np.sort(new.measurement_ycoord.unique())
    md = 14725+(yy-50)/.3048
    def matrix(df, column):
        return df.pivot(index='measurement_ycoord', columns='measurement_time', values=column).reindex(index=yy, columns=times).to_numpy()
    obs, coarse, fine, raw = matrix(new, 'measurement_values'), matrix(old, 'simulation_values'), matrix(new, 'simulation_values'), matrix(signed, 'simulation_values')
    def metrics(values):
        r = (values-obs)[:, 1:]
        return dict(J_data=float(.5*np.sum(r*r)), rms_microstrain=float(np.sqrt(np.mean(r*r))*1e6))
    summary = dict(coarse=metrics(coarse), refined=metrics(fine),
                   fixed_parameter_comparison=True, optimization_performed=False,
                   relative_J_change=metrics(fine)['J_data']/metrics(coarse)['J_data']-1,
                   coarse_to_refined_rms_microstrain=float(np.sqrt(np.mean((fine[:, 1:]-coarse[:, 1:])**2))*1e6),
                   signed_prediction_min_microstrain=float(raw.min()*1e6),
                   signed_prediction_negative_fraction=float(np.mean(raw[:, 1:] < 0)))
    save_json(RUN/'comparison_summary.json', summary)
    fig, axes = plt.subplots(1, 4, figsize=(16, 6), sharey=True, layout='constrained')
    for ax, arr, title in zip(axes, [obs, coarse, fine, fine-obs], ['Observation', 'Coarse: 200 x 200', 'Refined at same permeability', 'Refined minus observation']):
        im = ax.pcolormesh(times/1000, md, arr*1e6, shading='nearest', cmap='bwr', vmin=-obs.max()*1e6, vmax=obs.max()*1e6)
        ax.set(title=title, xlabel='Time [1000 s]', xlim=(0, 400), ylim=(14725, 14710))
        ax.axvspan(times[-1]/1000, 400, color='.9')
    axes[0].set_ylabel('Measured depth [ft]')
    fig.colorbar(im, ax=axes, label='Strain [microstrain]')
    fig.savefig(RUN/'mesh_comparison.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, layout='constrained')
    for ax, label, bounds in zip(axes, 'ABC', [(14709.5,14712), (14715,14717), (14721,14727)]):
        use = (md >= bounds[0]) & (md < bounds[1])
        for arr, name, style in [(obs, 'Observation (+)', 'k-'), (coarse, 'Coarse (+)', 'C1-'),
                                  (fine, 'Refined (+)', 'C2-'), (raw, 'Refined signed', 'C2--')]:
            ax.plot(times/1000, arr[use].mean(axis=0)*1e6, style, label=name)
        ax.set(title=f'Zone {label}: mean over visible channels', ylabel='Strain [microstrain]')
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[-1].set(xlabel='Time [1000 s]', xlim=(0, 400))
    fig.savefig(RUN/'zone_curves_signed.png', dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    # JIT expressions use mpicxx; Python's absolute path does not activate conda.
    os.environ['PATH'] = '/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin:'+os.environ['PATH']
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'check', 'run', 'summarize'])
    args = parser.parse_args()
    if args.action == 'prepare':
        return prepare()
    if args.action == 'summarize':
        return summarize()
    for name, digest in json.loads((RUN/'preflight.json').read_text())['input_sha256'].items():
        assert hashlib.sha256((RUN/name).read_bytes()).hexdigest() == digest, name
    if (RUN/'RUNNING').exists() or (RUN/'objective_history.csv').exists():
        raise RuntimeError('Case already started; refusing duplicate execution')
    model = Comparison(RUN)
    np.testing.assert_array_equal(model.expand(model.theta0), np.loadtxt(RUN/'initial_alpha.txt'))
    if args.action == 'check':
        ok, stdout, stderr = model.runner.run(str(RUN/'optimize.i'), str(RUN/'input_check'),
            num_processors=20, additional_args=['--check-input'], log_file_name='check.log',
            stream_output=False, clean_output_dir=False)
        (RUN/'input_check/stderr.log').write_text(stderr)
        if not ok:
            raise RuntimeError('Input check failed: '+stderr[-4000:])
        (RUN/'INPUT_CHECK_PASSED').write_text('MOOSE input check passed\n')
        return
    assert (RUN/'INPUT_CHECK_PASSED').exists()
    (RUN/'RUNNING').write_text(f'pid={os.getpid()}\n')
    try:
        model.evaluate(model.theta0, purpose='fixed_parameter_mesh_comparison')
        shutil.copy2(model.find_report(), RUN/'strain_refined.csv')
        files = list(model.output.glob('*_signed_data_*.csv'))
        last = max(files, key=lambda p: int(re.search(r'_signed_data_(\d+)\.csv$', p.name)[1]))
        shutil.copy2(last, RUN/'strain_signed.csv')
        summarize()
        (RUN/'COMPLETE').write_text('One fixed-parameter evaluation and QC complete; no optimization\n')
    except Exception as exc:
        (RUN/'FAILED').write_text(f'{type(exc).__name__}: {exc}\n')
        raise
    finally:
        (RUN/'RUNNING').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
