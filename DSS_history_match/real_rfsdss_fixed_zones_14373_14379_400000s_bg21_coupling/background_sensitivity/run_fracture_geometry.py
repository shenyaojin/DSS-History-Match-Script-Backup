"""One-cell deeper-edge trial on the existing bg=-18 physical forward.

Only the shared material boundary between layer_101 and layer_102 moves.
The mesh, observation coordinates, physical coefficients and solver are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

import run_experiment as base
import plot_weighted_refit_separate as qc

HERE = Path(__file__).resolve().parent
RUN = HERE / 'fracture_geometry_bg18' / 'deeper_edge_one_channel'
SOURCE = HERE / 'bg_18'
REFERENCE = HERE / 'coupling_refit_weighted3_unbounded'
KEYS = ['measurement_time', 'measurement_values', 'measurement_xcoord',
        'measurement_ycoord', 'measurement_zcoord']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_completed_forward():
    log = (RUN/'forward.log').read_text()
    # MOOSE logs the INITIAL output as Time Step 0; only 1..129 are solves.
    steps = list(map(int, re.findall(r'^Time Step (\d+),', log, re.M)))
    assert steps == list(range(130)), steps
    assert 'Finished Executing' in log


def pressure_check():
    """Track the previously observed overshoot at the same 315 ks snapshot."""
    from scipy.io import netcdf_file
    text = (RUN/'forward.i').read_text()
    block = re.search(r'^  \[injection_pressure_func\].*?^  \[\]', text, re.M|re.S)[0]
    sx = np.fromstring(re.search(r"^\s*x = '([^']*)'", block, re.M)[1], sep=' ')
    sy = np.fromstring(re.search(r"^\s*y = '([^']*)'", block, re.M)[1], sep=' ')
    with netcdf_file(RUN/'pressure.e', 'r', mmap=False) as ds:
        names = [r.tobytes().decode().strip('\x00 ') for r in ds.variables['name_nod_var'][:]]
        times = np.asarray(ds.variables['time_whole'][:])
        assert len(times) == 130
        index = int(np.argmin(abs(times-315033)))
        pressure = np.asarray(ds.variables[f'vals_nod_var{names.index("pp")+1}'][index])
        field_max = float(pressure.max())
    history_max = max(float(sy[sx<=times[index]].max()), float(np.interp(times[index], sx, sy)))
    return dict(time_s=float(times[index]), field_max_MPa=field_max/1e6,
        source_history_max_MPa=history_max/1e6, overshoot_percent=100*(field_max/history_max-1))


def prepare():
    if RUN.exists():
        raise RuntimeError('Refusing to overwrite existing geometry trial')
    cfg = json.loads((HERE.parent / 'inversion/config.json').read_text())
    original = (SOURCE / 'forward.i').read_text()
    edges = np.array(cfg['mesh_element_edges_m'])
    materials = np.array(cfg['mesh_edges_m'])
    old_lo, old_hi = materials[99], materials[101]
    hi_index = int(np.flatnonzero(np.isclose(edges, old_hi, rtol=0, atol=1e-10))[0])
    new_hi = float(edges[hi_index + 1])
    assert new_hi < materials[102]
    forward = base.replace_block(original, 'layer_101_bbox',
        lambda b: base.set_value(b, 'top_right', f'100 {new_hi:.16g} 0'))
    forward = base.replace_block(forward, 'layer_102_bbox',
        lambda b: base.set_value(b, 'bottom_left', f'0 {new_hi:.16g} 0'))
    # Exactly two coordinate lines change: all numerical and physical settings remain identical.
    changes = [(a,b) for a,b in zip(original.splitlines(), forward.splitlines()) if a != b]
    assert len(original.splitlines()) == len(forward.splitlines()) and len(changes) == 2
    for section in ['Kernels', 'Materials', 'AuxKernels', 'Reporters', 'Functions',
                    'Executioner', 'Preconditioning', 'BCs', 'Variables', 'FluidProperties']:
        assert base.section(original, section) == base.section(forward, section), section
    mid = (edges[:-1]+edges[1:])/2
    old_blocks = np.searchsorted(materials, mid, side='right')
    materials[101] = new_hi
    new_blocks = np.searchsorted(materials, mid, side='right')
    assert np.sum(old_blocks != new_blocks) == 1
    assert np.sum(np.isin(old_blocks, [100,101])) == 4
    assert np.sum(np.isin(new_blocks, [100,101])) == 5
    depth = lambda y: float(14376 + (y-50)/.3048)
    RUN.mkdir(parents=True)
    (RUN / 'forward.i').write_text(forward)
    for name in ['measurement_data.csv', 'solver_times.csv']:
        shutil.copy2(SOURCE / name, RUN / name)
    (RUN / '.gitignore').write_text('pressure.e*\nforward_csv*\n*.log\n')
    params = json.loads((SOURCE / 'parameters.json').read_text())
    source_c = params.pop('coupling_C')
    manifest = dict(scope='Extend only the deep edge by one existing DSS-scale mesh cell',
        background_alpha=-18, fixed_parameters=params,
        source_forward_reference_C=source_c,
        old_zone_depth_ft=[depth(old_lo), depth(old_hi)],
        new_zone_depth_ft=[depth(old_lo), depth(new_hi)],
        old_center_depth_ft=depth((old_lo+old_hi)/2),
        new_center_depth_ft=depth((old_lo+new_hi)/2),
        center_shift_ft=(new_hi-old_hi)/2/.3048,
        old_width_ft=(old_hi-old_lo)/.3048, new_width_ft=(new_hi-old_lo)/.3048,
        mesh=[200,216], old_zone_y_cells=4, new_zone_y_cells=5,
        n_forward_steps=129, max_steps=240, mpi_ranks=20,
        weights='Same original four observation channels retain SSE weight=3; all others=1',
        coupling='Refit unconstrained global C after the physical forward',
        changed_lines=changes, input_sha256=digest(RUN/'forward.i'),
        source_sha256={str(SOURCE/name):digest(SOURCE/name) for name in
            ['forward.i', 'measurement_data.csv', 'solver_times.csv', 'strain_prediction.csv']})
    base.save(RUN/'manifest.json', manifest)
    base.status(RUN/'status.json', 'prepared')
    print(json.dumps(manifest, indent=2), flush=True)


def solve(check=False):
    manifest = json.loads((RUN/'manifest.json').read_text())
    assert digest(RUN/'forward.i') == manifest['input_sha256']
    if not check:
        if (RUN/'STARTED').exists():
            raise RuntimeError('Refusing duplicate forward run')
        assert (RUN/'INPUT_CHECK_PASSED').exists()
        (RUN/'STARTED').write_text('Forward started\n')
    env = os.environ.copy()
    env['PATH'] = str(base.PYTHON_BIN) + ':' + env['PATH']
    for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'TBB_NUM_THREADS']:
        env[var] = '1'
    cmd = [str(base.PYTHON_BIN/'mpiexec'), '-n', '20',
        str(base.ROOT/'moose_env/moose/modules/combined/combined-opt'), '-i', 'forward.i', '--n-threads=1']
    if check:
        cmd.append('--check-input')
    base.status(RUN/'status.json', 'checking_input' if check else 'running', command=cmd)
    start = time.monotonic()
    with (RUN/('check.log' if check else 'forward.log')).open('w') as log:
        result = subprocess.run(cmd, cwd=RUN, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        base.status(RUN/'status.json', 'failed', returncode=result.returncode)
        raise RuntimeError('MOOSE failed; see check.log or forward.log')
    if not check:
        verify_completed_forward()
    (RUN/('INPUT_CHECK_PASSED' if check else 'FORWARD_COMPLETE')).write_text('Passed\n')
    base.status(RUN/'status.json', 'input_checked' if check else 'forward_complete', wall_seconds=time.monotonic()-start)


def summarize():
    # Allow postprocessing a completed solve after an interrupted Python driver.
    verify_completed_forward()
    (RUN/'FORWARD_COMPLETE').write_text('Verified 129 solves plus initial output\n')
    manifest = json.loads((RUN/'manifest.json').read_text())
    reference = json.loads((REFERENCE/'summary.json').read_text())
    old = next(r for r in reference['results'] if r['background_alpha'] == -18)
    files = [p for p in RUN.glob('forward*data_*.csv') if '_signed_data_' not in p.name]
    latest = max(files, key=lambda p:int(re.search(r'_(\d+)\.csv$',p.name)[1]))
    df = pd.read_csv(latest)
    canonical = pd.read_csv(SOURCE/'measurement_data.csv')
    np.testing.assert_allclose(df[KEYS], canonical[KEYS], rtol=1e-10, atol=1e-18)
    assert len(df) == 18*130 and df.measurement_time.nunique() == 130
    depth = 14376 + (df.measurement_ycoord.to_numpy()-50)/.3048
    core = np.isclose(depth[:,None], reference['central_channel_depths_ft'], rtol=0, atol=1e-6).any(axis=1)
    w = np.where(core, 3., 1.)
    use = df.measurement_time.to_numpy() > 0
    assert core.sum() == 4*130 and w[use].sum() == 26*129
    s = df.simulation_values.to_numpy().copy()
    d = df.measurement_values.to_numpy()
    assert np.isfinite([s,d]).all() and (s>=0).all() and (d>=0).all()
    ss = float(np.sum(w[use]*s[use]**2))
    sd = float(np.sum(w[use]*s[use]*d[use]))
    c = sd/ss
    residual = c*s-d
    assert abs(np.sum(w[use]*s[use]*residual[use])) < 1e-12*max(abs(sd), abs(c*ss))
    df['physical_simulation_values'] = s
    df['simulation_values'] = c*s
    df['misfit_values'] = np.where(use, residual, 0.)
    df['objective_weight'] = w
    df.to_csv(RUN/'bg_18_strain.csv', index=False)
    row = dict(background_alpha=-18, fitted_C=c,
        weighted_objective_after=float(.5*np.sum(w[use]*residual[use]**2)),
        weighted_rms_after_microstrain=float(np.sqrt(np.sum(w[use]*residual[use]**2)/w[use].sum())*1e6),
        unweighted_rms_after_microstrain=float(np.sqrt(np.mean(residual[use]**2))*1e6),
        core_rms_after_microstrain=float(np.sqrt(np.mean(residual[use & core]**2))*1e6))
    baseline = pd.read_csv(REFERENCE/'bg_18_strain.csv')
    channel_rows = []
    for y in np.sort(df.measurement_ycoord.unique()):
        selected = use & (df.measurement_ycoord.to_numpy() == y)
        channel_rows.append(dict(depth_ft=14376+(y-50)/.3048,
            weight=float(w[selected][0]),
            rms_before_microstrain=float(np.sqrt(np.mean(baseline.misfit_values.to_numpy()[selected]**2))*1e6),
            rms_after_microstrain=float(np.sqrt(np.mean(residual[selected]**2))*1e6)))
    pd.DataFrame(channel_rows).to_csv(RUN/'channel_comparison.csv', index=False)
    pd.DataFrame([dict(case='original', **old), dict(case='deeper_edge_one_channel', **row)]).to_csv(RUN/'comparison.csv', index=False)
    summary = dict(method='C=sum(w*s*d)/sum(w*s*s)', coupling_bounds=None,
        central_channel_depths_ft=reference['central_channel_depths_ft'], core_weight=3,
        geometry=manifest, results=[row], original_geometry_result=old,
        weighted_objective_reduction_percent=100*(1-row['weighted_objective_after']/old['weighted_objective_after']),
        pressure_diagnostic=pressure_check(),
        interpretation_limit='Single geometry trial, not a permeability inversion; known pressure overshoot remains unresolved')
    base.save(RUN/'summary.json', summary)
    lo, hi = manifest['new_zone_depth_ft']
    qc.main(out=RUN, alphas=(-18,), geometry_note=f'Zone: {lo:.3f}-{hi:.3f} ft | Deep edge extended by one channel')
    for path, expected in manifest['source_sha256'].items():
        assert digest(Path(path)) == expected
    (RUN/'COMPLETE').write_text('Forward, weighted global C refit and QC complete\n')
    base.status(RUN/'status.json', 'complete')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','check','run','summarize'])
    action = parser.parse_args().action
    if action == 'prepare':
        prepare()
    elif action == 'check':
        solve(check=True)
    elif action == 'run':
        if not (RUN/'INPUT_CHECK_PASSED').exists():
            solve(check=True)
        solve()
        summarize()
    else:
        summarize()
