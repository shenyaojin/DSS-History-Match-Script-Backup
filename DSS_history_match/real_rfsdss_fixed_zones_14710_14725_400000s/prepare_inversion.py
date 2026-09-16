"""Verify existing QC and adapt the successful field forward input, without solving."""
from pathlib import Path
import hashlib
import json
import re
import shutil

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'real_rfsdss_fixed_zones_14700_14750_400000s'
BASE = PREVIOUS / 'initial_forward_only/half_wave_positive'
V6 = HERE.parent / 'optimizer_input_file_test/perm5layer_100_v6strain_zonalgrid/inv/_template'
RUN = HERE / 'inversion'


def replace_one(text, pattern, replacement):
    text, count = re.subn(pattern, lambda m: replacement(m), text, count=1, flags=re.S)
    assert count == 1, pattern
    return text


def main():
    if (RUN / 'objective_history.csv').exists():
        raise RuntimeError('Existing inversion history: refusing to regenerate inputs')
    proposal = json.loads((HERE / 'PROPOSED_PARAMETERS.json').read_text())
    cfg = json.loads((BASE / 'config.json').read_text())
    with np.load(HERE / 'data/unrectified_native.npz', allow_pickle=True) as n:
        signed, md, native_times = n['data'], n['daxis'], n['taxis']
    with np.load(HERE / 'data/half_wave_positive_native.npz', allow_pickle=True) as n:
        positive = n['data']
    with np.load(HERE / 'data/half_wave_positive_solver.npz', allow_pickle=True) as n:
        sampled, times = n['data'], n['taxis']
    assert signed.shape == (45, 3318) and sampled.shape == (45, 130)
    assert np.isfinite(signed).all() and np.isfinite(sampled).all()
    assert np.array_equal(positive, np.maximum(signed, 0))
    expected = np.array([np.interp(times, native_times, row) for row in positive])
    assert np.array_equal(expected, sampled)
    template_times = np.fromstring(re.search(r"time_sequence = '([^']*)'", (V6 / 'forward_and_adjoint.i').read_text())[1], sep=' ')
    template_times[-1] = native_times[-1]
    assert np.array_equal(times, template_times)
    assert len(times) - 1 == 129 and np.all(np.diff(times) > 0)
    assert times[0] == 0 and times[-1] == 384301.087267
    frame = pd.read_csv(HERE / 'data/measurement_data.csv')
    np.testing.assert_allclose(frame.measurement_values, sampled.T.ravel(), rtol=1e-14, atol=1e-22)
    np.testing.assert_array_equal(frame.measurement_time, np.repeat(times, len(md)))
    edges = np.asarray(cfg['mesh_edges_m'])
    y = 50 + (md.astype(float) - 14725) * .3048
    np.testing.assert_allclose(frame.measurement_ycoord, np.tile(y, len(times)), rtol=0, atol=1e-13)
    assert np.all(frame.measurement_xcoord == 69.384)
    assert np.all(np.diff(edges) > 0) and len(edges) == 201
    zones = sorted(proposal['zones'], key=lambda z: z['lo_ft'])
    assert [(z['lo_ft'], z['hi_ft'], z['alpha']) for z in zones] == [
        (14701.5, 14704., -15.5), (14709.5, 14712., -15.5),
        (14715., 14717., -16.5), (14721., 14727., -16.5),
        (14737.5, 14739.5, -15.5)]
    alpha = np.full(200, -18.)
    for z in zones:
        indices = z['layer_indices_zero_based']
        alpha[indices] = z['alpha']
        assert np.isclose(edges[indices[0]], 50 + (z['lo_ft'] - 14725) * .3048)
        assert np.isclose(edges[indices[-1]+1], 50 + (z['hi_ft'] - 14725) * .3048)
    np.testing.assert_array_equal(alpha, np.loadtxt(BASE / 'initial_alpha.txt'))
    active = [z for z in zones if z['optimized']]
    cfg.update(zones=zones, initial_theta=[z['alpha'] for z in active],
               zone_layer_indices=[z['layer_indices_zero_based'] for z in active],
               n_parameters=3, n_channels=45, n_time=130, native_n_time=3318,
               beta_l1=proposal['beta_l1'], objective_scale=proposal['objective_scale'],
               regularization_delta=.05, regularization_reference_alpha=-18.,
               regularization_scope='one term per active zone; user handover formula',
               regularization_weights=[1, 1, 1],
               observation_sampling=proposal['transform'], depth_crop_ft=[14710, 14725])
    # Reuse the corrected, successfully solved forward input. Only observation
    # geometry/data and the executioner needed for the existing adjoint change.
    text = (BASE / 'initial_forward.i').read_text()
    text = text.replace('  type = Transient\n', "  type = 'TransientAndAdjoint'\n", 1)
    text = text.replace("  system_names = 'nl0'\n  solve_type = NEWTON", "  forward_system = 'nl0'\n  adjoint_system = 'adjoint'", 1)
    old_data = re.search(r'  \[data\].*?  \[\]', (PREVIOUS / 'half_wave_positive/forward_and_adjoint.i').read_text(), re.S)[0]
    text = replace_one(text, r'  \[data\].*?  \[\]', lambda m: old_data)
    layer = np.searchsorted(edges, y, side='right') - 1
    geometry = [np.tile(edges[layer+1], len(times)), np.tile(edges[layer], len(times)),
                np.tile(1/np.diff(edges)[layer], len(times)),
                np.full(sampled.size, (np.floor(cfg['receiver_x_m']/.5)+.5)*.5)]
    values = '; '.join(' '.join(f'{v:.16g}' for v in a) for a in geometry)
    text = replace_one(text, r"(  \[sampling_geometry\].*?real_vector_values = ')[^']*(')", lambda m: m[1]+values+m[2])
    for required in ['num_steps = 240', 'abort_on_solve_fail = true',
                     'l_tol = 0.001', 'nl_abs_tol = 0.001', 'nl_rel_tol = 0.001',
                     'l_max_its = 2000', 'nl_max_its = 200', "'lu mumps'",
                     "expression = 'max(strain_yy,0)'", 'if(s>0,1,0)']:
        assert required in text, required
    RUN.mkdir(exist_ok=True)
    (RUN / 'forward_and_adjoint.i').write_text(text)
    shutil.copy2(PREVIOUS / 'unrectified/optimize.i', RUN / 'optimize.i')
    shutil.copy2(HERE / 'data/measurement_data.csv', RUN / 'measurement_data.csv')
    shutil.copy2(HERE / 'data/solver_times.csv', RUN / 'solver_times.csv')
    np.savetxt(RUN / 'initial_alpha.txt', alpha)
    (RUN / 'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    # The full-window initial solve uses exactly the same physical model and
    # initial alpha. Crop its signed simulation and apply the requested operator.
    old = pd.read_csv(PREVIOUS / 'initial_forward_only/unrectified/strain_initial.csv')
    old = old[old.measurement_ycoord.between(y.min()-1e-10, y.max()+1e-10)].reset_index(drop=True)
    assert len(old) == len(frame)
    for column in ['measurement_time', 'measurement_xcoord', 'measurement_ycoord']:
        np.testing.assert_allclose(old[column], frame[column], rtol=0, atol=1e-10)
    frame['simulation_values'] = np.maximum(old.simulation_values, 0)
    frame['misfit_values'] = np.where(frame.measurement_time > 0, frame.simulation_values-frame.measurement_values, 0.)
    assert np.isfinite(frame.simulation_values).all() and np.any(frame.simulation_values > 0)
    frame.to_csv(RUN / 'strain_initial_reused.csv', index=False)
    audit = dict(native_shape=list(signed.shape), solver_shape=list(sampled.shape),
                 solver_steps=129, max_solver_steps=240, last_time_s=float(times[-1]),
                 negative_fraction=float(np.mean(signed < 0)), interpolation_exact=True,
                 initial_forward_source=str(PREVIOUS / 'initial_forward_only/unrectified'),
                 initial_data_objective=float(.5*np.sum(frame.misfit_values**2)),
                 initial_simulation_max=float(frame.simulation_values.max()),
                 regularization_note=cfg['regularization_scope'])
    audit['input_sha256'] = {name: hashlib.sha256((RUN/name).read_bytes()).hexdigest()
                             for name in ['forward_and_adjoint.i', 'optimize.i', 'measurement_data.csv', 'config.json']}
    (RUN / 'preflight.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
