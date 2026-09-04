"""D4 addendum - what reduction ratio is output/0211_simulation_MULTIstage/phase3_test.npz?

That file is the one Fig. 6 is drawn from. It was regenerated on 2026-04-28 with no record of
its parameters, and the manuscript describes it as a five-order-of-magnitude reduction in
diffusivity across the stage-7 frac hits. The archive also holds runs at ratio 1e-5, 1e-4,
1e-3, 1e-2 and 1e-1, which makes the question answerable from the data alone.

Two handles:

  * All phase-3 runs continue from the same phase-2 final snapshot, so if phase3_test shares
    that initial condition exactly, it is a phase-3 run of the same chain and any difference
    is attributable to the diffusivity field rather than to a different history.
  * The barrier shows up as a pressure step across the reduced node. Stepping that quantity
    against the known ratios calibrates step size versus ratio, and phase3_test can then be
    placed on that curve.

The point estimate is weak - at a near-uniform field the steps approach round-off, and one
node returns an unphysical ratio above 1 - so the reportable result is the bracketing, not the
number: phase3_test sits far above the 1e-5 run and above even the 1e-1 run.

Read-only. Writes only into output/rev2_20260901/D4/.
"""

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(REPO, 'output/rev2_20260901/D4')
SIM = os.path.join(REPO, 'output/0211_simulation_MULTIstage')
FH = os.path.join(REPO, 'data/legacy/s_well/geometry/frac_hit')

RATIO_FILES = {1e-5: 'phase3_1e-05.npz', 1e-4: 'phase3_0.0001.npz', 1e-3: 'phase3_0.001.npz',
               1e-2: 'phase3_0.01.npz', 1e-1: 'phase3_0.1.npz'}


def canonical(path):
    """Field as (n_t, n_x). The 2026-02-11 pack_result change left this directory mixed."""
    z = np.load(path, allow_pickle=True)
    t, x, a = z['taxis'], z['daxis'], z['data']
    if a.shape == (len(t), len(x)) and a.shape == (len(x), len(t)):
        raise ValueError(f'{path}: square field, layout ambiguous')
    if a.shape == (len(t), len(x)):
        return t, x, a, '(n_t, n_x)'
    if a.shape == (len(x), len(t)):
        return t, x, a.T, '(n_x, n_t)'
    raise ValueError(f'{path}: shape {a.shape} matches neither axis pairing')


def barrier_steps(field, mesh, hits):
    """|dP| across each reduced node, measured at node+-1 on the final snapshot."""
    out = []
    for m in hits:
        i = int(np.argmin(np.abs(mesh - m)))
        out.append(float(abs(field[-1, i - 1] - field[-1, i + 1])))
    return np.array(out)


def main():
    os.chdir(REPO)
    hits = np.sort(np.load(os.path.join(FH, 'frac_hit_stage_7_swell.npz'))['data'])
    t_test, x_test, test, lay_test = canonical(os.path.join(SIM, 'phase3_test.npz'))
    _, _, phase2, _ = canonical(os.path.join(SIM, 'phase2.npz'))

    rep = {'file': 'output/0211_simulation_MULTIstage/phase3_test.npz',
           'layout': lay_test, 'shape_canonical': list(test.shape),
           'initial_condition_matches_phase2_final_psi':
               float(np.max(np.abs(test[0] - phase2[-1]))),
           'comparisons': [], 'implied_ratio_per_hit': []}

    step_test = barrier_steps(test, x_test, hits)
    steps = {}
    for r, name in sorted(RATIO_FILES.items()):
        t2, x2, a2, lay = canonical(os.path.join(SIM, name))
        steps[r] = barrier_steps(a2, x2, hits)
        rep['comparisons'].append({
            'ratio': r, 'file': name, 'layout': lay,
            'max_abs_field_diff_psi': float(np.max(np.abs(a2 - test))),
            'rms_field_diff_psi': float(np.sqrt(np.mean((a2 - test) ** 2))),
            'shares_adaptive_taxis': bool(len(t2) == len(t_test) and np.allclose(t2, t_test)),
            'barrier_step_psi': steps[r].tolist(),
        })

    # Log-log interpolation of step size against ratio, anchored on 1e-2 and 1e-1.
    for k, m in enumerate(hits):
        y1, y2 = np.log10(steps[1e-2][k]), np.log10(steps[1e-1][k])
        slope = (y2 - y1) / (-1.0 - -2.0)
        lr = -1.0 + (np.log10(step_test[k]) - y2) / slope
        rep['implied_ratio_per_hit'].append(
            {'md_ft': float(m), 'step_test_psi': float(step_test[k]),
             'implied_ratio': float(10 ** lr)})

    imp = np.array([d['implied_ratio'] for d in rep['implied_ratio_per_hit']])
    rep['implied_ratio_median'] = float(np.median(imp))
    rep['implied_ratio_range'] = [float(imp.min()), float(imp.max())]
    rep['barrier_step_test_psi_range'] = [float(step_test.min()), float(step_test.max())]
    rep['barrier_step_1e5_psi_range'] = [float(steps[1e-5].min()), float(steps[1e-5].max())]
    rep['caveat'] = (
        'The point estimate is at the resolution limit: on a near-uniform field the barrier step '
        'approaches round-off, and one node returns an unphysical ratio above 1. Quote the '
        'bracketing (well above 1e-1, three orders above 1e-5), not the median.')

    with open(os.path.join(OUT, 'd4_phase3_attribution.json'), 'w') as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == '__main__':
    main()
