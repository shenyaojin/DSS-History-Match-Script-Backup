"""Bounded global coupling refit; central four channels have 2x SSE weight.

Reuses completed physical forwards. Does not change permeability or MOOSE inputs.
Run from any directory with the existing MOOSE Python environment.
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

import refit_coupling as qc

HERE = qc.HERE
OUT = HERE / 'coupling_refit_weighted_c005'
BOUNDS = (0.05, 1.0)
CORE_WEIGHT = 2.0


def main():
    global OUT, BOUNDS, CORE_WEIGHT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unbounded', action='store_true', help='Use the unconstrained least-squares optimum for C')
    parser.add_argument('--core-weight', type=float, default=CORE_WEIGHT)
    args = parser.parse_args()
    if not np.isfinite(args.core_weight) or args.core_weight <= 1:
        parser.error('--core-weight must be finite and greater than 1')
    CORE_WEIGHT = args.core_weight
    if args.unbounded:
        BOUNDS = None
        OUT = HERE / f'coupling_refit_weighted{CORE_WEIGHT:g}_unbounded'
    elif CORE_WEIGHT != 2:
        OUT = HERE / f'coupling_refit_weighted{CORE_WEIGHT:g}_c005'
    if OUT.exists():
        raise RuntimeError(f'Refusing to overwrite {OUT}')
    manifest = json.loads((HERE / 'manifest.json').read_text())
    old_c = float(manifest['fixed_theta'][2])
    frames = {a: pd.read_csv(HERE / f'bg_{abs(a)}/strain_prediction.csv')
              for a in [-21, -19, -18]}
    paths = [HERE / f'bg_{abs(a)}/strain_prediction.csv' for a in frames]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    canonical = pd.read_csv(HERE.parent / 'inversion/measurement_data.csv')
    keys = ['measurement_time', 'measurement_values', 'measurement_xcoord',
            'measurement_ycoord', 'measurement_zcoord']
    ys = np.sort(canonical.measurement_ycoord.unique())
    depths = 14376 + (ys - 50) / .3048
    zone = manifest['zone']
    core = (depths >= zone['lo_ft']) & (depths < zone['hi_ft'])
    assert len(depths) == 18 and core.sum() == 4
    channel_weights = pd.DataFrame(dict(depth_ft=depths, measurement_ycoord=ys,
        objective_weight=np.where(core, CORE_WEIGHT, 1.)))
    weight_map = dict(zip(ys, channel_weights.objective_weight))
    rows, fitted = [], {}
    for alpha, df in frames.items():
        np.testing.assert_allclose(df[keys], canonical[keys], rtol=1e-10, atol=1e-18)
        assert len(df) == 18 * 130 and df.measurement_time.nunique() == 130
        s = df.physical_simulation_values.to_numpy()
        d = df.measurement_values.to_numpy()
        w = df.measurement_ycoord.map(weight_map).to_numpy()
        use = df.measurement_time.to_numpy() > 0
        assert use.sum() == 18 * 129
        np.testing.assert_allclose(np.sum(w[use]), (14 + 4 * CORE_WEIGHT) * 129)
        assert np.isfinite([s, d, w]).all() and (s >= 0).all() and (d >= 0).all()
        np.testing.assert_allclose(df.simulation_values, old_c * s, rtol=1e-10, atol=1e-18)
        ss = float(np.sum(w[use] * s[use]**2))
        sd = float(np.sum(w[use] * s[use] * d[use]))
        assert ss > 0
        free_c = sd / ss
        refitted = alpha != -21
        c = float(free_c if BOUNDS is None else np.clip(free_c, *BOUNDS)) if refitted else old_c
        gradient = c * ss - sd
        # A previous C below the new bound is infeasible: error need not decrease.
        if refitted:
            eps = 1e-12 * max(abs(c * ss), abs(sd))
            if BOUNDS is None:
                assert abs(gradient) <= eps
            elif c == BOUNDS[0]:
                assert gradient >= -eps
            elif c == BOUNDS[1]:
                assert gradient <= eps
            else:
                assert abs(gradient) <= eps
            comparisons = [old_c, c - .001, c + .001] if BOUNDS is None else [*BOUNDS, np.clip(old_c, *BOUNDS)]
            for feasible_c in comparisons:
                assert np.sum(w[use]*(c*s[use]-d[use])**2) <= np.sum(w[use]*(feasible_c*s[use]-d[use])**2) + 1e-24
        row = dict(background_alpha=alpha, refitted=refitted,
            role=('unbounded weighted refit' if BOUNDS is None else 'bounded weighted refit') if refitted else 'historical reference',
            previous_C=old_c, fitted_C=c, unconstrained_optimum_C=free_c,
            coupling_at_bound=refitted and BOUNDS is not None and c in BOUNDS, derivative_dJ_dC=gradient)
        for name, coeff in [('before', old_c), ('after', c)]:
            residual = (coeff*s - d)[use]
            row[f'weighted_objective_{name}'] = .5 * float(np.sum(w[use]*residual**2))
            row[f'weighted_rms_{name}_microstrain'] = float(np.sqrt(np.sum(w[use]*residual**2)/w[use].sum())*1e6)
            row[f'unweighted_rms_{name}_microstrain'] = float(np.sqrt(np.mean(residual**2))*1e6)
            row[f'core_rms_{name}_microstrain'] = float(np.sqrt(np.mean(residual[w[use] == CORE_WEIGHT]**2))*1e6)
        rows.append(row)
        frame = df.copy()
        frame['simulation_values'] = c*s
        frame['misfit_values'] = np.where(use, c*s-d, 0.)
        frame['objective_weight'] = w
        fitted[alpha] = frame
    OUT.mkdir()
    channel_weights.to_csv(OUT / 'channel_weights.csv', index=False)
    for alpha, frame in fitted.items():
        frame.to_csv(OUT / f'bg_{abs(alpha)}_strain.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT / 'comparison.csv', index=False)
    result = dict(method='C=sum(w*s*d)/sum(w*s*s)' if BOUNDS is None else 'C=clip(sum(w*s*d)/sum(w*s*s), 0.05, 1)',
        objective='0.5 * sum(w * (C * positive_simulation - positive_observation)**2), t>0',
        coupling_bounds=None if BOUNDS is None else list(BOUNDS), core_weight=CORE_WEIGHT,
        central_channel_depths_ft=depths[core].tolist(), other_channel_weight=1.,
        scope='Only C refitted for bg=-19/-18; bg=-21 kept as historical reference',
        sample_count=18*129, weighted_sample_count=(14+4*CORE_WEIGHT)*129,
        regularization='Permeability and its regularization unchanged; no penalty on C',
        interpretation_limit='Existing pressure overshoot unresolved; scalar refit does not validate pressure numerics',
        source_sha256=hashes, results=rows)
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    qc.OUT = OUT
    qc.plot(frames, fitted, rows, manifest, core_weight=CORE_WEIGHT,
        heading=f'Bg -19/-18: global C {"unbounded" if BOUNDS is None else "in [0.05, 1]"}, central four channels weighted {CORE_WEIGHT:g}x | Bg -21: historical reference')
    for path, digest in hashes.items():
        assert hashlib.sha256(qc.Path(path).read_bytes()).hexdigest() == digest
    (OUT / 'COMPLETE').write_text('Weighted scalar refit complete; source forwards preserved\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
