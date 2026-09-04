"""Turn d2_summary_v2.json into the tables D2 has to report. Read-only.

    python3 scripts/manuscript_well_leakage/rev2/d2_report_v2.py \
        --summary output/rev2_20260901/D2/d2_summary_v2.json
"""

import argparse
import json

VAR_ORDER = ['baseline', 'base_frachit', 'baseline_far', 'extrap_g1md',
             'extrap_frachit', 'src_g2', 'src_g3']
NORMS = ('abs', 'norm')


def g(ev, n):
    return next(x for x in ev['per_gauge'] if x['gauge'] == n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', required=True)
    a = ap.parse_args()
    S = json.load(open(a.summary))
    V = S['variants']
    names = [n for n in VAR_ORDER if n in V]

    print('=' * 100)
    print('T1  CONTINUITY - published parameter vectors evaluated in this pipeline')
    print(f"{'variant':14s} {'published key':28s} {'psi (this)':>11s} "
          f"{'psi (pub)':>11s} {'d':>9s} {'norm (this)':>11s} {'norm (pub)':>11s} {'d':>9s}")
    for c in S['continuity_vs_published']['checks']:
        print(f"{c['variant']:14s} {c['published_key']:28s} "
              f"{c['reproduced_rmse_psi']:11.4f} {c['published_rmse_psi']:11.4f} "
              f"{c['abs_diff_psi']:+9.2e} {c['reproduced_norm']:11.5f} "
              f"{c['published_norm']:11.5f} {c['norm_diff']:+9.2e}")

    print()
    print('=' * 100)
    print('T2  CALIBRATION OBJECTIVE by variant x model x norm '
          '(cold-start restart envelope in brackets)')
    print(f"{'variant':15s} {'model':17s} {'norm':5s} {'objective':>12s} "
          f"{'min':>10s} {'median':>10s} {'max':>10s} {'spread%':>8s} {'D or D_near/D_far':>26s}")
    for n in names:
        for model, f in V[n]['fits'].items():
            for w in NORMS:
                p = f['per_norm'][w]
                e = p.get('restart_envelope')
                if model == 'uniform':
                    dd = f"D={p['D']:.1f}"
                else:
                    dd = f"{p['D_near']:.0f} / {p['D_far']:.1f}"
                if e:
                    print(f"{n:15s} {model:17s} {w:5s} {p['objective']:12.5f} "
                          f"{e['min']:10.5f} {e['median']:10.5f} {e['max']:10.5f} "
                          f"{100 * e['spread_frac_of_min']:8.1f} {dd:>26s}")
                else:
                    band = p.get('band_10pct_D')
                    cens = ('  CENSORED' if p.get('band_censored_low')
                            or p.get('band_censored_high') else '')
                    print(f"{n:15s} {model:17s} {w:5s} {p['objective']:12.5f} "
                          f"{'':10s} {'':10s} {'':10s} {'':8s} {dd:>26s}"
                          f"   +10% band [{band[0]:.0f}, {band[1]:.0f}]{cens}")

    print()
    print('=' * 100)
    print('T3  THE NEGATIVE CONTROL - every scored row for GAUGE 1 (MD 16645)')
    print(f"{'variant':15s} {'model':17s} {'norm':5s} {'role':22s} {'dist ft':>8s} "
          f"{'RMSE psi':>9s} {'RMSE/peak':>9s} {'amp':>7s} {'arrival s':>10s} {'bias psi':>9s}")
    for n in names:
        for key, ev in V[n]['scores'].items():
            model, w = key.split('::')
            r = g(ev, 1)
            print(f"{n:15s} {model:17s} {w:5s} {r['role']:22s} {r['distance_ft']:8.0f} "
                  f"{r['rmse_psi']:9.2f} {r['rmse_normalised']:9.4f} "
                  f"{r['amplitude_ratio']:7.3f} {r['arrival_err_s']:10.2f} "
                  f"{r['bias_psi']:9.2f}")
    print('-' * 100)
    print('    no-solver reference predictors of the SAME target:')
    for r in S['naive_gauge1_predictors']:
        print(f"    {r['predictor']:26s} RMSE {r['rmse_psi']:8.2f} psi "
              f"({r['rmse_normalised']:.4f})  amp {r['amplitude_ratio']:.3f}  "
              f"arrival {r['arrival_err_s']:+.1f} s  bias {r['bias_psi']:+.1f} psi")

    print()
    print('=' * 100)
    print('T4  PER-GAUGE DEGRADATION, two_zone anchored at the frac-hit centroid')
    for w in NORMS:
        print(f"--- norm = {w}")
        head = f"{'variant':15s}" + ''.join(f"{'g%d' % i:>18s}" for i in range(1, 8))
        print(head)
        for kind, fmt in (('rmse_psi', '{:8.1f}'), ('amplitude_ratio', '{:8.3f}'),
                          ('arrival_err_s', '{:8.1f}')):
            print(f"  [{kind}]")
            for n in names:
                key = f'two_zone_frachit::{w}'
                if key not in V[n]['scores']:
                    continue
                ev = V[n]['scores'][key]
                cells = ''
                for i in range(1, 8):
                    r = g(ev, i)
                    tag = {'source_dirichlet': 'S', 'imposed_extrapolation': 'X',
                           'bc_constituent': 'b', 'calibration': 'c',
                           'blind': 'B'}[r['role']]
                    cells += f"{fmt.format(r[kind])}({tag})".rjust(18)
                print(f"{n:15s}{cells}")
        print()
    print('    role tags: S source Dirichlet, X imposed extrapolation, '
          'b BC constituent, c calibration, B BLIND')

    print()
    print('=' * 100)
    print('T5  DOMAIN-TOP DEPENDENCE of the blind gauge-1 prediction')
    print(f"{'variant':9s} {'model':17s} {'norm':5s} {'top MD':>8s} {'nx':>7s} "
          f"{'g1 RMSE':>9s} {'g1 amp':>7s} {'g1 arr':>8s} {'calib obj':>11s}")
    for r in S['domain_top_sensitivity']:
        print(f"{r['variant']:9s} {r['model']:17s} {r['norm']:5s} "
              f"{r['top_md_ft']:8.0f} {r['nx']:7d} {r['g1_rmse_psi']:9.2f} "
              f"{r['g1_amplitude_ratio']:7.3f} {r['g1_arrival_err_s']:8.1f} "
              f"{(r['calib_gauge_mean_rmse_psi'] if r['norm'] == 'abs' else r['calib_gauge_mean_norm']):11.5f}")

    print()
    print('=' * 100)
    print('T6  NORM FLIP CHECK - variant ranking by calibration-set misfit')
    for w in NORMS:
        key = 'gauge_mean_rmse_calib' if w == 'abs' else 'gauge_mean_norm_calib'
        rows = []
        for n in names:
            k = f'two_zone_frachit::{w}'
            if k in V[n]['scores']:
                rows.append((V[n]['scores'][k][key], n))
        rows.sort()
        print(f"  {w:5s}: " + ' < '.join(f"{n}({v:.4f})" for v, n in rows))
    print()
    print('  blind gauge-1 RMSE ranking (two_zone, variants where g1 is blind):')
    for w in NORMS:
        rows = []
        for n in names:
            k = f'two_zone_frachit::{w}'
            if k in V[n]['scores']:
                r = g(V[n]['scores'][k], 1)
                if r['role'] == 'blind':
                    rows.append((r['rmse_psi'], n))
        rows.sort()
        print(f"  {w:5s}: " + ' < '.join(f"{n}({v:.1f} psi)" for v, n in rows))


if __name__ == '__main__':
    main()
