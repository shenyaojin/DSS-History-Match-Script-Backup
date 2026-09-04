"""D1 AMEND: full blind diagnostics for the BEST five-gauge two_zone fits.

The cold-start ensemble (`d1_amend_coldstart.py`) reports each subset's best
calibration criterion and the blind RMSE that goes with it, but not the
amplitude ratio, arrival error or the pre-registered pass/fail flag. Those are
what the README's per-gauge tables quote, so they have to be recomputed at the
best-of-search parameters rather than left at the warm-started ones -- otherwise
the tables and the amended headline would disagree.

Emits one CSV with, per (norm, held-out gauge): the best cold-start five-gauge
criterion, its parameters, and the full blind diagnostics of that fit, together
with the published warm-started values and the spread across restarts that are
within 1% of the best on the five gauges they were given.

    python scripts/manuscript_well_leakage/rev2/d1_amend_rescore.py \
        --config configs/rev2/d1_amend.json
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'manuscript_well_leakage',
                                'baseline_calibration'))
sys.path.insert(0, HERE)

import d1_loo_blind as d1        # noqa: E402
import rev2_manifest as rm       # noqa: E402
import r1_calibration_core as core  # noqa: E402
from r1_run_calibration import load_config  # noqa: E402
from d1_amend_coldstart import write_amend_manifest  # noqa: E402

COLDSTART_JSON = 'output/rev2_20260901/D1/amend/d1_coldstart_ensemble_v2.json'
SUMMARY_CSV = 'output/rev2_20260901/D1/d1_summary_v1.csv'
OUT_CSV = 'output/rev2_20260901/D1/amend/d1_twozone_bestfit_blind_v2.csv'
OUT_MAN = 'output/rev2_20260901/D1/amend/manifest_bestfit_rescore.json'


def log(m):
    print(f"[d1-amend-rescore] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg, _ = load_config(args.config)
    rm.assert_absent([OUT_CSV, OUT_MAN])

    S = d1.build_setup(cfg)
    gnums = [t['gauge'] for t in S['targets']]
    acc = cfg['acceptance']
    alo, ahi = acc['amplitude_ratio_band']
    thr = float(acc['max_normalised_rmse'])

    D = json.load(open(COLDSTART_JSON))
    rec = {(r['norm'], r['held_out_gauge']): r for r in D['reconciled_per_gauge']}
    with open(SUMMARY_CSV) as fh:
        pub = {(r['norm'], int(r['held_out_gauge'])): r
               for r in csv.DictReader(fh) if r['model'] == 'two_zone'}
    with open(SUMMARY_CSV) as fh:
        floors = {int(r['held_out_gauge']): float(r['single_gauge_floor_rmse_psi'])
                  for r in csv.DictReader(fh)
                  if r['model'] == 'uniform' and r['norm'] == 'absolute'}

    rows, taxis = [], None
    for norm in ('absolute', 'normalised'):
        for g in gnums:
            k = gnums.index(g)
            b = D['best_of_search'][f'{norm}|drop_g{g}']
            sc, _sims, _prof, _n = d1.full_score(S, 'two_zone', b['params'], cfg)
            r = sc[k]
            usable = bool(alo <= r['amplitude_ratio'] <= ahi
                          and r['rmse_normalised'] <= thr)
            rr = rec[(norm, g)]
            rows.append({
                'norm': norm, 'held_out_gauge': g,
                'distance_ft': r['distance_ft'], 'obs_max_psi': round(r['obs_max_psi'], 4),
                'best_coldstart_criterion': f"{b['criterion']:.6f}",
                'published_warmstart_criterion': f"{b['published_warmstart_criterion']:.6f}",
                'coldstart_beats_published': b['coldstart_beats_published'],
                'params_refit_5gauge': ";".join(f"{x:.5g}" for x in b['params_physical']),
                'blind_rmse_psi': round(r['rmse_psi'], 4),
                'blind_rmse_norm': round(r['rmse_normalised'], 5),
                'blind_amp_ratio': round(r['amplitude_ratio'], 4),
                'blind_bias_psi': round(r['bias_psi'], 4),
                'blind_arr_err_rel_s': round(r['arrival_err_relative_s'], 1),
                'blind_arr_err_abs10_s': round(r['arrival_err_abs10_s'], 1),
                'blind_arr_err_abs25_s': round(r['arrival_err_abs25_s'], 1),
                'single_gauge_floor_rmse_psi': round(floors[g], 4),
                'blind_over_floor': round(r['rmse_psi'] / floors[g], 4),
                'usable_blind_prediction': usable,
                'published_warmstart_blind_rmse_psi': float(pub[(norm, g)]['blind_rmse_psi']),
                'blind_rmse_psi_min_within_1pct': round(
                    rr['blind_rmse_psi_min_within_1pct_of_best'], 4),
                'blind_rmse_psi_max_within_1pct': round(
                    rr['blind_rmse_psi_max_within_1pct_of_best'], 4),
                'n_restarts_within_1pct': rr['n_restarts_within_1pct_of_best'],
            })
            log(f"{norm:11s} g{g} crit={b['criterion']:.5f} blind={r['rmse_psi']:7.3f} "
                f"({100*r['rmse_normalised']:.1f}% of peak) amp={r['amplitude_ratio']:.3f} "
                f"usable={usable}")

    cols = list(rows[0].keys())
    with open(OUT_CSV, 'w') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    log(f"wrote {OUT_CSV} ({len(rows)} rows)")

    # aggregates recomputed here so the CSV and the README cannot drift apart
    agg = {}
    for norm in ('absolute', 'normalised'):
        rs = [r for r in rows if r['norm'] == norm]
        b = np.array([r['blind_rmse_psi'] for r in rs])
        mn = np.array([r['blind_rmse_psi_min_within_1pct'] for r in rs])
        mx = np.array([r['blind_rmse_psi_max_within_1pct'] for r in rs])
        agg[norm] = {
            'loo_blind_best_of_search_psi': float(np.sqrt(np.mean(b ** 2))),
            'loo_blind_envelope_within_1pct_psi': [float(np.sqrt(np.mean(mn ** 2))),
                                                   float(np.sqrt(np.mean(mx ** 2)))],
            'n_usable_blind': int(sum(r['usable_blind_prediction'] for r in rs)),
            'blind_rmse_over_peak_max': float(max(r['blind_rmse_norm'] for r in rs)),
            'blind_over_floor_range': [float(min(r['blind_over_floor'] for r in rs)),
                                       float(max(r['blind_over_floor'] for r in rs))],
        }
        log(f"AGG {norm:11s} best-of-search {agg[norm]['loo_blind_best_of_search_psi']:.3f} psi; "
            f"envelope {agg[norm]['loo_blind_envelope_within_1pct_psi'][0]:.3f}-"
            f"{agg[norm]['loo_blind_envelope_within_1pct_psi'][1]:.3f} psi; "
            f"{agg[norm]['n_usable_blind']}/6 usable; "
            f"blind/floor {agg[norm]['blind_over_floor_range'][0]:.2f}-"
            f"{agg[norm]['blind_over_floor_range'][1]:.2f}")

    taxis, _ = core.solve_forward(
        S['mesh'], d1._profile(S, 'two_zone',
                               np.array(cfg['families']['two_zone']['warm_start'], float)),
        S['dt'], S['t_total'], S['src']['taxis'], S['src']['delta_psi'],
        S['source_idx'], record_idx=[0])
    write_amend_manifest(
        OUT_MAN, cfg, args.config, S, taxis,
        study_id='d1_amend_bestfit_rescore',
        results={'aggregate': agg, 'rows': rows,
                 'acceptance_rule': acc,
                 'n_forward_solves': 12},
        outputs=[(OUT_CSV, 'csv', None)],
        notes=['Full blind diagnostics at the BEST five-gauge two_zone fits found '
               'by the cold-start ensemble, so the per-gauge tables in the D1 '
               'README match the amended headline.',
               'blind_over_floor uses the UNIFORM single-gauge floor from '
               'd1_summary_v1.csv (absolute norm), the same denominator the v1 '
               'README used.'],
        extra_inputs=[(COLDSTART_JSON, 'prior_run_output', 'coldstart_ensemble')])
    log(f"wrote {OUT_MAN}")


if __name__ == '__main__':
    main()
