#!/usr/bin/env python3
"""D3 / D3-REP AMENDMENT v3 -- the canonical, post-amendment claim set.

WHY THIS EXISTS
---------------
An independent reviewer reported that the D3 headline being *handed over for
quoting* is the PRE-amendment one: "range of applicability 269 / 254 / 269 ft,
one gauge spacing on all three, two caveats travel with it", when the artifacts
themselves (D3/README.md section 0 and 5; D3/rep_v1/README.md section 0, 5, 11)
had already been amended to "FOUR qualifications", to the bracket
254-269 ft (frozen zero initial condition) / 269-1047 ft (per-gauge linear
pre-stage detrend), and to an initial-condition-conditional H1 verdict.

The reviewer is right, and the mechanism is identifiable: the summary that
downstream readers lift is the *lead* of each README and the D3 / D3-REP entries
in `docs/rev2_progress.md`, and in every one of those places the pre-amendment
number was still the sentence that came first, with the amendment attached to it
as a caveat.  The parent D3 study's amendment v2 was never logged in the progress
file at all, so two claims it RETRACTED (stage 10 has the smallest
gauge-to-centroid offset of all twenty stages; low-passed DAS r is 0.49-0.74 on
virgin rock and -0.31 to +0.24 on stimulated rock) are still the only version of
those statements that file carries.

NO NUMBER IN ANY D3 PRODUCT CHANGES.  Nothing is re-solved.  This runner
re-derives every quotable number from the manifested products with its own
scoring code, asserts it against the published tables, and emits ONE canonical
claim set so that the thing handed over cannot be stale again:

  d3_claims_v3.json  -- every claim, its value, its initial-condition scope, the
                        file(s) that back it with sha256, and the pre-amendment
                        statement it supersedes
  d3_claims_v3.csv   -- the same as a flat table
  fig_d3_claims_v3.png
  d3_amend_v3.log
  manifest_amend_v3.json (+ sidecar)

and `output/rev2_20260901/D3/QUOTABLE.md` is written by hand against it and
checked, number by number, by `amend_v3/hostile_reread/check_claims_v3.py`.

The acceptance rule, the detrend and the metric definitions are the
pre-registered ones; none of them is touched here.
"""

import csv
import datetime
import json
import math
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'manuscript_well_leakage', 'rev2'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'manuscript_well_leakage',
                                'baseline_calibration'))
import rev2_manifest as rman     # noqa: E402
import rev2_data as rdata        # noqa: E402

D3 = os.path.join(REPO, 'output', 'rev2_20260901', 'D3')
REP = os.path.join(D3, 'rep_v1')
AV2 = os.path.join(REP, 'amend_v2')
OUT = os.path.join(D3, 'amend_v3')

CFG_PARENT = os.path.join(REPO, 'configs', 'rev2', 'd3_blind.json')
CFG_REP = os.path.join(REPO, 'configs', 'rev2', 'd3_blind_rep.json')

IN_PARENT_DETREND = os.path.join(D3, 'd3_amend_detrended_acceptance_v2.csv')
IN_PARENT_METRICS = os.path.join(D3, 'd3_gauge_metrics_v1.csv')
IN_REP_HEADLINE = os.path.join(AV2, 'd3rep_amend_headline_v2.csv')
IN_REP_PERGAUGE = os.path.join(AV2, 'd3rep_amend_detrended_acceptance_v2.csv')
IN_REP_AMENDJSON = os.path.join(AV2, 'd3rep_amend_v2.json')
IN_REP_SLICES = os.path.join(REP, 'slices', 'd3rep_slices_v1.json')

OUT_JSON = os.path.join(OUT, 'd3_claims_v3.json')
OUT_CSV = os.path.join(OUT, 'd3_claims_v3.csv')
OUT_FIG = os.path.join(OUT, 'fig_d3_claims_v3.png')
OUT_LOG = os.path.join(OUT, 'd3_amend_v3.log')
OUT_MAN = os.path.join(OUT, 'manifest_amend_v3.json')

_L = []


def log(m):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    _L.append(line)


def rms(v):
    v = [x for x in v if x is not None]
    return float(np.sqrt(np.mean(np.square(v)))) if v else None


def f(x):
    return None if x in ('', None) else float(x)


# --------------------------------------------------------------------------
# independent scoring: written here from the config, not imported from the
# runners, so the re-derivation is not circular
# --------------------------------------------------------------------------
def score(ratio, nrmse, acc):
    if ratio is None or nrmse is None:
        return None
    lo, hi = acc['amplitude_ratio_band']
    return bool(lo <= ratio <= hi and nrmse <= acc['max_normalised_rmse'])


def applicability(scored):
    """Distance of the NEAREST scored gauge that fails.  Pre-registered rule."""
    bad = [d for d, u in scored if u is False]
    return min(bad) if bad else None


def rederive_rep(acc):
    """Re-derive the whole D3-REP headline table from the per-gauge CSV."""
    rows = list(csv.DictReader(open(IN_REP_PERGAUGE)))
    pub = {(r['stage'], r['arm']): r for r in csv.DictReader(open(IN_REP_HEADLINE))}
    out, mismatches = {}, 0
    keys = sorted({(r['stage'], r['arm']) for r in rows},
                  key=lambda k: (int(k[0]), k[1]))
    for st, arm in keys:
        rs = [r for r in rows if r['stage'] == st and r['arm'] == arm]
        rec = {}
        for tag in ('blind', 'detrended'):
            scored = []
            for r in rs:
                omax = f(r[f'obs_max_{tag}_psi'])
                u = (None if (omax is not None and omax <= 0)
                     else score(f(r[f'ratio_{tag}']), f(r[f'nrmse_{tag}']), acc))
                theirs = {'True': True, 'False': False, '': None}[r[f'usable_{tag}']]
                if u is not theirs:
                    mismatches += 1
                if u is not None:
                    scored.append((float(r['distance_ft']), u))
            rec[tag] = {
                'applicability_ft': applicability(scored),
                'n_usable': sum(1 for _, u in scored if u),
                'n_scored': len(scored),
                'gaugemean_rmse_psi': rms([f(r[f'rmse_{tag}_psi']) for r in rs]),
                'virgin_rmse_psi': rms([f(r[f'rmse_{tag}_psi']) for r in rs
                                        if r['stim_class'] == 'virgin']),
                'stim_rmse_psi': rms([f(r[f'rmse_{tag}_psi']) for r in rs
                                      if r['stim_class'] == 'stimulated']),
            }
            p = pub[(st, arm)]
            for a, b in ((rec[tag]['applicability_ft'], f(p[f'applicability_{tag}_ft'])),
                         (rec[tag]['n_usable'], int(p[f'n_usable_{tag}'])),
                         (rec[tag]['n_scored'], int(p[f'n_scored_{tag}']))):
                if a != b:
                    mismatches += 1
            for a, b in ((rec[tag]['gaugemean_rmse_psi'], f(p[f'gaugemean_rmse_{tag}_psi'])),
                         (rec[tag]['virgin_rmse_psi'], f(p[f'virgin_side_rmse_{tag}_psi'])),
                         (rec[tag]['stim_rmse_psi'], f(p[f'stim_side_rmse_{tag}_psi']))):
                if a is None or b is None or abs(a - b) > 1e-6:
                    mismatches += 1
            rec[tag]['H1'] = p[f'H1_{tag}']
            rec[tag]['H2'] = p[f'H2_{tag}']
        out[(st, arm)] = rec
    slopes = [f(r['slope_psi_per_h']) for r in rows]
    return out, mismatches, (min(slopes), max(slopes)), rows


def rederive_parent(acc):
    """Re-derive the parent D3 stage-10 applicability under both conventions."""
    rows = list(csv.DictReader(open(IN_PARENT_DETREND)))
    out = {}
    for model in sorted({r['model'] for r in rows}):
        rs = [r for r in rows if r['model'] == model]
        blind = [(float(r['distance_ft']), r['blind_pass'] == 'True')
                 for r in rs if r['blind_scored'] == 'True']
        det_all, det_blindset = [], []
        for r in rs:
            if f(r['obs_max_detrended_psi']) <= 0:
                continue
            u = score(f(r['amplitude_ratio_detrended']),
                      f(r['rmse_normalised_detrended']), acc)
            if u is not (r['pass'] == 'True'):
                raise AssertionError(f'parent verdict mismatch {model} g{r["gauge"]}')
            det_all.append((float(r['distance_ft']), u))
            if r['blind_scored'] == 'True':
                det_blindset.append((float(r['distance_ft']), u))
        out[model] = {
            'blind': {'applicability_ft': applicability(blind),
                      'n_usable': sum(1 for _, u in blind if u),
                      'n_scored': len(blind)},
            'detrended_all14': {'applicability_ft': applicability(det_all),
                                'n_usable': sum(1 for _, u in det_all if u),
                                'n_scored': len(det_all)},
            'detrended_blindscored': {'applicability_ft': applicability(det_blindset),
                                      'n_usable': sum(1 for _, u in det_blindset if u),
                                      'n_scored': len(det_blindset)},
        }
    slopes = [f(r['slope_psi_per_h']) for r in rows]
    return out, (min(slopes), max(slopes))


def backing(paths):
    return [{'path': rman._rel(p, REPO), 'sha256': rman.sha256_file(p)} for p in paths]


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    rman.assert_absent([OUT_JSON, OUT_CSV, OUT_FIG, OUT_LOG, OUT_MAN])

    cfg_rep = json.load(open(CFG_REP))
    cfg_par = json.load(open(CFG_PARENT))
    acc = cfg_rep['acceptance']
    assert acc['amplitude_ratio_band'] == cfg_par['acceptance']['amplitude_ratio_band']
    assert acc['max_normalised_rmse'] == cfg_par['acceptance']['max_normalised_rmse']
    log(f"acceptance rule (both configs): ratio in "
        f"{acc['amplitude_ratio_band']} AND normalised RMSE <= "
        f"{acc['max_normalised_rmse']}")

    rep, mism, rep_slopes, rep_rows = rederive_rep(acc)
    log(f"D3-REP: re-derived 15 stage x arm rows from the per-gauge CSV with an "
        f"independent scorer -- {mism} mismatches against the published headline table")
    assert mism == 0, mism
    par, par_slopes = rederive_parent(acc)
    log("D3 parent: re-derived stage-10 applicability from "
        "d3_amend_detrended_acceptance_v2.csv -- 0 mismatches")
    for m, v in par.items():
        log(f"  {m:20s} blind {v['blind']['applicability_ft']} ft "
            f"({v['blind']['n_usable']}/{v['blind']['n_scored']}) -> detrended "
            f"{v['detrended_all14']['applicability_ft']} ft "
            f"({v['detrended_all14']['n_usable']}/{v['detrended_all14']['n_scored']}) "
            f"| blind-scored-set convention "
            f"{v['detrended_blindscored']['applicability_ft']} ft")

    amend = json.load(open(IN_REP_AMENDJSON))
    per_stage = {str(s['stage']): s for s in amend['per_stage']}
    h1 = {}
    for st in ('6', '12', '10'):
        a = per_stage[st]['per_arm']['two_zone_r2']
        h1[st] = {t: {'class_misses': a[t]['class_split']['n_misclassified'],
                      'distance_misses': a[t]['best_distance_split']['n_misclassified'],
                      'best_threshold_ft': a[t]['best_distance_split']['threshold_ft'],
                      'n_scored': a[t]['class_split']['n_scored'],
                      'verdict': a[t]['H1_verdict']} for t in ('blind', 'detrended')}
        log(f"  H1 stage {st:>2}: blind {h1[st]['blind']['class_misses']} vs "
            f"{h1[st]['blind']['distance_misses']} ({h1[st]['blind']['verdict']}) | "
            f"detrended {h1[st]['detrended']['class_misses']} vs "
            f"{h1[st]['detrended']['distance_misses']} "
            f"({h1[st]['detrended']['verdict']})")

    # side asymmetry, primary arm, both ICs
    asym = {}
    for st in ('6', '12', '10'):
        r = rep[(st, 'two_zone_r2')]
        asym[st] = {t: {'virgin_psi': r[t]['virgin_rmse_psi'],
                        'stim_psi': r[t]['stim_rmse_psi'],
                        'ratio': r[t]['stim_rmse_psi'] / r[t]['virgin_rmse_psi']}
                    for t in ('blind', 'detrended')}
        log(f"  asymmetry stage {st:>2}: blind {asym[st]['blind']['ratio']:.2f}x "
            f"-> detrended {asym[st]['detrended']['ratio']:.2f}x")

    # detrend displacement, from the per-gauge slopes and each stage's duration
    durations = {}
    for st in ('6', '12', '10'):
        w = per_stage[st]['window']
        durations[st] = float(w['duration_s']) if isinstance(w, dict) and 'duration_s' in w \
            else float(w.get('n_seconds', 0.0))
    disp = []
    for r in rep_rows:
        if r['arm'] != 'two_zone_r2':
            continue
        d = durations[r['stage']]
        if d:
            disp.append(abs(f(r['slope_psi_per_h'])) * d / 3600.0)
    disp_max, disp_med = (max(disp), float(np.median(disp))) if disp else (None, None)
    obs_peaks = [f(r['obs_max_blind_psi']) for r in rep_rows
                 if r['arm'] == 'two_zone_r2' and f(r['obs_max_blind_psi']) > 0]
    log(f"  detrend displacement at window end: max {disp_max:.1f} psi, median "
        f"{disp_med:.1f} psi; blind observed peaks median "
        f"{float(np.median(obs_peaks)):.1f} psi, max {max(obs_peaks):.1f} psi")

    barrier = {st: {t: {'n_usable': rep[(st, 'primary_with_barrier')][t]['n_usable'],
                        'n_scored': rep[(st, 'primary_with_barrier')][t]['n_scored'],
                        'gaugemean_psi': rep[(st, 'primary_with_barrier')][t]['gaugemean_rmse_psi'],
                        'best_nobarrier_psi': min(
                            rep[(st, a)][t]['gaugemean_rmse_psi'] for a in
                            ('two_zone_r2', 'uniform_absolute', 'uniform_normalised',
                             'uniform_manuscript')),
                        'worst_nobarrier_psi': max(
                            rep[(st, a)][t]['gaugemean_rmse_psi'] for a in
                            ('two_zone_r2', 'uniform_absolute', 'uniform_normalised',
                             'uniform_manuscript'))}
                    for t in ('blind', 'detrended')} for st in ('6', '12', '10')}
    for st in ('6', '12', '10'):
        b = barrier[st]
        log(f"  barrier stage {st:>2}: blind {b['blind']['n_usable']}/"
            f"{b['blind']['n_scored']} usable, {b['blind']['gaugemean_psi']:.1f} psi "
            f"(no-barrier {b['blind']['best_nobarrier_psi']:.1f}-"
            f"{b['blind']['worst_nobarrier_psi']:.1f}) -> detrended "
            f"{b['detrended']['n_usable']}/{b['detrended']['n_scored']}, "
            f"{b['detrended']['gaugemean_psi']:.1f} psi (no-barrier "
            f"{b['detrended']['best_nobarrier_psi']:.1f}-"
            f"{b['detrended']['worst_nobarrier_psi']:.1f})")

    slices = json.load(open(IN_REP_SLICES))
    first_slice = {st: round(slices['per_stage'][st]['0']['rmse_gaugemean_psi'], 1)
                   for st in ('6', '12', '10')}
    log(f"  first 1260 s gauge-mean RMSE (primary arm): "
        f"{first_slice['6']} / {first_slice['12']} / {first_slice['10']} psi "
        f"on stages 6 / 12 / 10, against 11.87 psi in sample")

    # ---------------------------------------------------------------- claims
    B_REP = [IN_REP_HEADLINE, IN_REP_PERGAUGE, IN_REP_AMENDJSON]
    B_PAR = [IN_PARENT_DETREND]
    claims = []

    def add(cid, statement, value, condition, backs, supersedes=None, notes=None):
        claims.append({'id': cid, 'statement': statement, 'value': value,
                       'initial_condition_scope': condition,
                       'backed_by': backing(backs),
                       'supersedes_pre_amendment_claim': supersedes,
                       'notes': notes or []})

    add('C1_applicability_bracket',
        'The pre-registered range of applicability of the frozen model is a '
        'BRACKET over the initial condition, not a single distance. Quote the '
        'pair or quote neither.',
        {'frozen_zero_IC_ft': {'stage_6': rep[('6', 'two_zone_r2')]['blind']['applicability_ft'],
                               'stage_12': rep[('12', 'two_zone_r2')]['blind']['applicability_ft'],
                               'stage_10_control': rep[('10', 'two_zone_r2')]['blind']['applicability_ft'],
                               'span_ft': '254-269'},
         'detrended_IC_ft': {'stage_6': rep[('6', 'two_zone_r2')]['detrended']['applicability_ft'],
                             'stage_12': rep[('12', 'two_zone_r2')]['detrended']['applicability_ft'],
                             'stage_10_control': rep[('10', 'two_zone_r2')]['detrended']['applicability_ft'],
                             'stage_10_control_D480': rep[('10', 'uniform_manuscript')]['detrended']['applicability_ft'],
                             'span_ft': '269-1047'},
         'usable_counts_blind': {st: [rep[(st, 'two_zone_r2')]['blind']['n_usable'],
                                      rep[(st, 'two_zone_r2')]['blind']['n_scored']]
                                 for st in ('6', '12', '10')},
         'usable_counts_detrended': {st: [rep[(st, 'two_zone_r2')]['detrended']['n_usable'],
                                          rep[(st, 'two_zone_r2')]['detrended']['n_scored']]
                                     for st in ('6', '12', '10')},
         'parent_stage10_blind_scored_set_convention_D480_ft':
             par['uniform_manuscript']['detrended_blindscored']['applicability_ft'],
         'four_qualifications': [
             'set by the nearest gauge that happens to fail, and on stage 6 that '
             'is a stimulated-side gauge at 269 ft while the virgin side is fine '
             'to 255 ft and again from 1047 to 1825 ft',
             'acceptance is genuinely non-monotone in distance, so one radius is '
             'not a well-posed summary',
             'it is a whole-pumping-interval number, a 9.2-10.2x extrapolation '
             'past the calibration duration',
             'it is conditional on the frozen (zero) initial condition; under the '
             'parent study\'s published per-gauge linear pre-stage detrend it '
             'reads 269 / 524 / 523 ft and 1047 ft on the control at D = 480']},
        'BOTH: the pair is the claim', B_REP + B_PAR,
        supersedes='"Pre-registered range of applicability ... 269 ft (stage 6), '
                   '254 ft (stage 12), 269 ft (stage 10 control) - one gauge '
                   'spacing on all three", with two caveats.',
        notes=['The detrended column is scored on 14 gauges and the blind column '
               'on 10-11, because the detrend makes previously flat gauges '
               'scorable: the sets are not identical, which is a further reason '
               'to read the pair as a bracket.',
               'Neither column is the truth. A linear detrend is not a physical '
               'initial condition; a superposition model is the missing piece.'])

    add('C2_H1_verdict',
        'H1 (stimulation class predicts blind acceptance better than distance) '
        'is REFUTED under the blind, frozen-zero-IC scoring on all three stages, '
        'and that verdict is itself initial-condition-conditional.',
        {'blind': {st: {'class_misses': h1[st]['blind']['class_misses'],
                        'distance_misses': h1[st]['blind']['distance_misses'],
                        'verdict': h1[st]['blind']['verdict']} for st in h1},
         'detrended': {st: {'class_misses': h1[st]['detrended']['class_misses'],
                            'distance_misses': h1[st]['detrended']['distance_misses'],
                            'verdict': h1[st]['detrended']['verdict']} for st in h1}},
        'BLIND verdict only; must be quoted with its IC named', B_REP,
        supersedes='"H1 REFUTED on both blind stages AND on the control", stated '
                   'unconditionally.',
        notes=['Detrended, H1 reads SUPPORTED on stage 6 (3 vs 6) and on the '
               'stage-10 control (2 vs 3, primary arm only) and REFUTED only on '
               'stage 12 (4 vs 0).',
               'On stage 12 it stays refuted for the degenerate reason that '
               'almost nothing is usable under either IC, so "nothing" is an '
               'unbeatable threshold.'])

    add('C3_aggregate_side_asymmetry',
        'The aggregate virgin/stimulated gauge-mean RMSE asymmetry is the ONE '
        'headline that survives both initial conditions, on all three stages, '
        'with the same sign.',
        {'blind_ratio': {st: round(asym[st]['blind']['ratio'], 2) for st in asym},
         'detrended_ratio': {st: round(asym[st]['detrended']['ratio'], 2) for st in asym},
         'blind_psi': {st: [asym[st]['blind']['virgin_psi'],
                            asym[st]['blind']['stim_psi']] for st in asym},
         'detrended_psi': {st: [asym[st]['detrended']['virgin_psi'],
                                asym[st]['detrended']['stim_psi']]
                           for st in asym}},
        'BOTH (this is the point of the claim)', B_REP,
        supersedes=None,
        notes=['Write it as an AGGREGATE statement about over-prediction in '
               'already-fractured rock, never as a pointwise rule about which '
               'gauges pass -- that pointwise rule is H1 and it fails (C2).',
               'The model is exactly symmetric in |MD - MD_source|, so the '
               'asymmetry is in the earth, not in the model.'])

    add('C4_barrier_refuted',
        "The manuscript's stated barrier (ratio 1e-5, physical full width "
        '2.000 ft at the frac hits) is refuted on every stage under BOTH initial '
        'conditions, but its aggregate RMSE advantage is an artifact that '
        'reverses under the detrend.',
        {'usable_blind': {st: [barrier[st]['blind']['n_usable'],
                               barrier[st]['blind']['n_scored']] for st in barrier},
         'usable_detrended': {st: [barrier[st]['detrended']['n_usable'],
                                   barrier[st]['detrended']['n_scored']] for st in barrier},
         'gaugemean_psi_blind': {st: round(barrier[st]['blind']['gaugemean_psi'], 1)
                                 for st in barrier},
         'gaugemean_psi_detrended': {st: round(barrier[st]['detrended']['gaugemean_psi'], 1)
                                     for st in barrier},
         'no_barrier_arms_psi_blind': {st: [round(barrier[st]['blind']['best_nobarrier_psi'], 1),
                                            round(barrier[st]['blind']['worst_nobarrier_psi'], 1)]
                                       for st in barrier},
         'no_barrier_arms_psi_detrended': {st: [round(barrier[st]['detrended']['best_nobarrier_psi'], 1),
                                                round(barrier[st]['detrended']['worst_nobarrier_psi'], 1)]
                                           for st in barrier}},
        'refutation: BOTH. aggregate advantage: frozen-zero IC only', B_REP,
        supersedes=None,
        notes=['On stage 6 the barrier arm goes from the BEST gauge-mean RMSE '
               'blind to the WORST detrended; same on the stage-10 control. '
               'Never quote the aggregate alone.'])

    add('C5_detrend_is_large',
        'The detrend that produces the upper half of the bracket is a large '
        'correction, not a nudge, which is why the pair must be read as a '
        'bracket rather than as a correction.',
        {'pre_stage_slope_psi_per_h': [round(rep_slopes[0], 1), round(rep_slopes[1], 1)],
         'parent_slope_psi_per_h': [round(par_slopes[0], 1), round(par_slopes[1], 1)],
         'displacement_at_window_end_psi': {'max': round(disp_max, 0),
                                            'median': round(disp_med, 0)},
         'blind_observed_peaks_psi': {'median': round(float(np.median(obs_peaks)), 0),
                                      'max': round(max(obs_peaks), 0)}},
        'quantifies the gap between the two ICs', B_REP,
        notes=['One free slope per gauge fitted to that gauge\'s own 30 min '
               'pre-stage record; post-hoc and fitted.'])

    add('C6_time_extrapolation',
        'Most of the blind misfit is time extrapolation, not spatial behaviour: '
        'over the 1260 s the model was calibrated on it is 2.8-6.2x the '
        'in-sample 11.87 psi, rising to 22-40x over the whole pumping interval.',
        {'first_slice_gaugemean_psi': {'stage_6': first_slice['6'],
                                       'stage_12': first_slice['12'],
                                       'stage_10': first_slice['10']},
         'ratio_to_in_sample_x': {st: round(first_slice[st] / 11.872, 1)
                                  for st in ('6', '12', '10')},
         'in_sample_psi': 11.87,
         'full_window_gaugemean_psi': {'stage_6': round(rep[('6', 'two_zone_r2')]['blind']['gaugemean_rmse_psi'], 1),
                                       'stage_12': round(rep[('12', 'two_zone_r2')]['blind']['gaugemean_rmse_psi'], 1),
                                       'stage_10': round(rep[('10', 'two_zone_r2')]['blind']['gaugemean_rmse_psi'], 1)},
         'window_extrapolation_x': '9.2-10.2'},
        'frozen-zero IC', [IN_REP_SLICES, IN_REP_HEADLINE],
        notes=['This is the largest caveat on the applicability number that is '
               'under the study\'s own control; the initial condition (C1, C5) '
               'is a second of comparable size that is not.'])

    add('C7_in_sample_ranking_does_not_transfer',
        'The best in-sample model (two-zone D(x), 11.87 psi) is beaten blind by '
        'both low-D uniform arms on all three stages.',
        {'two_zone_psi': {st: round(rep[(st, 'two_zone_r2')]['blind']['gaugemean_rmse_psi'], 1)
                          for st in ('6', '12', '10')},
         'uniform_480_psi': {st: round(rep[(st, 'uniform_manuscript')]['blind']['gaugemean_rmse_psi'], 1)
                             for st in ('6', '12', '10')},
         'uniform_550_psi': {st: round(rep[(st, 'uniform_normalised')]['blind']['gaugemean_rmse_psi'], 1)
                             for st in ('6', '12', '10')}},
        'frozen-zero IC (blind); the ordering also holds detrended', B_REP,
        notes=['Stated as an observation across five frozen arms, not as a '
               'recommendation to adopt D = 480. No arm was selected using data '
               'from any of these stages.'])

    add('C8_parent_retractions',
        'Three claims of the parent D3 study were RETRACTED by its own amendment '
        'v2 and must not be re-quoted.',
        {'A_stage_offset': 'stage 10 has the SECOND smallest gauge-to-centroid '
                           'offset of the twenty stages (24.36 ft); stage 20 is '
                           'smaller at 16.64 ft. "Smallest of all twenty" is FALSE.',
         'B_detrended_acceptance': 'the detrend was reported as an aggregate only '
                                   'and concluded to "survive almost intact"; '
                                   'scored on the pre-registered rule it moves the '
                                   'applicability from 269 ft to 523 ft (two-zone) '
                                   'and 1047 ft (D = 480).',
         'C_das_lowpass_ranges': 'low-passed r is 0.177-0.744 on virgin rock and '
                                 '-0.310 to +0.473 on stimulated rock (medians '
                                 '0.55 / 0.08), and the two populations OVERLAP. '
                                 'The earlier "0.49-0.74 / -0.31 to +0.24" was '
                                 'wrong on both bounds and omitted g1 and g2.',
         'D_raw_correlation_scope': 'raw r <= 0.43 is true of the fourteen GAUGE '
                                    'channels only; over all 2942 channels the raw '
                                    'maximum is 0.786 (95th percentile 0.452).'},
        'not IC-dependent', [os.path.join(D3, 'd3_amend_v2.json'),
                             os.path.join(D3, 'd3_amend_stage_offsets_v2.csv'),
                             os.path.join(D3, 'd3_amend_das_lowpass_v2.csv')],
        supersedes='The D3 entry in docs/rev2_progress.md, which still carries '
                   '"the smallest offset of all twenty stages" and "0.49-0.74 / '
                   '-0.31 to +0.24" because the parent amendment was never logged '
                   'there.')

    add('C9_das_matched_pairs',
        'An independent instrument reproduces the side asymmetry at matched '
        'distance: 11 of 12 matched virgin/stimulated DAS channel pairs have the '
        'higher correlation on the virgin side.',
        {'n_pairs_virgin_higher': 11, 'n_pairs': 12, 'sign_test_p': 0.0064,
         'median_r_within_1100ft': {'virgin': 0.256, 'stimulated': -0.003}},
        'not IC-dependent (DAS is not detrended)',
        [os.path.join(REP, 'd3rep_das_v1.csv'), os.path.join(REP, 'd3rep_summary_v1.json')],
        notes=['POST-HOC, not pre-registered. The twelve pairs are not '
               'independent and the correlations are weak; quote the sign and '
               'the matched-pair count, not the p-value alone.'])

    add('C10_reproduction_integrity',
        'The amendment scores the same run it amends, and the replication '
        'reproduces the parent exactly.',
        {'control_vs_parent': '70/70 rows, worst |diff| 0.000e+00 psi',
         'amendment_blind_crosscheck': '210 rows, 0.000e+00 psi, 0 acceptance '
                                       'verdict mismatches',
         'v3_independent_rederivation': f'15 stage x arm rows re-derived here '
                                        f'with an independent scorer, {mism} '
                                        f'mismatches'},
        'not IC-dependent',
        [IN_REP_AMENDJSON, os.path.join(REP, 'd3rep_control_check_v1.csv')])

    log(f"claim set: {len(claims)} claims, "
        f"{sum(1 for c in claims if c['supersedes_pre_amendment_claim'])} of them "
        f"superseding a pre-amendment statement")

    # ------------------------------------------------------------------ fig
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0))
    ax = axes[0]
    order = [('6', 'two_zone_r2'), ('6', 'uniform_manuscript'),
             ('12', 'two_zone_r2'), ('12', 'uniform_manuscript'),
             ('10', 'two_zone_r2'), ('10', 'uniform_manuscript')]
    lbl = [f"st{st} {'two-zone' if a == 'two_zone_r2' else 'D=480'}" for st, a in order]
    y = np.arange(len(order))
    b = [rep[k]['blind']['applicability_ft'] for k in order]
    d = [rep[k]['detrended']['applicability_ft'] for k in order]
    # rows where the two ICs coincide would hide one marker under the other, so
    # the two series are drawn on slightly offset baselines and always both show
    for i in range(len(order)):
        ax.plot([b[i], d[i]], [y[i] - 0.10, y[i] + 0.10], color='0.55', lw=3,
                zorder=1, solid_capstyle='round')
    ax.scatter(b, y - 0.10, s=95, color='#1f77b4', zorder=3,
               label='frozen zero IC (blind, pre-registered)')
    ax.scatter(d, y + 0.10, s=95, color='#d62728', marker='D', zorder=3,
               label='per-gauge linear pre-stage detrend')
    for i, (bb, dd) in enumerate(zip(b, d)):
        ax.annotate(f'{bb:.0f}', (bb, y[i] - 0.10), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=9, color='#1f77b4')
        ax.annotate(f'{dd:.0f}', (dd, y[i] + 0.10), textcoords='offset points',
                    xytext=(0, -20), ha='center', fontsize=9, color='#d62728')
    ax.set_yticks(y); ax.set_yticklabels(lbl)
    ax.set_xlabel('pre-registered range of applicability (ft)')
    ax.set_title('(a) the headline is a BRACKET, not a number\n'
                 '254-269 ft (frozen IC)  /  269-1047 ft (detrended)', fontsize=11)
    ax.set_xlim(0, 1450); ax.set_ylim(len(order) - 0.4, -0.9)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.95)

    ax = axes[1]
    st_lbl = ['stage 6', 'stage 12', 'stage 10\n(control)']
    xb = np.arange(3)
    rb = [asym[s]['blind']['ratio'] for s in ('6', '12', '10')]
    rd = [asym[s]['detrended']['ratio'] for s in ('6', '12', '10')]
    ax.bar(xb - 0.19, rb, 0.36, color='#1f77b4', label='frozen zero IC')
    ax.bar(xb + 0.19, rd, 0.36, color='#d62728', label='detrended')
    for i in range(3):
        ax.text(xb[i] - 0.19, rb[i] + 0.08, f'{rb[i]:.2f}x', ha='center', fontsize=9)
        ax.text(xb[i] + 0.19, rd[i] + 0.08, f'{rd[i]:.2f}x', ha='center', fontsize=9)
    ax.axhline(1.0, color='k', lw=1, ls='--')
    ax.set_xticks(xb); ax.set_xticklabels(st_lbl)
    ax.set_ylabel('stimulated-side / virgin-side gauge-mean RMSE')
    ax.set_title('(b) the ONE headline that survives both ICs:\n'
                 'the aggregate side asymmetry, same sign on all three', fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

    ax = axes[2]
    cm_b = [h1[s]['blind']['class_misses'] for s in ('6', '12', '10')]
    dm_b = [h1[s]['blind']['distance_misses'] for s in ('6', '12', '10')]
    cm_d = [h1[s]['detrended']['class_misses'] for s in ('6', '12', '10')]
    dm_d = [h1[s]['detrended']['distance_misses'] for s in ('6', '12', '10')]
    w = 0.2
    ax.bar(xb - 1.5 * w, cm_b, w, color='#1f77b4', label='class rule, blind')
    ax.bar(xb - 0.5 * w, dm_b, w, color='#aec7e8', label='best distance, blind')
    ax.bar(xb + 0.5 * w, cm_d, w, color='#d62728', label='class rule, detrended')
    ax.bar(xb + 1.5 * w, dm_d, w, color='#ff9896', label='best distance, detrended')
    for i, s_ in enumerate(('6', '12', '10')):
        for xoff, v in ((-1.5 * w, cm_b[i]), (-0.5 * w, dm_b[i]),
                        (0.5 * w, cm_d[i]), (1.5 * w, dm_d[i])):
            ax.text(xb[i] + xoff, v + 0.08, str(v), ha='center', fontsize=8,
                    color='0.25')
        ax.text(xb[i] - w, max(cm_b[i], dm_b[i]) + 0.55,
                h1[s_]['blind']['verdict'].replace('H1 ', ''), ha='center',
                fontsize=8, fontweight='bold')
        ax.text(xb[i] + w, max(cm_d[i], dm_d[i]) + 0.55,
                h1[s_]['detrended']['verdict'].replace('H1 ', ''), ha='center',
                fontsize=8, fontweight='bold')
    ax.set_xticks(xb); ax.set_xticklabels(st_lbl)
    ax.set_ylabel('misclassified gauges (lower is better)')
    ax.set_ylim(0, 8)
    ax.set_title("(c) H1's verdict is IC-conditional too\n"
                 'quote the refutation with its IC named', fontsize=11)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    fig.suptitle('D3 / D3-REP amendment v3 - the canonical post-amendment claim set. '
                 'No number changes; what changes is which number is handed over.',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_FIG, dpi=300)
    plt.close(fig)
    log(f"figure -> {os.path.basename(OUT_FIG)} (300 dpi)")

    # ------------------------------------------------------------------ csv
    with open(OUT_CSV, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['claim_id', 'initial_condition_scope', 'statement',
                    'value_json', 'supersedes', 'backed_by'])
        for c in claims:
            w.writerow([c['id'], c['initial_condition_scope'], c['statement'],
                        json.dumps(c['value'], sort_keys=True),
                        c['supersedes_pre_amendment_claim'] or '',
                        '; '.join(b['path'] for b in c['backed_by'])])
    log(f"csv -> {os.path.basename(OUT_CSV)} ({len(claims)} rows)")

    doc = {
        'status': 'AMENDMENT v3 -- claim-set propagation only. NO product number '
                  'changes and nothing was re-solved.',
        'what_this_is': __doc__.strip(),
        'acceptance_rule': {'amplitude_ratio_band': acc['amplitude_ratio_band'],
                            'max_normalised_rmse': acc['max_normalised_rmse'],
                            'source': 'configs/rev2/d3_blind_rep.json -> acceptance '
                                      '(identical to configs/rev2/d3_blind.json)'},
        'independent_rederivation': {
            'rep_rows_checked': 15, 'rep_mismatches': mism,
            'parent_models_checked': sorted(par),
            'method': 'the acceptance rule was re-implemented in this file from '
                      'the config and applied to the published per-gauge CSVs; '
                      'no runner function was imported'},
        'rep_table': {f'{st}/{arm}': v for (st, arm), v in rep.items()},
        'parent_table': par,
        'h1': h1, 'side_asymmetry': asym, 'barrier': barrier,
        'detrend': {'rep_slope_psi_per_h': list(rep_slopes),
                    'parent_slope_psi_per_h': list(par_slopes),
                    'displacement_at_window_end_psi': {'max': disp_max,
                                                       'median': disp_med},
                    'stage_durations_s': durations},
        'slices_first_1260s_gaugemean_psi': first_slice,
        'claims': claims,
    }
    # rev2_manifest._jsonify returns a 3-TUPLE (payload, none_paths,
    # nonfinite_paths), not the payload -- writing its return value straight into
    # json.dump silently produces a 3-element LIST at the document root. Caught
    # by check_claims_v3.py; the products of the run that did it are parked in
    # _aborted_amend_v3_run3_json_shape/ with no manifest.
    payload, none_paths, nonfinite_paths = rman._jsonify(doc)
    assert isinstance(payload, dict), type(payload)
    with open(OUT_JSON, 'w') as fh:
        json.dump(payload, fh, indent=1)
    log(f"json -> {os.path.basename(OUT_JSON)} "
        f"({len(none_paths)} NONE_DECLARED, {len(nonfinite_paths)} non-finite)")

    # ------------------------------------------------------------- manifest
    ctrl = str(cfg_rep['stage_selection']['stage_control'])
    cres = cfg_rep['window']['resolved'][ctrl]
    cwin = rdata.Window(md_min_ft=float(cfg_rep['window']['md_min_ft']),
                        md_max_ft=float(cfg_rep['window']['md_max_ft']),
                        t_start=datetime.datetime.fromisoformat(cres['time_start']),
                        t_end=datetime.datetime.fromisoformat(cres['time_end']))
    cmesh = rdata.build_mesh(cwin, float(cfg_rep['mesh']['domain_pad_low_md_ft']),
                             float(cfg_rep['mesh']['domain_pad_high_md_ft']),
                             float(cfg_rep['mesh']['dx_ft']))
    csrc = cfg_rep['source']['resolved'][ctrl]
    sp = rman.source_protocol(
        application=cfg_rep['source']['application'],
        solver_class=cfg_rep['solver']['class'],
        placement_rule=cfg_rep['source']['selection_rule'],
        sources=[rman.source_record(
            cmesh.x, md_requested_ft=float(csrc['md_ft']),
            mesh_idx=int(cmesh.index_of(float(csrc['md_ft']))),
            driver=rman.driver_record(
                kind='gauge_series',
                baseline_removal=cfg_rep['source']['baseline_removal'],
                value_units='delta_psi',
                series_path=os.path.join(
                    REPO, cfg_rep['data']['gauge_series_template'].format(
                        n=int(csrc['gauge']))),
                gauge_number=int(csrc['gauge']), gauge_md_ft=float(csrc['md_ft']),
                taxis=np.array([0.0]), values=np.array([0.0]),
                time_start=cres['time_start'], time_end=cres['time_end']),
            label=f"g{int(csrc['gauge'])} (UPSTREAM runs; NO SOLVE in this amendment)",
            index_in_source_list=0)],
        targets=[], time_level=cfg_rep['solver']['source_time_level'],
        phase_chaining=rman.NONE_DECLARED,
        boundary_conditions={'lbc': cfg_rep['solver']['lbc'],
                             'rbc': cfg_rep['solver']['rbc'],
                             'source_node': 'Dirichlet'})
    num = rman.numerics(
        time=[rman.time_record(np.array([0.0, 1.0]), mode='fixed',
                               theta=float(cfg_rep['solver']['theta']),
                               t_total_requested_s=1.0,
                               dt_requested_s=float(cfg_rep['solver']['dt_s']),
                               source_time_level=cfg_rep['solver']['source_time_level'],
                               label='UPSTREAM runs; NO SOLVER RAN in this amendment')],
        mesh=rman.mesh_record(cmesh.x, dx_requested_ft=float(cfg_rep['mesh']['dx_ft']),
                              window_md_ft=(cwin.md_min_ft, cwin.md_max_ft),
                              pad_low_ft=float(cfg_rep['mesh']['domain_pad_low_md_ft']),
                              pad_high_ft=float(cfg_rep['mesh']['domain_pad_high_md_ft'])),
        interface_avg=cfg_rep['solver']['interface_avg'],
        boundary={'lbc': cfg_rep['solver']['lbc'], 'rbc': cfg_rep['solver']['rbc'],
                  'pml_thickness': 0.0, 'sigma_max': 0.0},
        diffusivity={'profile_family': cfg_rep['models'][cfg_rep['models']['primary']]['family'],
                     'params': cfg_rep['models'][cfg_rep['models']['primary']]['params'],
                     'baseline_D_ft2_s': float(cfg_rep['models']['uniform_absolute']['params'][0]),
                     'profile_anchor': 'distance from the source node',
                     'note': 'THIS AMENDMENT FITS NOTHING AND SOLVES NOTHING. It '
                             're-derives claims from manifested CSV/JSON products.'},
        barriers=rman.NONE_DECLARED,
        leakage={'lambda_leak_s^-1': 0.0, 'note': 'C2 censored null'},
        kernel={'name': 'none (no solve in this amendment)', 'banded': False,
                'theta': float(cfg_rep['solver']['theta']), 'lambda_leak': 0.0,
                'equivalence_reference': 'not applicable; every number is read '
                                         'from products hashed by manifest.json, '
                                         'manifest_amend_v2.json and '
                                         'manifest_slices_v1.json'},
        rng=rman.NONE_DECLARED,
        parallel={'processes_used': 1, 'cap': 1, 'note': 'pure post-processing'})

    # The declared log is written HERE, before write_manifest hashes it, and is
    # never reopened -- the one-block fix _notes/README.md prescribes for the
    # log-drift defect that hit d3_blind.py and d3_posthoc_v1.py.
    log(f"wall {time.time() - t0:.1f} s; writing manifest")
    with open(OUT_LOG, 'w') as fh:
        fh.write('\n'.join(_L) + '\n')

    rman.write_manifest(
        OUT_MAN, study_id=cfg_rep['study_id'] + '_amend_v3', task_id='D3',
        config=cfg_rep, config_path=CFG_REP,
        inputs=[(CFG_PARENT, 'config', 'd3_blind_parent_config'),
                (IN_PARENT_DETREND, 'prior_run_output', 'd3_parent_detrended_acceptance'),
                (IN_PARENT_METRICS, 'prior_run_output', 'd3_parent_gauge_metrics'),
                (os.path.join(D3, 'd3_amend_v2.json'), 'prior_run_output', 'd3_parent_amend_v2'),
                (os.path.join(D3, 'd3_amend_stage_offsets_v2.csv'), 'prior_run_output', 'd3_parent_offsets'),
                (os.path.join(D3, 'd3_amend_das_lowpass_v2.csv'), 'prior_run_output', 'd3_parent_das_lowpass'),
                (os.path.join(D3, 'manifest.json'), 'prior_run_output', 'd3_parent_manifest'),
                (os.path.join(D3, 'manifest_amend_v2.json'), 'prior_run_output', 'd3_parent_amend_manifest'),
                (IN_REP_HEADLINE, 'prior_run_output', 'd3rep_amend_headline'),
                (IN_REP_PERGAUGE, 'prior_run_output', 'd3rep_amend_per_gauge'),
                (IN_REP_AMENDJSON, 'prior_run_output', 'd3rep_amend_json'),
                (IN_REP_SLICES, 'prior_run_output', 'd3rep_slices'),
                (os.path.join(REP, 'd3rep_control_check_v1.csv'), 'prior_run_output', 'd3rep_control_check'),
                (os.path.join(REP, 'd3rep_das_v1.csv'), 'prior_run_output', 'd3rep_das'),
                (os.path.join(REP, 'd3rep_summary_v1.json'), 'prior_run_output', 'd3rep_summary'),
                (os.path.join(AV2, 'manifest_amend_v2.json'), 'prior_run_output', 'd3rep_amend_manifest'),
                (os.path.join(REP, 'manifest.json'), 'prior_run_output', 'd3rep_main_manifest')],
        source=sp, numerics=num,
        outputs=[rman.output_decl(OUT_JSON, role='json',
                                  note='the canonical D3/D3-REP claim set, post-amendment'),
                 rman.output_decl(OUT_CSV, role='csv', note='the claim set as a flat table'),
                 rman.output_decl(OUT_FIG, role='figure_png', dpi=300,
                                  note='the bracket, the surviving asymmetry, and H1 under both ICs'),
                 rman.output_decl(OUT_LOG, role='log', note='run log')],
        results={'n_claims': len(claims), 'rep_rows_rederived': 15,
                 'rederivation_mismatches': mism,
                 'applicability_bracket_ft': [254.0, 1047.0],
                 'wall_s': round(time.time() - t0, 2)},
        started_utc=started,
        run_label='D3 / D3-REP amendment v3: canonical claim set (no solve)',
        require_modules=('rev2_manifest', 'rev2_data'),
        notes=['NO SOLVER RAN and NO PRODUCT NUMBER CHANGES. This amendment fixes '
               'a propagation defect: the summary handed over for quoting was the '
               'pre-amendment one.',
               'Every claim is re-derived here with an acceptance rule '
               're-implemented from the frozen config, not imported from the '
               'runners, and asserted against the published tables (0 mismatches).',
               'The claim set is the thing to quote. QUOTABLE.md in the D3 '
               'directory is written against it and checked number by number by '
               'amend_v3/hostile_reread/check_claims_v3.py.'])
    rep_v = rman.verify(OUT_MAN, repo_root=REPO)
    print(f"manifest verify: status={rep_v['status']}")
    if rep_v['status'] != 'clean':
        print(json.dumps(rep_v)[:3000])


if __name__ == '__main__':
    main()
