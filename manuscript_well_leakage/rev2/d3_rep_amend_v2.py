#!/usr/bin/env python3
"""D3-REP amendment v2 -- POST-HOC robustness of the 254-269 ft headline to the
initial-condition (linear pre-stage drift) correction the PARENT study published.

WHY THIS EXISTS
---------------
An independent reviewer re-ran D3-REP with their own loader, mesh, solver and
metrics and reported that the headline

    "the range of applicability is 254-269 ft on all three stages"

is not robust to the per-gauge linear pre-stage detrend the parent D3 study had
ALREADY published in the same round (parent README: "moves the pre-registered
applicability distance from 269 ft to 523-1047 ft"), and that D3-REP's open
issue 2 scoped that caveat too narrowly ("the stimulated-side exclusion") when
the statistic actually affected is the study's own headline.

This script reproduces that claim from scratch inside the author's own code
path -- the same frozen config, the same `rev2_core.solve_forward` solve, the
same `d3_blind_rep.score_gauge` / `aggregate` acceptance rule -- on all three
stages and all five frozen arms, and records:

  * blind vs detrended applicability distance, per stage per arm;
  * every gauge whose pre-registered acceptance verdict FLIPS, with its side;
  * the H1 (class vs distance) and H2 (nearest virgin gauge) verdicts recomputed
    on the detrended series;
  * the SIDE AGGREGATES blind vs detrended, which is the part of the study the
    reviewer's own check found to survive -- recorded here so the correction to
    the README does not overcorrect.

NOTHING IS FITTED except the one per-gauge pre-stage slope, which is the
parent's published diagnostic, applied verbatim (30 min pre-window record,
`numpy.polyfit` degree 1, subtracted from the OBSERVATIONS). No model parameter,
window, norm or acceptance threshold is touched. The pre-registered blind
products in ../ are untouched and unchanged; this run RE-SCORES them.

Self-check built in: the blind half of every row is recomputed here from the
frozen config and compared against the pre-registered
`d3rep_gauge_metrics_v1.csv`. If the amendment's solve path did not reproduce
the pre-registered run bit for bit the amendment would be meaningless, so the
maximum blind-vs-recorded deviation is reported and asserted.

Usage:
  python3 scripts/manuscript_well_leakage/rev2/d3_rep_amend_v2.py \
      --config configs/rev2/d3_blind_rep.json
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'manuscript_well_leakage',
                                'baseline_calibration'))

import rev2_core as rcore        # noqa: E402
import rev2_data as rdata        # noqa: E402
import rev2_manifest as rman     # noqa: E402
import d3_blind_rep as d3r       # noqa: E402

VERSION = 'v2'
OUT = os.path.join(REPO, 'output', 'rev2_20260901', 'D3', 'rep_v1', 'amend_v2')
PARENT_RUN = os.path.join(REPO, 'output', 'rev2_20260901', 'D3', 'rep_v1')
PRE_S = 1800.0          # the parent's pre-stage detrend window, seconds
LOG = []


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)


def pre_stage_drift(gauge, t_start):
    """Per-gauge linear drift over the PRE_S seconds before the window opens.

    Verbatim the parent's published diagnostic (d3_amend_v2.py section B /
    d3_posthoc_drift_v1.csv): crop the raw gauge record to
    [t_start - PRE_S, t_start], drop non-increasing timestamps, fit degree 1.
    """
    from fiberis.analyzer.Data1D import Data1D_Gauge
    f = Data1D_Gauge.Data1DGauge()
    f.load_npz(rdata.repo_path(rdata.SWELL_GAUGE_TEMPLATE.format(n=gauge)))
    f.crop(t_start - datetime.timedelta(seconds=PRE_S), t_start)
    ta = np.asarray(f.taxis, float)
    da = np.asarray(f.data, float)
    if ta.size == 0:
        return None
    keep = np.concatenate(([True], np.diff(ta) > 0))
    ta, da = ta[keep], da[keep]
    if ta.size < 20:
        return None
    m, _c = np.polyfit(ta, da, 1)
    return {'slope_psi_per_s': float(m), 'n_pre_samples': int(ta.size),
            'span_s': float(ta[-1] - ta[0])}


def h2_verdict(rows):
    """H2: is the NEAREST VIRGIN gauge usable?  Returns (gauge, dist, verdict)."""
    virg = [r for r in rows if r['stim_class'] == 'virgin'
            and r.get('usable') is not None]
    if not virg:
        return None
    n = min(virg, key=lambda r: r['distance_ft'])
    return {'gauge': n['gauge'], 'distance_ft': n['distance_ft'],
            'usable': bool(n['usable']),
            'verdict': 'H2 SUPPORTED' if n['usable'] else 'H2 REFUTED'}


def run_stage_amend(stage, cfg, starts):
    """Re-solve the frozen arms on one stage and score blind AND detrended."""
    w = cfg['window']
    res = w['resolved'][str(stage)]
    win = rdata.Window(md_min_ft=float(w['md_min_ft']),
                       md_max_ft=float(w['md_max_ft']),
                       t_start=datetime.datetime.fromisoformat(res['time_start']),
                       t_end=datetime.datetime.fromisoformat(res['time_end']))
    log(f"--- stage {stage}: {win.t_start} .. {win.t_end} "
        f"({win.duration_s:.0f} s)")

    gw = rdata.load_window_gauges(win, baseline='first_sample',
                                  rebase=cfg['source']['rebase'])
    fhits = rdata.load_frac_hits(stage)
    src_gauge, centroid = rdata.pick_source_gauge(gw, fhits)
    src = gw.series[src_gauge]
    exp = cfg['source']['resolved'][str(stage)]
    if int(src_gauge) != int(exp['gauge']):
        raise SystemExit(f"stage {stage}: source gauge g{src_gauge} != frozen "
                         f"g{exp['gauge']}")
    mesh = rdata.build_mesh(win, float(cfg['mesh']['domain_pad_low_md_ft']),
                            float(cfg['mesh']['domain_pad_high_md_ft']),
                            float(cfg['mesh']['dx_ft']))
    if mesh.nx != int(cfg['mesh']['expected_nx']):
        raise SystemExit(f"mesh nx {mesh.nx} != frozen "
                         f"{cfg['mesh']['expected_nx']}")
    source_idx = mesh.index_of(src.md_ft)
    targets = rdata.make_targets(gw, mesh, src.md_ft, exclude_gauges=(src_gauge,))
    rec_idx = [t['idx'] for t in targets]
    dt_s = float(cfg['solver']['dt_s'])
    t_total = float(src.t_total_s)

    # ---- the ONE thing fitted here: the per-gauge pre-stage linear slope ----
    drift = {}
    for t in targets:
        drift[int(t['gauge'])] = pre_stage_drift(int(t['gauge']), win.t_start)
    n_missing = sum(1 for v in drift.values() if v is None)
    log(f"  pre-stage drift fitted on {len(drift) - n_missing}/{len(drift)} "
        f"gauges over {PRE_S:.0f} s before the window "
        f"({n_missing} without enough pre-record)")

    models = cfg['models']
    solves = {}
    for name in d3r.ARM_ORDER:
        prof = d3r.build_profile(mesh.x, source_idx, models[name])
        tic = time.time()
        taxis, rec = rcore.solve_forward(
            mesh.x, prof, dt_s, t_total, src.taxis_s, src.delta_psi, source_idx,
            record_idx=rec_idx, theta=float(cfg['solver']['theta']),
            source_time_level=cfg['solver']['source_time_level'],
            lambda_leak=float(cfg['solver']['lambda_leak_s^-1']),
            interface_avg=cfg['solver']['interface_avg'])
        solves[name] = (taxis, rec)
        log(f"  solve {name:22s} {rec.shape[0]} steps {time.time()-tic:5.1f} s")
    # the barrier sensitivity arm, so the amendment covers every frozen arm
    bsec = cfg['barrier']['secondary_sensitivity']
    prof_b, brep = d3r.apply_barrier(
        mesh.x, d3r.build_profile(mesh.x, source_idx,
                                  models[models['primary']]).copy(),
        list(fhits), float(bsec['ratio']), float(bsec['w_half_ft']))
    assert brep is not None and brep['n_fallback'] == 0, brep
    tic = time.time()
    taxis_b, rec_b = rcore.solve_forward(
        mesh.x, prof_b, dt_s, t_total, src.taxis_s, src.delta_psi, source_idx,
        record_idx=rec_idx, theta=float(cfg['solver']['theta']),
        source_time_level=cfg['solver']['source_time_level'],
        lambda_leak=0.0, interface_avg=cfg['solver']['interface_avg'])
    solves['primary_with_barrier'] = (taxis_b, rec_b)
    barrier_d_base = [float(d3r.build_profile(mesh.x, source_idx,
                                              models[models['primary']])[b['i0']])
                      for b in brep['barriers']]
    log(f"  solve primary_with_barrier {rec_b.shape[0]} steps "
        f"{time.time()-tic:5.1f} s")

    tol0 = float(cfg['stimulation_history']['tolerance_ft'])
    cls0, _earlier = d3r.classify_stimulation(stage, starts, tol0)
    metrics = cfg['metrics']
    frac = float(metrics['arrival_time']['relative_fraction'])
    thr_abs = list(metrics['arrival_time']['absolute_thresholds_psi'])
    acc = cfg['acceptance']

    per_arm = {}
    rows_out = []
    for arm, (tx, rc) in solves.items():
        blind_rows, det_rows = [], []
        for k, t in enumerate(targets):
            g = int(t['gauge'])
            common = dict(
                stage=stage, arm=arm, gauge=g, md_ft=float(t['md_ft']),
                distance_ft=float(t['distance_ft']),
                side=('above_source' if t['md_ft'] > src.md_ft
                      else 'below_source'),
                stim_class=cls0[g][0])
            rb, _ = d3r.score_gauge(tx, rc[:, k], t['taxis'], t['data'],
                                    thr_abs, frac)
            rb.update(common)
            blind_rows.append(rb)
            d = drift[g]
            if d is None:
                continue
            obs_d = t['data'] - d['slope_psi_per_s'] * t['taxis']
            rd, _ = d3r.score_gauge(tx, rc[:, k], t['taxis'], obs_d,
                                    thr_abs, frac)
            rd.update(common)
            rd['slope_psi_per_h'] = d['slope_psi_per_s'] * 3600.0
            det_rows.append(rd)
        agg_b = d3r.aggregate(blind_rows, acc)
        agg_b['sides'] = d3r.side_aggregates(blind_rows, src.md_ft)
        agg_b['class_split'] = d3r.class_split(blind_rows)
        agg_b['best_distance_split'] = d3r.best_distance_split(blind_rows)
        agg_b['H1_verdict'] = d3r.verdict(agg_b['class_split'],
                                          agg_b['best_distance_split'])
        agg_b['H2'] = h2_verdict(blind_rows)
        agg_d = d3r.aggregate(det_rows, acc)
        agg_d['sides'] = d3r.side_aggregates(det_rows, src.md_ft)
        agg_d['class_split'] = d3r.class_split(det_rows)
        agg_d['best_distance_split'] = d3r.best_distance_split(det_rows)
        agg_d['H1_verdict'] = d3r.verdict(agg_d['class_split'],
                                          agg_d['best_distance_split'])
        agg_d['H2'] = h2_verdict(det_rows)

        bmap = {r['gauge']: r for r in blind_rows}
        flips = []
        for r in det_rows:
            b = bmap[r['gauge']]
            if b['usable'] is r['usable']:
                continue
            flips.append({'gauge': r['gauge'], 'distance_ft': r['distance_ft'],
                          'side': r['side'], 'stim_class': r['stim_class'],
                          'blind_usable': b['usable'],
                          'detrended_usable': r['usable'],
                          'kind': ('unscored->pass' if b['usable'] is None
                                   and r['usable'] else
                                   'unscored->fail' if b['usable'] is None
                                   else 'fail->pass' if r['usable']
                                   else 'pass->fail')})
        per_arm[arm] = {'blind': agg_b, 'detrended': agg_d, 'flips': flips}
        log(f"  {arm:22s} applicability blind "
            f"{agg_b['applicability_distance_ft']} ft "
            f"({agg_b['n_usable']}/{agg_b['n_scored_for_acceptance']}) -> "
            f"detrended {agg_d['applicability_distance_ft']} ft "
            f"({agg_d['n_usable']}/{agg_d['n_scored_for_acceptance']}) | "
            f"gauge-mean {agg_b['rmse_gaugemean_psi']:.2f} -> "
            f"{agg_d['rmse_gaugemean_psi']:.2f} psi | "
            f"{len([f for f in flips if f['kind'] == 'fail->pass'])} fail->pass")

        for r in blind_rows:
            b = r
            d = next((x for x in det_rows if x['gauge'] == r['gauge']), None)
            rows_out.append({
                'stage': stage, 'arm': arm, 'gauge': r['gauge'],
                'md_ft': r['md_ft'], 'distance_ft': r['distance_ft'],
                'side': r['side'], 'stim_class': r['stim_class'],
                'slope_psi_per_h': (d['slope_psi_per_h'] if d else None),
                'obs_max_blind_psi': b['obs_max_psi'],
                'obs_max_detrended_psi': (d['obs_max_psi'] if d else None),
                'rmse_blind_psi': b['rmse_psi'],
                'rmse_detrended_psi': (d['rmse_psi'] if d else None),
                'ratio_blind': b['amplitude_ratio'],
                'ratio_detrended': (d['amplitude_ratio'] if d else None),
                'nrmse_blind': b['rmse_normalised'],
                'nrmse_detrended': (d['rmse_normalised'] if d else None),
                'usable_blind': b['usable'],
                'usable_detrended': (d['usable'] if d else None),
            })
    return {'stage': stage, 'source_gauge': int(src_gauge),
            'source_md_ft': float(src.md_ft),
            'centroid_ft': float(centroid),
            'window': {'t_start': win.t_start.isoformat(),
                       't_end': win.t_end.isoformat(),
                       'duration_s': float(win.duration_s)},
            'drift': {str(g): v for g, v in drift.items()},
            'per_arm': per_arm}, rows_out, dict(
        win=win, mesh=mesh, src=src, src_gauge=int(src_gauge),
        source_idx=int(source_idx), targets=targets, t_total=t_total,
        dt_s=dt_s, brep=brep, barrier_d_base=barrier_d_base)


def crosscheck_blind(rows_out):
    """The blind half must reproduce the pre-registered run exactly."""
    rec = {}
    with open(os.path.join(PARENT_RUN, 'd3rep_gauge_metrics_v1.csv')) as fh:
        for r in csv.DictReader(fh):
            rec[(int(r['stage']), r['arm'], int(r['gauge']))] = r
    dmax, rmax, n, mism = 0.0, 0.0, 0, 0
    for r in rows_out:
        k = (r['stage'], r['arm'], r['gauge'])
        if k not in rec:
            continue
        ref = rec[k]
        n += 1
        dmax = max(dmax, abs(r['rmse_blind_psi'] - float(ref['rmse_psi'])))
        if r['ratio_blind'] is not None and ref['amplitude_ratio']:
            rmax = max(rmax, abs(r['ratio_blind'] - float(ref['amplitude_ratio'])))
        ru = {'True': True, 'False': False, '': None}[ref['usable']]
        if ru is not r['usable_blind']:
            mism += 1
    return {'n_rows_compared': n, 'max_abs_rmse_diff_psi': dmax,
            'max_abs_amplitude_ratio_diff': rmax,
            'n_acceptance_verdict_mismatches': mism}


def fig_amend(path, stages, dpi):
    arms = ['two_zone_r2', 'uniform_manuscript']
    cols = {'below_source': '#1b7837', 'above_source': '#d95f02'}
    fig, ax = plt.subplots(2, 3, figsize=(15.0, 8.6))
    for j, S in enumerate(stages):
        st = S['stage']
        a = ax[0, j]
        pa = S['per_arm']['two_zone_r2']

        # normalised RMSE vs distance, blind (open) and detrended (filled)
        rows = S['_rows']
        sel = [r for r in rows if r['arm'] == 'two_zone_r2']
        for r in sel:
            c = cols[r['side']]
            if r['nrmse_blind'] is not None:
                a.plot(r['distance_ft'], r['nrmse_blind'], 'o', mfc='none',
                       mec=c, ms=7)
            if r['nrmse_detrended'] is not None:
                a.plot(r['distance_ft'], r['nrmse_detrended'], 's', color=c,
                       ms=6, alpha=0.8)
            if (r['nrmse_blind'] is not None
                    and r['nrmse_detrended'] is not None):
                a.plot([r['distance_ft']] * 2,
                       [r['nrmse_blind'], r['nrmse_detrended']], '-',
                       color=c, lw=0.8, alpha=0.5)
        a.axhline(0.25, color='k', ls='--', lw=1.0)
        ab = pa['blind']['applicability_distance_ft']
        ad = pa['detrended']['applicability_distance_ft']
        a.axvline(ab, color='#4a3aa7', ls=':', lw=1.4)
        if ad is not None:
            a.axvline(ad, color='#c1272d', ls='-.', lw=1.4)
        a.set_yscale('log')
        a.set_xlabel('distance from source (ft)')
        a.set_ylabel('normalised RMSE (RMSE / observed peak)')
        a.set_title(f"stage {st} — two_zone_r2\napplicability "
                    f"{ab:.0f} ft blind (purple) → "
                    f"{'none' if ad is None else f'{ad:.0f} ft'} detrended "
                    f"(red)", fontsize=9)
        a.text(0.02, 0.03, 'open = blind, filled = detrended\n'
                           'green = virgin side, orange = stimulated side',
               transform=a.transAxes, fontsize=7)

        b = ax[1, j]
        labels, blind_v, det_v = [], [], []
        for arm in arms:
            p = S['per_arm'][arm]
            labels.append(arm.replace('_', '\n'))
            blind_v.append(p['blind']['applicability_distance_ft'] or 0.0)
            det_v.append(p['detrended']['applicability_distance_ft'] or 0.0)
        xx = np.arange(len(labels))
        b.bar(xx - 0.18, blind_v, 0.34, label='blind (frozen IC)',
              color='#4a3aa7')
        b.bar(xx + 0.18, det_v, 0.34, label='detrended', color='#c1272d')
        for x, v in zip(xx - 0.18, blind_v):
            b.text(x, v + 12, f'{v:.0f}', ha='center', fontsize=8)
        for x, v in zip(xx + 0.18, det_v):
            b.text(x, v + 12, f'{v:.0f}', ha='center', fontsize=8)
        b.set_xticks(xx)
        b.set_xticklabels(labels, fontsize=8)
        b.set_ylabel('pre-registered applicability distance (ft)')
        b.set_ylim(0, 1200)
        b.set_title(f'stage {st}: the headline statistic under the two ICs',
                    fontsize=9)
        b.legend(fontsize=7.5)
    fig.suptitle('D3-REP amendment v2 — the 254–269 ft headline is conditional '
                 'on the frozen zero initial condition', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    cfg_path = os.path.abspath(args.config)
    with open(cfg_path) as fh:
        cfg = json.load(fh)

    os.makedirs(OUT, exist_ok=True)
    p_json = os.path.join(OUT, f'd3rep_amend_{VERSION}.json')
    p_csv = os.path.join(OUT, f'd3rep_amend_detrended_acceptance_{VERSION}.csv')
    p_sum = os.path.join(OUT, f'd3rep_amend_headline_{VERSION}.csv')
    p_fig = os.path.join(OUT, f'fig_d3rep_amend_{VERSION}.png')
    p_log = os.path.join(OUT, f'd3rep_amend_{VERSION}.log')
    p_man = os.path.join(OUT, 'manifest_amend_v2.json')
    rman.assert_absent([p_json, p_csv, p_sum, p_fig, p_log, p_man])

    log('D3-REP AMENDMENT v2 -- robustness of the 254-269 ft headline to the '
        "parent's published per-gauge linear pre-stage detrend")
    log('resolving pumping start times for all 20 stages')
    starts = d3r.stage_pumping_starts()

    stages = [int(s) for s in cfg['stage_selection']['stages_blind']] \
        + [int(cfg['stage_selection']['stage_control'])]
    out_stages, all_rows, ctx = [], [], None
    for s in stages:
        S, rows, c = run_stage_amend(s, cfg, starts)
        S['_rows'] = [r for r in rows]
        out_stages.append(S)
        all_rows.extend(rows)
        if s == stages[0]:
            ctx = c

    cross = crosscheck_blind(all_rows)
    log(f"crosscheck vs pre-registered d3rep_gauge_metrics_v1.csv: "
        f"{cross['n_rows_compared']} rows, max |dRMSE| "
        f"{cross['max_abs_rmse_diff_psi']:.3e} psi, max |dratio| "
        f"{cross['max_abs_amplitude_ratio_diff']:.3e}, "
        f"{cross['n_acceptance_verdict_mismatches']} verdict mismatches")
    assert cross['max_abs_rmse_diff_psi'] < 1e-9, cross
    assert cross['n_acceptance_verdict_mismatches'] == 0, cross

    # ---- the headline table --------------------------------------------
    head = []
    for S in out_stages:
        for arm, p in S['per_arm'].items():
            b, d = p['blind'], p['detrended']
            head.append({
                'stage': S['stage'], 'arm': arm,
                'applicability_blind_ft': b['applicability_distance_ft'],
                'applicability_detrended_ft': d['applicability_distance_ft'],
                'n_usable_blind': b['n_usable'],
                'n_scored_blind': b['n_scored_for_acceptance'],
                'n_usable_detrended': d['n_usable'],
                'n_scored_detrended': d['n_scored_for_acceptance'],
                'gaugemean_rmse_blind_psi': b['rmse_gaugemean_psi'],
                'gaugemean_rmse_detrended_psi': d['rmse_gaugemean_psi'],
                'virgin_side_rmse_blind_psi':
                    b['sides'].get('below_source', {}).get('rmse_gaugemean_psi'),
                'virgin_side_rmse_detrended_psi':
                    d['sides'].get('below_source', {}).get('rmse_gaugemean_psi'),
                'stim_side_rmse_blind_psi':
                    b['sides'].get('above_source', {}).get('rmse_gaugemean_psi'),
                'stim_side_rmse_detrended_psi':
                    d['sides'].get('above_source', {}).get('rmse_gaugemean_psi'),
                'H1_blind': b['H1_verdict'], 'H1_detrended': d['H1_verdict'],
                'H2_blind': (b['H2'] or {}).get('verdict'),
                'H2_detrended': (d['H2'] or {}).get('verdict'),
                'n_flip_fail_to_pass':
                    len([f for f in p['flips'] if f['kind'] == 'fail->pass']),
                'n_flip_pass_to_fail':
                    len([f for f in p['flips'] if f['kind'] == 'pass->fail']),
            })

    # side-asymmetry ratios, blind vs detrended, primary arm
    asym = {}
    for S in out_stages:
        p = S['per_arm']['two_zone_r2']
        r = {}
        for tag in ('blind', 'detrended'):
            v = p[tag]['sides']['below_source']['rmse_gaugemean_psi']
            a = p[tag]['sides']['above_source']['rmse_gaugemean_psi']
            r[tag] = {'virgin_psi': v, 'stimulated_psi': a, 'ratio': a / v}
        asym[str(S['stage'])] = r
        log(f"  stage {S['stage']} side asymmetry (two_zone): blind "
            f"{r['blind']['virgin_psi']:.2f}/{r['blind']['stimulated_psi']:.2f} "
            f"= {r['blind']['ratio']:.2f}x -> detrended "
            f"{r['detrended']['virgin_psi']:.2f}/"
            f"{r['detrended']['stimulated_psi']:.2f} "
            f"= {r['detrended']['ratio']:.2f}x")

    with open(p_csv, 'w', newline='') as fh:
        wri = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        wri.writeheader()
        for r in sorted(all_rows, key=lambda r: (r['stage'], r['arm'],
                                                 r['distance_ft'])):
            wri.writerow(r)
    with open(p_sum, 'w', newline='') as fh:
        wri = csv.DictWriter(fh, fieldnames=list(head[0]))
        wri.writeheader()
        for r in head:
            wri.writerow(r)
    fig_amend(p_fig, out_stages, int(cfg['outputs'].get('figure_dpi', 300)))

    for S in out_stages:
        S.pop('_rows', None)
    res = {
        'status': 'ok',
        'what_this_is': (
            'POST-HOC robustness check of the D3-REP headline "the '
            'pre-registered range of applicability is 254-269 ft on all three '
            'stages" against the per-gauge linear pre-stage detrend the PARENT '
            'D3 study published in the same round. No blind number changes; '
            'the pre-registered products are re-scored, not re-run.'),
        'detrend_definition': {
            'window_s_before_stage': PRE_S,
            'fit': 'numpy.polyfit degree 1 on the raw gauge record, slope only',
            'applied_to': 'the OBSERVED series (obs - slope * t)',
            'provenance': ('the parent study, output/rev2_20260901/D3/ '
                           'd3_posthoc_drift_v1.csv and d3_amend_v2.py '
                           'section B; reproduced here, not re-invented'),
            'limit': ('a linear detrend is NOT a physical initial condition. '
                      'It removes only the part of the relaxation from earlier '
                      'stages that is straight over the window, so it brackets '
                      'the effect rather than correcting it.')},
        'acceptance_rule': {
            'amplitude_ratio_band': list(cfg['acceptance']['amplitude_ratio_band']),
            'max_normalised_rmse': cfg['acceptance']['max_normalised_rmse'],
            'source': 'configs/rev2/d3_blind_rep.json -> acceptance '
                      '(pre-registered, unchanged)'},
        'crosscheck_blind_vs_preregistered_run': cross,
        'headline_table': head,
        'side_asymmetry_primary_arm': asym,
        'per_stage': out_stages,
    }
    with open(p_json, 'w') as fh:
        json.dump(res, fh, indent=1, default=d3r.jsonable)

    log(f"wall {time.time() - t0:.1f} s")
    log('log written before write_manifest hashes it; the manifest verify '
        'status is printed to the console and recorded in the README, not '
        'appended here.')
    with open(p_log, 'w') as fh:
        fh.write('\n'.join(LOG) + '\n')

    # ---- manifest -------------------------------------------------------
    win, mesh, src = ctx['win'], ctx['mesh'], ctx['src']
    sp = rman.source_protocol(
        application='dirichlet_node', solver_class='rev2_core.solve_forward',
        placement_rule=cfg['source']['selection_rule'],
        sources=[rman.source_record(
            mesh.x, md_requested_ft=float(src.md_ft),
            mesh_idx=int(ctx['source_idx']),
            driver=rman.driver_record(
                kind='gauge_series', baseline_removal='subtract_first_sample',
                value_units='delta_psi',
                series_path=os.path.join(
                    REPO, cfg['data']['gauge_series_template']
                    .format(n=ctx['src_gauge'])),
                gauge_number=int(ctx['src_gauge']),
                gauge_md_ft=float(src.md_ft),
                taxis=src.taxis_s, values=src.delta_psi,
                time_start=win.t_start.isoformat(),
                time_end=win.t_end.isoformat()),
            label=f"g{ctx['src_gauge']}", index_in_source_list=0)],
        targets=[{'gauge': int(t['gauge']), 'md_ft': float(t['md_ft']),
                  'mesh_idx': int(t['idx'])} for t in ctx['targets']],
        time_level='n', phase_chaining=rman.NONE_DECLARED,
        boundary_conditions={'lbc': 'Neumann', 'rbc': 'Neumann',
                             'source_node': 'Dirichlet'})
    bsec = cfg['barrier']['secondary_sensitivity']
    brecs = []
    for bi, b in enumerate(ctx['brep']['barriers']):
        msk = np.zeros(mesh.x.size, dtype=bool)
        msk[b['i0']:b['i1'] + 1] = True
        brecs.append(rman.barrier_record(
            mesh.x, msk, label=f"stage6_frachit_{bi}",
            centre_md_ft=float(b['md_ft']),
            w_requested_ft=float(bsec['w_half_ft']),
            ratio=float(bsec['ratio']),
            d_baseline=float(ctx['barrier_d_base'][bi]), report=b))
    txd = np.arange(0.0, ctx['t_total'] + ctx['dt_s'], ctx['dt_s'])
    num = rman.numerics(
        time=[rman.time_record(txd, mode='fixed', theta=1.0,
                               t_total_requested_s=ctx['t_total'],
                               dt_requested_s=ctx['dt_s'],
                               source_time_level='n', label='amend_v2 stage 6')],
        mesh=rman.mesh_record(mesh.x, dx_requested_ft=float(cfg['mesh']['dx_ft']),
                              window_md_ft=(win.md_min_ft, win.md_max_ft),
                              pad_low_ft=float(cfg['mesh']['domain_pad_low_md_ft']),
                              pad_high_ft=float(cfg['mesh']['domain_pad_high_md_ft'])),
        interface_avg='harmonic',
        boundary={'lbc': 'Neumann', 'rbc': 'Neumann', 'pml_thickness': 0.0,
                  'sigma_max': 0.0},
        diffusivity={'profile_family': 'two_zone + three uniforms + barrier arm',
                     'param_names': ['log10_D_near', 'log10_D_far',
                                     'log10_s_c', 'log10_width'],
                     'params': [float(x) for x in
                                cfg['models']['two_zone_r2']['params']],
                     'baseline_D_ft2_s':
                         float(cfg['models']['uniform_manuscript']['params'][0]),
                     'profile_anchor': 'distance from the source node',
                     'note': ('ALL FIVE FROZEN ARMS ARE RE-SOLVED WITH THE '
                              'FROZEN PARAMETERS. The only quantity fitted '
                              'anywhere in this run is the per-gauge pre-stage '
                              'linear slope, which is the parent study\'s '
                              'published post-hoc detrend.')},
        barriers=brecs,
        leakage={'lambda_leak_s^-1': 0.0, 'note': 'none'},
        kernel={'name': 'rev2_core.solve_forward', 'banded': True, 'theta': 1.0,
                'lambda_leak': 0.0,
                'equivalence_reference': 'bitwise identical to '
                                         'r1_calibration_core.solve_forward'},
        rng=rman.NONE_DECLARED,
        parallel={'processes_used': 1, 'cap': 8})
    inputs = [(os.path.join(REPO, cfg['data']['gauge_md_npz']), 'geometry',
               'gauge_md'),
              (os.path.join(PARENT_RUN, 'manifest.json'), 'prior_run_output',
               'd3rep_blind_manifest'),
              (os.path.join(PARENT_RUN, 'd3rep_gauge_metrics_v1.csv'),
               'prior_run_output', 'd3rep_gauge_metrics'),
              (os.path.join(REPO, 'output', 'rev2_20260901', 'D3',
                            'manifest_amend_v2.json'), 'prior_run_output',
               'parent_d3_amend_manifest'),
              (os.path.join(REPO, 'output', 'rev2_20260901', 'D3',
                            'd3_posthoc_drift_v1.csv'), 'prior_run_output',
               'parent_published_drift_slopes')]
    for n in range(1, 16):
        inputs.append((os.path.join(REPO, cfg['data']['gauge_series_template']
                                    .format(n=n)), 'gauge_series', f'gauge{n}'))
    for s in range(1, 21):
        inputs.append((os.path.join(
            REPO, rdata.SWELL_FRAC_HIT_TEMPLATE.format(stage=s)), 'geometry',
            f'frac_hit_stage{s}'))
    outs = [rman.output_decl(p_json, role='json', note='amendment record'),
            rman.output_decl(p_csv, role='csv',
                             note='per-gauge blind vs detrended acceptance, '
                                  '3 stages x 5 arms'),
            rman.output_decl(p_sum, role='csv',
                             note='the headline statistic under both initial '
                                  'conditions'),
            rman.output_decl(p_log, role='log', note='amend log'),
            rman.output_decl(p_fig, role='figure_png', dpi=300)]
    rman.write_manifest(
        p_man, study_id='d3rep_blind_amend_v2', task_id='D3', config=cfg,
        config_path=cfg_path, inputs=inputs, source=sp, numerics=num,
        outputs=outs, results=res, started_utc=started,
        run_label='D3-REP amendment v2 (POST-HOC; the headline is conditional '
                  'on the frozen initial condition; no blind number changes)',
        require_modules=('rev2_core', 'rev2_manifest', 'rev2_data',
                         'd3_blind_rep'),
        notes=['POST-HOC AMENDMENT. The pre-registered blind products in ../ '
               'are untouched and unchanged; this run RE-SCORES them under an '
               'alternative initial condition and reports what moves.',
               'configs/rev2/d3_blind_rep.json is deliberately NOT edited: it '
               'is pre-registered and hashed. This amendment changes claims in '
               'the README, not the frozen configuration.',
               'The blind half of every row was recomputed from the frozen '
               'config and matches the pre-registered '
               'd3rep_gauge_metrics_v1.csv to '
               f"{cross['max_abs_rmse_diff_psi']:.3e} psi over "
               f"{cross['n_rows_compared']} rows with "
               f"{cross['n_acceptance_verdict_mismatches']} verdict "
               'mismatches, so the amendment scores the same run.',
               'This amendment lives in its own subdirectory for the reason '
               'recorded in the parent README section 9: write_manifest scans '
               'its own directory for undeclared files.'])
    rep = rman.verify(p_man, repo_root=REPO)
    print(f"amend manifest verify: status={rep['status']} "
          f"findings={rep.get('findings')}")


if __name__ == '__main__':
    main()
