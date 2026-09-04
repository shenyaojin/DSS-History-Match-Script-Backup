#!/usr/bin/env python3
"""D3-REP -- replication of the D3 blind forward test on two never-used stages.

WHAT THIS IS
------------
D3 froze a model, ran it on fracturing stage 10, and found -- AFTER THE FACT --
that the blind skill separated by which SIDE of the source a gauge sat on rather
than by distance: every gauge above the source lay in rock earlier stages had
already fractured, every gauge below lay in rock no stage had touched, and only
the virgin side was predicted usefully. An observation made after seeing the data
cannot be tested on that data. This script tests it on two stages that have never
been used for anything, with the split pre-registered as a falsifiable hypothesis
BEFORE any of them was solved.

ORDER OF OPERATIONS -- the whole value of the study
---------------------------------------------------
1. configs/rev2/d3_blind_rep.json was written and hashed.
2. Its hash and the wall clock went into
   output/rev2_20260901/D3/rep_v1/PREREGISTERED.json, at a moment when THIS FILE
   DID NOT EXIST (the pre-registration records
   `runner_existed_at_preregistration: false`).
3. Only then was this file written.
The run refuses to start unless the config still hashes to the pre-registered
value, so the ordering is machine-enforced rather than asserted. There is no
optimiser, no search, no fit and no free parameter anywhere below: every physical
value is a literal read from the config, and every one of those is itself a
verbatim copy of a value the parent config had frozen before stage 10 was solved.
`_assert_inherited_verbatim` re-reads the parent config and checks the ten
inherited blocks block by block, so "copied, not tuned" is checked, not trusted.

IF THE REPLICATION FAILS IT IS REPORTED AS IT STANDS. Nothing here can be
repaired by a parameter change, because no parameter is free.
"""

import argparse
import csv
import datetime
import hashlib
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
import r1_calibration_core as r1c  # noqa: E402

TASK_ID = 'D3'
VERSION = 'v1'
PARENT_CFG = 'configs/rev2/d3_blind.json'
ARM_ORDER = ('two_zone_r2', 'uniform_absolute', 'uniform_normalised',
             'uniform_manuscript')
N_STAGES_TOTAL = 20

_LOG_LINES = []


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _LOG_LINES.append(line)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, datetime.datetime):
        return o.isoformat()
    raise TypeError(repr(type(o)))


# ---------------------------------------------------------------------------
# frozen model construction -- literal parameters, no fitting anywhere
# ---------------------------------------------------------------------------

def build_profile(mesh_x, source_idx, spec):
    fam = spec['family']
    params = list(spec['params'])
    if fam == 'uniform':
        return np.full(mesh_x.size, float(params[0]), dtype=float)
    if fam == 'two_zone':
        return r1c.profile_two_zone(mesh_x, source_idx, np.asarray(params, float))
    raise ValueError(f"unsupported frozen family {fam!r}")


def apply_barrier(mesh_x, prof, centres, ratio, w_half):
    if float(ratio) >= 1.0 or not len(centres):
        return prof, None
    d, rep = rcore.build_barrier_profile(
        mesh_x, prof, list(centres), float(w_half), float(ratio),
        ratio_reference='local', combine='min', on_empty='nearest',
        on_outside='raise', return_report=True)
    return d, rep


# ---------------------------------------------------------------------------
# metrics -- identical definitions to the parent study, so the stage-10 control
# can be compared row for row against d3_gauge_metrics_v1.csv
# ---------------------------------------------------------------------------

def score_gauge(t_sim, y_sim, t_obs, y_obs, thresholds_abs, frac):
    sim = np.interp(t_obs, t_sim, y_sim)
    resid = sim - y_obs
    obs_max = float(np.max(y_obs))
    sim_max = float(np.max(sim))
    out = {
        'n_samples': int(y_obs.size),
        'rmse_psi': float(np.sqrt(np.mean(resid ** 2))),
        'bias_psi': float(np.mean(resid)),
        'obs_max_psi': obs_max,
        'sim_max_psi': sim_max,
        'obs_min_psi': float(np.min(y_obs)),
        'sim_min_psi': float(np.min(sim)),
        'sum_sq': float(np.sum(resid ** 2)),
    }
    positive = obs_max > 0.0
    out['has_positive_response'] = bool(positive)
    out['amplitude_ratio'] = (sim_max / obs_max) if positive else None
    out['rmse_normalised'] = (out['rmse_psi'] / obs_max) if positive else None
    if positive:
        thr = float(frac) * obs_max
        t_o = r1c.arrival_time(t_obs, y_obs, thr)
        t_s = r1c.arrival_time(t_obs, sim, thr)
        out['arrival_obs_rel_s'] = None if np.isnan(t_o) else float(t_o)
        out['arrival_sim_rel_s'] = None if np.isnan(t_s) else float(t_s)
        out['arrival_err_rel_s'] = (None if (np.isnan(t_o) or np.isnan(t_s))
                                    else float(t_s - t_o))
        out['arrival_threshold_rel_psi'] = thr
    else:
        out['arrival_obs_rel_s'] = out['arrival_sim_rel_s'] = None
        out['arrival_err_rel_s'] = out['arrival_threshold_rel_psi'] = None
    for thr in thresholds_abs:
        t_o = r1c.arrival_time(t_obs, y_obs, float(thr))
        t_s = r1c.arrival_time(t_obs, sim, float(thr))
        k = f'{thr:g}'
        out[f'arrival_obs_abs{k}_s'] = None if np.isnan(t_o) else float(t_o)
        out[f'arrival_sim_abs{k}_s'] = None if np.isnan(t_s) else float(t_s)
        out[f'arrival_err_abs{k}_s'] = (None if (np.isnan(t_o) or np.isnan(t_s))
                                        else float(t_s - t_o))
    return out, sim


def aggregate(rows, acc):
    mse = [r['rmse_psi'] ** 2 for r in rows]
    norm = [r['rmse_normalised'] ** 2 for r in rows
            if r['rmse_normalised'] is not None]
    ss = sum(r['sum_sq'] for r in rows)
    n = sum(r['n_samples'] for r in rows)
    lo, hi = acc['amplitude_ratio_band']
    scored = []
    for r in rows:
        if r['amplitude_ratio'] is None or r['rmse_normalised'] is None:
            r['usable'] = None
            r['fail_reason'] = 'no positive observed response in window'
            continue
        ok_amp = lo <= r['amplitude_ratio'] <= hi
        ok_rms = r['rmse_normalised'] <= acc['max_normalised_rmse']
        r['usable'] = bool(ok_amp and ok_rms)
        r['fail_reason'] = None if r['usable'] else (
            ('amplitude_ratio out of band' if not ok_amp else '')
            + ('; ' if (not ok_amp and not ok_rms) else '')
            + ('normalised RMSE > threshold' if not ok_rms else ''))
        scored.append(r)
    failed = [r for r in scored if not r['usable']]
    applic = min((r['distance_ft'] for r in failed), default=None)
    return {
        'n_gauges': len(rows),
        'n_scored_for_acceptance': len(scored),
        'rmse_gaugemean_psi': float(np.sqrt(np.mean(mse))) if mse else None,
        'rmse_pooled_psi': float(np.sqrt(ss / n)) if n else None,
        'rmse_normalised_gaugemean': (float(np.sqrt(np.mean(norm)))
                                      if norm else None),
        'n_usable': int(sum(1 for r in scored if r['usable'])),
        'n_failed': len(failed),
        'applicability_distance_ft': (float(applic) if applic is not None
                                      else None),
        'applicability_note': (
            'distance of the NEAREST scored gauge that fails the pre-registered '
            'acceptance rule; None means no scored gauge failed'),
    }


def side_aggregates(rows, src_md):
    out = {}
    for side in ('below_source', 'above_source'):
        sel = [r for r in rows if r['side'] == side]
        if not sel:
            continue
        out[side] = {
            'n': len(sel),
            'rmse_gaugemean_psi': float(np.sqrt(np.mean(
                [r['rmse_psi'] ** 2 for r in sel]))),
            'n_usable': int(sum(1 for r in sel if r.get('usable') is True)),
            'n_scored': int(sum(1 for r in sel if r.get('usable') is not None)),
            'nearest_failing_ft': min(
                (r['distance_ft'] for r in sel if r.get('usable') is False),
                default=None),
        }
    return out


# ---------------------------------------------------------------------------
# stimulation history -- geometry and pumping timestamps ONLY, so it cannot be
# contaminated by the outcome it is used to explain
# ---------------------------------------------------------------------------

def stage_pumping_starts():
    starts = {}
    for s in range(1, N_STAGES_TOTAL + 1):
        p = rdata.load_pumping(s)
        starts[s] = p.pumping_start(threshold_bpm=1.0, hold_s=30.0)
    return starts


def classify_stimulation(stage, starts, tol_ft):
    """Return {gauge -> (class, [earlier stages whose hit span contains it])}."""
    t0 = starts[stage]
    earlier = [s for s in range(1, N_STAGES_TOTAL + 1) if starts[s] < t0]
    spans = {}
    for s in earlier:
        fh = rdata.load_frac_hits(s)
        spans[s] = (float(np.min(fh)) - tol_ft, float(np.max(fh)) + tol_ft)
    md_table = rdata.load_gauge_md_table()
    out = {}
    for g in range(1, 16):
        md = float(md_table.md_of(g))
        hits = [s for s, (lo, hi) in spans.items() if lo <= md <= hi]
        out[g] = ('stimulated' if hits else 'virgin', hits)
    return out, earlier


def best_distance_split(rows):
    """Fewest misclassifications achievable by ANY single distance threshold.

    Predicts usable for distance <= d*, not usable beyond. Candidates are the
    midpoints between consecutive sorted distances plus the two degenerate
    thresholds, so the optimum over all real thresholds is attained.
    """
    sc = [r for r in rows if r.get('usable') is not None]
    if not sc:
        return None
    d = np.array([r['distance_ft'] for r in sc], float)
    u = np.array([bool(r['usable']) for r in sc])
    ds = np.unique(d)
    cands = [0.0] + [float(0.5 * (ds[i] + ds[i + 1])) for i in range(ds.size - 1)] \
        + [float(ds[-1] + 1.0)]
    best = None
    for c in cands:
        pred = d <= c
        miss = int(np.sum(pred != u))
        if best is None or miss < best[1]:
            best = (c, miss)
    return {'threshold_ft': best[0], 'n_misclassified': best[1],
            'n_scored': len(sc)}


def class_split(rows):
    """Misclassifications of the pre-registered rule: virgin -> usable."""
    sc = [r for r in rows if r.get('usable') is not None]
    if not sc:
        return None
    miss = int(sum(1 for r in sc
                   if bool(r['usable']) != (r['stim_class'] == 'virgin')))
    tab = {'virgin_usable': 0, 'virgin_not': 0,
           'stimulated_usable': 0, 'stimulated_not': 0}
    for r in sc:
        key = r['stim_class'] + ('_usable' if r['usable'] else '_not')
        tab[key] += 1
    return {'n_misclassified': miss, 'n_scored': len(sc), 'table_2x2': tab}


def verdict(cls, dist):
    if cls is None or dist is None:
        return 'inconclusive (no scored gauge)'
    if cls['n_misclassified'] < dist['n_misclassified']:
        return 'H1 SUPPORTED'
    if cls['n_misclassified'] > dist['n_misclassified']:
        return 'H1 REFUTED'
    return 'H1 INCONCLUSIVE (tie)'


# ---------------------------------------------------------------------------
# integrity of the inheritance
# ---------------------------------------------------------------------------

def _deep_diff(a, b, path=''):
    """Every differing leaf between two JSON trees, as (path, a_leaf, b_leaf)."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f'{path}.{k}', a.get(k, '<absent>'), b.get(k, '<absent>')))
            else:
                out += _deep_diff(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, a, b))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += _deep_diff(x, y, f'{path}[{i}]')
    elif a != b:
        out.append((path, a, b))
    return out


def _assert_inherited_verbatim(cfg):
    """Check "copied, not tuned" against the parent, leaf by leaf.

    A block declared verbatim that differs in any NUMBER, boolean or null is a
    hard stop: that would be a tuned parameter wearing an inheritance label.
    A block that differs only in free-text prose is allowed through and every
    such leaf is RETURNED so it lands in the log, the summary and the manifest
    notes -- disclosed, never silently accepted.
    """
    parent_path = os.path.join(REPO, PARENT_CFG)
    parent_sha = sha256_file(parent_path)
    declared = cfg['inherits']['from_config_sha256']
    if parent_sha != declared:
        raise SystemExit(
            f"parent config {PARENT_CFG} hashes {parent_sha}, but this study was "
            f"frozen against {declared}. The parent is pre-registered and must "
            f"never be edited; refusing to run.")
    with open(parent_path) as fh:
        parent = json.load(fh)
    blocks = list(cfg['inherits']['verbatim_blocks'])
    deviations = []
    for b in blocks:
        for p, mine, theirs in _deep_diff(cfg[b], parent[b], path=b):
            if not (isinstance(mine, str) and isinstance(theirs, str)):
                raise SystemExit(
                    f"block '{b}' is declared verbatim but leaf {p} differs in a "
                    f"NON-PROSE value: {mine!r} vs parent {theirs!r}. That would "
                    f"be a tuned parameter; refusing to run.")
            deviations.append({'block': b, 'path': p, 'this_study': mine,
                               'parent': theirs})
    return parent_sha, blocks, deviations


def _load_parent_control(json_path, csv_path):
    """The parent study's stage-10 per-gauge metrics, for the control check.

    Two sources, deliberately. `d3_summary_v1.json` carries FULL precision and is
    what the pre-registered 1e-6 psi tolerance is checked against. The published
    CSV is rounded to six significant figures, so a CSV-only comparison could
    never resolve 1e-6 psi and quoting one against that tolerance would be
    meaningless; it is compared too, at its own precision, because it is the
    table a reader of the parent study actually sees.
    """
    full = {}
    with open(json_path) as fh:
        doc = json.load(fh)
    for arm, blk in doc['per_arm'].items():
        for r in blk['per_gauge']:
            full[(arm, int(r['gauge']))] = r
    rounded = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            rounded[(r['model'], int(r['gauge']))] = r
    return full, rounded


# ---------------------------------------------------------------------------
# one stage
# ---------------------------------------------------------------------------

def run_stage(stage, cfg, starts, blind):
    w = cfg['window']
    res = w['resolved'][str(stage)]
    win = rdata.Window(md_min_ft=float(w['md_min_ft']),
                       md_max_ft=float(w['md_max_ft']),
                       t_start=datetime.datetime.fromisoformat(res['time_start']),
                       t_end=datetime.datetime.fromisoformat(res['time_end']))
    tag = 'BLIND' if blind else 'CONTROL'
    log(f"--- stage {stage} [{tag}] : {win.t_start} .. {win.t_end} "
        f"({win.duration_s:.0f} s)")

    gw = rdata.load_window_gauges(win, baseline='first_sample',
                                 rebase=cfg['source']['rebase'])
    fhits = rdata.load_frac_hits(stage)
    src_gauge, centroid = rdata.pick_source_gauge(gw, fhits)
    src = gw.series[src_gauge]
    exp = cfg['source']['resolved'][str(stage)]
    if int(src_gauge) != int(exp['gauge']):
        raise SystemExit(f"stage {stage}: resolved source gauge g{src_gauge} "
                         f"differs from the pre-registered g{exp['gauge']}")
    if abs(float(centroid) - float(exp['frac_hit_centroid_ft'])) > 1e-3:
        raise SystemExit(f"stage {stage}: centroid {centroid} differs from the "
                         f"pre-registered {exp['frac_hit_centroid_ft']}")
    log(f"  frac hits ({fhits.size}) {float(np.min(fhits)):.2f}..{float(np.max(fhits)):.2f}; "
        f"centroid {centroid:.3f}; source g{src_gauge} MD {src.md_ft} "
        f"({src.md_ft - centroid:+.3f} ft)")

    mesh = rdata.build_mesh(win, float(cfg['mesh']['domain_pad_low_md_ft']),
                            float(cfg['mesh']['domain_pad_high_md_ft']),
                            float(cfg['mesh']['dx_ft']))
    source_idx = mesh.index_of(src.md_ft)
    if mesh.nx != int(cfg['mesh']['expected_nx']):
        raise SystemExit(f"mesh nx {mesh.nx} != pre-registered "
                         f"{cfg['mesh']['expected_nx']}")
    targets = rdata.make_targets(gw, mesh, src.md_ft, exclude_gauges=(src_gauge,))
    dt_s = float(cfg['solver']['dt_s'])
    t_total = float(src.t_total_s)
    rec_idx = [t['idx'] for t in targets]
    log(f"  mesh nx={mesh.nx} [{mesh.x[0]:.0f},{mesh.x[-1]:.0f}] source node "
        f"{source_idx}; {len(targets)} targets; t_total {t_total:.1f} s")

    models = cfg['models']
    primary = models['primary']
    profiles, solves = {}, {}
    for name in ARM_ORDER:
        prof = build_profile(mesh.x, source_idx, models[name])
        profiles[name] = prof
        tic = time.time()
        taxis, rec = rcore.solve_forward(
            mesh.x, prof, dt_s, t_total, src.taxis_s, src.delta_psi, source_idx,
            record_idx=rec_idx, theta=float(cfg['solver']['theta']),
            source_time_level=cfg['solver']['source_time_level'],
            lambda_leak=float(cfg['solver']['lambda_leak_s^-1']),
            interface_avg=cfg['solver']['interface_avg'])
        solves[name] = (taxis, rec)
        log(f"  solve {name:20s} {rec.shape[0]} steps  {time.time()-tic:5.1f} s")

    bsec = cfg['barrier']['secondary_sensitivity']
    prof_b, brep = apply_barrier(mesh.x, profiles[primary].copy(), list(fhits),
                                 float(bsec['ratio']), float(bsec['w_half_ft']))
    barrier_d_base = []
    if brep is not None:
        # HOUSE RULE (A1 / A4_repair2): the report is the authority, not warnings.
        assert brep['n_fallback'] == 0, brep
        assert brep.get('n_width_inflated', 0) == 0, brep
        barrier_d_base = [float(profiles[primary][b['i0']])
                          for b in brep['barriers']]
    tic = time.time()
    taxis_b, rec_b = rcore.solve_forward(
        mesh.x, prof_b, dt_s, t_total, src.taxis_s, src.delta_psi, source_idx,
        record_idx=rec_idx, theta=float(cfg['solver']['theta']),
        source_time_level=cfg['solver']['source_time_level'],
        lambda_leak=0.0, interface_avg=cfg['solver']['interface_avg'])
    solves['primary_with_barrier'] = (taxis_b, rec_b)
    log(f"  solve primary_with_barrier realised full width "
        f"{brep['realised_full_width_ft'] if brep else 'n/a'} ft  "
        f"{time.time()-tic:5.1f} s")

    # stimulation classification (geometry + timestamps only)
    tol0 = float(cfg['stimulation_history']['tolerance_ft'])
    cls0, earlier = classify_stimulation(stage, starts, tol0)
    tol_set = [0.0, 25.0, 50.0, 100.0]
    cls_alt = {t: classify_stimulation(stage, starts, t)[0] for t in tol_set}
    log(f"  earlier stages: {earlier}")

    metrics = cfg['metrics']
    frac = float(metrics['arrival_time']['relative_fraction'])
    thr_abs = list(metrics['arrival_time']['absolute_thresholds_psi'])
    acc = cfg['acceptance']

    per_arm, sim_store = {}, {}
    for arm, (tx, rc) in solves.items():
        rows = []
        for k, t in enumerate(targets):
            r, sim = score_gauge(tx, rc[:, k], t['taxis'], t['data'], thr_abs, frac)
            g = int(t['gauge'])
            r.update(stage=stage, arm=arm, gauge=g, md_ft=float(t['md_ft']),
                     distance_ft=float(t['distance_ft']),
                     side=('above_source' if t['md_ft'] > src.md_ft
                           else 'below_source'),
                     stim_class=cls0[g][0],
                     stim_by_stages=cls0[g][1],
                     stim_borderline=bool(len({cls_alt[t2][g][0]
                                               for t2 in tol_set}) > 1))
            rows.append(r)
            if arm == primary:
                sim_store[g] = sim
        agg = aggregate(rows, acc)
        agg['sides'] = side_aggregates(rows, src.md_ft)
        agg['class_split'] = class_split(rows)
        agg['best_distance_split'] = best_distance_split(rows)
        agg['H1_verdict'] = verdict(agg['class_split'], agg['best_distance_split'])
        per_arm[arm] = {'rows': rows, 'aggregate': agg}
        log(f"  {arm:22s} gauge-mean RMSE {agg['rmse_gaugemean_psi']:9.3f} psi | "
            f"usable {agg['n_usable']}/{agg['n_scored_for_acceptance']} | "
            f"applicability {agg['applicability_distance_ft']} ft | "
            f"{agg['H1_verdict']}")

    # ---- LF-DAS, native counts, SHAPE ONLY, no coefficient fitted -----------
    das = cfg['das']
    gamma = float(cfg['gamma']['value_psi^-1'])
    rec_das = rdata.load_das_stage(stage, kind=das['kind'],
                                   md_range=(float(mesh.x[0]), float(mesh.x[-1])),
                                   time_range=(win.t_start, win.t_end))
    keep = rec_das.artifact_mask(auto=True, z=20.0, pad_s=2.0)
    n_drop = int(keep.size - keep.sum())
    das_md = rec_das.daxis_ft.copy()
    das_idx = [mesh.index_of(m) for m in das_md]
    tic = time.time()
    tx_d, rec_d = rcore.solve_forward(
        mesh.x, profiles[primary], dt_s, t_total, src.taxis_s, src.delta_psi,
        source_idx, record_idx=das_idx, theta=1.0, source_time_level='n',
        interface_avg='harmonic')
    log(f"  DAS: {das_md.size} ch, {keep.size} samples ({n_drop} dropped), "
        f"node solve {rec_d.shape} in {time.time()-tic:.1f} s")

    off = (rec_das.t0_abs - src.t0_abs).total_seconds()
    t_das_model = rec_das.taxis_s + off
    n_ch = das_md.size
    model_sr = np.empty((n_ch, rec_das.taxis_s.size), dtype=np.float32)
    CH = 256
    for a in range(0, n_ch, CH):
        b = min(a + CH, n_ch)
        blk = np.gradient(rec_d[:, a:b], tx_d, axis=0) * gamma
        for j in range(a, b):
            model_sr[j, :] = np.interp(t_das_model, tx_d, blk[:, j - a])
        del blk
    del rec_d

    obs_all = rec_das.data
    rms_obs = np.empty(n_ch); rms_mod = np.empty(n_ch); corr = np.empty(n_ch)
    for a in range(0, n_ch, CH):
        b = min(a + CH, n_ch)
        o = np.asarray(obs_all[a:b, :], dtype=np.float64)[:, keep]
        m = model_sr[a:b, :].astype(np.float64)[:, keep]
        om = o.mean(axis=1); mm = m.mean(axis=1)
        vo = np.maximum((o ** 2).mean(axis=1) - om ** 2, 0.0)
        vm = np.maximum((m ** 2).mean(axis=1) - mm ** 2, 0.0)
        cross = (o * m).mean(axis=1) - om * mm
        rms_obs[a:b] = np.sqrt(vo)
        rms_mod[a:b] = np.sqrt(vm)
        with np.errstate(invalid='ignore', divide='ignore'):
            c = cross / np.sqrt(vo * vm)
        corr[a:b] = np.where((vo > 0) & (vm > 0), c, np.nan)
        del o, m
    del model_sr

    ch_dist = np.abs(das_md - src.md_ft)
    ref_d = float(das['reference_distance_ft']); ref_h = float(das['reference_halfwidth_ft'])
    ref_m = np.abs(ch_dist - ref_d) <= ref_h
    ref_obs = float(np.mean(rms_obs[ref_m])) if ref_m.any() else np.nan
    ref_mod = float(np.mean(rms_mod[ref_m])) if ref_m.any() else np.nan
    n_obs = rms_obs / ref_obs if ref_obs > 0 else np.full_like(rms_obs, np.nan)
    n_mod = rms_mod / ref_mod if ref_mod > 0 else np.full_like(rms_mod, np.nan)

    das_rows = []
    for j in range(n_ch):
        das_rows.append({'stage': stage, 'md_ft': float(das_md[j]),
                         'distance_ft': float(ch_dist[j]),
                         'side': ('above_source' if das_md[j] > src.md_ft
                                  else 'below_source'),
                         'rms_obs_counts': float(rms_obs[j]),
                         'rms_model_strainrate_s^-1': float(rms_mod[j]),
                         'rms_obs_normalised': float(n_obs[j]),
                         'rms_model_normalised': float(n_mod[j]),
                         'pearson_r': float(corr[j])})
    at_gauge = {}
    for t in targets:
        g = int(t['gauge'])
        j = int(np.argmin(np.abs(das_md - t['md_ft'])))
        at_gauge[g] = {'gauge': g, 'channel_md_ft': float(das_md[j]),
                       'channel_offset_ft': float(das_md[j] - t['md_ft']),
                       'distance_ft': float(t['distance_ft']),
                       'side': ('above_source' if t['md_ft'] > src.md_ft
                                else 'below_source'),
                       'stim_class': cls0[g][0],
                       'pearson_r': float(corr[j]),
                       'rms_obs_normalised': float(n_obs[j]),
                       'rms_model_normalised': float(n_mod[j])}
    for side in ('below_source', 'above_source'):
        rs = [v['pearson_r'] for v in at_gauge.values()
              if v['side'] == side and np.isfinite(v['pearson_r'])]
        if rs:
            log(f"  DAS r at gauge channels, {side}: "
                f"{min(rs):+.3f} .. {max(rs):+.3f} (median {np.median(rs):+.3f}, n={len(rs)})")

    return {
        'stage': stage, 'blind': blind,
        'window': {'t_start': res['time_start'], 't_end': res['time_end'],
                   'duration_s': res['duration_s'], 't_total_source_s': t_total},
        'source': {'gauge': int(src_gauge), 'md_ft': float(src.md_ft),
                   'centroid_ft': float(centroid),
                   'offset_ft': float(src.md_ft - centroid),
                   'mesh_idx': int(source_idx),
                   'frac_hits_ft': [float(x) for x in fhits],
                   'source_vs_hit_span': (
                       'inside' if np.min(fhits) <= src.md_ft <= np.max(fhits)
                       else ('above' if src.md_ft > np.max(fhits) else 'below'))},
        'earlier_stages': earlier,
        'stim_class': {g: cls0[g][0] for g in cls0},
        'stim_class_tol_sensitivity': {
            str(t2): {g: cls_alt[t2][g][0] for g in cls_alt[t2]} for t2 in tol_set},
        'per_arm': per_arm,
        'barrier_report': brep,
        'barrier_d_base': barrier_d_base,
        'das': {'n_channels': n_ch, 'n_samples': int(keep.size),
                'n_samples_dropped': n_drop,
                'reference_band_ft': [ref_d, ref_h],
                'reference_n_channels': int(ref_m.sum()),
                'reference_rms_obs_counts': ref_obs,
                'reference_rms_model_s^-1': ref_mod,
                'at_gauge': at_gauge, 'rows': das_rows},
        '_series': {'targets': targets, 'solves': solves, 'src': src,
                    'sim_primary': sim_store, 'mesh_nx': mesh.nx,
                    'das_dist': ch_dist, 'das_n_obs': n_obs,
                    'das_n_mod': n_mod, 'das_corr': corr, 'das_md': das_md},
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_overlay(path, results, primary, dpi):
    stages = [r for r in results if r['blind']]
    ncol = max(len(s['_series']['targets']) for s in stages)
    fig, axes = plt.subplots(len(stages), 1, figsize=(15, 5.2 * len(stages)))
    if len(stages) == 1:
        axes = [axes]
    for ax, st in zip(axes, stages):
        rows = {r['gauge']: r for r in st['per_arm'][primary]['rows']}
        ts = st['_series']
        for t in ts['targets']:
            g = int(t['gauge'])
            r = rows[g]
            col = 'tab:green' if r['stim_class'] == 'virgin' else 'tab:red'
            ax.plot(t['taxis'] / 3600.0, t['data'], color=col, lw=1.0, alpha=0.85)
            ax.plot(t['taxis'] / 3600.0, ts['sim_primary'][g], color='k', lw=0.8,
                    ls='--', alpha=0.7)
        ax.set_title(f"stage {st['stage']}: measured (green = virgin rock, "
                     f"red = already stimulated) vs frozen {primary} (black dashed). "
                     f"Source g{st['source']['gauge']} MD {st['source']['md_ft']:.0f} ft",
                     fontsize=10)
        ax.set_xlabel('time since window start, h')
        ax.set_ylabel(r'$\Delta P$, psi')
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def fig_class(path, results, primary, acc, dpi):
    stages = [r for r in results if r['blind']]
    fig, axes = plt.subplots(2, len(stages), figsize=(6.2 * len(stages), 8.4),
                             squeeze=False)
    lo, hi = acc['amplitude_ratio_band']
    for j, st in enumerate(stages):
        rows = st['per_arm'][primary]['rows']
        for ax, key, ylab, logy in (
                (axes[0][j], 'amplitude_ratio', 'amplitude ratio sim/obs', True),
                (axes[1][j], 'rmse_normalised', 'RMSE / observed peak', True)):
            for r in rows:
                if r[key] is None:
                    continue
                col = 'tab:green' if r['stim_class'] == 'virgin' else 'tab:red'
                mk = 'o' if r.get('usable') else 'x'
                ax.plot(r['distance_ft'], r[key], mk, color=col, ms=8, mew=2)
                ax.annotate(f"g{r['gauge']}", (r['distance_ft'], r[key]),
                            textcoords='offset points', xytext=(5, 4), fontsize=7)
            if logy:
                ax.set_yscale('log')
            ax.grid(alpha=0.25)
            ax.set_xlabel('distance from source, ft')
            ax.set_ylabel(ylab)
        axes[0][j].axhspan(lo, hi, color='0.85', zorder=0)
        axes[1][j].axhline(acc['max_normalised_rmse'], color='0.4', ls=':')
        ag = st['per_arm'][primary]['aggregate']
        axes[0][j].set_title(
            f"stage {st['stage']} -- {ag['H1_verdict']}\n"
            f"class rule misses {ag['class_split']['n_misclassified']}, "
            f"best distance threshold misses "
            f"{ag['best_distance_split']['n_misclassified']} "
            f"of {ag['class_split']['n_scored']}", fontsize=9)
    fig.suptitle('D3-REP: blind acceptance vs distance, coloured by PRE-REGISTERED '
                 'stimulation class (green virgin, red stimulated; o usable, x not)',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def fig_das(path, results, dpi):
    stages = [r for r in results if r['blind']]
    fig, axes = plt.subplots(2, len(stages), figsize=(6.2 * len(stages), 8.0),
                             squeeze=False)
    for j, st in enumerate(stages):
        ts = st['_series']
        src_md = st['source']['md_ft']
        above = ts['das_md'] > src_md
        ax = axes[0][j]
        for m, lab, c in ((~above, 'below source', 'tab:green'),
                          (above, 'above source', 'tab:red')):
            o = np.argsort(ts['das_dist'][m])
            ax.plot(ts['das_dist'][m][o], ts['das_n_obs'][m][o], color=c, lw=1.0,
                    label=f'LF-DAS, {lab}')
            ax.plot(ts['das_dist'][m][o], ts['das_n_mod'][m][o], color=c, lw=1.0,
                    ls='--', label=f'model, {lab}')
        ax.set_yscale('log')
        ax.set_xlim(0, 2600)
        ax.set_ylim(1e-3, 1e3)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        ax.set_xlabel('distance from source, ft')
        ax.set_ylabel('window RMS, each normalised\nat 500 ft (dimensionless)')
        ax.set_title(f"stage {st['stage']}: SHAPE ONLY. LF-DAS in native counts; "
                     f"no psi-strain coefficient fitted.", fontsize=9)
        ax = axes[1][j]
        for g, v in sorted(st['das']['at_gauge'].items()):
            c = 'tab:green' if v['stim_class'] == 'virgin' else 'tab:red'
            ax.plot(v['distance_ft'], v['pearson_r'], 'o', color=c, ms=8)
            ax.annotate(f"g{g}", (v['distance_ft'], v['pearson_r']),
                        textcoords='offset points', xytext=(5, 4), fontsize=7)
        ax.axhline(0, color='0.4', lw=0.8)
        ax.grid(alpha=0.25)
        ax.set_xlabel('distance from source, ft')
        ax.set_ylabel(r'Pearson $r$, model $\Gamma\,dP/dt$ vs LF-DAS counts')
        ax.set_title('green = virgin rock, red = already stimulated', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t_wall = time.time()
    cfg_path = os.path.abspath(args.config)
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    cfg_sha = sha256_file(cfg_path)

    outdir = os.path.join(REPO, cfg['outputs']['dir'])
    os.makedirs(outdir, exist_ok=True)
    O = {k: os.path.join(REPO, v) for k, v in cfg['outputs'].items()
         if isinstance(v, str) and v.startswith('output/')}

    # --- STEP 1 MUST HAVE PRECEDED STEP 2 ---------------------------------
    prereg_path = O['prereg_json']
    with open(prereg_path) as fh:
        prereg = json.load(fh)
    log(f"pre-registration : {os.path.relpath(prereg_path, REPO)}")
    log(f"  written_utc    : {prereg['written_utc']}")
    log(f"  runner existed : {prereg['runner_existed_at_preregistration']}")
    log(f"  config_sha256  : {prereg['config_sha256']}")
    log(f"  config now     : {cfg_sha}")
    if prereg['config_sha256'] != cfg_sha:
        raise SystemExit(
            "the config has changed since it was pre-registered. This study's "
            "entire value rests on the parameters being frozen; refusing to run.")
    if prereg['runner_existed_at_preregistration']:
        raise SystemExit("the runner already existed when the config was frozen; "
                         "the ordering claim would be false. Refusing to run.")
    log("  MATCH          : True")
    parent_sha, vblocks, deviations = _assert_inherited_verbatim(cfg)
    log(f"inheritance checked: {len(vblocks)} blocks against {PARENT_CFG} "
        f"({parent_sha[:16]}...); every numeric leaf identical; "
        f"{len(deviations)} prose-only deviation(s)")
    for d in deviations:
        log(f"  DISCLOSED prose deviation at {d['path']}: "
            f"this study {d['this_study']!r} vs parent {d['parent']!r}")

    rman.assert_absent([O['manifest'], O['gauge_csv'], O['class_csv'],
                        O['das_csv'], O['control_csv'], O['arrays_npz'],
                        O['summary_json'], O['run_log'], O['fig_overlay'],
                        O['fig_class'], O['fig_das']])

    log('resolving pumping start times for all 20 stages (classification rule)')
    starts = stage_pumping_starts()

    stages_blind = [int(s) for s in cfg['stage_selection']['stages_blind']]
    stage_ctrl = int(cfg['stage_selection']['stage_control'])
    results = []
    for s in stages_blind:
        results.append(run_stage(s, cfg, starts, blind=True))
    results.append(run_stage(stage_ctrl, cfg, starts, blind=False))

    # --- CONTROL: reproduce the parent study's stage-10 table --------------
    primary = cfg['models']['primary']
    parent_csv = os.path.join(REPO, 'output/rev2_20260901/D3/d3_gauge_metrics_v1.csv')
    parent_json = os.path.join(REPO, 'output/rev2_20260901/D3/d3_summary_v1.json')
    parent_full, parent_rounded = _load_parent_control(parent_json, parent_csv)
    ctrl = [r for r in results if r['stage'] == stage_ctrl][0]
    ctrl_rows, worst, worst_csv = [], 0.0, 0.0
    for arm, blk in ctrl['per_arm'].items():
        for r in blk['rows']:
            key = (arm, r['gauge'])
            if key not in parent_full:
                continue
            pf = parent_full[key]
            d_rmse = abs(float(pf['rmse_psi']) - r['rmse_psi'])
            d_peak = abs(float(pf['sim_max_psi']) - r['sim_max_psi'])
            d_amp = (abs(float(pf['amplitude_ratio']) - r['amplitude_ratio'])
                     if pf.get('amplitude_ratio') is not None
                     and r['amplitude_ratio'] is not None else 0.0)
            worst = max(worst, d_rmse, d_peak)
            row = {'arm': arm, 'gauge': r['gauge'],
                   'parent_rmse_psi': float(pf['rmse_psi']),
                   'rerun_rmse_psi': r['rmse_psi'],
                   'abs_diff_rmse_psi': d_rmse,
                   'parent_sim_max_psi': float(pf['sim_max_psi']),
                   'rerun_sim_max_psi': r['sim_max_psi'],
                   'abs_diff_sim_max_psi': d_peak,
                   'abs_diff_amplitude_ratio': d_amp,
                   'usable_parent': pf.get('usable'), 'usable_rerun': r['usable'],
                   'usable_agrees': bool(pf.get('usable') == r['usable'])}
            pr = parent_rounded.get(key)
            if pr is not None:
                dc = abs(float(pr['rmse_psi']) - r['rmse_psi'])
                worst_csv = max(worst_csv, dc)
                row['published_csv_rmse_psi'] = float(pr['rmse_psi'])
                row['abs_diff_vs_published_csv_psi'] = dc
            ctrl_rows.append(row)
    tol = float(cfg['stage_selection']['control_tolerance_psi'])
    ctrl_ok = bool(ctrl_rows) and worst <= tol
    n_usable_agree = sum(1 for r in ctrl_rows if r['usable_agrees'])
    log(f"CONTROL stage {stage_ctrl}: {len(ctrl_rows)} rows vs the parent's "
        f"full-precision d3_summary_v1.json, worst |diff| {worst:.3e} psi, "
        f"pre-registered tolerance {tol:g} -> {'PASS' if ctrl_ok else 'FAIL'}; "
        f"usable verdict agrees on {n_usable_agree}/{len(ctrl_rows)}")
    log(f"  vs the PUBLISHED CSV (rounded to 6 s.f.): worst |diff| "
        f"{worst_csv:.3e} psi -- limited by the CSV's own rounding, not by the run")

    # --- outputs -----------------------------------------------------------
    dpi = int(cfg['outputs']['figure_dpi'])
    gauge_fields = ['stage', 'arm', 'gauge', 'md_ft', 'distance_ft', 'side',
                    'stim_class', 'stim_borderline', 'n_samples', 'rmse_psi',
                    'rmse_normalised', 'bias_psi', 'obs_max_psi', 'sim_max_psi',
                    'amplitude_ratio', 'arrival_err_rel_s', 'arrival_err_abs10_s',
                    'arrival_err_abs25_s', 'has_positive_response', 'usable',
                    'fail_reason']
    with open(O['gauge_csv'], 'w', newline='') as fh:
        wtr = csv.DictWriter(fh, fieldnames=gauge_fields, extrasaction='ignore')
        wtr.writeheader()
        for st in results:
            for arm in list(ARM_ORDER) + ['primary_with_barrier']:
                for r in sorted(st['per_arm'][arm]['rows'],
                                key=lambda x: x['distance_ft']):
                    wtr.writerow(r)

    with open(O['class_csv'], 'w', newline='') as fh:
        wtr = csv.writer(fh)
        wtr.writerow(['stage', 'blind', 'arm', 'n_scored', 'n_usable',
                      'class_misclassified', 'best_distance_threshold_ft',
                      'distance_misclassified', 'H1_verdict',
                      'virgin_usable', 'virgin_not', 'stimulated_usable',
                      'stimulated_not', 'applicability_ft',
                      'rmse_gaugemean_psi', 'rmse_below_source', 'rmse_above_source'])
        for st in results:
            for arm in list(ARM_ORDER) + ['primary_with_barrier']:
                a = st['per_arm'][arm]['aggregate']
                cs, bd, t2 = a['class_split'], a['best_distance_split'], \
                    (a['class_split'] or {}).get('table_2x2', {})
                sd = a['sides']
                wtr.writerow([st['stage'], st['blind'], arm,
                              cs['n_scored'] if cs else None, a['n_usable'],
                              cs['n_misclassified'] if cs else None,
                              bd['threshold_ft'] if bd else None,
                              bd['n_misclassified'] if bd else None,
                              a['H1_verdict'],
                              t2.get('virgin_usable'), t2.get('virgin_not'),
                              t2.get('stimulated_usable'), t2.get('stimulated_not'),
                              a['applicability_distance_ft'],
                              round(a['rmse_gaugemean_psi'], 4),
                              round(sd.get('below_source', {}).get('rmse_gaugemean_psi', float('nan')), 4),
                              round(sd.get('above_source', {}).get('rmse_gaugemean_psi', float('nan')), 4)])

    with open(O['das_csv'], 'w', newline='') as fh:
        wtr = csv.DictWriter(fh, fieldnames=['stage', 'md_ft', 'distance_ft',
                                             'side', 'rms_obs_counts',
                                             'rms_model_strainrate_s^-1',
                                             'rms_obs_normalised',
                                             'rms_model_normalised', 'pearson_r'])
        wtr.writeheader()
        for st in results:
            for r in st['das']['rows']:
                wtr.writerow(r)

    with open(O['control_csv'], 'w', newline='') as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(ctrl_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(ctrl_rows)

    npz = {}
    for st in results:
        s = st['stage']
        ts = st['_series']
        for t in ts['targets']:
            g = int(t['gauge'])
            npz[f's{s}_g{g}_t'] = t['taxis']
            npz[f's{s}_g{g}_obs'] = t['data']
            npz[f's{s}_g{g}_sim_primary'] = ts['sim_primary'][g]
        npz[f's{s}_src_t'] = ts['src'].taxis_s
        npz[f's{s}_src_obs'] = ts['src'].delta_psi
        npz[f's{s}_das_dist'] = ts['das_dist']
        npz[f's{s}_das_md'] = ts['das_md']
        npz[f's{s}_das_n_obs'] = ts['das_n_obs']
        npz[f's{s}_das_n_mod'] = ts['das_n_mod']
        npz[f's{s}_das_corr'] = ts['das_corr']
    np.savez_compressed(O['arrays_npz'], **npz)

    fig_overlay(O['fig_overlay'], results, primary, dpi)
    fig_class(O['fig_class'], results, primary, cfg['acceptance'], dpi)
    fig_das(O['fig_das'], results, dpi)
    log('figures written')

    clean = []
    for st in results:
        c = {k: v for k, v in st.items() if k != '_series'}
        c['per_arm'] = {a: {'aggregate': b['aggregate'], 'rows': b['rows']}
                        for a, b in st['per_arm'].items()}
        c['das'] = {k: v for k, v in st['das'].items() if k != 'rows'}
        clean.append(c)
    results_block = {
        'stages_blind': stages_blind, 'stage_control': stage_ctrl,
        'primary_model': primary,
        'inheritance_check': {
            'parent_config': PARENT_CFG, 'parent_sha256': parent_sha,
            'blocks_declared_verbatim': vblocks,
            'n_prose_only_deviations': len(deviations),
            'prose_only_deviations': deviations,
            'note': ('Every NUMERIC, boolean and null leaf of the ten inherited '
                     'blocks is identical to the parent config; the runner '
                     'refuses to start otherwise. Any leaf listed here differs '
                     'in free text only and changes no computation.')},
        'control_check': {
            'n_rows': len(ctrl_rows),
            'worst_abs_diff_psi': worst,
            'worst_abs_diff_vs_published_csv_psi': worst_csv,
            'tolerance_psi': tol, 'pass': ctrl_ok,
            'n_usable_verdict_agrees': n_usable_agree,
            'compared_against': os.path.relpath(parent_json, REPO),
            'also_compared_against': os.path.relpath(parent_csv, REPO),
            'note': ('The pre-registered 1e-6 psi tolerance is checked against '
                     'the parent run\'s full-precision summary JSON. The '
                     'published CSV is rounded to six significant figures, so '
                     'its column is reported separately and its residual is a '
                     'property of that rounding.')},
        'H1_by_stage': {str(st['stage']):
                        st['per_arm'][primary]['aggregate']['H1_verdict']
                        for st in results if st['blind']},
        'headline': {str(st['stage']): {
            'gauge_mean_rmse_psi': st['per_arm'][primary]['aggregate']['rmse_gaugemean_psi'],
            'applicability_ft': st['per_arm'][primary]['aggregate']['applicability_distance_ft'],
            'n_usable': st['per_arm'][primary]['aggregate']['n_usable'],
            'n_scored': st['per_arm'][primary]['aggregate']['n_scored_for_acceptance'],
            'sides': st['per_arm'][primary]['aggregate']['sides'],
        } for st in results},
        'stages': clean,
    }
    with open(O['summary_json'], 'w') as fh:
        json.dump(results_block, fh, indent=1, default=jsonable)

    # --- manifest ----------------------------------------------------------
    ctrl_ts = ctrl['_series']
    drv = rman.driver_record(
        kind='gauge_series', baseline_removal=cfg['source']['baseline_removal'],
        value_units='delta_psi',
        series_path=os.path.join(REPO, cfg['data']['gauge_series_template']
                                 .format(n=ctrl['source']['gauge'])),
        gauge_number=int(ctrl['source']['gauge']),
        gauge_md_ft=float(ctrl['source']['md_ft']),
        taxis=ctrl_ts['src'].taxis_s, values=ctrl_ts['src'].delta_psi,
        time_start=ctrl['window']['t_start'], time_end=ctrl['window']['t_end'])
    # Every stage is solved on the SAME mesh (the window MD span is the gauge
    # array and the pads are fixed), so one mesh record describes all three.
    mesh_x = rdata.build_mesh(
        rdata.Window(md_min_ft=float(cfg['window']['md_min_ft']),
                     md_max_ft=float(cfg['window']['md_max_ft']),
                     t_start=datetime.datetime.fromisoformat(ctrl['window']['t_start']),
                     t_end=datetime.datetime.fromisoformat(ctrl['window']['t_end'])),
        float(cfg['mesh']['domain_pad_low_md_ft']),
        float(cfg['mesh']['domain_pad_high_md_ft']),
        float(cfg['mesh']['dx_ft'])).x
    srec = rman.source_record(
        mesh_x, md_requested_ft=float(ctrl['source']['md_ft']),
        mesh_idx=int(ctrl['source']['mesh_idx']), driver=drv,
        label=f"g{ctrl['source']['gauge']} (stage {stage_ctrl}, control; the two "
              f"blind stages use their own source gauges, recorded in results)",
        index_in_source_list=0)
    sp = rman.source_protocol(
        application=cfg['source']['application'],
        solver_class=cfg['solver']['class'],
        placement_rule=cfg['source']['selection_rule'],
        sources=[srec],
        targets=[{'gauge': int(t['gauge']), 'md_ft': float(t['md_ft']),
                  'distance_ft': float(t['distance_ft']),
                  'mesh_idx': int(t['idx'])} for t in ctrl_ts['targets']],
        time_level=cfg['solver']['source_time_level'],
        phase_chaining=rman.NONE_DECLARED,
        boundary_conditions={'lbc': cfg['solver']['lbc'],
                             'rbc': cfg['solver']['rbc'],
                             'source_node': 'Dirichlet'})
    brecs = rman.NONE_DECLARED
    bsec = cfg['barrier']['secondary_sensitivity']
    if ctrl['barrier_report'] is not None:
        brecs = []
        for bi, b in enumerate(ctrl['barrier_report']['barriers']):
            msk = np.zeros(mesh_x.size, dtype=bool)
            msk[b['i0']:b['i1'] + 1] = True
            brecs.append(rman.barrier_record(
                mesh_x, msk, label=f"stage{stage_ctrl}_frachit_{bi}",
                centre_md_ft=float(b['md_ft']),
                w_requested_ft=float(bsec['w_half_ft']),
                ratio=float(bsec['ratio']),
                d_baseline=float(ctrl['barrier_d_base'][bi]),
                report=b))
    models = cfg['models']
    num = rman.numerics(
        time=[rman.time_record(ctrl['_series']['solves'][primary][0],
                               mode='fixed', theta=float(cfg['solver']['theta']),
                               t_total_requested_s=ctrl['window']['t_total_source_s'],
                               dt_requested_s=float(cfg['solver']['dt_s']),
                               source_time_level=cfg['solver']['source_time_level'],
                               label=f'primary, control stage {stage_ctrl}')],
        mesh=rman.mesh_record(mesh_x, dx_requested_ft=float(cfg['mesh']['dx_ft']),
                              window_md_ft=(float(cfg['window']['md_min_ft']),
                                            float(cfg['window']['md_max_ft'])),
                              pad_low_ft=float(cfg['mesh']['domain_pad_low_md_ft']),
                              pad_high_ft=float(cfg['mesh']['domain_pad_high_md_ft'])),
        interface_avg=cfg['solver']['interface_avg'],
        boundary={'lbc': cfg['solver']['lbc'], 'rbc': cfg['solver']['rbc'],
                  'pml_thickness': 0.0, 'sigma_max': 0.0},
        diffusivity={'profile_family': 'multiple (four frozen arms + barrier arm)',
                     'primary_family': models[primary]['family'],
                     'param_names': models[primary].get('param_names', []),
                     'params': models[primary]['params'],
                     'baseline_D_ft2_s': float(models['uniform_absolute']['params'][0]),
                     'profile_anchor': 'distance from the source node of each stage',
                     'arms': {n: models[n]['params'] for n in ARM_ORDER},
                     'note': 'every value copied verbatim from the pre-registered '
                             'parent config; nothing fitted on any stage'},
        barriers=brecs,
        leakage={'lambda_leak_s^-1': 0.0,
                 'note': 'C2: fitted lambda is 0 exactly, censored null'},
        kernel={'name': 'rev2_core.solve_forward', 'banded': True,
                'theta': float(cfg['solver']['theta']), 'lambda_leak': 0.0,
                'equivalence_reference':
                    'bitwise identical to r1_calibration_core.solve_forward at '
                    'theta=1/harmonic/lambda=0 (rev2 self-test)'},
        rng=rman.NONE_DECLARED,
        parallel={'processes_used': 1, 'cap': int(cfg['compute']['processes']),
                  'note': 'three stages solved sequentially; no Pool, so no '
                          'worker code closure to declare'})

    all_stages = stages_blind + [stage_ctrl]
    inputs = [(os.path.join(REPO, cfg['data']['gauge_md_npz']), 'geometry', 'gauge_md'),
              (os.path.join(REPO, cfg['data']['well_geometry_npz']), 'geometry',
               'swell_geometry'),
              ('output/r1_baseline_calibration/r1_run_manifest.json',
               'prior_run_output', 'r1_manifest'),
              ('output/r2_diffusivity_profile/r2_manifest.json',
               'prior_run_output', 'r2_manifest'),
              (os.path.join(REPO, PARENT_CFG), 'config', 'parent_config_d3_blind'),
              (os.path.join(REPO, 'output/rev2_20260901/D3/PREREGISTERED.json'),
               'prior_run_output', 'parent_preregistration'),
              (parent_csv, 'prior_run_output', 'parent_stage10_gauge_metrics'),
              (parent_json, 'prior_run_output', 'parent_stage10_summary')]
    for n in range(1, 16):
        inputs.append((os.path.join(REPO, cfg['data']['gauge_series_template']
                                    .format(n=n)), 'gauge_series', f'gauge{n}'))
    for s in all_stages:
        inputs.append((rdata.repo_path(rdata.SWELL_DAS_TEMPLATE.format(
            stage=s, kind='')), 'das', f'lfdas_stage{s}'))
    # the classification rule reads every stage's hits and pumping curves
    for s in range(1, N_STAGES_TOTAL + 1):
        inputs.append((rdata.repo_path(rdata.SWELL_FRAC_HIT_TEMPLATE.format(
            stage=s)), 'geometry', f'frac_hit_stage{s}'))
        for k, v in rdata.PUMPING_CURVE_FILES.items():
            p = os.path.join(REPO, rdata.PUMPING_DIR_TEMPLATE.format(stage=s), v)
            if os.path.exists(p):
                inputs.append((p, 'pumping', f'stage{s}_{k}'))

    out_decls = [
        rman.output_decl(prereg_path, role='json',
                         note='pre-registration, written before this runner existed'),
        rman.output_decl(prereg_path + '.sha256', role='other',
                         note='sha256 sidecar of the pre-registration'),
        rman.output_decl(O['gauge_csv'], role='csv',
                         note='per-gauge blind metrics, three stages x five arms'),
        rman.output_decl(O['class_csv'], role='csv',
                         note='H1 test: class rule vs best distance threshold'),
        rman.output_decl(O['das_csv'], role='csv',
                         note='per-DAS-channel shape statistics, native counts'),
        rman.output_decl(O['control_csv'], role='csv',
                         note='stage-10 control against the parent study'),
        rman.output_decl(O['arrays_npz'], role='arrays_npz',
                         note='observed and simulated series, DAS statistics'),
        rman.output_decl(O['summary_json'], role='json', note='full results'),
        rman.output_decl(O['run_log'], role='log', note='run log'),
        rman.output_decl(O['fig_overlay'], role='figure_png', dpi=dpi),
        rman.output_decl(O['fig_class'], role='figure_png', dpi=dpi),
        rman.output_decl(O['fig_das'], role='figure_png', dpi=dpi)]

    log(f"total wall {time.time() - t_wall:.1f} s")
    # The log is written HERE, before write_manifest hashes it, and is never
    # reopened. Both parent runners reopened theirs and drifted their manifests
    # (output/rev2_20260901/D3/_notes/README.md); this is that fix.
    with open(O['run_log'], 'w') as fh:
        fh.write('\n'.join(_LOG_LINES) + '\n')

    doc = rman.write_manifest(
        O['manifest'], study_id=cfg['study_id'], task_id=TASK_ID, config=cfg,
        config_path=cfg_path, inputs=inputs, source=sp, numerics=num,
        outputs=out_decls, results=results_block, started_utc=started,
        run_label=f'D3-REP blind replication, stages {stages_blind} '
                  f'+ control {stage_ctrl} ({VERSION})',
        require_modules=('rev2_core', 'rev2_manifest', 'rev2_data',
                         'r1_calibration_core'),
        notes=[
            'PRE-REGISTERED. configs/rev2/d3_blind_rep.json was written and '
            'hashed at ' + prereg['written_utc'] + ', recorded in '
            'PREREGISTERED.json at a moment when this runner did not exist, and '
            'only then was the runner written. The run aborts unless the config '
            'still hashes to the pre-registered value.',
            'Every physical parameter is copied verbatim from the parent config '
            'configs/rev2/d3_blind.json (sha256 ' + parent_sha + '), which was '
            'itself pre-registered before stage 10 was solved. The runner '
            're-checks all ten inherited blocks against the parent before '
            'solving. There is no optimiser in this script.',
            'H1 (that blind skill separates by stimulation class rather than by '
            'distance) was registered as falsifiable BEFORE the solves, together '
            'with the classification rule, which uses frac-hit MDs, gauge MDs '
            'and pumping timestamps only and reads no pressure or DAS.',
            'LF-DAS is kept in native counts. The comparison is SHAPE ONLY '
            '(scale-invariant Pearson r and RMS profiles normalised at 500 ft); '
            'no psi<->strain coefficient is fitted anywhere.',
            ('Inheritance: all ten blocks checked leaf by leaf against the '
             'parent; every numeric leaf identical; '
             + str(len(deviations)) + ' prose-only deviation(s), listed in '
             'results.inheritance_check and in the config erratum.'),
            'Stage 10 is a CONTROL, not blind evidence: it re-runs the parent '
            'study under the modules as repaired by A4_repair2 and must '
            'reproduce output/rev2_20260901/D3/d3_gauge_metrics_v1.csv.',
        ])
    print(f"manifest written: {O['manifest']}")
    rep = rman.verify(O['manifest'], repo_root=REPO)
    print(f"manifest verify: status={rep['status']}")
    if rep['status'] != 'clean':
        print('  VERIFY DETAIL: ' + json.dumps(rep)[:3000])
    return doc


if __name__ == '__main__':
    main()
