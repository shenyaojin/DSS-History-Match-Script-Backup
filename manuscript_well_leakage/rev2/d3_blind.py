#!/usr/bin/env python3
"""D3 -- blind forward prediction of fracturing stage 10.

Every physical and numerical parameter is READ FROM THE CONFIG
(`configs/rev2/d3_blind.json`), which was written, hashed and pre-registered in
`output/rev2_20260901/D3/PREREGISTERED.json` before this file existed. There is
no optimiser anywhere in this script and no stage-10 observation is used to set
any parameter: the only things stage-10 data are used for are (a) the Dirichlet
drive at the source gauge, which is an input, and (b) scoring.

Run from the repo root:

    python3 scripts/manuscript_well_leakage/rev2/d3_blind.py \
        --config configs/rev2/d3_blind.json
"""

import argparse
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


def iso(dt):
    return dt.isoformat() if isinstance(dt, datetime.datetime) else str(dt)


# ---------------------------------------------------------------------------
# frozen model construction (no fitting anywhere)
# ---------------------------------------------------------------------------

def build_profile(mesh_x, source_idx, spec):
    """D(x) for one frozen model spec. Literal parameters only."""
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
# metrics
# ---------------------------------------------------------------------------

def score_gauge(t_sim, y_sim, t_obs, y_obs, thresholds_abs, frac):
    """RMSE / amplitude ratio / arrival errors for one gauge on its own clock."""
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
    """Gauge-mean and sample-pooled aggregates, plus the acceptance verdict."""
    mse = [r['rmse_psi'] ** 2 for r in rows]
    norm = [r['rmse_normalised'] ** 2 for r in rows
            if r['rmse_normalised'] is not None]
    ss = sum(r['sum_sq'] for r in rows)
    n = sum(r['n_samples'] for r in rows)
    lo, hi = acc['amplitude_ratio_band']
    usable = []
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
        usable.append(r)
    failed = [r for r in usable if not r['usable']]
    applic = min((r['distance_ft'] for r in failed), default=None)
    return {
        'n_gauges': len(rows),
        'n_scored_for_acceptance': len(usable),
        'rmse_gaugemean_psi': float(np.sqrt(np.mean(mse))) if mse else None,
        'rmse_pooled_psi': float(np.sqrt(ss / n)) if n else None,
        'rmse_normalised_gaugemean': (float(np.sqrt(np.mean(norm)))
                                      if norm else None),
        'n_usable': int(sum(1 for r in usable if r['usable'])),
        'n_failed': len(failed),
        'applicability_distance_ft': (float(applic) if applic is not None
                                      else None),
        'applicability_note': (
            'distance of the NEAREST scored gauge that fails the pre-registered '
            'acceptance rule; None means no scored gauge failed'),
    }


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

    # --- the pre-registration must exist and must match -------------------
    prereg_path = O['prereg_json']
    with open(prereg_path) as fh:
        prereg = json.load(fh)
    log(f"pre-registration : {os.path.relpath(prereg_path, REPO)}")
    log(f"  written_utc    : {prereg['written_utc']}")
    log(f"  config_sha256  : {prereg['config_sha256']}")
    log(f"  config now     : {cfg_sha}")
    prereg_ok = (prereg['config_sha256'] == cfg_sha)
    log(f"  MATCH          : {prereg_ok}")
    if not prereg_ok:
        raise SystemExit(
            "the config has changed since it was pre-registered. D3's entire "
            "value rests on the parameters being frozen; refusing to run.")

    rman.assert_absent([O['manifest'], O['summary_csv'], O['arrays_npz'],
                        O['summary_json'], O['fig_overlay'], O['fig_distance'],
                        O['fig_das'], O['slices_csv'], O['das_csv'],
                        O['run_log']])

    # --- data --------------------------------------------------------------
    stage = int(cfg['stage_selection']['stage'])
    w = cfg['window']
    win = rdata.Window(md_min_ft=float(w['md_min_ft']),
                       md_max_ft=float(w['md_max_ft']),
                       t_start=datetime.datetime.fromisoformat(w['time_start']),
                       t_end=datetime.datetime.fromisoformat(w['time_end']))
    log(f"stage {stage}; window MD [{win.md_min_ft}, {win.md_max_ft}] ft, "
        f"{win.t_start} .. {win.t_end} ({win.duration_s:.0f} s)")

    gw = rdata.load_window_gauges(win, baseline='first_sample',
                                 rebase=cfg['source']['rebase'])
    fhits = rdata.load_frac_hits(stage)
    src_gauge, centroid = rdata.pick_source_gauge(gw, fhits)
    src = gw.series[src_gauge]
    log(f"frac hits ({fhits.size}): {list(np.round(fhits, 3))}; centroid "
        f"{centroid:.3f} ft")
    log(f"source gauge g{src_gauge} at MD {src.md_ft} ft "
        f"({src.md_ft - centroid:+.3f} ft from the centroid)")
    if int(src_gauge) != int(cfg['source']['resolved_source_gauge']):
        raise SystemExit("resolved source gauge differs from the pre-registered "
                         "one; refusing to run")

    mesh = rdata.build_mesh(win, float(cfg['mesh']['domain_pad_low_md_ft']),
                            float(cfg['mesh']['domain_pad_high_md_ft']),
                            float(cfg['mesh']['dx_ft']))
    source_idx = mesh.index_of(src.md_ft)
    log(f"mesh nx={mesh.nx} [{mesh.x[0]:.1f}, {mesh.x[-1]:.1f}] ft, "
        f"dx={mesh.dx_ft} ft; source node {source_idx} "
        f"(snap {mesh.snap_error_ft(src.md_ft):+.3f} ft)")

    targets = rdata.make_targets(gw, mesh, src.md_ft,
                                 exclude_gauges=(src_gauge,))
    for t in targets:
        t['t0_abs'] = gw.series[t['gauge']].t0_abs
    t0_spread = max((t['t0_abs'] - src.t0_abs).total_seconds() for t in targets) \
        - min((t['t0_abs'] - src.t0_abs).total_seconds() for t in targets)
    log(f"{len(targets)} target gauges; per-gauge first-sample spread "
        f"{t0_spread:.3f} s (rebase='{gw.rebase}')")

    dt_s = float(cfg['solver']['dt_s'])
    t_total = float(src.t_total_s)
    rec_idx = [t['idx'] for t in targets]

    # --- frozen model arms -------------------------------------------------
    models = cfg['models']
    arm_names = [k for k in ('two_zone_r2', 'uniform_absolute',
                             'uniform_normalised', 'uniform_manuscript')
                 if k in models]
    primary = models['primary']

    solves = {}
    profiles = {}
    for name in arm_names:
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
        log(f"solve {name:20s} D[src]={prof[source_idx]:9.1f} "
            f"D[far]={prof[0]:8.1f}  {rec.shape[0]} steps  {time.time()-tic:5.1f} s")

    # barrier sensitivity arm (primary model only, pre-registered)
    bsec = cfg['barrier']['secondary_sensitivity']
    prof_b, brep = apply_barrier(mesh.x, profiles[primary].copy(), list(fhits),
                                 float(bsec['ratio']), float(bsec['w_half_ft']))
    tic = time.time()
    taxis_b, rec_b = rcore.solve_forward(
        mesh.x, prof_b, dt_s, t_total, src.taxis_s, src.delta_psi, source_idx,
        record_idx=rec_idx, theta=float(cfg['solver']['theta']),
        source_time_level=cfg['solver']['source_time_level'],
        lambda_leak=0.0, interface_avg=cfg['solver']['interface_avg'])
    solves['primary_with_barrier'] = (taxis_b, rec_b)
    log(f"solve primary_with_barrier ratio={bsec['ratio']:g} "
        f"w_half={bsec['w_half_ft']} ft, realised full width "
        f"{brep['realised_full_width_ft'] if brep else 'n/a'} ft, "
        f"n_fallback={brep['n_fallback'] if brep else 'n/a'}"
        f"  {time.time()-tic:5.1f} s")
    if brep is not None and brep['n_fallback'] != 0:
        log(f"  WARNING barrier fallback fired {brep['n_fallback']} time(s): "
            f"{brep['fallback_messages'][:2]}")

    # --- per-gauge scoring -------------------------------------------------
    frac = float(cfg['metrics']['arrival_time']['relative_fraction'])
    thr_abs = list(cfg['metrics']['arrival_time']['absolute_thresholds_psi'])
    acc = cfg['acceptance']

    per_arm = {}
    sim_series = {}
    for name, (taxis, rec) in solves.items():
        rows = []
        sims = {}
        for k, t in enumerate(targets):
            r, sim = score_gauge(taxis, rec[:, k], t['taxis'], t['data'],
                                 thr_abs, frac)
            r.update(gauge=int(t['gauge']), md_ft=float(t['md_ft']),
                     distance_ft=float(t['distance_ft']),
                     side=('above_source' if t['md_ft'] > src.md_ft
                           else 'below_source'))
            rows.append(r)
            sims[int(t['gauge'])] = sim
        rows.sort(key=lambda r: r['distance_ft'])
        per_arm[name] = {'per_gauge': rows, 'aggregate': aggregate(rows, acc)}
        sim_series[name] = sims
        a = per_arm[name]['aggregate']
        log(f"  {name:22s} gauge-mean RMSE {a['rmse_gaugemean_psi']:8.2f} psi, "
            f"normalised {a['rmse_normalised_gaugemean']}, usable "
            f"{a['n_usable']}/{a['n_scored_for_acceptance']}, applicability "
            f"{a['applicability_distance_ft']} ft")

    # --- absolute-UTC arrival cross-check (metric diagnostic, not a refit) --
    utc_rows = []
    taxis_p, rec_p = solves[primary]
    for k, t in enumerate(targets):
        obs_abs = (t['t0_abs'] - src.t0_abs).total_seconds() + t['taxis']
        sim_on_obs = np.interp(obs_abs, taxis_p, rec_p[:, k])
        obs_max = float(np.max(t['data']))
        if obs_max <= 0:
            utc_rows.append({'gauge': int(t['gauge']),
                             'distance_ft': float(t['distance_ft']),
                             'arrival_err_rel_s_utc': None})
            continue
        thr = frac * obs_max
        t_o = r1c.arrival_time(obs_abs, t['data'], thr)
        t_s = r1c.arrival_time(obs_abs, sim_on_obs, thr)
        utc_rows.append({'gauge': int(t['gauge']),
                         'distance_ft': float(t['distance_ft']),
                         'arrival_err_rel_s_utc': (None if (np.isnan(t_o) or np.isnan(t_s))
                                                   else float(t_s - t_o))})

    # --- reporting slices --------------------------------------------------
    slice_rows = []
    for (lo, hi) in cfg['window']['reporting_slices_s']:
        for name, (taxis, rec) in solves.items():
            for k, t in enumerate(targets):
                m = (t['taxis'] >= lo) & (t['taxis'] < hi)
                if m.sum() < 5:
                    continue
                sim = np.interp(t['taxis'][m], taxis, rec[:, k])
                obs = t['data'][m]
                resid = sim - obs
                omax = float(np.max(obs))
                slice_rows.append({
                    'slice_lo_s': float(lo), 'slice_hi_s': float(hi),
                    'model': name, 'gauge': int(t['gauge']),
                    'distance_ft': float(t['distance_ft']),
                    'n_samples': int(m.sum()),
                    'rmse_psi': float(np.sqrt(np.mean(resid ** 2))),
                    'obs_max_psi': omax, 'sim_max_psi': float(np.max(sim)),
                    'amplitude_ratio': (float(np.max(sim) / omax) if omax > 0
                                        else None)})

    # --- numerics verification (not a parameter search) --------------------
    checks = {}
    pads = [float(p) for p in cfg['numerics_checks']['padding_sensitivity']['pads_ft']]
    pad_res = []
    base_peak = np.max(solves[primary][1], axis=0)
    base_rmse = np.array([r['rmse_psi'] for r in
                          sorted(per_arm[primary]['per_gauge'],
                                 key=lambda r: r['gauge'])])
    for pad in pads:
        if pad == float(cfg['mesh']['domain_pad_low_md_ft']):
            continue
        m2 = rdata.build_mesh(win, pad, pad, float(cfg['mesh']['dx_ft']))
        si2 = m2.index_of(src.md_ft)
        p2 = build_profile(m2.x, si2, models[primary])
        ri2 = [m2.index_of(t['md_ft']) for t in targets]
        tic = time.time()
        tx2, rc2 = rcore.solve_forward(m2.x, p2, dt_s, t_total, src.taxis_s,
                                       src.delta_psi, si2, record_idx=ri2,
                                       theta=1.0, source_time_level='n',
                                       interface_avg='harmonic')
        pk2 = np.max(rc2, axis=0)
        rm2 = []
        for k, t in enumerate(targets):
            sim = np.interp(t['taxis'], tx2, rc2[:, k])
            rm2.append(float(np.sqrt(np.mean((sim - t['data']) ** 2))))
        rm2 = np.array(rm2)[np.argsort([t['gauge'] for t in targets])]
        d_peak = np.abs(pk2 - base_peak)
        pad_res.append({
            'pad_ft': pad, 'nx': int(m2.nx),
            'max_abs_peak_change_psi': float(np.max(d_peak)),
            'max_rel_peak_change_pct': float(np.max(
                100.0 * d_peak / np.maximum(np.abs(base_peak), 1e-12))),
            'max_abs_rmse_change_psi': float(np.max(np.abs(rm2 - base_rmse))),
            'wall_s': round(time.time() - tic, 1)})
        log(f"padding check pad={pad:.0f} ft: max |dpeak| "
            f"{pad_res[-1]['max_abs_peak_change_psi']:.4f} psi "
            f"({pad_res[-1]['max_rel_peak_change_pct']:.3f} %), max |dRMSE| "
            f"{pad_res[-1]['max_abs_rmse_change_psi']:.4f} psi")
    checks['padding_sensitivity'] = {'reference_pad_ft':
                                     float(cfg['mesh']['domain_pad_low_md_ft']),
                                     'rows': pad_res}

    dt_res = []
    for dtc in [float(x) for x in cfg['numerics_checks']['dt_check']['dt_s']]:
        if dtc == dt_s:
            continue
        tic = time.time()
        tx3, rc3 = rcore.solve_forward(mesh.x, profiles[primary], dtc, t_total,
                                       src.taxis_s, src.delta_psi, source_idx,
                                       record_idx=rec_idx, theta=1.0,
                                       source_time_level='n',
                                       interface_avg='harmonic')
        rm3 = []
        for k, t in enumerate(targets):
            sim = np.interp(t['taxis'], tx3, rc3[:, k])
            rm3.append(float(np.sqrt(np.mean((sim - t['data']) ** 2))))
        rm3 = np.array(rm3)[np.argsort([t['gauge'] for t in targets])]
        pk3 = np.max(rc3, axis=0)
        dt_res.append({
            'dt_s': dtc, 'n_steps': int(rc3.shape[0]),
            'max_abs_peak_change_psi': float(np.max(np.abs(pk3 - base_peak))),
            'max_rel_peak_change_pct': float(np.max(
                100.0 * np.abs(pk3 - base_peak)
                / np.maximum(np.abs(base_peak), 1e-12))),
            'max_abs_rmse_change_psi': float(np.max(np.abs(rm3 - base_rmse))),
            'wall_s': round(time.time() - tic, 1)})
        log(f"dt check dt={dtc} s: max |dpeak| "
            f"{dt_res[-1]['max_abs_peak_change_psi']:.4f} psi, max |dRMSE| "
            f"{dt_res[-1]['max_abs_rmse_change_psi']:.4f} psi")
    checks['dt_check'] = {'reference_dt_s': dt_s, 'rows': dt_res}

    # --- LF-DAS shape comparison ------------------------------------------
    das = cfg['das']
    gamma = float(cfg['gamma']['value_psi^-1'])
    log("loading LF-DAS ...")
    rec_das = rdata.load_das_stage(stage, kind=das['kind'],
                                   md_range=(float(mesh.x[0]), float(mesh.x[-1])),
                                   time_range=(win.t_start, win.t_end))
    log(f"DAS: {rec_das.data.shape[0]} channels MD "
        f"[{rec_das.daxis_ft[0]:.2f}, {rec_das.daxis_ft[-1]:.2f}], "
        f"{rec_das.data.shape[1]} samples from {rec_das.t0_abs}")
    keep = rec_das.artifact_mask(auto=True, z=20.0, pad_s=2.0)
    n_drop = int(keep.size - keep.sum())
    log(f"DAS artifact mask (auto, z=20): {n_drop} of {keep.size} samples dropped")

    das_md = rec_das.daxis_ft.copy()
    das_t0 = rec_das.t0_abs
    das_taxis = rec_das.taxis_s.copy()
    das_idx = [mesh.index_of(m) for m in das_md]
    tic = time.time()
    tx_d, rec_d = rcore.solve_forward(
        mesh.x, profiles[primary], dt_s, t_total, src.taxis_s, src.delta_psi,
        source_idx, record_idx=das_idx, theta=1.0, source_time_level='n',
        interface_avg='harmonic')
    log(f"DAS-node solve: {rec_d.shape} in {time.time()-tic:.1f} s")

    # model strain rate = Gamma * dP/dt, resampled onto the DAS absolute clock
    off = (das_t0 - src.t0_abs).total_seconds()
    t_das_model = das_taxis + off
    n_ch = das_md.size
    model_sr = np.empty((n_ch, das_taxis.size), dtype=np.float32)
    CH = 256
    for a in range(0, n_ch, CH):
        b = min(a + CH, n_ch)
        blk = np.gradient(rec_d[:, a:b], tx_d, axis=0) * gamma
        for j in range(a, b):
            model_sr[j, :] = np.interp(t_das_model, tx_d, blk[:, j - a])
        del blk
    del rec_d

    obs_all = rec_das.data                       # (n_ch, n_time)
    rms_obs_raw = np.empty(n_ch); rms_mod_raw = np.empty(n_ch)
    rms_obs_dc = np.empty(n_ch); rms_mod_dc = np.empty(n_ch)
    corr = np.empty(n_ch)
    nk = int(keep.sum())
    for a in range(0, n_ch, CH):
        b = min(a + CH, n_ch)
        o = np.asarray(obs_all[a:b, :], dtype=np.float64)[:, keep]
        m = model_sr[a:b, :].astype(np.float64)[:, keep]
        om = o.mean(axis=1); mm = m.mean(axis=1)
        o2 = (o ** 2).mean(axis=1); m2 = (m ** 2).mean(axis=1)
        cross = (o * m).mean(axis=1)
        vo = np.maximum(o2 - om ** 2, 0.0)
        vm = np.maximum(m2 - mm ** 2, 0.0)
        rms_obs_raw[a:b] = np.sqrt(o2)
        rms_mod_raw[a:b] = np.sqrt(m2)
        rms_obs_dc[a:b] = np.sqrt(vo)
        rms_mod_dc[a:b] = np.sqrt(vm)
        with np.errstate(invalid='ignore', divide='ignore'):
            c = (cross - om * mm) / np.sqrt(vo * vm)
        corr[a:b] = np.where((vo > 0) & (vm > 0), c, np.nan)
        del o, m
    del model_sr
    log(f"DAS statistics over {nk} retained samples, {n_ch} channels")

    ch_dist = np.abs(das_md - src.md_ft)
    ref_d = float(das['reference_distance_ft'])
    ref_h = float(das['reference_halfwidth_ft'])
    ref_m = np.abs(ch_dist - ref_d) <= ref_h
    ref_obs = float(np.mean(rms_obs_dc[ref_m])) if ref_m.any() else np.nan
    ref_mod = float(np.mean(rms_mod_dc[ref_m])) if ref_m.any() else np.nan
    log(f"DAS reference band {ref_d}+-{ref_h} ft: {int(ref_m.sum())} channels; "
        f"obs RMS {ref_obs:.4g} counts, model RMS {ref_mod:.4g} s^-1")

    n_obs = rms_obs_dc / ref_obs if ref_obs > 0 else np.full_like(rms_obs_dc, np.nan)
    n_mod = rms_mod_dc / ref_mod if ref_mod > 0 else np.full_like(rms_mod_dc, np.nan)

    das_rows = []
    for j in range(ch_dist.size):
        das_rows.append({'md_ft': float(das_md[j]),
                         'distance_ft': float(ch_dist[j]),
                         'side': ('above_source' if das_md[j] > src.md_ft
                                  else 'below_source'),
                         'rms_obs_counts_dc': float(rms_obs_dc[j]),
                         'rms_model_strainrate_dc': float(rms_mod_dc[j]),
                         'rms_obs_counts_raw': float(rms_obs_raw[j]),
                         'rms_model_strainrate_raw': float(rms_mod_raw[j]),
                         'norm_obs': float(n_obs[j]), 'norm_model': float(n_mod[j]),
                         'shape_ratio_model_over_obs': (
                             float(n_mod[j] / n_obs[j]) if n_obs[j] > 0 else None),
                         'pearson_r': (None if not np.isfinite(corr[j])
                                       else float(corr[j]))})

    das_at_gauge = []
    for t in targets:
        j = int(np.argmin(np.abs(das_md - t['md_ft'])))
        das_at_gauge.append({
            'gauge': int(t['gauge']), 'md_ft': float(t['md_ft']),
            'distance_ft': float(t['distance_ft']),
            'das_channel_md_ft': float(das_md[j]),
            'das_channel_offset_ft': float(das_md[j] - t['md_ft']),
            'pearson_r': (None if not np.isfinite(corr[j]) else float(corr[j])),
            'norm_obs': float(n_obs[j]), 'norm_model': float(n_mod[j]),
            'shape_ratio_model_over_obs': (float(n_mod[j] / n_obs[j])
                                           if n_obs[j] > 0 else None)})
    for r in sorted(das_at_gauge, key=lambda r: r['distance_ft']):
        log(f"  DAS g{r['gauge']:<3d} d={r['distance_ft']:7.1f} ft  r="
            f"{r['pearson_r'] if r['pearson_r'] is None else round(r['pearson_r'],4)}"
            f"  shape ratio {r['shape_ratio_model_over_obs']}")

    # --- products ----------------------------------------------------------
    hdr = ['model', 'gauge', 'md_ft', 'distance_ft', 'side', 'n_samples',
           'rmse_psi', 'rmse_normalised', 'bias_psi', 'obs_max_psi',
           'sim_max_psi', 'amplitude_ratio', 'arrival_obs_rel_s',
           'arrival_sim_rel_s', 'arrival_err_rel_s', 'arrival_err_abs10_s',
           'arrival_err_abs25_s', 'has_positive_response', 'usable',
           'fail_reason']
    with open(O['summary_csv'], 'w') as fh:
        fh.write(','.join(hdr) + '\n')
        for name in solves:
            for r in per_arm[name]['per_gauge']:
                vals = [name] + [('' if r.get(k) is None else
                                  (f"{r[k]:.6g}" if isinstance(r.get(k), float)
                                   else str(r.get(k)))) for k in hdr[1:]]
                fh.write(','.join(vals) + '\n')

    with open(O['slices_csv'], 'w') as fh:
        keys = ['slice_lo_s', 'slice_hi_s', 'model', 'gauge', 'distance_ft',
                'n_samples', 'rmse_psi', 'obs_max_psi', 'sim_max_psi',
                'amplitude_ratio']
        fh.write(','.join(keys) + '\n')
        for r in slice_rows:
            fh.write(','.join('' if r[k] is None else
                              (f"{r[k]:.6g}" if isinstance(r[k], float)
                               else str(r[k])) for k in keys) + '\n')

    with open(O['das_csv'], 'w') as fh:
        keys = list(das_rows[0].keys())
        fh.write(','.join(keys) + '\n')
        for r in das_rows:
            fh.write(','.join('' if r[k] is None else
                              (f"{r[k]:.6g}" if isinstance(r[k], float)
                               else str(r[k])) for k in keys) + '\n')

    npz = {'mesh_x': mesh.x, 'source_idx': np.array([source_idx]),
           'source_md_ft': np.array([src.md_ft]),
           'src_taxis_s': src.taxis_s, 'src_delta_psi': src.delta_psi,
           'frac_hits_ft': fhits,
           'das_distance_ft': ch_dist, 'das_md_ft': das_md,
           'das_rms_obs_dc': rms_obs_dc, 'das_rms_model_dc': rms_mod_dc,
           'das_norm_obs': n_obs, 'das_norm_model': n_mod, 'das_pearson_r': corr,
           'das_keep_mask': keep}
    for name in solves:
        npz[f'taxis_{name}'] = solves[name][0]
        npz[f'rec_{name}'] = solves[name][1].astype(np.float32)
    for t in targets:
        g = int(t['gauge'])
        npz[f'obs_taxis_g{g}'] = t['taxis']
        npz[f'obs_delta_g{g}'] = t['data']
    npz['profile_' + primary] = profiles[primary]
    npz['profile_primary_with_barrier'] = prof_b
    np.savez_compressed(O['arrays_npz'], **npz)

    results = {
        'preregistration': {
            'path': os.path.relpath(prereg_path, REPO),
            'written_utc': prereg['written_utc'],
            'config_sha256_prereg': prereg['config_sha256'],
            'config_sha256_at_run': cfg_sha,
            'match': bool(prereg_ok),
            'statement': 'the frozen parameter file was written and hashed '
                         'before this script existed; the run refuses to '
                         'proceed unless the hash still matches'},
        'setup': {
            'stage': stage, 'source_gauge': int(src_gauge),
            'source_md_ft': float(src.md_ft),
            'frac_hit_centroid_ft': float(centroid),
            'source_to_centroid_ft': float(src.md_ft - centroid),
            'frac_hits_ft': [float(x) for x in fhits],
            'n_targets': len(targets),
            'window_utc': [iso(win.t_start), iso(win.t_end)],
            'window_duration_s': float(win.duration_s),
            'source_series_t_total_s': t_total,
            'mesh_nx': int(mesh.nx),
            'mesh_md_ft': [float(mesh.x[0]), float(mesh.x[-1])],
            'source_node_snap_ft': float(mesh.snap_error_ft(src.md_ft)),
            'gauge_first_sample_spread_s': float(t0_spread),
            'calibration_window_duration_s': 1260.0,
            'extrapolation_in_time_factor': float(win.duration_s / 1260.0)},
        'models': {name: {'family': models[name]['family'],
                          'params': models[name]['params'],
                          'D_at_source_ft2_s': float(profiles[name][source_idx]),
                          'D_at_domain_end_ft2_s': float(profiles[name][0])}
                   for name in arm_names},
        'primary_model': primary,
        'per_arm': per_arm,
        'barrier_report': brep,
        'arrival_utc_crosscheck': utc_rows,
        'reporting_slices': slice_rows,
        'numerics_checks': checks,
        'das': {
            'unit_policy': cfg['das']['units'],
            'comparison': 'SHAPE ONLY. No psi<->strain coefficient is fitted. '
                          'Every statistic is invariant to a constant scale on '
                          'either series.',
            'gamma_psi^-1': gamma,
            'n_channels': int(ch_dist.size),
            'md_range_ft': [float(das_md[0]), float(das_md[-1])],
            'n_time_samples': int(keep.size),
            'n_samples_dropped_artifact': n_drop,
            'reference_band_ft': [ref_d, ref_h],
            'reference_n_channels': int(ref_m.sum()),
            'reference_rms_obs_counts': ref_obs,
            'reference_rms_model_strainrate': ref_mod,
            'at_gauges': das_at_gauge,
            'time_alignment': 'absolute UTC; DAS starts '
                              f'{off:.3f} s after the source gauge first sample'},
    }

    # ---- figures ----------------------------------------------------------
    fig_overlay(O['fig_overlay'], targets, solves, arm_names, primary, src,
                int(cfg['outputs']['figure_dpi']))
    fig_distance(O['fig_distance'], per_arm, arm_names, primary, acc,
                 int(cfg['outputs']['figure_dpi']))
    fig_das(O['fig_das'], ch_dist, n_obs, n_mod, corr, das_at_gauge, ref_d,
            src.md_ft, das_md, int(cfg['outputs']['figure_dpi']))

    with open(O['summary_json'], 'w') as fh:
        json.dump(results, fh, indent=1, default=str)
    with open(O['run_log'], 'w') as fh:
        fh.write('\n'.join(_LOG_LINES) + '\n')

    # ---- manifest ---------------------------------------------------------
    drv = rman.driver_record(
        kind='gauge_series', baseline_removal=cfg['source']['baseline_removal'],
        value_units='delta_psi',
        series_path=os.path.join(REPO, cfg['data']['gauge_series_template']
                                 .format(n=src_gauge)),
        gauge_number=int(src_gauge), gauge_md_ft=float(src.md_ft),
        taxis=src.taxis_s, values=src.delta_psi,
        time_start=iso(win.t_start), time_end=iso(win.t_end))
    srec = rman.source_record(mesh.x, md_requested_ft=float(src.md_ft),
                              mesh_idx=int(source_idx), driver=drv,
                              label=f'g{src_gauge}', index_in_source_list=0)
    sp = rman.source_protocol(
        application=cfg['source']['application'],
        solver_class=cfg['solver']['class'],
        placement_rule=cfg['source']['selection_rule'],
        sources=[srec],
        targets=[{'gauge': int(t['gauge']), 'md_ft': float(t['md_ft']),
                  'distance_ft': float(t['distance_ft']),
                  'mesh_idx': int(t['idx'])} for t in targets],
        time_level=cfg['solver']['source_time_level'],
        phase_chaining=rman.NONE_DECLARED,
        boundary_conditions={'lbc': cfg['solver']['lbc'],
                             'rbc': cfg['solver']['rbc'],
                             'source_node': 'Dirichlet'})
    brecs = rman.NONE_DECLARED
    if brep is not None:
        brecs = []
        for bi, b in enumerate(brep['barriers']):
            msk = np.zeros(mesh.nx, dtype=bool)
            msk[b['i0']:b['i1'] + 1] = True
            brecs.append(rman.barrier_record(
                mesh.x, msk, label=f"stage{stage}_frachit_{bi}",
                centre_md_ft=float(b['md_ft']),
                w_requested_ft=float(bsec['w_half_ft']),
                ratio=float(bsec['ratio']),
                d_baseline=float(profiles[primary][b['i0']]),
                report=b))
    num = rman.numerics(
        time=[rman.time_record(solves[primary][0], mode='fixed',
                               theta=float(cfg['solver']['theta']),
                               t_total_requested_s=t_total,
                               dt_requested_s=dt_s,
                               source_time_level=cfg['solver']['source_time_level'],
                               label='primary')],
        mesh=rman.mesh_record(mesh.x, dx_requested_ft=float(cfg['mesh']['dx_ft']),
                              window_md_ft=(win.md_min_ft, win.md_max_ft),
                              pad_low_ft=float(cfg['mesh']['domain_pad_low_md_ft']),
                              pad_high_ft=float(cfg['mesh']['domain_pad_high_md_ft'])),
        interface_avg=cfg['solver']['interface_avg'],
        boundary={'lbc': cfg['solver']['lbc'], 'rbc': cfg['solver']['rbc'],
                  'pml_thickness': 0.0, 'sigma_max': 0.0},
        diffusivity={'profile_family': 'multiple (four frozen arms)',
                     'primary_family': models[primary]['family'],
                     'param_names': models[primary].get('param_names', []),
                     'params': models[primary]['params'],
                     'baseline_D_ft2_s': float(profiles[primary][0]),
                     'profile_anchor': 'distance from the source node',
                     'arms': {n: models[n]['params'] for n in arm_names},
                     'note': 'every value copied from a prior manifest; nothing '
                             'fitted on stage-10 data'},
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
                  'note': 'the eight solves are cheap and run sequentially'})

    inputs = [(os.path.join(REPO, cfg['data']['gauge_md_npz']), 'geometry',
               'gauge_md'),
              (os.path.join(REPO, cfg['data']['frac_hit_npz']), 'geometry',
               f'frac_hit_stage{stage}'),
              (os.path.join(REPO, cfg['data']['das_npz']), 'das',
               f'lfdas_stage{stage}'),
              (os.path.join(REPO, cfg['data']['well_geometry_npz']), 'geometry',
               'swell_geometry'),
              ('output/r1_baseline_calibration/r1_run_manifest.json',
               'prior_run_output', 'r1_manifest'),
              ('output/r2_diffusivity_profile/r2_manifest.json',
               'prior_run_output', 'r2_manifest')]
    for n in range(1, 16):
        inputs.append((os.path.join(REPO, cfg['data']['gauge_series_template']
                                    .format(n=n)), 'gauge_series', f'gauge{n}'))
    for k, v in rdata.PUMPING_CURVE_FILES.items():
        inputs.append((os.path.join(REPO, cfg['data']['pumping_dir'], v),
                       'pumping', f'stage{stage}_{k}'))

    dpi = int(cfg['outputs']['figure_dpi'])
    out_decls = [
        rman.output_decl(prereg_path, role='json',
                         note='pre-registration written before the run'),
        rman.output_decl(prereg_path + '.sha256', role='other',
                         note='sha256 sidecar of the pre-registration'),
        rman.output_decl(O['summary_csv'], role='csv',
                         note='per-gauge blind metrics, every model arm'),
        rman.output_decl(O['slices_csv'], role='csv',
                         note='metrics on the three pre-registered 1260 s slices'),
        rman.output_decl(O['das_csv'], role='csv',
                         note='per-DAS-channel shape statistics'),
        rman.output_decl(O['arrays_npz'], role='arrays_npz',
                         note='simulated and observed series, DAS statistics'),
        rman.output_decl(O['summary_json'], role='json', note='full results'),
        rman.output_decl(O['run_log'], role='log', note='run log'),
        rman.output_decl(O['fig_overlay'], role='figure_png', dpi=dpi),
        rman.output_decl(O['fig_distance'], role='figure_png', dpi=dpi),
        rman.output_decl(O['fig_das'], role='figure_png', dpi=dpi)]

    doc = rman.write_manifest(
        O['manifest'], study_id=cfg['study_id'], task_id=TASK_ID, config=cfg,
        config_path=cfg_path, inputs=inputs, source=sp, numerics=num,
        outputs=out_decls, results=results, started_utc=started,
        run_label=f'D3 blind stage-{stage} forward {VERSION}',
        require_modules=('rev2_core', 'rev2_manifest', 'rev2_data',
                         'r1_calibration_core'),
        notes=[
            'D3 is a PRE-REGISTERED blind test. configs/rev2/d3_blind.json was '
            'written and hashed at ' + prereg['written_utc'] + ', recorded in '
            'PREREGISTERED.json, and only then was this script written and run.',
            'No parameter is fitted on stage-10 data; there is no optimiser in '
            'd3_blind.py.',
            'LF-DAS is kept in native counts. The comparison is SHAPE ONLY and '
            'no psi<->strain coefficient is fitted.',
        ])
    log(f"manifest written: {O['manifest']}")
    rep = rman.verify(O['manifest'], repo_root=REPO)
    log(f"manifest verify: status={rep['status']}")
    if rep['status'] != 'clean':
        log('  VERIFY DETAIL: ' + json.dumps(rep)[:2500])
    with open(O['run_log'], 'w') as fh:
        fh.write('\n'.join(_LOG_LINES) + '\n')
    log(f"total wall {time.time() - t_wall:.1f} s")
    return doc


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

COLS = {'two_zone_r2': '#4a3aa7', 'uniform_absolute': '#d95f02',
        'uniform_normalised': '#1b7837', 'uniform_manuscript': '#7b3294',
        'primary_with_barrier': '#999999'}
LBL = {'two_zone_r2': 'two-zone D(x) (R2 winner) — PRIMARY',
       'uniform_absolute': 'uniform D = 1150 ft²/s (R1, absolute norm)',
       'uniform_normalised': 'uniform D = 550 ft²/s (R1, normalised norm)',
       'uniform_manuscript': 'uniform D = 480 ft²/s (manuscript)',
       'primary_with_barrier': 'primary + barrier (ratio 1e-5, w = 2 ft)'}


def fig_overlay(path, targets, solves, arm_names, primary, src, dpi):
    order = sorted(targets, key=lambda t: t['distance_ft'])
    n = len(order)
    ncol, nrow = 4, int(np.ceil((n + 1) / 4))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16.0, 3.1 * nrow),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    ax0 = axes[0]
    ax0.plot(src.taxis_s / 60.0, src.delta_psi, color='k', lw=1.2)
    ax0.set_title(f"g{src.gauge} MD {src.md_ft:.0f} ft — SOURCE (Dirichlet input, "
                  f"not scored)", fontsize=8.5)
    ax0.set_ylabel('ΔP (psi)')
    ax0.grid(alpha=0.25)
    idx_of = {int(t['gauge']): k for k, t in enumerate(targets)}
    for a, t in zip(axes[1:], order):
        g = int(t['gauge'])
        k = idx_of[g]
        a.plot(t['taxis'] / 60.0, t['data'], color='k', lw=1.4, label='measured')
        for name in list(arm_names) + ['primary_with_barrier']:
            tx, rc = solves[name]
            a.plot(tx / 60.0, rc[:, k], color=COLS[name], lw=1.0,
                   ls=('--' if name == 'primary_with_barrier' else '-'),
                   alpha=(0.9 if name == primary else 0.75))
        a.set_title(f"g{g}  MD {t['md_ft']:.0f} ft   d = {t['distance_ft']:.0f} ft "
                    f"({'above' if t['md_ft'] > src.md_ft else 'below'})",
                    fontsize=8.5)
        a.grid(alpha=0.25)
    for a in axes[n + 1:]:
        a.axis('off')
    for a in axes[max(0, len(axes) - ncol):]:
        a.set_xlabel('time since pumping start (min)')
    h = [plt.Line2D([], [], color='k', lw=1.4, label='measured')]
    h += [plt.Line2D([], [], color=COLS[m],
                     ls=('--' if m == 'primary_with_barrier' else '-'),
                     lw=1.2, label=LBL[m])
          for m in list(arm_names) + ['primary_with_barrier']]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.suptitle('D3 — BLIND prediction of stage 10. Every parameter frozen '
                 'before the run (PREREGISTERED.json); nothing fitted here.',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.055, 1, 0.97))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def fig_distance(path, per_arm, arm_names, primary, acc, dpi):
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.6))
    for name in list(arm_names) + ['primary_with_barrier']:
        rows = per_arm[name]['per_gauge']
        d = np.array([r['distance_ft'] for r in rows])
        ax[0].plot(d, [r['rmse_psi'] for r in rows], 'o-', ms=4,
                   color=COLS[name], lw=1.1, label=LBL[name],
                   ls=('--' if name == 'primary_with_barrier' else '-'))
        m = [r['amplitude_ratio'] is not None for r in rows]
        ax[1].plot(d[m], [r['amplitude_ratio'] for r in rows
                          if r['amplitude_ratio'] is not None], 'o-', ms=4,
                   color=COLS[name], lw=1.1,
                   ls=('--' if name == 'primary_with_barrier' else '-'))
        m2 = [r['arrival_err_rel_s'] is not None for r in rows]
        ax[2].plot(d[m2], [r['arrival_err_rel_s'] for r in rows
                           if r['arrival_err_rel_s'] is not None], 'o-', ms=4,
                   color=COLS[name], lw=1.1,
                   ls=('--' if name == 'primary_with_barrier' else '-'))
    ax[0].set_yscale('log')
    ax[0].set_ylabel('blind RMSE (psi)')
    ax[1].axhspan(acc['amplitude_ratio_band'][0], acc['amplitude_ratio_band'][1],
                  color='0.85', zorder=0)
    ax[1].axhline(1.0, color='k', lw=0.7)
    ax[1].set_yscale('log')
    ax[1].set_ylabel('amplitude ratio  sim / obs')
    ax[2].axhline(0.0, color='k', lw=0.7)
    ax[2].set_ylabel('arrival error  sim − obs (s), 10% threshold')
    for a in ax:
        a.set_xlabel('distance from source gauge 9 (ft)')
        a.grid(alpha=0.25)
    ax[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle('D3 blind stage-10 prediction versus distance. Grey band in the '
                 'middle panel is the pre-registered acceptance band.',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def fig_das(path, ch_dist, n_obs, n_mod, corr, at_gauge, ref_d, src_md, das_md,
            dpi):
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    up = das_md > src_md
    for m, lab, c in ((up, 'above source', '#d95f02'),
                      (~up, 'below source', '#1b7837')):
        o = np.argsort(ch_dist[m])
        ax[0].semilogy(ch_dist[m][o], n_obs[m][o], lw=0.9, color=c,
                       label=f'LF-DAS, {lab}')
        ax[0].semilogy(ch_dist[m][o], n_mod[m][o], lw=1.4, ls='--', color=c,
                       label=f'model Γ·∂P/∂t, {lab}')
    ax[0].axvline(ref_d, color='k', lw=0.7, ls=':')
    ax[0].text(ref_d, ax[0].get_ylim()[1], ' normalisation\n distance',
               fontsize=7, va='top')
    ax[0].set_ylabel('window RMS, normalised at the reference distance')
    ax[0].set_title('shape only — both curves normalised, so the comparison\n'
                    'is invariant to any counts↔strain-rate coefficient',
                    fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)

    r = n_mod / np.where(n_obs > 0, n_obs, np.nan)
    for m, lab, c in ((up, 'above source', '#d95f02'),
                      (~up, 'below source', '#1b7837')):
        o = np.argsort(ch_dist[m])
        ax[1].semilogy(ch_dist[m][o], r[m][o], lw=0.9, color=c, label=lab)
    ax[1].axhline(1.0, color='k', lw=0.8)
    ax[1].set_ylabel('model shape / DAS shape')
    ax[1].set_title('> 1: the model decays more slowly than the DAS\n'
                    '< 1: the model decays faster', fontsize=9)
    ax[1].legend(fontsize=7.5, frameon=False)

    ax[2].plot(ch_dist, corr, '.', ms=1.6, color='0.65',
               label='per DAS channel')
    d = [g['distance_ft'] for g in at_gauge]
    rr = [np.nan if g['pearson_r'] is None else g['pearson_r'] for g in at_gauge]
    ax[2].plot(d, rr, 'o', ms=6, color='#4a3aa7', label='channel nearest a gauge')
    ax[2].axhline(0.0, color='k', lw=0.7)
    ax[2].set_ylabel('Pearson r,  model Γ·∂P/∂t vs DAS counts')
    ax[2].set_title('temporal shape agreement (scale-free)', fontsize=9)
    ax[2].legend(fontsize=7.5, frameon=False)
    for a in ax:
        a.set_xlabel('distance from source gauge 9 (ft)')
        a.grid(alpha=0.25)
    fig.suptitle('D3 — frozen model versus LF-DAS, stage 10. LF-DAS in native '
                 'counts; no psi↔strain coefficient fitted.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


if __name__ == '__main__':
    main()
