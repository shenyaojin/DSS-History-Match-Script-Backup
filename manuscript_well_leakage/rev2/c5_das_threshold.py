"""C5 - the LF-DAS "front" is a detection threshold, not a physical boundary.

Run from the repository root:

    python scripts/manuscript_well_leakage/rev2/c5_das_threshold.py \
        --config configs/rev2/c5_das_threshold.json

What it does
------------
1. Verifies every claim made about `LFDASdata_stg1_swell.npz` in the task package
   (channel count, sample count, MD span, NaN, time alignment) and reports the
   measured values rather than the claimed ones.
2. Replaces the legacy "5% of the global peak" front criterion with a PER-CHANNEL
   noise criterion: each channel's own RMS over an explicitly defined pre-pumping
   (shut-in) window, with detection declared at N x that RMS.  N = 2/3/5/10.
3. Detects the all-channel acquisition stripe near t ~ 600 s quantitatively, using
   the surface downlead (MD < 0) as a reference band that cannot carry reservoir
   signal, excludes it from every statistic, and quantifies how far the front moves
   if it is NOT excluded.
4. Separately, forward-models the same window with the verified R1 tridiagonal
   kernel at the established optima, takes dP/dt, and compares the SHAPE of its
   far-field amplitude decay against the LF-DAS strain-rate amplitude decay, each
   normalised to its own value at a common near-field reference channel.

LF-DAS is kept in NATIVE STRAIN-RATE UNITS throughout.  No psi<->strain conversion
is applied, used or fitted anywhere in this script.  Every DAS number reported is a
ratio (to the same channel's noise, or to a reference channel), hence invariant to
any constant scale factor.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import TwoSlopeNorm          # noqa: E402
from scipy.ndimage import median_filter, binary_closing  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', '..'))
BASE = os.path.join(REPO, 'scripts', 'manuscript_well_leakage',
                    'baseline_calibration')
sys.path.insert(0, BASE)
import r1_calibration_core as core        # noqa: E402
import r1_run_calibration as runner       # noqa: E402


VERSION = 'v2'   # output-file version tag; v1 was a first pass (single-channel
                 # normalisation, no spatial median) and is superseded.


def log(msg):
    print(f"[c5] {msg}", flush=True)


def _rel(p):
    p = os.path.abspath(p)
    return os.path.relpath(p, REPO) if p.startswith(REPO) else p


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def boxcar_nanmean(X, L):
    """Centred boxcar mean along axis 1, NaN-aware.

    Output is NaN where fewer than half the taps are finite, so masked stripe
    samples cannot leak into the filtered series.
    """
    M = np.isfinite(X).astype(float)
    Z = np.where(np.isfinite(X), X, 0.0)
    k = np.ones(int(L))
    num = np.stack([np.convolve(r, k, 'same') for r in Z])
    den = np.stack([np.convolve(r, k, 'same') for r in M])
    return np.where(den >= 0.5 * L, num / np.maximum(den, 1e-12), np.nan)


def contiguous_runs(idx):
    runs = []
    if idx.size == 0:
        return runs
    s = prev = idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            runs.append((s, prev))
            s = i
        prev = i
    runs.append((s, prev))
    return runs


def front_shallowest(det, md):
    idx = np.where(det)[0]
    return float(md[idx].min()) if idx.size else None


def front_connected(det, md, anchor, close_gap=2):
    """Shallowest MD of the contiguous detected run containing `anchor`.

    Gaps of up to `close_gap` undetected channels are bridged (binary closing
    with a structuring element of length close_gap + 1), so a single noisy
    channel cannot truncate the front.
    """
    d = binary_closing(det, structure=np.ones(close_gap + 1, bool))
    d = d | det
    if not d[anchor]:
        return None
    i = anchor
    while i > 0 and d[i - 1]:
        i -= 1
    return float(md[i])


# ---------------------------------------------------------------------------
# 0. data loading + claim verification
# ---------------------------------------------------------------------------

def load_and_verify(cfg):
    das_path = cfg['data']['das_npz']
    z = np.load(das_path, allow_pickle=True)
    t = z['taxis'].astype(float)
    da = z['daxis'].astype(float)
    st = z['start_time'].item()
    full = z['data']

    w = cfg['window']
    t0 = datetime.datetime.fromisoformat(w['time_start'])
    t1 = datetime.datetime.fromisoformat(w['time_end'])
    s0 = (t0 - st).total_seconds()
    s1 = (t1 - st).total_seconds()

    tm = np.where((t >= s0) & (t <= s1))[0]
    dm = np.where((da >= w['md_min_ft']) & (da <= w['md_max_ft']))[0]

    dt_unique = np.unique(np.diff(t))
    gaps = np.where(np.diff(t) > 1.5)[0]

    D = full[dm[0]:dm[-1] + 1, tm[0]:tm[-1] + 1].astype(np.float64)
    ver = {
        'file': _rel(das_path),
        'claim_channels_525': int(dm.size),
        'claim_samples_1260': int(tm.size),
        'claim_md_span_15003_16747': [float(da[dm[0]]), float(da[dm[-1]])],
        'channel_spacing_ft': float(np.median(np.diff(da))),
        'claim_no_nan': {'n_nan': int(np.isnan(D).sum()),
                         'n_inf': int(np.isinf(D).sum()),
                         'stored_dtype': str(full.dtype),
                         'note': ('the array is stored as int32, so "no NaN" is '
                                  'true by construction and is not evidence of '
                                  'gap-free data; the sample-clock check below is')},
        'das_start_time_utc': st.isoformat(),
        'window_start_offset_s': float(t[tm[0]] - s0),
        'window_end_offset_s': float(t[tm[-1]] - s1),
        'claim_time_aligned_within_0p7s': float(abs(t[tm[0]] - s0)),
        'sample_interval_s_unique_full_file': [float(x) for x in dt_unique],
        'sample_interval_s_unique_in_window':
            [float(x) for x in np.unique(np.diff(t[tm]))],
        'data_gaps_full_file': [
            {'t_s': float(t[i]), 'gap_s': float(t[i + 1] - t[i]),
             'utc': (st + datetime.timedelta(seconds=float(t[i]))).isoformat()}
            for i in gaps],
        'full_array_shape': [int(full.shape[0]), int(full.shape[1])],
        'full_md_range_ft': [float(da[0]), float(da[-1])],
    }
    return z, t, da, st, full, tm, dm, D, ver


# ---------------------------------------------------------------------------
# 1. pre-pumping window from the pumping curve
# ---------------------------------------------------------------------------

def pre_pumping_window(cfg, st_das, t_das, gaps_utc):
    p = cfg['data']['pump_rate_npz']
    zp = np.load(p, allow_pickle=True)
    stp = zp['start_time'].item()
    stp = datetime.datetime(stp.year, stp.month, stp.day, stp.hour, stp.minute,
                            stp.second, stp.microsecond)
    tp = zp['taxis'].astype(float)
    v = zp['data'].astype(float)

    spec = cfg['pre_pumping_window']
    q = v <= spec['shut_in_rate_threshold_bpm']
    runs = [(a, b) for a, b in contiguous_runs(np.where(q)[0])
            if (tp[b] - tp[a]) >= spec['shut_in_min_duration_s']]
    w0 = (datetime.datetime.fromisoformat(cfg['window']['time_start'])
          - stp).total_seconds()
    before = [r for r in runs if tp[r[1]] < w0]
    a, b = before[-1]
    inj_start = stp + datetime.timedelta(seconds=float(tp[b + 1]))
    shut_lo = stp + datetime.timedelta(seconds=float(tp[a]))
    shut_hi = stp + datetime.timedelta(seconds=float(tp[b]))

    # trim: keep clear of any DAS data gap, and of the injection ramp
    lo = shut_lo
    for g in gaps_utc:
        g_end = g['utc_end']
        if shut_lo <= g_end <= shut_hi:
            lo = max(lo, g_end + datetime.timedelta(seconds=spec['gap_guard_s']))
    lo = lo.replace(microsecond=0) + datetime.timedelta(seconds=1)
    hi = inj_start - datetime.timedelta(seconds=spec['injection_guard_s'])

    # alternative: everything before the very first pumping of the day
    first_run = [(a2, b2) for a2, b2 in runs if tp[a2] == tp[0]]
    alt_hi = (stp + datetime.timedelta(seconds=float(tp[first_run[0][1]])))

    info = {
        'pump_curve': _rel(p),
        'pump_curve_start_utc': stp.isoformat(),
        'shut_in_runs_utc': [
            [(stp + datetime.timedelta(seconds=float(tp[x]))).isoformat(),
             (stp + datetime.timedelta(seconds=float(tp[y]))).isoformat(),
             float(tp[y] - tp[x])] for x, y in runs],
        'selected_shut_in_utc': [shut_lo.isoformat(), shut_hi.isoformat()],
        'stage1_main_injection_start_utc': inj_start.isoformat(),
        'pre_pumping_window_utc': [lo.isoformat(), hi.isoformat()],
        'alternative_pre_any_pumping_utc': [
            (stp + datetime.timedelta(seconds=float(tp[0]))).isoformat(),
            alt_hi.isoformat()],
        'slurry_rate_bpm_in_analysis_window': [
            float(v[(tp >= w0) & (tp <= w0 + 1260)].min()),
            float(v[(tp >= w0) & (tp <= w0 + 1260)].max())],
    }
    return lo, hi, alt_hi, info, (stp, tp, v)


# ---------------------------------------------------------------------------
# 2. acquisition-stripe detection
# ---------------------------------------------------------------------------

def detect_stripe(cfg, full, da, t, tm, pm, tw):
    spec = cfg['stripe_detection']
    lo, hi = spec['reference_band_md_ft']
    ref = np.where((da >= lo) & (da < hi))[0]
    A = full[ref[0]:ref[-1] + 1, tm[0]:tm[-1] + 1].astype(np.float64)
    P = full[ref[0]:ref[-1] + 1, pm[0]:pm[-1] + 1].astype(np.float64)
    base = P.mean(axis=1)
    sig = np.sqrt(np.mean((P - base[:, None]) ** 2, axis=1))
    k = spec['per_channel_exceedance_sigma']
    f_win = np.mean(np.abs(A - base[:, None]) / sig[:, None] > k, axis=0)
    f_pre = np.mean(np.abs(P - base[:, None]) / sig[:, None] > k, axis=0)

    thr = spec['flag_fraction_threshold']
    flag = np.where(f_win > thr)[0]
    runs = contiguous_runs(flag)
    # merge runs separated by <= 5 samples (the stripe is oscillatory)
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= 5:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(list(r) if False else (r[0], r[1]))
    pad = spec['pad_s']
    windows = [[float(tw[a] - pad), float(tw[b] + pad)] for a, b in merged]
    mask = np.zeros(tw.size, bool)
    for a, b in windows:
        mask |= (tw >= a) & (tw <= b)

    info = {
        'reference_band_md_ft': [float(da[ref[0]]), float(da[ref[-1]])],
        'reference_band_n_channels': int(ref.size),
        'exceedance_sigma': float(k),
        'flag_fraction_threshold': float(thr),
        'false_alarm_floor_pre_pumping_max_fraction': float(f_pre.max()),
        'peak_fraction_in_window': float(f_win.max()),
        'peak_time_s_in_window': float(tw[int(np.argmax(f_win))]),
        'excluded_windows_t_win_s': windows,
        'n_samples_excluded': int(mask.sum()),
        'fraction_of_window_excluded': float(mask.mean()),
    }
    return mask, f_win, f_pre, info


# ---------------------------------------------------------------------------
# 3. per-channel noise criterion and the front
# ---------------------------------------------------------------------------

def snr_profile(D, Npre, stripe_mask, L, med_ch):
    """SNR per channel: window RMS / pre-pumping RMS of the low-passed signal."""
    base = Npre.mean(axis=1)
    Y = D - base[:, None]
    Ypre = Npre - base[:, None]
    if stripe_mask is not None:
        Y = Y.copy()
        Y[:, stripe_mask] = np.nan
    h = int(L) // 2
    sl = slice(h, -h) if h > 0 else slice(None)
    Ys = boxcar_nanmean(Y, L)[:, sl]
    Ps = boxcar_nanmean(Ypre, L)[:, sl]
    S = np.sqrt(np.nanmean(Ys ** 2, axis=1))
    sig = np.sqrt(np.nanmean(Ps ** 2, axis=1))
    snr = S / sig
    return snr, median_filter(snr, size=med_ch, mode='nearest'), S, sig, Ys, Ps


def sweep_fronts(snr, snr_med, md, anchor, n_grid, edge_tol):
    rows = []
    for N in n_grid:
        for label, s in (('raw', snr), ('median9', snr_med)):
            f_sh = front_shallowest(s > N, md)
            f_cn = front_connected(s > N, md, anchor)
            rows.append({
                'N': float(N), 'snr_variant': label,
                'front_shallowest_md_ft': f_sh,
                'front_shallowest_censored': (f_sh is not None
                                              and f_sh - md[0] <= edge_tol),
                'front_connected_md_ft': f_cn,
                'front_connected_censored': (f_cn is not None
                                             and f_cn - md[0] <= edge_tol),
                'n_channels_detected': int((s > N).sum()),
                'fraction_detected': float((s > N).mean()),
            })
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    with open(args.config, 'rb') as fh:
        raw = fh.read()
    cfg = json.loads(raw.decode('utf-8'))
    cfg_sha = hashlib.sha256(raw).hexdigest()
    outdir = cfg['outputs']['dir']
    os.makedirs(outdir, exist_ok=True)
    log(f"config {args.config} sha256={cfg_sha[:16]}")

    # ---- 0. load + verify ------------------------------------------------
    z, t, da, st, full, tm, dm, D, ver = load_and_verify(cfg)
    md = da[dm]
    tw = t[tm] - t[tm][0]
    log(f"DAS window block {D.shape}; MD {md[0]:.4f}..{md[-1]:.4f}; "
        f"first sample {ver['window_start_offset_s']:.3f} s after "
        f"{cfg['window']['time_start']}")

    gaps_utc = [{'utc_start': datetime.datetime.fromisoformat(g['utc']),
                 'utc_end': datetime.datetime.fromisoformat(g['utc'])
                 + datetime.timedelta(seconds=g['gap_s'])}
                for g in ver['data_gaps_full_file']]

    # ---- 1. pre-pumping window ------------------------------------------
    p_lo, p_hi, alt_hi, pre_info, pump = pre_pumping_window(cfg, st, t, gaps_utc)
    pm = np.where((t >= (p_lo - st).total_seconds())
                  & (t <= (p_hi - st).total_seconds()))[0]
    Npre = full[dm[0]:dm[-1] + 1, pm[0]:pm[-1] + 1].astype(np.float64)
    pre_info['n_das_samples'] = int(pm.size)
    pre_info['das_sample_interval_unique_s'] = \
        [float(x) for x in np.unique(np.diff(t[pm]))]
    log(f"pre-pumping window {p_lo}..{p_hi}  ({pm.size} DAS samples); "
        f"main injection starts {pre_info['stage1_main_injection_start_utc']}")

    # alternative noise window (before any pumping at all) - sensitivity only
    am = np.where(t <= (alt_hi - st).total_seconds())[0]
    Nalt = full[dm[0]:dm[-1] + 1, am[0]:am[-1] + 1].astype(np.float64)
    pre_info['alternative_n_das_samples'] = int(am.size)

    # ---- 2. stripe -------------------------------------------------------
    stripe_mask, f_win, f_pre, stripe_info = detect_stripe(
        cfg, full, da, t, tm, pm, tw)
    log(f"stripe: excluded {stripe_info['excluded_windows_t_win_s']} "
        f"({stripe_info['n_samples_excluded']} of {tw.size} samples, "
        f"{100*stripe_info['fraction_of_window_excluded']:.2f}%); "
        f"peak reference-band exceedance fraction "
        f"{stripe_info['peak_fraction_in_window']:.3f} at "
        f"t={stripe_info['peak_time_s_in_window']:.0f} s vs pre-pumping floor "
        f"{stripe_info['false_alarm_floor_pre_pumping_max_fraction']:.3f}")

    nc = cfg['noise_criterion']
    L = nc['lowpass_boxcar_s']
    medch = nc['spatial_median_filter_channels']
    n_grid = nc['N_grid']
    edge_tol = float(np.median(np.diff(md)))
    anchor_md = float(np.mean(np.unique(
        np.load(cfg['data']['frac_hit_stage1_npz'], allow_pickle=True)['data']
        .astype(float))))
    anchor = int(np.argmin(np.abs(md - anchor_md)))
    log(f"source anchor: stage-1 frac-hit centroid MD {anchor_md:.2f} -> "
        f"channel {anchor} at MD {md[anchor]:.2f}")

    # ---- 3. SNR + fronts -------------------------------------------------
    snr, snr_m, S_win, sig, Ys, Ps = snr_profile(D, Npre, stripe_mask, L, medch)
    rows_excl = sweep_fronts(snr, snr_m, md, anchor, n_grid, edge_tol)
    for r in rows_excl:
        r['stripe'] = 'excluded'

    snr_i, snr_mi, _, _, _, _ = snr_profile(D, Npre, None, L, medch)
    rows_incl = sweep_fronts(snr_i, snr_mi, md, anchor, n_grid, edge_tol)
    for r in rows_incl:
        r['stripe'] = 'included'

    # unsmoothed (no low-pass) variant, for completeness
    snr_r, snr_rm, _, _, _, _ = snr_profile(D, Npre, stripe_mask, 1, medch)
    rows_nolp = sweep_fronts(snr_r, snr_rm, md, anchor, n_grid, edge_tol)
    for r in rows_nolp:
        r['stripe'] = 'excluded_no_lowpass'

    # alternative noise window
    snr_a, snr_am, _, sig_alt, _, _ = snr_profile(D, Nalt, stripe_mask, L, medch)
    rows_alt = sweep_fronts(snr_a, snr_am, md, anchor, n_grid, edge_tol)
    for r in rows_alt:
        r['stripe'] = 'excluded_alt_noise_window'

    # ---- legacy 5%-of-global-peak rule -----------------------------------
    lg = cfg['legacy_rule']
    m100 = (tw >= 100) & (tw <= 450)
    a_leg = D[:, m100].mean(axis=1)
    pk = float(np.nanmax(a_leg))
    det_leg = a_leg >= 0.05 * pk
    legacy = {
        'statistic': lg['statistic'],
        'global_peak_native_units': pk,
        'global_peak_at_md_ft': float(md[int(np.argmax(a_leg))]),
        'threshold_native_units': 0.05 * pk,
        'front_shallowest_md_ft': front_shallowest(det_leg, md),
        'front_connected_md_ft': front_connected(det_leg, md, anchor),
        'n_channels_detected': int(det_leg.sum()),
        'published_value_md_ft': lg['published_value_md_ft'],
    }
    _f = legacy['front_shallowest_md_ft']
    legacy['reproduces_published'] = bool(
        _f is not None and abs(_f - lg['published_value_md_ft']) < 1.0)
    log(f"legacy rule reproduces MD {legacy['front_shallowest_md_ft']:.1f} "
        f"(published {lg['published_value_md_ft']:.0f}); global peak "
        f"{pk:.4g} native units at MD {legacy['global_peak_at_md_ft']:.1f}")

    # what fraction of the peak is the far-field signal?
    ref_far = int(np.argmin(np.abs(md - 15075.0)))
    legacy['far_field_g7_statistic_over_peak'] = float(a_leg[ref_far] / pk)
    legacy['far_field_g6_statistic_over_peak'] = float(
        a_leg[int(np.argmin(np.abs(md - 15344.0)))] / pk)

    # ---- 4. front as a function of time ----------------------------------
    tr = nc['time_resolved']
    binlen, stride = int(tr['bin_len_s']), int(tr['stride_s'])
    base = Npre.mean(axis=1)
    Yfull = D - base[:, None]
    Yfull[:, stripe_mask] = np.nan
    h = L // 2
    Yl = boxcar_nanmean(Yfull, L)
    centres = np.arange(binlen // 2, tw.size - binlen // 2, stride)
    fronts_t = {f"{N:g}": [] for N in n_grid}
    fronts_t_sh = {f"{N:g}": [] for N in n_grid}
    tcent = []
    snr_t = np.full((md.size, centres.size), np.nan)
    for j, c in enumerate(centres):
        sl = slice(c - binlen // 2, c + binlen // 2 + 1)
        s_t = np.sqrt(np.nanmean(Yl[:, sl] ** 2, axis=1)) / sig
        s_t = median_filter(np.nan_to_num(s_t, nan=0.0), size=medch, mode='nearest')
        snr_t[:, j] = s_t
        tcent.append(float(tw[c]))
        for N in n_grid:
            fronts_t[f"{N:g}"].append(front_connected(s_t > N, md, anchor))
            fronts_t_sh[f"{N:g}"].append(front_shallowest(s_t > N, md))
    tcent = np.asarray(tcent)

    # null spread of the same binned statistic over the quiet block
    Ypl = boxcar_nanmean(Npre - base[:, None], L)
    nullv = []
    for c in range(binlen // 2, Ypl.shape[1] - binlen // 2, stride):
        sl = slice(c - binlen // 2, c + binlen // 2 + 1)
        nullv.append(np.sqrt(np.nanmean(Ypl[:, sl] ** 2, axis=1)) / sig)
    nullv = np.concatenate(nullv)
    null_stats = {'median': float(np.nanmedian(nullv)),
                  'p95': float(np.nanpercentile(nullv, 95)),
                  'p99': float(np.nanpercentile(nullv, 99)),
                  'max': float(np.nanmax(nullv)),
                  'frac_above_2': float(np.nanmean(nullv > 2.0)),
                  'frac_above_3': float(np.nanmean(nullv > 3.0)),
                  'n': int(nullv.size)}
    log(f"null (quiet-block) binned SNR: median {null_stats['median']:.2f} "
        f"p99 {null_stats['p99']:.2f} max {null_stats['max']:.2f}; "
        f"P(SNR>2)={null_stats['frac_above_2']:.4f}")

    # ---- 5. forward model -------------------------------------------------
    fm = cfg['forward_model']
    fcfg = {'window': cfg['window'], 'data': cfg['data'],
            'source': {'selection_rule': fm['source']['selection_rule'],
                       'baseline_removal': fm['source']['baseline_removal']}}
    series, gnums, gmds, frac_hits, _, _ = runner.load_window_data(fcfg)
    src_gauge, fh_centroid = runner.pick_source_gauge(fcfg, series, frac_hits)
    src = series[src_gauge]
    pad_lo = fm['mesh']['domain_pad_low_md_ft']
    pad_hi = fm['mesh']['domain_pad_high_md_ft']
    dx = fm['mesh']['dx_ft']
    mesh = np.arange(cfg['window']['md_min_ft'] - pad_lo,
                     cfg['window']['md_max_ft'] + pad_hi + dx / 2.0, dx)
    source_idx = int(np.argmin(np.abs(mesh - src['md_ft'])))
    dt_s = fm['solver']['dt_s']
    t_total = float(src['taxis'][-1])
    rec_idx = np.array([int(np.argmin(np.abs(mesh - m))) for m in md])
    g_idx = np.array([int(np.argmin(np.abs(mesh - series[n]['md_ft'])))
                      for n in sorted(series)])
    log(f"forward mesh MD [{mesh[0]:.0f},{mesh[-1]:.0f}] nx={mesh.size}, "
        f"source gauge {src_gauge} at MD {src['md_ft']:.0f} "
        f"(idx {source_idx}), dt={dt_s} s, t_total={t_total:.0f} s")

    sims = {}
    for name, spec in fm['profiles'].items():
        fam = core.PROFILE_FAMILIES[spec['family']]
        prof = fam['fn'](mesh, source_idx, np.asarray(spec['params'], float))
        ts, rec = core.solve_forward(mesh, prof, dt_s, t_total,
                                     src['taxis'], src['delta_psi'],
                                     source_idx, record_idx=rec_idx)
        _, recg = core.solve_forward(mesh, prof, dt_s, t_total,
                                     src['taxis'], src['delta_psi'],
                                     source_idx, record_idx=g_idx)
        # pooled RMSE against the six target gauges - reproduces the R1 number
        sq, n, mse_each = 0.0, 0, []
        for k, gn in enumerate(sorted(series)):
            if gn == src_gauge:
                continue
            obs = series[gn]
            simv = np.interp(obs['taxis'], ts, recg[:, k])
            r = simv - obs['delta_psi']
            sq += float(np.sum(r ** 2))
            n += r.size
            mse_each.append(float(np.mean(r ** 2)))
        rmse = float(np.sqrt(sq / n))
        rmse_gaugemean = float(np.sqrt(np.mean(mse_each)))
        # dP/dt at the DAS channel depths, on the DAS sample grid
        dpdt = np.gradient(rec, ts, axis=0)
        dp_on_das = np.stack([np.interp(tw, ts, dpdt[:, k])
                              for k in range(rec.shape[1])])
        dp_on_das[:, stripe_mask] = np.nan       # identical masking
        dps = boxcar_nanmean(dp_on_das, L)[:, h:-h]
        amp = np.sqrt(np.nanmean(dps ** 2, axis=1))
        sims[name] = {'profile': spec, 'rmse_psi': rmse,
                      'rmse_gaugemean_psi': rmse_gaugemean, 'amp_dpdt': amp,
                      'n_steps': int(ts.size - 1), 'ts': ts, 'rec': rec}
        log(f"  {name}: pooled RMSE vs 6 target gauges = {rmse:.3f} psi "
            f"(gauge-mean form {rmse_gaugemean:.3f} psi)")

    # ---- instrument-response diagnostic ----------------------------------
    # The acquisition artifact is instrumental by construction, so its
    # per-channel amplitude maps the depth-dependent channel response.
    stripe_amp = np.sqrt(np.mean(D[:, stripe_mask] ** 2, axis=1))
    mch0 = int(fm['comparison']['spatial_median_channels'])
    sig_s = median_filter(sig, size=mch0, mode='nearest')
    art_s = median_filter(stripe_amp, size=mch0, mode='nearest')
    shal = (md >= 15003.0) & (md <= 15150.0)
    deep = (md >= 16400.0) & (md <= 16700.0)
    resp = {
        'shallow_band_md_ft': [15003.0, 15150.0],
        'deep_band_md_ft': [16400.0, 16700.0],
        'noise_floor_deep_over_shallow':
            float(np.median(sig_s[deep]) / np.median(sig_s[shal])),
        'artifact_amplitude_deep_over_shallow':
            float(np.median(art_s[deep]) / np.median(art_s[shal])),
        'log10_correlation_noise_vs_artifact':
            float(np.corrcoef(np.log10(art_s), np.log10(sig_s))[0, 1]),
        'artifact_per_channel_amplitude_decades':
            float(np.log10(stripe_amp.max() / stripe_amp.min())),
        'artifact_amplitude_over_noise_floor_median':
            float(np.median(stripe_amp / sig)),
        'interpretation': (
            'The artifact is instrumental (it is present on fibre above the '
            'wellhead) but it is NOT a usable calibration of the channel '
            'response: it exceeds the quiescent noise floor by ~3 orders of '
            'magnitude and its own per-channel amplitude spans ~3 decades, i.e. '
            'it is a saturating, non-linear event. It is reported only to show '
            'that the depth dependence of the channel response is not pinned '
            'down by these data. The two defensible normalisations of the DAS '
            'amplitude are therefore (i) raw, assuming a depth-independent '
            'response, and (ii) divided by each channel\'s own quiescent noise '
            'floor; results are bracketed between them.'),
    }
    log(f"instrument response: noise floor deep/shallow "
        f"{resp['noise_floor_deep_over_shallow']:.2f}x, artifact amplitude "
        f"deep/shallow {resp['artifact_amplitude_deep_over_shallow']:.2f}x, "
        f"log-log corr {resp['log10_correlation_noise_vs_artifact']:.3f}")

    # far-field decay, SHAPE only
    cmp = fm['comparison']
    lo_md, hi_md = cmp['valid_md_range_ft']
    valid = (md >= lo_md) & (md <= hi_md)
    ref_i = int(np.argmin(np.abs(md - cmp['reference_md_ft'])))
    rb = (md >= cmp['reference_band_md_ft'][0]) & (md <= cmp['reference_band_md_ft'][1])
    dist = src['md_ft'] - md
    mch = int(cmp['spatial_median_channels'])

    # Incoherent-noise subtraction: the window RMS of a far channel contains its
    # own noise floor in quadrature.  At SNR ~ 3 this inflates the amplitude by
    # ~5%, which would flatter the far field, so it is removed.
    S_sig = np.sqrt(np.maximum(S_win ** 2 - sig ** 2, 0.0))

    das_s = median_filter(S_sig, size=mch, mode='nearest')
    das_ratio = das_s / np.median(das_s[rb])
    das_snr_s = median_filter(S_sig / sig, size=mch, mode='nearest')
    das_snr_ratio = das_snr_s / np.median(das_snr_s[rb])
    das_art_s = median_filter(S_sig / stripe_amp, size=mch, mode='nearest')
    das_art_ratio = das_art_s / np.median(das_art_s[rb])
    for name, s in sims.items():
        s['amp_s'] = median_filter(s['amp_dpdt'], size=mch, mode='nearest')
        s['ratio'] = s['amp_s'] / np.median(s['amp_s'][rb])

    norms = {'raw': das_ratio, 'per_noise': das_snr_ratio,
             'per_artifact': das_art_ratio}
    decay = {
        'reference_band_md_ft': cmp['reference_band_md_ft'],
        'reference_band_distance_ft': [float(dist[rb].min()), float(dist[rb].max())],
        'spatial_median_channels': mch,
        'reference_channel_md_ft': float(md[ref_i]),
        'das_reference_value_native_units': float(np.median(das_s[rb])),
        'das_single_channel_reference_native_units': float(S_win[ref_i]),
        'noise_subtraction': ('DAS amplitude is sqrt(window RMS^2 - noise RMS^2) '
                              'per channel'),
        'normalisation_variants': {
            'raw': 'amplitude as recorded; assumes a depth-independent channel response',
            'per_noise': "amplitude divided by that channel's own quiescent noise "
                         'floor; assumes the noise floor tracks the response',
            'per_artifact': 'amplitude divided by that channel\'s amplitude during '
                            'the instrumental artifact. DIAGNOSTIC ONLY, NOT a '
                            'candidate normalisation: the artifact is a '
                            'saturating event ~1000x the noise floor whose own '
                            'per-channel amplitude spans ~3 decades. It is '
                            'reported to show how far the answer can move if the '
                            'channel response is depth dependent.'},
        'defensible_bracket': 'raw and per_noise; per_artifact is excluded',
        'instrument_response': resp,
    }
    for name, s in sims.items():
        decay[f'model_{name}_reference_value_psi_per_s'] = \
            float(np.median(s['amp_s'][rb]))

    # band summary: median of model/DAS over distance bands
    bands = []
    for blo, bhi in cmp['distance_bands_ft']:
        m_ = valid & (dist >= blo) & (dist <= bhi)
        row = {'band_lo_ft': blo, 'band_hi_ft': bhi,
               'n_channels': int(m_.sum()),
               'md_hi_ft': float(md[m_].max()), 'md_lo_ft': float(md[m_].min())}
        for nn, arr in norms.items():
            row[f'das_ratio_{nn}'] = float(np.median(arr[m_]))
        for name, s in sims.items():
            row[f'model_{name}_ratio_median'] = float(np.median(s['ratio'][m_]))
            for nn, arr in norms.items():
                row[f'model_over_das_{name}_{nn}'] = float(
                    np.median(s['ratio'][m_] / arr[m_]))
            # per_artifact is a diagnostic only (saturating event), so the
            # defensible bracket is raw vs per_noise.
            span = [row[f'model_over_das_{name}_raw'],
                    row[f'model_over_das_{name}_per_noise']]
            row[f'model_over_das_{name}_span'] = f"{min(span):.2f}-{max(span):.2f}"
            row[f'model_over_das_{name}_sign_consistent'] = bool(
                all(x > 1.0 for x in span) or all(x < 1.0 for x in span))
        bands.append(row)
        log("  band %4.0f-%4.0f ft (MD %.0f-%.0f): " % (blo, bhi, row['md_lo_ft'],
                                                        row['md_hi_ft'])
            + '  '.join("%s model/DAS raw x%.2f per_noise x%.2f per_artifact x%.2f"
                        % (n, row[f'model_over_das_{n}_raw'],
                           row[f'model_over_das_{n}_per_noise'],
                           row[f'model_over_das_{n}_per_artifact'])
                        for n in sims))

    # summary at the gauge depths
    gauge_rows = []
    for gn in sorted(series):
        m_ = series[gn]['md_ft']
        if not (lo_md <= m_ <= hi_md):
            continue
        k = int(np.argmin(np.abs(md - m_)))
        row = {'gauge': gn, 'md_ft': m_, 'distance_ft': float(dist[k]),
               'das_ratio': float(das_ratio[k]),
               'das_response_normalised_ratio': float(das_snr_ratio[k])}
        for name, s in sims.items():
            row[f'model_{name}_ratio'] = float(s['ratio'][k])
            row[f'over_prediction_factor_{name}'] = float(
                s['ratio'][k] / das_ratio[k])
            row[f'over_prediction_respnorm_{name}'] = float(
                s['ratio'][k] / das_snr_ratio[k])
        gauge_rows.append(row)
        log("  g%d d=%6.0f ft  DAS %.4f  " % (row['gauge'], row['distance_ft'],
                                              row['das_ratio'])
            + '  '.join(f"{n}={row['model_'+n+'_ratio']:.4f}"
                        f"(x{row['over_prediction_factor_'+n]:.2f})"
                        for n in sims))

    # ---- outputs ---------------------------------------------------------
    out_paths = []

    def w_csv(name, header, rows):
        p = os.path.join(outdir, name)
        with open(p, 'w') as fh:
            fh.write(','.join(header) + '\n')
            for r in rows:
                fh.write(','.join('' if r.get(k) is None else
                                  (f"{r[k]:.6g}" if isinstance(r[k], float)
                                   else str(r[k])) for k in header) + '\n')
        out_paths.append(p)
        return p

    all_rows = rows_excl + rows_incl + rows_nolp + rows_alt
    w_csv(f'c5_front_vs_threshold_{VERSION}.csv',
          ['stripe', 'snr_variant', 'N', 'front_shallowest_md_ft',
           'front_shallowest_censored', 'front_connected_md_ft',
           'front_connected_censored', 'n_channels_detected',
           'fraction_detected'], all_rows)

    w_csv(f'c5_channel_snr_{VERSION}.csv',
          ['md_ft', 'distance_from_source_ft', 'sigma_pre_native_units',
           'window_rms_native_units', 'snr', 'snr_median9',
           'snr_stripe_included', 'snr_no_lowpass', 'legacy_statistic',
           'legacy_over_global_peak'],
          [{'md_ft': float(md[i]), 'distance_from_source_ft': float(dist[i]),
            'sigma_pre_native_units': float(sig[i]),
            'window_rms_native_units': float(S_win[i]),
            'snr': float(snr[i]), 'snr_median9': float(snr_m[i]),
            'snr_stripe_included': float(snr_i[i]),
            'snr_no_lowpass': float(snr_r[i]),
            'legacy_statistic': float(a_leg[i]),
            'legacy_over_global_peak': float(a_leg[i] / pk)}
           for i in range(md.size)])

    rows_t = []
    for j, tc in enumerate(tcent):
        r = {'t_win_s': float(tc)}
        for N in n_grid:
            r[f'front_connected_N{N:g}_md_ft'] = fronts_t[f"{N:g}"][j]
            r[f'front_shallowest_N{N:g}_md_ft'] = fronts_t_sh[f"{N:g}"][j]
        rows_t.append(r)
    w_csv(f'c5_front_vs_time_{VERSION}.csv',
          ['t_win_s'] + [f'front_connected_N{N:g}_md_ft' for N in n_grid]
          + [f'front_shallowest_N{N:g}_md_ft' for N in n_grid], rows_t)

    dec_rows = []
    for i in range(md.size):
        if not valid[i]:
            continue
        r = {'md_ft': float(md[i]), 'distance_from_source_ft': float(dist[i]),
             'das_amp_window_rms_native_units': float(S_win[i]),
             'das_amp_noise_subtracted_native_units': float(S_sig[i]),
             'das_amp_median15_native_units': float(das_s[i]),
             'das_ratio_raw': float(das_ratio[i]),
             'das_ratio_per_noise': float(das_snr_ratio[i]),
             'das_ratio_per_artifact': float(das_art_ratio[i]),
             'channel_noise_floor_native_units': float(sig[i]),
             'artifact_amplitude_native_units': float(stripe_amp[i])}
        for name, s in sims.items():
            r[f'model_{name}_dpdt_amp_psi_per_s'] = float(s['amp_dpdt'][i])
            r[f'model_{name}_ratio_to_ref'] = float(s['ratio'][i])
            for nn, arr in norms.items():
                r[f'model_over_das_{name}_{nn}'] = float(s['ratio'][i] / arr[i])
        dec_rows.append(r)
    hdr = ['md_ft', 'distance_from_source_ft', 'das_amp_window_rms_native_units',
           'das_amp_noise_subtracted_native_units',
           'das_amp_median15_native_units', 'das_ratio_raw',
           'das_ratio_per_noise', 'das_ratio_per_artifact',
           'channel_noise_floor_native_units', 'artifact_amplitude_native_units']
    for name in sims:
        hdr += [f'model_{name}_dpdt_amp_psi_per_s', f'model_{name}_ratio_to_ref']
        hdr += [f'model_over_das_{name}_{nn}' for nn in norms]
    w_csv(f'c5_farfield_decay_{VERSION}.csv', hdr, dec_rows)

    bhdr = ['band_lo_ft', 'band_hi_ft', 'md_lo_ft', 'md_hi_ft', 'n_channels']
    bhdr += [f'das_ratio_{nn}' for nn in norms]
    for name in sims:
        bhdr += [f'model_{name}_ratio_median']
        bhdr += [f'model_over_das_{name}_{nn}' for nn in norms]
        bhdr += [f'model_over_das_{name}_span',
                 f'model_over_das_{name}_sign_consistent']
    w_csv(f'c5_farfield_decay_bands_{VERSION}.csv', bhdr, bands)

    w_csv(f'c5_stripe_detection_{VERSION}.csv',
          ['t_win_s', 'reference_band_exceedance_fraction', 'excluded'],
          [{'t_win_s': float(tw[j]),
            'reference_band_exceedance_fraction': float(f_win[j]),
            'excluded': bool(stripe_mask[j])} for j in range(tw.size)])

    # ---- figures ---------------------------------------------------------
    src_dpdt = np.gradient(src['delta_psi'], src['taxis'])
    out_paths.append(fig_front(os.path.join(outdir, f'fig01_front_vs_threshold_{VERSION}.png'),
                               md, snr, snr_m, snr_i, n_grid, rows_excl, rows_incl,
                               legacy, tcent, fronts_t,
                               src['taxis'], src['delta_psi'], src_dpdt))
    out_paths.append(fig_stripe(os.path.join(outdir, f'fig02_artifact_detection_{VERSION}.png'),
                                D, tw, md, f_win, f_pre, stripe_mask, stripe_info,
                                rows_excl, rows_incl, n_grid))
    out_paths.append(fig_decay(os.path.join(outdir, f'fig03_farfield_decay_{VERSION}.png'),
                               md, dist, valid, norms,
                               S_sig / np.median(das_s[rb]), sims, cmp,
                               gauge_rows, sig_s, art_s, resp))

    # ---- manifest --------------------------------------------------------
    results = {
        'das_verification': ver,
        'pre_pumping_window': pre_info,
        'stripe_detection': stripe_info,
        'noise_floor': {
            'sigma_native_units': {
                'min': float(sig.min()), 'p10': float(np.percentile(sig, 10)),
                'median': float(np.median(sig)),
                'p90': float(np.percentile(sig, 90)), 'max': float(sig.max())},
            'n_channels_sigma_gt_5x_median':
                int((sig > 5 * np.median(sig)).sum()),
            'per_channel_dc_offset_over_fluctuation_median': float(np.median(
                np.abs(Npre.mean(axis=1))
                / np.sqrt(np.mean((Npre - Npre.mean(axis=1)[:, None]) ** 2, axis=1)))),
            'null_binned_snr': null_stats,
        },
        'legacy_rule': legacy,
        'front_vs_threshold': all_rows,
        'front_vs_time_summary': {
            f"N={N:g}": {
                'first_defined_t_s': next((float(tcent[j]) for j in
                                           range(len(tcent))
                                           if fronts_t[f"{N:g}"][j] is not None),
                                          None),
                'shallowest_md_ft': (min([x for x in fronts_t[f"{N:g}"]
                                          if x is not None], default=None)),
                'final_md_ft': fronts_t[f"{N:g}"][-1],
            } for N in n_grid},
        'stripe_sensitivity': {
            f"N={N:g}": {
                'front_connected_excluded': next(
                    r['front_connected_md_ft'] for r in rows_excl
                    if r['N'] == N and r['snr_variant'] == 'median9'),
                'front_connected_included': next(
                    r['front_connected_md_ft'] for r in rows_incl
                    if r['N'] == N and r['snr_variant'] == 'median9'),
            } for N in n_grid},
        'forward_model': {
            'source_gauge': int(src_gauge),
            'frac_hit_centroid_md_ft': float(fh_centroid),
            'profiles': {n: {'spec': s['profile'],
                             'pooled_rmse_psi_vs_6_target_gauges': s['rmse_psi'],
                             'gauge_mean_rmse_psi': s['rmse_gaugemean_psi']}
                         for n, s in sims.items()},
            'r1_cross_check': ('uniform_1150 pooled RMSE should reproduce the '
                               'R1 value 82.33 psi; two_zone_r2 gauge-mean RMSE '
                               'should reproduce the r2 value 11.872 psi '
                               '(r2_profile_inversion scores with the gauge-mean '
                               'form, R1 with the pooled form)'),
        },
        'farfield_decay': decay,
        'farfield_decay_bands': bands,
        'farfield_decay_at_gauge_depths': gauge_rows,
        'unit_policy': cfg['units'],
    }

    man = write_manifest(os.path.join(outdir, 'manifest.json'), cfg, cfg_sha,
                         args.config, mesh, source_idx, src, dt_s,
                         int(sims[list(sims)[0]]['n_steps']), results,
                         out_paths, cfg, series)
    log(f"manifest written: {os.path.join(outdir, 'manifest.json')}")
    return man


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_front(path, md, snr, snr_m, snr_i, n_grid, rows_excl, rows_incl,
              legacy, tcent, fronts_t, src_t, src_dp, src_dpdt):
    fig, ax = plt.subplots(3, 1, figsize=(10.0, 13.0))
    cols = {2.0: '#1b7837', 3.0: '#2166ac', 5.0: '#d95f02', 10.0: '#7b3294'}

    a = ax[0]
    a.semilogy(md, snr, color='0.75', lw=0.7, label='SNR, per channel')
    a.semilogy(md, snr_m, color='k', lw=1.4,
               label='SNR, 9-channel median (30 ft)')
    for N in n_grid:
        a.axhline(N, color=cols[N], ls='--', lw=1.0)
        a.text(md[-1], N, f' N={N:g}', color=cols[N], va='center', fontsize=8)
    for gn, gm, pp in [(6, 15344.0, 160), (7, 15075.0, 57)]:
        a.axvline(gm, color='crimson', ls=':', lw=1.2)
        a.text(gm, a.get_ylim()[0] * 1.15, f' g{gn}: {pp} psi', color='crimson',
               fontsize=8, rotation=90, va='bottom')
    a.axvline(legacy['front_shallowest_md_ft'], color='b', lw=1.6)
    a.text(legacy['front_shallowest_md_ft'], 0.5, ' legacy "front"\n MD %.0f'
           % legacy['front_shallowest_md_ft'], color='b', fontsize=8, va='bottom')
    a.set_xlabel('measured depth MD (ft)')
    a.set_ylabel('SNR = window RMS / pre-pumping RMS\n(per channel, native strain-rate units)')
    a.set_title('(a) Per-channel detection SNR across the array\n'
                'signal is above the channel noise floor everywhere; the '
                '"front" is wherever a horizontal line is drawn', fontsize=10)
    a.legend(fontsize=8, loc='upper left')
    a.grid(alpha=0.3, which='both')

    a = ax[1]
    xs = [r['N'] for r in rows_excl if r['snr_variant'] == 'median9']
    for key, lab, mk, c in [
            ('front_connected_md_ft', 'source-connected run (primary)', 'o', 'k'),
            ('front_shallowest_md_ft', 'shallowest detected channel', 's', '0.5')]:
        ys = [r[key] for r in rows_excl if r['snr_variant'] == 'median9']
        a.plot(xs, ys, mk + '-', color=c, label=lab)
    ys = [r['front_connected_md_ft'] for r in rows_incl
          if r['snr_variant'] == 'median9']
    a.plot(xs, ys, '^--', color='#d95f02',
           label='source-connected, stripe NOT excluded')
    a.axhline(legacy['front_shallowest_md_ft'], color='b', lw=1.5,
              label='legacy 5%%-of-global-peak rule (MD %.0f)'
                    % legacy['front_shallowest_md_ft'])
    a.axhline(md[0], color='r', ls=':', lw=1.5,
              label='shallow limit of DAS coverage (MD %.0f) - fronts here are CENSORED'
                    % md[0])
    a.axhline(15344, color='crimson', ls='-.', lw=1.0)
    a.text(10.2, 15344, ' g6: 160 psi', color='crimson', fontsize=8, va='center')
    a.axhline(15075, color='crimson', ls='-.', lw=1.0)
    a.text(10.2, 15075, ' g7: 57 psi', color='crimson', fontsize=8, va='center')
    a.set_xscale('log')
    a.set_xticks(n_grid)
    a.set_xticklabels([f'{N:g}' for N in n_grid])
    a.set_xlabel('threshold multiplier N  (detection at N x per-channel pre-pumping RMS)')
    a.set_ylabel('up-hole edge of detection, MD (ft)')
    a.set_ylim(14950, 16250)
    a.set_title('(b) Front position is a function of the threshold, not of the reservoir',
                fontsize=10)
    a.legend(fontsize=7.5, loc='upper left')
    a.grid(alpha=0.3)

    a = ax[2]
    for N in n_grid:
        y = [np.nan if v is None else v for v in fronts_t[f"{N:g}"]]
        a.plot(tcent, y, '-', color=cols[N], lw=1.5, label=f'N={N:g}')
    a.axhline(md[0], color='r', ls=':', lw=1.5)
    a.text(tcent[-1], md[0], ' DAS coverage limit', color='r', fontsize=8,
           ha='right', va='bottom')
    a.axhline(legacy['front_shallowest_md_ft'], color='b', lw=1.2)
    a.text(tcent[0], legacy['front_shallowest_md_ft'], ' legacy MD 15632',
           color='b', fontsize=8, va='bottom')
    a.invert_yaxis()
    a.set_xlabel('time since 2020-03-16 11:24:00 (s)')
    a.set_ylabel('front MD (ft)')
    a.set_title('(c) The front moves - and then retreats, because LF-DAS measures a RATE.\n'
                'Source-connected front vs time (121 s bins, stripe excluded), '
                'with the source-gauge dP/dt behind it.', fontsize=10)
    a.legend(fontsize=8, loc='lower left', title='detection threshold',
             title_fontsize=8)
    a.text(0.99, 0.03, 'a line that stops = no detected channel remains\n'
           'connected to the source at that time',
           transform=a.transAxes, fontsize=7.5, ha='right', va='bottom',
           bbox=dict(fc='w', ec='0.7', alpha=0.85))
    a.grid(alpha=0.3)
    b = a.twinx()
    b.plot(src_t, src_dpdt, color='0.65', lw=1.0, zorder=0)
    b.set_ylabel('source gauge dP/dt (psi/s)', color='0.45', fontsize=9)
    b.tick_params(axis='y', colors='0.45')
    b.set_zorder(0)
    a.set_zorder(1)
    a.patch.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def fig_stripe(path, D, tw, md, f_win, f_pre, mask, info, rows_excl, rows_incl,
               n_grid):
    fig = plt.figure(figsize=(11.0, 9.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.0, 1.0, 1.1], hspace=0.42,
                          wspace=0.26)

    a = fig.add_subplot(gs[0, :])
    cl = float(np.nanpercentile(np.abs(D), 99))
    a.imshow(D, aspect='auto', cmap='bwr', norm=TwoSlopeNorm(0, -cl, cl),
             extent=[tw[0], tw[-1], md[-1], md[0]], interpolation='nearest')
    for lo, hi in info['excluded_windows_t_win_s']:
        a.axvline(lo, color='k', lw=0.8, ls='--')
        a.axvline(hi, color='k', lw=0.8, ls='--')
    a.set_xlabel('time since 11:24:00 (s)')
    a.set_ylabel('MD (ft)')
    a.set_title('(a) LF-DAS waterfall, native strain-rate units (no conversion applied).\n'
                'The all-channel vertical stripe near t = %.0f s is the acquisition '
                'artifact; dashed lines are the excluded window.'
                % info['peak_time_s_in_window'], fontsize=10)

    a = fig.add_subplot(gs[1, :])
    a.plot(tw, f_win, color='k', lw=1.0,
           label='analysis window: fraction of surface-downlead channels\n'
                 'exceeding %.0f x their own pre-pumping RMS'
                 % info['exceedance_sigma'])
    a.axhline(info['flag_fraction_threshold'], color='#d95f02', ls='--',
              label='flag threshold %.2f' % info['flag_fraction_threshold'])
    a.axhline(info['false_alarm_floor_pre_pumping_max_fraction'], color='g',
              ls=':', label='max over the quiet pre-pumping block (%.3f)'
                            % info['false_alarm_floor_pre_pumping_max_fraction'])
    a.fill_between(tw, 0, 1, where=mask, color='0.8', zorder=0)
    a.set_xlabel('time since 11:24:00 (s)')
    a.set_ylabel('exceedance fraction')
    a.set_ylim(0, 1.05)
    a.set_title('(b) Artifact detection. The reference band is fibre ABOVE the '
                'wellhead (MD < 0): it cannot carry reservoir signal, so a '
                'simultaneous transient there is instrumental.', fontsize=9.5)
    a.legend(fontsize=7.5, loc='upper left')
    a.grid(alpha=0.3)

    a = fig.add_subplot(gs[2, 0])
    j = int(np.argmin(np.abs(tw - info['peak_time_s_in_window'])))
    a.semilogy(md, np.abs(D[:, j]), 'k-', lw=0.8, label='t = %.0f s (stripe)'
               % tw[j])
    q = int(np.argmin(np.abs(tw - 300.0)))
    a.semilogy(md, np.abs(D[:, q]), color='0.6', lw=0.8,
               label='t = %.0f s (typical)' % tw[q])
    a.set_xlabel('MD (ft)')
    a.set_ylabel('|strain rate| (native units)')
    a.set_title('(c) The stripe is present on every channel', fontsize=9.5)
    a.legend(fontsize=7.5)
    a.grid(alpha=0.3, which='both')

    a = fig.add_subplot(gs[2, 1])
    xs = [r['N'] for r in rows_excl if r['snr_variant'] == 'median9']
    ye = [r['front_connected_md_ft'] for r in rows_excl
          if r['snr_variant'] == 'median9']
    yi = [r['front_connected_md_ft'] for r in rows_incl
          if r['snr_variant'] == 'median9']
    a.plot(xs, ye, 'o-', color='k', label='stripe excluded')
    a.plot(xs, yi, '^--', color='#d95f02', label='stripe NOT excluded')
    a.axhline(md[0], color='r', ls=':', lw=1.2, label='DAS coverage limit')
    a.set_xscale('log')
    a.set_xticks(n_grid)
    a.set_xticklabels([f'{N:g}' for N in n_grid])
    a.set_xlabel('N')
    a.set_ylabel('front MD (ft)')
    a.set_title('(d) Sensitivity: leaving the artifact in\nmoves the front to the '
                'array edge at every N', fontsize=9.5)
    a.legend(fontsize=7.5)
    a.grid(alpha=0.3)

    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def fig_decay(path, md, dist, valid, norms, das_raw_ratio,
              sims, cmp, gauge_rows, sig, stripe_amp, resp):
    das_ratio = norms['raw']
    das_snr_ratio = norms['per_noise']
    das_art_ratio = norms['per_artifact']
    fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.8))
    cols = {'uniform_1150': '#d95f02', 'uniform_550': '#7b3294',
            'two_zone_r2': '#1b7837'}
    lab = {'uniform_1150': 'model, uniform D = 1150 ft$^2$/s (R1 absolute-norm optimum)',
           'uniform_550': 'model, uniform D = 550 ft$^2$/s (R1 normalised-norm optimum)',
           'two_zone_r2': 'model, two-zone D(x) (r2 best fit, RMSE 11.9 psi)'}
    rb_lo, rb_hi = cmp['reference_band_md_ft']
    d_lo, d_hi = float(np.max(dist[md >= rb_hi])), float(np.min(dist[md <= rb_lo]))

    a = ax[0]
    a.semilogy(dist[valid], das_raw_ratio[valid], color='0.75', lw=0.7,
               label='observed LF-DAS, per channel')
    a.semilogy(dist[valid], das_ratio[valid], 'k-', lw=1.8,
               label='observed LF-DAS strain rate, %d-channel median'
                     % cmp['spatial_median_channels'])
    for n, s in sims.items():
        a.semilogy(dist[valid], s['ratio'][valid], color=cols.get(n, '0.4'),
                   lw=1.6, label=lab.get(n, n))
    a.axvspan(d_lo, d_hi, color='0.85', zorder=0)
    a.text(0.5 * (d_lo + d_hi), a.get_ylim()[0] * 1.4, 'reference\nband',
           fontsize=8, ha='center')
    for r in gauge_rows:
        a.plot(r['distance_ft'], r['das_ratio'], 'kv', ms=5)
        a.text(r['distance_ft'], r['das_ratio'] * 1.35, "g%d" % r['gauge'],
               fontsize=7.5, ha='center')
    a.set_xlabel('distance from the source node at MD 16645 (ft)')
    a.set_ylabel('amplitude / amplitude in the reference band')
    a.set_title('(a) SHAPE comparison of far-field amplitude decay.\n'
                'Each curve normalised to ITS OWN median over MD %.0f-%.0f. '
                'LF-DAS stays in native\nstrain-rate units: no psi<->strain '
                'coefficient is used or fitted anywhere here.'
                % (rb_lo, rb_hi), fontsize=9.5)
    a.legend(fontsize=7.5, loc='lower left')
    a.grid(alpha=0.3, which='both')

    a = ax[1]
    for n, s in sims.items():
        a.semilogy(dist[valid], (s['ratio'] / das_ratio)[valid],
                   color=cols.get(n, '0.4'), lw=1.7, label=lab.get(n, n))
        a.semilogy(dist[valid], (s['ratio'] / das_snr_ratio)[valid],
                   color=cols.get(n, '0.4'), lw=1.1, ls='--')
        a.semilogy(dist[valid], (s['ratio'] / das_art_ratio)[valid],
                   color=cols.get(n, '0.4'), lw=0.9, ls=':')
    a.axhline(1.0, color='k', lw=1.0)
    a.text(dist[valid].max(), 1.06, 'model = DAS', fontsize=8, ha='right')
    a.plot([], [], 'k-', lw=1.7, label='DAS normalisation: raw amplitude')
    a.plot([], [], 'k--', lw=1.1, label='             divided by channel noise floor')
    a.plot([], [], 'k:', lw=0.9,
           label='             divided by artifact amplitude (diagnostic only)')
    a.set_xlabel('distance from the source node at MD 16645 (ft)')
    a.set_ylabel('model / DAS  (normalised amplitude ratio)')
    a.set_title('(b) Above 1 = the model retains more far-field dP/dt amplitude\n'
                'than the LF-DAS retains strain-rate amplitude. The SIGN of the\n'
                'discrepancy depends on the DAS normalisation (solid vs dashed).',
                fontsize=9.5)
    a.legend(fontsize=7.0, loc='lower left', ncol=1)
    a.grid(alpha=0.3, which='both')

    a = ax[2]
    a.semilogy(md, sig / np.median(sig), 'k-', lw=1.2,
               label='quiescent noise floor per channel (%.1fx deep/shallow)'
                     % resp['noise_floor_deep_over_shallow'])
    a.semilogy(md, stripe_amp / np.median(stripe_amp), color='#d95f02', lw=1.2,
               label='amplitude of the acquisition artifact (%.0fx deep/shallow)'
                     % resp['artifact_amplitude_deep_over_shallow'])
    a.set_xlabel('MD (ft)')
    a.set_ylabel('amplitude / array median (native units)')
    a.set_title('(c) Why (b) is unresolved. Two independent probes of the channel\n'
                'response disagree by ~%.0fx: an instrumental transient varies '
                '%.0fx along\nthe array, the noise floor only %.1fx and in the '
                'opposite sense.'
                % (resp['artifact_amplitude_deep_over_shallow']
                   / resp['noise_floor_deep_over_shallow'],
                   resp['artifact_amplitude_deep_over_shallow'],
                   resp['noise_floor_deep_over_shallow']), fontsize=9.5)
    a.legend(fontsize=7.5, loc='upper left')
    a.grid(alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def code_hashes():
    out = {}
    for _, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if not f or not f.endswith('.py'):
            continue
        f = os.path.abspath(f)
        if not f.startswith(REPO) or not os.path.exists(f):
            continue
        out[_rel(f)] = core.file_sha256(f)
    return dict(sorted(out.items()))


def write_manifest(path, cfg, cfg_sha, cfg_path, mesh, source_idx, src, dt_s,
                   n_steps, results, out_paths, cfg_full, series):
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import rev2_manifest                                   # noqa: F401
        has_shared = hasattr(rev2_manifest, 'write_manifest')
    except Exception:
        has_shared = False
    import matplotlib as mpl
    import scipy
    import fiberis

    data_files = [cfg['data']['das_npz'], cfg['data']['pump_rate_npz'],
                  cfg['data']['gauge_md_npz'], cfg['data']['frac_hit_stage1_npz']]
    data_files += [cfg['data']['gauge_series_template'].format(n=n)
                   for n in sorted(series)]
    man = {
        'study_id': cfg['study_id'],
        'run_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_resolved': cfg,
        'config_sha256': cfg_sha,
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'matplotlib': mpl.__version__,
            'fiberis_path': os.path.dirname(os.path.abspath(fiberis.__file__)),
            'cwd': os.getcwd(),
            'code_sha256': code_hashes(),
            'input_data_sha256': {_rel(p): core.file_sha256(p)
                                  for p in data_files},
            'shared_rev2_manifest_module_available': bool(has_shared),
            'config_path': _rel(cfg_path),
            'note': ('.git is empty; code identity is pinned by sha256 of every '
                     '.py loaded at manifest-write time.'),
        },
        'source_protocol': {
            'source_md_ft': float(src['md_ft']),
            'source_gauge': int(src['gauge']),
            'driving_series_path': _rel(
                cfg['data']['gauge_series_template'].format(n=src['gauge'])),
            'application': 'dirichlet_node',
            'source_mesh_idx': int(source_idx),
            'md_snap_residual_ft': float(abs(mesh[source_idx] - src['md_ft'])),
        },
        'numerics': {
            'theta': 1.0,
            'interface_avg': 'harmonic',
            'dt_s': float(dt_s),
            'adaptive': None,
            'n_steps': int(n_steps),
            'domain_md_ft': [float(mesh[0]), float(mesh[-1])],
            'pad_low_ft': float(cfg['forward_model']['mesh']['domain_pad_low_md_ft']),
            'pad_high_ft': float(cfg['forward_model']['mesh']['domain_pad_high_md_ft']),
            'dx_ft': float(cfg['forward_model']['mesh']['dx_ft']),
            'nx': int(mesh.size),
            'barrier': None,
        },
        'results': results,
        'outputs': [],
    }
    for p in out_paths:
        if os.path.exists(p):
            man['outputs'].append({'path': _rel(p),
                                   'bytes': int(os.path.getsize(p)),
                                   'sha256': core.file_sha256(p)})
    with open(path, 'w') as fh:
        json.dump(man, fh, indent=2, default=float)
    man['outputs'].append({'path': _rel(path),
                           'bytes': int(os.path.getsize(path)),
                           'sha256': core.file_sha256(path),
                           'note': 'self-hash of the manifest as first written'})
    return man


if __name__ == '__main__':
    main()
