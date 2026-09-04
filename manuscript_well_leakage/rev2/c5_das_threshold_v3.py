"""C5 AMEND (v3) - the LF-DAS "front" is a detection threshold, not a physical
boundary, and the array does not run out where the v2 run said it did.

Run from the repository root:

    python scripts/manuscript_well_leakage/rev2/c5_das_threshold_v3.py \
        --config configs/rev2/c5_das_threshold_v3.json

Why this file exists
--------------------
Two independent reviewers re-derived the C5 v2 study from scratch and reproduced
a defect list (output/rev2_20260901/A4/challenge_defects/C5_defects.json).  Each
defect was reproduced again here before being acted on.  The substantive changes
against `c5_das_threshold.py` (v2), which is left untouched so that the v2
manifest's code hashes keep verifying:

  1. BLOCKER.  "MD 15003 is the shallow limit of DAS coverage" is false: the
     stage-1 file covers MD -548.97 to 16890.61 in 5240 channels.  15003 is the
     low edge of the R1 *pressure-comparison* window, inherited through
     `md_min_ft`.  The front statistic is now computed over MD 11000-16750, and
     the censoring narrative is deleted rather than restated.
  2. BLOCKER.  The "front advances then retreats" claim was an artifact of the
     source-connected front definition: at t = 460 s ~5 channels immediately
     below the frac-hit anchor drop under 2 sigma and sever the anchor from a
     block that still spans MD 15120-16644.  Both front definitions are now
     computed, plotted and recorded.
  3. BLOCKER.  P(SNR > 3) = 0 over the v2 binned null is arithmetic, not
     evidence: that statistic is a 121 s bin RMS divided by the RMS of the
     746 s block containing it, so it cannot exceed sqrt(746/121) = 2.483.
     Replaced by calibrations that CAN exceed the thresholds.
  4. BLOCKER/MAJOR.  The far-field shape comparison stopped at MD 15003 for the
     same reason as (1).  It now runs to MD 13000 on a pad-insensitive mesh, and
     every band ratio carries a reference-band sensitivity axis and an IQR.

LF-DAS is kept in NATIVE STRAIN-RATE UNITS throughout.  No psi<->strain
conversion is applied, used or fitted anywhere in this script.
"""

import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.colors import TwoSlopeNorm               # noqa: E402
from scipy.ndimage import median_filter, binary_closing  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
BASE = os.path.join(REPO, 'scripts', 'manuscript_well_leakage',
                    'baseline_calibration')
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)
import r1_calibration_core as core        # noqa: E402
import r1_run_calibration as runner       # noqa: E402
# Shared rev2 modules.  House rule 3: the manifest MUST come from rev2_manifest.
# v2 wrapped this import in `except Exception: has_shared = False` and then wrote
# its own document; a missing shared writer must be a hard failure, never a
# boolean in the output.
import rev2_core as rcore                 # noqa: E402
import rev2_manifest as rman              # noqa: E402
import rev2_data as rdata                 # noqa: E402

VERSION = 'v3'
TASK_ID = 'C5-amend'


def log(msg):
    print(f"[c5v3] {msg}", flush=True)


def _rel(p):
    p = os.path.abspath(p)
    return os.path.relpath(p, REPO) if p.startswith(REPO) else p


def _f(x):
    return None if x is None else float(x)


# ---------------------------------------------------------------------------
# small helpers (identical to v2 so the overlapping numbers reproduce exactly)
# ---------------------------------------------------------------------------

def boxcar_nanmean(X, L):
    """Centred boxcar mean along axis 1, NaN-aware."""
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


def _closed(det, close_gap=2):
    d = binary_closing(det, structure=np.ones(close_gap + 1, bool))
    return d | det


def front_connected(det, md, anchor, close_gap=2):
    d = _closed(det, close_gap)
    if not d[anchor]:
        return None
    i = anchor
    while i > 0 and d[i - 1]:
        i -= 1
    return float(md[i])


def front_longest_run(det, md, close_gap=2):
    """Shallowest MD of the LONGEST contiguous detected run.

    Anchor-free, so it cannot be severed by a near-source dropout, and it cannot
    be set by an isolated false alarm either.  Added in v3 because the v2
    "advance then retreat" claim was an artifact of the anchored definition.
    """
    d = _closed(det, close_gap)
    runs = contiguous_runs(np.where(d)[0])
    if not runs:
        return None, 0
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return float(md[a]), int(b - a + 1)


def run_length_containing(det, i, close_gap=2):
    d = _closed(det, close_gap)
    if not d[i]:
        return 0
    a = b = i
    while a > 0 and d[a - 1]:
        a -= 1
    while b < d.size - 1 and d[b + 1]:
        b += 1
    return int(b - a + 1)


def front_connected_idx(det, md, anchor, close_gap=2):
    d = _closed(det, close_gap)
    if not d[anchor]:
        return None
    i = anchor
    while i > 0 and d[i - 1]:
        i -= 1
    return i


def snr_block(A, P, mask, L, medch):
    """window RMS / pre-pumping RMS of the low-passed, baseline-removed signal."""
    base = P.mean(axis=1)
    Y = A - base[:, None]
    if mask is not None:
        Y = Y.copy()
        Y[:, mask] = np.nan
    Ypre = P - base[:, None]
    h = int(L) // 2
    sl = slice(h, -h) if h > 0 else slice(None)
    Ys = boxcar_nanmean(Y, L)[:, sl]
    Ps = boxcar_nanmean(Ypre, L)[:, sl]
    S = np.sqrt(np.nanmean(Ys ** 2, axis=1))
    sig = np.sqrt(np.nanmean(Ps ** 2, axis=1))
    snr = S / sig
    return snr, median_filter(snr, size=medch, mode='nearest'), S, sig


def sweep_fronts(snr, snr_med, md, anchor, n_grid, edge_tol):
    rows = []
    for N in n_grid:
        for label, s in (('raw', snr), ('median9', snr_med)):
            det = s > N
            f_sh = front_shallowest(det, md)
            i_cn = front_connected_idx(det, md, anchor)
            f_cn = None if i_cn is None else float(md[i_cn])
            f_lr, n_lr = front_longest_run(det, md)
            i_sh = int(np.argmax(det)) if det.any() else None
            rows.append({
                'N': float(N), 'snr_variant': label,
                'front_longest_run_md_ft': f_lr,
                'longest_run_n_channels': n_lr,
                'front_shallowest_md_ft': f_sh,
                'shallowest_run_n_channels': (None if i_sh is None else
                                              run_length_containing(det, i_sh)),
                'front_shallowest_at_domain_edge': (f_sh is not None
                                                    and f_sh - md[0] <= edge_tol),
                'front_connected_md_ft': f_cn,
                'front_connected_at_domain_edge': (f_cn is not None
                                                   and f_cn - md[0] <= edge_tol),
                'n_undetected_below_front': (None if i_cn is None
                                             else int((~det[:i_cn]).sum())),
                'n_channels_detected': int(det.sum()),
                'fraction_detected': float(det.mean()),
            })
    return rows


# ---------------------------------------------------------------------------
# pre-pumping window (unchanged rule; the QUOTED interval is now DAS samples)
# ---------------------------------------------------------------------------

def pre_pumping_window(cfg, gaps_utc):
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

    lo = shut_lo
    for g in gaps_utc:
        if shut_lo <= g['utc_end'] <= shut_hi:
            lo = max(lo, g['utc_end'] + datetime.timedelta(seconds=spec['gap_guard_s']))
    lo = lo.replace(microsecond=0) + datetime.timedelta(seconds=1)
    hi = inj_start - datetime.timedelta(seconds=spec['injection_guard_s'])

    first_run = [(a2, b2) for a2, b2 in runs if tp[a2] == tp[0]]
    alt_hi = stp + datetime.timedelta(seconds=float(tp[first_run[0][1]]))

    info = {
        'pump_curve': _rel(p),
        'pump_curve_start_utc': stp.isoformat(),
        'selected_shut_in_utc': [shut_lo.isoformat(), shut_hi.isoformat()],
        'stage1_main_injection_start_utc': inj_start.isoformat(),
        'pre_pumping_window_requested_utc': [lo.isoformat(), hi.isoformat()],
        'alternative_pre_any_pumping_pump_curve_utc': [
            (stp + datetime.timedelta(seconds=float(tp[0]))).isoformat(),
            alt_hi.isoformat()],
        'slurry_rate_bpm_in_analysis_window': [
            float(v[(tp >= w0) & (tp <= w0 + 1260)].min()),
            float(v[(tp >= w0) & (tp <= w0 + 1260)].max())],
    }
    return lo, hi, alt_hi, info


# ---------------------------------------------------------------------------
# stripe detection
# ---------------------------------------------------------------------------

def band_exceedance(full, da, tm, pm, lo, hi, k):
    idx = np.where((da >= lo) & (da < hi))[0]
    A = full[idx[0]:idx[-1] + 1, tm[0]:tm[-1] + 1].astype(np.float64)
    P = full[idx[0]:idx[-1] + 1, pm[0]:pm[-1] + 1].astype(np.float64)
    base = P.mean(axis=1)
    sig = np.sqrt(np.mean((P - base[:, None]) ** 2, axis=1))
    sig = np.where(sig > 0, sig, np.inf)
    f_win = np.mean(np.abs(A - base[:, None]) / sig[:, None] > k, axis=0)
    f_pre = np.mean(np.abs(P - base[:, None]) / sig[:, None] > k, axis=0)
    return idx, f_win, f_pre


def detect_stripe(cfg, full, da, tm, pm, tw):
    spec = cfg['stripe_detection']
    lo, hi = spec['reference_band_md_ft']
    k = spec['per_channel_exceedance_sigma']
    ref, f_win, f_pre = band_exceedance(full, da, tm, pm, lo, hi, k)

    thr = spec['flag_fraction_threshold']
    runs = contiguous_runs(np.where(f_win > thr)[0])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= 5:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append((r[0], r[1]))
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

    # The 'present on every band tested' claim is load-bearing, so it is MEASURED
    # and recorded here rather than asserted (v2 recorded only the downlead).
    bands = []
    t_lo, t_hi = windows[0]
    inw = (tw >= t_lo - 8) & (tw <= t_hi + 8)
    for blo, bhi in spec['multi_band_check_md_ft']:
        bidx, bw, bp = band_exceedance(full, da, tm, pm, blo, bhi, k)
        bands.append({
            'md_lo_ft': float(da[bidx[0]]), 'md_hi_ft': float(da[bidx[-1]]),
            'n_channels': int(bidx.size),
            'max_exceedance_fraction_in_stripe_window': float(bw[inw].max()),
            't_win_s_at_max': float(tw[inw][int(np.argmax(bw[inw]))]),
            'max_exceedance_fraction_elsewhere_in_window': float(bw[~inw].max()),
            'pre_pumping_false_alarm_floor': float(bp.max()),
        })
    info['multi_band_exceedance'] = bands
    return mask, f_win, f_pre, info


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(args.config, 'rb') as fh:
        raw = fh.read()
    cfg = json.loads(raw.decode('utf-8'))
    cfg_sha = hashlib.sha256(raw).hexdigest()
    outdir = cfg['outputs']['dir']
    os.makedirs(outdir, exist_ok=True)
    log(f"config {args.config} sha256={cfg_sha[:16]}")

    # ---- 0. load the FULL fibre and verify what the array actually covers --
    das_path = cfg['data']['das_npz']
    z = np.load(das_path, allow_pickle=True)
    t = z['taxis'].astype(float)
    da = z['daxis'].astype(float)
    st = z['start_time'].item()
    full = z['data']

    w = cfg['window']
    t0 = datetime.datetime.fromisoformat(w['time_start'])
    t1 = datetime.datetime.fromisoformat(w['time_end'])
    tm = np.where((t >= (t0 - st).total_seconds())
                  & (t <= (t1 - st).total_seconds()))[0]
    tw = t[tm] - t[tm][0]
    spacing = float(np.median(np.diff(da)))
    gaps = np.where(np.diff(t) > 1.5)[0]
    gaps_utc = [{'utc_start': st + datetime.timedelta(seconds=float(t[i])),
                 'utc_end': st + datetime.timedelta(seconds=float(t[i + 1])),
                 'gap_s': float(t[i + 1] - t[i])} for i in gaps]

    fw = cfg['front_window']
    dmF = np.where((da >= fw['md_min_ft']) & (da <= fw['md_max_ft']))[0]
    dmR = np.where((da >= w['md_min_ft']) & (da <= w['md_max_ft']))[0]
    mdF = da[dmF]
    mdR = da[dmR]

    ver = {
        'file': _rel(das_path),
        'full_array_shape': [int(full.shape[0]), int(full.shape[1])],
        'full_md_range_ft': [float(da[0]), float(da[-1])],
        'channel_spacing_ft': spacing,
        'stored_dtype': str(full.dtype),
        'das_start_time_utc': st.isoformat(),
        'n_channels_below_md_15000': int((da < 15000.0).sum()),
        'n_channels_in_front_window': int(dmF.size),
        'n_channels_in_r1_window': int(dmR.size),
        'r1_window_md_span_ft': [float(mdR[0]), float(mdR[-1])],
        'front_window_md_span_ft': [float(mdF[0]), float(mdF[-1])],
        'n_samples_in_window': int(tm.size),
        'window_start_offset_s': float(t[tm[0]] - (t0 - st).total_seconds()),
        'data_gaps_full_file': [{'utc': g['utc_start'].isoformat(),
                                 'gap_s': g['gap_s']} for g in gaps_utc],
        'CORRECTION_v2': ('v2 called MD 15003 "the shallow limit of DAS '
                          'coverage" and flagged fronts there as censored. That '
                          'is false: 15003 is the low edge of the R1 pressure-'
                          'comparison window (config md_min_ft), and the fibre '
                          'continues %d channels further up-hole to MD %.2f. No '
                          'front in this run is censored.'
                          % (int((da < 15000.0).sum()), float(da[0]))),
    }
    log(f"fibre MD {da[0]:.2f}..{da[-1]:.2f} in {da.size} channels; "
        f"front window {mdF[0]:.1f}..{mdF[-1]:.1f} ({dmF.size} ch); "
        f"{int((da < 15000).sum())} channels lie BELOW MD 15000")

    # ---- 1. pre-pumping window -------------------------------------------
    p_lo, p_hi, alt_hi, pre_info = pre_pumping_window(cfg, gaps_utc)
    pm = np.where((t >= (p_lo - st).total_seconds())
                  & (t <= (p_hi - st).total_seconds()))[0]
    am = np.where(t <= (alt_hi - st).total_seconds())[0]
    pre_info['n_das_samples'] = int(pm.size)
    pre_info['pre_pumping_window_das_samples_utc'] = [
        (st + datetime.timedelta(seconds=float(t[pm[0]]))).isoformat(),
        (st + datetime.timedelta(seconds=float(t[pm[-1]]))).isoformat()]
    pre_info['alternative_n_das_samples'] = int(am.size)
    pre_info['alternative_das_samples_utc'] = [
        (st + datetime.timedelta(seconds=float(t[am[0]]))).isoformat(),
        (st + datetime.timedelta(seconds=float(t[am[-1]]))).isoformat()]
    pre_info['alternative_quoting_correction'] = (
        'v2 quoted this window as 10:27:08-10:29:30 (the PUMP-CURVE interval, '
        '142 s) alongside 111 DAS samples. The DAS file starts at %s, so the '
        'block actually used is %s to %s = %.1f s = %d samples. After the 31 s '
        'boxcar that leaves ~%.1f effective independent samples, so it is a '
        'worst case, not a co-equal alternative.'
        % (st.isoformat(),
           pre_info['alternative_das_samples_utc'][0][11:],
           pre_info['alternative_das_samples_utc'][1][11:],
           float(t[am[-1]] - t[am[0]]), int(am.size),
           (float(t[am[-1]] - t[am[0]]) + 1.0) / 31.0))
    log(f"pre-pumping DAS block {pre_info['pre_pumping_window_das_samples_utc']}"
        f" ({pm.size} samples); alternative "
        f"{pre_info['alternative_das_samples_utc']} ({am.size} samples)")

    # ---- 2. stripe --------------------------------------------------------
    stripe_mask, f_win, f_pre, stripe_info = detect_stripe(cfg, full, da, tm,
                                                           pm, tw)
    log(f"stripe {stripe_info['excluded_windows_t_win_s']} "
        f"({stripe_info['n_samples_excluded']}/{tw.size} samples); per-band "
        "max exceedance in the stripe window: "
        + ', '.join("%.0f-%.0f:%.3f" % (b['md_lo_ft'], b['md_hi_ft'],
                                        b['max_exceedance_fraction_in_stripe_window'])
                    for b in stripe_info['multi_band_exceedance']))

    nc = cfg['noise_criterion']
    L = int(nc['lowpass_boxcar_s'])
    medch = int(nc['spatial_median_filter_channels'])
    n_grid = [float(x) for x in nc['N_grid']]
    edge_tol = spacing

    frac_hits = np.unique(np.load(cfg['data']['frac_hit_stage1_npz'],
                                  allow_pickle=True)['data'].astype(float))
    anchor_md = float(np.mean(frac_hits))
    anchor = int(np.argmin(np.abs(mdF - anchor_md)))
    log(f"anchor: frac-hit centroid MD {anchor_md:.2f} -> channel {anchor} "
        f"at MD {mdF[anchor]:.2f}")

    # ---- 3. SNR + fronts over the FULL fibre ------------------------------
    AF = full[dmF[0]:dmF[-1] + 1, tm[0]:tm[-1] + 1].astype(np.float64)
    PF = full[dmF[0]:dmF[-1] + 1, pm[0]:pm[-1] + 1].astype(np.float64)
    AltF = full[dmF[0]:dmF[-1] + 1, am[0]:am[-1] + 1].astype(np.float64)

    snr, snr_m, S_win, sig = snr_block(AF, PF, stripe_mask, L, medch)
    rows_excl = sweep_fronts(snr, snr_m, mdF, anchor, n_grid, edge_tol)
    for r in rows_excl:
        r['stripe'] = 'excluded'
    snr_i, snr_mi, _, _ = snr_block(AF, PF, None, L, medch)
    rows_incl = sweep_fronts(snr_i, snr_mi, mdF, anchor, n_grid, edge_tol)
    for r in rows_incl:
        r['stripe'] = 'included'
    snr_r, snr_rm, _, _ = snr_block(AF, PF, stripe_mask, 1, medch)
    rows_nolp = sweep_fronts(snr_r, snr_rm, mdF, anchor, n_grid, edge_tol)
    for r in rows_nolp:
        r['stripe'] = 'excluded_no_lowpass'
    snr_a, snr_am, _, sig_alt = snr_block(AF, AltF, stripe_mask, L, medch)
    rows_alt = sweep_fronts(snr_a, snr_am, mdF, anchor, n_grid, edge_tol)
    for r in rows_alt:
        r['stripe'] = 'excluded_alt_noise_window'
    all_rows = rows_excl + rows_incl + rows_nolp + rows_alt

    # the same sweep restricted to the R1 window, to show exactly what the
    # truncation cost v2
    iR = np.searchsorted(mdF, mdR[0])
    rows_r1 = sweep_fronts(snr[iR:], snr_m[iR:], mdF[iR:],
                           anchor - iR, n_grid, edge_tol)
    for r in rows_r1:
        r['stripe'] = 'excluded_R1_window_only'
    all_rows += rows_r1

    for r in rows_excl:
        if r['snr_variant'] == 'median9':
            log(f"  N={r['N']:g} connected front MD {r['front_connected_md_ft']}"
                f"  shallowest {r['front_shallowest_md_ft']}"
                f"  frac det {r['fraction_detected']:.3f}"
                f"  undetected channels below the front: "
                f"{r['n_undetected_below_front']}")

    # ---- 4. false-alarm calibration that CAN exceed the thresholds --------
    fa = cfg['false_alarm_calibration']
    controls = []
    for blo, bhi in fa['control_bands_md_ft']:
        idx = np.where((da >= blo) & (da <= bhi))[0]
        Ab = full[idx[0]:idx[-1] + 1, tm[0]:tm[-1] + 1].astype(np.float64)
        Pb = full[idx[0]:idx[-1] + 1, pm[0]:pm[-1] + 1].astype(np.float64)
        s_b, s_bm, _, _ = snr_block(Ab, Pb, stripe_mask, L, medch)
        controls.append({
            'md_lo_ft': float(da[idx[0]]), 'md_hi_ft': float(da[idx[-1]]),
            'n_channels': int(idx.size),
            'median_snr_median9': float(np.median(s_bm)),
            **{f'frac_median9_gt_{N:g}': float((s_bm > N).mean())
               for N in n_grid},
            **{f'frac_raw_gt_{N:g}': float((s_b > N).mean()) for N in n_grid},
        })
        log("  control band MD %8.1f-%8.1f n=%4d median SNR9 %.2f  "
            % (controls[-1]['md_lo_ft'], controls[-1]['md_hi_ft'],
               controls[-1]['n_channels'], controls[-1]['median_snr_median9'])
            + ' '.join("P9>%g=%.3f" % (N, controls[-1][f'frac_median9_gt_{N:g}'])
                       for N in n_grid))
    # the R1 window itself, for contrast
    s_w, s_wm, _, _ = snr_block(full[dmR[0]:dmR[-1] + 1, tm[0]:tm[-1] + 1].astype(float),
                                full[dmR[0]:dmR[-1] + 1, pm[0]:pm[-1] + 1].astype(float),
                                stripe_mask, L, medch)
    _sigR = snr_block(full[dmR[0]:dmR[-1] + 1, tm[0]:tm[-1] + 1].astype(float),
                      full[dmR[0]:dmR[-1] + 1, pm[0]:pm[-1] + 1].astype(float),
                      stripe_mask, L, medch)[3]
    sigma_r1 = {'min': float(_sigR.min()), 'p10': float(np.percentile(_sigR, 10)),
                'median': float(np.median(_sigR)),
                'p90': float(np.percentile(_sigR, 90)), 'max': float(_sigR.max()),
                'n_channels_gt_5x_median': int((_sigR > 5 * np.median(_sigR)).sum()),
                'n_channels': int(_sigR.size)}
    gauge_snr = {}
    for gn, gmd_ in ((6, 15344.0), (7, 15075.0), (8, 14821.0), (9, 14552.0),
                     (10, 14297.0), (11, 14028.0)):
        k_ = int(np.argmin(np.abs(mdF - gmd_)))
        gauge_snr[f'g{gn}'] = {'md_ft': float(mdF[k_]), 'snr': float(snr[k_]),
                               'snr_median9': float(snr_m[k_])}
    signal_band = {'md_lo_ft': float(mdR[0]), 'md_hi_ft': float(mdR[-1]),
                   'n_channels': int(dmR.size),
                   'median_snr_median9': float(np.median(s_wm)),
                   **{f'frac_median9_gt_{N:g}': float((s_wm > N).mean())
                      for N in n_grid}}

    base_p = PF.mean(axis=1)
    Pl = boxcar_nanmean(PF - base_p[:, None], L)[:, L // 2:-(L // 2)]
    PR_pre = full[dmR[0]:dmR[-1] + 1, pm[0]:pm[-1] + 1].astype(np.float64)

    # split-half of the quiet block: a within-block null with NO upper bound.
    # Reported two ways (each half about the whole-block mean, which is the
    # convention the detection statistic itself uses, and each half about its
    # OWN mean, which is the reviewers' convention and removes a DC drift), and
    # on two bands, because the drift is a property of the R1-window channels.
    def split_half_and_drift(P, label):
        base_all = P.mean(axis=1)
        Pl_c = boxcar_nanmean(P - base_all[:, None], L)[:, L // 2:-(L // 2)]
        nh = Pl_c.shape[1] // 2
        out = {'band': label, 'n_channels': int(P.shape[0])}
        for tag, arr in (('common_baseline', Pl_c),):
            a1 = np.sqrt(np.nanmean(arr[:, :nh] ** 2, axis=1))
            a2 = np.sqrt(np.nanmean(arr[:, nh:2 * nh] ** 2, axis=1))
            rr = a1 / a2
            rr9 = median_filter(rr, size=medch, mode='nearest')
            out[tag] = {'median_raw': float(np.median(rr)),
                        'median_median9': float(np.median(rr9)),
                        'max_median9': float(rr9.max()),
                        **{f'frac_raw_gt_{N:g}': float((rr > N).mean())
                           for N in n_grid},
                        **{f'frac_median9_gt_{N:g}': float((rr9 > N).mean())
                           for N in n_grid}}
        half = P.shape[1] // 2
        A1, B1 = P[:, :half], P[:, half:]
        As = boxcar_nanmean(A1 - A1.mean(axis=1, keepdims=True), L)[:, L // 2:-(L // 2)]
        Bs = boxcar_nanmean(B1 - B1.mean(axis=1, keepdims=True), L)[:, L // 2:-(L // 2)]
        rr = np.sqrt(np.nanmean(As ** 2, axis=1)) / np.sqrt(np.nanmean(Bs ** 2, axis=1))
        rr9 = median_filter(rr, size=medch, mode='nearest')
        out['own_baseline'] = {'median_raw': float(np.median(rr)),
                               'median_median9': float(np.median(rr9)),
                               'max_median9': float(rr9.max()),
                               **{f'frac_raw_gt_{N:g}': float((rr > N).mean())
                                  for N in n_grid},
                               **{f'frac_median9_gt_{N:g}': float((rr9 > N).mean())
                                  for N in n_grid}}
        # quarters, each about its OWN mean (the reviewers' convention: it
        # measures the FLUCTUATION level, free of any DC drift)
        qn = P.shape[1] // 4
        qr = []
        for i in range(4):
            Q = P[:, i * qn:(i + 1) * qn]
            Qs = boxcar_nanmean(Q - Q.mean(axis=1, keepdims=True), L)[:, L // 2:-(L // 2)]
            qr.append(np.sqrt(np.nanmean(Qs ** 2, axis=1)))
        qr = np.asarray(qr)
        q = [float(np.median(x)) for x in qr]
        out['per_quarter_median_rms_native_units'] = q
        out['quarter_over_last_quarter_median_of_ratios'] = [
            float(np.median(x / qr[3])) for x in qr]
        out['drift_first_over_last_quarter'] = float(np.median(qr[0] / qr[3]))
        qn2 = Pl_c.shape[1] // 4
        out['per_quarter_median_rms_common_baseline'] = [
            float(np.median(np.sqrt(np.nanmean(Pl_c[:, i * qn2:(i + 1) * qn2] ** 2,
                                               axis=1)))) for i in range(4)]
        return out

    split_half = [split_half_and_drift(PF, 'front window MD 11000-16750'),
                  split_half_and_drift(PR_pre, 'R1 window MD 15000-16750')]
    stationarity = {
        'note': ('The "quiet" block is not stationary over the channels that '
                 'carry the signal: across the R1 window its own noise level '
                 'falls by %.2fx from the first quarter to the last. That is why '
                 'the alternative (earlier, quieter) noise window pushes every '
                 'front up-hole, and it is why a split-half of the SAME block '
                 'already exceeds N = 3 for %.1f%% of R1-window channels under '
                 'the reviewers\' own-baseline convention.'
                 % (split_half[1]['drift_first_over_last_quarter'],
                    100 * split_half[1]['own_baseline']['frac_median9_gt_3'])),
        'per_band': [{'band': d['band'],
                      'per_quarter_median_rms_native_units':
                          d['per_quarter_median_rms_native_units'],
                      'quarter_over_last_quarter_median_of_ratios':
                          d['quarter_over_last_quarter_median_of_ratios'],
                      'drift_first_over_last_quarter':
                          d['drift_first_over_last_quarter']}
                     for d in split_half]}

    # the v2 binned null, retained WITH the bound that produced P(>3) = 0
    tr = nc['time_resolved']
    binlen, stride = int(tr['bin_len_s']), int(tr['stride_s'])
    nullv = []
    centres_n = list(range(binlen // 2, Pl.shape[1] - binlen // 2, stride))
    for c in centres_n:
        sl = slice(c - binlen // 2, c + binlen // 2 + 1)
        nullv.append(np.sqrt(np.nanmean(Pl[:, sl] ** 2, axis=1)) / sig)
    nullv = np.concatenate(nullv)
    n_sigma_samples = int(Pl.shape[1])
    hard_bound = float(np.sqrt(n_sigma_samples / binlen))
    binned_null = {
        'median': float(np.nanmedian(nullv)),
        'p95': float(np.nanpercentile(nullv, 95)),
        'p99': float(np.nanpercentile(nullv, 99)),
        'max': float(np.nanmax(nullv)),
        'frac_above_2': float(np.nanmean(nullv > 2.0)),
        'frac_above_3': float(np.nanmean(nullv > 3.0)),
        'n_overlapping_bins': int(nullv.size),
        'n_non_overlapping_bins': int(n_sigma_samples // binlen) * int(mdF.size),
        'n_channels': int(mdF.size),
        'bin_overlap_fraction': float(1.0 - stride / binlen),
        'hard_upper_bound': hard_bound,
        'observed_max_over_bound': float(np.nanmax(nullv) / hard_bound),
        'INVALID_AS_A_FALSE_ALARM_RATE': (
            'This statistic is a %d s bin RMS divided by the RMS of the %d s '
            'block that CONTAINS it, so it cannot exceed sqrt(%d/%d) = %.3f. '
            'P(>3) = 0 is therefore arithmetic, not evidence, and the v2 '
            'inference "N >= 3 admits no false alarms, so MD 15110 is the most '
            'defensible single number" does not follow. Its n was also quoted at '
            '%d overlapping bins (%.0f%% overlap); the non-overlapping count is '
            '%d. Use control_bands and split_half instead.'
            % (binlen, n_sigma_samples, n_sigma_samples, binlen, hard_bound,
               int(nullv.size), 100 * (1.0 - stride / binlen),
               int(n_sigma_samples // binlen) * int(mdF.size))),
    }
    log(f"binned null: max {binned_null['max']:.3f} vs hard bound {hard_bound:.3f} "
        "-> P(>3)=0 is arithmetic. R1-window split-half of the SAME quiet block: "
        f"P(med9>3) = {split_half[1]['own_baseline']['frac_median9_gt_3']:.3f} "
        f"(own-baseline) / {split_half[1]['common_baseline']['frac_median9_gt_3']:.3f} "
        f"(common baseline); quiet-block drift "
        f"{split_half[1]['drift_first_over_last_quarter']:.2f}x")

    # ---- 5. the legacy rule, raw and DC-removed ---------------------------
    lg = cfg['legacy_rule']
    m100 = (tw >= 100) & (tw <= 450)
    AR = full[dmR[0]:dmR[-1] + 1, tm[0]:tm[-1] + 1].astype(np.float64)
    PR = full[dmR[0]:dmR[-1] + 1, pm[0]:pm[-1] + 1].astype(np.float64)
    baseR = PR.mean(axis=1)
    a_leg = AR[:, m100].mean(axis=1)                  # as published: about ZERO
    a_dc = (AR - baseR[:, None])[:, m100].mean(axis=1)  # about the channel baseline
    anchorR = int(np.argmin(np.abs(mdR - anchor_md)))
    pk = float(np.nanmax(a_leg))
    pk_dc = float(np.nanmax(a_dc))
    a_dc_m9 = median_filter(a_dc, size=medch, mode='nearest')
    i6 = int(np.argmin(np.abs(mdR - 15344.0)))
    i7 = int(np.argmin(np.abs(mdR - 15075.0)))
    legacy = {
        'statistic': lg['statistic'],
        'baseline_convention': lg['baseline_convention'],
        'global_peak_native_units': pk,
        'global_peak_at_md_ft': float(mdR[int(np.argmax(a_leg))]),
        'threshold_native_units': 0.05 * pk,
        'front_shallowest_md_ft': front_shallowest(a_leg >= 0.05 * pk, mdR),
        'front_connected_md_ft': front_connected(a_leg >= 0.05 * pk, mdR, anchorR),
        'n_channels_detected': int((a_leg >= 0.05 * pk).sum()),
        'published_value_md_ft': lg['published_value_md_ft'],
        'g6_md_ft': float(mdR[i6]), 'g7_md_ft': float(mdR[i7]),
        'g6_statistic_over_peak': float(a_leg[i6] / pk),
        'g7_statistic_over_peak': float(a_leg[i7] / pk),
        'dc_removed': {
            'global_peak_native_units': pk_dc,
            'global_peak_at_md_ft': float(mdR[int(np.argmax(a_dc))]),
            'front_shallowest_md_ft': front_shallowest(a_dc >= 0.05 * pk_dc, mdR),
            'n_channels_detected': int((a_dc >= 0.05 * pk_dc).sum()),
            'g6_statistic_over_peak': float(a_dc[i6] / pk_dc),
            'g6_statistic_over_peak_median9': float(a_dc_m9[i6] / pk_dc),
            'g7_statistic_over_peak': float(a_dc[i7] / pk_dc),
            'g7_statistic_over_peak_median9': float(a_dc_m9[i7] / pk_dc),
            'g6_pre_pumping_mean_native_units': float(baseR[i6]),
            'g7_pre_pumping_mean_native_units': float(baseR[i7]),
            'g6_neighbour_pre_pumping_means': [float(x) for x in
                                               baseR[i6 - 4:i6 + 5]],
        },
        'MECHANISM_CORRECTION_v2': (
            'v2 said the legacy statistic is near zero at g6 "because the '
            'far-field response there is a slow oscillation whose 350 s signed '
            'mean nearly cancels". That is false at g6. The legacy statistic is '
            'an UNBASELINED signed mean, and at g6 the channel sits on a DC '
            'strain-rate pedestal of %.1f counts (neighbours %.0f to %.0f) '
            'against a signal of +%.1f: the cancellation is the pedestal, not an '
            'oscillation. Removing the pedestal takes g6 from %.3f%% to %+.2f%% '
            'of the peak (9-channel median %+.2f%%), i.e. to or above the 5%% '
            'line, and moves the legacy front from MD %.2f to MD %.2f. The '
            'cancellation explanation holds only at g7 (%+.3f%%).'),
    }
    legacy['MECHANISM_CORRECTION_v2'] = legacy['MECHANISM_CORRECTION_v2'] % (
        baseR[i6], min(baseR[i6 - 4:i6 + 5]), max(baseR[i6 - 4:i6 + 5]), a_dc[i6],
        100 * a_leg[i6] / pk, 100 * a_dc[i6] / pk_dc, 100 * a_dc_m9[i6] / pk_dc,
        legacy['front_shallowest_md_ft'],
        legacy['dc_removed']['front_shallowest_md_ft'], 100 * a_dc[i7] / pk_dc)
    log(f"legacy front MD {legacy['front_shallowest_md_ft']:.2f} (published "
        f"{lg['published_value_md_ft']:.0f}); DC-removed front MD "
        f"{legacy['dc_removed']['front_shallowest_md_ft']:.2f}; g6 raw "
        f"{100*a_leg[i6]/pk:.3f}% -> dc {100*a_dc[i6]/pk_dc:+.2f}% "
        f"(med9 {100*a_dc_m9[i6]/pk_dc:+.2f}%); g7 raw {100*a_leg[i7]/pk:.3f}% "
        f"-> dc {100*a_dc[i7]/pk_dc:+.3f}%")

    # ---- 6. front versus time, BOTH definitions ---------------------------
    Yfull = AF - PF.mean(axis=1)[:, None]
    Yfull[:, stripe_mask] = np.nan
    Yl = boxcar_nanmean(Yfull, L)
    centres = np.arange(binlen // 2, tw.size - binlen // 2, stride)
    fronts_c = {f"{N:g}": [] for N in n_grid}
    fronts_s = {f"{N:g}": [] for N in n_grid}
    fronts_l = {f"{N:g}": [] for N in n_grid}
    frac_t = {f"{N:g}": [] for N in n_grid}
    frac_t_r1 = {f"{N:g}": [] for N in n_grid}
    anchor_det = {f"{N:g}": [] for N in n_grid}
    tcent = []
    for c in centres:
        sl = slice(c - binlen // 2, c + binlen // 2 + 1)
        s_t = np.sqrt(np.nanmean(Yl[:, sl] ** 2, axis=1)) / sig
        s_t = median_filter(np.nan_to_num(s_t, nan=0.0), size=medch, mode='nearest')
        tcent.append(float(tw[c]))
        for N in n_grid:
            det = s_t > N
            fronts_c[f"{N:g}"].append(front_connected(det, mdF, anchor))
            fronts_s[f"{N:g}"].append(front_shallowest(det, mdF))
            fronts_l[f"{N:g}"].append(front_longest_run(det, mdF)[0])
            frac_t[f"{N:g}"].append(float(det.mean()))
            frac_t_r1[f"{N:g}"].append(float(det[iR:].mean()))
            anchor_det[f"{N:g}"].append(bool(_closed(det)[anchor]))
    tcent = np.asarray(tcent)

    fvt = {}
    for N in n_grid:
        k = f"{N:g}"
        cser = [x for x in fronts_c[k] if x is not None]
        sser = [x for x in fronts_s[k] if x is not None]
        lser = [x for x in fronts_l[k] if x is not None]
        # is the shallowest-detected series monotone non-increasing after its min?
        fvt[f"N={N:g}"] = {
            'connected_first_defined_t_s': next(
                (float(tcent[j]) for j in range(len(tcent))
                 if fronts_c[k][j] is not None), None),
            'connected_shallowest_md_ft': min(cser) if cser else None,
            'connected_final_md_ft': fronts_c[k][-1],
            'connected_n_bins_undefined': int(sum(1 for x in fronts_c[k]
                                                  if x is None)),
            'longest_run_first_md_ft': lser[0] if lser else None,
            'longest_run_min_md_ft': min(lser) if lser else None,
            'longest_run_final_md_ft': fronts_l[k][-1],
            'longest_run_retreat_ft': (float(fronts_l[k][-1] - min(lser))
                                       if lser and fronts_l[k][-1] is not None
                                       else None),
            'shallowest_first_md_ft': sser[0] if sser else None,
            'shallowest_min_md_ft': min(sser) if sser else None,
            'shallowest_final_md_ft': fronts_s[k][-1],
            'shallowest_retreat_ft': (float(fronts_s[k][-1] - min(sser))
                                      if sser and fronts_s[k][-1] is not None
                                      else None),
            'max_fraction_detected': float(max(frac_t[k])),
            'fraction_detected_at_t460': float(
                frac_t[k][int(np.argmin(np.abs(tcent - 460.0)))]),
        }
    # diagnosis of the v2 "retreat"
    j460 = int(np.argmin(np.abs(tcent - 460.0)))
    j450 = int(np.argmin(np.abs(tcent - 450.0)))
    def _runs_at(j):
        sl_ = slice(centres[j] - binlen // 2, centres[j] + binlen // 2 + 1)
        s_ = median_filter(np.nan_to_num(np.sqrt(np.nanmean(Yl[:, sl_] ** 2, axis=1))
                                         / sig, nan=0.0), size=medch, mode='nearest')
        return s_, ["%.0f-%.0f" % (mdF[a], mdF[b]) for a, b in
                    contiguous_runs(np.where(_closed(s_ > 2.0))[0])]
    _, runs450 = _runs_at(j450)
    s460, runs460 = _runs_at(j460)
    retreat = {
        't_before_s': float(tcent[j450]), 't_after_s': float(tcent[j460]),
        'connected_front_before_md_ft': fronts_c['2'][j450],
        'connected_front_after_md_ft': fronts_c['2'][j460],
        'shallowest_front_before_md_ft': fronts_s['2'][j450],
        'shallowest_front_after_md_ft': fronts_s['2'][j460],
        'longest_run_front_before_md_ft': fronts_l['2'][j450],
        'longest_run_front_after_md_ft': fronts_l['2'][j460],
        'fraction_detected_before_full_fibre': frac_t['2'][j450],
        'fraction_detected_after_full_fibre': frac_t['2'][j460],
        'fraction_detected_before_R1_window': frac_t_r1['2'][j450],
        'fraction_detected_after_R1_window': frac_t_r1['2'][j460],
        'detected_runs_at_t_before_N2': runs450[:12],
        'detected_runs_at_t_after_N2': runs460[:12],
        'snr_at_the_9_channels_below_the_anchor': [
            float(x) for x in s460[anchor - 8:anchor + 1]],
        'CLAIM_CORRECTION_v2': (
            'v2 wrote "the front moves and then retreats". It does not. Across '
            't = %.0f -> %.0f s, %.1f%% of the R1-window channels are still '
            'detected (was %.1f%%) and the LONGEST detected run still starts at '
            'MD %.0f (was MD %.0f); what changed is that ~5 channels immediately '
            'below the frac-hit anchor fell under 2 sigma and severed the ANCHOR '
            'from that run, so the anchored definition alone jumps from MD %.0f '
            'to MD %.0f. The anchor-free longest-run front at N=2 runs %.0f -> '
            '%.0f -> %.0f ft over the whole window with a net retreat of %.0f ft. '
            'The physical point survives in a different form and is worth '
            'keeping: near-source channels drop below their OWN noise as dP/dt '
            'decays while the far-field detections persist, because LF-DAS '
            'measures a rate.'
            % (tcent[j450], tcent[j460], 100 * frac_t_r1['2'][j460],
               100 * frac_t_r1['2'][j450], fronts_l['2'][j460],
               fronts_l['2'][j450], fronts_c['2'][j450], fronts_c['2'][j460],
               fronts_l['2'][0], min(x for x in fronts_l['2'] if x is not None),
               fronts_l['2'][-1],
               fronts_l['2'][-1] - min(x for x in fronts_l['2'] if x is not None))),
    }
    log("retreat diagnosis: t=%.0f->%.0f s, R1-window frac detected %.3f->%.3f, "
        "anchored front %.1f -> %.1f, longest-run front %.1f -> %.1f"
        % (tcent[j450], tcent[j460], frac_t_r1['2'][j450], frac_t_r1['2'][j460],
           fronts_c['2'][j450] or -1, fronts_c['2'][j460] or -1,
           fronts_l['2'][j450] or -1, fronts_l['2'][j460] or -1))

    # ---- 7. forward model, far field extended to MD 13000 ------------------
    fm = cfg['forward_model']
    cmp_ = fm['comparison']
    fcfg = {'window': cfg['window'], 'data': cfg['data'],
            'source': {'selection_rule': fm['source']['selection_rule'],
                       'baseline_removal': fm['source']['baseline_removal']}}
    series, gnums, gmds, fh_arr, _, _ = runner.load_window_data(fcfg)
    src_gauge, fh_centroid = runner.pick_source_gauge(fcfg, series, fh_arr)
    src = series[src_gauge]

    # absolute time of the source gauge's first in-window sample, so the model
    # can be put on the DAS clock rather than on a co-rebased one
    zg = np.load(cfg['data']['gauge_series_template'].format(n=src_gauge),
                 allow_pickle=True)
    stg = zg['start_time'].item()
    tg = zg['taxis'].astype(float)
    gm = (tg >= (t0 - stg).total_seconds()) & (tg <= (t1 - stg).total_seconds())
    src_t0_abs = stg + datetime.timedelta(seconds=float(tg[gm][0]))
    das_minus_gauge_s = (st + datetime.timedelta(seconds=float(t[tm[0]]))
                         - src_t0_abs).total_seconds()

    dmC = np.where((da >= cmp_['valid_md_range_ft'][0])
                   & (da <= w['md_max_ft']))[0]
    mdC = da[dmC]
    AC = full[dmC[0]:dmC[-1] + 1, tm[0]:tm[-1] + 1].astype(np.float64)
    PC = full[dmC[0]:dmC[-1] + 1, pm[0]:pm[-1] + 1].astype(np.float64)
    snrC, snrC_m, S_C, sig_C = snr_block(AC, PC, stripe_mask, L, medch)
    stripe_amp = np.sqrt(np.mean(AC[:, stripe_mask] ** 2, axis=1))

    dx = float(fm['mesh']['dx_ft'])
    dt_s = float(fm['solver']['dt_s'])
    t_total = float(src['taxis'][-1])
    h = L // 2

    def run_models(pad_lo):
        mesh = np.arange(w['md_min_ft'] - pad_lo,
                         w['md_max_ft'] + fm['mesh']['domain_pad_high_md_ft']
                         + dx / 2.0, dx)
        si = int(np.argmin(np.abs(mesh - src['md_ft'])))
        rec_idx = np.array([int(np.argmin(np.abs(mesh - m))) for m in mdC])
        g_idx = np.array([int(np.argmin(np.abs(mesh - series[n]['md_ft'])))
                          for n in sorted(series)])
        out = {}
        for name, spec in fm['profiles'].items():
            fam = core.PROFILE_FAMILIES[spec['family']]
            prof = fam['fn'](mesh, si, np.asarray(spec['params'], float))
            ts, rec = rcore.solve_forward(mesh, prof, dt_s, t_total,
                                          src['taxis'], src['delta_psi'], si,
                                          record_idx=rec_idx, theta=1.0,
                                          interface_avg='harmonic')
            _, recg = rcore.solve_forward(mesh, prof, dt_s, t_total,
                                          src['taxis'], src['delta_psi'], si,
                                          record_idx=g_idx, theta=1.0,
                                          interface_avg='harmonic')
            sq, n, mse_each = 0.0, 0, []
            for k, gn in enumerate(sorted(series)):
                if gn == src_gauge:
                    continue
                obs = series[gn]
                r = np.interp(obs['taxis'], ts, recg[:, k]) - obs['delta_psi']
                sq += float(np.sum(r ** 2))
                n += r.size
                mse_each.append(float(np.mean(r ** 2)))
            dpdt = np.gradient(rec, ts, axis=0)
            amps = {}
            for tag, off in (('abs', das_minus_gauge_s), ('v2_rebased', 0.0)):
                on_das = np.stack([np.interp(tw + off, ts, dpdt[:, k])
                                   for k in range(rec.shape[1])])
                on_das[:, stripe_mask] = np.nan
                amps[tag] = np.sqrt(np.nanmean(
                    boxcar_nanmean(on_das, L)[:, h:-h] ** 2, axis=1))
            out[name] = {'profile': spec,
                         'rmse_pooled_psi': float(np.sqrt(sq / n)),
                         'rmse_gaugemean_psi': float(np.sqrt(np.mean(mse_each))),
                         'amp': amps['abs'], 'amp_v2_align': amps['v2_rebased'],
                         'n_steps': int(ts.size - 1)}
        out['_mesh'] = mesh
        out['_source_idx'] = si
        return out

    pad_lo = float(fm['mesh']['domain_pad_low_md_ft'])
    sims = run_models(pad_lo)
    mesh = sims.pop('_mesh')
    source_idx = sims.pop('_source_idx')
    sims_alt = run_models(float(fm['mesh']['pad_check_alternative_low_md_ft']))
    sims_alt.pop('_mesh')
    sims_alt.pop('_source_idx')
    valid = (mdC >= cmp_['valid_md_range_ft'][0]) & (mdC <= cmp_['valid_md_range_ft'][1])
    pad_check = {name: float(np.nanmax(np.abs(sims[name]['amp'][valid]
                                              / sims_alt[name]['amp'][valid] - 1.0)))
                 for name in sims}
    align_check = {name: float(np.nanmax(np.abs(sims[name]['amp'][valid]
                                                / sims[name]['amp_v2_align'][valid] - 1.0)))
                   for name in sims}
    log("pad sensitivity (MD %g pad vs %g pad), max |rel diff| in amplitude: %s"
        % (pad_lo, fm['mesh']['pad_check_alternative_low_md_ft'],
           {k: round(v, 6) for k, v in pad_check.items()}))
    log("time-alignment sensitivity (absolute vs v2's co-rebased clock, "
        "offset %.3f s): %s" % (das_minus_gauge_s,
                                {k: round(v, 6) for k, v in align_check.items()}))
    for name, s in sims.items():
        log(f"  {name}: gauge-mean RMSE {s['rmse_gaugemean_psi']:.3f} psi, "
            f"pooled {s['rmse_pooled_psi']:.3f} psi")

    # ---- 8. shape comparison with THREE axes of sensitivity ---------------
    mch = int(cmp_['spatial_median_channels'])
    dist = src['md_ft'] - mdC
    S_sig = np.sqrt(np.maximum(S_C ** 2 - sig_C ** 2, 0.0))
    n_clipped = int((S_sig <= 0).sum())
    das_s = median_filter(S_sig, size=mch, mode='nearest')
    das_pn_s = median_filter(np.where(sig_C > 0, S_sig / sig_C, 0.0),
                             size=mch, mode='nearest')
    das_pa_s = median_filter(np.where(stripe_amp > 0, S_sig / stripe_amp, 0.0),
                             size=mch, mode='nearest')
    sig_s = median_filter(sig_C, size=mch, mode='nearest')
    art_s = median_filter(stripe_amp, size=mch, mode='nearest')
    for name, s in sims.items():
        s['amp_s'] = median_filter(s['amp'], size=mch, mode='nearest')

    ref_bands = [tuple(b) for b in cmp_['reference_band_sensitivity_md_ft']]
    prim = tuple(cmp_['reference_band_md_ft'])

    def ratios_for(rb_lo, rb_hi):
        rb = (mdC >= rb_lo) & (mdC <= rb_hi)
        d = {'raw': das_s / np.median(das_s[rb]),
             'per_noise': das_pn_s / np.median(das_pn_s[rb]),
             'per_artifact': das_pa_s / np.median(das_pa_s[rb])}
        m = {name: s['amp_s'] / np.median(s['amp_s'][rb])
             for name, s in sims.items()}
        return d, m

    norms, mod = ratios_for(*prim)

    def band_row(blo, bhi, dsets, msets, label=None):
        m_ = valid & (dist >= blo) & (dist <= bhi)
        if m_.sum() < 5:
            return None
        row = {'band_lo_ft': float(blo), 'band_hi_ft': float(bhi),
               'label': label or f"{blo:g}-{bhi:g}",
               'n_channels': int(m_.sum()),
               'md_lo_ft': float(mdC[m_].min()), 'md_hi_ft': float(mdC[m_].max()),
               'n_eff_after_median15': float(m_.sum() / mch),
               'das_median_snr9': float(np.median(snrC_m[m_])),
               'n_channels_noise_clipped_to_zero': int((S_sig[m_] <= 0).sum()),
               'noise_floor_median_native_units': float(np.median(sig_s[m_])),
               'artifact_amp_median_native_units': float(np.median(art_s[m_]))}
        for nn, arr in dsets.items():
            row[f'das_ratio_{nn}'] = float(np.median(arr[m_]))
        for name in msets:
            row[f'model_{name}_ratio_median'] = float(np.median(msets[name][m_]))
            for nn, arr in dsets.items():
                with np.errstate(divide='ignore', invalid='ignore'):
                    q = msets[name][m_] / arr[m_]
                q = q[np.isfinite(q)]
                row[f'model_over_das_{name}_{nn}'] = float(np.median(q)) if q.size else None
                if nn in ('raw', 'per_noise'):
                    row[f'model_over_das_{name}_{nn}_q25'] = (
                        float(np.percentile(q, 25)) if q.size else None)
                    row[f'model_over_das_{name}_{nn}_q75'] = (
                        float(np.percentile(q, 75)) if q.size else None)
        return row

    bands = []
    for blo, bhi in cmp_['distance_bands_ft']:
        r = band_row(blo, bhi, norms, mod)
        if r is not None:
            bands.append(r)
    split_rows = []
    for blo, bhi in cmp_['split_bands_ft']:
        r = band_row(blo, bhi, norms, mod,
                     label=f"SPLIT {blo:g}-{bhi:g}")
        if r is not None:
            split_rows.append(r)

    # reference-band sensitivity: the THIRD axis
    refsens = []
    for rb_lo, rb_hi in ref_bands:
        d2, m2 = ratios_for(rb_lo, rb_hi)
        row = {'ref_band_md_ft': [float(rb_lo), float(rb_hi)],
               'is_primary': bool((rb_lo, rb_hi) == prim), 'bands': []}
        for blo, bhi in list(cmp_['distance_bands_ft']) + list(cmp_['split_bands_ft']):
            r = band_row(blo, bhi, d2, m2)
            if r is not None:
                row['bands'].append({k: r[k] for k in r
                                     if k.startswith('model_over_das_')
                                     and not k.endswith(('_q25', '_q75'))}
                                    | {'band_lo_ft': r['band_lo_ft'],
                                       'band_hi_ft': r['band_hi_ft']})
        refsens.append(row)

    # the full three-axis envelope per band per model
    envelope = []
    for blo, bhi in list(cmp_['distance_bands_ft']) + list(cmp_['split_bands_ft']):
        e = {'band_lo_ft': float(blo), 'band_hi_ft': float(bhi)}
        for name in sims:
            vals = []
            for row in refsens:
                for b in row['bands']:
                    if b['band_lo_ft'] == blo and b['band_hi_ft'] == bhi:
                        for nn in ('raw', 'per_noise'):
                            v = b.get(f'model_over_das_{name}_{nn}')
                            if v is not None:
                                vals.append(v)
            if vals:
                e[f'{name}_min'] = float(min(vals))
                e[f'{name}_max'] = float(max(vals))
                e[f'{name}_sign_consistent'] = bool(all(v > 1.0 for v in vals)
                                                    or all(v < 1.0 for v in vals))
                e[f'{name}_n_variants'] = len(vals)
        envelope.append(e)
        log("  band %4.0f-%4.0f ft (MD %.0f-%.0f): " % (
            blo, bhi, (src['md_ft'] - bhi), (src['md_ft'] - blo))
            + '  '.join("%s [%.2f,%.2f]%s" % (n, e.get(f'{n}_min', np.nan),
                                              e.get(f'{n}_max', np.nan),
                                              '*' if e.get(f'{n}_sign_consistent')
                                              else '')
                        for n in sims))

    resp = {
        'shallow_band_md_ft': [15003.0, 15150.0],
        'deep_band_md_ft': [16400.0, 16700.0],
        'noise_floor_deep_over_shallow': float(
            np.median(sig_s[(mdC >= 16400) & (mdC <= 16700)])
            / np.median(sig_s[(mdC >= 15003) & (mdC <= 15150)])),
        'artifact_amplitude_deep_over_shallow': float(
            np.median(art_s[(mdC >= 16400) & (mdC <= 16700)])
            / np.median(art_s[(mdC >= 15003) & (mdC <= 15150)])),
        'artifact_level_histogram_top6_kcounts': (
            lambda lv, ct: [[float(lv[i] * 1000.0), int(ct[i])]
                            for i in np.argsort(ct)[::-1][:6]])(
            *np.unique(np.round(art_s / 1000.0), return_counts=True)),
        'artifact_level_note': ('15-channel-median artifact amplitude rounded to '
                                '1000 counts: it clusters on a few discrete '
                                'levels rather than varying smoothly with MD, '
                                'i.e. it behaves like an acquisition gain regime '
                                'and cannot be read as a smooth channel-response '
                                'curve.'),
        'artifact_per_channel_amplitude_decades':
            float(np.log10(stripe_amp.max() / max(stripe_amp.min(), 1e-9))),
        'interpretation': (
            'Two independent probes of the depth-dependent channel response '
            'disagree, and the artifact proxy takes DISCRETE steps (~2.2k, 13.4k, '
            '24.5k, 35.7k counts) rather than varying smoothly, i.e. it behaves '
            'like a gain regime. Neither is a usable calibration, so amplitude '
            'ratios are reported under two normalisations and their disagreement '
            'is part of the result.'),
    }

    # sub-band diagnostic for the v2 headline
    inner = valid & (dist >= 1400) & (dist < 1520)
    outer = valid & (dist >= 1520) & (dist <= 1645)
    edge_diag = {
        'inner_half_md_ft': [float(mdC[inner].min()), float(mdC[inner].max())],
        'outer_half_md_ft': [float(mdC[outer].min()), float(mdC[outer].max())],
        'n_inner': int(inner.sum()), 'n_outer': int(outer.sum()),
        'artifact_ratio_outer_over_inner': float(np.median(art_s[outer])
                                                 / np.median(art_s[inner])),
        'noise_floor_ratio_outer_over_inner': float(np.median(sig_s[outer])
                                                    / np.median(sig_s[inner])),
        'note': ("v2's headline (uniform D=1150 over-predicts the farthest 250 ft "
                 "by 1.5-1.8x) is carried entirely by the OUTER half of that "
                 "band, where both channel-response proxies step DOWN by about a "
                 "factor of two relative to their immediate neighbours. The inner "
                 "half shows no over-prediction under the raw norm."),
    }

    decay = {
        'reference_band_md_ft': [float(prim[0]), float(prim[1])],
        'reference_band_sensitivity': refsens,
        'three_axis_envelope': envelope,
        'spatial_median_channels': mch,
        'valid_md_range_ft': cmp_['valid_md_range_ft'],
        'n_channels_compared': int(valid.sum()),
        'n_channels_noise_clipped_to_zero': n_clipped,
        'noise_subtraction': 'sqrt(window RMS^2 - noise RMS^2) per channel',
        'pad_sensitivity_max_rel_diff': pad_check,
        'time_alignment_offset_s': float(das_minus_gauge_s),
        'time_alignment_max_rel_diff_vs_v2': align_check,
        'instrument_response': resp,
        'edge_subband_diagnostic': edge_diag,
        'defensible_bracket': ('raw and per_noise x five reference bands. '
                               'per_artifact is a diagnostic only.'),
    }

    # ---- outputs ----------------------------------------------------------
    out_decls = []

    def w_csv(name, header, rows, note=None):
        p = os.path.join(outdir, name)
        with open(p, 'w') as fh:
            fh.write(','.join(header) + '\n')
            for r in rows:
                fh.write(','.join(
                    '' if r.get(k) is None else
                    (f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]))
                    for k in header) + '\n')
        out_decls.append(rman.output_decl(p, role='csv', note=note))
        return p

    w_csv(f'c5_front_vs_threshold_{VERSION}.csv',
          ['stripe', 'snr_variant', 'N', 'front_connected_md_ft',
           'front_connected_at_domain_edge', 'front_longest_run_md_ft',
           'longest_run_n_channels', 'front_shallowest_md_ft',
           'shallowest_run_n_channels', 'front_shallowest_at_domain_edge',
           'n_undetected_below_front',
           'n_channels_detected', 'fraction_detected'], all_rows,
          note='front vs N over the full fibre (MD 11000-16750) plus the '
               'R1-window-only sweep that shows what v2 truncation cost')

    w_csv(f'c5_channel_snr_{VERSION}.csv',
          ['md_ft', 'distance_from_source_ft', 'sigma_pre_native_units',
           'window_rms_native_units', 'snr', 'snr_median9',
           'snr_stripe_included', 'snr_no_lowpass', 'snr_alt_noise_window'],
          [{'md_ft': float(mdF[i]),
            'distance_from_source_ft': float(16645.0 - mdF[i]),
            'sigma_pre_native_units': float(sig[i]),
            'window_rms_native_units': float(S_win[i]),
            'snr': float(snr[i]), 'snr_median9': float(snr_m[i]),
            'snr_stripe_included': float(snr_i[i]),
            'snr_no_lowpass': float(snr_r[i]),
            'snr_alt_noise_window': float(snr_a[i])}
           for i in range(mdF.size)],
          note='per-channel detection statistic over the full fibre')

    w_csv(f'c5_legacy_statistic_{VERSION}.csv',
          ['md_ft', 'legacy_statistic_raw', 'legacy_over_peak_raw',
           'pre_pumping_mean', 'legacy_statistic_dc_removed',
           'legacy_over_peak_dc_removed'],
          [{'md_ft': float(mdR[i]), 'legacy_statistic_raw': float(a_leg[i]),
            'legacy_over_peak_raw': float(a_leg[i] / pk),
            'pre_pumping_mean': float(baseR[i]),
            'legacy_statistic_dc_removed': float(a_dc[i]),
            'legacy_over_peak_dc_removed': float(a_dc[i] / pk_dc)}
           for i in range(mdR.size)],
          note='the legacy 5%-of-global-peak rule, raw (as published, about '
               'zero) and with the per-channel DC pedestal removed')

    rows_t = []
    for j, tc in enumerate(tcent):
        r = {'t_win_s': float(tc)}
        for N in n_grid:
            k = f"{N:g}"
            r[f'front_connected_N{k}_md_ft'] = fronts_c[k][j]
            r[f'front_longest_run_N{k}_md_ft'] = fronts_l[k][j]
            r[f'front_shallowest_N{k}_md_ft'] = fronts_s[k][j]
            r[f'fraction_detected_N{k}'] = frac_t[k][j]
            r[f'fraction_detected_R1_window_N{k}'] = frac_t_r1[k][j]
            r[f'anchor_detected_N{k}'] = anchor_det[k][j]
        rows_t.append(r)
    hdr_t = ['t_win_s']
    for N in n_grid:
        k = f"{N:g}"
        hdr_t += [f'front_connected_N{k}_md_ft', f'front_longest_run_N{k}_md_ft',
                  f'front_shallowest_N{k}_md_ft', f'fraction_detected_N{k}',
                  f'fraction_detected_R1_window_N{k}', f'anchor_detected_N{k}']
    w_csv(f'c5_front_vs_time_{VERSION}.csv', hdr_t, rows_t,
          note='both front definitions plus the detected fraction and whether '
               'the frac-hit anchor is itself detected in each bin')

    dec_rows = []
    for i in range(mdC.size):
        if not valid[i]:
            continue
        r = {'md_ft': float(mdC[i]), 'distance_from_source_ft': float(dist[i]),
             'das_amp_window_rms_native_units': float(S_C[i]),
             'das_amp_noise_subtracted_native_units': float(S_sig[i]),
             'das_amp_median15_native_units': float(das_s[i]),
             'das_snr_median9': float(snrC_m[i]),
             'das_ratio_raw': float(norms['raw'][i]),
             'das_ratio_per_noise': float(norms['per_noise'][i]),
             'das_ratio_per_artifact': float(norms['per_artifact'][i]),
             'channel_noise_floor_native_units': float(sig_C[i]),
             'artifact_amplitude_native_units': float(stripe_amp[i])}
        for name in sims:
            r[f'model_{name}_ratio_to_ref'] = float(mod[name][i])
            for nn in ('raw', 'per_noise', 'per_artifact'):
                v = norms[nn][i]
                r[f'model_over_das_{name}_{nn}'] = (float(mod[name][i] / v)
                                                    if v > 0 else None)
        dec_rows.append(r)
    hdrd = ['md_ft', 'distance_from_source_ft', 'das_amp_window_rms_native_units',
            'das_amp_noise_subtracted_native_units', 'das_amp_median15_native_units',
            'das_snr_median9', 'das_ratio_raw', 'das_ratio_per_noise',
            'das_ratio_per_artifact', 'channel_noise_floor_native_units',
            'artifact_amplitude_native_units']
    for name in sims:
        hdrd += [f'model_{name}_ratio_to_ref']
        hdrd += [f'model_over_das_{name}_{nn}'
                 for nn in ('raw', 'per_noise', 'per_artifact')]
    w_csv(f'c5_farfield_decay_{VERSION}.csv', hdrd, dec_rows,
          note='per-channel far-field comparison, MD 13000-16645')

    bhdr = ['label', 'band_lo_ft', 'band_hi_ft', 'md_lo_ft', 'md_hi_ft',
            'n_channels', 'n_eff_after_median15', 'das_median_snr9',
            'n_channels_noise_clipped_to_zero', 'noise_floor_median_native_units',
            'artifact_amp_median_native_units',
            'das_ratio_raw', 'das_ratio_per_noise', 'das_ratio_per_artifact']
    for name in sims:
        bhdr += [f'model_{name}_ratio_median']
        for nn in ('raw', 'per_noise'):
            bhdr += [f'model_over_das_{name}_{nn}',
                     f'model_over_das_{name}_{nn}_q25',
                     f'model_over_das_{name}_{nn}_q75']
        bhdr += [f'model_over_das_{name}_per_artifact']
    for e in envelope:
        for b in bands + split_rows:
            if b['band_lo_ft'] == e['band_lo_ft'] and b['band_hi_ft'] == e['band_hi_ft']:
                for name in sims:
                    b[f'{name}_envelope_min'] = e.get(f'{name}_min')
                    b[f'{name}_envelope_max'] = e.get(f'{name}_max')
                    b[f'{name}_sign_consistent'] = e.get(f'{name}_sign_consistent')
    for name in sims:
        bhdr += [f'{name}_envelope_min', f'{name}_envelope_max',
                 f'{name}_sign_consistent']
    w_csv(f'c5_farfield_decay_bands_{VERSION}.csv', bhdr, bands + split_rows,
          note='band table with IQR, n_eff, DAS SNR, response proxies and the '
               'three-axis (raw/per_noise x 5 reference bands) envelope; the '
               'last two rows split the v2 headline band in half')

    w_csv(f'c5_control_bands_{VERSION}.csv',
          ['md_lo_ft', 'md_hi_ft', 'n_channels', 'median_snr_median9']
          + [f'frac_median9_gt_{N:g}' for N in n_grid]
          + [f'frac_raw_gt_{N:g}' for N in n_grid],
          controls + [dict(signal_band, **{f'frac_raw_gt_{N:g}': None
                                           for N in n_grid})],
          note='false-alarm calibration with the ACTUAL detection statistic on '
               'fibre bands that cannot carry a stage-1 reservoir signal; the '
               'last row is the R1 window itself, for contrast')

    w_csv(f'c5_stripe_detection_{VERSION}.csv',
          ['t_win_s', 'reference_band_exceedance_fraction', 'excluded'],
          [{'t_win_s': float(tw[j]),
            'reference_band_exceedance_fraction': float(f_win[j]),
            'excluded': bool(stripe_mask[j])} for j in range(tw.size)],
          note='per-sample downlead exceedance fraction and the exclusion mask')

    # ---- figures ----------------------------------------------------------
    gmd_all = np.load(cfg['data']['gauge_md_npz'], allow_pickle=True)['data'].astype(float)
    falsifiers = []
    for n in cfg['data']['falsifier_gauges']:
        zz = np.load(cfg['data']['gauge_series_template'].format(n=n),
                     allow_pickle=True)
        s2 = zz['start_time'].item()
        ta = zz['taxis'].astype(float)
        dv = zz['data'].astype(float)
        mk = (ta >= (t0 - s2).total_seconds()) & (ta <= (t1 - s2).total_seconds())
        d = dv[mk] - dv[mk][0]
        falsifiers.append({'gauge': int(n), 'md_ft': float(gmd_all[n - 1]),
                           'rise_psi': float(d.max()),
                           'final_psi': float(d[-1])})
    log('falsifier gauges: ' + ', '.join("g%d MD %.0f %.1f psi"
                                         % (f['gauge'], f['md_ft'], f['rise_psi'])
                                         for f in falsifiers))

    p1 = os.path.join(outdir, f'fig01_front_vs_threshold_{VERSION}.png')
    fig_front(p1, mdF, snr, snr_m, snr_i, n_grid, rows_excl, rows_incl, rows_r1,
              legacy, tcent, fronts_c, fronts_s, fronts_l, falsifiers, controls)
    out_decls.append(rman.output_decl(p1, role='figure_png', dpi=300,
                                      note='(a) SNR over the full fibre; '
                                           '(b) front vs N, no censoring; '
                                           '(c) both front definitions vs time'))

    p2 = os.path.join(outdir, f'fig02_artifact_detection_{VERSION}.png')
    fig_stripe(p2, AF, tw, mdF, f_win, f_pre, stripe_mask, stripe_info,
               rows_excl, rows_incl, n_grid)
    out_decls.append(rman.output_decl(p2, role='figure_png', dpi=300,
                                      note='artifact detection, multi-band '
                                           'exceedance and front sensitivity'))

    p3 = os.path.join(outdir, f'fig03_farfield_decay_{VERSION}.png')
    fig_decay(p3, mdC, dist, valid, norms, mod, sims, cmp_, refsens, envelope,
              sig_s, art_s, snrC_m, S_sig, das_s, prim, falsifiers)
    out_decls.append(rman.output_decl(p3, role='figure_png', dpi=300,
                                      note='far-field shape comparison to '
                                           'MD 13000 with the three-axis envelope'))

    # ---- manifest, via the shared writer ----------------------------------
    results = {
        'amendment': {
            'defect_list': 'output/rev2_20260901/A4/challenge_defects/C5_defects.json',
            'supersedes': 'output/rev2_20260901/C5/manifest.json (v2 run)',
            'note': 'Every v2 output is retained unmodified; v3 products are in '
                    'output/rev2_20260901/C5/v3/.',
        },
        'das_verification': ver,
        'pre_pumping_window': pre_info,
        'stripe_detection': stripe_info,
        'noise_floor': {
            'sigma_native_units': {
                'min': float(sig.min()), 'p10': float(np.percentile(sig, 10)),
                'median': float(np.median(sig)),
                'p90': float(np.percentile(sig, 90)), 'max': float(sig.max())},
            'n_channels_sigma_gt_5x_median': int((sig > 5 * np.median(sig)).sum()),
            'sigma_native_units_R1_window_only': sigma_r1,
            'snr_at_gauge_depths': gauge_snr,
            'false_alarm_control_bands': controls,
            'signal_band_for_contrast': signal_band,
            'split_half_null': split_half,
            'quiet_block_stationarity': stationarity,
            'binned_null_RETAINED_BUT_INVALID': binned_null,
        },
        'legacy_rule': legacy,
        'front_vs_threshold': all_rows,
        'front_vs_time_summary': fvt,
        'retreat_diagnosis': retreat,
        'stripe_sensitivity': {
            f"N={N:g}": {
                'front_connected_excluded': next(
                    r['front_connected_md_ft'] for r in rows_excl
                    if r['N'] == N and r['snr_variant'] == 'median9'),
                'front_connected_included': next(
                    r['front_connected_md_ft'] for r in rows_incl
                    if r['N'] == N and r['snr_variant'] == 'median9'),
            } for N in n_grid},
        'falsifier_gauges': falsifiers,
        'forward_model': {
            'source_gauge': int(src_gauge),
            'frac_hit_centroid_md_ft': float(fh_centroid),
            'profiles': {n: {'spec': s['profile'],
                             'gauge_mean_rmse_psi': s['rmse_gaugemean_psi'],
                             'pooled_rmse_psi_vs_6_target_gauges':
                                 s['rmse_pooled_psi']}
                         for n, s in sims.items()},
            'r1_cross_check': (
                "BOTH published headline numbers are the GAUGE-MEAN RMSE returned "
                "by r1_calibration_core.misfit_for_profile (see its docstring, "
                "r1_calibration_core.py:553-569). uniform_1150 gauge-mean = "
                "%.4f psi reproduces R1's /results/uniform/best_rmse = 82.3373; "
                "two_zone_r2 gauge-mean = %.4f psi reproduces r2's 11.872. The "
                "pooled-residual value %.4f psi is an extra diagnostic only and "
                "matches R1's /results/uniform/best_row/rmse_pooled_psi = "
                "78.1890; it has no published headline counterpart. The v2 "
                "manifest said the POOLED form should reproduce 82.33, which is "
                "backwards and would make an auditor conclude the cross-check "
                "failed."
                % (sims['uniform_1150']['rmse_gaugemean_psi'],
                   sims['two_zone_r2']['rmse_gaugemean_psi'],
                   sims['uniform_1150']['rmse_pooled_psi'])),
        },
        'farfield_decay': decay,
        'farfield_decay_bands': bands,
        'farfield_decay_split_bands': split_rows,
        'unit_policy': cfg['units'],
    }

    drv = rman.driver_record(kind='gauge_series',
                             baseline_removal=fm['source']['baseline_removal'],
                             value_units='delta_psi',
                             series_path=cfg['data']['gauge_series_template'].format(n=src_gauge),
                             gauge_number=int(src_gauge),
                             gauge_md_ft=float(src['md_ft']),
                             taxis=src['taxis'], values=src['delta_psi'],
                             time_start=w['time_start'], time_end=w['time_end'])
    srec = rman.source_record(mesh, md_requested_ft=float(src['md_ft']),
                              mesh_idx=int(source_idx), driver=drv,
                              label=f'g{src_gauge}', index_in_source_list=0)
    sp = rman.source_protocol(
        application='dirichlet_node',
        solver_class='rev2_core.solve_forward',
        placement_rule=fm['source']['selection_rule'],
        sources=[srec],
        targets=[{'gauge': int(gn), 'md_ft': float(series[gn]['md_ft'])}
                 for gn in sorted(series) if gn != src_gauge],
        time_level='n', phase_chaining=rman.NONE_DECLARED,
        boundary_conditions={'lbc': fm['solver']['lbc'],
                             'rbc': fm['solver']['rbc']})
    taxis_model = np.arange(0.0, t_total + dt_s / 2.0, dt_s)
    num = rman.numerics(
        time=rman.time_record(taxis_model, mode='fixed', theta=1.0,
                              t_total_requested_s=t_total, dt_requested_s=dt_s,
                              source_time_level='n'),
        mesh=rman.mesh_record(mesh, dx_requested_ft=dx,
                              window_md_ft=(w['md_min_ft'], w['md_max_ft']),
                              pad_low_ft=pad_lo,
                              pad_high_ft=float(fm['mesh']['domain_pad_high_md_ft'])),
        interface_avg='harmonic',
        boundary={'lbc': fm['solver']['lbc'], 'rbc': fm['solver']['rbc'],
                  'pml_thickness': 0.0, 'sigma_max': 0.0},
        diffusivity={'profile_families': {n: s['profile'] for n, s in sims.items()},
                     'baseline_D_ft2_s': 1150.0, 'profile_family': 'multiple',
                     'param_names': ['log10 params, see profile_families'],
                     'params': [], 'profile_anchor': 'physical_md',
                     'note': 'three fixed profiles, no fitting in this study'},
        barriers=rman.NONE_DECLARED, leakage=rman.NONE_DECLARED,
        kernel={'name': 'rev2_core.solve_forward', 'banded': True,
                'theta': 1.0, 'lambda_leak': 0.0,
                'equivalence_reference':
                    'bitwise identical to r1_calibration_core.solve_forward at '
                    'theta=1/harmonic/lambda=0 (rev2 self-test, 60/60)'},
        rng=rman.NONE_DECLARED, parallel=rman.NONE_DECLARED)

    inputs = [(cfg['data']['das_npz'], 'das', 'stage1_lfdas'),
              (cfg['data']['pump_rate_npz'], 'pumping', 'stage1_slurry_rate'),
              (cfg['data']['gauge_md_npz'], 'geometry', 'gauge_md'),
              (cfg['data']['frac_hit_stage1_npz'], 'geometry', 'frac_hit_stage1')]
    for n in sorted(set(list(series) + list(cfg['data']['falsifier_gauges']))):
        inputs.append((cfg['data']['gauge_series_template'].format(n=n),
                       'gauge_series', f'gauge{n}'))

    man_path = os.path.join(outdir, 'manifest.json')
    doc = rman.write_manifest(
        man_path, study_id=cfg['study_id'], task_id=TASK_ID, config=cfg,
        config_path=args.config, inputs=inputs, source=sp, numerics=num,
        outputs=out_decls, results=results, started_utc=started,
        run_label=f'C5 amend {VERSION}',
        require_modules=('rev2_core', 'rev2_manifest', 'rev2_data',
                         'r1_calibration_core'),
        notes=['C5 amend: fixes the defects reproduced in '
               'output/rev2_20260901/A4/challenge_defects/C5_defects.json.',
               'LF-DAS kept in native strain-rate counts; no psi<->strain '
               'coefficient used or fitted.'])
    log(f"manifest written via rev2_manifest.write_manifest: {man_path}")
    rep = rman.verify(man_path, repo_root=REPO)
    log(f"manifest verify: status={rep['status']}")
    if rep['status'] != 'clean':
        log(f"  VERIFY DETAIL: {json.dumps(rep)[:2000]}")
    return doc


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_front(path, md, snr, snr_m, snr_i, n_grid, rows_excl, rows_incl, rows_r1,
              legacy, tcent, fronts_c, fronts_s, fronts_l, falsifiers, controls):
    fig, ax = plt.subplots(3, 1, figsize=(11.0, 14.0))
    cols = {2.0: '#1b7837', 3.0: '#2166ac', 5.0: '#d95f02', 10.0: '#7b3294'}

    a = ax[0]
    a.semilogy(md, snr, color='0.78', lw=0.6, label='SNR, per channel')
    a.semilogy(md, snr_m, color='k', lw=1.3,
               label='SNR, 9-channel median (30 ft)')
    for N in n_grid:
        a.axhline(N, color=cols[N], ls='--', lw=1.0)
        a.text(md[-1], N, f' N={N:g}', color=cols[N], va='center', fontsize=8)
    for f in falsifiers:
        a.axvline(f['md_ft'], color='crimson', ls=':', lw=1.0)
        a.text(f['md_ft'], 1.02 * a.get_ylim()[0],
               " g%d %.0f psi" % (f['gauge'], f['rise_psi']), color='crimson',
               fontsize=7.5, rotation=90, va='bottom')
    a.axvline(legacy['front_shallowest_md_ft'], color='b', lw=1.4)
    a.text(legacy['front_shallowest_md_ft'], 0.115,
           ' legacy "front"\n MD %.0f' % legacy['front_shallowest_md_ft'],
           color='b', fontsize=8, va='bottom', ha='left')
    a.axvline(15000.0, color='0.35', ls='-.', lw=1.2)
    a.text(14930.0, 45.0, 'low edge of the R1 PRESSURE window. \n'
           'NOT an array limit: the fibre continues \n'
           'to MD %.0f (%d more channels). ' % (-548.97, 4672), color='0.2',
           fontsize=7.5, va='top', ha='right',
           bbox=dict(fc='w', ec='0.7', alpha=0.9))
    a.set_ylim(0.09, 130)
    a.set_xlabel('measured depth MD (ft)')
    a.set_ylabel('SNR = window RMS / pre-pumping RMS\n(per channel, native strain-rate units)')
    a.set_title('(a) Per-channel detection SNR over the FULL fibre (MD %.0f-%.0f, '
                '%d channels).\nDetection stops on its own near MD 14.7 kft, '
                '3.7 kft above the low end of the analysed range.'
                % (md[0], md[-1], md.size), fontsize=10)
    a.legend(fontsize=8, loc='upper left')
    a.grid(alpha=0.3, which='both')

    a = ax[1]
    xs = [r['N'] for r in rows_excl if r['snr_variant'] == 'median9']
    for key, lab, mk, c in [
            ('front_connected_md_ft', 'source-connected run (primary)', 'o', 'k'),
            ('front_shallowest_md_ft', 'shallowest detected channel (loosest)',
             's', '0.5')]:
        ys = [r[key] for r in rows_excl if r['snr_variant'] == 'median9']
        a.plot(xs, ys, mk + '-', color=c, label=lab)
    ys = [r['front_connected_md_ft'] for r in rows_r1
          if r['snr_variant'] == 'median9']
    a.plot(xs, ys, 'd:', color='#2166ac',
           label='source-connected, analysed over the R1 window only (what v2 did)')
    ys = [r['front_connected_md_ft'] for r in rows_incl
          if r['snr_variant'] == 'median9']
    a.plot(xs, ys, '^--', color='#d95f02',
           label='source-connected, acquisition stripe NOT excluded')
    a.axhline(legacy['front_shallowest_md_ft'], color='b', lw=1.4,
              label='legacy 5%%-of-global-peak rule (MD %.0f)'
                    % legacy['front_shallowest_md_ft'])
    a.axhline(15000.0, color='0.35', ls='-.', lw=1.2,
              label='low edge of the R1 pressure window (MD 15000) - NOT an array limit')
    for f in falsifiers:
        a.axhline(f['md_ft'], color='crimson', ls='-', lw=0.7, alpha=0.6)
        a.text(10.4, f['md_ft'], ' g%d: %.0f psi' % (f['gauge'], f['rise_psi']),
               color='crimson', fontsize=7, va='center')
    a.set_xscale('log')
    a.set_xticks(n_grid)
    a.set_xticklabels([f'{N:g}' for N in n_grid])
    a.set_xlim(1.8, 14.5)
    a.set_ylim(13480, 16400)
    for r in rows_excl:
        if r['snr_variant'] == 'median9' and r['shallowest_run_n_channels'] is not None:
            a.annotate('%d ch' % r['shallowest_run_n_channels'],
                       (r['N'], r['front_shallowest_md_ft']),
                       textcoords='offset points', xytext=(4, -10),
                       fontsize=6.5, color='0.35')
    a.text(1.9, 13520, 'the "shallowest detected" points at N = 2/3/5 are '
           'ISOLATED 2-25 channel patches near MD 13.9 kft,\ndisconnected from '
           'the 274-603 channel main run; at the control-band exceedance rates '
           'they are not fronts.',
           fontsize=6.8, color='0.35', va='bottom')
    a.set_xlabel('threshold multiplier N  (detection at N x per-channel pre-pumping RMS)')
    a.set_ylabel('up-hole edge of detection, MD (ft)')
    a.set_title('(b) The front is a function of the threshold. Nothing here is '
                'censored: at every N the\nfront has undetected fibre below it, '
                'while gauges further up-hole still record 17-36 psi.',
                fontsize=10)
    a.legend(fontsize=7.2, loc='lower right')
    a.grid(alpha=0.3)

    a = ax[2]
    for N in n_grid:
        k = f"{N:g}"
        a.plot(tcent, [np.nan if v is None else v for v in fronts_c[k]], '-',
               color=cols[N], lw=1.6, label=f'N={N:g}, anchored at the frac hit')
        a.plot(tcent, [np.nan if v is None else v for v in fronts_l[k]], '--',
               color=cols[N], lw=1.3, label=f'N={N:g}, longest detected run')
    a.axhline(legacy['front_shallowest_md_ft'], color='b', lw=1.0)
    a.text(tcent[0], legacy['front_shallowest_md_ft'], ' legacy MD 15632',
           color='b', fontsize=8, va='bottom')
    a.invert_yaxis()
    a.set_xlabel('time since 2020-03-16 11:24:00 (s)')
    a.set_ylabel('front MD (ft)')
    a.set_title('(c) There is no far-field retreat. The ANCHORED front (solid) '
                'jumps back after t ~ 450 s only because\n~5 channels BELOW the '
                'frac-hit anchor drop under 2 sigma and sever it from a run that '
                'still reaches\nMD 15120. The anchor-free longest-run front '
                '(dashed) does not retreat.', fontsize=9.5)
    a.legend(fontsize=6.8, loc='lower left', ncol=2)
    a.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def fig_stripe(path, D, tw, md, f_win, f_pre, mask, info, rows_excl, rows_incl,
               n_grid):
    fig = plt.figure(figsize=(11.0, 9.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.0, 1.0, 1.1], hspace=0.45,
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
    a.set_title('(a) LF-DAS waterfall over the full front window, native '
                'strain-rate units (no conversion).\nThe all-channel stripe near '
                't = %.0f s is the acquisition artifact; dashed lines bound the '
                'excluded window.' % info['peak_time_s_in_window'], fontsize=10)

    a = fig.add_subplot(gs[1, :])
    a.plot(tw, f_win, color='k', lw=1.0,
           label='surface downlead: fraction of channels exceeding %.0f x their '
                 'own pre-pumping RMS' % info['exceedance_sigma'])
    a.axhline(info['flag_fraction_threshold'], color='#d95f02', ls='--',
              label='flag threshold %.2f' % info['flag_fraction_threshold'])
    a.axhline(info['false_alarm_floor_pre_pumping_max_fraction'], color='g',
              ls=':', label='max over the quiet pre-pumping block (%.3f)'
                            % info['false_alarm_floor_pre_pumping_max_fraction'])
    a.fill_between(tw, 0, 1, where=mask, color='0.8', zorder=0)
    a.set_xlabel('time since 11:24:00 (s)')
    a.set_ylabel('exceedance fraction')
    a.set_ylim(0, 1.05)
    a.set_title('(b) Artifact detection on fibre ABOVE the wellhead (MD < 0), '
                'which cannot carry reservoir signal.', fontsize=9.5)
    a.legend(fontsize=7.5, loc='upper left')
    a.grid(alpha=0.3)

    a = fig.add_subplot(gs[2, 0])
    mb = info['multi_band_exceedance']
    y = np.arange(len(mb))
    a.barh(y, [b['max_exceedance_fraction_in_stripe_window'] for b in mb],
           color='#d95f02', label='max in the stripe window')
    a.barh(y, [b['pre_pumping_false_alarm_floor'] for b in mb], height=0.35,
           color='g', label='pre-pumping false-alarm floor')
    a.set_yticks(y)
    a.set_yticklabels(['MD %.0f-%.0f\n(n=%d)' % (b['md_lo_ft'], b['md_hi_ft'],
                                                 b['n_channels']) for b in mb],
                      fontsize=6.5)
    a.set_xlabel('exceedance fraction')
    a.set_title('(c) The transient is on EVERY band tested\n(this is what makes '
                'it instrumental)', fontsize=9)
    a.legend(fontsize=6.5, loc='lower right')
    a.grid(alpha=0.3, axis='x')

    a = fig.add_subplot(gs[2, 1])
    xs = [r['N'] for r in rows_excl if r['snr_variant'] == 'median9']
    ye = [r['front_connected_md_ft'] for r in rows_excl
          if r['snr_variant'] == 'median9']
    yi = [r['front_connected_md_ft'] for r in rows_incl
          if r['snr_variant'] == 'median9']
    a.plot(xs, ye, 'o-', color='k', label='stripe excluded')
    a.plot(xs, yi, '^--', color='#d95f02', label='stripe NOT excluded')
    a.set_xscale('log')
    a.set_xticks(n_grid)
    a.set_xticklabels([f'{N:g}' for N in n_grid])
    a.set_xlabel('N')
    a.set_ylabel('source-connected front MD (ft)')
    a.set_title('(d) Leaving the artifact in moves the\nfront by up to %.0f ft'
                % max(abs(e - i) for e, i in zip(ye, yi)
                      if e is not None and i is not None), fontsize=9)
    a.legend(fontsize=7)
    a.grid(alpha=0.3)

    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def fig_decay(path, md, dist, valid, norms, mod, sims, cmp_, refsens, envelope,
              sig_s, art_s, snr9, S_sig, das_s, prim, falsifiers):
    fig, ax = plt.subplots(1, 3, figsize=(18.0, 6.0))
    cols = {'uniform_1150': '#d95f02', 'uniform_550': '#7b3294',
            'two_zone_r2': '#1b7837'}
    lab = {'uniform_1150': 'model, uniform D = 1150 ft$^2$/s',
           'uniform_550': 'model, uniform D = 550 ft$^2$/s',
           'two_zone_r2': 'model, two-zone D(x) (r2 best fit)'}

    a = ax[0]
    a.semilogy(dist[valid], (S_sig / np.median(das_s[(md >= prim[0])
                                                     & (md <= prim[1])]))[valid],
               color='0.82', lw=0.6, label='observed LF-DAS, per channel')
    a.set_ylim(2e-4, 3.0)
    a.semilogy(dist[valid], norms['raw'][valid], 'k-', lw=1.8,
               label='observed LF-DAS, %d-channel median' % cmp_['spatial_median_channels'])
    for n, s in sims.items():
        a.semilogy(dist[valid], mod[n][valid], color=cols[n], lw=1.5, label=lab[n])
    a.axvspan(16645 - prim[1], 16645 - prim[0], color='0.85', zorder=0)
    a.axvline(16645 - 15003, color='r', ls=':', lw=1.2)
    a.text(16645 - 15003, a.get_ylim()[1] * 0.5, ' v2 stopped here\n (MD 15003)',
           color='r', fontsize=7.5, ha='left', va='top')
    for f in falsifiers:
        d = 16645.0 - f['md_ft']
        if d <= dist[valid].max():
            a.plot([d], [1.1e-3], 'v', color='crimson', ms=5)
            a.text(d, 1.4e-3, 'g%d\n%.0f psi' % (f['gauge'], f['rise_psi']),
                   color='crimson', fontsize=6.5, ha='center')
    a.set_xlabel('distance from the source node at MD 16645 (ft)')
    a.set_ylabel('amplitude / amplitude in the reference band')
    a.set_title('(a) SHAPE comparison, extended to MD 13000.\nEach curve '
                'normalised to its own median over MD %.0f-%.0f. LF-DAS stays in '
                'native\nstrain-rate units: no psi<->strain coefficient anywhere.'
                % prim, fontsize=9.5)
    a.legend(fontsize=7.2, loc='lower left')
    a.grid(alpha=0.3, which='both')

    a = ax[1]
    for n in sims:
        lo = [e.get(f'{n}_min', np.nan) for e in envelope]
        hi = [e.get(f'{n}_max', np.nan) for e in envelope]
        xc = [0.5 * (e['band_lo_ft'] + e['band_hi_ft']) for e in envelope]
        a.fill_between(xc, lo, hi, color=cols[n], alpha=0.25)
        a.semilogy(xc, [0.5 * (l + h) for l, h in zip(lo, hi)], 'o-',
                   color=cols[n], lw=1.5, ms=4, label=lab[n])
    a.axhline(1.0, color='k', lw=1.0)
    a.axvline(16645 - 15003, color='r', ls=':', lw=1.2)
    a.text(16645 - 15003, 0.02, ' v2 stopped here', color='r', fontsize=7.5)
    a.set_yscale('log')
    a.set_xlabel('distance from the source node at MD 16645 (ft)')
    a.set_ylabel('model / DAS  (normalised amplitude ratio)')
    a.set_title('(b) Bands are the FULL envelope over 2 normalisations x 5 '
                'reference bands.\nThe envelope is as wide as the effect in every '
                'band beyond 1400 ft, so no\nsingle over-prediction factor is '
                'quotable there.', fontsize=9.5)
    a.legend(fontsize=7.2, loc='lower left')
    a.grid(alpha=0.3, which='both')

    a = ax[2]
    a.semilogy(md, sig_s / np.median(sig_s), 'k-', lw=1.2,
               label='quiescent noise floor (15-ch median)')
    a.semilogy(md, art_s / np.median(art_s), color='#d95f02', lw=1.2,
               label='acquisition-artifact amplitude (15-ch median)')
    a.semilogy(md, snr9, color='#2166ac', lw=1.0, label='DAS detection SNR (9-ch median)')
    a.axhline(2.0, color='#2166ac', ls=':', lw=1.0)
    a.text(md[0], 2.05, ' SNR = 2', color='#2166ac', fontsize=7.5)
    a.axvline(15003, color='r', ls=':', lw=1.2)
    a.set_xlabel('MD (ft)')
    a.set_ylabel('amplitude / array median  |  SNR')
    a.set_title('(c) Why (b) cannot be tightened. The artifact proxy takes '
                'DISCRETE steps\n(gain regimes), the noise floor moves with it '
                'only partly, and below MD ~15150\nthe DAS itself sits at SNR ~2, '
                'so its amplitude is a bound, not a measurement.', fontsize=9.5)
    a.legend(fontsize=7.2, loc='upper left')
    a.grid(alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


if __name__ == '__main__':
    main()
