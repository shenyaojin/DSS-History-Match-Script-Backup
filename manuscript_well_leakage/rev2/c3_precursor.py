"""C3 -- the far-field negative precursor: timing, refit, and the surviving spread.

Run from the repository root:

    python3 scripts/manuscript_well_leakage/rev2/c3_precursor.py \
        --config configs/rev2/c3_precursor.json

What this does, in order:

1. Reproduces the measured precursor table (per-gauge minimum delta-P, time of the
   minimum, fraction of that gauge's signal) directly from the gauge records, as a
   check on the numbers quoted in the house rules.

2. TIMING. Places the seven minima on the same absolute time axis as the stage-1
   Slurry Rate record and reports the lag of each from pumping start and from the
   first pressure arrival at that gauge. It then tests the ONSET as well as the
   minimum, because the two carry different information: a minimum can migrate
   while the onset does not. The onset is tested two ways -- inside the comparison
   window (where it turns out to be censored at the window edge) and on an extended
   record that reaches back to a genuinely quiescent state.

3. The decisive forward test. The R1 comparison window opens at 11:24, in the middle
   of the falloff of the PRECEDING injection cycle. This run therefore also drives
   the same verified diffusion kernel from a quiescent 10:20 state through the whole
   preceding cycle, and asks whether the negative excursion then comes out of pure
   diffusion without any poroelastic term.

4. REFIT. Two independently defensible post-precursor start-time rules, misfit scored
   only inside each gauge's own retained window, uniform-D and per-gauge single fits
   before and after.

5. THE SPREAD. How much of the per-gauge single-fit spread survives excluding the
   precursor. The spread is reported as a PATH-AVERAGE effect: every single-gauge fit
   assumes a uniform medium over the whole source-to-gauge path, so a farther gauge
   integrates more low-diffusivity rock and returns a lower equivalent value.

The forward solver is not reimplemented here: r1_calibration_core.solve_forward is
the verified kernel (bit-equivalent to fibeRIS, ~1800x faster). Only the SCORING is
new, because the misfit has to be restricted to a per-gauge time window.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import sys
from multiprocessing import Pool

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.join(os.path.dirname(_HERE), 'baseline_calibration')
sys.path.insert(0, _BASE)
import r1_calibration_core as core  # noqa: E402
from r1_run_calibration import load_config, load_window_data, pick_source_gauge  # noqa: E402

REPO = os.getcwd()
_G = {}


def log(m):
    print(f"[c3] {m}", flush=True)


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------

def first_crossing(taxis, series, level, start_index=0):
    """First time at or after start_index at which series reaches level (upward)."""
    for i in range(start_index, len(series)):
        if series[i] >= level:
            if i == start_index or series[i - 1] >= level:
                return float(taxis[i])
            y0, y1 = series[i - 1], series[i]
            f = (level - y0) / (y1 - y0)
            return float(taxis[i - 1] + f * (taxis[i] - taxis[i - 1]))
    return float('nan')


def moving_average(taxis, series, width_s):
    """Centred moving average over a fixed time width on a possibly uneven axis."""
    out = np.empty_like(series)
    half = width_s / 2.0
    lo = np.searchsorted(taxis, taxis - half, side='left')
    hi = np.searchsorted(taxis, taxis + half, side='right')
    csum = np.concatenate([[0.0], np.cumsum(series)])
    out = (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)
    return out


def _score(resid_list, targets):
    mse, nrm, sq_pool, n_pool = [], [], 0.0, 0
    for r, tgt in zip(resid_list, targets):
        v = float(np.mean(r ** 2))
        mse.append(v)
        nrm.append(v / tgt['amp_scale'] ** 2)
        sq_pool += float(np.sum(r ** 2))
        n_pool += int(r.size)
    return {
        'per_gauge_mse': mse,
        'per_gauge_rmse': [float(np.sqrt(v)) for v in mse],
        'rmse_gauge_mean_psi': float(np.sqrt(np.mean(mse))),
        'rmse_normalised': float(np.sqrt(np.mean(nrm))),
        'rmse_pooled_psi': float(np.sqrt(sq_pool / n_pool)),
        'n_residuals': n_pool,
    }


def masked_metrics(taxis_sim, sim_at_targets, targets):
    """Per-gauge and pooled misfit with each gauge scored only inside its own mask.

    Returns per-gauge MSE, the gauge-mean RMSE (the norm the house-rules baselines
    82.33 / 11.87 psi are quoted in), the amplitude-normalised RMSE, and the pooled
    RMSE over all retained residuals (the R1 norm). Both pooled conventions are
    reported because they are not the same number and the round has already been
    burned once by an unnamed norm.
    """
    resid = []
    for k, tgt in enumerate(targets):
        m = tgt['mask']
        resid.append(np.interp(tgt['taxis'][m], taxis_sim, sim_at_targets[:, k])
                     - tgt['data'][m])
    return _score(resid, targets)


def masked_metrics_presampled(sim_list, targets):
    """Same scoring, but the simulation is already on each gauge's own time axis."""
    resid = [sim_list[k][t['mask']] - t['data'][t['mask']] for k, t in enumerate(targets)]
    return _score(resid, targets)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_pumping(cfg):
    """Stage-1 surface channels on an absolute time axis."""
    out = {}
    for ch in cfg['data']['pumping_channels']:
        p = os.path.join(cfg['data']['pumping_dir'], ch + '.npz')
        f = np.load(p, allow_pickle=True)
        st = f['start_time'].item()
        if hasattr(st, 'to_pydatetime'):
            st = st.to_pydatetime()
        out[ch] = {'path': p, 'start': st,
                   'taxis': np.asarray(f['taxis'], float),
                   'data': np.asarray(f['data'], float)}
    return out


def pumping_events(pump, cfg):
    """On/off intervals of the rate channel, plus a threshold-sensitivity table."""
    ch = cfg['data']['pumping_start_channel']
    p = pump[ch]
    st, t, d = p['start'], p['taxis'], p['data']
    ev = {}
    for thr in cfg['pumping_events']['sensitivity_thresholds_bpm']:
        on = d > thr
        chg = np.diff(on.astype(int))
        idx = np.arange(len(d))
        starts = idx[1:][chg == 1]
        stops = idx[1:][chg == -1]
        rows = []
        for s in starts:
            e = stops[stops > s]
            e = int(e[0]) if len(e) else len(d) - 1
            if (e - s) >= cfg['pumping_events']['min_on_duration_s']:
                rows.append({
                    'start_utc': (st + datetime.timedelta(seconds=float(t[s]))).isoformat(),
                    'stop_utc': (st + datetime.timedelta(seconds=float(t[e]))).isoformat(),
                    'duration_s': int(e - s),
                    'max_rate_bpm': float(d[s:e].max()),
                })
        ev[str(thr)] = rows
    return ev


def load_extended_gauges(cfg, gauge_numbers):
    """Gauge records over the extended window, all on one absolute time origin."""
    from fiberis.analyzer.Data1D import Data1D_Gauge
    e = cfg['extended_window']
    t0 = datetime.datetime.fromisoformat(e['time_start'])
    t1 = datetime.datetime.fromisoformat(e['time_end'])
    out = {}
    for n in gauge_numbers:
        g = Data1D_Gauge.Data1DGauge()
        g.load_npz(cfg['data']['gauge_series_template'].format(n=int(n)))
        g.crop(t0, t1)
        st = g.start_time
        st = st.to_pydatetime() if hasattr(st, 'to_pydatetime') else st
        taxis = np.asarray(g.taxis, float) + (st - t0).total_seconds()
        raw = np.asarray(g.data, float)
        out[int(n)] = {'t0_abs': t0, 'taxis_from_t0': taxis, 'raw_psi': raw,
                       'delta_psi': raw - raw[0], 'first_sample_utc': st.isoformat()}
    return out


# ---------------------------------------------------------------------------
# parallel sweep
# ---------------------------------------------------------------------------

def _init_worker(payload):
    _G.update(payload)


def _sweep_one(D):
    mesh = _G['mesh']
    taxis, rec = core.solve_forward(mesh, core.build_uniform_profile(mesh, float(D)),
                                    _G['dt'], _G['t_total'], _G['src_taxis'],
                                    _G['src_data'], _G['source_idx'],
                                    record_idx=_G['record_idx'])
    res = {}
    for label, targets in _G['target_sets'].items():
        res[label] = masked_metrics(taxis, rec, targets)
    return float(D), res


def _sweep_ext(D):
    mesh = _G['mesh']
    taxis, rec = core.solve_forward(mesh, core.build_uniform_profile(mesh, float(D)),
                                    _G['dt'], _G['ext_t_total'], _G['ext_src_taxis'],
                                    _G['ext_src_data'], _G['source_idx'],
                                    record_idx=_G['record_idx'])
    res = {}
    for label, targets in _G['ext_target_sets'].items():
        sim_list = []
        for k, tgt in enumerate(targets):
            s = np.interp(tgt['w0'] + tgt['taxis'], taxis, rec[:, k])
            sim_list.append(s - np.interp(tgt['w0'], taxis, rec[:, k]))
        res[label] = masked_metrics_presampled(sim_list, targets)
        res[label]['sim_min_psi'] = [float(np.min(s)) for s in sim_list]
        res[label]['sim_tmin_s'] = [float(targets[k]['taxis'][int(np.argmin(s))])
                                    for k, s in enumerate(sim_list)]
    return float(D), res


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg, cfg_hash = load_config(args.config)
    outdir = cfg['outputs']['dir']
    os.makedirs(outdir, exist_ok=True)
    log(f"config {args.config} sha256={cfg_hash[:16]}")

    # -- data ---------------------------------------------------------------
    series, gauge_numbers, gauge_mds, frac_hits, t_start, t_end = load_window_data(cfg)
    src_gauge, fh_centroid = pick_source_gauge(cfg, series, frac_hits)
    src = series[src_gauge]
    log(f"source gauge g{src_gauge} at MD {src['md_ft']:.0f}; targets "
        f"{[n for n in sorted(series) if n != src_gauge]}")

    m = cfg['mesh']
    pad_lo, pad_hi, dx = (float(m['domain_pad_low_md_ft']),
                          float(m['domain_pad_high_md_ft']), float(m['dx_ft']))
    mesh = np.arange(cfg['window']['md_min_ft'] - pad_lo,
                     cfg['window']['md_max_ft'] + pad_hi + dx / 2.0, dx)
    source_idx = int(np.argmin(np.abs(mesh - src['md_ft'])))
    md_snap_residual = float(mesh[source_idx] - src['md_ft'])
    dt = float(cfg['solver']['dt_s'])
    t_total = float(src['taxis'][-1])
    log(f"domain MD [{mesh[0]:.0f}, {mesh[-1]:.0f}] nx={len(mesh)}; "
        f"dt={dt} s, t_total={t_total:.1f} s")

    pump = load_pumping(cfg)
    events = pumping_events(pump, cfg)
    ext = load_extended_gauges(cfg, list(gauge_numbers))

    # -- 1. reproduce the precursor table -----------------------------------
    thr_rel = float(cfg['arrival']['relative_fraction'])
    abs_thr = [float(x) for x in cfg['arrival']['absolute_thresholds_psi']]
    precursor = []
    for n in sorted(series):
        s = series[n]
        t, d = s['taxis'], s['delta_psi']
        i = int(np.argmin(d))
        obs_max = float(np.max(d))
        row = {
            'gauge': n,
            'md_ft': s['md_ft'],
            'distance_ft': float(abs(s['md_ft'] - src['md_ft'])),
            'min_dP_psi': float(d[i]),
            't_min_s': float(t[i]),
            'window_max_psi': obs_max,
            'pct_of_gauge_signal': float(abs(d[i]) / obs_max * 100.0),
            'arrival_rel10pct_s': core.arrival_time(t, d, thr_rel * obs_max),
        }
        for a in abs_thr:
            row[f'arrival_abs{a:g}psi_s'] = core.arrival_time(t, d, a)
        precursor.append(row)

    # absolute time of window t=0 for every gauge (crop resets start_time)
    from fiberis.analyzer.Data1D import Data1D_Gauge
    for row in precursor:
        g = Data1D_Gauge.Data1DGauge()
        g.load_npz(cfg['data']['gauge_series_template'].format(n=row['gauge']))
        g.crop(t_start, t_end)
        st = g.start_time
        st = st.to_pydatetime() if hasattr(st, 'to_pydatetime') else st
        row['window_t0_utc'] = st.isoformat()
        row['t_min_utc'] = (st + datetime.timedelta(seconds=row['t_min_s'])).isoformat()

    log("precursor table reproduced:")
    for r in precursor:
        log(f"  g{r['gauge']} dist {r['distance_ft']:6.0f} ft  min {r['min_dP_psi']:7.2f} psi "
            f"at {r['t_min_s']:6.1f} s ({r['t_min_utc'][11:19]})  "
            f"{r['pct_of_gauge_signal']:5.2f}% of signal")

    # -- 2. timing against the pumping record --------------------------------
    rate_ch = cfg['data']['pumping_start_channel']
    on_thr = float(cfg['pumping_events']['on_threshold_bpm'])
    rows = events[str(on_thr)]
    win_t0 = datetime.datetime.fromisoformat(precursor[0]['window_t0_utc'])
    prior = [r for r in rows if datetime.datetime.fromisoformat(r['start_utc']) < win_t0]
    pump_start = datetime.datetime.fromisoformat(prior[-1]['start_utc'])
    # The reference for the falloff is the shut-in of the LARGEST preceding
    # injection cycle, not simply the previous interval: a 3.6 bpm blip does not
    # set the far-field pressure field, the 22.8 bpm cycle does.
    prev_major = max(prior[:-1], key=lambda r: r['max_rate_bpm']) if len(prior) >= 2 else None
    prev_stop = (datetime.datetime.fromisoformat(prev_major['stop_utc'])
                 if prev_major else None)
    log(f"pumping start for this window = {pump_start.isoformat()} "
        f"(rule: last sustained 0 -> >{on_thr} bpm transition before the window)")
    log(f"prior on-intervals: " + "; ".join(
        f"{r['start_utc'][11:19]}-{r['stop_utc'][11:19]} ({r['max_rate_bpm']:.1f} bpm)"
        for r in prior))

    for r in precursor:
        tmin = datetime.datetime.fromisoformat(r['t_min_utc'])
        r['lag_min_from_pump_start_s'] = (tmin - pump_start).total_seconds()
        r['pump_start_utc'] = pump_start.isoformat()
        r['pump_start_window_time_s'] = (
            pump_start - datetime.datetime.fromisoformat(r['window_t0_utc'])).total_seconds()
        r['lag_min_from_arrival_rel10pct_s'] = r['t_min_s'] - r['arrival_rel10pct_s']
        for a in abs_thr:
            r[f'lag_min_from_arrival_abs{a:g}psi_s'] = (
                r['t_min_s'] - r[f'arrival_abs{a:g}psi_s'])

    # -- 3. onset ------------------------------------------------------------
    hold = float(cfg['precursor']['onset_rules']['in_window_derivative']['hold_s'])
    smooth_list = cfg['precursor']['onset_rules']['absolute_turnover']['smooth_s']
    ext_t0 = datetime.datetime.fromisoformat(cfg['extended_window']['time_start'])

    # quiescence check on the extended record
    q_end = (datetime.datetime.fromisoformat(cfg['extended_window']['quiescence_check_end'])
             - ext_t0).total_seconds()
    quiescence = {}
    for n in sorted(ext):
        t, d = ext[n]['taxis_from_t0'], ext[n]['raw_psi']
        mq = t <= q_end
        quiescence[f'g{n}'] = float(np.ptp(d[mq]))
    log("extended-record quiescence (peak-to-peak psi over 10:20-10:28): " +
        ", ".join(f"{k}={v:.3f}" for k, v in quiescence.items()))

    for r in precursor:
        n = r['gauge']
        # (a) in-window derivative rule
        t, d = series[n]['taxis'], series[n]['delta_psi']
        dd = np.gradient(d, t)
        onset_idx = None
        for i in range(len(t)):
            j = np.searchsorted(t, t[i] + hold, side='right')
            if np.all(dd[i:max(j, i + 1)] < 0):
                onset_idx = i
                break
        r['onset_in_window_s'] = float(t[onset_idx]) if onset_idx is not None else float('nan')
        r['onset_in_window_censored'] = bool(onset_idx == 0)
        # (b) absolute turnover on the extended record
        te, de = ext[n]['taxis_from_t0'], ext[n]['raw_psi']
        tmin_from_ext_t0 = (datetime.datetime.fromisoformat(r['t_min_utc'])
                            - ext_t0).total_seconds()
        r['onset_turnover_utc'] = {}
        r['onset_turnover_lag_from_pump_start_s'] = {}
        r['onset_turnover_is_last_local_max'] = {}
        for w in smooth_list:
            sm = moving_average(te, de, float(w))
            k = int(np.searchsorted(te, tmin_from_ext_t0, side='right')) - 1
            # The onset is the maximum of the smoothed record preceding the
            # precursor minimum: the moment that gauge turns over from rising to
            # falling. Taking the argmax over [record start, t_min] rather than
            # walking back to the nearest local maximum makes the estimate
            # insensitive to small wiggles; the check below records whether the
            # smoothed record really is monotone-decreasing from there to the
            # minimum, i.e. whether the two definitions coincide.
            j = int(np.argmax(sm[:k + 1]))
            seg = sm[j:k + 1]
            monotone = bool(np.all(np.diff(seg) <= 1e-9)) if len(seg) > 1 else True
            tt = ext_t0 + datetime.timedelta(seconds=float(te[j]))
            r['onset_turnover_utc'][str(w)] = tt.isoformat()
            r['onset_turnover_lag_from_pump_start_s'][str(w)] = (tt - pump_start).total_seconds()
            r['onset_turnover_is_last_local_max'][str(w)] = monotone

    log("onset: in-window derivative rule -> " + ", ".join(
        f"g{r['gauge']}={r['onset_in_window_s']:.0f}s"
        f"{'[CENSORED]' if r['onset_in_window_censored'] else ''}" for r in precursor))
    log("onset: absolute turnover (60 s smoother) -> " + ", ".join(
        f"g{r['gauge']}={r['onset_turnover_utc']['60.0'][11:19]}" for r in precursor))

    # previous-cycle peak arrival -> implied diffusivity, an independent check that
    # the turnover times are a diffusive front and not a simultaneous poroelastic step
    if prev_stop is not None:
        for r in precursor:
            tt = datetime.datetime.fromisoformat(r['onset_turnover_utc']['60.0'])
            lag = (tt - prev_stop).total_seconds()
            r['onset_lag_from_prev_shutin_s'] = lag
            r['implied_D_from_onset_ft2_s'] = (
                float(r['distance_ft'] ** 2 / lag) if lag > 0 and r['distance_ft'] > 0
                else float('nan'))

    # -- 4. targets and refit masks -----------------------------------------
    tgt_gauges = [n for n in sorted(series) if n != src_gauge]

    def build_targets(rule):
        out = []
        for n in tgt_gauges:
            s = series[n]
            t, d = s['taxis'], s['delta_psi']
            amp = float(np.max(d))
            i = int(np.argmin(d))
            if rule is None:
                t_start_s, mask = float(t[0]), np.ones(len(t), bool)
            else:
                spec = cfg['refit']['rules'][rule]
                level = (spec['level_psi'] if spec['level_mode'] == 'absolute'
                         else spec['level_frac'] * amp)
                t_start_s = first_crossing(t, d, level, i)
                mask = t >= t_start_s
            out.append({'gauge': n, 'md_ft': s['md_ft'],
                        'distance_ft': float(abs(s['md_ft'] - src['md_ft'])),
                        'idx': int(np.argmin(np.abs(mesh - s['md_ft']))),
                        'taxis': t, 'data': d, 'mask': mask,
                        'amp_scale': amp, 't_start_s': t_start_s,
                        'n_kept': int(mask.sum()), 'n_total': int(len(t)),
                        'frac_kept': float(mask.sum() / len(t))})
        return out

    target_sets = {'full': build_targets(None)}
    for rule in cfg['refit']['rules']:
        target_sets[rule] = build_targets(rule)
    for label, ts in target_sets.items():
        log(f"mask '{label}': " + ", ".join(
            f"g{t['gauge']} t>={t['t_start_s']:.0f}s ({t['frac_kept']*100:.0f}%)" for t in ts))

    record_idx = [t['idx'] for t in target_sets['full']]

    payload = {'mesh': mesh, 'dt': dt, 't_total': t_total,
               'src_taxis': src['taxis'], 'src_data': src['delta_psi'],
               'source_idx': source_idx, 'record_idx': record_idx,
               'target_sets': target_sets}

    # -- 5. sweeps -----------------------------------------------------------
    sw = cfg['sweeps']['coarse']
    coarse = np.logspace(np.log10(sw['min']), np.log10(sw['max']), int(sw['n_points']))
    nproc = int(cfg['search']['processes'])
    log(f"uniform-D sweep: {len(coarse)} points, {nproc} workers")
    with Pool(nproc, initializer=_init_worker, initargs=(payload,)) as pool:
        res_coarse = pool.map(_sweep_one, coarse, chunksize=2)

    def collect(results):
        grid = np.array([r[0] for r in results])
        o = np.argsort(grid)
        grid = grid[o]
        out = {}
        for label in target_sets:
            out[label] = {
                'grid': grid,
                'rmse_gauge_mean': np.array([results[i][1][label]['rmse_gauge_mean_psi'] for i in o]),
                'rmse_normalised': np.array([results[i][1][label]['rmse_normalised'] for i in o]),
                'rmse_pooled': np.array([results[i][1][label]['rmse_pooled_psi'] for i in o]),
                'per_gauge_mse': np.array([results[i][1][label]['per_gauge_mse'] for i in o]),
                'n_residuals': results[o[0]][1][label]['n_residuals'],
            }
        return out

    C = collect(res_coarse)

    # refine over the union of every optimum found
    opts = []
    for label in target_sets:
        for key in ('rmse_gauge_mean', 'rmse_normalised'):
            opts.append(C[label]['grid'][int(np.argmin(C[label][key]))])
        pg = C[label]['per_gauge_mse']
        for k in range(pg.shape[1]):
            opts.append(C[label]['grid'][int(np.argmin(pg[:, k]))])
    rf = cfg['sweeps']['refine']
    lo = max(np.log10(min(opts)) - rf['half_width_decades'], np.log10(sw['min']))
    hi = min(np.log10(max(opts)) + rf['half_width_decades'], np.log10(sw['max']))
    n_ref = int(min(rf['max_points'], np.ceil((hi - lo) / rf['spacing_decades']) + 1))
    refine = np.logspace(lo, hi, n_ref)
    log(f"refine sweep: {n_ref} points over D = {10**lo:.0f} .. {10**hi:.0f} "
        f"({(10**rf['spacing_decades']-1)*100:.1f}% spacing)")
    with Pool(nproc, initializer=_init_worker, initargs=(payload,)) as pool:
        res_ref = pool.map(_sweep_one, refine, chunksize=2)
    ALL = collect(res_coarse + res_ref)

    def optimum(label, key):
        g, c = ALL[label]['grid'], ALL[label][key]
        i = int(np.argmin(c))
        return {'D_ft2_s': float(g[i]), 'value': float(c[i]),
                'at_grid_edge': bool(i == 0 or i == len(g) - 1)}

    summary = {'uniform': {}, 'per_gauge': {}}
    for label in target_sets:
        summary['uniform'][label] = {
            'absolute_norm_gauge_mean': optimum(label, 'rmse_gauge_mean'),
            'absolute_norm_pooled': optimum(label, 'rmse_pooled'),
            'normalised_norm': optimum(label, 'rmse_normalised'),
            'n_residuals': int(ALL[label]['n_residuals']),
        }
        g = ALL[label]['grid']
        pg = ALL[label]['per_gauge_mse']
        rows = []
        for k, tg in enumerate(target_sets[label]):
            i = int(np.argmin(pg[:, k]))
            rows.append({'gauge': tg['gauge'], 'distance_ft': tg['distance_ft'],
                         'D_ft2_s': float(g[i]), 'rmse_psi': float(np.sqrt(pg[i, k])),
                         'at_grid_edge': bool(i == 0 or i == len(g) - 1),
                         't_start_s': tg['t_start_s'], 'frac_kept': tg['frac_kept']})
        summary['per_gauge'][label] = rows
        Ds = [r['D_ft2_s'] for r in rows]
        summary['per_gauge'][label + '_spread'] = {
            'D_max': float(max(Ds)), 'D_min': float(min(Ds)),
            'ratio': float(max(Ds) / min(Ds)),
            'log10_std': float(np.std(np.log10(Ds), ddof=1)),
        }

    for label in target_sets:
        u = summary['uniform'][label]
        s = summary['per_gauge'][label + '_spread']
        log(f"[{label}] uniform optimum D={u['absolute_norm_gauge_mean']['D_ft2_s']:.0f} "
            f"(gauge-mean RMSE {u['absolute_norm_gauge_mean']['value']:.2f} psi); "
            f"normalised optimum D={u['normalised_norm']['D_ft2_s']:.0f} "
            f"(norm {u['normalised_norm']['value']:.4f}); "
            f"per-gauge D spans {s['D_min']:.0f}-{s['D_max']:.0f} ({s['ratio']:.1f}x)")

    # cross-norm misfit at the other norm's optimum, for the before/after table
    for label in target_sets:
        for key, other in (('rmse_gauge_mean', 'normalised_norm'),
                           ('rmse_normalised', 'absolute_norm_gauge_mean')):
            D = summary['uniform'][label][other]['D_ft2_s']
            i = int(np.argmin(np.abs(ALL[label]['grid'] - D)))
            summary['uniform'][label][f'{key}_at_{other}_optimum'] = float(ALL[label][key][i])

    # -- 6. extended-window forward test -------------------------------------
    ext_src = ext[src_gauge]
    ext_t_total = float(max(ext[n]['taxis_from_t0'][-1] for n in ext))
    ext_targets = []
    for tg in target_sets['full']:
        n = tg['gauge']
        w0 = (datetime.datetime.fromisoformat(
            [r for r in precursor if r['gauge'] == n][0]['window_t0_utc']) - ext_t0).total_seconds()
        ext_targets.append(dict(tg, w0=w0))
    payload_ext = dict(payload)
    payload_ext.update({'ext_t_total': ext_t_total,
                        'ext_src_taxis': ext_src['taxis_from_t0'],
                        'ext_src_data': ext_src['delta_psi'],
                        'ext_target_sets': {'full': ext_targets}})
    esw = cfg['sweeps']['extended_window']
    ext_grid = np.unique(np.concatenate([
        np.logspace(np.log10(esw['min']), np.log10(esw['max']), int(esw['n_points'])),
        np.array(cfg['sweeps']['extended_window_report_D'], float)]))
    log(f"extended-window sweep ({ext_t_total:.0f} s from a quiescent state): "
        f"{len(ext_grid)} points")
    with Pool(nproc, initializer=_init_worker, initargs=(payload_ext,)) as pool:
        res_ext = pool.map(_sweep_ext, ext_grid, chunksize=1)
    res_ext.sort(key=lambda r: r[0])
    ext_grid_s = np.array([r[0] for r in res_ext])
    ext_rmse = np.array([r[1]['full']['rmse_gauge_mean_psi'] for r in res_ext])
    ext_best_i = int(np.argmin(ext_rmse))

    ext_report = {}
    for D in cfg['sweeps']['extended_window_report_D'] + [float(ext_grid_s[ext_best_i])]:
        i = int(np.argmin(np.abs(ext_grid_s - D)))
        r = res_ext[i][1]['full']
        rows = []
        for k, tg in enumerate(ext_targets):
            obs = [p for p in precursor if p['gauge'] == tg['gauge']][0]
            rows.append({
                'gauge': tg['gauge'], 'distance_ft': tg['distance_ft'],
                'obs_min_psi': obs['min_dP_psi'], 'obs_t_min_s': obs['t_min_s'],
                'sim_min_psi': r['sim_min_psi'][k], 'sim_t_min_s': r['sim_tmin_s'][k],
                'explained_fraction_of_min': float(r['sim_min_psi'][k] / obs['min_dP_psi']),
                'sim_t_min_minus_obs_t_min_s': float(r['sim_tmin_s'][k] - obs['t_min_s']),
            })
        ext_report[f"D={ext_grid_s[i]:.0f}"] = {
            'D_ft2_s': float(ext_grid_s[i]),
            'rmse_gauge_mean_psi_over_full_window': float(r['rmse_gauge_mean_psi']),
            'per_gauge': rows}

    # the same comparison for the R1-style run (starts AT the window, IC = 0)
    taxis_w, rec_w = core.solve_forward(
        mesh, core.build_uniform_profile(mesh, summary['uniform']['full']
                                         ['absolute_norm_gauge_mean']['D_ft2_s']),
        dt, t_total, src['taxis'], src['delta_psi'], source_idx, record_idx=record_idx)
    r1_style = []
    for k, tg in enumerate(target_sets['full']):
        s = np.interp(tg['taxis'], taxis_w, rec_w[:, k])
        obs = [p for p in precursor if p['gauge'] == tg['gauge']][0]
        r1_style.append({'gauge': tg['gauge'], 'distance_ft': tg['distance_ft'],
                         'obs_min_psi': obs['min_dP_psi'],
                         'sim_min_psi': float(np.min(s)),
                         'sim_t_min_s': float(tg['taxis'][int(np.argmin(s))]),
                         'explained_fraction_of_min': float(np.min(s) / obs['min_dP_psi'])})
    log("extended-window test, explained fraction of the observed minimum:")
    for key in ext_report:
        log(f"  {key}: " + ", ".join(
            f"g{r['gauge']}={r['explained_fraction_of_min']*100:.0f}%"
            for r in ext_report[key]['per_gauge']))
    log("  R1-style run starting AT the window: " + ", ".join(
        f"g{r['gauge']}={r['explained_fraction_of_min']*100:.0f}%" for r in r1_style))

    # -- 7. figures ----------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dpi = int(cfg['outputs']['figure_dpi'])
    written = []

    cmap = plt.get_cmap('viridis')
    gcol = {n: cmap(0.08 + 0.82 * i / max(len(gauge_numbers) - 1, 1))
            for i, n in enumerate(sorted(gauge_numbers))}

    def hhmm(dt_abs):
        return (dt_abs - ext_t0).total_seconds()

    # --- Figure 1: precursor against the pumping curve, absolute time -------
    fig, ax = plt.subplots(3, 1, figsize=(11, 11.5),
                           gridspec_kw={'height_ratios': [1.0, 1.35, 1.35]})
    ax[0].sharex(ax[1])
    p = pump[rate_ch]
    tp = np.array([(p['start'] + datetime.timedelta(seconds=float(x)) - ext_t0).total_seconds()
                   for x in p['taxis']])
    sel = (tp >= 0) & (tp <= ext_t_total)
    ax[0].plot(tp[sel] / 60.0, p['data'][sel], color='#1f77b4', lw=1.4, label='Slurry Rate (bpm)')
    ax[0].set_ylabel('Slurry rate (bpm)', color='#1f77b4')
    ax0b = ax[0].twinx()
    q = pump['Treating Pressure']
    tq = np.array([(q['start'] + datetime.timedelta(seconds=float(x)) - ext_t0).total_seconds()
                   for x in q['taxis']])
    selq = (tq >= 0) & (tq <= ext_t_total)
    ax0b.plot(tq[selq] / 60.0, q['data'][selq], color='#888888', lw=1.0,
              label='Treating pressure (psi)')
    ax0b.set_ylabel('Treating pressure (psi)', color='#666666')
    ax[0].set_title('C3  Far-field negative precursor on the stage-1 absolute time axis '
                    '(2020-03-16)', fontsize=11)

    for a in ax:
        a.axvspan(hhmm(win_t0) / 60.0,
                  (hhmm(win_t0) + t_total) / 60.0, color='#ffe9b0', alpha=0.55, zorder=0)
        a.axvline(hhmm(pump_start) / 60.0, color='#d62728', lw=1.6, ls='--', zorder=1)
        if prev_stop is not None:
            a.axvline(hhmm(prev_stop) / 60.0, color='#2ca02c', lw=1.4, ls=':', zorder=1)
        a.grid(alpha=0.25, lw=0.5)

    for n in sorted(ext):
        ax[1].plot(ext[n]['taxis_from_t0'] / 60.0, ext[n]['raw_psi'],
                   color=gcol[n], lw=1.2, label=f'g{n}')
    for r in precursor:
        tt = datetime.datetime.fromisoformat(r['onset_turnover_utc']['60.0'])
        n = r['gauge']
        y = float(np.interp(hhmm(tt), ext[n]['taxis_from_t0'], ext[n]['raw_psi']))
        ax[1].plot(hhmm(tt) / 60.0, y, marker='v', ms=8, mfc='white',
                   mec=gcol[n], mew=1.8, zorder=5)
    ax[1].set_ylabel('Gauge pressure (psi)')
    ax[1].set_xlim(0.0, ext_t_total / 60.0)
    ax[1].set_xlabel('minutes after 2020-03-16 10:20:00 UTC')
    ax[1].legend(ncol=7, fontsize=8, loc='lower right', framealpha=0.9)
    ax[1].text(0.012, 0.95, 'v  onset = last local maximum before the decline '
                            '(60 s smoother)', transform=ax[1].transAxes,
               fontsize=8.5, va='top')

    for r in precursor:
        n = r['gauge']
        w0 = hhmm(datetime.datetime.fromisoformat(r['window_t0_utc']))
        t, d = series[n]['taxis'], series[n]['delta_psi']
        keep = t <= 900.0
        ax[2].plot((w0 + t[keep]) / 60.0, d[keep], color=gcol[n], lw=1.5, label=f'g{n}')
        ax[2].plot((w0 + r['t_min_s']) / 60.0, r['min_dP_psi'], marker='o', ms=7,
                   mfc='white', mec=gcol[n], mew=1.8, zorder=5)
    ax[2].axhline(0.0, color='k', lw=0.8)
    ax[2].set_ylabel('Window-referenced $\\Delta P$ (psi)')
    ax[2].set_xlabel('minutes after 2020-03-16 10:20:00 UTC   '
                     '(shaded = R1 comparison window 11:24-11:45)')
    ax[2].set_xlim(hhmm(win_t0) / 60.0 - 3.0, (hhmm(win_t0) + 900.0) / 60.0)
    ax[2].set_ylim(-38, 120)
    ax[2].legend(ncol=7, fontsize=8, loc='upper left', framealpha=0.9)
    ax[2].text(0.012, 0.62,
               'o  precursor minimum;  red dashed = pumping start '
               f'{pump_start.strftime("%H:%M:%S")};\n'
               'green dotted = shut-in of the preceding cycle '
               f'{prev_stop.strftime("%H:%M:%S") if prev_stop else "n/a"}',
               transform=ax[2].transAxes, fontsize=8.5, va='top')
    fig.tight_layout()
    f1 = os.path.join(outdir, 'fig_c3_precursor_vs_pumping_v1.png')
    fig.savefig(f1, dpi=dpi)
    plt.close(fig)
    written.append(f1)

    # --- Figure 2: the extended-window forward test -------------------------
    Dbest_ext = float(ext_grid_s[ext_best_i])
    taxis_e, rec_e = core.solve_forward(mesh, core.build_uniform_profile(mesh, Dbest_ext),
                                        dt, ext_t_total, ext_src['taxis_from_t0'],
                                        ext_src['delta_psi'], source_idx,
                                        record_idx=record_idx)
    D_r1 = summary['uniform']['full']['absolute_norm_gauge_mean']['D_ft2_s']
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
    for k, tg in enumerate(target_sets['full']):
        a = axes.flat[k]
        n = tg['gauge']
        w0 = ext_targets[k]['w0']
        t = tg['taxis']
        keep = t <= 900.0
        se = np.interp(w0 + t, taxis_e, rec_e[:, k]) - np.interp(w0, taxis_e, rec_e[:, k])
        sw_ = np.interp(t, taxis_w, rec_w[:, k])
        a.plot(t[keep], tg['data'][keep], color='k', lw=1.8, label='observed')
        a.plot(t[keep], sw_[keep], color='#d62728', lw=1.3, ls='--',
               label=f'starts at window, D={D_r1:.0f}')
        a.plot(t[keep], se[keep], color='#1f77b4', lw=1.5,
               label=f'starts quiescent 10:20, D={Dbest_ext:.0f}')
        a.axhline(0, color='k', lw=0.6)
        a.set_title(f"g{n}  {tg['distance_ft']:.0f} ft from source", fontsize=10)
        a.grid(alpha=0.25, lw=0.5)
        if k >= 3:
            a.set_xlabel('s after window start (11:24:04.8)')
        if k % 3 == 0:
            a.set_ylabel('$\\Delta P$ (psi)')
        a.set_ylim(-40, 90)
    axes.flat[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('C3  The negative precursor is reproduced by pure diffusion once the '
                 'solver is not started in the middle of the previous falloff',
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    f2 = os.path.join(outdir, 'fig_c3_extended_window_v1.png')
    fig.savefig(f2, dpi=dpi)
    plt.close(fig)
    written.append(f2)

    # --- Figure 3: before/after refit ---------------------------------------
    lab_style = {'full': ('#333333', '-', 'full window (before)'),
                 'A_zero_crossing': ('#1f77b4', '--', 'rule A: after zero crossing'),
                 'B_ten_percent': ('#d62728', '-.', 'rule B: after 10% of window max')}
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9))
    for label in target_sets:
        c, ls, nm = lab_style[label]
        g = ALL[label]['grid']
        ax[0, 0].loglog(g, ALL[label]['rmse_gauge_mean'], color=c, ls=ls, lw=1.6, label=nm)
        o = summary['uniform'][label]['absolute_norm_gauge_mean']
        ax[0, 0].plot(o['D_ft2_s'], o['value'], 'o', color=c, ms=7)
        ax[0, 1].loglog(g, ALL[label]['rmse_normalised'], color=c, ls=ls, lw=1.6, label=nm)
        o = summary['uniform'][label]['normalised_norm']
        ax[0, 1].plot(o['D_ft2_s'], o['value'], 'o', color=c, ms=7)
    ax[0, 0].set_xlabel('uniform $D$ (ft$^2$/s)')
    ax[0, 0].set_ylabel('gauge-mean RMSE (psi)')
    ax[0, 0].set_title('(a) absolute norm', fontsize=10)
    ax[0, 1].set_xlabel('uniform $D$ (ft$^2$/s)')
    ax[0, 1].set_ylabel('amplitude-normalised RMSE')
    ax[0, 1].set_title('(b) normalised norm', fontsize=10)
    for a in (ax[0, 0], ax[0, 1]):
        a.grid(alpha=0.25, which='both', lw=0.5)
        a.legend(fontsize=8.5)

    for label in target_sets:
        c, ls, nm = lab_style[label]
        rows = summary['per_gauge'][label]
        ax[1, 0].semilogy([r['distance_ft'] for r in rows], [r['D_ft2_s'] for r in rows],
                          marker='o', color=c, ls=ls, lw=1.6, ms=6, label=nm)
    ax[1, 0].set_xlabel('distance from source (ft)')
    ax[1, 0].set_ylabel('single-gauge path-averaged $D$ (ft$^2$/s)')
    ax[1, 0].set_title('(c) per-gauge single fits: a PATH-AVERAGE, not a local contrast',
                       fontsize=10)
    ax[1, 0].grid(alpha=0.25, which='both', lw=0.5)
    ax[1, 0].legend(fontsize=8.5)

    labels = list(target_sets)
    x = np.arange(len(labels))
    v1 = [summary['uniform'][l]['absolute_norm_gauge_mean']['value'] for l in labels]
    v2 = [summary['uniform'][l]['normalised_norm']['value'] for l in labels]
    ax[1, 1].bar(x - 0.2, v1, 0.4, color='#4c72b0', label='gauge-mean RMSE (psi)')
    axb = ax[1, 1].twinx()
    axb.bar(x + 0.2, v2, 0.4, color='#dd8452', label='normalised RMSE')
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels([lab_style[l][2].replace(': ', ':\n') for l in labels],
                             fontsize=8)
    ax[1, 1].set_ylabel('gauge-mean RMSE (psi)', color='#4c72b0')
    axb.set_ylabel('amplitude-normalised RMSE', color='#dd8452')
    ax[1, 1].set_title('(d) misfit at each criterion\'s own optimum', fontsize=10)
    for xi, (a_, b_) in enumerate(zip(v1, v2)):
        ax[1, 1].text(xi - 0.2, a_, f'{a_:.1f}', ha='center', va='bottom', fontsize=8)
        axb.text(xi + 0.2, b_, f'{b_:.3f}', ha='center', va='bottom', fontsize=8)
    fig.suptitle('C3  Recalibration with the precursor excluded from the misfit',
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    f3 = os.path.join(outdir, 'fig_c3_refit_before_after_v1.png')
    fig.savefig(f3, dpi=dpi)
    plt.close(fig)
    written.append(f3)

    # -- 8. csv --------------------------------------------------------------
    def write_csv(path, rows, fields):
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow(r)
        written.append(path)

    prec_rows = []
    for r in precursor:
        d = {k: v for k, v in r.items() if not isinstance(v, dict)}
        d['onset_turnover_utc_60s'] = r['onset_turnover_utc']['60.0']
        d['onset_turnover_lag_from_pump_start_s_60s'] = \
            r['onset_turnover_lag_from_pump_start_s']['60.0']
        d['onset_turnover_utc_30s'] = r['onset_turnover_utc']['30.0']
        d['onset_turnover_utc_120s'] = r['onset_turnover_utc']['120.0']
        prec_rows.append(d)
    write_csv(os.path.join(outdir, 'c3_precursor_timing.csv'), prec_rows,
              ['gauge', 'md_ft', 'distance_ft', 'min_dP_psi', 't_min_s', 'window_max_psi',
               'pct_of_gauge_signal', 'window_t0_utc', 't_min_utc', 'pump_start_utc',
               'pump_start_window_time_s', 'lag_min_from_pump_start_s',
               'arrival_rel10pct_s', 'lag_min_from_arrival_rel10pct_s',
               'arrival_abs10psi_s', 'lag_min_from_arrival_abs10psi_s',
               'arrival_abs25psi_s', 'lag_min_from_arrival_abs25psi_s',
               'onset_in_window_s', 'onset_in_window_censored',
               'onset_turnover_utc_30s', 'onset_turnover_utc_60s', 'onset_turnover_utc_120s',
               'onset_turnover_lag_from_pump_start_s_60s',
               'onset_lag_from_prev_shutin_s', 'implied_D_from_onset_ft2_s'])

    pg_rows = []
    for label in target_sets:
        for r in summary['per_gauge'][label]:
            pg_rows.append(dict(r, criterion=label))
    write_csv(os.path.join(outdir, 'c3_per_gauge_D.csv'), pg_rows,
              ['criterion', 'gauge', 'distance_ft', 't_start_s', 'frac_kept',
               'D_ft2_s', 'rmse_psi', 'at_grid_edge'])

    uni_rows = []
    for label in target_sets:
        u = summary['uniform'][label]
        uni_rows.append({
            'criterion': label,
            'n_residuals': u['n_residuals'],
            'D_absolute_norm_ft2_s': u['absolute_norm_gauge_mean']['D_ft2_s'],
            'rmse_gauge_mean_psi': u['absolute_norm_gauge_mean']['value'],
            'rmse_pooled_psi_at_pooled_optimum': u['absolute_norm_pooled']['value'],
            'D_pooled_norm_ft2_s': u['absolute_norm_pooled']['D_ft2_s'],
            'D_normalised_norm_ft2_s': u['normalised_norm']['D_ft2_s'],
            'rmse_normalised': u['normalised_norm']['value'],
            'rmse_normalised_at_absolute_optimum':
                u['rmse_normalised_at_absolute_norm_gauge_mean_optimum'],
            'rmse_gauge_mean_at_normalised_optimum':
                u['rmse_gauge_mean_at_normalised_norm_optimum'],
            'D_spread_ratio': summary['per_gauge'][label + '_spread']['ratio'],
            'D_spread_min': summary['per_gauge'][label + '_spread']['D_min'],
            'D_spread_max': summary['per_gauge'][label + '_spread']['D_max'],
        })
    write_csv(os.path.join(outdir, 'c3_refit_summary.csv'), uni_rows,
              list(uni_rows[0].keys()))

    ext_rows = []
    for key, blk in ext_report.items():
        for r in blk['per_gauge']:
            ext_rows.append(dict(r, run=f"extended_quiescent_{key}",
                                 D_ft2_s=blk['D_ft2_s'],
                                 rmse_gauge_mean_psi=blk['rmse_gauge_mean_psi_over_full_window']))
    for r in r1_style:
        ext_rows.append(dict(r, run=f'window_start_D={D_r1:.0f}', D_ft2_s=D_r1))
    write_csv(os.path.join(outdir, 'c3_extended_window_test.csv'), ext_rows,
              ['run', 'D_ft2_s', 'gauge', 'distance_ft', 'obs_min_psi', 'obs_t_min_s',
               'sim_min_psi', 'sim_t_min_s', 'explained_fraction_of_min',
               'sim_t_min_minus_obs_t_min_s', 'rmse_gauge_mean_psi'])

    arrays = os.path.join(outdir, 'c3_arrays.npz')
    np.savez(arrays,
             **{f'grid_{l}': ALL[l]['grid'] for l in target_sets},
             **{f'rmse_gauge_mean_{l}': ALL[l]['rmse_gauge_mean'] for l in target_sets},
             **{f'rmse_normalised_{l}': ALL[l]['rmse_normalised'] for l in target_sets},
             **{f'rmse_pooled_{l}': ALL[l]['rmse_pooled'] for l in target_sets},
             **{f'per_gauge_mse_{l}': ALL[l]['per_gauge_mse'] for l in target_sets},
             ext_grid=ext_grid_s, ext_rmse_gauge_mean=ext_rmse,
             target_gauges=np.array(tgt_gauges))
    written.append(arrays)

    # -- 9. manifest ---------------------------------------------------------
    code_sha = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if not f:
            continue
        f = os.path.abspath(f)
        if f.startswith(REPO + os.sep) and f.endswith('.py') and os.path.exists(f):
            code_sha[os.path.relpath(f, REPO)] = core.file_sha256(f)
    code_sha[os.path.relpath(os.path.abspath(__file__), REPO)] = \
        core.file_sha256(os.path.abspath(__file__))

    inputs = [cfg['data']['gauge_md_npz'], cfg['data']['frac_hit_stage1_npz']]
    inputs += [cfg['data']['gauge_series_template'].format(n=int(n)) for n in gauge_numbers]
    inputs += [pump[c]['path'] for c in cfg['data']['pumping_channels']]
    input_sha = {p: core.file_sha256(p) for p in inputs}

    import scipy
    import matplotlib as mpl
    import fiberis
    manifest = {
        'study_id': cfg['study_id'],
        'run_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_resolved': cfg,
        'config_sha256': cfg_hash,
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'matplotlib': mpl.__version__,
            'fiberis_path': os.path.dirname(os.path.abspath(fiberis.__file__)),
            'cwd': REPO,
            'code_sha256': code_sha,
            'input_data_sha256': input_sha,
        },
        'source_protocol': {
            'source_md_ft': float(src['md_ft']),
            'source_gauge': int(src_gauge),
            'driving_series_path': cfg['data']['gauge_series_template'].format(n=src_gauge),
            'application': 'dirichlet_node',
            'source_mesh_idx': int(source_idx),
            'md_snap_residual_ft': md_snap_residual,
        },
        'numerics': {
            'theta': float(cfg['solver']['theta']),
            'interface_avg': cfg['solver']['interface_avg'],
            'dt_s': dt,
            'adaptive': None,
            'n_steps': int(np.ceil(t_total / dt)),
            'domain_md_ft': [float(mesh[0]), float(mesh[-1])],
            'pad_low_ft': pad_lo,
            'pad_high_ft': pad_hi,
            'dx_ft': dx,
            'nx': int(len(mesh)),
            'barrier': None,
            'extended_window_n_steps': int(np.ceil(ext_t_total / dt)),
            'n_forward_solves': int(len(coarse) + len(refine) + len(ext_grid) + 2),
        },
        'results': {
            'precursor_table': precursor,
            'pumping_events_by_threshold_bpm': events,
            'pumping_start_utc': pump_start.isoformat(),
            'pumping_start_rule': cfg['pumping_events']['rule'],
            'previous_cycle_shutin_utc': prev_stop.isoformat() if prev_stop else None,
            'extended_record_quiescence_ptp_psi': quiescence,
            'refit_masks': {l: [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                 for k, v in t.items()
                                 if k in ('gauge', 't_start_s', 'n_kept', 'n_total',
                                          'frac_kept', 'amp_scale')}
                                for t in target_sets[l]] for l in target_sets},
            'uniform_optima': summary['uniform'],
            'per_gauge_single_fits': {l: summary['per_gauge'][l] for l in target_sets},
            'per_gauge_spread': {l: summary['per_gauge'][l + '_spread'] for l in target_sets},
            'extended_window_forward_test': ext_report,
            'extended_window_best_D_ft2_s': Dbest_ext,
            'extended_window_best_rmse_gauge_mean_psi': float(ext_rmse[ext_best_i]),
            'window_start_run_for_comparison': {'D_ft2_s': D_r1, 'per_gauge': r1_style},
        },
        'outputs': [],
    }
    with open(os.path.join(outdir, 'c3_results.json'), 'w') as fh:
        json.dump(manifest['results'], fh, indent=2, default=float)
    written.append(os.path.join(outdir, 'c3_results.json'))

    for p in written:
        manifest['outputs'].append({'path': p, 'bytes': os.path.getsize(p),
                                    'sha256': core.file_sha256(p)})
    mpath = os.path.join(outdir, 'manifest.json')
    with open(mpath, 'w') as fh:
        json.dump(manifest, fh, indent=2, default=float)
    log(f"wrote {mpath} and {len(written)} product files")
    log('done')


if __name__ == '__main__':
    main()
