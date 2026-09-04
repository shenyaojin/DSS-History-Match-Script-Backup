"""C3 addendum -- three diagnostics that the main C3 run left open, plus a v2 figure.

Additive only: it writes new files beside the c3_precursor.py products and never
touches them.

    python3 scripts/manuscript_well_leakage/rev2/c3_precursor_addendum.py \
        --config configs/rev2/c3_precursor.json

1. CROSS-EVALUATION. The before/after RMSEs from the main run are not directly
   comparable, because truncating the record changes which residuals are averaged as
   well as which D minimises them. The honest comparison is the same misfit
   definition evaluated at both optima: how much does moving D from the full-window
   optimum to the post-precursor optimum actually buy, judged on the post-precursor
   data alone?

2. MISFIT PARTITION. At the full-window optimum, how much of the total squared misfit
   lives inside the excluded precursor segment? That is the direct measure of
   "precursor contamination" of the calibration.

3. ONSET SHARPNESS. The onset estimate is the maximum of a smoothed record; on a flat
   maximum that is poorly determined. This reports the width of the interval within
   1 psi of the maximum, so the onset times are quoted with a resolution.

Figure v2 of the precursor/pumping panel widens the bottom panel so that the pumping
start is visible on the same axis as the minima.
"""

import argparse
import csv
import datetime
import json
import os
import platform
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.join(os.path.dirname(_HERE), 'baseline_calibration')
sys.path.insert(0, _BASE)
sys.path.insert(0, _HERE)
import r1_calibration_core as core  # noqa: E402
from r1_run_calibration import load_config, load_window_data, pick_source_gauge  # noqa: E402
from c3_precursor import (first_crossing, moving_average, load_pumping,  # noqa: E402
                          pumping_events, load_extended_gauges)

REPO = os.getcwd()


def log(m):
    print(f"[c3+] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg, cfg_hash = load_config(args.config)
    outdir = cfg['outputs']['dir']
    prev = json.load(open(os.path.join(outdir, 'manifest.json')))
    log(f"building on {outdir}/manifest.json (run {prev['run_utc']})")

    series, gauge_numbers, gauge_mds, frac_hits, t_start, t_end = load_window_data(cfg)
    src_gauge, _ = pick_source_gauge(cfg, series, frac_hits)
    src = series[src_gauge]
    m = cfg['mesh']
    pad_lo, pad_hi, dx = (float(m['domain_pad_low_md_ft']),
                          float(m['domain_pad_high_md_ft']), float(m['dx_ft']))
    mesh = np.arange(cfg['window']['md_min_ft'] - pad_lo,
                     cfg['window']['md_max_ft'] + pad_hi + dx / 2.0, dx)
    source_idx = int(np.argmin(np.abs(mesh - src['md_ft'])))
    dt = float(cfg['solver']['dt_s'])
    t_total = float(src['taxis'][-1])
    tgt_gauges = [n for n in sorted(series) if n != src_gauge]

    masks = {}
    for lab in prev['results']['refit_masks']:
        masks[lab] = {int(r['gauge']): float(r['t_start_s'])
                      for r in prev['results']['refit_masks'][lab]}
    targets = []
    for n in tgt_gauges:
        s = series[n]
        targets.append({'gauge': n, 'distance_ft': float(abs(s['md_ft'] - src['md_ft'])),
                        'idx': int(np.argmin(np.abs(mesh - s['md_ft']))),
                        'taxis': s['taxis'], 'data': s['delta_psi'],
                        'amp_scale': float(np.max(s['delta_psi']))})

    # ---- 1. cross-evaluation, straight off the saved sweep arrays -----------
    A = np.load(os.path.join(outdir, 'c3_arrays.npz'))
    labels = list(masks)
    opt = {l: prev['results']['uniform_optima'][l] for l in labels}
    cross_rows = []
    for scored_on in labels:
        g = A[f'grid_{scored_on}']
        for key, nm in (('rmse_gauge_mean', 'gauge_mean_rmse_psi'),
                        ('rmse_normalised', 'normalised_rmse')):
            c = A[f'{key}_{scored_on}']
            o = np.argsort(g)
            gs, cs = g[o], c[o]
            row = {'scored_on': scored_on, 'metric': nm}
            for at in labels:
                D = opt[at][('absolute_norm_gauge_mean' if key == 'rmse_gauge_mean'
                             else 'normalised_norm')]['D_ft2_s']
                row[f'at_D_of_{at}'] = float(np.interp(D, gs, cs))
                row[f'D_of_{at}'] = float(D)
            best = float(np.min(cs))
            row['own_minimum'] = best
            row['penalty_using_full_window_D_pct'] = float(
                (row[f'at_D_of_full'] / best - 1.0) * 100.0)
            cross_rows.append(row)
    for r in cross_rows:
        log(f"scored on {r['scored_on']:16s} {r['metric']:20s}: own min {r['own_minimum']:.4g}, "
            f"using the full-window D costs {r['penalty_using_full_window_D_pct']:+.2f}%")

    # ---- 2. misfit partition at the full-window optimum ---------------------
    D_full = opt['full']['absolute_norm_gauge_mean']['D_ft2_s']
    taxis, rec = core.solve_forward(mesh, core.build_uniform_profile(mesh, D_full),
                                    dt, t_total, src['taxis'], src['delta_psi'],
                                    source_idx, record_idx=[t['idx'] for t in targets])
    part_rows = []
    for lab in labels:
        if lab == 'full':
            continue
        tot_ex, tot_in = 0.0, 0.0
        for k, tg in enumerate(targets):
            t0 = masks[lab][tg['gauge']]
            sim = np.interp(tg['taxis'], taxis, rec[:, k])
            r = sim - tg['data']
            ex = tg['taxis'] < t0
            ss_ex = float(np.sum(r[ex] ** 2))
            ss_in = float(np.sum(r[~ex] ** 2))
            tot_ex += ss_ex
            tot_in += ss_in
            part_rows.append({
                'rule': lab, 'gauge': tg['gauge'], 'distance_ft': tg['distance_ft'],
                'D_ft2_s': D_full, 't_start_s': t0,
                'n_excluded': int(ex.sum()), 'n_retained': int((~ex).sum()),
                'rmse_excluded_psi': float(np.sqrt(ss_ex / max(ex.sum(), 1))),
                'rmse_retained_psi': float(np.sqrt(ss_in / max((~ex).sum(), 1))),
                'ss_excluded': ss_ex, 'ss_retained': ss_in,
                'precursor_share_of_squared_misfit_pct': float(
                    100.0 * ss_ex / (ss_ex + ss_in)),
            })
        part_rows.append({
            'rule': lab, 'gauge': 'ALL', 'distance_ft': '', 'D_ft2_s': D_full,
            't_start_s': '', 'n_excluded': '', 'n_retained': '',
            'rmse_excluded_psi': '', 'rmse_retained_psi': '',
            'ss_excluded': tot_ex, 'ss_retained': tot_in,
            'precursor_share_of_squared_misfit_pct': float(
                100.0 * tot_ex / (tot_ex + tot_in))})
        log(f"rule {lab}: the excluded pre-start segment carries "
            f"{100.0*tot_ex/(tot_ex+tot_in):.1f}% of the total squared misfit at D={D_full:.0f}")

    # ---- 3. onset sharpness -------------------------------------------------
    pump = load_pumping(cfg)
    events = pumping_events(pump, cfg)
    ext = load_extended_gauges(cfg, list(gauge_numbers))
    ext_t0 = datetime.datetime.fromisoformat(cfg['extended_window']['time_start'])
    prec = {int(r['gauge']): r for r in prev['results']['precursor_table']}
    on_thr = float(cfg['pumping_events']['on_threshold_bpm'])
    rows = events[str(on_thr)]
    win_t0 = datetime.datetime.fromisoformat(prec[1]['window_t0_utc'])
    prior = [r for r in rows if datetime.datetime.fromisoformat(r['start_utc']) < win_t0]
    pump_start = datetime.datetime.fromisoformat(prior[-1]['start_utc'])
    prev_major = max(prior[:-1], key=lambda r: r['max_rate_bpm'])
    prev_stop = datetime.datetime.fromisoformat(prev_major['stop_utc'])

    onset_rows = []
    for n in sorted(ext):
        te, de = ext[n]['taxis_from_t0'], ext[n]['raw_psi']
        tmin = (datetime.datetime.fromisoformat(prec[n]['t_min_utc']) - ext_t0).total_seconds()
        k = int(np.searchsorted(te, tmin, side='right')) - 1
        sm = moving_average(te, de, 60.0)
        j = int(np.argmax(sm[:k + 1]))
        near = np.where(sm[:k + 1] >= sm[j] - 1.0)[0]
        t_on = ext_t0 + datetime.timedelta(seconds=float(te[j]))
        lo = ext_t0 + datetime.timedelta(seconds=float(te[near[0]]))
        hi = ext_t0 + datetime.timedelta(seconds=float(te[near[-1]]))
        onset_rows.append({
            'gauge': n, 'distance_ft': prec[n]['distance_ft'],
            'onset_utc_60s': t_on.isoformat(),
            'onset_within_1psi_lo_utc': lo.isoformat(),
            'onset_within_1psi_hi_utc': hi.isoformat(),
            'onset_plateau_width_s': float(te[near[-1]] - te[near[0]]),
            'lag_from_prev_cycle_shutin_s': (t_on - prev_stop).total_seconds(),
            'lag_from_pump_start_s': (t_on - pump_start).total_seconds(),
            'implied_D_x2_over_lag_ft2_s': (
                float(prec[n]['distance_ft'] ** 2 / (t_on - prev_stop).total_seconds())
                if (t_on - prev_stop).total_seconds() > 0 and prec[n]['distance_ft'] > 0
                else ''),
        })
        log(f"g{n} onset {t_on.strftime('%H:%M:%S')} "
            f"(+-1 psi plateau {onset_rows[-1]['onset_plateau_width_s']:.0f} s), "
            f"lag from prior shut-in {onset_rows[-1]['lag_from_prev_cycle_shutin_s']:+.0f} s")

    # ---- write ------------------------------------------------------------
    written = []

    def write_csv(path, rows_, fields):
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            for r in rows_:
                w.writerow(r)
        written.append(path)

    write_csv(os.path.join(outdir, 'c3_cross_evaluation.csv'), cross_rows,
              list(cross_rows[0].keys()))
    write_csv(os.path.join(outdir, 'c3_misfit_partition.csv'), part_rows,
              list(part_rows[0].keys()))
    write_csv(os.path.join(outdir, 'c3_onset_sharpness.csv'), onset_rows,
              list(onset_rows[0].keys()))

    # ---- figure v2 ---------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dpi = int(cfg['outputs']['figure_dpi'])
    ext_t_total = float(max(ext[n]['taxis_from_t0'][-1] for n in ext))
    cmap = plt.get_cmap('viridis')
    gcol = {n: cmap(0.08 + 0.82 * i / max(len(gauge_numbers) - 1, 1))
            for i, n in enumerate(sorted(gauge_numbers))}

    def mins(d):
        return (d - ext_t0).total_seconds() / 60.0

    fig, ax = plt.subplots(3, 1, figsize=(11.5, 12.0),
                           gridspec_kw={'height_ratios': [0.85, 1.3, 1.3]})
    ax[0].sharex(ax[1])
    p = pump[cfg['data']['pumping_start_channel']]
    tp = np.array([(p['start'] + datetime.timedelta(seconds=float(x)) - ext_t0).total_seconds()
                   for x in p['taxis']])
    s0 = (tp >= 0) & (tp <= ext_t_total)
    ax[0].plot(tp[s0] / 60.0, p['data'][s0], color='#1f77b4', lw=1.5)
    ax[0].set_ylabel('Slurry rate (bpm)', color='#1f77b4')
    a0b = ax[0].twinx()
    q = pump['Treating Pressure']
    tq = np.array([(q['start'] + datetime.timedelta(seconds=float(x)) - ext_t0).total_seconds()
                   for x in q['taxis']])
    s1 = (tq >= 0) & (tq <= ext_t_total)
    a0b.plot(tq[s1] / 60.0, q['data'][s1], color='#999999', lw=1.0)
    a0b.set_ylabel('Treating pressure (psi)', color='#777777')
    ax[0].set_title('C3  The "precursor" is the falloff of the PRECEDING injection cycle, '
                    'not a response to this one', fontsize=11.5)

    for a in ax:
        a.axvspan(mins(win_t0), mins(win_t0) + t_total / 60.0,
                  color='#ffe9b0', alpha=0.6, zorder=0)
        a.axvline(mins(pump_start), color='#d62728', lw=1.7, ls='--', zorder=1)
        a.axvline(mins(prev_stop), color='#2ca02c', lw=1.5, ls=':', zorder=1)
        a.grid(alpha=0.25, lw=0.5)

    for n in sorted(ext):
        ax[1].plot(ext[n]['taxis_from_t0'] / 60.0, ext[n]['raw_psi'],
                   color=gcol[n], lw=1.3, label=f'g{n}')
    for r in onset_rows:
        n = r['gauge']
        tt = mins(datetime.datetime.fromisoformat(r['onset_utc_60s']))
        y = float(np.interp(tt * 60.0, ext[n]['taxis_from_t0'], ext[n]['raw_psi']))
        ax[1].plot(tt, y, marker='v', ms=9, mfc='white', mec=gcol[n], mew=1.9, zorder=5)
    ax[1].set_ylabel('Gauge pressure (psi)')
    ax[1].set_xlim(0.0, ext_t_total / 60.0)
    ax[1].set_xlabel('minutes after 2020-03-16 10:20:00 UTC')
    ax[1].legend(ncol=7, fontsize=8.5, loc='lower right', framealpha=0.92)
    ax[1].text(0.015, 0.97,
               'v = onset: maximum of the 60 s-smoothed record before the precursor '
               'minimum.\nThe onsets migrate outward from the 10:50:34 shut-in of the '
               'preceding cycle;\nsix of seven PRECEDE the 11:18:57 pumping restart.',
               transform=ax[1].transAxes, fontsize=8.8, va='top',
               bbox=dict(fc='white', ec='0.7', alpha=0.85, boxstyle='round,pad=0.35'))

    for n in sorted(series):
        w0 = mins(datetime.datetime.fromisoformat(prec[n]['window_t0_utc']))
        t, d = series[n]['taxis'], series[n]['delta_psi']
        keep = t <= 900.0
        ax[2].plot(w0 + t[keep] / 60.0, d[keep], color=gcol[n], lw=1.6, label=f'g{n}')
        ax[2].plot(w0 + prec[n]['t_min_s'] / 60.0, prec[n]['min_dP_psi'], marker='o',
                   ms=7.5, mfc='white', mec=gcol[n], mew=1.9, zorder=5)
    ax[2].axhline(0.0, color='k', lw=0.9)
    ax[2].set_ylabel('Window-referenced $\\Delta P$ (psi)')
    ax[2].set_xlabel('minutes after 2020-03-16 10:20:00 UTC   '
                     '(shaded = R1 comparison window, opens 11:24:04.8)')
    ax[2].set_xlim(mins(pump_start) - 1.5, mins(win_t0) + 900.0 / 60.0)
    ax[2].set_ylim(-36, 46)
    ax[2].legend(ncol=7, fontsize=8.5, loc='upper left', framealpha=0.92)
    ax[2].annotate('pumping start\n11:18:57', xy=(mins(pump_start), 30),
                   xytext=(mins(pump_start) + 0.5, 34), fontsize=8.8, color='#d62728')
    ax[2].text(0.985, 0.06,
               'o = precursor minimum.  The minima lag pumping start by 399-794 s and '
               'migrate with distance,\nso they are not locked to the start of pumping.',
               transform=ax[2].transAxes, fontsize=8.8, va='bottom', ha='right',
               bbox=dict(fc='white', ec='0.7', alpha=0.85, boxstyle='round,pad=0.35'))
    fig.tight_layout()
    f = os.path.join(outdir, 'fig_c3_precursor_vs_pumping_v2.png')
    fig.savefig(f, dpi=dpi)
    plt.close(fig)
    written.append(f)

    # ---- manifest ----------------------------------------------------------
    code_sha = {}
    for mod in list(sys.modules.values()):
        fp = getattr(mod, '__file__', None)
        if not fp:
            continue
        fp = os.path.abspath(fp)
        if fp.startswith(REPO + os.sep) and fp.endswith('.py') and os.path.exists(fp):
            code_sha[os.path.relpath(fp, REPO)] = core.file_sha256(fp)
    code_sha[os.path.relpath(os.path.abspath(__file__), REPO)] = \
        core.file_sha256(os.path.abspath(__file__))

    inputs = [cfg['data']['gauge_md_npz'], cfg['data']['frac_hit_stage1_npz']]
    inputs += [cfg['data']['gauge_series_template'].format(n=int(n)) for n in gauge_numbers]
    inputs += [pump[c]['path'] for c in cfg['data']['pumping_channels']]
    inputs += [os.path.join(outdir, 'c3_arrays.npz'), os.path.join(outdir, 'manifest.json')]

    import scipy
    import matplotlib as mpl
    import fiberis
    manifest = {
        'study_id': cfg['study_id'] + '_addendum',
        'run_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_resolved': cfg,
        'config_sha256': cfg_hash,
        'environment': {
            'python': sys.version.split()[0], 'platform': platform.platform(),
            'numpy': np.__version__, 'scipy': scipy.__version__,
            'matplotlib': mpl.__version__,
            'fiberis_path': os.path.dirname(os.path.abspath(fiberis.__file__)),
            'cwd': REPO, 'code_sha256': code_sha,
            'input_data_sha256': {p: core.file_sha256(p) for p in inputs},
        },
        'source_protocol': {
            'source_md_ft': float(src['md_ft']), 'source_gauge': int(src_gauge),
            'driving_series_path': cfg['data']['gauge_series_template'].format(n=src_gauge),
            'application': 'dirichlet_node', 'source_mesh_idx': int(source_idx),
            'md_snap_residual_ft': float(mesh[source_idx] - src['md_ft']),
        },
        'numerics': {
            'theta': float(cfg['solver']['theta']),
            'interface_avg': cfg['solver']['interface_avg'], 'dt_s': dt,
            'adaptive': None, 'n_steps': int(np.ceil(t_total / dt)),
            'domain_md_ft': [float(mesh[0]), float(mesh[-1])],
            'pad_low_ft': pad_lo, 'pad_high_ft': pad_hi, 'dx_ft': dx,
            'nx': int(len(mesh)), 'barrier': None, 'n_forward_solves': 1,
        },
        'results': {
            'builds_on_manifest_sha256': core.file_sha256(
                os.path.join(outdir, 'manifest.json')),
            'cross_evaluation': cross_rows,
            'misfit_partition_at_full_window_optimum': part_rows,
            'onset_sharpness': onset_rows,
            'pumping_start_utc': pump_start.isoformat(),
            'preceding_cycle': prev_major,
        },
        'outputs': [],
    }
    for pth in written:
        manifest['outputs'].append({'path': pth, 'bytes': os.path.getsize(pth),
                                    'sha256': core.file_sha256(pth)})
    mp = os.path.join(outdir, 'manifest_addendum.json')
    with open(mp, 'w') as fh:
        json.dump(manifest, fh, indent=2, default=float)
    log(f"wrote {mp} and {len(written)} product files")


if __name__ == '__main__':
    main()
