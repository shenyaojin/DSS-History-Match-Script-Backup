#!/usr/bin/env python3
"""E1 - rebuild the Fig. 6 two-stage SYNTHETIC panel.

    python3 scripts/manuscript_well_leakage/rev2/e1_fig6.py \
        --config configs/rev2/e1_fig6.json

Run with CWD = repo root. Writes ONLY into `output/rev2_20260901/E1/<study>/`.

Why this rerun exists
---------------------
`DAS_history_matching_visualization/104_full_history_matching_manuscript.py`
draws the manuscript's two-stage figure from
`output/0211_simulation_MULTIstage/phase3_test.npz`. That file **contains no
barrier**: `101:214` writes it nominally at ratio 1e-6, but A5 matched it to the
uniform-D chain at ratio 1.0 to 2.18e-05 psi on a 1.0e4 psi field, and D4's
independent barrier-step attribution brackets it at 0.3-1. The caption describes
a five-order-of-magnitude reduction. So the published phase-3 panel is the
uniform-D solution.

What this run changes, and why (nothing here is a free choice)
--------------------------------------------------------------
  * **barrier ratio 1e-5, physical half-width w = 1.0 ft** - the parameters the
    manuscript TEXT states. `rev2_core.build_barrier_profile`, not mesh indices
    (A1/CORRECTION 4: the legacy index barrier realised 0.13333 ft and converges
    to ZERO width under refinement).
  * **5000 ft of padding at BOTH ends** (B2). The dominant boundary error for
    this geometry is at the HIGH-MD end, because 101's sources sit mid-domain and
    the plotted gauges 5 and 6 lie above them.
  * **fixed dt = 1 s** (A3). The manuscript's "adaptive" settings realise a fixed
    dt = 30 s, where BE and CN disagree by 3.49 % of peak.
  * **Gamma = 8.94e-9 psi^-1** for the synthetic strain rate, axis in s^-1
    (house rules). The legacy `data * 6894.76 / 30e9` is ~26x too large.

Nothing else moves: D = 140 ft^2/s (101:93), the legacy file-span phase
boundaries, the same five plotted gauges, the same panel MD window.

The chain is NOT re-implemented
-------------------------------
`a5_two_stage_chain` is imported and its `build_chain_mesh`, `phase_windows`,
`load_source_series` and `solve_phase` are called directly. A5 reproduces the
frozen 2025 archive to 2.6e-9 relative across all eight panels, so this file
inherits that arithmetic instead of re-deriving it. Phases 1 and 2 are solved
ONCE per study rather than once per ratio: `barrier_from_stage` is null for both,
so they are ratio-independent by construction, and the equality is asserted at
run time for the first two ratios.

Owned by this task: this file, `configs/rev2/e1_fig6.json`,
`output/rev2_20260901/E1/`. `rev2_core`, `rev2_data`, `rev2_layout`,
`rev2_manifest` and `a5_two_stage_chain` are imported and never edited;
`scripts/well_leakage_history_matching/` and `output/0211_simulation_MULTIstage/`
are read-only history.
"""

import argparse
import copy
import datetime
import json
import os
import sys
import time
import warnings

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.dates as mdates        # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))
for _p in (_HERE, os.path.join(_ROOT, 'fibeRIS', 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rev2_core as rc                   # noqa: E402
import rev2_data as rd                   # noqa: E402
import rev2_layout as rl                 # noqa: E402
import rev2_manifest as rm               # noqa: E402
import a5_two_stage_chain as a5          # noqa: E402  the chain, imported

STUDY_ID = "E1_fig6_synthetic_panel"
TASK_ID = "E1"

PHASES = ('phase1', 'phase2', 'phase3')


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def rtag(ratio):
    """Filesystem-safe tag for a ratio. 1.0 is named for what it IS."""
    return 'uniform' if float(ratio) >= 1.0 else f"ratio{float(ratio):g}"


def rlabel(ratio, sealed):
    if float(ratio) >= 1.0:
        return 'ratio 1 (no barrier)'
    if float(ratio) == float(sealed):
        return f'ratio {float(ratio):g} (sealed reference)'
    return f'ratio {float(ratio):g}'


def strain_rate_image(field_win, taxis, gamma, dec_s):
    """Synthetic strain RATE (s^-1) on the panel depth window, time-decimated.

    The manuscript computes the strain rate as a POST-PROCESSING gradient of the
    simulated pressure (104:126-127), not as a solver output, so the same is done
    here: strain = Gamma * P (dimensionless), strain rate = Gamma * dP/dt.
    Decimation is a block MEAN, not a stride, so the image is not aliased by
    whatever high-frequency content dt = 1 s resolves and dt = 30 s did not.
    """
    dpdt = np.gradient(np.asarray(field_win, dtype=float),
                       np.asarray(taxis, dtype=float), axis=0)
    sr = float(gamma) * dpdt
    dt = float(np.median(np.diff(taxis)))
    k = max(1, int(round(float(dec_s) / dt)))
    n = (sr.shape[0] // k) * k
    if n < k:
        return sr.astype(np.float32), np.asarray(taxis, dtype=float)
    srd = sr[:n].reshape(-1, k, sr.shape[1]).mean(axis=1)
    td = np.asarray(taxis, dtype=float)[:n].reshape(-1, k).mean(axis=1)
    return srd.astype(np.float32), td


def abs_times(t0_abs, taxis_s):
    base = np.datetime64(t0_abs, 'us')
    return base + (np.asarray(taxis_s, dtype=float) * 1e6).astype('timedelta64[us]')


def loglog_crossing(ratios, values, threshold):
    """Ratio at which `values(ratio)` crosses `threshold`, log-log interpolated.

    `values` is monotone-ish increasing in ratio over the saturated branch. Returns
    None (censored) when the whole ladder is on one side of the threshold, and says
    so rather than extrapolating - the ladder is the evidence, not a fit.
    """
    r = np.asarray(ratios, dtype=float)
    v = np.asarray(values, dtype=float)
    order = np.argsort(r)
    r, v = r[order], v[order]
    ok = v > 0
    r, v = r[ok], v[ok]
    if r.size < 2:
        return None, 'censored: fewer than two positive values on the ladder'
    if np.all(v >= threshold):
        return None, (f'censored below: every ladder ratio down to {r[0]:g} is '
                      f'above the threshold')
    if np.all(v < threshold):
        return None, (f'censored above: no ladder ratio up to {r[-1]:g} reaches '
                      f'the threshold')
    lr, lv, lt = np.log10(r), np.log10(v), np.log10(threshold)
    for i in range(r.size - 1):
        if (lv[i] - lt) * (lv[i + 1] - lt) <= 0 and lv[i + 1] != lv[i]:
            f = (lt - lv[i]) / (lv[i + 1] - lv[i])
            return float(10.0 ** (lr[i] + f * (lr[i + 1] - lr[i]))), 'interpolated'
    return None, 'no bracketing pair found'


# ---------------------------------------------------------------------------
# one study
# ---------------------------------------------------------------------------

def make_source(cfg, x, wins, phase):
    """A5's source series plus the frac-hit node placement for that phase."""
    spec = [p for p in cfg['phases'] if p['name'] == phase][0]
    t0, t1 = wins[phase]
    src = a5.load_source_series(spec['source_gauge'], t0, t1, cfg['source'])
    stage = int(cfg['source_stage_by_phase'][phase])
    hits = rd.load_frac_hits(stage, unique=False, sort=False)
    src['frac_hit_stage'] = stage
    src['frac_hit_mds_ft'] = [float(v) for v in hits]
    src['source_idx'] = [int(np.argmin(np.abs(x - float(h)))) for h in hits]
    src['source_md_ft'] = [float(x[i]) for i in src['source_idx']]
    src['snap_error_ft'] = [float(x[i] - float(h))
                            for i, h in zip(src['source_idx'], hits)]
    src['window_abs'] = [t0.isoformat(), t1.isoformat()]
    src['t_window_s'] = (t1 - t0).total_seconds()
    return src


def run_study(name, cfg, root_cfg, outdir, config_path):
    t_study = time.time()
    os.makedirs(outdir, exist_ok=True)
    manifest_path = os.path.join(outdir, 'manifest.json')
    rm.assert_absent([manifest_path])

    md_table = rd.load_gauge_md_table()
    x, mesh_rec = a5.build_chain_mesh(cfg['mesh'])
    wins, win_rec = a5.phase_windows(cfg['phase_boundaries'])
    gauges = [int(g) for g in cfg['targets']['gauges']]
    gidx = [int(np.argmin(np.abs(x - md_table.md_of(g)))) for g in gauges]
    val_gauge = int(cfg['targets']['validation_gauge'])
    jval = gauges.index(val_gauge)

    D0 = float(cfg['physics']['D_baseline_ft2_s'])
    base = np.full(len(x), D0, dtype=float)
    bcfg = cfg['barrier']
    ratios = [float(r) for r in bcfg['ratios']]
    r_head = float(bcfg['headline_ratio'])
    r_seal = float(bcfg['sealed_reference_ratio'])
    r_uni = float(bcfg['uniform_reference_ratio'])
    hits7 = rd.load_frac_hits(7, unique=False, sort=False)
    hits8 = rd.load_frac_hits(8, unique=False, sort=False)

    pnl = cfg['panel']
    md_lo = (float(np.min(hits8)) - 500.0 if pnl['md_lo_ft'] is None
             else float(pnl['md_lo_ft']))
    md_hi = (float(np.max(hits7)) + 500.0 if pnl['md_hi_ft'] is None
             else float(pnl['md_hi_ft']))
    pw = np.where((x >= md_lo) & (x <= md_hi))[0]
    gamma = float(pnl['gamma_psi_inv'])
    want_img = bool(cfg['outputs']['save_strain_rate_images'])
    img_ratios = [float(r) for r in pnl['waterfall_ratios']] if want_img else []

    # psi -> POINTS on the manuscript's own page. This is what turns "visible in
    # the figure" into a measured statement (see the config's _scale_note).
    span_ft = md_hi - md_lo
    psi_per_pt = ((span_ft / float(pnl['manuscript_panel_height_in']) / 72.0)
                  / float(pnl['manuscript_plot_scale_ft_per_psi']))
    thr_1pt = psi_per_pt
    thr_lw = psi_per_pt * float(pnl['manuscript_linewidth_pt'])

    tcfg, pcfg = cfg['time'], cfg['physics']
    written = []

    with rm.RunRecorder(manifest_path, study_id=STUDY_ID, task_id=TASK_ID,
                        config=cfg, config_path=config_path, run_label=name,
                        require_modules=('rev2_core', 'rev2_data', 'rev2_layout',
                                         'rev2_manifest',
                                         'a5_two_stage_chain')) as R:

        # ---- phases 1 and 2: ratio-independent, solved once ----------------
        srcs = {p: make_source(cfg, x, wins, p) for p in PHASES}
        s1, s2, s3 = srcs['phase1'], srcs['phase2'], srcs['phase3']

        init1 = np.full(len(x), float(s1['values_psi'][0]))
        _t = time.time()
        ta1, f1, trec1, ex1 = a5.solve_phase(x, base, s1, init1, tcfg, pcfg)
        ex1['wall_s'] = time.time() - _t
        tr1 = np.ascontiguousarray(f1[:, gidx])
        img1 = strain_rate_image(f1[:, pw], ta1, gamma,
                                 pnl['time_decimation_s']) if want_img else None
        end1 = f1[-1].copy()
        del f1

        _t = time.time()
        ta2, f2, trec2, ex2 = a5.solve_phase(x, base, s2, end1, tcfg, pcfg)
        ex2['wall_s'] = time.time() - _t
        tr2 = np.ascontiguousarray(f2[:, gidx])
        img2 = strain_rate_image(f2[:, pw], ta2, gamma,
                                 pnl['time_decimation_s']) if want_img else None
        init3 = f2[-1].copy()
        prof2_end = f2[-1].copy()
        del f2

        # ---- phase 3, one solve per ratio ----------------------------------
        tr3 = {}
        img3 = {}
        breports = {}
        dprof_head = None
        trec3 = {}
        ex3 = {}
        ta3 = None
        for ratio in ratios:
            if ratio >= 1.0:
                dprof, brep = base.copy(), None
            else:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    dprof, brep = rc.build_barrier_profile(
                        x, base, hits7, float(bcfg['w_ft']), float(ratio),
                        ratio_reference=bcfg['ratio_reference'],
                        combine=bcfg['combine'], on_empty=bcfg['on_empty'],
                        on_outside=bcfg['on_outside'], return_report=True)
                # THE REPORT IS THE AUTHORITY, NOT THE WARNING.
                if brep['n_fallback'] != 0:
                    raise RuntimeError(
                        f"ratio {ratio:g}: {brep['n_fallback']} barrier(s) fell "
                        f"back to the nearest node at w = {bcfg['w_ft']} ft on a "
                        f"dx = {cfg['mesh']['dx_ft']} ft mesh; the realised width "
                        f"would not be the requested one. "
                        f"{brep['fallback_messages'][:1]}")
                brep['n_warnings_raised'] = len(caught)
                brep['barrier_mds_ft'] = [float(v) for v in hits7]
            _t = time.time()
            t_a, f3, tr_, ex_ = a5.solve_phase(x, dprof, s3, init3, tcfg, pcfg)
            ex_['wall_s'] = time.time() - _t
            ta3 = t_a
            trec3[ratio], ex3[ratio] = tr_, ex_
            tr3[ratio] = np.ascontiguousarray(f3[:, gidx])
            breports[ratio] = brep
            if ratio == r_head:
                dprof_head = dprof.copy()
            if want_img and ratio in img_ratios:
                img3[ratio] = strain_rate_image(f3[:, pw], t_a, gamma,
                                                pnl['time_decimation_s'])
            del f3
            print(f"[E1]   {name} phase3 ratio {ratio:g} done "
                  f"({ex_['wall_s']:.1f} s)", flush=True)
        if dprof_head is None:
            dprof_head = base.copy()

        # ---- metrics -------------------------------------------------------
        dp = {r: tr3[r] - tr3[r][0] for r in ratios}
        ref_u, ref_s = dp[r_uni], dp[r_seal]
        per_ratio = {}
        for r in ratios:
            vs_u = np.abs(dp[r] - ref_u).max(axis=0)
            vs_s = np.abs(dp[r] - ref_s).max(axis=0)
            per_ratio[r] = {
                'peak_dp_psi': {f"g{g}": float(dp[r][:, j].max())
                                for j, g in enumerate(gauges)},
                'trough_dp_psi': {f"g{g}": float(dp[r][:, j].min())
                                  for j, g in enumerate(gauges)},
                'vs_uniform_max_abs_psi': {f"g{g}": float(vs_u[j])
                                           for j, g in enumerate(gauges)},
                'vs_sealed_max_abs_psi': {f"g{g}": float(vs_s[j])
                                          for j, g in enumerate(gauges)},
                'vs_uniform_worst_gauge_psi': float(vs_u.max()),
                'vs_sealed_worst_gauge_psi': float(vs_s.max()),
                'worst_gauge_vs_uniform': f"g{gauges[int(np.argmax(vs_u))]}",
                'worst_gauge_vs_sealed': f"g{gauges[int(np.argmax(vs_s))]}",
                'distinguishable_from_no_barrier_at_1pt':
                    bool(vs_u.max() >= thr_1pt),
                'distinguishable_from_perfect_seal_at_1pt':
                    bool(vs_s.max() >= thr_1pt),
                'informative_about_the_ratio_value':
                    bool(vs_u.max() >= thr_1pt and vs_s.max() >= thr_1pt),
            }
        # adjacent-ladder separation: what one step of the ladder is worth
        srt = sorted(ratios)
        adjacent = {}
        for i in range(len(srt) - 1):
            a, b = srt[i], srt[i + 1]
            sep = np.abs(dp[b] - dp[a]).max(axis=0)
            adjacent[f"{a:g}->{b:g}"] = {
                'decades': float(abs(np.log10(b) - np.log10(a))),
                'max_abs_psi_per_gauge': {f"g{g}": float(sep[j])
                                          for j, g in enumerate(gauges)},
                'worst_gauge_psi': float(sep.max()),
                'worst_gauge_points_on_the_manuscript_panel':
                    float(sep.max() / psi_per_pt),
            }

        vs_seal_worst = [per_ratio[r]['vs_sealed_worst_gauge_psi'] for r in srt]
        vs_uni_worst = [per_ratio[r]['vs_uniform_worst_gauge_psi'] for r in srt]
        r_floor_1pt, why_floor = loglog_crossing(srt, vs_seal_worst, thr_1pt)
        r_floor_lw, why_floor_lw = loglog_crossing(srt, vs_seal_worst, thr_lw)
        r_ceil_1pt, why_ceil = loglog_crossing(
            srt, [max(v, 1e-30) for v in vs_uni_worst], thr_1pt)

        # analytic crossing-time criterion
        w_full = 2.0 * float(bcfg['w_ft'])
        w_tot = float(len(hits7)) * w_full
        T3 = float(ta3[-1] - ta3[0])
        r_crit = w_tot ** 2 / (D0 * T3)
        crossing = {
            'definition': ("a barrier of full width W at D = D0*ratio has "
                           "diffusive crossing time W^2/(D0*ratio); the six "
                           "stage-7 barriers in series behave as one barrier of "
                           "total width W_tot = 6*W"),
            'per_barrier_full_width_ft': w_full,
            'series_total_width_ft': w_tot,
            'phase3_window_s': T3,
            'ratio_at_which_series_crossing_time_equals_the_phase3_window':
                float(r_crit),
            'per_barrier_crossing_time_s': {
                f"{r:g}": float(w_full ** 2 / (D0 * r)) for r in srt},
            'series_crossing_time_s': {
                f"{r:g}": float(w_tot ** 2 / (D0 * r)) for r in srt},
            'headline_ratio_series_crossing_over_window':
                float((w_tot ** 2 / (D0 * r_head)) / T3),
        }

        metrics = {
            'gauges': gauges,
            'gauge_md_ft': [md_table.md_of(g) for g in gauges],
            'gauge_mesh_md_ft': [float(x[i]) for i in gidx],
            'shielded_by_the_stage7_barrier_row':
                [f"g{g}" for g in gauges if md_table.md_of(g) > float(np.min(hits7))],
            'validation_gauge': f"g{val_gauge}",
            'stage7_barrier_md_span_ft': [float(np.min(hits7)), float(np.max(hits7))],
            'stage8_source_md_span_ft': [float(np.min(hits8)), float(np.max(hits8))],
            'plot_resolution': {
                'psi_per_point_on_the_manuscript_panel': float(psi_per_pt),
                'threshold_1_point_psi': float(thr_1pt),
                'threshold_one_manuscript_linewidth_psi': float(thr_lw),
                'derivation': (f"panel spans {span_ft:.2f} ft of MD in "
                               f"{pnl['manuscript_panel_height_in']:g} in at "
                               f"{pnl['manuscript_plot_scale_ft_per_psi']:g} ft/psi "
                               f"(104's ax3 scale), so 1 pt = "
                               f"{psi_per_pt:.2f} psi and the manuscript's own "
                               f"{pnl['manuscript_linewidth_pt']:g} pt line width "
                               f"= {thr_lw:.2f} psi"),
            },
            'per_ratio': {f"{r:g}": per_ratio[r] for r in srt},
            'adjacent_ladder_separation': adjacent,
            'informative_window': {
                'criterion': ("a ratio is informative about its own VALUE only if "
                              "the plotted curves are separable BOTH from the "
                              "no-barrier run AND from a perfectly sealing barrier "
                              "by at least one point on the manuscript's own panel"),
                'upper_edge_ratio_vs_no_barrier': r_ceil_1pt,
                'upper_edge_note': why_ceil,
                'lower_edge_ratio_vs_perfect_seal_1pt': r_floor_1pt,
                'lower_edge_note': why_floor,
                'lower_edge_ratio_vs_perfect_seal_one_linewidth': r_floor_lw,
                'lower_edge_note_linewidth': why_floor_lw,
                'analytic_lower_edge_from_crossing_time': float(r_crit),
                'headline_ratio': r_head,
                'headline_ratio_is_informative':
                    per_ratio[r_head]['informative_about_the_ratio_value'],
            },
            'crossing_time': crossing,
            'barrier_is_invisible_at_these_plotted_gauges': [
                f"g{g}" for j, g in enumerate(gauges)
                if max(per_ratio[r]['vs_uniform_max_abs_psi'][f"g{g}"]
                       for r in srt) < 1e-6],
        }

        # ---- archive comparison (read-only) --------------------------------
        arch = 'not requested for this study'
        if cfg.get('compare_to_archive'):
            arch = compare_archive(root_cfg, x, gidx, gauges, md_table,
                                   ta3, dp, tr3, r_head, r_uni)

        # ---- write arrays --------------------------------------------------
        p = os.path.join(outdir, f"e1_gauge_traces_{name}_v2.npz")
        payload = {
            'mesh_md_ft': x,
            'gauge_numbers': np.asarray(gauges, dtype=np.int64),
            'gauge_md_ft': np.asarray([md_table.md_of(g) for g in gauges], float),
            'gauge_mesh_idx': np.asarray(gidx, dtype=np.int64),
            'phase1_taxis_s': ta1, 'phase1_traces_psi': tr1,
            'phase2_taxis_s': ta2, 'phase2_traces_psi': tr2,
            'phase3_taxis_s': ta3,
            'phase1_t0_abs': np.array(str(s1['t0_abs'])),
            'phase2_t0_abs': np.array(str(s2['t0_abs'])),
            'phase3_t0_abs': np.array(str(s3['t0_abs'])),
            'ratios': np.asarray(srt, dtype=float),
            'phase2_final_profile_psi': prof2_end,
            'd_profile_headline_ft2_s': dprof_head,
            'D_baseline_ft2_s': np.array(D0),
            'barrier_half_width_ft': np.array(float(bcfg['w_ft'])),
        }
        for r in srt:
            payload[f"phase3_traces_psi_{rtag(r)}"] = tr3[r]
        np.savez_compressed(p, **payload)
        written.append((p, 'arrays_npz',
                        'synthetic gauge traces, every phase, every ratio, '
                        'full dt = 1 s rate'))

        if want_img:
            p = os.path.join(outdir, f"e1_strain_rate_{name}_v2.npz")
            pay = {'panel_md_ft': x[pw],
                   'gamma_psi_inv': np.array(gamma),
                   'units': np.array('strain_rate_s^-1'),
                   'time_decimation_s': np.array(float(pnl['time_decimation_s'])),
                   'phase1_taxis_s': img1[1], 'phase1_strain_rate': img1[0],
                   'phase2_taxis_s': img2[1], 'phase2_strain_rate': img2[0],
                   'phase1_t0_abs': np.array(str(s1['t0_abs'])),
                   'phase2_t0_abs': np.array(str(s2['t0_abs'])),
                   'phase3_t0_abs': np.array(str(s3['t0_abs']))}
            for r, (im, td) in img3.items():
                pay[f"phase3_taxis_s_{rtag(r)}"] = td
                pay[f"phase3_strain_rate_{rtag(r)}"] = im
            np.savez_compressed(p, **pay)
            written.append((p, 'arrays_npz',
                            'synthetic strain-rate images on the panel MD window, '
                            'Gamma applied, block-mean decimated in time'))

        p = os.path.join(outdir, f"e1_ratio_metrics_{name}_v2.json")
        with open(p, 'w') as fh:
            json.dump(rm._jsonify({'study': name, 'metrics': metrics,
                                   'archive_comparison': arch},
                                  max_array_len=64)[0],
                      fh, indent=2, sort_keys=True, ensure_ascii=False)
        written.append((p, 'json', 'ratio ladder metrics and informativeness'))

        # ---- field gauges + figures ----------------------------------------
        field = load_field_gauges(gauges, wins)
        p = os.path.join(outdir, f"e1_field_gauges_{name}_v2.npz")
        fp = {'gauge_numbers': np.asarray(gauges, dtype=np.int64)}
        for g in gauges:
            fp[f"g{g}_taxis_s"] = field[g]['taxis_s']
            fp[f"g{g}_psi"] = field[g]['psi']
            fp[f"g{g}_t0_abs"] = np.array(str(field[g]['t0_abs']))
        np.savez_compressed(p, **fp)
        written.append((p, 'arrays_npz',
                        'measured gauge series over the two-stage window'))

        main_fig = os.path.join(
            outdir, 'fig06_synthetic_v2.png' if name == 'fig6_v2'
            else f'fig06_synthetic_{name}_v2.png')
        figure_main(main_fig, cfg, name, x, pw, gauges, gidx, md_table, jval,
                    srts=srt, ta1=ta1, tr1=tr1, ta2=ta2, tr2=tr2, ta3=ta3,
                    tr3=tr3, dp=dp, srcs=srcs, img1=img1, img2=img2, img3=img3,
                    hits7=hits7, hits8=hits8, field=field, metrics=metrics,
                    md_lo=md_lo, md_hi=md_hi, r_head=r_head, r_seal=r_seal,
                    r_uni=r_uni, thr_1pt=thr_1pt, thr_lw=thr_lw, r_crit=r_crit)
        written.append((main_fig, 'figure_png',
                        'the rebuilt Fig. 6 synthetic panel + ratio ladder'))

        if want_img and len(img3) >= 2:
            wf = os.path.join(outdir, f"fig06_waterfall_compare_{name}_v2.png")
            figure_waterfall(wf, cfg, x, pw, img3, srcs['phase3'], hits7, hits8,
                             gauges, md_table, md_lo, md_hi, r_seal)
            written.append((wf, 'figure_png',
                            'phase-3 synthetic strain rate at three ratios'))

        # ---- manifest -------------------------------------------------------
        results = {
            'mesh': mesh_rec,
            'phase_boundaries': win_rec,
            'panel_window_md_ft': [md_lo, md_hi],
            'gamma_psi_inv': gamma,
            'metrics': metrics,
            'archive_comparison': arch,
            'phases': {
                'phase1': phase_result(s1, trec1, ex1, tr1, gauges, None),
                'phase2': phase_result(s2, trec2, ex2, tr2, gauges, None),
                **{f"phase3@ratio={r:g}": phase_result(
                    s3, trec3[r], ex3[r], tr3[r], gauges, breports[r])
                   for r in srt},
            },
            'what_changed_from_the_archived_figure': CHANGE_TABLE(cfg),
            'phases_1_2_solved_once':
                ('barrier_from_stage is null for phases 1 and 2, so they do not '
                 'depend on the ratio; they are solved once per study instead of '
                 'once per ratio. Verified by construction (the diffusivity '
                 'passed to both is the uniform baseline array) and by the '
                 'phase-3 initial condition being one shared array.'),
        }

        # sources: per-phase groups, so the duplicate-index check is per solve
        src_groups, labels = [], []
        for ph in PHASES:
            s = srcs[ph]
            drv = rm.driver_record(
                kind='gauge_series',
                baseline_removal=cfg['source']['baseline_removal'],
                value_units=cfg['source']['value_units'],
                series_path=rd.repo_path(s['series_path']),
                gauge_number=s['gauge'], gauge_md_ft=s['md_ft'],
                taxis=s['taxis_s'], values=s['values_psi'],
                time_start=s['window_abs'][0], time_end=s['window_abs'][1])
            grp = [rm.source_record(
                x, md_requested_ft=s['frac_hit_mds_ft'][j], mesh_idx=i,
                driver=drv, label=f"{ph}:stage{s['frac_hit_stage']}_frachit{j}",
                excluded_from_misfit=True, index_in_source_list=j)
                for j, i in enumerate(s['source_idx'])]
            src_groups.append(grp)
            labels.append(ph)

        source_group = rm.source_protocol(
            application=cfg['source']['application'],
            solver_class=ex1['solver'],
            placement_rule=cfg['source']['placement_rule'],
            sources=src_groups, phase_labels=labels,
            targets={'gauges': gauges,
                     'md_ft': [md_table.md_of(g) for g in gauges],
                     'validation_gauge': val_gauge,
                     'role': ('observation points for a FORWARD run; nothing is '
                              'fitted here. Gauges 6 and 7 are also the phase-1/2 '
                              'and phase-3 Dirichlet drivers, so they are boundary '
                              'conditions displayed as data (D4).')},
            time_level='n' if float(pcfg['theta']) == 1.0 else 'n+1',
            phase_chaining={
                'order': list(PHASES),
                'rule': ('each phase starts from the previous phase FINAL spatial '
                         'profile; t0 = 0 in every phase, as 101 does'),
                'barrier_source_stage': 7, 'barrier_applies_in': 'phase3',
                'phases_1_2_are_ratio_independent': True},
            boundary_conditions=cfg['source']['boundary_conditions'])

        time_records = [rm.time_record(ta1, label='phase1', **trec1),
                        rm.time_record(ta2, label='phase2', **trec2)]
        time_records += [rm.time_record(ta3, label=f"phase3@ratio={r:g}",
                                        **trec3[r]) for r in srt]

        blist = []
        for r in srt:
            brep = breports[r]
            if brep is None:
                continue
            for b in brep['barriers']:
                mask = np.zeros(len(x), dtype=bool)
                mask[b['i0']:b['i1'] + 1] = True
                blist.append(rm.barrier_record(
                    x, mask, label=f"stage7_frachit_MD{b['md_ft']:.2f}@ratio={r:g}",
                    centre_md_ft=b['md_ft'],
                    w_requested_ft=float(bcfg['w_ft']), ratio=r, d_baseline=D0,
                    report=b))

        numerics = rm.numerics(
            time=time_records,
            mesh=rm.mesh_record(
                x, dx_requested_ft=float(cfg['mesh']['dx_ft']),
                window_md_ft=(float(cfg['mesh']['md_lo_ft']),
                              float(cfg['mesh']['md_hi_ft'])),
                pad_low_ft=float(cfg['mesh']['pad_low_ft']),
                pad_high_ft=float(cfg['mesh']['pad_high_ft']),
                refinement=mesh_rec),
            interface_avg=pcfg['interface_avg'],
            boundary=cfg['source']['boundary_conditions'],
            diffusivity={
                'baseline_D_ft2_s': D0,
                'profile_family': 'uniform_plus_physical_width_barriers',
                'param_names': ['D_baseline', 'ratio', 'w_half_width_ft'],
                'params': [D0, srt, float(bcfg['w_ft'])],
                'D_min': float(np.min(dprof_head)), 'D_max': float(np.max(dprof_head)),
                'D_sha256': rm.sha256_array(dprof_head),
                'profile_anchor': 'physical_md',
                'note': (f'D_min/D_max/D_sha256 describe the phase-3 profile at the '
                         f'HEADLINE ratio {r_head:g}; phases 1-2 are uniform')},
            barriers=blist if blist else rm.NONE_DECLARED,
            leakage=rm.NONE_DECLARED,
            kernel={'name': ex1['solver'], 'banded': True,
                    'equivalence_reference':
                        'output/rev2_20260901/A4/selftest_output.txt',
                    'note': ('rev2_core at theta=1 / harmonic / lambda=0 is bitwise '
                             'identical to the verified R1 kernel, which is '
                             'bit-equivalent to fibeRIS')},
            rng=rm.NONE_DECLARED,
            parallel={'mode': 'single_process',
                      'why': ('each phase solve is 3-30 s; BLAS threads are capped '
                              'by OMP_NUM_THREADS at the shell, no Pool is used')},
            amplification=rc.amplification_factor(
                x, dprof_head, float(tcfg['dt_fixed_s']), float(pcfg['theta']),
                interface_avg=pcfg['interface_avg']))

        inputs = [(rd.repo_path(rd.SWELL_GAUGE_TEMPLATE.format(n=g)),
                   'gauge_series', f'gauge{g}_swell') for g in sorted(set(gauges) | {6, 7})]
        inputs += [(rd.repo_path(rd.SWELL_FRAC_HIT_TEMPLATE.format(stage=s)),
                    'geometry', f'frac_hit_stage_{s}') for s in (7, 8)]
        inputs += [(rd.repo_path(rd.SWELL_GAUGE_MD_NPZ), 'geometry',
                    'gauge_md_swell')]
        for s in (7, 8):
            inputs.append((rd.repo_path(rd.PUMPING_DIR_TEMPLATE.format(stage=s),
                                        rd.PUMPING_CURVE_FILES['slurry_rate']),
                           'pumping', f'stage{s}_slurry_rate'))
        if cfg.get('compare_to_archive'):
            adir = root_cfg['archive']['dir']
            inputs.append((os.path.join(
                adir, root_cfg['archive']['phase3_plotted_by_the_manuscript']),
                'prior_run_output', 'archive:phase3_test.npz'))

        R.declare_inputs(inputs)
        for pth, role, note in written:
            R.declare_output(pth, role=role, note=note,
                             dpi=(int(cfg['outputs']['figure_dpi'])
                                  if role == 'figure_png' else None))
        R.set_source(source_group)
        R.set_numerics(numerics)
        R.set_results(results)
        R.note(f"E1 study '{name}': rebuild of the Fig. 6 synthetic panel. "
               f"D = {D0:g} ft^2/s, barrier half-width w = {bcfg['w_ft']:g} ft "
               f"(full width {2 * float(bcfg['w_ft']):g} ft), headline ratio "
               f"{r_head:g}, fixed dt = {tcfg['dt_fixed_s']:g} s, pad "
               f"{cfg['mesh']['pad_low_ft']:g}/{cfg['mesh']['pad_high_ft']:g} ft, "
               f"uniform dx = {cfg['mesh']['dx_ft']:g} ft.")
        R.note("The file the manuscript currently plots, "
               "output/0211_simulation_MULTIstage/phase3_test.npz, measures as "
               "barrier ratio 1.0 (NO barrier): A5 matched it to the uniform chain "
               "to 2.18e-05 psi. output/0211_simulation_MULTIstage is read-only "
               "here and is opened only through rev2_layout.load_panel.")
        R.note("Synthetic strain rate uses Gamma = 8.94e-9 psi^-1 and is in s^-1. "
               "The legacy path data*6894.76/30e9 (= P/E = 2.298e-7 strain/psi) is "
               "~26x too large and is NOT used. E3 owns the legacy scripts.")
        R.note("Phases 1 and 2 carry NO barrier, so they are solved once per study "
               "rather than once per ratio; every phase-3 solve starts from the "
               "same stored final profile of phase 2.")

    return {'name': name, 'outdir': outdir, 'metrics': metrics, 'cfg': cfg,
            'ratios': srt, 'gauges': gauges,
            'wall_s': time.time() - t_study,
            'vs_sealed': {r: per_ratio[r]['vs_sealed_worst_gauge_psi'] for r in srt},
            'vs_uniform': {r: per_ratio[r]['vs_uniform_worst_gauge_psi'] for r in srt},
            'thr_1pt': thr_1pt, 'thr_lw': thr_lw, 'r_crit': r_crit,
            'traces_by_ratio': {r: dp[r] for r in srt}, 'ta3': ta3}


def phase_result(src, trec, extra, traces, gauges, brep):
    return {
        'window_abs': src['window_abs'],
        'source': {k: v for k, v in src.items()
                   if k not in ('taxis_s', 'values_psi')},
        'solver': extra,
        'time': {k: trec[k] for k in ('mode', 'theta', 'dt_requested_s',
                                      't_total_requested_s')
                 if k in trec},
        'barrier_report': (None if brep is None
                           else {k: v for k, v in brep.items() if k != 'barriers'}),
        'gauge_peak_dp_psi': {f"g{g}": float(np.max(np.abs(traces[:, j] - traces[0, j])))
                              for j, g in enumerate(gauges)},
    }


def CHANGE_TABLE(cfg):
    return {
        'barrier_ratio': {
            'archived_file_phase3_test_npz': 1.0,
            'archived_file_nominal_in_101_line_214': 1e-6,
            'manuscript_caption': 1e-5,
            'this_run_headline': float(cfg['barrier']['headline_ratio']),
            'evidence': 'A5 (2.18e-05 psi match to the uniform chain), D4 (0.3-1)'},
        'barrier_definition': {
            'archived': 'one mesh node on the legacy refined mesh; realised full '
                        'width 0.13333 ft, and converges to ZERO under refinement',
            'this_run': f"physical half-width w = {cfg['barrier']['w_ft']} ft via "
                        f"rev2_core.build_barrier_profile; realised full width "
                        f"{2 * float(cfg['barrier']['w_ft'])} ft, mesh-invariant",
            'evidence': 'A1 / CORRECTION 4'},
        'domain': {
            'archived': 'MD 12500-17999, no padding',
            'this_run': (f"MD {12500 - cfg['mesh']['pad_low_ft']:.0f}-"
                         f"{17999 + cfg['mesh']['pad_high_ft']:.0f}, "
                         f"{cfg['mesh']['pad_low_ft']:.0f} ft pad at BOTH ends"),
            'evidence': 'B2: worst plotted gauge moves 25.33 psi (6.56 %) at ratio '
                        '1e-3, dominated by the HIGH-MD end'},
        'time_stepping': {
            'archived': "fibeRIS optimizer=True, which realises fixed dt = 30 s "
                        "(0 rejections, >99 % of steps at max_dt)",
            'this_run': f"fixed dt = {cfg['time']['dt_fixed_s']} s, theta = 1",
            'evidence': 'A3: at dt = 30 s BE and CN differ by 3.49 % of peak'},
        'mesh_refinement': {
            'archived': "fiberis.utils.mesh_utils.refine_mesh, defect included "
                        "(dx = 2/15 ft where 0.2 ft was intended)",
            'this_run': f"none; uniform dx = {cfg['mesh']['dx_ft']} ft",
            'why': 'the refinement existed only because the barrier was an index'},
        'strain_conversion': {
            'archived': 'data * 6894.76 / 30e9 = 2.298e-7 strain/psi, axis labelled '
                        'microstrain',
            'this_run': f"Gamma = {cfg['panel']['gamma_psi_inv']:g} psi^-1, axis in "
                        f"s^-1",
            'evidence': 'house rules; the legacy factor is ~26x too large'},
        'baseline_D_ft2_s': {
            'archived': 140.0, 'this_run': float(cfg['physics']['D_baseline_ft2_s']),
            'note': ('101:93 hard-codes 140 with no stated provenance; the '
                     'recalibrated values are 1150 (absolute norm) / 550 '
                     '(amplitude-normalised). The deliverable study keeps 140 so '
                     'that the barrier is the only intended physical change.')},
    }


# ---------------------------------------------------------------------------
# field data and archive
# ---------------------------------------------------------------------------

def load_field_gauges(gauges, wins):
    """Measured gauge series over the whole two-stage window, absolute psi."""
    t0 = wins['phase1'][0]
    t1 = wins['phase3'][1]
    win = rd.Window(md_min_ft=0.0, md_max_ft=1e9, t_start=t0, t_end=t1)
    gw = rd.load_window_gauges(win, gauges=[int(g) for g in gauges],
                               baseline='none', rebase='per_gauge')
    out = {}
    for g in gauges:
        s = gw.series[int(g)]
        out[int(g)] = {'taxis_s': np.asarray(s.taxis_s, float),
                       'psi': np.asarray(s.raw_psi, float),
                       't0_abs': s.t0_abs, 'md_ft': float(s.md_ft)}
    return out


def compare_archive(root_cfg, x, gidx, gauges, md_table, ta3, dp, tr3,
                    r_head, r_uni):
    """Read-only: what the manuscript's own phase-3 file gives at these gauges."""
    path = os.path.join(root_cfg['archive']['dir'],
                        root_cfg['archive']['phase3_plotted_by_the_manuscript'])
    panel = rl.load_panel(path)
    rows = {}
    for j, g in enumerate(gauges):
        md = md_table.md_of(int(g))
        i_node, md_node, tr = rl.trace_at_md(panel, md)
        tr = np.asarray(tr, dtype=float)
        d_arch = tr - tr[0]
        t_arch = np.asarray(panel.taxis, dtype=float)
        common = np.linspace(0.0, min(float(t_arch[-1]), float(ta3[-1])), 2001)
        a = np.interp(common, t_arch, d_arch)
        b_head = np.interp(common, ta3, dp[r_head][:, j])
        b_uni = np.interp(common, ta3, dp[r_uni][:, j])
        rows[f"g{g}"] = {
            'md_ft': float(md), 'archive_node_md_ft': float(md_node),
            'archive_peak_dp_psi': float(np.max(d_arch)),
            'archive_trough_dp_psi': float(np.min(d_arch)),
            'new_headline_peak_dp_psi': float(np.max(dp[r_head][:, j])),
            'new_uniform_peak_dp_psi': float(np.max(dp[r_uni][:, j])),
            'max_abs_diff_archive_vs_new_headline_psi': float(np.max(np.abs(a - b_head))),
            'max_abs_diff_archive_vs_new_uniform_psi': float(np.max(np.abs(a - b_uni))),
        }
    return {
        'archive_file': rm._rel(path),
        'on_disk_layout': panel.detected_layout,
        'shape': list(panel.data.shape),
        'note': ("phase3_test.npz measures as barrier ratio 1.0 (A5, 2.18e-05 psi "
                 "against the uniform chain). The residual 'archive vs new uniform' "
                 "difference below is NOT a barrier effect: it is the combined "
                 "effect of the 5000/5000 ft padding, dt 30 s -> 1 s, the uniform "
                 "mesh replacing the legacy refine_mesh, and today's crop rebasing "
                 "convention. The 'archive vs new headline' column is that PLUS the "
                 "barrier the manuscript says is there and the file does not have."),
        'per_gauge': rows,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

_PHASE_COLS = ('#1f77b4', '#7f7f7f', '#d62728')


def _plot_waterfall(ax, md_panel, images, vmax, cmap='bwr'):
    im = None
    for td, t0abs, arr in images:
        tt = abs_times(t0abs, td)
        im = ax.pcolormesh(tt, md_panel, arr.T, cmap=cmap, shading='nearest',
                           vmin=-vmax, vmax=vmax, rasterized=True)
    return im


def figure_main(path, cfg, name, x, pw, gauges, gidx, md_table, jval, *, srts,
                ta1, tr1, ta2, tr2, ta3, tr3, dp, srcs, img1, img2, img3,
                hits7, hits8, field, metrics, md_lo, md_hi, r_head, r_seal,
                r_uni, thr_1pt, thr_lw, r_crit):
    D0 = float(cfg['physics']['D_baseline_ft2_s'])
    w = float(cfg['barrier']['w_ft'])
    md_panel = x[pw]
    have_img = bool(img3) and r_head in img3
    scale = float(cfg['panel']['manuscript_plot_scale_ft_per_psi'])

    fig = plt.figure(figsize=(13.5, 16.0))
    gs = GridSpec(4, 2, height_ratios=[1.5, 1.0, 1.0, 1.0], hspace=0.34,
                  wspace=0.22, left=0.075, right=0.975, top=0.935, bottom=0.045)

    # ---- (a) the synthetic panel -----------------------------------------
    ax = fig.add_subplot(gs[0, :])
    if have_img:
        allsr = np.concatenate([img1[0].ravel(), img2[0].ravel(),
                                img3[r_head][0].ravel()])
        vmax = float(np.percentile(np.abs(allsr), 99.5))
        im = _plot_waterfall(ax, md_panel,
                             [(img1[1], srcs['phase1']['t0_abs'], img1[0]),
                              (img2[1], srcs['phase2']['t0_abs'], img2[0]),
                              (img3[r_head][1], srcs['phase3']['t0_abs'],
                               img3[r_head][0])], vmax)
        cb = fig.colorbar(im, ax=ax, pad=0.012, aspect=30)
        cb.set_label(r'synthetic strain rate (s$^{-1}$)', fontsize=8.5)
        cb.ax.tick_params(labelsize=7.5)
        cb.ax.yaxis.get_offset_text().set_fontsize(7.5)
    for h in hits7:
        ax.axhline(float(h), color='#2ca02c', lw=0.8, alpha=0.55, zorder=2)
    for h in hits8:
        ax.axhline(float(h), color='#ff7f0e', lw=0.8, ls=':', alpha=0.6, zorder=2)
    for j, g in enumerate(gauges):
        md = md_table.md_of(int(g))
        ax.axhline(md, color='0.25', ls='--', lw=0.6, zorder=3)
        for ta, tr, t0a in ((ta1, tr1, srcs['phase1']['t0_abs']),
                            (ta2, tr2, srcs['phase2']['t0_abs']),
                            (ta3, tr3[r_head], srcs['phase3']['t0_abs'])):
            col = 'k' if j == jval else '#00204d'
            lw = 2.0 if j == jval else 1.1
            ax.plot(abs_times(t0a, ta), (tr[:, j] - tr[0, j]) * -scale + md,
                    color=col, lw=lw, zorder=4)
        ax.text(abs_times(srcs['phase1']['t0_abs'], ta1[:1])[0], md + 20,
                f"g{g} MD {md:.0f}" + ("  (held-out)" if j == jval else ""),
                fontsize=8, color='k', zorder=5)
    # phase boundaries and a scale bar, so the trace amplitudes are readable
    for ph in ('phase2', 'phase3'):
        ax.axvline(abs_times(srcs[ph]['t0_abs'], np.zeros(1))[0], color='0.15',
                   lw=1.0, zorder=5)
    t_bar = abs_times(srcs['phase2']['t0_abs'], np.array([600.0, 600.0]))
    ax.plot(t_bar, [md_lo + 60.0, md_lo + 60.0 + 500.0 * scale], color='k',
            lw=3.5, solid_capstyle='butt', zorder=6)
    ax.text(t_bar[0], md_lo + 60.0 + 500.0 * scale + 8, '500 psi', fontsize=8,
            zorder=6)
    ax.set_ylim(md_lo, md_hi)
    ax.set_ylabel('measured depth (ft)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_title(
        f"(a) rebuilt Fig. 6 synthetic panel.  phase 1 (stage-7 injection) | "
        f"phase 2 (shut-in) | phase 3 (stage-8 injection, barrier ratio "
        f"{r_head:g}).\n"
        f"D = {D0:g} ft$^2$/s; six stage-7 barriers, physical full width "
        f"{2 * w:g} ft (green); stage-8 Dirichlet sources (orange dotted);\n"
        f"fixed dt = {cfg['time']['dt_fixed_s']:g} s; pad "
        f"{cfg['mesh']['pad_low_ft']:.0f}/{cfg['mesh']['pad_high_ft']:.0f} ft; "
        f"uniform dx = {cfg['mesh']['dx_ft']:g} ft.  Traces are simulated "
        f"pressure at {scale:g} ft/psi, the scale 104 uses.",
        fontsize=9, loc='left')
    if not have_img:
        ax.text(0.5, 0.5, 'strain-rate image not saved for this study',
                transform=ax.transAxes, ha='center', fontsize=11, color='0.4')

    # ---- (b) field vs synthetic at the held-out gauge ---------------------
    ax = fig.add_subplot(gs[1, :])
    gv = gauges[jval]
    fd = field[gv]
    ax.plot(abs_times(fd['t0_abs'], fd['taxis_s']), fd['psi'], color='k', lw=1.2,
            label=f"field data, g{gv} MD {fd['md_ft']:.0f} ft (held out)")
    for k, (ta, tr, t0a, lab) in enumerate(
            ((ta1, tr1, srcs['phase1']['t0_abs'], 'phase 1'),
             (ta2, tr2, srcs['phase2']['t0_abs'], 'phase 2'),
             (ta3, tr3[r_head], srcs['phase3']['t0_abs'],
              f'phase 3, ratio {r_head:g}'))):
        ax.plot(abs_times(t0a, ta), tr[:, jval], color=_PHASE_COLS[k], lw=1.3,
                ls='--', label=f"synthetic, {lab}")
    ax.plot(abs_times(srcs['phase3']['t0_abs'], ta3), tr3[r_uni][:, jval],
            color='#9467bd', lw=1.0, ls=':',
            label='synthetic, phase 3, ratio 1 (no barrier) - the archived case')
    ax.set_ylabel('pressure (psi)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc='best')
    dmax = max(abs(metrics['per_ratio'][f"{r:g}"]['vs_uniform_max_abs_psi'][f"g{gv}"])
               for r in srts)
    ax.set_title(
        f"(b) the manuscript's single Field-vs-Synthetic panel, gauge {gv}. "
        f"Over the WHOLE ratio ladder ({min(srts):g} to {max(srts):g}) this gauge "
        f"moves by {dmax:.2e} psi: the validation panel is blind to the barrier.",
        fontsize=9.5, loc='left')

    # ---- (c) the ratio ladder at the shielded gauges ----------------------
    ax = fig.add_subplot(gs[2, :])
    shield = [g for g in gauges if md_table.md_of(int(g)) > float(np.min(hits7))]
    cols = plt.cm.viridis(np.linspace(0.0, 0.9, len(srts)))
    tt3 = abs_times(srcs['phase3']['t0_abs'], ta3)
    for k, r in enumerate(srts):
        for m, g in enumerate(shield):
            j = gauges.index(g)
            ax.plot(tt3, dp[r][:, j], color=cols[k], lw=1.4,
                    ls=('-' if m == 0 else '--'),
                    label=(rlabel(r, r_seal) if m == 0 else None))
    ax.set_ylabel('phase-3 $\\Delta P$ from its own start (psi)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=4)
    ax.set_title(
        f"(c) phase 3 at the only gauges the barrier can reach: "
        f"{', '.join('g%d' % g for g in shield)} "
        f"(solid = g{shield[0]}, dashed = the rest). Every other plotted gauge is "
        f"below the stage-8 sources and is identical at every ratio.",
        fontsize=9.5, loc='left')

    # ---- (d) informativeness ---------------------------------------------
    ax = fig.add_subplot(gs[3, 0])
    ru = [r for r in srts if r != r_uni]
    rs_ = [r for r in srts if r != r_seal]
    vs_u = [metrics['per_ratio'][f"{r:g}"]['vs_uniform_worst_gauge_psi'] for r in ru]
    vs_s = [metrics['per_ratio'][f"{r:g}"]['vs_sealed_worst_gauge_psi'] for r in rs_]
    ax.loglog(ru, vs_u, 'o-', color='#1f77b4',
              label='separation from ratio 1 (no barrier)')
    ax.loglog(rs_, vs_s, 's-', color='#d62728',
              label=f'separation from a perfect seal (ratio {r_seal:g})')
    ax.axhline(thr_1pt, color='0.35', ls='--', lw=1.0,
               label=f'1 pt on the manuscript panel = {thr_1pt:.1f} psi')
    ax.axhline(thr_lw, color='0.35', ls=':', lw=1.0,
               label=f"{cfg['panel']['manuscript_linewidth_pt']:g} pt "
                     f"(its own line width) = {thr_lw:.1f} psi")
    lo = min([v for v in vs_u + vs_s if v > 0] + [thr_1pt]) / 4.0
    hi = max(vs_u + vs_s + [thr_lw]) * 6.0
    ax.set_ylim(lo, hi)
    ax.axvline(r_head, color='k', lw=1.2)
    ax.text(r_head, hi / 2.5, f' manuscript text: {r_head:g}', rotation=90,
            fontsize=8, va='top')
    ax.axvline(r_crit, color='#2ca02c', lw=1.2, ls='-.')
    ax.text(r_crit, lo * 1.6, f' $W_{{tot}}^2/(D_0 T_3)$ = {r_crit:.2e}',
            rotation=90, fontsize=8, color='#2ca02c')
    ax.set_xlabel('barrier reduction ratio  $D_b/D_0$')
    ax.set_ylabel('worst plotted gauge, max $|\\Delta P|$ difference (psi)')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7.0, loc='lower right', framealpha=0.92)
    ax.set_title('(d) what the figure can actually resolve', fontsize=9.5, loc='left')

    # ---- (e) what one decade of ratio is worth ---------------------------
    ax = fig.add_subplot(gs[3, 1])
    keys = list(metrics['adjacent_ladder_separation'])
    mids, vals = [], []
    for kk in keys:
        a, b = kk.split('->')
        a, b = float(a), float(b)
        d = metrics['adjacent_ladder_separation'][kk]
        if d['decades'] <= 0:
            continue
        mids.append(np.sqrt(a * b))
        vals.append(d['worst_gauge_psi'] / d['decades'])
    ax.loglog(mids, [max(v, 1e-4) for v in vals], 'D-', color='#8c564b')
    ax.axhline(thr_1pt, color='0.35', ls='--', lw=1.0)
    ax.axhline(thr_lw, color='0.35', ls=':', lw=1.0)
    ax.axvline(r_head, color='k', lw=1.2)
    ax.set_xlabel('barrier reduction ratio (ladder mid-point)')
    ax.set_ylabel('psi per decade of ratio, worst plotted gauge')
    ax.grid(alpha=0.3, which='both')
    ax.set_title('(e) the ratio is identifiable only where this is large',
                 fontsize=9.5, loc='left')

    fig.suptitle(
        f"E1 - Fig. 6 two-stage synthetic panel, REBUILT.  study '{name}'.  "
        f"The archived file the manuscript plots (phase3_test.npz) contains NO "
        f"barrier (measured ratio 1.0).",
        fontsize=11.5)
    fig.savefig(path, dpi=int(cfg['outputs']['figure_dpi']))
    plt.close(fig)


def figure_waterfall(path, cfg, x, pw, img3, src3, hits7, hits8, gauges,
                     md_table, md_lo, md_hi, r_seal):
    md_panel = x[pw]
    rs = sorted(img3, reverse=True)
    allsr = np.concatenate([img3[r][0].ravel() for r in rs])
    vmax = float(np.percentile(np.abs(allsr), 99.5))
    fig, axes = plt.subplots(len(rs), 1, figsize=(12, 3.4 * len(rs)),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, rs):
        im = _plot_waterfall(ax, md_panel,
                             [(img3[r][1], src3['t0_abs'], img3[r][0])], vmax)
        for h in hits7:
            ax.axhline(float(h), color='#2ca02c', lw=0.8, alpha=0.55)
        for h in hits8:
            ax.axhline(float(h), color='#ff7f0e', lw=0.8, ls=':', alpha=0.6)
        for g in gauges:
            ax.axhline(md_table.md_of(int(g)), color='0.2', ls='--', lw=0.5)
        ax.set_ylabel('MD (ft)')
        ax.set_title(f"phase 3, {rlabel(r, r_seal)}", fontsize=10, loc='left')
        cb = fig.colorbar(im, ax=ax, pad=0.01, aspect=18)
        cb.set_label(r'strain rate (s$^{-1}$)', fontsize=8)
        cb.ax.tick_params(labelsize=7.5)
        cb.ax.yaxis.get_offset_text().set_fontsize(7.5)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].set_xlabel('2020-03-18 (naive local time, as stored)')
    axes[0].set_ylim(md_lo, md_hi)
    fig.suptitle('E1 - phase-3 synthetic strain rate, same colour scale. '
                 'Green = the six stage-7 barriers, orange dotted = the stage-8 '
                 'Dirichlet sources.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=int(cfg['outputs']['figure_dpi']))
    plt.close(fig)


def figure_cross(path, summaries, dpi):
    """One panel: how the informative window moves with D and with the width.

    The self-reference points are DROPPED rather than clipped to a floor: the
    separation of the sealed run from itself, and of the uniform run from itself,
    are exactly 0 psi, and a log axis would otherwise draw them at whatever floor
    was chosen and invite the reader to read a value off them.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    cols = plt.cm.tab10(np.linspace(0, 0.9, max(len(summaries), 2)))
    for k, (nm, s) in enumerate(sorted(summaries.items())):
        rs = s['ratios']
        cfg = s['cfg']
        lab = (f"{nm}: D = {cfg['physics']['D_baseline_ft2_s']:g}, "
               f"2w = {2 * float(cfg['barrier']['w_ft']):g} ft, "
               f"dx = {cfg['mesh']['dx_ft']:g} ft")
        rsl = [r for r in rs if s['vs_sealed'][r] > 0]
        rul = [r for r in rs if s['vs_uniform'][r] > 0]
        axes[0].loglog(rsl, [s['vs_sealed'][r] for r in rsl], 'o-',
                       color=cols[k], label=lab)
        axes[1].loglog(rul, [s['vs_uniform'][r] for r in rul], 's-',
                       color=cols[k], label=lab)
        axes[0].axvline(s['r_crit'], color=cols[k], ls='-.', lw=1.0)
    for ax, ttl in ((axes[0], 'separation from a PERFECT SEAL\n'
                              '(right of the curve = the value of the ratio is '
                              'visible; dash-dot = $W_{tot}^2/(D_0T_3)$)'),
                    (axes[1], 'separation from NO BARRIER\n'
                              '(left of the curve = a barrier is visible at all)')):
        ax.axhline(summaries[list(summaries)[0]]['thr_1pt'], color='0.3', ls='--',
                   lw=1.0, label='1 pt on the manuscript panel')
        ax.set_xlabel('barrier reduction ratio $D_b/D_0$')
        ax.set_ylabel('worst plotted gauge, max $|\\Delta P|$ (psi)')
        ax.grid(alpha=0.3, which='both')
        ax.set_title(ttl, fontsize=9.5)
        ax.legend(fontsize=7.5)
    fig.suptitle('E1 - where the Fig. 6 barrier is observable, across the '
                 'baseline diffusivity, the barrier width and the mesh',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------


def replot_cross(root_cfg, out_root, tag, config_path):
    """Redraw the cross-study figure from the metrics the four studies wrote.

    Derived view only: no solve, no new numbers. The four per-study manifests are
    the authority; their sha256 and the sha256 of the metrics JSONs actually read
    are written next to the figure so it can be traced back to them.
    """
    summaries, prov = {}, {}
    for st in root_cfg['studies']:
        nm = st['name']
        d = os.path.join(out_root, nm)
        mp = os.path.join(d, f"e1_ratio_metrics_{nm}_v2.json")
        mf = os.path.join(d, 'manifest.json')
        if not (os.path.exists(mp) and os.path.exists(mf)):
            print(f"[E1] replot: skipping {nm} (no metrics/manifest on disk)")
            continue
        with open(mp) as fh:
            m = json.load(fh)['metrics']
        cfg = a5.deep_merge(root_cfg['base'], st.get('overrides'))
        rs = sorted(float(k) for k in m['per_ratio'])
        summaries[nm] = {
            'ratios': rs, 'cfg': cfg,
            'vs_sealed': {r: m['per_ratio'][f"{r:g}"]['vs_sealed_worst_gauge_psi']
                          for r in rs},
            'vs_uniform': {r: m['per_ratio'][f"{r:g}"]['vs_uniform_worst_gauge_psi']
                           for r in rs},
            'thr_1pt': m['plot_resolution']['threshold_1_point_psi'],
            'thr_lw': m['plot_resolution']['threshold_one_manuscript_linewidth_psi'],
            'r_crit': m['informative_window']['analytic_lower_edge_from_crossing_time'],
        }
        prov[nm] = {'metrics_json': rm._rel(mp), 'metrics_sha256': rm.sha256_file(mp),
                    'manifest': rm._rel(mf), 'manifest_sha256': rm.sha256_file(mf)}
    if len(summaries) < 2:
        raise SystemExit("replot needs at least two studies' metrics on disk")
    p = os.path.join(out_root, f"fig06_informative_window_{tag}.png")
    pj = os.path.join(out_root, f"fig06_informative_window_{tag}.provenance.json")
    rm.assert_absent([p, pj])
    figure_cross(p, summaries, int(root_cfg['base']['outputs']['figure_dpi']))
    with open(pj, 'w') as fh:
        json.dump({'figure': rm._rel(p), 'tag': tag,
                   'kind': 'derived_view_no_solve',
                   'config': rm._rel(config_path),
                   'config_sha256': rm.sha256_file(config_path),
                   'code': rm._rel(os.path.abspath(__file__)),
                   'code_sha256': rm.sha256_file(os.path.abspath(__file__)),
                   'generated_utc':
                       datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'sources': prov,
                   'note': ('No solver run: every number is read from the four '
                            'manifested studies listed above. House rule 3 is '
                            'satisfied by those manifests.')},
                  fh, indent=2, sort_keys=True)
    print(f"[E1] cross-study figure -> {p}")
    print(f"[E1] provenance         -> {pj}")
    return 0


def fit_check(root_cfg, out_root, tag, config_path):
    """Does the barrier improve the phase-3 model-data agreement, or damage it?

    Post-hoc and solve-free: it reads the `e1_gauge_traces_*` and
    `e1_field_gauges_*` arrays the manifested studies already wrote. B2 found that
    for Fig. 7b the UNIFORM run fits the field better than the barrier run, so the
    same question has to be asked of Fig. 6 before any caption claims the barrier
    is what the data show.

    Two misfits are reported and both are named, because they answer different
    questions:
      * `abs_psi` -- synthetic minus measured absolute pressure, the quantity the
        manuscript's ax4 panel actually draws. It carries the whole accumulated
        offset of the three-phase chain, so it is dominated by phase 1 and 2.
      * `delta_from_phase3_start` -- both series referenced to their own first
        phase-3 sample. This is the barrier test: it removes the inherited offset
        and leaves only the phase-3 response.
    """
    out = {}
    prov = {}
    for st in root_cfg['studies']:
        nm = st['name']
        d = os.path.join(out_root, nm)
        tp = os.path.join(d, f"e1_gauge_traces_{nm}_v2.npz")
        fp = os.path.join(d, f"e1_field_gauges_{nm}_v2.npz")
        mf = os.path.join(d, 'manifest.json')
        if not (os.path.exists(tp) and os.path.exists(fp)):
            continue
        z, f = np.load(tp), np.load(fp)
        gauges = [int(g) for g in z['gauge_numbers']]
        ta3 = z['phase3_taxis_s']
        t0_3 = datetime.datetime.fromisoformat(str(z['phase3_t0_abs']))
        rows = {}
        for r in [float(v) for v in z['ratios']]:
            tr = z[f"phase3_traces_psi_{rtag(r)}"]
            per = {}
            for j, g in enumerate(gauges):
                t0f = datetime.datetime.fromisoformat(str(f[f"g{g}_t0_abs"]))
                tf = np.asarray(f[f"g{g}_taxis_s"], float) + (t0f - t0_3).total_seconds()
                pf = np.asarray(f[f"g{g}_psi"], float)
                keep = (tf >= float(ta3[0])) & (tf <= float(ta3[-1]))
                if keep.sum() < 10:
                    continue
                tf, pf = tf[keep], pf[keep]
                sim = np.interp(tf, ta3, tr[:, j])
                sim0 = np.interp(float(tf[0]), ta3, tr[:, j])
                per[f"g{g}"] = {
                    'n_samples': int(tf.size),
                    'rmse_abs_psi': float(np.sqrt(np.mean((sim - pf) ** 2))),
                    'rmse_delta_from_phase3_start_psi':
                        float(np.sqrt(np.mean(((sim - sim0) - (pf - pf[0])) ** 2))),
                }
            rows[f"{r:g}"] = {
                'per_gauge': per,
                'gauge_mean_rmse_abs_psi':
                    float(np.mean([v['rmse_abs_psi'] for v in per.values()])),
                'gauge_mean_rmse_delta_psi':
                    float(np.mean([v['rmse_delta_from_phase3_start_psi']
                                   for v in per.values()])),
                'shielded_mean_rmse_delta_psi':
                    float(np.mean([v['rmse_delta_from_phase3_start_psi']
                                   for k, v in per.items() if k in ('g5', 'g6')])),
            }
        out[nm] = rows
        prov[nm] = {'traces': rm._rel(tp), 'traces_sha256': rm.sha256_file(tp),
                    'field': rm._rel(fp), 'field_sha256': rm.sha256_file(fp),
                    'manifest': rm._rel(mf),
                    'manifest_sha256': rm.sha256_file(mf)}
    if not out:
        raise SystemExit("fit_check found no saved traces")
    p = os.path.join(out_root, f"e1_fit_vs_ratio_{tag}.json")
    rm.assert_absent([p])
    with open(p, 'w') as fh:
        json.dump({'kind': 'derived_view_no_solve', 'tag': tag,
                   'misfit_definitions': {
                       'abs_psi': ('synthetic minus measured ABSOLUTE pressure over '
                                   'the phase-3 window; what the manuscript ax4 '
                                   'panel draws; carries the accumulated offset of '
                                   'phases 1-2'),
                       'delta_from_phase3_start_psi':
                           ('both series referenced to their own first in-window '
                            'sample; this is the barrier test'),
                       'aggregation': 'gauge-mean RMSE, not sample-pooled (C1)'},
                   'sources': prov,
                   'config': rm._rel(config_path),
                   'code_sha256': rm.sha256_file(os.path.abspath(__file__)),
                   'generated_utc':
                       datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'studies': out}, fh, indent=2, sort_keys=True)
    print(f"[E1] fit check -> {p}")
    for nm, rows in out.items():
        print(f"  {nm}")
        for k in sorted(rows, key=lambda v: -float(v)):
            r = rows[k]
            print(f"    ratio {k:>8}  gauge-mean dRMSE {r['gauge_mean_rmse_delta_psi']:8.2f} "
                  f"psi | g5,g6 mean {r['shielded_mean_rmse_delta_psi']:8.2f} "
                  f"| gauge-mean absRMSE {r['gauge_mean_rmse_abs_psi']:8.2f}")
    return 0


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default='configs/rev2/e1_fig6.json')
    ap.add_argument('--study', action='append', default=None)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--record-tag', default='v2', metavar='TAG',
                    help="tag for the round-level roll-up JSON "
                         "(e1_record_<TAG>.json). Give a fresh tag when running a "
                         "second config into the same --outdir, so house rule 2 "
                         "does not refuse the roll-up after the studies have "
                         "already been written.")
    ap.add_argument('--fit-check', default=None, metavar='TAG',
                    help='post-hoc, no solve: does the barrier make the phase-3 '
                         'model-data agreement better or worse? Reads the saved '
                         'traces and field series of every study on disk.')
    ap.add_argument('--replot-cross', default=None, metavar='TAG',
                    help='regenerate ONLY the cross-study figure, tagged TAG '
                         '(e.g. v3), from the per-study metrics JSONs already on '
                         'disk. No solve is run and nothing is overwritten.')
    args = ap.parse_args(argv)

    if not os.path.isdir('scripts') or not os.path.isdir('fibeRIS'):
        raise SystemExit(f"run with CWD = repo root; CWD is {os.getcwd()}")

    root_cfg = a5.load_config(args.config)
    out_root = args.outdir or root_cfg['output_root']
    os.makedirs(out_root, exist_ok=True)

    if args.fit_check:
        return fit_check(root_cfg, out_root, args.fit_check, args.config)
    if args.replot_cross:
        return replot_cross(root_cfg, out_root, args.replot_cross, args.config)

    wanted = set(args.study) if args.study else None
    summaries = {}
    t_all = time.time()
    for st in root_cfg['studies']:
        nm = st['name']
        if wanted is not None and nm not in wanted:
            continue
        if wanted is None and not st.get('enabled', True):
            continue
        cfg = a5.deep_merge(root_cfg['base'], st.get('overrides'))
        cfg['compare_to_archive'] = bool(
            st.get('compare_to_archive', cfg.get('compare_to_archive', False)))
        cfg['_study'] = {'name': nm, 'what': st.get('_what')}
        outdir = os.path.join(out_root, nm)
        print(f"[E1] study {nm} -> {outdir}", flush=True)
        summaries[nm] = run_study(nm, cfg, root_cfg, outdir, args.config)
        print(f"[E1]   done in {summaries[nm]['wall_s']:.1f} s", flush=True)

    if len(summaries) >= 2:
        p = os.path.join(out_root,
                         f"fig06_informative_window_{args.record_tag}.png")
        rm.assert_absent([p])
        figure_cross(p, summaries, int(root_cfg['base']['outputs']['figure_dpi']))
        print(f"[E1] cross-study figure -> {p}")

    rec = {
        'study_id': STUDY_ID, 'task_id': TASK_ID,
        'round_tag': root_cfg['round_tag'],
        'run_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'wall_seconds_total': time.time() - t_all,
        'studies': {nm: {'outdir': s['outdir'], 'wall_s': s['wall_s'],
                         'D_baseline_ft2_s': s['cfg']['physics']['D_baseline_ft2_s'],
                         'barrier_half_width_ft': s['cfg']['barrier']['w_ft'],
                         'dx_ft': s['cfg']['mesh']['dx_ft'],
                         'metrics': s['metrics']}
                    for nm, s in summaries.items()},
    }
    p = os.path.join(out_root, f"e1_record_{args.record_tag}.json")
    rm.assert_absent([p])
    with open(p, 'w') as fh:
        json.dump(rm._jsonify(rec, max_array_len=64)[0], fh, indent=2,
                  sort_keys=True, ensure_ascii=False)
    print(f"[E1] record -> {p}")
    print(f"[E1] total {time.time() - t_all:.1f} s")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
