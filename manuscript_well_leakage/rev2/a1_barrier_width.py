"""A1 DELIVERABLE -- barrier width as a physical quantity: mesh-independence demo.

The A1 INVESTIGATION is already done (output/rev2_20260901/A1/README.md,
effective_widths.csv, A1_spec_build_barrier_profile.md; summarised as CORRECTION 4
in the house rules). Nothing here re-measures the legacy call sites. This script
produces the DELIVERABLE the task package asks for:

  fix w = 1.0 ft (half-width; requested full width 2w = 2.0 ft) and the ratio,
  sweep dx = 1.0 / 0.5 / 0.2 / 0.1 ft, forward-model gauges g2-g7 from a Dirichlet
  source at g1, and overlay the four dx cases per gauge for

    physical_w             rev2_core.build_barrier_profile, |x - x_hit| <= w   (THE FIX)
    physical_w_rounded_md  same, but centres rounded to integer MD             (worst-case alignment,
                                                                               and 101's own idiom)
    legacy_single_node     d[locate(x, md)] = D0*ratio                          (101/106, 102r, 103r)
    legacy_tent9           d[idx-4:idx+5] *= tent(1 ... ratio ... 1)            (103_*fatal_wrong*)
    no_barrier             uniform D                                            (control)

ACCEPTANCE: per-gauge RMSE(sim, obs) changes < 2 % between dx = 1.0 and dx = 0.1.
Nothing is tuned to make it pass. Where it fails the residual mesh dependence is
located and quantified instead.

Run from the repo root:
    python3 scripts/manuscript_well_leakage/rev2/a1_barrier_width.py

Owns: configs/rev2/a1_barrier_width.json, output/rev2_20260901/A1/deliverable/.
Imports the shared rev2 modules; edits none of them.
"""

import argparse
import datetime
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import rev2_core as core          # noqa: E402
import rev2_data as rd            # noqa: E402
import rev2_manifest as rm        # noqa: E402

DEFAULT_CONFIG = 'configs/rev2/a1_barrier_width.json'

# The four definitions that carry a ratio, in figure/legend order.
BARRIER_DEFS = ('physical_w', 'physical_w_rounded_md',
                'legacy_single_node', 'legacy_tent9')
DEF_LABEL = {
    'no_barrier': 'no barrier (uniform D)',
    'physical_w': 'physical $w$ = 1.0 ft  (the fix)',
    'physical_w_rounded_md': 'physical $w$, centres rounded to integer MD',
    'legacy_single_node': 'legacy: single node',
    'legacy_tent9': 'legacy: 9-node tent [idx-4:idx+5]',
}
DEF_COLOR = {'physical_w': '#1b6ca8', 'physical_w_rounded_md': '#3fa34d',
             'legacy_single_node': '#c1272d', 'legacy_tent9': '#e08214',
             'no_barrier': '#555555'}
DX_COLOR = {1.0: '#08306b', 0.5: '#2171b5', 0.2: '#6baed6', 0.1: '#c6dbef'}
DX_STYLE = {1.0: '-', 0.5: '--', 0.2: '-.', 0.1: ':'}


# ---------------------------------------------------------------------------
# config / small helpers
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as fh:
        return json.load(fh)


def _retag(path, tag):
    """Insert a version tag into an output filename, replacing an existing _v1."""
    d, b = os.path.split(path)
    if '_v1.' in b:
        b = b.replace('_v1.', '_%s.' % tag)
    else:
        root, ext = os.path.splitext(b)
        b = '%s_%s%s' % (root, tag, ext)
    return os.path.join(d, b)


def dx_tag(dx):
    return ('%g' % dx).replace('.', 'p')


def ratio_tag(r):
    return ('%g' % r).replace('-0', '-').replace('.', 'p')


def _rmse(sim_t, sim_y, tgt):
    """Per-gauge RMSE on that gauge's own sample times (misfit_for_profile)."""
    r = np.interp(tgt['taxis'], sim_t, sim_y) - tgt['data']
    return float(np.sqrt(np.mean(r ** 2)))


def _interp_on(tgt, sim_t, sim_y):
    return np.interp(tgt['taxis'], sim_t, sim_y)


# ---------------------------------------------------------------------------
# barrier profiles -- the three definitions plus the control
# ---------------------------------------------------------------------------

def legacy_tent_profile(mesh, d0, mds, ratio):
    """The hard-coded 9-node window of 103_matching_final_fatal_wrong.py:86.

        drop = concat(linspace(1, ratio, 5), reversed(linspace(1, ratio, 5))[1:])
             = [1, a, b, c, ratio, c, b, a, 1]
        d[idx-4:idx+5] = d0 * drop

    Reproduced verbatim except that `min` replaces assignment (rev2_core's rule):
    assignment lets a shoulder value of exactly 1.0 reset a neighbouring barrier
    back to baseline, which happens for 69 window pairs in
    103_matching_prod_final_fatalwrong2.py. The six stage-2 centres used here are
    >= 25.46 ft apart, so at every dx in this sweep no two 9-node windows touch and
    the two combination rules give the same array.
    """
    x = np.asarray(mesh, dtype=float)
    prof = np.full(x.size, float(d0))
    half = np.linspace(1.0, float(ratio), 5)
    drop = np.concatenate([half, half[::-1][1:]])
    spans = []
    for md in np.atleast_1d(np.asarray(mds, dtype=float)):
        k = int(np.argmin(np.abs(x - md)))          # == mesh_utils.locate
        i0, i1 = k - 4, k + 5
        if i0 < 0 or i1 > x.size:
            raise ValueError(f"9-node window for MD {md} falls off the mesh")
        prof[i0:i1] = np.minimum(prof[i0:i1], float(d0) * drop)
        spans.append((i0, i1 - 1))
    return prof, spans


def make_profile(mesh, d0, mds, defn, ratio, cfg):
    """Return (profile, report, spans) for one definition at one ratio.

    `spans` is the list of (i0, i1) node index ranges that were ACTUALLY reduced
    (D strictly below baseline), which is what rev2_manifest.barrier_record wants.
    """
    b = cfg['barrier']
    x = np.asarray(mesh, dtype=float)
    if defn == 'no_barrier':
        return np.full(x.size, float(d0)), None, []

    if defn in ('physical_w', 'physical_w_rounded_md'):
        centres = np.round(mds) if defn == 'physical_w_rounded_md' else np.asarray(mds)
        w = float(b['w_half_width_ft'])
        on_empty = 'raise'          # a physical w must never silently fall back
    elif defn == 'legacy_single_node':
        centres = np.asarray(mds)
        w = 0.0
        on_empty = 'nearest'        # w = 0 + nearest IS the legacy assignment
    elif defn == 'legacy_tent9':
        prof, spans = legacy_tent_profile(x, d0, mds, ratio)
        return prof, None, spans
    else:
        raise ValueError(f"unknown definition {defn!r}")

    with warnings.catch_warnings():
        # The report is the authority, not the warning (build_barrier_profile
        # docstring); n_fallback is asserted on below.
        warnings.simplefilter('ignore', core.BarrierWidthWarning)
        warnings.simplefilter('ignore', core.BarrierOverlapWarning)
        prof, report = core.build_barrier_profile(
            x, float(d0), centres, w, float(ratio),
            ratio_reference=b['ratio_reference'], combine=b['combine'],
            on_empty=on_empty, on_outside=b['on_outside'], return_report=True)
    if defn.startswith('physical') and report['n_fallback'] != 0:
        raise RuntimeError(f"{defn}: {report['n_fallback']} barrier(s) fell back "
                           f"to a single node; w is not resolved by this mesh")
    spans = [(r['i0'], r['i1']) for r in report['barriers']]
    return prof, report, spans


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

_G = {}


def _init_worker(cfg):
    gw = rd.load_window_gauges(rd.R1_WINDOW)
    src_gauge = 1
    src = gw.series[src_gauge]
    _G['cfg'] = cfg
    _G['gw'] = gw
    _G['src'] = src
    _G['mds'] = rd.load_frac_hits(int(cfg['barrier']['centres_rule'].split('_')[1]))
    _G['tgt_gauges'] = [int(g) for g in cfg['targets']['gauges']]


def _mesh_for(cfg, dx):
    return rd.build_mesh(rd.R1_WINDOW,
                         float(cfg['mesh']['domain_pad_low_md_ft']),
                         float(cfg['mesh']['domain_pad_high_md_ft']), dx)


def _work(job):
    dx, defn, ratio = job
    cfg, src = _G['cfg'], _G['src']
    d0 = float(cfg['baseline_diffusivity']['D0_ft2_s'])
    mesh = _mesh_for(cfg, dx)
    x = mesh.x
    prof, report, spans = make_profile(x, d0, _G['mds'], defn, ratio, cfg)

    rec_idx = [mesh.index_of(_G['gw'].series[g].md_ft) for g in _G['tgt_gauges']]
    s = cfg['solver']
    t0 = time.time()
    taxis, rec = core.solve_forward(
        x, prof, float(s['dt_s']), float(src.t_total_s), src.taxis_s,
        src.delta_psi, mesh.index_of(float(src.md_ft)), record_idx=rec_idx,
        theta=float(s['theta']), source_time_level=s['source_time_level'],
        lambda_leak=float(s['lambda_leak']), p0=float(s['p0']),
        interface_avg=s['interface_avg'],
        theta_startup_steps=int(s['theta_startup_steps']))
    wall = time.time() - t0

    eq_w = (0.0 if ratio >= 1.0
            else core.barrier_equivalent_width(x, prof, d0, ratio))
    info = {
        'dx': dx, 'defn': defn, 'ratio': ratio, 'nx': int(x.size),
        'wall_s': wall, 'spans': [[int(a), int(b)] for a, b in spans],
        'equivalent_full_width_ft': float(eq_w),
        'excess_resistance_s_per_ft': float(
            core._series_resistance(x, prof) - core._series_resistance(x, np.full(x.size, d0))),
        'report': report,
        'D_min': float(prof.min()), 'D_max': float(prof.max()),
    }
    if report is not None:
        info['realised_full_width_ft'] = [r['realised_full_width_ft']
                                          for r in report['barriers']]
        info['total_control_volume_width_ft'] = report['total_equivalent_width_ft']
    else:
        wf = []
        for i0, i1 in spans:
            dl = x[i0] - x[i0 - 1] if i0 > 0 else 0.0
            drr = x[i1 + 1] - x[i1] if i1 < x.size - 1 else 0.0
            wf.append(float(x[i1] - x[i0] + dl / 2 + drr / 2))
        info['realised_full_width_ft'] = wf
        info['total_control_volume_width_ft'] = float(np.sum(wf))
    return job, np.asarray(taxis), np.asarray(rec), info


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--tag', default='',
                    help="output-filename version tag, e.g. 'v2'. The config is "
                         "NEVER edited (that would drift every earlier manifest's "
                         "config hash); only the output names move.")
    args = ap.parse_args(argv)
    t_start = time.time()

    cfg = load_config(args.config)
    if args.tag:
        for _k, _v in list(cfg['outputs'].items()):
            if isinstance(_v, str) and _k != 'dir':
                cfg['outputs'][_k] = _retag(_v, args.tag)
    outdir = cfg['outputs']['dir']
    os.makedirs(outdir, exist_ok=True)
    dpi = int(cfg['outputs']['figure_dpi'])
    dxs = [float(v) for v in cfg['mesh']['dx_ft_sweep']]
    dx_ref = float(cfg['mesh']['reference_dx_ft'])
    dx_coarse = max(dxs)
    ratios = [float(v) for v in cfg['barrier']['ratios']]
    d0 = float(cfg['baseline_diffusivity']['D0_ft2_s'])
    thr = float(cfg['acceptance']['threshold_percent'])

    def log(msg):
        print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)

    # ---- refuse to overwrite -------------------------------------------
    planned = [cfg['outputs'][k] for k in
               ('metrics_csv', 'widths_csv', 'acceptance_json',
                'fig_manuscript_ratio', 'fig_transmitting_ratio',
                'fig_width_and_drift', 'manifest_rollup')]
    planned += [cfg['outputs']['curves_npz_template'].format(tag=dx_tag(d)) for d in dxs]
    planned += [cfg['outputs']['manifest_per_dx_template'].format(tag=dx_tag(d)) for d in dxs]
    rm.assert_absent(planned)

    # ---- setup in the parent (targets, obs) -----------------------------
    gw = rd.load_window_gauges(rd.R1_WINDOW)
    src = gw.series[1]
    mds = rd.load_frac_hits(2)
    tgt_gauges = [int(g) for g in cfg['targets']['gauges']]
    log(f"source g1 MD {src.md_ft:.0f} ft, {src.taxis_s.size} samples, "
        f"t_total {src.t_total_s:.3f} s; targets {tgt_gauges}")
    log(f"barrier centres (stage-2 frac hits): {np.round(mds, 3).tolist()}")

    meshes = {dx: _mesh_for(cfg, dx) for dx in dxs}
    targets = {dx: [{'gauge': g, 'md_ft': float(gw.series[g].md_ft),
                     'distance_ft': abs(float(gw.series[g].md_ft) - float(src.md_ft)),
                     'idx': meshes[dx].index_of(gw.series[g].md_ft),
                     'taxis': gw.series[g].taxis_s, 'data': gw.series[g].delta_psi}
                    for g in tgt_gauges] for dx in dxs}
    for dx in dxs:
        snap = max(abs(meshes[dx].snap_error_ft(gw.series[g].md_ft)) for g in tgt_gauges)
        log(f"dx={dx}: nx={meshes[dx].nx}, max gauge snap error {snap:.3e} ft")

    # ---- jobs -----------------------------------------------------------
    jobs = []
    for dx in dxs:
        jobs.append((dx, 'no_barrier', 1.0))
        for defn in BARRIER_DEFS:
            for r in ratios:
                jobs.append((dx, defn, r))
    log(f"{len(jobs)} forward solves queued")

    nproc = min(int(cfg['run']['processes']), 6)
    results = {}
    with mp.Pool(nproc, initializer=_init_worker, initargs=(cfg,)) as pool:
        for job, taxis, rec, info in pool.imap_unordered(_work, jobs, chunksize=1):
            results[job] = (taxis, rec, info)
            log(f"  done dx={job[0]:<4} {job[1]:<22} ratio={job[2]:<8g} "
                f"nx={info['nx']:>6d} {info['wall_s']:6.2f}s "
                f"eqW={info['equivalent_full_width_ft']:.4f} ft")
    log("all solves finished")

    taxis_ref = results[(dxs[0], 'no_barrier', 1.0)][0]
    for job, (t, _, _) in results.items():
        if not np.array_equal(t, taxis_ref):
            raise RuntimeError(f"time axis differs for {job}; the sweep is spatial only")

    # ---- metrics --------------------------------------------------------
    rows, wrows = [], []
    obs_peak = {g: float(np.max(gw.series[g].delta_psi)) for g in tgt_gauges}
    for defn in ('no_barrier',) + BARRIER_DEFS:
        rlist = [1.0] if defn == 'no_barrier' else ratios
        for r in rlist:
            ref_t, ref_rec, _ = results[(dx_ref, defn, r)]
            for dx in dxs:
                t, rec, info = results[(dx, defn, r)]
                wrows.append({
                    'definition': defn, 'ratio': r, 'dx_ft': dx, 'nx': info['nx'],
                    'n_barriers': len(info['spans']),
                    'realised_full_width_min_ft': (float(np.min(info['realised_full_width_ft']))
                                                   if info['spans'] else 0.0),
                    'realised_full_width_med_ft': (float(np.median(info['realised_full_width_ft']))
                                                   if info['spans'] else 0.0),
                    'realised_full_width_max_ft': (float(np.max(info['realised_full_width_ft']))
                                                   if info['spans'] else 0.0),
                    'total_control_volume_width_ft': info['total_control_volume_width_ft'],
                    'equivalent_full_width_ft': info['equivalent_full_width_ft'],
                    'excess_resistance_s_per_ft': info['excess_resistance_s_per_ft'],
                    'D_barrier_ft2_s': d0 * r,
                    'wall_s': info['wall_s'],
                })
                for k, g in enumerate(tgt_gauges):
                    tgt = targets[dx][k]
                    sim = _interp_on(tgt, t, rec[:, k])
                    simref = _interp_on(tgt, ref_t, ref_rec[:, k])
                    denom = float(np.sqrt(np.mean(simref ** 2)))
                    rows.append({
                        'definition': defn, 'ratio': r, 'dx_ft': dx, 'gauge': g,
                        'md_ft': tgt['md_ft'], 'distance_ft': tgt['distance_ft'],
                        'rmse_psi': _rmse(t, rec[:, k], tgt),
                        'peak_sim_psi': float(np.max(sim)),
                        'peak_obs_psi': obs_peak[g],
                        'amplitude_ratio': float(np.max(sim)) / obs_peak[g],
                        'self_conv_rel': (float(np.sqrt(np.mean((sim - simref) ** 2))) / denom
                                          if denom > 0 else float('nan')),
                        'rms_sim_psi': float(np.sqrt(np.mean(sim ** 2))),
                    })

    idx = {(d['definition'], d['ratio'], d['dx_ft'], d['gauge']): d for d in rows}

    # ---- acceptance -----------------------------------------------------
    acc = {}
    for defn in ('no_barrier',) + BARRIER_DEFS:
        for r in ([1.0] if defn == 'no_barrier' else ratios):
            per_g = {}
            for g in tgt_gauges:
                a = idx[(defn, r, dx_coarse, g)]
                b = idx[(defn, r, dx_ref, g)]
                pct = 100.0 * (a['rmse_psi'] - b['rmse_psi']) / b['rmse_psi']
                pk = (100.0 * (a['peak_sim_psi'] - b['peak_sim_psi']) / b['peak_sim_psi']
                      if b['peak_sim_psi'] > 0 else float('nan'))
                per_g[g] = {
                    'rmse_dx_coarse_psi': a['rmse_psi'],
                    'rmse_dx_ref_psi': b['rmse_psi'],
                    'rmse_change_pct': pct,
                    'rmse_psi_by_dx': {dd: idx[(defn, r, dd, g)]['rmse_psi']
                                       for dd in dxs},
                    'rmse_change_pct_vs_ref_by_dx': {
                        dd: 100.0 * (idx[(defn, r, dd, g)]['rmse_psi'] - b['rmse_psi'])
                        / b['rmse_psi'] for dd in dxs},
                    'peak_sim_psi_by_dx': {dd: idx[(defn, r, dd, g)]['peak_sim_psi']
                                           for dd in dxs},
                    'self_conv_rel_by_dx': {dd: idx[(defn, r, dd, g)]['self_conv_rel']
                                            for dd in dxs},
                    'peak_dx_coarse_psi': a['peak_sim_psi'],
                    'peak_dx_ref_psi': b['peak_sim_psi'],
                    'peak_change_pct': pk,
                    'self_conv_rel_at_dx_coarse': a['self_conv_rel'],
                }
            gm = {}
            for dx in dxs:
                mses = [idx[(defn, r, dx, g)]['rmse_psi'] ** 2 for g in tgt_gauges]
                gm[dx] = float(np.sqrt(np.mean(mses)))
            worst = max(abs(v['rmse_change_pct']) for v in per_g.values())
            peaks_ref = [per_g[g]['peak_dx_ref_psi'] for g in tgt_gauges]
            acc[f"{defn}|{r:g}"] = {
                'definition': defn, 'ratio': r,
                'per_gauge': per_g,
                'gauge_mean_rmse_psi_by_dx': gm,
                'gauge_mean_rmse_change_pct': 100.0 * (gm[dx_coarse] - gm[dx_ref]) / gm[dx_ref],
                'max_abs_rmse_change_pct': worst,
                'max_abs_rmse_change_pct_by_dx': {
                    dd: max(abs(per_g[g]['rmse_change_pct_vs_ref_by_dx'][dd])
                            for g in tgt_gauges) for dd in dxs},
                'passes_2pct': bool(worst < thr),
                'max_abs_peak_change_pct': max(
                    abs(v['peak_change_pct']) for v in per_g.values()
                    if np.isfinite(v['peak_change_pct'])) if any(
                    np.isfinite(v['peak_change_pct']) for v in per_g.values()) else float('nan'),
                'max_peak_sim_psi_at_dx_ref': float(np.max(peaks_ref)),
                'degenerate_opaque': bool(np.max(peaks_ref) < 0.5),
                'equivalent_full_width_ft_by_dx': {
                    dx: [w['equivalent_full_width_ft'] for w in wrows
                         if w['definition'] == defn and w['ratio'] == r
                         and w['dx_ft'] == dx][0] for dx in dxs},
            }

    # ---- write tables ---------------------------------------------------
    def write_csv(path, dicts, cols):
        with open(path, 'w') as fh:
            fh.write(','.join(cols) + '\n')
            for d in dicts:
                fh.write(','.join(
                    ('%.10g' % d[c]) if isinstance(d[c], float) else str(d[c])
                    for c in cols) + '\n')

    mcols = ['definition', 'ratio', 'dx_ft', 'gauge', 'md_ft', 'distance_ft',
             'rmse_psi', 'peak_sim_psi', 'peak_obs_psi', 'amplitude_ratio',
             'rms_sim_psi', 'self_conv_rel']
    write_csv(cfg['outputs']['metrics_csv'], rows, mcols)
    wcols = ['definition', 'ratio', 'dx_ft', 'nx', 'n_barriers',
             'realised_full_width_min_ft', 'realised_full_width_med_ft',
             'realised_full_width_max_ft', 'total_control_volume_width_ft',
             'equivalent_full_width_ft', 'excess_resistance_s_per_ft',
             'D_barrier_ft2_s', 'wall_s']
    write_csv(cfg['outputs']['widths_csv'], wrows, wcols)
    log(f"wrote {cfg['outputs']['metrics_csv']} ({len(rows)} rows) and "
        f"{cfg['outputs']['widths_csv']} ({len(wrows)} rows)")

    # ---- curve archives, one per mesh -----------------------------------
    npz_paths = {}
    for dx in dxs:
        arrs = {'taxis_s': taxis_ref,
                'gauges': np.asarray(tgt_gauges),
                'gauge_md_ft': np.asarray([gw.series[g].md_ft for g in tgt_gauges]),
                'mesh_x_ft': meshes[dx].x,
                }
        for defn in ('no_barrier',) + BARRIER_DEFS:
            for r in ([1.0] if defn == 'no_barrier' else ratios):
                arrs[f"sim__{defn}__r{ratio_tag(r)}"] = results[(dx, defn, r)][1]
        for k, g in enumerate(tgt_gauges):
            arrs[f"obs_g{g}_taxis_s"] = gw.series[g].taxis_s
            arrs[f"obs_g{g}_psi"] = gw.series[g].delta_psi
        p = cfg['outputs']['curves_npz_template'].format(tag=dx_tag(dx))
        np.savez_compressed(p, **arrs)
        npz_paths[dx] = p
    log(f"wrote {len(npz_paths)} curve archives")

    # ---- figures --------------------------------------------------------
    fig_ratio_a = float(cfg['barrier']['manuscript_ratio'])
    fig_ratio_b = 0.01
    _fig_overlay(cfg, fig_ratio_a, cfg['outputs']['fig_manuscript_ratio'],
                 results, targets, gw, tgt_gauges, dxs, acc, dpi, logy=True)
    _fig_overlay(cfg, fig_ratio_b, cfg['outputs']['fig_transmitting_ratio'],
                 results, targets, gw, tgt_gauges, dxs, acc, dpi, logy=False)
    peak_lut = {(d['definition'], d['ratio'], d['dx_ft'], d['gauge']):
                d['peak_sim_psi'] for d in rows}
    _fig_width_drift(cfg, wrows, acc, tgt_gauges, dxs, ratios,
                     cfg['outputs']['fig_width_and_drift'], dpi, thr, peak_lut)
    log("wrote 3 figures")

    # ---- acceptance json -------------------------------------------------
    summary = {
        'study_id': cfg['study_id'], 'task_id': cfg['task_id'],
        'written_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'acceptance_rule': cfg['acceptance']['rule'],
        'threshold_percent': thr,
        'dx_compared': [dx_coarse, dx_ref],
        'misfit_convention': cfg['criteria']['primary']['aggregate'],
        'baseline_D_ft2_s': d0,
        'barrier_centres_md_ft': np.round(mds, 6).tolist(),
        'w_half_width_ft': float(cfg['barrier']['w_half_width_ft']),
        'cases': acc,
        'wall_seconds': time.time() - t_start,
    }
    with open(cfg['outputs']['acceptance_json'], 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    log(f"wrote {cfg['outputs']['acceptance_json']}")

    # ---- manifests -------------------------------------------------------
    _write_manifests(cfg, args.config, gw, src, mds, meshes, targets, tgt_gauges,
                     dxs, ratios, results, taxis_ref, npz_paths, acc, summary, d0)
    log("manifests written")

    # ---- console summary -------------------------------------------------
    print('\n=== ACCEPTANCE: per-gauge RMSE change, dx %g -> %g ft ===' % (dx_coarse, dx_ref))
    print('%-24s %8s %12s %12s %10s %10s %8s' %
          ('definition', 'ratio', 'maxRMSE%', 'maxRMSE%@0.2', 'maxPeak%',
           'eqW1.0/0.1', 'verdict'))
    for key, a in acc.items():
        eq = a['equivalent_full_width_ft_by_dx']
        drift = eq[dx_coarse] / eq[dx_ref] if eq[dx_ref] > 0 else float('nan')
        nxt = sorted(d for d in dxs if d > dx_ref)[0] if len(dxs) > 1 else dx_ref
        print('%-24s %8.0e %12.6g %12.6g %10.4g %10.4f %8s%s' %
              (a['definition'], a['ratio'], a['max_abs_rmse_change_pct'],
               a['max_abs_rmse_change_pct_by_dx'][nxt],
               a['max_abs_peak_change_pct'], drift,
               'PASS' if a['passes_2pct'] else 'FAIL',
               '  (opaque: max sim peak %.3g psi -- test degenerate)'
               % a['max_peak_sim_psi_at_dx_ref'] if a['degenerate_opaque'] else ''))
    print('\nMax simulated peak at the reference mesh, per case (psi):')
    for key, a in acc.items():
        print('  %-24s ratio %-8g  %.6g psi' %
              (a['definition'], a['ratio'], a['max_peak_sim_psi_at_dx_ref']))
    print('\nTOTAL wall %.0f s' % (time.time() - t_start))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _fig_overlay(cfg, ratio, path, results, targets, gw, tgt_gauges, dxs, acc,
                 dpi, logy):
    """Six gauge panels; four dx curves per definition overlaid."""
    show = ['physical_w', 'legacy_single_node', 'legacy_tent9']
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.6), sharex=True)
    for k, g in enumerate(tgt_gauges):
        ax = axes.flat[k]
        tgt = targets[dxs[0]][k]
        if not logy:
            ax.plot(tgt['taxis'], tgt['data'], color='0.75', lw=2.2, zorder=0,
                    label='observed')
            t, rec, _ = results[(min(dxs), 'no_barrier', 1.0)]
            ax.plot(t, rec[:, k], color='k', lw=0.9, ls=(0, (1, 1)), zorder=1,
                    label='no barrier')
        for defn in show:
            for dx in dxs:
                t, rec, _ = results[(dx, defn, ratio)]
                y = rec[:, k]
                if logy:
                    y = np.maximum(y, 1e-16)
                lw = {'physical_w': 2.6, 'legacy_single_node': 3.4,
                      'legacy_tent9': 1.3}[defn]
                al = {'physical_w': 0.95, 'legacy_single_node': 0.45,
                      'legacy_tent9': 1.0}[defn]
                ax.plot(t, y, color=DEF_COLOR[defn], ls=DX_STYLE[dx], lw=lw,
                        alpha=al,
                        label=(f"{DEF_LABEL[defn]}" if dx == dxs[0] else None),
                        zorder={'physical_w': 4, 'legacy_single_node': 2,
                                'legacy_tent9': 3}[defn])
        if logy:
            ax.set_yscale('log')
            ax.set_ylim(1e-12, max(1e-2, ax.get_ylim()[1]))
        ax.set_title(f"g{g}  MD {tgt['md_ft']:.0f} ft   ({tgt['distance_ft']:.0f} ft "
                     f"from source)", fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        if k >= 3:
            ax.set_xlabel('time since window start (s)')
        if k % 3 == 0:
            ax.set_ylabel(r'simulated $\Delta P$ (psi)')
    # legend: definition colours + dx line styles
    h1 = [plt.Line2D([], [], color=DEF_COLOR[d],
                     lw={'physical_w': 2.6, 'legacy_single_node': 3.4,
                         'legacy_tent9': 1.3}[d],
                     alpha={'physical_w': 0.95, 'legacy_single_node': 0.45,
                            'legacy_tent9': 1.0}[d], label=DEF_LABEL[d])
          for d in show]
    h2 = [plt.Line2D([], [], color='0.35', ls=DX_STYLE[dx], lw=1.6,
                     label=f"$\\Delta x$ = {dx} ft") for dx in dxs]
    if not logy:
        h1 = [plt.Line2D([], [], color='0.75', lw=2.2, label='observed'),
              plt.Line2D([], [], color='k', lw=0.9, ls=(0, (1, 1)),
                         label='no barrier (uniform $D$)')] + h1
    fig.legend(handles=h1 + h2, loc='lower center', ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.005))
    key = f"physical_w|{ratio:g}"
    eq = acc[key]['equivalent_full_width_ft_by_dx']
    eql = acc[f"legacy_single_node|{ratio:g}"]['equivalent_full_width_ft_by_dx']
    fig.suptitle(
        f"A1 mesh independence — barrier ratio {ratio:g}, $w$ = "
        f"{cfg['barrier']['w_half_width_ft']:g} ft (full width 2$w$ = "
        f"{2 * cfg['barrier']['w_half_width_ft']:g} ft), $D_0$ = "
        f"{cfg['baseline_diffusivity']['D0_ft2_s']:g} ft$^2$/s\n"
        f"equivalent full width PER BARRIER across the sweep — physical $w$: "
        f"{eq[max(dxs)] / 6:.3f} → {eq[min(dxs)] / 6:.3f} ft   |   legacy single node: "
        f"{eql[max(dxs)] / 6:.3f} → {eql[min(dxs)] / 6:.3f} ft  (= $\\Delta x$)"
        + ("   |   log axis: at this ratio a 2 ft barrier is opaque"
           if logy else "")
        + "\nthe two legacy curves coincide (the tent's shoulders are only a 4x "
          "reduction, so its resistance is its centre node's)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0.075, 1, 0.93))
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def _fig_width_drift(cfg, wrows, acc, tgt_gauges, dxs, ratios, path, dpi, thr,
                     peak_lut):
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))

    # (a) equivalent full width vs dx
    ax = axes[0]
    r_show = 0.01
    nb = 6.0  # barriers in the set; plot PER BARRIER so 2w is the right reference
    for defn in BARRIER_DEFS:
        y = [[w['equivalent_full_width_ft'] / nb for w in wrows
              if w['definition'] == defn and w['ratio'] == r_show
              and w['dx_ft'] == dx][0] for dx in dxs]
        ax.plot(dxs, y, 'o-', color=DEF_COLOR[defn], label=DEF_LABEL[defn], lw=2)
        ax.annotate('%.3g' % y[0], (dxs[0], y[0]), textcoords='offset points',
                    xytext=(4, -9), fontsize=7, color=DEF_COLOR[defn])
        ax.annotate('%.3g' % y[-1], (dxs[-1], y[-1]), textcoords='offset points',
                    xytext=(-6, 6), fontsize=7, color=DEF_COLOR[defn], ha='right')
    ax.axhline(2.0 * float(cfg['barrier']['w_half_width_ft']), color='k', ls='--',
               lw=1.2, label='requested full width 2$w$ = 2.0 ft')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\Delta x$ (ft)')
    ax.set_ylabel('equivalent full width PER BARRIER (ft)')
    ax.set_title(f'(a) what the solver actually sees\n(ratio {r_show:g}; '
                 f'series-resistance equivalent, 6 barriers)', fontsize=10)
    ax.grid(alpha=0.3, which='both', lw=0.5)
    ax.legend(fontsize=7.5, loc='lower right')

    # (b) per-gauge RMSE change, dx 1.0 -> 0.1
    ax = axes[1]
    width = 0.2
    xs = np.arange(len(tgt_gauges))
    for i, defn in enumerate(BARRIER_DEFS):
        v = [acc[f"{defn}|{r_show:g}"]['per_gauge'][g]['rmse_change_pct']
             for g in tgt_gauges]
        ax.bar(xs + (i - 1.5) * width, v, width, color=DEF_COLOR[defn],
               label=DEF_LABEL[defn])
    ax.axhspan(-thr, thr, color='0.82', zorder=0,
               label=f'\u00b1{thr:g}% acceptance band')
    ax.set_yscale('symlog', linthresh=0.01, linscale=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([f'g{g}' for g in tgt_gauges])
    ax.set_ylabel(r'RMSE($\Delta x$=1.0) − RMSE($\Delta x$=0.1), %')
    ax.set_title(f'(b) acceptance criterion  (symlog axis)\n'
                 f'(RMSE vs observed data, ratio {r_show:g})', fontsize=10)
    ax.grid(alpha=0.3, axis='y', lw=0.5)
    ax.legend(fontsize=7.5)

    # (c) peak amplitude vs dx, normalised to dx = 0.1, mean over gauges
    ax = axes[2]
    for defn in BARRIER_DEFS:
        for r, ls in ((0.01, '-'), (float(cfg['barrier']['manuscript_ratio']), '--')):
            a = acc[f"{defn}|{r:g}"]
            base = np.array([a['per_gauge'][g]['peak_dx_ref_psi'] for g in tgt_gauges])
            if np.any(base <= 0):
                continue
            y = []
            for dx in dxs:
                pk = np.array([peak_lut[(defn, r, dx, g)] for g in tgt_gauges])
                y.append(float(np.mean(pk / base)))
            ax.plot(dxs, y, ls, marker='o', color=DEF_COLOR[defn], lw=1.8)
    ax.axhline(1.0, color='k', lw=1.0, ls=':')
    hc = [plt.Line2D([], [], color=DEF_COLOR[d], lw=1.8, label=DEF_LABEL[d])
          for d in BARRIER_DEFS]
    hc += [plt.Line2D([], [], color='0.35', ls='-', lw=1.8, label='ratio 0.01'),
           plt.Line2D([], [], color='0.35', ls='--', lw=1.8,
                      label='ratio %g (manuscript)'
                            % float(cfg['barrier']['manuscript_ratio']))]
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\Delta x$ (ft)')
    ax.set_ylabel(r'mean over g2–g7 of  peak($\Delta x$) / peak($\Delta x$=0.1)')
    ax.set_title('(c) simulated peak amplitude drift\n(solid ratio 0.01, dashed '
                 'manuscript ratio)', fontsize=10)
    ax.grid(alpha=0.3, which='both', lw=0.5)
    ax.legend(handles=hc, fontsize=7, loc='lower left')

    fig.suptitle('A1 — physical barrier half-width $w$ vs the legacy index '
                 'definitions: what the fix buys', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# manifests
# ---------------------------------------------------------------------------

def _barrier_records(cfg, mesh_x, mds, d0, dx, results, ratios):
    recs = []
    for defn in BARRIER_DEFS:
        for r in ratios:
            info = results[(dx, defn, r)][2]
            rep = info['report']
            centres = np.round(mds) if defn == 'physical_w_rounded_md' else np.asarray(mds)
            if defn == 'legacy_tent9':
                w_req = 4.0 * dx
            elif defn == 'legacy_single_node':
                w_req = 0.0
            else:
                w_req = float(cfg['barrier']['w_half_width_ft'])
            for j, (i0, i1) in enumerate(info['spans']):
                mask = np.zeros(mesh_x.size, dtype=bool)
                mask[i0:i1 + 1] = True
                recs.append(rm.barrier_record(
                    mesh_x, mask,
                    label=f"{defn}|ratio={r:g}|hit{j}",
                    centre_md_ft=float(centres[j]), w_requested_ft=w_req,
                    ratio=r, d_baseline=d0,
                    report=(rep if (j == 0 and rep is not None) else None)))
    return recs


def _write_manifests(cfg, cfg_path, gw, src, mds, meshes, targets, tgt_gauges,
                     dxs, ratios, results, taxis_ref, npz_paths, acc, summary, d0):
    s = cfg['solver']
    drv = rm.driver_record(
        kind='gauge_series', baseline_removal='subtract_first_sample',
        value_units='delta_psi',
        series_path=cfg['data']['gauge_series_template'].format(n=1),
        gauge_number=1, gauge_md_ft=float(src.md_ft),
        taxis=src.taxis_s, values=src.delta_psi,
        time_start=cfg['window']['time_start'], time_end=cfg['window']['time_end'])

    inputs = [(cfg['data']['gauge_md_npz'], 'geometry', 'gauge_md_npz'),
              (cfg['data']['frac_hit_stage1_npz'], 'geometry', 'frac_hit_stage1'),
              (cfg['data']['frac_hit_barrier_npz'], 'geometry', 'frac_hit_stage2'),
              ('output/rev2_20260901/A1/effective_widths.csv', 'prior_run_output',
               'a1_investigation_widths'),
              ('output/rev2_20260901/A1/A1_spec_build_barrier_profile.md',
               'prior_run_output', 'a1_spec')]
    inputs += [(cfg['data']['gauge_series_template'].format(n=n), 'gauge_series',
                f'gauge{n}') for n in [1] + list(tgt_gauges)]

    def _src_group(dx):
        return rm.source_protocol(
            application='dirichlet_node',
            solver_class='rev2_core.solve_forward (tridiagonal solve_banded); '
                         'bitwise identical to r1_calibration_core.solve_forward '
                         'at theta=1/harmonic/lambda=0 (A4 self-test T1a)',
            placement_rule=cfg['source']['selection_rule'],
            sources=[rm.source_record(meshes[dx].x, md_requested_ft=float(src.md_ft),
                                      mesh_idx=meshes[dx].index_of(float(src.md_ft)),
                                      driver=drv, label='g1', index_in_source_list=0)],
            targets=[{'gauge': t['gauge'], 'md_ft': t['md_ft'],
                      'distance_ft': t['distance_ft'], 'mesh_idx': t['idx']}
                     for t in targets[dx]],
            time_level=s['source_time_level'],
            phase_chaining=rm.NONE_DECLARED,
            boundary_conditions={'lbc': s['lbc'], 'rbc': s['rbc']})

    def _num_group(dx, barriers):
        base = np.full(meshes[dx].nx, d0)
        return rm.numerics(
            time=rm.time_record(taxis_ref, mode='fixed', theta=float(s['theta']),
                                t_total_requested_s=float(src.t_total_s),
                                dt_requested_s=float(s['dt_s']),
                                source_time_level=s['source_time_level'],
                                theta_startup_steps=int(s['theta_startup_steps']),
                                label=f"dx={dx} ft, all runs share this axis"),
            mesh=rm.mesh_record(meshes[dx].x, dx_requested_ft=dx,
                                window_md_ft=(cfg['window']['md_min_ft'],
                                              cfg['window']['md_max_ft']),
                                pad_low_ft=float(cfg['mesh']['domain_pad_low_md_ft']),
                                pad_high_ft=float(cfg['mesh']['domain_pad_high_md_ft']),
                                refinement=cfg['mesh']['refinement']),
            interface_avg=s['interface_avg'],
            boundary={'lbc': s['lbc'], 'rbc': s['rbc'], 'pml_thickness': 0.0,
                      'sigma_max': 0.0},
            diffusivity={'profile_family': 'uniform_plus_barriers',
                         'D0_ft2_s': d0,
                         'D0_provenance': cfg['baseline_diffusivity']['provenance'],
                         'baseline_D_sha256': rm.sha256_array(base),
                         'note': 'The barrier records below give the reduced field '
                                 'for every (definition, ratio) run on this mesh; '
                                 'the no_barrier control uses the baseline itself.'},
            barriers=barriers, leakage=rm.NONE_DECLARED,
            kernel={'name': 'rev2_core.solve_forward', 'banded': True,
                    'theta': float(s['theta']),
                    'equivalence_reference':
                        'output/rev2_20260901/A4/selftest_output.txt T1a: bitwise '
                        'identical (max|diff| = 0.000e+00 psi) to '
                        'r1_calibration_core.solve_forward, which is proven '
                        'bit-equivalent to fibeRIS in the R1 manifest'},
            rng={'engine': 'none', 'seeds': None,
                 'note': 'fully deterministic sweep, no random component'},
            parallel={'processes': int(cfg['run']['processes']),
                      'backend': 'multiprocessing.Pool'})

    for dx in dxs:
        p = cfg['outputs']['manifest_per_dx_template'].format(tag=dx_tag(dx))
        rm.write_manifest(
            p, study_id=cfg['study_id'] + f"__dx{dx_tag(dx)}",
            task_id=cfg['task_id'], config=cfg, config_path=cfg_path,
            inputs=inputs, source=_src_group(dx),
            numerics=_num_group(dx, _barrier_records(cfg, meshes[dx].x, mds, d0,
                                                     dx, results, ratios)),
            outputs=[rm.output_decl(npz_paths[dx], role='arrays_npz',
                                    note=f'all 17 recorded gauge series on the '
                                         f'dx = {dx} ft mesh, plus the mesh and '
                                         f'the observed series')],
            results={'dx_ft': dx, 'nx': meshes[dx].nx,
                     'runs': {f"{d}|{r:g}": {
                         'equivalent_full_width_ft':
                             results[(dx, d, r)][2]['equivalent_full_width_ft'],
                         'total_control_volume_width_ft':
                             results[(dx, d, r)][2]['total_control_volume_width_ft'],
                         'excess_resistance_s_per_ft':
                             results[(dx, d, r)][2]['excess_resistance_s_per_ft'],
                         'wall_s': results[(dx, d, r)][2]['wall_s']}
                         for d in BARRIER_DEFS for r in ratios}},
            notes=['One manifest per mesh. The acceptance roll-up across meshes is '
                   'output/rev2_20260901/A1/deliverable/manifest.json.',
                   'dt is held at 1 s for every dx: this is a SPATIAL refinement '
                   'study and the time discretisation must not move with it.',
                   'Barrier records are listed for every (definition, ratio) run on '
                   'this mesh; the build_barrier_profile report is attached to the '
                   'first record of each group.'],
            require_modules=('rev2_core', 'rev2_data', 'rev2_manifest'),
            extra_code_files=(os.path.abspath(__file__),),
            allow_undeclared_outputs=True)

    rm.write_manifest(
        cfg['outputs']['manifest_rollup'], study_id=cfg['study_id'],
        task_id=cfg['task_id'], config=cfg, config_path=cfg_path,
        inputs=inputs, source=_src_group(min(dxs)),
        numerics=_num_group(min(dxs),
                            _barrier_records(cfg, meshes[min(dxs)].x, mds, d0,
                                             min(dxs), results, ratios)),
        outputs=[rm.output_decl(cfg['outputs']['metrics_csv'], role='csv'),
                 rm.output_decl(cfg['outputs']['widths_csv'], role='csv'),
                 rm.output_decl(cfg['outputs']['acceptance_json'], role='json'),
                 rm.output_decl(cfg['outputs']['fig_manuscript_ratio'],
                                role='figure_png', dpi=int(cfg['outputs']['figure_dpi'])),
                 rm.output_decl(cfg['outputs']['fig_transmitting_ratio'],
                                role='figure_png', dpi=int(cfg['outputs']['figure_dpi'])),
                 rm.output_decl(cfg['outputs']['fig_width_and_drift'],
                                role='figure_png', dpi=int(cfg['outputs']['figure_dpi']))],
        results=summary,
        notes=['ROLL-UP manifest for the whole dx sweep. `numerics.mesh` records '
               'the FINEST mesh (dx = 0.1 ft, the reference of the acceptance '
               'comparison); the other three meshes have their own manifests '
               'manifest_dx1p0/0p5/0p2.json in the same directory, each with its '
               'own mesh_sha256 and barrier records.',
               'Every solver run in this study is covered by exactly one per-dx '
               'manifest (house rule 3).',
               'The A1 investigation (output/rev2_20260901/A1/) is an INPUT here, '
               'not re-run.'],
        require_modules=('rev2_core', 'rev2_data', 'rev2_manifest'),
        extra_code_files=(os.path.abspath(__file__),),
        allow_undeclared_outputs=True)


if __name__ == '__main__':
    main()
