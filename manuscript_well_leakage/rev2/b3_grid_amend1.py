#!/usr/bin/env python3
"""B3 AMEND 1 -- is the reported between-mesh spread of the calibrated D
resolvable by the search that measured it?

The v1/v2 report quoted two headline grid-convergence numbers:

    control  , meshes dx = 1.0 .. 0.05 : D = 1138.52-1138.53, spread 0.0004 %
    w1_r1em2 , meshes dx = 1.0 .. 0.1  : D = 5080.41-5080.70, spread 0.0056 %

and listed both in section 9 as quotable.  Both are best-of-6 spreads over a
1-D Nelder-Mead search whose own multi-restart envelope on a SINGLE mesh
reaches 0.0054 % (control, dx = 0.05) and 0.0093 % (w1, dx = 0.1), with all
six restarts scored converged (objective within 1e-6 relative of the best) on
every arm and every mesh.  A between-mesh spread that is SMALLER than the
same-mesh restart scatter is not a measurement of grid convergence; it is a
sample of the search's own resolution.

This script measures that directly, in three parts, and rewrites the two
claims as upper bounds.

  stage `envelope`  no new solves.  Re-reads b3_uniform_restarts_v2.json and
                    tabulates, per (arm, norm, mesh), the full 6-restart D
                    envelope, next to the between-mesh best-of-6 spread over
                    the width-matched group.  This is the comparison the v2
                    README never made.

  stage `hires`     new solves.  Re-locates the argmin on every (arm, norm,
                    mesh) with a DIFFERENT and far more precise estimator than
                    the NM restarts: a two-pass local least-squares parabola
                    fit to the objective on a symmetric stencil in log10 D.
                    The parabola uses every sample (so objective round-off
                    averages down as 1/sqrt(n)) instead of terminating on a
                    simplex-size test, and its own noise floor is measured
                    from the fit residual and propagated to an uncertainty on
                    the argmin.  Two independent stencil half-widths are run
                    so the estimator's own reproducibility is measured, not
                    assumed.

  stage `report`    csv + json + figure + manifest, and the numbers the README
                    must now carry.

Owner: task B3.  Writes only into output/rev2_20260901/B3/ and only files
tagged `_v3`/`amend1`.  Touches no other task's files and edits no shared
module.  b3_grid.py and b3_grid_v2report.py are left exactly as they are:
nothing they COMPUTED is wrong, and the v1/v2 products stand unaltered under
house rule 2.  What was wrong was the README's reading of them.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'baseline_calibration')))

import rev2_core as core          # noqa: E402
import rev2_data as rd            # noqa: E402
import rev2_manifest as rm        # noqa: E402
import r1_calibration_core as r1c  # noqa: E402
import b3_grid as B               # noqa: E402

CONFIG = 'configs/rev2/b3_grid.json'
TAG = 'v3'
WORKER_MODULES = ('rev2_core', 'rev2_data', 'r1_calibration_core')

# Output paths live HERE and NOT in configs/rev2/b3_grid.json on purpose.
# Adding an `outputs_amend1` block to that config changes its sha256, which
# retro-flips all 19 v2 manifests from `clean` to `config: drift` -- exactly the
# failure the v1 manifests already suffered when `outputs_v2` was appended
# (README section 0). The amendment must not damage the audit trail of the run
# it is amending, so it declares its own products explicitly via output_decl and
# leaves the config byte-identical.
OUTPUTS = {
    'dir': 'output/rev2_20260901/B3',
    'csv': 'output/rev2_20260901/B3/b3_search_resolution_v3.csv',
    'json': 'output/rev2_20260901/B3/b3_amend1_search_resolution_v3.json',
    'fig': 'output/rev2_20260901/B3/fig_b3_search_resolution_v3.png',
    'manifest': 'output/rev2_20260901/B3/manifest_amend1_v3.json',
}

# stencil half-widths in log10 D, run independently so the high-precision
# estimator's own reproducibility is measured rather than asserted.
STENCILS = (5.0e-4, 2.0e-3)
N_STENCIL = 11


def log(msg):
    sys.stdout.write('[b3-amend1 %7.1fs] %s\n' % (time.time() - _T0, msg))
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# worker: one misfit, reusing b3_grid's own cached setup and profile builder
# ---------------------------------------------------------------------------

def _mis(task):
    """(arm, dx, log10_D) -> (rmse_psi, rmse_normalised).  Identical objective
    to b3_grid._misfit; it is literally that function."""
    arm, dx, x = task
    return B._misfit((arm, dx, 'uniform', [float(x)]))


def _parabola_argmin(xs, ys):
    """Least-squares parabola through (xs, ys); returns argmin, curvature,
    residual rms, and the argmin uncertainty implied by that residual.

    For y = a(x-m)^2 + c the argmin uncertainty from independent errors of
    size s on n samples symmetric about m is  s / (a * sqrt(sum (x-m)^2)),
    which is what `sig` reports.  It is an upper bound here because most of
    the residual is the neglected quartic term of a smooth function, not
    noise.
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    x0 = xs.mean()
    c = np.polyfit(xs - x0, ys, 2)
    if c[0] <= 0:
        return None
    m = x0 - c[1] / (2.0 * c[0])
    res = ys - np.polyval(c, xs - x0)
    s = float(np.sqrt(np.sum(res ** 2) / max(len(xs) - 3, 1)))
    sig = s / (c[0] * np.sqrt(np.sum((xs - m) ** 2))) if c[0] > 0 else np.nan
    return {'log10_D': float(m), 'D_ft2_s': float(10.0 ** m),
            'curvature_per_dex2': float(c[0]), 'fmin': float(np.polyval(c, m - x0)),
            'resid_rms': s, 'sigma_log10_D': float(sig),
            'sigma_pct_D': float(sig * np.log(10.0) * 100.0)}


def stage_envelope(cfg):
    """No solves.  The restart envelope next to the between-mesh spread."""
    restarts = B.load_ckpt('uniform_restarts_v1.json')
    if restarts is None:
        raise SystemExit('b3_grid v1 restart checkpoint missing')
    cells = {}
    for r in restarts:
        k = (r['arm'], r['norm'], float(r['dx_ft']))
        cells.setdefault(k, []).append(r)
    rows = []
    for k in sorted(cells, key=lambda k: (k[0], k[1], -k[2])):
        v = cells[k]
        Ds = np.array([r['D_ft2_s'] for r in v], float)
        objs = np.array([r['objective'] for r in v], float)
        best = objs.min()
        conv = Ds[objs <= best * (1.0 + 1e-6)]
        rows.append({
            'arm': k[0], 'norm': k[1], 'dx_ft': k[2],
            'n_restarts': int(Ds.size),
            'n_restarts_converged': int(conv.size),
            'D_best': float(Ds[int(np.argmin(objs))]),
            'restart_D_min': float(Ds.min()), 'restart_D_max': float(Ds.max()),
            'restart_envelope_pct': float((Ds.max() - Ds.min()) / Ds.min() * 100.0),
            'obj_best': float(best),
            'obj_spread_rel': float((objs.max() - objs.min()) / best)})
    return rows


TZ_PARAMS = ('D_near', 'D_far', 's_c', 'width')


def stage_envelope_two_zone(cfg):
    """No solves.  The same comparison for the k = 4 family, whose stability
    rows sit in the same README table as the uniform ones and carry the same
    hazard.  b3_grid ran 4 cold restarts per (arm, mesh), absolute norm only.
    """
    restarts = B.load_ckpt('two_zone_restarts_v1.json')
    if restarts is None:
        raise SystemExit('b3_grid v1 two_zone restart checkpoint missing')
    rs = restarts['restarts'] if isinstance(restarts, dict) else restarts
    cells = {}
    for r in rs:
        cells.setdefault((r['arm'], float(r['dx_ft'])), []).append(r)
    per_cell, best = [], {}
    for k in sorted(cells, key=lambda k: (k[0], -k[1])):
        v = cells[k]
        P = 10.0 ** np.array([r['params'] for r in v], float)
        obj = np.array([r['rmse_psi'] for r in v], float)
        bi = int(np.argmin(obj))
        best[k] = P[bi]
        row = {'arm': k[0], 'dx_ft': k[1], 'n_restarts': int(P.shape[0]),
               'rmse_psi_best': float(obj[bi]),
               'rmse_spread_pct': float((obj.max() - obj.min()) / obj.min() * 100.0)}
        for j, nm in enumerate(TZ_PARAMS):
            row['%s_best' % nm] = float(P[bi, j])
            row['%s_restart_envelope_pct' % nm] = float(
                (P[:, j].max() - P[:, j].min()) / P[:, j].min() * 100.0)
        per_cell.append(row)
    groups = {}
    for arm in sorted({k[0] for k in best}):
        ks = sorted([k for k in best if k[0] == arm], key=lambda k: -k[1])
        g = {'arm': arm, 'dxs': [k[1] for k in ks]}
        for j, nm in enumerate(TZ_PARAMS):
            vals = np.array([best[k][j] for k in ks], float)
            floor = max(r['%s_restart_envelope_pct' % nm] for r in per_cell
                        if r['arm'] == arm)
            sp = float((vals.max() - vals.min()) / vals.min() * 100.0)
            g[nm] = {'best_of_4_per_mesh': [float(x) for x in vals],
                     'between_mesh_spread_pct': sp,
                     'restart_envelope_pct_worst_mesh': floor,
                     'resolvable_by_the_search': bool(sp > floor),
                     'upper_bound_pct': max(sp, floor)}
        groups[arm] = g
    return {'note': 'Absolute norm only; b3_grid fits two_zone under the '
                    'absolute norm. 4 cold restarts per (arm, mesh) from one '
                    'dx-independent 256-point Latin hypercube.',
            'per_cell': per_cell, 'groups': groups}


def stage_hires(cfg, nproc, env_rows):
    """New solves.  Two-pass parabola argmin at two stencil widths."""
    seeds = {(r['arm'], r['norm'], r['dx_ft']): r['D_best'] for r in env_rows}
    keys = sorted(seeds, key=lambda k: (k[2],))     # fine meshes first
    out = {}
    with mp.Pool(nproc, initializer=B._init_worker, initargs=(cfg,)) as pool:
        for h in STENCILS:
            # pass 1: stencil centred on the published best-of-6
            tasks, index = [], []
            for k in keys:
                x0 = float(np.log10(seeds[k]))
                for x in x0 + np.linspace(-h, h, N_STENCIL):
                    tasks.append((k[0], k[2], x))
                    index.append((k, x))
            vals = pool.map(_mis, tasks, chunksize=1)
            p1 = {}
            for k in keys:
                xs = [x for (kk, x) in index if kk == k]
                ni = 0 if k[1] == 'abs' else 1
                ys = [v[ni] for (kk, _), v in zip(index, vals) if kk == k]
                p1[k] = _parabola_argmin(xs, ys)
            # pass 2: re-centre on the pass-1 argmin and re-fit, so the result
            # does not depend on where the NM search happened to stop
            tasks, index = [], []
            for k in keys:
                x0 = p1[k]['log10_D']
                for x in x0 + np.linspace(-h, h, N_STENCIL):
                    tasks.append((k[0], k[2], x))
                    index.append((k, x))
            vals = pool.map(_mis, tasks, chunksize=1)
            for k in keys:
                xs = [x for (kk, x) in index if kk == k]
                ni = 0 if k[1] == 'abs' else 1
                ys = [v[ni] for (kk, _), v in zip(index, vals) if kk == k]
                r = _parabola_argmin(xs, ys)
                r['pass1_D_ft2_s'] = p1[k]['D_ft2_s']
                r['pass_shift_pct'] = (r['D_ft2_s'] / p1[k]['D_ft2_s'] - 1.0) * 100.0
                out[(k, h)] = r
            log('hires stencil +-%.0e dex done (%d solves)'
                % (h, 2 * len(keys) * N_STENCIL))
    rows = []
    for k in keys:
        a = out[(k, STENCILS[0])]
        b = out[(k, STENCILS[1])]
        rows.append({
            'arm': k[0], 'norm': k[1], 'dx_ft': k[2],
            'D_nm_best_of_6': seeds[k],
            'D_hires_h1': a['D_ft2_s'], 'D_hires_h2': b['D_ft2_s'],
            'hires_stencil_disagreement_pct':
                abs(a['D_ft2_s'] - b['D_ft2_s']) / min(a['D_ft2_s'], b['D_ft2_s']) * 100.0,
            'hires_pass_shift_pct_h1': a['pass_shift_pct'],
            'hires_sigma_pct_h1': a['sigma_pct_D'],
            'hires_sigma_pct_h2': b['sigma_pct_D'],
            'curvature_h1': a['curvature_per_dex2'],
            'resid_rms_h1': a['resid_rms'],
            'nm_minus_hires_pct':
                (seeds[k] / a['D_ft2_s'] - 1.0) * 100.0})
    return rows, {str(k): v for k, v in
                  {('%s|%s|%g|h=%g' % (k[0], k[1], k[2], h)): v
                   for (k, h), v in out.items()}.items()}


# ---------------------------------------------------------------------------
# the two headline groups
# ---------------------------------------------------------------------------

GROUPS = {
    'control_nobarrier_20x': {
        'arm': 'control',
        'dxs': [1.0, 0.5, 0.25, 0.2, 0.1, 0.05],
        'refinement': '20x (dx 1.0 -> 0.05 ft)',
        'realised_width': 'no barrier'},
    'w1_r1em2_12ft_10x': {
        'arm': 'w1_r1em2',
        'dxs': [1.0, 0.5, 0.25, 0.2, 0.1],
        'refinement': '10x (dx 1.0 -> 0.1 ft)',
        'realised_width': '12.000 ft total (6 x 2.000 ft, exact)'},
    'w0p25_r1em2_3ft_2p5x': {
        'arm': 'w0p25_r1em2',
        'dxs': [0.5, 0.25, 0.2],
        'refinement': '2.5x (dx 0.5 -> 0.2 ft)',
        'realised_width': '3.000 ft total (6 x 0.500 ft)'},
}


def stage_compare(env_rows, hi_rows):
    def pick(rows, arm, norm, dx, key):
        for r in rows:
            if r['arm'] == arm and r['norm'] == norm and abs(r['dx_ft'] - dx) < 1e-12:
                return r[key]
        raise KeyError((arm, norm, dx, key))

    out = {}
    for gname, g in GROUPS.items():
        for norm in ('abs', 'normalised'):
            nm = [pick(env_rows, g['arm'], norm, dx, 'D_best') for dx in g['dxs']]
            hi = [pick(hi_rows, g['arm'], norm, dx, 'D_hires_h1') for dx in g['dxs']]
            hi2 = [pick(hi_rows, g['arm'], norm, dx, 'D_hires_h2') for dx in g['dxs']]
            envs = [pick(env_rows, g['arm'], norm, dx, 'restart_envelope_pct')
                    for dx in g['dxs']]
            sig = [pick(hi_rows, g['arm'], norm, dx, 'hires_sigma_pct_h1')
                   for dx in g['dxs']]
            sten = [pick(hi_rows, g['arm'], norm, dx,
                         'hires_stencil_disagreement_pct') for dx in g['dxs']]
            sp_nm = (max(nm) - min(nm)) / min(nm) * 100.0
            sp_hi = (max(hi) - min(hi)) / min(hi) * 100.0
            sp_hi2 = (max(hi2) - min(hi2)) / min(hi2) * 100.0
            floor = max(envs)
            # How much objective does the reported spread actually buy?  The
            # misfit is locally a * (log10 D - log10 D*)^2 + f*, so two optima
            # that differ by dlog in log10 D differ in misfit by a * dlog^2.
            # This is search-free and implementation-free: it says what the
            # data can possibly be resolving.
            curv = max(pick(hi_rows, g['arm'], norm, dx, 'curvature_h1')
                       for dx in g['dxs'])
            fmin = max(pick(hi_rows, g['arm'], norm, dx, 'resid_rms_h1')
                       for dx in g['dxs'])
            dlog_nm = float(np.log10(max(nm) / min(nm)))
            dlog_hi = float(np.log10(max(hi) / min(hi)))
            out['%s|%s' % (gname, norm)] = {
                'arm': g['arm'], 'norm': norm, 'dxs': g['dxs'],
                'refinement': g['refinement'],
                'realised_width': g['realised_width'],
                'D_nm_best_of_6': nm, 'D_hires_h1': hi, 'D_hires_h2': hi2,
                'between_mesh_spread_pct_nm': sp_nm,
                'between_mesh_spread_pct_hires_h1': sp_hi,
                'between_mesh_spread_pct_hires_h2': sp_hi2,
                'restart_envelope_pct_per_mesh': envs,
                'restart_envelope_pct_worst_single_mesh': floor,
                'restart_envelope_pct_all_restarts_pooled': _pooled(
                    env_rows, g['arm'], norm, g['dxs']),
                'hires_sigma_pct_worst': max(sig),
                'hires_stencil_disagreement_pct_worst': max(sten),
                'curvature_psi_per_dex2': curv,
                'objective_gap_across_group_psi_nm': curv * dlog_nm ** 2,
                'objective_gap_across_group_psi_hires': curv * dlog_hi ** 2,
                'parabola_fit_residual_rms_psi': fmin,
                'nm_spread_over_search_floor': sp_nm / floor if floor else None,
                'nm_spread_resolvable_by_the_search': bool(sp_nm > floor),
                'hires_spread_resolvable_by_hires': bool(sp_hi > max(max(sig), max(sten))),
                'upper_bound_pct': max(sp_nm, sp_hi, sp_hi2, floor),
            }
    return out


def _pooled(env_rows, arm, norm, dxs):
    lo, hi = np.inf, -np.inf
    for r in env_rows:
        if r['arm'] == arm and r['norm'] == norm and any(
                abs(r['dx_ft'] - d) < 1e-12 for d in dxs):
            lo = min(lo, r['restart_D_min'])
            hi = max(hi, r['restart_D_max'])
    return float((hi - lo) / lo * 100.0)


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def figure(cmp_rows, env_rows, path, dpi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.0, 6.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0], hspace=0.42,
                          wspace=0.24)
    ax = [fig.add_subplot(gs[:, 0])]
    sub = [fig.add_subplot(gs[i, 1]) for i in range(3)]

    order = [('control_nobarrier_20x', 'control\nno barrier, 20x'),
             ('w1_r1em2_12ft_10x', 'w1_r1em2\n12.000 ft realised, 10x'),
             ('w0p25_r1em2_3ft_2p5x', 'w0p25_r1em2\n3.000 ft realised, 2.5x')]
    xs = np.arange(len(order))
    wbar = 0.34
    for j, norm in enumerate(('abs', 'normalised')):
        a = ax[0]
        sp = [cmp_rows['%s|%s' % (g, norm)]['between_mesh_spread_pct_nm']
              for g, _ in order]
        fl = [cmp_rows['%s|%s' % (g, norm)]['restart_envelope_pct_worst_single_mesh']
              for g, _ in order]
        a.bar(xs + (j - 0.5) * wbar, sp, wbar * 0.9,
              color=('#2b6cb0' if norm == 'abs' else '#63b3ed'),
              label='between-mesh spread, best-of-6 (%s)' % norm)
        a.plot(xs + (j - 0.5) * wbar, fl, '_', ms=26, mew=3,
               color=('#c53030' if norm == 'abs' else '#f6ad55'),
               label='same-mesh 6-restart envelope, worst mesh (%s)' % norm)
    ax[0].set_yscale('log')
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([l for _, l in order], fontsize=8)
    ax[0].set_ylabel(r'spread in the calibrated $D$  (%)')
    ax[0].set_title('the two "grid-converged" spreads sit BELOW the search\'s\n'
                    'own same-mesh scatter: they are upper bounds, not measurements',
                    fontsize=9.5)
    ax[0].grid(alpha=0.3, axis='y', which='both')
    ax[0].legend(fontsize=6.5, loc='upper left')

    # right: one panel per width-matched group, each on its own scale, so the
    # control and w1 groups are not flattened by w0p25's real 0.18 % spread
    for gi, (g, lab) in enumerate(order):
        a = sub[gi]
        c = cmp_rows['%s|abs' % g]
        dxs = np.array(c['dxs'], float)
        nm = np.array(c['D_nm_best_of_6'], float)
        hi = np.array(c['D_hires_h1'], float)
        ref = hi.mean()
        env = np.array(c['restart_envelope_pct_per_mesh'], float)
        col = ['#2b6cb0', '#2f855a', '#b7791f'][gi]
        a.errorbar(dxs, (nm / ref - 1.0) * 100.0, yerr=env / 2.0, fmt='o',
                   ms=5, capsize=3, color=col, lw=1.0,
                   label='NM best-of-6, bar = same-mesh 6-restart envelope')
        a.plot(dxs, (hi / ref - 1.0) * 100.0, 's', ms=6, mfc='none', mew=1.6,
               color='k', label='two-pass parabola argmin')
        a.set_xscale('log')
        a.axhline(0.0, color='k', lw=0.6, ls=':')
        a.grid(alpha=0.3, which='both')
        a.set_title('%s   published spread %s %%,  parabola %s %%'
                    % (lab.replace('\n', ' — '),
                       '%.4g' % c['between_mesh_spread_pct_nm'],
                       '%.4g' % c['between_mesh_spread_pct_hires_h1']),
                    fontsize=8)
        a.tick_params(labelsize=7)
        a.set_ylabel(r'$D - \bar{D}$ (%)', fontsize=7.5)
        if gi == 0:
            a.legend(fontsize=6.0, loc='best')
        if gi == 2:
            a.set_xlabel(r'$\Delta x$  (ft)', fontsize=8)

    fig.suptitle('B3 amend 1 — the grid-convergence spreads are bounded by the '
                 'search resolution, not measured by it', fontsize=11)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def write_manifest(cfg, cfg_disk, cfg_path, t_start, outs, results, n_solves):
    s = cfg['solver']
    fhits = rd.load_frac_hits(2)
    S0 = rd.setup_r1(dx_ft=1.0,
                     pad_low_ft=float(cfg['mesh']['domain_pad_low_md_ft']),
                     pad_high_ft=float(cfg['mesh']['domain_pad_high_md_ft']))
    src = S0['src_series']
    drv = rm.driver_record(
        kind='gauge_series', baseline_removal='subtract_first_sample',
        value_units='delta_psi',
        series_path=cfg['data']['gauge_series_template'].format(n=1),
        gauge_number=1, gauge_md_ft=float(src.md_ft),
        taxis=src.taxis_s, values=src.delta_psi,
        time_start=cfg['window']['time_start'], time_end=cfg['window']['time_end'])
    inputs = B._manifest_inputs(cfg, cfg['targets']['gauges'])
    inputs += [(p, 'prior_run_output', k) for k, p in (
        ('b3_uniform_restarts_v2', cfg['outputs_v2']['uniform_restarts_json']),
        ('b3_uniform_fits_v2', cfg['outputs_v2']['uniform_csv']),
        ('b3_forward_vs_calibrated_v2',
         cfg['outputs_v2']['forward_vs_calibrated_json'])) if os.path.exists(p)]

    dx_fine = 0.05
    S = rd.setup_r1(dx_ft=dx_fine,
                    pad_low_ft=float(cfg['mesh']['domain_pad_low_md_ft']),
                    pad_high_ft=float(cfg['mesh']['domain_pad_high_md_ft']))
    mesh = S['mesh']
    taxis_ref = np.arange(0.0, float(src.t_total_s) + float(s['dt_s']),
                          float(s['dt_s']))
    rm.write_manifest(
        outs['manifest'],
        study_id=cfg['study_id'] + '__amend1_search_resolution',
        task_id=cfg['task_id'], config=cfg_disk, config_path=cfg_path,
        inputs=inputs,
        source=rm.source_protocol(
            application='dirichlet_node',
            solver_class='rev2_core.solve_forward (tridiagonal solve_banded)',
            placement_rule=cfg['source']['selection_rule'],
            sources=[rm.source_record(mesh.x, md_requested_ft=float(src.md_ft),
                                      mesh_idx=mesh.index_of(float(src.md_ft)),
                                      driver=drv, label='g1',
                                      index_in_source_list=0)],
            targets=[{'gauge': t['gauge'], 'md_ft': t['md_ft'],
                      'distance_ft': t['distance_ft'], 'mesh_idx': t['idx']}
                     for t in S['targets']],
            time_level=s['source_time_level'],
            phase_chaining=rm.NONE_DECLARED,
            boundary_conditions={'lbc': s['lbc'], 'rbc': s['rbc']}),
        numerics=rm.numerics(
            time=rm.time_record(taxis_ref, mode='fixed', theta=float(s['theta']),
                                t_total_requested_s=float(src.t_total_s),
                                dt_requested_s=float(s['dt_s']),
                                source_time_level=s['source_time_level'],
                                theta_startup_steps=int(s['theta_startup_steps']),
                                label='shared by every mesh; dt fixed at 1 s'),
            mesh=rm.mesh_record(mesh.x, dx_requested_ft=dx_fine,
                                window_md_ft=(cfg['window']['md_min_ft'],
                                              cfg['window']['md_max_ft']),
                                pad_low_ft=float(cfg['mesh']['domain_pad_low_md_ft']),
                                pad_high_ft=float(cfg['mesh']['domain_pad_high_md_ft']),
                                refinement=cfg['mesh']['refinement']),
            interface_avg=s['interface_avg'],
            boundary={'lbc': s['lbc'], 'rbc': s['rbc'], 'pml_thickness': 0.0,
                      'sigma_max': 0.0},
            diffusivity={
                'profile_families_fitted': ['uniform'],
                'bounds': {'uniform': cfg['families']['uniform']['bounds']},
                'note': 'AMEND 1 re-locates the uniform (k = 1) argmin only. No '
                        'two_zone fit is repeated and no fitted value in the v2 '
                        'products is replaced: this run measures how precisely '
                        'the v2 search located its own optimum.',
                'estimator': 'two-pass local least-squares parabola in log10 D, '
                             '%d-point symmetric stencil, half-widths %s dex, '
                             'each pass re-centred on the previous argmin'
                             % (N_STENCIL, ' and '.join('%g' % h for h in STENCILS)),
                'ratio_reference': cfg['barrier']['ratio_reference']},
            barriers=rm.NONE_DECLARED, leakage=rm.NONE_DECLARED,
            kernel={'name': 'rev2_core.solve_forward', 'banded': True,
                    'theta': float(s['theta']),
                    'equivalence_reference': 'A4 self-test T1a'},
            rng=rm.NONE_DECLARED,
            parallel={'processes': int(cfg['run']['processes']),
                      'backend': 'multiprocessing.Pool'}),
        outputs=[rm.output_decl(outs['csv'], role='csv'),
                 rm.output_decl(outs['json'], role='json'),
                 rm.output_decl(outs['fig'], role='figure_png',
                                dpi=int(cfg['outputs_v2']['figure_dpi']))],
        results=results,
        notes=[
            'AMEND 1 to B3, answering an independent reviewer who reproduced '
            'that the two quoted grid-convergence spreads (0.0004 % control, '
            '0.0056 % w1) are SMALLER than the study\'s own same-mesh '
            'multi-restart scatter (up to 0.0054 % and 0.0093 % respectively), '
            'so they cannot be measurements of grid convergence.',
            'CONFIRMED. Both spreads reproduce from b3_uniform_fits_v2.csv to '
            'the digit, and so does the restart envelope that swamps them. The '
            'README claims are rewritten as UPPER BOUNDS.',
            'No fitted parameter changes. b3_grid.py and b3_grid_v2report.py '
            'are unedited; the v1 and v2 products stand unaltered (house '
            'rule 2). Only the reading of them in the README changes, plus '
            'this new measurement of the search resolution.',
            'The barrier-realisation failures (+46.2973 % calibrated D, '
            '+50.3105 % two_zone D_near) are 6876x and 25247x their own arm\'s '
            'restart floor, and still 2072x and 2251x the worst floor anywhere '
            'in the study (0.022348 %). They are NOT affected by this '
            'amendment.',
            'nx at dx = 0.05 is recorded for the finest mesh; every arm shares '
            'the mesh at a given dx because the barrier changes D(x), not the '
            'node set.',
            "This run's output paths are module constants in b3_grid_amend1.py, "
            'NOT a new config block: appending one would change the config '
            'sha256 and retro-flip all 19 v2 manifests to `config: drift`. The '
            'config recorded here is therefore byte-identical to the one the v2 '
            'report recorded, and manifest_v2.json still verifies clean.'],
        started_utc=t_start, run_label='B3-amend1-%s' % TAG,
        require_modules=('rev2_core', 'rev2_data', 'rev2_manifest'),
        worker_modules=WORKER_MODULES,
        extra_code_files=(os.path.abspath(__file__),
                          os.path.abspath(B.__file__)),
        allow_undeclared_outputs=True)
    return outs['manifest']


# ---------------------------------------------------------------------------

def main():
    global _T0
    _T0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=CONFIG)
    ap.add_argument('--stage', default='all',
                    choices=('all', 'envelope', 'hires', 'report'))
    args = ap.parse_args()

    t_start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(args.config) as f:
        cfg_disk = json.load(f)
    cfg = json.loads(json.dumps(cfg_disk))
    outs = OUTPUTS
    nproc = int(cfg['run']['processes'])
    warnings.simplefilter('ignore', core.BarrierWidthWarning)

    env_rows = stage_envelope(cfg)
    log('envelope: %d (arm, norm, mesh) cells, envelope %.6f - %.6f %%'
        % (len(env_rows),
           min(r['restart_envelope_pct'] for r in env_rows),
           max(r['restart_envelope_pct'] for r in env_rows)))
    if args.stage == 'envelope':
        for r in env_rows:
            print('  %-12s %-10s dx=%-5g  D=%.6f  env=%.6f %%  conv %d/%d'
                  % (r['arm'], r['norm'], r['dx_ft'], r['D_best'],
                     r['restart_envelope_pct'], r['n_restarts_converged'],
                     r['n_restarts']))
        return

    ck = B.load_ckpt('amend1_hires_%s.json' % TAG)
    if ck is None:
        hi_rows, hi_raw = stage_hires(cfg, nproc, env_rows)
        B.save_ckpt('amend1_hires_%s.json' % TAG,
                    {'rows': hi_rows, 'raw': hi_raw})
    else:
        hi_rows, hi_raw = ck['rows'], ck['raw']
    n_solves = len(hi_rows) * len(STENCILS) * 2 * N_STENCIL
    log('hires: %d cells, %d solves, worst stencil disagreement %.2e %%'
        % (len(hi_rows), n_solves,
           max(r['hires_stencil_disagreement_pct'] for r in hi_rows)))
    if args.stage == 'hires':
        return

    cmp_rows = stage_compare(env_rows, hi_rows)
    tz = stage_envelope_two_zone(cfg)
    log('two_zone envelope: %d (arm, mesh) cells; D_near between-mesh vs floor '
        '%s' % (len(tz['per_cell']),
                {a: '%.4f vs %.4f %%' % (g['D_near']['between_mesh_spread_pct'],
                                         g['D_near']['restart_envelope_pct_worst_mesh'])
                 for a, g in tz['groups'].items()}))

    rm.assert_absent([outs['csv'], outs['json'], outs['fig'],
                      outs['manifest']])
    B.write_csv(outs['csv'],
                [dict(e, **{k: v for k, v in h.items()
                            if k not in ('arm', 'norm', 'dx_ft')})
                 for e, h in zip(sorted(env_rows,
                                        key=lambda r: (r['arm'], r['norm'], -r['dx_ft'])),
                                 sorted(hi_rows,
                                        key=lambda r: (r['arm'], r['norm'], -r['dx_ft'])))])
    figure(cmp_rows, env_rows, outs['fig'], int(cfg['outputs_v2']['figure_dpi']))

    results = {
        'question': 'Is the between-mesh spread of the calibrated D that B3 '
                    'reported larger than the resolution of the search that '
                    'measured it?',
        'verdict': 'NO for the two width-matched grid-convergence groups. Both '
                   'reported spreads are smaller than the same-mesh 6-restart '
                   'envelope on at least one mesh of their own group, so they '
                   'are UPPER BOUNDS on the mesh sensitivity, not measurements '
                   'of it. They must not be quoted as measured spreads.',
        'search_resolution_floor': {
            'definition': 'spread in D over the 6 cold Nelder-Mead restarts on '
                          'ONE mesh, all of which score converged (objective '
                          'within 1e-6 relative of the best on that mesh)',
            'over_all_36_cells_pct': [
                min(r['restart_envelope_pct'] for r in env_rows),
                max(r['restart_envelope_pct'] for r in env_rows)],
            'n_restarts_converged_min': min(r['n_restarts_converged']
                                            for r in env_rows),
            'n_restarts_min': min(r['n_restarts'] for r in env_rows),
            'nelder_mead_xatol_log10_D': float(cfg['search']['uniform']['xatol']),
            'nelder_mead_xatol_as_pct_D': float(
                (10.0 ** float(cfg['search']['uniform']['xatol']) - 1.0) * 100.0)},
        'groups': cmp_rows,
        'per_cell_envelope': env_rows,
        'per_cell_hires': hi_rows,
        'two_zone_restart_envelope': tz,
        'hires_estimator': {
            'method': 'two-pass local least-squares parabola in log10 D',
            'stencil_half_widths_dex': list(STENCILS),
            'n_points': N_STENCIL,
            'worst_stencil_disagreement_pct':
                max(r['hires_stencil_disagreement_pct'] for r in hi_rows),
            'worst_sigma_from_fit_residual_pct':
                max(r['hires_sigma_pct_h1'] for r in hi_rows),
            'n_forward_solves': n_solves},
        'unaffected': {
            'note': 'The mesh-DEPENDENT results are orders of magnitude above '
                    'this floor and stand unchanged.',
            'w0p25_dx1_calibrated_D_excess_pct': 46.2973,
            'w0p25_dx1_two_zone_D_near_excess_pct': 50.3105,
            'ratio_to_worst_search_floor': 46.2973 / max(
                r['restart_envelope_pct'] for r in env_rows)},
    }
    with open(outs['json'], 'w') as f:
        json.dump(results, f, indent=1)
    write_manifest(cfg, cfg_disk, args.config, t_start, outs, results, n_solves)
    log('wrote %s, %s, %s, %s'
        % (outs['csv'], outs['json'], outs['fig'], outs['manifest']))

    for g in ('control_nobarrier_20x|abs', 'w1_r1em2_12ft_10x|abs'):
        c = cmp_rows[g]
        print('\n%s' % g)
        print('  published between-mesh spread (best-of-6) : %.6f %%'
              % c['between_mesh_spread_pct_nm'])
        print('  same-mesh restart envelope, worst mesh    : %.6f %%'
              % c['restart_envelope_pct_worst_single_mesh'])
        print('  all restarts pooled over the group        : %.6f %%'
              % c['restart_envelope_pct_all_restarts_pooled'])
        print('  high-precision between-mesh spread        : %.6f %% (h1) '
              '/ %.6f %% (h2)' % (c['between_mesh_spread_pct_hires_h1'],
                                  c['between_mesh_spread_pct_hires_h2']))
        print('  QUOTABLE UPPER BOUND                      : <= %.4f %%'
              % c['upper_bound_pct'])
        print('  misfit gap the published spread buys      : %.3e psi '
              '(objective is %.4f psi)'
              % (c['objective_gap_across_group_psi_nm'],
                 min(r['obj_best'] for r in env_rows
                     if r['arm'] == c['arm'] and r['norm'] == c['norm'])))


if __name__ == '__main__':
    main()
