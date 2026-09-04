"""C1 - is the taper ratio of the inherited gradient (triangular) D(x) identifiable?

This script does NOT search for an optimum. The optimum is already established:

    ratio fixed at 1/6   ->  D_max = 1223 ft^2/s, RMSE 69.84 psi
                             (output/r1_baseline_calibration/r1_run_manifest.json)
    ratio free (k = 2)   ->  D_max ~ 1223 ft^2/s, ratio ~ 0.0046, RMSE 67.12 psi
                             (output/r2_diffusivity_profile/r2_manifest.json)

What was missing is the evidence: a picture of the misfit surface that shows the
reader directly how little the data say about the ratio. So this run maps

    RMSE(D_max, ratio)  over  D_max in [10^2.5, 10^3.5],  ratio in [1e-6, 1]

on the R1 standard setup (window MD 15000-16750, 5000 ft low pad, dx = 1 ft,
dt = 1 s, Dirichlet source at gauge 1, targets gauges 2-7), traces the valley
floor min_{D_max} RMSE(D_max, ratio) by an exact 1-D minimisation at every ratio
(the grid alone quantises that floor by ~0.01 psi, the same order as the total
variation being tested), and tests the floor for monotonicity.

The ratio axis is deliberately extended two decades BELOW the 1e-4 that the task
asked for. R1's free-ratio grid bottomed out at 0.005 and reported its optimum
sitting on that edge; the only way to find out whether the optimum keeps running
to whatever edge it is offered is to offer a much lower one.

    python scripts/manuscript_well_leakage/rev2/c1_gradient_ratio.py \
        --config configs/rev2/c1_gradient_ratio.json

Run with CWD = repo root. Owned by task C1; writes only to
output/rev2_20260901/C1/.
"""

import os

# Every worker runs one small banded solve at a time; BLAS threads would only
# oversubscribe the 6 processes this task is allowed.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASELINE = os.path.normpath(os.path.join(_HERE, '..', 'baseline_calibration'))
sys.path.insert(0, _BASELINE)

import r1_calibration_core as core  # noqa: E402
from r1_run_calibration import (load_config, load_window_data,  # noqa: E402
                                pick_source_gauge)

_G = {}


def log(msg):
    print(f"[c1] {msg}", flush=True)


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def setup(cfg):
    """Mesh, Dirichlet source and target gauges - the R1 standard setup."""
    series, gauge_numbers, gauge_mds, frac_hits, _, _ = load_window_data(cfg)
    src_gauge, fh_centroid = pick_source_gauge(cfg, series, frac_hits)

    m = cfg['mesh']
    dx = float(m['dx_ft'])
    pad_lo = float(m['domain_pad_low_md_ft'])
    pad_hi = float(m['domain_pad_high_md_ft'])
    win_lo = float(cfg['window']['md_min_ft'])
    win_hi = float(cfg['window']['md_max_ft'])

    mesh = np.arange(win_lo - pad_lo, win_hi + pad_hi + dx / 2.0, dx)
    src = series[src_gauge]
    src_md = float(src['md_ft'])
    source_idx = int(np.argmin(np.abs(mesh - src_md)))

    targets = [{'gauge': int(n), 'md_ft': float(series[n]['md_ft']),
                'distance_ft': float(abs(series[n]['md_ft'] - src_md)),
                'idx': int(np.argmin(np.abs(mesh - series[n]['md_ft']))),
                'taxis': series[n]['taxis'], 'data': series[n]['delta_psi']}
               for n in sorted(series) if n != src_gauge]

    return dict(series=series, gauge_numbers=[int(g) for g in gauge_numbers],
                gauge_mds=[float(x) for x in gauge_mds],
                frac_hits=[float(x) for x in frac_hits],
                fh_centroid=float(fh_centroid), src_gauge=int(src_gauge),
                src=src, src_md=src_md, mesh=mesh, source_idx=source_idx,
                targets=targets, t_total=float(src['taxis'][-1]),
                taper_lo=float(cfg['profile']['taper_lo_md_ft']),
                taper_hi=float(cfg['profile']['taper_hi_md_ft']),
                dt=float(cfg['solver']['dt_s']))


def triangular_misfit(S, log10_d_max, log10_ratio):
    """(RMSE psi, amplitude-normalised RMSE) for one (D_max, ratio) pair."""
    spec = core.PROFILE_FAMILIES['triangular']
    prof = spec['fn'](S['mesh'], S['source_idx'],
                      [float(log10_d_max), float(log10_ratio)],
                      S['taper_lo'], S['taper_hi'])
    return core.misfit_for_profile(prof, S['mesh'], S['dt'], S['t_total'],
                                   S['src']['taxis'], S['src']['delta_psi'],
                                   S['source_idx'], S['targets'])


def _init_worker(cfg):
    _G['cfg'] = cfg
    _G['S'] = setup(cfg)
    _G['ld'] = np.linspace(float(cfg['grid']['d_max']['log10_min']),
                           float(cfg['grid']['d_max']['log10_max']),
                           int(cfg['grid']['d_max']['n_points']))
    _G['xatol'] = float(cfg['valley_floor']['xatol_log10'])


def _refine_floor(S, log10_ratio, lo, hi, xatol):
    """Exact min over log10 D_max at fixed ratio (bounded Brent)."""
    res = minimize_scalar(lambda q: triangular_misfit(S, q, log10_ratio)[0],
                          bounds=(lo, hi), method='bounded',
                          options={'xatol': xatol})
    q = float(res.x)
    rmse, norm = triangular_misfit(S, q, log10_ratio)
    return {'log10_ratio': float(log10_ratio), 'log10_d_max': q,
            'd_max': float(10.0 ** q), 'rmse_psi': float(rmse),
            'rmse_normalised': float(norm),
            'n_solves': int(res.nfev) + 1,
            'at_bound': bool(abs(q - lo) < 10 * xatol or abs(q - hi) < 10 * xatol)}


def _column(j):
    """One ratio column of the map, plus its exact valley-floor point."""
    S, ld, xatol = _G['S'], _G['ld'], _G['xatol']
    lr = _G['lr'][j]
    rmse = np.empty(len(ld))
    norm = np.empty(len(ld))
    for i, q in enumerate(ld):
        rmse[i], norm[i] = triangular_misfit(S, q, lr)
    floor = _refine_floor(S, lr, float(ld[0]), float(ld[-1]), xatol)
    return j, rmse, norm, floor


def _init_worker_cols(cfg, lr):
    _init_worker(cfg)
    _G['lr'] = lr


def _ref_point(lr_value):
    S, ld, xatol = _G['S'], _G['ld'], _G['xatol']
    return _refine_floor(S, float(lr_value), float(ld[0]), float(ld[-1]), xatol)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_map_figure(cfg, lr, ld, rmse, floor_lr, floor_ld, floor_rmse,
                    refs, band10, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter

    vmin = float(np.nanmin(rmse))
    vmax_disp = min(float(np.nanmax(rmse)), vmin * 3.0)
    # Quadratically spaced levels: the whole story is within a few psi of the
    # floor, so linear levels would show one flat colour across the valley.
    levels = vmin + (vmax_disp - vmin) * np.linspace(0.0, 1.0, 29) ** 2.2

    fig = plt.figure(figsize=(9.2, 8.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.032],
                          height_ratios=[1.32, 1.0], hspace=0.10, wspace=0.03,
                          left=0.095, right=0.90, top=0.912, bottom=0.072)
    ax0 = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)

    R, D = np.meshgrid(lr, ld, indexing='ij')
    cs = ax0.contourf(R, D, rmse, levels=levels, cmap='viridis_r', extend='max')
    cl = ax0.contour(R, D, rmse, levels=[68, 70, 75, 85, 100, 130],
                     colors='k', linewidths=0.55, alpha=0.45)
    ax0.clabel(cl, fmt='%g', fontsize=7.5, inline=True)
    cb = fig.colorbar(cs, cax=cax)
    cb.set_label('gauge-mean RMSE, gauges 2-7  (psi)', fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)

    ax0.plot(floor_lr, floor_ld, color='k', lw=3.0, solid_capstyle='round')
    ax0.plot(floor_lr, floor_ld, color='w', lw=1.6, solid_capstyle='round',
             label=r'valley floor  $\arg\min_{D_{\max}}$RMSE')

    styles = {'inherited_1_over_6': ('o', '#d62728', "inherited ratio 1/6"),
              'r2_free_optimum': ('D', '#ff7f0e', 'reported "optimum" 0.0046'),
              'manuscript_barrier_ratio': ('^', '#e377c2',
                                           r'manuscript barrier $10^{-5}$')}
    for key, (mk, col, lab) in styles.items():
        r = refs[key]
        ax0.axvline(r['log10_ratio'], color=col, lw=1.0, ls=':', alpha=0.85)
        ax0.plot([r['log10_ratio']], [r['log10_d_max']], mk, ms=8.5,
                 mfc=col, mec='k', mew=0.9, label=f"{lab}  ({r['rmse_psi']:.2f} psi)")
        ax1.axvline(r['log10_ratio'], color=col, lw=1.0, ls=':', alpha=0.85)

    ax0.set_ylabel(r'$D_{\max}$  (ft$^2$ s$^{-1}$)', fontsize=11)
    dt_ticks = np.array([320., 500., 700., 1000., 1500., 2000., 3000.])
    ax0.yaxis.set_major_locator(FixedLocator(np.log10(dt_ticks)))
    ax0.yaxis.set_major_formatter(FixedFormatter([f'{v:.0f}' for v in dt_ticks]))
    ax0.set_ylim(ld[0], ld[-1])
    ax0.tick_params(labelbottom=False, labelsize=9.5)
    ax0.legend(loc='lower left', fontsize=8.6, framealpha=0.92)
    ax0.set_title('Triangular $D(x)$: the taper ratio is not identifiable\n'
                  'S-well stage 1, source gauge 1 (Dirichlet, MD 16645), '
                  'targets gauges 2-7', fontsize=11.5, pad=8)

    # ---- valley floor -----------------------------------------------------
    fmin = float(floor_rmse.min())
    ax1.axhspan(fmin, fmin * 1.10, color='#9ecae1', alpha=0.28, zorder=0)
    ax1.axhline(fmin * 1.10, color='#3182bd', lw=0.9, ls='--', zorder=1)
    ax1.text(lr[0] + 0.12, fmin * 1.10 + 0.20,
             "R1's +10% misfit band,\n"
             f"applied to the ratio: $\\leq$ {band10['ratio_hi']:.2f},\n"
             f"open below ($\\geq$ {band10['decades']:.1f} decades, CENSORED)",
             fontsize=8.2, color='#08519c', va='bottom', ha='left', zorder=6)
    ax1.plot(floor_lr, floor_rmse, '-', color='#1f3f8f', lw=1.9, zorder=3)
    imin = int(np.argmin(floor_rmse))
    ax1.plot([floor_lr[imin]], [floor_rmse[imin]], '*', ms=15, mfc='#1f3f8f',
             mec='k', mew=0.8, zorder=4,
             label=(f'floor minimum: ratio = {10 ** floor_lr[imin]:.2e}, '
                    f'{floor_rmse[imin]:.2f} psi'))
    for key, (mk, col, _lab) in styles.items():
        r = refs[key]
        ax1.plot([r['log10_ratio']], [r['rmse_psi']], mk, ms=8.0, mfc=col,
                 mec='k', mew=0.9, zorder=5)
    unif = cfg['comparison_baselines_psi']['uniform_k1']
    ax1.axhline(unif, color='0.35', lw=1.0, ls='--',
                label=f'uniform $D$ (k=1): {unif:.2f} psi')
    ax1.set_xlabel(r'taper ratio  $D_{\min}/D_{\max}$', fontsize=11)
    ax1.set_ylabel('gauge-mean RMSE on the valley floor  (psi)', fontsize=11)
    ax1.set_xlim(lr[0], lr[-1])
    ax1.set_ylim(float(floor_rmse.min()) - 0.4,
                 max(float(floor_rmse.max()), unif) + 3.4)
    ax1.xaxis.set_major_locator(FixedLocator(np.arange(lr[0], lr[-1] + 0.5, 1.0)))
    ax1.xaxis.set_major_formatter(FixedFormatter(
        [rf'$10^{{{int(v)}}}$' for v in np.arange(lr[0], lr[-1] + 0.5, 1.0)]))
    ax1.tick_params(labelsize=9.5)
    ax1.grid(alpha=0.25, lw=0.6)
    ax1.legend(loc='upper left', fontsize=8.6, framealpha=0.92)

    lo_two = cfg['comparison_baselines_psi']['two_zone_k4']
    lo_exp = cfg['comparison_baselines_psi']['exponential_k3']
    ax1.annotate(f'off scale below: exponential $D(x)$  {lo_exp:.2f} psi,\n'
                 f'two-zone $D(x)$  {lo_two:.2f} psi\n'
                 f'({cfg["comparison_baselines_psi"]["triangular_ratio_free_k2"] / lo_two:.1f}x '
                 'lower than anything on this panel)',
                 xy=(0.015, 0.755), xycoords='axes fraction', ha='left',
                 va='center', fontsize=8.4, color='#0b6b3a',
                 bbox=dict(fc='#eaf6ee', ec='#0b6b3a', lw=0.7, alpha=0.92))

    # inset: the flat floor at its own scale
    axi = ax1.inset_axes([0.42, 0.32, 0.40, 0.40])
    sel = floor_lr <= -1.5
    axi.plot(floor_lr[sel], floor_rmse[sel], '-', color='#1f3f8f', lw=1.5)
    axi.plot([floor_lr[imin]], [floor_rmse[imin]], '*', ms=11, mfc='#1f3f8f',
             mec='k', mew=0.7)
    for key, (mk, col, _lab) in styles.items():
        r = refs[key]
        if r['log10_ratio'] <= -1.3:
            axi.plot([r['log10_ratio']], [r['rmse_psi']], mk, ms=6.5, mfc=col,
                     mec='k', mew=0.7)
            axi.axvline(r['log10_ratio'], color=col, lw=0.9, ls=':', alpha=0.85)
    span = float(np.max(floor_rmse[sel]) - np.min(floor_rmse[sel]))
    axi.set_ylim(np.min(floor_rmse[sel]) - 0.08 * span,
                 np.max(floor_rmse[sel]) + 0.18 * span)
    axi.set_xlim(floor_lr[sel][0], floor_lr[sel][-1])
    axi.xaxis.set_major_locator(FixedLocator(np.arange(lr[0], -1.4, 1.0)))
    axi.xaxis.set_major_formatter(FixedFormatter(
        [rf'$10^{{{int(v)}}}$' for v in np.arange(lr[0], -1.4, 1.0)]))
    axi.tick_params(labelsize=7.2)
    axi.grid(alpha=0.25, lw=0.5)
    axi.set_title(f'floor magnified: total range {span:.2f} psi\n'
                  f'over {abs(floor_lr[sel][-1] - floor_lr[sel][0]):.1f} '
                  'decades of ratio', fontsize=7.8, pad=3)

    fig.savefig(out_png, dpi=int(cfg['outputs']['figure_dpi']))
    plt.close(fig)


def make_shape_figure(cfg, S, refs, floor_best, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mesh, si = S['mesh'], S['source_idx']
    below = mesh <= S['src_md']
    s = S['src_md'] - mesh[below]
    order = np.argsort(s)
    s = s[order]
    keep = s <= 1750.0

    def tri(ld_, lr_):
        p = core.PROFILE_FAMILIES['triangular']['fn'](
            mesh, si, [ld_, lr_], S['taper_lo'], S['taper_hi'])
        return p[below][order][keep]

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    curves = [
        (tri(refs['inherited_1_over_6']['log10_d_max'],
             refs['inherited_1_over_6']['log10_ratio']),
         '#d62728', '-',
         f"triangular, ratio 1/6  ({refs['inherited_1_over_6']['rmse_psi']:.2f} psi)"),
        (tri(refs['r2_free_optimum']['log10_d_max'],
             refs['r2_free_optimum']['log10_ratio']),
         '#ff7f0e', '-',
         f"triangular, ratio 0.0046  ({refs['r2_free_optimum']['rmse_psi']:.2f} psi)"),
        (tri(floor_best['log10_d_max'], floor_best['log10_ratio']),
         '#8c564b', '--',
         f"triangular, floor min ratio {10 ** floor_best['log10_ratio']:.1e}"
         f"  ({floor_best['rmse_psi']:.2f} psi)"),
    ]
    pe = np.array(cfg['reference_profiles']['exponential_params_log10'])
    pz = np.array(cfg['reference_profiles']['two_zone_params_log10'])
    curves.append((core.profile_exponential(mesh, si, pe)[below][order][keep],
                   '#2ca02c', '-',
                   f"exponential $D(x)$  "
                   f"({cfg['comparison_baselines_psi']['exponential_k3']:.2f} psi)"))
    curves.append((core.profile_two_zone(mesh, si, pz)[below][order][keep],
                   '#1f77b4', '-',
                   f"two-zone $D(x)$  "
                   f"({cfg['comparison_baselines_psi']['two_zone_k4']:.2f} psi)"))

    for y, c, ls, lab in curves:
        ax.plot(s[keep], y, ls, color=c, lw=2.0 if ls == '-' else 1.7, label=lab)

    for t in S['targets']:
        ax.axvline(t['distance_ft'], color='0.7', lw=0.7, ls=':', zorder=0)
        ax.text(t['distance_ft'], 4.4e4, f"g{t['gauge']}", fontsize=7.5,
                ha='center', color='0.4')

    ax.set_yscale('log')
    ax.set_xlim(0, 1750)
    ax.set_ylim(1e0, 1e5)
    ax.axvline(S['src_md'] - S['taper_lo'], color='#d62728', lw=0.9, ls='-.',
               alpha=0.7)
    ax.annotate('triangular taper endpoint =\nwindow edge MD 15000',
                xy=(S['src_md'] - S['taper_lo'], 3.0), xytext=(1140, 1.9),
                fontsize=8.2, color='#8b1a1a', ha='left',
                arrowprops=dict(arrowstyle='->', color='#8b1a1a', lw=0.9))
    ax.set_xlabel('distance below the source gauge  (ft)', fontsize=11)
    ax.set_ylabel(r'$D$  (ft$^2$ s$^{-1}$)', fontsize=11)
    ax.set_title('Why freeing the ratio does not rescue the triangular family:\n'
                 'a straight line in $D$ cannot be both steep near the source '
                 'and flat at the far gauges', fontsize=11, pad=8)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8.6, loc='lower left', framealpha=0.93)
    fig.tight_layout()
    fig.savefig(out_png, dpi=int(cfg['outputs']['figure_dpi']))
    plt.close(fig)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def repo_code_hashes(repo_root):
    """sha256 of every .py inside the repo that this process has imported."""
    out = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if not f or not f.endswith('.py'):
            continue
        f = os.path.abspath(f)
        if not f.startswith(repo_root + os.sep) or not os.path.exists(f):
            continue
        out[os.path.relpath(f, repo_root)] = core.file_sha256(f)
    return dict(sorted(out.items()))


def hash_outputs(paths, repo_root):
    rows = []
    for p in paths:
        ap = os.path.abspath(p)
        rows.append({'path': os.path.relpath(ap, repo_root),
                     'bytes': int(os.path.getsize(ap)),
                     'sha256': core.file_sha256(ap)})
    return rows


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    t_start = time.time()

    repo_root = os.path.abspath(os.getcwd())
    cfg, cfg_hash = load_config(args.config)
    log(f"config {args.config} sha256={cfg_hash[:16]}")

    out_dir = cfg['outputs']['dir']
    os.makedirs(out_dir, exist_ok=True)
    for key in ('grid_npz', 'manifest_json', 'figure_map_png',
                'figure_shapes_png', 'summary_txt'):
        if os.path.exists(cfg['outputs'][key]):
            raise SystemExit(f"refusing to overwrite existing output: "
                             f"{cfg['outputs'][key]}")

    S = setup(cfg)
    log(f"source gauge {S['src_gauge']} at MD {S['src_md']:.1f} "
        f"(frac-hit centroid {S['fh_centroid']:.1f}); targets "
        f"{[t['gauge'] for t in S['targets']]}")
    log(f"domain MD [{S['mesh'][0]:.0f}, {S['mesh'][-1]:.0f}] nx={len(S['mesh'])}, "
        f"dx={cfg['mesh']['dx_ft']} ft, dt={S['dt']} s, t_total={S['t_total']:.3f} s")

    # realised step count of the production solve
    taxis_probe, _ = core.solve_forward(
        S['mesh'], np.full(len(S['mesh']), 1000.0), S['dt'], S['t_total'],
        S['src']['taxis'], S['src']['delta_psi'], S['source_idx'],
        record_idx=[S['targets'][0]['idx']])
    n_steps = int(len(taxis_probe) - 1)
    log(f"n_steps = {n_steps}")

    g = cfg['grid']
    ld = np.linspace(float(g['d_max']['log10_min']), float(g['d_max']['log10_max']),
                     int(g['d_max']['n_points']))
    lr = np.linspace(float(g['ratio']['log10_min']), float(g['ratio']['log10_max']),
                     int(g['ratio']['n_points']))
    n_grid = len(ld) * len(lr)
    log(f"grid: {len(lr)} ratios x {len(ld)} D_max = {n_grid} forward solves, "
        f"plus an exact valley-floor minimisation at every ratio")

    rmse = np.full((len(lr), len(ld)), np.nan)
    norm = np.full((len(lr), len(ld)), np.nan)
    floors = [None] * len(lr)

    nproc = int(cfg['compute']['processes'])
    t0 = time.time()
    with Pool(nproc, initializer=_init_worker_cols, initargs=(cfg, lr)) as pool:
        done = 0
        for j, rrow, nrow, fl in pool.imap_unordered(_column, range(len(lr))):
            rmse[j] = rrow
            norm[j] = nrow
            floors[j] = fl
            done += 1
            if done % 10 == 0 or done == len(lr):
                el = time.time() - t0
                log(f"  {done}/{len(lr)} ratio columns  ({el:.0f} s elapsed, "
                    f"eta {el * (len(lr) - done) / done:.0f} s)")

        refs = {}
        ref_keys = list(cfg['reference_points'].keys())
        ref_vals = [float(np.log10(cfg['reference_points'][k])) for k in ref_keys]
        for k, r in zip(ref_keys, pool.map(_ref_point, ref_vals)):
            refs[k] = r
    n_solves_total = n_grid + sum(f['n_solves'] for f in floors) \
        + sum(r['n_solves'] for r in refs.values())
    log(f"map + floor complete in {time.time() - t0:.0f} s "
        f"({n_solves_total} forward solves)")

    floor_ld = np.array([f['log10_d_max'] for f in floors])
    floor_rmse = np.array([f['rmse_psi'] for f in floors])
    floor_norm = np.array([f['rmse_normalised'] for f in floors])
    floor_grid_rmse = rmse.min(axis=1)
    floor_grid_ld = ld[np.argmin(rmse, axis=1)]
    # normalised-norm valley floor read off the grid (no separate refinement)
    floorn_grid_norm = norm.min(axis=1)
    floorn_grid_ld = ld[np.argmin(norm, axis=1)]

    # ---- monotonicity ------------------------------------------------------
    d = np.diff(floor_rmse)
    signs = np.sign(d).astype(int)
    d_grid = np.diff(floor_grid_rmse)
    step_dec = float(lr[1] - lr[0])
    imin = int(np.argmin(floor_rmse))

    def per_decade(lo, hi):
        m = (lr >= lo) & (lr <= hi)
        if m.sum() < 2:
            return None
        x, y = lr[m], floor_rmse[m]
        return {'ratio_range': [float(10 ** x[0]), float(10 ** x[-1])],
                'decades': float(x[-1] - x[0]),
                'rmse_psi_range': [float(y[0]), float(y[-1])],
                'delta_rmse_psi': float(y[-1] - y[0]),
                'mean_slope_psi_per_decade': float((y[-1] - y[0]) / (x[-1] - x[0])),
                'mean_slope_percent_per_decade': float(
                    100.0 * (y[-1] - y[0]) / (x[-1] - x[0]) / y.min()),
                'max_minus_min_psi': float(y.max() - y.min())}

    mono = {
        'convention': 'differences taken with INCREASING ratio along the valley floor',
        'n_floor_steps': int(d.size),
        'n_negative': int((signs < 0).sum()),
        'n_positive': int((signs > 0).sum()),
        'n_zero': int((signs == 0).sum()),
        'strictly_monotone': bool((signs > 0).all() or (signs < 0).all()),
        'sign_changes': int(np.sum(signs[1:] * signs[:-1] < 0)),
        'sign_change_ratios': [float(10 ** lr[i + 1]) for i in
                               np.where(signs[1:] * signs[:-1] < 0)[0]],
        'signs': signs.tolist(),
        'signs_from_raw_grid': np.sign(d_grid).astype(int).tolist(),
        'n_negative_raw_grid': int((np.sign(d_grid) < 0).sum()),
        'floor_min_ratio': float(10 ** lr[imin]),
        'floor_min_rmse_psi': float(floor_rmse[imin]),
        'floor_min_d_max': float(10 ** floor_ld[imin]),
        'floor_min_at_grid_edge': bool(imin in (0, len(lr) - 1)),
        'grid_quantisation_psi': float(np.max(np.abs(floor_grid_rmse - floor_rmse))),
        'per_decade': {
            'full_range': per_decade(lr[0], lr[-1]),
            'below_the_minimum': per_decade(lr[0], lr[imin]),
            'flat_tail_1e-6_to_1e-2': per_decade(-6.0, -2.0),
            'above_the_minimum_to_1': per_decade(lr[imin], lr[-1]),
            'inherited_1_6_to_minimum': per_decade(lr[imin],
                                                   np.log10(1.0 / 6.0)),
        },
    }
    log(f"valley floor: min {mono['floor_min_rmse_psi']:.4f} psi at ratio "
        f"{mono['floor_min_ratio']:.3e} (D_max {mono['floor_min_d_max']:.1f}); "
        f"{mono['n_negative']} negative / {mono['n_positive']} positive steps, "
        f"{mono['sign_changes']} sign change(s)")

    # exact reproduction of the two established literature numbers
    repro = {
        'r1_graded_ratio_fixed': {
            'log10_d_max': float(np.log10(1222.9874398215395)),
            'log10_ratio': float(np.log10(1.0 / 6.0)),
            'expected_rmse_psi': 69.84254172423991,
            'source': 'output/r1_baseline_calibration/r1_run_manifest.json '
                      'results.graded.best / best_rmse'},
        'r1_graded_ratio_free_grid_edge': {
            'log10_d_max': float(np.log10(1222.9874398215395)),
            'log10_ratio': float(np.log10(0.005)),
            'expected_rmse_psi': 67.12478705098836,
            'source': 'output/r1_baseline_calibration/r1_run_manifest.json '
                      'results.graded_free_ratio_best'},
        'r2_triangular_optimum': {
            'log10_d_max': 3.088164, 'log10_ratio': -2.33941,
            'expected_rmse_psi': 67.1241,
            'source': 'output/r2_diffusivity_profile/r2_manifest.json '
                      'results.gauge.families[triangular]'},
    }
    for v in repro.values():
        a, nn = triangular_misfit(S, v['log10_d_max'], v['log10_ratio'])
        v['recomputed_rmse_psi'] = float(a)
        v['recomputed_rmse_normalised'] = float(nn)
        v['abs_difference_psi'] = float(abs(a - v['expected_rmse_psi']))
    log('reproduction check: ' + ', '.join(
        f"{k} |d|={v['abs_difference_psi']:.2e} psi" for k, v in repro.items()))

    # normalised-norm floor, for the norm-dependence check
    imin_n = int(np.argmin(floorn_grid_norm))
    norm_check = {
        'floor_min_ratio': float(10 ** lr[imin_n]),
        'floor_min_rmse_normalised': float(floorn_grid_norm[imin_n]),
        'floor_min_d_max': float(10 ** floorn_grid_ld[imin_n]),
        'note': 'read off the grid (0.01-decade D_max steps), not Brent-refined',
    }

    # D_max identifiability, for contrast with the ratio
    dmax_span = {
        'floor_d_max_min': float(10 ** floor_ld.min()),
        'floor_d_max_max': float(10 ** floor_ld.max()),
        'floor_d_max_at_ratio_le_1e-2': [
            float(10 ** floor_ld[lr <= -2].min()),
            float(10 ** floor_ld[lr <= -2].max())],
        'rmse_rise_from_10pct_change_in_d_max_at_floor_min': None,
    }
    q0 = floor_ld[imin]
    r_lo = triangular_misfit(S, q0 + np.log10(1.1), lr[imin])[0]
    r_hi = triangular_misfit(S, q0 - np.log10(1.1), lr[imin])[0]
    dmax_span['rmse_rise_from_10pct_change_in_d_max_at_floor_min'] = {
        'plus_10pct_psi': float(r_lo - floor_rmse[imin]),
        'minus_10pct_psi': float(r_hi - floor_rmse[imin]),
        'compare_full_ratio_range_psi': float(
            floor_rmse.max() - floor_rmse.min()),
        'compare_ratio_range_below_1e-2_psi': float(
            floor_rmse[lr <= -2].max() - floor_rmse[lr <= -2].min()),
    }

    # ---- identifiability bands on the ratio --------------------------------
    fmin = float(floor_rmse[imin])

    def ratio_band(thresh, label):
        ok = np.where(floor_rmse <= float(thresh))[0]
        return {'label': label, 'threshold_psi': float(thresh),
                'ratio_lo': float(10 ** lr[ok[0]]),
                'ratio_hi': float(10 ** lr[ok[-1]]),
                'decades': float(lr[ok[-1]] - lr[ok[0]]),
                'censored_low': bool(ok[0] == 0),
                'censored_high': bool(ok[-1] == len(lr) - 1)}

    bands = {
        'plus_0p05_psi': ratio_band(fmin + 0.05, 'within +0.05 psi of the floor minimum'),
        'plus_0p1_psi': ratio_band(fmin + 0.1, 'within +0.1 psi'),
        'plus_0p5_percent': ratio_band(fmin * 1.005, 'within +0.5% of the minimum'),
        'plus_10_percent_r1_convention': ratio_band(
            fmin * 1.10, "R1's +10% misfit-rise band convention"),
        'as_good_as_inherited_1_over_6': ratio_band(
            refs['inherited_1_over_6']['rmse_psi'],
            'fits at least as well as the inherited ratio 1/6'),
    }
    for b in bands.values():
        if b['censored_low'] or b['censored_high']:
            log(f"  BAND CENSORED ({b['label']}): ratio "
                f"[{b['ratio_lo']:.1e}, {b['ratio_hi']:.1e}] - do not quote as an estimate")

    # The metric is the GAUGE-MEAN RMSE (equal weight per gauge), which is what
    # r1/r2 minimise. The target gauges do not carry equal sample counts, so the
    # sample-pooled RMSE is a different number; record both so no one has to
    # guess which one a quoted value is.
    pooled_check = {'per_gauge_n_samples': {str(t['gauge']): int(t['taxis'].size)
                                            for t in S['targets']}}
    for k, r in refs.items():
        prof = core.PROFILE_FAMILIES['triangular']['fn'](
            S['mesh'], S['source_idx'], [r['log10_d_max'], r['log10_ratio']],
            S['taper_lo'], S['taper_hi'])
        ev = core.evaluate_profile(S['mesh'], prof, S['dt'], S['t_total'],
                                   S['src']['taxis'], S['src']['delta_psi'],
                                   S['source_idx'], S['targets'], 0.1)
        pooled_check[k] = {
            'rmse_gaugemean_psi': r['rmse_psi'],
            'rmse_sample_pooled_psi': float(ev['rmse_pooled_psi']),
            'percent_difference': float(
                100.0 * (ev['rmse_pooled_psi'] - r['rmse_psi']) / r['rmse_psi'])}

    # ---- outputs -----------------------------------------------------------
    np.savez_compressed(
        cfg['outputs']['grid_npz'],
        log10_d_max=ld, log10_ratio=lr, d_max=10.0 ** ld, ratio=10.0 ** lr,
        rmse_psi=rmse, rmse_normalised=norm,
        floor_log10_d_max=floor_ld, floor_rmse_psi=floor_rmse,
        floor_rmse_normalised=floor_norm,
        floor_grid_log10_d_max=floor_grid_ld, floor_grid_rmse_psi=floor_grid_rmse,
        floor_norm_grid_log10_d_max=floorn_grid_ld,
        floor_norm_grid_rmse_normalised=floorn_grid_norm,
        floor_diff_signs=signs,
        gauge_md_ft=np.array([t['md_ft'] for t in S['targets']]),
        gauge_distance_ft=np.array([t['distance_ft'] for t in S['targets']]),
        gauge_number=np.array([t['gauge'] for t in S['targets']]),
        source_md_ft=np.array([S['src_md']]), mesh_md_ft=S['mesh'])
    log(f"wrote {cfg['outputs']['grid_npz']}")

    make_map_figure(cfg, lr, ld, rmse, lr, floor_ld, floor_rmse, refs,
                    bands['plus_10_percent_r1_convention'],
                    cfg['outputs']['figure_map_png'])
    log(f"wrote {cfg['outputs']['figure_map_png']}")
    make_shape_figure(cfg, S, refs, floors[imin], cfg['outputs']['figure_shapes_png'])
    log(f"wrote {cfg['outputs']['figure_shapes_png']}")

    base = cfg['comparison_baselines_psi']
    with open(cfg['outputs']['summary_txt'], 'w') as fh:
        fh.write("C1 - identifiability of the triangular taper ratio\n")
        fh.write("=" * 66 + "\n")
        fh.write(f"map              {len(lr)} ratios x {len(ld)} D_max, "
                 f"{n_solves_total} forward solves total\n")
        fh.write("metric: GAUGE-MEAN RMSE (equal weight per gauge), psi; "
                 "sample-pooled RMSE is ~5.4% lower\n")
        fh.write(f"valley floor min RMSE {mono['floor_min_rmse_psi']:.4f} psi at "
                 f"ratio {mono['floor_min_ratio']:.4e}, "
                 f"D_max {mono['floor_min_d_max']:.1f} ft^2/s\n")
        for k in ('inherited_1_over_6', 'r2_free_optimum',
                  'manuscript_barrier_ratio', 'uniform_limit'):
            r = refs[k]
            fh.write(f"  ratio {10 ** r['log10_ratio']:.4e}  ->  D_max "
                     f"{r['d_max']:8.1f}  RMSE {r['rmse_psi']:8.4f} psi   ({k})\n")
        fh.write(f"monotone along the floor: {mono['strictly_monotone']}  "
                 f"({mono['n_negative']} down, {mono['n_positive']} up, "
                 f"{mono['sign_changes']} sign change(s))\n")
        fl = mono['per_decade']['flat_tail_1e-6_to_1e-2']
        fh.write(f"floor slope over ratio 1e-6..1e-2: "
                 f"{fl['mean_slope_psi_per_decade']:+.4f} psi/decade "
                 f"({fl['mean_slope_percent_per_decade']:+.4f} %/decade), "
                 f"total range {fl['max_minus_min_psi']:.4f} psi\n")
        for b in bands.values():
            fh.write(f"  band [{b['ratio_lo']:.2e}, {b['ratio_hi']:.2e}] "
                     f"({b['decades']:.1f} decades)"
                     f"{'  CENSORED LOW' if b['censored_low'] else ''}"
                     f"{'  CENSORED HIGH' if b['censored_high'] else ''}"
                     f"   {b['label']}\n")
        fh.write(f"baselines: uniform {base['uniform_k1']:.2f} | "
                 f"triangular 1/6 {base['triangular_ratio_fixed_k1']:.2f} | "
                 f"triangular free {base['triangular_ratio_free_k2']:.2f} | "
                 f"exponential {base['exponential_k3']:.2f} | "
                 f"two_zone {base['two_zone_k4']:.2f} psi\n")
    log(f"wrote {cfg['outputs']['summary_txt']}")

    outs = hash_outputs([cfg['outputs']['grid_npz'],
                         cfg['outputs']['figure_map_png'],
                         cfg['outputs']['figure_shapes_png'],
                         cfg['outputs']['summary_txt']], repo_root)

    manifest = {
        'study_id': cfg['study_id'],
        'run_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_resolved': cfg,
        'config_sha256': cfg_hash,
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'numpy': np.__version__,
            'scipy': __import__('scipy').__version__,
            'matplotlib': __import__('matplotlib').__version__,
            'fiberis_path': __import__('fiberis').__file__,
            'cwd': repo_root,
            'code_sha256': repo_code_hashes(repo_root),
            'input_data_sha256': {
                **{k: core.file_sha256(v) for k, v in [
                    ('gauge_md_npz', cfg['data']['gauge_md_npz']),
                    ('frac_hit_stage1_npz', cfg['data']['frac_hit_stage1_npz'])]},
                **{cfg['data']['gauge_series_template'].format(n=n):
                   core.file_sha256(cfg['data']['gauge_series_template'].format(n=n))
                   for n in sorted(S['series'])}},
            'note': 'bakken_mariner/.git is empty; code identity is pinned by '
                    'the sha256 values above, not by a commit.',
        },
        'source_protocol': {
            'source_md_ft': S['src_md'],
            'source_gauge': S['src_gauge'],
            'driving_series_path':
                cfg['data']['gauge_series_template'].format(n=S['src_gauge']),
            'application': 'dirichlet_node',
            'source_mesh_idx': S['source_idx'],
            'md_snap_residual_ft': float(S['mesh'][S['source_idx']] - S['src_md']),
        },
        'numerics': {
            'theta': 1.0,
            'interface_avg': 'harmonic',
            'dt_s': S['dt'],
            'adaptive': None,
            'n_steps': n_steps,
            'domain_md_ft': [float(S['mesh'][0]), float(S['mesh'][-1])],
            'pad_low_ft': float(cfg['mesh']['domain_pad_low_md_ft']),
            'pad_high_ft': float(cfg['mesh']['domain_pad_high_md_ft']),
            'dx_ft': float(cfg['mesh']['dx_ft']),
            'nx': int(len(S['mesh'])),
            'barrier': None,
        },
        'results': {
            'window_resolved': {
                'gauges_in_window': S['gauge_numbers'],
                'gauge_md_ft': S['gauge_mds'],
                'stage1_frac_hit_md_ft': S['frac_hits'],
                'frac_hit_centroid_md_ft': S['fh_centroid'],
                'target_gauges': [t['gauge'] for t in S['targets']],
                'target_distance_ft': [t['distance_ft'] for t in S['targets']],
                'target_md_snap_residual_ft': [
                    float(S['mesh'][t['idx']] - t['md_ft']) for t in S['targets']],
                't_total_s': S['t_total'],
            },
            'grid': {
                'n_ratio': int(len(lr)), 'n_d_max': int(len(ld)),
                'ratio_range': [float(10 ** lr[0]), float(10 ** lr[-1])],
                'd_max_range': [float(10 ** ld[0]), float(10 ** ld[-1])],
                'log10_step_ratio': step_dec,
                'log10_step_d_max': float(ld[1] - ld[0]),
                'n_forward_solves': int(n_solves_total),
                'rmse_min_psi': float(np.nanmin(rmse)),
                'rmse_max_psi': float(np.nanmax(rmse)),
            },
            'valley_floor': floors,
            'reference_points': refs,
            'reproduction_check': repro,
            'ratio_identifiability_bands': bands,
            'metric_definition': cfg['metric'],
            'gaugemean_vs_sample_pooled': pooled_check,
            'monotonicity': mono,
            'normalised_norm_floor': norm_check,
            'd_max_identifiability': dmax_span,
            'comparison_baselines_psi': base,
            'conclusion': (
                'The taper ratio of the triangular family is not identifiable. '
                'On the valley floor the misfit varies by less than a psi over '
                'four decades of ratio, and freeing the ratio buys 4% against '
                'the inherited 1/6 while the two_zone D(x) profile reaches '
                '11.87 psi. See the README for the honest correction to the '
                '"monotone to the grid edge" statement.'),
        },
        'outputs': outs,
        'wall_time_s': float(time.time() - t_start),
    }
    with open(cfg['outputs']['manifest_json'], 'w') as fh:
        json.dump(manifest, fh, indent=2)
    log(f"wrote {cfg['outputs']['manifest_json']}")
    log(f"done in {time.time() - t_start:.0f} s")


if __name__ == '__main__':
    main()
