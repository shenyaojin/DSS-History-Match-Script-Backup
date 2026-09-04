"""B3 -- grid independence under a FULL RE-FIT.

A1 established that with a PHYSICAL barrier half-width the FORWARD solution is
mesh-independent: at a fixed D0 = 1150 ft^2/s the worst per-gauge RMSE changes by
0.0118 % from dx = 1.0 to 0.1 ft. That is a statement about one point in parameter
space. B3 asks the harder question the task package poses: does the CALIBRATED
answer move with the mesh?

Protocol: hold w and ratio fixed, sweep dx, and RE-FIT the diffusivity model from a
COLD START on every mesh -- for two families of very different flexibility, because
a flexible family has more freedom to absorb a mesh change and whether it does is
the interesting part:

    uniform   k = 1   log10 D
    two_zone  k = 4   log10 D_near, log10 D_far, log10 s_c, log10 width

Three arms, each with w and ratio frozen:

    control        no barrier                    -- the control; any dx dependence
                                                    here is not the barrier's
    w1_r1em2       w = 1.0 ft, ratio = 1e-2      -- the headline; 2w/dx = 2 .. 40
    w0p25_r1em2    w = 0.25 ft, ratio = 1e-2     -- deliberately UNDER-RESOLVED at
                                                    dx = 1.0 (2w/dx = 0.5), where
                                                    three of six barriers fall back
                                                    to the nearest node and the
                                                    realised width doubles

Nothing is warm-started. D1's blocker was a warm start from an optimum computed
with the held-out gauge, which hid exactly the variability that must be measured
here, so every fit begins from a start set that does not depend on dx and every fit
reports a multi-restart envelope rather than a single number.

Run from the repo root:

    python3 scripts/manuscript_well_leakage/rev2/b3_grid.py --stage all

Stages are independently resumable (`--stage geom|uniform|two_zone|crossmesh|report`)
and checkpoint into output/rev2_20260901/B3/_ckpt/.

Owns: configs/rev2/b3_grid.json, output/rev2_20260901/B3/,
      scripts/manuscript_well_leakage/rev2/b3_grid.py.
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
from scipy.optimize import minimize
from scipy.stats import qmc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_BC = os.path.join(os.path.dirname(_HERE), 'baseline_calibration')
if _BC not in sys.path:
    sys.path.insert(0, _BC)

import rev2_core as core        # noqa: E402
import rev2_data as rd          # noqa: E402
import rev2_manifest as rm      # noqa: E402
import r1_calibration_core as r1c  # noqa: E402

DEFAULT_CONFIG = 'configs/rev2/b3_grid.json'
CKPT_DIR = 'output/rev2_20260901/B3/_ckpt'


def log(msg):
    print('[%s] %s' % (datetime.datetime.now().strftime('%H:%M:%S'), msg),
          flush=True)


def dx_tag(dx):
    return ('%g' % dx).replace('.', 'p')


def load_config(path):
    with open(path) as fh:
        return json.load(fh)


def ckpt_path(name):
    return os.path.join(CKPT_DIR, name)


def save_ckpt(name, obj):
    os.makedirs(CKPT_DIR, exist_ok=True)
    tmp = ckpt_path(name) + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, ckpt_path(name))


def load_ckpt(name):
    p = ckpt_path(name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# worker side: one cached setup per (dx), shared by every arm and family
# ---------------------------------------------------------------------------

_W = {}


def _init_worker(cfg):
    _W['cfg'] = cfg
    _W['setups'] = {}
    _W['fh'] = rd.load_frac_hits(2)
    warnings.simplefilter('ignore', core.BarrierWidthWarning)


def _setup(dx):
    """Cache rev2_data.setup_r1 per dx inside the worker process."""
    S = _W['setups'].get(dx)
    if S is None:
        S = rd.setup_r1(dx_ft=float(dx),
                        pad_low_ft=float(_W['cfg']['mesh']['domain_pad_low_md_ft']),
                        pad_high_ft=float(_W['cfg']['mesh']['domain_pad_high_md_ft']))
        S['_mesh_x'] = S['mesh'].x
        S['_rec'] = [t['idx'] for t in S['targets']]
        _W['setups'][dx] = S
    return S


def _arm_by_name(cfg, name):
    for a in cfg['arms']:
        if a['name'] == name:
            return a
    raise KeyError(name)


def build_profile(cfg, S, fh, family, params, arm):
    """Baseline D(x) from the family, then the frozen barrier on top."""
    mesh = S['_mesh_x']
    base = r1c.PROFILE_FAMILIES[family]['fn'](mesh, S['source_idx'],
                                              np.asarray(params, dtype=float))
    if not arm['barrier']:
        return base
    b = cfg['barrier']
    return core.build_barrier_profile(
        mesh, base, fh, float(arm['w_half_width_ft']), float(arm['ratio']),
        ratio_reference=b['ratio_reference'], combine=b['combine'],
        on_empty=b['on_empty'], on_outside=b['on_outside'])


def _misfit(task):
    """(arm_name, dx, family, params) -> (rmse_psi, rmse_normalised)."""
    arm_name, dx, family, params = task
    cfg = _W['cfg']
    S = _setup(dx)
    arm = _arm_by_name(cfg, arm_name)
    prof = build_profile(cfg, S, _W['fh'], family, params, arm)
    s = cfg['solver']
    return r1c.misfit_for_profile(
        prof, S['_mesh_x'], float(s['dt_s']), float(S['t_total_s']),
        S['src_series'].taxis_s, S['src_series'].delta_psi,
        S['source_idx'], S['targets'])


def _nm_uniform(task):
    """One cold 1-D Nelder-Mead restart. Runs entirely inside a worker."""
    arm_name, dx, x0, norm_idx, lo, hi, opts = task
    n = [0]

    def f(q):
        n[0] += 1
        return _misfit((arm_name, dx, 'uniform',
                        [float(np.clip(q[0], lo, hi))]))[norm_idx]

    res = minimize(f, [float(x0)], method='Nelder-Mead', options=opts)
    x = float(np.clip(res.x[0], lo, hi))
    a, nn = _misfit((arm_name, dx, 'uniform', [x]))
    return {'arm': arm_name, 'dx_ft': dx, 'norm': ('abs' if norm_idx == 0
                                                   else 'normalised'),
            'x0_log10_D': float(x0), 'log10_D': x, 'D_ft2_s': float(10.0 ** x),
            'rmse_psi': float(a), 'rmse_normalised': float(nn),
            'objective': float(res.fun), 'nfev': int(n[0]),
            'at_bound': bool(abs(x - lo) < 1e-9 or abs(x - hi) < 1e-9)}


def _nm_two_zone(task):
    """One cold 4-D Nelder-Mead restart plus a simplex-restart polish."""
    arm_name, dx, x0, bounds, opts1, opts2 = task
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    n = [0]
    t0 = time.time()

    def f(q):
        n[0] += 1
        return _misfit((arm_name, dx, 'two_zone', np.clip(q, lo, hi)))[0]

    r1 = minimize(f, np.asarray(x0, float), method='Nelder-Mead', options=opts1)
    r2 = minimize(f, np.clip(r1.x, lo, hi), method='Nelder-Mead', options=opts2)
    best = r2 if float(r2.fun) <= float(r1.fun) else r1
    x = np.clip(np.asarray(best.x, float), lo, hi)
    a, nn = _misfit((arm_name, dx, 'two_zone', x))
    return {'arm': arm_name, 'dx_ft': dx, 'x0': [float(v) for v in x0],
            'params': [float(v) for v in x],
            'rmse_psi': float(a), 'rmse_normalised': float(nn),
            'nfev': int(n[0]), 'wall_s': float(time.time() - t0),
            'at_bound': [bool(abs(v - b[0]) < 1e-9 or abs(v - b[1]) < 1e-9)
                         for v, b in zip(x, bounds)]}


# ---------------------------------------------------------------------------
# stage: geometry / barrier realisation (no solves)
# ---------------------------------------------------------------------------

def stage_geom(cfg, tag):
    fh = rd.load_frac_hits(2)
    dxs = sorted(set(cfg['families']['uniform']['dx_ft_sweep']
                     + cfg['families']['two_zone']['dx_ft_sweep']), reverse=True)
    rows = []
    for dx in dxs:
        S = rd.setup_r1(dx_ft=dx,
                        pad_low_ft=float(cfg['mesh']['domain_pad_low_md_ft']),
                        pad_high_ft=float(cfg['mesh']['domain_pad_high_md_ft']))
        mesh = S['mesh']
        snap_src = mesh.snap_error_ft(S['src_md'])
        snap_tgt = max(abs(mesh.snap_error_ft(t['md_ft'])) for t in S['targets'])
        for arm in cfg['arms']:
            if not arm['barrier']:
                rows.append(dict(arm=arm['name'], dx_ft=dx, nx=mesh.nx,
                                 w_ft='', two_w_over_dx='', n_fallback=0,
                                 realised_full_width_min_ft='',
                                 realised_full_width_med_ft='',
                                 realised_full_width_max_ft='',
                                 excess_resistance_s_per_ft=0.0,
                                 max_centre_offset_ft='',
                                 snap_source_ft=snap_src, snap_target_max_ft=snap_tgt))
                continue
            w = float(arm['w_half_width_ft'])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', core.BarrierWidthWarning)
                _, rep = core.build_barrier_profile(
                    mesh.x, np.full(mesh.nx, 1150.0), fh, w, float(arm['ratio']),
                    ratio_reference=cfg['barrier']['ratio_reference'],
                    combine=cfg['barrier']['combine'],
                    on_empty=cfg['barrier']['on_empty'],
                    on_outside=cfg['barrier']['on_outside'], return_report=True)
            rw = rep['realised_full_width_ft']
            rows.append(dict(
                arm=arm['name'], dx_ft=dx, nx=mesh.nx, w_ft=w,
                two_w_over_dx=2.0 * w / dx, n_fallback=int(rep['n_fallback']),
                realised_full_width_min_ft=rw['min'],
                realised_full_width_med_ft=rw['median'],
                realised_full_width_max_ft=rw['max'],
                excess_resistance_s_per_ft=rep['excess_resistance_s_per_ft'],
                max_centre_offset_ft=max(abs(b['center_offset_ft'])
                                         for b in rep['barriers']),
                snap_source_ft=snap_src, snap_target_max_ft=snap_tgt))
    save_ckpt('geom_%s.json' % tag, rows)
    log('geom: %d rows' % len(rows))
    return rows


# ---------------------------------------------------------------------------
# stage: uniform family
# ---------------------------------------------------------------------------

def stage_uniform(cfg, tag, pool):
    fam = cfg['families']['uniform']
    lo, hi = fam['bounds'][0]
    g = fam['profile_grid']
    grid = np.linspace(g['lo'], g['hi'], int(g['n']))
    dxs = list(fam['dx_ft_sweep'])
    arms = [a['name'] for a in cfg['arms']]

    # --- misfit profiles (both norms from the same solve) -------------------
    prof = load_ckpt('uniform_grid_%s.json' % tag)
    if prof is None:
        tasks = [(a, dx, 'uniform', [float(x)])
                 for a in arms for dx in dxs for x in grid]
        # coarse meshes first so the pool stays busy while the fine ones run
        order = sorted(range(len(tasks)), key=lambda i: -tasks[i][1])
        t0 = time.time()
        vals = [None] * len(tasks)
        for k, res in zip(order, pool.imap(_misfit, [tasks[i] for i in order],
                                           chunksize=4)):
            vals[k] = res
        prof = {}
        i = 0
        for a in arms:
            for dx in dxs:
                n = len(grid)
                prof['%s|%g' % (a, dx)] = {
                    'log10_D': grid.tolist(),
                    'rmse_psi': [float(v[0]) for v in vals[i:i + n]],
                    'rmse_normalised': [float(v[1]) for v in vals[i:i + n]]}
                i += n
        save_ckpt('uniform_grid_%s.json' % tag, prof)
        log('uniform grid: %d solves in %.0f s' % (len(tasks), time.time() - t0))

    # --- cold multi-restart Nelder-Mead, both norms -------------------------
    restarts = load_ckpt('uniform_restarts_%s.json' % tag)
    if restarts is None:
        opts = {'maxiter': int(cfg['search']['uniform']['maxiter']),
                'xatol': float(cfg['search']['uniform']['xatol']),
                'fatol': float(cfg['search']['uniform']['fatol']),
                'disp': False}
        tasks = [(a, dx, x0, ni, lo, hi, opts)
                 for a in arms for dx in dxs
                 for x0 in fam['restart_starts_log10_D'] for ni in (0, 1)]
        tasks.sort(key=lambda t: t[1])          # fine meshes first: longest jobs
        t0 = time.time()
        restarts = list(pool.imap_unordered(_nm_uniform, tasks, chunksize=1))
        save_ckpt('uniform_restarts_%s.json' % tag, restarts)
        log('uniform restarts: %d fits, %d solves, %.0f s'
            % (len(restarts), sum(r['nfev'] for r in restarts), time.time() - t0))
    return prof, restarts


# ---------------------------------------------------------------------------
# stage: two_zone family
# ---------------------------------------------------------------------------

def _separated_starts(pts, vals, n_want, min_sep):
    """Best points that are mutually >= min_sep apart in the unit cube."""
    order = np.argsort(vals)
    chosen = []
    for i in order:
        if len(chosen) >= n_want:
            break
        if all(np.linalg.norm(pts[i] - pts[j]) >= min_sep for j in chosen):
            chosen.append(i)
    for i in order:                     # top up if separation was too strict
        if len(chosen) >= n_want:
            break
        if i not in chosen:
            chosen.append(i)
    return [int(i) for i in chosen[:n_want]]


def stage_two_zone(cfg, tag, pool):
    fam = cfg['families']['two_zone']
    sc = cfg['search']['two_zone']
    bounds = [tuple(b) for b in fam['bounds']]
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    dxs = list(fam['dx_ft_sweep'])
    arms = [a['name'] for a in cfg['arms']]

    # One LHS design, drawn once, used on EVERY mesh and in EVERY arm.
    unit = qmc.LatinHypercube(d=len(bounds),
                              seed=int(sc['lhs_seed'])).random(int(sc['n_lhs']))
    pts = lo + unit * (hi - lo)

    screen = load_ckpt('two_zone_screen_%s.json' % tag)
    if screen is None:
        tasks = [(a, dx, 'two_zone', p) for a in arms for dx in dxs for p in pts]
        tasks.sort(key=lambda t: -t[1])
        t0 = time.time()
        vals = list(pool.imap(_misfit, tasks, chunksize=4))
        screen = {}
        for (a, dx, _f, p), v in zip(tasks, vals):
            screen.setdefault('%s|%g' % (a, dx), []).append(float(v[0]))
        save_ckpt('two_zone_screen_%s.json' % tag, screen)
        log('two_zone LHS screen: %d solves in %.0f s'
            % (len(tasks), time.time() - t0))

    restarts = load_ckpt('two_zone_restarts_%s.json' % tag)
    if restarts is None:
        opts1 = {'maxiter': int(sc['nm_maxiter_pass1']),
                 'xatol': float(sc['xatol']), 'fatol': float(sc['fatol']),
                 'disp': False}
        opts2 = {'maxiter': int(sc['nm_maxiter_pass2']),
                 'xatol': float(sc['xatol']) / 10.0,
                 'fatol': float(sc['fatol']) / 10.0, 'disp': False}
        tasks = []
        starts = {}
        for a in arms:
            for dx in dxs:
                v = np.asarray(screen['%s|%g' % (a, dx)], float)
                sel = _separated_starts(unit, v, int(sc['n_restarts']),
                                        float(sc['min_start_separation_unitcube']))
                starts['%s|%g' % (a, dx)] = sel
                for i in sel:
                    tasks.append((a, dx, pts[i], bounds, opts1, opts2))
        save_ckpt('two_zone_starts_%s.json' % tag, starts)
        tasks.sort(key=lambda t: t[1])          # fine meshes first
        t0 = time.time()
        restarts = list(pool.imap_unordered(_nm_two_zone, tasks, chunksize=1))
        save_ckpt('two_zone_restarts_%s.json' % tag, restarts)
        log('two_zone restarts: %d fits, %d solves, %.0f s'
            % (len(restarts), sum(r['nfev'] for r in restarts), time.time() - t0))
    return screen, restarts


# ---------------------------------------------------------------------------
# stage: cross-mesh forward evaluation of frozen parameters
# ---------------------------------------------------------------------------

def best_of(restarts, arm, dx, key='rmse_psi', extra=None):
    sub = [r for r in restarts if r['arm'] == arm and abs(r['dx_ft'] - dx) < 1e-12
           and (extra is None or extra(r))]
    if not sub:
        return None
    return min(sub, key=lambda r: r[key])


def stage_crossmesh(cfg, tag, pool, uni_restarts, tz_restarts):
    out = load_ckpt('crossmesh_%s.json' % tag)
    if out is not None:
        return out
    rows = []
    tasks = []
    meta = []
    for fam_name, restarts, key in (
            ('uniform', uni_restarts, 'log10_D'),
            ('two_zone', tz_restarts, 'params')):
        dxs = list(cfg['families'][fam_name]['dx_ft_sweep'])
        for arm in [a['name'] for a in cfg['arms']]:
            for dx_fit in (max(dxs), min(dxs)):
                extra = (lambda r: r['norm'] == 'abs') if fam_name == 'uniform' \
                    else None
                b = best_of(restarts, arm, dx_fit, extra=extra)
                if b is None:
                    continue
                p = [b[key]] if fam_name == 'uniform' else b[key]
                for dx_run in dxs:
                    tasks.append((arm, dx_run, fam_name, p))
                    meta.append((fam_name, arm, dx_fit, dx_run, p,
                                 float(b['rmse_psi'])))
    vals = list(pool.imap(_misfit, tasks, chunksize=2))
    for (fam_name, arm, dx_fit, dx_run, p, rmse_fit), v in zip(meta, vals):
        rows.append(dict(family=fam_name, arm=arm, dx_fit_ft=dx_fit,
                         dx_run_ft=dx_run, params=list(map(float, p)),
                         rmse_psi=float(v[0]), rmse_normalised=float(v[1]),
                         rmse_at_own_mesh_psi=rmse_fit))
    save_ckpt('crossmesh_%s.json' % tag, rows)
    log('crossmesh: %d solves' % len(tasks))
    return rows


# ---------------------------------------------------------------------------
# stage: report -- tables, stability metrics, figures, manifests
# ---------------------------------------------------------------------------

REF_DX_SUMMARY = 0.05   # mesh used for mesh-INDEPENDENT derived scalars only


def _band_from_curve(x, y, factor=1.1):
    """Contiguous +factor band around the grid minimum, with censoring flags."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    i = int(np.argmin(y))
    thr = y[i] * factor
    lo_i = i
    while lo_i > 0 and y[lo_i - 1] <= thr:
        lo_i -= 1
    hi_i = i
    while hi_i < len(x) - 1 and y[hi_i + 1] <= thr:
        hi_i += 1
    if lo_i > 0:
        f = (thr - y[lo_i]) / (y[lo_i - 1] - y[lo_i])
        xlo = x[lo_i] + f * (x[lo_i - 1] - x[lo_i])
    else:
        xlo = x[0]
    if hi_i < len(x) - 1:
        f = (thr - y[hi_i]) / (y[hi_i + 1] - y[hi_i])
        xhi = x[hi_i] + f * (x[hi_i + 1] - x[hi_i])
    else:
        xhi = x[-1]
    return {'grid_argmin_log10_D': float(x[i]), 'grid_min': float(y[i]),
            'lo_log10_D': float(xlo), 'hi_log10_D': float(xhi),
            'lo_D_ft2_s': float(10.0 ** xlo), 'hi_D_ft2_s': float(10.0 ** xhi),
            'censored_low': bool(lo_i == 0), 'censored_high': bool(hi_i == len(x) - 1)}


def _profile_D(family, params, xs, x_src):
    """D(x) for a family on an arbitrary abscissa, independent of any fit mesh."""
    mesh = np.asarray(xs, float)
    i_src = int(np.argmin(np.abs(mesh - x_src)))
    return r1c.PROFILE_FAMILIES[family]['fn'](mesh, i_src,
                                              np.asarray(params, float))


def _path_resistance(family, params, x_lo, x_hi, x_src, step=0.05):
    """int dx / D(x) over [x_lo, x_hi], s/ft, on a fixed fine reference grid."""
    xs = np.arange(x_lo, x_hi + step / 2.0, step)
    d = _profile_D(family, params, xs, x_src)
    dmid = 0.5 * (d[:-1] + d[1:])
    return float(np.sum(np.diff(xs) / dmid))


def _pct(a, b):
    return float(100.0 * (a / b - 1.0)) if b else float('nan')


def _stable_dx(dxs_sorted_coarse_first, pct_vs_ref, tol):
    """Coarsest dx from which every mesh at least as fine stays within tol."""
    best = None
    for i, dx in enumerate(dxs_sorted_coarse_first):
        if all(abs(pct_vs_ref[j]) <= tol
               for j in range(i, len(dxs_sorted_coarse_first))):
            best = dx
            break
    return best


def _a1_forward_reference(path='output/rev2_20260901/A1/deliverable/'
                               'a1_mesh_independence_v3.csv'):
    """A1's FORWARD sweep at fixed D0 = 1150, reused rather than recomputed."""
    if not os.path.exists(path):
        return None
    import csv
    agg = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r['definition'] != 'physical_w':
                continue
            k = (float(r['ratio']), float(r['dx_ft']))
            agg.setdefault(k, []).append(float(r['rmse_psi']) ** 2)
    return {'%g|%g' % k: float(np.sqrt(np.mean(v))) for k, v in agg.items()}


def summarise_uniform(cfg, prof, restarts):
    fam = cfg['families']['uniform']
    dxs = list(fam['dx_ft_sweep'])
    rows = []
    for arm in [a['name'] for a in cfg['arms']]:
        for norm in ('abs', 'normalised'):
            for dx in dxs:
                rs = [r for r in restarts if r['arm'] == arm
                      and abs(r['dx_ft'] - dx) < 1e-12 and r['norm'] == norm]
                best = min(rs, key=lambda r: r['objective'])
                obj = np.array([r['objective'] for r in rs], float)
                Ds = np.array([r['D_ft2_s'] for r in rs], float)
                conv = Ds[obj <= best['objective'] * (1.0 + 1e-6)]
                g = prof['%s|%g' % (arm, dx)]
                band = _band_from_curve(
                    g['log10_D'],
                    g['rmse_psi'] if norm == 'abs' else g['rmse_normalised'])
                rows.append(dict(
                    arm=arm, norm=norm, dx_ft=dx,
                    D_ft2_s=best['D_ft2_s'], log10_D=best['log10_D'],
                    rmse_psi=best['rmse_psi'],
                    rmse_normalised=best['rmse_normalised'],
                    objective=best['objective'],
                    n_restarts=len(rs),
                    n_restarts_converged=int(conv.size),
                    restart_D_min=float(Ds.min()), restart_D_max=float(Ds.max()),
                    restart_spread_pct=_pct(float(Ds.max()), float(Ds.min())),
                    converged_spread_pct=(_pct(float(conv.max()), float(conv.min()))
                                          if conv.size else float('nan')),
                    nfev_total=int(sum(r['nfev'] for r in rs)),
                    at_bound=bool(best['at_bound']),
                    band10_lo_D=band['lo_D_ft2_s'], band10_hi_D=band['hi_D_ft2_s'],
                    band10_censored=bool(band['censored_low']
                                         or band['censored_high'])))
    return rows


def summarise_two_zone(cfg, restarts):
    fam = cfg['families']['two_zone']
    dxs = list(fam['dx_ft_sweep'])
    names = fam['param_names']
    rows = []
    for arm in [a['name'] for a in cfg['arms']]:
        for dx in dxs:
            rs = [r for r in restarts if r['arm'] == arm
                  and abs(r['dx_ft'] - dx) < 1e-12]
            best = min(rs, key=lambda r: r['rmse_psi'])
            v = np.array([r['rmse_psi'] for r in rs], float)
            P = np.array([r['params'] for r in rs], float)
            conv = P[v <= best['rmse_psi'] * 1.001]        # within 0.1 % of best
            row = dict(arm=arm, dx_ft=dx, rmse_psi=best['rmse_psi'],
                       rmse_normalised=best['rmse_normalised'],
                       n_restarts=len(rs),
                       n_restarts_within_0p1pct=int(conv.shape[0]),
                       restart_rmse_min=float(v.min()),
                       restart_rmse_max=float(v.max()),
                       nfev_total=int(sum(r['nfev'] for r in rs)),
                       wall_s=float(sum(r['wall_s'] for r in rs)),
                       any_at_bound=bool(any(best['at_bound'])))
            for j, nm in enumerate(names):
                row[nm] = float(best['params'][j])
                row[nm.replace('log10_', '') + '_value'] = float(10.0 ** best['params'][j])
                row[nm + '_restart_min'] = float(P[:, j].min())
                row[nm + '_restart_max'] = float(P[:, j].max())
                row[nm + '_conv_min'] = float(conv[:, j].min()) if conv.size else float('nan')
                row[nm + '_conv_max'] = float(conv[:, j].max()) if conv.size else float('nan')
            rows.append(row)
    return rows


def derived_scalars(cfg, uni_rows, tz_rows, geom_rows):
    """Mesh-independent summaries: total series resistance source -> g7."""
    x_src, x_g7 = 16645.0, 15075.0
    L = x_src - x_g7
    exc = {}
    for g in geom_rows:
        exc[(g['arm'], g['dx_ft'])] = float(g['excess_resistance_s_per_ft'] or 0.0)
    out = []
    for r in uni_rows:
        R = L / r['D_ft2_s'] + exc.get((r['arm'], r['dx_ft']), 0.0)
        out.append(dict(family='uniform', arm=r['arm'], norm=r['norm'],
                        dx_ft=r['dx_ft'], R_total_s_per_ft=R, D_eq_ft2_s=L / R,
                        rmse_psi=r['rmse_psi']))
    for r in tz_rows:
        p = [r[n] for n in cfg['families']['two_zone']['param_names']]
        R = _path_resistance('two_zone', p, x_g7, x_src, x_src) \
            + exc.get((r['arm'], r['dx_ft']), 0.0)
        out.append(dict(family='two_zone', arm=r['arm'], norm='abs',
                        dx_ft=r['dx_ft'], R_total_s_per_ft=R, D_eq_ft2_s=L / R,
                        rmse_psi=r['rmse_psi']))
    return out


def stability_table(cfg, uni_rows, tz_rows, deriv):
    tol = float(cfg['criteria']['stability_criterion']['tolerance_percent'])
    out = []

    def add(family, arm, norm, quantity, dxs, vals):
        ref = vals[-1]                      # finest mesh
        pct = [_pct(v, ref) for v in vals]
        out.append(dict(family=family, arm=arm, norm=norm, quantity=quantity,
                        dx_ft=list(dxs), value=list(map(float, vals)),
                        pct_vs_finest=pct,
                        coarse_to_fine_pct=pct[0],
                        max_abs_pct=float(np.max(np.abs(pct))),
                        stable_from_dx_ft=_stable_dx(dxs, pct, tol),
                        tolerance_pct=tol))

    for arm in [a['name'] for a in cfg['arms']]:
        for norm in ('abs', 'normalised'):
            sub = sorted([r for r in uni_rows if r['arm'] == arm
                          and r['norm'] == norm], key=lambda r: -r['dx_ft'])
            add('uniform', arm, norm, 'D_ft2_s', [r['dx_ft'] for r in sub],
                [r['D_ft2_s'] for r in sub])
            add('uniform', arm, norm, 'rmse_psi', [r['dx_ft'] for r in sub],
                [r['rmse_psi'] for r in sub])
        sub = sorted([r for r in tz_rows if r['arm'] == arm],
                     key=lambda r: -r['dx_ft'])
        if sub:
            for nm in cfg['families']['two_zone']['param_names']:
                add('two_zone', arm, 'abs', nm.replace('log10_', '') + '_ft_or_ft2s',
                    [r['dx_ft'] for r in sub], [10.0 ** r[nm] for r in sub])
            add('two_zone', arm, 'abs', 'rmse_psi', [r['dx_ft'] for r in sub],
                [r['rmse_psi'] for r in sub])
        for fam in ('uniform', 'two_zone'):
            sub = sorted([d for d in deriv if d['family'] == fam
                          and d['arm'] == arm and d['norm'] == 'abs'],
                         key=lambda r: -r['dx_ft'])
            if sub:
                add(fam, arm, 'abs', 'D_eq_ft2_s', [r['dx_ft'] for r in sub],
                    [r['D_eq_ft2_s'] for r in sub])
    return out


def write_csv(path, rows, fields=None):
    import csv
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

ARM_COLOR = {'control': '#555555', 'w1_r1em2': '#1b6ca8',
             'w0p25_r1em2': '#c1272d'}
ARM_LABEL = {'control': 'control: no barrier',
             'w1_r1em2': r'$w$ = 1.0 ft, ratio = $10^{-2}$  ($2w/\Delta x$ = 2–40)',
             'w0p25_r1em2': r'$w$ = 0.25 ft, ratio = $10^{-2}$  ($2w/\Delta x$ = 0.5–10)'}


def fig_calibrated(cfg, uni_rows, tz_rows, deriv, path, dpi):
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.0))
    tol = float(cfg['criteria']['stability_criterion']['tolerance_percent'])

    for arm in ARM_COLOR:
        for norm, a in (('abs', ax[0, 0]), ('normalised', ax[0, 1])):
            sub = sorted([r for r in uni_rows if r['arm'] == arm
                          and r['norm'] == norm], key=lambda r: r['dx_ft'])
            if not sub:
                continue
            dx = [r['dx_ft'] for r in sub]
            D = [r['D_ft2_s'] for r in sub]
            lo = [r['band10_lo_D'] for r in sub]
            hi = [r['band10_hi_D'] for r in sub]
            a.fill_between(dx, lo, hi, color=ARM_COLOR[arm], alpha=0.12, lw=0)
            a.plot(dx, D, 'o-', color=ARM_COLOR[arm], ms=5,
                   label=ARM_LABEL[arm])
            a.plot(dx, [r['restart_D_min'] for r in sub], '_',
                   color=ARM_COLOR[arm], ms=9)
            a.plot(dx, [r['restart_D_max'] for r in sub], '_',
                   color=ARM_COLOR[arm], ms=9)
    for a, ttl in ((ax[0, 0], 'absolute norm'), (ax[0, 1], 'amplitude-normalised norm')):
        a.set_xscale('log'), a.set_yscale('log')
        a.set_xlabel(r'$\Delta x$  (ft)'), a.set_ylabel(r'calibrated $D$  (ft$^2$/s)')
        a.set_title('uniform family (k = 1), %s\nshaded: +10%% misfit band; '
                    'ticks: multi-restart envelope' % ttl, fontsize=10)
        a.grid(alpha=0.3, which='both'), a.legend(fontsize=7.5, loc='best')

    a = ax[1, 0]
    for arm in ARM_COLOR:
        sub = sorted([r for r in uni_rows if r['arm'] == arm and r['norm'] == 'abs'],
                     key=lambda r: r['dx_ft'])
        ref = sub[0]['D_ft2_s']
        a.plot([r['dx_ft'] for r in sub], [_pct(r['D_ft2_s'], ref) for r in sub],
               'o-', color=ARM_COLOR[arm], ms=5, label='uniform ' + arm)
        sub = sorted([r for r in tz_rows if r['arm'] == arm], key=lambda r: r['dx_ft'])
        if sub:
            ref = 10.0 ** sub[0]['log10_D_far']
            a.plot([r['dx_ft'] for r in sub],
                   [_pct(10.0 ** r['log10_D_far'], ref) for r in sub], 's--',
                   color=ARM_COLOR[arm], ms=5, mfc='none',
                   label=r'two_zone $D_{far}$ ' + arm)
    a.axhspan(-tol, tol, color='0.75', alpha=0.45, lw=0,
              label=r'$\pm$%g%% (what the grid supports)' % tol)
    a.axhline(0, color='k', lw=0.6)
    a.set_xscale('log')
    a.set_xlabel(r'$\Delta x$  (ft)')
    a.set_ylabel('change vs the FINEST mesh  (%)')
    a.set_title('Does the calibrated answer move with the mesh?', fontsize=10)
    a.grid(alpha=0.3), a.legend(fontsize=7, loc='best')

    a = ax[1, 1]
    for arm in ARM_COLOR:
        sub = sorted([r for r in uni_rows if r['arm'] == arm and r['norm'] == 'abs'],
                     key=lambda r: r['dx_ft'])
        a.plot([r['dx_ft'] for r in sub], [r['rmse_psi'] for r in sub], 'o-',
               color=ARM_COLOR[arm], ms=5, label='uniform ' + arm)
        sub = sorted([r for r in tz_rows if r['arm'] == arm], key=lambda r: r['dx_ft'])
        if sub:
            a.plot([r['dx_ft'] for r in sub], [r['rmse_psi'] for r in sub], 's--',
                   color=ARM_COLOR[arm], ms=5, mfc='none', label='two_zone ' + arm)
    a.set_xscale('log'), a.set_yscale('log')
    a.set_xlabel(r'$\Delta x$  (ft)')
    a.set_ylabel('gauge-mean RMSE at the optimum  (psi)')
    a.set_title('misfit at the re-fitted optimum', fontsize=10)
    a.grid(alpha=0.3, which='both'), a.legend(fontsize=7, loc='best')

    fig.suptitle('B3 — grid independence under a FULL RE-FIT: '
                 r'$w$ and ratio frozen, $D$ re-fitted cold at every $\Delta x$',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def fig_profiles(cfg, prof, uni_rows, path, dpi):
    arms = [a['name'] for a in cfg['arms']]
    dxs = list(cfg['families']['uniform']['dx_ft_sweep'])
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(2, len(arms), figsize=(4.4 * len(arms), 7.4),
                           sharex=True)
    for j, arm in enumerate(arms):
        for i, norm in enumerate(('rmse_psi', 'rmse_normalised')):
            a = ax[i, j]
            for m, dx in enumerate(dxs):
                g = prof['%s|%g' % (arm, dx)]
                a.plot(10.0 ** np.asarray(g['log10_D']), g[norm],
                       color=cmap(m / max(len(dxs) - 1, 1)), lw=1.4,
                       label=r'$\Delta x$ = %g ft' % dx)
            sub = [r for r in uni_rows if r['arm'] == arm
                   and r['norm'] == ('abs' if i == 0 else 'normalised')]
            for r in sub:
                a.axvline(r['D_ft2_s'], color='r', lw=0.5, alpha=0.6)
            a.set_xscale('log'), a.set_yscale('log')
            a.grid(alpha=0.3, which='both')
            if i == 0:
                a.set_title(ARM_LABEL[arm], fontsize=8.5)
            if i == 1:
                a.set_xlabel(r'uniform $D$  (ft$^2$/s)')
            if j == 0:
                a.set_ylabel('gauge-mean RMSE (psi)' if i == 0
                             else 'gauge-mean normalised RMSE')
            if i == 0 and j == 0:
                a.legend(fontsize=6.5, loc='best')
    fig.suptitle('B3 — the misfit profile of the uniform family on every mesh '
                 '(red: the located optima)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def fig_two_zone(cfg, tz_rows, deriv, path, dpi):
    names = cfg['families']['two_zone']['param_names']
    tol = float(cfg['criteria']['stability_criterion']['tolerance_percent'])
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.6))
    axes = list(ax.ravel())
    for k, nm in enumerate(names):
        a = axes[k]
        for arm in ARM_COLOR:
            sub = sorted([r for r in tz_rows if r['arm'] == arm],
                         key=lambda r: r['dx_ft'])
            if not sub:
                continue
            dx = [r['dx_ft'] for r in sub]
            a.plot(dx, [10.0 ** r[nm] for r in sub], 'o-', ms=5,
                   color=ARM_COLOR[arm], label=arm)
            a.fill_between(dx, [10.0 ** r[nm + '_conv_min'] for r in sub],
                           [10.0 ** r[nm + '_conv_max'] for r in sub],
                           color=ARM_COLOR[arm], alpha=0.15, lw=0)
        a.set_xscale('log'), a.set_yscale('log')
        a.set_xlabel(r'$\Delta x$  (ft)')
        a.set_ylabel(nm.replace('log10_', '') +
                     (' (ft$^2$/s)' if '_D' in nm else ' (ft)'))
        a.grid(alpha=0.3, which='both')
        if k == 0:
            a.legend(fontsize=7)
    a = axes[4]
    for arm in ARM_COLOR:
        sub = sorted([d for d in deriv if d['family'] == 'two_zone'
                      and d['arm'] == arm], key=lambda r: r['dx_ft'])
        if sub:
            a.plot([r['dx_ft'] for r in sub], [r['D_eq_ft2_s'] for r in sub],
                   'o-', ms=5, color=ARM_COLOR[arm], label='two_zone ' + arm)
        sub = sorted([d for d in deriv if d['family'] == 'uniform'
                      and d['arm'] == arm and d['norm'] == 'abs'],
                     key=lambda r: r['dx_ft'])
        if sub:
            a.plot([r['dx_ft'] for r in sub], [r['D_eq_ft2_s'] for r in sub],
                   's--', ms=5, mfc='none', color=ARM_COLOR[arm],
                   label='uniform ' + arm)
    a.set_xscale('log'), a.set_xlabel(r'$\Delta x$  (ft)')
    a.set_ylabel(r'$D_{eq}$ source$\to$g7  (ft$^2$/s)')
    a.set_title('path-equivalent D (profile + barrier resistance)', fontsize=9)
    a.grid(alpha=0.3), a.legend(fontsize=6.5)

    a = axes[5]
    for arm in ARM_COLOR:
        sub = sorted([r for r in tz_rows if r['arm'] == arm],
                     key=lambda r: r['dx_ft'])
        if not sub:
            continue
        ref = sub[0]['rmse_psi']
        a.plot([r['dx_ft'] for r in sub],
               [_pct(r['rmse_psi'], ref) for r in sub], 'o-', ms=5,
               color=ARM_COLOR[arm], label=arm)
    a.axhspan(-tol, tol, color='0.75', alpha=0.45, lw=0)
    a.axhline(0, color='k', lw=0.6)
    a.set_xscale('log'), a.set_xlabel(r'$\Delta x$  (ft)')
    a.set_ylabel('RMSE change vs finest mesh (%)')
    a.set_title('two_zone misfit stability', fontsize=9)
    a.grid(alpha=0.3), a.legend(fontsize=7)

    fig.suptitle('B3 — the four-parameter two_zone family re-fitted cold on '
                 'every mesh (shaded: restarts within 0.1% of the best)',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# manifests
# ---------------------------------------------------------------------------

def _manifest_inputs(cfg, tgt_gauges):
    inputs = [(cfg['data']['gauge_md_npz'], 'geometry', 'gauge_md_npz'),
              (cfg['data']['frac_hit_stage1_npz'], 'geometry', 'frac_hit_stage1'),
              (cfg['data']['frac_hit_barrier_npz'], 'geometry', 'frac_hit_stage2'),
              ('output/rev2_20260901/A1/deliverable/a1_mesh_independence_v3.csv',
               'prior_run_output', 'a1_forward_sweep'),
              ('output/rev2_20260901/A1/deliverable/README.md',
               'prior_run_output', 'a1_deliverable_readme')]
    inputs += [(cfg['data']['gauge_series_template'].format(n=n), 'gauge_series',
                'gauge%d' % n) for n in [1] + list(tgt_gauges)]
    return [i for i in inputs if os.path.exists(i[0])]


def _write_manifests(cfg, cfg_path, tag, t_start, geom_rows, uni_rows, tz_rows,
                     summary, outputs_decl):
    s = cfg['solver']
    b = cfg['barrier']
    fh = rd.load_frac_hits(2)
    dxs = sorted(set(cfg['families']['uniform']['dx_ft_sweep']
                     + cfg['families']['two_zone']['dx_ft_sweep']), reverse=True)
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
    inputs = _manifest_inputs(cfg, cfg['targets']['gauges'])

    written = []
    for dx in dxs:
        S = rd.setup_r1(dx_ft=dx,
                        pad_low_ft=float(cfg['mesh']['domain_pad_low_md_ft']),
                        pad_high_ft=float(cfg['mesh']['domain_pad_high_md_ft']))
        mesh = S['mesh']
        taxis_ref = np.arange(0.0, float(src.t_total_s) + float(s['dt_s']),
                              float(s['dt_s']))
        for arm in cfg['arms']:
            fams = [f for f in ('uniform', 'two_zone')
                    if dx in cfg['families'][f]['dx_ft_sweep']]
            if not fams:
                continue
            u = [r for r in uni_rows if r['arm'] == arm['name']
                 and r['norm'] == 'abs' and abs(r['dx_ft'] - dx) < 1e-12]
            d0 = float(u[0]['D_ft2_s']) if u else 1150.0
            if arm['barrier']:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', core.BarrierWidthWarning)
                    _, rep = core.build_barrier_profile(
                        mesh.x, np.full(mesh.nx, d0), fh,
                        float(arm['w_half_width_ft']), float(arm['ratio']),
                        ratio_reference=b['ratio_reference'], combine=b['combine'],
                        on_empty=b['on_empty'], on_outside=b['on_outside'],
                        return_report=True)
                recs = []
                for j, br in enumerate(rep['barriers']):
                    mask = np.zeros(mesh.nx, dtype=bool)
                    mask[br['i0']:br['i1'] + 1] = True
                    recs.append(rm.barrier_record(
                        mesh.x, mask,
                        label='%s|hit%d' % (arm['name'], j),
                        centre_md_ft=float(br['md_ft']),
                        w_requested_ft=float(arm['w_half_width_ft']),
                        ratio=float(arm['ratio']), d_baseline=d0,
                        report=(rep if j == 0 else None)))
            else:
                recs = rm.NONE_DECLARED

            src_grp = rm.source_protocol(
                application='dirichlet_node',
                solver_class='rev2_core.solve_forward (tridiagonal solve_banded); '
                             'bitwise identical to r1_calibration_core.solve_forward '
                             'at theta=1 / harmonic / lambda=0 (A4 self-test T1a)',
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
                boundary_conditions={'lbc': s['lbc'], 'rbc': s['rbc']})

            fitted = {f: [r for r in (uni_rows if f == 'uniform' else tz_rows)
                          if r['arm'] == arm['name']
                          and abs(r['dx_ft'] - dx) < 1e-12] for f in fams}
            num_grp = rm.numerics(
                time=rm.time_record(taxis_ref, mode='fixed', theta=float(s['theta']),
                                    t_total_requested_s=float(src.t_total_s),
                                    dt_requested_s=float(s['dt_s']),
                                    source_time_level=s['source_time_level'],
                                    theta_startup_steps=int(s['theta_startup_steps']),
                                    label='dx=%g ft; every solve in this study '
                                          'shares this axis' % dx),
                mesh=rm.mesh_record(mesh.x, dx_requested_ft=dx,
                                    window_md_ft=(cfg['window']['md_min_ft'],
                                                  cfg['window']['md_max_ft']),
                                    pad_low_ft=float(cfg['mesh']['domain_pad_low_md_ft']),
                                    pad_high_ft=float(cfg['mesh']['domain_pad_high_md_ft']),
                                    refinement=cfg['mesh']['refinement']),
                interface_avg=s['interface_avg'],
                boundary={'lbc': s['lbc'], 'rbc': s['rbc'], 'pml_thickness': 0.0,
                          'sigma_max': 0.0},
                diffusivity={
                    'profile_families_fitted': fams,
                    'bounds': {f: cfg['families'][f]['bounds'] for f in fams},
                    'param_names': {f: cfg['families'][f]['param_names']
                                    for f in fams},
                    'fitted_optimum': fitted,
                    'barrier_applied_on_top_of_fitted_profile': bool(arm['barrier']),
                    'ratio_reference': b['ratio_reference'],
                    'D0_for_barrier_records_ft2_s': d0,
                    'note': 'D(x) is RE-FITTED on this mesh; the barrier records '
                            'below are built at the fitted absolute-norm uniform '
                            'optimum D0 so their D_barrier is a real number from '
                            'this run. Every barrier property that matters here '
                            '(realised width, excess resistance, node set) is '
                            'independent of D0.'},
                barriers=recs, leakage=rm.NONE_DECLARED,
                kernel={'name': 'rev2_core.solve_forward', 'banded': True,
                        'theta': float(s['theta']),
                        'equivalence_reference':
                            'output/rev2_20260901/A4/selftest_output.txt T1a: '
                            'bitwise identical (max|diff| = 0.000e+00 psi) to '
                            'r1_calibration_core.solve_forward, itself proven '
                            'bit-equivalent to fibeRIS in the R1 manifest'},
                rng={'engine': 'scipy.stats.qmc.LatinHypercube',
                     'seeds': {'two_zone_lhs': int(cfg['search']['two_zone']['lhs_seed'])},
                     'note': 'The LHS seed does NOT depend on dx or on the arm, so '
                             'every mesh is searched from the identical candidate '
                             'start set. The uniform family uses six fixed cold '
                             'starts and no random component.'},
                parallel={'processes': int(cfg['run']['processes']),
                          'backend': 'multiprocessing.Pool'})

            p = cfg['outputs']['manifest_per_run_template'].format(
                arm=arm['name'], tag=dx_tag(dx))
            rm.write_manifest(
                p, study_id=cfg['study_id'] + '__%s_dx%s' % (arm['name'], dx_tag(dx)),
                task_id=cfg['task_id'], config=cfg, config_path=cfg_path,
                inputs=inputs, source=src_grp, numerics=num_grp,
                outputs=outputs_decl, results={
                    'arm': arm['name'], 'dx_ft': dx, 'nx': mesh.nx,
                    'families_fitted': fams,
                    'barrier_realisation': [g for g in geom_rows
                                            if g['arm'] == arm['name']
                                            and abs(g['dx_ft'] - dx) < 1e-12],
                    'fits': fitted},
                notes=['One manifest per (arm, mesh). Every forward solve in B3 '
                       'belongs to exactly one of them (house rule 3).',
                       'dt is held at 1 s on every mesh: B3 is a SPATIAL '
                       'refinement study and the time discretisation must not '
                       'move with it.',
                       'Products are shared across the whole sweep, so the same '
                       'output inventory is declared in every manifest; the '
                       'roll-up is manifest.json in the same directory.'],
                started_utc=t_start, run_label='B3-%s' % tag,
                require_modules=('rev2_core', 'rev2_data', 'rev2_manifest'),
                extra_code_files=(os.path.abspath(__file__),),
                allow_undeclared_outputs=True)
            written.append(p)

    # roll-up
    dx_fine = min(dxs)
    S = rd.setup_r1(dx_ft=dx_fine,
                    pad_low_ft=float(cfg['mesh']['domain_pad_low_md_ft']),
                    pad_high_ft=float(cfg['mesh']['domain_pad_high_md_ft']))
    mesh = S['mesh']
    taxis_ref = np.arange(0.0, float(src.t_total_s) + float(s['dt_s']),
                          float(s['dt_s']))
    rm.write_manifest(
        cfg['outputs']['manifest_rollup'], study_id=cfg['study_id'],
        task_id=cfg['task_id'], config=cfg, config_path=cfg_path,
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
                                label='shared by every mesh in the sweep'),
            mesh=rm.mesh_record(mesh.x, dx_requested_ft=dx_fine,
                                window_md_ft=(cfg['window']['md_min_ft'],
                                              cfg['window']['md_max_ft']),
                                pad_low_ft=float(cfg['mesh']['domain_pad_low_md_ft']),
                                pad_high_ft=float(cfg['mesh']['domain_pad_high_md_ft']),
                                refinement=cfg['mesh']['refinement']),
            interface_avg=s['interface_avg'],
            boundary={'lbc': s['lbc'], 'rbc': s['rbc'], 'pml_thickness': 0.0,
                      'sigma_max': 0.0},
            diffusivity={'profile_families_fitted': ['uniform', 'two_zone'],
                         'bounds': {f: cfg['families'][f]['bounds']
                                    for f in ('uniform', 'two_zone')},
                         'note': 'ROLL-UP. The per-(arm, mesh) manifests carry the '
                                 'barrier records and the fitted optima.'},
            barriers=rm.NONE_DECLARED, leakage=rm.NONE_DECLARED,
            kernel={'name': 'rev2_core.solve_forward', 'banded': True,
                    'theta': float(s['theta']),
                    'equivalence_reference': 'A4 self-test T1a'},
            rng={'engine': 'scipy.stats.qmc.LatinHypercube',
                 'seeds': {'two_zone_lhs': int(cfg['search']['two_zone']['lhs_seed'])},
                 'note': 'seed independent of dx and arm'},
            parallel={'processes': int(cfg['run']['processes']),
                      'backend': 'multiprocessing.Pool'}),
        outputs=outputs_decl, results=summary,
        notes=['ROLL-UP manifest for the whole B3 study. `numerics.mesh` records '
               'the FINEST mesh; every (arm, mesh) pair has its own manifest '
               'manifest_<arm>_dx<tag>.json in this directory, each with its own '
               'mesh hash and barrier records.',
               'The A1 deliverable is an INPUT here, not re-run: its forward sweep '
               'at fixed D0 = 1150 is quoted alongside the re-fitted results.',
               'barriers=NONE_DECLARED in this roll-up because the three arms have '
               'different barriers; they are declared per arm.'],
        started_utc=t_start, run_label='B3-%s' % tag,
        require_modules=('rev2_core', 'rev2_data', 'rev2_manifest'),
        extra_code_files=(os.path.abspath(__file__),),
        allow_undeclared_outputs=True)
    written.append(cfg['outputs']['manifest_rollup'])
    return written


def main_report(cfg, cfg_path, tag, t_start):
    o = cfg['outputs']
    dpi = int(o['figure_dpi'])
    os.makedirs(o['dir'], exist_ok=True)

    geom_rows = load_ckpt('geom_%s.json' % tag) or stage_geom(cfg, tag)
    prof = load_ckpt('uniform_grid_%s.json' % tag)
    uni_restarts = load_ckpt('uniform_restarts_%s.json' % tag)
    tz_restarts = load_ckpt('two_zone_restarts_%s.json' % tag)
    tz_screen = load_ckpt('two_zone_screen_%s.json' % tag)
    cross = load_ckpt('crossmesh_%s.json' % tag) or []
    if prof is None or uni_restarts is None:
        raise SystemExit('report: uniform checkpoints missing; run --stage uniform')
    if tz_restarts is None:
        raise SystemExit('report: two_zone checkpoints missing; run --stage two_zone')

    uni_rows = summarise_uniform(cfg, prof, uni_restarts)
    tz_rows = summarise_two_zone(cfg, tz_restarts)
    deriv = derived_scalars(cfg, uni_rows, tz_rows, geom_rows)
    stab = stability_table(cfg, uni_rows, tz_rows, deriv)
    a1fwd = _a1_forward_reference()

    write_csv(o['barrier_csv'], geom_rows)
    write_csv(o['uniform_csv'], uni_rows)
    write_csv(o['two_zone_csv'], tz_rows)
    write_csv(o['crossmesh_csv'],
              [dict(r, params='|'.join('%.6f' % v for v in r['params']))
               for r in cross])
    with open(o['uniform_restarts_json'], 'w') as fh:
        json.dump({'restarts': uni_restarts, 'derived': deriv,
                   'stability': stab}, fh, indent=1)
    with open(o['two_zone_restarts_json'], 'w') as fh:
        json.dump({'restarts': tz_restarts,
                   'lhs_screen_best': {k: float(np.min(v))
                                       for k, v in (tz_screen or {}).items()},
                   'starts': load_ckpt('two_zone_starts_%s.json' % tag)},
                  fh, indent=1)
    grid = np.asarray(prof[list(prof)[0]]['log10_D'], float)
    np.savez_compressed(
        o['uniform_grid_npz'], log10_D=grid,
        **{('%s__%s' % (k.replace('|', '_dx'), n)).replace('.', 'p'):
           np.asarray(v[n], float)
           for k, v in prof.items() for n in ('rmse_psi', 'rmse_normalised')})

    fig_calibrated(cfg, uni_rows, tz_rows, deriv, o['fig_calibrated'], dpi)
    fig_profiles(cfg, prof, uni_rows, o['fig_profiles'], dpi)
    fig_two_zone(cfg, tz_rows, deriv, o['fig_two_zone'], dpi)

    tol = float(cfg['criteria']['stability_criterion']['tolerance_percent'])
    headline = {}
    for st in stab:
        if st['quantity'] in ('D_ft2_s', 'D_far_ft_or_ft2s', 'D_eq_ft2_s'):
            headline['%s|%s|%s|%s' % (st['family'], st['arm'], st['norm'],
                                      st['quantity'])] = {
                'coarse_to_fine_pct': st['coarse_to_fine_pct'],
                'max_abs_pct': st['max_abs_pct'],
                'stable_from_dx_ft': st['stable_from_dx_ft'],
                'dx_ft': st['dx_ft'], 'value': st['value']}
    summary = {
        'task': 'B3', 'tag': tag,
        'tolerance_pct': tol,
        'n_solves_total': int(sum(r['nfev'] for r in uni_restarts)
                              + sum(r['nfev'] for r in tz_restarts)
                              + len(grid) * len(cfg['families']['uniform']['dx_ft_sweep'])
                              * len(cfg['arms'])
                              + int(cfg['search']['two_zone']['n_lhs'])
                              * len(cfg['families']['two_zone']['dx_ft_sweep'])
                              * len(cfg['arms']) + len(cross)),
        'barrier_realisation': geom_rows,
        'uniform_fits': uni_rows, 'two_zone_fits': tz_rows,
        'derived_path_equivalent': deriv,
        'stability': stab, 'headline': headline,
        'cross_mesh_forward': cross,
        'a1_forward_reference_gauge_mean_rmse_psi_at_D0_1150': a1fwd,
    }
    with open(o['summary_json'], 'w') as fh:
        json.dump(summary, fh, indent=1)

    outputs_decl = [
        rm.output_decl(o['barrier_csv'], role='csv'),
        rm.output_decl(o['uniform_csv'], role='csv'),
        rm.output_decl(o['two_zone_csv'], role='csv'),
        rm.output_decl(o['crossmesh_csv'], role='csv'),
        rm.output_decl(o['uniform_grid_npz'], role='arrays_npz'),
        rm.output_decl(o['uniform_restarts_json'], role='json'),
        rm.output_decl(o['two_zone_restarts_json'], role='json'),
        rm.output_decl(o['summary_json'], role='json'),
        rm.output_decl(o['fig_calibrated'], role='figure_png', dpi=dpi),
        rm.output_decl(o['fig_profiles'], role='figure_png', dpi=dpi),
        rm.output_decl(o['fig_two_zone'], role='figure_png', dpi=dpi)]
    written = _write_manifests(cfg, cfg_path, tag, t_start, geom_rows, uni_rows,
                               tz_rows, summary, outputs_decl)
    log('report: %d manifests written' % len(written))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--tag', default='v1')
    ap.add_argument('--stage', default='all',
                    choices=('all', 'geom', 'uniform', 'two_zone', 'crossmesh',
                             'report'))
    args = ap.parse_args()
    cfg = load_config(args.config)
    nproc = int(cfg['run']['processes'])
    t_start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    T0 = time.time()

    if args.stage in ('all', 'geom'):
        stage_geom(cfg, args.tag)

    if args.stage in ('all', 'uniform', 'two_zone', 'crossmesh'):
        with mp.Pool(nproc, initializer=_init_worker, initargs=(cfg,)) as pool:
            uni = tz = None
            if args.stage in ('all', 'uniform'):
                uni = stage_uniform(cfg, args.tag, pool)
            if args.stage in ('all', 'two_zone'):
                tz = stage_two_zone(cfg, args.tag, pool)
            if args.stage in ('all', 'crossmesh'):
                ur = (uni[1] if uni else load_ckpt('uniform_restarts_%s.json'
                                                   % args.tag))
                tr = (tz[1] if tz else load_ckpt('two_zone_restarts_%s.json'
                                                 % args.tag))
                if ur and tr:
                    stage_crossmesh(cfg, args.tag, pool, ur, tr)
                else:
                    log('crossmesh SKIPPED: fit checkpoints not present yet')

    if args.stage in ('all', 'report'):
        main_report(cfg, args.config, args.tag, t_start)

    log('B3 stage %s done in %.0f s' % (args.stage, time.time() - T0))
