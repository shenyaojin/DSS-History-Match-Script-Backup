"""D2 v2 - remove the SOURCE gauge, under BOTH norms, with cold-start restart envelopes.

Reviewer 2's charge is circularity: gauge 1 (MD 16645) is imposed as a Dirichlet
boundary value, so it is fit trivially and is excluded from every misfit. This
study rebuilds the boundary condition WITHOUT gauge 1 under four substitutions,
recalibrates every model FROM A COLD START on an identical held-in gauge set
under the absolute AND the amplitude-normalised norm, and then rescores every
gauge - including gauge 1, which becomes a genuine blind target in three of the
substitutions.

Substitutions
  (a)  extrap_g1md     linear two-point extrapolation of gauges 2 and 3 to
                       MD 16645. BC VALUE changes, BC LOCATION does not.
  (a') extrap_frachit  the same rule extrapolated to the stage-1 frac-hit
                       centroid MD 16683.2, where the fluid actually enters.
                       Gauge 1 is then a blind target 38 ft below the boundary.
  (b)  src_g2          gauge 2 drives the Dirichlet node at its own MD 16384.
                       Gauge 1 is a blind target 261 ft on the injection side.
  (c)  src_g3          the same one gauge further out (MD 16122), which turns a
                       single number into a degradation trend.

Controls
  baseline      the published protocol, calibrated on {2..7}.
  base_frachit  the published series applied at the frac-hit centroid, i.e. the
                established 82.33 -> 81.24 / 11.87 -> 11.31 experiment, redone
                here so this study can say whether it agrees.
  baseline_far  the published protocol recalibrated on {4,5,6,7} only, so the
                substitutions differ from it in the boundary condition ALONE.

What v2 adds over v1 (output/rev2_20260901/D2/*_v1.*, kept, not deleted):
  * both norms everywhere - the absolute gauge-mean psi norm and the
    amplitude-normalised norm - because the D1 amendment showed the uniform
    range-of-applicability headline flips between them;
  * cold-start MULTI-RESTART envelopes instead of one number per fit (D1's
    blocker was a warm start from a fit that had seen the withheld gauge);
  * a domain-top rule that pads above the Dirichlet node whenever a target sits
    above it, because v1's own sweep shows the blind gauge-1 prediction is not
    converged at the well TD;
  * naive no-solver predictors of gauge 1, so "the model degrades" can be
    compared against "a straight line through two neighbours";
  * manifests written by rev2_manifest.write_manifest (house rule 3).

Run from the repository root:

    python3 scripts/manuscript_well_leakage/rev2/d2_source_loo_v2.py \
        --config configs/rev2/d2_source_loo_v2.json
"""

import argparse
import csv
import datetime
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', '..'))
_BC = os.path.join(REPO, 'scripts', 'manuscript_well_leakage', 'baseline_calibration')
_R2 = os.path.dirname(os.path.abspath(__file__))
for _p in (_BC, _R2, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import r1_calibration_core as core            # noqa: E402
from r1_run_calibration import (load_config, load_window_data,   # noqa: E402
                                pick_source_gauge)
import rev2_core as rc                        # noqa: E402
import rev2_manifest as rm                    # noqa: E402
import rev2_data as rd                        # noqa: E402

ALL_GAUGES = [1, 2, 3, 4, 5, 6, 7]
NORMS = ('abs', 'norm')
_G = {}
_LOG_FH = None


def log(m):
    line = f"[d2v2] {m}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + '\n')
        _LOG_FH.flush()


# --------------------------------------------------------------------------
# profiles
# --------------------------------------------------------------------------
def profile_uniform(mesh, anchor_md, p):
    return np.full(len(mesh), 10.0 ** p[0])


def profile_two_zone_anchored(mesh, anchor_md, p):
    """core.profile_two_zone with the anchor given as a physical MD.

    core's version measures distance from the SOURCE NODE, so moving the
    boundary condition silently moves the high-diffusivity zone with it. Passing
    the anchor explicitly lets the same shape stay pinned to the injection point
    in every variant, which is what isolates the boundary substitution.
    """
    d_n, d_f, sc, w = 10.0 ** p[0], 10.0 ** p[1], 10.0 ** p[2], 10.0 ** p[3]
    s = np.abs(np.asarray(mesh, dtype=float) - float(anchor_md))
    frac = 0.5 * (1.0 + np.tanh((s - sc) / max(w, 1e-6)))
    return d_n * (1.0 - frac) + d_f * frac


PROFILE_FN = {
    'uniform': profile_uniform,
    'two_zone_bc': profile_two_zone_anchored,
    'two_zone_frachit': profile_two_zone_anchored,
}


# --------------------------------------------------------------------------
# boundary-condition construction
# --------------------------------------------------------------------------
def extrapolate_linear(series, from_gauges, target_md):
    """Two-point linear-in-MD extrapolation of the delta-pressure series.

    With gauges a (nearer the target MD) and b (further),

        P(md) = P_a + (md - md_a)/(md_a - md_b) * (P_a - P_b)

    evaluated sample by sample on gauge a's time axis, gauge b interpolated onto
    it. This is the lowest-order rule that uses no information from the removed
    gauge; it is deliberately naive, and its error against gauge 1 is reported
    separately so the reader can tell "the extrapolation is wrong" from "the
    model is wrong".
    """
    ga, gb = from_gauges
    a, b = series[ga], series[gb]
    if abs(a['md_ft'] - target_md) > abs(b['md_ft'] - target_md):
        a, b = b, a
    taxis = a['taxis']
    pa = a['delta_psi']
    pb = np.interp(taxis, b['taxis'], b['delta_psi'])
    w = (target_md - a['md_ft']) / (a['md_ft'] - b['md_ft'])
    return taxis, pa + w * (pa - pb), {
        'near_gauge': int(a['gauge']), 'far_gauge': int(b['gauge']),
        'near_md_ft': float(a['md_ft']), 'far_md_ft': float(b['md_ft']),
        'target_md_ft': float(target_md), 'weight': float(w),
        'formula': ('P(md) = P_near + w*(P_near - P_far), '
                    'w = (md - md_near)/(md_near - md_far)'),
    }


def variant_top_md(cfg, src_md, target_mds):
    """Domain top: pad above the Dirichlet node only when a target sits above it.

    A Dirichlet row decouples the sub-domain above it from the one below (A5
    verified ~1e-10 psi leak), so a variant with every target below the node
    needs no high-end pad and reproduces the published domain exactly. Once a
    blind target sits ABOVE the node it is inside a closed box whose far wall is
    the domain top, and B2's lesson applies: v1 measured the blind gauge-1 RMSE
    moving 134 -> 225 psi as the top went 16750 -> 18000.
    """
    m = cfg['mesh']
    top = float(cfg['window']['md_max_ft'])
    if float(src_md) > top:
        top = float(src_md) + float(m['domain_pad_above_source_ft'])
    above = [md for md in target_mds if md > src_md + 1e-9]
    if above:
        top = max(top, max(above) + float(m['domain_pad_above_highest_target_ft']))
    return float(top)


def build_variant(cfg, series, fh_centroid, vspec, top_md=None):
    """Mesh, Dirichlet node, driving series, targets and gauge roles."""
    dx = float(cfg['mesh']['dx_ft'])
    pad_lo = float(cfg['mesh']['domain_pad_low_md_ft'])
    win_lo = float(cfg['window']['md_min_ft'])

    bc = vspec['bc']
    driver_gauges, extrap_info, src_gauge = [], None, None
    if bc['kind'] == 'gauge':
        src_gauge = int(bc['gauge'])
        src_md = (float(fh_centroid) if bc.get('md_from') == 'frac_hit_centroid'
                  else float(bc['md_ft']))
        s_taxis = series[src_gauge]['taxis']
        s_data = series[src_gauge]['delta_psi']
        driver_gauges = [src_gauge]
        driving_paths = [cfg['data']['gauge_series_template'].format(n=src_gauge)]
    elif bc['kind'] == 'extrapolate':
        src_md = (float(fh_centroid) if bc.get('md_from') == 'frac_hit_centroid'
                  else float(bc['md_ft']))
        driver_gauges = [int(g) for g in bc['from_gauges']]
        s_taxis, s_data, extrap_info = extrapolate_linear(series, driver_gauges,
                                                          src_md)
        driving_paths = [cfg['data']['gauge_series_template'].format(n=g)
                         for g in driver_gauges]
    else:
        raise ValueError(bc['kind'])

    if top_md is None:
        top_md = variant_top_md(cfg, src_md,
                                [float(series[n]['md_ft']) for n in ALL_GAUGES])
    mesh = np.arange(win_lo - pad_lo, float(top_md) + dx / 2.0, dx)
    source_idx = int(np.argmin(np.abs(mesh - src_md)))
    node_md = float(mesh[source_idx])

    targets, roles = [], {}
    for n in ALL_GAUGES:
        md = float(series[n]['md_ft'])
        idx = int(np.argmin(np.abs(mesh - md)))
        targets.append({'gauge': n, 'md_ft': md,
                        'distance_ft': abs(md - src_md), 'idx': idx,
                        'taxis': series[n]['taxis'],
                        'data': series[n]['delta_psi']})
        if idx == source_idx:
            roles[n] = ('source_dirichlet' if bc['kind'] == 'gauge'
                        else 'imposed_extrapolation')
        elif n in driver_gauges:
            roles[n] = 'bc_constituent'
        elif n in vspec['calib_gauges']:
            roles[n] = 'calibration'
        else:
            roles[n] = 'blind'

    calib = [n for n in vspec['calib_gauges'] if roles[n] == 'calibration']
    if bc['kind'] == 'gauge' and bc.get('md_from') == 'frac_hit_centroid':
        # base_frachit deliberately scores the driving gauge as a target: the
        # node has moved off its MD, so it is a (very short-range) prediction.
        calib = [n for n in vspec['calib_gauges'] if roles[n] != 'source_dirichlet']
    if not calib:
        raise RuntimeError(f"{vspec['name']}: empty calibration set")

    obs_max = np.array([float(np.max(t['data'])) for t in targets])
    return {
        'name': vspec['name'], 'label': vspec['label'], 'spec': vspec,
        'models': list(vspec['models']),
        'mesh': mesh, 'source_idx': source_idx, 'src_md_ft': src_md,
        'node_md_ft': node_md, 'md_snap_residual_ft': float(node_md - src_md),
        'src_gauge': src_gauge, 'driver_gauges': driver_gauges,
        'driving_paths': driving_paths, 'extrap_info': extrap_info,
        's_taxis': np.asarray(s_taxis, float), 's_data': np.asarray(s_data, float),
        't_total': float(np.asarray(s_taxis, float)[-1]),
        'targets': targets, 'roles': roles, 'calib_gauges': calib,
        'calib_pos': [i for i, t in enumerate(targets) if t['gauge'] in calib],
        'blind_gauges': [n for n in ALL_GAUGES if roles[n] == 'blind'],
        'obs_max': obs_max, 'frac_hit_centroid': float(fh_centroid),
        'top_md_ft': float(mesh[-1]), 'dx_ft': dx, 'pad_low_ft': pad_lo,
    }


# --------------------------------------------------------------------------
# forward evaluation - one solve gives every gauge's MSE, hence BOTH norms
# --------------------------------------------------------------------------
def per_gauge_mse(V, prof, dt):
    if not np.all(np.isfinite(prof)) or np.any(prof <= 0):
        return np.full(len(V['targets']), np.inf)
    taxis, rec = rc.solve_forward(
        V['mesh'], prof, dt, V['t_total'], V['s_taxis'], V['s_data'],
        V['source_idx'], record_idx=[t['idx'] for t in V['targets']])
    if not np.all(np.isfinite(rec)):
        return np.full(len(V['targets']), np.inf)
    out = np.empty(len(V['targets']))
    for k, tgt in enumerate(V['targets']):
        r = np.interp(tgt['taxis'], taxis, rec[:, k]) - tgt['data']
        out[k] = float(np.mean(r ** 2))
    return out


def objectives_from_mse(V, mse):
    """(absolute gauge-mean RMSE in psi, amplitude-normalised RMSE)."""
    pos = V['calib_pos']
    m = np.asarray(mse, float)[pos]
    o_abs = float(np.sqrt(np.mean(m)))
    o_nrm = float(np.sqrt(np.mean(m / V['obs_max'][pos] ** 2)))
    return o_abs, o_nrm


def _init_worker(V, dt):
    _G['V'], _G['dt'] = V, dt


def _task(args):
    """One forward solve. args = (model, params, anchor, variant_name).

    The variant name is checked against the one this process was initialised
    with. `_G` is a process global, so a call made in the PARENT picks up
    whatever variant was last passed to `_init_worker` there - which silently
    scored one variant's parameters against another variant's mesh, boundary
    condition and calibration set until this guard was added. Never remove it.
    """
    model, params, anchor, vname = args
    V = _G['V']
    if V['name'] != vname:
        raise RuntimeError(
            f"_task called for variant {vname!r} in a process initialised for "
            f"{V['name']!r}: the result would be scored against the wrong mesh "
            f"and the wrong calibration set")
    prof = PROFILE_FN[model](V['mesh'], anchor, np.asarray(params, float))
    return per_gauge_mse(V, prof, _G['dt']).tolist()


def _scalar(V, mse, which):
    a, n = objectives_from_mse(V, mse)
    return a if which == 'abs' else n


def _polish(args):
    """One Nelder-Mead polish under one norm, run inside a worker.

    Nelder-Mead is serial, so farming the restarts out to the pool makes an
    R-restart polish cost roughly the wall time of ceil(2R/nproc) of them.
    """
    model, start, anchor, maxiter, which, vname = args
    V = _G['V']

    def f(q):
        return _scalar(V, _task((model, q, anchor, vname)), which)

    res = minimize(f, np.asarray(start, float), method='Nelder-Mead',
                   options={'maxiter': maxiter, 'xatol': 1e-3, 'fatol': 1e-4,
                            'disp': False})
    return float(res.fun), [float(x) for x in np.asarray(res.x, float)], int(res.nfev)


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
def fit_uniform(pool, V, mcfg, band_frac=0.10):
    """One log10-D grid; both norms read off the same forward solves."""
    c, r = mcfg['coarse'], mcfg['refine']
    grid = np.logspace(c['log10_min'], c['log10_max'], c['n'])
    mse = np.array(pool.map(
        _task, [('uniform', [np.log10(d)], None, V['name']) for d in grid],
        chunksize=2))
    obj = np.array([objectives_from_mse(V, m) for m in mse])          # (n, 2)

    out = {'model': 'uniform', 'k': 1, 'per_norm': {}}
    all_grid, all_mse = [grid], [mse]
    for j, which in enumerate(NORMS):
        i = int(np.argmin(obj[:, j]))
        lo = max(c['log10_min'], np.log10(grid[i]) - r['half_width_dex'])
        hi = min(c['log10_max'], np.log10(grid[i]) + r['half_width_dex'])
        g2 = np.logspace(lo, hi, r['n'])
        m2 = np.array(pool.map(
            _task, [('uniform', [np.log10(d)], None, V['name']) for d in g2],
            chunksize=2))
        all_grid.append(g2)
        all_mse.append(m2)

    full_grid = np.concatenate(all_grid)
    full_mse = np.concatenate(all_mse)
    order = np.argsort(full_grid)
    full_grid, full_mse = full_grid[order], full_mse[order]
    full_obj = np.array([objectives_from_mse(V, m) for m in full_mse])

    for j, which in enumerate(NORMS):
        curve = full_obj[:, j]
        k = int(np.argmin(curve))
        thr = curve[k] * (1.0 + band_frac)
        inside = np.where(curve <= thr)[0]
        lo_i, hi_i = int(inside[0]), int(inside[-1])
        out['per_norm'][which] = {
            'params': [float(np.log10(full_grid[k]))], 'param_names': ['log10_D'],
            'D': float(full_grid[k]), 'objective': float(curve[k]),
            'objective_abs': float(full_obj[k, 0]),
            'objective_norm': float(full_obj[k, 1]),
            'anchor_md_ft': None,
            'at_grid_edge': bool(k in (0, len(full_grid) - 1)),
            'band_10pct_D': [float(full_grid[lo_i]), float(full_grid[hi_i])],
            'band_censored_low': bool(lo_i == 0),
            'band_censored_high': bool(hi_i == len(full_grid) - 1),
        }
    # single-gauge equivalent D under the absolute norm, for the path-average story
    single = {}
    for k, tgt in enumerate(V['targets']):
        kk = int(np.argmin(full_mse[:, k]))
        single[tgt['gauge']] = {'best_D': float(full_grid[kk]),
                                'rmse_psi': float(np.sqrt(full_mse[kk, k])),
                                'at_grid_edge': bool(kk in (0, len(full_grid) - 1))}
    out['single_gauge_fit'] = single
    out['grid'] = full_grid
    out['curve_abs'] = full_obj[:, 0]
    out['curve_norm'] = full_obj[:, 1]
    return out


def fit_two_zone(pool, V, model, anchor, bounds, search):
    """R independent COLD restarts; both norms share the coarse LHS evaluations.

    Restart r draws its own Latin hypercube from the FULL box with its own seed,
    narrows to `refine_frac` of the box around its own best point under each
    norm, then polishes from that point. Nothing is seeded from another fit, from
    the published optimum, or from anything that saw a withheld gauge (D1).
    """
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    R = int(search['n_restarts'])
    seeds = [int(search['seed']) + int(search['restart_seed_stride']) * r
             for r in range(R)]

    # ---- phase 1: coarse LHS, all restarts in one map, both norms scored -----
    coarse_pts, tasks = [], []
    for s in seeds:
        pts = lo + qmc.LatinHypercube(d=len(bounds), seed=s).random(
            int(search['n_coarse'])) * (hi - lo)
        coarse_pts.append(pts)
        tasks += [(model, q, anchor, V['name']) for q in pts]
    coarse_mse = np.array(pool.map(_task, tasks, chunksize=4))
    per = int(search['n_coarse'])

    # ---- phase 2: per restart per norm, refine around that restart's best ----
    ref_tasks, ref_key = [], []
    for r in range(R):
        mse_r = coarse_mse[r * per:(r + 1) * per]
        obj_r = np.array([objectives_from_mse(V, m) for m in mse_r])
        for j, which in enumerate(NORMS):
            c = coarse_pts[r][int(np.argmin(obj_r[:, j]))]
            half = (hi - lo) * float(search['refine_frac']) / 2.0
            l2 = np.maximum(lo, c - half)
            h2 = np.minimum(hi, c + half)
            pts = l2 + qmc.LatinHypercube(
                d=len(bounds), seed=seeds[r] + 7 + j).random(
                    int(search['n_refine'])) * (h2 - l2)
            ref_key.append((r, which, pts))
            ref_tasks += [(model, q, anchor, V['name']) for q in pts]
    ref_mse = np.array(pool.map(_task, ref_tasks, chunksize=4))

    # ---- phase 3: one Nelder-Mead polish per (restart, norm), concurrently ---
    starts, pol_tasks = {}, []
    off = 0
    nref = int(search['n_refine'])
    for (r, which, pts) in ref_key:
        mse_r = ref_mse[off:off + nref]
        off += nref
        j = NORMS.index(which)
        obj_r = np.array([objectives_from_mse(V, m) for m in mse_r])
        best_ref = pts[int(np.argmin(obj_r[:, j]))]
        # compare with this restart's coarse best under the same norm
        cm = coarse_mse[r * per:(r + 1) * per]
        co = np.array([objectives_from_mse(V, m) for m in cm])
        best_coarse = coarse_pts[r][int(np.argmin(co[:, j]))]
        cand = (best_ref if np.min(obj_r[:, j]) < np.min(co[:, j]) else best_coarse)
        starts[(r, which)] = np.asarray(cand, float)
        pol_tasks.append((model, cand, anchor,
                          int(search['nelder_mead_maxiter']), which, V['name']))
    polished = pool.map(_polish, pol_tasks)

    # Re-evaluate every polished point IN THE POOL. Calling _task in the parent
    # would use the parent's `_G`, which belongs to whichever variant was last
    # initialised there.
    final_p = [np.clip(np.asarray(fp, float), lo, hi) for (fv, fp, nf) in polished]
    final_mse = pool.map(_task, [(model, p, anchor, V['name']) for p in final_p])

    restarts = {which: [] for which in NORMS}
    for (key, (fv, fp, nfev), p, mse) in zip(ref_key, polished, final_p, final_mse):
        r, which = key[0], key[1]
        a, n = objectives_from_mse(V, np.asarray(mse, float))
        restarts[which].append({
            'restart': int(r), 'seed': seeds[r],
            'params': [float(x) for x in p],
            'objective': float(a if which == 'abs' else n),
            'objective_abs': a, 'objective_norm': n,
            'nfev_polish': int(nfev),
            # The value Nelder-Mead itself reported, recomputed independently
            # above. A non-zero difference means the polish and the rescore did
            # not see the same problem - the failure mode the _task guard exists
            # to catch - so it is recorded rather than assumed away.
            'polish_reported_objective': float(fv),
            'polish_recheck_diff': float((a if which == 'abs' else n) - fv),
            'start_params': [float(x) for x in starts[(r, which)]],
            'D_near': float(10 ** p[0]), 'D_far': float(10 ** p[1]),
            's_c_ft': float(10 ** p[2]), 'width_ft': float(10 ** p[3]),
            'at_bound': [bool(abs(x - b[0]) < 1e-6 or abs(x - b[1]) < 1e-6)
                         for x, b in zip(p, bounds)],
        })

    out = {'model': model, 'k': 4, 'anchor_md_ft': float(anchor),
           'param_names': ['log10_D_near', 'log10_D_far', 'log10_s_c', 'log10_width'],
           'per_norm': {}, 'restarts': restarts}
    for which in NORMS:
        rs = sorted(restarts[which], key=lambda d: d['objective'])
        vals = np.array([d['objective'] for d in rs])
        best = rs[0]
        out['per_norm'][which] = {
            'params': best['params'], 'objective': best['objective'],
            'objective_abs': best['objective_abs'],
            'objective_norm': best['objective_norm'],
            'anchor_md_ft': float(anchor),
            'D_near': best['D_near'], 'D_far': best['D_far'],
            's_c_ft': best['s_c_ft'], 'width_ft': best['width_ft'],
            'at_bound': best['at_bound'],
            'restart_envelope': {
                'n': int(len(vals)), 'min': float(vals.min()),
                'median': float(np.median(vals)), 'max': float(vals.max()),
                'spread_frac_of_min': float(vals.max() / vals.min() - 1.0),
                'values': [float(x) for x in vals]},
            'param_envelope': {
                nm: {'min': float(np.min([d['params'][i] for d in rs])),
                     'max': float(np.max([d['params'][i] for d in rs]))}
                for i, nm in enumerate(out['param_names'])},
        }
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score(V, prof, dt, thr_frac, keep_sim=False):
    ev = core.evaluate_profile(
        V['mesh'], prof, dt, V['t_total'], V['s_taxis'], V['s_data'],
        V['source_idx'], V['targets'], thr_frac)
    # Maximum principle for the heat equation: on each side of the Dirichlet
    # node the sub-domain has Dirichlet data at the node, no flux at its outer
    # end and a zero initial condition, so the interior can never exceed
    # max(0, max BC). Where a gauge's observed peak EXCEEDS the imposed peak, no
    # diffusivity whatsoever can reproduce it - "structurally incapable", not
    # "badly fitted".
    bc_peak = float(np.max(V['s_data']))
    for g in ev['per_gauge']:
        g['role'] = V['roles'][g['gauge']]
        g['is_prediction'] = g['role'] not in ('source_dirichlet',
                                               'imposed_extrapolation')
        g['rmse_normalised'] = (g['rmse_psi'] / g['obs_max_psi']
                                if g['obs_max_psi'] else float('nan'))
        g['bc_peak_psi'] = bc_peak
        g['max_principle_amp_bound'] = (bc_peak / g['obs_max_psi']
                                        if g['obs_max_psi'] else float('nan'))
        g['amp_bound_binding'] = bool(g['max_principle_amp_bound'] < 1.0)
    mse = np.array([g['rmse_psi'] ** 2 for g in ev['per_gauge']])
    omax = np.array([g['obs_max_psi'] for g in ev['per_gauge']])
    gg = [g['gauge'] for g in ev['per_gauge']]

    def gm(subset, normalised=False):
        pos = [i for i, n in enumerate(gg) if n in subset]
        if not pos:
            return float('nan')
        v = mse[pos] / omax[pos] ** 2 if normalised else mse[pos]
        return float(np.sqrt(np.mean(v)))

    ev['gauge_mean_rmse_calib'] = gm(V['calib_gauges'])
    ev['gauge_mean_norm_calib'] = gm(V['calib_gauges'], True)
    ev['gauge_mean_rmse_common4'] = gm([4, 5, 6, 7])
    ev['gauge_mean_norm_common4'] = gm([4, 5, 6, 7], True)
    ev['gauge_mean_rmse_common6'] = gm([2, 3, 4, 5, 6, 7])
    ev['gauge_mean_norm_common6'] = gm([2, 3, 4, 5, 6, 7], True)
    pred = [g['gauge'] for g in ev['per_gauge'] if g['is_prediction']]
    ev['gauge_mean_rmse_predicted'] = gm(pred)
    if keep_sim:
        taxis, rec = rc.solve_forward(
            V['mesh'], prof, dt, V['t_total'], V['s_taxis'], V['s_data'],
            V['source_idx'], record_idx=[t['idx'] for t in V['targets']])
        ev['sim_taxis'], ev['sim'] = taxis, rec
    return ev


def g_of(ev, n):
    return next(g for g in ev['per_gauge'] if g['gauge'] == n)


# --------------------------------------------------------------------------
# fit cache - so a killed run resumes instead of repeating an hour of solves
# --------------------------------------------------------------------------
def _cache_key(cfg, V, model):
    """Everything a fit depends on, hashed. A change anywhere invalidates it."""
    import hashlib
    payload = json.dumps({
        'variant': V['name'], 'model': model,
        'spec': V['spec'],
        'mesh': [float(V['mesh'][0]), float(V['mesh'][-1]), int(len(V['mesh']))],
        'source_idx': int(V['source_idx']),
        'calib_gauges': V['calib_gauges'],
        'bc_sha256': rm.sha256_array(V['s_data']),
        'taxis_sha256': rm.sha256_array(V['s_taxis']),
        'search': cfg['search'], 'model_cfg': cfg['models'][model],
        'solver': cfg['solver'],
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_path(cfg, V, model):
    return os.path.join(cfg['outputs']['dir'], 'cache_v2',
                        f"{V['name']}__{model}.json")


def cache_load(cfg, V, model):
    p = _cache_path(cfg, V, model)
    if not os.path.exists(p):
        return None
    try:
        doc = json.load(open(p))
    except Exception:
        return None
    if doc.get('key') != _cache_key(cfg, V, model):
        return None
    f = doc['fit']
    for k in ('grid', 'curve_abs', 'curve_norm'):
        if k in f:
            f[k] = np.asarray(f[k], float)
    return f


def cache_store(cfg, V, model, f):
    p = _cache_path(cfg, V, model)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    doc = {'key': _cache_key(cfg, V, model), 'variant': V['name'],
           'model': model,
           'written_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'fit': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in f.items()}}
    tmp = p + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(doc, fh)
    os.replace(tmp, p)


# --------------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()
                if k not in ('grid', 'curve_abs', 'curve_norm', 'sim', 'sim_taxis',
                             'mesh', 's_taxis', 's_data', 'targets', 'obs_max',
                             'spec')}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# --------------------------------------------------------------------------
def main():
    global _LOG_FH
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--dry-run', action='store_true',
                    help='build variants, evaluate continuity, skip the searches')
    args = ap.parse_args()
    t0_wall = datetime.datetime.now()
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    cfg, cfg_hash = load_config(args.config)
    out = cfg['outputs']
    os.makedirs(out['dir'], exist_ok=True)
    _LOG_FH = open(out['log'], 'a')
    log(f"=== run start {started_utc}  config sha256={cfg_hash[:16]}")

    dt = float(cfg['solver']['dt_s'])
    thr = float(cfg['metrics']['arrival_time']['threshold_frac'])
    search = cfg['search']
    nproc = int(search['processes'])

    series, gnums, gmds, frac_hits, t_start, t_end = load_window_data(cfg)
    _, fh_centroid = pick_source_gauge(cfg, series, frac_hits)
    log(f"gauges {sorted(series)}  frac-hit centroid MD {fh_centroid:.3f}")

    # Cross-check the shared loader against the r1 loader that produced the
    # established numbers, so the provenance chain is explicit.
    loader_check = {}
    try:
        gw = rd.load_window_gauges()
        for n in ALL_GAUGES:
            a = series[n]['delta_psi']
            b = np.asarray(gw.series[n].delta_psi, float)
            loader_check[f'g{n}'] = {
                'n_samples_r1': int(a.size), 'n_samples_rev2': int(b.size),
                'max_abs_diff_psi': (float(np.max(np.abs(a - b)))
                                     if a.size == b.size else None)}
        log("loader cross-check vs rev2_data.load_window_gauges: "
            + ", ".join(f"g{n}:{loader_check[f'g{n}']['max_abs_diff_psi']}"
                        for n in ALL_GAUGES))
    except Exception as exc:                              # pragma: no cover
        loader_check = {'error': repr(exc)}
        log(f"loader cross-check FAILED: {exc!r}")

    variants = [build_variant(cfg, series, fh_centroid, v) for v in cfg['variants']]
    for V in variants:
        log(f"--- {V['name']}: node MD {V['node_md_ft']:.1f} "
            f"(requested {V['src_md_ft']:.3f}, snap {V['md_snap_residual_ft']:+.3f} ft), "
            f"mesh {V['mesh'][0]:.0f}-{V['mesh'][-1]:.0f} nx={len(V['mesh'])}, "
            f"calib {V['calib_gauges']}, blind {V['blind_gauges']}")

    # ---- kernel identity assertion (cheap, not a re-verification) ----------
    Vb = variants[0]
    prof_chk = np.full(len(Vb['mesh']), 1150.0)
    ridx = [t['idx'] for t in Vb['targets']]
    ta1, r1r = core.solve_forward(Vb['mesh'], prof_chk, dt, Vb['t_total'],
                                  Vb['s_taxis'], Vb['s_data'], Vb['source_idx'],
                                  record_idx=ridx)
    ta2, r2r = rc.solve_forward(Vb['mesh'], prof_chk, dt, Vb['t_total'],
                                Vb['s_taxis'], Vb['s_data'], Vb['source_idx'],
                                record_idx=ridx)
    kernel_check = {'max_abs_diff_psi': float(np.max(np.abs(r1r - r2r))),
                    'taxis_identical': bool(np.array_equal(ta1, ta2)),
                    'note': 'rev2_core at theta=1/harmonic/lambda=0 vs r1 kernel'}
    log(f"kernel identity: max|diff| = {kernel_check['max_abs_diff_psi']:.3e} psi")

    # ---- continuity: EVALUATE the published parameter vectors --------------
    cont_cfg = cfg['continuity']
    cont = {'note': cont_cfg['note'], 'checks': []}
    pub = cont_cfg['published_params']
    pubv = cont_cfg['published_values']
    for vname, key, model, anchor_from in (
            ('baseline', 'gauge_uniform', 'uniform', None),
            ('baseline', 'gauge_two_zone', 'two_zone_bc', 'node'),
            ('base_frachit', 'frac_centroid_uniform', 'uniform', None),
            ('base_frachit', 'frac_centroid_two_zone', 'two_zone_bc', 'node')):
        V = next(v for v in variants if v['name'] == vname)
        anchor = V['node_md_ft'] if anchor_from == 'node' else None
        prof = PROFILE_FN[model](V['mesh'], anchor, np.asarray(pub[key], float))
        _init_worker(V, dt)
        mse = per_gauge_mse(V, prof, dt)
        a, n = objectives_from_mse(V, mse)
        pa = pubv[key.replace('_uniform', '_uniform_rmse_psi')
                  .replace('_two_zone', '_two_zone_rmse_psi')]
        pn = pubv[key.replace('_uniform', '_uniform_norm')
                  .replace('_two_zone', '_two_zone_norm')]
        cont['checks'].append({
            'variant': vname, 'published_key': key, 'model': model,
            'reproduced_rmse_psi': a, 'published_rmse_psi': pa,
            'abs_diff_psi': float(a - pa),
            'reproduced_norm': n, 'published_norm': pn,
            'norm_diff': float(n - pn)})
        log(f"continuity {vname}/{key}: {a:.4f} psi vs published {pa:.4f} "
            f"(d={a - pa:+.4f}); norm {n:.5f} vs {pn:.5f} (d={n - pn:+.5f})")

    # ---- naive, solver-free predictors of gauge 1 --------------------------
    obs1 = series[1]
    o1max = float(np.max(obs1['delta_psi']))
    naive = []

    def naive_row(label, taxis, vals, note):
        v = np.interp(obs1['taxis'], taxis, vals)
        r = v - obs1['delta_psi']
        naive.append({
            'predictor': label, 'note': note,
            'rmse_psi': float(np.sqrt(np.mean(r ** 2))),
            'rmse_normalised': float(np.sqrt(np.mean(r ** 2)) / o1max),
            'amplitude_ratio': float(np.max(v) / o1max),
            'arrival_err_s': float(
                core.arrival_time(obs1['taxis'], v, thr * o1max)
                - core.arrival_time(obs1['taxis'], obs1['delta_psi'], thr * o1max)),
            'bias_psi': float(np.mean(r))})

    naive_row('copy_g2', series[2]['taxis'], series[2]['delta_psi'],
              'predict gauge 1 by copying gauge 2 (261 ft away) - no model at all')
    naive_row('copy_g3', series[3]['taxis'], series[3]['delta_psi'],
              'predict gauge 1 by copying gauge 3 (523 ft away)')
    for tgt_md, lbl in ((16645.0, 'extrap_g2g3_to_16645'),
                        (float(fh_centroid), 'extrap_g2g3_to_frachit')):
        ta, va, info = extrapolate_linear(series, [2, 3], tgt_md)
        naive_row(lbl, ta, va,
                  f"two-point linear extrapolation of g2,g3 to MD {tgt_md:.1f}; "
                  f"this is exactly the imposed BC of the extrap_* variants")
    for r in naive:
        log(f"naive {r['predictor']:24s} RMSE {r['rmse_psi']:7.2f} psi "
            f"({r['rmse_normalised']:.4f}) amp {r['amplitude_ratio']:.3f} "
            f"arr {r['arrival_err_s']:+.1f} s")

    if args.dry_run:
        log('dry run: stopping before the searches')
        return

    # ---- fit every variant -------------------------------------------------
    fits, scores, sims = {}, {}, {}
    for V in variants:
        log(f"=== fitting {V['name']}: {V['label']}")
        fits[V['name']], scores[V['name']], sims[V['name']] = {}, {}, {}
        # The PARENT's `_G` must point at this variant too: anything evaluated
        # outside the pool reads it (see the guard in _task).
        _init_worker(V, dt)
        with Pool(nproc, initializer=_init_worker, initargs=(V, dt)) as pool:
            for model in V['models']:
                mc = cfg['models'][model]
                cached = cache_load(cfg, V, model)
                if cached is not None:
                    fits[V['name']][model] = cached
                    for which in NORMS:
                        p = cached['per_norm'][which]
                        log(f"    {model}[{which:4s}] CACHED  "
                            f"obj={p['objective']:9.4f}")
                    continue
                if model == 'uniform':
                    f = fit_uniform(pool, V, mc)
                    for which in NORMS:
                        p = f['per_norm'][which]
                        log(f"    uniform[{which:4s}] D={p['D']:9.1f} "
                            f"obj={p['objective']:9.4f} band "
                            f"[{p['band_10pct_D'][0]:.0f},{p['band_10pct_D'][1]:.0f}]"
                            f"{' CENSORED' if p['band_censored_low'] or p['band_censored_high'] else ''}")
                else:
                    anchor = (V['node_md_ft'] if mc['anchor'] == 'dirichlet_node_md'
                              else V['frac_hit_centroid'])
                    f = fit_two_zone(pool, V, model, anchor, mc['bounds'], search)
                    for which in NORMS:
                        p = f['per_norm'][which]
                        e = p['restart_envelope']
                        log(f"    {model}[{which:4s}] Dn={p['D_near']:9.1f} "
                            f"Df={p['D_far']:7.1f} sc={p['s_c_ft']:7.1f} "
                            f"w={p['width_ft']:6.1f} obj={p['objective']:9.4f} "
                            f"envelope[{e['min']:.4f},{e['median']:.4f},{e['max']:.4f}] "
                            f"spread {100 * e['spread_frac_of_min']:.1f}%"
                            f"{' AT BOUND' if any(p['at_bound']) else ''}")
                fits[V['name']][model] = f
                cache_store(cfg, V, model, f)
        # score the best parameters of each (model, norm)
        _init_worker(V, dt)
        for model in V['models']:
            f = fits[V['name']][model]
            for which in NORMS:
                p = f['per_norm'][which]
                prof = PROFILE_FN[model](V['mesh'], p['anchor_md_ft'],
                                         np.asarray(p['params'], float))
                ev = score(V, prof, dt, thr, keep_sim=True)
                sims[V['name']][(model, which)] = (ev.pop('sim_taxis'), ev.pop('sim'))
                scores[V['name']][(model, which)] = ev
            # per-restart scores of the blind gauges: the envelope of the ANSWER,
            # not just of the objective
            if 'restarts' in f:
                for which in NORMS:
                    for rec in f['restarts'][which]:
                        prof = PROFILE_FN[model](V['mesh'], f['anchor_md_ft'],
                                                 np.asarray(rec['params'], float))
                        evr = score(V, prof, dt, thr)
                        rec['blind_scores'] = {
                            f"g{n}": {'rmse_psi': g_of(evr, n)['rmse_psi'],
                                      'rmse_normalised': g_of(evr, n)['rmse_normalised'],
                                      'amplitude_ratio': g_of(evr, n)['amplitude_ratio'],
                                      'arrival_err_s': g_of(evr, n)['arrival_err_s']}
                            for n in V['blind_gauges']}

    # ---- domain-top sensitivity for the blind gauge-1 prediction -----------
    log("=== domain-top sensitivity (blind gauge 1 above the Dirichlet node)")
    top_rows = []
    for vname in ('src_g2', 'src_g3'):
        vspec = next(v for v in cfg['variants'] if v['name'] == vname)
        for top in cfg['mesh']['domain_top_sensitivity_md_ft']:
            Vt = build_variant(cfg, series, fh_centroid, vspec, top_md=top)
            _init_worker(Vt, dt)
            for model in Vt['models']:
                for which in NORMS:
                    p = fits[vname][model]['per_norm'][which]
                    prof = PROFILE_FN[model](Vt['mesh'], p['anchor_md_ft'],
                                             np.asarray(p['params'], float))
                    ev = score(Vt, prof, dt, thr)
                    g1 = g_of(ev, 1)
                    top_rows.append({
                        'variant': vname, 'model': model, 'norm': which,
                        'top_md_ft': float(top), 'nx': int(len(Vt['mesh'])),
                        'g1_rmse_psi': g1['rmse_psi'],
                        'g1_rmse_normalised': g1['rmse_normalised'],
                        'g1_amplitude_ratio': g1['amplitude_ratio'],
                        'g1_arrival_err_s': g1['arrival_err_s'],
                        'calib_gauge_mean_rmse_psi': ev['gauge_mean_rmse_calib'],
                        'calib_gauge_mean_norm': ev['gauge_mean_norm_calib']})
        r0 = [r for r in top_rows
              if r['variant'] == vname and r['model'] == 'uniform' and r['norm'] == 'abs']
        log(f"    {vname} uniform/abs: g1 RMSE {r0[0]['g1_rmse_psi']:.1f} psi at top "
            f"{r0[0]['top_md_ft']:.0f} -> {r0[-1]['g1_rmse_psi']:.1f} psi at top "
            f"{r0[-1]['top_md_ft']:.0f}; calibration objective "
            f"{r0[0]['calib_gauge_mean_rmse_psi']:.4f} -> "
            f"{r0[-1]['calib_gauge_mean_rmse_psi']:.4f}")

    # ---- comparison csv ----------------------------------------------------
    rows = []
    for V in variants:
        for model in V['models']:
            for which in NORMS:
                f = fits[V['name']][model]
                p = f['per_norm'][which]
                ev = scores[V['name']][(model, which)]
                ref_key = (model if (model, which) in scores['baseline_far']
                           else 'two_zone_frachit')
                base = scores['baseline_far'].get((ref_key, which))
                bg = {g['gauge']: g for g in base['per_gauge']} if base else {}
                for g in ev['per_gauge']:
                    b = bg.get(g['gauge'])
                    rows.append({
                        'variant': V['name'], 'variant_label': V['label'],
                        'model': model, 'norm': which,
                        'dirichlet_md_ft': round(V['node_md_ft'], 2),
                        'domain_top_md_ft': round(V['top_md_ft'], 1),
                        'driver_gauges': '+'.join(str(x) for x in V['driver_gauges']),
                        'calib_gauges': '+'.join(str(x) for x in V['calib_gauges']),
                        'gauge': g['gauge'], 'gauge_md_ft': g['md_ft'],
                        'role': g['role'], 'is_prediction': g['is_prediction'],
                        'distance_from_bc_ft': round(g['distance_ft'], 1),
                        'rmse_psi': round(g['rmse_psi'], 4),
                        'rmse_normalised': round(g['rmse_normalised'], 6),
                        'amplitude_ratio': round(g['amplitude_ratio'], 5),
                        'arrival_err_s': (round(g['arrival_err_s'], 3)
                                          if np.isfinite(g['arrival_err_s']) else ''),
                        'obs_max_psi': round(g['obs_max_psi'], 2),
                        'sim_max_psi': round(g['sim_max_psi'], 2),
                        'bc_peak_psi': round(g['bc_peak_psi'], 2),
                        'max_principle_amp_bound': round(g['max_principle_amp_bound'], 4),
                        'amp_bound_binding': g['amp_bound_binding'],
                        'bias_psi': round(g['bias_psi'], 3),
                        'rmse_psi_ref_baseline_far': (round(b['rmse_psi'], 4) if b else ''),
                        'delta_rmse_vs_baseline_far_psi': (round(g['rmse_psi'] - b['rmse_psi'], 4)
                                                           if b else ''),
                        'rmse_ratio_vs_baseline_far': (round(g['rmse_psi'] / b['rmse_psi'], 4)
                                                       if b and b['rmse_psi'] else ''),
                        'calibrated_D': (round(p['D'], 2) if model == 'uniform' else ''),
                        'calibrated_D_near': (round(p['D_near'], 2) if model != 'uniform' else ''),
                        'calibrated_D_far': (round(p['D_far'], 2) if model != 'uniform' else ''),
                        'calibrated_s_c_ft': (round(p['s_c_ft'], 2) if model != 'uniform' else ''),
                        'objective': round(p['objective'], 6),
                        'objective_abs_psi': round(p['objective_abs'], 4),
                        'objective_norm': round(p['objective_norm'], 6),
                        'calib_gauge_mean_rmse_psi': round(ev['gauge_mean_rmse_calib'], 4),
                        'calib_gauge_mean_norm': round(ev['gauge_mean_norm_calib'], 6),
                        'common4_rmse_psi': round(ev['gauge_mean_rmse_common4'], 4),
                        'common4_norm': round(ev['gauge_mean_norm_common4'], 6),
                        'common6_rmse_psi': round(ev['gauge_mean_rmse_common6'], 4),
                        'common6_norm': round(ev['gauge_mean_norm_common6'], 6),
                    })
    with open(out['comparison_csv'], 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {out['comparison_csv']} ({len(rows)} rows)")

    # ---- restart envelope csv ---------------------------------------------
    erows = []
    for V in variants:
        for model in V['models']:
            f = fits[V['name']][model]
            if 'restarts' not in f:
                continue
            for which in NORMS:
                for rec in f['restarts'][which]:
                    row = {'variant': V['name'], 'model': model, 'norm': which,
                           'restart': rec['restart'], 'seed': rec['seed'],
                           'objective': rec['objective'],
                           'objective_abs_psi': rec['objective_abs'],
                           'objective_norm': rec['objective_norm'],
                           'D_near': rec['D_near'], 'D_far': rec['D_far'],
                           's_c_ft': rec['s_c_ft'], 'width_ft': rec['width_ft'],
                           'nfev_polish': rec['nfev_polish'],
                           'at_bound': any(rec['at_bound'])}
                    for gk, gv in rec.get('blind_scores', {}).items():
                        row[f'blind_{gk}_rmse_psi'] = round(gv['rmse_psi'], 4)
                        row[f'blind_{gk}_amp'] = round(gv['amplitude_ratio'], 4)
                        row[f'blind_{gk}_arr_s'] = (round(gv['arrival_err_s'], 2)
                                                    if np.isfinite(gv['arrival_err_s'])
                                                    else '')
                    erows.append(row)
    keys = sorted({k for r in erows for k in r})
    head = [k for k in ('variant', 'model', 'norm', 'restart', 'seed', 'objective',
                        'objective_abs_psi', 'objective_norm', 'D_near', 'D_far',
                        's_c_ft', 'width_ft', 'nfev_polish', 'at_bound') if k in keys]
    head += [k for k in keys if k not in head]
    with open(out['envelope_csv'], 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=head, extrasaction='ignore')
        w.writeheader()
        for r in erows:
            w.writerow({k: r.get(k, '') for k in head})
    log(f"wrote {out['envelope_csv']} ({len(erows)} rows)")

    # ---- arrays ------------------------------------------------------------
    npz = {}
    for V in variants:
        npz[f"bc_taxis__{V['name']}"] = V['s_taxis']
        npz[f"bc_data__{V['name']}"] = V['s_data']
        npz[f"mesh_span__{V['name']}"] = np.array([V['mesh'][0], V['mesh'][-1],
                                                   float(len(V['mesh']))])
        for (model, which), (ta, rec) in sims[V['name']].items():
            npz[f"sim_taxis__{V['name']}__{model}__{which}"] = ta
            npz[f"sim__{V['name']}__{model}__{which}"] = rec
        u = fits[V['name']].get('uniform')
        if u is not None:
            npz[f"uniform_grid__{V['name']}"] = u['grid']
            npz[f"uniform_curve_abs__{V['name']}"] = u['curve_abs']
            npz[f"uniform_curve_norm__{V['name']}"] = u['curve_norm']
    for n in ALL_GAUGES:
        npz[f"obs_taxis__g{n}"] = series[n]['taxis']
        npz[f"obs_data__g{n}"] = series[n]['delta_psi']
    npz['gauge_numbers'] = np.array(ALL_GAUGES)
    npz['gauge_md_ft'] = np.array([series[n]['md_ft'] for n in ALL_GAUGES])
    np.savez_compressed(out['arrays_npz'], **npz)
    log(f"wrote {out['arrays_npz']}")

    # ---- summary -----------------------------------------------------------
    summary = {
        'study_id': cfg['study_id'], 'task_id': cfg['task_id'],
        'started_utc': started_utc,
        'kernel_identity_check': kernel_check,
        'loader_cross_check': loader_check,
        'continuity_vs_published': cont,
        'naive_gauge1_predictors': naive,
        'variants': {V['name']: {
            'label': V['label'], 'dirichlet_md_ft': V['node_md_ft'],
            'requested_md_ft': V['src_md_ft'],
            'domain_md_ft': [float(V['mesh'][0]), float(V['mesh'][-1])],
            'nx': int(len(V['mesh'])),
            'driver_gauges': V['driver_gauges'], 'calib_gauges': V['calib_gauges'],
            'blind_gauges': V['blind_gauges'], 'roles': V['roles'],
            'bc_peak_psi': float(np.max(V['s_data'])),
            'extrapolation': V['extrap_info'],
            'fits': _jsonable(fits[V['name']]),
            'scores': {f"{m}::{w}": _jsonable(ev)
                       for (m, w), ev in scores[V['name']].items()},
        } for V in variants},
        'domain_top_sensitivity': top_rows,
    }
    with open(out['summary_json'], 'w') as fh:
        json.dump(_jsonable(summary), fh, indent=2)
    log(f"wrote {out['summary_json']}")

    # ---- figures -----------------------------------------------------------
    try:
        import d2_figures_v2 as figmod
        figmod.make_all(cfg, variants, series, sims, scores, fits, summary)
        log('figures written')
    except Exception as exc:                              # pragma: no cover
        log(f"FIGURES FAILED: {exc!r}")

    # ---- manifests ---------------------------------------------------------
    inputs = [(cfg['data']['gauge_md_npz'], 'geometry', 'gauge_md_swell'),
              (cfg['data']['frac_hit_stage1_npz'], 'geometry', 'frac_hit_stage_1'),
              (cfg['data']['well_geometry_npz'], 'geometry', 'swell_geometry'),
              (cfg['data']['r2_manifest'], 'prior_run_output', 'r2_profile_manifest')]
    for n in ALL_GAUGES:
        inputs.append((cfg['data']['gauge_series_template'].format(n=n),
                       'gauge_series', f'gauge{n}_swell'))

    # The run log is NOT declared as an output: it is still being appended to
    # after the manifest is hashed, so declaring it guarantees a permanent DRIFT
    # against the manifest's own record. Its path is named in the notes instead.
    prod = [out['comparison_csv'], out['envelope_csv'], out['summary_json'],
            out['arrays_npz'], out['figure_main_png'],
            out['figure_degradation_png'], out['figure_control_png']]

    def decls():
        d = [rm.output_decl(out['comparison_csv'], role='csv',
                            note='per variant x model x norm x gauge'),
             rm.output_decl(out['envelope_csv'], role='csv',
                            note='one row per cold restart'),
             rm.output_decl(out['summary_json'], role='json'),
             rm.output_decl(out['arrays_npz'], role='arrays_npz')]
        for k in ('figure_main_png', 'figure_degradation_png', 'figure_control_png'):
            if os.path.exists(out[k]):
                d.append(rm.output_decl(out[k], role='figure_png',
                                        dpi=int(out['figure_dpi'])))
        return d

    def groups_for(V):
        if V['extrap_info']:
            drv = rm.driver_record(
                kind='synthetic',
                baseline_removal=cfg['source']['baseline_removal'],
                value_units='delta_psi', series_path=None,
                gauge_number=None, gauge_md_ft=float(V['src_md_ft']),
                taxis=V['s_taxis'], values=V['s_data'],
                time_start=cfg['window']['time_start'],
                time_end=cfg['window']['time_end'])
        else:
            drv = rm.driver_record(
                kind='gauge_series',
                baseline_removal=cfg['source']['baseline_removal'],
                value_units='delta_psi',
                series_path=cfg['data']['gauge_series_template'].format(
                    n=int(V['src_gauge'])),
                gauge_number=int(V['src_gauge']),
                gauge_md_ft=float(series[int(V['src_gauge'])]['md_ft']),
                taxis=V['s_taxis'], values=V['s_data'],
                time_start=cfg['window']['time_start'],
                time_end=cfg['window']['time_end'])
        src = rm.source_record(V['mesh'], md_requested_ft=float(V['src_md_ft']),
                               mesh_idx=int(V['source_idx']), driver=drv,
                               label=V['name'],
                               excluded_from_misfit=bool(
                                   all(g not in V['calib_gauges']
                                       for g in V['driver_gauges'])),
                               index_in_source_list=0)
        sg = rm.source_protocol(
            application=cfg['source']['application'],
            solver_class=cfg['solver']['class'],
            placement_rule=f"{V['name']}: {V['label']} "
                           f"[bc.kind={V['spec']['bc']['kind']}, "
                           f"node MD {V['node_md_ft']:.2f}]",
            sources=[src],
            targets={'gauges': [int(t['gauge']) for t in V['targets']],
                     'md_ft': [float(t['md_ft']) for t in V['targets']],
                     'distance_ft': [float(t['distance_ft']) for t in V['targets']],
                     'n_samples': [int(t['taxis'].size) for t in V['targets']],
                     'roles': V['roles'],
                     'calibration_gauges': V['calib_gauges'],
                     'blind_gauges': V['blind_gauges'],
                     'role': ('all seven gauges are scored; only calibration_gauges '
                              'enter the objective')},
            time_level='n',
            phase_chaining={'mode': 'single_phase',
                            'note': 'one window, one solve per parameter vector'},
            boundary_conditions=cfg['source']['boundary_conditions'])
        best = fits[V['name']][V['models'][-1]]['per_norm']['abs']
        prof = PROFILE_FN[V['models'][-1]](V['mesh'], best['anchor_md_ft'],
                                           np.asarray(best['params'], float))
        num = rm.numerics(
            time=rm.time_record(sims[V['name']][(V['models'][-1], 'abs')][0],
                                mode='fixed', theta=float(cfg['solver']['theta']),
                                t_total_requested_s=float(V['t_total']),
                                dt_requested_s=dt, source_time_level='n',
                                theta_startup_steps=0,
                                label=f"{V['name']}: every solve shares this taxis"),
            mesh=rm.mesh_record(V['mesh'], dx_requested_ft=float(cfg['mesh']['dx_ft']),
                                window_md_ft=(float(cfg['window']['md_min_ft']),
                                              float(cfg['window']['md_max_ft'])),
                                pad_low_ft=float(cfg['mesh']['domain_pad_low_md_ft']),
                                pad_high_ft=float(V['mesh'][-1]
                                                  - cfg['window']['md_max_ft']),
                                refinement={'mode': 'none', 'uniform': True}),
            interface_avg=cfg['solver']['interface_avg'],
            boundary=cfg['source']['boundary_conditions'],
            diffusivity={'profile_family': V['models'][-1],
                         'param_names': ['log10_D_near', 'log10_D_far',
                                         'log10_s_c', 'log10_width'],
                         'params': best['params'],
                         'anchor_md_ft': best['anchor_md_ft'],
                         'D_min': float(np.min(prof)), 'D_max': float(np.max(prof)),
                         'D_sha256': rm.sha256_array(prof),
                         'note': ('the recorded profile is this variant absolute-norm '
                                  'optimum; every fitted profile of this study is in '
                                  'results.fits')},
            barriers=rm.NONE_DECLARED,
            leakage={'model': 'none (lambda_leak = 0)',
                     'note': 'C2 measured lambda_leak* = 0 exactly under both norms'},
            kernel={'name': 'rev2_core.solve_forward', 'banded': True,
                    'equivalence_reference':
                        'output/rev2_20260901/A4/selftest_output.txt',
                    'asserted_in_this_run': kernel_check},
            rng={'latin_hypercube_seeds': [int(search['seed'])
                                           + int(search['restart_seed_stride']) * r
                                           for r in range(int(search['n_restarts']))],
                 'used_for': 'the cold-start two_zone restarts; the uniform search '
                             'is a deterministic grid'},
            parallel={'mode': 'multiprocessing.Pool',
                      'processes': int(search['processes']),
                      'why': 'house rules cap concurrent tasks at 6 workers'},
            amplification=rc.amplification_factor(
                V['mesh'], prof, dt, float(cfg['solver']['theta']),
                interface_avg=cfg['solver']['interface_avg'], lambda_leak=0.0))
        return sg, num

    for V in variants:
        p = out['per_variant_manifest_template'].format(name=V['name'])
        os.makedirs(os.path.dirname(p), exist_ok=True)
        sg, num = groups_for(V)
        rm.write_manifest(
            p, study_id=f"{cfg['study_id']}::{V['name']}", task_id=cfg['task_id'],
            config=cfg, config_path=args.config, inputs=inputs,
            source=sg, numerics=num, outputs=decls(),
            results={'variant': V['name'], 'label': V['label'],
                     'roles': V['roles'], 'calib_gauges': V['calib_gauges'],
                     'blind_gauges': V['blind_gauges'],
                     'fits': _jsonable(fits[V['name']]),
                     'scores': {f"{m}::{w}": _jsonable(ev)
                                for (m, w), ev in scores[V['name']].items()}},
            notes=[V['spec']['role'], cfg['search']['cold_start_rule']],
            started_utc=started_utc, run_label=f"D2v2::{V['name']}",
            require_modules=('rev2_core', 'rev2_data', 'rev2_manifest',
                             'r1_calibration_core'),
            extra_code_files=(os.path.abspath(__file__),),
            overwrite=True, allow_undeclared_outputs=True)
        log(f"wrote {p}")

    sg, num = groups_for(variants[0])
    rm.write_manifest(
        out['manifest_json'], study_id=cfg['study_id'], task_id=cfg['task_id'],
        config=cfg, config_path=args.config, inputs=inputs,
        source=sg, numerics=num, outputs=decls(),
        results=_jsonable(summary),
        notes=[cfg['description'], cfg['search']['cold_start_rule'],
               cfg['mesh']['top_rule'],
               ('per-variant manifests are under '
                + os.path.dirname(out['per_variant_manifest_template'])),
               ('the run log is ' + out['log'] + '; it is deliberately NOT a '
                'declared output because it keeps growing after the manifest '
                'is hashed')],
        started_utc=started_utc, run_label='D2v2',
        require_modules=('rev2_core', 'rev2_data', 'rev2_manifest',
                             'r1_calibration_core'),
        extra_code_files=(os.path.abspath(__file__),),
        overwrite=True, allow_undeclared_outputs=True)
    log(f"wrote {out['manifest_json']}")
    log(f"done, wall {(datetime.datetime.now() - t0_wall).total_seconds():.0f} s")


if __name__ == '__main__':
    main()
