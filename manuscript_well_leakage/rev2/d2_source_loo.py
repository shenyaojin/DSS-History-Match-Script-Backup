"""D2 - remove the SOURCE gauge and see how much the prediction degrades.

Reviewer 2's charge is circularity: gauge 1 (MD 16645) is imposed as a Dirichlet
boundary value, so it is fit trivially and is excluded from every misfit. This
study rebuilds the boundary condition WITHOUT gauge 1 under four substitutions,
recalibrates each model on an identical held-in gauge set {4,5,6,7}, and then
rescores every gauge - including gauge 1, which becomes a genuine blind target
in three of the four substitutions.

Substitutions
  (a)  extrap_g1md     linear two-point extrapolation of gauges 2 and 3 to
                       MD 16645. BC VALUE changes, BC LOCATION does not.
  (a') extrap_frachit  the same rule extrapolated to the stage-1 frac-hit
                       centroid MD 16683.2, which is where the fluid actually
                       enters. Gauge 1 is then a blind target 38 ft away.
  (b)  src_g2          gauge 2 drives the Dirichlet node at its own MD 16384.
                       Gauge 1 is a blind target 261 ft on the injection side.
  (c)  src_g3          the same one gauge further out (MD 16122), which turns a
                       single number into a degradation trend.

Two controls make the comparison fair:
  baseline      the published protocol, calibrated on {2..7}. It must reproduce
                the established uniform 82.33 psi and two_zone 11.87 psi or the
                pipeline is wrong.
  baseline_far  the published protocol recalibrated on {4,5,6,7} only, so that
                the substitutions differ from it in the boundary condition ALONE
                and not in the calibration set.

Run from the repository root:

    python scripts/manuscript_well_leakage/rev2/d2_source_loo.py \
        --config configs/rev2/d2_source_loo.json
"""

import argparse
import csv
import datetime
import json
import os
import platform
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', '..'))
_BC = os.path.join(REPO, 'scripts', 'manuscript_well_leakage', 'baseline_calibration')
sys.path.insert(0, _BC)
import r1_calibration_core as core  # noqa: E402
from r1_run_calibration import load_config, load_window_data, pick_source_gauge  # noqa: E402

ALL_GAUGES = [1, 2, 3, 4, 5, 6, 7]
_G = {}


def log(m):
    print(f"[d2] {m}", flush=True)


# --------------------------------------------------------------------------
# profiles
# --------------------------------------------------------------------------
def profile_uniform(mesh, anchor_md, p):
    return np.full(len(mesh), 10.0 ** p[0])


def profile_two_zone_anchored(mesh, anchor_md, p):
    """core.profile_two_zone but with the anchor given as a physical MD.

    core's version measures distance from the SOURCE NODE, so moving the
    boundary condition silently moves the high-diffusivity zone with it. Passing
    the anchor explicitly lets the same shape be pinned to the injection point
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
    as a separate diagnostic so the reader can separate 'the extrapolation is
    wrong' from 'the model is wrong'.
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
        'formula': 'P(md) = P_near + w*(P_near - P_far), w = (md - md_near)/(md_near - md_far)',
    }


def build_variant(cfg, series, fh_centroid, vspec, top_md):
    """Mesh, Dirichlet node, driving series, targets and gauge roles."""
    dx = float(cfg['mesh']['dx_ft'])
    pad_lo = float(cfg['mesh']['domain_pad_low_md_ft'])
    win_lo = float(cfg['window']['md_min_ft'])
    mesh = np.arange(win_lo - pad_lo, float(top_md) + dx / 2.0, dx)

    bc = vspec['bc']
    driver_gauges, extrap_info, src_gauge = [], None, None
    if bc['kind'] == 'gauge':
        src_gauge = int(bc['gauge'])
        src_md = float(series[src_gauge]['md_ft'])
        s_taxis = series[src_gauge]['taxis']
        s_data = series[src_gauge]['delta_psi']
        driver_gauges = [src_gauge]
        driving_paths = [cfg['data']['gauge_series_template'].format(n=src_gauge)]
    elif bc['kind'] == 'extrapolate':
        src_md = (float(fh_centroid) if bc.get('md_from') == 'frac_hit_centroid'
                  else float(bc['md_ft']))
        driver_gauges = [int(g) for g in bc['from_gauges']]
        s_taxis, s_data, extrap_info = extrapolate_linear(
            series, driver_gauges, src_md)
        driving_paths = [cfg['data']['gauge_series_template'].format(n=g)
                         for g in driver_gauges]
    else:
        raise ValueError(bc['kind'])

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
            roles[n] = 'source_dirichlet' if bc['kind'] == 'gauge' else 'imposed_extrapolation'
        elif n in driver_gauges:
            roles[n] = 'bc_constituent'
        elif n in vspec['calib_gauges']:
            roles[n] = 'calibration'
        else:
            roles[n] = 'blind'

    calib = [n for n in vspec['calib_gauges'] if roles[n] == 'calibration']
    if sorted(calib) != sorted(vspec['calib_gauges']):
        raise RuntimeError(f"{vspec['name']}: calibration gauge is also a driver")

    return {
        'name': vspec['name'], 'label': vspec['label'], 'spec': vspec,
        'mesh': mesh, 'source_idx': source_idx, 'src_md_ft': src_md,
        'node_md_ft': node_md, 'md_snap_residual_ft': float(node_md - src_md),
        'src_gauge': src_gauge, 'driver_gauges': driver_gauges,
        'driving_paths': driving_paths, 'extrap_info': extrap_info,
        's_taxis': np.asarray(s_taxis, float), 's_data': np.asarray(s_data, float),
        't_total': float(np.asarray(s_taxis, float)[-1]),
        'targets': targets, 'roles': roles, 'calib_gauges': calib,
        'calib_pos': [i for i, t in enumerate(targets) if t['gauge'] in calib],
        'blind_gauges': [n for n in ALL_GAUGES if roles[n] == 'blind'],
        'frac_hit_centroid': float(fh_centroid),
        'top_md_ft': float(mesh[-1]), 'dx_ft': dx, 'pad_low_ft': pad_lo,
    }


# --------------------------------------------------------------------------
# forward evaluation
# --------------------------------------------------------------------------
def per_gauge_mse(V, prof, dt):
    """Simulate once, return per-gauge MSE for all seven gauges."""
    if not np.all(np.isfinite(prof)) or np.any(prof <= 0):
        return np.full(len(V['targets']), np.inf)
    taxis, rec = core.solve_forward(
        V['mesh'], prof, dt, V['t_total'], V['s_taxis'], V['s_data'],
        V['source_idx'], record_idx=[t['idx'] for t in V['targets']])
    if not np.all(np.isfinite(rec)):
        return np.full(len(V['targets']), np.inf)
    out = np.empty(len(V['targets']))
    for k, tgt in enumerate(V['targets']):
        r = np.interp(tgt['taxis'], taxis, rec[:, k]) - tgt['data']
        out[k] = float(np.mean(r ** 2))
    return out


def _init_worker(V, dt):
    _G['V'], _G['dt'] = V, dt


def _task(args):
    model, params, anchor = args
    V = _G['V']
    prof = PROFILE_FN[model](V['mesh'], anchor, np.asarray(params, float))
    mse = per_gauge_mse(V, prof, _G['dt'])
    obj = float(np.sqrt(np.mean(mse[V['calib_pos']])))
    return obj, mse.tolist()


def objective(model, params, anchor):
    return _task((model, params, anchor))[0]


def _polish(args):
    """One Nelder-Mead polish, run inside a worker.

    Nelder-Mead is serial, so a multi-start polish would otherwise cost one
    full NM run per start. Farming the starts out to the pool makes a 4-start
    polish cost the wall time of one, which is what buys enough convergence for
    the baseline variant to reproduce the published two_zone optimum.
    """
    model, start, anchor, maxiter = args
    res = minimize(lambda q: objective(model, q, anchor), np.asarray(start, float),
                   method='Nelder-Mead',
                   options={'maxiter': maxiter, 'xatol': 1e-3, 'fatol': 1e-4,
                            'disp': False})
    return float(res.fun), [float(x) for x in np.asarray(res.x, float)]


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
def fit_uniform(pool, V, mcfg):
    c, r = mcfg['coarse'], mcfg['refine']
    grid = np.logspace(c['log10_min'], c['log10_max'], c['n'])
    res = pool.map(_task, [('uniform', [np.log10(d)], None) for d in grid], chunksize=2)
    obj = np.array([x[0] for x in res])
    i = int(np.argmin(obj))
    lo = max(c['log10_min'], np.log10(grid[i]) - r['half_width_dex'])
    hi = min(c['log10_max'], np.log10(grid[i]) + r['half_width_dex'])
    grid2 = np.logspace(lo, hi, r['n'])
    res2 = pool.map(_task, [('uniform', [np.log10(d)], None) for d in grid2], chunksize=2)
    obj2 = np.array([x[0] for x in res2])

    full_grid = np.concatenate([grid, grid2])
    full_obj = np.concatenate([obj, obj2])
    full_mse = np.array([x[1] for x in res] + [x[1] for x in res2])
    order = np.argsort(full_grid)
    full_grid, full_obj, full_mse = full_grid[order], full_obj[order], full_mse[order]

    j = int(np.argmin(full_obj))
    single = {}
    for k, tgt in enumerate(V['targets']):
        kk = int(np.argmin(full_mse[:, k]))
        single[tgt['gauge']] = {
            'best_D': float(full_grid[kk]),
            'rmse_psi': float(np.sqrt(full_mse[kk, k])),
            'at_grid_edge': bool(kk in (0, len(full_grid) - 1)),
        }
    return {
        'model': 'uniform', 'k': 1, 'params': [float(np.log10(full_grid[j]))],
        'param_names': ['log10_D'], 'D': float(full_grid[j]),
        'objective_psi': float(full_obj[j]), 'anchor_md_ft': None,
        'at_bound': [bool(j in (0, len(full_grid) - 1))],
        'grid': full_grid, 'curve': full_obj, 'per_gauge_mse_grid': full_mse,
        'single_gauge_fit': single,
    }


def fit_two_zone(pool, V, model, anchor, bounds, search):
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)

    def draw(centre, frac, n, sd):
        if centre is None:
            l2, h2 = lo, hi
        else:
            half = (hi - lo) * frac / 2.0
            l2 = np.maximum(lo, centre - half)
            h2 = np.minimum(hi, centre + half)
        pts = l2 + qmc.LatinHypercube(d=len(bounds), seed=sd).random(n) * (h2 - l2)
        vals = np.array([x[0] for x in pool.map(
            _task, [(model, q, anchor) for q in pts], chunksize=4)])
        return pts, vals

    pts, v = draw(None, None, search['n_coarse'], search['seed'])
    order = np.argsort(v)
    best_p, best_v = pts[order[0]].copy(), float(v[order[0]])
    pts2, v2 = draw(best_p, search['refine_frac'], search['n_refine'], search['seed'] + 1)
    if float(v2.min()) < best_v:
        best_p, best_v = pts2[int(np.argmin(v2))].copy(), float(v2.min())

    # Multi-start polish: the best LHS point plus the next few distinct basins
    # from the coarse round, so a single deep-but-wrong basin cannot decide the
    # answer. All starts are polished concurrently in the pool.
    starts = [best_p]
    for idx in order[:search.get('n_polish_starts', 4) * 4]:
        cand = pts[idx]
        if all(np.max(np.abs(cand - s)) > 0.25 for s in starts):
            starts.append(cand.copy())
        if len(starts) >= search.get('n_polish_starts', 4):
            break
    polished = pool.map(_polish, [(model, s, anchor, search['nelder_mead_maxiter'])
                                  for s in starts])
    for fv, fp in polished:
        if fv < best_v:
            best_v, best_p = float(fv), np.asarray(fp, float)
    best_p = np.clip(best_p, lo, hi)
    return {
        'model': model, 'k': 4, 'params': [float(x) for x in best_p],
        'param_names': ['log10_D_near', 'log10_D_far', 'log10_s_c', 'log10_width'],
        'D_near': float(10 ** best_p[0]), 'D_far': float(10 ** best_p[1]),
        's_c_ft': float(10 ** best_p[2]), 'width_ft': float(10 ** best_p[3]),
        'objective_psi': objective(model, best_p, anchor),
        'anchor_md_ft': float(anchor),
        'at_bound': [bool(abs(x - b[0]) < 1e-6 or abs(x - b[1]) < 1e-6)
                     for x, b in zip(best_p, bounds)],
    }


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score(V, prof, dt, thr_frac):
    ev = core.evaluate_profile(
        V['mesh'], prof, dt, V['t_total'], V['s_taxis'], V['s_data'],
        V['source_idx'], V['targets'], thr_frac)
    taxis, rec = core.solve_forward(
        V['mesh'], prof, dt, V['t_total'], V['s_taxis'], V['s_data'],
        V['source_idx'], record_idx=[t['idx'] for t in V['targets']])
    # Maximum principle for the heat equation: on each side of the Dirichlet
    # node the sub-domain has Dirichlet data at the node, no flux at its outer
    # end and a zero initial condition, so the interior can never exceed
    # max(0, max BC). Whenever a gauge's observed peak is LARGER than the peak
    # of the imposed series, no diffusivity whatsoever can reproduce it: the
    # amplitude ratio is capped at bc_peak/obs_peak. That converts "the fit is
    # bad" into "the configuration is structurally incapable", which is a much
    # stronger and more honest statement.
    bc_peak = float(np.max(V['s_data']))
    for g in ev['per_gauge']:
        g['role'] = V['roles'][g['gauge']]
        g['is_prediction'] = g['role'] not in ('source_dirichlet', 'imposed_extrapolation')
        g['bc_peak_psi'] = bc_peak
        g['max_principle_amp_bound'] = (bc_peak / g['obs_max_psi']
                                        if g['obs_max_psi'] else float('nan'))
        g['amp_bound_binding'] = bool(g['max_principle_amp_bound'] < 1.0)
    mse = np.array([g['rmse_psi'] ** 2 for g in ev['per_gauge']])
    gg = [g['gauge'] for g in ev['per_gauge']]

    def gm(subset):
        pos = [i for i, n in enumerate(gg) if n in subset]
        return float(np.sqrt(np.mean(mse[pos]))) if pos else float('nan')

    ev['gauge_mean_rmse_calib'] = gm(V['calib_gauges'])
    ev['gauge_mean_rmse_common4'] = gm([4, 5, 6, 7])
    ev['gauge_mean_rmse_predicted'] = gm(
        [g['gauge'] for g in ev['per_gauge'] if g['is_prediction']])
    ev['sim_taxis'] = taxis
    ev['sim'] = rec
    return ev


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()
                if k not in ('grid', 'curve', 'per_gauge_mse_grid', 'sim', 'sim_taxis')}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def code_hashes():
    """sha256 of every .py file this run actually imported from inside the repo."""
    out = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if not f or not f.endswith('.py'):
            continue
        f = os.path.abspath(f)
        if not f.startswith(REPO + os.sep) or not os.path.exists(f):
            continue
        out[os.path.relpath(f, REPO)] = core.file_sha256(f)
    out[os.path.relpath(os.path.abspath(__file__), REPO)] = \
        core.file_sha256(os.path.abspath(__file__))
    return dict(sorted(out.items()))


def output_records(paths):
    recs = []
    for p in paths:
        if os.path.exists(p):
            recs.append({'path': p, 'bytes': os.path.getsize(p),
                         'sha256': core.file_sha256(p)})
    return recs


def make_manifest(study_id, cfg, cfg_path, cfg_hash, inputs, source_protocol,
                  numerics, results, outputs):
    return {
        'study_id': study_id,
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
            'cwd': os.getcwd(),
            'config_path': os.path.abspath(cfg_path),
            'note': ('bakken_mariner/.git is empty; no commit hash is recoverable, '
                     'so code identity is pinned by the sha256 values below.'),
            'code_sha256': code_hashes(),
            'input_data_sha256': inputs,
        },
        'source_protocol': source_protocol,
        'numerics': numerics,
        'results': results,
        'outputs': outputs,
    }


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    t_wall0 = datetime.datetime.now()

    cfg, cfg_hash = load_config(args.config)
    log(f"config {args.config} sha256={cfg_hash[:16]}")
    out = cfg['outputs']
    os.makedirs(out['dir'], exist_ok=True)

    dt = float(cfg['solver']['dt_s'])
    thr = float(cfg['metrics']['arrival_time']['threshold_frac'])
    search = cfg['search']
    nproc = int(search['processes'])

    series, gnums, gmds, frac_hits, t_start, t_end = load_window_data(cfg)
    _, fh_centroid = pick_source_gauge(cfg, series, frac_hits)
    geom = np.load(cfg['data']['well_geometry_npz'], allow_pickle=True)
    well_td = float(np.asarray(geom['data'], float).max())
    top_md = float(cfg['mesh']['domain_top_md_ft'])
    log(f"gauges {sorted(series)}; frac-hit centroid MD {fh_centroid:.2f}; "
        f"well TD MD {well_td:.1f}; domain top MD {top_md:.1f}")
    if abs(well_td - top_md) > 1e-6:
        log(f"WARNING: config domain top {top_md} != surveyed well TD {well_td}")

    inputs = {cfg['data']['gauge_md_npz']: core.file_sha256(cfg['data']['gauge_md_npz']),
              cfg['data']['frac_hit_stage1_npz']: core.file_sha256(cfg['data']['frac_hit_stage1_npz']),
              cfg['data']['well_geometry_npz']: core.file_sha256(cfg['data']['well_geometry_npz'])}
    for n in ALL_GAUGES:
        p = cfg['data']['gauge_series_template'].format(n=n)
        inputs[p] = core.file_sha256(p)

    variants = [build_variant(cfg, series, fh_centroid, v, top_md)
                for v in cfg['variants']]

    fits, scores, sims = {}, {}, {}
    for V in variants:
        log(f"=== {V['name']}: {V['label']}")
        log(f"    Dirichlet node MD {V['node_md_ft']:.1f} (requested {V['src_md_ft']:.2f}, "
            f"snap {V['md_snap_residual_ft']:+.2f} ft, idx {V['source_idx']}); "
            f"calib {V['calib_gauges']}; roles " +
            ", ".join(f"g{n}:{V['roles'][n]}" for n in ALL_GAUGES))
        if V['extrap_info']:
            log(f"    extrapolation: {V['extrap_info']['formula']} "
                f"near g{V['extrap_info']['near_gauge']} far g{V['extrap_info']['far_gauge']} "
                f"w={V['extrap_info']['weight']:.5f}; peak {V['s_data'].max():.1f} psi "
                f"(g1 observed peak {series[1]['delta_psi'].max():.1f} psi)")
        _init_worker(V, dt)
        fits[V['name']], scores[V['name']], sims[V['name']] = {}, {}, {}
        with Pool(nproc, initializer=_init_worker, initargs=(V, dt)) as pool:
            fu = fit_uniform(pool, V, cfg['models']['uniform'])
            fits[V['name']]['uniform'] = fu
            log(f"    uniform          D={fu['D']:8.1f} ft^2/s  obj={fu['objective_psi']:8.3f} psi"
                f"{'  [AT GRID EDGE]' if any(fu['at_bound']) else ''}")
            for mname, anchor in (('two_zone_bc', V['node_md_ft']),
                                  ('two_zone_frachit', V['frac_hit_centroid'])):
                mc = cfg['models'][mname]
                f = fit_two_zone(pool, V, mname, anchor, mc['bounds'], search)
                fits[V['name']][mname] = f
                log(f"    {mname:16s} D_near={f['D_near']:8.1f} D_far={f['D_far']:7.1f} "
                    f"s_c={f['s_c_ft']:7.1f} w={f['width_ft']:6.1f}  obj={f['objective_psi']:8.3f} psi"
                    f"{'  [AT BOUND]' if any(f['at_bound']) else ''}")
        for mname, f in fits[V['name']].items():
            anchor = f['anchor_md_ft']
            prof = PROFILE_FN[mname](V['mesh'], anchor, np.asarray(f['params'], float))
            ev = score(V, prof, dt, thr)
            sims[V['name']][mname] = (ev.pop('sim_taxis'), ev.pop('sim'))
            scores[V['name']][mname] = ev

    # ---- continuity check against the established numbers -------------------
    cont = {
        'baseline_uniform_objective_psi': fits['baseline']['uniform']['objective_psi'],
        'baseline_uniform_D': fits['baseline']['uniform']['D'],
        'established_uniform_rmse_psi': 82.33,
        'baseline_two_zone_bc_objective_psi': fits['baseline']['two_zone_bc']['objective_psi'],
        'established_two_zone_rmse_psi': 11.87,
    }
    cont['uniform_reproduced'] = bool(abs(cont['baseline_uniform_objective_psi'] - 82.33) < 0.5)
    cont['two_zone_reproduced'] = bool(cont['baseline_two_zone_bc_objective_psi'] < 12.5)
    log(f"continuity: uniform {cont['baseline_uniform_objective_psi']:.2f} psi "
        f"(established 82.33), two_zone {cont['baseline_two_zone_bc_objective_psi']:.2f} psi "
        f"(established 11.87)")

    # ---- top-boundary sensitivity for the blind gauge-1 prediction ----------
    log("=== top-boundary sensitivity for the blind gauge-1 prediction")
    top_rows = []
    for vname in ('src_g2', 'src_g3', 'extrap_frachit'):
        vspec = next(v for v in cfg['variants'] if v['name'] == vname)
        for top in cfg['mesh']['domain_top_sensitivity_md_ft']:
            Vt = build_variant(cfg, series, fh_centroid, vspec, top)
            for mname in ('uniform', 'two_zone_frachit'):
                f = fits[vname][mname]
                anchor = f['anchor_md_ft']
                prof = PROFILE_FN[mname](Vt['mesh'], anchor, np.asarray(f['params'], float))
                ev = score(Vt, prof, dt, thr)
                ev.pop('sim_taxis'); ev.pop('sim')
                g1 = next(g for g in ev['per_gauge'] if g['gauge'] == 1)
                top_rows.append({'variant': vname, 'model': mname, 'top_md_ft': float(top),
                                 'g1_rmse_psi': g1['rmse_psi'],
                                 'g1_amplitude_ratio': g1['amplitude_ratio'],
                                 'g1_arrival_err_s': g1['arrival_err_s'],
                                 'gauge_mean_rmse_common4': ev['gauge_mean_rmse_common4']})
        r0 = [r for r in top_rows if r['variant'] == vname and r['model'] == 'uniform']
        log(f"    {vname}: uniform g1 RMSE {r0[0]['g1_rmse_psi']:.1f} psi at top "
            f"{r0[0]['top_md_ft']:.0f} -> {r0[-1]['g1_rmse_psi']:.1f} psi at top {r0[-1]['top_md_ft']:.0f}")

    # ---- refit uniform at the extreme top, to bound the sensitivity ---------
    vspec = next(v for v in cfg['variants'] if v['name'] == 'src_g2')
    top_ext = float(cfg['mesh']['domain_top_sensitivity_md_ft'][-1])
    Vext = build_variant(cfg, series, fh_centroid, vspec, top_ext)
    with Pool(nproc, initializer=_init_worker, initargs=(Vext, dt)) as pool:
        f_ext = fit_uniform(pool, Vext, cfg['models']['uniform'])
    _init_worker(Vext, dt)
    prof_ext = profile_uniform(Vext['mesh'], None, np.asarray(f_ext['params'], float))
    ev_ext = score(Vext, prof_ext, dt, thr)
    ev_ext.pop('sim_taxis'); ev_ext.pop('sim')
    g1_ext = next(g for g in ev_ext['per_gauge'] if g['gauge'] == 1)
    top_refit = {'variant': 'src_g2', 'top_md_ft': top_ext, 'D': f_ext['D'],
                 'objective_psi': f_ext['objective_psi'],
                 'g1_rmse_psi': g1_ext['rmse_psi'],
                 'g1_amplitude_ratio': g1_ext['amplitude_ratio'],
                 'D_at_production_top': fits['src_g2']['uniform']['D']}
    log(f"    src_g2 refit at top {top_ext:.0f}: D={f_ext['D']:.1f} "
        f"(production top D={fits['src_g2']['uniform']['D']:.1f}), g1 RMSE {g1_ext['rmse_psi']:.1f} psi")

    # ---- extrapolation-rule diagnostic (pure data, no solver) ---------------
    extrap_diag = []
    for V in variants:
        if not V['extrap_info']:
            continue
        obs = series[1]
        sim_on_obs = np.interp(obs['taxis'], V['s_taxis'], V['s_data'])
        r = sim_on_obs - obs['delta_psi']
        omax = float(np.max(obs['delta_psi']))
        extrap_diag.append({
            'variant': V['name'], 'target_md_ft': V['src_md_ft'],
            'gauge1_md_ft': obs['md_ft'],
            'rmse_vs_gauge1_psi': float(np.sqrt(np.mean(r ** 2))),
            'peak_extrapolated_psi': float(np.max(V['s_data'])),
            'peak_gauge1_psi': omax,
            'peak_ratio': float(np.max(V['s_data']) / omax),
            'arrival_err_s': (core.arrival_time(obs['taxis'], sim_on_obs, thr * omax)
                              - core.arrival_time(obs['taxis'], obs['delta_psi'], thr * omax)),
            'note': ('For extrap_g1md this IS the gauge-1 row: the Dirichlet node sits on '
                     'gauge 1s MD, so the simulated value there is the imposed extrapolation, '
                     'not a model prediction.'),
        })
        log(f"    extrapolation {V['name']} -> MD {V['src_md_ft']:.1f}: "
            f"RMSE vs g1 {extrap_diag[-1]['rmse_vs_gauge1_psi']:.1f} psi, "
            f"peak ratio {extrap_diag[-1]['peak_ratio']:.3f}")

    # ---- comparison csv ----------------------------------------------------
    rows = []
    for V in variants:
        for mname in ('uniform', 'two_zone_bc', 'two_zone_frachit'):
            f = fits[V['name']][mname]
            ev = scores[V['name']][mname]
            base = scores['baseline_far'][mname]
            base_by_g = {g['gauge']: g for g in base['per_gauge']}
            for g in ev['per_gauge']:
                b = base_by_g[g['gauge']]
                rows.append({
                    'variant': V['name'], 'variant_label': V['label'],
                    'model': mname,
                    'dirichlet_md_ft': round(V['node_md_ft'], 1),
                    'driver_gauges': '+'.join(str(x) for x in V['driver_gauges']),
                    'calib_gauges': '+'.join(str(x) for x in V['calib_gauges']),
                    'gauge': g['gauge'], 'gauge_md_ft': g['md_ft'],
                    'role': g['role'], 'is_prediction': g['is_prediction'],
                    'distance_from_bc_ft': round(g['distance_ft'], 1),
                    'rmse_psi': round(g['rmse_psi'], 4),
                    'amplitude_ratio': round(g['amplitude_ratio'], 5),
                    'arrival_err_s': (round(g['arrival_err_s'], 3)
                                      if np.isfinite(g['arrival_err_s']) else ''),
                    'obs_max_psi': round(g['obs_max_psi'], 2),
                    'sim_max_psi': round(g['sim_max_psi'], 2),
                    'bc_peak_psi': round(g['bc_peak_psi'], 2),
                    'max_principle_amp_bound': round(g['max_principle_amp_bound'], 4),
                    'amp_bound_binding': g['amp_bound_binding'],
                    'bias_psi': round(g['bias_psi'], 3),
                    'rmse_psi_baseline_far': round(b['rmse_psi'], 4),
                    'delta_rmse_vs_baseline_far_psi': round(g['rmse_psi'] - b['rmse_psi'], 4),
                    'rmse_ratio_vs_baseline_far': (round(g['rmse_psi'] / b['rmse_psi'], 4)
                                                   if b['rmse_psi'] else ''),
                    'calibrated_D': (round(f['D'], 2) if mname == 'uniform' else ''),
                    'calibrated_D_near': (round(f['D_near'], 2) if mname != 'uniform' else ''),
                    'calibrated_D_far': (round(f['D_far'], 2) if mname != 'uniform' else ''),
                    'calibrated_s_c_ft': (round(f['s_c_ft'], 2) if mname != 'uniform' else ''),
                    'objective_psi': round(f['objective_psi'], 4),
                    'gauge_mean_rmse_common4_psi': round(ev['gauge_mean_rmse_common4'], 4),
                })
    with open(out['comparison_csv'], 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {out['comparison_csv']} ({len(rows)} rows)")

    # ---- arrays ------------------------------------------------------------
    npz = {}
    for V in variants:
        npz[f"bc_taxis__{V['name']}"] = V['s_taxis']
        npz[f"bc_data__{V['name']}"] = V['s_data']
        for mname, (ta, rec) in sims[V['name']].items():
            npz[f"sim_taxis__{V['name']}__{mname}"] = ta
            npz[f"sim__{V['name']}__{mname}"] = rec
        npz[f"uniform_grid__{V['name']}"] = fits[V['name']]['uniform']['grid']
        npz[f"uniform_curve__{V['name']}"] = fits[V['name']]['uniform']['curve']
    for n in ALL_GAUGES:
        npz[f"obs_taxis__g{n}"] = series[n]['taxis']
        npz[f"obs_data__g{n}"] = series[n]['delta_psi']
    npz['gauge_numbers'] = np.array(ALL_GAUGES)
    npz['gauge_md_ft'] = np.array([series[n]['md_ft'] for n in ALL_GAUGES])
    np.savez_compressed(out['arrays_npz'], **npz)
    log(f"wrote {out['arrays_npz']}")

    # ---- figures -----------------------------------------------------------
    make_figures(cfg, variants, series, sims, scores, fits)

    # ---- summary json ------------------------------------------------------
    summary = {
        'study_id': cfg['study_id'],
        'continuity_check': cont,
        'variants': {V['name']: {
            'label': V['label'], 'dirichlet_md_ft': V['node_md_ft'],
            'driver_gauges': V['driver_gauges'],
            'calib_gauges': V['calib_gauges'], 'blind_gauges': V['blind_gauges'],
            'roles': V['roles'], 'bc_peak_psi': float(np.max(V['s_data'])),
            'fits': _jsonable(fits[V['name']]),
            'scores': _jsonable(scores[V['name']]),
        } for V in variants},
        'top_boundary_sensitivity': top_rows,
        'top_boundary_refit': top_refit,
        'extrapolation_diagnostic': extrap_diag,
    }
    with open(out['summary_json'], 'w') as fh:
        json.dump(_jsonable(summary), fh, indent=2)
    log(f"wrote {out['summary_json']}")

    # ---- manifests ---------------------------------------------------------
    prod_paths = [out['comparison_csv'], out['summary_json'], out['arrays_npz'],
                  out['figure_png'], out['figure_bc_png']]
    n_steps = int(np.ceil(variants[0]['t_total'] / dt)) + 1

    def numerics_for(V):
        return {
            'theta': 1.0, 'interface_avg': 'harmonic', 'dt_s': dt, 'adaptive': None,
            'n_steps': int(np.ceil(V['t_total'] / dt)) + 1,
            'domain_md_ft': [float(V['mesh'][0]), float(V['mesh'][-1])],
            'pad_low_ft': V['pad_low_ft'],
            'pad_high_ft': float(V['mesh'][-1] - cfg['window']['md_max_ft']),
            'dx_ft': V['dx_ft'], 'nx': int(len(V['mesh'])), 'barrier': None,
        }

    def source_for(V):
        return {
            'source_md_ft': V['src_md_ft'],
            'source_gauge': V['src_gauge'],
            'driving_series_path': (V['driving_paths'][0] if len(V['driving_paths']) == 1
                                    else V['driving_paths']),
            'application': 'dirichlet_node',
            'source_mesh_idx': V['source_idx'],
            'md_snap_residual_ft': V['md_snap_residual_ft'],
            'driver_gauges': V['driver_gauges'],
            'driving_series_construction': (V['extrap_info'] if V['extrap_info']
                                            else 'observed gauge delta-pressure, baseline = first sample'),
        }

    for V in variants:
        d = out['per_variant_manifest_template'].format(name=V['name'])
        os.makedirs(os.path.dirname(d), exist_ok=True)
        man = make_manifest(
            f"{cfg['study_id']}::{V['name']}", cfg, args.config, cfg_hash, inputs,
            source_for(V), numerics_for(V),
            {'variant': V['name'], 'label': V['label'], 'roles': V['roles'],
             'calib_gauges': V['calib_gauges'],
             'fits': _jsonable(fits[V['name']]),
             'scores': _jsonable(scores[V['name']]),
             'continuity_check': cont if V['name'] == 'baseline' else None},
            output_records(prod_paths))
        with open(d, 'w') as fh:
            json.dump(man, fh, indent=2)
        log(f"wrote {d}")

    man = make_manifest(
        cfg['study_id'], cfg, args.config, cfg_hash, inputs,
        {**source_for(variants[0]),
         'per_variant': {V['name']: source_for(V) for V in variants}},
        {**numerics_for(variants[0]), 'n_steps': n_steps,
         'per_variant': {V['name']: numerics_for(V) for V in variants}},
        _jsonable(summary),
        output_records(prod_paths + [out['per_variant_manifest_template'].format(name=V['name'])
                                     for V in variants]))
    man['wall_time_s'] = (datetime.datetime.now() - t_wall0).total_seconds()
    with open(out['manifest_json'], 'w') as fh:
        json.dump(man, fh, indent=2)
    log(f"wrote {out['manifest_json']}  (wall {man['wall_time_s']:.0f} s)")
    log('done')


# --------------------------------------------------------------------------
def make_figures(cfg, variants, series, sims, scores, fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = cfg['outputs']
    dpi = int(out['figure_dpi'])
    present = {V['name'] for V in variants}
    order = [n for n in ('baseline_far', 'extrap_g1md', 'extrap_frachit',
                         'src_g2', 'src_g3') if n in present]
    colors = {'baseline_far': '#444444', 'extrap_g1md': '#1b9e77',
              'extrap_frachit': '#7570b3', 'src_g2': '#d95f02', 'src_g3': '#e7298a'}
    short = {'baseline_far': 'baseline (g1 @ 16645)',
             'extrap_g1md': '(a) extrap g2,g3 -> 16645',
             'extrap_frachit': "(a') extrap g2,g3 -> 16683",
             'src_g2': '(b) g2 @ 16384', 'src_g3': '(c) g3 @ 16122'}
    Vby = {V['name']: V for V in variants}

    models = ['uniform', 'two_zone_frachit']
    mtitle = {'uniform': 'uniform D (k=1)',
              'two_zone_frachit': 'two_zone D(x), anchored at the frac hits (k=4)'}

    fig, axes = plt.subplots(7, 2, figsize=(11.0, 16.5), sharex=True)
    for r, n in enumerate(ALL_GAUGES):
        for c, mname in enumerate(models):
            ax = axes[r, c]
            ax.plot(series[n]['taxis'], series[n]['delta_psi'], color='k', lw=2.0,
                    zorder=5, label='observed')
            for vn in order:
                V, ev = Vby[vn], scores[vn][mname]
                g = next(x for x in ev['per_gauge'] if x['gauge'] == n)
                ta, rec = sims[vn][mname]
                k = [t['gauge'] for t in V['targets']].index(n)
                ls = '-' if g['is_prediction'] else ':'
                ax.plot(ta, rec[:, k], color=colors[vn], lw=1.3, ls=ls,
                        label=short[vn] if (r == 0 and c == 0) else None)
            if r == 0:
                ax.set_title(mtitle[mname], fontsize=10)
            ax.set_ylabel(f"g{n} (MD {series[n]['md_ft']:.0f})\n$\\Delta P$ [psi]",
                          fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.25, lw=0.4)
            rl = ", ".join(f"{vn.split('_')[0][:3]}:{Vby[vn]['roles'][n][:4]}" for vn in order)
            ax.text(0.01, 0.96, rl, transform=ax.transAxes, fontsize=6.0,
                    va='top', color='#666666')
    for c in range(2):
        axes[-1, c].set_xlabel('time since 11:24:00 [s]', fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.suptitle('D2 - removing the source gauge: observed vs baseline vs four '
                 'boundary-condition substitutions\n'
                 'dotted = the Dirichlet node sits on that gauge (imposed, not a prediction)',
                 fontsize=11, y=1.022)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out['figure_png'], dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    log(f"wrote {out['figure_png']}")

    # BC figure
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ax.plot(series[1]['taxis'], series[1]['delta_psi'], color='k', lw=2.2,
            label='gauge 1 observed (MD 16645) - REMOVED')
    for vn in order:
        V = Vby[vn]
        ax.plot(V['s_taxis'], V['s_data'], color=colors[vn], lw=1.3,
                label=f"{short[vn]}")
    ax.set_xlabel('time since 11:24:00 [s]')
    ax.set_ylabel('$\\Delta P$ [psi]')
    ax.set_title('the driving series each variant imposes')
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25, lw=0.4)

    ax = axes[1]
    for mname, mk in (('uniform', 'o'), ('two_zone_frachit', 's')):
        for vn in order:
            ev = scores[vn][mname]
            gs = [g for g in ev['per_gauge'] if g['is_prediction']]
            ax.plot([g['md_ft'] for g in gs], [g['rmse_psi'] for g in gs],
                    marker=mk, ms=5, lw=1.2, color=colors[vn],
                    ls='-' if mname == 'uniform' else '--',
                    label=f"{short[vn]} | {mname}" if True else None)
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.set_xlabel('gauge MD [ft]')
    ax.set_ylabel('per-gauge RMSE [psi]')
    ax.set_title('per-gauge RMSE (solid = uniform, dashed = two_zone)')
    ax.legend(fontsize=6.2, frameon=False, ncol=2)
    ax.grid(alpha=0.25, lw=0.4, which='both')
    fig.tight_layout()
    fig.savefig(out['figure_bc_png'], dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    log(f"wrote {out['figure_bc_png']}")


if __name__ == '__main__':
    main()
