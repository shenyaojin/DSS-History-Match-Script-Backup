"""D1 -- leave-one-gauge-out BLIND prediction.

For each observation gauge j in 2..7: remove gauge j COMPLETELY, recalibrate on
the remaining five under exactly the R1 criterion, window and numerics, then
forward-model and score gauge j against data that never entered the fit.

This is deliberately NOT the "leave-one-out" already in the R1 manifest. That
one dropped a gauge, recalibrated, and reported only how far the argmin moved
(899-1290 ft^2/s). It never asked whether the refit model can predict the gauge
it dropped. D1 asks exactly that, which is what Reviewer 2 asked for.

Everything is run twice over:
  * two norms   -- absolute (gauge-mean RMSE, psi) and amplitude-normalised.
                   For the uniform model they disagree by 2.4x, and the
                   disagreement is itself a reported result.
  * two model families -- uniform D (the published baseline) and the two_zone
                   D(x) family that won the R2 profile inversion. The
                   comparison "blind error falls from X to Y once the model is
                   well specified" is the point of the exercise.

    python scripts/manuscript_well_leakage/rev2/d1_loo_blind.py \
        --config configs/rev2/d1_loo_blind.json

Uses the verified kernel r1_calibration_core.solve_forward unchanged (proven
bit-equivalent to fibeRIS, ~1800x faster). Nothing under baseline_calibration/
is modified.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', '..'))
BASELINE = os.path.join(REPO, 'scripts', 'manuscript_well_leakage',
                        'baseline_calibration')
sys.path.insert(0, BASELINE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import r1_calibration_core as core  # noqa: E402
from r1_run_calibration import (load_config, load_window_data,  # noqa: E402
                                pick_source_gauge)

_G = {}


def log(m):
    print(f"[d1] {m}", flush=True)


# ---------------------------------------------------------------------------
# setup shared by main process and every worker
# ---------------------------------------------------------------------------

def build_setup(cfg):
    series, gnums, gmds, frac_hits, t_start, t_end = load_window_data(cfg)
    src_gauge, fh_centroid = pick_source_gauge(cfg, series, frac_hits)
    src = series[src_gauge]

    m = cfg['mesh']
    dx = float(m['dx_ft'])
    pad_lo = float(m['domain_pad_low_md_ft'])
    pad_hi = float(m['domain_pad_high_md_ft'])
    win_lo = float(cfg['window']['md_min_ft'])
    win_hi = float(cfg['window']['md_max_ft'])
    mesh = np.arange(win_lo - pad_lo, win_hi + pad_hi + dx / 2.0, dx)
    source_idx = int(np.argmin(np.abs(mesh - src['md_ft'])))

    tgt_nums = [n for n in sorted(series) if n != src_gauge]
    targets = [{'gauge': int(n), 'md_ft': float(series[n]['md_ft']),
                'distance_ft': float(abs(series[n]['md_ft'] - src['md_ft'])),
                'idx': int(np.argmin(np.abs(mesh - series[n]['md_ft']))),
                'taxis': series[n]['taxis'],
                'data': series[n]['delta_psi']} for n in tgt_nums]
    obs_max = np.array([float(np.max(t['data'])) for t in targets])
    return dict(series=series, src_gauge=int(src_gauge), src=src, mesh=mesh,
                source_idx=source_idx, targets=targets, obs_max=obs_max,
                t_total=float(src['taxis'][-1]), dt=float(cfg['solver']['dt_s']),
                fh_centroid=float(fh_centroid), gauge_nums=tgt_nums,
                src_md=float(src['md_ft']))


def _init_worker(cfg):
    _G['cfg'] = cfg
    _G['S'] = build_setup(cfg)


def _profile(S, family, p):
    if family == 'uniform':
        return core.build_uniform_profile(S['mesh'], float(p[0]))
    return core.PROFILE_FAMILIES[family]['fn'](S['mesh'], S['source_idx'],
                                               np.asarray(p, dtype=float))


def per_gauge_misfit(S, prof):
    """Per-gauge MSE (psi^2) and per-gauge normalised MSE, one forward solve.

    Identical arithmetic to core.misfit_for_profile, only without the final
    aggregation, so any gauge subset and either norm can be scored afterwards
    from the same solve.
    """
    if not np.all(np.isfinite(prof)) or np.any(prof <= 0):
        n = len(S['targets'])
        return np.full(n, np.inf), np.full(n, np.inf)
    taxis, rec = core.solve_forward(
        S['mesh'], prof, S['dt'], S['t_total'], S['src']['taxis'],
        S['src']['delta_psi'], S['source_idx'],
        record_idx=[t['idx'] for t in S['targets']])
    if not np.all(np.isfinite(rec)):
        n = len(S['targets'])
        return np.full(n, np.inf), np.full(n, np.inf)
    mse = np.empty(len(S['targets']))
    for k, tgt in enumerate(S['targets']):
        r = np.interp(tgt['taxis'], taxis, rec[:, k]) - tgt['data']
        mse[k] = float(np.mean(r ** 2))
    return mse, mse / S['obs_max'] ** 2


def _eval_point(args):
    family, p = args
    S = _G['S']
    return per_gauge_misfit(S, _profile(S, family, p))


def subset_criterion(mse, norm2, cols, norm):
    v = norm2 if norm == 'normalised' else mse
    return float(np.sqrt(np.mean(np.asarray(v)[..., cols], axis=-1)))


def crit_curve(mat, cols):
    return np.sqrt(np.mean(mat[:, cols], axis=1))


# ---------------------------------------------------------------------------
# full scoring of one accepted model (all six gauges, full diagnostics)
# ---------------------------------------------------------------------------

def full_score(S, family, p, cfg):
    prof = _profile(S, family, p)
    taxis, rec = core.solve_forward(
        S['mesh'], prof, S['dt'], S['t_total'], S['src']['taxis'],
        S['src']['delta_psi'], S['source_idx'],
        record_idx=[t['idx'] for t in S['targets']])
    mcfg = cfg['metrics']['arrival_time']
    rows, sims = [], {}
    for k, tgt in enumerate(S['targets']):
        sim_on_obs = np.interp(tgt['taxis'], taxis, rec[:, k])
        resid = sim_on_obs - tgt['data']
        obs_max = float(np.max(tgt['data']))
        arr = core.arrival_robustness(taxis, rec[:, k], tgt['taxis'],
                                      tgt['data'],
                                      mcfg['absolute_thresholds_psi'],
                                      mcfg['relative_fraction'])
        rows.append({
            'gauge': tgt['gauge'], 'md_ft': tgt['md_ft'],
            'distance_ft': tgt['distance_ft'],
            'rmse_psi': float(np.sqrt(np.mean(resid ** 2))),
            'rmse_normalised': float(np.sqrt(np.mean(resid ** 2)) / obs_max),
            'bias_psi': float(np.mean(resid)),
            'obs_max_psi': obs_max,
            'sim_max_psi': float(np.max(sim_on_obs)),
            'amplitude_ratio': float(np.max(sim_on_obs) / obs_max),
            'arrival_err_relative_s': float(arr['relative']),
            'arrival_err_abs10_s': float(arr['abs_10psi']),
            'arrival_err_abs25_s': float(arr['abs_25psi']),
        })
        sims[tgt['gauge']] = sim_on_obs
    return rows, sims, prof, int(len(taxis) - 1)


# ---------------------------------------------------------------------------
# uniform family
# ---------------------------------------------------------------------------

def run_uniform(cfg, S, pool, subsets):
    fam = cfg['families']['uniform']
    g = fam['grid']
    coarse = np.logspace(np.log10(g['min']), np.log10(g['max']), g['n_points'])
    floor_cfg = fam['single_gauge_floor_grid']
    floor_grid = np.logspace(np.log10(floor_cfg['min']),
                             np.log10(floor_cfg['max']), floor_cfg['n_points'])

    store = {}   # D value -> (mse vector, norm2 vector)

    def evaluate(values):
        todo = [v for v in values if float(v) not in store]
        if not todo:
            return
        res = pool.map(_eval_point, [('uniform', [v]) for v in todo],
                       chunksize=2)
        for v, (m, n2) in zip(todo, res):
            store[float(v)] = (m, n2)

    t0 = time.time()
    evaluate(np.concatenate([coarse, floor_grid]))
    log(f"uniform: {len(store)} coarse+floor solves in {time.time()-t0:.1f} s")

    def curve_on(values, cols, norm):
        mat = np.array([store[float(v)][1 if norm == 'normalised' else 0]
                        for v in values])
        return crit_curve(mat, cols)

    ref = fam['refine']
    fits = {}
    for norm in ('absolute', 'normalised'):
        for label, cols in subsets.items():
            c = curve_on(coarse, cols, norm)
            best_coarse = float(coarse[int(np.argmin(c))])
            grid = coarse
            if ref['enabled']:
                extra = np.logspace(np.log10(best_coarse) - ref['half_width_decades'],
                                    np.log10(best_coarse) + ref['half_width_decades'],
                                    ref['n_points'])
                evaluate(extra)
                grid = np.unique(np.concatenate([coarse, extra]))
            c = curve_on(grid, cols, norm)
            i = int(np.argmin(c))
            fits[(norm, label)] = {
                'params': [float(grid[i])], 'criterion': float(c[i]),
                'grid_lo': float(grid[0]), 'grid_hi': float(grid[-1]),
                'at_grid_edge': bool(i == 0 or i == len(grid) - 1),
                'best_coarse': best_coarse, 'n_grid': int(len(grid)),
            }
    log(f"uniform: {len(store)} total solves in {time.time()-t0:.1f} s")

    # per-gauge floor: best achievable by a uniform D fitting that gauge alone
    floors = {}
    all_vals = np.array(sorted(store))
    mat = np.array([store[float(v)][0] for v in all_vals])
    for k, tgt in enumerate(S['targets']):
        i = int(np.argmin(mat[:, k]))
        floors[tgt['gauge']] = {'D': float(all_vals[i]),
                                'rmse_psi': float(np.sqrt(mat[i, k])),
                                'at_grid_edge': bool(i == 0 or i == len(all_vals) - 1)}
    return fits, floors, len(store)


# ---------------------------------------------------------------------------
# two_zone family
# ---------------------------------------------------------------------------

def _fit_job(args):
    """One (subset, norm) two_zone inversion, run entirely inside a worker."""
    label, cols, norm, start_pts, bounds, n_local, frac, seed, maxiter = args
    S = _G['S']
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    cols = list(cols)

    def f(p):
        p = np.clip(np.asarray(p, float), lo, hi)
        mse, n2 = per_gauge_misfit(S, _profile(S, 'two_zone', p))
        v = n2 if norm == 'normalised' else mse
        v = np.asarray(v)[cols]
        if not np.all(np.isfinite(v)):
            return np.inf
        return float(np.sqrt(np.mean(v)))

    best_p = np.asarray(start_pts[0], float)
    best_v = f(best_p)
    for sp in start_pts[1:]:
        v = f(sp)
        if v < best_v:
            best_p, best_v = np.asarray(sp, float), v

    # local Latin-hypercube round around the incumbent, then Nelder-Mead polish
    half = (hi - lo) * frac / 2.0
    l2 = np.maximum(lo, best_p - half)
    h2 = np.minimum(hi, best_p + half)
    pts = l2 + qmc.LatinHypercube(d=len(bounds), seed=seed).random(n_local) * (h2 - l2)
    for q in pts:
        v = f(q)
        if v < best_v:
            best_p, best_v = q.copy(), v

    for start in (best_p.copy(), np.asarray(start_pts[0], float)):
        res = minimize(f, start, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-3,
                                'fatol': 1e-4, 'disp': False})
        if float(res.fun) < best_v:
            best_p, best_v = np.clip(np.asarray(res.x, float), lo, hi), float(res.fun)

    return (norm, label, {
        'params': [float(x) for x in best_p], 'criterion': float(best_v),
        'at_bound': [bool(abs(x - b[0]) < 1e-6 or abs(x - b[1]) < 1e-6)
                     for x, b in zip(best_p, bounds)],
    })


def run_two_zone(cfg, S, pool, subsets):
    fam = cfg['families']['two_zone']
    sc = cfg['search']
    bounds = [tuple(b) for b in fam['bounds']]
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)

    pts = lo + qmc.LatinHypercube(d=len(bounds), seed=sc['seed']).random(
        sc['n_global_lhs']) * (hi - lo)
    pts = np.vstack([np.asarray(fam['warm_start'], float)[None, :], pts])
    t0 = time.time()
    res = pool.map(_eval_point, [('two_zone', p) for p in pts], chunksize=4)
    mse = np.array([r[0] for r in res])
    nrm = np.array([r[1] for r in res])
    log(f"two_zone: global LHS {len(pts)} solves in {time.time()-t0:.1f} s")

    jobs = []
    for jn, norm in enumerate(('absolute', 'normalised')):
        mat = nrm if norm == 'normalised' else mse
        for si, (label, cols) in enumerate(subsets.items()):
            c = crit_curve(mat, cols)
            order = np.argsort(c)[:3]
            starts = [pts[i] for i in order] + [np.asarray(fam['warm_start'], float)]
            jobs.append((label, tuple(cols), norm, [list(map(float, s)) for s in starts],
                         bounds, sc['n_local_lhs'], sc['local_lhs_frac'],
                         int(sc['seed']) + 100 * jn + si,
                         int(sc['nelder_mead_maxiter'])))

    t0 = time.time()
    fits = {}
    for norm, label, r in pool.imap_unordered(_fit_job, jobs):
        fits[(norm, label)] = r
        log(f"  two_zone {norm:11s} subset {label:9s} crit={r['criterion']:.4f} "
            f"params={[round(x, 3) for x in r['params']]}"
            f"{'  [AT BOUND]' if any(r['at_bound']) else ''}")
    log(f"two_zone: {len(jobs)} subset inversions in {time.time()-t0:.1f} s")
    return fits, int(len(pts))


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def _rel(p):
    p = os.path.abspath(p)
    return os.path.relpath(p, REPO) if p.startswith(REPO) else p


def code_hashes():
    out = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if not f or not f.endswith('.py'):
            continue
        f = os.path.abspath(f)
        if not f.startswith(REPO) or not os.path.exists(f):
            continue
        out[_rel(f)] = core.file_sha256(f)
    return dict(sorted(out.items()))


def write_manifest(path, cfg, cfg_sha, cfg_path, S, results, out_paths,
                   n_steps, family):
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import rev2_manifest  # noqa: F401
        has_shared = hasattr(rev2_manifest, 'write_manifest')
    except Exception:
        has_shared = False
    import matplotlib
    import scipy
    import fiberis

    data_files = [cfg['data']['gauge_md_npz'], cfg['data']['frac_hit_stage1_npz']]
    data_files += [cfg['data']['gauge_series_template'].format(n=n)
                   for n in sorted(S['series'])]
    man = {
        'study_id': f"{cfg['study_id']}__{family}",
        'run_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_resolved': cfg,
        'config_sha256': cfg_sha,
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'matplotlib': matplotlib.__version__,
            'fiberis_path': os.path.dirname(os.path.abspath(fiberis.__file__)),
            'cwd': os.getcwd(),
            'code_sha256': code_hashes(),
            'input_data_sha256': {_rel(p): core.file_sha256(p)
                                  for p in data_files},
            'shared_rev2_manifest_module_available': bool(has_shared),
            'config_path': _rel(cfg_path),
            'note': ('bakken_mariner/.git is empty; code identity is pinned by '
                     'sha256 of every .py loaded at write time.'),
        },
        'source_protocol': {
            'source_md_ft': S['src_md'],
            'source_gauge': S['src_gauge'],
            'driving_series_path': _rel(
                cfg['data']['gauge_series_template'].format(n=S['src_gauge'])),
            'application': 'dirichlet_node',
            'source_mesh_idx': S['source_idx'],
            'md_snap_residual_ft': float(abs(S['mesh'][S['source_idx']] - S['src_md'])),
        },
        'numerics': {
            'theta': 1.0,
            'interface_avg': 'harmonic',
            'dt_s': S['dt'],
            'adaptive': None,
            'n_steps': int(n_steps),
            'domain_md_ft': [float(S['mesh'][0]), float(S['mesh'][-1])],
            'pad_low_ft': float(cfg['mesh']['domain_pad_low_md_ft']),
            'pad_high_ft': float(cfg['mesh']['domain_pad_high_md_ft']),
            'dx_ft': float(cfg['mesh']['dx_ft']),
            'nx': int(len(S['mesh'])),
            'barrier': None,
        },
        'results': results,
        'outputs': [],
    }
    for p in out_paths:
        if os.path.exists(p):
            man['outputs'].append({'path': _rel(p),
                                   'bytes': int(os.path.getsize(p)),
                                   'sha256': core.file_sha256(p)})
    with open(path, 'w') as fh:
        json.dump(man, fh, indent=2, default=float)
    return man


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _guard(paths):
    exist = [p for p in paths if os.path.exists(p)]
    if exist:
        raise SystemExit("refusing to overwrite existing outputs (house rule 2): "
                         + ", ".join(exist))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg, cfg_sha = load_config(args.config)
    out = cfg['outputs']
    os.makedirs(out['dir'], exist_ok=True)
    managed = [out[k] for k in ('summary_csv', 'calibration_csv', 'arrays_npz',
                                'manifest_uniform', 'manifest_two_zone',
                                'fig_blind_vs_incal', 'fig_overlay_absolute',
                                'fig_overlay_normalised', 'fig_refit_shift')]
    _guard(managed)
    log(f"config {args.config} sha256={cfg_sha[:16]}")

    S = build_setup(cfg)
    gnums = [t['gauge'] for t in S['targets']]
    log(f"source gauge {S['src_gauge']} MD {S['src_md']:.0f} (mesh idx "
        f"{S['source_idx']}, snap residual "
        f"{abs(S['mesh'][S['source_idx']]-S['src_md']):.3g} ft); targets {gnums}; "
        f"domain MD [{S['mesh'][0]:.0f}, {S['mesh'][-1]:.0f}] nx={len(S['mesh'])}")

    subsets = {'all': list(range(len(gnums)))}
    for k, gn in enumerate(gnums):
        subsets[f'drop_g{gn}'] = [c for c in range(len(gnums)) if c != k]

    nproc = int(cfg['search']['processes'])
    with Pool(nproc, initializer=_init_worker, initargs=(cfg,)) as pool:
        uni_fits, floors, n_uni = run_uniform(cfg, S, pool, subsets)
        tz_fits, n_tz = run_two_zone(cfg, S, pool, subsets)

    # ---- score every accepted model -------------------------------------
    scored, n_steps = {}, None
    for family, fits in (('uniform', uni_fits), ('two_zone', tz_fits)):
        for (norm, label), r in fits.items():
            rows, sims, prof, nst = full_score(S, family, r['params'], cfg)
            n_steps = nst
            scored[(family, norm, label)] = {'rows': rows, 'sims': sims,
                                             'fit': r, 'profile': prof}

    acc = cfg['acceptance']
    amp_lo, amp_hi = acc['amplitude_ratio_band']
    max_nrmse = float(acc['max_normalised_rmse'])

    summary, series_store = [], {}
    for family in ('uniform', 'two_zone'):
        for norm in ('absolute', 'normalised'):
            allfit = scored[(family, norm, 'all')]
            for k, gn in enumerate(gnums):
                label = f'drop_g{gn}'
                blind = scored[(family, norm, label)]
                b = blind['rows'][k]
                i = allfit['rows'][k]
                pa = np.array(allfit['fit']['params'], float)
                pb = np.array(blind['fit']['params'], float)
                if family == 'uniform':
                    shift = f"{100.0*(pb[0]-pa[0])/pa[0]:.1f}"
                    pa_s = f"{pa[0]:.1f}"
                    pb_s = f"{pb[0]:.1f}"
                else:
                    shift = ";".join(f"{100.0*(10**y-10**x)/10**x:.1f}"
                                     for x, y in zip(pa, pb))
                    pa_s = ";".join(f"{10**x:.4g}" for x in pa)
                    pb_s = ";".join(f"{10**x:.4g}" for x in pb)
                usable = bool(amp_lo <= b['amplitude_ratio'] <= amp_hi
                              and b['rmse_normalised'] <= max_nrmse)
                fl = floors[gn]
                summary.append({
                    'model': family, 'norm': norm, 'held_out_gauge': gn,
                    'md_ft': b['md_ft'], 'distance_ft': b['distance_ft'],
                    'obs_max_psi': round(b['obs_max_psi'], 4),
                    'params_all_gauge': pa_s, 'params_refit_5gauge': pb_s,
                    'param_shift_pct': shift,
                    'incal_rmse_psi': round(i['rmse_psi'], 4),
                    'blind_rmse_psi': round(b['rmse_psi'], 4),
                    'blind_over_incal': round(b['rmse_psi'] / i['rmse_psi'], 4),
                    'incal_rmse_norm': round(i['rmse_normalised'], 5),
                    'blind_rmse_norm': round(b['rmse_normalised'], 5),
                    'incal_amp_ratio': round(i['amplitude_ratio'], 4),
                    'blind_amp_ratio': round(b['amplitude_ratio'], 4),
                    'incal_arr_err_rel_s': round(i['arrival_err_relative_s'], 3),
                    'blind_arr_err_rel_s': round(b['arrival_err_relative_s'], 3),
                    'blind_arr_err_abs10_s': round(b['arrival_err_abs10_s'], 3),
                    'blind_arr_err_abs25_s': round(b['arrival_err_abs25_s'], 3),
                    'blind_bias_psi': round(b['bias_psi'], 4),
                    'single_gauge_floor_D': round(fl['D'], 2),
                    'single_gauge_floor_rmse_psi': round(fl['rmse_psi'], 4),
                    'blind_over_floor': round(b['rmse_psi'] / fl['rmse_psi'], 4),
                    'usable_blind_prediction': usable,
                })
                series_store[f'{family}_{norm}_blind_g{gn}'] = blind['sims'][gn]
                series_store[f'{family}_{norm}_incal_g{gn}'] = allfit['sims'][gn]

    # ---- aggregate leave-one-out blind error -----------------------------
    # The single number the argument turns on: pool the six blind predictions
    # the same way the calibration pools its gauges, so "blind error falls from
    # X to Y" is stated in the same norm the calibration minimises.
    aggregate = {}
    for family in ('uniform', 'two_zone'):
        for norm in ('absolute', 'normalised'):
            rs = [r for r in summary if r['model'] == family and r['norm'] == norm]
            b = np.array([r['blind_rmse_psi'] for r in rs], float)
            i = np.array([r['incal_rmse_psi'] for r in rs], float)
            bn = np.array([r['blind_rmse_norm'] for r in rs], float)
            inn = np.array([r['incal_rmse_norm'] for r in rs], float)
            aggregate[f'{family}|{norm}'] = {
                'loo_blind_rmse_gaugemean_psi': float(np.sqrt(np.mean(b ** 2))),
                'incal_rmse_gaugemean_psi': float(np.sqrt(np.mean(i ** 2))),
                'loo_blind_rmse_normalised': float(np.sqrt(np.mean(bn ** 2))),
                'incal_rmse_normalised': float(np.sqrt(np.mean(inn ** 2))),
                'worst_gauge': int(rs[int(np.argmax(b))]['held_out_gauge']),
                'worst_blind_rmse_psi': float(b.max()),
                'n_usable_blind': int(sum(r['usable_blind_prediction'] for r in rs)),
                'first_failing_distance_ft': (
                    min((r['distance_ft'] for r in rs
                         if not r['usable_blind_prediction']), default=None)),
            }
            a = aggregate[f'{family}|{norm}']
            log(f"AGGREGATE {family:9s} {norm:11s}: LOO blind gauge-mean RMSE "
                f"{a['loo_blind_rmse_gaugemean_psi']:8.3f} psi  "
                f"(in-calibration {a['incal_rmse_gaugemean_psi']:8.3f} psi); "
                f"normalised {a['loo_blind_rmse_normalised']:.4f} vs "
                f"{a['incal_rmse_normalised']:.4f}; "
                f"{a['n_usable_blind']}/6 gauges usable")

    cols = list(summary[0].keys())
    with open(out['summary_csv'], 'w') as fh:
        fh.write(",".join(cols) + "\n")
        for r in summary:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")
    log(f"wrote {out['summary_csv']} ({len(summary)} rows)")

    calib = []
    for family, fits in (('uniform', uni_fits), ('two_zone', tz_fits)):
        for (norm, label), r in sorted(fits.items()):
            calib.append({
                'model': family, 'norm': norm, 'subset': label,
                'params': ";".join(f"{x:.10g}" for x in r['params']),
                'params_physical': (f"{r['params'][0]:.4f}" if family == 'uniform'
                                    else ";".join(f"{10**x:.5g}" for x in r['params'])),
                'criterion_value': f"{r['criterion']:.6g}",
                'at_bound_or_grid_edge': str(r.get('at_grid_edge',
                                                   any(r.get('at_bound', [False])))),
            })
    ccols = list(calib[0].keys())
    with open(out['calibration_csv'], 'w') as fh:
        fh.write(",".join(ccols) + "\n")
        for r in calib:
            fh.write(",".join(str(r[c]) for c in ccols) + "\n")
    log(f"wrote {out['calibration_csv']}")

    npz = {f'obs_g{t["gauge"]}': t['data'] for t in S['targets']}
    npz.update({f'taxis_g{t["gauge"]}': t['taxis'] for t in S['targets']})
    npz.update(series_store)
    npz['mesh'] = S['mesh']
    for family in ('uniform', 'two_zone'):
        for norm in ('absolute', 'normalised'):
            npz[f'profile_{family}_{norm}_all'] = scored[(family, norm, 'all')]['profile']
    np.savez_compressed(out['arrays_npz'], **npz)
    log(f"wrote {out['arrays_npz']}")

    make_figures(cfg, S, gnums, summary, scored, floors, out)

    # ---- manifests -------------------------------------------------------
    shared_out = [out['summary_csv'], out['calibration_csv'], out['arrays_npz'],
                  out['fig_blind_vs_incal'], out['fig_overlay_absolute'],
                  out['fig_overlay_normalised'], out['fig_refit_shift']]
    for family, fits, mpath, nsolve in (
            ('uniform', uni_fits, out['manifest_uniform'], n_uni),
            ('two_zone', tz_fits, out['manifest_two_zone'], n_tz)):
        res = {
            'model_family': family,
            'n_forward_solves_in_search': int(nsolve),
            'target_gauges': gnums,
            'target_distance_ft': [t['distance_ft'] for t in S['targets']],
            'observed_peak_psi': {str(t['gauge']): float(np.max(t['data']))
                                  for t in S['targets']},
            'calibrations': {f'{norm}|{label}': fits[(norm, label)]
                             for (norm, label) in fits},
            'all_gauge_fit_per_gauge': {
                norm: scored[(family, norm, 'all')]['rows']
                for norm in ('absolute', 'normalised')},
            'blind_predictions': [r for r in summary if r['model'] == family],
            'aggregate_leave_one_out': {k: v for k, v in aggregate.items()
                                        if k.startswith(family + '|')},
            'aggregate_leave_one_out_other_family': {
                k: v for k, v in aggregate.items()
                if not k.startswith(family + '|')},
            'single_gauge_uniform_floor': floors,
            'acceptance_rule': acc,
        }
        write_manifest(mpath, cfg, cfg_sha, args.config, S, res,
                       shared_out + [mpath], n_steps, family)
        log(f"wrote {mpath}")

    # ---- headline --------------------------------------------------------
    for family in ('uniform', 'two_zone'):
        for norm in ('absolute', 'normalised'):
            bad = [r for r in summary if r['model'] == family and r['norm'] == norm
                   and not r['usable_blind_prediction']]
            d = min((r['distance_ft'] for r in bad), default=None)
            log(f"HEADLINE {family:9s} {norm:11s}: blind prediction fails from "
                + (f"{d:.0f} ft onward" if d is not None else "nowhere in range"))


def make_figures(cfg, S, gnums, summary, scored, floors, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dpi = int(out['figure_dpi'])
    dist = np.array([t['distance_ft'] for t in S['targets']])
    C = {'uniform': '#1f77b4', 'two_zone': '#d62728'}

    def get(family, norm, key):
        d = {r['held_out_gauge']: r[key] for r in summary
             if r['model'] == family and r['norm'] == norm}
        return np.array([d[g] for g in gnums], float)

    # ---- figure 1: blind vs in-calibration -------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.6))
    for row, norm in enumerate(('absolute', 'normalised')):
        for family in ('uniform', 'two_zone'):
            for key, a, ylab, logy in (
                    ('rmse_psi', ax[row, 0], 'RMSE at held-out gauge (psi)', True),
                    ('rmse_norm', ax[row, 1], 'RMSE / observed peak', True),
                    ('amp_ratio', ax[row, 2], 'amplitude ratio sim/obs', True)):
                a.plot(dist, get(family, norm, f'incal_{key}'), 'o--',
                       color=C[family], alpha=0.45, mfc='none',
                       label=f'{family}, in calibration')
                a.plot(dist, get(family, norm, f'blind_{key}'), 's-',
                       color=C[family], label=f'{family}, BLIND')
                a.set_xlabel('distance from source gauge 1 (ft)')
                a.set_ylabel(ylab)
                if logy:
                    a.set_yscale('log')
        fl = np.array([floors[g]['rmse_psi'] for g in gnums])
        ax[row, 0].plot(dist, fl, ':', color='0.35',
                        label='floor: uniform D fit to that gauge alone')
        ax[row, 1].axhline(cfg['acceptance']['max_normalised_rmse'], color='0.35',
                           ls=':', label='acceptance threshold 0.25')
        for lim in cfg['acceptance']['amplitude_ratio_band']:
            ax[row, 2].axhline(lim, color='0.35', ls=':')
        ax[row, 2].axhline(1.0, color='k', lw=0.6)
        letters = 'abcdef'[3 * row:3 * row + 3]
        ax[row, 0].set_title(f'({letters[0]}) {norm} norm: absolute error')
        ax[row, 1].set_title(f'({letters[1]}) {norm} norm: relative error')
        ax[row, 2].set_title(f'({letters[2]}) {norm} norm: amplitude')
        for a in ax[row]:
            a.grid(alpha=0.25)
            a.legend(fontsize=7, loc='best')
    for a in ax.ravel():
        for g, d in zip(gnums, dist):
            a.annotate(f'g{g}', (d, a.get_ylim()[0]), fontsize=6,
                       color='0.4', ha='center', va='bottom')
    fig.suptitle('D1  leave-one-gauge-out BLIND prediction: gauge j removed from the '
                 'calibration entirely, model refit on the other five, then j predicted',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out['fig_blind_vs_incal'], dpi=dpi)
    plt.close(fig)
    log(f"wrote {out['fig_blind_vs_incal']}")

    # ---- figures 2 & 3: per-gauge overlays -------------------------------
    for norm, path in (('absolute', out['fig_overlay_absolute']),
                       ('normalised', out['fig_overlay_normalised'])):
        fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        for k, (g, a) in enumerate(zip(gnums, ax.ravel())):
            tgt = S['targets'][k]
            a.plot(tgt['taxis'], tgt['data'], 'k-', lw=2, label='observed')
            for family in ('uniform', 'two_zone'):
                a.plot(tgt['taxis'], scored[(family, norm, 'all')]['sims'][g],
                       '--', color=C[family], alpha=0.6,
                       label=f'{family}, in calibration')
                a.plot(tgt['taxis'],
                       scored[(family, norm, f'drop_g{g}')]['sims'][g],
                       '-', color=C[family], label=f'{family}, BLIND')
            a.set_title(f'gauge {g}   MD {tgt["md_ft"]:.0f} ft   '
                        f'{tgt["distance_ft"]:.0f} ft from source', fontsize=9)
            txt = []
            for family in ('uniform', 'two_zone'):
                r = [s for s in summary if s['model'] == family
                     and s['norm'] == norm and s['held_out_gauge'] == g][0]
                txt.append(f"{family}: blind RMSE {r['blind_rmse_psi']:.1f} psi "
                           f"({100*r['blind_rmse_norm']:.0f}% of peak), "
                           f"amp {r['blind_amp_ratio']:.2f}")
            a.text(0.02, 0.97, "\n".join(txt), transform=a.transAxes,
                   fontsize=7, va='top', ha='left',
                   bbox=dict(fc='white', ec='0.7', alpha=0.85, pad=2.5))
            a.grid(alpha=0.25)
            a.set_xlabel('time since window start (s)')
            a.set_ylabel('delta pressure (psi)')
            if k == 0:
                a.legend(fontsize=7, loc='lower right')
        fig.suptitle(f'D1 per-gauge overlay -- {norm} norm. Dashed = gauge was in the '
                     'calibration; solid = gauge was withheld and blind-predicted.',
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        log(f"wrote {path}")

    # ---- figure 4: refit parameter shift ---------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(len(gnums))
    for norm, mk in (('absolute', 'o-'), ('normalised', 's--')):
        du = np.array([float(r['params_refit_5gauge']) for r in summary
                       if r['model'] == 'uniform' and r['norm'] == norm])
        dall = float([r['params_all_gauge'] for r in summary
                      if r['model'] == 'uniform' and r['norm'] == norm][0])
        ax[0].plot(x, du, mk, label=f'{norm} norm refit')
        ax[0].axhline(dall, ls=':', color='0.4')
        ax[0].annotate(f'all-gauge {dall:.0f}', (0, dall), fontsize=7, color='0.3')
    ax[0].set_yscale('log')
    ax[0].set_ylabel('refit uniform D (ft$^2$/s)')
    ax[0].set_title('(a) uniform: refit D with gauge j removed')

    for pi, (pname, a) in enumerate(zip(['D_near', 'D_far', 's_c'], ax[1:])):
        for norm, mk in (('absolute', 'o-'), ('normalised', 's--')):
            vals = np.array([float(r['params_refit_5gauge'].split(';')[pi])
                             for r in summary if r['model'] == 'two_zone'
                             and r['norm'] == norm])
            allv = float([r['params_all_gauge'] for r in summary
                          if r['model'] == 'two_zone' and r['norm'] == norm
                          ][0].split(';')[pi])
            a.plot(x, vals, mk, label=f'{norm} norm refit')
            a.axhline(allv, ls=':', color='0.4')
        a.set_yscale('log')
        a.set_ylabel(f'two_zone {pname}' + (' (ft)' if pname == 's_c' else ' (ft$^2$/s)'))
        a.set_title(f'({"bcd"[pi]}) two_zone: refit {pname}')
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels([f'drop g{g}' for g in gnums], fontsize=8)
        a.grid(alpha=0.25)
        a.legend(fontsize=7)
    fig.suptitle('D1  how far the calibrated parameters move when one gauge is removed',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out['fig_refit_shift'], dpi=dpi)
    plt.close(fig)
    log(f"wrote {out['fig_refit_shift']}")


if __name__ == '__main__':
    main()
