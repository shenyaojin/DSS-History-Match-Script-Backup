"""R2 - invert the diffusivity PROFILE D(x), not a single scalar.

R1 established two things this study follows up on:
  * no single D describes the window (each gauge alone requires D spanning 50x),
  * the 104r triangular family is too stiff - freeing its taper ratio drives the
    misfit monotonically to the grid edge with no turning point.

So here the data choose the SHAPE of the decay. Every family is parameterised in
terms of distance from the source, so the parameters are physical and invariant
to mesh and padding choices.

    python scripts/manuscript_well_leakage/baseline_calibration/r2_profile_inversion.py \
        --config configs/r2_diffusivity_profile.json
"""

import argparse
import datetime
import json
import os
import platform
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_calibration_core as core  # noqa: E402
from r1_run_calibration import load_config, load_window_data, pick_source_gauge  # noqa: E402

_G = {}


def log(m):
    print(f"[r2] {m}", flush=True)


def _setup(cfg, source_mode):
    """Build mesh/source/targets for one source-placement mode."""
    series, gnums, gmds, frac_hits, _, _ = load_window_data(cfg)
    src_gauge, fh_centroid = pick_source_gauge(cfg, series, frac_hits)
    m = cfg['mesh']
    pad_lo = float(m['domain_pad_low_md_ft'])
    dx = float(m['dx_ft'])
    win_lo, win_hi = cfg['window']['md_min_ft'], cfg['window']['md_max_ft']
    src = series[src_gauge]

    if source_mode == 'gauge':
        src_md = src['md_ft']
        hi = win_hi
        tgt_gauges = [n for n in sorted(series) if n != src_gauge]
    elif source_mode == 'frac_centroid':
        # Prescribe pressure at the stage-1 frac-hit centroid instead, driven by
        # the same gauge series. This turns the source gauge into an independent
        # target and tests whether the near-source D is an artifact of pinning
        # the boundary condition 38 ft away from the actual injection.
        src_md = float(fh_centroid)
        hi = max(win_hi, src_md + float(cfg['source_variants']['pad_above_source_ft']))
        tgt_gauges = sorted(series)
    else:
        raise ValueError(source_mode)

    mesh = np.arange(win_lo - pad_lo, hi + dx / 2.0, dx)
    source_idx = int(np.argmin(np.abs(mesh - src_md)))
    targets = [{'gauge': n, 'md_ft': series[n]['md_ft'],
                'distance_ft': abs(series[n]['md_ft'] - src_md),
                'idx': int(np.argmin(np.abs(mesh - series[n]['md_ft']))),
                'taxis': series[n]['taxis'], 'data': series[n]['delta_psi']}
               for n in tgt_gauges]
    return dict(series=series, src_gauge=src_gauge, src=src, mesh=mesh,
                source_idx=source_idx, targets=targets, src_md=src_md,
                fh_centroid=fh_centroid, win_lo=win_lo, win_hi=win_hi)


def _init_worker(cfg, source_mode):
    _G['cfg'] = cfg
    _G['S'] = _setup(cfg, source_mode)


def _misfit(args):
    fam, p = args
    cfg, S = _G['cfg'], _G['S']
    spec = core.PROFILE_FAMILIES[fam]
    if fam == 'triangular':
        prof = spec['fn'](S['mesh'], S['source_idx'], p,
                          S['win_lo'], S['win_hi'])
    else:
        prof = spec['fn'](S['mesh'], S['source_idx'], p)
    return core.misfit_for_profile(
        prof, S['mesh'], cfg['solver']['dt_s'],
        float(S['src']['taxis'][-1]), S['src']['taxis'], S['src']['delta_psi'],
        S['source_idx'], S['targets'])


def fit_family(pool, fam, bounds, n_coarse, seed, maxiter):
    """Two parallel Latin-hypercube rounds, then a short Nelder-Mead polish.

    The LHS rounds carry the search (they parallelise perfectly across cores);
    Nelder-Mead is serial so it is used only to polish, not to explore.
    """
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)

    def sample(centre, frac, n, sd):
        if centre is None:
            pts = lo + qmc.LatinHypercube(d=len(bounds), seed=sd).random(n) * (hi - lo)
        else:
            half = (hi - lo) * frac / 2.0
            l2 = np.maximum(lo, centre - half)
            h2 = np.minimum(hi, centre + half)
            pts = l2 + qmc.LatinHypercube(d=len(bounds), seed=sd).random(n) * (h2 - l2)
        vals = pool.map(_misfit, [(fam, q) for q in pts], chunksize=4)
        v = np.array([x[0] for x in vals])
        return pts, v

    pts, v = sample(None, None, n_coarse, seed)
    i = int(np.argmin(v))
    best_p, best_v = pts[i].copy(), float(v[i])
    log(f"  {fam}: LHS round 1 best {best_v:.3f} psi ({n_coarse} pts)")

    pts2, v2 = sample(best_p, 0.25, max(n_coarse // 2, 100), seed + 1)
    j = int(np.argmin(v2))
    if v2[j] < best_v:
        best_p, best_v = pts2[j].copy(), float(v2[j])
    log(f"  {fam}: LHS round 2 best {best_v:.3f} psi")

    res = minimize(lambda q: _misfit((fam, q))[0], best_p, method='Nelder-Mead',
                   options={'maxiter': maxiter, 'xatol': 1e-3, 'fatol': 1e-4,
                            'disp': False})
    if float(res.fun) < best_v:
        best_p, best_v = np.asarray(res.x, float), float(res.fun)
    best_p = np.clip(best_p, lo, hi)

    a, n = _misfit((fam, best_p))
    return {'family': fam, 'params': [float(x) for x in best_p],
            'param_names': core.PROFILE_FAMILIES[fam]['names'],
            'k': core.PROFILE_FAMILIES[fam]['k'],
            'rmse_psi': float(a), 'rmse_normalised': float(n),
            'at_bound': [bool(abs(x - b[0]) < 1e-6 or abs(x - b[1]) < 1e-6)
                         for x, b in zip(best_p, bounds)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg, cfg_hash = load_config(args.config)
    log(f"config {args.config} sha256={cfg_hash[:16]}")

    fam_cfg = cfg['families']
    n_coarse = cfg['search']['n_coarse']
    maxiter = cfg['search']['nelder_mead_maxiter']
    seed = cfg['search']['seed']
    nproc = cfg['search']['processes']

    all_results = {}
    for mode in cfg['source_variants']['modes']:
        S = _setup(cfg, mode)
        log(f"=== source mode '{mode}': source MD {S['src_md']:.1f}, "
            f"domain MD [{S['mesh'][0]:.0f}, {S['mesh'][-1]:.0f}] "
            f"(nx={len(S['mesh'])}), targets "
            f"{[t['gauge'] for t in S['targets']]}")
        res = []
        # The Nelder-Mead polish runs in THIS process, so give it the same
        # globals the pool workers get from their initializer.
        _init_worker(cfg, mode)
        with Pool(nproc, initializer=_init_worker, initargs=(cfg, mode)) as pool:
            for fam, spec in fam_cfg.items():
                r = fit_family(pool, fam, spec['bounds'], n_coarse, seed, maxiter)
                res.append(r)
                log(f"  {fam:12s} k={r['k']} RMSE={r['rmse_psi']:7.3f} psi  "
                    f"norm={r['rmse_normalised']:.4f}  "
                    f"params={[round(x,3) for x in r['params']]}"
                    f"{'  [AT BOUND]' if any(r['at_bound']) else ''}")
        res.sort(key=lambda r: r['rmse_psi'])
        all_results[mode] = {'setup': {
            'source_md_ft': S['src_md'], 'n_mesh': int(len(S['mesh'])),
            'domain_md': [float(S['mesh'][0]), float(S['mesh'][-1])],
            'target_gauges': [t['gauge'] for t in S['targets']],
            'target_distance_ft': [t['distance_ft'] for t in S['targets']]},
            'families': res}

    # D(x) of the winning family, sampled at the gauge MDs, for reporting
    base = cfg['source_variants']['modes'][0]
    S = _setup(cfg, base)
    win = all_results[base]['families'][0]
    spec = core.PROFILE_FAMILIES[win['family']]
    p = np.array(win['params'])
    prof = (spec['fn'](S['mesh'], S['source_idx'], p, S['win_lo'], S['win_hi'])
            if win['family'] == 'triangular'
            else spec['fn'](S['mesh'], S['source_idx'], p))
    at_gauges = {str(t['gauge']): float(prof[t['idx']]) for t in S['targets']}
    log(f"WINNER ({base}): {win['family']} RMSE {win['rmse_psi']:.3f} psi; "
        f"D at gauges: " + ", ".join(f"g{k}={v:.0f}" for k, v in at_gauges.items()))

    out = cfg['outputs']
    os.makedirs(os.path.dirname(out['manifest_json']), exist_ok=True)
    manifest = {
        'study_id': cfg['study_id'],
        'run_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_sha256': cfg_hash, 'config_resolved': cfg,
        'environment': {
            'python': sys.version.split()[0], 'platform': platform.platform(),
            'numpy': np.__version__, 'scipy': __import__('scipy').__version__,
            'code_sha256': {
                'runner': core.file_sha256(os.path.abspath(__file__)),
                'core': core.file_sha256(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'r1_calibration_core.py'))},
            'cwd': os.getcwd(),
            'note': 'bakken_mariner/.git is empty; code identity pinned by sha256.'},
        'results': all_results,
        'winner': {'source_mode': base, **win, 'D_at_gauges': at_gauges},
    }
    with open(out['manifest_json'], 'w') as fh:
        json.dump(manifest, fh, indent=2)
    log(f"wrote {out['manifest_json']}")

    np.savez(out['arrays_npz'], mesh=S['mesh'], winner_profile=prof,
             gauge_md=np.array([t['md_ft'] for t in S['targets']]),
             gauge_distance=np.array([t['distance_ft'] for t in S['targets']]),
             gauge_D=np.array([prof[t['idx']] for t in S['targets']]))
    log(f"wrote {out['arrays_npz']}")
    log('done')


if __name__ == '__main__':
    main()
