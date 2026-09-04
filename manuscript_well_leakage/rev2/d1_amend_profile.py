"""D1 AMEND, blocker 2: the near-zone identifiability probe was a conditional
slice, not an identifiability band.

`d1_nearzone_identifiability.py:73-77` builds every scan point as
`q = base.copy(); q[pi] = v`, so `log10_D_near` is moved with `log10_D_far`,
`log10_s_c` and `log10_width` FROZEN at the fitted values. A one-at-a-time slice
is blind to a parameter trade-off by construction, and the trade-off is exactly
what the README then used the probe to rule out. Three further problems in the
same file:

  * the scan is 41 points across the FULL bounds -- 0.0875 decade per step for
    log10_D_near -- so the reported "+10% band = 0.175 decades" is two grid
    steps and every "+1% band" in `d1_identifiability_v1.json` is 0.0000
    decades, i.e. a single point, i.e. unresolved;
  * the reference is the minimum OVER THE GRID, and the grid misses the fitted
    optimum (`absolute|all|log10_s_c` reports 14.0419 psi where the fit is
    11.8719), so those bands are taken relative to an inflated criterion;
  * bands that reach a search bound were not flagged as censored.

This script replaces it with a PROFILE LIKELIHOOD: at every scanned value the
three remaining parameters are re-optimised, the grid is refined to 0.03 decade
near the optimum, the reference is the fitted subset optimum (best of the
published warm-started fit and the cold-start ensemble), and any band touching a
search bound is reported as CENSORED. The frozen slice is recomputed on the same
grid so the two are directly comparable.

    python scripts/manuscript_well_leakage/rev2/d1_amend_profile.py \
        --config configs/rev2/d1_amend.json
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'manuscript_well_leakage',
                                'baseline_calibration'))
sys.path.insert(0, HERE)

import d1_loo_blind as d1        # noqa: E402
import rev2_manifest as rm       # noqa: E402
import r1_calibration_core as core  # noqa: E402
from r1_run_calibration import load_config  # noqa: E402
from d1_amend_coldstart import write_amend_manifest, read_published  # noqa: E402

COLDSTART_JSON = 'output/rev2_20260901/D1/amend/d1_coldstart_ensemble_v2.json'
PARAM_IDX = {'log10_D_near': 0, 'log10_s_c': 2}


def log(m):
    print(f"[d1-amend-prof] {m}", flush=True)


# ---------------------------------------------------------------------------
# worker: one continuation chain of profile points
# ---------------------------------------------------------------------------

def _chain(args):
    """Walk a monotone list of scanned values, re-optimising the other three
    parameters at each, warm-started from the previous point's solution."""
    (norm, subset_cols, held_out_col, pi, values, base, bounds, seed_starts,
     maxiter, maxfev) = args
    S = d1._G['S']
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    free = [i for i in range(4) if i != pi]
    cols = list(subset_cols)
    base = np.asarray(base, float)

    def eval_full(q):
        mse, n2 = d1.per_gauge_misfit(S, d1._profile(S, 'two_zone',
                                                    np.clip(q, lo, hi)))
        v = np.asarray(n2 if norm == 'normalised' else mse)[cols]
        crit = (float(np.sqrt(np.mean(v)))
                if np.all(np.isfinite(v)) else float('inf'))
        return crit, float(np.sqrt(mse[held_out_col])), float(np.sqrt(n2[held_out_col]))

    out = []
    prev = np.array(base[free], float)
    for j, v in enumerate(values):
        def f(x):
            q = base.copy()
            q[pi] = v
            q[free] = np.clip(x, lo[free], hi[free])
            return eval_full(q)[0]

        starts = [prev]
        if j == 0:
            starts += [np.asarray(s, float) for s in seed_starts]
        best_x, best_c = None, np.inf
        for s in starts:
            r = minimize(f, np.asarray(s, float), method='Nelder-Mead',
                         options={'maxiter': maxiter, 'maxfev': maxfev,
                                  'xatol': 1e-3, 'fatol': 1e-4})
            if float(r.fun) < best_c:
                best_c, best_x = float(r.fun), np.clip(np.asarray(r.x, float),
                                                       lo[free], hi[free])
        q = base.copy()
        q[pi] = v
        q[free] = best_x
        crit_p, blind_psi, blind_norm = eval_full(q)
        # frozen (one-at-a-time) counterpart on the same grid point
        qf = base.copy()
        qf[pi] = v
        crit_f, blind_f_psi, _ = eval_full(qf)
        out.append({'value_log10': float(v), 'profiled_criterion': crit_p,
                    'profiled_params': [float(x) for x in q],
                    'profiled_blind_rmse_psi': blind_psi,
                    'profiled_blind_rmse_normalised': blind_norm,
                    'frozen_criterion': crit_f,
                    'frozen_blind_rmse_psi': blind_f_psi})
        prev = best_x
    return (norm, pi, tuple(float(v) for v in values), out)


def _repair(args):
    """Re-polish one grid point from a neighbour's solution."""
    (norm, subset_cols, held_out_col, pi, value, start_params, bounds,
     maxiter, maxfev) = args
    S = d1._G['S']
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    free = [i for i in range(4) if i != pi]
    cols = list(subset_cols)
    base = np.asarray(start_params, float)

    def eval_full(q):
        mse, n2 = d1.per_gauge_misfit(S, d1._profile(S, 'two_zone',
                                                    np.clip(q, lo, hi)))
        v = np.asarray(n2 if norm == 'normalised' else mse)[cols]
        crit = (float(np.sqrt(np.mean(v)))
                if np.all(np.isfinite(v)) else float('inf'))
        return crit, float(np.sqrt(mse[held_out_col])), float(np.sqrt(n2[held_out_col]))

    def f(x):
        q = base.copy()
        q[pi] = value
        q[free] = np.clip(x, lo[free], hi[free])
        return eval_full(q)[0]

    r = minimize(f, base[free], method='Nelder-Mead',
                 options={'maxiter': maxiter, 'maxfev': maxfev,
                          'xatol': 1e-3, 'fatol': 1e-4})
    q = base.copy()
    q[pi] = value
    q[free] = np.clip(np.asarray(r.x, float), lo[free], hi[free])
    c, bp, bn = eval_full(q)
    return (norm, pi, float(value), {'profiled_criterion': c,
                                     'profiled_params': [float(x) for x in q],
                                     'profiled_blind_rmse_psi': bp,
                                     'profiled_blind_rmse_normalised': bn})


# ---------------------------------------------------------------------------

def band(values, crit, blind, ref, tol, bounds_lo, bounds_hi, step):
    """Bracket of scanned values whose PROFILED criterion stays <= ref*(1+tol).

    Reported as the connected interval containing the argmin, so a disconnected
    low-criterion island far away cannot silently widen it.
    """
    v = np.asarray(values, float)
    c = np.asarray(crit, float)
    b = np.asarray(blind, float)
    thr = ref * (1.0 + tol)
    i0 = int(np.argmin(c))
    lo = i0
    while lo - 1 >= 0 and c[lo - 1] <= thr:
        lo -= 1
    hi = i0
    while hi + 1 < len(c) and c[hi + 1] <= thr:
        hi += 1
    inside = slice(lo, hi + 1)
    return {
        'tol': tol, 'reference_criterion': float(ref),
        'threshold_criterion': float(thr),
        'log10_interval': [float(v[lo]), float(v[hi])],
        'physical_interval': [float(10 ** v[lo]), float(10 ** v[hi])],
        'decades': float(v[hi] - v[lo]),
        'factor': float(10 ** (v[hi] - v[lo])),
        'censored_low': bool(abs(v[lo] - bounds_lo) < 1e-9),
        'censored_high': bool(abs(v[hi] - bounds_hi) < 1e-9),
        'grid_step_decades_near_optimum': float(step),
        'resolved': bool((v[hi] - v[lo]) > 1.5 * step),
        'blind_rmse_g_range_psi': [float(np.nanmin(b[inside])),
                                   float(np.nanmax(b[inside]))],
        'n_grid_points_inside': int(hi - lo + 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg, cfg_sha = load_config(args.config)
    out = cfg['outputs']
    pc = cfg['amend']['profile_identifiability']
    os.makedirs(out['dir'], exist_ok=True)
    rm.assert_absent([out['profile_json'], out['manifest_profile'],
                      out['fig_profile']])

    t_start = time.time()
    S = d1.build_setup(cfg)
    gnums = [t['gauge'] for t in S['targets']]
    g2col = gnums.index(2)
    cols5 = tuple(c for c in range(len(gnums)) if c != g2col)
    bounds = [tuple(b) for b in cfg['families']['two_zone']['bounds']]

    pub_calib, pub_blind = read_published()
    cold = json.load(open(COLDSTART_JSON))['best_of_search']

    # reference optimum per norm: the best fit known for the drop_g2 subset
    refs = {}
    for norm in ('absolute', 'normalised'):
        pp, pcrit = pub_calib[(norm, 'drop_g2')]
        cbest = cold[f'{norm}|drop_g2']
        if cbest['criterion'] < pcrit:
            refs[norm] = {'params': cbest['params'], 'criterion': cbest['criterion'],
                          'origin': f"cold-start restart {cbest['restart']}"}
        else:
            refs[norm] = {'params': pp, 'criterion': pcrit,
                          'origin': 'published warm-started fit (d1_calibrations_v1.csv)'}
        log(f"reference {norm}: crit={refs[norm]['criterion']:.6f} "
            f"from {refs[norm]['origin']}  params="
            f"{[round(10**x, 4) for x in refs[norm]['params']]}")

    fine_h = float(pc['fine_half_width_decades'])
    fine_s = float(pc['fine_step_decades'])
    coarse_s = float(pc['coarse_step_decades'])
    maxiter = int(pc['nelder_mead_maxiter'])
    maxfev = int(pc['nelder_mead_maxfev'])

    jobs, grids = [], {}
    for norm in ('absolute', 'normalised'):
        base = np.asarray(refs[norm]['params'], float)
        for pname, pi in PARAM_IDX.items():
            blo, bhi = bounds[pi]
            c0 = float(base[pi])
            fine_up = [c0 + fine_s * k for k in range(1, int(fine_h / fine_s) + 1)
                       if c0 + fine_s * k <= bhi]
            fine_dn = [c0 - fine_s * k for k in range(1, int(fine_h / fine_s) + 1)
                       if c0 - fine_s * k >= blo]
            top = (fine_up[-1] if fine_up else c0)
            bot = (fine_dn[-1] if fine_dn else c0)
            coarse_up = list(np.arange(top + coarse_s, bhi + 1e-9, coarse_s))
            if not coarse_up or abs(coarse_up[-1] - bhi) > 1e-9:
                coarse_up.append(bhi)
            coarse_dn = list(np.arange(bot - coarse_s, blo - 1e-9, -coarse_s))
            if not coarse_dn or abs(coarse_dn[-1] - blo) > 1e-9:
                coarse_dn.append(blo)
            grids[(norm, pi)] = dict(c0=c0, fine_up=fine_up, fine_dn=fine_dn,
                                     coarse_up=coarse_up, coarse_dn=coarse_dn)
            seeds = [[base[i] for i in range(4) if i != pi],
                     [2.4, 2.5, 2.0], [2.3, 2.8, 1.0]]
            seeds = [[s[0], s[1], s[2]] for s in seeds]
            # four sub-chains keep the pool balanced; each is a continuation walk
            # outward from the fitted optimum (or from it via 3 seeded starts)
            jobs.append((norm, cols5, g2col, pi, tuple([c0] + fine_up), base,
                         bounds, seeds, maxiter, maxfev))
            if fine_dn:
                jobs.append((norm, cols5, g2col, pi, tuple(fine_dn), base,
                             bounds, seeds, maxiter, maxfev))
            if coarse_up:
                jobs.append((norm, cols5, g2col, pi, tuple(coarse_up), base,
                             bounds, seeds, maxiter, maxfev))
            if coarse_dn:
                jobs.append((norm, cols5, g2col, pi, tuple(coarse_dn), base,
                             bounds, seeds, maxiter, maxfev))
    npoints = sum(len(j[4]) for j in jobs)
    log(f"{len(jobs)} continuation chains, {npoints} profile points "
        f"(fine step {fine_s} dec, coarse {coarse_s} dec)")

    nproc = int(cfg['search']['processes'])
    store = {}
    with Pool(nproc, initializer=d1._init_worker, initargs=(cfg,)) as pool:
        t0 = time.time()
        for norm, pi, vals, recs in pool.imap_unordered(_chain, jobs):
            for r in recs:
                store[(norm, pi, round(r['value_log10'], 9))] = r
            log(f"  chain {norm:11s} p{pi} {len(recs):3d} pts "
                f"[{10**vals[0]:.4g} .. {10**vals[-1]:.4g}] "
                f"({time.time()-t0:.0f} s elapsed)")

        # ---- neighbour repair: a continuation chain can lag behind a better
        # branch found by the chain walking the other way.
        rounds = 0
        while rounds < 3:
            rounds += 1
            fix = []
            for (norm, pi) in {(k[0], k[1]) for k in store}:
                ks = sorted(k[2] for k in store if k[0] == norm and k[1] == pi)
                for a, b in list(zip(ks[:-1], ks[1:])) + list(zip(ks[1:], ks[:-1])):
                    ra, rb = store[(norm, pi, a)], store[(norm, pi, b)]
                    if rb['profiled_criterion'] < ra['profiled_criterion'] * (1 - 1e-9):
                        fix.append((norm, cols5, g2col, pi, a,
                                    rb['profiled_params'], bounds, maxiter, maxfev))
            # de-duplicate by (norm, pi, value)
            seen, uniq = set(), []
            for j in fix:
                key = (j[0], j[3], round(j[4], 9))
                if key not in seen:
                    seen.add(key)
                    uniq.append(j)
            if not uniq:
                break
            nimp = 0
            for norm, pi, val, rec in pool.imap_unordered(_repair, uniq):
                cur = store[(norm, pi, round(val, 9))]
                if rec['profiled_criterion'] < cur['profiled_criterion']:
                    cur.update(rec)
                    nimp += 1
            log(f"  repair round {rounds}: {len(uniq)} candidates, {nimp} improved")
            if nimp == 0:
                break

    # ---- assemble ---------------------------------------------------------
    scans, summary = {}, {}
    for norm in ('absolute', 'normalised'):
        ref = refs[norm]['criterion']
        for pname, pi in PARAM_IDX.items():
            ks = sorted(k[2] for k in store if k[0] == norm and k[1] == pi)
            recs = [store[(norm, pi, k)] for k in ks]
            v = [r['value_log10'] for r in recs]
            key = f'{norm}|drop_g2|{pname}'
            scans[key] = {
                'param': pname, 'norm': norm, 'subset': 'drop_g2',
                'reference_criterion': ref,
                'reference_origin': refs[norm]['origin'],
                'grid_log10': v,
                'profiled_criterion': [r['profiled_criterion'] for r in recs],
                'frozen_criterion': [r['frozen_criterion'] for r in recs],
                'profiled_blind_rmse_g2_psi': [r['profiled_blind_rmse_psi'] for r in recs],
                'frozen_blind_rmse_g2_psi': [r['frozen_blind_rmse_psi'] for r in recs],
                'profiled_params': [r['profiled_params'] for r in recs],
            }
            blo, bhi = bounds[pi]
            s = {'argmin_profiled_log10': float(v[int(np.argmin(
                    [r['profiled_criterion'] for r in recs]))]),
                 'min_profiled_criterion': float(min(
                     r['profiled_criterion'] for r in recs)),
                 'reference_criterion': float(ref),
                 'reference_origin': refs[norm]['origin'],
                 'search_bounds_log10': [blo, bhi],
                 'n_grid_points': len(v),
                 'fine_step_decades': fine_s, 'coarse_step_decades': coarse_s}
            for tag, tol in (('1pct', 0.01), ('10pct', 0.10)):
                s[tag] = band(v, [r['profiled_criterion'] for r in recs],
                              [r['profiled_blind_rmse_psi'] for r in recs],
                              ref, tol, blo, bhi, fine_s)
                s[tag + '_frozen'] = band(
                    v, [r['frozen_criterion'] for r in recs],
                    [r['frozen_blind_rmse_psi'] for r in recs],
                    ref, tol, blo, bhi, fine_s)
            summary[key] = s
            b10 = s['10pct']
            log(f"{key:36s} +10% band {b10['physical_interval'][0]:9.1f}.."
                f"{b10['physical_interval'][1]:11.1f} = {b10['decades']:.3f} dec "
                f"(factor {b10['factor']:.1f})"
                f"{' CENSORED' if b10['censored_low'] or b10['censored_high'] else ''}"
                f"  blind g2 {b10['blind_rmse_g_range_psi'][0]:.1f}.."
                f"{b10['blind_rmse_g_range_psi'][1]:.1f} psi ; "
                f"+1% {s['1pct']['decades']:.3f} dec")

    doc = {'what': ('profile-likelihood identifiability of the two_zone near-zone '
                    'parameters for the drop_g2 subset: at every scanned value the '
                    'other three parameters are RE-OPTIMISED'),
           'reference': refs, 'scans': scans, 'summary': summary,
           'wall_seconds': float(time.time() - t_start),
           'note': ('The frozen_* series reproduce d1_nearzone_identifiability.py\'s '
                    'one-at-a-time slice on the SAME grid, so the two are directly '
                    'comparable. Bands are the connected interval containing the '
                    'profiled argmin, measured against the fitted subset optimum '
                    '(not the grid minimum), and flagged CENSORED where they reach '
                    'a search bound.')}
    with open(out['profile_json'], 'w') as fh:
        json.dump(doc, fh, indent=2)
    log(f"wrote {out['profile_json']}")

    make_figure(cfg, scans, summary, refs, out)

    taxis, _ = core.solve_forward(
        S['mesh'], d1._profile(S, 'two_zone',
                               np.array(cfg['families']['two_zone']['warm_start'], float)),
        S['dt'], S['t_total'], S['src']['taxis'], S['src']['delta_psi'],
        S['source_idx'], record_idx=[0])
    write_amend_manifest(
        out['manifest_profile'], cfg, args.config, S, taxis,
        study_id='d1_amend_profile_identifiability',
        results={'reference': refs, 'summary': summary,
                 'n_profile_points': npoints,
                 'wall_seconds': doc['wall_seconds']},
        outputs=[(out['profile_json'], 'json', None),
                 (out['fig_profile'], 'figure_png', int(out['figure_dpi']))],
        notes=['Amends blocker 2 / major defect 9 of output/rev2_20260901/A4/'
               'challenge_defects/D1_defects.json.',
               'Replaces the one-at-a-time slice of '
               'd1_nearzone_identifiability.py with a profile likelihood.',
               'Bands reaching a search bound are CENSORED and must not be '
               'quoted as estimates (HOUSE_RULES statistics rules).'],
        extra_inputs=[(COLDSTART_JSON, 'prior_run_output', 'coldstart_ensemble'),
                      ('output/rev2_20260901/D1/d1_identifiability_v1.json',
                       'prior_run_output', 'identifiability_v1')])
    log(f"wrote {out['manifest_profile']}")
    log(f"TOTAL wall {time.time()-t_start:.0f} s")


def make_figure(cfg, scans, summary, refs, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.2))
    for r, norm in enumerate(('absolute', 'normalised')):
        for c, pname in enumerate(('log10_D_near', 'log10_s_c')):
            a = ax[r, c]
            a2 = a.twinx()
            key = f'{norm}|drop_g2|{pname}'
            d = scans[key]
            s = summary[key]
            g = 10 ** np.asarray(d['grid_log10'])
            ref = d['reference_criterion']
            a.plot(g, np.asarray(d['profiled_criterion']) / ref, '-o', ms=2.6,
                   color='#d62728', lw=1.4,
                   label='PROFILED (other 3 re-optimised)')
            a.plot(g, np.asarray(d['frozen_criterion']) / ref, '--', color='0.45',
                   lw=1.2, label='FROZEN slice (as published)')
            a2.plot(g, d['profiled_blind_rmse_g2_psi'], ':', color='#1f77b4',
                    lw=1.5, label='blind RMSE at g2, profiled (right axis)')
            b = s['10pct']
            a.axhspan(1.0, 1.10, color='#ffd9d9', alpha=0.55, zorder=0)
            a.axvspan(b['physical_interval'][0], b['physical_interval'][1],
                      color='#d62728', alpha=0.09, zorder=0)
            a.axhline(1.10, color='0.3', ls='-.', lw=0.8)
            a.axhline(1.01, color='0.6', ls='--', lw=0.8)
            for edge, cens in ((b['physical_interval'][0], b['censored_low']),
                               (b['physical_interval'][1], b['censored_high'])):
                if cens:
                    a.axvline(edge, color='#d62728', lw=1.6)
                    a.annotate('CENSORED\nat search bound', (edge, 1.13),
                               fontsize=6.5, color='#d62728', ha='left'
                               if edge == b['physical_interval'][0] else 'right')
            a.set_xscale('log')
            a.set_yscale('log')
            a2.set_yscale('log')
            a.set_xlabel(pname.replace('log10_', '')
                         + (' (ft)' if 's_c' in pname else ' (ft$^2$/s)'))
            a.set_ylabel('criterion / fitted subset optimum')
            a2.set_ylabel('blind RMSE at gauge 2 (psi)', color='#1f77b4')
            a.set_title(
                f"({'abcd'[2*r+c]}) {norm} norm, drop-g2 subset: profile of {pname}\n"
                f"+10% band {b['physical_interval'][0]:.0f}-{b['physical_interval'][1]:.0f}"
                f" = {b['decades']:.2f} dec"
                + (' [CENSORED]' if b['censored_low'] or b['censored_high'] else '')
                + f"; blind g2 {b['blind_rmse_g_range_psi'][0]:.1f}-"
                  f"{b['blind_rmse_g_range_psi'][1]:.1f} psi", fontsize=8.5)
            a.grid(alpha=0.25)
            h1, l1 = a.get_legend_handles_labels()
            h2, l2 = a2.get_legend_handles_labels()
            a.legend(h1 + h2, l1 + l2, fontsize=6.5, loc='upper center')
    fig.suptitle('D1 AMEND -- near-zone identifiability as a PROFILE likelihood. The published probe '
                 'froze the other three parameters (grey dashed);\nre-optimising them (red) widens every '
                 'band by more than an order of magnitude, and the blind error at the withheld gauge '
                 'varies across it.', fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out['fig_profile'], dpi=int(out['figure_dpi']))
    plt.close(fig)
    log(f"wrote {out['fig_profile']}")


if __name__ == '__main__':
    main()
