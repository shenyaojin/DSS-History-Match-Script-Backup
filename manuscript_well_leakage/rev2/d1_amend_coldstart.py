"""D1 AMEND, blocker 1: the two_zone leave-one-out refits were warm-started at
the ALL-SIX-GAUGE optimum, i.e. at a point computed WITH the held-out gauge.

`configs/rev2/d1_loo_blind.json` -> `families.two_zone.warm_start` is the r2
winner, and `d1_loo_blind.py:322` injects it both as a candidate and as a
Nelder-Mead start for every subset. Two independent reviewers re-inverted the
subsets from scratch and found (a) better five-gauge fits on drop_g3 and drop_g6
whose blind predictions are WORSE, and (b) mutually inconsistent aggregates
(14.54 and 16.73 psi against the published 14.14). The published per-subset
optima are therefore optimiser-dependent, and the published number cannot be
quoted as a point value.

This script replaces the single warm-started search with a RESTART ENSEMBLE:
`n_restarts` independent cold searches, each with its own Latin hypercube seed
and no knowledge of the published optimum, on every (subset, norm). It reports

  * the leave-one-out blind aggregate for each restart -> a RANGE, which is what
    is quotable;
  * the best-of-search optimum per subset (pooling all restarts and, for
    reference only, the published warm-started fit), with its blind score;
  * the spread of the blind RMSE across the restarts whose calibration criterion
    agrees with the best to within 1% -- the honest per-gauge uncertainty, since
    those restarts are indistinguishable on the data they were given.

Nothing under baseline_calibration/ or the shared rev2 modules is modified; the
verified kernel and d1_loo_blind's own setup/misfit helpers are imported.

    python scripts/manuscript_well_leakage/rev2/d1_amend_coldstart.py \
        --config configs/rev2/d1_amend.json
"""

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'manuscript_well_leakage',
                                'baseline_calibration'))
sys.path.insert(0, HERE)

import d1_loo_blind as d1        # noqa: E402  (setup + per_gauge_misfit only)
import rev2_manifest as rm       # noqa: E402
import r1_calibration_core as core  # noqa: E402
from r1_run_calibration import load_config  # noqa: E402

PUBLISHED_CALIB_CSV = 'output/rev2_20260901/D1/d1_calibrations_v1.csv'
PUBLISHED_SUMMARY_CSV = 'output/rev2_20260901/D1/d1_summary_v1.csv'


def log(m):
    print(f"[d1-amend-cold] {m}", flush=True)


# ---------------------------------------------------------------------------
# one (restart, subset, norm) cold inversion, run inside a worker
# ---------------------------------------------------------------------------

def _fit_job(args):
    (restart, label, cols, norm, start_pts, bounds, n_local, frac, seed,
     maxiter, maxfev) = args
    S = d1._G['S']
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    cols = list(cols)

    def f(p):
        p = np.clip(np.asarray(p, float), lo, hi)
        mse, n2 = d1.per_gauge_misfit(S, d1._profile(S, 'two_zone', p))
        v = np.asarray(n2 if norm == 'normalised' else mse)[cols]
        if not np.all(np.isfinite(v)):
            return np.inf
        return float(np.sqrt(np.mean(v)))

    best_p = np.asarray(start_pts[0], float)
    best_v = f(best_p)
    for sp in start_pts[1:]:
        v = f(sp)
        if v < best_v:
            best_p, best_v = np.asarray(sp, float), v

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
                       options={'maxiter': maxiter, 'maxfev': maxfev,
                                'xatol': 1e-3, 'fatol': 1e-4, 'disp': False})
        if float(res.fun) < best_v:
            best_p = np.clip(np.asarray(res.x, float), lo, hi)
            best_v = float(res.fun)

    # score the held-out gauge with the accepted parameters
    mse, n2 = d1.per_gauge_misfit(S, d1._profile(S, 'two_zone', best_p))
    gnums = [t['gauge'] for t in S['targets']]
    blind = None
    if label != 'all':
        g = int(label.split('_g')[1])
        k = gnums.index(g)
        blind = {'gauge': g, 'rmse_psi': float(np.sqrt(mse[k])),
                 'rmse_normalised': float(np.sqrt(n2[k]))}
    on_bound = [bool(abs(x - b[0]) < 1e-3 or abs(x - b[1]) < 1e-3)
                for x, b in zip(best_p, bounds)]
    return {'restart': int(restart), 'norm': norm, 'subset': label,
            'params': [float(x) for x in best_p], 'criterion': float(best_v),
            'blind': blind, 'at_or_within_1e-3_of_bound': on_bound}


# ---------------------------------------------------------------------------

def read_published():
    calib, summ = {}, {}
    with open(PUBLISHED_CALIB_CSV) as fh:
        for r in csv.DictReader(fh):
            if r['model'] == 'two_zone':
                calib[(r['norm'], r['subset'])] = (
                    [float(x) for x in r['params'].split(';')],
                    float(r['criterion_value']))
    with open(PUBLISHED_SUMMARY_CSV) as fh:
        for r in csv.DictReader(fh):
            if r['model'] == 'two_zone':
                summ[(r['norm'], int(r['held_out_gauge']))] = float(r['blind_rmse_psi'])
    return calib, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg, cfg_sha = load_config(args.config)
    out = cfg['outputs']
    am = cfg['amend']['coldstart_ensemble']
    os.makedirs(out['dir'], exist_ok=True)
    rm.assert_absent([out['coldstart_csv'], out['coldstart_json'],
                      out['coldstart_summary_csv'], out['manifest_coldstart']])

    t_start = time.time()
    S = d1.build_setup(cfg)
    gnums = [t['gauge'] for t in S['targets']]
    log(f"source g{S['src_gauge']} MD {S['src_md']:.0f} idx {S['source_idx']}; "
        f"targets {gnums}; nx={len(S['mesh'])}")

    subsets = {'all': list(range(len(gnums)))}
    for k, gn in enumerate(gnums):
        subsets[f'drop_g{gn}'] = [c for c in range(len(gnums)) if c != k]

    fam = cfg['families']['two_zone']
    bounds = [tuple(b) for b in fam['bounds']]
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    nproc = int(cfg['search']['processes'])
    R = int(am['n_restarts'])
    NG = int(am['n_global_lhs_per_restart'])
    seeds = [int(am['seed_base']) + 1000 * r for r in range(R)]

    records = []
    with Pool(nproc, initializer=d1._init_worker, initargs=(cfg,)) as pool:
        for r in range(R):
            t0 = time.time()
            pts = lo + qmc.LatinHypercube(d=4, seed=seeds[r]).random(NG) * (hi - lo)
            res = pool.map(d1._eval_point, [('two_zone', p) for p in pts],
                           chunksize=4)
            mse = np.array([x[0] for x in res])
            nrm = np.array([x[1] for x in res])
            log(f"restart {r}: LHS {NG} solves in {time.time()-t0:.0f} s "
                f"(seed {seeds[r]}, NO warm start)")
            jobs = []
            for jn, norm in enumerate(('absolute', 'normalised')):
                mat = nrm if norm == 'normalised' else mse
                for si, (label, cols) in enumerate(subsets.items()):
                    c = np.sqrt(np.mean(mat[:, cols], axis=1))
                    order = np.argsort(c)[:3]
                    jobs.append((r, label, tuple(cols), norm,
                                 [list(map(float, pts[i])) for i in order],
                                 bounds, int(am['n_local_lhs']),
                                 float(am['local_lhs_frac']),
                                 seeds[r] + 7 * (100 * jn + si),
                                 int(am['nelder_mead_maxiter']),
                                 int(am['nelder_mead_maxfev'])))
            for rec in pool.imap_unordered(_fit_job, jobs):
                records.append(rec)
                b = rec['blind']
                log(f"  r{rec['restart']} {rec['norm']:11s} {rec['subset']:9s} "
                    f"crit={rec['criterion']:.5f} "
                    f"params={[round(10**x, 4) for x in rec['params']]}"
                    + (f"  BLIND={b['rmse_psi']:.4f}" if b else ""))
            log(f"restart {r} done in {time.time()-t0:.0f} s "
                f"({len(records)} records so far)")
            with open(out['coldstart_json'] + '.partial', 'w') as fh:
                json.dump(records, fh, indent=1)

    # ---- per-restart aggregates ------------------------------------------
    pub_calib, pub_blind = read_published()
    by = {(rec['restart'], rec['norm'], rec['subset']): rec for rec in records}
    per_restart = {}
    for norm in ('absolute', 'normalised'):
        for r in range(R):
            bl = [by[(r, norm, f'drop_g{g}')]['blind']['rmse_psi'] for g in gnums]
            per_restart[f'{norm}|restart{r}'] = {
                'loo_blind_rmse_gaugemean_psi': float(np.sqrt(np.mean(np.square(bl)))),
                'per_gauge_blind_rmse_psi': {str(g): v for g, v in zip(gnums, bl)},
                'subset_criteria': {s: by[(r, norm, s)]['criterion']
                                    for s in subsets},
            }

    # ---- best of search per (norm, subset), pooling restarts --------------
    best, reconciled = {}, []
    for norm in ('absolute', 'normalised'):
        for label in subsets:
            cand = [by[(r, norm, label)] for r in range(R)]
            cbest = min(cand, key=lambda z: z['criterion'])
            pub_p, pub_c = pub_calib[(norm, label)]
            best[(norm, label)] = {
                'coldstart_best': cbest,
                'published_warmstart': {'params': pub_p, 'criterion': pub_c},
                'coldstart_beats_published': bool(cbest['criterion'] < pub_c),
                'criterion_ratio_cold_over_published': float(cbest['criterion'] / pub_c),
                'restart_criteria': [z['criterion'] for z in cand],
            }
            # restarts indistinguishable from the best cold-start fit on the
            # five gauges they were given (criterion within 1%)
            near = [z for z in cand
                    if z['criterion'] <= cbest['criterion'] * 1.01]
            if label != 'all':
                bl = [z['blind']['rmse_psi'] for z in near]
                bl_all = [z['blind']['rmse_psi'] for z in cand]
                g = int(label.split('_g')[1])
                reconciled.append({
                    'norm': norm, 'held_out_gauge': g,
                    'distance_ft': float(S['targets'][gnums.index(g)]['distance_ft']),
                    'published_criterion': pub_c,
                    'published_blind_rmse_psi': pub_blind[(norm, g)],
                    'best_coldstart_criterion': cbest['criterion'],
                    'best_coldstart_blind_rmse_psi': cbest['blind']['rmse_psi'],
                    'blind_rmse_psi_min_over_restarts': float(min(bl_all)),
                    'blind_rmse_psi_max_over_restarts': float(max(bl_all)),
                    'blind_rmse_psi_min_within_1pct_of_best': float(min(bl)),
                    'blind_rmse_psi_max_within_1pct_of_best': float(max(bl)),
                    'n_restarts_within_1pct_of_best': len(near),
                    'best_coldstart_params_physical': ";".join(
                        f"{10**x:.5g}" for x in cbest['params']),
                })
    # aggregate of the best-of-search fits
    agg_best = {}
    for norm in ('absolute', 'normalised'):
        bl = [best[(norm, f'drop_g{g}')]['coldstart_best']['blind']['rmse_psi']
              for g in gnums]
        vals = [per_restart[f'{norm}|restart{r}']['loo_blind_rmse_gaugemean_psi']
                for r in range(R)]
        agg_best[norm] = {
            'loo_blind_rmse_gaugemean_psi_best_of_search': float(
                np.sqrt(np.mean(np.square(bl)))),
            'loo_blind_rmse_gaugemean_psi_per_restart': vals,
            'loo_blind_range_over_restarts_psi': [float(min(vals)), float(max(vals))],
            'published_warmstart_value_psi': float(np.sqrt(np.mean(np.square(
                [pub_blind[(norm, g)] for g in gnums])))),
        }
        log(f"AGG {norm:11s}: published(warm) "
            f"{agg_best[norm]['published_warmstart_value_psi']:.3f} psi ; "
            f"cold restarts {min(vals):.3f}-{max(vals):.3f} psi ; "
            f"best-of-search {agg_best[norm]['loo_blind_rmse_gaugemean_psi_best_of_search']:.3f} psi")

    # ---- write products ---------------------------------------------------
    cols = ['restart', 'norm', 'subset', 'criterion', 'log10_D_near',
            'log10_D_far', 'log10_s_c', 'log10_width', 'D_near_ft2_s',
            'D_far_ft2_s', 's_c_ft', 'width_ft', 'blind_gauge',
            'blind_rmse_psi', 'blind_rmse_normalised', 'near_bound_flags']
    with open(out['coldstart_csv'], 'w') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for rec in sorted(records, key=lambda z: (z['norm'], z['subset'], z['restart'])):
            p = rec['params']
            b = rec['blind'] or {}
            w.writerow([rec['restart'], rec['norm'], rec['subset'],
                        f"{rec['criterion']:.6g}"]
                       + [f"{x:.10g}" for x in p]
                       + [f"{10**x:.6g}" for x in p]
                       + [b.get('gauge', ''),
                          f"{b['rmse_psi']:.6g}" if b else '',
                          f"{b['rmse_normalised']:.6g}" if b else '',
                          ";".join('1' if z else '0'
                                   for z in rec['at_or_within_1e-3_of_bound'])])
    log(f"wrote {out['coldstart_csv']} ({len(records)} rows)")

    rcols = list(reconciled[0].keys())
    with open(out['coldstart_summary_csv'], 'w') as fh:
        w = csv.writer(fh)
        w.writerow(rcols)
        for r in reconciled:
            w.writerow([r[c] for c in rcols])
    log(f"wrote {out['coldstart_summary_csv']} ({len(reconciled)} rows)")

    doc = {
        'what': ('cold-start restart ensemble for the two_zone leave-one-out '
                 'inversions; no warm start from the all-six-gauge optimum'),
        'n_restarts': R, 'n_global_lhs_per_restart': NG, 'seeds': seeds,
        'per_restart_aggregate': per_restart,
        'aggregate': agg_best,
        'reconciled_per_gauge': reconciled,
        'best_of_search': {f'{n}|{l}': {
            'params': best[(n, l)]['coldstart_best']['params'],
            'params_physical': [10 ** x for x in best[(n, l)]['coldstart_best']['params']],
            'criterion': best[(n, l)]['coldstart_best']['criterion'],
            'restart': best[(n, l)]['coldstart_best']['restart'],
            'published_warmstart_criterion': best[(n, l)]['published_warmstart']['criterion'],
            'coldstart_beats_published': best[(n, l)]['coldstart_beats_published'],
            'criterion_ratio_cold_over_published': best[(n, l)]['criterion_ratio_cold_over_published'],
            'restart_criteria': best[(n, l)]['restart_criteria'],
        } for (n, l) in best},
        'records': records,
        'wall_seconds': float(time.time() - t_start),
    }
    with open(out['coldstart_json'], 'w') as fh:
        json.dump(doc, fh, indent=2)
    if os.path.exists(out['coldstart_json'] + '.partial'):
        os.remove(out['coldstart_json'] + '.partial')
    log(f"wrote {out['coldstart_json']}")

    # ---- manifest (house rule 3, via the shared module) -------------------
    taxis, _ = core.solve_forward(
        S['mesh'], d1._profile(S, 'two_zone', np.array(fam['warm_start'], float)),
        S['dt'], S['t_total'], S['src']['taxis'], S['src']['delta_psi'],
        S['source_idx'], record_idx=[0])
    write_amend_manifest(out['manifest_coldstart'], cfg, args.config, S, taxis,
                         study_id='d1_amend_coldstart_ensemble',
                         results={
                             'aggregate': agg_best,
                             'per_restart_aggregate': per_restart,
                             'reconciled_per_gauge': reconciled,
                             'best_of_search': doc['best_of_search'],
                             'n_forward_solves_estimate': int(
                                 R * NG + len(records) * (int(am['n_local_lhs'])
                                                          + 2 * int(am['nelder_mead_maxfev']))),
                             'wall_seconds': doc['wall_seconds'],
                         },
                         outputs=[(out['coldstart_csv'], 'csv', None),
                                  (out['coldstart_summary_csv'], 'csv', None),
                                  (out['coldstart_json'], 'json', None)],
                         notes=[
                             'Amends blocker 1 of output/rev2_20260901/A4/'
                             'challenge_defects/D1_defects.json.',
                             'NO warm start: the published all-six-gauge optimum '
                             'never enters the search. It is re-scored for '
                             'comparison only.',
                             'The quotable leave-one-out blind aggregate is the '
                             'RANGE over restarts, not any single restart.'])
    log(f"wrote {out['manifest_coldstart']}")
    log(f"TOTAL wall {time.time()-t_start:.0f} s")


# ---------------------------------------------------------------------------
# shared manifest assembly (used by the profile script too)
# ---------------------------------------------------------------------------

def write_amend_manifest(path, cfg, cfg_path, S, taxis, *, study_id, results,
                         outputs, notes, extra_inputs=()):
    fam = cfg['families']['two_zone']
    prof = d1._profile(S, 'two_zone', np.array(fam['warm_start'], float))
    drv = rm.driver_record(
        kind='gauge_series', baseline_removal='subtract_first_sample',
        value_units='delta_psi',
        series_path=cfg['data']['gauge_series_template'].format(n=S['src_gauge']),
        gauge_number=S['src_gauge'], gauge_md_ft=S['src_md'],
        taxis=S['src']['taxis'], values=S['src']['delta_psi'],
        time_start=cfg['window']['time_start'], time_end=cfg['window']['time_end'])
    src = rm.source_protocol(
        application='dirichlet_node',
        solver_class='r1_calibration_core.solve_forward (tridiagonal solve_banded)',
        placement_rule=cfg['source']['selection_rule'],
        sources=[rm.source_record(S['mesh'], md_requested_ft=S['src_md'],
                                  mesh_idx=S['source_idx'], driver=drv,
                                  label=f"g{S['src_gauge']}",
                                  index_in_source_list=0)],
        targets=[{'gauge': t['gauge'], 'md_ft': t['md_ft'],
                  'distance_ft': t['distance_ft'], 'mesh_idx': t['idx']}
                 for t in S['targets']],
        time_level=cfg['solver']['source_time_level'],
        phase_chaining=rm.NONE_DECLARED,
        boundary_conditions={'lbc': cfg['solver']['lbc'],
                             'rbc': cfg['solver']['rbc']})
    num = rm.numerics(
        time=rm.time_record(taxis, mode='fixed', theta=float(cfg['solver']['theta']),
                            t_total_requested_s=S['t_total'],
                            dt_requested_s=float(cfg['solver']['dt_s']),
                            source_time_level=cfg['solver']['source_time_level']),
        mesh=rm.mesh_record(S['mesh'], dx_requested_ft=float(cfg['mesh']['dx_ft']),
                            window_md_ft=(cfg['window']['md_min_ft'],
                                          cfg['window']['md_max_ft']),
                            pad_low_ft=float(cfg['mesh']['domain_pad_low_md_ft']),
                            pad_high_ft=float(cfg['mesh']['domain_pad_high_md_ft']),
                            refinement=cfg['mesh']['refinement']),
        interface_avg=cfg['solver']['interface_avg'],
        boundary={'lbc': cfg['solver']['lbc'], 'rbc': cfg['solver']['rbc'],
                  'pml_thickness': 0.0, 'sigma_max': 0.0},
        diffusivity={'profile_family': 'two_zone',
                     'param_names': fam['param_names'],
                     'bounds_log10': fam['bounds'],
                     'reference_params_for_hash': fam['warm_start'],
                     'D_min': float(prof.min()), 'D_max': float(prof.max()),
                     'D_sha256': rm.sha256_array(prof),
                     'profile_anchor': 'source mesh node (physical MD 16645)',
                     'note': ('D(x) varies per search point; the hashed profile is '
                              'the published all-gauge optimum, recorded as the '
                              'reference point the amend compares against.')},
        barriers=rm.NONE_DECLARED, leakage=rm.NONE_DECLARED,
        kernel={'name': 'r1_calibration_core.solve_forward', 'banded': True,
                'equivalence_reference':
                    'output/r1_baseline_calibration/r1_run_manifest.json; '
                    'rev2_core.solve_forward is bitwise identical to it at '
                    'theta=1/harmonic/lambda=0 (A4 self-test)'},
        rng={'engine': 'scipy.stats.qmc.LatinHypercube + numpy default',
             'seeds': cfg['amend']},
        parallel={'processes': int(cfg['search']['processes']),
                  'backend': 'multiprocessing.Pool'})
    inputs = [(cfg['data']['gauge_md_npz'], 'geometry', 'gauge_md_npz'),
              (cfg['data']['frac_hit_stage1_npz'], 'geometry', 'frac_hit_stage1')]
    inputs += [(cfg['data']['gauge_series_template'].format(n=n), 'gauge_series',
                f'gauge{n}') for n in sorted(S['series'])]
    inputs += [(PUBLISHED_CALIB_CSV, 'prior_run_output', 'd1_calibrations_v1'),
               (PUBLISHED_SUMMARY_CSV, 'prior_run_output', 'd1_summary_v1'),
               ('output/rev2_20260901/A4/challenge_defects/D1_defects.json',
                'prior_run_output', 'D1_defect_list')]
    inputs += list(extra_inputs)
    return rm.write_manifest(
        path, study_id=study_id, task_id='D1-amend', config=cfg,
        config_path=cfg_path, inputs=inputs, source=src, numerics=num,
        outputs=[rm.output_decl(p, role=role, dpi=dpi)
                 for p, role, dpi in outputs],
        results=results, notes=notes,
        require_modules=('d1_loo_blind', 'r1_calibration_core', 'rev2_manifest'),
        allow_undeclared_outputs=True)


if __name__ == '__main__':
    main()
