"""R1 baseline-diffusivity calibration study.

Runs the whole study from one config file:

    python scripts/manuscript_well_leakage/baseline_calibration/r1_run_calibration.py \
        --config configs/r1_baseline_calibration.json

Writes the arrays .npz, the run manifest .json, and the misfit-curve .png to the
paths named in the config. Nothing about the study is hard-coded here: the
window, gauge paths, sweep ranges, metric definitions and output paths all come
from the config, so re-running the committed config reproduces the numbers.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_calibration_core as core  # noqa: E402


def load_config(path):
    with open(path, 'rb') as fh:
        raw = fh.read()
    return json.loads(raw.decode('utf-8')), hashlib.sha256(raw).hexdigest()


def log(msg):
    print(f"[r1] {msg}", flush=True)


def load_window_data(cfg):
    from fiberis.analyzer.Data1D import Data1D_Gauge
    from fiberis.analyzer.Geometry3D import DataG3D_md

    w = cfg['window']
    t_start = datetime.datetime.fromisoformat(w['time_start'])
    t_end = datetime.datetime.fromisoformat(w['time_end'])

    md_frame = DataG3D_md.G3DMeasuredDepth()
    md_frame.load_npz(cfg['data']['gauge_md_npz'])
    all_md = np.asarray(md_frame.data, dtype=float)
    in_window = (all_md >= w['md_min_ft']) & (all_md <= w['md_max_ft'])
    gauge_numbers = np.where(in_window)[0] + 1  # gauge n <-> index n-1
    gauge_mds = all_md[in_window]

    fh_frame = DataG3D_md.G3DMeasuredDepth()
    fh_frame.load_npz(cfg['data']['frac_hit_stage1_npz'])
    frac_hits = np.unique(np.asarray(fh_frame.data, dtype=float))

    series = {}
    for n, md in zip(gauge_numbers, gauge_mds):
        gframe = Data1D_Gauge.Data1DGauge()
        gframe.load_npz(cfg['data']['gauge_series_template'].format(n=n))
        gframe.crop(t_start, t_end)
        taxis = np.asarray(gframe.taxis, dtype=float)
        taxis = taxis - taxis[0]
        raw = np.asarray(gframe.data, dtype=float)
        # A frame carrying exactly the series the study uses, so the fibeRIS
        # cross-check runs on the same numbers rather than on raw ~8300 psi.
        delta_frame = Data1D_Gauge.Data1DGauge()
        delta_frame.load_npz(cfg['data']['gauge_series_template'].format(n=n))
        delta_frame.crop(t_start, t_end)
        delta_frame.taxis = taxis
        delta_frame.data = raw - raw[0]
        series[int(n)] = {
            'gauge': int(n),
            'md_ft': float(md),
            'taxis': taxis,
            'raw_psi': raw,
            'delta_psi': raw - raw[0],
            'frame': delta_frame,
        }
    return series, gauge_numbers, gauge_mds, frac_hits, t_start, t_end


def pick_source_gauge(cfg, series, frac_hits):
    rule = cfg['source']['selection_rule']
    if rule != 'nearest_gauge_to_stage1_frac_hits':
        raise ValueError(f"Unsupported source selection_rule: {rule}")
    target_md = float(np.mean(frac_hits))
    best = min(series.values(), key=lambda s: abs(s['md_ft'] - target_md))
    return best['gauge'], target_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg, cfg_hash = load_config(args.config)
    log(f"config {args.config} sha256={cfg_hash[:16]}")

    # All data/output paths in the config are repo-relative by design, so the
    # study must run with CWD = repo root. Fail here with an actionable message
    # rather than deep inside a loader.
    missing = [p for p in (cfg['data']['gauge_md_npz'],
                           cfg['data']['frac_hit_stage1_npz'])
               if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "Config data paths are repo-relative; run this from the repository "
            f"root. Not found from CWD={os.getcwd()}: {missing}")

    series, gauge_numbers, gauge_mds, frac_hits, t_start, t_end = load_window_data(cfg)
    log(f"in-window gauges {list(gauge_numbers)} at MD {list(gauge_mds)}")
    log(f"stage-1 frac hits {list(frac_hits)}")

    src_gauge, fh_centroid = pick_source_gauge(cfg, series, frac_hits)
    log(f"source gauge = {src_gauge} (MD {series[src_gauge]['md_ft']}), "
        f"frac-hit centroid MD {fh_centroid:.2f}")

    m = cfg['mesh']
    pad_lo = float(m.get('domain_pad_low_md_ft', 0.0))
    pad_hi = float(m.get('domain_pad_high_md_ft', 0.0))
    # The window bounds say where gauges are compared; the solver domain is
    # padded so the no-flux boundary cannot reflect onto the farthest target.
    mesh = np.arange(cfg['window']['md_min_ft'] - pad_lo,
                     cfg['window']['md_max_ft'] + pad_hi + m['dx_ft'] / 2.0,
                     m['dx_ft'])
    mesh_window = np.arange(cfg['window']['md_min_ft'],
                            cfg['window']['md_max_ft'] + m['dx_ft'] / 2.0,
                            m['dx_ft'])
    log(f"domain MD [{mesh[0]:.0f}, {mesh[-1]:.0f}] (nx={len(mesh)}), "
        f"pad_low={pad_lo:.0f} ft; comparison window MD "
        f"[{cfg['window']['md_min_ft']:.0f}, {cfg['window']['md_max_ft']:.0f}]")
    # Graded-profile taper endpoints are PHYSICAL MDs (the comparison window),
    # not mesh ends, so the graded model does not change meaning when the domain
    # is padded for boundary reasons.
    taper_lo = float(cfg['sweeps']['graded'].get(
        'taper_lo_md_ft', cfg['window']['md_min_ft']))
    taper_hi = float(cfg['sweeps']['graded'].get(
        'taper_hi_md_ft', cfg['window']['md_max_ft']))
    log(f"graded taper pinned to MD [{taper_lo:.0f}, {taper_hi:.0f}] "
        "(mesh-independent)")
    source_idx = int(np.argmin(np.abs(mesh - series[src_gauge]['md_ft'])))
    src = series[src_gauge]
    t_total = float(src['taxis'][-1])

    targets = []
    for n in sorted(series):
        if n == src_gauge:
            continue
        s = series[n]
        targets.append({
            'gauge': n,
            'md_ft': s['md_ft'],
            'distance_ft': abs(s['md_ft'] - src['md_ft']),
            'idx': int(np.argmin(np.abs(mesh - s['md_ft']))),
            'taxis': s['taxis'],
            'data': s['delta_psi'],
        })
    log(f"targets: gauges {[t['gauge'] for t in targets]} "
        f"at distances {[t['distance_ft'] for t in targets]} ft")

    sol = cfg['solver']
    thr = cfg['metrics']['arrival_time']['threshold_frac']

    # ---- kernel equivalence proof (gates everything downstream) -------------
    equiv, resid_checks = [], []
    src_idx_w = int(np.argmin(np.abs(mesh_window - series[src_gauge]['md_ft'])))
    for d_test in (200.0, 500.0, 1200.0, 2000.0):
        prof = core.build_uniform_profile(mesh_window, d_test)
        rec_e = core.verify_against_fiberis(mesh_window, prof,
                                            dt_used_probe := 10.0,
                                            200.0, src['frame'], src_idx_w)
        rec_e.update({'model': 'uniform', 'param': d_test, 'dt_s': dt_used_probe})
        equiv.append(rec_e)
        resid = core.dense_residual_check(mesh_window, prof, dt_used_probe,
                                          src['frame'], src_idx_w)
        resid.update({'model': 'uniform', 'param': d_test})
        resid_checks.append(resid)
    for dmax in (840.0, 2000.0):
        prof = core.build_triangular_profile(
            mesh_window, dmax, cfg['sweeps']['graded']['d_min_over_d_max'],
            src_idx_w, taper_lo, taper_hi)
        rec_e = core.verify_against_fiberis(mesh_window, prof, 10.0, 200.0,
                                           src['frame'], src_idx_w)
        rec_e.update({'model': 'triangular', 'param': dmax, 'dt_s': 10.0})
        equiv.append(rec_e)
        resid = core.dense_residual_check(mesh_window, prof, 10.0,
                                          src['frame'], src_idx_w)
        resid.update({'model': 'triangular', 'param': dmax})
        resid_checks.append(resid)

    worst_abs = max(e['max_abs_diff_psi'] for e in equiv)
    worst_rel = max(e['max_relative_diff'] for e in equiv)
    tol_rel = sol['fast_kernel_equivalence_rel_tol']
    all_tridiag = all(r['matrix_is_exactly_tridiagonal'] for r in resid_checks)
    worst_resid = max(max(r['residual_inf_dense'], r['residual_inf_banded'])
                      for r in resid_checks)
    log(f"kernel equivalence: worst abs {worst_abs:.3e} psi, "
        f"worst relative {worst_rel:.3e} (tol {tol_rel:g}); "
        f"fibeRIS matrix exactly tridiagonal for all cases: {all_tridiag}; "
        f"worst ||Au-b||_inf either solver {worst_resid:.3e}")
    if not all_tridiag:
        raise SystemExit("fibeRIS matrix is not tridiagonal; fast kernel invalid.")
    if worst_rel > tol_rel:
        raise SystemExit(f"Fast kernel does not match fibeRIS "
                         f"(worst relative {worst_rel:.3e} > {tol_rel:g}).")

    def score(diffusivity, dt):
        return core.evaluate_profile(mesh, diffusivity, dt, t_total,
                                     src['taxis'], src['delta_psi'],
                                     source_idx, targets, thr)

    # The gate that actually matters: does the round-off move the reported metric?
    metric_checks = []
    for d_test in (300.0, 1000.0, 2000.0):
        targets_w = [dict(t, idx=int(np.argmin(np.abs(mesh_window - t['md_ft']))))
                     for t in targets]
        mc = core.metric_equivalence_check(
            mesh_window, core.build_uniform_profile(mesh_window, d_test), 10.0,
            t_total, src['frame'], src_idx_w, targets_w, thr)
        mc['model'] = 'uniform'
        mc['param'] = d_test
        metric_checks.append(mc)
        log(f"metric check D={d_test:.0f}: RMSE fibeRIS {mc['rmse_fiberis_psi']:.9f} "
            f"vs fast {mc['rmse_fast_psi']:.9f} psi "
            f"(|diff| {mc['abs_rmse_difference_psi']:.3e})")
    # The gate above uses dt=10 s; round-off accumulates with step count, so
    # also check at the production dt over the full window on the compact mesh
    # (the dense fibeRIS solve is O(nx^3), infeasible on the padded mesh).
    for d_test in cfg['solver'].get('production_dt_equivalence_D', [1150.0]):
        mc = core.metric_equivalence_check(
            mesh_window, core.build_uniform_profile(mesh_window, float(d_test)),
            sol['dt_s'], t_total, src['frame'], src_idx_w, targets_w, thr)
        mc['model'] = 'uniform'
        mc['param'] = float(d_test)
        mc['note'] = 'production dt, full duration'
        metric_checks.append(mc)
        log(f"metric check D={d_test:.0f} at PRODUCTION dt={sol['dt_s']} s "
            f"({int(t_total/sol['dt_s'])} steps): |dRMSE| "
            f"{mc['abs_rmse_difference_psi']:.3e} psi")

    worst_metric = max(mc['abs_rmse_difference_psi'] for mc in metric_checks)
    tol_metric = sol['metric_equivalence_tol_psi']
    if worst_metric > tol_metric:
        raise SystemExit(f"Kernel round-off moves the reported RMSE by "
                         f"{worst_metric:.3e} psi > {tol_metric:g}.")
    log(f"metric equivalence worst |dRMSE| = {worst_metric:.3e} psi "
        f"(tol {tol_metric:g}) -> fast kernel adopted for sweeps")

    # ---- dt convergence ----------------------------------------------------
    dt_rows = []
    for dt in sol['dt_convergence_check_s']:
        r = score(core.build_uniform_profile(mesh, 500.0), dt)
        dt_rows.append({'dt_s': dt, 'rmse_psi': r['rmse_pooled_psi']})
        log(f"dt {dt:>5} s -> pooled RMSE {r['rmse_psi'] if False else r['rmse_pooled_psi']:.6f} psi")
    dt_used = sol['dt_s']
    finest = dt_rows[-1]['rmse_psi']
    at_used = [row for row in dt_rows if row['dt_s'] == dt_used][0]['rmse_psi']
    log(f"dt discretisation error at dt={dt_used}s vs {dt_rows[-1]['dt_s']}s: "
        f"{abs(at_used - finest):.4f} psi ({abs(at_used-finest)/finest*100:.3f}%)")

    # ---- domain-padding convergence ---------------------------------------
    pad_rows = []
    for pad in m.get('padding_convergence_check_ft', [pad_lo]):
        mp = np.arange(cfg['window']['md_min_ft'] - float(pad),
                       cfg['window']['md_max_ft'] + pad_hi + m['dx_ft'] / 2.0,
                       m['dx_ft'])
        sip = int(np.argmin(np.abs(mp - src['md_ft'])))
        tp = [dict(t, idx=int(np.argmin(np.abs(mp - t['md_ft'])))) for t in targets]
        rp = core.evaluate_profile(mp, core.build_uniform_profile(mp, 2000.0),
                                   dt_used, t_total, src['taxis'],
                                   src['delta_psi'], sip, tp, thr)
        far = rp['per_gauge'][-1]
        pad_rows.append({'pad_ft': float(pad), 'nx': int(len(mp)),
                         'rmse_pooled_psi_at_D2000': rp['rmse_pooled_psi'],
                         'farthest_gauge': far['gauge'],
                         'farthest_gauge_sim_max_psi': far['sim_max_psi'],
                         'farthest_gauge_obs_max_psi': far['obs_max_psi']})
        log(f"padding {pad:>6.0f} ft (nx={len(mp)}): pooled RMSE at D=2000 "
            f"{rp['rmse_pooled_psi']:.4f} psi, g{far['gauge']} sim max "
            f"{far['sim_max_psi']:.2f} vs obs {far['obs_max_psi']:.2f} psi")
    if len(pad_rows) > 1:
        log(f"padding artifact removed: RMSE at D=2000 changes "
            f"{pad_rows[0]['rmse_pooled_psi_at_D2000']:.2f} -> "
            f"{pad_rows[-1]['rmse_pooled_psi_at_D2000']:.2f} psi "
            f"from pad 0 to {pad_rows[-1]['pad_ft']:.0f} ft")

    # ---- sweeps ------------------------------------------------------------
    def run_sweep(kind, grid):
        rows, per_gauge_mse, per_gauge_norm2 = [], [], []
        for val in grid:
            if kind == 'uniform':
                prof = core.build_uniform_profile(mesh, val)
            else:
                prof = core.build_triangular_profile(
                    mesh, val, cfg['sweeps']['graded']['d_min_over_d_max'],
                    source_idx, taper_lo, taper_hi)
            r = score(prof, dt_used)
            mse = np.array([g['rmse_psi'] ** 2 for g in r['per_gauge']])
            per_gauge_mse.append(mse)
            per_gauge_norm2.append(np.array(
                [(g['rmse_psi'] / g['obs_max_psi']) ** 2 for g in r['per_gauge']]))
            rows.append({
                'value': float(val),
                'profile_mean': float(np.mean(prof)),
                'rmse_pooled_psi': r['rmse_pooled_psi'],
                'rmse_gaugemean_psi': float(np.sqrt(np.mean(mse))),
                'rmse_normalized': r['rmse_normalized'],
                'arrival_err_mean_s': r['arrival_err_mean_s'],
                'arrival_err_absmean_s': r['arrival_err_absmean_s'],
                'amplitude_ratio_mean': r['amplitude_ratio_mean'],
                'per_gauge': r['per_gauge'],
            })
        return rows, np.asarray(per_gauge_mse), np.asarray(per_gauge_norm2)

    sw = cfg['sweeps']
    grid_u = np.logspace(np.log10(sw['uniform']['min']), np.log10(sw['uniform']['max']),
                         sw['uniform']['n_points'])
    grid_g = np.logspace(np.log10(sw['graded']['min']), np.log10(sw['graded']['max']),
                         sw['graded']['n_points'])

    log(f"uniform sweep: {len(grid_u)} points {grid_u[0]:.1f}..{grid_u[-1]:.1f}")
    rows_u, mse_u, norm_u = run_sweep('uniform', grid_u)
    log(f"graded sweep: {len(grid_g)} points")
    rows_g, mse_g, norm_g = run_sweep('graded', grid_g)

    # refine around each minimum
    ref = sw['refine_minimum']
    extra = {}
    if ref['enabled']:
        for kind, rows, grid in (('uniform', rows_u, grid_u), ('graded', rows_g, grid_g)):
            key = 'rmse_gaugemean_psi'
            best = min(rows, key=lambda r: r[key])['value']
            lo = np.log10(best) - ref['half_width_decades']
            hi = np.log10(best) + ref['half_width_decades']
            g2 = np.logspace(lo, hi, ref['n_points'])
            r2, m2, n2 = run_sweep(kind, g2)
            extra[kind] = (g2, r2, m2, n2)
            log(f"{kind} refine around {best:.1f}: "
                f"{g2[0]:.1f}..{g2[-1]:.1f} ({len(g2)} pts)")

    def merge(grid, rows, mse, norm2, kind):
        if kind in extra:
            g2, r2, m2, n2 = extra[kind]
            grid = np.concatenate([grid, g2])
            rows = rows + r2
            mse = np.vstack([mse, m2])
            norm2 = np.vstack([norm2, n2])
        order = np.argsort(grid)
        uniq = np.concatenate([[True], np.diff(grid[order]) > 1e-9])
        keep = order[uniq]
        return grid[keep], [rows[i] for i in keep], mse[keep], norm2[keep]

    grid_u, rows_u, mse_u, norm_u = merge(grid_u, rows_u, mse_u, norm_u, 'uniform')
    grid_g, rows_g, mse_g, norm_g = merge(grid_g, rows_g, mse_g, norm_g, 'graded')

    # ---- free-ratio (2-parameter) triangular variant ----------------------
    ratio_grid = np.array(cfg['sweeps']['graded_free_ratio']['ratio_grid'])
    free_rows = []
    for rat in ratio_grid:
        for val in grid_g[::2]:
            prof = core.build_triangular_profile(mesh, val, rat, source_idx,
                                                taper_lo, taper_hi)
            r = score(prof, dt_used)
            mse = np.array([g['rmse_psi'] ** 2 for g in r['per_gauge']])
            free_rows.append({'d_max': float(val), 'ratio': float(rat),
                              'rmse_gaugemean_psi': float(np.sqrt(np.mean(mse))),
                              'profile_mean': float(np.mean(prof))})
    best_free = min(free_rows, key=lambda r: r['rmse_gaugemean_psi'])
    best_free['ratio_at_grid_edge'] = bool(
        best_free['ratio'] == ratio_grid.min() or best_free['ratio'] == ratio_grid.max())
    log(f"free-ratio best: D_max={best_free['d_max']:.1f} ratio={best_free['ratio']:.3f} "
        f"RMSE={best_free['rmse_gaugemean_psi']:.3f} "
        f"(ratio at grid edge: {best_free['ratio_at_grid_edge']})")

    # ---- per-gauge single-gauge best-fit D --------------------------------
    # If one homogeneous D described this window, every gauge would prefer the
    # same D. Fitting each gauge alone separates "wrong D" from "wrong model".
    pg_cfg = cfg['diagnostics']['single_gauge_fit']
    pg_grid = np.logspace(np.log10(pg_cfg['min']), np.log10(pg_cfg['max']),
                          pg_cfg['n_points'])
    pg_curves = np.zeros((len(pg_grid), len(targets)))
    for i, val in enumerate(pg_grid):
        r = score(core.build_uniform_profile(mesh, val), dt_used)
        pg_curves[i] = [g['rmse_psi'] for g in r['per_gauge']]
    single_gauge = []
    for k, tgt in enumerate(targets):
        j = int(np.argmin(pg_curves[:, k]))
        single_gauge.append({
            'gauge': tgt['gauge'],
            'distance_ft': tgt['distance_ft'],
            'best_D': float(pg_grid[j]),
            'own_rmse_psi': float(pg_curves[j, k]),
            'at_grid_edge': bool(j in (0, len(pg_grid) - 1)),
        })
    sg_vals = np.array([x['best_D'] for x in single_gauge])
    sg_spread = float(sg_vals.max() / sg_vals.min())
    log(f"single-gauge best-fit D: " +
        ", ".join(f"g{x['gauge']}={x['best_D']:.0f}" for x in single_gauge) +
        f" -> spread {sg_spread:.1f}x, own RMSE "
        f"{min(x['own_rmse_psi'] for x in single_gauge):.1f}-"
        f"{max(x['own_rmse_psi'] for x in single_gauge):.1f} psi")

    # ---- solver-free erfc cross-check on the single-gauge spread -----------
    src_amp = float(np.max(src['delta_psi']))
    erfc_rows = []
    for tgt in targets:
        ratio = float(np.max(tgt['data'])) / src_amp
        erfc_rows.append({
            'gauge': tgt['gauge'], 'distance_ft': tgt['distance_ft'],
            'amplitude_ratio_to_source': ratio,
            'erfc_implied_D': core.erfc_implied_diffusivity(
                tgt['distance_ft'], ratio, t_total),
        })
    ev = np.array([r['erfc_implied_D'] for r in erfc_rows], dtype=float)
    log("erfc-implied D: " + ", ".join(
        f"g{r['gauge']}={r['erfc_implied_D']:.0f}" for r in erfc_rows) +
        f" -> spread {np.nanmax(ev)/np.nanmin(ev):.1f}x")

    # ---- arrival-threshold robustness -------------------------------------
    abs_thr = cfg['metrics']['arrival_time']['absolute_thresholds_psi_check']
    prof_b = core.build_uniform_profile(mesh, sum_u_pre_best := float(
        grid_u[int(np.argmin([r['rmse_gaugemean_psi'] for r in rows_u]))]))
    tx_b, rec_b = core.solve_forward(mesh, prof_b, dt_used, t_total,
                                     src['taxis'], src['delta_psi'], source_idx,
                                     record_idx=[t['idx'] for t in targets])
    arr_rows = []
    for k, tgt in enumerate(targets):
        r = core.arrival_robustness(tx_b, rec_b[:, k], tgt['taxis'], tgt['data'],
                                    abs_thr, thr)
        r.update({'gauge': tgt['gauge'], 'distance_ft': tgt['distance_ft']})
        arr_rows.append(r)
    signs = {key: [np.sign(r[key]) for r in arr_rows]
             for key in arr_rows[0] if key.startswith(('relative', 'abs_'))}
    consistent = all(v == list(signs.values())[0] for v in signs.values())
    log(f"arrival sign pattern identical across all thresholds "
        f"({', '.join(signs)}): {consistent}")

    # ---- right-boundary decoupling check ----------------------------------
    hi_trim = cfg['mesh'].get('high_side_trim_check_ft', [])
    trim_rows = []
    ref_vals = None
    for trim in hi_trim:
        mt = np.arange(mesh[0], cfg['window']['md_max_ft'] - float(trim)
                       + m['dx_ft'] / 2.0, m['dx_ft'])
        if mt[-1] < src['md_ft']:
            continue
        sit = int(np.argmin(np.abs(mt - src['md_ft'])))
        it = [int(np.argmin(np.abs(mt - t['md_ft']))) for t in targets]
        txt, rct = core.solve_forward(mt, core.build_uniform_profile(mt, 1150.0),
                                      dt_used, t_total, src['taxis'],
                                      src['delta_psi'], sit, record_idx=it)
        vals = rct.max(axis=0)
        if ref_vals is None:
            ref_vals = vals
        trim_rows.append({'trim_ft': float(trim), 'md_max_ft': float(mt[-1]),
                          'max_abs_diff_vs_untrimmed_psi':
                              float(np.max(np.abs(vals - ref_vals)))})
    if trim_rows:
        log(f"high-MD trim: max target change "
            f"{max(r['max_abs_diff_vs_untrimmed_psi'] for r in trim_rows):.2e} psi "
            f"-> Dirichlet source decouples the high-MD side")

    # ---- is the minimum interior to the specified range? ------------------
    ext = cfg['sweeps'].get('range_adequacy_check_D', [])
    ext_rows = []
    for D in ext:
        r = score(core.build_uniform_profile(mesh, float(D)), dt_used)
        mse = np.array([g['rmse_psi'] ** 2 for g in r['per_gauge']])
        ext_rows.append({'D': float(D),
                         'rmse_gaugemean_psi': float(np.sqrt(np.mean(mse)))})
    if ext_rows:
        log("beyond-range check: " + ", ".join(
            f"D={r['D']:.0f}->{r['rmse_gaugemean_psi']:.1f}" for r in ext_rows))

    # ---- summarise a curve ------------------------------------------------
    def summarise(grid, rows, mse, label, key='rmse_gaugemean_psi'):
        curve = np.array([r[key] for r in rows])
        i = int(np.argmin(curve))
        best, best_rmse = float(grid[i]), float(curve[i])

        rise = cfg['uncertainty']['misfit_rise_fraction']
        thresh = best_rmse * (1.0 + rise)
        within = np.where(curve <= thresh)[0]
        band = (float(grid[within[0]]), float(grid[within[-1]]))
        band_open_low = bool(within[0] == 0)
        band_open_high = bool(within[-1] == len(grid) - 1)

        # parabola in log10 D through the 5 points nearest the minimum
        lo, hi = max(0, i - 2), min(len(grid), i + 3)
        x = np.log10(grid[lo:hi])
        coef = np.polyfit(x, curve[lo:hi], 2)
        curv_log = float(2.0 * coef[0])          # d2 RMSE / d(log10 D)2
        vertex = float(10 ** (-coef[1] / (2 * coef[0]))) if coef[0] != 0 else np.nan

        # Gauge-level bootstrap, enumerated EXACTLY (462 multisets for 6 gauges)
        # rather than sampled: a finite draw can report probability 0 or 1 for an
        # event that actually has support.
        n_g = mse.shape[1]
        boot = core.exact_gauge_bootstrap(grid, mse)
        argmins = boot['argmins']
        ci = tuple(boot['ci95'])
        ci_censored_low = bool(boot['mass_at_grid_min'] > 1e-12
                               and ci[0] <= grid[0])
        ci_censored_high = bool(boot['mass_at_grid_max'] > 1e-12
                                and ci[1] >= grid[-1])
        if ci_censored_high or ci_censored_low:
            log(f"  WARNING {label}: bootstrap CI95 is CENSORED at the grid "
                f"({boot['mass_at_grid_max']*100:.1f}% of exact mass sits on the "
                f"ceiling {grid[-1]:.0f}); quote CI68 "
                f"[{boot['ci68'][0]:.0f}, {boot['ci68'][1]:.0f}] instead")

        # Leave-one-gauge-out argmin: with only 6 target gauges this says more
        # about how much any single gauge drives the answer than the bootstrap.
        loo = []
        for drop in range(n_g):
            keep_cols = [c for c in range(n_g) if c != drop]
            curve_loo = np.sqrt(np.mean(mse[:, keep_cols], axis=1))
            loo.append({'dropped_gauge': int(targets[drop]['gauge']),
                        'argmin': float(grid[int(np.argmin(curve_loo))])})
        loo_vals = np.array([x['argmin'] for x in loo])

        # Does the answer depend on how residuals are weighted?
        pooled = np.array([r['rmse_pooled_psi'] for r in rows])
        argmin_pooled = float(grid[int(np.argmin(pooled))])

        refs = {}
        for rv in cfg['uncertainty']['reference_values_to_report']:
            if grid[0] <= rv <= grid[-1]:
                r_rmse = float(np.interp(rv, grid, curve))
                refs[str(rv)] = {
                    'rmse_psi': r_rmse,
                    'excess_over_min_psi': r_rmse - best_rmse,
                    'excess_fraction': r_rmse / best_rmse - 1.0,
                    'inside_10pct_band': bool(band[0] <= rv <= band[1]),
                    'exact_bootstrap_fraction_of_argmins_above':
                        core.exact_fraction_above(boot, rv),
                }
        log(f"{label}: min at {best:.1f} (RMSE {best_rmse:.3f} psi), "
            f"10% band [{band[0]:.0f}, {band[1]:.0f}]"
            f"{' OPEN-LOW' if band_open_low else ''}"
            f"{' OPEN-HIGH' if band_open_high else ''}, "
            f"bootstrap CI95 [{ci[0]:.0f}, {ci[1]:.0f}]"
            f"{' (CENSORED)' if (ci_censored_high or ci_censored_low) else ''}"
            f", CI68 [{boot['ci68'][0]:.0f}, {boot['ci68'][1]:.0f}]")
        return {
            'label': label, 'grid': grid, 'curve': curve, 'rows': rows,
            'per_gauge_mse': mse, 'best': best, 'best_rmse': best_rmse,
            'best_index': i, 'band_10pct': band,
            'band_open_low': band_open_low, 'band_open_high': band_open_high,
            'curvature_per_log10_decade': curv_log, 'parabola_vertex': vertex,
            'bootstrap_argmins': argmins, 'bootstrap_ci95': ci,
            'bootstrap_method': 'exact_enumeration',
            'bootstrap_n_multisets': boot['n_multisets'],
            'bootstrap_ci68': boot['ci68'],
            'bootstrap_median': boot['median'],
            'bootstrap_ci95_censored_low': ci_censored_low,
            'bootstrap_ci95_censored_high': ci_censored_high,
            'bootstrap_mass_at_grid_max': boot['mass_at_grid_max'],
            'bootstrap_mass_at_grid_min': boot['mass_at_grid_min'],
            'leave_one_gauge_out': loo,
            'leave_one_gauge_out_range': [float(loo_vals.min()), float(loo_vals.max())],
            'argmin_pooled_weighting': argmin_pooled,
            'argmin_shift_from_weighting': argmin_pooled - best,
            'reference_values': refs,
            'profile_mean_at_best': float(rows[i]['profile_mean']),
            'best_row': rows[i],
        }

    sum_u = summarise(grid_u, rows_u, mse_u, 'uniform')
    sum_g = summarise(grid_g, rows_g, mse_g, 'graded')
    # Same curves under the amplitude-normalised norm.
    sum_u_n = summarise(grid_u, rows_u, norm_u, 'uniform (normalised)',
                        key='rmse_normalized')
    sum_g_n = summarise(grid_g, rows_g, norm_g, 'graded (normalised)',
                        key='rmse_normalized')
    log(f"norm-sensitivity: uniform argmin {sum_u['best']:.0f} (psi RMSE) vs "
        f"{sum_u_n['best']:.0f} (normalised); graded {sum_g['best']:.0f} vs "
        f"{sum_g_n['best']:.0f}")

    # ---- model comparison --------------------------------------------------
    # Effective sample size: deflate pooled N by the residual autocorrelation
    # time at the best uniform fit, so AIC is not inflated by oversampling.
    prof_best = core.build_uniform_profile(mesh, sum_u['best'])
    rec = core.evaluate_profile(mesh, prof_best, dt_used, t_total, src['taxis'],
                               src['delta_psi'], source_idx, targets, thr)
    n_pool = rec['n_residuals']
    ac_lengths = []
    for k, tgt in enumerate(targets):
        taxis_s, r_s = core.solve_forward(mesh, prof_best, dt_used, t_total,
                                          src['taxis'], src['delta_psi'],
                                          source_idx, record_idx=[tgt['idx']])
        resid = np.interp(tgt['taxis'], taxis_s, r_s[:, 0]) - tgt['data']
        resid = resid - resid.mean()
        denom = np.sum(resid ** 2)
        if denom <= 0:
            continue
        ac = np.correlate(resid, resid, mode='full')[len(resid) - 1:] / denom
        first = np.where(ac < 1.0 / np.e)[0]
        ac_lengths.append(float(first[0]) if first.size else float(len(ac)))
    ac_mean = float(np.mean(ac_lengths)) if ac_lengths else 1.0
    n_eff = max(len(targets), int(round(n_pool / max(ac_mean, 1.0))))

    def aic(rmse, k, n):
        return n * np.log(rmse ** 2) + 2 * k

    models = {
        'uniform_D': {'k': 1, 'rmse': sum_u['best_rmse'], 'best_param': sum_u['best']},
        'graded_ratio_fixed': {'k': 1, 'rmse': sum_g['best_rmse'], 'best_param': sum_g['best']},
        'graded_ratio_free': {'k': 2, 'rmse': best_free['rmse_gaugemean_psi'],
                              'best_param': best_free['d_max'],
                              'best_ratio': best_free['ratio']},
    }
    for name, mm in models.items():
        mm['aic_neff'] = float(aic(mm['rmse'], mm['k'], n_eff))
        mm['aic_npooled'] = float(aic(mm['rmse'], mm['k'], n_pool))
    best_aic = min(m['aic_neff'] for m in models.values())
    for mm in models.values():
        mm['delta_aic_neff'] = mm['aic_neff'] - best_aic
    log(f"autocorr length {ac_mean:.1f} samples -> n_eff {n_eff} (pooled N {n_pool})")
    for name, mm in models.items():
        log(f"  {name:22s} k={mm['k']} RMSE={mm['rmse']:.4f} dAIC={mm['delta_aic_neff']:+.2f}")

    # ---- interval method spread --------------------------------------------
    # The +10% band is a convention whose confidence content depends entirely on
    # the assumed sample size, so report the spread across conventions instead of
    # a single "CI".
    n_conv = {'gauges': len(targets), 'autocorr_neff': n_eff,
              'pooled_residuals': n_pool}
    refs_to_test = cfg['uncertainty']['reference_values_to_report']
    spread = {
        'uniform_absolute_norm': core.interval_method_spread(
            sum_u['grid'], sum_u['curve'], sum_u['per_gauge_mse'], n_conv,
            refs_to_test),
        'uniform_normalised_norm': core.interval_method_spread(
            sum_u_n['grid'], sum_u_n['curve'], norm_u, n_conv, refs_to_test),
    }
    for norm_label, sp in spread.items():
        for conv in ('gauges', 'autocorr_neff'):
            if conv in sp:
                e = sp[conv]
                r480 = e['reference_values'].get('480.0', {})
                log(f"  {norm_label} F-95% @n={e['n']}: "
                    f"[{e['interval'][0]:.0f}, {e['interval'][1]:.0f}]"
                    f"{' OPEN-HIGH' if e['open_high'] else ''}"
                    f"  (+{e['percent_rmse_rise_at_95']:.1f}% RMSE); "
                    f"480 inside={r480.get('inside_95')} p={r480.get('p_value', float('nan')):.4f}")
        jk = sp['jackknife_log10']
        log(f"  {norm_label} jackknife 95%: "
            f"[{jk['interval'][0]:.0f}, {jk['interval'][1]:.0f}]")

    # ---- figure ------------------------------------------------------------
    fig_path = cfg['outputs']['figure_png']
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    make_figure(cfg, fig_path, sum_u, sum_g, rows_u, targets, mesh, source_idx,
                src, dt_used, t_total, thr, sum_u_n, sum_g_n, single_gauge)
    log(f"wrote {fig_path}")

    # ---- arrays ------------------------------------------------------------
    npz_path = cfg['outputs']['arrays_npz']
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez(
        npz_path,
        uniform_D_grid=sum_u['grid'],
        uniform_rmse_gaugemean=sum_u['curve'],
        uniform_rmse_pooled=np.array([r['rmse_pooled_psi'] for r in rows_u]),
        uniform_arrival_err_mean_s=np.array([r['arrival_err_mean_s'] for r in rows_u]),
        uniform_arrival_err_absmean_s=np.array([r['arrival_err_absmean_s'] for r in rows_u]),
        uniform_amplitude_ratio_mean=np.array([r['amplitude_ratio_mean'] for r in rows_u]),
        uniform_rmse_normalized=np.array([r['rmse_normalized'] for r in rows_u]),
        graded_rmse_normalized_curve=np.array([r['rmse_normalized'] for r in rows_g]),
        uniform_per_gauge_mse=sum_u['per_gauge_mse'],
        uniform_bootstrap_argmins=sum_u['bootstrap_argmins'],
        graded_Dmax_grid=sum_g['grid'],
        graded_rmse_gaugemean=sum_g['curve'],
        graded_rmse_pooled=np.array([r['rmse_pooled_psi'] for r in rows_g]),
        graded_profile_mean=np.array([r['profile_mean'] for r in rows_g]),
        graded_arrival_err_mean_s=np.array([r['arrival_err_mean_s'] for r in rows_g]),
        graded_amplitude_ratio_mean=np.array([r['amplitude_ratio_mean'] for r in rows_g]),
        graded_per_gauge_mse=sum_g['per_gauge_mse'],
        graded_bootstrap_argmins=sum_g['bootstrap_argmins'],
        free_ratio_dmax=np.array([r['d_max'] for r in free_rows]),
        free_ratio_ratio=np.array([r['ratio'] for r in free_rows]),
        free_ratio_rmse=np.array([r['rmse_gaugemean_psi'] for r in free_rows]),
        erfc_implied_D=np.array([r['erfc_implied_D'] for r in erfc_rows]),
        range_check_D=np.array([r['D'] for r in ext_rows]) if ext_rows else np.array([]),
        range_check_rmse=np.array(
            [r['rmse_gaugemean_psi'] for r in ext_rows]) if ext_rows else np.array([]),
        single_gauge_D_grid=pg_grid,
        single_gauge_rmse_curves=pg_curves,
        single_gauge_best_D=sg_vals,
        target_gauges=np.array([t['gauge'] for t in targets]),
        target_md_ft=np.array([t['md_ft'] for t in targets]),
        target_distance_ft=np.array([t['distance_ft'] for t in targets]),
        source_gauge=np.array([src_gauge]),
        source_md_ft=np.array([src['md_ft']]),
        mesh=mesh,
        padding_check_pad_ft=np.array([r['pad_ft'] for r in pad_rows]),
        padding_check_rmse_at_D2000=np.array(
            [r['rmse_pooled_psi_at_D2000'] for r in pad_rows]),
        dt_convergence_dt=np.array([r['dt_s'] for r in dt_rows]),
        dt_convergence_rmse=np.array([r['rmse_psi'] for r in dt_rows]),
    )
    log(f"wrote {npz_path}")

    # ---- manifest ----------------------------------------------------------
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()
                    if k not in ('grid', 'curve', 'rows', 'per_gauge_mse',
                                 'bootstrap_argmins')}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    manifest = {
        'study_id': cfg['study_id'],
        'run_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'config_path': os.path.abspath(args.config),
        'config_sha256': cfg_hash,
        'config_resolved': cfg,
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'numpy': np.__version__,
            'scipy': __import__('scipy').__version__,
            'matplotlib': __import__('matplotlib').__version__,
            'fiberis_path': __import__('fiberis').__file__,
            'note': 'bakken_mariner/.git is empty; no commit hash is recoverable, '
                    'so code identity is pinned by the sha256 values below.',
            'code_sha256': {
                'runner': core.file_sha256(os.path.abspath(__file__)),
                'core': core.file_sha256(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'r1_calibration_core.py')),
            },
            'input_data_sha256': {
                k: core.file_sha256(v) for k, v in [
                    ('gauge_md_npz', cfg['data']['gauge_md_npz']),
                    ('frac_hit_stage1_npz', cfg['data']['frac_hit_stage1_npz'])]
            },
            'gauge_series_sha256': {
                str(n): core.file_sha256(
                    cfg['data']['gauge_series_template'].format(n=n))
                for n in sorted(series)
            },
            'cwd': os.getcwd(),
        },
        'kernel_equivalence_vs_fiberis': clean(equiv),
        'kernel_equivalence_worst_abs_psi': worst_abs,
        'kernel_equivalence_worst_relative': worst_rel,
        'dense_vs_banded_residual_checks': clean(resid_checks),
        'metric_equivalence_checks': clean(metric_checks),
        'window_resolved': {
            'gauges_in_window': [int(g) for g in gauge_numbers],
            'gauge_md_ft': [float(x) for x in gauge_mds],
            'stage1_frac_hit_md_ft': [float(x) for x in frac_hits],
            'frac_hit_centroid_md_ft': fh_centroid,
            'source_gauge': int(src_gauge),
            'source_md_ft': float(src['md_ft']),
            'source_mesh_idx': source_idx,
            'target_gauges': [t['gauge'] for t in targets],
            'target_distance_ft': [t['distance_ft'] for t in targets],
            't_total_s': t_total,
            'n_mesh': int(len(mesh)),
            'observed_max_delta_psi': {str(n): float(np.max(series[n]['delta_psi']))
                                       for n in sorted(series)},
            'sample_dt_s': {str(n): float(np.median(np.diff(series[n]['taxis'])))
                            for n in sorted(series)},
            'n_samples': {str(n): int(series[n]['taxis'].size) for n in sorted(series)},
        },
        'dt_convergence': dt_rows,
        'domain_padding_convergence': pad_rows,
        'domain': {'md_min_ft': float(mesh[0]), 'md_max_ft': float(mesh[-1]),
                   'nx': int(len(mesh)), 'pad_low_ft': pad_lo,
                   'pad_high_ft': pad_hi},
        'dt_used_s': dt_used,
        'results': {
            'uniform': clean(sum_u),
            'graded': clean(sum_g),
            'graded_free_ratio_best': best_free,
            'uniform_normalized_norm': clean(sum_u_n),
            'graded_normalized_norm': clean(sum_g_n),
        },
        'erfc_cross_check': erfc_rows,
        'arrival_threshold_robustness': {
            'rows': arr_rows,
            'sign_pattern_consistent_across_thresholds': bool(consistent),
        },
        'high_md_trim_check': trim_rows,
        'range_adequacy_check': ext_rows,
        'single_gauge_diagnostic': {
            'grid_min': float(pg_grid[0]), 'grid_max': float(pg_grid[-1]),
            'n_points': int(len(pg_grid)),
            'per_gauge': single_gauge,
            'spread_ratio': sg_spread,
            'interpretation': (
                'Each target gauge is individually fit to 6-19 psi RMSE, but the '
                'D required rises monotonically toward the source. A single '
                'homogeneous D is therefore not merely mis-tuned but '
                'mis-specified for this window.'),
        },
        'interval_method_spread': clean(spread),
        'model_comparison': {
            'residual_autocorr_length_samples': ac_mean,
            'n_residuals_pooled': n_pool,
            'n_effective': n_eff,
            'models': clean(models),
            'aic_caveat': (
                'For uniform vs graded_ratio_fixed both k=1, so dAIC reduces to '
                'n_eff*ln(RMSE ratio^2) and carries no information beyond the '
                'RMSE ratio itself. AIC also assumes residuals are noise, while '
                'here bias^2/MSE is 0.59-0.83 at five of six gauges. The AIC row '
                'for graded_ratio_free is additionally invalid because its '
                'optimum sits on the ratio grid boundary with misfit still '
                'decreasing (unidentified). Use the RMSE ratio and the residual '
                'structure, not these AIC values, as evidence.'),
            'bias_fraction_of_mse_per_gauge': {
                str(g['gauge']): float(g['bias_psi'] ** 2 / g['rmse_psi'] ** 2)
                for g in sum_u['best_row']['per_gauge']},
        },
        'outputs': cfg['outputs'],
    }
    man_path = cfg['outputs']['manifest_json']
    os.makedirs(os.path.dirname(man_path), exist_ok=True)
    with open(man_path, 'w') as fh:
        json.dump(manifest, fh, indent=2)
    log(f"wrote {man_path}")

    with open(os.path.join(os.path.dirname(man_path), 'r1_summary.txt'), 'w') as fh:
        fh.write(f"uniform best D            = {sum_u['best']:.1f} ft^2/s "
                 f"(RMSE {sum_u['best_rmse']:.3f} psi)\n")
        fh.write(f"uniform 10% band          = [{sum_u['band_10pct'][0]:.0f}, "
                 f"{sum_u['band_10pct'][1]:.0f}] ft^2/s\n")
        _cens = (' [RIGHT-CENSORED at grid ceiling - DO NOT QUOTE]'
                 if sum_u['bootstrap_ci95_censored_high'] else '')
        fh.write(f"uniform gauge-resample CI68 = [{sum_u['bootstrap_ci68'][0]:.0f}, "
                 f"{sum_u['bootstrap_ci68'][1]:.0f}] ft^2/s  (quote this one)\n")
        fh.write(f"uniform gauge-resample CI95 = [{sum_u['bootstrap_ci95'][0]:.0f}, "
                 f"{sum_u['bootstrap_ci95'][1]:.0f}] ft^2/s{_cens}\n")
        fh.write("  NOTE: gauge resampling is a sensitivity spread, NOT a confidence "
                 "interval (n=6, argmin functional, non-exchangeable units).\n")
        fh.write(f"graded best D_max         = {sum_g['best']:.1f} ft^2/s "
                 f"(RMSE {sum_g['best_rmse']:.3f} psi), "
                 f"D_min = {sum_g['best'] * cfg['sweeps']['graded']['d_min_over_d_max']:.1f}"
                 f"  [full-domain profile mean omitted: it is domain-dependent]\n")
        fh.write(f"uniform leave-one-out     = {sum_u['leave_one_gauge_out_range']}\n")
        fh.write(f"uniform argmin (pooled w) = {sum_u['argmin_pooled_weighting']:.1f}\n")
        fh.write(f"uniform argmin (normalised norm) = {sum_u_n['best']:.1f} ft^2/s\n")
        fh.write(f"graded  argmin (normalised norm) = {sum_g_n['best']:.1f} ft^2/s\n")
        fh.write("single-gauge best-fit D: " + ", ".join(
            f"g{x['gauge']}={x['best_D']:.0f}" for x in single_gauge) +
            f" (spread {sg_spread:.1f}x)\n")
        for rv, info in sum_u['reference_values'].items():
            fh.write(f"uniform D={rv:>6}: RMSE {info['rmse_psi']:.3f} psi "
                     f"(+{info['excess_fraction']*100:.1f}% over min), "
                     f"inside 10% band: {info['inside_10pct_band']}\n")
    log("done")


def make_figure(cfg, path, sum_u, sum_g, rows_u, targets, mesh, source_idx,
                src, dt_used, t_total, thr, sum_u_n=None, sum_g_n=None,
                single_gauge=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    refs = [(480.0, '--'), (140.0, ':'), (320.0, '-.'), (840.0, (0, (3, 1, 1, 1)))]

    def ref_lines(ax, label_y=0.97):
        """Reference D values, labelled in axes coordinates so the text cannot
        escape the axes when the limits change afterwards."""
        tr = ax.get_xaxis_transform()
        for rv, style in refs:
            if ax.get_xlim()[0] <= rv <= ax.get_xlim()[1]:
                ax.axvline(rv, color='0.35', ls=style, lw=1.0, alpha=0.85)
                ax.text(rv, label_y, f'{rv:.0f}', transform=tr, rotation=90,
                        va='top', ha='right', fontsize=7.5, color='0.25')

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

    # (1) misfit curves, absolute psi norm
    ax = axes[0, 0]
    ax.semilogx(sum_u['grid'], sum_u['curve'], '-', color='C0', lw=1.9,
                label='uniform $D$')
    ax.semilogx(sum_g['grid'], sum_g['curve'], '-', color='C1', lw=1.9,
                label=r'graded $D_{max}$ ($D_{min}{=}D_{max}/6$)')
    b = sum_u['band_10pct']
    ax.axvspan(b[0], b[1], color='C0', alpha=0.10, zorder=0)
    ax.plot(sum_u['best'], sum_u['best_rmse'], 'o', color='C0', ms=7, zorder=5)
    ax.plot(sum_g['best'], sum_g['best_rmse'], 's', color='C1', ms=7, zorder=5)
    ax.set_ylim(top=min(ax.get_ylim()[1], sum_u['best_rmse'] * 3.4))
    ref_lines(ax)
    ax.set_xlabel(r'diffusivity (ft$^2$/s)')
    ax.set_ylabel('RMSE on target gauges (psi)')
    ax.set_title(f"(a) Misfit, absolute norm\nmin $D$={sum_u['best']:.0f}, "
                 f"+10% band [{b[0]:.0f}, {b[1]:.0f}]", fontsize=10)
    ax.legend(fontsize=8, loc='upper center')
    ax.grid(alpha=0.3, which='both')

    # (2) misfit curves, amplitude-normalised norm
    ax = axes[0, 1]
    if sum_u_n is not None:
        ax.semilogx(sum_u_n['grid'], sum_u_n['curve'], '-', color='C0', lw=1.9,
                    label='uniform $D$')
        if sum_g_n is not None:
            ax.semilogx(sum_g_n['grid'], sum_g_n['curve'], '-', color='C1',
                        lw=1.9, label='graded $D_{max}$')
        bn = sum_u_n['band_10pct']
        ax.axvspan(bn[0], bn[1], color='C0', alpha=0.10, zorder=0)
        ax.plot(sum_u_n['best'], sum_u_n['best_rmse'], 'o', color='C0', ms=7,
                zorder=5)
        ax.set_ylim(top=min(ax.get_ylim()[1], sum_u_n['best_rmse'] * 3.4))
        ref_lines(ax)
        ax.set_title(f"(b) Misfit, amplitude-normalised\nmin $D$="
                     f"{sum_u_n['best']:.0f}, +10% band "
                     f"[{bn[0]:.0f}, {bn[1]:.0f}]", fontsize=10)
        ax.legend(fontsize=8, loc='upper center')
    ax.set_xlabel(r'diffusivity (ft$^2$/s)')
    ax.set_ylabel('RMSE / gauge amplitude (dimensionless)')
    ax.grid(alpha=0.3, which='both')

    # (3) single-gauge best-fit D vs distance -- the model-adequacy evidence
    ax = axes[0, 2]
    if single_gauge:
        d = np.array([x['distance_ft'] for x in single_gauge])
        v = np.array([x['best_D'] for x in single_gauge])
        ax.semilogy(d, v, 'o-', color='C3', lw=1.6, ms=7)
        for x in single_gauge:
            ax.annotate(f"g{x['gauge']}", (x['distance_ft'], x['best_D']),
                        textcoords='offset points', xytext=(6, 6), fontsize=8)
        ax.axhline(sum_u['best'], color='C0', ls='--', lw=1.2,
                   label=f"joint best $D$={sum_u['best']:.0f}")
        ax.axhline(480.0, color='0.35', ls='--', lw=1.0, label='480')
        ax.set_xlabel('distance from source gauge (ft)')
        ax.set_ylabel(r'best-fit $D$ for that gauge alone (ft$^2$/s)')
        ax.set_title('(c) Each gauge fitted alone\n'
                     f"required $D$ spans {v.max()/v.min():.0f}$\\times$",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which='both')

    # (4) constraint quality, both norms
    ax = axes[1, 0]
    ax.semilogx(sum_u['grid'], sum_u['curve'] / sum_u['best_rmse'] - 1.0, '-',
                color='C0', lw=1.8, label='absolute norm')
    if sum_u_n is not None:
        ax.semilogx(sum_u_n['grid'],
                    sum_u_n['curve'] / sum_u_n['best_rmse'] - 1.0, '-',
                    color='C4', lw=1.8, label='normalised norm')
    ax.axhline(cfg['uncertainty']['misfit_rise_fraction'], color='k', ls='--',
               lw=1.0, label='+10%')
    ax.set_ylim(-0.02, 1.0)
    ref_lines(ax)
    ax.set_xlabel(r'uniform $D$ (ft$^2$/s)')
    ax.set_ylabel('fractional RMSE rise above minimum')
    ax.set_title('(d) Constraint quality', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    # (5) fit at the best uniform D
    ax = axes[1, 1]
    prof = core.build_uniform_profile(mesh, sum_u['best'])
    taxis_s, rec = core.solve_forward(mesh, prof, dt_used, t_total, src['taxis'],
                                      src['delta_psi'], source_idx,
                                      record_idx=[t['idx'] for t in targets])
    colors = plt.cm.viridis(np.linspace(0, 0.88, len(targets)))
    ax.plot(src['taxis'], src['delta_psi'], '-', color='k', lw=1.4,
            label=f"g{src['gauge']} (source, prescribed)")
    for k, tgt in enumerate(targets):
        ax.plot(tgt['taxis'], tgt['data'], '-', color=colors[k], lw=1.6,
                label=f"g{tgt['gauge']} ({tgt['distance_ft']:.0f} ft)")
        ax.plot(taxis_s, rec[:, k], '--', color=colors[k], lw=1.2)
    ax.set_xlabel('time since window start (s)')
    ax.set_ylabel(r'$\Delta P$ (psi)')
    ax.set_title(f"(e) Fit at joint best $D$={sum_u['best']:.0f}\n"
                 'solid = observed, dashed = simulated', fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # (6) per-gauge diagnostics at the joint optimum
    ax = axes[1, 2]
    pg = sum_u['best_row']['per_gauge']
    d = np.array([g['distance_ft'] for g in pg])
    amp = np.array([g['amplitude_ratio'] for g in pg])
    arr = np.array([g['arrival_err_s'] for g in pg])
    ax.plot(d, amp, 'o-', color='C2', lw=1.6, ms=6, label='amplitude ratio')
    ax.axhline(1.0, color='C2', lw=0.9, ls=':')
    ax.set_ylabel('simulated / observed amplitude', color='C2')
    ax.tick_params(axis='y', labelcolor='C2')
    ax.set_xlabel('distance from source gauge (ft)')
    ax2 = ax.twinx()
    ax2.plot(d, arr, 's--', color='C3', lw=1.6, ms=6, label='arrival error')
    ax2.axhline(0.0, color='C3', lw=0.9, ls=':')
    ax2.set_ylabel('arrival-time error (s), + = late', color='C3')
    ax2.tick_params(axis='y', labelcolor='C3')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='center left')
    ax.set_title('(f) Residual structure at the optimum\n'
                 'near: under-predicted and late; far: over and early',
                 fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle('R1 baseline hydraulic diffusivity calibration — S well stage 1, '
                 'comparison window MD 15000-16750 ft, '
                 f"solver domain MD {mesh[0]:.0f}-{mesh[-1]:.0f} ft",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(path, dpi=cfg['outputs']['figure_dpi'])
    plt.close(fig)


if __name__ == '__main__':
    main()
