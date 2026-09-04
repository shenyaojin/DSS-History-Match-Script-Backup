#!/usr/bin/env python3
"""A3 DELIVERABLE -- time discretisation at the manuscript working point.

The A3 *investigation* is finished and is not redone here; its results are the
premises (output/rev2_20260901/A3/README.md, house-rules CORRECTION 3):

  * the scheme is fully implicit backward Euler, theta = 1 (matbuilder.py:41,45);
  * the Dirichlet datum is evaluated at time level n (matbuilder.py:73), i.e. a
    full-dt lag;
  * the manuscript's "adaptive" settings are not an accuracy control -- the error
    estimator never approaches tol, so dt pins to max_dt = 30 s.

This script produces the DELIVERABLE: three comparisons on one case that carries
both a barrier and a refined region -- manuscript phase 3 of
scripts/well_leakage_history_matching/101_fiberis_matching.py.

  1. theta = 1 vs theta = 0.5 (Rannacher start-up), side-by-side dP/dt waterfalls
     plus a difference panel, quantified near the barrier and inside the refined
     region: max, RMS, and fraction of signal.
  2. fixed dt = 1 s vs the manuscript adaptive settings on the SAME case: the
     realised dt trace (min/median/max, step count, rejections) and the error.
  3. the actual r = D*dt/dx^2 at the working point in the refined region.

Everything imports the shared rev2 modules; nothing here reimplements the solver.

Usage (CWD must be the repo root):
    python3 scripts/manuscript_well_leakage/rev2/a3_time_scheme.py \
        --config configs/rev2/a3_time_scheme.json
"""

import argparse
import datetime
import json
import os
import sys
import time
import warnings

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import TwoSlopeNorm                         # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
for _p in (_HERE, os.path.join(_ROOT, 'fibeRIS', 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rev2_core as rc                                             # noqa: E402
import rev2_data as rd                                             # noqa: E402
import rev2_layout as rl                                           # noqa: E402
import rev2_manifest as rm                                         # noqa: E402


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

class Tee:
    def __init__(self, path):
        self.fh = open(path, 'w')

    def __call__(self, *args):
        line = ' '.join(str(a) for a in args)
        print(line)
        self.fh.write(line + '\n')
        self.fh.flush()

    def close(self):
        self.fh.close()


def _stats(diff, ref):
    """max / RMS of `diff`, and both normalised by `ref`'s max and RMS."""
    d = np.asarray(diff, dtype=float)
    r = np.asarray(ref, dtype=float)
    dmax = float(np.max(np.abs(d))) if d.size else float('nan')
    drms = float(np.sqrt(np.mean(d ** 2))) if d.size else float('nan')
    rmax = float(np.max(np.abs(r))) if r.size else float('nan')
    rrms = float(np.sqrt(np.mean(r ** 2))) if r.size else float('nan')
    return {'max_abs': dmax, 'rms': drms,
            'ref_max_abs': rmax, 'ref_rms': rrms,
            'frac_of_ref_max': dmax / rmax if rmax else float('nan'),
            'rms_frac_of_ref_rms': drms / rrms if rrms else float('nan'),
            'n_samples': int(d.size)}


def _sign_reversals(inc):
    """Number of sign changes in a sequence of per-step increments."""
    s = np.sign(np.asarray(inc, dtype=float))
    s = s[s != 0.0]
    return int(np.sum(s[1:] != s[:-1])) if s.size > 1 else 0


def _lag_search(t, a, b, lo, hi, step, t_min=None):
    """argmin_L RMSE( a(t) - b(t-L) ) on the interior of t, by linear interp."""
    t = np.asarray(t, dtype=float)
    lags = np.arange(lo, hi + step / 2, step)
    m = t >= (t[0] + hi)                      # keep t-L inside b's support
    if t_min is not None:
        m &= t >= float(t_min)
    best, best_r = float('nan'), float('inf')
    curve = []
    for L in lags:
        bl = np.interp(t[m] - L, t, b)
        r = float(np.sqrt(np.mean((a[m] - bl) ** 2)))
        curve.append(r)
        if r < best_r:
            best_r, best = r, float(L)
    r0 = float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))
    return best, best_r, r0, lags, np.asarray(curve)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/rev2/a3_time_scheme.json')
    args = ap.parse_args(argv)

    t_wall = time.time()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()

    with open(args.config) as fh:
        cfg = json.load(fh)

    outdir = cfg['outputs']['dir']
    dpi = int(cfg['outputs']['figure_dpi'])
    os.makedirs(outdir, exist_ok=True)

    P = lambda name: os.path.join(outdir, name)                    # noqa: E731
    OUT = {
        'log': P('a3_deliverable_v1.log'),
        'results': P('a3_results_v1.json'),
        'csv': P('a3_key_numbers_v1.csv'),
        'npz': P('a3_panels_v1.npz'),
        'trace': P('a3_adaptive_trace_v1.json'),
        'fig1': P('fig01_a3_be_vs_cn_waterfall_v1.png'),
        'fig2': P('fig02_a3_be_vs_cn_traces_v1.png'),
        'fig3': P('fig03_a3_fixed_vs_adaptive_v1.png'),
        'fig4': P('fig04_a3_grid_and_r_v1.png'),
        'manifest': P('manifest.json'),
    }
    rm.assert_absent(list(OUT.values()) + [OUT['manifest'] + '.sha256'])
    log = Tee(OUT['log'])
    log(f"# A3 deliverable -- started {started}")
    log(f"# config {args.config}")

    # -----------------------------------------------------------------------
    # 1. mesh, barrier, sources, initial condition
    # -----------------------------------------------------------------------
    from fiberis.utils import mesh_utils

    fh7 = rd.load_frac_hits(7)
    fh8 = rd.load_frac_hits(8)
    mc = cfg['mesh']
    x = (float(mc['md_lo_ft'])
         + float(mc['dx_ft']) * np.arange(int(mc['n_uniform_nodes']), dtype=float))
    half = float(mc['refinement']['half_span_ft'])
    fac = int(mc['refinement']['factor'])
    for f in np.round(fh7):
        x = mesh_utils.refine_mesh(x, [f - half, f + half], fac)
    for f in np.round(fh8):
        x = mesh_utils.refine_mesh(x, [f - half, f + half], fac)
    nx = int(x.size)
    dxs = np.diff(x)
    log(f"\n## mesh: nx={nx} MD {x[0]:.1f}-{x[-1]:.1f} ft, "
        f"dx {dxs.min():.8f}-{dxs.max():.4f} ft")
    assert nx == int(mc['refinement']['expected_nx']), (nx, mc)

    locate = lambda md: int(np.argmin(np.abs(x - float(md))))      # noqa: E731
    idx7 = [locate(f) for f in fh7]        # barrier nodes (stage-7 frac hits)
    idx8 = [locate(f) for f in fh8]        # phase-3 source nodes (stage-8)

    D0 = float(cfg['physics']['D_baseline_ft2_s'])
    bc = cfg['physics']['barrier']
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        d_phase3, brep = rc.build_barrier_profile(
            x, D0, fh7, float(bc['w_half_width_ft']), float(bc['ratio']),
            ratio_reference=bc['ratio_reference'], combine=bc['combine'],
            on_empty=bc['on_empty'], return_report=True)
    n_width_warn = sum(1 for w in caught
                       if issubclass(w.category, rc.BarrierWidthWarning))
    # THE REPORT IS THE AUTHORITY (rev2_core docstring): assert on it, not on the
    # warnings, which do not survive a filter or a Pool boundary.
    assert brep['n_fallback'] == len(fh7), brep['n_fallback']
    assert brep['n_overlapping_pairs'] == 0
    d_uniform = np.full(nx, D0)

    # the legacy single-node assignment this must reproduce EXACTLY
    d_legacy = d_uniform.copy()
    for i in idx7:
        d_legacy[i] = D0 * float(bc['ratio'])
    legacy_identical = bool(np.array_equal(d_legacy, d_phase3))
    log(f"## barrier: {brep['n_barriers']} single nodes, ratio {bc['ratio']:g}, "
        f"realised full width "
        f"{brep['realised_full_width_ft']['median']:.6f} ft "
        f"(= 2/15); identical to the legacy index assignment: {legacy_identical}")
    assert legacy_identical

    md_tab = rd.load_gauge_md_table()
    mon_g = [int(g) for g in cfg['analysis']['monitor_gauges']]
    mon_md = [md_tab.md_of(g) for g in mon_g]
    mon_idx = [locate(m) for m in mon_md]
    log("## monitor gauges (101:72 plots gauge_md[4:10]): "
        + ", ".join(f"g{g}@{m:.0f}ft" for g, m in zip(mon_g, mon_md)))

    pw = rd.manuscript_phase_windows()
    ph = {}
    for key, (g, a, b) in pw.items():
        W = rd.Window(md_min_ft=0.0, md_max_ft=1e9, t_start=a, t_end=b)
        gw = rd.load_window_gauges(W, gauges=[g], baseline='none')
        ph[key] = {'gauge': g, 't_start': a, 't_end': b,
                   'series': gw.series[g], 'window': W}
        s = ph[key]['series']
        log(f"## {key}: gauge {g}, {s.n} samples, t_total {s.t_total_s:.3f} s, "
            f"raw {s.raw_psi.min():.1f}-{s.raw_psi.max():.1f} psi")

    # the 2025 archive of the same chain, used as a fibeRIS reference below
    p2_path = cfg['phases']['phase2_archive']
    panel2 = rl.load_panel(p2_path)
    mesh_matches_panel = bool(np.array_equal(
        np.asarray(panel2.daxis, dtype=float), x))
    log(f"## archived phase-2 panel {p2_path} (on-disk layout "
        f"{panel2.detected_layout}); daxis bit-identical to the rebuilt mesh: "
        f"{mesh_matches_panel}")
    assert mesh_matches_panel

    s3 = ph['phase3']['series']
    t_total3 = float(s3.t_total_s)

    # -----------------------------------------------------------------------
    # 2. chain verification: re-run phases 1 and 2 with rev2_core
    # -----------------------------------------------------------------------
    ad = cfg['adaptive']
    akw = dict(dt_init=float(ad['dt_init']), tol=float(ad['tol']),
               safety_factor=float(ad['safety_factor']),
               order_p=int(ad['order_p']), max_dt=float(ad['max_dt']),
               min_dt=float(ad['min_dt']),
               controller_tol=float(ad['controller_tol']),
               zero_field_policy=ad['zero_field_policy'],
               max_attempts=int(ad['max_attempts']))

    log("\n## chain verification (phases 1+2, adaptive, theta=1, level n)")
    s1 = ph['phase1']['series']
    s2 = ph['phase2']['series']
    panel1 = rl.load_panel(cfg['phases']['phase1_archive'])
    F1 = np.asarray(panel1.data, dtype=float)
    T1 = np.asarray(panel1.taxis, dtype=float)
    F2 = np.asarray(panel2.data, dtype=float)
    T2 = np.asarray(panel2.taxis, dtype=float)
    u_archive2 = F2[-1].copy()

    def run_chain(shift_s):
        """phases 1 then 2 with the drive series shifted by `shift_s` seconds."""
        ta1 = s1.taxis_s + shift_s
        ta2 = s2.taxis_s + shift_s
        ui = np.full(nx, float(np.interp(0.0, ta1, s1.raw_psi)))   # 101:131-133
        t_1, f_1, r_1 = rc.solve_forward_adaptive(
            x, d_uniform, float(s1.t_total_s), [ta1] * len(idx7),
            [s1.raw_psi] * len(idx7), idx7, initial=ui, **akw)
        e1 = f_1[-1].copy()
        del f_1
        t_2, f_2, r_2 = rc.solve_forward_adaptive(
            x, d_uniform, float(s2.t_total_s), [ta2] * len(idx7),
            [s2.raw_psi] * len(idx7), idx7, initial=e1, **akw)
        e2 = f_2[-1].copy()
        del f_2
        return (t_1, e1, r_1), (t_2, e2, r_2)

    t1 = time.time()
    (tax1, u_end1, tr1), (tax2, u_end2, tr2) = run_chain(0.0)
    chain_wall = time.time() - t1
    u0 = u_end2.copy()          # THE phase-3 initial condition (current data)

    # Why the archive is not reproduced bit-for-bit: scan a constant offset of the
    # drive series against the archived phase-1 SOURCE-NODE values, which are a
    # prescribed datum and therefore read the source series directly.
    sc = cfg['phases']['archive_shift_scan_s']
    shifts = np.arange(float(sc[0]), float(sc[1]) + float(sc[2]) / 2, float(sc[2]))
    jsrc = idx7[1]
    resid = np.array([np.max(np.abs(F1[1:, jsrc]
                                    - np.interp(T1[:-1] + d_, s1.taxis_s,
                                                s1.raw_psi)))
                      for d_ in shifts])
    best_shift = float(shifts[int(np.argmin(resid))])
    best_resid = float(resid.min())
    resid0 = float(resid[int(np.argmin(np.abs(shifts)))])
    (tax1s, u_end1s, tr1s), (tax2s, u_end2s, tr2s) = run_chain(-best_shift)

    keep = ('n_attempts', 'n_accepted', 'n_rejected', 'dt_min_s', 'dt_max_s',
            'dt_mean_s', 'frac_at_max_dt', 't_end_s', 'overshoot_s', 'err_min',
            'err_max', 'flip_margin')
    chain = {
        'phase1': {k: tr1[k] for k in keep},
        'phase2': {k: tr2[k] for k in keep},
        'phase1_taxis_identical_to_archive': bool(
            tax1.size == T1.size and np.array_equal(tax1, T1)),
        'phase2_taxis_identical_to_archive': bool(
            tax2.size == T2.size and np.array_equal(tax2, T2)),
        'archive_drive_time_offset': {
            'best_shift_s': best_shift,
            'max_abs_source_node_residual_at_best_shift_psi': best_resid,
            'max_abs_source_node_residual_at_zero_shift_psi': resid0,
            'gauge_sample_interval_s': float(np.median(np.diff(s1.taxis_s))),
            'first_sample_offset_from_window_start_s':
                (s1.t0_abs - ph['phase1']['t_start']).total_seconds(),
            'interpretation':
                'the 2025 archive keyed its cropped drive series to the crop '
                'WINDOW start, current fibeRIS crop rebases to the first '
                'in-window sample -- one gauge sample apart',
        },
        'as_run_vs_archive': {
            'phase1_end_max_abs_psi': float(np.max(np.abs(u_end1 - F1[-1]))),
            'phase2_end_max_abs_psi': float(np.max(np.abs(u_end2 - u_archive2))),
        },
        'shift_corrected_vs_archive': {
            'phase1_end_max_abs_psi': float(np.max(np.abs(u_end1s - F1[-1]))),
            'phase1_end_rms_psi': float(np.sqrt(np.mean((u_end1s - F1[-1]) ** 2))),
            'phase2_end_max_abs_psi': float(np.max(np.abs(u_end2s - u_archive2))),
            'phase2_end_rms_psi': float(np.sqrt(np.mean((u_end2s
                                                         - u_archive2) ** 2))),
            'phase1_taxis_identical': bool(tax1s.size == T1.size
                                           and np.array_equal(tax1s, T1)),
            'phase2_taxis_identical': bool(tax2s.size == T2.size
                                           and np.array_equal(tax2s, T2)),
        },
        'field_range_psi': [float(u0.min()), float(u0.max())],
        'wall_s': chain_wall,
    }
    for nm, tr in (('phase1', tr1), ('phase2', tr2)):
        log(f"   {nm} {tr['n_attempts']} attempts / {tr['n_accepted']} accepted / "
            f"{tr['n_rejected']} rejected, dt {tr['dt_min_s']:g}-"
            f"{tr['dt_max_s']:.6g} s, frac at max_dt {tr['frac_at_max_dt']:.4f}, "
            f"err {tr['err_min']:.3e}-{tr['err_max']:.3e}")
    log(f"   phase-1 taxis identical to the archive: "
        f"{chain['phase1_taxis_identical_to_archive']}; as-run end-state "
        f"max|diff| {chain['as_run_vs_archive']['phase1_end_max_abs_psi']:.4g} psi")
    log(f"   archive drive-series offset: best shift {best_shift:+.3f} s reduces "
        f"the prescribed source-node residual from {resid0:.4g} to "
        f"{best_resid:.4g} psi (gauge sampling "
        f"{chain['archive_drive_time_offset']['gauge_sample_interval_s']:.3f} s)")
    log(f"   with that shift undone, phases 1+2 reproduce the archived fibeRIS "
        f"run: phase-1 end max|diff| "
        f"{chain['shift_corrected_vs_archive']['phase1_end_max_abs_psi']:.4g} psi, "
        f"phase-2 end max|diff| "
        f"{chain['shift_corrected_vs_archive']['phase2_end_max_abs_psi']:.4g} psi "
        f"over a {u0.min():.1f}-{u0.max():.1f} psi field")
    log(f"   the phase-3 initial condition used below is the CURRENT-data chain "
        f"(shift 0); it differs from the archived phase-2 end state by "
        f"{chain['as_run_vs_archive']['phase2_end_max_abs_psi']:.4g} psi max "
        f"({chain_wall:.2f} s)")

    jump = s3.raw_psi[0] - u0[idx8]
    log(f"## phase-3 restart discontinuity at the six source nodes: "
        f"{jump.min():.2f} to {jump.max():.2f} psi "
        f"(s(0) = {s3.raw_psi[0]:.3f} psi)")

    # -----------------------------------------------------------------------
    # 3. the recorded node set
    # -----------------------------------------------------------------------
    an = cfg['analysis']
    b_lo, b_hi = [float(v) for v in an['waterfall_band_md_ft']]
    band_nodes = np.where((x >= b_lo) & (x <= b_hi))[0]
    extra = []
    for i in mon_idx + idx7 + idx8:
        extra += [i - 1, i, i + 1]
    rec_idx = np.unique(np.concatenate([band_nodes, np.asarray(extra)]))
    rec_idx = rec_idx[(rec_idx >= 0) & (rec_idx < nx)]
    rec_x = x[rec_idx]
    pos = {int(i): k for k, i in enumerate(rec_idx)}
    # control-volume size of each recorded node, used to classify refined/coarse
    cv = np.empty(nx)
    cv[1:-1] = (x[2:] - x[:-2]) / 2.0
    cv[0] = x[1] - x[0]
    cv[-1] = x[-1] - x[-2]
    rec_cv = cv[rec_idx]
    thr = float(an['refined_dx_threshold_ft'])
    m_refined = rec_cv < thr
    m_coarse = ~m_refined
    nb = float(an['barrier_neighbourhood_ft'])
    m_barrier = np.zeros(rec_x.size, dtype=bool)
    for f in fh7:
        m_barrier |= np.abs(rec_x - f) <= nb
    m_band = (rec_x >= b_lo) & (rec_x <= b_hi)
    log(f"\n## recorded nodes: {rec_idx.size} "
        f"({int(m_refined.sum())} refined, {int(m_coarse.sum())} coarse, "
        f"{int(m_barrier.sum())} within {nb:g} ft of a stage-7 frac hit)")

    # -----------------------------------------------------------------------
    # 4. comparison 3 -- the grid numbers (cheap, do them first)
    # -----------------------------------------------------------------------
    dx_ref = float(np.median(dxs[dxs < thr]))
    dx_cor = float(np.median(dxs[dxs >= thr]))
    D_r1 = 1150.0     # the R1 baseline optimum, for context only
    grid = {'dx_refined_ft': dx_ref, 'dx_coarse_ft': dx_cor,
            'D_baseline_ft2_s': D0, 'D_barrier_ft2_s': D0 * float(bc['ratio']),
            'r_D_dt_over_dx2': {}, 'amplification': {}}
    for dt in (0.5, 1.0, 2.0, 30.0):
        grid['r_D_dt_over_dx2'][f'dt={dt:g}'] = {
            'refined_D140': D0 * dt / dx_ref ** 2,
            'coarse_D140': D0 * dt / dx_cor ** 2,
            'refined_D1150': D_r1 * dt / dx_ref ** 2,
            'coarse_D1150': D_r1 * dt / dx_cor ** 2,
            'barrier_node_D0.0014': D0 * float(bc['ratio']) * dt / dx_ref ** 2,
        }
    for dt in (1.0, 30.0):
        for th in (1.0, 0.5):
            g = rc.amplification_factor(x, d_phase3, dt, th)
            grid['amplification'][f'dt={dt:g},theta={th:g}'] = g
    log("\n## comparison 3 -- r = D dt/dx^2 at the working point")
    for k, v in grid['r_D_dt_over_dx2'].items():
        log(f"   {k:9s} refined(dx={dx_ref:.5f}) r={v['refined_D140']:11.1f}   "
            f"coarse(dx={dx_cor:.1f}) r={v['coarse_D140']:8.1f}   "
            f"barrier node r={v['barrier_node_D0.0014']:.4g}")
    for k, v in grid['amplification'].items():
        log(f"   {k:18s} lam_max={v['lam_max']:.6g}  g={v['g']:+.8f}")

    # -----------------------------------------------------------------------
    # 5. the phase-3 runs
    # -----------------------------------------------------------------------
    log("\n## phase-3 runs (initial = the phase-2 end state, held fixed)")
    src_t = [s3.taxis_s] * len(idx8)
    src_d = [s3.raw_psi] * len(idx8)
    runs = {}
    for spec in cfg['runs']:
        key = spec['key']
        t0r = time.time()
        if spec['mode'] == 'fixed':
            tax, fld = rc.solve_forward_multi(
                x, d_phase3, float(spec['dt_s']), t_total3, src_t, src_d, idx8,
                initial=u0, record_idx=rec_idx, theta=float(spec['theta']),
                source_time_level=spec['source_time_level'],
                theta_startup_steps=int(spec['rannacher']))
            tr = None
        else:
            tax, fld, tr = rc.solve_forward_adaptive(
                x, d_phase3, t_total3, src_t, src_d, idx8, initial=u0,
                record_idx=rec_idx, theta=float(spec['theta']),
                source_time_level=spec['source_time_level'], **akw)
        runs[key] = {'spec': spec, 't': tax, 'f': fld, 'trace': tr,
                     'wall_s': time.time() - t0r}
        log(f"   {key:16s} theta={spec['theta']:g} level={spec['source_time_level']:3s} "
            f"steps={tax.size - 1:6d} t_end={tax[-1]:.3f} s  "
            f"({runs[key]['wall_s']:.2f} s)")

    # -----------------------------------------------------------------------
    # 6. comparison 1 -- theta = 1 vs theta = 0.5
    # -----------------------------------------------------------------------
    def dpdt(key):
        r = runs[key]
        dt = float(np.median(np.diff(r['t'])))
        return (r['t'][:-1] + r['t'][1:]) / 2.0, np.diff(r['f'], axis=0) / dt

    regions = {'band': m_band, 'refined': m_refined, 'coarse': m_coarse,
               'barrier_pm1ft': m_barrier}

    def compare(a, b, label):
        """Region-wise comparison of pressure and dP/dt; `a` is the reference."""
        assert np.allclose(runs[a]['t'], runs[b]['t'])
        Pa, Pb = runs[a]['f'], runs[b]['f']
        ta, Ga = dpdt(a)
        _, Gb = dpdt(b)
        # the phase-3 signal is the response relative to the restart field
        Sa = Pa - Pa[0]
        t_settle = float(an['restart_settle_s'])
        rowP = runs[a]['t'] >= t_settle
        rowG = ta >= t_settle
        out = {'label': label, 'reference': a, 'other': b, 'regions': {},
               'regions_after_settle': {}, 'settle_time_s': t_settle}
        for rn, msk in regions.items():
            out['regions'][rn] = {
                'pressure_psi': _stats((Pb - Pa)[:, msk], Sa[:, msk]),
                'dPdt_psi_per_s': _stats((Gb - Ga)[:, msk], Ga[:, msk]),
            }
            out['regions_after_settle'][rn] = {
                'pressure_psi': _stats((Pb - Pa)[np.ix_(rowP, msk)],
                                       Sa[np.ix_(rowP, msk)]),
                'dPdt_psi_per_s': _stats((Gb - Ga)[np.ix_(rowG, msk)],
                                         Ga[np.ix_(rowG, msk)]),
            }
        # where the worst dP/dt disagreement sits, over the whole band
        DG = np.abs(Gb - Ga)[:, m_band]
        it, ix = np.unravel_index(int(np.argmax(DG)), DG.shape)
        out['worst_dPdt_location'] = {
            't_s': float(ta[it]), 'md_ft': float(rec_x[m_band][ix]),
            'value_psi_per_s': float((Gb - Ga)[:, m_band][it, ix])}
        # per-monitor-gauge
        mg = {}
        for g, md, i in zip(mon_g, mon_md, mon_idx):
            k = pos[i]
            sig = Pa[:, k] - Pa[0, k]
            mg[f'g{g}'] = {
                'md_ft': md,
                'peak_abs_dP_psi': float(np.max(np.abs(sig))),
                'rmse_psi': float(np.sqrt(np.mean((Pb[:, k] - Pa[:, k]) ** 2))),
                'max_abs_psi': float(np.max(np.abs(Pb[:, k] - Pa[:, k]))),
            }
            mg[f'g{g}']['rmse_pct_of_peak'] = (
                100.0 * mg[f'g{g}']['rmse_psi'] / mg[f'g{g}']['peak_abs_dP_psi'])
            mg[f'g{g}']['max_pct_of_peak'] = (
                100.0 * mg[f'g{g}']['max_abs_psi'] / mg[f'g{g}']['peak_abs_dP_psi'])
        out['monitor_gauges'] = mg
        return out

    log("\n## comparison 1 -- theta = 1 vs theta = 0.5")
    comp1 = {
        'cn_vs_be_dt1': compare('be_dt1_np1', 'cn_dt1_rann',
                                'CN(Rannacher) - BE, both level n+1, dt = 1 s'),
        'cn_vs_be_dt30': compare('be_dt30_np1', 'cn_dt30_rann',
                                 'CN(Rannacher) - BE, both level n+1, dt = 30 s'),
        'sourcelevel_dt1': compare('be_dt1_n', 'be_dt1_np1',
                                   'BE level n+1 - BE level n, dt = 1 s'),
        'sourcelevel_dt30': compare('be_dt30_n', 'be_dt30_np1',
                                    'BE level n+1 - BE level n, dt = 30 s'),
        'rannacher_dt1': compare('cn_dt1_rann', 'cn_dt1_norann',
                                 'CN without Rannacher - CN with, dt = 1 s'),
        'rannacher_dt30': compare('cn_dt30_rann', 'cn_dt30_norann',
                                  'CN without Rannacher - CN with, dt = 30 s'),
        'convergence_dt1_vs_dt0p5': None,   # filled below
    }
    for k, v in comp1.items():
        if v is None:
            continue
        log(f"   {v['label']}   worst dP/dt disagreement at t = "
            f"{v['worst_dPdt_location']['t_s']:.1f} s, MD "
            f"{v['worst_dPdt_location']['md_ft']:.2f} ft")
        for rn in ('band', 'refined', 'barrier_pm1ft'):
            for tag, blk in (('all t ', v['regions']),
                             (f">{v['settle_time_s']:.0f}s", v['regions_after_settle'])):
                s = blk[rn]['dPdt_psi_per_s']
                p = blk[rn]['pressure_psi']
                log(f"      {rn:14s} {tag} dP/dt max|d| {s['max_abs']:.4g} "
                    f"({100 * s['frac_of_ref_max']:.4g}% of peak |dP/dt|), "
                    f"RMS {s['rms']:.4g} ({100 * s['rms_frac_of_ref_rms']:.4g}% of "
                    f"RMS) | P max|d| {p['max_abs']:.4g} psi "
                    f"({100 * p['frac_of_ref_max']:.4g}% of peak dP)")

    # convergence of the dt = 1 s reference against dt = 0.5 s (same level n)
    ta1, tb1 = runs['be_dt1_n']['t'], runs['be_dt0p5_n']['t']
    conv = {'label': 'BE dt = 0.5 s vs dt = 1 s, level n', 'monitor_gauges': {}}
    for g, md, i in zip(mon_g, mon_md, mon_idx):
        k = pos[i]
        a = runs['be_dt1_n']['f'][:, k]
        b = np.interp(ta1, tb1, runs['be_dt0p5_n']['f'][:, k])
        sig = a - a[0]
        conv['monitor_gauges'][f'g{g}'] = {
            'md_ft': md, 'peak_abs_dP_psi': float(np.max(np.abs(sig))),
            'rmse_psi': float(np.sqrt(np.mean((b - a) ** 2))),
            'max_abs_psi': float(np.max(np.abs(b - a)))}
        e = conv['monitor_gauges'][f'g{g}']
        e['rmse_pct_of_peak'] = 100.0 * e['rmse_psi'] / e['peak_abs_dP_psi']
    comp1['convergence_dt1_vs_dt0p5'] = conv
    log("   BE dt=1 s vs dt=0.5 s (is the reference converged?): "
        + ", ".join(f"g{g} {conv['monitor_gauges'][f'g{g}']['rmse_pct_of_peak']:.3f}%"
                    for g in mon_g))

    # ringing diagnostic at the restart
    rw = float(an['ringing_window_s'])
    probe_nodes = {'source0_plus1': idx8[0] + 1,
                   'source5_plus1': idx8[-1] + 1,
                   'barrier0_minus1': idx7[0] - 1,
                   'barrier0_plus1': idx7[0] + 1}
    ring = {}
    for pk, node in probe_nodes.items():
        k = pos[node]
        ring[pk] = {'mesh_idx': node, 'md_ft': float(x[node])}
        for key in ('be_dt1_np1', 'cn_dt1_rann', 'cn_dt1_norann',
                    'be_dt30_np1', 'cn_dt30_rann', 'cn_dt30_norann'):
            t = runs[key]['t']
            f = runs[key]['f'][:, k]
            m = t <= t[0] + rw
            inc = np.diff(f[m])
            ring[pk][key] = {
                'n_steps_in_window': int(inc.size),
                'sign_reversals': _sign_reversals(inc),
                'max_abs_2nd_time_difference_psi':
                    float(np.max(np.abs(np.diff(f[m], n=2)))) if m.sum() > 2
                    else float('nan')}
    log(f"   ringing in the first {rw:g} s after the restart "
        f"(increment sign reversals):")
    for pk in probe_nodes:
        log(f"      {pk:16s} (MD {ring[pk]['md_ft']:.2f}) "
            + "  ".join(f"{key.split('_', 1)[1] if False else key}="
                        f"{ring[pk][key]['sign_reversals']}"
                        f"/{ring[pk][key]['n_steps_in_window']}"
                        for key in ('be_dt1_np1', 'cn_dt1_rann', 'cn_dt1_norann',
                                    'be_dt30_np1', 'cn_dt30_rann',
                                    'cn_dt30_norann')))

    # -----------------------------------------------------------------------
    # 7. comparison 2 -- fixed dt = 1 s vs the manuscript adaptive settings
    # -----------------------------------------------------------------------
    log("\n## comparison 2 -- fixed dt = 1 s vs the manuscript adaptive settings")
    tra = runs['adaptive_ms']['trace']
    dts = np.diff(runs['adaptive_ms']['t'])
    errs = np.array([a['err'] for a in tra['attempts']], dtype=float)
    adapt = {k: tra[k] for k in
             ('n_attempts', 'n_accepted', 'n_rejected', 'dt_init_s', 'tol',
              'controller_tol', 'safety_factor', 'order_p', 'max_dt_s',
              'min_dt_s', 'zero_field_policy', 'dt_min_s', 'dt_max_s',
              'dt_mean_s', 'frac_at_max_dt', 't_end_s', 't_total_requested_s',
              'overshoot_s', 'err_min', 'err_max', 'flip_margin', 'error_norm')}
    adapt['dt_median_s'] = float(np.median(dts))
    adapt['n_steps_at_max_dt'] = int(np.sum(np.isclose(dts, tra['max_dt_s'])))
    adapt['tol_over_err_max'] = float(tra['tol'] / tra['err_max'])
    adapt['field_l2_norm_at_restart_psi'] = float(np.linalg.norm(u0))
    log(f"   attempts {adapt['n_attempts']}, accepted {adapt['n_accepted']}, "
        f"rejected {adapt['n_rejected']}")
    log(f"   realised dt  min {adapt['dt_min_s']:g}  median "
        f"{adapt['dt_median_s']:g}  max {adapt['dt_max_s']:g} s; "
        f"{adapt['n_steps_at_max_dt']}/{adapt['n_accepted']} steps "
        f"({100 * adapt['frac_at_max_dt']:.2f}%) sit on max_dt = "
        f"{adapt['max_dt_s']:g} s")
    log(f"   error estimate {adapt['err_min']:.3e} .. {adapt['err_max']:.3e} "
        f"against tol {adapt['tol']:g}  -> the control is never within a factor "
        f"{adapt['tol_over_err_max']:.4g} of binding; flip_margin "
        f"{adapt['flip_margin']:.4g}")
    log(f"   t_end {adapt['t_end_s']:.3f} s vs t_total "
        f"{adapt['t_total_requested_s']:.3f} s (overshoot "
        f"{adapt['overshoot_s']:+.3f} s -- `while t < t_total`, no clipping)")

    ref_t = runs['be_dt1_n']['t']
    ref_f = runs['be_dt1_n']['f']

    def against_ref(key):
        t, f = runs[key]['t'], runs[key]['f']
        gi = {}
        for g, md, i in zip(mon_g, mon_md, mon_idx):
            k = pos[i]
            a = ref_f[:, k]
            b = np.interp(ref_t, t, f[:, k])
            sig = a - a[0]
            peak = float(np.max(np.abs(sig)))
            gi[f'g{g}'] = {
                'md_ft': md, 'peak_abs_dP_psi': peak,
                'rmse_psi': float(np.sqrt(np.mean((b - a) ** 2))),
                'max_abs_psi': float(np.max(np.abs(b - a)))}
            gi[f'g{g}']['rmse_pct_of_peak'] = 100.0 * gi[f'g{g}']['rmse_psi'] / peak
            gi[f'g{g}']['max_pct_of_peak'] = (100.0 * gi[f'g{g}']['max_abs_psi']
                                              / peak)
        # region-wise on the recorded band, interpolated column by column
        B = np.empty_like(ref_f)
        for k in range(ref_f.shape[1]):
            B[:, k] = np.interp(ref_t, t, f[:, k])
        S = ref_f - ref_f[0]
        reg = {rn: _stats((B - ref_f)[:, msk], S[:, msk])
               for rn, msk in regions.items()}
        return {'monitor_gauges': gi, 'regions_pressure_psi': reg}

    comp2 = {'adaptive_trace': adapt,
             'adaptive_vs_dt1': against_ref('adaptive_ms'),
             'dt30_vs_dt1': against_ref('be_dt30_n'),
             'dt0p5_vs_dt1': against_ref('be_dt0p5_n')}
    for nm in ('adaptive_vs_dt1', 'dt30_vs_dt1'):
        log(f"   {nm}:")
        for g in mon_g:
            e = comp2[nm]['monitor_gauges'][f'g{g}']
            log(f"      g{g:<2d} MD {e['md_ft']:.0f}  peak dP "
                f"{e['peak_abs_dP_psi']:8.2f} psi  RMSE {e['rmse_psi']:7.3f} psi "
                f"({e['rmse_pct_of_peak']:5.2f}% of peak)  max "
                f"{e['max_abs_psi']:7.3f} psi ({e['max_pct_of_peak']:5.2f}%)")

    # The source-level lag. The investigation's exact "one full dt" identity holds
    # only for a smooth start: at a restart the level-n and level-(n+1) runs impose
    # different data on the very first step, and that seed difference diffuses, so
    # the lag is measured by a continuous search restricted to t >= settle.
    lag = {}
    t_settle = float(an['restart_settle_s'])
    for dtv, kn, knp1 in ((30.0, 'be_dt30_n', 'be_dt30_np1'),
                          (1.0, 'be_dt1_n', 'be_dt1_np1')):
        tn = runs[kn]['t']
        e = {}
        for g, i in zip(mon_g, mon_idx):
            k = pos[i]
            a = runs[kn]['f'][:, k]
            b = runs[knp1]['f'][:, k]
            noshift_max = float(np.max(np.abs(a - b)))
            L, r, r0, _, _ = _lag_search(
                tn, a, b, float(an['lag_search_s'][0]),
                min(float(an['lag_search_s'][1]), 3 * dtv),
                float(an['lag_search_step_s']), t_min=t_settle)
            e[f'g{g}'] = {
                'max_abs_unshifted_psi': noshift_max,
                'max_abs_one_step_index_shift_psi':
                    float(np.max(np.abs(a[1:] - b[:-1]))),
                'best_fit_lag_s': L, 'best_fit_lag_in_dt': L / dtv,
                'rmse_at_best_lag_psi': r, 'rmse_at_zero_lag_psi': r0,
                'lag_removes_fraction': (1.0 - r / r0) if r0 else float('nan')}
        lag[f'dt={dtv:g}'] = e
        log(f"   source-level lag at dt = {dtv:g} s (level n vs level n+1, "
            f"measured for t >= {t_settle:g} s):")
        for g in mon_g:
            q = e[f'g{g}']
            log(f"      g{g:<2d} max|n - n+1| {q['max_abs_unshifted_psi']:8.3f} psi; "
                f"best-fit lag {q['best_fit_lag_s']:7.3f} s = "
                f"{q['best_fit_lag_in_dt']:.4f} dt; RMSE {q['rmse_at_zero_lag_psi']:.4g}"
                f" -> {q['rmse_at_best_lag_psi']:.4g} psi "
                f"({100 * q['lag_removes_fraction']:.1f}% removed)")
    comp2['source_level_lag'] = lag

    # -----------------------------------------------------------------------
    # 8. figures
    # -----------------------------------------------------------------------
    log("\n## figures")
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8,
                         'axes.labelsize': 8, 'figure.dpi': 110})

    z_lo, z_hi = [float(v) for v in an['zoom_band_md_ft']]
    zt0, zt1 = [float(v) for v in an['zoom_time_s']]

    tG, G_be = dpdt('be_dt1_np1')
    _, G_cn = dpdt('cn_dt1_rann')
    G_d = G_cn - G_be

    def wf(ax, T, MD, Z, title, cmap, vmax, cb_label):
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.pcolormesh(MD, T, Z, cmap=cmap, norm=norm, shading='auto',
                           rasterized=True)
        ax.set_title(title)
        ax.set_xlabel('MD (ft)')
        ax.set_ylabel('t since phase-3 restart (s)')
        cb = plt.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(cb_label)
        return im

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    dec = int(an['panel_time_decimation'])
    sl = slice(None, None, dec)
    v1 = float(np.percentile(np.abs(G_be), 99.5))
    # The difference panels are clipped at their own 99.9th percentile: a single
    # first-step cell carries the maximum, and scaling to it would render the
    # panel blank and hide where the two schemes actually disagree. Both the clip
    # and the true maximum are printed.
    v2 = float(np.percentile(np.abs(G_d), 99.9)) or 1e-30
    m2 = float(np.max(np.abs(G_d)))
    wf(axes[0, 0], tG[sl], rec_x, G_be[sl],
       r'(a) $\theta=1$ (backward Euler), $dt=1$ s', 'RdBu_r', v1, 'dP/dt (psi/s)')
    wf(axes[0, 1], tG[sl], rec_x, G_cn[sl],
       r'(b) $\theta=0.5$ (Crank-Nicolson + Rannacher)', 'RdBu_r', v1,
       'dP/dt (psi/s)')
    wf(axes[0, 2], tG[sl], rec_x, G_d[sl],
       f'(c) (b) - (a), clipped at p99.9 = {v2:.3g} (max {m2:.3g}) psi/s',
       'PuOr_r', v2, r'$\Delta$ dP/dt (psi/s)')
    mz = (rec_x >= z_lo) & (rec_x <= z_hi)
    tz = (tG >= zt0) & (tG <= zt1)
    v3 = float(np.percentile(np.abs(G_be[np.ix_(tz, mz)]), 99.5))
    v4 = float(np.percentile(np.abs(G_d[np.ix_(tz, mz)]), 99.9)) or 1e-30
    m4 = float(np.max(np.abs(G_d[np.ix_(tz, mz)])))
    wf(axes[1, 0], tG[tz], rec_x[mz], G_be[np.ix_(tz, mz)],
       f'(d) zoom on the barrier / refined region, t < {zt1:.0f} s', 'RdBu_r', v3,
       'dP/dt (psi/s)')
    wf(axes[1, 1], tG[tz], rec_x[mz], G_cn[np.ix_(tz, mz)],
       '(e) same, Crank-Nicolson', 'RdBu_r', v3, 'dP/dt (psi/s)')
    wf(axes[1, 2], tG[tz], rec_x[mz], G_d[np.ix_(tz, mz)],
       f'(f) (e) - (d), clipped at p99.9 = {v4:.3g} (max {m4:.3g}) psi/s',
       'PuOr_r', v4, r'$\Delta$ dP/dt (psi/s)')
    for ax in axes[1]:
        for f in fh7:
            ax.axvline(f, color='k', lw=0.4, ls=':')
    st = comp1['cn_vs_be_dt1']['regions']['refined']['dPdt_psi_per_s']
    fig.suptitle(
        'A3: dP/dt at the manuscript working point (phase 3, D = 140 ft$^2$/s, '
        'six single-node barriers at ratio 1e-5 on a 0.1333 ft refined mesh).  '
        f"Inside the refined region CN - BE is at most {st['max_abs']:.3g} psi/s "
        f"= {100 * st['frac_of_ref_max']:.3g}% of the peak |dP/dt|.", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT['fig1'], dpi=dpi)
    plt.close(fig)
    log(f"   {OUT['fig1']}")

    # ---- figure 2: traces and the ringing diagnostic
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.8))
    ax = axes[0, 0]
    node = probe_nodes['source0_plus1']
    k = pos[node]
    for key, c, ls in (('be_dt1_np1', 'k', '-'), ('cn_dt1_rann', 'C0', '-'),
                       ('cn_dt1_norann', 'C3', '--')):
        t, f = runs[key]['t'], runs[key]['f'][:, k]
        m = t <= 120
        ax.plot(t[m], f[m], ls, color=c, lw=1.0, label=key)
    ax.set_title(f'(a) P(t) at the node next to source 1 (MD {x[node]:.2f} ft), '
                 f'dt = 1 s')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('P (psi)')
    ax.legend(fontsize=6)

    ax = axes[0, 1]
    for key, c, ls in (('be_dt30_np1', 'k', '-o'), ('cn_dt30_rann', 'C0', '-s'),
                       ('cn_dt30_norann', 'C3', '--^')):
        t, f = runs[key]['t'], runs[key]['f'][:, k]
        m = t <= 600
        ax.plot(t[m][1:], np.diff(f[m]), ls, color=c, lw=1.0, ms=2.5, label=key)
    ax.axhline(0, color='0.6', lw=0.5)
    ax.set_title('(b) per-step increment, same node, dt = 30 s (the manuscript '
                 'dt):\nCN rings; Rannacher removes it')
    ax.set_xlabel('t (s)')
    ax.set_ylabel(r'$P^{n+1}-P^{n}$ (psi)')
    ax.legend(fontsize=6)

    ax = axes[1, 0]
    kb = pos[probe_nodes['barrier0_minus1']]
    for key, c, lw, ls in (('be_dt1_np1', 'k', 1.8, '-'),
                           ('cn_dt1_rann', 'C1', 0.8, '--')):
        t, f = runs[key]['t'], runs[key]['f'][:, kb]
        ax.plot(t[1:], np.diff(f) / float(np.median(np.diff(t))), ls, color=c,
                lw=lw, label=key)
    ax.set_title(f"(c) dP/dt just outside barrier 1 "
                 f"(MD {x[probe_nodes['barrier0_minus1']]:.2f} ft), dt = 1 s")
    ax.set_xlabel('t (s)')
    ax.set_ylabel('dP/dt (psi/s)')
    ax.legend(fontsize=6)

    ax = axes[1, 1]
    for g, i in zip(mon_g, mon_idx):
        k2 = pos[i]
        d = runs['cn_dt1_rann']['f'][:, k2] - runs['be_dt1_np1']['f'][:, k2]
        ax.plot(runs['be_dt1_np1']['t'], d, lw=0.8, label=f'g{g} @ {x[i]:.0f} ft')
    ax.set_title('(d) CN - BE at the six monitor gauges, dt = 1 s')
    ax.set_xlabel('t (s)')
    ax.set_ylabel(r'$P_{CN}-P_{BE}$ (psi)')
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT['fig2'], dpi=dpi)
    plt.close(fig)
    log(f"   {OUT['fig2']}")

    # ---- figure 3: fixed vs adaptive
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.8))
    at = np.array([a['t'] for a in tra['attempts']], dtype=float)
    adt = np.array([a['dt'] for a in tra['attempts']], dtype=float)
    acc = np.array([a['accepted'] for a in tra['attempts']], dtype=bool)
    ax = axes[0, 0]
    ax.step(at, adt, where='post', color='C0', lw=1.0, label='attempted dt')
    ax.axhline(tra['max_dt_s'], color='C3', ls='--', lw=0.8,
               label=f"max_dt = {tra['max_dt_s']:g} s")
    ax.axhline(tra['min_dt_s'], color='C2', ls=':', lw=0.8,
               label=f"min_dt = {tra['min_dt_s']:g} s")
    ax.set_yscale('log')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('dt (s)')
    ax.set_title(f"(a) realised dt: {int(acc.sum())} accepted / "
                 f"{int((~acc).sum())} rejected, "
                 f"{100 * adapt['frac_at_max_dt']:.1f}% at max_dt")
    ax.legend(fontsize=6)

    ax = axes[0, 1]
    ax.semilogy(at, errs, '.', ms=2.5, color='C0', label='error estimate')
    ax.axhline(tra['tol'], color='C3', ls='--', lw=0.9,
               label=f"tol = {tra['tol']:g}")
    ax.set_xlabel('t (s)')
    ax.set_ylabel('relative L2 error estimate')
    ax.set_title(f"(b) the control never binds: max err "
                 f"{adapt['err_max']:.2e} = tol/{adapt['tol_over_err_max']:.0f}")
    ax.legend(fontsize=6)

    ax = axes[1, 0]
    gi = mon_g.index(7) if 7 in mon_g else 0
    k3 = pos[mon_idx[gi]]
    ax.plot(ref_t, ref_f[:, k3] - ref_f[0, k3], 'k-', lw=1.0, label='fixed dt = 1 s')
    ax.plot(runs['adaptive_ms']['t'],
            runs['adaptive_ms']['f'][:, k3] - runs['adaptive_ms']['f'][0, k3],
            'C0-o', ms=2.0, lw=0.8, label='manuscript adaptive')
    ax.plot(runs['be_dt30_n']['t'],
            runs['be_dt30_n']['f'][:, k3] - runs['be_dt30_n']['f'][0, k3],
            'C3--', lw=0.8, label='fixed dt = 30 s')
    ax.set_xlim(0, 1500)
    ax.set_xlabel('t (s)')
    ax.set_ylabel(r'$\Delta P$ (psi)')
    ax.set_title(f'(c) gauge {mon_g[gi]} (MD {mon_md[gi]:.0f} ft), first 1500 s')
    ax.legend(fontsize=6)

    ax = axes[1, 1]
    for g, i in zip(mon_g, mon_idx):
        k4 = pos[i]
        b = np.interp(ref_t, runs['adaptive_ms']['t'],
                      runs['adaptive_ms']['f'][:, k4])
        ax.plot(ref_t, b - ref_f[:, k4], lw=0.8, label=f'g{g}')
    ax.set_xlabel('t (s)')
    ax.set_ylabel(r'$P_{adaptive}-P_{dt=1}$ (psi)')
    ax.set_title('(d) adaptive - fixed dt = 1 s at the monitor gauges')
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT['fig3'], dpi=dpi)
    plt.close(fig)
    log(f"   {OUT['fig3']}")

    # ---- figure 4: the grid and r
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.0), sharex=True)
    m4 = (x >= b_lo) & (x <= b_hi)
    xc = 0.5 * (x[:-1] + x[1:])
    m4c = (xc >= b_lo) & (xc <= b_hi)
    axes[0].semilogy(xc[m4c], dxs[m4c], 'k-', lw=0.7)
    axes[0].set_ylabel('dx (ft)')
    axes[0].set_title('(a) cell size: 1.0 ft, refined to 2/15 = 0.13333 ft over '
                      r'$\pm$1 ft at each frac hit')
    axes[1].semilogy(x[m4], d_phase3[m4], 'k-', lw=0.7)
    axes[1].set_ylabel(r'D (ft$^2$/s)')
    axes[1].set_title('(b) diffusivity: D = 140 ft$^2$/s with six single-node '
                      'barriers at ratio 1e-5 (stage-7 frac hits)')
    for dt, c in ((1.0, 'C0'), (30.0, 'C3')):
        cvloc = np.minimum(np.r_[dxs, dxs[-1]], np.r_[dxs[0], dxs])
        axes[2].semilogy(x[m4], d_phase3[m4] * dt / cvloc[m4] ** 2, color=c,
                         lw=0.7, label=f'dt = {dt:g} s')
    axes[2].axhline(0.5, color='0.5', ls='--', lw=0.8, label='r = 1/2')
    axes[2].set_ylabel(r'$r=D\,dt/dx^2$')
    axes[2].set_xlabel('MD (ft)')
    axes[2].set_title(f"(c) cell Fourier number: {grid['r_D_dt_over_dx2']['dt=1']['refined_D140']:.0f} "
                      f"in the refined region at dt = 1 s, "
                      f"{grid['r_D_dt_over_dx2']['dt=30']['refined_D140']:.0f} at "
                      f"dt = 30 s")
    axes[2].legend(fontsize=6)
    for ax in axes:
        for f in fh7:
            ax.axvline(f, color='C3', lw=0.3, ls=':')
        for f in fh8:
            ax.axvline(f, color='C0', lw=0.3, ls=':')
    fig.tight_layout()
    fig.savefig(OUT['fig4'], dpi=dpi)
    plt.close(fig)
    log(f"   {OUT['fig4']}")

    # -----------------------------------------------------------------------
    # 9. saved arrays, results, csv
    # -----------------------------------------------------------------------
    mon_traces = {f'P_{key}_g{g}': runs[key]['f'][:, pos[i]]
                  for key in runs for g, i in zip(mon_g, mon_idx)}
    np.savez_compressed(
        OUT['npz'],
        mesh_md_ft=x, dx_ft=dxs, D_phase3=d_phase3,
        recorded_idx=rec_idx, recorded_md_ft=rec_x, recorded_cv_ft=rec_cv,
        barrier_idx=np.asarray(idx7), source_idx=np.asarray(idx8),
        monitor_gauge=np.asarray(mon_g), monitor_idx=np.asarray(mon_idx),
        monitor_md_ft=np.asarray(mon_md),
        initial_phase2_end_psi=u0, archive_phase2_end_psi=u_archive2,
        chain_shift_scan_s=shifts, chain_shift_scan_resid_psi=resid,
        dpdt_t_s=tG[sl], dpdt_be_dt1_np1=G_be[sl].astype(np.float32),
        dpdt_cn_dt1_rann=G_cn[sl].astype(np.float32),
        adaptive_taxis_s=runs['adaptive_ms']['t'],
        adaptive_dt_s=dts, adaptive_err=errs,
        **{k: np.asarray(v) for k, v in mon_traces.items()},
        **{f't_{key}': runs[key]['t'] for key in runs})
    log(f"   {OUT['npz']}")

    with open(OUT['trace'], 'w') as fhh:
        json.dump({'run': 'adaptive_ms (manuscript settings, phase 3)',
                   'settings': akw, 'summary': adapt,
                   'attempts': tra['attempts'],
                   'phase1_summary': chain['phase1'],
                   'phase2_summary': chain['phase2'],
                   'phase1_attempts': tr1['attempts'],
                   'phase2_attempts': tr2['attempts']}, fhh, indent=1,
                  default=float)
    log(f"   {OUT['trace']}")

    results = {
        'case': cfg['case'],
        'mesh': {'nx': nx, 'md_range_ft': [float(x[0]), float(x[-1])],
                 'dx_refined_ft': dx_ref, 'dx_coarse_ft': dx_cor,
                 'n_refined_nodes': int(np.sum(cv < thr))},
        'barrier_report': brep,
        'barrier_identical_to_legacy_index_assignment': legacy_identical,
        'barrier_width_warnings_raised': int(n_width_warn),
        'phase3_initial_condition': 'phases 1+2 re-run here (adaptive, theta=1, '
                                    'level n) from current gauge data',
        'phase3_restart_discontinuity_psi': {
            's0_psi': float(s3.raw_psi[0]),
            'jump_at_source_nodes_psi': [float(v) for v in jump]},
        'chain_verification': chain,
        'comparison_1_theta': comp1,
        'comparison_1_ringing': ring,
        'comparison_2_time_stepping': comp2,
        'comparison_3_grid': grid,
        'runs': {k: {'spec': v['spec'], 'n_steps': int(v['t'].size - 1),
                     't_end_s': float(v['t'][-1]), 'wall_s': v['wall_s']}
                 for k, v in runs.items()},
        'wall_seconds_total': time.time() - t_wall,
    }
    with open(OUT['results'], 'w') as fhh:
        json.dump(results, fhh, indent=1, default=float)
    log(f"   {OUT['results']}")

    rows = [('quantity', 'value', 'units', 'where')]
    r1c = comp1['cn_vs_be_dt1']['regions']
    r30c = comp1['cn_vs_be_dt30']['regions']
    for rn in ('band', 'refined', 'barrier_pm1ft'):
        s = r1c[rn]['dPdt_psi_per_s']
        rows.append((f'CN-BE dP/dt max|diff| ({rn}, dt=1s)',
                     f"{s['max_abs']:.6g}", 'psi/s',
                     f"{100 * s['frac_of_ref_max']:.4g}% of peak |dP/dt|"))
        rows.append((f'CN-BE dP/dt RMS ({rn}, dt=1s)', f"{s['rms']:.6g}", 'psi/s',
                     f"{100 * s['rms_frac_of_ref_rms']:.4g}% of RMS |dP/dt|"))
        p = r1c[rn]['pressure_psi']
        rows.append((f'CN-BE P max|diff| ({rn}, dt=1s)', f"{p['max_abs']:.6g}",
                     'psi', f"{100 * p['frac_of_ref_max']:.4g}% of peak dP"))
        s = r30c[rn]['dPdt_psi_per_s']
        rows.append((f'CN-BE dP/dt max|diff| ({rn}, dt=30s)',
                     f"{s['max_abs']:.6g}", 'psi/s',
                     f"{100 * s['frac_of_ref_max']:.4g}% of peak |dP/dt|"))
    for g in mon_g:
        e = comp2['adaptive_vs_dt1']['monitor_gauges'][f'g{g}']
        rows.append((f'adaptive - dt=1s RMSE at g{g}', f"{e['rmse_psi']:.6g}",
                     'psi', f"{e['rmse_pct_of_peak']:.3g}% of the "
                            f"{e['peak_abs_dP_psi']:.1f} psi peak"))
    rows += [
        ('adaptive attempts', adapt['n_attempts'], 'count', 'phase 3'),
        ('adaptive accepted', adapt['n_accepted'], 'count', 'phase 3'),
        ('adaptive rejected', adapt['n_rejected'], 'count', 'phase 3'),
        ('adaptive dt min/median/max',
         f"{adapt['dt_min_s']:g}/{adapt['dt_median_s']:g}/{adapt['dt_max_s']:g}",
         's', 'phase 3'),
        ('adaptive fraction of steps at max_dt',
         f"{adapt['frac_at_max_dt']:.6f}", 'dimensionless', 'phase 3'),
        ('adaptive max error estimate', f"{adapt['err_max']:.6g}",
         'relative L2', f"tol = {adapt['tol']:g}, i.e. tol/"
                        f"{adapt['tol_over_err_max']:.4g}"),
        ('r = D dt/dx^2, refined region, dt = 1 s',
         f"{grid['r_D_dt_over_dx2']['dt=1']['refined_D140']:.6g}",
         'dimensionless', 'D = 140, dx = 0.13333 ft'),
        ('r = D dt/dx^2, refined region, dt = 30 s',
         f"{grid['r_D_dt_over_dx2']['dt=30']['refined_D140']:.6g}",
         'dimensionless', 'D = 140, dx = 0.13333 ft'),
        ('r = D dt/dx^2, coarse region, dt = 1 s',
         f"{grid['r_D_dt_over_dx2']['dt=1']['coarse_D140']:.6g}",
         'dimensionless', 'D = 140, dx = 1 ft'),
        ('CN amplification factor g, dt = 1 s',
         f"{grid['amplification']['dt=1,theta=0.5']['g']:.8f}", 'dimensionless',
         'stiffest mode'),
        ('BE amplification factor g, dt = 1 s',
         f"{grid['amplification']['dt=1,theta=1']['g']:.3e}", 'dimensionless',
         'stiffest mode'),
    ]
    for g in mon_g:
        q = comp2['source_level_lag']['dt=30'][f'g{g}']
        rows.append((f'source-level lag at dt=30 s, g{g}',
                     f"{q['best_fit_lag_s']:.4g}", 's',
                     f"{q['best_fit_lag_in_dt']:.4f} dt; the lag removes "
                     f"{100 * q['lag_removes_fraction']:.1f}% of the level-n vs "
                     f"level-(n+1) RMSE ({q['rmse_at_zero_lag_psi']:.4g} -> "
                     f"{q['rmse_at_best_lag_psi']:.3g} psi)"))
    with open(OUT['csv'], 'w') as fhh:
        for r in rows:
            fhh.write(','.join('"' + str(v).replace('"', "'") + '"'
                               for v in r) + '\n')
    log(f"   {OUT['csv']}")

    # -----------------------------------------------------------------------
    # 10. manifest
    # -----------------------------------------------------------------------
    drv3 = rm.driver_record(
        kind='gauge_series', baseline_removal='none_absolute_psi',
        value_units='psi',
        series_path='data/fiberis_format/s_well/gauges/gauge7_data_swell.npz',
        gauge_number=7, gauge_md_ft=md_tab.md_of(7), taxis=s3.taxis_s,
        values=s3.raw_psi, time_start=str(ph['phase3']['t_start']),
        time_end=str(ph['phase3']['t_end']))
    sp = rm.source_protocol(
        application='dirichlet_node',
        solver_class='rev2_core.solve_forward_multi / solve_forward_adaptive '
                     '(banded; theta=1/harmonic/lambda=0 is bitwise identical to '
                     'r1_calibration_core.solve_forward)',
        placement_rule='mesh_utils.locate(x, md) = argmin|x - md| on each stage-8 '
                       'frac-hit MD (101_fiberis_matching.py:89)',
        sources=[rm.source_record(x, md_requested_ft=float(f), mesh_idx=int(i),
                                  driver=drv3, label=f'stage-8 frac hit {j}',
                                  index_in_source_list=j)
                 for j, (f, i) in enumerate(zip(fh8, idx8))],
        targets=[{'gauge': g, 'md_ft': md, 'mesh_idx': int(i),
                  'role': 'monitor (101:72 plots gauge_md[4:10])'}
                 for g, md, i in zip(mon_g, mon_md, mon_idx)],
        time_level='n',
        phase_chaining={
            'phases': ['phase1 (stage-7 file span, gauge 6 drive)',
                       'phase2 (stage-7 end -> stage-8 start, gauge 6 drive)',
                       'phase3 (stage-8 file span, gauge 7 drive) -- THE CASE'],
            'phase3_initial_condition': 'phases 1+2 re-run here with rev2_core '
                                        'at the manuscript adaptive settings '
                                        'from current gauge data',
            'archive_reference': [cfg['phases']['phase1_archive'],
                                  cfg['phases']['phase2_archive']],
            'archive_drive_time_offset_s':
                chain['archive_drive_time_offset']['best_shift_s'],
            'phase2_end_vs_archive_max_abs_psi':
                chain['as_run_vs_archive']['phase2_end_max_abs_psi'],
            'restart_discontinuity_psi': [float(v) for v in jump]},
        boundary_conditions={'lbc': cfg['physics']['boundary']['lbc'],
                             'rbc': cfg['physics']['boundary']['rbc']})

    trecs = []
    for key, r in runs.items():
        sp_ = r['spec']
        if sp_['mode'] == 'fixed':
            trecs.append(rm.time_record(
                r['t'], mode='fixed', theta=float(sp_['theta']),
                t_total_requested_s=t_total3, dt_requested_s=float(sp_['dt_s']),
                source_time_level=sp_['source_time_level'],
                theta_startup_steps=int(sp_['rannacher']), label=key))
        else:
            tr = r['trace']
            trecs.append(rm.time_record(
                r['t'], mode='adaptive', theta=float(sp_['theta']),
                t_total_requested_s=t_total3, dt_init_s=tr['dt_init_s'],
                tol=tr['tol'], controller_tol=tr['controller_tol'],
                max_dt_s=tr['max_dt_s'], min_dt_s=tr['min_dt_s'],
                safety_factor=tr['safety_factor'], order_p=tr['order_p'],
                n_steps_rejected=tr['n_rejected'],
                source_time_level=sp_['source_time_level'],
                zero_field_policy=tr['zero_field_policy'],
                flip_margin=tr['flip_margin'], label=key))
    for nm, tr, tt, tx in (('chain_phase1', tr1, float(s1.t_total_s), tax1),
                           ('chain_phase2', tr2, float(s2.t_total_s), tax2),
                           ('chain_phase1_archive_shift', tr1s,
                            float(s1.t_total_s), tax1s),
                           ('chain_phase2_archive_shift', tr2s,
                            float(s2.t_total_s), tax2s)):
        trecs.append(rm.time_record(
            tx, mode='adaptive', theta=1.0, t_total_requested_s=tt,
            dt_init_s=tr['dt_init_s'], tol=tr['tol'],
            controller_tol=tr['controller_tol'], max_dt_s=tr['max_dt_s'],
            min_dt_s=tr['min_dt_s'], safety_factor=tr['safety_factor'],
            order_p=tr['order_p'], n_steps_rejected=tr['n_rejected'],
            source_time_level='n', zero_field_policy=tr['zero_field_policy'],
            flip_margin=tr['flip_margin'], label=nm))

    brecs = []
    for j, f in enumerate(fh7):
        msk = np.zeros(nx, dtype=bool)
        msk[idx7[j]] = True
        brecs.append(rm.barrier_record(
            x, msk, label=f'stage-7 frac hit {j}', centre_md_ft=float(f),
            w_requested_ft=float(bc['w_half_width_ft']), ratio=float(bc['ratio']),
            d_baseline=D0, report=brep['barriers'][j]))

    nmx = rm.numerics(
        time=trecs,
        mesh=rm.mesh_record(x, dx_requested_ft=float(mc['dx_ft']),
                            window_md_ft=(float(x[0]), float(x[-1])),
                            pad_low_ft=0.0, pad_high_ft=0.0,
                            refinement={'function': 'fiberis.utils.mesh_utils.'
                                                    'refine_mesh',
                                        'half_span_ft': half, 'factor': fac,
                                        'centres': 'np.round(stage-7 then '
                                                   'stage-8 frac hits)',
                                        'n_calls': int(fh7.size + fh8.size),
                                        'dx_refined_ft': dx_ref,
                                        'n_refined_nodes': int(np.sum(cv < thr))}),
        interface_avg=cfg['physics']['interface_avg'],
        boundary={'lbc': cfg['physics']['boundary']['lbc'],
                  'rbc': cfg['physics']['boundary']['rbc'],
                  'pml_thickness': 0.0, 'sigma_max': 0.0},
        diffusivity={'profile_family': 'uniform_with_single_node_barriers',
                     'param_names': ['D', 'ratio'],
                     'params': [D0, float(bc['ratio'])],
                     'profile_anchor': 'D = 140 everywhere except the six '
                                       'stage-7 frac-hit nodes',
                     'profile_sha256': rm.sha256_array(d_phase3)},
        barriers=brecs,
        leakage=rm.NONE_DECLARED,
        kernel={'name': 'rev2_core.solve_forward_multi / '
                        'solve_forward_adaptive', 'banded': True,
                'equivalence_reference':
                    'output/rev2_20260901/A4/selftest_output.txt (T1: max|diff| '
                    '= 0.000e+00 psi vs r1_calibration_core.solve_forward); this '
                    'run adds a full-size fibeRIS cross-check at nx = 5656: '
                    'with the archive drive-time offset undone, phases 1+2 '
                    'reproduce output/0211_simulation_MULTIstage/phase{1,2}.npz '
                    'to '
                    f"{chain['shift_corrected_vs_archive']['phase1_end_max_abs_psi']:.3e}"
                    ' / '
                    f"{chain['shift_corrected_vs_archive']['phase2_end_max_abs_psi']:.3e}"
                    ' psi',
                'n_forward_solves': int(len(runs) + 4)},
        rng=rm.NONE_DECLARED,
        parallel={'backend': 'none (serial)', 'processes': 1,
                  'deterministic': True},
        amplification=grid['amplification'])

    inputs = [
        (args.config, 'config', 'a3_time_scheme.json'),
        ('data/fiberis_format/s_well/gauges/gauge6_data_swell.npz',
         'gauge_series', 'phase 1/2 drive'),
        ('data/fiberis_format/s_well/gauges/gauge7_data_swell.npz',
         'gauge_series', 'phase 3 drive'),
        ('data/legacy/s_well/geometry/frac_hit/frac_hit_stage_7_swell.npz',
         'geometry', 'stage-7 frac hits (barrier + phase 1/2 sources)'),
        ('data/legacy/s_well/geometry/frac_hit/frac_hit_stage_8_swell.npz',
         'geometry', 'stage-8 frac hits (phase 3 sources)'),
        ('data/legacy/s_well/geometry/gauge_md_swell.npz', 'geometry',
         'gauge MD table'),
        (cfg['phases']['phase1_archive'], 'prior_run_output',
         'archived 2025 fibeRIS phase-1 panel, used as a full-size reference'),
        (cfg['phases']['phase2_archive'], 'prior_run_output',
         'archived 2025 fibeRIS phase-2 panel, used as a full-size reference'),
        ('scripts/well_leakage_history_matching/101_fiberis_matching.py', 'other',
         'the manuscript script this case reproduces'),
    ]
    for st in (7, 8):
        inputs.append((f'data/fiberis_format/prod/pumping_data/stage{st}/'
                       'Slurry Rate.npz', 'pumping',
                       f'stage {st} file span -> phase boundaries'))

    outs = []
    for k, p in OUT.items():
        if k == 'manifest':
            continue
        role = ('figure_png' if k.startswith('fig') else
                'log' if k == 'log' else
                'csv' if k == 'csv' else
                'arrays_npz' if k == 'npz' else 'json')
        outs.append(rm.output_decl(p, role=role,
                                   dpi=dpi if role == 'figure_png' else None,
                                   note=k))

    # The log is hashed by write_manifest, so it must be final BEFORE that call.
    log(f"   {OUT['manifest']}")
    log(f"\n# done in {time.time() - t_wall:.1f} s (the manifest is written "
        f"immediately after this line, which is why the log ends here)")
    log.close()

    rm.write_manifest(
        OUT['manifest'], study_id=cfg['study_id'], task_id='A3-deliverable',
        config=cfg, config_path=args.config, inputs=inputs, source=sp,
        numerics=nmx, outputs=outs, results=results, started_utc=started,
        run_label='A3 deliverable: BE vs CN, fixed vs adaptive, and r at the '
                  'manuscript working point',
        require_modules=('rev2_core', 'rev2_data', 'rev2_layout',
                         'rev2_manifest'),
        notes=[
            'The A3 investigation (output/rev2_20260901/A3/README.md, 613 lines) '
            'is NOT redone here; its findings are the premises. Nothing in '
            'output/rev2_20260901/A3/ outside deliverable/ was read-modified or '
            'overwritten.',
            'Every phase-3 comparison starts from the SAME initial field '
            '(phases 1+2 re-run here at the manuscript adaptive settings), so '
            'the time scheme is the only variable. Absolute pressures inherit the manuscript domain, which '
            'is padded 1797 ft below the lowest plotted gauge rather than the '
            '5000 ft the house rules require; that boundary is identical in '
            'every run compared here and cancels out of every difference '
            'reported, but the absolute levels are not quoted as physics (B2 '
            'owns that question).',
            'The archived output/0211_simulation_MULTIstage/phase{1,2}.npz CANNOT '
            'be reproduced bit-for-bit from current inputs: their drive series is '
            'offset by one gauge sample '
            f"({chain['archive_drive_time_offset']['best_shift_s']:+.3f} s) from "
            'what the current crop produces, because the 2025 crop keyed its '
            'taxis to the crop window start while current fibeRIS rebases to the '
            'first in-window sample (and the gauge npz files were themselves '
            'regenerated on 2025-03-13, after those outputs were written). With '
            'that offset undone the reproduction is exact to '
            f"{chain['shift_corrected_vs_archive']['phase1_end_max_abs_psi']:.3g} "
            f"/ {chain['shift_corrected_vs_archive']['phase2_end_max_abs_psi']:.3g}"
            ' psi, which is the full-size fibeRIS verification of rev2_core '
            '(nx = 5656, dense LAPACK vs banded, 485 + 309 adaptive steps).',
            'theta = 0.5 requires source_time_level = "n+1" (rev2_core raises '
            'otherwise), while the fibeRIS convention at theta = 1 is level "n". '
            'A naive BE-vs-CN comparison would therefore confound the time '
            'scheme with a full-dt source lag, so BE was run at BOTH levels and '
            'the two effects are reported separately.',
            'The barrier is the LEGACY single-node one (w = 0, on_empty = '
            '"nearest"), asserted here to be bit-identical to the index '
            'assignment d[idx] = D*ratio that 101 uses. On this mesh it realises '
            'a full width of 2/15 = 0.13333 ft (house-rules CORRECTION 4).',
        ])
    print(f"manifest written: {OUT['manifest']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
