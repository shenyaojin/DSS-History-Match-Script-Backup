#!/usr/bin/env python3
"""C4 -- are the barrier half-width `w` and the reduction ratio separately identifiable?

Round `rev2_20260901`. Task C4. Owns `configs/rev2/c4_identifiability.json` and
`output/rev2_20260901/C4/`.

THE QUESTION. A barrier is a slab of width W = 2w with diffusivity
D_b = D0 * ratio embedded in a background D0. If the slab is thin enough that its
own storage is negligible over the observation window it acts as a resistive
MEMBRANE: the only thing the surrounding field can sense is the series resistance
W / D_b = W / (D0 * ratio) (units s/ft). Then w, ratio and D0 are not three
parameters but one, and the misfit valley is a BAND along constant W/D_b rather
than a point. If instead the slab's own diffusion time W^2 / D_b sets the
response, the invariant is W^2/D_b and the band has a different slope. The two
are distinguishable by the EXPONENT q in w^q / D_b, and that exponent is
measured here rather than assumed.

WHAT IS RUN. E1's / 101's two-stage chain, imported from `a5_two_stage_chain`
(which reproduces the frozen 2025 archive to 2.6e-9 relative), with phase 3
carrying six stage-7 barriers built by `rev2_core.build_barrier_profile` at a
PHYSICAL half-width. Phases 1 and 2 are barrier-free, so for a given background D
they are solved ONCE and every phase-3 cell restarts from the same stored final
profile. Phase 3 is solved with `record_idx` at the five gauge nodes, which is
bitwise identical to the full-field solve at those columns (asserted once per
run, `record_idx_identity_max_abs_psi` in the manifest results).

WHY THIS GEOMETRY. It is the only place in this round where a barrier is both
physically present and observable against measured data: B2/E2 showed the Fig. 7b
production case is contaminated beyond rescue and prefers NO barrier at all, and
the R1 pressure window contains no barrier. Only gauges 5 and 6 lie above the
stage-7 barrier row, so they are the only gauges carrying barrier information
(E1: gauge 8 moves 1.6e-10 psi across eight decades of ratio).

MISFIT. Gauge-mean RMSE over the shielded gauges {5, 6} of the phase-3 pressure
CHANGE, each series referenced to its own first in-window sample -- E1's
`delta_from_phase3_start`. Gauge-mean, never sample-pooled (C1). The
all-five-gauge and absolute-psi variants are computed and stored alongside.

MODES
  --mode sweep     solve; writes arrays + metrics + a manifest per study dir
  --mode analyse   solve-free; valley trace, exponent fit, +10% bands, collapse
  --mode figures   solve-free; the contour plane, the 1-D curves, the collapse

Shared modules (`rev2_core`, `rev2_data`, `rev2_manifest`) and `a5_two_stage_chain`
are IMPORTED and never edited.
"""

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')          # before numpy loads its BLAS

import argparse                              # noqa: E402
import datetime                              # noqa: E402
import json                                  # noqa: E402
import multiprocessing as mp                 # noqa: E402
import sys                                   # noqa: E402
import time                                  # noqa: E402
import warnings                              # noqa: E402

import numpy as np                           # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rev2_core as rc                       # noqa: E402
import rev2_data as rd                       # noqa: E402
import rev2_manifest as rm                   # noqa: E402
import a5_two_stage_chain as a5              # noqa: E402

STUDY_ID = "C4_w_ratio_identifiability"
TASK_ID = "C4"
PHASES = ('phase1', 'phase2', 'phase3')
MAX_WORKERS = 6                              # house rule


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as fh:
        return json.load(fh)


def cell_tag(D0, w, ratio):
    return f"D{D0:g}_w{w:g}_r{ratio:g}".replace('.', 'p').replace('-', 'm')


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def make_source(cfg, x, wins, phase):
    """A5's source series plus the frac-hit node placement (E1's make_source)."""
    spec = [p for p in cfg['phases'] if p['name'] == phase][0]
    t0, t1 = wins[phase]
    src = a5.load_source_series(spec['source_gauge'], t0, t1, cfg['source'])
    stage = int(cfg['source_stage_by_phase'][phase])
    hits = rd.load_frac_hits(stage, unique=False, sort=False)
    src['frac_hit_stage'] = stage
    src['frac_hit_mds_ft'] = [float(v) for v in hits]
    src['source_idx'] = [int(np.argmin(np.abs(x - float(h)))) for h in hits]
    src['source_md_ft'] = [float(x[i]) for i in src['source_idx']]
    src['snap_error_ft'] = [float(x[i] - float(h))
                            for i, h in zip(src['source_idx'], hits)]
    src['window_abs'] = [t0.isoformat(), t1.isoformat()]
    return src


def load_field(gauges, wins):
    """Measured gauge series over the two-stage window, absolute psi (E1)."""
    t0, t1 = wins['phase1'][0], wins['phase3'][1]
    win = rd.Window(md_min_ft=0.0, md_max_ft=1e9, t_start=t0, t_end=t1)
    gw = rd.load_window_gauges(win, gauges=[int(g) for g in gauges],
                               baseline='none', rebase='per_gauge')
    out = {}
    for g in gauges:
        s = gw.series[int(g)]
        out[int(g)] = {'taxis_s': np.asarray(s.taxis_s, float),
                       'psi': np.asarray(s.raw_psi, float),
                       't0_abs': s.t0_abs, 'md_ft': float(s.md_ft)}
    return out


def field_on_phase3(field, gauges, t0_3, ta3):
    """Measured samples inside the phase-3 sim window, in sim-relative seconds.

    Exactly E1's fit_check bookkeeping: the measured axis is shifted by
    (gauge t0 - phase-3 source t0) and clipped to the simulated span.
    """
    out = {}
    for g in gauges:
        f = field[int(g)]
        tf = f['taxis_s'] + (f['t0_abs'] - t0_3).total_seconds()
        keep = (tf >= float(ta3[0])) & (tf <= float(ta3[-1]))
        if keep.sum() < 10:
            raise RuntimeError(f"gauge {g}: only {int(keep.sum())} measured "
                               f"samples inside the phase-3 window")
        out[int(g)] = {'t_rel_s': tf[keep], 'psi': f['psi'][keep],
                       'md_ft': f['md_ft']}
    return out


def misfit(tr, ta3, fld, gauges, shielded):
    """Per-gauge and aggregated RMSE. `tr` is (n_t, n_gauges) simulated psi."""
    per = {}
    for j, g in enumerate(gauges):
        d = fld[int(g)]
        sim = np.interp(d['t_rel_s'], ta3, tr[:, j])
        sim0 = float(np.interp(float(d['t_rel_s'][0]), ta3, tr[:, j]))
        pf = d['psi']
        per[int(g)] = {
            'n_samples': int(d['t_rel_s'].size),
            'rmse_abs_psi': float(np.sqrt(np.mean((sim - pf) ** 2))),
            'rmse_delta_psi': float(np.sqrt(np.mean(
                ((sim - sim0) - (pf - pf[0])) ** 2))),
        }
    sh = [per[int(g)]['rmse_delta_psi'] for g in shielded]
    return {
        'per_gauge': {f"g{g}": per[int(g)] for g in gauges},
        'shielded_mean_rmse_delta_psi': float(np.mean(sh)),
        'gauge_mean_rmse_delta_psi': float(np.mean(
            [per[int(g)]['rmse_delta_psi'] for g in gauges])),
        'gauge_mean_rmse_abs_psi': float(np.mean(
            [per[int(g)]['rmse_abs_psi'] for g in gauges])),
    }


# ---------------------------------------------------------------------------
# the worker: one (D0, w, ratio) phase-3 solve
# ---------------------------------------------------------------------------

_G = {}          # inherited by fork; never pickled


def _cell(job):
    D0, w, ratio = float(job['D0']), float(job['w']), float(job['ratio'])
    x = _G['x']
    base = np.full(len(x), D0, dtype=float)
    if ratio >= 1.0:
        dprof, brep = base, None
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            dprof, brep = rc.build_barrier_profile(
                x, base, _G['hits7'], w, ratio,
                ratio_reference=_G['bcfg']['ratio_reference'],
                combine=_G['bcfg']['combine'],
                on_empty=_G['bcfg']['on_empty'],
                on_outside=_G['bcfg']['on_outside'],
                return_report=True)
        # THE REPORT IS THE AUTHORITY, NOT THE WARNING (rev2_core docstring).
        if brep['n_fallback'] != 0:
            raise RuntimeError(
                f"D0={D0:g} w={w:g} ratio={ratio:g}: {brep['n_fallback']} "
                f"barrier(s) fell back to the nearest node; the realised width "
                f"is not the requested one. {brep['fallback_messages'][:1]}")
        if brep['n_overlapping_pairs'] != 0:
            raise RuntimeError(
                f"D0={D0:g} w={w:g}: {brep['n_overlapping_pairs']} overlapping "
                f"barrier pair(s); the six stage-7 hits are 32.60 ft apart at "
                f"the closest, so this grid was sized not to merge them")
        brep['n_warnings_raised'] = len(caught)

    s3 = _G['s3']
    idx = [int(i) for i in s3['source_idx']]
    t_w = time.time()
    ta3, tr = rc.solve_forward_multi(
        x, dprof, float(_G['tcfg']['dt_fixed_s']), float(s3['t_total_s']),
        [s3['taxis_s']] * len(idx), [s3['values_psi']] * len(idx), idx,
        initial=_G['init3'][f"{D0:g}"], t0=0.0, record_idx=_G['gidx'],
        theta=float(_G['pcfg']['theta']),
        lambda_leak=float(_G['pcfg']['lambda_leak']),
        p0=float(_G['pcfg']['p0_psi']),
        interface_avg=_G['pcfg']['interface_avg'],
        theta_startup_steps=int(_G['pcfg']['theta_startup_steps']))
    wall = time.time() - t_w

    m = misfit(tr, ta3, _G['fld'], _G['gauges'], _G['shielded'])
    dp = tr - tr[0]
    dec = max(1, int(round(float(_G['dec_s']) / float(_G['tcfg']['dt_fixed_s']))))
    out = {
        'D0': D0, 'w_ft': w, 'ratio': ratio,
        'misfit': m,
        'peak_dp_psi': {f"g{g}": float(dp[:, j].max())
                        for j, g in enumerate(_G['gauges'])},
        'trough_dp_psi': {f"g{g}": float(dp[:, j].min())
                          for j, g in enumerate(_G['gauges'])},
        'wall_s': wall,
        'taxis_sha256': rm.sha256_array(ta3),
        'n_t': int(ta3.size),
        'trace_sha256': rm.sha256_array(tr),
        'barrier': None if brep is None else {
            'n_barriers': brep['n_barriers'],
            'n_fallback': brep['n_fallback'],
            'n_overlapping_pairs': brep['n_overlapping_pairs'],
            'n_merged_groups': brep['n_merged_groups'],
            'realised_full_width_ft': brep['realised_full_width_ft'],
            'total_equivalent_width_ft': brep['total_equivalent_width_ft'],
            'excess_resistance_s_per_ft': brep['excess_resistance_s_per_ft'],
            'i0': [b['i0'] for b in brep['barriers']],
            'i1': [b['i1'] for b in brep['barriers']],
            'd_barrier_ft2_s': float(D0 * ratio),
        },
    }
    return out, np.ascontiguousarray(dp[::dec]), np.ascontiguousarray(dp)


# ---------------------------------------------------------------------------
# the chain prelude: phases 1 and 2, once per background D
# ---------------------------------------------------------------------------

def chain_prelude(cfg, x, srcs, D0):
    base = np.full(len(x), float(D0), dtype=float)
    tcfg, pcfg = cfg['time'], dict(cfg['physics'])
    pcfg['D_baseline_ft2_s'] = float(D0)
    s1, s2 = srcs['phase1'], srcs['phase2']

    t = time.time()
    init1 = np.full(len(x), float(s1['values_psi'][0]))
    ta1, f1, trec1, ex1 = a5.solve_phase(x, base, s1, init1, tcfg, pcfg)
    ex1['wall_s'] = time.time() - t
    end1 = f1[-1].copy()
    del f1

    t = time.time()
    ta2, f2, trec2, ex2 = a5.solve_phase(x, base, s2, end1, tcfg, pcfg)
    ex2['wall_s'] = time.time() - t
    init3 = f2[-1].copy()
    del f2
    return {'init3': init3, 'ta1': ta1, 'ta2': ta2, 'trec1': trec1,
            'trec2': trec2, 'ex1': ex1, 'ex2': ex2,
            'ic_phase1_psi': float(s1['values_psi'][0])}


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def build_jobs(cfg, grid):
    lo, hi = float(grid['log10_ratio_start']), float(grid['log10_ratio_stop'])
    step = float(grid['log10_ratio_step'])
    n = int(round(abs(hi - lo) / step)) + 1
    logs = lo + np.sign(hi - lo) * step * np.arange(n)
    ratios = [float(10.0 ** v) for v in logs]
    r_uni = float(grid['uniform_reference_ratio'])
    r_seal = float(grid['sealed_reference_ratio'])
    ws = [float(v) for v in grid['w_ft']]

    plane_jobs, dcurve_jobs = [], []
    for D0 in [float(v) for v in grid['planes_D_ft2_s']]:
        for w in ws:
            for r in ratios:
                plane_jobs.append({'kind': 'plane', 'D0': D0, 'w': w, 'ratio': r})
        # references: the uniform control is w-independent; the sealed control is
        # run at the reference width so a "flat because sealed" cell has something
        # to be compared against.
        plane_jobs.append({'kind': 'plane_ref', 'D0': D0, 'w': ws[0],
                           'ratio': r_uni})
        plane_jobs.append({'kind': 'plane_ref', 'D0': D0, 'w': 1.0,
                           'ratio': r_seal})
    dc = grid['dcurve']
    for D0 in [float(v) for v in dc['D_ft2_s']]:
        for r in [float(v) for v in dc['ratios']]:
            dcurve_jobs.append({'kind': 'dcurve', 'D0': D0,
                                'w': float(dc['w_ft']), 'ratio': r})
    return ratios, ws, plane_jobs, dcurve_jobs


def run_sweep(cfg_path, outroot, tag, n_workers):
    t_all = time.time()
    root = load_config(cfg_path)
    cfg = root['base']
    grid = root['grid']
    os.makedirs(outroot, exist_ok=True)

    md_table = rd.load_gauge_md_table()
    x, mesh_rec = a5.build_chain_mesh(cfg['mesh'])
    wins, win_rec = a5.phase_windows(cfg['phase_boundaries'])
    gauges = [int(g) for g in cfg['targets']['gauges']]
    shielded = [int(g) for g in cfg['targets']['shielded_gauges']]
    gidx = [int(np.argmin(np.abs(x - md_table.md_of(g)))) for g in gauges]
    hits7 = rd.load_frac_hits(7, unique=False, sort=False)
    srcs = {p: make_source(cfg, x, wins, p) for p in PHASES}
    s3 = srcs['phase3']
    field = load_field(gauges, wins)

    ratios, ws, plane_jobs, dcurve_jobs = build_jobs(cfg, grid)
    all_D = sorted({j['D0'] for j in plane_jobs} | {j['D0'] for j in dcurve_jobs})

    print(f"[C4] nx={len(x)} gauges={gauges} shielded={shielded}", flush=True)
    print(f"[C4] {len(ws)} w x {len(ratios)} ratios x "
          f"{len(grid['planes_D_ft2_s'])} planes + {len(dcurve_jobs)} dcurve "
          f"cells = {len(plane_jobs) + len(dcurve_jobs)} phase-3 solves",
          flush=True)

    # ---- phases 1 and 2, once per background D ---------------------------
    prel = {}
    for D0 in all_D:
        t = time.time()
        prel[f"{D0:g}"] = chain_prelude(cfg, x, srcs, D0)
        print(f"[C4] prelude D={D0:g} done ({time.time() - t:.1f} s)", flush=True)

    ta3_ref = None
    fld = None

    # one full-field phase-3 solve to prove record_idx changes nothing
    base140 = np.full(len(x), float(all_D[0]), dtype=float)
    dpr, _ = rc.build_barrier_profile(
        x, base140, hits7, 1.0, 1e-4, ratio_reference=cfg['barrier']['ratio_reference'],
        combine=cfg['barrier']['combine'], on_empty=cfg['barrier']['on_empty'],
        on_outside=cfg['barrier']['on_outside'], return_report=True)
    idx3 = [int(i) for i in s3['source_idx']]
    common = dict(theta=float(cfg['physics']['theta']),
                  lambda_leak=float(cfg['physics']['lambda_leak']),
                  p0=float(cfg['physics']['p0_psi']),
                  interface_avg=cfg['physics']['interface_avg'],
                  theta_startup_steps=int(cfg['physics']['theta_startup_steps']))
    ta_a, tr_a = rc.solve_forward_multi(
        x, dpr, float(cfg['time']['dt_fixed_s']), float(s3['t_total_s']),
        [s3['taxis_s']] * len(idx3), [s3['values_psi']] * len(idx3), idx3,
        initial=prel[f"{all_D[0]:g}"]['init3'], t0=0.0, record_idx=gidx, **common)
    ta_b, f_b = rc.solve_forward_multi(
        x, dpr, float(cfg['time']['dt_fixed_s']), float(s3['t_total_s']),
        [s3['taxis_s']] * len(idx3), [s3['values_psi']] * len(idx3), idx3,
        initial=prel[f"{all_D[0]:g}"]['init3'], t0=0.0, **common)
    rec_identity = float(np.abs(tr_a - f_b[:, gidx]).max())
    rec_taxis_equal = bool(np.array_equal(ta_a, ta_b))
    del f_b
    ta3_ref = ta_a
    fld = field_on_phase3(field, gauges, s3['t0_abs'], ta3_ref)
    print(f"[C4] record_idx identity: max|diff| = {rec_identity:.3e} psi, "
          f"taxis equal = {rec_taxis_equal}", flush=True)

    # ---- fork the pool with everything already in globals ----------------
    _G.update({'x': x, 'hits7': hits7, 's3': s3, 'gidx': gidx, 'gauges': gauges,
               'shielded': shielded, 'fld': fld,
               'tcfg': cfg['time'], 'pcfg': cfg['physics'],
               'bcfg': cfg['barrier'],
               'dec_s': float(cfg['outputs']['trace_decimation_s']),
               'init3': {k: v['init3'] for k, v in prel.items()}})

    jobs = plane_jobs + dcurve_jobs
    n_workers = max(1, min(int(n_workers), MAX_WORKERS))
    t_pool = time.time()
    results, dec_traces, full_traces = [], [], {}
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=n_workers) as pool:
        for k, (res, dec, full) in enumerate(
                pool.imap(_cell, jobs, chunksize=1)):
            res['kind'] = jobs[k]['kind']
            results.append(res)
            dec_traces.append(dec)
            if jobs[k]['kind'] == 'plane_ref' or (
                    jobs[k]['kind'] == 'plane' and jobs[k]['w'] == 1.0):
                full_traces[cell_tag(res['D0'], res['w_ft'], res['ratio'])] = full
            if (k + 1) % 25 == 0 or k + 1 == len(jobs):
                print(f"[C4]   {k + 1}/{len(jobs)} cells "
                      f"({time.time() - t_pool:.0f} s)", flush=True)
    pool_wall = time.time() - t_pool

    # every phase-3 solve shares one time axis; assert it rather than declare
    # 600-odd identical time records
    tsha = sorted({r['taxis_sha256'] for r in results})
    if len(tsha) != 1 or tsha[0] != rm.sha256_array(ta3_ref):
        raise RuntimeError(f"phase-3 time axes differ across cells: {tsha[:3]}")

    # the barrier NODE MASK is a function of w alone; the ratio only scales D
    # inside it. Assert it, so one barrier_record per (w, barrier) is complete.
    masks = {}
    for r in results:
        if r['barrier'] is None:
            continue
        key = f"{r['w_ft']:g}"
        sig = (tuple(r['barrier']['i0']), tuple(r['barrier']['i1']))
        masks.setdefault(key, set()).add(sig)
    bad = {k: len(v) for k, v in masks.items() if len(v) != 1}
    if bad:
        raise RuntimeError(f"barrier node mask is not ratio-independent: {bad}")

    payload = {'x_md_ft': x, 'gauge_numbers': np.asarray(gauges),
               'gauge_md_ft': np.asarray([md_table.md_of(g) for g in gauges]),
               'gauge_mesh_idx': np.asarray(gidx),
               'phase3_taxis_s': ta3_ref,
               'phase3_t0_abs': np.array(str(s3['t0_abs'])),
               'trace_decimation_s': np.array(
                   float(cfg['outputs']['trace_decimation_s'])),
               'trace_units': np.array('dP_psi_from_phase3_first_sample'),
               'ratios': np.asarray(ratios), 'w_ft': np.asarray(ws),
               'planes_D': np.asarray([float(v) for v in grid['planes_D_ft2_s']]),
               'dcurve_D': np.asarray([float(v) for v in grid['dcurve']['D_ft2_s']]),
               'dcurve_ratios': np.asarray(
                   [float(v) for v in grid['dcurve']['ratios']])}
    for r, dec in zip(results, dec_traces):
        payload[f"dp_{cell_tag(r['D0'], r['w_ft'], r['ratio'])}"] = dec.astype(
            np.float32)
    for k, v in full_traces.items():
        payload[f"full_{k}"] = v
    for D0 in all_D:
        payload[f"init3_D{D0:g}"] = prel[f"{D0:g}"]['init3']

    arr_path = os.path.join(outroot, f"c4_cells_{tag}.npz")
    rm.assert_absent([arr_path])
    np.savez_compressed(arr_path, **payload)

    cells = [{k: v for k, v in r.items()} for r in results]
    met_path = os.path.join(outroot, f"c4_cells_{tag}.json")
    rm.assert_absent([met_path])
    with open(met_path, 'w') as fh:
        json.dump({'study_id': STUDY_ID, 'task_id': TASK_ID, 'tag': tag,
                   'generated_utc': utcnow(),
                   'misfit_definition': {
                       'primary': ('gauge-mean RMSE over the SHIELDED gauges '
                                   '{5, 6} of dP referenced to each series own '
                                   'first in-window phase-3 sample'),
                       'aggregation': 'gauge-mean, not sample-pooled (C1)',
                       'secondary': ['gauge_mean_rmse_delta_psi (all five)',
                                     'gauge_mean_rmse_abs_psi (all five)']},
                   'w_ft': ws, 'ratios': ratios,
                   'record_idx_identity_max_abs_psi': rec_identity,
                   'record_idx_taxis_equal': rec_taxis_equal,
                   'pool_wall_s': pool_wall, 'n_workers': n_workers,
                   'cells': cells}, fh, indent=1, sort_keys=True)

    # ---- manifest --------------------------------------------------------
    manifest_path = os.path.join(outroot, 'manifest.json')
    with rm.RunRecorder(manifest_path, study_id=STUDY_ID, task_id=TASK_ID,
                        config=root, config_path=cfg_path, run_label=tag,
                        require_modules=('rev2_core', 'rev2_data',
                                         'rev2_manifest',
                                         'a5_two_stage_chain')) as R:
        drv = {}
        for p in PHASES:
            s = srcs[p]
            drv[p] = rm.driver_record(
                kind='gauge_series',
                baseline_removal=cfg['source']['baseline_removal'],
                value_units=cfg['source']['value_units'],
                series_path=rd.repo_path(
                    rd.SWELL_GAUGE_TEMPLATE.format(n=int(s['gauge']))),
                gauge_number=int(s['gauge']), gauge_md_ft=float(s['md_ft']),
                taxis=s['taxis_s'], values=s['values_psi'],
                time_start=s['window_abs'][0], time_end=s['window_abs'][1])
        src_groups = []
        for p in PHASES:
            s = srcs[p]
            src_groups.append([
                rm.source_record(x, md_requested_ft=float(h), mesh_idx=int(i),
                                 driver=drv[p],
                                 label=f"{p}:stage{s['frac_hit_stage']}_hit_"
                                       f"MD{float(h):.2f}",
                                 index_in_source_list=k)
                for k, (h, i) in enumerate(zip(s['frac_hit_mds_ft'],
                                               s['source_idx']))])
        source_group = rm.source_protocol(
            application=cfg['source']['application'],
            solver_class='rev2_core.solve_forward_multi',
            placement_rule=cfg['source']['placement_rule'],
            sources=src_groups, phase_labels=list(PHASES),
            targets={'gauges': gauges,
                     'md_ft': [md_table.md_of(g) for g in gauges],
                     'shielded_gauges': shielded,
                     'role': ('observation points for a FORWARD sweep; nothing '
                              'is fitted by an optimiser, the misfit is '
                              'evaluated on a grid. Gauges 6 and 7 are also the '
                              'phase-1/2 and phase-3 Dirichlet drivers, so they '
                              'are boundary conditions displayed as data (D4); '
                              'gauge 6 is a free observation in phase 3 only.')},
            time_level='n' if float(cfg['physics']['theta']) == 1.0 else 'n+1',
            phase_chaining={
                'order': list(PHASES),
                'rule': ('each phase starts from the previous phase FINAL '
                         'spatial profile; t0 = 0 in every phase, as 101 does'),
                'barrier_source_stage': 7, 'barrier_applies_in': 'phase3',
                'phases_1_2_are_ratio_and_w_independent': True,
                'phases_1_2_solved_once_per_background_D': [float(v)
                                                            for v in all_D]},
            boundary_conditions=cfg['source']['boundary_conditions'])

        trecs = []
        for D0 in all_D:
            pr = prel[f"{D0:g}"]
            trecs.append(rm.time_record(pr['ta1'], label=f"phase1@D={D0:g}",
                                        **pr['trec1']))
            trecs.append(rm.time_record(pr['ta2'], label=f"phase2@D={D0:g}",
                                        **pr['trec2']))
        trecs.append(rm.time_record(
            ta3_ref, mode='fixed', theta=float(cfg['physics']['theta']),
            t_total_requested_s=float(s3['t_total_s']),
            dt_requested_s=float(cfg['time']['dt_fixed_s']),
            source_time_level='n',
            theta_startup_steps=int(cfg['physics']['theta_startup_steps']),
            label=(f"phase3 (ALL {len(jobs)} cells share this axis; asserted "
                   f"identical by sha256 of the realised taxis)")))

        blist = []
        seen_w = set()
        for r in sorted(results, key=lambda c: abs(np.log10(c['ratio']) + 4.0)):
            if r['barrier'] is None or r['w_ft'] in seen_w:
                continue
            seen_w.add(r['w_ft'])
            for b, (i0, i1) in enumerate(zip(r['barrier']['i0'],
                                             r['barrier']['i1'])):
                mask = np.zeros(len(x), dtype=bool)
                mask[i0:i1 + 1] = True
                blist.append(rm.barrier_record(
                    x, mask,
                    label=(f"stage7_frachit_MD{hits7[b]:.2f}@w={r['w_ft']:g}ft "
                           f"(node mask asserted ratio-independent)"),
                    centre_md_ft=float(hits7[b]),
                    w_requested_ft=float(r['w_ft']),
                    ratio=float(r['ratio']), d_baseline=float(r['D0'])))

        head = [r for r in results
                if r['D0'] == 140.0 and r['w_ft'] == 1.0
                and abs(np.log10(r['ratio']) + 4.0) < 1e-9]
        numerics = rm.numerics(
            time=trecs,
            mesh=rm.mesh_record(
                x, dx_requested_ft=float(cfg['mesh']['dx_ft']),
                window_md_ft=(float(cfg['mesh']['md_lo_ft']),
                              float(cfg['mesh']['md_hi_ft'])),
                pad_low_ft=float(cfg['mesh']['pad_low_ft']),
                pad_high_ft=float(cfg['mesh']['pad_high_ft']),
                refinement=mesh_rec),
            interface_avg=cfg['physics']['interface_avg'],
            boundary=cfg['source']['boundary_conditions'],
            diffusivity={
                'baseline_D_ft2_s': [float(v) for v in all_D],
                'profile_family': 'uniform_plus_physical_width_barriers',
                'param_names': ['D_baseline', 'ratio', 'w_half_width_ft'],
                'params': {'D_baseline_ft2_s': [float(v) for v in all_D],
                           'ratio': ratios + [float(grid['uniform_reference_ratio']),
                                              float(grid['sealed_reference_ratio'])],
                           'w_half_width_ft': ws},
                'D_min': float(min(r['D0'] * r['ratio'] for r in results)),
                'D_max': float(max(r['D0'] for r in results)),
                'D_sha256': rm.sha256_array(dpr),
                'profile_anchor': 'physical_md',
                'note': ('D_sha256 is the D = 140, w = 1 ft, ratio = 1e-4 '
                         'phase-3 profile, the cell used for the record_idx '
                         'identity check; every cell profile is the uniform '
                         'baseline with six stage-7 barriers built by '
                         'rev2_core.build_barrier_profile. Phases 1 and 2 are '
                         'uniform.')},
            barriers=blist if blist else rm.NONE_DECLARED,
            leakage=rm.NONE_DECLARED,
            kernel={'name': 'rev2_core.solve_forward_multi', 'banded': True,
                    'equivalence_reference':
                        'output/rev2_20260901/A4/selftest_output.txt',
                    'note': ('rev2_core at theta=1 / harmonic / lambda=0 is '
                             'bitwise identical to the verified R1 kernel. '
                             'record_idx was checked against the full-field '
                             'solve in THIS run: see results.')},
            rng=rm.NONE_DECLARED,
            parallel={'mode': 'multiprocessing.Pool', 'n_workers': n_workers,
                      'start_method': 'fork',
                      'blas_threads_per_worker': 1,
                      'why': ('phase-3 cells are independent; phases 1 and 2 are '
                              'solved serially in the parent because they are '
                              'shared by every cell at that background D'),
                      'determinism': ('each cell is a deterministic banded solve; '
                                      'the pool only changes the ORDER of '
                                      'completion, and results are re-keyed by '
                                      '(D0, w, ratio)')},
            amplification=rc.amplification_factor(
                x, dpr, float(cfg['time']['dt_fixed_s']),
                float(cfg['physics']['theta']),
                interface_avg=cfg['physics']['interface_avg']))

        inputs = [(rd.repo_path(rd.SWELL_GAUGE_TEMPLATE.format(n=g)),
                   'gauge_series', f'gauge{g}_swell')
                  for g in sorted(set(gauges) | {6, 7})]
        inputs += [(rd.repo_path(rd.SWELL_FRAC_HIT_TEMPLATE.format(stage=s)),
                    'geometry', f'frac_hit_stage_{s}') for s in (7, 8)]
        inputs += [(rd.repo_path(rd.SWELL_GAUGE_MD_NPZ), 'geometry',
                    'gauge_md_swell')]
        for s in (7, 8):
            inputs.append((rd.repo_path(rd.PUMPING_DIR_TEMPLATE.format(stage=s),
                                        rd.PUMPING_CURVE_FILES['slurry_rate']),
                           'pumping', f'stage{s}_slurry_rate'))
        R.declare_inputs(inputs)
        R.declare_output(arr_path, role='arrays_npz',
                         note=('per-cell dP traces (decimated to '
                               f"{cfg['outputs']['trace_decimation_s']:g} s by "
                               'STRIDE, float32) plus full 1 s traces for the '
                               'reference cells and the whole w = 1 ft row, and '
                               'the phase-2 final profile per background D'))
        R.declare_output(met_path, role='json',
                         note='per-cell misfit, peaks and barrier report summary')
        R.set_source(source_group)
        R.set_numerics(numerics)
        R.set_results({
            'n_phase3_solves': len(jobs),
            'n_plane_cells': len(plane_jobs), 'n_dcurve_cells': len(dcurve_jobs),
            'record_idx_identity_max_abs_psi': rec_identity,
            'record_idx_taxis_equal': rec_taxis_equal,
            'phase3_taxis_sha256': rm.sha256_array(ta3_ref),
            'all_cells_share_one_phase3_taxis': True,
            'barrier_node_mask_is_ratio_independent': True,
            'headline_cell_D140_w1_r1e-4_shielded_rmse_psi':
                (head[0]['misfit']['shielded_mean_rmse_delta_psi']
                 if head else None),
            'wall_s_total': time.time() - t_all,
            'wall_s_pool': pool_wall,
        })
        R.note("C4: forward misfit sweep over the (w, ratio) plane at three "
               "background D, plus a 1-D curve in D. Nothing is fitted by an "
               "optimiser; the misfit is evaluated on a grid so the SHAPE of the "
               "valley, not just its argmin, is measurable.")
        R.note("Geometry and misfit are E1's, deliberately: it is the only case "
               "in this round where a barrier is both present and observable "
               "against measured data (B2/E2 retired the Fig. 7b production "
               "case; the R1 window has no barrier). The D = 140, w = 1 ft row "
               "must reproduce E1's published ladder exactly, and that check is "
               "in the analysis product.")
        R.note("Phases 1 and 2 carry no barrier and are solved ONCE per "
               "background D; every phase-3 cell restarts from that stored final "
               "profile. Phase 3 is solved with record_idx at the five gauge "
               "nodes, verified in this run to be bitwise identical to the "
               "full-field solve at those columns.")
        R.note("Only ONE time_record is declared for phase 3 even though "
               f"{len(jobs)} phase-3 solves were run: fixed dt = 1 s and one "
               "common source window make every realised time axis identical, "
               "which is asserted by comparing the sha256 of each cell's taxis. "
               "Likewise one barrier_record per (w, barrier): the node mask is a "
               "function of w alone and the ratio only scales D inside it, also "
               "asserted per cell.")
    print(f"[C4] sweep done in {time.time() - t_all:.0f} s -> {outroot}",
          flush=True)
    return 0


# ---------------------------------------------------------------------------
# analysis (solve-free)
# ---------------------------------------------------------------------------

def _parabola_min(xs, ys):
    """Vertex of the parabola through three points; None if not a minimum."""
    (x1, x2, x3), (y1, y2, y3) = xs, ys
    d = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if d == 0:
        return None, None
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / d
    b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / d
    c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2
         + x1 * x2 * (x1 - x2) * y3) / d
    if a <= 0:
        return None, None
    xv = -b / (2 * a)
    return float(xv), float(a * xv * xv + b * xv + c)


def profile_band(xg, yg, level):
    """The contiguous <= level interval containing the argmin, with censoring."""
    xg = np.asarray(xg, float)
    yg = np.asarray(yg, float)
    if xg[0] > xg[-1]:                      # log10(ratio) runs downward
        xg, yg = xg[::-1], yg[::-1]
    k = int(np.argmin(yg))
    lo_i = k
    while lo_i > 0 and yg[lo_i - 1] <= level:
        lo_i -= 1
    hi_i = k
    while hi_i < len(xg) - 1 and yg[hi_i + 1] <= level:
        hi_i += 1
    if lo_i == 0:
        lo, lo_cens = float(xg[0]), True
    else:
        y0, y1 = yg[lo_i - 1], yg[lo_i]
        lo = float(xg[lo_i - 1] + (level - y0) * (xg[lo_i] - xg[lo_i - 1])
                   / (y1 - y0))
        lo_cens = False
    if hi_i == len(xg) - 1:
        hi, hi_cens = float(xg[-1]), True
    else:
        y0, y1 = yg[hi_i], yg[hi_i + 1]
        hi = float(xg[hi_i] + (level - y0) * (xg[hi_i + 1] - xg[hi_i])
                   / (y1 - y0))
        hi_cens = False
    return {'lo': lo, 'hi': hi, 'lo_censored': lo_cens, 'hi_censored': hi_cens,
            'width': hi - lo,
            'argmin_at_grid_edge': bool(k == 0 or k == len(xg) - 1)}


def _cross_log(xg, yg, level):
    """Largest x where the piecewise-linear y(x) crosses `level`, else None.

    `xg` is log10(ratio) and runs downward; `yg` is a separation in psi that
    falls monotonically as the barrier seals. Returns None when the curve never
    crosses inside the grid, which is a CENSORED edge, not an estimate.
    """
    xg = np.asarray(xg, float)
    yg = np.asarray(yg, float)
    hits = []
    for i in range(len(xg) - 1):
        y0, y1 = yg[i], yg[i + 1]
        if (y0 - level) * (y1 - level) <= 0 and y0 != y1:
            hits.append(float(xg[i] + (level - y0) * (xg[i + 1] - xg[i])
                              / (y1 - y0)))
    if not hits:
        return None
    return float(max(hits))


def linfit(u, v):
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    n = u.size
    A = np.vstack([u, np.ones(n)]).T
    coef, res, *_ = np.linalg.lstsq(A, v, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((v - pred) ** 2))
    ss_tot = float(np.sum((v - v.mean()) ** 2))
    dof = max(n - 2, 1)
    s2 = ss_res / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return {'slope': float(coef[0]), 'intercept': float(coef[1]),
            'slope_stderr': float(np.sqrt(cov[0, 0])),
            'r2': float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
            'n': int(n), 'residual_rms': float(np.sqrt(ss_res / n))}


def collapse_scatter(xv, yv, bin_width):
    """Leave-one-out residual RMS of y about a binned master curve of x."""
    xv = np.asarray(xv, float)
    yv = np.asarray(yv, float)
    edges = np.arange(np.floor(xv.min() / bin_width) * bin_width,
                      xv.max() + bin_width, bin_width)
    b = np.clip(np.digitize(xv, edges) - 1, 0, len(edges) - 1)
    res, used = [], 0
    for k in np.unique(b):
        m = b == k
        if m.sum() < 3:
            continue
        ys = yv[m]
        n = ys.size
        loo = (ys.sum() - ys) / (n - 1)
        res.extend(list(ys - loo))
        used += n
    if not res:
        return None
    res = np.asarray(res)
    return {'bin_width_decades': float(bin_width),
            'n_used': int(used), 'n_total': int(xv.size),
            'loo_residual_rms_psi': float(np.sqrt(np.mean(res ** 2))),
            'loo_residual_max_psi': float(np.max(np.abs(res)))}


def run_analyse(cfg_path, outroot, sweep_dir, tag, sweep_tag,
                padcheck=None):
    root = load_config(cfg_path)
    grid = root['grid']
    met_path = os.path.join(sweep_dir, f"c4_cells_{sweep_tag}.json")
    arr_path = os.path.join(sweep_dir, f"c4_cells_{sweep_tag}.npz")
    man_path = os.path.join(sweep_dir, 'manifest.json')
    with open(met_path) as fh:
        doc = json.load(fh)
    cells = doc['cells']
    ws = [float(v) for v in doc['w_ft']]
    ratios = [float(v) for v in doc['ratios']]
    planes = [float(v) for v in grid['planes_D_ft2_s']]

    def get(D0, w, r, kinds=('plane', 'plane_ref', 'dcurve')):
        for c in cells:
            if (c['kind'] in kinds and abs(c['D0'] - D0) < 1e-9
                    and abs(c['w_ft'] - w) < 1e-12
                    and abs(np.log10(c['ratio']) - np.log10(r)) < 1e-9):
                return c
        return None

    M = {}          # M[D0] = (n_w, n_r) shielded misfit
    for D0 in planes:
        A = np.full((len(ws), len(ratios)), np.nan)
        for i, w in enumerate(ws):
            for j, r in enumerate(ratios):
                c = get(D0, w, r, kinds=('plane',))
                A[i, j] = c['misfit']['shielded_mean_rmse_delta_psi']
        M[D0] = A

    out = {'kind': 'derived_view_no_solve', 'tag': tag,
           'generated_utc': utcnow(),
           'sources': {'cells_json': rm._rel(met_path),
                       'cells_json_sha256': rm.sha256_file(met_path),
                       'cells_npz': rm._rel(arr_path),
                       'cells_npz_sha256': rm.sha256_file(arr_path),
                       'sweep_manifest': rm._rel(man_path),
                       'sweep_manifest_sha256': rm.sha256_file(man_path),
                       'config': rm._rel(cfg_path),
                       'config_sha256': rm.sha256_file(cfg_path)},
           'code_sha256': rm.sha256_file(os.path.abspath(__file__)),
           'misfit_definition': doc['misfit_definition'],
           'grid': {'w_ft': ws, 'ratios': ratios, 'planes_D': planes,
                    'log10_ratio_step': float(grid['log10_ratio_step'])}}

    # ---- E1 cross-check ---------------------------------------------------
    xc = root['cross_checks']['e1_reference']
    rows = []
    for k, v in xc['expect_shielded_mean_rmse_delta_psi'].items():
        r = float(k)
        c = get(140.0, 1.0, r)
        got = None if c is None else c['misfit']['shielded_mean_rmse_delta_psi']
        rows.append({'ratio': r, 'E1_published_psi': float(v),
                     'C4_psi': got,
                     'difference_psi': None if got is None else got - float(v)})
    out['e1_cross_check'] = {
        'reference': xc['path'], 'study': xc['study'],
        'note': ('E1 published these to 2 decimals; the comparison is against '
                 'those rounded values, so agreement to < 0.005 psi is exact '
                 'agreement at the published precision.'),
        'rows': rows,
        'max_abs_difference_psi': (
            float(np.nanmax([abs(r['difference_psi']) for r in rows
                             if r['difference_psi'] is not None]))
            if any(r['difference_psi'] is not None for r in rows) else None),
        'n_rows_present': int(sum(1 for r in rows
                                  if r['difference_psi'] is not None))}

    # ---- free reproducibility check: cells solved twice -------------------
    # Three of the D-curve backgrounds (140, 550, 1150) also carry a full plane,
    # and the D curve runs at w = 1 ft, so every half-decade ratio on those three
    # rows is solved TWICE through the same code path in the same pool. The two
    # answers must be bit-identical, and the trace sha256 says so directly.
    seen = {}
    dups = []
    for c in cells:
        if c['kind'] not in ('plane', 'dcurve'):
            continue
        k = (c['D0'], c['w_ft'], round(float(np.log10(c['ratio'])), 9))
        if k in seen:
            o = seen[k]
            import struct as _st
            dups.append({'D0': c['D0'], 'w_ft': c['w_ft'], 'ratio': c['ratio'],
                         'kinds': [o['kind'], c['kind']],
                         'ratio_bits': [_st.pack('>d', o['ratio']).hex(),
                                        _st.pack('>d', c['ratio']).hex()],
                         'ratio_bits_equal': bool(o['ratio'] == c['ratio']),
                         'ratio_ulp_difference': int(abs(
                             _st.unpack('>q', _st.pack('>d', o['ratio']))[0]
                             - _st.unpack('>q', _st.pack('>d', c['ratio']))[0])),
                         'trace_sha256_equal':
                             bool(o['trace_sha256'] == c['trace_sha256']),
                         'misfit_difference_psi': float(
                             c['misfit']['shielded_mean_rmse_delta_psi']
                             - o['misfit']['shielded_mean_rmse_delta_psi'])})
        else:
            seen[k] = c
    out['duplicate_cell_check'] = {
        'note': ('the plane ratios are built as 10**(-2 - 0.25k) and the D-curve '
                 'ratios are decimal literals in the config, so a half-decade '
                 'value can differ by 1 ulp between the two paths; where the '
                 'bits agree the traces are bit-identical, and where they differ '
                 'by 1 ulp the misfit moves by ~1e-10 psi, which is the '
                 'conditioning of the whole chain with respect to the barrier '
                 'diffusivity'),
        'n_bitwise_identical': int(sum(1 for q in dups
                                       if q['trace_sha256_equal'])),
        'n_ratio_bits_differ': int(sum(1 for q in dups
                                       if not q['ratio_bits_equal'])),
        'max_abs_misfit_difference_where_ratio_bits_equal_psi': float(max(
            [abs(q['misfit_difference_psi']) for q in dups
             if q['ratio_bits_equal']] or [0.0])),
        'n_duplicated_cells': len(dups),
        'all_trace_sha256_equal': bool(all(d['trace_sha256_equal'] for d in dups))
        if dups else None,
        'max_abs_misfit_difference_psi': (
            float(max(abs(d['misfit_difference_psi']) for d in dups))
            if dups else None),
        'rows': dups}

    # ---- guard against the batch-5 build_barrier_profile width defect ----
    # An independent verifier found that build_barrier_profile can realise
    # 2w + dx instead of 2w when BOTH endpoints of [x_hit-w, x_hit+w] land on
    # mesh nodes; report['n_fallback'] stays 0, so the assertion used by A1, A2,
    # E1 and E4 does not catch it, and the excess series resistance is then up to
    # 50% too large. Every cell's own report is recorded, so the check is made
    # here rather than assumed.
    infl = []
    for c in cells:
        b = c.get('barrier')
        if not b:
            continue
        for k in ('min', 'max'):
            infl.append(abs(b['realised_full_width_ft'][k]
                            / (2.0 * c['w_ft']) - 1.0))
    out['barrier_width_check'] = {
        'defect': ('rev2_core.build_barrier_profile realises 2w + dx when both '
                   'endpoints of [x_hit-w, x_hit+w] land on nodes (batch-5 '
                   'independent verification); n_fallback stays 0'),
        'worst_relative_width_error': float(max(infl)) if infl else None,
        'n_cells_checked': int(len(infl) // 2),
        'verdict': ('does not bite C4: every barrier on every cell realises '
                    'exactly 2w' if infl and max(infl) < 1e-12
                    else 'CHECK FAILED -- widths are not 2w'),
        'n_fallback_max': int(max(c['barrier']['n_fallback'] for c in cells
                                  if c.get('barrier'))),
        'n_overlapping_pairs_max': int(max(c['barrier']['n_overlapping_pairs']
                                           for c in cells if c.get('barrier'))),
    }

    # ---- references and regimes ------------------------------------------
    refs = {}
    for D0 in planes:
        u = get(D0, ws[0], float(grid['uniform_reference_ratio']),
                kinds=('plane_ref',))
        s = get(D0, 1.0, float(grid['sealed_reference_ratio']),
                kinds=('plane_ref',))
        refs[f"{D0:g}"] = {
            'uniform_misfit_psi': u['misfit']['shielded_mean_rmse_delta_psi'],
            'sealed_misfit_psi': s['misfit']['shielded_mean_rmse_delta_psi'],
            'uniform_peak_dp_psi': u['peak_dp_psi'],
            'sealed_peak_dp_psi': s['peak_dp_psi']}
    out['references'] = refs

    # ---- valley trace per plane ------------------------------------------
    lr = np.log10(np.asarray(ratios))
    lw = np.log10(np.asarray(ws))
    valley = {}
    for D0 in planes:
        A = M[D0]
        Mmin = float(np.nanmin(A))
        i_min, j_min = np.unravel_index(int(np.nanargmin(A)), A.shape)
        per_w = []
        for i, w in enumerate(ws):
            y = A[i]
            j = int(np.argmin(y))
            edge = (j == 0 or j == len(y) - 1)
            if edge:
                rv, mv = float(lr[j]), float(y[j])
            else:
                rv, mv = _parabola_min(lr[j - 1:j + 2], y[j - 1:j + 2])
                if rv is None:
                    rv, mv = float(lr[j]), float(y[j])
            band = profile_band(lr, y, 1.1 * float(y[j]))
            per_w.append({
                'w_ft': w,
                'log10_ratio_opt': rv, 'ratio_opt': float(10.0 ** rv),
                'misfit_at_opt_psi': mv,
                'grid_misfit_min_psi': float(y[j]),
                'argmin_on_grid_edge': bool(edge),
                'D_barrier_opt_ft2_s': float(D0 * 10.0 ** rv),
                'log10_ratio_band_10pct': band,
                'total_barrier_width_ft': 6.0 * 2.0 * w})
        good = [p for p in per_w if not p['argmin_on_grid_edge']]
        fit_r = linfit([np.log10(p['w_ft']) for p in good],
                       [p['log10_ratio_opt'] for p in good]) if len(good) > 2 \
            else None
        fit_db = linfit([np.log10(p['w_ft']) for p in good],
                        [np.log10(p['D_barrier_opt_ft2_s']) for p in good]) \
            if len(good) > 2 else None
        floor = [p['misfit_at_opt_psi'] for p in good]
        floor_grid = [p['grid_misfit_min_psi'] for p in per_w]
        # profiles
        prof_w = np.nanmin(A, axis=1)
        prof_r = np.nanmin(A, axis=0)
        lvl = 1.1 * Mmin
        valley[f"{D0:g}"] = {
            'global_min_psi': Mmin,
            'global_min_at': {'w_ft': ws[i_min], 'ratio': ratios[j_min],
                              'D_barrier_ft2_s': D0 * ratios[j_min]},
            'global_min_on_w_edge': bool(i_min in (0, len(ws) - 1)),
            'global_min_on_ratio_edge': bool(j_min in (0, len(ratios) - 1)),
            'per_w': per_w,
            'floor_variation_psi': (float(max(floor) - min(floor))
                                    if floor else None),
            'floor_variation_pct_of_min': (
                float(100.0 * (max(floor) - min(floor)) / min(floor))
                if floor else None),
            'floor_variation_grid_psi': float(max(floor_grid) - min(floor_grid)),
            'floor_variation_grid_pct_of_min': float(
                100.0 * (max(floor_grid) - min(floor_grid)) / min(floor_grid)),
            'floor_grid_psi': [float(v) for v in floor_grid],
            'w_span_decades': float(np.log10(ws[-1] / ws[0])),
            'fit_log10_ratio_vs_log10_w': fit_r,
            'fit_log10_Dbarrier_vs_log10_w': fit_db,
            'profile_over_ratio_min_psi': [float(v) for v in prof_w],
            'profile_over_w_min_psi': [float(v) for v in prof_r],
            'band_10pct_in_log10_w': profile_band(lw, prof_w, lvl),
            'band_10pct_in_log10_ratio': profile_band(lr, prof_r, lvl),
            'band_10pct_level_psi': lvl,
        }
    out['valley'] = valley

    # ---- along vs across the band ----------------------------------------
    anis = {}
    for D0 in planes:
        v = valley[f"{D0:g}"]
        p = (v['fit_log10_Dbarrier_vs_log10_w'] or {}).get('slope')
        A = M[D0]
        lvl = 1.1 * v['global_min_psi']
        # cells inside the +10% set, in (u = log10 w, v = log10 ratio)
        ii, jj = np.where(A <= lvl)
        if p is None or ii.size == 0:
            anis[f"{D0:g}"] = None
            continue
        u = lw[ii]
        vv = lr[jj]
        # band direction: v = p*u + c  (from log10 D_b = p log10 w + c)
        n = np.sqrt(1.0 + p * p)
        s_along = (u + p * vv) / n
        s_across = (-p * u + vv) / n
        anis[f"{D0:g}"] = {
            'exponent_p_used': float(p),
            'n_cells_within_10pct': int(ii.size),
            'extent_along_band_decades': float(s_along.max() - s_along.min()),
            'extent_across_band_decades': float(s_across.max() - s_across.min()),
            'aspect_ratio': (float((s_along.max() - s_along.min())
                                   / (s_across.max() - s_across.min()))
                             if s_across.max() > s_across.min() else None),
            'note': ('extent measured over GRID CELLS inside the +10% set, so '
                     'both numbers are quantised to the grid step '
                     f"({float(grid['log10_ratio_step'])} decade in ratio, "
                     'irregular in w) and the across-band extent is a lower '
                     'bound of one grid step when only one row qualifies'),
        }
    out['anisotropy'] = anis


    # ---- along the band versus across it ---------------------------------
    # "Along" = w swept over the whole grid with the ratio re-optimised at each w
    # (i.e. walking the valley floor). "Across" = ratio moved at FIXED w until the
    # misfit rises by the same amount. The quotient is the anisotropy, in decades.
    across = {}
    for D0 in planes:
        v = valley[f"{D0:g}"]
        A = M[D0]
        delta = v['floor_variation_grid_psi']
        half = []
        for i, w in enumerate(ws):
            y = A[i]
            j = int(np.argmin(y))
            b = profile_band(lr, y, float(y[j]) + delta)
            if b['lo_censored'] or b['hi_censored']:
                half.append(None)
            else:
                half.append(0.5 * b['width'])
        ok = [h for h in half if h is not None]
        across[f"{D0:g}"] = {
            'delta_psi': delta,
            'note': ('delta is the FULL misfit variation along the valley floor '
                     'over the whole w grid; the across-band half-width is the '
                     'ratio offset at fixed w that costs the same delta'),
            'w_span_decades': v['w_span_decades'],
            'across_halfwidth_decades_by_w': half,
            'across_halfwidth_decades_median': (float(np.median(ok))
                                                if ok else None),
            'anisotropy_along_over_across': (
                float(v['w_span_decades'] / float(np.median(ok)))
                if ok and float(np.median(ok)) > 0 else None),
        }
    out['along_vs_across'] = across

    # ---- the band in the COMBINATION coordinate ---------------------------
    # The contrast the paper needs: each parameter separately is censored, but
    # the combination w/D_barrier is pinned. Cells are binned in
    # kappa = log10(w/D_barrier) and the LOWER ENVELOPE is taken, because at one
    # kappa several (w, ratio) pairs exist and the profile likelihood takes the
    # best of them.
    comb = {}
    for D0 in planes:
        A = M[D0]
        kap, val = [], []
        for i, w in enumerate(ws):
            for j, r in enumerate(ratios):
                kap.append(np.log10(w / (D0 * r)))
                val.append(A[i, j])
        kap = np.asarray(kap)
        val = np.asarray(val)
        bw = 0.10
        edges = np.arange(np.floor(kap.min() / bw) * bw, kap.max() + bw, bw)
        ctr, env = [], []
        for k in range(len(edges) - 1):
            m = (kap >= edges[k]) & (kap < edges[k + 1])
            if m.sum() == 0:
                continue
            ctr.append(float(0.5 * (edges[k] + edges[k + 1])))
            env.append(float(val[m].min()))
        ctr = np.asarray(ctr)
        env = np.asarray(env)
        lvl = 1.1 * float(env.min())
        b = profile_band(ctr, env, lvl)
        comb[f"{D0:g}"] = {
            'coordinate': 'kappa = log10(w / D_barrier), units log10(s/ft)',
            'bin_width_decades': bw,
            'n_bins': int(ctr.size),
            'envelope_min_psi': float(env.min()),
            'kappa_at_min': float(ctr[int(np.argmin(env))]),
            'band_10pct_in_kappa': b,
            'w_over_Dbarrier_at_min_s_per_ft': float(10.0 ** ctr[int(np.argmin(env))]),
            'band_10pct_in_w_over_Dbarrier_s_per_ft': [float(10.0 ** b['lo']),
                                                       float(10.0 ** b['hi'])],
            'note': ('the envelope is the profile over every (w, ratio) pair '
                     'that shares a kappa bin, so this band is the combination '
                     'band to compare against the censored single-parameter '
                     'bands above'),
        }
    out['combination_band'] = comb

    # ---- conditional (not profiled) bands at the manuscript's own values --
    cond = {}
    for D0 in planes:
        A = M[D0]
        iw = int(np.argmin(np.abs(np.asarray(ws) - 1.0)))
        y = A[iw]
        j = int(np.argmin(y))
        r_ms = 1e-5
        jr = int(np.argmin(np.abs(lr - np.log10(r_ms))))
        col = A[:, jr]
        i2 = int(np.argmin(col))
        cond[f"{D0:g}"] = {
            'at_w_1ft': {
                'w_ft': ws[iw], 'grid_min_psi': float(y[j]),
                'ratio_at_min': float(ratios[j]),
                'band_10pct_in_log10_ratio': profile_band(lr, y, 1.1 * float(y[j])),
                'misfit_at_ratio_1e-5_psi': float(A[iw, jr]),
                'penalty_of_1e-5_vs_best_psi': float(A[iw, jr] - y[j]),
                'penalty_of_1e-5_vs_best_pct': float(
                    100.0 * (A[iw, jr] - y[j]) / y[j])},
            'at_ratio_1e-5': {
                'ratio': float(ratios[jr]), 'grid_min_psi': float(col[i2]),
                'w_at_min_ft': float(ws[i2]),
                'band_10pct_in_log10_w': profile_band(lw, col, 1.1 * float(col[i2])),
                'misfit_at_w_1ft_psi': float(A[iw, jr])},
        }
    out['conditional_bands_at_manuscript_values'] = cond

    # ---- regimes: flat because unidentifiable, or flat because SEALED? ----
    # Two separate diagnostics, because A1 warned they are different statements:
    #   * misfit-space: how far this cell sits from the sealed and the uniform
    #     references, as a fraction of the whole misfit range the barrier spans;
    #   * signal-space: the largest difference between this cell's simulated
    #     shielded-gauge dP trace and the sealed/uniform ones, in psi and in
    #     typographic points on the manuscript's own panel.
    # 104's ax3 draws (P - P0)*(-0.15 ft/psi) on an MD axis spanning
    # [min(stage-8 hits) - 500, max(stage-7 hits) + 500] = 1424.29 ft in a ~6 in
    # tall subplot, so 1 pt = 21.98 psi and its own linewidth=2 is 43.96 psi (E1).
    z = np.load(arr_path)
    span_ft = 1424.2857142857142
    psi_per_pt = (span_ft / 6.0 / 72.0) / 0.15
    gnums = [int(g) for g in z['gauge_numbers']]
    sh_idx = [gnums.index(int(g)) for g in (5, 6)]
    r_seal = float(grid['sealed_reference_ratio'])
    reg = {}
    for D0 in planes:
        seal = np.asarray(z[f"dp_{cell_tag(D0, 1.0, r_seal)}"], float)
        uni = np.asarray(z[f"dp_{cell_tag(D0, ws[0], 1.0)}"], float)
        Mu = refs[f"{D0:g}"]['uniform_misfit_psi']
        Ms = refs[f"{D0:g}"]['sealed_misfit_psi']
        Mmin = valley[f"{D0:g}"]['global_min_psi']
        A = M[D0]
        rows, windows = [], []
        for i, w in enumerate(ws):
            ds_w, du_w = [], []
            for j, r in enumerate(ratios):
                dpc = np.asarray(z[f"dp_{cell_tag(D0, w, r)}"], float)
                d_seal = float(np.max(np.abs(dpc[:, sh_idx] - seal[:, sh_idx])))
                d_uni = float(np.max(np.abs(dpc[:, sh_idx] - uni[:, sh_idx])))
                ds_w.append(d_seal)
                du_w.append(d_uni)
                # local misfit slope in psi per decade of ratio
                if j == 0:
                    sl = (A[i, 1] - A[i, 0]) / (lr[1] - lr[0])
                elif j == len(ratios) - 1:
                    sl = (A[i, -1] - A[i, -2]) / (lr[-1] - lr[-2])
                else:
                    sl = (A[i, j + 1] - A[i, j - 1]) / (lr[j + 1] - lr[j - 1])
                rows.append({
                    'w_ft': w, 'ratio': r, 'misfit_psi': float(A[i, j]),
                    'D_barrier_ft2_s': D0 * r,
                    'log10_w_over_Dbarrier': float(np.log10(w / (D0 * r))),
                    'sep_from_sealed_psi': d_seal,
                    'sep_from_uniform_psi': d_uni,
                    'sep_from_sealed_pt': d_seal / psi_per_pt,
                    'dM_dlog10ratio_psi_per_decade': float(sl),
                    'regime': ('indistinguishable_from_a_perfect_seal'
                               if d_seal < psi_per_pt else
                               'indistinguishable_from_no_barrier'
                               if d_uni < psi_per_pt else 'resolved')})
            # the informative window in ratio at this w, by E1's one-point rule
            ds_w = np.asarray(ds_w)
            du_w = np.asarray(du_w)
            lo_edge = _cross_log(lr, ds_w, psi_per_pt)     # sealing edge
            hi_edge = _cross_log(lr, du_w, psi_per_pt)     # transparency edge
            windows.append({
                'w_ft': w,
                'log10_ratio_sealing_edge': lo_edge,
                'log10_ratio_transparency_edge': hi_edge,
                'ratio_sealing_edge': (None if lo_edge is None
                                       else float(10.0 ** lo_edge)),
                'ratio_transparency_edge': (None if hi_edge is None
                                            else float(10.0 ** hi_edge)),
                'window_decades': (None if (lo_edge is None or hi_edge is None)
                                   else float(hi_edge - lo_edge)),
                'sealing_edge_censored': bool(lo_edge is None),
                'transparency_edge_censored': bool(hi_edge is None)})
        okw = [q for q in windows if q['log10_ratio_sealing_edge'] is not None]
        reg[f"{D0:g}"] = {
            'uniform_misfit_psi': Mu, 'sealed_misfit_psi': Ms,
            'grid_min_misfit_psi': Mmin,
            'misfit_range_uniform_minus_min_psi': float(Mu - Mmin),
            'psi_per_typographic_point': psi_per_pt,
            'threshold_note': ("1 typographic point on 104's own ax3 scale "
                               '(0.15 ft/psi over a 1424.29 ft MD axis in a '
                               '~6 in subplot) = 21.98 psi; E1 measured this '
                               'and the manuscript own linewidth=2 is 43.96 psi'),
            'counts': {k: int(sum(1 for q in rows if q['regime'] == k))
                       for k in ('indistinguishable_from_a_perfect_seal',
                                 'indistinguishable_from_no_barrier',
                                 'resolved')},
            'n_cells': len(rows),
            'informative_window_by_w': windows,
            'fit_log10_sealing_edge_vs_log10_w': (
                linfit([np.log10(q['w_ft']) for q in okw],
                       [q['log10_ratio_sealing_edge'] for q in okw])
                if len(okw) > 2 else None),
            'cells': rows,
        }
    out['regimes'] = reg

    # ---- barrier timescales at the valley floor ---------------------------
    T3 = float(z['phase3_taxis_s'][-1])
    ts = {}
    for D0 in planes:
        rows = []
        for pw_ in valley[f"{D0:g}"]['per_w']:
            w = pw_['w_ft']
            Db = pw_['D_barrier_opt_ft2_s']
            Wtot = 6.0 * 2.0 * w
            rows.append({
                'w_ft': w, 'D_barrier_opt_ft2_s': Db,
                'single_barrier_full_width_ft': 2.0 * w,
                'series_full_width_ft': Wtot,
                'crossing_time_single_s': float((2.0 * w) ** 2 / Db),
                'crossing_time_series_s': float(Wtot ** 2 / Db),
                'crossing_time_series_over_window': float(Wtot ** 2 / Db / T3),
                'series_resistance_s_per_ft': float(Wtot / Db),
            })
        ts[f"{D0:g}"] = {'phase3_window_s': T3, 'rows': rows}
    out['barrier_timescales_at_optimum'] = ts

    # ---- the collapse: which combination does the data constrain? --------
    # Reference limits per background D, needed twice below: the sealed and the
    # no-barrier misfit both depend on D0, so a raw-misfit collapse cannot hold
    # in the saturated wings even if the barrier physics collapses perfectly.
    # That is a real limitation, not a nuisance, so it is measured three ways:
    # raw over everything, raw over the INFORMATIVE cells only, and on a misfit
    # normalised by each background's own two limits.
    ref_by_D = {}
    for D0 in planes:
        ref_by_D[D0] = (refs[f"{D0:g}"]['uniform_misfit_psi'],
                        refs[f"{D0:g}"]['sealed_misfit_psi'])
    dcg = grid['dcurve']
    for D0 in [float(v) for v in dcg['D_ft2_s']]:
        cu = get(D0, float(dcg['w_ft']), 1.0, kinds=('dcurve',))
        cs = get(D0, float(dcg['w_ft']), float(grid['sealed_reference_ratio']),
                 kinds=('dcurve',))
        if cu is not None and cs is not None:
            ref_by_D[D0] = (cu['misfit']['shielded_mean_rmse_delta_psi'],
                            cs['misfit']['shielded_mean_rmse_delta_psi'])

    pts = [(c['w_ft'], c['ratio'], c['D0'],
            c['misfit']['shielded_mean_rmse_delta_psi'])
           for c in cells if c['kind'] in ('plane', 'dcurve')
           and c['ratio'] < 1.0
           and abs(np.log10(c['ratio'])
                   - np.log10(float(grid['sealed_reference_ratio']))) > 1e-9]
    W = np.array([p[0] for p in pts])
    R = np.array([p[1] for p in pts])
    D = np.array([p[2] for p in pts])
    Y = np.array([p[3] for p in pts])
    Db = D * R
    Mu = np.array([ref_by_D[p[2]][0] for p in pts])
    Msl = np.array([ref_by_D[p[2]][1] for p in pts])
    Ynorm = (Y - Msl) / (Mu - Msl)
    # informative = not pinned to either limit, at 10% of that background's own
    # uniform-to-sealed span
    span = np.abs(Mu - Msl)
    info = (np.abs(Y - Msl) > 0.10 * span) & (np.abs(Y - Mu) > 0.10 * span)

    def qscan(xw, xdb, yv, mask=None):
        m = np.ones(yv.size, bool) if mask is None else mask
        rows = []
        for q in np.arange(0.30, 3.001, 0.01):
            sc = collapse_scatter(np.log10(xw[m] ** q / xdb[m]), yv[m], 0.10)
            if sc:
                rows.append((float(q), sc['loo_residual_rms_psi'], sc['n_used']))
        bst = min(rows, key=lambda t: t[1]) if rows else None
        return rows, bst

    scan, best = qscan(W, Db, Y)
    scan_i, best_i = qscan(W, Db, Y, info)
    scan_n, best_n = qscan(W, Db, Ynorm)

    def _rows(rows):
        return [{'q': q, 'loo_rms': r, 'n_used': n} for q, r, n in rows
                if abs(q * 20 - round(q * 20)) < 1e-6]

    out['collapse'] = {
        'definition': ('every barrier cell from the three planes AND the D '
                       'curve (ratio < 1, the sealed reference excluded) is '
                       'placed at x = log10(w^q / D_barrier) with D_barrier = '
                       'D0*ratio, and the residual about a leave-one-out binned '
                       'master curve is measured. q = 1 is a resistive membrane '
                       '(the invariant is w/D_barrier); q = 2 is barrier-storage '
                       'control (the invariant is w^2/D_barrier). The comparison '
                       'coordinates ratio-alone and w-alone are the null '
                       'hypotheses that one parameter suffices.'),
        'n_cells': int(Y.size),
        'n_informative_cells': int(info.sum()),
        'informative_rule': ('|M - M_sealed| and |M - M_uniform| both greater '
                             'than 10% of that background D0 own '
                             'uniform-to-sealed span'),
        'misfit_range_psi': [float(Y.min()), float(Y.max())],
        'reference_limits_by_D0': {f"{k:g}": {'uniform_psi': v[0],
                                              'sealed_psi': v[1]}
                                   for k, v in sorted(ref_by_D.items())},
        'raw_all_cells': {
            'q_scan': _rows(scan),
            'best_q': None if best is None else best[0],
            'best_loo_rms_psi': None if best is None else best[1],
            'q1': collapse_scatter(np.log10(W / Db), Y, 0.10),
            'q2': collapse_scatter(np.log10(W ** 2 / Db), Y, 0.10),
            'ratio_only': collapse_scatter(np.log10(R), Y, 0.10),
            'w_only': collapse_scatter(np.log10(W), Y, 0.10),
            'Dbarrier_only': collapse_scatter(np.log10(Db), Y, 0.10)},
        'raw_informative_only': {
            'q_scan': _rows(scan_i),
            'best_q': None if best_i is None else best_i[0],
            'best_loo_rms_psi': None if best_i is None else best_i[1],
            'q1': collapse_scatter(np.log10(W[info] / Db[info]), Y[info], 0.10),
            'q2': collapse_scatter(np.log10(W[info] ** 2 / Db[info]), Y[info],
                                   0.10),
            'ratio_only': collapse_scatter(np.log10(R[info]), Y[info], 0.10),
            'w_only': collapse_scatter(np.log10(W[info]), Y[info], 0.10),
            'Dbarrier_only': collapse_scatter(np.log10(Db[info]), Y[info], 0.10)},
        'normalised_misfit': {
            'definition': ('nu = (M - M_sealed(D0)) / (M_uniform(D0) - '
                           'M_sealed(D0)); dimensionless, so the D0-dependent '
                           'asymptotes are removed and only the barrier '
                           'transmission is left'),
            'q_scan': _rows(scan_n),
            'best_q': None if best_n is None else best_n[0],
            'best_loo_rms': None if best_n is None else best_n[1],
            'q1': collapse_scatter(np.log10(W / Db), Ynorm, 0.10),
            'q2': collapse_scatter(np.log10(W ** 2 / Db), Ynorm, 0.10),
            'ratio_only': collapse_scatter(np.log10(R), Ynorm, 0.10),
            'w_only': collapse_scatter(np.log10(W), Ynorm, 0.10),
            'nu_range': [float(Ynorm.min()), float(Ynorm.max())]},
        'bin_width_sensitivity': {
            f"{bw:g}": collapse_scatter(np.log10(W / Db), Y, bw)
            for bw in (0.05, 0.10, 0.20, 0.40)},
    }

    # ---- the D curve ------------------------------------------------------
    dc = grid['dcurve']
    dcD = [float(v) for v in dc['D_ft2_s']]
    dcR = [float(v) for v in dc['ratios'] if float(v) < 1.0
           and abs(float(v) - float(grid['sealed_reference_ratio'])) > 1e-30]
    curve = []
    for D0 in dcD:
        row = []
        for r in dcR:
            c = get(D0, float(dc['w_ft']), r, kinds=('dcurve',))
            row.append(None if c is None
                       else c['misfit']['shielded_mean_rmse_delta_psi'])
        cu = get(D0, float(dc['w_ft']), 1.0, kinds=('dcurve',))
        cs = get(D0, float(dc['w_ft']),
                 float(grid['sealed_reference_ratio']), kinds=('dcurve',))
        y = np.array([v for v in row if v is not None], float)
        j = int(np.argmin(y))
        edge = (j == 0 or j == y.size - 1)
        lrr = np.log10(np.asarray(dcR))
        if edge:
            rv, mv = float(lrr[j]), float(y[j])
        else:
            rv, mv = _parabola_min(lrr[j - 1:j + 2], y[j - 1:j + 2])
            if rv is None:
                rv, mv = float(lrr[j]), float(y[j])
        curve.append({
            'D0_ft2_s': D0,
            'misfit_by_ratio_psi': row,
            'uniform_misfit_psi': cu['misfit']['shielded_mean_rmse_delta_psi'],
            'sealed_misfit_psi': cs['misfit']['shielded_mean_rmse_delta_psi'],
            'profiled_min_psi': mv, 'ratio_opt': float(10.0 ** rv),
            'D_barrier_opt_ft2_s': float(D0 * 10.0 ** rv),
            'argmin_on_grid_edge': bool(edge),
            'band_10pct_in_log10_ratio': profile_band(lrr, y, 1.1 * float(y[j])),
        })
    prof_D = np.array([c['profiled_min_psi'] for c in curve])
    kD = int(np.argmin(prof_D))
    out['d_curve'] = {
        'w_ft': float(dc['w_ft']), 'ratios': dcR, 'rows': curve,
        'profiled_over_ratio_min_psi': [float(v) for v in prof_D],
        'band_10pct_in_log10_D': profile_band(np.log10(dcD), prof_D,
                                              1.1 * float(prof_D.min())),
        'argmin_D_ft2_s': dcD[kD],
        'argmin_on_grid_edge': bool(kD in (0, len(dcD) - 1)),
        'fit_log10_Dbarrier_opt_vs_log10_D0': linfit(
            np.log10([c['D0_ft2_s'] for c in curve if not c['argmin_on_grid_edge']]),
            np.log10([c['D_barrier_opt_ft2_s'] for c in curve
                      if not c['argmin_on_grid_edge']]))
        if sum(1 for c in curve if not c['argmin_on_grid_edge']) > 2 else None,
        'fit_log10_ratio_opt_vs_log10_D0': linfit(
            np.log10([c['D0_ft2_s'] for c in curve if not c['argmin_on_grid_edge']]),
            np.log10([c['ratio_opt'] for c in curve
                      if not c['argmin_on_grid_edge']]))
        if sum(1 for c in curve if not c['argmin_on_grid_edge']) > 2 else None,
    }

    # ---- plane-to-plane: is a plane at another D the same plane shifted? --
    shift = {}
    for D0 in planes[1:]:
        A0, A1 = M[planes[0]], M[D0]
        rows = []
        for i, w in enumerate(ws):
            # interpolate plane D0 onto the D_barrier of plane planes[0]
            x0 = np.log10(planes[0] * np.asarray(ratios))
            x1 = np.log10(D0 * np.asarray(ratios))
            lo, hi = max(x0.min(), x1.min()), min(x0.max(), x1.max())
            m = (x0 >= lo) & (x0 <= hi)
            if m.sum() < 3:
                continue
            y1 = np.interp(x0[m], x1[::-1], A1[i][::-1])
            rows.append({'w_ft': w,
                         'n': int(m.sum()),
                         'rms_difference_psi': float(np.sqrt(np.mean(
                             (A0[i][m] - y1) ** 2))),
                         'max_difference_psi': float(np.max(np.abs(
                             A0[i][m] - y1)))})
        shift[f"{planes[0]:g}_vs_{D0:g}"] = {
            'note': ('plane at D0 re-indexed by D_barrier = D0*ratio and '
                     'compared with the reference plane at the same '
                     'D_barrier and the same w. If the data constrained '
                     'D_barrier rather than ratio these would agree.'),
            'rows': rows,
            'worst_rms_psi': float(max(r['rms_difference_psi'] for r in rows)),
        }
    out['plane_vs_plane_at_equal_Dbarrier'] = shift

    # ---- the padding check ------------------------------------------------
    # B2 settled 5000 ft of padding at both ends for this geometry, but at the
    # archive's own D. Over the whole chain (36898 s) sqrt(4*D*t) is 4550 ft at
    # D = 140 and 26060 ft at D = 4600, so the high-D rows of the D curve may be
    # boundary-contaminated. A separate manifested run repeats a few cells at a
    # 20000 ft pad; the difference is the contamination.
    if padcheck:
        paths = [q for q in str(padcheck).split(',') if q]
        pcells, psrc = [], []
        for q in paths:
            with open(q) as fh:
                pdoc = json.load(fh)
            pcells.extend(pdoc['cells'])
            psrc.append({'path': rm._rel(q), 'sha256': rm.sha256_file(q),
                         'n_cells': len(pdoc['cells']), 'tag': pdoc.get('tag')})
        rows = []
        for c in pcells:
            base = None
            for b in cells:
                if (abs(b['D0'] - c['D0']) < 1e-9
                        and abs(b['w_ft'] - c['w_ft']) < 1e-12
                        and abs(np.log10(b['ratio'])
                                - np.log10(c['ratio'])) < 1e-9):
                    base = b
                    break
            if base is None:
                continue
            m5 = base['misfit']['shielded_mean_rmse_delta_psi']
            m20 = c['misfit']['shielded_mean_rmse_delta_psi']
            rows.append({'D0': c['D0'], 'w_ft': c['w_ft'], 'ratio': c['ratio'],
                         'misfit_pad5000_psi': m5, 'misfit_pad20000_psi': m20,
                         'difference_psi': m20 - m5,
                         'difference_pct': 100.0 * (m20 - m5) / m5,
                         'peak_dp_g5_pad5000': base['peak_dp_psi']['g5'],
                         'peak_dp_g5_pad20000': c['peak_dp_psi']['g5'],
                         'peak_dp_g6_pad5000': base['peak_dp_psi']['g6'],
                         'peak_dp_g6_pad20000': c['peak_dp_psi']['g6']})
        by_D = {}
        for r in rows:
            by_D.setdefault(f"{r['D0']:g}", []).append(abs(r['difference_pct']))
        t_chain = 14495.0 + 9221.0 + 13182.0
        out['padding_check'] = {
            'sources': psrc,
            'pad_ft': 20000.0,
            'baseline_pad_ft': 5000.0,
            'chain_duration_s': t_chain,
            'diffusion_length_ft_by_D0': {
                f"{d:g}": float(np.sqrt(4.0 * d * t_chain))
                for d in sorted({r['D0'] for r in rows})},
            'worst_abs_difference_pct_by_D0': {k: float(max(v))
                                               for k, v in by_D.items()},
            'rows': rows}

    p = os.path.join(outroot, f"c4_analysis_{tag}.json")
    rm.assert_absent([p])
    os.makedirs(outroot, exist_ok=True)
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    print(f"[C4] analysis -> {p}")

    # console summary
    _x = out['e1_cross_check']['max_abs_difference_psi']
    print("  E1 cross-check max|diff| = "
          + ("not applicable (no D = 140, w = 1 ft row in this sweep)"
             if _x is None else f"{_x:.4f} psi"))
    for D0 in planes:
        v = valley[f"{D0:g}"]
        f = v['fit_log10_Dbarrier_vs_log10_w']
        fit_s = ('slope p = n/a' if f is None else
                 f"slope p = {f['slope']:.3f} +- {f['slope_stderr']:.3f} "
                 f"(R2 {f['r2']:.4f})")
        fv = v['floor_variation_psi']
        fs = ('n/a' if fv is None else
              f"{fv:.2f} psi ({v['floor_variation_pct_of_min']:.2f}%)")
        print(f"  D0={D0:g}: min {v['global_min_psi']:.2f} psi at w="
              f"{v['global_min_at']['w_ft']:g} ratio="
              f"{v['global_min_at']['ratio']:.3g}; floor varies {fs}; {fit_s}")
    for key in ('raw_all_cells', 'raw_informative_only'):
        cc = out['collapse'][key]
        if cc['best_q'] is None:
            continue
        print(f"  collapse[{key}]: best q = {cc['best_q']:.2f} (LOO rms "
              f"{cc['best_loo_rms_psi']:.2f} psi); q=1 "
              f"{cc['q1']['loo_residual_rms_psi']:.2f}, q=2 "
              f"{cc['q2']['loo_residual_rms_psi']:.2f}, ratio-only "
              f"{cc['ratio_only']['loo_residual_rms_psi']:.2f}, w-only "
              f"{cc['w_only']['loo_residual_rms_psi']:.2f}")
    return 0


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# figures (solve-free)
# ---------------------------------------------------------------------------
# Encoding choices, stated so they can be checked rather than argued about:
#   * the misfit is a MAGNITUDE, so the plane uses a single perceptually uniform
#     sequential ramp (viridis), never a rainbow, with explicit levels;
#   * w and D0 are ORDERED quantities, so the 1-D families are coloured by an
#     ordered ramp rather than by categorical hues, and every family also carries
#     a direct label at its own curve;
#   * no dual axes anywhere; grid and spines are recessive; text is ink-coloured
#     and never takes a series colour.

INK = '#1a1a1a'
INK2 = '#555555'
GRIDC = '#d9d9d9'


def _style(ax):
    ax.tick_params(colors=INK2, labelsize=8, width=0.8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(INK2)
        ax.spines[sp].set_linewidth(0.8)
    ax.grid(True, color=GRIDC, lw=0.5, alpha=0.8)
    ax.set_axisbelow(True)


def run_figures(cfg_path, outroot, sweep_dir, tag, sweep_tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root = load_config(cfg_path)
    grid = root['grid']
    dpi = int(root['base']['outputs']['figure_dpi'])
    an_path = os.path.join(outroot, f"c4_analysis_{tag}.json")
    met_path = os.path.join(sweep_dir, f"c4_cells_{sweep_tag}.json")
    with open(an_path) as fh:
        an = json.load(fh)
    with open(met_path) as fh:
        doc = json.load(fh)
    cells = doc['cells']
    ws = [float(v) for v in an['grid']['w_ft']]
    ratios = [float(v) for v in an['grid']['ratios']]
    planes = [float(v) for v in an['grid']['planes_D']]
    lw_ = np.log10(np.asarray(ws))
    lr_ = np.log10(np.asarray(ratios))
    figdir = os.path.join(outroot, 'figs')
    os.makedirs(figdir, exist_ok=True)
    written = []

    def Mplane(D0):
        A = np.full((len(ws), len(ratios)), np.nan)
        for c in cells:
            if c['kind'] != 'plane' or abs(c['D0'] - D0) > 1e-9:
                continue
            i = int(np.argmin(np.abs(np.asarray(ws) - c['w_ft'])))
            j = int(np.argmin(np.abs(lr_ - np.log10(c['ratio']))))
            A[i, j] = c['misfit']['shielded_mean_rmse_delta_psi']
        return A

    # ---- figure 1: the plane at three background D ------------------------
    Ms = {D0: Mplane(D0) for D0 in planes}
    vmin = min(float(np.nanmin(A)) for A in Ms.values())
    vmax = max(float(np.nanmax(A)) for A in Ms.values())
    levels = np.linspace(vmin, vmax, 41)
    fig, axes = plt.subplots(1, len(planes), figsize=(4.1 * len(planes), 4.3),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for k, D0 in enumerate(planes):
        ax = axes[k]
        A = Ms[D0]
        cf = ax.contourf(lw_, lr_, A.T, levels=levels, cmap='viridis_r',
                         extend='neither')
        v = an['valley'][f"{D0:g}"]
        lvl = 1.1 * v['global_min_psi']
        ax.contour(lw_, lr_, A.T, levels=[lvl], colors='#ffffff',
                   linewidths=1.6)
        pw_ = v['per_w']
        ax.plot([np.log10(p['w_ft']) for p in pw_],
                [p['log10_ratio_opt'] for p in pw_], 'o', ms=4.5,
                mfc='#ffffff', mec=INK, mew=0.9, zorder=5)
        f = v['fit_log10_Dbarrier_vs_log10_w']
        if f is not None:
            xx = np.array([lw_[0] - 0.1, lw_[-1] + 0.1])
            yy = f['slope'] * xx + f['intercept'] - np.log10(D0)
            ax.plot(xx, yy, '-', color='#ffffff', lw=1.2, zorder=4)
            ax.text(0.03, 0.05,
                    f"valley: ratio $\\propto w^{{{f['slope']:.2f}}}$",
                    transform=ax.transAxes, fontsize=8, color='#ffffff')
        # the region the signal cannot tell from a perfect seal
        reg = an['regimes'][f"{D0:g}"]
        S = np.full((len(ws), len(ratios)), np.nan)
        for q in reg['cells']:
            i = int(np.argmin(np.abs(np.asarray(ws) - q['w_ft'])))
            j = int(np.argmin(np.abs(lr_ - np.log10(q['ratio']))))
            S[i, j] = q['sep_from_sealed_pt']
        ax.contour(lw_, lr_, S.T, levels=[1.0], colors='#d43d51',
                   linewidths=1.3, linestyles='--')
        ax.plot([0.0], [-5.0], marker='*', ms=13, mfc='#f2c14e', mec=INK,
                mew=0.8, zorder=6, ls='none')
        ax.set_xlim(lw_[0], lw_[-1])
        ax.set_ylim(lr_.min(), lr_.max())
        ax.set_xticks(lw_)
        ax.set_xticklabels([f"{w:g}" for w in ws], fontsize=8)
        ax.set_xlabel('barrier half-width $w$  (ft)', fontsize=9, color=INK)
        if k == 0:
            ax.set_ylabel('reduction ratio  $D_{barrier}/D_0$', fontsize=9,
                          color=INK)
            tks = np.arange(np.ceil(lr_.min()), np.floor(lr_.max()) + 0.1, 1.0)
            ax.set_yticks(tks)
            ax.set_yticklabels([f"$10^{{{int(t)}}}$" for t in tks], fontsize=8)
        ax.set_title(f"$D_0$ = {D0:g} ft$^2$/s", fontsize=9.5, color=INK)
        _style(ax)
        ax.grid(False)
    cb = fig.colorbar(cf, ax=list(axes), fraction=0.028, pad=0.02)
    cb.set_label('gauge-mean RMSE at gauges 5, 6  (psi)', fontsize=8.5,
                 color=INK)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    handles = [Line2D([], [], color='#ffffff', lw=1.6, label='+10% of the plane minimum'),
               Line2D([], [], marker='o', ls='none', mfc='#ffffff', mec=INK,
                      ms=5, label='best ratio at that $w$'),
               Line2D([], [], color='#d43d51', lw=1.3, ls='--',
                      label='below this the trace is within 1 typographic point\nof a perfect seal (saturated, not identified)'),
               Line2D([], [], marker='*', ls='none', mfc='#f2c14e', mec=INK,
                      ms=11, label="manuscript's stated $w$ = 1 ft, ratio = $10^{-5}$")]
    fig.legend(handles=handles, loc='lower center', fontsize=7.4,
               frameon=False, labelcolor=INK, ncol=2, handlelength=2.0,
               bbox_to_anchor=(0.45, -0.16))
    fig.suptitle('The misfit valley is a BAND along constant $w/D_{barrier}$, '
                 'not a point', fontsize=11, color=INK, y=0.99)
    p1 = os.path.join(figdir, f"fig_c4_plane_{tag}.png")
    rm.assert_absent([p1])
    fig.savefig(p1, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    written.append((p1, 'the (w, ratio) misfit plane at three background D'))

    # ---- figure 2: the collapse -------------------------------------------
    r_seal_f = float(grid['sealed_reference_ratio'])
    pts = [(c['w_ft'], c['ratio'], c['D0'],
            c['misfit']['shielded_mean_rmse_delta_psi'])
           for c in cells if c['kind'] in ('plane', 'dcurve')
           and c['ratio'] < 1.0
           and abs(np.log10(c['ratio']) - np.log10(r_seal_f)) > 1e-9]
    W = np.array([q[0] for q in pts])
    Rr = np.array([q[1] for q in pts])
    Dd = np.array([q[2] for q in pts])
    Yy = np.array([q[3] for q in pts])
    Db = Dd * Rr
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(np.log10(W).min(), np.log10(W).max())
    lim = an['collapse']['reference_limits_by_D0']
    Mu_ = np.array([lim[f"{q[2]:g}"]['uniform_psi'] for q in pts])
    Ms_ = np.array([lim[f"{q[2]:g}"]['sealed_psi'] for q in pts])
    Yn = (Yy - Ms_) / (Mu_ - Ms_)
    cn = an['collapse']['normalised_misfit']
    fig, axs = plt.subplots(1, 4, figsize=(15.6, 4.0))
    panels = [('reduction ratio alone', np.log10(Rr), 'ratio',
               an['collapse']['raw_all_cells']['ratio_only'], Yy, 'psi'),
              ('barrier diffusivity alone  $D_0\\cdot$ratio', np.log10(Db),
               '$D_{barrier}$ (ft$^2$/s)',
               an['collapse']['raw_all_cells']['Dbarrier_only'], Yy, 'psi'),
              ('the combination  $w/D_{barrier}$', np.log10(W / Db),
               '$w/D_{barrier}$ (s/ft)',
               an['collapse']['raw_all_cells']['q1'], Yy, 'psi'),
              ('the combination, normalised misfit',
               np.log10(W / Db), '$w/D_{barrier}$ (s/ft)', cn['q1'], Yn, '')]
    for ax, (ttl, xv, xlab, sc, yv, unit) in zip(axs, panels):
        ax.scatter(xv, yv, s=16, c=np.log10(W), cmap=cmap, norm=norm,
                   edgecolors='white', linewidths=0.35, zorder=3)
        ax.set_title(ttl, fontsize=9.5, color=INK)
        ax.set_xlabel(f"$\\log_{{10}}$  {xlab}", fontsize=9, color=INK)
        if sc:
            v = sc['loo_residual_rms_psi']
            ax.text(0.03, 0.94,
                    'scatter about the master curve\n'
                    + (f"{v:.1f} psi rms" if unit else f"{v:.3f} rms"),
                    transform=ax.transAxes, fontsize=8, color=INK, va='top')
        _style(ax)
    axs[0].set_ylabel('gauge-mean RMSE at gauges 5, 6  (psi)', fontsize=9,
                      color=INK)
    axs[3].set_ylabel(r'$\nu$ = (M $-$ M$_{sealed}$)/(M$_{uniform}$ $-$ M$_{sealed}$)',
                      fontsize=9, color=INK)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=list(axs), fraction=0.02, pad=0.015)
    cb.set_label('$\\log_{10}$  $w$ (ft)', fontsize=8.5, color=INK)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    fig.suptitle(f"Every barrier cell of every plane and of the $D$ curve "
                 f"({len(Yy)} forward solves): only the combination collapses "
                 'them onto one curve', fontsize=10.5, color=INK, y=1.06)
    p2 = os.path.join(figdir, f"fig_c4_collapse_{tag}.png")
    rm.assert_absent([p2])
    fig.savefig(p2, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    written.append((p2, 'the (w, ratio, D0) -> w/D_barrier collapse'))

    # ---- figure 3: the one-dimensional curves -----------------------------
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.0))
    ax = axs[0]
    D0 = 140.0
    A = Ms[D0]
    for i, w in enumerate(ws):
        c = cmap(norm(np.log10(w)))
        ax.plot(lr_, A[i], '-', color=c, lw=1.6, zorder=3)
        j = int(np.argmin(A[i]))
        ax.plot([lr_[j]], [A[i][j]], 'o', ms=4, mfc='white', mec=c, mew=1.2,
                zorder=4)
        if w in (ws[0], 1.0, ws[-1]):
            ax.annotate(f"$w$ = {w:g} ft", (lr_[0], A[i][0]),
                        textcoords='offset points', xytext=(-4, 2),
                        ha='right', fontsize=7.5, color=INK)
    ref = an['references'][f"{D0:g}"]
    ax.axhline(ref['uniform_misfit_psi'], color=INK2, lw=1.0, ls=':', zorder=2)
    ax.axhline(ref['sealed_misfit_psi'], color='#d43d51', lw=1.0, ls='--',
               zorder=2)
    ax.text(lr_[-1], ref['uniform_misfit_psi'], ' no barrier', fontsize=7.5,
            color=INK2, va='bottom', ha='right')
    ax.text(lr_[-1], ref['sealed_misfit_psi'], ' perfect seal', fontsize=7.5,
            color='#d43d51', va='bottom', ha='right')
    ax.set_xlabel('$\\log_{10}$  reduction ratio', fontsize=9, color=INK)
    ax.set_ylabel('gauge-mean RMSE at gauges 5, 6  (psi)', fontsize=9, color=INK)
    ax.set_title(f"misfit vs ratio, one curve per $w$  ($D_0$ = {D0:g})",
                 fontsize=9.5, color=INK)
    _style(ax)

    ax = axs[1]
    dc = an['d_curve']
    dcD = [r['D0_ft2_s'] for r in dc['rows']]
    ax.plot(np.log10(dcD), dc['profiled_over_ratio_min_psi'], '-o', ms=4,
            color='#3b7ea1', lw=1.8, zorder=4,
            label='ratio re-optimised at each $D_0$')
    for r_show, sty in ((1e-4, '--'), (1e-5, '-.')):
        jr = int(np.argmin(np.abs(np.log10(np.asarray(dc['ratios']))
                                  - np.log10(r_show))))
        yv = [r['misfit_by_ratio_psi'][jr] for r in dc['rows']]
        ax.plot(np.log10(dcD), yv, sty, lw=1.3, color=INK2,
                label=f"ratio fixed at $10^{{{int(round(np.log10(r_show)))}}}$")
        ax.annotate(f"ratio $10^{{{int(round(np.log10(r_show)))}}}$",
                    (np.log10(dcD[-1]), yv[-1]), textcoords='offset points',
                    xytext=(-4, 4), ha='right', fontsize=7.5, color=INK2)
    ax.plot(np.log10(dcD), [r['uniform_misfit_psi'] for r in dc['rows']], ':',
            color=INK2, lw=1.0, label='no barrier')
    ax.set_xlabel('$\\log_{10}$  background $D_0$  (ft$^2$/s)', fontsize=9,
                  color=INK)
    ax.set_ylabel('gauge-mean RMSE at gauges 5, 6  (psi)', fontsize=9, color=INK)
    ax.set_title(f"misfit vs $D_0$  ($w$ = {dc['w_ft']:g} ft)", fontsize=9.5,
                 color=INK)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK, loc='upper left')
    _style(ax)

    ax = axs[2]
    for k, D0 in enumerate(planes):
        v = an['valley'][f"{D0:g}"]
        c = plt.get_cmap('cividis')(k / max(len(planes) - 1, 1) * 0.8)
        ax.plot([np.log10(p['w_ft']) for p in v['per_w']],
                [np.log10(p['D_barrier_opt_ft2_s']) for p in v['per_w']],
                '-o', ms=4, lw=1.6, color=c, label=f"$D_0$ = {D0:g}")
    xx = np.array([lw_[0], lw_[-1]])
    v0 = an['valley'][f"{planes[0]:g}"]
    b = np.log10(v0['per_w'][0]['D_barrier_opt_ft2_s']) - 1.0 * lw_[0]
    ax.plot(xx, 1.0 * xx + b, '-', color=INK2, lw=1.0)
    ax.text(xx[-1], 1.0 * xx[-1] + b, ' slope 1\n (membrane)', fontsize=7.5,
            color=INK2, va='center')
    b2 = np.log10(v0['per_w'][0]['D_barrier_opt_ft2_s']) - 2.0 * lw_[0]
    ax.plot(xx, 2.0 * xx + b2, '--', color=INK2, lw=1.0)
    ax.text(xx[-1], 2.0 * xx[-1] + b2, ' slope 2\n (storage)', fontsize=7.5,
            color=INK2, va='center')
    ax.set_xlabel('$\\log_{10}$  $w$  (ft)', fontsize=9, color=INK)
    ax.set_ylabel('$\\log_{10}$  best-fitting $D_{barrier}$  (ft$^2$/s)',
                  fontsize=9, color=INK)
    ax.set_title('the valley floor, three background $D$', fontsize=9.5,
                 color=INK)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK, loc='upper left')
    _style(ax)
    p3 = os.path.join(figdir, f"fig_c4_curves_{tag}.png")
    rm.assert_absent([p3])
    fig.savefig(p3, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    written.append((p3, '1-D misfit curves in ratio and in D, and the valley '
                        'floor locus'))

    prov = {'kind': 'figures_no_solve', 'tag': tag, 'generated_utc': utcnow(),
            'dpi': dpi,
            'code_sha256': rm.sha256_file(os.path.abspath(__file__)),
            'sources': {'analysis_json': rm._rel(an_path),
                        'analysis_sha256': rm.sha256_file(an_path),
                        'cells_json': rm._rel(met_path),
                        'cells_sha256': rm.sha256_file(met_path),
                        'sweep_manifest': rm._rel(
                            os.path.join(sweep_dir, 'manifest.json')),
                        'sweep_manifest_sha256': rm.sha256_file(
                            os.path.join(sweep_dir, 'manifest.json'))},
            'figures': [{'path': rm._rel(q[0]), 'sha256': rm.sha256_file(q[0]),
                         'note': q[1]} for q in written]}
    pp = os.path.join(figdir, f"figures_provenance_{tag}.json")
    rm.assert_absent([pp])
    with open(pp, 'w') as fh:
        json.dump(prov, fh, indent=1, sort_keys=True)
    for q in written:
        print(f"[C4] figure -> {q[0]}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/rev2/c4_identifiability.json')
    ap.add_argument('--mode', default='sweep',
                    choices=('sweep', 'analyse', 'figures'))
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--tag', default='v1')
    ap.add_argument('--sweep-dir', default=None)
    ap.add_argument('--sweep-tag', default='v1')
    ap.add_argument('--padcheck', default=None,
                    help=('comma-separated pad-check cells JSON paths '
                          '(analyse mode)'))
    ap.add_argument('--workers', type=int, default=MAX_WORKERS)
    a = ap.parse_args(argv)

    root = load_config(a.config)
    outroot = a.outdir or os.path.join(root['output_root'], 'sweep')
    if a.mode == 'sweep':
        return run_sweep(a.config, outroot, a.tag, a.workers)
    if a.mode == 'analyse':
        sweep_dir = a.sweep_dir or os.path.join(root['output_root'], 'sweep')
        return run_analyse(a.config, a.outdir or root['output_root'],
                           sweep_dir, a.tag, a.sweep_tag, padcheck=a.padcheck)
    sweep_dir = a.sweep_dir or os.path.join(root['output_root'], 'sweep')
    return run_figures(a.config, a.outdir or root['output_root'],
                       sweep_dir, a.tag, a.sweep_tag)


if __name__ == '__main__':
    raise SystemExit(main())
