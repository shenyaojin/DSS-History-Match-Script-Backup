#!/usr/bin/env python3
"""C4 AMEND 1 -- the ratio-grid convergence of the per-w optimum.

WHY THIS FILE EXISTS
--------------------
`c4_stage2.py --mode analyse` locates each per-`w` optimum by fitting a parabola
through the three grid points around the discrete argmin of a **0.25-decade**
ratio ladder, and quotes

  * the vertex ABSCISSA as `ratio_opt`, hence `D_barrier*` and the series
    resistance `W_tot/D_barrier*` (645 / 682 / 679 s/ft), and
  * the vertex ORDINATE as `misfit_at_opt_psi`, hence `floor_variation_psi`
    (0.69-0.82 psi).

Neither was checked for convergence in the ratio step `h`.  The stage-2 report
itself measured that going from `h` = 0.5 to `h` = 0.25 decade moves the vertex
by 0.080-0.131 decades (`ratio_grid_resolution_bias`) and concluded "quote the
planes for the VALUE" -- i.e. it assumed, without testing, that `h` = 0.25 had
converged.

This script tests it, on the SAME solver, the SAME chain and the SAME misfit,
by re-solving a **0.05-decade** ladder of 17 cells centred on each published
vertex, and then a **0.0125-decade** ladder of 5 cells centred on the 0.05-decade
vertex, so the vertex is measured on a three-point `h` sequence 0.25 / 0.05 /
0.0125 and the observed order of convergence is reported rather than assumed.

Because every ladder is centred on the published vertex, the `k` = 0 cell of the
fine ladder is the SIMULATED misfit AT the published vertex ratio.  That gives
the parabola-interpolation error of `misfit_at_opt_psi` for free, which is the
second thing this script measures.

MODES
-----
  finegrid --spec {plane140,plane550,plane1150,dcurve}   solver run, manifested
  analyse                                                solve-free, writes
                                                         c4_stage2_amend1_<tag>.json
  figures                                                solve-free, writes v5

NOT EDITED, ON PURPOSE
----------------------
`c4_identifiability.py` is hashed as `code.main_script` in five manifests on
disk, so it is IMPORTED here (its `_cell`, `chain_prelude`, `make_source`,
`load_field`, `misfit` are reused verbatim) and never modified.  `rev2_core`,
`rev2_data`, `rev2_manifest` and `a5_two_stage_chain` are imported and never
modified.  `c4_stage2.py` is NOT hashed in any manifest (checked) and its
figure code IS edited by this amendment -- see amend defect 3.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import multiprocessing as mp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rev2_core as rc          # noqa: E402
import rev2_data as rd          # noqa: E402
import rev2_manifest as rm      # noqa: E402
import a5_two_stage_chain as a5  # noqa: E402
import c4_identifiability as c4  # noqa: E402  (the stage-1 solver, unmodified)

STUDY_ID = 'C4_w_ratio_identifiability_amend1_ratio_grid_convergence'
TASK_ID = 'C4'
ROUND = 'rev2_20260901'
OUTROOT = os.path.join(ROOT, 'output', ROUND, 'C4')
STAGE2_ANALYSIS = os.path.join(OUTROOT, 'c4_stage2_analysis_v1.json')
MISFIT_KEY = 'shielded_mean_rmse_delta_psi'
N_BARRIERS = 6
PHASES = ('phase1', 'phase2', 'phase3')

FINE_STEP = 0.05        # decades, the primary refinement
FINE_HALF = 8           # +- 8 steps  -> +- 0.40 decades, 17 cells
ULTRA_STEP = 0.0125     # decades, the h-convergence check
ULTRA_HALF = 2          # +- 2 steps  -> +- 0.025 decades, 5 cells

SPECS = {
    'plane140': {
        'config': 'configs/rev2/c4_identifiability.json',
        'plane_key': 'pad5000:140', 'D0': [140.0], 'pad_ft': 5000.0,
        'w_from': 'plane', 'ultra_w': [1.0],
        'outdir': 'finegrid_pad5000_140', 'tag': 'fine140_v1'},
    'plane550': {
        'config': 'output/rev2_20260901/C4/c4_pad20000.json',
        'plane_key': 'pad20000:550', 'D0': [550.0], 'pad_ft': 20000.0,
        'w_from': 'plane', 'ultra_w': [1.0],
        'outdir': 'finegrid_pad20000_550', 'tag': 'fine550_v1'},
    'plane1150': {
        'config': 'output/rev2_20260901/C4/c4_pad20000.json',
        'plane_key': 'pad20000:1150', 'D0': [1150.0], 'pad_ft': 20000.0,
        'w_from': 'plane', 'ultra_w': [1.0],
        'outdir': 'finegrid_pad20000_1150', 'tag': 'fine1150_v1'},
    'dcurve': {
        'config': 'output/rev2_20260901/C4/c4_dcurve_pad20000.json',
        'plane_key': None, 'D0': None, 'pad_ft': 20000.0,
        'w_from': 'dcurve', 'ultra_w': [],
        'outdir': 'finegrid_dcurve_pad20000', 'tag': 'finedcurve_v1'},
}


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def rp(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


# ---------------------------------------------------------------------------
# centres: where the published (coarse-ladder) vertices are
# ---------------------------------------------------------------------------

def centres_for(spec):
    """[(D0, w, log10_ratio_centre, provenance), ...] from the stage-2 analysis."""
    A = json.load(open(STAGE2_ANALYSIS))
    out = []
    if spec['w_from'] == 'plane':
        pl = A['planes'][spec['plane_key']]
        D0 = float(pl['D0_ft2_s'])
        for p in pl['per_w']:
            out.append((D0, float(p['w_ft']), float(p['log10_ratio_opt']),
                        {'source': f"planes.{spec['plane_key']}.per_w",
                         'coarse_step_decades': 0.25,
                         'coarse_log10_ratio_opt': float(p['log10_ratio_opt']),
                         'coarse_vertex_misfit_psi': float(p['misfit_at_opt_psi']),
                         'coarse_grid_min_psi': float(p['grid_min_psi']),
                         'coarse_resistance_s_per_ft':
                             float(p['resistance_s_per_ft'])}))
    else:
        for row in A['d_curve_pad20000']['rows']:
            out.append((float(row['D0_ft2_s']), float(row['w_ft']),
                        float(np.log10(float(row['ratio_opt']))),
                        {'source': 'd_curve_pad20000.rows',
                         'coarse_step_decades': 0.5,
                         'coarse_log10_ratio_opt':
                             float(np.log10(float(row['ratio_opt']))),
                         'coarse_vertex_misfit_psi':
                             float(row['profiled_min_psi']),
                         'coarse_grid_min_psi':
                             float(min(row['misfit_by_ratio_psi'])),
                         'coarse_resistance_s_per_ft':
                             float(row['resistance_s_per_ft'])}))
    return out


def parab_vertex(x, y):
    """Vertex of the parabola through three points (x ascending)."""
    a, b, c = np.polyfit(np.asarray(x, float), np.asarray(y, float), 2)
    if a <= 0:
        return None, None
    xv = -b / (2.0 * a)
    return float(xv), float(a * xv * xv + b * xv + c)


# ---------------------------------------------------------------------------
# the solver run
# ---------------------------------------------------------------------------

def run_finegrid(spec_name, workers, smoke_dir=None):
    """smoke_dir: write to that directory with a 3-cell ladder and 2 centres.

    The smoke path exists only to exercise the manifest block cheaply before a
    ~10 minute production run; it writes nothing into the task output tree.
    """
    t_all = time.time()
    spec = SPECS[spec_name]
    cfg_path = rp(spec['config'])
    outdir = smoke_dir or os.path.join(OUTROOT, spec['outdir'])
    tag = spec['tag'] + ('_smoke' if smoke_dir else '')
    os.makedirs(outdir, exist_ok=True)

    root = c4.load_config(cfg_path)
    cfg = root['base']

    md_table = rd.load_gauge_md_table()
    x, mesh_rec = a5.build_chain_mesh(cfg['mesh'])
    wins, _ = a5.phase_windows(cfg['phase_boundaries'])
    gauges = [int(g) for g in cfg['targets']['gauges']]
    shielded = [int(g) for g in cfg['targets']['shielded_gauges']]
    gidx = [int(np.argmin(np.abs(x - md_table.md_of(g)))) for g in gauges]
    hits7 = rd.load_frac_hits(7, unique=False, sort=False)
    srcs = {p: c4.make_source(cfg, x, wins, p) for p in PHASES}
    s3 = srcs['phase3']
    field = c4.load_field(gauges, wins)

    cen = centres_for(spec)
    if smoke_dir:
        cen = cen[:2]
    all_D = sorted({d for d, _, _, _ in cen})
    print(f"[C4a1] spec={spec_name} nx={len(x)} pad={spec['pad_ft']:g} ft "
          f"backgrounds={all_D}", flush=True)

    prel = {}
    for D0 in all_D:
        t = time.time()
        prel[f"{D0:g}"] = c4.chain_prelude(cfg, x, srcs, D0)
        print(f"[C4a1] prelude D={D0:g} done ({time.time() - t:.1f} s)",
              flush=True)

    # the phase-3 time axis: taken from one cheap uniform solve, NOT re-derived.
    # record_idx equivalence was proven in the parent runs (0.000e+00 psi) and
    # the house rules say not to re-verify a proven kernel property.
    common = dict(theta=float(cfg['physics']['theta']),
                  lambda_leak=float(cfg['physics']['lambda_leak']),
                  p0=float(cfg['physics']['p0_psi']),
                  interface_avg=cfg['physics']['interface_avg'],
                  theta_startup_steps=int(cfg['physics']['theta_startup_steps']))
    idx3 = [int(i) for i in s3['source_idx']]
    D_ref = float(all_D[0])
    dpr, _ = rc.build_barrier_profile(
        x, np.full(len(x), D_ref), hits7, 1.0, 1e-4,
        ratio_reference=cfg['barrier']['ratio_reference'],
        combine=cfg['barrier']['combine'], on_empty=cfg['barrier']['on_empty'],
        on_outside=cfg['barrier']['on_outside'], return_report=True)
    ta3_ref, _ = rc.solve_forward_multi(
        x, dpr, float(cfg['time']['dt_fixed_s']), float(s3['t_total_s']),
        [s3['taxis_s']] * len(idx3), [s3['values_psi']] * len(idx3), idx3,
        initial=prel[f"{D_ref:g}"]['init3'], t0=0.0, record_idx=gidx, **common)
    fld = c4.field_on_phase3(field, gauges, s3['t0_abs'], ta3_ref)

    c4._G.update({'x': x, 'hits7': hits7, 's3': s3, 'gidx': gidx,
                  'gauges': gauges, 'shielded': shielded, 'fld': fld,
                  'tcfg': cfg['time'], 'pcfg': cfg['physics'],
                  'bcfg': cfg['barrier'],
                  'dec_s': float(cfg['outputs']['trace_decimation_s']),
                  'init3': {k: v['init3'] for k, v in prel.items()}})

    # ---- pass 1: the 0.05-decade ladder -----------------------------------
    fine_half = 1 if smoke_dir else FINE_HALF
    ultra_half = 1 if smoke_dir else ULTRA_HALF
    jobs, meta = [], []
    for D0, w, c0, prov in cen:
        for k in range(-fine_half, fine_half + 1):
            lr = c0 + FINE_STEP * k
            jobs.append({'D0': D0, 'w': w, 'ratio': float(10.0 ** lr)})
            meta.append({'D0': D0, 'w_ft': w, 'log10_ratio': float(lr),
                         'k': k, 'ladder': 'fine', 'step_decades': FINE_STEP,
                         'centre_log10_ratio': c0})
    n_workers = max(1, min(int(workers), 8))
    ctx = mp.get_context('fork')
    t_pool = time.time()
    res = []
    with ctx.Pool(processes=n_workers) as pool:
        for i, (r, _dec, _full) in enumerate(pool.imap(c4._cell, jobs,
                                                       chunksize=1)):
            res.append(r)
            if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                print(f"[C4a1]   fine {i + 1}/{len(jobs)} "
                      f"({time.time() - t_pool:.0f} s)", flush=True)
    pool1 = time.time() - t_pool

    cells = []
    for m, r in zip(meta, res):
        cells.append({**m, 'ratio': r['ratio'],
                      'misfit_psi': r['misfit'][MISFIT_KEY],
                      'misfit_all': r['misfit'],
                      'taxis_sha256': r['taxis_sha256'],
                      'trace_sha256': r['trace_sha256'],
                      'barrier': r['barrier'], 'wall_s': r['wall_s']})

    # ---- pass 2: the 0.0125-decade ladder around the fine vertex ----------
    fine_vertex = {}
    for D0, w, c0, prov in cen:
        rows = sorted([c for c in cells
                       if c['ladder'] == 'fine'
                       and abs(c['D0'] - D0) < 1e-9 and abs(c['w_ft'] - w) < 1e-9],
                      key=lambda c: c['log10_ratio'])
        lr = np.array([c['log10_ratio'] for c in rows])
        y = np.array([c['misfit_psi'] for c in rows])
        j = int(np.argmin(y))
        if j in (0, len(y) - 1):
            fine_vertex[(D0, w)] = (float(lr[j]), float(y[j]), True)
        else:
            xv, yv = parab_vertex(lr[j - 1:j + 2], y[j - 1:j + 2])
            fine_vertex[(D0, w)] = (xv, yv, False)

    ultra_jobs, ultra_meta = [], []
    for D0, w, c0, prov in cen:
        if w not in spec['ultra_w']:
            continue
        xv = fine_vertex[(D0, w)][0]
        for k in range(-ultra_half, ultra_half + 1):
            lr = xv + ULTRA_STEP * k
            ultra_jobs.append({'D0': D0, 'w': w, 'ratio': float(10.0 ** lr)})
            ultra_meta.append({'D0': D0, 'w_ft': w, 'log10_ratio': float(lr),
                               'k': k, 'ladder': 'ultra',
                               'step_decades': ULTRA_STEP,
                               'centre_log10_ratio': float(xv)})
    pool2 = 0.0
    if ultra_jobs:
        t_pool = time.time()
        with ctx.Pool(processes=n_workers) as pool:
            for m, (r, _d, _f) in zip(ultra_meta,
                                      pool.imap(c4._cell, ultra_jobs,
                                                chunksize=1)):
                cells.append({**m, 'ratio': r['ratio'],
                              'misfit_psi': r['misfit'][MISFIT_KEY],
                              'misfit_all': r['misfit'],
                              'taxis_sha256': r['taxis_sha256'],
                              'trace_sha256': r['trace_sha256'],
                              'barrier': r['barrier'], 'wall_s': r['wall_s']})
        pool2 = time.time() - t_pool
        print(f"[C4a1]   ultra {len(ultra_jobs)} cells ({pool2:.0f} s)",
              flush=True)

    # ---- the same two assertions the parent sweep makes --------------------
    tsha = sorted({c['taxis_sha256'] for c in cells})
    if len(tsha) != 1 or tsha[0] != rm.sha256_array(ta3_ref):
        raise RuntimeError(f"phase-3 time axes differ across cells: {tsha[:3]}")
    masks = {}
    for c in cells:
        if c['barrier'] is None:
            continue
        masks.setdefault(f"{c['w_ft']:g}", set()).add(
            (tuple(c['barrier']['i0']), tuple(c['barrier']['i1'])))
    bad = {k: len(v) for k, v in masks.items() if len(v) != 1}
    if bad:
        raise RuntimeError(f"barrier node mask is not ratio-independent: {bad}")
    wbad = [c for c in cells if c['barrier'] is not None
            and (c['barrier']['n_fallback'] or c['barrier']['n_overlapping_pairs']
                 or abs(c['barrier']['realised_full_width_ft']['min']
                        - 2.0 * c['w_ft']) > 1e-9
                 or abs(c['barrier']['realised_full_width_ft']['max']
                        - 2.0 * c['w_ft']) > 1e-9)]
    if wbad:
        raise RuntimeError(f"{len(wbad)} cells realised the wrong barrier width")

    met_path = os.path.join(outdir, f"c4_finegrid_{tag}.json")
    rm.assert_absent([met_path])
    doc = {'study_id': STUDY_ID, 'task_id': TASK_ID, 'tag': tag,
           'spec': spec_name, 'spec_detail': spec,
           'generated_utc': utcnow(),
           'centres_from': STAGE2_ANALYSIS,
           'centres_sha256': rm.sha256_file(STAGE2_ANALYSIS)
           if hasattr(rm, 'file_sha256') else None,
           'ladder': {'fine_step_decades': FINE_STEP, 'fine_half': FINE_HALF,
                      'ultra_step_decades': ULTRA_STEP,
                      'ultra_half': ULTRA_HALF,
                      'note': ('every ladder is CENTRED on the published '
                               '0.25-decade (planes) / 0.5-decade (D curve) '
                               'parabola vertex, so the k = 0 cell is the '
                               'SIMULATED misfit at that published vertex')},
           'misfit_definition': {
               'primary': ('gauge-mean RMSE over the SHIELDED gauges {5, 6} of '
                           'dP referenced to each series own first in-window '
                           'phase-3 sample -- identical to stage 1 and 2'),
               'aggregation': 'gauge-mean, not sample-pooled (C1)'},
           'centres': [{'D0': d, 'w_ft': w, 'log10_ratio': c0, **prov}
                       for d, w, c0, prov in cen],
           'pool_wall_s_fine': pool1, 'pool_wall_s_ultra': pool2,
           'n_workers': n_workers, 'cells': cells}
    with open(met_path, 'w') as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)

    arr_path = os.path.join(outdir, f"c4_finegrid_{tag}.npz")
    rm.assert_absent([arr_path])
    np.savez_compressed(
        arr_path,
        x_md_ft=x, gauge_numbers=np.asarray(gauges),
        gauge_mesh_idx=np.asarray(gidx), phase3_taxis_s=ta3_ref,
        D0=np.asarray([c['D0'] for c in cells]),
        w_ft=np.asarray([c['w_ft'] for c in cells]),
        log10_ratio=np.asarray([c['log10_ratio'] for c in cells]),
        misfit_psi=np.asarray([c['misfit_psi'] for c in cells]),
        ladder=np.asarray([c['ladder'] for c in cells]))

    # ---- manifest ---------------------------------------------------------
    manifest_path = os.path.join(outdir, 'manifest.json')
    with rm.RunRecorder(manifest_path, study_id=STUDY_ID, task_id=TASK_ID,
                        config=root, config_path=cfg_path, run_label=tag,
                        require_modules=('rev2_core', 'rev2_data',
                                         'rev2_manifest', 'a5_two_stage_chain',
                                         'c4_identifiability')) as R:
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
                     'role': ('observation points for a FORWARD ladder; the '
                              'misfit is evaluated on a grid, nothing is fitted '
                              'by an optimiser')},
            time_level='n' if float(cfg['physics']['theta']) == 1.0 else 'n+1',
            phase_chaining={
                'order': list(PHASES),
                'rule': ('each phase starts from the previous phase FINAL '
                         'spatial profile; t0 = 0 in every phase'),
                'barrier_source_stage': 7, 'barrier_applies_in': 'phase3',
                'phases_1_2_are_ratio_and_w_independent': True,
                'phases_1_2_solved_once_per_background_D': all_D},
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
            label=(f"phase3 (ALL {len(cells)} cells share this axis; asserted "
                   f"identical by sha256 of the realised taxis)")))

        blist = []
        seen_w = set()
        for c in cells:
            if c['barrier'] is None or c['w_ft'] in seen_w:
                continue
            seen_w.add(c['w_ft'])
            for b, (i0, i1) in enumerate(zip(c['barrier']['i0'],
                                             c['barrier']['i1'])):
                mask = np.zeros(len(x), dtype=bool)
                mask[i0:i1 + 1] = True
                blist.append(rm.barrier_record(
                    x, mask,
                    label=(f"stage7_frachit_MD{hits7[b]:.2f}@w={c['w_ft']:g}ft "
                           f"(node mask asserted ratio-independent)"),
                    centre_md_ft=float(hits7[b]),
                    w_requested_ft=float(c['w_ft']),
                    ratio=float(c['ratio']), d_baseline=float(c['D0'])))

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
                'baseline_D_ft2_s': all_D,
                'profile_family': 'uniform_plus_physical_width_barriers',
                'param_names': ['D_baseline', 'ratio', 'w_half_width_ft'],
                'params': {'D_baseline_ft2_s': all_D,
                           'ratio': sorted({float(c['ratio']) for c in cells}),
                           'w_half_width_ft':
                               sorted({float(c['w_ft']) for c in cells})},
                'D_min': float(min(c['D0'] * c['ratio'] for c in cells)),
                'D_max': float(max(c['D0'] for c in cells)),
                'D_sha256': rm.sha256_array(dpr),
                'profile_anchor': 'physical_md',
                'note': ('D_sha256 is the uniform+barrier profile of the single '
                         'time-axis reference solve; every ladder cell is the '
                         'uniform baseline with six stage-7 barriers built by '
                         'rev2_core.build_barrier_profile')},
            barriers=blist if blist else rm.NONE_DECLARED,
            leakage=rm.NONE_DECLARED,
            kernel={'name': 'rev2_core.solve_forward_multi', 'banded': True,
                    'equivalence_reference':
                        'output/rev2_20260901/A4/selftest_output.txt',
                    'note': ('record_idx equivalence to the full-field solve was '
                             'proven at 0.000e+00 psi in the parent C4 sweeps on '
                             'these same meshes and is not re-verified here')},
            rng=rm.NONE_DECLARED,
            parallel={'mode': 'multiprocessing.Pool', 'n_workers': n_workers,
                      'start_method': 'fork', 'blas_threads_per_worker': 1,
                      'why': 'ladder cells are independent phase-3 solves',
                      'determinism': ('each cell is a deterministic banded '
                                      'solve; the pool only changes completion '
                                      'ORDER and cells are re-keyed by '
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
        inputs.append((STAGE2_ANALYSIS, 'prior_run_output',
                       'stage2_analysis_v1_the_ladder_centres'))
        R.declare_inputs(inputs)
        R.declare_output(met_path, role='json',
                         note='per-cell misfit on the fine and ultra ladders')
        R.declare_output(arr_path, role='arrays_npz',
                         note='ladder coordinates and misfits, plus the mesh')
        R.set_source(source_group)
        R.set_numerics(numerics)
        R.set_results({
            'n_phase3_solves': len(cells),
            'n_fine_cells': int(len(jobs)), 'n_ultra_cells': int(len(ultra_jobs)),
            'phase3_taxis_sha256': rm.sha256_array(ta3_ref),
            'all_cells_share_one_phase3_taxis': True,
            'barrier_node_mask_is_ratio_independent': True,
            'all_barriers_realise_exactly_2w': True,
            'wall_s_total': time.time() - t_all,
            'wall_s_pool_fine': pool1, 'wall_s_pool_ultra': pool2})
        R.note('C4 amend 1: the per-w optimum of the stage-2 report was located '
               'by a parabola on a 0.25-decade ratio ladder (0.5-decade for the '
               'D curve) and never checked for convergence in that step. This '
               'run re-measures every published vertex on a 0.05-decade ladder '
               'and, at w = 1 ft, on a 0.0125-decade ladder, so the vertex is '
               'measured on an h-sequence and the observed order is reported.')
        R.note('Each ladder is CENTRED on the published vertex, so the k = 0 '
               'cell is the simulated misfit AT that vertex: it measures the '
               'parabola-interpolation error of misfit_at_opt_psi directly, '
               'which is what floor_variation_psi was built from.')
        R.note('The solver, the chain, the mesh, the misfit and the barrier '
               'construction are IMPORTED from c4_identifiability and '
               'a5_two_stage_chain unmodified; only the ratio ladder differs '
               'from the parent run at the same config.')
    print(f"[C4a1] {spec_name} done in {time.time() - t_all:.0f} s -> {outdir}",
          flush=True)
    return 0


# ---------------------------------------------------------------------------
# analysis (solve-free)
# ---------------------------------------------------------------------------

def _ladder_rows(doc, ladder):
    rows = {}
    for c in doc['cells']:
        if c['ladder'] != ladder:
            continue
        rows.setdefault((round(c['D0'], 9), round(c['w_ft'], 9)), []).append(c)
    for k in rows:
        rows[k].sort(key=lambda c: c['log10_ratio'])
    return rows


def _vertex(rows):
    lr = np.array([c['log10_ratio'] for c in rows])
    y = np.array([c['misfit_psi'] for c in rows])
    j = int(np.argmin(y))
    edge = j in (0, len(y) - 1)
    if edge:
        return float(lr[j]), float(y[j]), True, float(y[j]), float(lr[j])
    xv, yv = parab_vertex(lr[j - 1:j + 2], y[j - 1:j + 2])
    return xv, yv, False, float(y[j]), float(lr[j])


def linfit(u, v):
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    n = u.size
    A = np.vstack([u, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((v - pred) ** 2))
    ss_tot = float(np.sum((v - v.mean()) ** 2))
    cov = (ss_res / max(n - 2, 1)) * np.linalg.inv(A.T @ A)
    return {'slope': float(coef[0]), 'intercept': float(coef[1]),
            'slope_stderr': float(np.sqrt(cov[0, 0])),
            'r2': float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
            'n': int(n)}


def run_analyse(tag):
    out = {'study_id': STUDY_ID, 'task_id': TASK_ID, 'tag': tag,
           'kind': 'c4_amend1_ratio_grid_convergence',
           'generated_utc': utcnow(),
           'conventions': {
               'minimum': ('THE minimum of a (w, ratio) row is the parabola '
                           'vertex on the 0.05-decade ladder ("converged '
                           'vertex"). Its own interpolation error is measured '
                           'against the 0.0125-decade ladder at w = 1 ft and '
                           'reported. Grid minima and coarse-ladder vertices '
                           'are carried alongside but are never THE minimum.'),
               'resistance': 'W_tot / D_barrier*, W_tot = 6 * 2w = 12w ft',
               'misfit': MISFIT_KEY}}

    docs = {}
    for name, spec in SPECS.items():
        p = os.path.join(OUTROOT, spec['outdir'], f"c4_finegrid_{spec['tag']}.json")
        if os.path.exists(p):
            docs[name] = json.load(open(p))
    out['sources'] = {n: {'path': os.path.relpath(
        os.path.join(OUTROOT, SPECS[n]['outdir'],
                     f"c4_finegrid_{SPECS[n]['tag']}.json"), ROOT),
        'sha256': rm.sha256_file(os.path.join(
            OUTROOT, SPECS[n]['outdir'], f"c4_finegrid_{SPECS[n]['tag']}.json"))}
        for n in docs}
    out['stage2_analysis_sha256'] = rm.sha256_file(STAGE2_ANALYSIS)

    A2 = json.load(open(STAGE2_ANALYSIS))

    # ---- the planes -------------------------------------------------------
    planes = {}
    for name in ('plane140', 'plane550', 'plane1150'):
        if name not in docs:
            continue
        doc = docs[name]
        spec = SPECS[name]
        pl = A2['planes'][spec['plane_key']]
        D0 = float(pl['D0_ft2_s'])
        fine = _ladder_rows(doc, 'fine')
        ultra = _ladder_rows(doc, 'ultra')
        per_w = []
        for p in pl['per_w']:
            w = float(p['w_ft'])
            rows = fine[(round(D0, 9), round(w, 9))]
            xv, yv, edge, gmin, glr = _vertex(rows)
            k0 = [c for c in rows if c['k'] == 0][0]
            Wtot = N_BARRIERS * 2.0 * w
            rec = {
                'w_ft': w,
                'coarse_log10_ratio_opt': float(p['log10_ratio_opt']),
                'coarse_vertex_misfit_psi': float(p['misfit_at_opt_psi']),
                'coarse_grid_min_psi': float(p['grid_min_psi']),
                'coarse_resistance_s_per_ft': float(p['resistance_s_per_ft']),
                'simulated_misfit_at_coarse_vertex_psi': float(k0['misfit_psi']),
                'coarse_vertex_interpolation_error_psi':
                    float(p['misfit_at_opt_psi'] - k0['misfit_psi']),
                'fine_log10_ratio_opt': xv,
                'fine_min_psi': yv,
                'fine_grid_min_psi': gmin,
                'fine_argmin_on_ladder_edge': bool(edge),
                'fine_ratio_opt': float(10.0 ** xv),
                'fine_D_barrier_opt_ft2_s': float(D0 * 10.0 ** xv),
                'fine_resistance_s_per_ft': float(Wtot / (D0 * 10.0 ** xv)),
                'shift_decades_coarse_to_fine':
                    float(xv - float(p['log10_ratio_opt'])),
                'resistance_bias_pct_of_fine':
                    float(100.0 * (float(p['resistance_s_per_ft'])
                                   - Wtot / (D0 * 10.0 ** xv))
                          / (Wtot / (D0 * 10.0 ** xv))),
                'total_width_ft': Wtot,
                'crossing_time_s': float(Wtot * Wtot / (D0 * 10.0 ** xv)),
            }
            key = (round(D0, 9), round(w, 9))
            if key in ultra:
                uv, uy, uedge, ug, ulr = _vertex(ultra[key])
                rec['ultra_log10_ratio_opt'] = uv
                rec['ultra_min_psi'] = uy
                rec['ultra_resistance_s_per_ft'] = float(Wtot / (D0 * 10.0 ** uv))
                rec['shift_decades_fine_to_ultra'] = float(uv - xv)
                rec['fine_resistance_bias_pct_of_ultra'] = float(
                    100.0 * (Wtot / (D0 * 10.0 ** xv) - Wtot / (D0 * 10.0 ** uv))
                    / (Wtot / (D0 * 10.0 ** uv)))
            per_w.append(rec)

        Rf = np.array([r['fine_resistance_s_per_ft'] for r in per_w])
        Rc = np.array([r['coarse_resistance_s_per_ft'] for r in per_w])
        Yf = np.array([r['fine_min_psi'] for r in per_w])
        Yc = np.array([r['coarse_vertex_misfit_psi'] for r in per_w])
        Ys = np.array([r['simulated_misfit_at_coarse_vertex_psi']
                       for r in per_w])
        ws = np.array([r['w_ft'] for r in per_w])
        fit = linfit(np.log10(ws),
                     np.log10([r['fine_D_barrier_opt_ft2_s'] for r in per_w]))
        d = np.diff(Yf)
        planes[spec['plane_key']] = {
            'D0_ft2_s': D0, 'pad_ft': spec['pad_ft'], 'per_w': per_w,
            'resistance_s_per_ft': {
                'converged_mean': float(Rf.mean()),
                'converged_min': float(Rf.min()), 'converged_max': float(Rf.max()),
                'converged_spread_pct_of_mean':
                    float(100.0 * (Rf.max() - Rf.min()) / Rf.mean()),
                'coarse_mean': float(Rc.mean()),
                'coarse_spread_pct_of_mean':
                    float(100.0 * (Rc.max() - Rc.min()) / Rc.mean()),
                'coarse_bias_pct_of_converged':
                    float(100.0 * (Rc.mean() - Rf.mean()) / Rf.mean())},
            'floor_variation_psi': {
                'converged_vertex': float(Yf.max() - Yf.min()),
                'converged_vertex_pct_of_min':
                    float(100.0 * (Yf.max() - Yf.min()) / Yf.min()),
                'coarse_vertex_as_published': float(Yc.max() - Yc.min()),
                'simulated_at_coarse_vertex_ratios': float(Ys.max() - Ys.min()),
                'coarse_vertex_interpolation_error_range_psi':
                    float((Yc - Ys).max() - (Yc - Ys).min()),
                'monotone_in_w': bool(np.all(d > 0)),
                'n_positive_differences': int((d > 0).sum()),
                'n_differences': int(d.size),
                'floor_psi_by_w': [float(v) for v in Yf]},
            'exponent_fit_log10_Dbarrier_vs_log10_w_fine': fit,
            'converged_minimum_psi': float(Yf.min()),
            'converged_minimum_at_w_ft': float(ws[int(np.argmin(Yf))]),
        }
    out['planes'] = planes

    # ---- the h-convergence sequence at w = 1 ft ---------------------------
    hconv = {}
    for name in ('plane140', 'plane550', 'plane1150'):
        if name not in docs:
            continue
        spec = SPECS[name]
        pl = A2['planes'][spec['plane_key']]
        D0 = float(pl['D0_ft2_s'])
        rec = [p for p in planes[spec['plane_key']]['per_w']
               if abs(p['w_ft'] - 1.0) < 1e-9]
        if not rec or 'ultra_log10_ratio_opt' not in rec[0]:
            continue
        r = rec[0]
        e_coarse = r['coarse_log10_ratio_opt'] - r['ultra_log10_ratio_opt']
        e_fine = r['fine_log10_ratio_opt'] - r['ultra_log10_ratio_opt']
        hconv[spec['plane_key']] = {
            'D0_ft2_s': D0, 'w_ft': 1.0,
            'log10_ratio_opt_by_step': {
                '0.25': r['coarse_log10_ratio_opt'],
                '0.05': r['fine_log10_ratio_opt'],
                '0.0125': r['ultra_log10_ratio_opt']},
            'resistance_s_per_ft_by_step': {
                '0.25': r['coarse_resistance_s_per_ft'],
                '0.05': r['fine_resistance_s_per_ft'],
                '0.0125': r['ultra_resistance_s_per_ft']},
            'error_vs_finest_decades': {'0.25': e_coarse, '0.05': e_fine},
            'error_ratio_coarse_over_fine':
                float(e_coarse / e_fine) if e_fine != 0 else None,
            'step_ratio': 5.0,
            'observed_order_p':
                float(np.log(abs(e_coarse / e_fine)) / np.log(5.0))
                if e_fine != 0 else None,
            'interpretation': ('a 5x step reduction that divides the error by '
                               '~5 is first order (O(h)); by ~25 would be '
                               'second order')}
    out['h_convergence_at_w1'] = hconv

    # ---- anisotropy and the +delta geometry, on the converged ladder ------
    # Stage 1 measured "along vs across" with delta = the GRID floor variation
    # and the across-band half-width read off the 0.25-decade ladder; stage 2
    # quoted a vertex floor variation instead. Both estimators are noisier than
    # the quantity itself. Here ONE convention is used throughout: delta is the
    # converged (0.05-decade) floor variation, and the across-band half-width is
    # read off the same converged ladder by linear interpolation between cells.
    anis = {}
    for name in ('plane140', 'plane550', 'plane1150'):
        if name not in docs:
            continue
        spec = SPECS[name]
        key = spec['plane_key']
        doc = docs[name]
        D0 = float(A2['planes'][key]['D0_ft2_s'])
        fine = _ladder_rows(doc, 'fine')
        delta = planes[key]['floor_variation_psi']['converged_vertex']
        half, censored = [], 0
        for p_ in planes[key]['per_w']:
            w = float(p_['w_ft'])
            rows = fine[(round(D0, 9), round(w, 9))]
            lr = np.array([c['log10_ratio'] for c in rows])
            y = np.array([c['misfit_psi'] for c in rows])
            xv, yv = float(p_['fine_log10_ratio_opt']), float(p_['fine_min_psi'])
            lvl = yv + delta
            lo = hi = None
            for i in range(len(lr) - 1):
                dy = y[i + 1] - y[i]
                if dy == 0:
                    continue
                if lr[i + 1] <= xv and y[i] >= lvl >= y[i + 1]:
                    lo = lr[i] + (lvl - y[i]) * (lr[i + 1] - lr[i]) / dy
                if lr[i] >= xv and y[i] <= lvl <= y[i + 1] and hi is None:
                    hi = lr[i] + (lvl - y[i]) * (lr[i + 1] - lr[i]) / dy
            if lo is None or hi is None:
                half.append(None)
                censored += 1
            else:
                half.append(float(0.5 * (hi - lo)))
        ok = [h for h in half if h is not None]
        span = float(np.log10(max(p_['w_ft'] for p_ in planes[key]['per_w'])
                              / min(p_['w_ft'] for p_ in planes[key]['per_w'])))
        anis[key] = {
            'D0_ft2_s': D0,
            'delta_psi': delta,
            'delta_convention': ('the converged (0.05-decade vertex) floor '
                                 'variation over the whole w grid'),
            'w_span_decades': span,
            'across_halfwidth_decades_by_w': half,
            'n_censored_by_ladder_span': censored,
            'across_halfwidth_decades_median':
                float(np.median(ok)) if ok else None,
            'anisotropy_along_over_across':
                float(span / float(np.median(ok))) if ok else None,
            'published_anisotropy_stage1_or_2': None,
            'note': ('a LOWER bound: both ends of the along-band direction are '
                     'w-grid edges, so the along extent is censored'),
        }
    out['anisotropy_converged'] = anis

    # ---- one convention for THE minimum, applied to the published claims ---
    try:
        import c4_stage2 as st2
        runs, _ = st2.load_runs()
    except Exception as exc:          # pragma: no cover
        runs, st2 = None, None
        out['minimum_convention_note'] = f'plane matrices unavailable: {exc!r}'
    mins = {}
    if runs is not None:
        for name in ('plane140', 'plane550', 'plane1150'):
            if name not in docs:
                continue
            spec = SPECS[name]
            key = spec['plane_key']
            run_label = key.split(':')[0]
            D0 = float(A2['planes'][key]['D0_ft2_s'])
            ws, lrs, Amat = st2.plane_matrix(runs[run_label], D0)
            iw = int(np.argmin(np.abs(np.asarray(ws) - 1.0)))
            jr = int(np.argmin(np.abs(np.asarray(lrs) + 5.0)))
            p1 = [q for q in planes[key]['per_w'] if abs(q['w_ft'] - 1.0) < 1e-9][0]
            m_ms = float(Amat[iw, jr])
            mins[key] = {
                'D0_ft2_s': D0, 'w_ft': 1.0,
                'misfit_at_manuscript_ratio_1e-5_psi': m_ms,
                'grid_min_at_w1_psi': float(p1['coarse_grid_min_psi']),
                'coarse_vertex_min_at_w1_psi':
                    float(p1['coarse_vertex_misfit_psi']),
                'converged_min_at_w1_psi': float(p1['fine_min_psi']),
                'penalty_pct_vs_grid_min':
                    float(100.0 * (m_ms - p1['coarse_grid_min_psi'])
                          / p1['coarse_grid_min_psi']),
                'penalty_pct_vs_coarse_vertex':
                    float(100.0 * (m_ms - p1['coarse_vertex_misfit_psi'])
                          / p1['coarse_vertex_misfit_psi']),
                'penalty_pct_vs_converged_min':
                    float(100.0 * (m_ms - p1['fine_min_psi'])
                          / p1['fine_min_psi']),
                'plane_min_converged_psi': planes[key]['converged_minimum_psi'],
                'floor_variation_as_pct_of_converged_min':
                    float(100.0 * planes[key]['floor_variation_psi']
                          ['converged_vertex']
                          / planes[key]['converged_minimum_psi']),
            }
    out['minimum_convention'] = {
        'rule': ('THE minimum of a row is the converged (0.05-decade) parabola '
                 'vertex. The same convention is used for the floor variation, '
                 'the anisotropy delta and the manuscript-ratio penalty, so the '
                 'three are mutually consistent.'),
        'rows': mins}

    # ---- the resistance across backgrounds, converged ---------------------
    keys = [('pad5000:140', 140.0), ('pad20000:550', 550.0),
            ('pad20000:1150', 1150.0)]
    have = [(k, d) for k, d in keys if k in planes]
    if have:
        Rm = [planes[k]['resistance_s_per_ft']['converged_mean'] for k, _ in have]
        Rc = [planes[k]['resistance_s_per_ft']['coarse_mean'] for k, _ in have]
        D0s = [d for _, d in have]
        out['resistance_invariance_converged'] = {
            'D0_ft2_s': D0s,
            'converged_mean_resistance_s_per_ft': Rm,
            'coarse_mean_resistance_s_per_ft': Rc,
            'converged_spread_pct_of_mean':
                float(100.0 * (max(Rm) - min(Rm)) / float(np.mean(Rm))),
            'coarse_spread_pct_of_mean':
                float(100.0 * (max(Rc) - min(Rc)) / float(np.mean(Rc))),
            'per_barrier_s_per_ft': [v / N_BARRIERS for v in Rm],
            'fit_log10_R_vs_log10_D0': linfit(np.log10(D0s), np.log10(Rm))
            if len(Rm) > 2 else None,
            'grid_bias_pct_by_plane':
                {k: planes[k]['resistance_s_per_ft']
                 ['coarse_bias_pct_of_converged'] for k, _ in have},
        }

    # ---- the D curve, converged ------------------------------------------
    if 'dcurve' in docs:
        doc = docs['dcurve']
        fine = _ladder_rows(doc, 'fine')
        # which backgrounds are PADDING-converged, read off the 40000 ft
        # probe rather than hard-coded: a background fails if the worst
        # 20000 -> 40000 change on the pad ladder reaches 1 % (the criterion
        # stage 2 stated in advance).
        worst = {}
        for r in A2['pad_ladder']['rows']:
            d = float(r['D0'])
            worst[d] = max(worst.get(d, 0.0),
                           abs(float(r['change_20000_to_40000_pct'])))
        not_conv = sorted(d for d, v in worst.items() if v >= 1.0)
        rows = []
        for row in A2['d_curve_pad20000']['rows']:
            D0 = float(row['D0_ft2_s'])
            w = float(row['w_ft'])
            cells = fine[(round(D0, 9), round(w, 9))]
            xv, yv, edge, gmin, glr = _vertex(cells)
            k0 = [c for c in cells if c['k'] == 0][0]
            Wtot = N_BARRIERS * 2.0 * w
            rows.append({
                'D0': D0, 'w_ft': w,
                'padding_converged': bool(D0 not in not_conv),
                'coarse_log10_ratio_opt': float(np.log10(float(row['ratio_opt']))),
                'coarse_min_psi': float(row['profiled_min_psi']),
                'coarse_resistance_s_per_ft': float(row['resistance_s_per_ft']),
                'coarse_D_barrier_opt_ft2_s':
                    float(row['D_barrier_opt_ft2_s']),
                'simulated_misfit_at_coarse_vertex_psi': float(k0['misfit_psi']),
                'fine_log10_ratio_opt': xv, 'fine_min_psi': yv,
                'fine_argmin_on_ladder_edge': bool(edge),
                'fine_ratio_opt': float(10.0 ** xv),
                'fine_D_barrier_opt_ft2_s': float(D0 * 10.0 ** xv),
                'fine_resistance_s_per_ft': float(Wtot / (D0 * 10.0 ** xv)),
                'shift_decades_coarse_to_fine':
                    float(xv - np.log10(float(row['ratio_opt']))),
            })
        conv = [r for r in rows if r['padding_converged']]
        out['d_curve_converged_ladder'] = {
            'padding_not_converged_D0': not_conv,
            'padding_convergence_criterion':
                ('worst 20000 -> 40000 ft change on the pad ladder >= 1 % of '
                 'the misfit; read from pad_ladder, not hard-coded'),
            'rows': rows,
            'n_rows': len(rows), 'n_padding_converged_rows': len(conv),
            'fit_log10_ratio_vs_log10_D0_8rows_fine':
                linfit([np.log10(r['D0']) for r in conv],
                       [r['fine_log10_ratio_opt'] for r in conv]),
            'fit_log10_ratio_vs_log10_D0_8rows_coarse':
                linfit([np.log10(r['D0']) for r in conv],
                       [r['coarse_log10_ratio_opt'] for r in conv]),
            'fit_log10_ratio_vs_log10_D0_9rows_coarse':
                linfit([np.log10(r['D0']) for r in rows],
                       [r['coarse_log10_ratio_opt'] for r in rows]),
            'fit_log10_R_vs_log10_D0_8rows_fine':
                linfit([np.log10(r['D0']) for r in conv],
                       [np.log10(r['fine_resistance_s_per_ft']) for r in conv]),
            'D_barrier_opt_range_8rows_fine':
                [float(min(r['fine_D_barrier_opt_ft2_s'] for r in conv)),
                 float(max(r['fine_D_barrier_opt_ft2_s'] for r in conv))],
            'D_barrier_opt_range_8rows_coarse':
                [float(min(r['coarse_D_barrier_opt_ft2_s'] for r in conv)),
                 float(max(r['coarse_D_barrier_opt_ft2_s'] for r in conv))],
            'D_barrier_opt_range_9rows_coarse':
                [float(min(r['coarse_D_barrier_opt_ft2_s'] for r in rows)),
                 float(max(r['coarse_D_barrier_opt_ft2_s'] for r in rows))],
            'D0_span_factor_8rows': float(max(r['D0'] for r in conv)
                                          / min(r['D0'] for r in conv)),
            'D0_span_factor_9rows': float(max(r['D0'] for r in rows)
                                          / min(r['D0'] for r in rows)),
        }

    out['code_sha256'] = {
        os.path.relpath(os.path.join(HERE, f), ROOT):
            rm.sha256_file(os.path.join(HERE, f))
        for f in ('c4_stage2_amend1.py', 'c4_stage2.py', 'c4_identifiability.py',
                  'rev2_core.py', 'rev2_data.py', 'rev2_manifest.py',
                  'a5_two_stage_chain.py')}

    path = os.path.join(OUTROOT, f"c4_stage2_amend1_{tag}.json")
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"[C4a1] analysis -> {path}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=('finegrid', 'analyse'))
    ap.add_argument('--spec', choices=sorted(SPECS))
    ap.add_argument('--tag', default='v1')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--smoke-dir', default=None,
                    help='cheap end-to-end check of the manifest block; writes '
                         'a 3-cell ladder to this directory instead of the '
                         'task output tree')
    a = ap.parse_args(argv)
    if a.mode == 'finegrid':
        if not a.spec:
            ap.error('--spec is required for --mode finegrid')
        return run_finegrid(a.spec, a.workers, smoke_dir=a.smoke_dir)
    return run_analyse(a.tag)


if __name__ == '__main__':
    raise SystemExit(main())
