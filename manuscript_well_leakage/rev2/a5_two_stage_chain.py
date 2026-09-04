#!/usr/bin/env python3
"""A5 - the manuscript two-stage chain, runnable end to end.

    python3 scripts/manuscript_well_leakage/rev2/a5_two_stage_chain.py \
        --config configs/rev2/a5_two_stage_chain.json

Run with CWD = repo root. Writes ONLY into `output/rev2_20260901/A5/<study>/`.

What this reproduces
--------------------
`scripts/well_leakage_history_matching/101_fiberis_matching.py` chains three
solves on one mesh:

    phase 1  stage-7 injection, six stage-7 frac-hit nodes driven by gauge 6
    phase 2  shut-in, same six nodes, same gauge, initial = phase-1 final field
    phase 3  stage-8 injection, six stage-8 frac-hit nodes driven by gauge 7,
             initial = phase-2 final field, and the six STAGE-7 nodes now carry a
             reduced diffusivity - the frac hits of the previous stage acting as
             barriers.

That script cannot run under current fibeRIS: it reads
`prev_result['data'][-1, :]` (correct only for the pre-2025-02-11 time-major npz
layout), and the archive it wrote is mixed-layout. Nothing here imports it; the
legacy scripts and `output/0211_simulation_MULTIstage/` are read-only history.

Everything numerical goes through the shared rev2 modules:
`rev2_core` (solver + physical-width barriers), `rev2_data` (windows, gauges,
frac hits, mesh), `rev2_layout` (the mixed-layout archive reader/writer),
`rev2_manifest` (house rule 3). None of them is edited by this file.

Two things this script deliberately does NOT decide, per the task assignment.
Both are config parameters, both run at the LEGACY value in the deliverable
study, and both get a flagged sensitivity study of their own:
  * `mesh.pad_low_ft` - the legacy domain leaves 1797 ft below the lowest
    plotted gauge. Task B2 owns the decision.
  * `physics.D_baseline_ft2_s` - 101 hard-codes 140; the recalibrated values are
    1150 (absolute norm) / 550 (amplitude-normalised norm).

Three legacy behaviours are reproduced on purpose and are flagged in the
manifest rather than silently fixed; see the config comments for each:
  * `mesh.refinement = "legacy_refine_mesh"` keeps the buggy
    `mesh_utils.refine_mesh` (dx = 2/15 ft, not the intended 0.2 ft);
  * `source.crop_rebase = "requested_start"` reproduces the Feb-2025 fibeRIS
    `crop()` convention, without which the archive cannot be matched;
  * `phase_boundaries.mode = "legacy_file_span"` keeps 101's phase boundaries,
    which are pumping-FILE spans and not pumping events.
"""

import argparse
import copy
import datetime
import json
import os
import sys
import time
import warnings

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.dates as mdates        # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))
for _p in (_HERE, os.path.join(_ROOT, 'fibeRIS', 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rev2_core as rc                   # noqa: E402
import rev2_data as rd                   # noqa: E402
import rev2_layout as rl                 # noqa: E402
import rev2_manifest as rm               # noqa: E402

STUDY_ID = "A5_two_stage_chain"
TASK_ID = "A5"

# The five legacy consumers of output/0211_simulation_MULTIstage and what each
# one gets wrong under CURRENT fibeRIS. Measured in the A5 layout investigation
# (output/rev2_20260901/A5/a5_consumer_chains.json); reproduced here so the
# repro record this run emits is self-contained.
LEGACY_CONSUMERS = {
    "101_fiberis_matching.py": {
        "role": "producer of the archive AND consumer of its own phase1/phase2",
        "reads": ["phase1.npz -> :156", "phase2.npz -> :199, :233"],
        "idiom": "np.load(...)['data'][-1, :]",
        "assumed_layout": "time_major (n_t, n_x)",
        "verdict_today": ("correct for the six 2025-02-11 files as they sit on "
                          "disk; WRONG for anything current fibeRIS writes, "
                          "where [-1, :] returns the time history of the last "
                          "mesh node instead of the final spatial profile"),
        "other_defects": [
            ":42-47 phase boundaries come from get_start_time/get_end_time, "
            "which ignore their threshold and return FILE spans, not pumping "
            "events",
            ":142 phase2_total_time is computed and never used - solve() takes "
            "no t_total, so t_total = source[0].taxis[-1]",
            ":178 phase3_total_time likewise computed and never used",
            ":77 np.arange(12500, 12500+5500, 1) ends at 17999, not 18000",
            ":80-84 refine_mesh's point count uses a NODE count where an "
            "interval count is meant, so dx = 2/15 ft, not the intended 0.2 ft",
            ":190 the barrier is one mesh node, so its physical width is "
            "whatever the local dx happens to be",
            ":214 writes phase3_test.npz nominally at ratio 1e-6; the file on "
            "disk measures as ratio 1.0 (no barrier) - see the run results",
        ],
    },
    "101p_fiberis_matching_postprocessing.py": {
        "role": "post-processing / figures",
        "reads": ["phase1.npz", "phase2.npz", "phase3_0.1.npz"],
        "idiom": "phase1/2 transposes COMMENTED OUT at :22-23; phase3 still "
                 "transposed at :35",
        "assumed_layout": "inconsistent: depth_major for phase1/2, time_major "
                          "for phase3",
        "verdict_today": ("IndexError - boolean index did not match indexed "
                          "array along dimension 1 (5656 vs 310); phase1/2 are "
                          "left time-major and every depth mask then misfires"),
    },
    "102_hist_matching_sensitivity_analysis.py": {
        "role": "sensitivity analysis",
        "reads": ["phase1.npz", "phase2.npz"],
        "idiom": ".T at :26-27",
        "assumed_layout": "time_major on disk",
        "verdict_today": ("COMPLETES, and is correct - it only ever touches the "
                          "two old-layout files. Its :14 file list names "
                          "phase3_test.npz alongside the ratio files, so the "
                          "same .T would be wrong the moment phase 3 is loaded"),
    },
    "DAS_history_matching_visualization/102_IMAGE25abstract.py": {
        "role": "abstract figure",
        "reads": ["phase1.npz", "phase2.npz", "phase3_test.npz"],
        "idiom": ".T at :24-25 and :37",
        "assumed_layout": "time_major for all three",
        "verdict_today": ("IndexError (5656 vs 443): phase3_test.npz was "
                          "regenerated 2025-04-28 in the NEW depth-major layout, "
                          "so the .T at :37 puts it back to time-major"),
    },
    "DAS_history_matching_visualization/102r_IMAGE25abstract_with_scalar.py": {
        "role": "abstract figure with a scalar overlay",
        "reads": ["phase1.npz", "phase2.npz", "phase3_test.npz"],
        "idiom": ".T at :25-26 and :38",
        "assumed_layout": "time_major for all three",
        "verdict_today": "IndexError (5656 vs 443), same cause as 102",
    },
    "DAS_history_matching_visualization/104_full_history_matching_manuscript.py": {
        "role": "THE manuscript two-stage figure",
        "reads": ["phase1.npz", "phase2.npz", "phase3_test.npz"],
        "idiom": ".T at :24-25; :37 is a bare `data = data` where the .T was "
                 "hand-deleted after phase3_test.npz was regenerated",
        "assumed_layout": "time_major for phase1/2, depth_major for phase3_test",
        "verdict_today": ("layout handling COMPLETES and is correct. The "
                          "surviving defect is not layout: the phase-3 panel is "
                          "fed phase3_test.npz, which contains no barrier at "
                          "all (measured ratio 1.0), so the figure's phase-3 "
                          "curve is the UNIFORM D solution"),
    },
}

# Every call site raises TypeError on the legacy select_time idiom under current
# fibeRIS (core2D.py:473 demands both bounds have the same type).
SELECT_TIME_TRAP = ("obj.select_time(30, obj.get_end_time()) now raises "
                    "TypeError: pass two floats, e.g. "
                    "obj.select_time(30.0, float(obj.taxis[-1]))")


# ---------------------------------------------------------------------------
# config plumbing
# ---------------------------------------------------------------------------

def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path):
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------

def build_chain_mesh(mcfg):
    """Base mesh + optional refinement around the frac hits.

    Returns (x, record). `record` is JSON-serialisable and goes verbatim into
    `mesh_record(refinement=...)`, because "refined around the frac hits" is not
    a reproducible statement and the realised dx is the thing that matters.
    """
    lo = float(mcfg['md_lo_ft']) - float(mcfg['pad_low_ft'])
    hi = float(mcfg['md_hi_ft']) + float(mcfg['pad_high_ft'])
    dx = float(mcfg['dx_ft'])
    # rev2_data.build_mesh's count-based construction; window=(lo, hi) and zero
    # pads because the padding is already folded into lo/hi above, so the record
    # reports the requested pad against the ORIGINAL window below.
    base = rd.build_mesh((lo, hi), 0.0, 0.0, dx)
    x = base.x
    mode = mcfg['refinement']
    calls = []

    if mode == 'none':
        pass
    elif mode in ('legacy_refine_mesh', 'corrected_refine'):
        from fiberis.utils import mesh_utils
        w = float(mcfg['refine_half_width_ft'])
        factor = int(mcfg['refine_factor'])
        for stage in mcfg['refine_stages']:
            hits = rd.load_frac_hits(int(stage), unique=False, sort=False)
            if mcfg.get('refine_round_centres', True):
                hits = np.round(hits)
            for c in hits:
                c = float(c)
                start, end = c - w, c + w
                i0 = int(np.searchsorted(x, start, side='left'))
                i1 = int(np.searchsorted(x, end, side='right'))
                n_nodes_in = i1 - i0
                before = len(x)
                if mode == 'legacy_refine_mesh':
                    # Verbatim legacy call, defect included.
                    x = mesh_utils.refine_mesh(x, [start, end], factor)
                    n_pts = n_nodes_in * factor + 1
                else:
                    # The intended behaviour: refine each INTERVAL by `factor`.
                    n_int = max(n_nodes_in - 1, 1)
                    n_pts = n_int * factor + 1
                    refined = np.linspace(start, end, n_pts)
                    coarse = np.concatenate((x[:i0], x[i1:]))
                    x = np.sort(np.concatenate((coarse, refined)))
                calls.append({
                    'stage': int(stage), 'centre_md_ft': c,
                    'range_ft': [start, end],
                    'nodes_in_range_before': int(n_nodes_in),
                    'n_points_inserted': int(n_pts),
                    'realised_dx_ft': float((end - start) / (n_pts - 1))
                                      if n_pts > 1 else None,
                    'degenerate': bool(n_nodes_in == 0),
                    'nx_before': int(before), 'nx_after': int(len(x)),
                })
    else:
        raise ValueError(f"unknown mesh.refinement {mode!r}")

    x = np.asarray(x, dtype=float)
    d = np.diff(x)
    record = {
        'mode': mode,
        'base_nx': int(base.nx),
        'base_span_md_ft': [float(base.x[0]), float(base.x[-1])],
        'refine_half_width_ft': float(mcfg.get('refine_half_width_ft', 0.0)),
        'refine_factor': int(mcfg.get('refine_factor', 0)),
        'refine_stages': list(mcfg.get('refine_stages', [])),
        'n_calls': len(calls),
        'n_degenerate_calls': int(sum(c['degenerate'] for c in calls)),
        'realised_dx_ft': {'min': float(d.min()), 'max': float(d.max()),
                           'median': float(np.median(d))},
        'distinct_dx_ft': [float(v) for v in np.unique(np.round(d, 9))][:12],
        'calls': calls,
        'legacy_defect_note': (
            "mesh_utils.refine_mesh builds linspace(start, end, "
            "(end_idx-start_idx)*factor + 1) where end_idx-start_idx is a NODE "
            "count. For a +-1 ft window on a 1 ft mesh that is 3, giving 16 "
            "points across 2 ft (dx = 2/15 = 0.13333 ft) where factor = 5 over "
            "+-1 ft implies 0.2 ft. It also degenerates to a single inserted "
            "point at MD-1 when the window contains no node, which is why the "
            "10 ft meshes of 102r/103r place their barriers 1 ft off the frac "
            "hit; on this 1 ft mesh it does not degenerate (0 of 12 calls)."
            if mode == 'legacy_refine_mesh' else
            "not applicable" if mode == 'none' else
            "corrected: linspace over INTERVALS, giving dx = dx_base/factor"),
    }
    return x, record


# ---------------------------------------------------------------------------
# phase windows and sources
# ---------------------------------------------------------------------------

def phase_windows(pcfg):
    """{'phase1'|'phase2'|'phase3': (t_start, t_end)} plus a provenance record."""
    mode = pcfg['mode']
    if mode == 'legacy_file_span':
        w = rd.manuscript_phase_windows()
        wins = {k: (v[1], v[2]) for k, v in w.items()}
        kind = ('pumping FILE spans, via Data1DPumpingCurve.get_start_time / '
                'get_end_time, which ignore their threshold (101:42-47)')
    elif mode == 'pumping_event':
        thr = float(pcfg.get('pumping_threshold_bpm', 1.0))
        hold = float(pcfg.get('pumping_hold_s', 30.0))
        p7 = rd.load_pumping(7, curves=('slurry_rate',))
        p8 = rd.load_pumping(8, curves=('slurry_rate',))
        a7 = p7.pumping_start(threshold_bpm=thr, hold_s=hold)
        b7 = p7.pumping_end(threshold_bpm=thr)
        a8 = p8.pumping_start(threshold_bpm=thr, hold_s=hold)
        b8 = p8.pumping_end(threshold_bpm=thr)
        wins = {'phase1': (a7, b7), 'phase2': (b7, a8), 'phase3': (a8, b8)}
        kind = (f'slurry-rate events, threshold {thr} bpm held {hold} s '
                '(rev2_data.PumpingStage)')
    else:
        raise ValueError(f"unknown phase_boundaries.mode {mode!r}")

    ref = rd.manuscript_phase_windows()
    rec = {'mode': mode, 'definition': kind,
           'windows': {k: [v[0].isoformat(), v[1].isoformat(),
                           (v[1] - v[0]).total_seconds()]
                       for k, v in wins.items()},
           'legacy_file_span_windows': {
               k: [v[1].isoformat(), v[2].isoformat()] for k, v in ref.items()}}
    if mode == 'legacy_file_span':
        p7 = rd.load_pumping(7, curves=('slurry_rate',))
        p8 = rd.load_pumping(8, curves=('slurry_rate',))
        rec['consequence'] = {
            'phase1_starts_before_stage7_pumping_s': (
                p7.pumping_start() - wins['phase1'][0]).total_seconds(),
            'phase1_ends_after_stage7_pumping_s': (
                wins['phase1'][1] - p7.pumping_end()).total_seconds(),
            'phase3_starts_before_stage8_pumping_s': (
                p8.pumping_start() - wins['phase3'][0]).total_seconds(),
            'phase3_ends_after_stage8_pumping_s': (
                wins['phase3'][1] - p8.pumping_end()).total_seconds(),
            'note': ('The phase-1 and phase-3 windows each open before the pump '
                     'does and close after it stops. The Dirichlet datum is the '
                     'GAUGE reading, not the pump, so the extra seconds are not '
                     'fabricated pressure - they are quiet pre/post-injection '
                     'gauge samples. The consequence is a labelling one: t = 0 '
                     'in every archived phase npz, and therefore on every '
                     'manuscript two-stage time axis, is a file boundary, not '
                     'the start of injection. Any arrival time or lag quoted '
                     'from those axes is offset by the amounts above.'),
        }
    return wins, rec


def load_source_series(gauge, t_start, t_end, scfg):
    """Absolute-psi Dirichlet datum for one phase, on the legacy time base.

    Goes through rev2_data.load_window_gauges (which validates the crop, rejects
    an empty window and rejects a non-increasing taxis) and then applies the
    Feb-2025 rebase convention as a pure shift, so the two conventions differ by
    one recorded number rather than by a second load path.
    """
    win = rd.Window(md_min_ft=0.0, md_max_ft=1e9, t_start=t_start, t_end=t_end)
    gw = rd.load_window_gauges(win, gauges=[int(gauge)],
                               baseline='none', rebase='per_gauge')
    s = gw.series[int(gauge)]
    lead_s = (s.t0_abs - t_start).total_seconds()
    mode = scfg['crop_rebase']
    if mode == 'first_sample':
        taxis = np.asarray(s.taxis_s, dtype=float)
        t0_abs = s.t0_abs
    elif mode == 'requested_start':
        taxis = np.asarray(s.taxis_s, dtype=float) + lead_s
        t0_abs = t_start
    else:
        raise ValueError(f"unknown source.crop_rebase {mode!r}")
    return {
        'gauge': int(gauge), 'md_ft': float(s.md_ft),
        'taxis_s': taxis, 'values_psi': np.asarray(s.raw_psi, dtype=float),
        't0_abs': t0_abs, 'first_sample_abs': s.t0_abs,
        'lead_s': float(lead_s), 'crop_rebase': mode,
        't_total_s': float(taxis[-1]), 'n_samples': int(taxis.size),
        'series_path': rd.SWELL_GAUGE_TEMPLATE.format(n=int(gauge)),
    }


# ---------------------------------------------------------------------------
# one chain
# ---------------------------------------------------------------------------

def solve_phase(x, dprof, src, initial, tcfg, pcfg):
    """One phase. Returns (taxis, field, time_record kwargs, extra)."""
    idx = [int(i) for i in src['source_idx']]
    if len(set(idx)) != len(idx):
        dup = sorted({i for i in idx if idx.count(i) > 1})
        raise ValueError(
            f"duplicate source mesh indices {dup}: two Dirichlet sources on one "
            f"node silently overwrite each other, so the run would apply fewer "
            f"independent sources than it declares")
    n = len(idx)
    taxes = [src['taxis_s']] * n
    vals = [src['values_psi']] * n
    common = dict(theta=float(pcfg['theta']),
                  lambda_leak=float(pcfg['lambda_leak']),
                  p0=float(pcfg['p0_psi']),
                  interface_avg=pcfg['interface_avg'])
    startup = int(pcfg['theta_startup_steps'])
    t_total = float(src['t_total_s'])

    if tcfg['mode'] == 'fixed':
        dt = float(tcfg['dt_fixed_s'])
        taxis, field = rc.solve_forward_multi(
            x, dprof, dt, t_total, taxes, vals, idx, initial=initial, t0=0.0,
            theta_startup_steps=startup, **common)
        trec = dict(mode='fixed', theta=common['theta'],
                    t_total_requested_s=t_total, dt_requested_s=dt,
                    source_time_level='n' if common['theta'] == 1.0 else 'n+1',
                    theta_startup_steps=startup)
        extra = {'solver': 'rev2_core.solve_forward_multi', 'n_rejected': 0}
    elif tcfg['mode'] == 'adaptive':
        if startup:
            # rev2_core.solve_forward_adaptive has no theta_startup_steps
            # parameter (rev2_core.py:742-750). CORRECTION 3 requires Rannacher
            # start-up for any multi-stage rerun with theta < 1, and the
            # manuscript's stepping IS the adaptive one, so that combination is
            # currently unreachable. Refuse rather than run without it.
            raise NotImplementedError(
                "physics.theta_startup_steps > 0 with time.mode='adaptive': "
                "rev2_core.solve_forward_adaptive exposes no "
                "theta_startup_steps, so Rannacher start-up cannot be applied "
                "on the adaptive path. Use time.mode='fixed', or leave "
                "theta = 1 where no start-up is needed. Reported to the module "
                "owner rather than worked around here.")
        taxis, field, trace = rc.solve_forward_adaptive(
            x, dprof, t_total, taxes, vals, idx, initial=initial, t0=0.0,
            dt_init=float(tcfg['dt_init_s']), tol=float(tcfg['tol']),
            controller_tol=float(tcfg['controller_tol']),
            safety_factor=float(tcfg['safety_factor']),
            order_p=int(tcfg['order_p']), max_dt=float(tcfg['max_dt_s']),
            min_dt=float(tcfg['min_dt_s']),
            zero_field_policy=tcfg['zero_field_policy'], **common)
        trec = dict(mode='adaptive', theta=common['theta'],
                    t_total_requested_s=t_total,
                    dt_init_s=trace['dt_init_s'], tol=trace['tol'],
                    controller_tol=trace['controller_tol'],
                    max_dt_s=trace['max_dt_s'], min_dt_s=trace['min_dt_s'],
                    safety_factor=trace['safety_factor'],
                    order_p=trace['order_p'],
                    n_steps_rejected=trace['n_rejected'],
                    zero_field_policy=trace['zero_field_policy'],
                    flip_margin=trace['flip_margin'],
                    source_time_level='n' if common['theta'] == 1.0 else 'n+1',
                    theta_startup_steps=startup)
        extra = {'solver': 'rev2_core.solve_forward_adaptive',
                 'n_attempts': trace['n_attempts'],
                 'n_rejected': trace['n_rejected'],
                 'frac_at_max_dt': trace['frac_at_max_dt'],
                 'err_min': trace['err_min'], 'err_max': trace['err_max'],
                 'flip_margin': trace['flip_margin'],
                 'overshoot_s': trace['overshoot_s'],
                 'dt_realised_s': {'min': trace['dt_min_s'],
                                   'max': trace['dt_max_s'],
                                   'mean': trace['dt_mean_s']}}
    else:
        raise ValueError(f"unknown time.mode {tcfg['mode']!r}")
    return taxis, field, trec, extra


def run_chain(cfg, x, wins, ratio):
    """phase 1 -> phase 2 -> phase 3 at one barrier ratio."""
    D0 = float(cfg['physics']['D_baseline_ft2_s'])
    bcfg = cfg['barrier']
    uniform = np.full(len(x), D0, dtype=float)

    out = {}
    prev_field = None
    for spec in cfg['phases']:
        name = spec['name']
        t0, t1 = wins[name]
        src = load_source_series(spec['source_gauge'], t0, t1, cfg['source'])
        stage = int(cfg['source_stage_by_phase'][name])
        hits = rd.load_frac_hits(stage, unique=False, sort=False)
        src['frac_hit_stage'] = stage
        src['frac_hit_mds_ft'] = [float(v) for v in hits]
        src['source_idx'] = [int(np.argmin(np.abs(x - float(h)))) for h in hits]
        src['source_md_ft'] = [float(x[i]) for i in src['source_idx']]
        src['snap_error_ft'] = [float(x[i] - float(h))
                                for i, h in zip(src['source_idx'], hits)]

        if spec['barrier_from_stage'] is None or ratio >= 1.0:
            dprof, brep = uniform.copy(), None
        else:
            bhits = rd.load_frac_hits(int(spec['barrier_from_stage']),
                                      unique=False, sort=False)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                dprof, brep = rc.build_barrier_profile(
                    x, uniform, bhits, float(bcfg['w_ft']), float(ratio),
                    ratio_reference=bcfg['ratio_reference'],
                    combine=bcfg['combine'], on_empty=bcfg['on_empty'],
                    on_outside=bcfg['on_outside'], return_report=True)
            # THE REPORT IS THE AUTHORITY, NOT THE WARNING (rev2_core docstring).
            expected_fallback = len(bhits) if float(bcfg['w_ft']) == 0.0 else 0
            if brep['n_fallback'] != expected_fallback:
                raise RuntimeError(
                    f"{name}: {brep['n_fallback']} barrier(s) fell back to the "
                    f"nearest node, expected {expected_fallback} for "
                    f"w = {bcfg['w_ft']} ft. Messages: "
                    f"{brep['fallback_messages'][:2]}")
            brep['n_warnings_raised'] = len(caught)
            brep['barrier_mds_ft'] = [float(v) for v in bhits]

        if spec['initial'] == 'uniform_first_source_sample':
            initial = np.full(len(x), float(src['values_psi'][0]))
            ic = {'kind': 'uniform', 'value_psi': float(src['values_psi'][0]),
                  'note': "101:131-133 -- absolute psi, not a delta field"}
        elif spec['initial'] == 'previous_phase_final':
            if prev_field is None:
                raise ValueError(f"{name}: no previous phase to chain from")
            initial = prev_field[-1].copy()
            ic = {'kind': 'previous_phase_final_profile',
                  'note': ("101 does this as prev_result['data'][-1, :], which "
                           "is layout-dependent; here it is the in-memory field, "
                           "and rev2_layout.final_profile is the correct reader "
                           "for the archived files")}
        else:
            raise ValueError(f"{name}: unknown initial {spec['initial']!r}")

        t_wall = time.time()
        taxis, field, trec, extra = solve_phase(
            x, dprof, src, initial, cfg['time'], cfg['physics'])
        extra['wall_s'] = time.time() - t_wall
        out[name] = {'taxis': taxis, 'field': field, 'src': src,
                     'time_record_kw': trec, 'extra': extra,
                     'barrier_report': brep, 'initial_condition': ic,
                     'window_abs': [t0.isoformat(), t1.isoformat()],
                     'd_profile': dprof}
        prev_field = field
    return out


# ---------------------------------------------------------------------------
# comparison with the frozen archive
# ---------------------------------------------------------------------------

def compare_with_archive(chain_by_ratio, acfg, x):
    """Every archived panel against the reproduction, read via rev2_layout."""
    adir = acfg['dir']
    rows = {}

    def _cmp(fname, taxis, field):
        p = rl.load_panel(os.path.join(adir, fname))
        r = {'file': fname, 'sha256': p.sha256,
             'on_disk_layout': p.detected_layout,
             'detection_rule': p.evidence.get('rule'),
             'archive_shape_canonical': [p.n_t, p.n_x],
             'repro_shape': [int(field.shape[0]), int(field.shape[1])],
             'archive_start_time': str(p.start_time),
             'daxis_identical': bool(np.array_equal(np.asarray(p.daxis, float),
                                                    x))}
        if [p.n_t, p.n_x] != list(field.shape):
            r['status'] = 'SHAPE_MISMATCH'
            return r
        d = np.asarray(p.data, float) - field
        ta = np.asarray(p.taxis, float)
        r.update(status='COMPARED',
                 taxis_identical=bool(np.array_equal(ta, taxis)),
                 taxis_max_abs_diff_s=float(np.max(np.abs(ta - taxis))),
                 field_max_abs_diff_psi=float(np.max(np.abs(d))),
                 field_rmse_psi=float(np.sqrt(np.mean(d ** 2))),
                 field_max_abs_value_psi=float(np.max(np.abs(p.data))),
                 relative_max_diff=float(np.max(np.abs(d))
                                         / np.max(np.abs(p.data))))
        return r

    ref = chain_by_ratio[sorted(chain_by_ratio)[0]]
    rows['phase1'] = _cmp(acfg['phase1'], ref['phase1']['taxis'],
                          ref['phase1']['field'])
    rows['phase2'] = _cmp(acfg['phase2'], ref['phase2']['taxis'],
                          ref['phase2']['field'])
    rows['phase3'] = {}
    for key, fname in acfg['phase3_by_ratio'].items():
        ratio = float(key)
        if ratio not in chain_by_ratio:
            rows['phase3'][key] = {'file': fname, 'status': 'RATIO_NOT_RUN'}
            continue
        c = chain_by_ratio[ratio]['phase3']
        rows['phase3'][key] = _cmp(fname, c['taxis'], c['field'])

    # Which ratio does each archived phase-3 panel actually contain?
    ident = {}
    for key, fname in acfg['phase3_by_ratio'].items():
        p = rl.load_panel(os.path.join(adir, fname))
        best, best_d = None, None
        for ratio, c in chain_by_ratio.items():
            f = c['phase3']['field']
            if f.shape != (p.n_t, p.n_x):
                continue
            dd = float(np.max(np.abs(np.asarray(p.data, float) - f)))
            if best_d is None or dd < best_d:
                best, best_d = ratio, dd
        ident[fname] = {'nominal_ratio_from_filename': key,
                        'best_matching_run_ratio': best,
                        'max_abs_diff_psi': best_d}
    rows['phase3_ratio_identification'] = ident
    return rows


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def gauge_traces(x, field, md_table, gauges):
    idx = [int(np.argmin(np.abs(x - md_table.md_of(int(g))))) for g in gauges]
    return (np.asarray(idx, dtype=np.int64),
            np.asarray([md_table.md_of(int(g)) for g in gauges], dtype=float),
            np.asarray([float(x[i]) for i in idx], dtype=float),
            np.ascontiguousarray(field[:, idx]))


def save_phase_npz(outdir, tag, x, taxis, field, start_abs, ocfg, written):
    paths = []
    if ocfg['save_full_field']:
        p = os.path.join(outdir, f"{tag}_field.npz")
        panel = rl.Panel(data=np.ascontiguousarray(field),
                         taxis=np.asarray(taxis, float),
                         daxis=np.asarray(x, float), start_time=start_abs,
                         detected_layout=rl.LAYOUT_TIME_MAJOR,
                         evidence={'origin': 'a5_two_stage_chain', 'rule': None},
                         source_path=p)
        rl.save_panel(panel, p, layout=ocfg['field_layout'])
        paths.append((p, 'arrays_npz',
                      f"{tag} full field, stamped layout={ocfg['field_layout']}"))
    written.extend(paths)
    return paths


def make_figure(path, cfg, chain_by_ratio, md_table, dpi):
    """Three chained phases, then the phase-3 barrier-ratio sweep.

    Panel 4 plots only the SHIELDED gauges - those the stage-7 barrier row
    (MD 15187-15364) separates from the stage-8 sources (MD 14940-15118). They
    are the only gauges whose phase-3 response carries any information about the
    reduction ratio; plotting the unshielded ones alongside them is what makes a
    barrier study look more conclusive than it is.
    """
    gauges = list(cfg['targets']['gauges'])
    ratios = sorted(chain_by_ratio)
    ref_ratio = ratios[0]
    hits7 = rd.load_frac_hits(7, unique=False)
    lo7, hi7 = float(np.min(hits7)), float(np.max(hits7))
    # "Shielded" = above the LOWEST stage-7 frac hit, so at least part of the
    # barrier row lies between the stage-8 sources (MD 14940-15118) and the
    # gauge. g6 (MD 15344) sits between the 5th and 6th barriers and is
    # shielded by five of the six.
    shielded = [g for g in gauges if md_table.md_of(int(g)) > lo7]
    if not shielded:
        shielded = gauges[:1]

    fig, axes = plt.subplots(4, 1, figsize=(11, 14))
    colors = plt.cm.viridis(np.linspace(0, 0.88, len(gauges)))
    for ax, phase in zip(axes[:3], ('phase1', 'phase2', 'phase3')):
        c = chain_by_ratio[ref_ratio][phase]
        t0 = datetime.datetime.fromisoformat(c['window_abs'][0])
        tt = [t0 + datetime.timedelta(seconds=float(s)) for s in c['taxis']]
        tr = c['traces'][phase]
        for j, g in enumerate(gauges):
            ax.plot(tt, tr[:, j], color=colors[j], lw=1.1,
                    label=f"g{g} MD {md_table.md_of(int(g)):.0f} ft"
                          + (" (shielded)" if g in shielded else ""))
        ax.set_ylabel('simulated pressure (psi)')
        extra = (f"  barrier ratio {ref_ratio:g}" if phase == 'phase3' else "")
        ax.set_title(f"{phase}  {c['window_abs'][0]} -> {c['window_abs'][1]}"
                     f"{extra}", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=3)

    ax = axes[3]
    cmap = plt.cm.plasma(np.linspace(0, 0.85, max(len(ratios), 2)))
    for k, ratio in enumerate(ratios):
        c = chain_by_ratio[ratio]['phase3']
        t0 = datetime.datetime.fromisoformat(c['window_abs'][0])
        tt = [t0 + datetime.timedelta(seconds=float(s)) for s in c['taxis']]
        for j, g in enumerate(gauges):
            if g not in shielded:
                continue
            ax.plot(tt, c['traces']['phase3'][:, j] - c['traces']['phase3'][0, j],
                    color=cmap[k], lw=1.2,
                    ls=('-' if g == shielded[0] else '--'),
                    label=(f"ratio {ratio:g}" if g == shielded[0] else None))
    ax.set_ylabel('phase-3 dP from its own start (psi)')
    ax.set_xlabel('2020-03-18 (naive local time, as stored)')
    ax.set_title(f"phase 3, shielded gauges "
                 f"{', '.join('g%d' % g for g in shielded)} "
                 f"(at or behind the stage-7 barrier row MD {lo7:.0f}-{hi7:.0f}), "
                 f"barrier ratio sweep", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3)

    fig.suptitle(
        f"A5 two-stage chain - D = {cfg['physics']['D_baseline_ft2_s']:g} "
        f"ft$^2$/s, barrier half-width w = {cfg['barrier']['w_ft']:g} ft, "
        f"mesh '{cfg['mesh']['refinement']}', "
        f"{cfg['time']['mode']} stepping", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# a study = one chain configuration
# ---------------------------------------------------------------------------

def run_study(name, cfg, root_cfg, outdir, config_path):
    os.makedirs(outdir, exist_ok=True)
    manifest_path = os.path.join(outdir, 'manifest.json')
    rm.assert_absent([manifest_path])

    md_table = rd.load_gauge_md_table()
    x, mesh_rec = build_chain_mesh(cfg['mesh'])
    wins, win_rec = phase_windows(cfg['phase_boundaries'])
    gauges = cfg['targets']['gauges']

    with rm.RunRecorder(manifest_path, study_id=STUDY_ID, task_id=TASK_ID,
                        config=cfg, config_path=config_path,
                        run_label=name,
                        require_modules=('rev2_core', 'rev2_data',
                                         'rev2_layout', 'rev2_manifest')) as R:
        chain_by_ratio = {}
        for ratio in cfg['barrier']['ratios']:
            ratio = float(ratio)
            ch = run_chain(cfg, x, wins, ratio)
            for phase, c in ch.items():
                idx, md_req, md_node, tr = gauge_traces(
                    x, c['field'], md_table, gauges)
                c['traces'] = {phase: tr}
                c['trace_meta'] = {'gauges': [int(g) for g in gauges],
                                   'md_requested_ft': md_req.tolist(),
                                   'md_node_ft': md_node.tolist(),
                                   'mesh_idx': idx.tolist()}
            chain_by_ratio[ratio] = ch

        written = []
        ocfg = cfg['outputs']
        for ratio in sorted(chain_by_ratio):
            ch = chain_by_ratio[ratio]
            rtag = ('uniform' if ratio >= 1.0
                    else f"ratio{ratio:g}".replace('.', 'p'))
            for phase, c in ch.items():
                tag = phase if phase != 'phase3' else f"phase3_{rtag}"
                if phase != 'phase3' and ratio != sorted(chain_by_ratio)[0]:
                    continue  # phases 1-2 do not depend on the barrier ratio
                start_abs = c['src']['t0_abs']
                save_phase_npz(outdir, tag, x, c['taxis'], c['field'],
                               start_abs, ocfg, written)
                if ocfg['save_gauge_traces'] or ocfg['save_final_profile']:
                    p = os.path.join(outdir, f"{tag}_summary.npz")
                    payload = {'taxis_s': np.asarray(c['taxis'], float),
                               'mesh_md_ft': x,
                               'gauge_numbers': np.asarray(gauges,
                                                           dtype=np.int64),
                               'gauge_md_ft': np.asarray(
                                   c['trace_meta']['md_requested_ft'], float),
                               'gauge_mesh_md_ft': np.asarray(
                                   c['trace_meta']['md_node_ft'], float),
                               'start_time': np.array(str(start_abs)),
                               'd_profile_ft2_s': c['d_profile']}
                    if ocfg['save_gauge_traces']:
                        payload['gauge_traces_psi'] = c['traces'][phase]
                    if ocfg['save_final_profile']:
                        payload['final_profile_psi'] = c['field'][-1]
                    np.savez(p, **payload)
                    written.append((p, 'arrays_npz',
                                    f"{tag} gauge traces + final profile"))

        fig_path = os.path.join(outdir, f"fig01_{name}_chain_v1.png")
        make_figure(fig_path, cfg, chain_by_ratio, md_table,
                    int(cfg['outputs']['figure_dpi']))
        written.append((fig_path, 'figure_png', 'three-phase chain at gauges'))

        # ---- comparison with the frozen archive ---------------------------
        arch_rows = None
        if cfg.get('compare_to_archive'):
            arch_rows = compare_with_archive(chain_by_ratio,
                                             root_cfg['archive'], x)

        # ---- results ------------------------------------------------------
        results = {
            'mesh': mesh_rec,
            'phase_boundaries': win_rec,
            'phases': {},
            'archive_comparison': arch_rows if arch_rows is not None
                                  else 'not requested for this study',
            'open_parameters_NOT_settled_here': {
                'mesh.pad_low_ft': {
                    'value_used': cfg['mesh']['pad_low_ft'],
                    'legacy_value': 0.0,
                    'owner': 'task B2',
                    'why': ('the legacy domain leaves 1797 ft below gauge 10 '
                            '(MD 14297), inside the region the house rules call '
                            'unconverged; >= 5000 ft is the converged pad')},
                'physics.D_baseline_ft2_s': {
                    'value_used': cfg['physics']['D_baseline_ft2_s'],
                    'legacy_value': 140.0,
                    'recalibrated_absolute_norm': 1150.0,
                    'recalibrated_normalised_norm': 550.0,
                    'owner': 'the baseline-D task',
                    'why': '101:93 hard-codes 140 with no stated provenance'},
            },
            'legacy_behaviours_reproduced': {
                'mesh_refinement': cfg['mesh']['refinement'],
                'source_crop_rebase': cfg['source']['crop_rebase'],
                'phase_boundary_mode': cfg['phase_boundaries']['mode'],
                'barrier_half_width_ft': cfg['barrier']['w_ft'],
            },
        }
        for ratio in sorted(chain_by_ratio):
            for phase, c in chain_by_ratio[ratio].items():
                if phase != 'phase3' and ratio != sorted(chain_by_ratio)[0]:
                    continue
                key = phase if phase != 'phase3' else f"phase3@ratio={ratio:g}"
                results['phases'][key] = {
                    'window_abs': c['window_abs'],
                    'source': {k: v for k, v in c['src'].items()
                               if k not in ('taxis_s', 'values_psi')},
                    'initial_condition': c['initial_condition'],
                    'solver': c['extra'],
                    'barrier_report': (None if c['barrier_report'] is None
                                       else {k: v for k, v
                                             in c['barrier_report'].items()
                                             if k != 'barriers'}),
                    'trace_meta': c['trace_meta'],
                    'gauge_peak_psi': {
                        f"g{g}": float(np.max(np.abs(
                            c['traces'][phase][:, j] - c['traces'][phase][0, j])))
                        for j, g in enumerate(gauges)},
                }

        # ---- manifest groups ----------------------------------------------
        ref_ratio = sorted(chain_by_ratio)[0]
        srcs = []
        k = 0
        for phase in ('phase1', 'phase2', 'phase3'):
            c = chain_by_ratio[ref_ratio][phase]
            s = c['src']
            drv = rm.driver_record(
                kind='gauge_series',
                baseline_removal=cfg['source']['baseline_removal'],
                value_units=cfg['source']['value_units'],
                series_path=rd.repo_path(s['series_path']),
                gauge_number=s['gauge'], gauge_md_ft=s['md_ft'],
                taxis=s['taxis_s'], values=s['values_psi'],
                time_start=c['window_abs'][0], time_end=c['window_abs'][1])
            for j, i in enumerate(s['source_idx']):
                srcs.append(rm.source_record(
                    x, md_requested_ft=s['frac_hit_mds_ft'][j], mesh_idx=i,
                    driver=drv,
                    label=f"{phase}:stage{s['frac_hit_stage']}_frachit{j}",
                    excluded_from_misfit=True, index_in_source_list=k))
                k += 1
        source_group = rm.source_protocol(
            application=cfg['source']['application'],
            solver_class=chain_by_ratio[ref_ratio]['phase1']['extra']['solver'],
            placement_rule=cfg['source']['placement_rule'],
            sources=srcs, targets={
                'gauges': [int(g) for g in gauges],
                'md_ft': [md_table.md_of(int(g)) for g in gauges],
                'role': ('observation points; no misfit is minimised in this '
                         'run - it is a forward reproduction')},
            time_level='n' if float(cfg['physics']['theta']) == 1.0 else 'n+1',
            phase_chaining={
                'order': ['phase1', 'phase2', 'phase3'],
                'rule': ('each phase starts from the previous phase FINAL '
                         'spatial profile, held in memory; t0 = 0 in every '
                         'phase, as 101 does'),
                'legacy_reader': ("101 read it as prev_result['data'][-1, :], "
                                  "valid only for the pre-2025-02-11 layout; "
                                  "rev2_layout.final_profile is the correct "
                                  "reader for both layouts"),
                'barrier_source_stage': 7, 'barrier_applies_in': 'phase3'},
            boundary_conditions=cfg['source']['boundary_conditions'])

        time_records = []
        for ratio in sorted(chain_by_ratio):
            for phase in ('phase1', 'phase2', 'phase3'):
                if phase != 'phase3' and ratio != ref_ratio:
                    continue
                c = chain_by_ratio[ratio][phase]
                lbl = phase if phase != 'phase3' else f"phase3@ratio={ratio:g}"
                time_records.append(rm.time_record(
                    c['taxis'], label=lbl, **c['time_record_kw']))

        barriers = rm.NONE_DECLARED
        blist = []
        for ratio in sorted(chain_by_ratio):
            c = chain_by_ratio[ratio]['phase3']
            if c['barrier_report'] is None:
                continue
            D0 = float(cfg['physics']['D_baseline_ft2_s'])
            for b in c['barrier_report']['barriers']:
                mask = np.zeros(len(x), dtype=bool)
                mask[b['i0']:b['i1'] + 1] = True
                blist.append(rm.barrier_record(
                    x, mask, label=f"stage7_frachit_MD{b['md_ft']:.2f}"
                                   f"@ratio={ratio:g}",
                    centre_md_ft=b['md_ft'],
                    w_requested_ft=float(cfg['barrier']['w_ft']),
                    ratio=ratio, d_baseline=D0, report=b))
        if blist:
            barriers = blist

        d_ref = chain_by_ratio[ref_ratio]['phase3']['d_profile']
        numerics = rm.numerics(
            time=time_records,
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
                'baseline_D_ft2_s': float(cfg['physics']['D_baseline_ft2_s']),
                'profile_family': 'uniform_plus_barriers',
                'param_names': ['D_baseline', 'ratio', 'w'],
                'params': [float(cfg['physics']['D_baseline_ft2_s']),
                           [float(r) for r in cfg['barrier']['ratios']],
                           float(cfg['barrier']['w_ft'])],
                'D_min': float(np.min(d_ref)), 'D_max': float(np.max(d_ref)),
                'D_sha256': rm.sha256_array(d_ref),
                'profile_anchor': 'physical_md',
                'note': ('D_min/D_max/D_sha256 describe the phase-3 profile at '
                         f'ratio {ref_ratio:g}; phases 1-2 are uniform')},
            barriers=barriers,
            leakage=rm.NONE_DECLARED
                    if float(cfg['physics']['lambda_leak']) == 0.0
                    else {'lambda_leak_s^-1': float(cfg['physics']['lambda_leak']),
                          'p0_psi': float(cfg['physics']['p0_psi'])},
            kernel={'name': chain_by_ratio[ref_ratio]['phase1']['extra']['solver'],
                    'banded': True,
                    'equivalence_reference':
                        'output/rev2_20260901/A4/selftest_output.txt',
                    'note': ('rev2_core at theta=1 / harmonic / lambda=0 is '
                             'bitwise identical to the verified R1 kernel, which '
                             'is bit-equivalent to fibeRIS')},
            rng=rm.NONE_DECLARED,
            parallel={'mode': 'single_process',
                      'why': 'each chain is ~1-40 s; no pool is needed'},
            amplification=rc.amplification_factor(
                x, d_ref, float(cfg['time'].get('max_dt_s', 30.0)),
                float(cfg['physics']['theta']),
                interface_avg=cfg['physics']['interface_avg']))

        inputs = [(rd.repo_path(rd.SWELL_GAUGE_TEMPLATE.format(n=g)),
                   'gauge_series', f'gauge{g}_swell') for g in (6, 7)]
        inputs += [(rd.repo_path(rd.SWELL_FRAC_HIT_TEMPLATE.format(stage=s)),
                    'geometry', f'frac_hit_stage_{s}') for s in (7, 8)]
        inputs += [(rd.repo_path(rd.SWELL_GAUGE_MD_NPZ), 'geometry',
                    'gauge_md_swell')]
        for s in (7, 8):
            for k2, fn in rd.PUMPING_CURVE_FILES.items():
                if k2 != 'slurry_rate':
                    continue
                inputs.append((rd.repo_path(
                    rd.PUMPING_DIR_TEMPLATE.format(stage=s), fn), 'pumping',
                    f'stage{s}_slurry_rate'))
        if cfg.get('compare_to_archive'):
            adir = root_cfg['archive']['dir']
            for fn in ([root_cfg['archive']['phase1'],
                        root_cfg['archive']['phase2']]
                       + list(root_cfg['archive']['phase3_by_ratio'].values())):
                inputs.append((os.path.join(adir, fn), 'prior_run_output',
                               f'archive:{fn}'))

        R.declare_inputs(inputs)
        for p, role, note in written:
            R.declare_output(p, role=role, note=note,
                             dpi=(int(cfg['outputs']['figure_dpi'])
                                  if role == 'figure_png' else None))
        R.set_source(source_group)
        R.set_numerics(numerics)
        R.set_results(results)
        R.note(f"study '{name}': " + json.dumps(
            results['legacy_behaviours_reproduced'], sort_keys=True))
        R.note("output/0211_simulation_MULTIstage is read-only; it is opened "
               "through rev2_layout.load_panel and never written.")
        R.note(SELECT_TIME_TRAP)
        R.note(
            "DISCREPANCY 'duplicate_source_mesh_idx' is a FALSE POSITIVE here "
            "and must not be read as a defect in this run. rev2_manifest."
            "source_protocol assumes one solve, so it sees the 18 declared "
            "source nodes of a three-phase chain as one list; phases 1 and 2 "
            "legitimately drive the SAME six stage-7 nodes. Within each "
            "individual solve the six indices are unique - solve_phase asserts "
            "it before calling the kernel, and results.phases[*].source records "
            "the per-phase index list. Reported to the module owner.")

    return {'name': name, 'outdir': outdir, 'mesh': mesh_rec,
            'results': results,
            'traces': {r: {p: chain_by_ratio[r][p]['traces'][p]
                           for p in ('phase1', 'phase2', 'phase3')}
                       for r in chain_by_ratio},
            'taxes': {r: {p: chain_by_ratio[r][p]['taxis']
                          for p in ('phase1', 'phase2', 'phase3')}
                      for r in chain_by_ratio},
            'gauges': [int(g) for g in gauges]}


# ---------------------------------------------------------------------------
# cross-study comparison
# ---------------------------------------------------------------------------

def cross_study(summaries, reference):
    """Gauge-trace differences between studies, interpolated onto one axis."""
    if reference not in summaries:
        return {'status': f'reference study {reference} not run'}
    ref = summaries[reference]
    out = {
        'reference': reference,
        'metric': ('per-gauge comparison on the phase-3 window. Both traces are '
                   're-based to their own first sample and the study trace is '
                   'linearly interpolated onto the reference taxis.'),
        'read_max_abs_diff_with_care': (
            'max_abs_diff_psi is NOT an amplitude error on this problem. Every '
            'phase-3 trace has a near-vertical injection front, and any study '
            'that shifts the time base - a different adaptive dt sequence, a '
            'fixed 1 s step against 30 s adaptive steps, or a shifted phase '
            'window - turns a small time offset into a large instantaneous '
            'difference on that front. Read peak_change_pct and end_diff_psi '
            'for amplitude, and t_at_max_diff_s to see whether the maximum sits '
            'on the front.'),
        'studies': {}}
    for name, s in summaries.items():
        if name == reference:
            continue
        rows = {}
        for ratio in sorted(set(s['traces']) & set(ref['traces'])):
            ta_ref = np.asarray(ref['taxes'][ratio]['phase3'], float)
            tr_ref = np.asarray(ref['traces'][ratio]['phase3'], float)
            ta = np.asarray(s['taxes'][ratio]['phase3'], float)
            tr = np.asarray(s['traces'][ratio]['phase3'], float)
            per = {}
            for j, g in enumerate(ref['gauges']):
                a = tr_ref[:, j] - tr_ref[0, j]
                b = np.interp(ta_ref, ta, tr[:, j] - tr[0, j])
                pa, pb = np.max(np.abs(a)), np.max(np.abs(b))
                k = int(np.argmax(np.abs(b - a)))
                per[f"g{g}"] = {
                    'max_abs_diff_psi': float(np.max(np.abs(b - a))),
                    't_at_max_diff_s': float(ta_ref[k]),
                    'rmse_psi': float(np.sqrt(np.mean((b - a) ** 2))),
                    'end_diff_psi': float(b[-1] - a[-1]),
                    'peak_ref_psi': float(pa), 'peak_study_psi': float(pb),
                    'peak_change_pct': (float(100.0 * (pb - pa) / pa)
                                        if pa > 0 else None)}
            rows[f"ratio={ratio:g}"] = per
        out['studies'][name] = rows
    return out


# ---------------------------------------------------------------------------
# repro record
# ---------------------------------------------------------------------------

def _first_phase3_barrier(summary):
    for k, v in summary['results']['phases'].items():
        if k.startswith('phase3') and v.get('barrier_report'):
            return k, v['barrier_report']
    return None, None


def headline_findings(summaries, cross):
    """The three numbers this task exists to produce, assembled from the runs."""
    out = {}

    # 1. how closely the archive is reproduced
    for name, s in summaries.items():
        ac = s['results'].get('archive_comparison')
        if not isinstance(ac, dict):
            continue
        worst = 0.0
        rows = [ac['phase1'], ac['phase2']] + list(ac['phase3'].values())
        for r in rows:
            if r.get('status') == 'COMPARED':
                worst = max(worst, r['field_max_abs_diff_psi'])
        out['archive_reproduction'] = {
            'study': name,
            'n_panels_compared': sum(1 for r in rows
                                     if r.get('status') == 'COMPARED'),
            'worst_max_abs_diff_psi': worst,
            'interpretation': ('round-off between the dense LAPACK solve fibeRIS '
                               'used and the banded solve rev2_core uses, '
                               'accumulated over the run; no model difference '
                               'survives at this level'),
            'phase3_test_identification': ac.get(
                'phase3_ratio_identification', {}).get('phase3_test.npz'),
        }

    # 2. barrier width vs mesh, index-defined against physically defined
    widths = {}
    for name, s in summaries.items():
        key, b = _first_phase3_barrier(s)
        if b is None:
            continue
        widths[name] = {
            'phase3_key': key,
            'mesh_mode': s['mesh']['mode'],
            'mesh_dx_min_ft': s['mesh']['realised_dx_ft']['min'],
            'w_requested_ft': b['w_requested_ft'],
            'realised_full_width_ft': b['realised_full_width_ft'],
            'n_nodes_captured': b['n_nodes_captured'],
            'total_equivalent_width_ft': b['total_equivalent_width_ft'],
            'excess_resistance_s_per_ft': b['excess_resistance_s_per_ft'],
            'n_fallback_to_nearest_node': b['n_fallback'],
        }
    out['barrier_width_vs_mesh'] = {
        'per_study': widths,
        'question': ('does the choice to KEEP or REPLACE the buggy '
                     'mesh_utils.refine_mesh change the answer?'),
        'answer': ('only while the barrier is defined by mesh INDEX. A '
                   'single-node barrier realises whatever the local control '
                   'volume is, so the same nominal reduction ratio is a '
                   'different physical barrier on every mesh. Giving the '
                   'barrier a physical half-width w makes the three meshes '
                   'agree - compare total_equivalent_width_ft and '
                   'excess_resistance_s_per_ft across the w>0 studies.'),
    }
    out['pressure_consequence'] = {
        'source': 'cross_study.studies[*][ratio=...] peak_change_pct',
        'note': ('read at the SHIELDED gauges (g5 MD 15599, g6 MD 15344 - the '
                 'only ones the stage-7 barrier row separates from the stage-8 '
                 'sources); the unshielded gauges are insensitive to the '
                 'barrier by construction and their ~0% change is not evidence '
                 'of anything'),
        'studies': {k: {rk: {g: v['peak_change_pct']
                             for g, v in per.items() if g in ('g5', 'g6')}
                        for rk, per in rows.items()}
                    for k, rows in cross.get('studies', {}).items()},
    }
    return out


def repro_record(root_cfg, summaries, cross):
    adir = root_cfg['archive']['dir']
    files = sorted(f for f in os.listdir(adir) if f.endswith('.npz'))
    layout_rows = rl.layout_report([os.path.join(adir, f) for f in files])
    rec = {
        'what': ('A5 repro record: the on-disk layout of every file in '
                 f'{adir}, what each legacy consumer does with it, and how '
                 'closely the rev2 chain reproduces the archive.'),
        'generated_utc': datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        'archive_dir': adir,
        'archive_is_read_only': True,
        'layout_history': {
            'cause': ("fibeRIS commit aabffe2 'Bug fixed in pds packing "
                      "function', dated 2025-02-11, changed "
                      "PDS1D.pack_result from data=self.snapshot (n_t, n_x) to "
                      "data=self.snapshot.T (n_x, n_t)."),
            'shape_validation_added': ("commit a78bf16, 2025-06-03; "
                                       "Data2D.load_npz (core2D.py:239-244) "
                                       "now hard-rejects all six old-layout "
                                       "files"),
            'year_correction': ('the task package says 2026 for both dates; '
                                'the commits are 2025 (HOUSE_RULES CORRECTION 2)'),
            'mtime_is_not_evidence': ('three old-layout files (phase3_0.001, '
                                      '0.0001, 1e-05) were written 27-96 min '
                                      'AFTER the layout-flipping commit because '
                                      'the running interpreter still held the '
                                      'old code. Only CONTENT is authoritative; '
                                      'rev2_layout decides on len(taxis)/'
                                      'len(daxis) against data.shape'),
        },
        'file_layouts': [
            {'file': r['basename'], 'sha256': r.get('sha256'),
             'mtime_utc': r.get('mtime_utc'), 'data_shape': r.get('data_shape'),
             'n_t': r.get('n_t'), 'n_x': r.get('n_x'),
             'layout': r.get('layout'),
             'detection_rule': (r.get('evidence') or {}).get('rule'),
             'error': r.get('error')}
            for r in layout_rows],
        'legacy_consumers': LEGACY_CONSUMERS,
        'select_time_trap': SELECT_TIME_TRAP,
        'crop_convention_finding': {
            'claim': ('fibeRIS Data1D.crop rebased the cropped taxis to the '
                      'REQUESTED crop start in Feb 2025 and rebases it to the '
                      'FIRST IN-WINDOW SAMPLE now (core1D.py:158-160).'),
            'evidence': ('pack_result stores source[0].start_time (pds.py:411-415). '
                         'The archived phase npz files carry 04:24:28 / 08:26:03 '
                         '/ 10:59:44 with no sub-second part, whereas the first '
                         'in-window gauge samples are at 04:24:28.981 / '
                         '08:26:03.976 / 10:59:45.206.'),
            'consequence': ('the archived source axes lead the current ones by '
                            '0.981 / 0.976 / 1.206 s. Reproducing phase 1 with '
                            "today's convention leaves a 3.12 psi residual "
                            'against the archive; with the 2025 convention the '
                            'residual is 1.6e-5 psi.'),
            'handled_by': "config source.crop_rebase",
        },
        'reproduction': {name: s['results'].get('archive_comparison')
                         for name, s in summaries.items()
                         if isinstance(s['results'].get('archive_comparison'),
                                       dict)},
        'cross_study': cross,
        'headline_findings': headline_findings(summaries, cross),
        'studies': {name: {'outdir': s['outdir'],
                           'mesh': {k: v for k, v in s['mesh'].items()
                                    if k != 'calls'},
                           'open_parameters':
                               s['results']['open_parameters_NOT_settled_here'],
                           'legacy_behaviours':
                               s['results']['legacy_behaviours_reproduced']}
                    for name, s in summaries.items()},
    }
    return rec


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default='configs/rev2/a5_two_stage_chain.json')
    ap.add_argument('--study', action='append', default=None,
                    help='run only this study (repeatable); default: all '
                         'enabled studies in the config')
    ap.add_argument('--outdir', default=None,
                    help='override output_root (must be new)')
    args = ap.parse_args(argv)

    if not os.path.isdir('scripts') or not os.path.isdir('fibeRIS'):
        raise SystemExit(f"run with CWD = repo root; CWD is {os.getcwd()}")

    root_cfg = load_config(args.config)
    out_root = args.outdir or root_cfg['output_root']
    os.makedirs(out_root, exist_ok=True)

    wanted = set(args.study) if args.study else None
    summaries = {}
    t_all = time.time()
    for st in root_cfg['studies']:
        name = st['name']
        if wanted is not None and name not in wanted:
            continue
        if wanted is None and not st.get('enabled', True):
            continue
        cfg = deep_merge(root_cfg['base'], st.get('overrides'))
        cfg['compare_to_archive'] = bool(
            st.get('compare_to_archive', cfg.get('compare_to_archive', False)))
        cfg['_study'] = {'name': name, 'what': st.get('_what')}
        outdir = os.path.join(out_root, name)
        t0 = time.time()
        print(f"[A5] study {name} -> {outdir}", flush=True)
        summaries[name] = run_study(name, cfg, root_cfg, outdir, args.config)
        print(f"[A5]   done in {time.time() - t0:.1f} s", flush=True)

    cross = cross_study(summaries, root_cfg.get('cross_study_reference'))
    rec = repro_record(root_cfg, summaries, cross)
    rec['wall_seconds_total'] = time.time() - t_all
    rec_path = os.path.join(out_root, 'a5_repro_record.json')
    rm.assert_absent([rec_path])
    with open(rec_path, 'w') as fh:
        json.dump(rm._jsonify(rec, max_array_len=64)[0], fh, indent=2,
                  sort_keys=True, ensure_ascii=False)
    print(f"[A5] repro record -> {rec_path}")
    print(f"[A5] total {time.time() - t_all:.1f} s")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
