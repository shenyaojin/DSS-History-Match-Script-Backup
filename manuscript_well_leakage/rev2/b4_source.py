#!/usr/bin/env python3
"""B4 - settle the source-application protocol on the two-stage, with-barrier case.

    python3 scripts/manuscript_well_leakage/rev2/b4_source.py \
        --config configs/rev2/b4_source.json                    # the nine chains
    python3 scripts/manuscript_well_leakage/rev2/b4_source.py \
        --config configs/rev2/b4_source.json --identity-check   # a5 equivalence
    python3 scripts/manuscript_well_leakage/rev2/b4_source.py \
        --config configs/rev2/b4_source.json --report           # solve-free rollup

Run with CWD = repo root. Writes ONLY into `output/rev2_20260901/B4/`.

The question
------------
Three places disagree about where the Dirichlet pressure datum is applied:

  TEXT   "pressure is applied at the frac hits";
  R1/R2  the calibration applies it at gauge 1 (MD 16645), 19-52 ft from the
         stage-1 frac hits (MD 16663.8 / 16688.8 / 16696.9);
  CODE   101_fiberis_matching.py - and therefore A5 and E1 - applies it at the
         six stage-7 frac-hit nodes driven by gauge 6, then at the six stage-8
         frac-hit nodes driven by gauge 7.

Two answers already exist and are NOT recomputed here:
  * on the STAGE-1 CALIBRATION, moving the source to the frac-hit centroid
    changes the misfit by ~1 % (82.33 -> 81.24 uniform, 11.87 -> 11.31 two_zone,
    R2), so the near-source high D is not an artifact of boundary placement;
  * the fitted UNIFORM D depends strongly on how far DOWN-HOLE the boundary sits
    (1135 -> 254 as it moves g1 -> g3, D2) - the path-average effect.
Those are different statements and this file keeps them apart.

What is new here is the same comparison on the case the manuscript FIGURE uses:
the two-stage chain with the stage-7 barrier row in phase 3.

How the chain is obtained
-------------------------
It is NOT re-implemented. This file imports `a5_two_stage_chain` and calls its
`build_chain_mesh` / `phase_windows` / `load_source_series` / `solve_phase`,
exactly as `e1_fig6.py` does. The only thing added is the selection of the
source mesh nodes. `--identity-check` runs `a5.run_chain()` itself on the same
config and requires the target-gauge traces of all three phases to be BITWISE
identical, so "only the placement changed" is a measured statement rather than
an assurance.

Metric conventions (inherited, not invented)
--------------------------------------------
  * `rmse_delta_psi` - simulated and measured each referenced to their own first
    in-window sample. This is the placement test. `rmse_abs_psi` - absolute psi,
    what the manuscript's ax4 panel draws; it carries the accumulated offset of
    the whole chain. Both are reported and both are named (E1 fit_check).
  * amplitude ratio = max(simulated dP) / max(measured dP); arrival at ONE
    threshold 0.1 x max(measured dP) applied to both traces on the measured time
    axis; arrival_err_s = t_sim - t_obs
    (`r1_calibration_core.evaluate_profile` / `arrival_time`).
  * aggregation is the GAUGE-MEAN RMSE, never sample-pooled (C1).
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
for _p in (_HERE,
           os.path.join(_ROOT, 'scripts', 'manuscript_well_leakage',
                        'baseline_calibration'),
           os.path.join(_ROOT, 'fibeRIS', 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rev2_core as rc                   # noqa: E402
import rev2_data as rd                   # noqa: E402
import rev2_manifest as rm               # noqa: E402
import a5_two_stage_chain as a5          # noqa: E402  the chain, imported
import r1_calibration_core as r1c        # noqa: E402  arrival_time convention

STUDY_ID = "B4_source_protocol"
TASK_ID = "B4"
PHASES = ('phase1', 'phase2', 'phase3')
PLACEMENTS = ('driving_gauge_md', 'frac_hit_nodes', 'frac_hit_centroid')

PLACEMENT_RULE = {
    'driving_gauge_md': 'nearest_mesh_node_to_the_DRIVING_GAUGE_MD',
    'frac_hit_nodes': 'nearest_mesh_node_to_each_frac_hit_MD',
    'frac_hit_centroid': 'nearest_mesh_node_to_the_frac_hit_CENTROID_MD',
}

PLACEMENT_WHAT = {
    'driving_gauge_md': ("one Dirichlet node at the MD of the gauge whose series "
                         "supplies the datum; what the R1/R2 calibration does"),
    'frac_hit_nodes': ("one Dirichlet node at each of the driving stage's six "
                       "frac hits; what 101/A5/E1 do and what the text says"),
    'frac_hit_centroid': ("one Dirichlet node at the arithmetic centroid of the "
                          "driving stage's frac hits; the R2 frac_centroid "
                          "variant carried onto this case"),
}


def rtag(ratio):
    return 'uniform' if float(ratio) >= 1.0 else f"r{float(ratio):g}".replace(
        '.', 'p').replace('-', 'm')


def _rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, dtype=float) ** 2)))


# ---------------------------------------------------------------------------
# source placement - the ONLY thing this file adds to the A5 chain
# ---------------------------------------------------------------------------

def make_source(cfg, x, wins, phase):
    """A5's source series, plus the mesh nodes the configured placement selects."""
    spec = [p for p in cfg['phases'] if p['name'] == phase][0]
    t0, t1 = wins[phase]
    src = a5.load_source_series(spec['source_gauge'], t0, t1, cfg['source'])
    stage = int(cfg['source_stage_by_phase'][phase])
    hits = np.asarray(rd.load_frac_hits(stage, unique=False, sort=False),
                      dtype=float)
    placement = cfg['source']['placement']
    if placement not in PLACEMENTS:
        raise ValueError(f"unknown source.placement {placement!r}; "
                         f"expected one of {PLACEMENTS}")

    if placement == 'frac_hit_nodes':
        req = [float(h) for h in hits]
        labels = [f"{phase}:stage{stage}_frachit{j}" for j in range(len(req))]
    elif placement == 'driving_gauge_md':
        req = [float(src['md_ft'])]
        labels = [f"{phase}:gauge{src['gauge']}_md"]
    else:  # frac_hit_centroid
        req = [float(np.mean(hits))]
        labels = [f"{phase}:stage{stage}_frachit_centroid"]

    idx = [int(np.argmin(np.abs(x - float(m)))) for m in req]
    if len(set(idx)) != len(idx):
        dup = sorted({i for i in idx if idx.count(i) > 1})
        raise ValueError(
            f"{phase}/{placement}: requested MDs {req} snap to duplicate mesh "
            f"indices {dup} on dx = {float(np.median(np.diff(x))):g} ft; the run "
            f"would apply fewer independent Dirichlet nodes than it declares")

    src.update(
        placement=placement,
        placement_rule=PLACEMENT_RULE[placement],
        frac_hit_stage=stage,
        frac_hit_mds_ft=[float(v) for v in hits],
        frac_hit_centroid_md_ft=float(np.mean(hits)),
        frac_hit_span_md_ft=[float(np.min(hits)), float(np.max(hits))],
        source_md_requested_ft=req,
        source_labels=labels,
        source_idx=idx,
        source_md_ft=[float(x[i]) for i in idx],
        snap_error_ft=[float(x[i] - float(m)) for i, m in zip(idx, req)],
        n_dirichlet_nodes=len(idx),
        source_centroid_md_ft=float(np.mean([x[i] for i in idx])),
        offset_gauge_to_source_centroid_ft=float(
            np.mean([x[i] for i in idx]) - float(src['md_ft'])),
        window_abs=[t0.isoformat(), t1.isoformat()],
        t_window_s=(t1 - t0).total_seconds(),
    )
    return src


# ---------------------------------------------------------------------------
# measured data + metrics
# ---------------------------------------------------------------------------

def load_field_gauges(gauges, wins):
    """Measured gauge series over the whole two-stage window, absolute psi."""
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


def phase_metrics(gauges, field, t0_abs, taxis, traces, thr_frac, src_md_nodes,
                  src_idx, gidx, phase, md_table):
    """Per-target-gauge model-data metrics for one solved phase.

    Both series are put on the MEASURED time axis, clipped to the simulated
    window. Every gauge that is itself a Dirichlet node in this phase is flagged:
    its misfit is zero by construction and must never be read as a fit.
    """
    per = {}
    for j, g in enumerate(gauges):
        f = field[int(g)]
        tf = f['taxis_s'] + (f['t0_abs'] - t0_abs).total_seconds()
        pf = f['psi']
        keep = (tf >= float(taxis[0])) & (tf <= float(taxis[-1]))
        if int(keep.sum()) < 10:
            per[f"g{g}"] = {'status': 'TOO_FEW_SAMPLES',
                            'n_samples': int(keep.sum())}
            continue
        tf, pf = tf[keep], pf[keep]
        sim = np.interp(tf, taxis, traces[:, j])
        sim0 = float(np.interp(float(tf[0]), taxis, traces[:, j]))
        dsim = sim - sim0
        dobs = pf - float(pf[0])

        obs_max = float(np.max(dobs))
        sim_max = float(np.max(dsim))
        obs_ptp = float(np.ptp(dobs))
        sim_ptp = float(np.ptp(dsim))
        thr = thr_frac * obs_max
        if obs_max > 0.0:
            t_obs = r1c.arrival_time(tf, dobs, thr)
            t_sim = r1c.arrival_time(tf, dsim, thr)
            amp = sim_max / obs_max
        else:
            t_obs = t_sim = np.nan
            amp = np.nan
        is_src = int(gidx[j]) in set(int(i) for i in src_idx)
        per[f"g{g}"] = {
            'status': 'OK',
            'md_ft': float(md_table.md_of(int(g))),
            'mesh_md_ft': float(field[int(g)]['md_ft']),
            'n_samples': int(tf.size),
            'distance_to_nearest_source_node_ft': float(
                np.min(np.abs(np.asarray(src_md_nodes, float)
                              - md_table.md_of(int(g))))),
            'is_dirichlet_node_in_this_phase': bool(is_src),
            'rmse_abs_psi': _rms(sim - pf),
            'rmse_delta_psi': _rms(dsim - dobs),
            'bias_delta_psi': float(np.mean(dsim - dobs)),
            'obs_max_dp_psi': obs_max,
            'sim_max_dp_psi': sim_max,
            'obs_ptp_dp_psi': obs_ptp,
            'sim_ptp_dp_psi': sim_ptp,
            'amplitude_ratio': float(amp),
            'ptp_ratio': float(sim_ptp / obs_ptp) if obs_ptp != 0 else np.nan,
            'arrival_obs_s': float(t_obs),
            'arrival_sim_s': float(t_sim),
            'arrival_err_s': float(t_sim - t_obs),
        }
    ok = [v for v in per.values() if v.get('status') == 'OK']
    free = [v for v in ok if not v['is_dirichlet_node_in_this_phase']]
    agg = {
        'n_gauges_scored': len(ok),
        'n_gauges_free': len(free),
        'n_gauges_pinned_as_dirichlet': len(ok) - len(free),
        'gauge_mean_rmse_delta_psi': float(np.mean(
            [v['rmse_delta_psi'] for v in ok])) if ok else None,
        'gauge_mean_rmse_abs_psi': float(np.mean(
            [v['rmse_abs_psi'] for v in ok])) if ok else None,
        'gauge_mean_rmse_delta_psi_FREE_ONLY': float(np.mean(
            [v['rmse_delta_psi'] for v in free])) if free else None,
        'gauge_mean_rmse_abs_psi_FREE_ONLY': float(np.mean(
            [v['rmse_abs_psi'] for v in free])) if free else None,
        'mean_amplitude_ratio': float(np.nanmean(
            [v['amplitude_ratio'] for v in ok])) if ok else None,
        'mean_abs_arrival_err_s': float(np.nanmean(
            [abs(v['arrival_err_s']) for v in ok])) if ok else None,
        'aggregation': 'gauge-mean RMSE, not sample-pooled (C1)',
        'phase': phase,
    }
    return {'per_gauge': per, 'aggregate': agg}


# ---------------------------------------------------------------------------
# one study = one placement at one (D, pad) arm
# ---------------------------------------------------------------------------

def run_study(name, cfg, root_cfg, outdir, config_path):
    t_study = time.time()
    os.makedirs(outdir, exist_ok=True)
    manifest_path = os.path.join(outdir, 'manifest.json')
    rm.assert_absent([manifest_path])

    md_table = rd.load_gauge_md_table()
    x, mesh_rec = a5.build_chain_mesh(cfg['mesh'])
    wins, win_rec = a5.phase_windows(cfg['phase_boundaries'])
    gauges = [int(g) for g in cfg['targets']['gauges']]
    gidx = [int(np.argmin(np.abs(x - md_table.md_of(g)))) for g in gauges]

    D0 = float(cfg['physics']['D_baseline_ft2_s'])
    base = np.full(len(x), D0, dtype=float)
    bcfg = cfg['barrier']
    ratios = [float(r) for r in bcfg['ratios']]
    r_head = float(bcfg['headline_ratio'])
    r_uni = float(bcfg['uniform_reference_ratio'])
    hits7 = rd.load_frac_hits(7, unique=False, sort=False)
    tcfg, pcfg = cfg['time'], cfg['physics']
    thr_frac = float(cfg['metrics']['arrival_threshold_frac'])
    placement = cfg['source']['placement']
    written = []

    with rm.RunRecorder(manifest_path, study_id=STUDY_ID, task_id=TASK_ID,
                        config=cfg, config_path=config_path, run_label=name,
                        require_modules=('rev2_core', 'rev2_data',
                                         'rev2_manifest', 'a5_two_stage_chain',
                                         'r1_calibration_core')) as R:

        srcs = {p: make_source(cfg, x, wins, p) for p in PHASES}
        s1, s2, s3 = srcs['phase1'], srcs['phase2'], srcs['phase3']

        # ---- phases 1 and 2: no barrier, so ratio-independent ---------------
        init1 = np.full(len(x), float(s1['values_psi'][0]))
        _t = time.time()
        ta1, f1, trec1, ex1 = a5.solve_phase(x, base, s1, init1, tcfg, pcfg)
        ex1['wall_s'] = time.time() - _t
        tr1 = np.ascontiguousarray(f1[:, gidx])
        end1 = f1[-1].copy()
        del f1
        print(f"[B4]   {name} phase1 done ({ex1['wall_s']:.1f} s)", flush=True)

        _t = time.time()
        ta2, f2, trec2, ex2 = a5.solve_phase(x, base, s2, end1, tcfg, pcfg)
        ex2['wall_s'] = time.time() - _t
        tr2 = np.ascontiguousarray(f2[:, gidx])
        init3 = f2[-1].copy()
        del f2
        print(f"[B4]   {name} phase2 done ({ex2['wall_s']:.1f} s)", flush=True)

        # ---- phase 3, one solve per barrier ratio ---------------------------
        tr3, breports, trec3, ex3 = {}, {}, {}, {}
        ta3 = None
        dprof_head = None
        for ratio in ratios:
            if ratio >= 1.0:
                dprof, brep = base.copy(), None
            else:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    dprof, brep = rc.build_barrier_profile(
                        x, base, hits7, float(bcfg['w_ft']), float(ratio),
                        ratio_reference=bcfg['ratio_reference'],
                        combine=bcfg['combine'], on_empty=bcfg['on_empty'],
                        on_outside=bcfg['on_outside'], return_report=True)
                # THE REPORT IS THE AUTHORITY, NOT THE WARNING.
                if brep['n_fallback'] != 0:
                    raise RuntimeError(
                        f"ratio {ratio:g}: {brep['n_fallback']} barrier(s) fell "
                        f"back to the nearest node at w = {bcfg['w_ft']} ft; the "
                        f"realised width would not be the requested one. "
                        f"{brep['fallback_messages'][:1]}")
                brep['n_warnings_raised'] = len(caught)
                brep['barrier_mds_ft'] = [float(v) for v in hits7]
            _t = time.time()
            t_a, f3, tr_, ex_ = a5.solve_phase(x, dprof, s3, init3, tcfg, pcfg)
            ex_['wall_s'] = time.time() - _t
            ta3 = t_a
            trec3[ratio], ex3[ratio] = tr_, ex_
            tr3[ratio] = np.ascontiguousarray(f3[:, gidx])
            breports[ratio] = brep
            if ratio == r_head:
                dprof_head = dprof.copy()
            del f3
            print(f"[B4]   {name} phase3 ratio {ratio:g} done "
                  f"({ex_['wall_s']:.1f} s)", flush=True)
        if dprof_head is None:
            dprof_head = base.copy()

        # ---- metrics against the measured gauges ----------------------------
        field = load_field_gauges(gauges, wins)
        metrics = {
            'phase1': phase_metrics(gauges, field, s1['t0_abs'], ta1, tr1,
                                    thr_frac, s1['source_md_ft'],
                                    s1['source_idx'], gidx, 'phase1', md_table),
            'phase2': phase_metrics(gauges, field, s2['t0_abs'], ta2, tr2,
                                    thr_frac, s2['source_md_ft'],
                                    s2['source_idx'], gidx, 'phase2', md_table),
        }
        for ratio in ratios:
            metrics[f"phase3@ratio={ratio:g}"] = phase_metrics(
                gauges, field, s3['t0_abs'], ta3, tr3[ratio], thr_frac,
                s3['source_md_ft'], s3['source_idx'], gidx, 'phase3', md_table)

        # ---- how big is the BARRIER signal in this placement? ---------------
        dp3 = {r: tr3[r] - tr3[r][0] for r in ratios}
        vs_uni = np.abs(dp3[r_head] - dp3[r_uni]).max(axis=0)
        barrier_signal = {
            'definition': ("max over the phase-3 window of |dP(headline ratio) - "
                           "dP(no barrier)| at each target gauge, dP referenced "
                           "to each run's own first phase-3 sample (E1's "
                           "vs_uniform)"),
            'headline_ratio': r_head, 'uniform_ratio': r_uni,
            'per_gauge_psi': {f"g{g}": float(vs_uni[j])
                              for j, g in enumerate(gauges)},
            'worst_gauge': f"g{gauges[int(np.argmax(vs_uni))]}",
            'worst_psi': float(vs_uni.max()),
        }

        # ---- saved arrays ---------------------------------------------------
        p = os.path.join(outdir, f"b4_traces_{name}_v1.npz")
        rm.assert_absent([p])
        pay = {'gauge_numbers': np.asarray(gauges, dtype=np.int64),
               'gauge_md_ft': np.asarray([md_table.md_of(g) for g in gauges],
                                         dtype=float),
               'gauge_mesh_idx': np.asarray(gidx, dtype=np.int64),
               'ratios': np.asarray(ratios, dtype=float),
               'mesh_md_ft': x,
               'phase1_taxis_s': ta1, 'phase1_traces_psi': tr1,
               'phase1_t0_abs': np.array(str(s1['t0_abs'])),
               'phase2_taxis_s': ta2, 'phase2_traces_psi': tr2,
               'phase2_t0_abs': np.array(str(s2['t0_abs'])),
               'phase3_taxis_s': ta3,
               'phase3_t0_abs': np.array(str(s3['t0_abs'])),
               'phase3_final_profile_psi': None}
        pay.pop('phase3_final_profile_psi')
        for r in ratios:
            pay[f"phase3_traces_psi_{rtag(r)}"] = tr3[r]
        for ph, s in (('phase1', s1), ('phase2', s2), ('phase3', s3)):
            pay[f"{ph}_source_idx"] = np.asarray(s['source_idx'], dtype=np.int64)
            pay[f"{ph}_source_md_ft"] = np.asarray(s['source_md_ft'], dtype=float)
        np.savez_compressed(p, **pay)
        written.append((p, 'arrays_npz',
                        'target-gauge traces for all three phases + the phase-3 '
                        'ratio ladder, plus the applied source nodes'))

        p = os.path.join(outdir, f"b4_field_gauges_{name}_v1.npz")
        rm.assert_absent([p])
        fp = {'gauge_numbers': np.asarray(gauges, dtype=np.int64)}
        for g in gauges:
            fp[f"g{g}_taxis_s"] = field[g]['taxis_s']
            fp[f"g{g}_psi"] = field[g]['psi']
            fp[f"g{g}_t0_abs"] = np.array(str(field[g]['t0_abs']))
        np.savez_compressed(p, **fp)
        written.append((p, 'arrays_npz',
                        'measured gauge series over the two-stage window'))

        p = os.path.join(outdir, f"b4_metrics_{name}_v1.json")
        rm.assert_absent([p])
        with open(p, 'w') as fh:
            json.dump(rm._jsonify({'study': name, 'placement': placement,
                                   'metrics': metrics,
                                   'barrier_signal': barrier_signal},
                                  max_array_len=64)[0],
                      fh, indent=2, sort_keys=True, ensure_ascii=False)
        written.append((p, 'json', 'per-gauge model-data metrics per phase'))

        fig = os.path.join(outdir, f"fig01_{name}_traces_v1.png")
        figure_study(fig, cfg, name, gauges, md_table, ta1, tr1, ta2, tr2,
                     ta3, tr3, srcs, field, ratios, r_head,
                     int(cfg['outputs']['figure_dpi']))
        written.append((fig, 'figure_png',
                        'three chained phases at the target gauges, measured '
                        'overlaid'))

        # ---- manifest -------------------------------------------------------
        results = {
            'placement': placement,
            'placement_rule': PLACEMENT_RULE[placement],
            'placement_what': PLACEMENT_WHAT[placement],
            'mesh': mesh_rec,
            'phase_boundaries': win_rec,
            'metrics': metrics,
            'barrier_signal_vs_no_barrier': barrier_signal,
            'source_geometry': {
                ph: {k: srcs[ph][k] for k in
                     ('gauge', 'md_ft', 'frac_hit_stage', 'frac_hit_mds_ft',
                      'frac_hit_centroid_md_ft', 'frac_hit_span_md_ft',
                      'source_md_requested_ft', 'source_md_ft', 'source_idx',
                      'snap_error_ft', 'n_dirichlet_nodes',
                      'source_centroid_md_ft',
                      'offset_gauge_to_source_centroid_ft')}
                for ph in PHASES},
            'phases': {
                'phase1': {'window_abs': s1['window_abs'], 'solver': ex1,
                           'time': trec1},
                'phase2': {'window_abs': s2['window_abs'], 'solver': ex2,
                           'time': trec2},
                **{f"phase3@ratio={r:g}": {'window_abs': s3['window_abs'],
                                           'solver': ex3[r], 'time': trec3[r],
                                           'barrier_report':
                                               None if breports[r] is None else
                                               {k: v for k, v
                                                in breports[r].items()
                                                if k != 'barriers'}}
                   for r in ratios},
            },
            'phases_1_2_solved_once': (
                'barrier_from_stage is null for phases 1 and 2, so they carry no '
                'barrier and are solved once per study rather than once per '
                'ratio; every phase-3 solve starts from the same stored final '
                'profile of phase 2'),
            'what_is_NOT_settled_here': {
                'D_baseline_ft2_s': float(D0),
                'pad_ft': [float(cfg['mesh']['pad_low_ft']),
                           float(cfg['mesh']['pad_high_ft'])],
                'note': ('this study varies ONLY the source placement; the D and '
                         'pad arms exist to show the placement verdict does not '
                         'depend on either'),
            },
        }

        src_groups, labels = [], []
        for ph in PHASES:
            s = srcs[ph]
            drv = rm.driver_record(
                kind='gauge_series',
                baseline_removal=cfg['source']['baseline_removal'],
                value_units=cfg['source']['value_units'],
                series_path=rd.repo_path(s['series_path']),
                gauge_number=s['gauge'], gauge_md_ft=s['md_ft'],
                taxis=s['taxis_s'], values=s['values_psi'],
                time_start=s['window_abs'][0], time_end=s['window_abs'][1])
            grp = [rm.source_record(
                x, md_requested_ft=s['source_md_requested_ft'][j], mesh_idx=i,
                driver=drv, label=s['source_labels'][j],
                excluded_from_misfit=True, index_in_source_list=j)
                for j, i in enumerate(s['source_idx'])]
            src_groups.append(grp)
            labels.append(ph)

        pinned = sorted({int(g) for ph in PHASES for j, g in enumerate(gauges)
                         if int(gidx[j]) in set(srcs[ph]['source_idx'])})
        source_group = rm.source_protocol(
            application=cfg['source']['application'],
            solver_class=ex1['solver'],
            placement_rule=PLACEMENT_RULE[placement],
            sources=src_groups, phase_labels=labels,
            targets={'gauges': gauges,
                     'md_ft': [md_table.md_of(g) for g in gauges],
                     'in_manuscript_figure':
                         [int(g) for g in cfg['targets']['in_manuscript_figure']],
                     'validation_gauge': int(cfg['targets']['validation_gauge']),
                     'target_gauges_that_are_ALSO_dirichlet_nodes': pinned,
                     'role': ('observation points for a FORWARD run; nothing is '
                              'fitted. Gauges 6 and 7 supply the phase-1/2 and '
                              'phase-3 driving series in every placement (D4); '
                              'under placement="driving_gauge_md" they are ALSO '
                              'the applied nodes, so their own misfit is zero by '
                              'construction')},
            time_level='n' if float(pcfg['theta']) == 1.0 else 'n+1',
            phase_chaining={
                'order': list(PHASES),
                'rule': ('each phase starts from the previous phase FINAL spatial '
                         'profile; t0 = 0 in every phase, as 101 does'),
                'barrier_source_stage': 7, 'barrier_applies_in': 'phase3',
                'phases_1_2_are_ratio_independent': True},
            boundary_conditions=cfg['source']['boundary_conditions'])

        time_records = [rm.time_record(ta1, label='phase1', **trec1),
                        rm.time_record(ta2, label='phase2', **trec2)]
        time_records += [rm.time_record(ta3, label=f"phase3@ratio={r:g}",
                                        **trec3[r]) for r in ratios]

        blist = []
        for r in ratios:
            brep = breports[r]
            if brep is None:
                continue
            for b in brep['barriers']:
                mask = np.zeros(len(x), dtype=bool)
                mask[b['i0']:b['i1'] + 1] = True
                blist.append(rm.barrier_record(
                    x, mask, label=f"stage7_frachit_MD{b['md_ft']:.2f}@ratio={r:g}",
                    centre_md_ft=b['md_ft'],
                    w_requested_ft=float(bcfg['w_ft']), ratio=r, d_baseline=D0,
                    report=b))

        numerics = rm.numerics(
            time=time_records,
            mesh=rm.mesh_record(
                x, dx_requested_ft=float(cfg['mesh']['dx_ft']),
                window_md_ft=(float(cfg['mesh']['md_lo_ft']),
                              float(cfg['mesh']['md_hi_ft'])),
                pad_low_ft=float(cfg['mesh']['pad_low_ft']),
                pad_high_ft=float(cfg['mesh']['pad_high_ft']),
                refinement=mesh_rec),
            interface_avg=pcfg['interface_avg'],
            boundary=cfg['source']['boundary_conditions'],
            diffusivity={
                'baseline_D_ft2_s': D0,
                'profile_family': 'uniform_plus_physical_width_barriers',
                'param_names': ['D_baseline', 'ratio', 'w_half_width_ft'],
                'params': [D0, ratios, float(bcfg['w_ft'])],
                'D_min': float(np.min(dprof_head)),
                'D_max': float(np.max(dprof_head)),
                'D_sha256': rm.sha256_array(dprof_head),
                'profile_anchor': 'physical_md',
                'note': (f'D_min/D_max/D_sha256 describe the phase-3 profile at '
                         f'the headline ratio {r_head:g}; phases 1-2 are uniform')},
            barriers=blist if blist else rm.NONE_DECLARED,
            leakage=rm.NONE_DECLARED,
            kernel={'name': ex1['solver'], 'banded': True,
                    'equivalence_reference':
                        'output/rev2_20260901/A4/selftest_output.txt',
                    'note': ('rev2_core at theta=1 / harmonic / lambda=0 is '
                             'bitwise identical to the verified R1 kernel, which '
                             'is bit-equivalent to fibeRIS')},
            rng=rm.NONE_DECLARED,
            parallel={'mode': 'single_process',
                      'why': ('each phase solve is 8-15 s and holds a ~1.8 GB '
                              'field; studies are run serially so peak memory '
                              'stays near one field')},
            amplification=rc.amplification_factor(
                x, dprof_head, float(tcfg['dt_fixed_s']), float(pcfg['theta']),
                interface_avg=pcfg['interface_avg']))

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
        for pth, role, note in written:
            R.declare_output(pth, role=role, note=note,
                             dpi=(int(cfg['outputs']['figure_dpi'])
                                  if role == 'figure_png' else None))
        R.set_source(source_group)
        R.set_numerics(numerics)
        R.set_results(results)
        R.note(f"B4 study '{name}': source placement '{placement}' "
               f"({PLACEMENT_WHAT[placement]}). D = {D0:g} ft^2/s, pad "
               f"{cfg['mesh']['pad_low_ft']:g}/{cfg['mesh']['pad_high_ft']:g} ft, "
               f"uniform dx = {cfg['mesh']['dx_ft']:g} ft, fixed dt = "
               f"{tcfg['dt_fixed_s']:g} s, barrier half-width w = "
               f"{bcfg['w_ft']:g} ft. Only the placement differs between the "
               f"studies of one arm.")
        R.note("The chain is NOT re-implemented here: build_chain_mesh, "
               "phase_windows, load_source_series and solve_phase all come from "
               "a5_two_stage_chain, as in e1_fig6.py. b4_source.py adds only the "
               "selection of the source mesh nodes. The --identity-check run "
               "asserts that placement='frac_hit_nodes' reproduces a5.run_chain "
               "bitwise at the target gauges.")
        if pinned:
            R.note(f"DEGENERACY, not a fit: target gauge(s) {pinned} are "
                   f"themselves Dirichlet nodes in at least one phase under this "
                   f"placement, so their simulated trace is the measured driving "
                   f"series and their misfit is zero by construction. The "
                   f"aggregate metrics are therefore reported twice, once over "
                   f"all scored gauges and once over the FREE gauges only.")
        R.note("Established elsewhere and deliberately not recomputed: on the "
               "STAGE-1 calibration the frac-centroid variant moves the misfit by "
               "~1% (82.33 -> 81.24 uniform, 11.87 -> 11.31 two_zone, R2/D2), and "
               "the fitted uniform D depends strongly on how far DOWN-HOLE the "
               "boundary sits (1135 -> 254, g1 -> g3, D2). Those are two separate "
               "statements and neither is re-derived here.")

    return {'name': name, 'outdir': outdir, 'placement': placement,
            'cfg': cfg, 'gauges': gauges, 'ratios': ratios,
            'r_head': r_head, 'r_uni': r_uni,
            'metrics': metrics, 'barrier_signal': barrier_signal,
            'srcs': {ph: {k: v for k, v in srcs[ph].items()
                          if k not in ('taxis_s', 'values_psi')}
                     for ph in PHASES},
            'traces': {'phase1': (ta1, tr1), 'phase2': (ta2, tr2),
                       'phase3': (ta3, {r: tr3[r] for r in ratios})},
            'wall_s': time.time() - t_study}


# ---------------------------------------------------------------------------
# per-study figure
# ---------------------------------------------------------------------------

def figure_study(path, cfg, name, gauges, md_table, ta1, tr1, ta2, tr2, ta3,
                 tr3, srcs, field, ratios, r_head, dpi):
    rm.assert_absent([path])
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=False)
    colors = plt.cm.viridis(np.linspace(0, 0.88, len(gauges)))
    panels = (('phase1', ta1, tr1, srcs['phase1']),
              ('phase2', ta2, tr2, srcs['phase2']),
              ('phase3', ta3, tr3[r_head], srcs['phase3']))
    for ax, (ph, ta, tr, s) in zip(axes, panels):
        t0 = s['t0_abs']
        tt = [t0 + datetime.timedelta(seconds=float(v)) for v in ta]
        for j, g in enumerate(gauges):
            is_src = any(abs(float(m) - md_table.md_of(int(g))) < 0.5
                         for m in s['source_md_ft'])
            ax.plot(tt, tr[:, j] - tr[0, j], color=colors[j], lw=1.2,
                    label=f"g{g} MD {md_table.md_of(int(g)):.0f}"
                          + (" [PINNED: is a source]" if is_src else ""))
            f = field[int(g)]
            tf = f['taxis_s'] + (f['t0_abs'] - t0).total_seconds()
            keep = (tf >= float(ta[0])) & (tf <= float(ta[-1]))
            if keep.sum() > 5:
                dobs = f['psi'][keep] - float(f['psi'][keep][0])
                tto = [t0 + datetime.timedelta(seconds=float(v))
                       for v in tf[keep]]
                ax.plot(tto, dobs, color=colors[j], lw=0.9, ls=':', alpha=0.85)
        ax.set_ylabel('dP from phase start (psi)')
        ttl = ph + (f"   barrier ratio {r_head:g}" if ph == 'phase3' else "")
        ax.set_title(f"{ttl}   source at "
                     f"{', '.join('%.1f' % m for m in s['source_md_ft'])} ft "
                     f"({s['n_dirichlet_nodes']} Dirichlet node"
                     f"{'s' if s['n_dirichlet_nodes'] > 1 else ''}), driven by "
                     f"gauge {s['gauge']} (MD {s['md_ft']:.0f} ft)", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel('2020-03-18 (naive local time, as stored). '
                        'solid = simulated, dotted = measured')
    fig.suptitle(f"B4 {name}: placement '{cfg['source']['placement']}' - "
                 f"D = {cfg['physics']['D_baseline_ft2_s']:g} ft$^2$/s, pad "
                 f"{cfg['mesh']['pad_low_ft']:g}/{cfg['mesh']['pad_high_ft']:g} ft, "
                 f"w = {cfg['barrier']['w_ft']:g} ft, dt = "
                 f"{cfg['time']['dt_fixed_s']:g} s", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# identity check against a5.run_chain
# ---------------------------------------------------------------------------

def identity_check(root_cfg, out_root, config_path):
    """placement='frac_hit_nodes' must reproduce a5.run_chain BITWISE.

    Without this, "only the placement changed" would be an assurance rather than
    a measurement: b4_source.py runs its own phase loop (it has to, because the
    placement is chosen inside it), and a silent divergence in the initial
    condition, the barrier construction or the phase chaining would look exactly
    like a placement effect.
    """
    icfg = root_cfg['identity_check']
    st = [s for s in root_cfg['studies'] if s['name'] == icfg['study']][0]
    cfg = a5.deep_merge(root_cfg['base'], st.get('overrides'))
    ratio = float(icfg['ratio'])
    md_table = rd.load_gauge_md_table()
    x, _ = a5.build_chain_mesh(cfg['mesh'])
    wins, _ = a5.phase_windows(cfg['phase_boundaries'])
    gauges = [int(g) for g in cfg['targets']['gauges']]
    gidx = [int(np.argmin(np.abs(x - md_table.md_of(g)))) for g in gauges]

    D0 = float(cfg['physics']['D_baseline_ft2_s'])
    base = np.full(len(x), D0, dtype=float)
    hits7 = rd.load_frac_hits(7, unique=False, sort=False)
    bcfg = cfg['barrier']

    # --- B4's own path ---------------------------------------------------
    srcs = {p: make_source(cfg, x, wins, p) for p in PHASES}
    init1 = np.full(len(x), float(srcs['phase1']['values_psi'][0]))
    ta1, f1, _, _ = a5.solve_phase(x, base, srcs['phase1'], init1,
                                   cfg['time'], cfg['physics'])
    b_tr1 = np.ascontiguousarray(f1[:, gidx]); end1 = f1[-1].copy(); del f1
    ta2, f2, _, _ = a5.solve_phase(x, base, srcs['phase2'], end1,
                                   cfg['time'], cfg['physics'])
    b_tr2 = np.ascontiguousarray(f2[:, gidx]); init3 = f2[-1].copy(); del f2
    dprof, brep = rc.build_barrier_profile(
        x, base, hits7, float(bcfg['w_ft']), ratio,
        ratio_reference=bcfg['ratio_reference'], combine=bcfg['combine'],
        on_empty=bcfg['on_empty'], on_outside=bcfg['on_outside'],
        return_report=True)
    ta3, f3, _, _ = a5.solve_phase(x, dprof, srcs['phase3'], init3,
                                   cfg['time'], cfg['physics'])
    b_tr3 = np.ascontiguousarray(f3[:, gidx]); del f3

    # --- A5's own path ---------------------------------------------------
    ch = a5.run_chain(cfg, x, wins, ratio)
    a_tr = {ph: np.ascontiguousarray(ch[ph]['field'][:, gidx]) for ph in PHASES}
    a_src = {ph: [int(i) for i in ch[ph]['src']['source_idx']] for ph in PHASES}
    del ch

    rows = {}
    for ph, b, ta in (('phase1', b_tr1, ta1), ('phase2', b_tr2, ta2),
                      ('phase3', b_tr3, ta3)):
        a = a_tr[ph]
        rows[ph] = {
            'shape_b4': list(b.shape), 'shape_a5': list(a.shape),
            'source_idx_b4': [int(i) for i in srcs[ph]['source_idx']],
            'source_idx_a5': a_src[ph],
            'source_idx_identical': (
                [int(i) for i in srcs[ph]['source_idx']] == a_src[ph]),
            'bitwise_identical': bool(np.array_equal(a, b)),
            'max_abs_diff_psi': float(np.max(np.abs(a - b))),
            'max_abs_value_psi': float(np.max(np.abs(a))),
        }
    ok = all(r['bitwise_identical'] and r['source_idx_identical']
             for r in rows.values())
    doc = {'kind': 'equivalence_check_no_manifest_solve',
           'what': ("b4_source.py's phase loop vs a5_two_stage_chain.run_chain on "
                    "the identical config, placement='frac_hit_nodes'"),
           'config': rm._rel(config_path),
           'study': icfg['study'], 'ratio': ratio,
           'barrier_n_fallback': int(brep['n_fallback']),
           'gauges': gauges, 'PASS': bool(ok), 'per_phase': rows,
           'code_sha256': rm.sha256_file(os.path.abspath(__file__)),
           'a5_sha256': rm.sha256_file(a5.__file__),
           'generated_utc':
               datetime.datetime.now(datetime.timezone.utc).isoformat()}
    p = os.path.join(out_root, 'b4_identity_check_v1.json')
    os.makedirs(out_root, exist_ok=True)
    rm.assert_absent([p])
    with open(p, 'w') as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    print(f"[B4] identity check {'PASS' if ok else 'FAIL'} -> {p}")
    for ph, r in rows.items():
        print(f"    {ph}: bitwise={r['bitwise_identical']} "
              f"max|diff|={r['max_abs_diff_psi']:.3e} psi on a "
              f"{r['max_abs_value_psi']:.4g} psi field")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# cross-study rollup and the decision
# ---------------------------------------------------------------------------

def _load_study(out_root, name):
    mp = os.path.join(out_root, name, 'manifest.json')
    jp = os.path.join(out_root, name, f"b4_metrics_{name}_v1.json")
    tp = os.path.join(out_root, name, f"b4_traces_{name}_v1.npz")
    if not (os.path.exists(mp) and os.path.exists(jp) and os.path.exists(tp)):
        return None
    with open(jp) as fh:
        j = json.load(fh)
    return {'name': name, 'metrics': j['metrics'],
            'placement': j['placement'],
            'barrier_signal': j['barrier_signal'],
            'npz': tp, 'manifest': mp, 'metrics_json': jp}


def cross_report(root_cfg, out_root, config_path, tag='v1'):
    """Solve-free rollup: reads the manifested studies' saved products."""
    md_table = rd.load_gauge_md_table()
    studies = {}
    for st in root_cfg['studies']:
        s = _load_study(out_root, st['name'])
        if s is not None:
            studies[st['name']] = s
    if not studies:
        raise SystemExit("cross_report found no completed studies")

    ref_name = root_cfg['cross_study_reference']
    arms = root_cfg['arms']
    r_head = float(root_cfg['base']['barrier']['headline_ratio'])
    r_uni = float(root_cfg['base']['barrier']['uniform_reference_ratio'])
    gauges = [int(g) for g in root_cfg['base']['targets']['gauges']]
    in_fig = set(int(g) for g in root_cfg['base']['targets']['in_manuscript_figure'])

    # ---- table 1: per target gauge, per placement, per phase ---------------
    key_phases = ['phase1', 'phase2', f"phase3@ratio={r_head:g}",
                  f"phase3@ratio={r_uni:g}"]
    table = {}
    for arm, names in arms.items():
        table[arm] = {}
        for ph in key_phases:
            table[arm][ph] = {}
            for nm in names:
                if nm not in studies:
                    continue
                m = studies[nm]['metrics'].get(ph)
                if m is None:
                    continue
                table[arm][ph][studies[nm]['placement']] = {
                    'study': nm,
                    'per_gauge': {k: {kk: v[kk] for kk in
                                      ('rmse_delta_psi', 'rmse_abs_psi',
                                       'amplitude_ratio', 'arrival_err_s',
                                       'obs_max_dp_psi', 'sim_max_dp_psi',
                                       'is_dirichlet_node_in_this_phase',
                                       'distance_to_nearest_source_node_ft')}
                                  for k, v in m['per_gauge'].items()
                                  if v.get('status') == 'OK'},
                    'aggregate': m['aggregate'],
                }

    # ---- table 2: placement-vs-placement trace differences -----------------
    def dpair(a, b, ph, ratio=None):
        za, zb = np.load(studies[a]['npz']), np.load(studies[b]['npz'])
        if ph == 'phase3':
            ta, tb = za['phase3_taxis_s'], zb['phase3_taxis_s']
            A = za[f"phase3_traces_psi_{rtag(ratio)}"]
            B = zb[f"phase3_traces_psi_{rtag(ratio)}"]
        else:
            ta, tb = za[f"{ph}_taxis_s"], zb[f"{ph}_taxis_s"]
            A, B = za[f"{ph}_traces_psi"], zb[f"{ph}_traces_psi"]
        dA, dB = A - A[0], B - B[0]
        if not np.array_equal(ta, tb):
            common = np.linspace(0.0, min(float(ta[-1]), float(tb[-1])), 4001)
            dA = np.stack([np.interp(common, ta, dA[:, j])
                           for j in range(dA.shape[1])], axis=1)
            dB = np.stack([np.interp(common, tb, dB[:, j])
                           for j in range(dB.shape[1])], axis=1)
        d = np.abs(dA - dB).max(axis=0)
        peak = np.abs(dA).max(axis=0)
        return d, peak

    pairs = {}
    for arm, names in arms.items():
        by_pl = {studies[n]['placement']: n for n in names if n in studies}
        if len(by_pl) < 2:
            continue
        pairs[arm] = {}
        combos = [('driving_gauge_md', 'frac_hit_centroid', 'LOCATION only '
                   '(1 node each)'),
                  ('frac_hit_centroid', 'frac_hit_nodes', 'MULTIPLICITY only '
                   '(1 vs 6 nodes, same centroid)'),
                  ('driving_gauge_md', 'frac_hit_nodes', 'the full text-vs-code '
                   'gap')]
        for a_pl, b_pl, what in combos:
            if a_pl not in by_pl or b_pl not in by_pl:
                continue
            key = f"{a_pl} vs {b_pl}"
            pairs[arm][key] = {'what_it_isolates': what, 'phases': {}}
            for ph, ratio in (('phase1', None), ('phase2', None),
                              ('phase3', r_head), ('phase3', r_uni)):
                d, peak = dpair(by_pl[a_pl], by_pl[b_pl], ph, ratio)
                lbl = ph if ratio is None else f"phase3@ratio={ratio:g}"
                pairs[arm][key]['phases'][lbl] = {
                    'max_abs_dP_difference_psi': {
                        f"g{g}": float(d[j]) for j, g in enumerate(gauges)},
                    'as_pct_of_that_gauge_peak': {
                        f"g{g}": (float(100.0 * d[j] / peak[j])
                                  if peak[j] > 0 else None)
                        for j, g in enumerate(gauges)},
                    'worst_gauge': f"g{gauges[int(np.argmax(d))]}",
                    'worst_psi': float(d.max()),
                    'worst_gauge_in_figure': f"g{gauges[int(np.argmax(np.where([g in in_fig for g in gauges], d, -1)))]}",
                    'worst_psi_in_figure': float(
                        np.max([d[j] for j, g in enumerate(gauges)
                                if g in in_fig])),
                }

    # ---- table 3: placement effect vs BARRIER effect ------------------------
    bar_vs_place = {}
    for arm, names in arms.items():
        by_pl = {studies[n]['placement']: n for n in names if n in studies}
        if 'frac_hit_nodes' not in by_pl:
            continue
        bsig = studies[by_pl['frac_hit_nodes']]['barrier_signal']
        row = {'barrier_signal_reference_placement': 'frac_hit_nodes',
               'barrier_per_gauge_psi': bsig['per_gauge_psi'],
               'placement_per_gauge_psi': {}, 'ratio_placement_over_barrier': {}}
        for other in ('driving_gauge_md', 'frac_hit_centroid'):
            if other not in by_pl:
                continue
            d, _ = dpair(by_pl[other], by_pl['frac_hit_nodes'], 'phase3', r_head)
            row['placement_per_gauge_psi'][other] = {
                f"g{g}": float(d[j]) for j, g in enumerate(gauges)}
            row['ratio_placement_over_barrier'][other] = {
                f"g{g}": (float(d[j] / bsig['per_gauge_psi'][f"g{g}"])
                          if bsig['per_gauge_psi'][f"g{g}"] > 0 else None)
                for j, g in enumerate(gauges)}
        bar_vs_place[arm] = row

    doc = {
        'kind': 'derived_view_no_solve',
        'tag': tag,
        'config': rm._rel(config_path),
        'code_sha256': rm.sha256_file(os.path.abspath(__file__)),
        'generated_utc':
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'sources': {n: {'manifest': rm._rel(s['manifest']),
                        'manifest_sha256': rm.sha256_file(s['manifest']),
                        'metrics_json': rm._rel(s['metrics_json']),
                        'metrics_sha256': rm.sha256_file(s['metrics_json']),
                        'traces_npz': rm._rel(s['npz']),
                        'traces_sha256': rm.sha256_file(s['npz'])}
                    for n, s in studies.items()},
        'reference_study': ref_name,
        'metric_definitions': {
            'rmse_delta_psi': ('simulated minus measured, both referenced to '
                               'their own first in-window sample; the placement '
                               'test'),
            'rmse_abs_psi': ('simulated minus measured ABSOLUTE psi; what the '
                             "manuscript's ax4 panel draws; carries the whole "
                             'chain offset'),
            'amplitude_ratio': 'max(simulated dP) / max(measured dP)',
            'arrival_err_s': ('t_sim - t_obs at one threshold, 0.1 x max(measured '
                              'dP), applied to both traces on the measured time '
                              'axis (r1_calibration_core convention)'),
            'aggregation': 'gauge-mean RMSE, not sample-pooled (C1)',
        },
        'per_gauge_table': table,
        'placement_pair_differences': pairs,
        'placement_effect_vs_barrier_effect': bar_vs_place,
        'gauges_in_manuscript_figure': sorted(in_fig),
        'gauge_md_ft': {f"g{g}": md_table.md_of(int(g)) for g in gauges},
    }
    p = os.path.join(out_root, f"b4_cross_report_{tag}.json")
    rm.assert_absent([p])
    with open(p, 'w') as fh:
        json.dump(rm._jsonify(doc, max_array_len=64)[0], fh, indent=2,
                  sort_keys=True, ensure_ascii=False)
    print(f"[B4] cross report -> {p}")
    return doc, studies, p


# ---------------------------------------------------------------------------
# cross-study figures
# ---------------------------------------------------------------------------

_PL_COL = {'frac_hit_nodes': '#1f77b4', 'driving_gauge_md': '#d62728',
           'frac_hit_centroid': '#2ca02c'}
_PL_SHORT = {'frac_hit_nodes': 'frac-hit nodes (6)',
             'driving_gauge_md': 'driving gauge MD (1)',
             'frac_hit_centroid': 'frac-hit centroid (1)'}


def figure_geometry(path, root_cfg, md_table, dpi):
    rm.assert_absent([path])
    hits7 = np.asarray(rd.load_frac_hits(7, unique=False, sort=False), float)
    hits8 = np.asarray(rd.load_frac_hits(8, unique=False, sort=False), float)
    gauges = [int(g) for g in root_cfg['base']['targets']['gauges']]
    in_fig = set(int(g) for g in root_cfg['base']['targets']['in_manuscript_figure'])

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    for ax, (stage, hits, drv) in zip(
            axes, ((7, hits7, 6), (8, hits8, 7))):
        c = float(np.mean(hits))
        gmd = md_table.md_of(drv)
        ax.plot(hits, np.zeros_like(hits), 'o', ms=9, mfc='none',
                color=_PL_COL['frac_hit_nodes'], mew=2,
                label=f"stage-{stage} frac hits (n={len(hits)})")
        ax.plot([c], [0], 's', ms=11, color=_PL_COL['frac_hit_centroid'],
                label=f"frac-hit centroid  MD {c:.1f}")
        ax.plot([gmd], [0], 'v', ms=12, color=_PL_COL['driving_gauge_md'],
                label=f"driving gauge {drv}  MD {gmd:.0f}")
        for g in gauges:
            m = md_table.md_of(g)
            ax.axvline(m, color='0.6', lw=0.8,
                       ls='-' if g in in_fig else ':')
            ax.text(m, 0.62, f"g{g}", ha='center', va='bottom', fontsize=8,
                    color='0.25')
        ax.annotate("", xy=(c, -0.35), xytext=(gmd, -0.35),
                    arrowprops=dict(arrowstyle='<->', color='0.3', lw=1.2))
        ax.text(0.5 * (c + gmd), -0.52,
                f"gauge -> centroid  {abs(gmd - c):.1f} ft",
                ha='center', va='top', fontsize=9, color='0.2')
        ax.axvspan(float(np.min(hits7)), float(np.max(hits7)), color='0.85',
                   zorder=0,
                   label=('stage-7 row = the phase-3 BARRIER row'
                          if stage == 7 else None))
        ax.set_ylim(-1.0, 1.0)
        ax.set_yticks([])
        ax.set_title(f"phase {'1 and 2' if stage == 7 else '3'}: "
                     f"stage-{stage} injection driven by gauge {drv}",
                     fontsize=10)
        ax.legend(fontsize=8, loc='upper left', ncol=2)
    axes[-1].set_xlabel('measured depth (ft)')
    axes[-1].set_xlim(14700, 15750)
    fig.suptitle("B4 - the three defensible source placements, and where the "
                 "target gauges sit\n(solid grey = gauge drawn in Fig. 6; "
                 "dotted = carried here but not in the figure)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def figure_metrics(path, doc, root_cfg, arm, dpi):
    rm.assert_absent([path])
    r_head = float(root_cfg['base']['barrier']['headline_ratio'])
    gauges = [int(g) for g in root_cfg['base']['targets']['gauges']]
    in_fig = set(int(g) for g in root_cfg['base']['targets']['in_manuscript_figure'])
    phases = ['phase1', f"phase3@ratio={r_head:g}"]
    quantities = [('rmse_delta_psi', 'RMSE vs measured, dP (psi)', True),
                  ('amplitude_ratio', 'amplitude ratio  sim/obs', False),
                  ('arrival_err_s', 'arrival error  t$_{sim}$-t$_{obs}$ (s)',
                   False)]
    fig, axes = plt.subplots(len(phases), 3, figsize=(14, 7.2))
    width = 0.26
    for i, ph in enumerate(phases):
        blk = doc['per_gauge_table'][arm].get(ph, {})
        for k, (q, lbl, logy) in enumerate(quantities):
            ax = axes[i, k]
            for m, pl in enumerate(('driving_gauge_md', 'frac_hit_nodes',
                                    'frac_hit_centroid')):
                if pl not in blk:
                    continue
                pg = blk[pl]['per_gauge']
                xs, ys, pin = [], [], []
                for j, g in enumerate(gauges):
                    v = pg.get(f"g{g}")
                    if v is None:
                        continue
                    xs.append(j + (m - 1) * width)
                    ys.append(v[q] if v[q] is not None
                              and np.isfinite(v[q]) else np.nan)
                    pin.append(v['is_dirichlet_node_in_this_phase'])
                bars = ax.bar(xs, np.abs(ys) if logy else ys, width=width,
                              color=_PL_COL[pl], label=_PL_SHORT[pl],
                              edgecolor='none')
                for b, p_ in zip(bars, pin):
                    if p_:
                        b.set_hatch('///')
                        b.set_edgecolor('k')
            if logy:
                ax.set_yscale('log')
            if q == 'amplitude_ratio':
                ax.axhline(1.0, color='0.3', lw=0.9, ls='--')
            if q == 'arrival_err_s':
                ax.axhline(0.0, color='0.3', lw=0.9, ls='--')
            ax.set_xticks(range(len(gauges)))
            ax.set_xticklabels([f"g{g}" + ("" if g in in_fig else "*")
                                for g in gauges], fontsize=8)
            ax.set_ylabel(lbl, fontsize=8)
            ax.grid(alpha=0.3, axis='y')
            if i == 0 and k == 0:
                ax.legend(fontsize=7)
            ax.set_title(ph, fontsize=9)
    fig.suptitle(f"B4 arm '{arm}': per target gauge, per placement. "
                 f"hatched = that gauge IS a Dirichlet node in that phase "
                 f"(misfit zero by construction). * = not drawn in Fig. 6",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def figure_place_vs_barrier(path, doc, root_cfg, dpi):
    rm.assert_absent([path])
    gauges = [int(g) for g in root_cfg['base']['targets']['gauges']]
    in_fig = set(int(g) for g in root_cfg['base']['targets']['in_manuscript_figure'])
    arms = list(doc['placement_effect_vs_barrier_effect'].keys())
    fig, axes = plt.subplots(1, len(arms), figsize=(5.2 * len(arms), 4.6),
                             squeeze=False)
    for k, arm in enumerate(arms):
        ax = axes[0, k]
        row = doc['placement_effect_vs_barrier_effect'][arm]
        xs = np.arange(len(gauges))
        bar = [row['barrier_per_gauge_psi'][f"g{g}"] for g in gauges]
        r_head = float(root_cfg['base']['barrier']['headline_ratio'])
        ax.bar(xs - 0.22, np.maximum(bar, 1e-12), width=0.22, color='0.35',
               label=f"BARRIER: |dP(ratio {r_head:g}) - dP(no barrier)|")
        for m, pl in enumerate(('driving_gauge_md', 'frac_hit_centroid')):
            if pl not in row['placement_per_gauge_psi']:
                continue
            v = [row['placement_per_gauge_psi'][pl][f"g{g}"] for g in gauges]
            ax.bar(xs + (m) * 0.22, np.maximum(v, 1e-12), width=0.22,
                   color=_PL_COL[pl],
                   label=f"PLACEMENT: {_PL_SHORT[pl]} vs frac-hit nodes")
        ax.set_yscale('log')
        ax.set_xticks(xs)
        ax.set_xticklabels([f"g{g}" + ("" if g in in_fig else "*")
                            for g in gauges], fontsize=8)
        ax.set_ylabel('max |dP difference| over phase 3 (psi)')
        ax.set_title(arm, fontsize=10)
        ax.grid(alpha=0.3, axis='y')
        if k == 0:
            ax.legend(fontsize=7, loc='best')
    fig.suptitle("B4 - is the Fig. 6 barrier signal larger than the ambiguity "
                 "in where the source is applied?  (phase 3, headline ratio)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--only', default=None,
                    help='comma-separated study names')
    ap.add_argument('--identity-check', action='store_true')
    ap.add_argument('--report', action='store_true',
                    help='solve-free cross-study rollup + figures')
    ap.add_argument('--tag', default='v1')
    args = ap.parse_args(argv)

    config_path = os.path.abspath(args.config)
    root_cfg = a5.load_config(config_path)
    out_root = os.path.join(rm.repo_root(), root_cfg['output_root'])
    os.makedirs(out_root, exist_ok=True)
    dpi = int(root_cfg['base']['outputs']['figure_dpi'])

    if args.identity_check:
        return identity_check(root_cfg, out_root, config_path)

    if args.report:
        doc, studies, p = cross_report(root_cfg, out_root, config_path,
                                       tag=args.tag)
        md_table = rd.load_gauge_md_table()
        figure_geometry(os.path.join(out_root,
                                     f"fig02_placement_geometry_{args.tag}.png"),
                        root_cfg, md_table, dpi)
        for arm in doc['per_gauge_table']:
            figure_metrics(os.path.join(
                out_root, f"fig03_metrics_{arm}_{args.tag}.png"),
                doc, root_cfg, arm, dpi)
        figure_place_vs_barrier(os.path.join(
            out_root, f"fig04_placement_vs_barrier_{args.tag}.png"),
            doc, root_cfg, dpi)
        print(f"[B4] figures written under {out_root}")
        return 0

    only = set(args.only.split(',')) if args.only else None
    t_all = time.time()
    for st in root_cfg['studies']:
        if not st.get('enabled', True):
            continue
        if only and st['name'] not in only:
            continue
        cfg = a5.deep_merge(root_cfg['base'], st.get('overrides'))
        outdir = os.path.join(out_root, st['name'])
        print(f"[B4] study {st['name']} "
              f"(placement {cfg['source']['placement']}, "
              f"D {cfg['physics']['D_baseline_ft2_s']:g}, "
              f"pad {cfg['mesh']['pad_low_ft']:g})", flush=True)
        s = run_study(st['name'], cfg, root_cfg, outdir, config_path)
        print(f"[B4] study {st['name']} done in {s['wall_s']:.1f} s -> {outdir}",
              flush=True)
    print(f"[B4] all studies in {time.time() - t_all:.1f} s")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
