"""D0 = 480 ft^2/s rerun of the two-stage chain (Tasks 1-5).

    python3 scripts/manuscript_well_leakage/revision_figs/d480_two_stage.py

Run with CWD = repo root.  Writes into `output/revision_d480/` and puts the
regenerated panel (b) in `figs/manuscript_revision/`.

WHY THIS EXISTS
---------------
The manuscript reports a baseline hydraulic diffusivity of 480 ft^2/s.  Every
run behind the two-stage figure so far used D0 = 140, the value hard-coded at
`101_fiberis_matching.py:93`.  Figure and text describe different models.  This
rebuilds the chain at 480 and recomputes the four numbers the text quotes.

WHAT IT DOES NOT DO
-------------------
No ratio sweep, no D0 sweep, no barrier-width study, no leave-one-out, no
production scripts, no manuscript files.  Those studies are complete and are
inputs here, not questions.

HOW THE CHAIN IS RUN
--------------------
`a5_two_stage_chain` is IMPORTED, not copied and not edited: its
`build_chain_mesh` / `phase_windows` / `run_chain` are the reference
implementation of the three-phase chain (phase 1 stage-7 injection driven by
Gauge B, phase 2 shut-in, phase 3 stage-8 injection driven by Gauge C with the
stage-7 frac hits carrying the reduced diffusivity).  Only the config differs:
D0, the padding, fixed dt = 1 s and the barrier ratio.  Everything else -
legacy mesh refinement, w = 0 barriers, harmonic faces, backward Euler,
`requested_start` crop, legacy phase boundaries - is left at the A5 base value.

GAUGE LETTERING (for the report)
    A = gauge 5  (MD 15599, deepest)      D = gauge 8  (MD 14821, withheld)
    B = gauge 6  (MD 15344, phase 1-2 driver)
    C = gauge 7  (MD 15075, phase 3 driver)
"""

import argparse
import datetime
import gc
import json
import os
import platform
import sys
import time

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                    # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REV2 = os.path.abspath(os.path.join(_HERE, os.pardir, 'rev2'))
for _p in (_HERE, _REV2):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import a5_two_stage_chain as a5                    # noqa: E402  (imported, never edited)
import rev2_data as rd                             # noqa: E402
import revfig_common as rc                         # noqa: E402

OUTDIR = 'output/revision_d480'
FIGDIR = 'figs/manuscript_revision'
A5_CONFIG = 'configs/rev2/a5_two_stage_chain.json'

GAUGE_LETTER = {5: 'A', 6: 'B', 7: 'C', 8: 'D', 9: 'E', 10: 'F'}
TARGETS = [5, 6, 7, 8, 9, 10]

D_NEW = 480.0        # the manuscript value
D_OLD = 140.0        # 101:93, the value every earlier run used
RATIO = 1e-5
PADS = [5000.0, 10000.0, 20000.0]
PAD_TOL_PSI = 1.0    # Task 1 acceptance

# Task 5: the transfer coefficient and the colour limit are UNDER SEPARATE
# REVIEW and are deliberately left exactly as `fig6_history_matching` has them.
GAMMA = 8.94e-9                                  # 1/psi
K_LEGACY = 6894.76 / 30e9
CLIM_PANEL_B = 1e-7 / (K_LEGACY / GAMMA)         # 3.88991059877356e-09 strain/s


# --------------------------------------------------------------------------- chain

def make_cfg(d0, pad):
    """A5's base config with exactly four fields changed."""
    cfg = json.load(open(A5_CONFIG))['base']
    cfg['physics']['D_baseline_ft2_s'] = float(d0)
    cfg['mesh']['pad_low_ft'] = float(pad)
    cfg['mesh']['pad_high_ft'] = float(pad)      # B2: 5000 ft at BOTH ends
    cfg['time']['mode'] = 'fixed'
    cfg['time']['dt_fixed_s'] = 1.0              # backward Euler, theta = 1
    cfg['targets']['gauges'] = list(TARGETS)
    return cfg


def run_one(d0, pad, ratio, want_field=False, field_stride=8, log=print):
    """One three-phase chain.  Returns only what the report needs; the full
    field (up to 5 GB per phase at pad 20000) is freed as soon as it is cropped."""
    cfg = make_cfg(d0, pad)
    x, mesh_rec = a5.build_chain_mesh(cfg['mesh'])
    wins, _ = a5.phase_windows(cfg['phase_boundaries'])
    t0 = time.time()
    log(f"    solving D0={d0:g} pad={pad:g} ratio={ratio:g}  nx={len(x)} ...")
    ch = a5.run_chain(cfg, x, wins, float(ratio))
    log(f"      done in {time.time() - t0:.1f} s")

    gidx = [int(np.argmin(np.abs(x - md))) for md in
            [rd.load_gauge_md_table().md_of(g) for g in TARGETS]]

    out = {'D0': float(d0), 'pad_ft': float(pad), 'ratio': float(ratio),
           'nx': int(len(x)), 'gauges': list(TARGETS),
           'gauge_md_ft': [float(x[i]) for i in gidx],
           'wall_s': time.time() - t0, 'mesh': mesh_rec,
           'phase_windows': {k: [v[0].isoformat(), v[1].isoformat()]
                             for k, v in wins.items()},
           'phases': {}}

    if want_field:
        fh7 = rd.load_frac_hits(7, unique=False)
        fh8 = rd.load_frac_hits(8, unique=False)
        lo, hi = float(np.min(fh8)) - 500.0, float(np.max(fh7)) + 500.0
        dsel = np.nonzero((x >= lo) & (x <= hi))[0]
        out['panel_depth_window_ft'] = [lo, hi]
        out['panel_md_ft'] = x[dsel]
        out['panel_stride'] = int(field_stride)

    for name in ('phase1', 'phase2', 'phase3'):
        c = ch[name]
        rec = {'taxis_s': np.asarray(c['taxis'], float),
               'traces_psi': np.ascontiguousarray(c['field'][:, gidx]),
               'window_abs': c['window_abs'],
               'source_gauge': int(c['src']['gauge']),
               'n_steps': int(len(c['taxis']))}
        if want_field:
            # dP/dt is stored GAMMA-FREE so the transfer coefficient can be
            # changed later without re-running the chain; it must be differenced
            # on the FULL 1 s grid and only then thinned.
            dpdt = np.gradient(c['field'][:, dsel],
                               np.asarray(c['taxis'], float), axis=0)
            rec['panel_dpdt_psi_per_s'] = np.ascontiguousarray(dpdt[::field_stride])
            rec['panel_taxis_s'] = np.asarray(c['taxis'], float)[::field_stride]
            del dpdt
        out['phases'][name] = rec
        c['field'] = None
    del ch
    gc.collect()
    return out


# ------------------------------------------------------------------- assembly

def concat_traces(run):
    """Phases 1-3 end to end on one absolute-seconds axis, duplicate joints dropped."""
    t, y = [], []
    t0_abs = datetime.datetime.fromisoformat(run['phases']['phase1']['window_abs'][0])
    for k, name in enumerate(('phase1', 'phase2', 'phase3')):
        p = run['phases'][name]
        off = (datetime.datetime.fromisoformat(p['window_abs'][0]) - t0_abs).total_seconds()
        s = 1 if k else 0                      # drop the repeated joint sample
        t.append(p['taxis_s'][s:] + off)
        y.append(p['traces_psi'][s:])
    return np.concatenate(t), np.concatenate(y, axis=0), t0_abs


def observed_delta(t_rel_s, t0_abs, gauges):
    """Measured gauge series, baseline-removed, on the simulation time axis."""
    w = rd.Window(12000.0, 17000.0, t0_abs,
                  t0_abs + datetime.timedelta(seconds=float(t_rel_s[-1])))
    gw = rd.load_window_gauges(w, gauges=list(gauges))
    out = {}
    for g in gauges:
        s = gw.series[int(g)]
        lead = (s.t0_abs - t0_abs).total_seconds()
        out[int(g)] = np.interp(t_rel_s, s.taxis_s + lead,
                                s.raw_psi - float(s.raw_psi[0]))
    return out


def rmse_vs_field(run, gauges):
    """RMSE of the pressure CHANGE, simulated vs measured, per gauge.

    Both series are referenced to their own first sample of the chain window, so
    this measures the change, not the absolute level - the simulated field is
    seeded with a uniform absolute pressure and carries no meaningful datum.
    Reported over the whole chain and, separately, over phase 3, where the
    barrier is the only thing acting.
    """
    t, y, t0_abs = concat_traces(run)
    obs = observed_delta(t, t0_abs, gauges)
    p3_start = (datetime.datetime.fromisoformat(run['phases']['phase3']['window_abs'][0])
                - t0_abs).total_seconds()
    m3 = t >= p3_start
    out = {}
    for g in gauges:
        j = TARGETS.index(int(g))
        sim = y[:, j] - y[0, j]
        r = sim - obs[int(g)]
        out[int(g)] = {
            'rmse_full_chain_psi': float(np.sqrt(np.mean(r ** 2))),
            'rmse_phase3_psi': float(np.sqrt(np.mean(r[m3] ** 2))),
            'n_samples': int(t.size)}
    return out


def barrier_signal(run_b, run_u, gauges):
    """max_t |P_barrier - P_nobarrier| over phase 3, per gauge."""
    out = {}
    for g in gauges:
        j = TARGETS.index(int(g))
        a = run_b['phases']['phase3']['traces_psi'][:, j]
        b = run_u['phases']['phase3']['traces_psi'][:, j]
        n = min(a.size, b.size)
        d = (a[:n] - a[0]) - (b[:n] - b[0])
        out[int(g)] = float(np.max(np.abs(d)))
    return out


# ----------------------------------------------------------------- Task 1

def padding_study(runs, log=print):
    rows = []
    for lo, hi in zip(PADS[:-1], PADS[1:]):
        a, b = runs[lo], runs[hi]
        row = {'from_pad_ft': lo, 'to_pad_ft': hi, 'per_gauge_max_abs_dP_psi': {}}
        for g in (5, 6, 7, 8):
            ja = TARGETS.index(g)
            ta, ya, _ = concat_traces(a)
            tb, yb, _ = concat_traces(b)
            n = min(ya.shape[0], yb.shape[0])
            if not np.allclose(ta[:n], tb[:n]):
                raise RuntimeError("padding runs are not on a common time grid")
            d = (ya[:n, ja] - ya[0, ja]) - (yb[:n, ja] - yb[0, ja])
            row['per_gauge_max_abs_dP_psi'][GAUGE_LETTER[g]] = float(np.max(np.abs(d)))
        row['worst_gauge_psi'] = max(row['per_gauge_max_abs_dP_psi'].values())
        rows.append(row)
    chosen = None
    for row in rows:
        if row['worst_gauge_psi'] < PAD_TOL_PSI:
            chosen = row['from_pad_ft']
            break
    if chosen is None:
        chosen = PADS[-1]
    return rows, chosen


# ----------------------------------------------------------------- Task 5

def panel_b(run, path_stem, dpi=400):
    """Panel (b) of the two-stage figure, standalone, from this chain.

    Same construction as `fig6_history_matching` panel (b): merged phase 1-3
    synthetic strain rate over the plotted depth window, `bwr`, the gauge
    overlays at -0.15 psi/ft, the black dashed gauge locations.  THE COLOUR
    LIMIT IS NOT TOUCHED - see CLIM_PANEL_B.
    """
    md_table = rd.load_gauge_md_table()
    fh7 = rd.load_frac_hits(7, unique=False)
    fh8 = rd.load_frac_hits(8, unique=False)
    lo, hi = float(np.min(fh8)) - 500.0, float(np.max(fh7)) + 500.0
    gauges = [g for g in range(1, 16) if lo <= md_table.md_of(g) <= hi]

    t0_abs = datetime.datetime.fromisoformat(run['phases']['phase1']['window_abs'][0])
    T, SR = [], []
    for k, name in enumerate(('phase1', 'phase2', 'phase3')):
        p = run['phases'][name]
        off = (datetime.datetime.fromisoformat(p['window_abs'][0]) - t0_abs).total_seconds()
        tt = p['panel_taxis_s'] + off
        m = tt > (T[-1][-1] if T else -np.inf)
        if k:                       # legacy select_time(30, ...) on phases 2 and 3
            m &= (p['panel_taxis_s'] >= 30.0)
        T.append(tt[m])
        SR.append(p['panel_dpdt_psi_per_s'][m] * GAMMA)
    taxis = np.concatenate(T)
    sr = np.concatenate(SR, axis=0)
    tt = [t0_abs + datetime.timedelta(seconds=float(s)) for s in taxis]

    tp, yp, _ = concat_traces(run)
    fig, ax = plt.subplots(figsize=(7, 6))
    for g in gauges:
        md = md_table.md_of(g)
        ax.axhline(y=md, color='black', linestyle='--', lw=1.0)
        if g in TARGETS:
            j = TARGETS.index(g)
            y = (yp[:, j] - yp[0, j]) * -0.15 + md
            tabs = [t0_abs + datetime.timedelta(seconds=float(s)) for s in tp]
            ax.plot(tabs, y, color=('black' if g == 8 else rc.GAUGE_COLOR),
                    lw=rc.GAUGE_LW, zorder=4)
    img = ax.pcolormesh(tt, run['panel_md_ft'], sr.T, cmap='bwr', shading='auto')
    img.set_clim(-CLIM_PANEL_B, CLIM_PANEL_B)
    ax.set_ylim(hi, lo)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(path_stem), exist_ok=True)
    png, tif = path_stem + '.png', path_stem + '.tiff'
    fig.savefig(png, dpi=dpi)
    fig.savefig(tif, dpi=dpi, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    return {'png': png, 'tiff': tif, 'dpi': dpi,
            'clim_strain_per_s': [-CLIM_PANEL_B, CLIM_PANEL_B],
            'gamma_per_psi': GAMMA,
            'colour_limit_status': 'UNCHANGED - pending the transfer-coefficient review',
            'depth_window_ft': [lo, hi], 'n_time_columns': int(taxis.size)}


def export_chain(path, d0, pad, ratio, field_stride=8, log=print):
    """Persist the merged chain products the figure needs.

    The D0 = 480 study did not save its field, only the numbers and the
    standalone panel.  This re-executes THE SAME frozen configuration - same
    A5 module, same config, same D0/pad/ratio/dt - and writes the merged
    products so no later figure change has to touch the solver again.  It is a
    deterministic regeneration, not a new study: nothing about the run is
    varied and no number in the D0 = 480 report changes.
    """
    run = run_one(d0, pad, ratio, want_field=True, field_stride=field_stride, log=log)
    t0_abs = datetime.datetime.fromisoformat(run['phases']['phase1']['window_abs'][0])
    T, G = [], []
    for k, name in enumerate(('phase1', 'phase2', 'phase3')):
        pph = run['phases'][name]
        off = (datetime.datetime.fromisoformat(pph['window_abs'][0]) - t0_abs).total_seconds()
        tt = pph['panel_taxis_s'] + off
        m = tt > (T[-1][-1] if T else -np.inf)
        if k:                      # legacy select_time(30, ...) on phases 2 and 3
            m &= (pph['panel_taxis_s'] >= 30.0)
        T.append(tt[m])
        G.append(pph['panel_dpdt_psi_per_s'][m])
    tp, yp, _ = concat_traces(run)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        panel_md_ft=run['panel_md_ft'],
        panel_taxis_s=np.concatenate(T),
        panel_dpdt_psi_per_s=np.concatenate(G, axis=0),
        trace_taxis_s=tp, trace_psi=yp,
        gauge_numbers=np.asarray(TARGETS, dtype=np.int64),
        gauge_md_ft=np.asarray(run['gauge_md_ft'], float),
        t0_abs=np.array(t0_abs.isoformat()),
        D0=np.array(float(d0)), pad_ft=np.array(float(pad)),
        ratio=np.array(float(ratio)), dt_s=np.array(1.0),
        panel_depth_window_ft=np.asarray(run['panel_depth_window_ft'], float),
        field_stride=np.array(int(field_stride)))
    log(f"  wrote {path}")
    return path


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=OUTDIR)
    ap.add_argument('--figdir', default=FIGDIR)
    ap.add_argument('--dpi', type=int, default=400)
    ap.add_argument('--export-chain', metavar='NPZ', default=None,
                    help='re-execute the frozen D0=480 config and persist its '
                         'merged products; does nothing else')
    a = ap.parse_args()

    if a.export_chain:
        export_chain(a.export_chain, D_NEW, 10000.0, RATIO)
        return
    os.makedirs(a.outdir, exist_ok=True)
    t_start = time.time()
    res = {}

    # ---- TASK 1 ---------------------------------------------------------
    print("[TASK 1] domain padding at D0 = 480, ratio 1e-5")
    pad_runs = {}
    for pad in PADS:
        pad_runs[pad] = run_one(D_NEW, pad, RATIO)
    rows, chosen = padding_study(pad_runs)
    for r in rows:
        pretty = '  '.join(f"{k}={v:8.3f}" for k, v in
                           r['per_gauge_max_abs_dP_psi'].items())
        print(f"  pad {r['from_pad_ft']:.0f} -> {r['to_pad_ft']:.0f} ft : {pretty}"
              f"   worst {r['worst_gauge_psi']:.3f} psi")
    print(f"  smallest padding with change < {PAD_TOL_PSI} psi: {chosen:.0f} ft")
    res['task1_padding'] = {'tolerance_psi': PAD_TOL_PSI,
                            'pads_tested_ft': PADS,
                            'metric': 'max over the whole chain of |dP(pad_hi) - '
                                      'dP(pad_lo)|, per gauge, both ends padded',
                            'successive_changes': rows,
                            'chosen_pad_ft': chosen}

    # ---- TASK 2 ---------------------------------------------------------
    print(f"[TASK 2] chain at D0 = 480 and D0 = 140, pad {chosen:.0f} ft")
    runs = {('480', '1e-05'): pad_runs[chosen]}
    runs[('480', '1.0')] = run_one(D_NEW, chosen, 1.0)
    runs[('140', '1e-05')] = run_one(D_OLD, chosen, RATIO)
    runs[('140', '1.0')] = run_one(D_OLD, chosen, 1.0)
    res['task2_runs'] = {f"D{d}_ratio{r}": {
        k: v for k, v in runs[(d, r)].items()
        if k in ('D0', 'pad_ft', 'ratio', 'nx', 'wall_s', 'gauge_md_ft',
                 'phase_windows')} for (d, r) in runs}

    # ---- TASKS 3 and 4 --------------------------------------------------
    print("[TASK 3/4] recomputing the quoted numbers")
    res['task3'] = {}
    res['task4'] = {}
    for d in ('480', '140'):
        rb, ru = runs[(d, '1e-05')], runs[(d, '1.0')]
        fb = rmse_vs_field(rb, [5, 6, 8])
        fu = rmse_vs_field(ru, [5, 6, 8])
        sig = barrier_signal(rb, ru, [5, 6])
        for key in ('rmse_full_chain_psi', 'rmse_phase3_psi'):
            gm_b = float(np.mean([fb[5][key], fb[6][key]]))
            gm_u = float(np.mean([fu[5][key], fu[6][key]]))
            res['task3'].setdefault(key, {})[d] = {
                'per_gauge_no_barrier_psi': {GAUGE_LETTER[g]: fu[g][key] for g in (5, 6)},
                'per_gauge_barrier_1e-5_psi': {GAUGE_LETTER[g]: fb[g][key] for g in (5, 6)},
                'gauge_mean_no_barrier_psi': gm_u,
                'gauge_mean_barrier_1e-5_psi': gm_b,
                'improvement_factor_nobarrier_over_barrier': gm_u / gm_b,
                'improvement_percent': 100.0 * (gm_u - gm_b) / gm_u}
        res['task3'].setdefault('barrier_signal_amplitude_psi', {})[d] = {
            GAUGE_LETTER[g]: sig[g] for g in (5, 6)}
        res['task4'][d] = {
            'gauge': 'D (gauge 8, MD 14821, withheld from calibration)',
            'field_vs_synthetic_rmse_full_chain_psi': fb[8]['rmse_full_chain_psi'],
            'field_vs_synthetic_rmse_phase3_psi': fb[8]['rmse_phase3_psi']}

    for key in ('rmse_full_chain_psi', 'rmse_phase3_psi'):
        for d in ('480', '140'):
            e = res['task3'][key][d]
            print(f"  [{key} D0={d}] no-barrier {e['gauge_mean_no_barrier_psi']:9.2f} "
                  f"barrier {e['gauge_mean_barrier_1e-5_psi']:9.2f}  "
                  f"factor {e['improvement_factor_nobarrier_over_barrier']:6.3f}  "
                  f"({e['improvement_percent']:+.1f} %)")
    for d in ('480', '140'):
        print(f"  [gauge D  D0={d}] field-vs-synthetic RMSE "
              f"{res['task4'][d]['field_vs_synthetic_rmse_full_chain_psi']:9.2f} psi "
              f"(phase 3 only {res['task4'][d]['field_vs_synthetic_rmse_phase3_psi']:.2f})")

    # ---- TASK 5 ---------------------------------------------------------
    print("[TASK 5] regenerating panel (b) from the D0 = 480 barrier run")
    del runs[('480', '1e-05')], pad_runs
    gc.collect()
    panel_run = run_one(D_NEW, chosen, RATIO, want_field=True)
    res['task5_panel_b'] = panel_b(panel_run,
                                   os.path.join(a.figdir, 'Fig6_panelb_D480_no_scale'),
                                   dpi=a.dpi)
    print(f"  -> {res['task5_panel_b']['png']}")

    # ---- manifest -------------------------------------------------------
    inputs = [A5_CONFIG,
              'data/fiberis_format/s_well/geometry/gauge_md_swell.npz',
              'data/fiberis_format/s_well/geometry/frac_hit/frac_hit_stage_7_swell.npz',
              'data/fiberis_format/s_well/geometry/frac_hit/frac_hit_stage_8_swell.npz']
    inputs += [rd.SWELL_GAUGE_TEMPLATE.format(n=g) for g in TARGETS]
    for stage in (7, 8):
        for f in ('Proppant Concentration', 'Slurry Rate', 'Treating Pressure'):
            inputs.append(f"data/fiberis_format/prod/pumping_data/stage{stage}/{f}.npz")
    man = {
        'study': 'D0=480 rerun of the two-stage chain (Tasks 1-5)',
        'manuscript': 'SJ-0626-0151 (SPE Journal), revision 1',
        'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'generating_script': {'path': os.path.relpath(os.path.abspath(__file__), os.getcwd()),
                              'sha256': rc.sha256(os.path.abspath(__file__))},
        'chain_implementation': {
            'module': 'scripts/manuscript_well_leakage/rev2/a5_two_stage_chain.py',
            'sha256': rc.sha256(os.path.join(_REV2, 'a5_two_stage_chain.py')),
            'note': 'imported unmodified; only the config differs'},
        'rev2_modules': {m: rc.sha256(os.path.join(_REV2, m + '.py'))
                         for m in ('rev2_core', 'rev2_data', 'rev2_layout')},
        'inputs': [{'path': p, 'sha256': rc.sha256(p)}
                   for p in sorted(set(inputs)) if os.path.exists(p)],
        'parameters_changed_from_a5_base': {
            'physics.D_baseline_ft2_s': [D_OLD, D_NEW],
            'mesh.pad_low_ft / pad_high_ft': PADS,
            'time.mode': 'fixed', 'time.dt_fixed_s': 1.0,
            'physics.theta': 1.0, 'barrier.ratios': [RATIO, 1.0],
            'barrier.w_ft': 0.0},
        'parameters_left_at_a5_base': [
            'mesh.refinement = legacy_refine_mesh', 'barrier.w_ft = 0.0',
            'physics.interface_avg = harmonic', 'source.crop_rebase = requested_start',
            'phase_boundaries.mode = legacy_file_span'],
        'gauge_lettering': {v: f"gauge {k}" for k, v in GAUGE_LETTER.items()},
        'results': res,
        'not_done_by_instruction': [
            'no reduction-ratio sweep', 'no D0 sweep or baseline re-derivation',
            'no barrier-width study', 'no leave-one-out or source-gauge removal',
            'no production scripts or Figure 7', 'no manuscript file touched',
            'panel (b) colour limit NOT changed - pending the transfer-coefficient review'],
        'environment': {'python': sys.version.split()[0],
                        'platform': platform.platform(),
                        'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
                        'cwd': os.getcwd()},
        'wall_time_s': time.time() - t_start,
    }
    p = os.path.join(a.outdir, 'd480_manifest.json')
    with open(p, 'w') as fh:
        json.dump(man, fh, indent=2, default=lambda o: (o.tolist() if isinstance(o, np.ndarray)
                                                        else str(o)))
    print(f"[manifest] {p}   total {man['wall_time_s']:.1f} s")


if __name__ == '__main__':
    main()
