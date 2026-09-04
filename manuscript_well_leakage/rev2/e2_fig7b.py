"""E2 -- rebuild the Fig. 7b production-period forward model.

Fig. 7b (`figs/manuscript/production/dropdown_to202051_noscalar.png`, drawn by
`scripts/sponsor_meeting_report_2025/production_sim/
103p_forward_modeling_viz_without_scalar.py` from `output/0324_forward_simulator/`,
which `scripts/well_leakage_history_matching/103r_forwarding_modeling_prod_QC_swell.py`
produced) carries four independent, already-established defects. This script does
not re-derive them; it fixes them and measures what is left.

  1 LEGEND (D4). 103p:81,83 hard-codes three labels onto os.listdir positions
    [0,2,4]. "min D = 1 x baseline" is the ratio 0.001 file, "min D = 1e-5 x
    baseline" is the uniform ratio 1.0 file, and the genuine 1e-5 run is dropped
    by `dataframe_full[:-1]`. Here every legend string is read back out of the
    run's own manifest by `label_from_manifest` -- never hard-coded, never
    inferred from a directory listing.
  2 AXIS (D4). The field curve is assembled in os.listdir order while the model
    curves are sampled in gauge-number order, both plotted against one axis
    labelled "Gauge Number", so field point 1 is gauge 10. The field profile here
    comes from `rev2_data.production_drawdown` (numeric order; `sorted()` is also
    wrong). The listdir-ordered curve is drawn once, dashed, in fig02 only, to
    show the size of the defect.
  3 SOURCE (D4). The archive was driven by PRODUCER GAUGE 1 (IC 7028.35 psi,
    MD 16196) though the filenames say gauge3, and 103r cannot run today because
    `os.listdir(prod/gauges)[2]` is now `pressure_g1.npz` (zero samples in the
    window). Drivers here are named by explicit path. 103r:81 also anchors the
    Dirichlet node to the MESH CENTRE, so padding moves the boundary condition;
    every source here is pinned to a physical MD.
  4 DOMAIN (B2). The plotted barrier curve moves -55.6% at a 5000 ft pad and
    -75.7% at 20000 ft, and B2 found no convergence over the pads it tested
    (0 .. 20000 ft). This script extends the sweep to 400000 ft (800000 ft at
    D = 1150) and reports where it converges.

Nothing here edits a shared module: `rev2_core`, `rev2_data` and `rev2_manifest`
are imported. Every solver run writes its own manifest through
`rev2_manifest.RunRecorder` and refuses to overwrite an existing one.

Run:
    python3 scripts/manuscript_well_leakage/rev2/e2_fig7b.py \
        --config configs/rev2/e2_fig7b.json
`--sweep S3_pad` (repeatable) runs a subset, `--outdir` writes elsewhere,
`--figures-only` redraws from manifests and summaries already on disk.
"""

import argparse
import datetime
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.join(os.path.dirname(_HERE), 'baseline_calibration')
for _p in (_HERE, _BASE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rev2_core as rc          # noqa: E402
import rev2_data as rd          # noqa: E402
import rev2_manifest as rm      # noqa: E402
from fiberis.analyzer.Data1D import Data1D_Gauge  # noqa: E402

# The legacy sweep (S6) requests w = 0, which falls back to the nearest node and
# warns once per frac hit. Every other sweep uses on_empty='raise'.
warnings.simplefilter('ignore', rc.BarrierWidthWarning)
warnings.simplefilter('ignore', rc.BarrierOverlapWarning)

FRAC_HIT_DIR = 'data/legacy/s_well/geometry/frac_hit/'
SWELL_GAUGE = rd.SWELL_GAUGE_TEMPLATE
PROD_GAUGE = rd.PROD_GAUGE_TEMPLATE


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

class Inputs:
    """Everything read from disk, loaded once in the parent process."""

    def __init__(self, cfg):
        m = cfg['model']
        self.t0 = datetime.datetime.fromisoformat(m['window']['t_start'])
        self.t1 = datetime.datetime.fromisoformat(m['window']['t_end'])
        self.md_table = rd.load_gauge_md_table()
        self.prod_md_table = rd.load_gauge_md_table(rd.PROD_GAUGE_MD_NPZ,
                                                    check_reference=False)
        # All 113 frac hits of all 20 stages, exactly the set 102r/103r barrier.
        # os.listdir order is irrelevant here because the set is concatenated and
        # `build_barrier_profile` combines with `min`, which is order-independent.
        self.fh_files = sorted(os.listdir(rd.repo_path(FRAC_HIT_DIR)))
        hits = []
        for f in self.fh_files:
            z = np.load(rd.repo_path(FRAC_HIT_DIR, f), allow_pickle=True)
            try:
                hits.append(np.atleast_1d(np.asarray(z['data'], dtype=float)))
            finally:
                z.close()
        self.hits_raw = np.concatenate(hits)
        self.hits = np.unique(self.hits_raw)   # 16696.914 is stored twice

        self.gauges = [int(g) for g in m['targets']['gauges']]
        self.gauge_md = np.array([self.md_table.md_of(g) for g in self.gauges])

        # Observed S-well series over the whole production window.
        self.obs = {}
        for g in self.gauges:
            self.obs[g] = self._series(SWELL_GAUGE.format(n=g))

        # Drivers, by explicit path.
        self.drivers = {}
        for name, d in m['drivers'].items():
            self.drivers[name] = dict(d, series=self._series(d['path']))

        # Field drawdown profiles, in NUMERIC gauge order.
        self.field = {}
        for iso in m['drawdown_eval_dates']:
            te = datetime.datetime.fromisoformat(iso)
            n, md, dd = rd.production_drawdown(self.t0, te, gauges=self.gauges)
            self.field[iso[:10]] = {'gauge': n.tolist(), 'md_ft': md.tolist(),
                                    'drawdown_psi': dd.tolist()}
        # The defect-2 reconstruction: the same fifteen numbers in the order the
        # manuscript figure put them on its "Gauge Number" axis. This is the
        # order of THIS filesystem TODAY, which need not be the order of the
        # filesystem in April 2025; it is drawn only to show the size of the
        # defect, and no number derived from it is quoted as the archive's.
        import re
        self.listdir_order = [
            int(re.search(r'gauge(\d+)_data_swell', f).group(1))
            for f in os.listdir(rd.repo_path('data/fiberis_format/s_well/gauges'))
            if re.search(r'gauge(\d+)_data_swell', f)]

    def _series(self, relpath):
        f = Data1D_Gauge.Data1DGauge()
        f.load_npz(rd.repo_path(relpath))
        f.crop(self.t0, self.t1)
        t = np.asarray(f.taxis, dtype=float)
        v = np.asarray(f.data, dtype=float)
        if t.size == 0:
            raise ValueError(f"{relpath}: crop to [{self.t0}, {self.t1}] "
                             f"returned no samples")
        return t, v


# ---------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------

def build_mesh(hits, *, pad_ft, core, dx_bg, dx_mid, dx_fine, r_mid, r_fine,
               snap):
    """Piecewise-uniform mesh: dx_fine near every frac hit, dx_bg in the pads.

    Built from shared breakpoints with an integer cell count per sub-interval,
    NOT as a de-duplicated union of concatenated grids. A1's open issue 1 records
    a reference solve poisoned by node pairs 9e-13 ft apart that `np.unique` left
    behind; a breakpoint construction cannot produce them, and breakpoints closer
    than `snap` are merged so the smallest cell is bounded below by `snap`.
    """
    lo, hi = float(core[0]) - float(pad_ft), float(core[1]) + float(pad_ft)
    zones = []
    for h in np.asarray(hits, dtype=float):
        zones.append((h - r_fine, h + r_fine, dx_fine))
        zones.append((h - r_mid, h - r_fine, dx_mid))
        zones.append((h + r_fine, h + r_mid, dx_mid))
    bps = [lo, hi]
    for a, b, _ in zones:
        for v in (a, b):
            if lo < v < hi:
                bps.append(float(v))
    bps = np.array(sorted(bps))
    keep = [bps[0]]
    for v in bps[1:]:
        if v - keep[-1] > snap:
            keep.append(float(v))
    keep[-1] = hi
    bps = np.array(keep)
    za = np.array([z[0] for z in zones])
    zb = np.array([z[1] for z in zones])
    zd = np.array([z[2] for z in zones])
    pieces = []
    for a, b in zip(bps[:-1], bps[1:]):
        mid = 0.5 * (a + b)
        sel = (za - 1e-12 <= mid) & (mid <= zb + 1e-12)
        d = float(zd[sel].min()) if sel.any() else float(dx_bg)
        n = max(1, int(np.ceil((b - a) / d - 1e-9)))
        pieces.append(a + (b - a) * np.arange(n) / n)
    x = np.concatenate(pieces + [np.array([hi])])
    dx = np.diff(x)
    meta = {'mode': 'piecewise_uniform_breakpoints',
            'dx_background_ft': float(dx_bg), 'dx_mid_ft': float(dx_mid),
            'dx_fine_ft': float(dx_fine), 'r_mid_ft': float(r_mid),
            'r_fine_ft': float(r_fine), 'breakpoint_snap_ft': float(snap),
            'n_zones': len(zones), 'n_breakpoints': int(bps.size),
            'dx_min_ft': float(dx.min()), 'dx_max_ft': float(dx.max()),
            'n_degenerate_calls': 0,
            'note': ('no legacy mesh_utils.refine_mesh call is made, so its '
                     'degenerate branch (84/113 calls in 103r, A1) cannot fire')}
    return x, meta


def build_mesh_uniform(*, pad_ft, core, dx_bg):
    """The legacy dx = 10 ft mesh with no refinement, for sweep S6."""
    lo, hi = float(core[0]) - float(pad_ft), float(core[1]) + float(pad_ft)
    n = int(round((hi - lo) / dx_bg)) + 1
    x = lo + float(dx_bg) * np.arange(n, dtype=float)
    dx = np.diff(x)
    return x, {'mode': 'uniform_no_refinement',
               'dx_background_ft': float(dx_bg),
               'dx_min_ft': float(dx.min()), 'dx_max_ft': float(dx.max()),
               'n_degenerate_calls': 0,
               'note': ('reproduces the coarse skeleton of the legacy 103r mesh '
                        'without mesh_utils.refine_mesh; combined with w = 0 the '
                        'barrier is one control volume of 10 ft')}


# ---------------------------------------------------------------------------
# one run (worker)
# ---------------------------------------------------------------------------

def solve_spec(spec):
    """Pure numerics for one run. Returns everything the parent needs to write a
    manifest; writes nothing itself, so a Pool worker never races on a file."""
    t_wall = time.time()
    hits = np.asarray(spec['hits'], dtype=float)
    if spec['mesh_kind'] == 'uniform':
        x, mrec = build_mesh_uniform(pad_ft=spec['pad_ft'], core=spec['core'],
                                     dx_bg=spec['dx_bg'])
    else:
        x, mrec = build_mesh(hits, pad_ft=spec['pad_ft'], core=spec['core'],
                             dx_bg=spec['dx_bg'], dx_mid=spec['dx_mid'],
                             dx_fine=spec['dx_fine'], r_mid=spec['r_mid'],
                             r_fine=spec['r_fine'], snap=spec['snap'])
    D0 = float(spec['D_ft2_s'])
    ratio = float(spec['ratio'])
    if ratio == 1.0:
        # ratio 1 is the uniform control: rev2_core returns the baseline
        # bit-for-bit, and the manifest must say NONE_DECLARED rather than list
        # 112 barriers of zero strength.
        dprof = np.full(x.size, D0)
        brep = None
    else:
        dprof, brep = rc.build_barrier_profile(
            x, D0, hits, float(spec['w_ft']), ratio,
            on_empty=spec['on_empty'], return_report=True)

    sidx = int(np.argmin(np.abs(x - float(spec['source_md_ft']))))
    gidx = [int(np.argmin(np.abs(x - m))) for m in spec['gauge_md']]
    dt_ax = np.asarray(spec['driver_taxis'], dtype=float)
    dt_v = np.asarray(spec['driver_values'], dtype=float)
    dt_s = float(spec['dt_s'])
    t_total = float(dt_ax[-1])
    u0 = np.full(x.size, float(dt_v[0]))

    taxis, trace = rc.solve_forward(
        x, dprof, dt_s, t_total, dt_ax, dt_v, sidx, initial=u0, t0=0.0,
        record_idx=gidx, theta=float(spec['theta']),
        lambda_leak=float(spec['lambda_leak']), p0=float(spec['p0_psi']),
        interface_avg=spec['interface_avg'],
        theta_startup_steps=int(spec['theta_startup_steps']))

    out = {'mesh': x, 'taxis': taxis, 'trace': trace,
           'mesh_record_extra': mrec, 'source_mesh_idx': sidx,
           'gauge_mesh_idx': gidx, 't_total_s': t_total,
           'source_mesh_md_ft': float(x[sidx]),
           'barrier_report': brep, 'wall_s': time.time() - t_wall}

    # --- metrics ----------------------------------------------------------
    per_date = {}
    for label, te_s in spec['eval_times']:
        sim = np.array([np.interp(te_s, taxis, trace[:, k])
                        for k in range(trace.shape[1])])
        dd = trace[0] - sim
        obs = np.asarray(spec['field'][label], dtype=float)
        res = dd - obs
        per_date[label] = {
            'sim_drawdown_psi': dd.tolist(),
            'obs_drawdown_psi': obs.tolist(),
            'residual_psi': res.tolist(),
            'profile_rmse_psi': float(np.sqrt(np.mean(res ** 2))),
            'profile_mae_psi': float(np.mean(np.abs(res))),
            'profile_bias_psi': float(np.mean(res)),
            'profile_max_abs_resid_psi': float(np.max(np.abs(res))),
            'sim_profile_range_psi': float(np.ptp(dd)),
            'obs_profile_range_psi': float(np.ptp(obs)),
        }

    # Time-series misfit, both series referenced to their own first sample, the
    # simulation interpolated onto each observed axis (B2's convention).
    per_gauge, ns, rs = [], [], []
    for k, g in enumerate(spec['gauges']):
        ot = np.asarray(spec['obs_taxis'][k], dtype=float)
        ov = np.asarray(spec['obs_values'][k], dtype=float)
        sim = np.interp(ot, taxis, trace[:, k])
        r = (sim - sim[0]) - (ov - ov[0])
        per_gauge.append({'gauge': int(g), 'md_ft': float(spec['gauge_md'][k]),
                          'mesh_idx': int(gidx[k]),
                          'rmse_psi': float(np.sqrt(np.mean(r ** 2))),
                          'bias_psi': float(np.mean(r)),
                          'max_abs_resid_psi': float(np.max(np.abs(r))),
                          'n_samples': int(ot.size)})
        ns.append(ot.size)
        rs.append(per_gauge[-1]['rmse_psi'])
    ns = np.asarray(ns, dtype=float)
    rs = np.asarray(rs, dtype=float)
    out['metrics'] = {
        'drawdown': per_date,
        'per_gauge_timeseries': per_gauge,
        'pooled_timeseries': {
            'gauge_mean_rmse_psi': float(np.mean(rs)),
            'sample_pooled_rmse_psi': float(np.sqrt(np.sum(ns * rs ** 2)
                                                    / np.sum(ns))),
            'n_gauges': int(rs.size), 'n_samples_total': int(np.sum(ns))},
    }
    out['wall_s'] = time.time() - t_wall
    return out


# ---------------------------------------------------------------------------
# manifest helpers
# ---------------------------------------------------------------------------

def boundary_group():
    return {'lbc': 'Neumann', 'rbc': 'Neumann', 'pml_thickness': 0.0,
            'sigma_max': 0.0,
            'note': ('both ends no-flux, as 103r:111 sets them. sigma is '
                     'identically zero, so the fibeRIS PML diagonal loop -- '
                     'which would corrupt the Neumann and Dirichlet rows -- is '
                     'not exercised')}


def kernel_group():
    return {'module': 'scripts/manuscript_well_leakage/rev2/rev2_core.py',
            'function': 'rev2_core.solve_forward',
            'equivalence': ('theta=1 / harmonic / lambda=0 is bitwise identical '
                            'to r1_calibration_core.solve_forward, which is '
                            'proven bit-equivalent to fibeRIS PDS1D_SingleSource; '
                            'asserted in rev2_selftest T1'),
            'caveat': ('the four adversarial verifiers of the rev2 modules were '
                       'killed by a session limit; the modules are self-tested, '
                       'not independently verified')}


def barrier_records(mesh, report, d_base, ratio, w_ft):
    out = []
    for k, b in enumerate(report['barriers']):
        mask = np.zeros(mesh.size, dtype=bool)
        mask[int(b['i0']):int(b['i1']) + 1] = True
        out.append(rm.barrier_record(
            mesh, mask, label=f"frachit[{k}]",
            centre_md_ft=float(b['md_ft']), w_requested_ft=float(w_ft),
            ratio=float(ratio), d_baseline=float(d_base),
            report={'fallback': bool(b['fallback']),
                    'realised_full_width_ft': float(b['realised_full_width_ft']),
                    'center_offset_ft': float(b['center_offset_ft']),
                    'group': int(b['group'])}))
    return out


def declare_inputs(inp, spec):
    items = [(rd.repo_path(rd.SWELL_GAUGE_MD_NPZ), 'geometry', 'gauge_md_swell'),
             (rd.repo_path(rd.PROD_GAUGE_MD_NPZ), 'geometry', 'gauge_md_prod')]
    for f in inp.fh_files:
        items.append((rd.repo_path(FRAC_HIT_DIR, f), 'geometry',
                      os.path.splitext(f)[0]))
    for g in inp.gauges:
        items.append((rd.repo_path(SWELL_GAUGE.format(n=g)), 'gauge_series',
                      f's_well_gauge{g}'))
    items.append((rd.repo_path(spec['driver_path']), 'gauge_series',
                  f"driver_{spec['driver']}"))
    return items


# ---------------------------------------------------------------------------
# THE LEGEND. Read from the manifest, never hard-coded.
# ---------------------------------------------------------------------------

def label_from_manifest(manifest_path):
    """Build a legend string from what the run's manifest says it actually did.

    This function is the whole point of defect 1. 103p wrote its labels as a
    literal list and indexed them by position in an `os.listdir`; the labels
    therefore described neither the file nor the physics. Here the ratio, the
    barrier diffusivity and the realised physical width come out of
    `numerics.diffusivity` and `numerics.barriers`, which `write_manifest`
    derived from the arrays the solver was handed.
    """
    with open(manifest_path) as fh:
        doc = json.load(fh)
    num = doc['numerics']
    D0 = float(num['diffusivity']['D_ft2_s'])
    bars = num['barriers']
    if not isinstance(bars, list):          # NONE_DECLARED
        return {'label': f"uniform, no barrier ($D = {D0:g}$ ft$^2$/s)",
                'ratio': 1.0, 'D0': D0, 'n_barriers': 0, 'width_ft': 0.0}
    ratio = float(bars[0]['ratio'])
    width = float(np.median([b['full_width_realised_control_volume_ft']
                             for b in bars]))
    return {'label': (f"barrier ratio {ratio:g} "
                      f"($D_b = {ratio * D0:.3g}$ ft$^2$/s, "
                      f"{len(bars)} x {width:.2f} ft)"),
            'ratio': ratio, 'D0': D0, 'n_barriers': len(bars),
            'width_ft': width}


# ---------------------------------------------------------------------------
# spec construction
# ---------------------------------------------------------------------------

def make_specs(cfg, inp, only=None):
    m = cfg['model']
    ph = cfg['physics']
    core = [float(v) for v in m['core_md_ft']]
    eval_times = [(iso[:10],
                   (datetime.datetime.fromisoformat(iso) - inp.t0).total_seconds())
                  for iso in m['drawdown_eval_dates']]
    field = {lbl: inp.field[lbl]['drawdown_psi'] for lbl, _ in eval_times}
    obs_taxis = [inp.obs[g][0].tolist() for g in inp.gauges]
    obs_values = [inp.obs[g][1].tolist() for g in inp.gauges]

    base = dict(core=core, hits=inp.hits.tolist(),
                dx_bg=float(m['dx_background_ft']), dx_mid=float(m['dx_mid_ft']),
                dx_fine=float(m['dx_fine_ft']), r_mid=float(m['r_mid_ft']),
                r_fine=float(m['r_fine_ft']), snap=float(m['breakpoint_snap_ft']),
                D_ft2_s=float(m['D_ft2_s']), w_ft=float(m['barrier']['w_ft']),
                on_empty=m['barrier']['on_empty'], mesh_kind='refined',
                dt_s=float(m['time']['dt_s']), theta=float(ph['theta']),
                theta_startup_steps=int(ph['theta_startup_steps']),
                lambda_leak=float(ph['lambda_leak']), p0_psi=float(ph['p0_psi']),
                interface_avg=ph['interface_avg'],
                gauges=inp.gauges, gauge_md=inp.gauge_md.tolist(),
                obs_taxis=obs_taxis, obs_values=obs_values,
                eval_times=eval_times, field=field,
                driver='prod_g1', source_placement='md_16196')

    def with_driver(s, name):
        d = inp.drivers[name]
        s = dict(s, driver=name, driver_path=d['path'],
                 driver_well=d['well'], driver_gauge=int(d['gauge']),
                 driver_md_ft=float(d['md_ft']), driver_note=d['note'],
                 driver_taxis=d['series'][0].tolist(),
                 driver_values=d['series'][1].tolist())
        return s

    def with_source(s, name):
        p = m['source_placements'][name]
        return dict(s, source_placement=name,
                    source_md_ft=float(p['md_ft']), source_note=p['note'])

    specs = []

    def add(sweep, tag, **kw):
        if only and sweep not in only:
            return
        s = dict(base)
        s.update(kw)
        s = with_driver(s, s['driver'])
        s = with_source(s, s['source_placement'])
        s['sweep'] = sweep
        s['tag'] = tag
        specs.append(s)

    def rtag(r):
        return ('r%g' % r).replace('.', 'p').replace('-', 'm').replace('+', '')

    sw = cfg['sweeps']

    if sw['S1_dt']['enabled']:
        for pad in sw['S1_dt']['pad_ft']:
            for r in sw['S1_dt']['ratios']:
                for dt in sw['S1_dt']['dt_s']:
                    add('S1_dt', f"pad{int(pad):07d}_{rtag(r)}_dt{int(dt):06d}",
                        pad_ft=float(pad), ratio=float(r), dt_s=float(dt))

    if sw['S2_dx']['enabled']:
        for pad in sw['S2_dx']['pad_ft']:
            for r in sw['S2_dx']['ratios']:
                for dxb, dxm, dxf in sw['S2_dx']['meshes']:
                    add('S2_dx',
                        f"pad{int(pad):07d}_{rtag(r)}_dx{dxb:g}_{dxf:g}",
                        pad_ft=float(pad), ratio=float(r), dx_bg=float(dxb),
                        dx_mid=float(dxm), dx_fine=float(dxf),
                        snap=min(float(m['breakpoint_snap_ft']), float(dxf) / 2))

    if sw['S3_pad']['enabled']:
        for pad in sw['S3_pad']['pad_ft']:
            for r in sw['S3_pad']['ratios']:
                add('S3_pad', f"pad{int(pad):07d}_{rtag(r)}",
                    pad_ft=float(pad), ratio=float(r))

    if sw['S4_ratio']['enabled']:
        for pad in sw['S4_ratio']['pad_ft']:
            for r in sw['S4_ratio']['ratios']:
                add('S4_ratio', f"pad{int(pad):07d}_{rtag(r)}",
                    pad_ft=float(pad), ratio=float(r))

    if sw['S5_source']['enabled']:
        for pad in sw['S5_source']['pad_ft']:
            for drv, src in sw['S5_source']['variants']:
                for r in sw['S5_source']['ratios']:
                    add('S5_source',
                        f"pad{int(pad):07d}_{drv}_{src}_{rtag(r)}",
                        pad_ft=float(pad), ratio=float(r), driver=drv,
                        source_placement=src)

    if sw['S6_legacy']['enabled']:
        for pad in sw['S6_legacy']['pad_ft']:
            for r in sw['S6_legacy']['ratios']:
                add('S6_legacy', f"pad{int(pad):07d}_{rtag(r)}_legacybarrier",
                    pad_ft=float(pad), ratio=float(r), mesh_kind='uniform',
                    w_ft=0.0, on_empty='nearest')

    if sw['S7_D']['enabled']:
        for pad in sw['S7_D']['pad_ft']:
            for r in sw['S7_D']['ratios']:
                add('S7_D', f"pad{int(pad):07d}_{rtag(r)}_D1150",
                    pad_ft=float(pad), ratio=float(r),
                    D_ft2_s=float(sw['S7_D']['D_ft2_s']))

    return specs


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_all(cfg, inp, specs, outdir, config_path, processes):
    todo = []
    for s in specs:
        rundir = os.path.join(outdir, s['sweep'], s['tag'])
        mpath = os.path.join(rundir, 'manifest.json')
        rm.assert_absent([mpath])
        todo.append((s, rundir, mpath))

    results = []
    n = len(todo)
    if processes > 1:
        pool = mp.Pool(processes)
        it = pool.imap(solve_spec, [t[0] for t in todo])
    else:
        pool = None
        it = (solve_spec(t[0]) for t in todo)
    try:
        for k, res in enumerate(it):
            s, rundir, mpath = todo[k]
            results.append(write_run(cfg, inp, s, res, rundir, mpath,
                                     config_path))
            r = results[-1]
            print(f"  [{k + 1}/{n}] {s['sweep']}/{s['tag']}: nx={r['nx']} "
                  f"steps={r['n_steps']} "
                  f"profileRMSE(2mo)={r['drawdown']['2020-06-01']['profile_rmse_psi']:.1f} psi "
                  f"({r['wall_s']:.1f} s)", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return results


def write_run(cfg, inp, spec, res, rundir, mpath, config_path):
    os.makedirs(rundir, exist_ok=True)
    x = res['mesh']
    taxis = res['taxis']
    npz = os.path.join(rundir, 'summary.npz')
    np.savez_compressed(
        npz, mesh_md_ft=x, taxis_s=taxis, traces_psi=res['trace'],
        gauge_numbers=np.array(spec['gauges']),
        gauge_md_ft=np.asarray(spec['gauge_md']),
        gauge_mesh_idx=np.asarray(res['gauge_mesh_idx']),
        source_mesh_idx=np.array([res['source_mesh_idx']]),
        driver_taxis_s=np.asarray(spec['driver_taxis']),
        driver_psi=np.asarray(spec['driver_values']))

    with rm.RunRecorder(mpath, study_id=cfg['study_id'], task_id=cfg['task_id'],
                        config=cfg, config_path=config_path,
                        run_label=f"{spec['sweep']}/{spec['tag']}",
                        require_modules=('rev2_core', 'rev2_data')) as rec:
        drec = rm.driver_record(
            kind='gauge_series', baseline_removal='none_absolute_psi',
            value_units='psi', series_path=rd.repo_path(spec['driver_path']),
            gauge_number=spec['driver_gauge'], gauge_md_ft=spec['driver_md_ft'],
            taxis=np.asarray(spec['driver_taxis']),
            values=np.asarray(spec['driver_values']),
            time_start=inp.t0.isoformat(), time_end=inp.t1.isoformat())
        rec.declare_inputs(declare_inputs(inp, spec))
        rec.declare_output(npz, role='arrays_npz',
                           note='mesh, time axis, simulated traces at the fifteen '
                                'S-well gauge MDs, and the Dirichlet driver')
        rec.set_source(rm.source_protocol(
            application='dirichlet_node',
            solver_class='rev2_core.solve_forward',
            placement_rule=(f"pinned to physical MD {spec['source_md_ft']:.1f} ft "
                            f"({spec['source_placement']}); NOT the legacy "
                            f"(mesh[0]+mesh[-1])/2 rule, which moves with the pad"),
            sources=[rm.source_record(x, md_requested_ft=spec['source_md_ft'],
                                      mesh_idx=res['source_mesh_idx'],
                                      driver=drec, label='prod_dirichlet',
                                      index_in_source_list=0)],
            targets=[{'gauge': int(g), 'md_ft': float(md), 'mesh_idx': int(i),
                      'well': 's_well'}
                     for g, md, i in zip(spec['gauges'], spec['gauge_md'],
                                         res['gauge_mesh_idx'])],
            time_level='n', phase_chaining='none',
            boundary_conditions=boundary_group()))

        if res['barrier_report'] is None:
            bars = rm.NONE_DECLARED
            brep_summary = {'n_barriers': 0,
                            'note': 'ratio = 1.0: the uniform control, no barrier'}
        else:
            bars = barrier_records(x, res['barrier_report'],
                                   spec['D_ft2_s'], spec['ratio'], spec['w_ft'])
            b = res['barrier_report']
            brep_summary = {k: b[k] for k in
                            ('n_barriers', 'n_merged_groups', 'n_fallback',
                             'n_touching_domain_edge',
                             'total_equivalent_width_ft',
                             'excess_resistance_s_per_ft')}
        rec.set_numerics(rm.numerics(
            time=rm.time_record(taxis, mode='fixed', theta=spec['theta'],
                                t_total_requested_s=res['t_total_s'],
                                dt_requested_s=spec['dt_s'],
                                source_time_level='n',
                                theta_startup_steps=spec['theta_startup_steps'],
                                label='production_forward'),
            mesh=rm.mesh_record(x, dx_requested_ft=spec['dx_bg'],
                                window_md_ft=spec['core'],
                                pad_low_ft=spec['pad_ft'],
                                pad_high_ft=spec['pad_ft'],
                                refinement=res['mesh_record_extra']),
            interface_avg=spec['interface_avg'], boundary=boundary_group(),
            diffusivity={'family': 'uniform_with_frac_hit_barriers',
                         'D_ft2_s': spec['D_ft2_s']},
            barriers=bars,
            leakage={'lambda_leak': spec['lambda_leak'], 'p0_psi': spec['p0_psi']},
            kernel=kernel_group(), rng=rm.NONE_DECLARED,
            parallel={'processes': int(cfg.get('processes', 1)),
                      'note': 'solve in a worker process; manifest written in the '
                              'parent, so no two processes touch one file'}))
        out = {
            'sweep': spec['sweep'], 'tag': spec['tag'],
            'pad_ft': spec['pad_ft'], 'ratio': spec['ratio'],
            'D_ft2_s': spec['D_ft2_s'], 'w_ft': spec['w_ft'],
            'mesh_kind': spec['mesh_kind'], 'dt_s': spec['dt_s'],
            'dx_bg_ft': spec['dx_bg'], 'dx_fine_ft': spec['dx_fine'],
            'driver': spec['driver'], 'driver_gauge': spec['driver_gauge'],
            'driver_md_ft': spec['driver_md_ft'],
            'source_placement': spec['source_placement'],
            'source_md_requested_ft': spec['source_md_ft'],
            'source_md_realised_ft': res['source_mesh_md_ft'],
            'nx': int(x.size), 'n_steps': int(taxis.size - 1),
            'mesh_span_md_ft': [float(x[0]), float(x[-1])],
            'barrier_report': brep_summary,
            'wall_s': res['wall_s'],
        }
        out.update(res['metrics'])
        rec.set_results(out)
        rec.note(f"driver: {spec['driver_note']}")
        rec.note(f"source: {spec['source_note']}")
    out['manifest'] = os.path.relpath(mpath, rd.REPO_ROOT)
    out['summary_npz'] = os.path.relpath(npz, rd.REPO_ROOT)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/rev2/e2_fig7b.json')
    ap.add_argument('--sweep', action='append', default=None)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--processes', type=int, default=None)
    ap.add_argument('--figures-only', action='store_true')
    args = ap.parse_args(argv)

    config_path = os.path.abspath(args.config)
    with open(config_path) as fh:
        cfg = json.load(fh)
    outdir = args.outdir or rd.repo_path(cfg['output_root'])
    os.makedirs(outdir, exist_ok=True)
    nproc = args.processes if args.processes is not None \
        else int(cfg.get('processes', 1))

    t0 = time.time()
    inp = Inputs(cfg)
    print(f"inputs: {inp.hits.size} unique frac hits "
          f"({inp.hits_raw.size} raw), {len(inp.gauges)} S-well gauges, "
          f"drivers {sorted(inp.drivers)}", flush=True)
    for lbl, f in inp.field.items():
        ref = rd.DRAWDOWN_REFERENCE.get(('2020-04-01', lbl))
        ok = ('n/a' if ref is None else
              'MATCHES rev2_data.DRAWDOWN_REFERENCE'
              if np.allclose(f['drawdown_psi'], ref, atol=5e-3)
              else 'DIFFERS from rev2_data.DRAWDOWN_REFERENCE')
        print(f"  field drawdown to {lbl}: {ok}", flush=True)

    results_path = os.path.join(outdir, 'e2_runs.json')
    if args.figures_only:
        with open(results_path) as fh:
            results = json.load(fh)['runs']
    else:
        specs = make_specs(cfg, inp, only=set(args.sweep) if args.sweep else None)
        print(f"{len(specs)} runs queued on {nproc} process(es)", flush=True)
        results = run_all(cfg, inp, specs, outdir, config_path, nproc)
        blob = {'study_id': cfg['study_id'], 'task_id': cfg['task_id'],
                'round_tag': cfg['round_tag'],
                'run_utc': datetime.datetime.now(
                    datetime.timezone.utc).isoformat(),
                'config_path': os.path.relpath(config_path, rd.REPO_ROOT),
                'n_runs': len(results),
                'wall_s': time.time() - t0,
                'field_drawdown': inp.field,
                'listdir_order_today': inp.listdir_order,
                'runs': results}
        with open(results_path, 'w') as fh:
            json.dump(blob, fh, indent=2, sort_keys=True)
        print(f"wrote {results_path}", flush=True)
    print(f"wall {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
