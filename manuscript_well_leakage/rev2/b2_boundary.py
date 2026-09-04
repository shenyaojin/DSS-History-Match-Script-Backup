"""B2 -- boundary contamination of the three legacy computational domains.

THE DELIVERABLE IS A DECISION: for each manuscript figure, is the archived result
affected by the finite no-flux domain, by how much, and must it be re-run?

The task package's phrase "the manuscript 6000 ft domain" does not correspond to
anything in the code. Three legacy scripts use three different domains and are
reported separately here:

  101_fiberis_matching.py   MD 12500-17999, dx 1 ft   -> Fig. 6 (two-stage)
  102r_..._QC_swell.py      MD  9187-21357, dx 10 ft  -> production QC (S well)
  103r_..._QC_swell.py      MD 11890-16880, dx 10 ft  -> production QC (producer)

Each is run first exactly as it stands, then with low-MD padding of 2000 / 5000 /
8000 / 20000 ft, and SEPARATELY with high-MD padding of the same sizes. The
high-end sweep exists because the established claim -- "no high-end pad is needed,
the Dirichlet source node decouples everything above it" -- was established on the
R1 window, where the source is the topmost node and no target gauge lies above it.
In 101 the sources sit in the MIDDLE of the domain and gauges lie on BOTH sides, so
the claim has to be verified rather than assumed. It is verified here, and its
scope turns out to be narrower than the sentence suggests (see the README).

Nothing in this file edits a shared module. `rev2_core`, `rev2_data` and
`rev2_manifest` are imported; `mesh_utils.refine_mesh` and the legacy scripts'
arithmetic are reproduced, defects included, because the question is what the
ARCHIVED numbers are worth.

Run:
    python3 scripts/manuscript_well_leakage/rev2/b2_boundary.py \
        --config configs/rev2/b2_boundary.json
Add --case r1_reference|c101|c102r|c103r (repeatable) to run a subset, --outdir to
write elsewhere. The script never overwrites: every manifest path is guarded by
rev2_manifest.assert_absent.
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
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.join(os.path.dirname(_HERE), 'baseline_calibration')
for _p in (_HERE, _BASE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rev2_core as rc          # noqa: E402
import rev2_data as rd          # noqa: E402
import rev2_manifest as rm      # noqa: E402
from r1_calibration_core import arrival_time  # noqa: E402
from fiberis.utils import mesh_utils          # noqa: E402
from fiberis.analyzer.Data1D import Data1D_Gauge  # noqa: E402

FRAC_HIT_DIR = 'data/legacy/s_well/geometry/frac_hit/'
SWELL_GAUGE = 'data/fiberis_format/s_well/gauges/gauge{n}_data_swell.npz'
PROD_GAUGE = 'data/fiberis_format/prod/gauges/gauge{n}_data_prod.npz'
# The legacy copy (data/legacy/prod/geometry/gauge_md_prod.npz) stores its
# trajectory under ew/ns/tvd and is rejected by the current G3D loader; the
# fiberis_format copy carries the identical `data` array (16196, 15197, 14198,
# 13199, 12200 ft) under xaxis/yaxis/zaxis and is the one used here.
PROD_MD_NPZ = rd.PROD_GAUGE_MD_NPZ

# 102r/103r place a barrier on every one of these; the requested half-width is 0,
# so rev2_core falls back to the nearest node and warns 113 times per build.
warnings.simplefilter('ignore', rc.BarrierWidthWarning)
warnings.simplefilter('ignore', rc.BarrierOverlapWarning)


# ---------------------------------------------------------------------------
# inputs, cached
# ---------------------------------------------------------------------------

class Inputs:
    """Everything read from disk, loaded once. Also records provenance facts that
    a later reader would otherwise have to re-derive (os.listdir orders, the
    hard-zero dropout in the producer gauges, the frac-hit duplicate)."""

    def __init__(self):
        self.md_table = rd.load_gauge_md_table()
        self.prod_md_table = rd.load_gauge_md_table(PROD_MD_NPZ,
                                                    check_reference=False)
        self.fh_listdir = sorted(os.listdir(rd.repo_path(FRAC_HIT_DIR)))
        self.fh_listdir_raw = os.listdir(rd.repo_path(FRAC_HIT_DIR))
        hits = []
        for f in self.fh_listdir_raw:
            z = np.load(rd.repo_path(FRAC_HIT_DIR, f), allow_pickle=True)
            try:
                hits.append(np.atleast_1d(np.asarray(z['data'], dtype=float)))
            finally:
                z.close()
        self.all_hits = np.concatenate(hits)
        self.fh7 = rd.load_frac_hits(7, unique=False, sort=False)
        self.fh8 = rd.load_frac_hits(8, unique=False, sort=False)
        self.phase_windows = rd.manuscript_phase_windows()
        self._series = {}
        self._phase_src = {}

    # -- production-window observations -------------------------------------
    def gauge_window(self, well, n, t0, t1):
        key = (well, int(n), t0.isoformat(), t1.isoformat())
        if key not in self._series:
            tmpl = SWELL_GAUGE if well == 's_well' else PROD_GAUGE
            f = Data1D_Gauge.Data1DGauge()
            f.load_npz(rd.repo_path(tmpl.format(n=int(n))))
            f.crop(t0, t1)
            t = np.asarray(f.taxis, dtype=float)
            v = np.asarray(f.data, dtype=float)
            if t.size == 0:
                raise ValueError(f"{well} gauge {n}: crop to [{t0}, {t1}] "
                                 f"returned no samples")
            self._series[key] = (t, v)
        return self._series[key]

    # -- 101 phase drivers ---------------------------------------------------
    def phase_source(self, phase):
        """Absolute-psi Dirichlet datum for one 101 phase, on the LEGACY time base.

        `crop_rebase='requested_start'`: fibeRIS's Data1D.crop rebased the cropped
        axis to the requested start in Feb 2025 and rebases it to the first
        in-window sample today (core1D.py:158-160). A5 identified the old
        convention from the archived start_time values; reproducing the archive
        needs it, and it is applied here as a pure shift.
        """
        if phase not in self._phase_src:
            gauge, t0, t1 = self.phase_windows[phase]
            win = rd.Window(md_min_ft=0.0, md_max_ft=1e9, t_start=t0, t_end=t1)
            gw = rd.load_window_gauges(win, gauges=[int(gauge)], baseline='none',
                                       rebase='per_gauge')
            s = gw.series[int(gauge)]
            lead = (s.t0_abs - t0).total_seconds()
            self._phase_src[phase] = {
                'gauge': int(gauge), 'md_ft': float(s.md_ft),
                'taxis_s': np.asarray(s.taxis_s, dtype=float) + lead,
                'values_psi': np.asarray(s.raw_psi, dtype=float),
                't0_requested': t0, 't1_requested': t1,
                'first_sample_abs': s.t0_abs, 'lead_s': float(lead),
                'series_path': SWELL_GAUGE.format(n=int(gauge)),
            }
        return self._phase_src[phase]


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------

def _legacy_refine(x, centres, half_width, factor, round_centres=False):
    """fiberis.utils.mesh_utils.refine_mesh, called exactly as the legacy scripts
    call it, with the per-call record A1 needs to be reproducible.

    The defect (kept): the inserted point count is
    (end_idx - start_idx)*factor + 1 where end_idx-start_idx is a NODE count, not
    an interval count. On a 1 ft mesh over +-1 ft that is 3 -> 16 points across
    2 ft -> dx = 2/15 ft, not the 0.2 ft that "factor 5" implies. On a 10 ft mesh
    the window usually contains NO node, the count collapses to 1, nothing is
    refined, and the single inserted point lands at centre-1 ft.
    """
    calls = []
    for c in centres:
        c = float(np.round(c)) if round_centres else float(c)
        s, e = c - half_width, c + half_width
        i0 = int(np.searchsorted(x, s, side='left'))
        i1 = int(np.searchsorted(x, e, side='right'))
        n_in = i1 - i0
        n_pts = n_in * factor + 1
        before = len(x)
        x = mesh_utils.refine_mesh(x, [s, e], factor)
        calls.append({'centre_md_ft': c, 'nodes_in_range_before': int(n_in),
                      'n_points_inserted': int(n_pts),
                      'degenerate': bool(n_in == 0),
                      'nx_before': int(before), 'nx_after': int(len(x))})
    return np.asarray(x, dtype=float), calls


def build_mesh_101(inp, pad_low, pad_high, cfg):
    lo = float(cfg['md_lo_ft']) - float(pad_low)
    hi = float(cfg['md_hi_ft']) + float(pad_high)
    x = rd.build_mesh((lo, hi), 0.0, 0.0, float(cfg['dx_ft'])).x
    rcfg = cfg['refine']
    calls = []
    for stage in rcfg['stages']:
        hits = inp.fh7 if int(stage) == 7 else inp.fh8
        x, c = _legacy_refine(x, hits, float(rcfg['half_width_ft']),
                              int(rcfg['factor']),
                              round_centres=bool(rcfg['round_centres']))
        for e in c:
            e['stage'] = int(stage)
        calls.extend(c)
    return x, {'mode': rcfg['mode'], 'n_calls': len(calls),
               'n_degenerate_calls': int(sum(c['degenerate'] for c in calls)),
               'calls': calls}


def build_mesh_prod(inp, pad_low, pad_high, cfg):
    if cfg['mesh_from'] == 'frac_hit_stage7_pm6000':
        lo = float(np.min(inp.fh7)) - 6000.0 - float(pad_low)
        hi = float(np.max(inp.fh7)) + 6000.0 + float(pad_high)
    elif cfg['mesh_from'] == 'fixed_11890_16890':
        lo = 11890.0 - float(pad_low)
        hi = 16890.0 + float(pad_high)
    else:
        raise ValueError(f"unknown mesh_from {cfg['mesh_from']!r}")
    # np.arange, verbatim: the legacy scripts build the base mesh this way and it
    # stops one dx BELOW hi, which is why 102r's domain ends at 21356.83 and not
    # 21364.29 and 103r's at 16880 and not 16890.
    x = np.arange(lo, hi, float(cfg['dx_ft']))
    rcfg = cfg['refine']
    x, calls = _legacy_refine(x, inp.all_hits, float(rcfg['half_width_ft']),
                              int(rcfg['factor']), round_centres=False)
    return x, {'mode': 'legacy_refine_mesh', 'n_calls': len(calls),
               'n_degenerate_calls': int(sum(c['degenerate'] for c in calls)),
               'base_arange': [lo, hi, float(cfg['dx_ft'])],
               'calls': calls}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def gauge_metrics(sim_taxis, sim_trace, obs_taxis, obs_raw, *,
                  thr_frac=0.1, thr_abs=10.0):
    """One gauge, one run. Both series referenced to their own first sample.

    The simulation is interpolated onto the OBSERVED time axis, never the other
    way round: the observed axis is the one the manuscript figures are drawn on,
    and interpolating the data instead would let a run with a coarser time grid
    look better simply by smoothing the target.
    """
    sim = np.interp(obs_taxis, sim_taxis, sim_trace)
    sdp = sim - sim[0]
    odp = obs_raw - obs_raw[0]
    res = sdp - odp
    i = int(np.argmax(np.abs(sdp)))
    j = int(np.argmax(np.abs(odp)))
    thr_s = thr_frac * float(np.max(np.abs(odp)))
    out = {
        'peak_abs_dp_psi': float(np.abs(sdp[i])),
        'peak_signed_dp_psi': float(sdp[i]),
        't_peak_s': float(obs_taxis[i]),
        'obs_peak_abs_dp_psi': float(np.abs(odp[j])),
        'obs_peak_signed_dp_psi': float(odp[j]),
        'obs_t_peak_s': float(obs_taxis[j]),
        'end_dp_psi': float(sdp[-1]),
        'obs_end_dp_psi': float(odp[-1]),
        'rmse_psi': float(np.sqrt(np.mean(res ** 2))),
        'bias_psi': float(np.mean(res)),
        'max_abs_resid_psi': float(np.max(np.abs(res))),
        'n_samples': int(obs_taxis.size),
        'arrival_rel_sim_s': arrival_time(obs_taxis, np.abs(sdp), thr_s),
        'arrival_rel_obs_s': arrival_time(obs_taxis, np.abs(odp), thr_s),
        'arrival_abs_sim_s': arrival_time(obs_taxis, np.abs(sdp), thr_abs),
        'arrival_abs_obs_s': arrival_time(obs_taxis, np.abs(odp), thr_abs),
        'arrival_threshold_rel_psi': float(thr_s),
        'arrival_threshold_abs_psi': float(thr_abs),
    }
    for k in ('rel', 'abs'):
        a, b = out[f'arrival_{k}_sim_s'], out[f'arrival_{k}_obs_s']
        out[f'arrival_{k}_lag_s'] = (float(a - b) if np.isfinite(a) and
                                     np.isfinite(b) else float('nan'))
    return out


def pooled(per_gauge):
    """Both misfit conventions, named. C1 measured that the gauge-mean and the
    sample-pooled RMSE differ by 5.1-5.5% on this data, so quoting one unnamed is
    not reproducible."""
    n = np.array([g['n_samples'] for g in per_gauge], dtype=float)
    r = np.array([g['rmse_psi'] for g in per_gauge], dtype=float)
    return {'gauge_mean_rmse_psi': float(np.mean(r)),
            'sample_pooled_rmse_psi': float(np.sqrt(np.sum(n * r ** 2) / np.sum(n))),
            'n_gauges': int(r.size), 'n_samples_total': int(np.sum(n))}


# ---------------------------------------------------------------------------
# manifest helpers
# ---------------------------------------------------------------------------

def boundary_group():
    return {'lbc': 'Neumann', 'rbc': 'Neumann', 'pml_thickness': 0.0,
            'sigma_max': 0.0,
            'note': ('both ends no-flux, as every legacy script sets them; the '
                     'PML diagonal loop in fibeRIS matbuilder is not exercised '
                     'because sigma is identically zero')}


def kernel_group(solver):
    return {'module': 'scripts/manuscript_well_leakage/rev2/rev2_core.py',
            'function': solver,
            'equivalence': ('theta=1 / harmonic / lambda=0 is bitwise identical '
                            'to r1_calibration_core.solve_forward, which is '
                            'proven bit-equivalent to fibeRIS; verified in '
                            'rev2_selftest T1 (60/60)'),
            'caveat': ('the four adversarial verifiers of the rev2 modules were '
                       'killed by a session limit and never ran; the modules are '
                       'self-tested, not independently verified')}


def barrier_records(mesh, report, d_base, ratio, label_prefix):
    """One barrier_record per applied barrier, built from the node span the
    profile builder actually reduced -- never from a mask this script re-derives.

    `report['barriers']` is the per-frac-hit list; `report['merged_groups']` is the
    overlap-merged view whose total width is the one that may be summed. Both are
    carried: the first as barrier_records, the second in the run results.
    """
    out = []
    for k, b in enumerate(report['barriers']):
        mask = np.zeros(mesh.size, dtype=bool)
        mask[int(b['i0']):int(b['i1']) + 1] = True
        out.append(rm.barrier_record(
            mesh, mask, label=f"{label_prefix}[{k}]",
            centre_md_ft=float(b['md_ft']), w_requested_ft=float(
                report['w_requested_ft']),
            ratio=float(ratio), d_baseline=float(d_base),
            report={'fallback': bool(b['fallback']),
                    'realised_full_width_ft': float(b['realised_full_width_ft']),
                    'center_offset_ft': float(b['center_offset_ft']),
                    'group': int(b['group'])}))
    return out


def declare_common_inputs(inp, case, cfg):
    items = [(rd.repo_path(rd.SWELL_GAUGE_MD_NPZ), 'geometry', 'gauge_md_swell')]
    if 'prod' in (cfg.get('well'), cfg.get('target_well')):
        items.append((rd.repo_path(PROD_MD_NPZ), 'geometry', 'gauge_md_prod'))
    if cfg['kind'] == 'prod_forward':
        for f in sorted(inp.fh_listdir_raw):
            items.append((rd.repo_path(FRAC_HIT_DIR, f), 'geometry',
                          os.path.splitext(f)[0]))
    else:
        for st in (7, 8):
            items.append((rd.repo_path(rd.SWELL_FRAC_HIT_TEMPLATE.format(stage=st)),
                          'geometry', f'frac_hit_stage_{st}'))
            items.append((rd.repo_path(FRAC_HIT_DIR,
                                       f'frac_hit_stage_{st}_swell.npz'),
                          'geometry', f'legacy_frac_hit_stage_{st}'))
    return items


# ---------------------------------------------------------------------------
# case: r1_reference (harness validation)
# ---------------------------------------------------------------------------

def run_r1_reference(inp, cfg, root_cfg, outdir, config_path):
    results = []
    win = rd.Window(md_min_ft=cfg['window_md_ft'][0],
                    md_max_ft=cfg['window_md_ft'][1],
                    t_start=rd.R1_WINDOW.t_start, t_end=rd.R1_WINDOW.t_end)
    for pad_low, pad_high in cfg['pads']:
        tag = f"pad_lo{int(pad_low):05d}_hi{int(pad_high):05d}"
        rundir = os.path.join(outdir, tag)
        mpath = os.path.join(rundir, 'manifest.json')
        rm.assert_absent([mpath])
        os.makedirs(rundir, exist_ok=True)
        t_wall = time.time()
        with rm.RunRecorder(mpath, study_id=root_cfg['study_id'] + ':r1_reference',
                            task_id='B2', config=cfg, config_path=config_path,
                            run_label=tag,
                            require_modules=('rev2_core', 'rev2_data')) as rec:
            s = rd.setup_r1(win, pad_low_ft=float(pad_low),
                            pad_high_ft=float(pad_high), dx_ft=float(cfg['dx_ft']))
            mesh = s['mesh']
            D = np.full(mesh.nx, float(cfg['D_ft2_s']))
            tgts = s['targets']
            taxis, trace = rc.solve_forward(
                mesh.x, D, float(cfg['dt_s']), s['t_total_s'],
                s['src_series'].taxis_s, s['src_series'].delta_psi,
                s['source_idx'], record_idx=[t['idx'] for t in tgts])
            per = []
            for k, t in enumerate(tgts):
                m = gauge_metrics(taxis, trace[:, k], t['taxis'], t['data'])
                m.update(gauge=int(t['gauge']), md_ft=float(t['md_ft']),
                         distance_ft=float(t['distance_ft']))
                per.append(m)
            npz = os.path.join(rundir, 'summary.npz')
            np.savez_compressed(
                npz, taxis_s=taxis, traces_psi=trace,
                gauge_numbers=np.array([t['gauge'] for t in tgts]),
                gauge_md_ft=np.array([t['md_ft'] for t in tgts]),
                mesh_md_ft=mesh.x)
            rec.declare_inputs(
                [(rd.repo_path(rd.SWELL_GAUGE_TEMPLATE.format(n=n)), 'gauge_series',
                  f'gauge{n}') for n in s['gauge_window'].numbers]
                + [(rd.repo_path(rd.SWELL_GAUGE_MD_NPZ), 'geometry', 'gauge_md_swell'),
                   (rd.repo_path(rd.SWELL_FRAC_HIT_TEMPLATE.format(stage=1)),
                    'geometry', 'frac_hit_stage_1')])
            rec.declare_output(npz, role='arrays_npz',
                               note='sim traces at the six target gauges')
            rec.set_source(rm.source_protocol(
                application='dirichlet_node',
                solver_class='rev2_core.solve_forward',
                placement_rule='source gauge node (r1 protocol, gauge 1 MD 16645)',
                sources=[rm.source_record(
                    mesh.x, md_requested_ft=s['src_md'],
                    mesh_idx=s['source_idx'],
                    driver=rm.driver_record(
                        kind='gauge_series',
                        baseline_removal='subtract_first_sample',
                        value_units='delta_psi',
                        series_path=rd.repo_path(
                            rd.SWELL_GAUGE_TEMPLATE.format(n=s['src_gauge'])),
                        gauge_number=s['src_gauge'],
                        gauge_md_ft=s['src_series'].md_ft,
                        taxis=s['src_series'].taxis_s,
                        values=s['src_series'].delta_psi),
                    label='r1_source')],
                targets=[{'gauge': int(t['gauge']), 'md_ft': float(t['md_ft']),
                          'mesh_idx': int(t['idx'])} for t in tgts],
                time_level='n', phase_chaining='none',
                boundary_conditions=boundary_group()))
            rec.set_numerics(rm.numerics(
                time=rm.time_record(taxis, mode='fixed', theta=1.0,
                                    t_total_requested_s=s['t_total_s'],
                                    dt_requested_s=float(cfg['dt_s']),
                                    source_time_level='n', label='r1'),
                mesh=rm.mesh_record(mesh.x, dx_requested_ft=float(cfg['dx_ft']),
                                    window_md_ft=cfg['window_md_ft'],
                                    pad_low_ft=float(pad_low),
                                    pad_high_ft=float(pad_high),
                                    refinement={'mode': 'none'}),
                interface_avg='harmonic', boundary=boundary_group(),
                diffusivity={'family': 'uniform', 'D_ft2_s': float(cfg['D_ft2_s'])},
                barriers=rm.NONE_DECLARED,
                leakage={'lambda_leak': 0.0, 'p0_psi': 0.0},
                kernel=kernel_group('solve_forward'),
                rng=rm.NONE_DECLARED,
                parallel={'processes': 1, 'note': 'serial'}))
            res = {'pad_low_ft': float(pad_low), 'pad_high_ft': float(pad_high),
                   'nx': int(mesh.nx), 'per_gauge': per, 'pooled': pooled(per),
                   'wall_s': time.time() - t_wall}
            rec.set_results(res)
            rec.note('Harness validation against the established R1 padding '
                     'result, not a legacy script. Expected: g7 peak ~490 -> ~260 '
                     'psi and sample-pooled RMSE 161 -> 100 as the low-end pad '
                     'goes 0 -> 5000 ft at D = 2000 ft^2/s.')
        res['manifest'] = os.path.relpath(mpath, rd.REPO_ROOT)
        results.append(res)
        print(f"  r1_reference {tag}: nx={res['nx']} pooled="
              f"{res['pooled']['sample_pooled_rmse_psi']:.2f} psi "
              f"g7peak={per[-1]['peak_abs_dp_psi']:.1f} psi")
    return results


# ---------------------------------------------------------------------------
# case: c101 (the two-stage chain)
# ---------------------------------------------------------------------------

def solve_one(mesh, dprof, taxes, vals, idx, initial, t_total, tmode,
              record_idx=None):
    common = dict(theta=1.0, lambda_leak=0.0, p0=0.0, interface_avg='harmonic')
    if tmode['mode'] == 'fixed':
        dt = float(tmode['dt_fixed_s'])
        taxis, out = rc.solve_forward_multi(mesh, dprof, dt, t_total, taxes, vals,
                                            idx, initial=initial, t0=0.0,
                                            record_idx=record_idx, **common)
        trec = dict(mode='fixed', theta=1.0, t_total_requested_s=t_total,
                    dt_requested_s=dt, source_time_level='n')
        extra = {'solver': 'rev2_core.solve_forward_multi', 'n_rejected': 0}
    else:
        taxis, out, tr = rc.solve_forward_adaptive(
            mesh, dprof, t_total, taxes, vals, idx, initial=initial, t0=0.0,
            record_idx=record_idx, dt_init=float(tmode['dt_init_s']),
            tol=float(tmode['tol']), controller_tol=float(tmode['controller_tol']),
            safety_factor=float(tmode['safety_factor']),
            order_p=int(tmode['order_p']), max_dt=float(tmode['max_dt_s']),
            min_dt=float(tmode['min_dt_s']),
            zero_field_policy=tmode['zero_field_policy'], **common)
        trec = dict(mode='adaptive', theta=1.0, t_total_requested_s=t_total,
                    dt_init_s=tr['dt_init_s'], tol=tr['tol'],
                    controller_tol=tr['controller_tol'], max_dt_s=tr['max_dt_s'],
                    min_dt_s=tr['min_dt_s'], safety_factor=tr['safety_factor'],
                    order_p=tr['order_p'], n_steps_rejected=tr['n_rejected'],
                    zero_field_policy=tr['zero_field_policy'],
                    flip_margin=tr['flip_margin'], source_time_level='n')
        extra = {'solver': 'rev2_core.solve_forward_adaptive',
                 'n_attempts': tr['n_attempts'], 'n_rejected': tr['n_rejected'],
                 'frac_at_max_dt': tr['frac_at_max_dt'],
                 'err_max': tr['err_max'], 'flip_margin': tr['flip_margin'],
                 'overshoot_s': tr['overshoot_s']}
    return taxis, out, trec, extra


def run_c101(inp, cfg, root_cfg, outdir, config_path):
    gauges = [int(g) for g in cfg['targets']['gauges']]
    gauge_md = [inp.md_table.md_of(g) for g in gauges]
    obs = {}
    for g in gauges:
        obs[g] = {}
        for ph in ('phase1', 'phase2', 'phase3'):
            _, t0, t1 = inp.phase_windows[ph]
            obs[g][ph] = inp.gauge_window('s_well', g, t0, t1)
    src = {ph: inp.phase_source(ph) for ph in ('phase1', 'phase2', 'phase3')}
    D0 = float(cfg['D_ft2_s'])
    bcfg = cfg['barrier']
    results = []

    specs = []
    for block in cfg['runs']:
        for pad in block['pads']:
            for ratio in block['ratios']:
                specs.append((block['time_mode'], float(pad[0]), float(pad[1]),
                              float(ratio)))

    for tmode_name, pad_low, pad_high, ratio in specs:
        tmode = cfg['time_modes'][tmode_name]
        rtag = ('r%g' % ratio).replace('.', 'p').replace('-', 'm')
        tag = f"{tmode_name}_lo{int(pad_low):05d}_hi{int(pad_high):05d}_{rtag}"
        rundir = os.path.join(outdir, tag)
        mpath = os.path.join(rundir, 'manifest.json')
        rm.assert_absent([mpath])
        os.makedirs(rundir, exist_ok=True)
        t_wall = time.time()
        with rm.RunRecorder(mpath, study_id=root_cfg['study_id'] + ':c101',
                            task_id='B2', config=cfg, config_path=config_path,
                            run_label=tag,
                            require_modules=('rev2_core', 'rev2_data')) as rec:
            x, mrec = build_mesh_101(inp, pad_low, pad_high, cfg)
            i7 = [int(mesh_utils.locate(x, float(f))[0]) for f in inp.fh7]
            i8 = [int(mesh_utils.locate(x, float(f))[0]) for f in inp.fh8]
            gidx = [int(mesh_utils.locate(x, m)[0]) for m in gauge_md]
            uni = np.full(x.size, D0)
            dbar, brep = rc.build_barrier_profile(
                x, D0, inp.fh7, float(bcfg['w_ft']), ratio,
                ratio_reference=bcfg['ratio_reference'], combine=bcfg['combine'],
                on_empty=bcfg['on_empty'], return_report=True)
            if brep['n_skipped_outside'] or len(brep['barriers']) != len(inp.fh7):
                raise RuntimeError(
                    "barrier build did not apply one barrier per stage-7 frac "
                    f"hit: {brep['n_barriers']} barriers, "
                    f"{brep['n_skipped_outside']} skipped")

            u = np.full(x.size, float(src['phase1']['values_psi'][0]))
            traces, taxes_out, trecs, extras = {}, {}, [], {}
            for ph, idxs, prof in (('phase1', i7, uni), ('phase2', i7, uni),
                                   ('phase3', i8, dbar)):
                s = src[ph]
                n = len(idxs)
                taxis, field, tr, ex = solve_one(
                    x, prof, [s['taxis_s']] * n, [s['values_psi']] * n, idxs, u,
                    float(s['taxis_s'][-1]), tmode, record_idx=None)
                u = field[-1].copy()
                traces[ph] = field[:, gidx].copy()
                taxes_out[ph] = taxis
                tr['label'] = ph
                trecs.append(rm.time_record(taxis, **tr))
                extras[ph] = ex
                del field

            per_phase = {}
            for ph in ('phase1', 'phase2', 'phase3'):
                per = []
                for k, g in enumerate(gauges):
                    ot, ov = obs[g][ph]
                    m = gauge_metrics(taxes_out[ph], traces[ph][:, k], ot, ov)
                    m.update(gauge=int(g), md_ft=float(gauge_md[k]),
                             mesh_idx=int(gidx[k]))
                    per.append(m)
                per_phase[ph] = {'per_gauge': per, 'pooled': pooled(per)}

            npz = os.path.join(rundir, 'summary.npz')
            np.savez_compressed(
                npz, mesh_md_ft=x,
                gauge_numbers=np.array(gauges), gauge_md_ft=np.array(gauge_md),
                **{f'{ph}_taxis_s': taxes_out[ph] for ph in taxes_out},
                **{f'{ph}_traces_psi': traces[ph] for ph in traces})

            src_records = []
            for ph, idxs in (('phase1', i7), ('phase2', i7), ('phase3', i8)):
                s = src[ph]
                hits = inp.fh7 if ph != 'phase3' else inp.fh8
                for j, i in enumerate(idxs):
                    src_records.append(rm.source_record(
                        x, md_requested_ft=float(hits[j]), mesh_idx=int(i),
                        driver=rm.driver_record(
                            kind='gauge_series',
                            baseline_removal='none_absolute_psi',
                            value_units='psi',
                            series_path=rd.repo_path(s['series_path']),
                            gauge_number=s['gauge'], gauge_md_ft=s['md_ft'],
                            taxis=s['taxis_s'], values=s['values_psi'],
                            time_start=s['t0_requested'].isoformat(),
                            time_end=s['t1_requested'].isoformat()),
                        label=f"{ph}:frachit{j}", index_in_source_list=j))

            rec.declare_inputs(
                [(rd.repo_path(SWELL_GAUGE.format(n=g)), 'gauge_series',
                  f'gauge{g}') for g in sorted(set(gauges) | {6, 7})]
                + declare_common_inputs(inp, 'c101', cfg)
                + [(rd.repo_path(rd.PUMPING_DIR_TEMPLATE.format(stage=s),
                                 rd.PUMPING_CURVE_FILES['slurry_rate']),
                    'pumping', f'stage{s}_slurry_rate') for s in (7, 8)])
            rec.declare_output(npz, role='arrays_npz',
                               note='per-phase gauge traces and the mesh')
            rec.set_source(rm.source_protocol(
                application='dirichlet_node',
                solver_class='rev2_core.solve_forward_multi'
                             if tmode['mode'] == 'fixed'
                             else 'rev2_core.solve_forward_adaptive',
                placement_rule='nearest mesh node to each frac-hit MD '
                               '(mesh_utils.locate), 101:87-89',
                sources=src_records,
                targets=[{'gauge': g, 'md_ft': float(m), 'mesh_idx': int(i)}
                         for g, m, i in zip(gauges, gauge_md, gidx)],
                time_level='n',
                phase_chaining='phase1 -> phase2 -> phase3, each seeded with the '
                               'previous final field (101:155-157, :198-200)',
                boundary_conditions=boundary_group()))
            rec.set_numerics(rm.numerics(
                time=trecs,
                mesh=rm.mesh_record(x, dx_requested_ft=float(cfg['dx_ft']),
                                    window_md_ft=[cfg['md_lo_ft'], cfg['md_hi_ft']],
                                    pad_low_ft=pad_low, pad_high_ft=pad_high,
                                    refinement=mrec),
                interface_avg='harmonic', boundary=boundary_group(),
                diffusivity={'family': 'uniform', 'D_ft2_s': D0,
                             'provenance': 'hard-coded at 101:93, no stated source'},
                barriers=barrier_records(x, brep, D0, ratio, 'phase3_stage7'),
                leakage={'lambda_leak': 0.0, 'p0_psi': 0.0},
                kernel=kernel_group('solve_forward_multi'),
                rng=rm.NONE_DECLARED,
                parallel={'processes': 1, 'note': 'serial'}))
            res = {'time_mode': tmode_name, 'pad_low_ft': pad_low,
                   'pad_high_ft': pad_high, 'ratio': ratio, 'nx': int(x.size),
                   'mesh_span_md_ft': [float(x[0]), float(x[-1])],
                   'n_degenerate_refine_calls': mrec['n_degenerate_calls'],
                   'phases': per_phase, 'solver_extra': extras,
                   'barrier_report': {
                       'n_barriers': brep['n_barriers'],
                       'n_merged_groups': brep['n_merged_groups'],
                       'n_fallback': brep['n_fallback'],
                       'realised_full_width_ft': brep['realised_full_width_ft'],
                       'total_equivalent_width_ft':
                           brep['total_equivalent_width_ft'],
                       'excess_resistance_s_per_ft':
                           brep['excess_resistance_s_per_ft']},
                   'wall_s': time.time() - t_wall}
            rec.set_results(res)
            rec.note('The manifest raises duplicate_source_mesh_idx at severity '
                     "'error'. It is a FALSE POSITIVE here: source_protocol "
                     'assumes one solve, while this record declares 18 nodes '
                     'across three phases of which phases 1 and 2 legitimately '
                     'drive the same six. Within each solve the indices are '
                     'unique (solve_forward_multi asserts it).')
            rec.note('crop_rebase = requested_start reproduces the Feb-2025 '
                     'fibeRIS Data1D.crop convention that the archive was written '
                     'under (A5). rev2_data implements the current convention; '
                     'the difference is applied here as a pure shift of the source '
                     f"axis by {src['phase1']['lead_s']:.3f} s in phase 1.")
        res['manifest'] = os.path.relpath(mpath, rd.REPO_ROOT)
        results.append(res)
        p1 = per_phase['phase1']['per_gauge']
        p3 = per_phase['phase3']['per_gauge']
        print(f"  c101 {tag}: nx={x.size} "
              f"p1 g10peak={p1[-1]['peak_abs_dp_psi']:.2f} "
              f"p3 g5peak={p3[0]['peak_abs_dp_psi']:.2f} "
              f"({res['wall_s']:.1f} s)")
    return results


# ---------------------------------------------------------------------------
# cases: c102r / c103r (production forwards)
# ---------------------------------------------------------------------------

def run_prod(inp, case, cfg, root_cfg, outdir, config_path):
    t0 = datetime.datetime.fromisoformat(cfg['window']['t_start'])
    t1 = datetime.datetime.fromisoformat(cfg['window']['t_end'])
    well = cfg['well']
    # Where the field is SAMPLED can differ from the well the driver comes from:
    # Fig. 7b drives 103r from a producer gauge and then reads the simulated
    # drawdown at the fifteen S-WELL gauge MDs, comparing it against S-well field
    # data (103p_forward_modeling_viz_without_scalar.py:24-26, :34-40, :58-63).
    twell = cfg.get('target_well', well)
    gauges = [int(g) for g in cfg['targets']['gauges']]
    mdt = inp.md_table if twell == 's_well' else inp.prod_md_table
    gauge_md = [mdt.md_of(g) for g in gauges]
    obs = {g: inp.gauge_window(twell, g, t0, t1) for g in gauges}
    D0 = float(cfg['D_ft2_s'])
    results = []

    driver_defs = {
        'numeric_g6': ('s_well', 6, 'documented intent (comment "gauge 6" and the '
                                    'output filename "..._gauge6.npz")', False),
        'listdir_g2': ('s_well', 2, 'os.listdir(...)[5] on this filesystem today',
                       False),
        'prod_g3': ('prod', 3, 'documented intent (comment "gauge 3 (center at the '
                               'producer)")', False),
        'prod_g1': ('prod', 1, "D4's initial-condition fingerprint: the archived "
                               'output/0324_forward_simulator/*.npz that Fig. 7b '
                               'is drawn from was driven by PRODUCER GAUGE 1 '
                               '(IC 7028.35 psi, MD 16196), not by gauge 3',
                    False),
        'swell_g15': ('s_well', 15, "D4's fingerprint for the 102r archive "
                                    'output/0224_forward_modeling_mix/*.npz: '
                                    'S-well gauge 15, IC 7984.85 psi, MD 12098',
                      False),
        'prod_g3_zeromasked': ('prod', 3, 'same series with its single hard-zero '
                                          'dropout removed', True),
    }

    specs = []
    for block in cfg['runs']:
        place = block.get('placement', cfg['source']['placement'])
        for pad in block['pads']:
            for ratio in block['ratios']:
                specs.append((block['driver'], block['dt_mode'], place,
                              float(pad[0]), float(pad[1]), float(ratio)))

    for drv, dt_mode, place, pad_low, pad_high, ratio in specs:
        dwell, dn, dnote, dmask = driver_defs[drv]
        dt_ax, dt_v = inp.gauge_window(dwell, dn, t0, t1)
        n_zero = int(np.sum(dt_v == 0.0))
        if dmask:
            keep = dt_v != 0.0
            dt_ax, dt_v = dt_ax[keep], dt_v[keep]
        dt_s = ((dt_ax[1] - dt_ax[0]) * 100.0 if dt_mode == 'legacy'
                else float(dt_ax[1] - dt_ax[0]))
        t_total = float(dt_ax[-1])
        rtag = ('r%g' % ratio).replace('.', 'p').replace('-', 'm')
        ptag = ('' if place == cfg['source']['placement'] else f"_{place}")
        tag = (f"{drv}_{dt_mode}{ptag}_lo{int(pad_low):05d}"
               f"_hi{int(pad_high):05d}_{rtag}")
        rundir = os.path.join(outdir, tag)
        mpath = os.path.join(rundir, 'manifest.json')
        rm.assert_absent([mpath])
        os.makedirs(rundir, exist_ok=True)
        t_wall = time.time()
        with rm.RunRecorder(mpath, study_id=root_cfg['study_id'] + f':{case}',
                            task_id='B2', config=cfg, config_path=config_path,
                            run_label=tag,
                            require_modules=('rev2_core', 'rev2_data')) as rec:
            x, mrec = build_mesh_prod(inp, pad_low, pad_high, cfg)
            if place == 'stage7_frac_hit_nodes':
                src_md = [float(f) for f in inp.fh7]
            elif place == 'mesh_centre_node':
                # 103r:84, verbatim. Anchored to the MESH: one-sided padding moves
                # the source, which is a property of the placement rule and not of
                # the boundary.
                src_md = [float((x[0] + x[-1]) / 2.0)]
            elif place.startswith('fixed_md_'):
                src_md = [float(place.split('fixed_md_')[1])]
            else:
                raise ValueError(f"unknown source placement {place!r}")
            sidx = [int(mesh_utils.locate(x, m)[0]) for m in src_md]
            gidx = [int(mesh_utils.locate(x, m)[0]) for m in gauge_md]
            dprof, brep = rc.build_barrier_profile(
                x, D0, inp.all_hits, float(cfg['barrier']['w_ft']), ratio,
                on_empty=cfg['barrier']['on_empty'], return_report=True)
            u0 = np.full(x.size, float(dt_v[0]))
            n = len(sidx)
            taxis, trace, trec, extra = solve_one(
                x, dprof, [dt_ax] * n, [dt_v] * n, sidx, u0, t_total,
                {'mode': 'fixed', 'dt_fixed_s': dt_s}, record_idx=gidx)
            evals = [(d, (datetime.datetime.fromisoformat(d) - t0).total_seconds())
                     for d in cfg.get('drawdown_eval_dates', [])]
            per = []
            for k, g in enumerate(gauges):
                ot, ov = obs[g]
                m = gauge_metrics(taxis, trace[:, k], ot, ov)
                sim_on_obs = np.interp(ot, taxis, trace[:, k])
                m.update(gauge=int(g), md_ft=float(gauge_md[k]),
                         mesh_idx=int(gidx[k]),
                         sim_drawdown_psi=float(sim_on_obs[0] - sim_on_obs[-1]),
                         obs_drawdown_psi=float(ov[0] - ov[-1]))
                # Fig. 7b calls select_time(2020-04-01, 2020-06-01) on a run that
                # was integrated to 2021-07-01, so the drawdown it PLOTS is the
                # two-month one, not the fifteen-month one. Both are recorded.
                for label, te in evals:
                    m[f'sim_drawdown_at_{label[:10]}_psi'] = float(
                        sim_on_obs[0] - np.interp(te, taxis, trace[:, k]))
                    m[f'obs_drawdown_at_{label[:10]}_psi'] = float(
                        ov[0] - np.interp(te, ot, ov))
                per.append(m)
            npz = os.path.join(rundir, 'summary.npz')
            np.savez_compressed(npz, mesh_md_ft=x, taxis_s=taxis,
                                traces_psi=trace,
                                gauge_numbers=np.array(gauges),
                                gauge_md_ft=np.array(gauge_md),
                                source_mesh_idx=np.array(sidx),
                                driver_taxis_s=dt_ax, driver_psi=dt_v)
            trec['label'] = f'{case}_forward'
            dmdt = inp.md_table if dwell == 's_well' else inp.prod_md_table
            drec = rm.driver_record(
                kind='gauge_series', baseline_removal='none_absolute_psi',
                value_units='psi',
                series_path=rd.repo_path(
                    (SWELL_GAUGE if dwell == 's_well' else PROD_GAUGE).format(n=dn)),
                gauge_number=dn, gauge_md_ft=dmdt.md_of(dn),
                taxis=dt_ax, values=dt_v,
                time_start=t0.isoformat(), time_end=t1.isoformat())
            rec.declare_inputs(
                [(rd.repo_path((SWELL_GAUGE if twell == 's_well' else PROD_GAUGE)
                               .format(n=g)), 'gauge_series', f'{twell}_gauge{g}')
                 for g in gauges]
                + [(rd.repo_path((SWELL_GAUGE if dwell == 's_well' else PROD_GAUGE)
                                 .format(n=dn)), 'gauge_series', f'driver_{drv}')]
                + declare_common_inputs(inp, case, cfg))
            rec.declare_output(npz, role='arrays_npz',
                               note='sim traces at every target gauge, the mesh '
                                    'and the Dirichlet driver')
            rec.set_source(rm.source_protocol(
                application='dirichlet_node',
                solver_class='rev2_core.solve_forward_multi',
                placement_rule=place + ' via mesh_utils.locate',
                sources=[rm.source_record(x, md_requested_ft=m, mesh_idx=i,
                                          driver=drec, label=f'src{j}',
                                          index_in_source_list=j)
                         for j, (m, i) in enumerate(zip(src_md, sidx))],
                targets=[{'gauge': g, 'md_ft': float(m), 'mesh_idx': int(i)}
                         for g, m, i in zip(gauges, gauge_md, gidx)],
                time_level='n', phase_chaining='none',
                boundary_conditions=boundary_group()))
            rec.set_numerics(rm.numerics(
                time=rm.time_record(taxis, **trec),
                mesh=rm.mesh_record(x, dx_requested_ft=float(cfg['dx_ft']),
                                    window_md_ft=[float(x[0]) + pad_low,
                                                  float(x[-1]) - pad_high],
                                    pad_low_ft=pad_low, pad_high_ft=pad_high,
                                    refinement=mrec),
                interface_avg='harmonic', boundary=boundary_group(),
                diffusivity={'family': 'uniform', 'D_ft2_s': D0},
                barriers=barrier_records(x, brep, D0, ratio, 'frachit'),
                leakage={'lambda_leak': 0.0, 'p0_psi': 0.0},
                kernel=kernel_group('solve_forward_multi'),
                rng=rm.NONE_DECLARED,
                parallel={'processes': 1, 'note': 'serial'}))
            res = {'driver': drv, 'driver_note': dnote, 'dt_mode': dt_mode,
                   'placement': place, 'driver_well': dwell,
                   'driver_gauge': dn, 'driver_md_ft': float(dmdt.md_of(dn)),
                   'target_well': twell,
                   'dt_s': float(dt_s), 't_total_s': t_total,
                   'n_zero_samples_in_raw_driver': n_zero,
                   'pad_low_ft': pad_low, 'pad_high_ft': pad_high,
                   'ratio': ratio, 'nx': int(x.size),
                   'mesh_span_md_ft': [float(x[0]), float(x[-1])],
                   'source_md_ft': [float(x[i]) for i in sidx],
                   'n_degenerate_refine_calls': mrec['n_degenerate_calls'],
                   'per_gauge': per, 'pooled': pooled(per),
                   'solver_extra': extra,
                   'barrier_report': {
                       'n_barriers': brep['n_barriers'],
                       'n_merged_groups': brep['n_merged_groups'],
                       'n_fallback': brep['n_fallback'],
                       'n_touching_domain_edge': brep['n_touching_domain_edge'],
                       'total_equivalent_width_ft':
                           brep['total_equivalent_width_ft'],
                       'excess_resistance_s_per_ft':
                           brep['excess_resistance_s_per_ft']},
                   'wall_s': time.time() - t_wall}
            rec.set_results(res)
            rec.note(f"driver: {dnote}. {n_zero} hard-zero sample(s) in the raw "
                     f"cropped driver series"
                     + (' (removed for this run)' if dmask else
                        ' (kept, as the legacy script does)') + '.')
        res['manifest'] = os.path.relpath(mpath, rd.REPO_ROOT)
        results.append(res)
        print(f"  {case} {tag}: nx={x.size} steps={taxis.size} "
              f"pooled={res['pooled']['sample_pooled_rmse_psi']:.1f} psi "
              f"({res['wall_s']:.1f} s)")
    return results


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def aggregate_c101(runs):
    gauges = [5, 6, 7, 8, 9, 10]
    agg = {}
    for tmode in sorted({r['time_mode'] for r in runs}):
        for ratio in sorted({r['ratio'] for r in runs}):
            sel = [r for r in runs if r['time_mode'] == tmode
                   and r['ratio'] == ratio]
            if not sel:
                continue
            ref = [r for r in sel if r['pad_low_ft'] == 0 and r['pad_high_ft'] == 0]
            if not ref:
                continue
            ref = ref[0]
            block = {}
            for ph in ('phase1', 'phase2', 'phase3'):
                def val(r, g, ph=ph, field='peak_abs_dp_psi'):
                    return next(x[field] for x in r['phases'][ph]['per_gauge']
                                if x['gauge'] == g)
                rows = []
                for r in sel:
                    row = {'pad_low_ft': r['pad_low_ft'],
                           'pad_high_ft': r['pad_high_ft'], 'nx': r['nx'],
                           'sample_pooled_rmse_psi':
                               r['phases'][ph]['pooled']['sample_pooled_rmse_psi'],
                           'gauge_mean_rmse_psi':
                               r['phases'][ph]['pooled']['gauge_mean_rmse_psi'],
                           'gauges': {}}
                    for g in gauges:
                        pk, pkr = val(r, g), val(ref, g)
                        rm_, rmr = (val(r, g, field='rmse_psi'),
                                    val(ref, g, field='rmse_psi'))
                        ar, arr = (val(r, g, field='arrival_rel_sim_s'),
                                   val(ref, g, field='arrival_rel_sim_s'))
                        row['gauges'][str(g)] = {
                            'peak_abs_dp_psi': pk,
                            'peak_delta_vs_legacy_psi': pk - pkr,
                            'peak_pct_vs_legacy': 100.0 * (pk - pkr) / pkr,
                            'rmse_psi': rm_, 'rmse_delta_vs_legacy_psi': rm_ - rmr,
                            'arrival_rel_sim_s': ar,
                            'arrival_delta_vs_legacy_s': ar - arr,
                            'obs_peak_abs_dp_psi':
                                val(r, g, field='obs_peak_abs_dp_psi')}
                    rows.append(row)
                block[ph] = rows
            agg[f'{tmode}_ratio{ratio:g}'] = block
    return agg


def aggregate_prod(runs, gauges):
    agg = {}
    keys = sorted({(r['driver'], r['dt_mode'], r['placement'], r['ratio'])
                   for r in runs})
    for drv, dtm, place, ratio in keys:
        sel = [r for r in runs
               if (r['driver'], r['dt_mode'], r['placement'], r['ratio'])
               == (drv, dtm, place, ratio)]
        ref = [r for r in sel if r['pad_low_ft'] == 0 and r['pad_high_ft'] == 0]
        if not ref:
            continue
        ref = ref[0]

        def val(r, g, field):
            return next(x[field] for x in r['per_gauge'] if x['gauge'] == g)
        rows = []
        for r in sel:
            row = {'pad_low_ft': r['pad_low_ft'], 'pad_high_ft': r['pad_high_ft'],
                   'nx': r['nx'],
                   'sample_pooled_rmse_psi': r['pooled']['sample_pooled_rmse_psi'],
                   'gauge_mean_rmse_psi': r['pooled']['gauge_mean_rmse_psi'],
                   'gauges': {}}
            for g in gauges:
                pk, pkr = (val(r, g, 'peak_abs_dp_psi'),
                           val(ref, g, 'peak_abs_dp_psi'))
                rm_, rmr = val(r, g, 'rmse_psi'), val(ref, g, 'rmse_psi')
                ar, arr = (val(r, g, 'arrival_rel_sim_s'),
                           val(ref, g, 'arrival_rel_sim_s'))
                row['gauges'][str(g)] = {
                    'peak_abs_dp_psi': pk, 'peak_delta_vs_legacy_psi': pk - pkr,
                    'peak_pct_vs_legacy': 100.0 * (pk - pkr) / pkr,
                    'rmse_psi': rm_, 'rmse_delta_vs_legacy_psi': rm_ - rmr,
                    'arrival_rel_sim_s': ar, 'arrival_delta_vs_legacy_s': ar - arr,
                    'sim_drawdown_psi': val(r, g, 'sim_drawdown_psi'),
                    'obs_drawdown_psi': val(r, g, 'obs_drawdown_psi')}
                for kk in sorted(r['per_gauge'][0]):
                    if kk.startswith('sim_drawdown_at_') or \
                            kk.startswith('obs_drawdown_at_'):
                        row['gauges'][str(g)][kk] = val(r, g, kk)
            row['source_md_ft'] = r['source_md_ft']
            rows.append(row)
        agg[f'{drv}_{dtm}_{place}_ratio{ratio:g}'] = rows
    return agg


def decoupling_check(runs, outdir):
    """Verify -- not assume -- that a Dirichlet source node decouples the mesh.

    The fixed-dt sweep shares one time grid across pads, so the difference between
    two runs' gauge traces is exact rather than an interpolation residual. For each
    phase this reports, per gauge, max_t |dP(pad) - dP(legacy)| and whether that
    gauge lies above or below the phase's source cluster. The claim under test is
    that a gauge BELOW the lowest source is untouched by the high-MD boundary and a
    gauge ABOVE the highest source is untouched by the low-MD boundary.
    """
    src_span = {'phase1': None, 'phase2': None, 'phase3': None}
    out = {'definition': ('max over time of |dP(padded) - dP(legacy)| at each '
                          'gauge, on the shared fixed-dt = 30 s time grid; dP is '
                          "each run's own first sample removed"),
           'phases': {}}
    ref = next(r for r in runs if r['time_mode'] == 'fixed30'
               and r['ratio'] == 1.0 and r['pad_low_ft'] == 0
               and r['pad_high_ft'] == 0)
    zr = np.load(os.path.join(rd.REPO_ROOT, os.path.dirname(ref['manifest']),
                              'summary.npz'))
    gauges = [int(g) for g in zr['gauge_numbers']]
    for ph in ('phase1', 'phase2', 'phase3'):
        base = zr[f'{ph}_traces_psi']
        base = base - base[0]
        rows = []
        for r in runs:
            if r['time_mode'] != 'fixed30' or r['ratio'] != 1.0:
                continue
            if r['pad_low_ft'] == 0 and r['pad_high_ft'] == 0:
                continue
            z = np.load(os.path.join(rd.REPO_ROOT, os.path.dirname(r['manifest']),
                                     'summary.npz'))
            tr = z[f'{ph}_traces_psi']
            tr = tr - tr[0]
            if tr.shape != base.shape:
                z.close()
                continue
            rows.append({'pad_low_ft': r['pad_low_ft'],
                         'pad_high_ft': r['pad_high_ft'],
                         'max_abs_dtrace_psi':
                             {str(g): float(np.max(np.abs(tr[:, k] - base[:, k])))
                              for k, g in enumerate(gauges)}})
            z.close()
        out['phases'][ph] = rows
    zr.close()
    out['source_span_note'] = (
        'phase 1 and 2 drive the six stage-7 frac hits, MD 15186.83-15364.29; '
        'phase 3 drives the six stage-8 frac hits, MD 14940.0-15118.18. Gauge MDs: '
        '5=15599, 6=15344, 7=15075, 8=14821, 9=14552, 10=14297.')
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _pad_axis(rows, which):
    other = 'pad_high_ft' if which == 'pad_low_ft' else 'pad_low_ft'
    sel = [r for r in rows if r[other] == 0]
    sel.sort(key=lambda r: r[which])
    return sel


def fig_c101(agg, path, dpi, key):
    block = agg[key]
    gauges = ['5', '6', '7', '8', '9', '10']
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(gauges)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex='col')
    for col, (which, title) in enumerate(
            [('pad_low_ft', 'low-MD padding (deeper end, MD < 12500)'),
             ('pad_high_ft', 'high-MD padding (shallower end, MD > 17999)')]):
        for row, ph in enumerate(['phase1', 'phase3']):
            ax = axes[row][col]
            sel = _pad_axis(block[ph], which)
            pads = [r[which] for r in sel]
            for gi, g in enumerate(gauges):
                y = [r['gauges'][g]['peak_pct_vs_legacy'] for r in sel]
                ax.plot(pads, y, 'o-', color=colors[gi], lw=1.4, ms=4,
                        label=f'gauge {g}')
            ax.axhline(0, color='k', lw=0.6)
            ax.axvline(5000, color='crimson', ls='--', lw=0.9)
            ax.grid(alpha=0.3)
            ax.set_ylabel(f'{ph} peak |dP| change (%)\nrelative to the legacy domain')
            if row == 0:
                ax.set_title(title, fontsize=10)
            if row == 1:
                ax.set_xlabel(f'{which.replace("_ft", "")} (ft)')
    axes[0][0].legend(fontsize=8, ncol=2, loc='best')
    fig.suptitle('B2 / 101_fiberis_matching.py (Fig. 6): peak amplitude vs domain '
                 f'padding\n{key}; red dashed = the 5000 ft house rule',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def fig_c101_traces(runs, path, dpi, ratio):
    sel = {(r['pad_low_ft'], r['pad_high_ft']): r for r in runs
           if r['time_mode'] == 'fixed30' and r['ratio'] == ratio}
    want = [((0.0, 0.0), 'legacy domain (no pad)', 'crimson'),
            ((0.0, 8000.0), '+8000 ft high-MD pad', 'tab:blue'),
            ((8000.0, 0.0), '+8000 ft low-MD pad', 'tab:green'),
            ((20000.0, 20000.0), '+20000 ft both ends', 'k')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (ph, g, gi) in zip(axes, [('phase3', 5, 0), ('phase1', 10, 5)]):
        for key, lab, c in want:
            r = sel.get(key)
            if r is None:
                continue
            z = np.load(os.path.join(rd.REPO_ROOT, os.path.dirname(r['manifest']),
                                     'summary.npz'))
            t = z[f'{ph}_taxis_s']
            y = z[f'{ph}_traces_psi'][:, gi]
            ax.plot(t, y - y[0], color=c, lw=1.3, label=lab)
            z.close()
        ax.set_xlabel('time within the phase (s)')
        ax.set_ylabel('dP (psi)')
        ax.set_title(f'{ph}, gauge {g}')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('B2 / 101: the two traces the domain actually moves '
                 f'(barrier ratio {ratio:g}, fixed dt = 30 s)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def fig_prod(agg, key, gauges, md, path, dpi, title):
    rows = agg[key]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    sel = sorted([r for r in rows if r['pad_low_ft'] == r['pad_high_ft']],
                 key=lambda r: r['pad_low_ft'])
    cmap = plt.cm.plasma(np.linspace(0, 0.85, max(len(sel), 1)))
    ax = axes[0]
    for i, r in enumerate(sel):
        y = [r['gauges'][str(g)]['sim_drawdown_psi'] for g in gauges]
        ax.plot(md, y, 'o-', color=cmap[i], lw=1.3, ms=4,
                label=f"pad {int(r['pad_low_ft'])} ft both ends")
    obs = [rows[0]['gauges'][str(g)]['obs_drawdown_psi'] for g in gauges]
    ax.plot(md, obs, 'ks--', lw=1.6, ms=6, label='observed')
    ax.set_xlabel('gauge MD (ft)')
    ax.set_ylabel('drawdown over the window (psi)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title('drawdown profile')
    ax = axes[1]
    for which, mk in (('pad_low_ft', 'o-'), ('pad_high_ft', 's--')):
        s2 = _pad_axis(rows, which)
        ax.plot([r[which] for r in s2],
                [r['sample_pooled_rmse_psi'] for r in s2], mk,
                label=f'{which.replace("_ft", "")} only')
    ax.plot([r['pad_low_ft'] for r in sel],
            [r['sample_pooled_rmse_psi'] for r in sel], '^-', color='crimson',
            label='both ends')
    ax.set_xlabel('padding (ft)')
    ax.set_ylabel('sample-pooled RMSE vs data (psi)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title('misfit vs padding')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def fig_r1(runs, path, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    pads = [r['pad_low_ft'] for r in runs]
    axes[0].plot(pads, [r['pooled']['sample_pooled_rmse_psi'] for r in runs],
                 'o-', color='crimson')
    axes[0].set_xlabel('low-MD pad (ft)')
    axes[0].set_ylabel('sample-pooled RMSE (psi)')
    axes[0].grid(alpha=0.3)
    axes[0].set_title('R1 window, D = 2000 ft$^2$/s')
    for gi, g in enumerate([2, 3, 4, 5, 6, 7]):
        y = [next(x['peak_abs_dp_psi'] for x in r['per_gauge'] if x['gauge'] == g)
             for r in runs]
        axes[1].plot(pads, y, 'o-', label=f'gauge {g}')
    axes[1].axhline(next(x['obs_peak_abs_dp_psi'] for x in runs[0]['per_gauge']
                         if x['gauge'] == 7), color='k', ls=':',
                    label='gauge 7 measured')
    axes[1].set_xlabel('low-MD pad (ft)')
    axes[1].set_ylabel('simulated peak dP (psi)')
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].set_title('per-gauge peak')
    fig.suptitle('B2 harness validation: the established R1 padding result, '
                 'reproduced by this task', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# case-level manifest
# ---------------------------------------------------------------------------

def write_case_manifest(case, cfg, root_cfg, outdir, config_path, runs, files,
                        summary, note):
    """Aggregate record for one case. Its source/numerics groups are the LEGACY
    run's, and it declares only the case-level figures and json; every solver run
    carries its own manifest in its own subdirectory."""
    mpath = os.path.join(outdir, 'manifest.json')
    rm.assert_absent([mpath])
    ref_manifest = os.path.join(rd.REPO_ROOT, runs[0]['manifest'])
    with open(ref_manifest) as fh:
        ref = json.load(fh)
    with rm.RunRecorder(mpath, study_id=root_cfg['study_id'] + f':{case}:aggregate',
                        task_id='B2', config=cfg, config_path=config_path,
                        run_label=f'{case}_aggregate',
                        require_modules=('rev2_core', 'rev2_data'),
                        allow_undeclared_outputs=False) as rec:
        for p, role, dpi, n in files:
            rec.declare_output(p, role=role, dpi=dpi, note=n)
        rec.declare_inputs([(ref_manifest, 'prior_run_output',
                             f'{case}_reference_run_manifest')])
        sp = dict(ref['source_protocol'])
        sp[rm._BUILDER] = 'source_protocol'
        nm = dict(ref['numerics'])
        nm[rm._BUILDER] = 'numerics'
        rec.set_source(sp)
        rec.set_numerics(nm)
        rec.set_results(summary)
        rec.note(note)
        rec.note(f'Aggregate of {len(runs)} solver runs, each with its own '
                 f'manifest under {os.path.relpath(outdir, rd.REPO_ROOT)}/. The '
                 f'source and numerics groups copied here are the LEGACY-domain '
                 f'run\'s ({runs[0]["manifest"]}); every other run differs from '
                 f'it only in mesh.pad_low_ft / mesh.pad_high_ft (and, where the '
                 f'name says so, in the time mode, driver or barrier ratio).')
    return mpath


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/rev2/b2_boundary.json')
    ap.add_argument('--case', action='append', default=None)
    ap.add_argument('--outdir', default=None)
    a = ap.parse_args(argv)

    cfg_path = os.path.abspath(a.config)
    with open(cfg_path) as fh:
        root = json.load(fh)
    out_root = os.path.abspath(a.outdir or os.path.join(rd.REPO_ROOT,
                                                        root['output_root']))
    os.makedirs(out_root, exist_ok=True)
    want = a.case or [k for k, v in root['cases'].items() if v.get('enabled')]
    dpi = int(root['figures']['dpi'])

    t_all = time.time()
    inp = Inputs()
    record = {'study_id': root['study_id'], 'task_id': 'B2',
              'config_path': os.path.relpath(cfg_path, rd.REPO_ROOT),
              'run_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'cases': {}}

    for case in want:
        cfg = root['cases'][case]
        outdir = os.path.join(out_root, case)
        os.makedirs(outdir, exist_ok=True)
        print(f"[{case}] {cfg['kind']}")
        if cfg['kind'] == 'r1_window':
            runs = run_r1_reference(inp, cfg, root, outdir, cfg_path)
            fp = os.path.join(outdir, 'fig01_r1_reference_padding_v1.png')
            fig_r1(runs, fp, dpi)
            jp = os.path.join(outdir, 'r1_reference_summary.json')
            summary = {'runs': runs}
            with open(jp, 'w') as fh:
                json.dump(rm._jsonify(summary)[0], fh, indent=2, sort_keys=True)
            mp = write_case_manifest(
                case, cfg, root, outdir, cfg_path, runs,
                [(fp, 'figure_png', dpi, 'R1 padding convergence'),
                 (jp, 'json', None, 'per-run metrics')],
                summary,
                'Harness validation. Reproduces the established R1 numbers with '
                'this task\'s own code before any legacy domain is judged.')
            record['cases'][case] = {'n_runs': len(runs), 'summary_json':
                                     os.path.relpath(jp, rd.REPO_ROOT),
                                     'case_manifest':
                                     os.path.relpath(mp, rd.REPO_ROOT)}
        elif cfg['kind'] == 'chain101':
            runs = run_c101(inp, cfg, root, outdir, cfg_path)
            agg = aggregate_c101(runs)
            files = []
            for i, key in enumerate(sorted(agg)):
                fp = os.path.join(outdir, f'fig{i + 1:02d}_c101_{key}_v1.png')
                fig_c101(agg, fp, dpi, key)
                files.append((fp, 'figure_png', dpi,
                              f'peak change vs padding, {key}'))
            fp = os.path.join(outdir, 'fig90_c101_traces_ratio1p0_v1.png')
            fig_c101_traces(runs, fp, dpi, 1.0)
            files.append((fp, 'figure_png', dpi, 'the two moved traces'))
            jp = os.path.join(outdir, 'c101_summary.json')
            summary = {'runs': runs, 'aggregate': agg,
                       'decoupling_check': decoupling_check(runs, outdir)}
            with open(jp, 'w') as fh:
                json.dump(rm._jsonify(summary)[0], fh, indent=2, sort_keys=True)
            files.append((jp, 'json', None, 'per-run metrics and delta tables'))
            mp = write_case_manifest(
                case, cfg, root, outdir, cfg_path, runs, files, summary,
                'Boundary contamination of 101_fiberis_matching.py, the script '
                'behind the manuscript two-stage figure.')
            record['cases'][case] = {'n_runs': len(runs), 'summary_json':
                                     os.path.relpath(jp, rd.REPO_ROOT),
                                     'case_manifest':
                                     os.path.relpath(mp, rd.REPO_ROOT)}
        elif cfg['kind'] == 'prod_forward':
            runs = run_prod(inp, case, cfg, root, outdir, cfg_path)
            gauges = [int(g) for g in cfg['targets']['gauges']]
            mdt = (inp.md_table if cfg.get('target_well', cfg['well']) == 's_well'
                   else inp.prod_md_table)
            md = [mdt.md_of(g) for g in gauges]
            agg = aggregate_prod(runs, gauges)
            files = []
            for i, key in enumerate(sorted(agg)):
                if 'legacy' not in key:
                    continue
                fp = os.path.join(outdir, f'fig{i + 1:02d}_{case}_{key}_v1.png')
                fig_prod(agg, key, gauges, md, fp, dpi,
                         f'B2 / {case}: drawdown profile and misfit vs domain '
                         f'padding ({key})')
                files.append((fp, 'figure_png', dpi, f'drawdown + misfit, {key}'))
            jp = os.path.join(outdir, f'{case}_summary.json')
            summary = {'runs': runs, 'aggregate': agg,
                       'gauge_md_ft': {str(g): float(m)
                                       for g, m in zip(gauges, md)}}
            with open(jp, 'w') as fh:
                json.dump(rm._jsonify(summary)[0], fh, indent=2, sort_keys=True)
            files.append((jp, 'json', None, 'per-run metrics and delta tables'))
            mp = write_case_manifest(
                case, cfg, root, outdir, cfg_path, runs, files, summary,
                f'Boundary contamination of {case}.')
            record['cases'][case] = {'n_runs': len(runs), 'summary_json':
                                     os.path.relpath(jp, rd.REPO_ROOT),
                                     'case_manifest':
                                     os.path.relpath(mp, rd.REPO_ROOT)}
        else:
            raise ValueError(f"unknown case kind {cfg['kind']!r}")

    record['wall_seconds'] = time.time() - t_all
    rp = os.path.join(out_root, 'b2_boundary_record.json')
    with open(rp, 'w') as fh:
        json.dump(rm._jsonify(record)[0], fh, indent=2, sort_keys=True)
    print(f"done in {record['wall_seconds']:.1f} s -> {rp}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
