"""E4 -- forward-model the scenario the manuscript actually adopts.

The manuscript adopts hypothesis B: MANY conductive fractures, each isolated by
its own barrier, whose signals sum along a poor cement annulus into a
near-uniform production drawdown. Fig. 7b shows two counter-examples (one
fracture, no barrier; one fracture, with barrier) and no positive example --
the adopted scenario has never been forward-modelled. This task supplies it.

Model, stated once:

  * 1-D diffusion along the S-well measured depth, `rev2_core.solve_forward_multi`
    (theta = 1, harmonic faces, lambda = 0 -- the configuration that is bitwise
    identical to the verified R1 kernel).
  * One DIRICHLET node per conductive fracture, all driven by the same producer
    pressure history. That is what "conductive fracture" means here: the
    fracture ties the annulus at that MD to the depleting reservoir.
  * A barrier of physical half-width w = 1.0 ft (A1's accepted 2.000 ft
    equivalent full width) centred on each fracture, reduction ratio `ratio`.
    This is the "isolated by its own barrier" element.
  * Everything between the fractures is the poor cement annulus at the baseline
    diffusivity D0. That is the "signal sums along the annulus" element.
  * Target: the two-month drawdown at the fifteen S-well gauges, in NUMERIC
    gauge order (`rev2_data.production_drawdown`).

Fracture sets come from the DSS record (channels where tensile strain appears)
and, as an independent control, from the archived LF-DAS frac-hit catalogue.
Several selections are run and the spread reported, because a scenario built
from one hand-picked fracture set proves nothing.

Owned by task E4: this file, `configs/rev2/e4_multifrac.json`,
`output/rev2_20260901/E4/`. Shared modules are imported, never edited.
"""

import argparse
import datetime
import fnmatch
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rev2_core as rc          # noqa: E402
import rev2_data as rd          # noqa: E402
import rev2_manifest as rm      # noqa: E402

TASK = 'E4'
OUTDIR = os.path.join('output', rm.ROUND_TAG, TASK)
CONFIG = os.path.join('configs', 'rev2', 'e4_multifrac.json')


# ---------------------------------------------------------------------------
# manifest groups shared by every run
# ---------------------------------------------------------------------------

def boundary_group(cfg):
    s = cfg['solver']
    return {'lbc': s['lbc'], 'rbc': s['rbc'], 'pml_thickness': 0.0,
            'sigma_max': 0.0,
            'note': ('both ends no-flux, as every legacy production script sets '
                     'them; sigma is identically zero so the fibeRIS PML '
                     'diagonal loop (a known defect, HOUSE_RULES CORRECTION 3) '
                     'is never exercised')}


def kernel_group():
    return {'module': 'scripts/manuscript_well_leakage/rev2/rev2_core.py',
            'function': 'solve_forward_multi',
            'equivalence': ('theta=1 / harmonic / lambda=0 is bitwise identical '
                            'to r1_calibration_core.solve_forward, which is '
                            'proven bit-equivalent to fibeRIS; '
                            'solve_forward_multi matches fibeRIS '
                            'PDS1D_MultiSource to 7.36e-11 psi (rev2_selftest)'),
            'caveat': ('the four adversarial verifiers of the rev2 modules were '
                       'killed by a session limit; the modules are self-tested, '
                       'not independently verified')}


def barrier_records(mesh, report, d_base, ratio, label_prefix):
    if report is None:
        return rm.NONE_DECLARED
    out = []
    for k, b in enumerate(report['barriers']):
        mask = np.zeros(mesh.size, dtype=bool)
        mask[int(b['i0']):int(b['i1']) + 1] = True
        out.append(rm.barrier_record(
            mesh, mask, label=f"{label_prefix}[{k}]",
            centre_md_ft=float(b['md_ft']),
            w_requested_ft=float(report['w_requested_ft']),
            ratio=float(ratio), d_baseline=float(d_base),
            report={'fallback': bool(b['fallback']),
                    'realised_full_width_ft': float(b['realised_full_width_ft']),
                    'center_offset_ft': float(b['center_offset_ft']),
                    'group': int(b['group'])}))
    return out


# ---------------------------------------------------------------------------
# DSS fracture picking
# ---------------------------------------------------------------------------

def dss_net_tensile(cfg):
    """Per-channel net tensile strain change over the DSS record.

    Returns (daxis_ft, net_ue, meta). Everything is a stated, reproducible rule:
    no channel is hand-picked anywhere in this function.
    """
    dcfg = cfg['dss']
    path = rd.repo_path(dcfg['path'])
    z = np.load(path, allow_pickle=True)
    daxis = np.asarray(z['daxis'], dtype=float)
    taxis = np.asarray(z['taxis'], dtype=float)
    data = z['data']

    # common-mode drift removal, verbatim from 105s_DSSfrac_picker.py
    dr = dcfg['drift_removal']
    band = (daxis > float(dr['daxis_lo_ft'])) & (daxis < float(dr['daxis_hi_ft']))
    drift = np.median(data[band, :], axis=0).astype(float)

    tlo, thi = [float(v) for v in dcfg['time_window_s']]
    dlo, dhi = [float(v) for v in dcfg['daxis_window_ft']]
    mt = (taxis >= tlo) & (taxis <= thi)
    md_ = (daxis >= dlo) & (daxis <= dhi)
    sub = data[md_, :][:, mt].astype(np.float64) - drift[mt][None, :]
    t = taxis[mt]
    d = daxis[md_]

    e0, e1 = [float(v) for v in dcfg['early_window_s']]
    l0, l1 = [float(v) for v in dcfg['late_window_s']]
    early = (t >= e0) & (t <= e1)
    late = (t >= l0) & (t <= l1)
    net = np.median(sub[:, late], axis=1) - np.median(sub[:, early], axis=1)
    mad = float(1.4826 * np.median(np.abs(net - np.median(net))))
    meta = {'dss_path': dcfg['path'], 'n_channels_used': int(d.size),
            'n_times_used': int(t.size),
            'daxis_span_ft': [float(d[0]), float(d[-1])],
            'taxis_span_s': [float(t[0]), float(t[-1])],
            'n_early_samples': int(early.sum()), 'n_late_samples': int(late.sum()),
            'sigma_mad_ue': mad, 'std_ue': float(net.std()),
            'channel_spacing_ft': float(np.median(np.diff(d)))}
    return d, net, meta


def dss_picks(cfg, outdir, cache=True):
    """Every DSS-derived fracture selection, plus the QC record."""
    cpath = os.path.join(outdir, 'dss_picks_v1.json')
    if cache and os.path.exists(cpath):
        with open(cpath) as fh:
            return json.load(fh)
    from scipy.signal import find_peaks
    d, net, meta = dss_net_tensile(cfg)
    dcfg = cfg['dss']
    mad = meta['sigma_mad_ue']
    dist = int(dcfg['peak_min_separation_channels'])
    cal = float(dcfg['md_calibration_ft'])
    cal_alt = float(dcfg['md_calibration_alt_ft'])
    out = {'meta': meta, 'sets': {}}
    for k in dcfg['k_sigma_levels']:
        p, props = find_peaks(net, height=k * mad, prominence=k * mad,
                              distance=dist)
        name = 'dss_k%d' % int(k)
        out['sets'][name] = {
            'rule': (f"find_peaks(net_tensile, height={k}*sigma_MAD, "
                     f"prominence={k}*sigma_MAD, distance={dist} channels "
                     f"= {dist * meta['channel_spacing_ft']:.2f} ft), "
                     f"sigma_MAD = {mad:.4f} ue; MD = daxis + {cal}"),
            'k_sigma': float(k),
            'daxis_ft': [float(v) for v in d[p]],
            'md_ft': [float(v + cal) for v in d[p]],
            'net_tensile_ue': [float(v) for v in net[p]],
            'prominence_ue': [float(v) for v in props['prominences']]}
    kp = int(cfg['dss']['primary_k'])
    prim = out['sets']['dss_k%d' % kp]
    out['sets']['dss_k%d_eofcal' % kp] = {
        'rule': prim['rule'].replace(f"MD = daxis + {cal}",
                                     f"MD = daxis + {cal_alt}") +
                ' [end-of-fibre depth calibration, 151.4 ft below the '
                'manuscript one]',
        'k_sigma': float(kp),
        'daxis_ft': list(prim['daxis_ft']),
        'md_ft': [float(v + cal_alt) for v in prim['daxis_ft']],
        'net_tensile_ue': list(prim['net_tensile_ue']),
        'prominence_ue': list(prim['prominence_ue'])}
    # the three depths the manuscript's own DSS figure marks by hand
    out['manuscript_handpicked_md_ft'] = [13705.0, 15069.0, 16276.0]
    out['manuscript_handpick_recovery'] = []
    for h in out['manuscript_handpicked_md_ft']:
        cand = np.asarray(out['sets']['dss_k3']['md_ft'])
        j = int(np.argmin(np.abs(cand - h)))
        out['manuscript_handpick_recovery'].append(
            {'manuscript_md_ft': h, 'nearest_automatic_pick_md_ft': float(cand[j]),
             'offset_ft': float(cand[j] - h)})
    if cache:
        with open(cpath, 'w') as fh:
            json.dump(out, fh, indent=1)
    return out


def selection_md(name, picks, cfg):
    """MDs of one named fracture selection, sorted, plus a provenance string."""
    if name in picks['sets']:
        s = picks['sets'][name]
        return np.sort(np.asarray(s['md_ft'], dtype=float)), s['rule']
    if name == 'frachit_all':
        hits = np.sort(np.concatenate(
            [rd.load_frac_hits(st) for st in range(1, 21)]))
        return hits, ('every catalogued LF-DAS frac hit on the S well, stages '
                      '1-20, rev2_data.load_frac_hits(unique=True) per stage '
                      '(112 distinct MDs; stage 1 stores 16696.914 twice)')
    if name == 'frachit_stage7':
        return rd.load_frac_hits(7), ('the six stage-7 frac hits, the set '
                                      '101/106 drive')
    if name == 'fig7b_single':
        md = float(cfg['fig7b_single_source_md_ft'])
        return np.array([md]), cfg['fig7b_single_note']
    raise ValueError(f'unknown selection {name!r}')


# ---------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------

def build_mesh(lo, hi, centres, mcfg):
    """Uniform base mesh refined around every barrier/source centre.

    De-duplicated with a finite tolerance: A1 open issue 1 records that
    `np.unique` on concatenated float grids left node pairs 9e-13 ft apart and
    raised a no-barrier reference solve's RMSE from 7.5e-7 to 22.12 psi.
    """
    dxb = float(mcfg['dx_base_ft'])
    n = int(np.floor((hi - lo) / dxb)) + 1
    parts = [lo + dxb * np.arange(n)]
    fh, dxf = float(mcfg['fine_half_ft']), float(mcfg['dx_fine_ft'])
    th, dxt = float(mcfg['trans_half_ft']), float(mcfg['dx_trans_ft'])
    for c in np.atleast_1d(centres):
        parts.append(c - fh + dxf * np.arange(int(np.floor(2 * fh / dxf)) + 1))
        parts.append(c - th + dxt * np.arange(int(np.floor(2 * th / dxt)) + 1))
    m = np.sort(np.concatenate(parts))
    m = m[(m >= lo) & (m <= hi)]
    tol = float(mcfg['dedupe_tol_ft'])
    keep = np.ones(m.size, dtype=bool)
    last = m[0]
    for i in range(1, m.size):
        if m[i] - last < tol:
            keep[i] = False
        else:
            last = m[i]
    x = m[keep]
    rec = {'mode': 'uniform_base_plus_local_refinement',
           'dx_base_ft': dxb, 'dx_fine_ft': dxf, 'fine_half_ft': fh,
           'dx_trans_ft': dxt, 'trans_half_ft': th, 'dedupe_tol_ft': tol,
           'n_refine_centres': int(np.atleast_1d(centres).size),
           'n_nodes_before_dedupe': int(m.size),
           'n_nodes_removed_by_dedupe': int((~keep).sum()),
           'n_degenerate_calls': 0,
           'note': ('no legacy refine_mesh call; its degenerate branch (A1: '
                    '94/113 calls in 102r, 84/113 in 103r) is avoided entirely')}
    return x, rec


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

class Inputs:
    """Everything read from disk, loaded once per process."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.t0 = datetime.datetime.fromisoformat(cfg['window']['t_start'])
        self.t1 = datetime.datetime.fromisoformat(cfg['window']['t_end'])
        self.md_table = rd.load_gauge_md_table()
        self.prod_md = rd.load_gauge_md_table(rd.PROD_GAUGE_MD_NPZ,
                                              check_reference=False)
        g, md, dd = rd.production_drawdown(self.t0, self.t1)
        self.gauges, self.gauge_md, self.obs_dd = g, md, dd
        self.obs_series = {}
        self._drv = {}

    def gauge_series(self, well, n):
        key = (well, int(n))
        if key not in self._drv:
            from fiberis.analyzer.Data1D import Data1D_Gauge
            tmpl = (rd.SWELL_GAUGE_TEMPLATE if well == 's_well'
                    else rd.PROD_GAUGE_TEMPLATE)
            f = Data1D_Gauge.Data1DGauge()
            f.load_npz(rd.repo_path(tmpl.format(n=int(n))))
            f.crop(self.t0, self.t1)
            t = np.asarray(f.taxis, dtype=float)
            v = np.asarray(f.data, dtype=float)
            if t.size == 0:
                raise ValueError(f'{well} gauge {n}: crop returned no samples')
            self._drv[key] = (t, v)
        return self._drv[key]

    def driver(self, well, n):
        """Producer driver series, hard-zero dropouts removed.

        `rev2_data.production_drawdown(well='prod')` masks the exact-zero
        dropout every producer gauge carries; the same mask is applied here.
        At the legacy dt = 360000 s the dropout fell between steps and was
        harmless (B2 3.3), but at dt = 60 s a single 0 psi sample would be
        interpolated into a ~3400 psi notch lasting two sample intervals.
        """
        t, v = self.gauge_series(well, n)
        n_zero = int(np.sum(v == 0.0))
        keep = v != 0.0
        t, v = t[keep], v[keep]
        return t - t[0], v, n_zero


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------

def spec_tag(sp):
    def f(x):
        return ('%g' % x).replace('.', 'p').replace('-', 'm').replace('+', '')
    return (f"{sp['selection']}_{sp['barrier_mode']}_r{f(sp['ratio'])}"
            f"_pad{int(sp['pad']):05d}_D{f(sp['D'])}_{sp['driver']}"
            f"_dt{f(sp['dt'])}")


def barrier_centres_for(sp, frac_md, inp):
    mode = sp['barrier_mode']
    if mode == 'none':
        return np.array([]), 'no barrier anywhere (null control)'
    if mode == 'at_source':
        return frac_md, ('one barrier centred on every conductive fracture, '
                         'i.e. on every Dirichlet node -- the literal reading '
                         'of "each fracture isolated by its own barrier"')
    if mode == 'between':
        if frac_md.size < 2:
            return np.array([]), 'fewer than two fractures: no midpoints exist'
        return (0.5 * (frac_md[:-1] + frac_md[1:]),
                ('one barrier at the midpoint of every adjacent fracture pair '
                 '-- the alternative reading, in which the barriers cut the '
                 'annulus into one compartment per fracture'))
    if mode == 'frachit_all':
        hits = np.sort(np.concatenate(
            [rd.load_frac_hits(st) for st in range(1, 21)]))
        return hits, ('barriers at all 112 catalogued frac hits with a single '
                      'Dirichlet source, reproducing 103r -- the configuration '
                      'the two curves in Fig. 7b actually come from')
    raise ValueError(f"unknown barrier_mode {mode!r}")


def run_one(args):
    sp, cfg, outdir, config_path = args
    inp = _INPUTS if _INPUTS is not None else Inputs(cfg)
    picks = _PICKS
    tag = spec_tag(sp)
    rundir = os.path.join(outdir, sp.get('run_subdir', 'runs'), tag)
    mpath = os.path.join(rundir, 'manifest.json')
    if os.path.exists(mpath):
        # House rule 2: never overwrite an existing output. Re-read what that
        # run recorded instead, so an interrupted sweep can be resumed without
        # losing the runs it already did.
        with open(mpath) as fh:
            prev = json.load(fh)
        out = dict(prev.get('results') or {})
        out.update(tag=tag, reused_existing_run=True,
                   manifest=os.path.relpath(mpath, rd.REPO_ROOT),
                   summary_npz=os.path.relpath(
                       os.path.join(rundir, 'summary.npz'), rd.REPO_ROOT))
        return out
    rm.assert_absent([mpath])
    os.makedirs(rundir, exist_ok=True)
    t_wall = time.time()

    frac_md, sel_rule = selection_md(sp['selection'], picks, cfg)
    bcent, bnote = barrier_centres_for(sp, frac_md, inp)
    D0, ratio, pad, dt = sp['D'], sp['ratio'], sp['pad'], sp['dt']
    mcfg = cfg['mesh']
    margin = float(mcfg['core_margin_ft'])
    anchors = np.concatenate([frac_md, bcent, inp.gauge_md]) if bcent.size \
        else np.concatenate([frac_md, inp.gauge_md])
    lo = float(anchors.min()) - margin - pad
    hi = float(anchors.max()) + margin + pad
    refine_at = np.unique(np.concatenate([frac_md, bcent])) if bcent.size \
        else np.unique(frac_md)

    with rm.RunRecorder(mpath, study_id=cfg['study_id'], task_id=TASK,
                        config=cfg, config_path=config_path, run_label=tag,
                        require_modules=('rev2_core', 'rev2_data')) as rec:
        x, mrec = build_mesh(lo, hi, refine_at, mcfg)
        mrec['refine_centres_md_ft'] = [float(v) for v in refine_at]

        if ratio >= 1.0 or bcent.size == 0:
            dprof = np.full(x.size, float(D0))
            brep = None
        else:
            dprof, brep = rc.build_barrier_profile(
                x, float(D0), bcent, float(cfg['barrier']['w_ft']), float(ratio),
                on_empty=cfg['barrier']['on_empty'], return_report=True)
            if brep['n_fallback'] != 0:
                raise RuntimeError(
                    f"{tag}: {brep['n_fallback']} barrier(s) fell back to the "
                    f"nearest node -- the mesh does not resolve w")

        sidx = [int(np.argmin(np.abs(x - m))) for m in frac_md]
        if len(set(sidx)) != len(sidx):
            raise RuntimeError(f'{tag}: two fractures snapped to one node')
        gidx = [int(np.argmin(np.abs(x - m))) for m in inp.gauge_md]

        dwell = cfg['drivers'][sp['driver']]['well']
        dg = cfg['drivers'][sp['driver']]['gauge']
        if dg == 'nearest':
            # one driver per fracture: the producer gauge nearest in MD
            per_src = [int(inp.prod_md.numbers[
                int(np.argmin(np.abs(inp.prod_md.md_ft - m)))]) for m in frac_md]
        else:
            per_src = [int(dg)] * len(sidx)
        series, n_zero_tot = {}, 0
        for n in sorted(set(per_src)):
            ta, vv, nz = inp.driver(dwell, n)
            series[n] = (ta, vv)
            n_zero_tot += nz
        ref_n = per_src[0]
        ta0, vv0 = series[ref_n]
        t_total = float(max(series[n][0][-1] for n in series))
        u0 = np.full(x.size, float(vv0[0]))

        taxis, trace = rc.solve_forward_multi(
            x, dprof, float(dt), t_total,
            [series[n][0] for n in per_src], [series[n][1] for n in per_src],
            sidx, initial=u0, t0=0.0, record_idx=gidx,
            theta=float(cfg['solver']['theta']),
            lambda_leak=float(cfg['solver']['lambda_leak']),
            p0=float(cfg['solver']['p0_psi']),
            interface_avg=cfg['solver']['interface_avg'])

        per = []
        for k, g in enumerate(inp.gauges):
            ot, ov = inp.gauge_series('s_well', int(g))
            ot = ot - ot[0]
            sim = np.interp(ot, taxis, trace[:, k])
            per.append({'gauge': int(g), 'md_ft': float(inp.gauge_md[k]),
                        'mesh_idx': int(gidx[k]),
                        'sim_drawdown_psi': float(sim[0] - sim[-1]),
                        'sim_drawdown_raw_psi': float(trace[0, k] - trace[-1, k]),
                        'obs_drawdown_psi': float(inp.obs_dd[k]),
                        'residual_psi': float((sim[0] - sim[-1])
                                              - inp.obs_dd[k])})
        sim_dd = np.array([p['sim_drawdown_psi'] for p in per])
        obs_dd = np.asarray(inp.obs_dd, dtype=float)
        res = sim_dd - obs_dd
        prof = {
            'sim_mean_psi': float(sim_dd.mean()),
            'sim_std_psi': float(sim_dd.std(ddof=0)),
            'sim_spread_psi': float(sim_dd.max() - sim_dd.min()),
            'obs_mean_psi': float(obs_dd.mean()),
            'obs_std_psi': float(obs_dd.std(ddof=0)),
            'obs_spread_psi': float(obs_dd.max() - obs_dd.min()),
            'rmse_psi': float(np.sqrt(np.mean(res ** 2))),
            'bias_psi': float(res.mean()),
            'rmse_after_bias_removal_psi': float(np.sqrt(
                np.mean((res - res.mean()) ** 2))),
            'max_abs_residual_psi': float(np.abs(res).max()),
            'rmse_of_constant_at_obs_mean_psi': float(obs_dd.std(ddof=0)),
            'uniformity_index': float(sim_dd.std(ddof=0) / obs_dd.std(ddof=0)),
            'corr_with_obs': (float(np.corrcoef(sim_dd, obs_dd)[0, 1])
                              if sim_dd.std() > 1e-9 else None)}

        step = max(1, int(np.ceil(taxis.size / int(cfg['trace_save_max_rows']))))
        npz = os.path.join(rundir, 'summary.npz')
        np.savez_compressed(
            npz, mesh_md_ft=x, diffusivity_ft2_s=dprof,
            taxis_sub_s=taxis[::step], traces_sub_psi=trace[::step],
            trace_first_psi=trace[0], trace_last_psi=trace[-1],
            trace_subsample_step=np.array([step]),
            gauge_numbers=np.asarray(inp.gauges), gauge_md_ft=inp.gauge_md,
            gauge_mesh_idx=np.asarray(gidx),
            obs_drawdown_psi=obs_dd, sim_drawdown_psi=sim_dd,
            fracture_md_ft=frac_md, source_mesh_idx=np.asarray(sidx),
            barrier_centre_md_ft=bcent,
            driver_taxis_s=ta0, driver_psi=vv0)

        drecs = {}
        for n in sorted(set(per_src)):
            ta, vv = series[n]
            drecs[n] = rm.driver_record(
                kind='gauge_series', baseline_removal='none_absolute_psi',
                value_units='psi',
                series_path=rd.repo_path(
                    (rd.SWELL_GAUGE_TEMPLATE if dwell == 's_well'
                     else rd.PROD_GAUGE_TEMPLATE).format(n=n)),
                gauge_number=n,
                gauge_md_ft=(inp.md_table if dwell == 's_well'
                             else inp.prod_md).md_of(n),
                taxis=ta, values=vv,
                time_start=inp.t0.isoformat(), time_end=inp.t1.isoformat())

        inputs = [(rd.repo_path(rd.SWELL_GAUGE_TEMPLATE.format(n=int(g))),
                   'gauge_series', f's_well_gauge{int(g)}')
                  for g in inp.gauges]
        inputs += [(rd.repo_path(rd.PROD_GAUGE_TEMPLATE.format(n=n)),
                    'gauge_series', f'driver_prod_gauge{n}')
                   for n in sorted(set(per_src))]
        inputs += [(rd.repo_path(rd.SWELL_GAUGE_MD_NPZ), 'geometry',
                    'gauge_md_swell'),
                   (rd.repo_path(rd.PROD_GAUGE_MD_NPZ), 'geometry',
                    'gauge_md_prod')]
        if sp['selection'].startswith('dss'):
            inputs.append((rd.repo_path(cfg['dss']['path']), 'das',
                           'swell_rfs_strain_change'))
        if sp['selection'].startswith('frachit') or \
                sp['barrier_mode'] == 'frachit_all':
            for st in range(1, 21):
                inputs.append((rd.repo_path(
                    rd.SWELL_FRAC_HIT_TEMPLATE.format(stage=st)), 'geometry',
                    f'frac_hit_stage_{st}'))
        rec.declare_inputs(inputs)
        rec.declare_output(npz, role='arrays_npz',
                           note='mesh, D(x), sub-sampled gauge traces, exact '
                                'first/last field, drawdown profiles, fracture '
                                'and barrier MDs, driver series')
        rec.set_source(rm.source_protocol(
            application='dirichlet_node',
            solver_class='rev2_core.solve_forward_multi',
            placement_rule=(f"selection={sp['selection']}: {sel_rule}; "
                            f"snapped to the nearest mesh node"),
            sources=[rm.source_record(x, md_requested_ft=float(m),
                                      mesh_idx=int(i), driver=drecs[n],
                                      label=f'frac{j}', index_in_source_list=j)
                     for j, (m, i, n) in enumerate(zip(frac_md, sidx, per_src))],
            targets=[{'gauge': int(g), 'md_ft': float(m), 'mesh_idx': int(i)}
                     for g, m, i in zip(inp.gauges, inp.gauge_md, gidx)],
            time_level=cfg['solver']['source_time_level'],
            phase_chaining='none', boundary_conditions=boundary_group(cfg)))
        rec.set_numerics(rm.numerics(
            time=rm.time_record(taxis, mode='fixed',
                                theta=float(cfg['solver']['theta']),
                                t_total_requested_s=t_total,
                                dt_requested_s=float(dt),
                                source_time_level='n'),
            mesh=rm.mesh_record(x, dx_requested_ft=float(mcfg['dx_base_ft']),
                                window_md_ft=[lo + pad, hi - pad],
                                pad_low_ft=float(pad), pad_high_ft=float(pad),
                                refinement=mrec),
            interface_avg=cfg['solver']['interface_avg'],
            boundary=boundary_group(cfg),
            diffusivity={'family': 'uniform', 'D_ft2_s': float(D0)},
            barriers=barrier_records(x, brep, D0, ratio, sp['barrier_mode']),
            leakage={'lambda_leak': float(cfg['solver']['lambda_leak']),
                     'p0_psi': float(cfg['solver']['p0_psi'])},
            kernel=kernel_group(), rng=rm.NONE_DECLARED,
            parallel={'processes': 1, 'note': 'one solve per worker process'}))
        out = {'tag': tag, 'block': sp.get('block'),
               'selection': sp['selection'], 'selection_rule': sel_rule,
               'n_fractures': int(frac_md.size),
               'fracture_md_ft': [float(v) for v in frac_md],
               'barrier_mode': sp['barrier_mode'], 'barrier_note': bnote,
               'n_barrier_centres': int(bcent.size),
               'ratio': float(ratio), 'pad_ft': float(pad),
               'D_ft2_s': float(D0), 'dt_s': float(dt),
               'driver': sp['driver'], 'driver_gauges': sorted(set(per_src)),
               'n_zero_samples_removed_from_driver': n_zero_tot,
               'nx': int(x.size), 'n_steps': int(taxis.size),
               't_total_s': t_total,
               'mesh_span_md_ft': [float(x[0]), float(x[-1])],
               'per_gauge': per, 'profile': prof,
               'barrier_report': (None if brep is None else {
                   'n_barriers': brep['n_barriers'],
                   'n_merged_groups': brep['n_merged_groups'],
                   'n_fallback': brep['n_fallback'],
                   'total_equivalent_width_ft':
                       brep['total_equivalent_width_ft'],
                   'excess_resistance_s_per_ft':
                       brep['excess_resistance_s_per_ft']}),
               'wall_s': time.time() - t_wall}
        rec.set_results(out)
        rec.note(f"{bnote}. Driver: {sp['driver']} "
                 f"({n_zero_tot} hard-zero sample(s) removed).")
    out['manifest'] = os.path.relpath(mpath, rd.REPO_ROOT)
    out['summary_npz'] = os.path.relpath(npz, rd.REPO_ROOT)
    print(f"  {tag}: nx={x.size} steps={taxis.size} "
          f"mean={prof['sim_mean_psi']:.1f} spread={prof['sim_spread_psi']:.2f} "
          f"rmse={prof['rmse_psi']:.1f} psi ({out['wall_s']:.1f} s)",
          flush=True)
    return out


_INPUTS = None
_PICKS = None


def _init(cfg, picks):
    global _INPUTS, _PICKS
    _INPUTS = Inputs(cfg)
    _PICKS = picks


# ---------------------------------------------------------------------------
# figures and the aggregate table
# ---------------------------------------------------------------------------

C_OBS = '#111111'
C_B = '#1f77b4'
C_B2 = '#17becf'
C_CE1 = '#d62728'
C_CE2 = '#ff7f0e'


def _dpi_save(fig, path, dpi=300):
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists; version the filename '
                              f'(house rule 9)')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f'  wrote {path}')
    return path


def fig_dss_picks(cfg, picks, outdir, ver='v1'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d, net, meta = dss_net_tensile(cfg)
    cal = float(cfg['dss']['md_calibration_ft'])
    mad = meta['sigma_mad_ue']
    hits = np.sort(np.concatenate([rd.load_frac_hits(s) for s in range(1, 21)]))
    md_table = rd.load_gauge_md_table()
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                            gridspec_kw={'height_ratios': [2, 1]})
    ax = axs[0]
    ax.plot(d + cal, net, lw=0.5, color='#444444')
    for k, c in [(3, '#c6dbef'), (4, '#6baed6'), (5, '#2171b5'), (6, '#08306b')]:
        s = picks['sets'].get('dss_k%d' % k)
        if s is None:
            continue
        ax.plot(s['md_ft'], s['net_tensile_ue'], 'v', ms=4 + (k - 3), color=c,
                label=f"k = {k} sigma_MAD  (n = {len(s['md_ft'])})")
    for lev, ls in [(3, ':'), (4, '--'), (5, '-.'), (6, '-')]:
        ax.axhline(lev * mad, color='grey', lw=0.6, ls=ls)
    ax.set_ylabel('net tensile strain change (ue)')
    ax.set_ylim(-3, max(6.0, float(np.max(net)) * 1.1))
    ax.legend(fontsize=7, loc='upper left', ncol=2)
    ax.set_title('DSS (RFS strain change), S well: net tensile change over the '
                 f"{meta['taxis_span_s'][1] / 3600:.0f} h record\n"
                 f"MD = daxis + {cal} ft (the manuscript DSS figure's own "
                 'calibration)', fontsize=9)
    ax = axs[1]
    ax.vlines(hits, 0, 1, color='#2ca02c', lw=0.6,
              label=f'catalogued frac hits (n = {hits.size})')
    ax.vlines(md_table.md_ft, 1.2, 2.2, color='#111111', lw=1.2,
              label='S-well gauges (n = 15)')
    s = picks['sets']['dss_k%d' % int(cfg['dss']['primary_k'])]
    ax.vlines(s['md_ft'], 2.4, 3.4, color=C_B, lw=1.0,
              label=f"DSS tensile picks, k = {int(cfg['dss']['primary_k'])} "
                    f"(n = {len(s['md_ft'])})")
    ax.set_yticks([])
    ax.set_ylim(-0.2, 3.8)
    ax.set_xlabel('measured depth (ft)')
    ax.legend(fontsize=7, loc='lower left', ncol=3)
    ax.set_xlim(11500, 17000)
    return _dpi_save(fig, os.path.join(outdir, f'fig01_e4_dss_picks_{ver}.png'))


_KEY_ALIAS = {'pad': 'pad_ft', 'D': 'D_ft2_s', 'dt': 'dt_s'}


def _get(runs, **kw):
    """Select runs by result-record fields; `pad`/`D`/`dt` alias the _ft/_s names."""
    def ok(r):
        for k, v in kw.items():
            rk = _KEY_ALIAS.get(k, k)
            if rk not in r:
                return False
            if isinstance(v, float):
                if abs(float(r[rk]) - v) > 1e-12 * max(1.0, abs(v)):
                    return False
            elif r[rk] != v:
                return False
        return True
    return [r for r in runs if ok(r)]


def fig_profile(cfg, runs, outdir, ver='v1'):
    """THE deliverable: one worked example plus the two counter-examples."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base = dict(pad=5000.0, D=140.0, driver='prod_g1', dt=60.0)
    curves = []
    hb = _get(runs, selection='dss_k4', barrier_mode='at_source',
              ratio=1e-05, **base)
    hb2 = _get(runs, selection='frachit_all', barrier_mode='at_source',
               ratio=1e-05, **base)
    ce1 = _get(runs, selection='fig7b_single', barrier_mode='frachit_all',
               ratio=1.0, **base)
    ce2 = _get(runs, selection='fig7b_single', barrier_mode='frachit_all',
               ratio=0.001, **base)
    ce2l = _get(runs, selection='fig7b_single', barrier_mode='frachit_all',
                ratio=0.001, pad=0.0, D=140.0, driver='prod_g1', dt=60.0)
    if not (hb and hb2 and ce1 and ce2):
        print('  fig_profile: missing runs, skipped')
        return None
    md = np.array([p['md_ft'] for p in hb[0]['per_gauge']])
    gn = np.array([p['gauge'] for p in hb[0]['per_gauge']])
    obs = np.array([p['obs_drawdown_psi'] for p in hb[0]['per_gauge']])

    def sim(r):
        return np.array([p['sim_drawdown_psi'] for p in r[0]['per_gauge']])

    curves = [
        (sim(hb), 's-', C_B, 1.6,
         f"HYPOTHESIS B: {hb[0]['n_fractures']} DSS-picked fractures, each with "
         'a 2 ft barrier at ratio 1e-5'),
        (sim(hb2), '^-', C_B2, 1.4,
         f"HYPOTHESIS B: {hb2[0]['n_fractures']} catalogued frac hits, each with "
         'a 2 ft barrier at ratio 1e-5'),
        (sim(ce1), 'v--', C_CE1, 1.3,
         'COUNTER-EXAMPLE 1: one fracture, no barrier'),
        (sim(ce2), 'd--', C_CE2, 1.3,
         'COUNTER-EXAMPLE 2: one fracture, 112 barriers at ratio 1e-3')]

    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.4),
                            gridspec_kw={'width_ratios': [1.35, 1]})
    for ax, zoom in zip(axs, (False, True)):
        ax.plot(md, obs, 'o-', color=C_OBS, lw=2.2, ms=6, zorder=5,
                label='MEASURED drawdown, 2020-04-01 to 2020-06-01')
        for y, st, c, lw, lab in curves:
            ax.plot(md, y, st, color=c, lw=lw, ms=5, label=lab)
        if ce2l and not zoom:
            ax.plot(md, sim(ce2l), ':', color=C_CE2, lw=1.2, alpha=0.8,
                    label='counter-example 2 on the legacy unpadded box '
                          '(what Fig. 7b actually plots)')
        ax.grid(alpha=0.25)
        ax.invert_xaxis()
        ax.set_xlabel('S-well measured depth (ft)')
    axs[0].set_ylabel('two-month drawdown (psi)')
    axs[0].legend(fontsize=7.2, loc='lower center')
    axs[0].set_title('(a) full range', fontsize=9)
    lo = min(float(np.min([c[0] for c in curves])), float(obs.min()))
    axs[1].set_ylim(3150, 3450)
    axs[1].set_title('(b) the flat family, magnified -- hypothesis B and the '
                     'no-barrier\nsingle fracture are not distinguishable by '
                     'this profile', fontsize=9)
    for m, g in zip(md, gn):
        axs[1].annotate(str(int(g)), (m, 3155), fontsize=6, ha='center',
                        va='bottom', color='grey')
    fig.suptitle('Production-period drawdown at the fifteen S-well gauges '
                 '(numeric gauge order): the adopted scenario against the two '
                 'counter-examples\n'
                 'converged domain (5000 ft pad both ends), sources pinned to '
                 'physical MDs, D = 140 ft$^2$/s, fixed dt = 60 s, driver = '
                 'producer gauge 1', fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _dpi_save(fig, os.path.join(outdir,
                                       f'fig02_e4_drawdown_profile_{ver}.png'))


def fig_selection_spread(cfg, runs, outdir, ver='v1'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base = dict(pad=5000.0, D=140.0, driver='prod_g1', dt=60.0,
                barrier_mode='at_source')
    sels = ['dss_k3', 'dss_k4', 'dss_k5', 'dss_k6', 'dss_k4_eofcal',
            'frachit_all', 'frachit_stage7']
    ratios = [1.0, 0.001, 1e-05, 1e-06, 1e-08]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    obs = None
    for s in sels:
        r = _get(runs, selection=s, ratio=1e-05, **base)
        if not r:
            continue
        md = np.array([p['md_ft'] for p in r[0]['per_gauge']])
        sd = np.array([p['sim_drawdown_psi'] for p in r[0]['per_gauge']])
        obs = np.array([p['obs_drawdown_psi'] for p in r[0]['per_gauge']])
        axs[0].plot(md, sd, '-o', ms=3, lw=1.1,
                    label=f"{s} (n = {r[0]['n_fractures']})")
    if obs is not None:
        axs[0].plot(md, obs, 'k-o', lw=2, ms=5, label='measured')
    axs[0].set_xlabel('MD (ft)')
    axs[0].set_ylabel('two-month drawdown (psi)')
    axs[0].set_title('(a) fracture selection, ratio 1e-5', fontsize=9)
    axs[0].legend(fontsize=6.5)
    axs[0].invert_xaxis()
    axs[0].grid(alpha=0.25)
    for s in sels:
        xs, ys, zs = [], [], []
        for rt in ratios:
            r = _get(runs, selection=s, ratio=rt, **base)
            if not r:
                continue
            xs.append(rt)
            ys.append(r[0]['profile']['rmse_psi'])
            zs.append(r[0]['profile']['sim_spread_psi'])
        if xs:
            axs[1].semilogx(xs, ys, '-o', ms=4, label=s)
            axs[2].loglog(xs, np.maximum(zs, 1e-3), '-o', ms=4, label=s)
    if obs is not None:
        axs[1].axhline(float(obs.std(ddof=0)), color='k', ls='--', lw=1.2,
                       label='constant at the observed mean')
        axs[2].axhline(float(obs.max() - obs.min()), color='k', ls='--',
                       lw=1.2, label='observed spread')
    axs[1].set_xlabel('barrier reduction ratio')
    axs[1].set_ylabel('profile RMSE over the 15 gauges (psi)')
    axs[1].set_title('(b) misfit vs barrier strength', fontsize=9)
    axs[1].legend(fontsize=6.5)
    axs[1].grid(alpha=0.25)
    axs[2].set_xlabel('barrier reduction ratio')
    axs[2].set_ylabel('simulated max - min over the 15 gauges (psi)')
    axs[2].set_title('(c) how non-uniform the prediction is', fontsize=9)
    axs[2].legend(fontsize=6.5)
    axs[2].grid(alpha=0.25)
    fig.tight_layout()
    return _dpi_save(fig, os.path.join(
        outdir, f'fig03_e4_selection_spread_{ver}.png'))


def fig_padding(cfg, runs, outdir, ver='v1'):
    """Domain robustness: the test B2 says every E-group re-run must pass."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    combos = [('dss_k4', 'at_source', 1e-05, C_B,
               'hypothesis B, 36 DSS picks'),
              ('frachit_all', 'at_source', 1e-05, C_B2,
               'hypothesis B, 112 frac hits'),
              ('dss_k4', 'at_source', 1.0, '#7f7f7f',
               'hypothesis B, 36 DSS picks, barrier OFF'),
              ('fig7b_single', 'frachit_all', 1.0, C_CE1,
               'counter-example 1: one fracture, no barrier'),
              ('fig7b_single', 'frachit_all', 0.001, C_CE2,
               'counter-example 2: one fracture, 112 barriers @1e-3')]
    for sel, bm, rt, c, lab in combos:
        pads, mean, spread = [], [], []
        for pad in [0.0, 2000.0, 5000.0, 20000.0]:
            r = _get(runs, selection=sel, barrier_mode=bm, ratio=rt, pad=pad,
                     D=140.0, driver='prod_g1', dt=60.0)
            if not r:
                continue
            pads.append(pad)
            mean.append(r[0]['profile']['sim_mean_psi'])
            spread.append(r[0]['profile']['sim_spread_psi'])
        if not pads:
            continue
        axs[0].plot(pads, mean, '-o', color=c, label=lab)
        ref = mean[0]
        axs[1].plot(pads, [100.0 * (v - ref) / max(abs(ref), 1e-9)
                           for v in mean], '-o', color=c, label=lab)
        axs[2].semilogy(pads, np.maximum(spread, 1e-3), '-o', color=c,
                        label=lab)
    axs[0].set_ylabel('mean simulated drawdown over the 15 gauges (psi)')
    axs[0].set_title('(a) level', fontsize=9)
    axs[1].set_ylabel('change from the unpadded legacy box (%)')
    axs[1].set_title('(b) the domain-robustness test', fontsize=9)
    axs[2].set_ylabel('simulated max - min over the 15 gauges (psi)')
    axs[2].set_title('(c) shape', fontsize=9)
    for ax in axs:
        ax.set_xlabel('symmetric pad added at both ends (ft)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    return _dpi_save(fig, os.path.join(outdir, f'fig04_e4_padding_{ver}.png'))


def fig_level(cfg, runs, outdir, ver='v1'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    drvs = ['prod_g1', 'prod_g2', 'prod_g3', 'prod_g4', 'prod_g5',
            'prod_nearest']
    for sel, c in [('dss_k4', C_B), ('frachit_all', C_B2)]:
        xs, ys = [], []
        for k, dv in enumerate(drvs):
            r = _get(runs, selection=sel, barrier_mode='at_source',
                     ratio=1e-05, pad=5000.0, D=140.0, driver=dv, dt=60.0)
            if not r:
                continue
            xs.append(k)
            ys.append(r[0]['profile']['sim_mean_psi'])
        axs[0].plot(xs, ys, '-o', color=c, label=sel)
    r0 = _get(runs, selection='dss_k4', barrier_mode='at_source', ratio=1e-05,
              pad=5000.0, D=140.0, driver='prod_g1', dt=60.0)
    if r0:
        o = np.array([p['obs_drawdown_psi'] for p in r0[0]['per_gauge']])
        axs[0].axhline(float(o.mean()), color='k', ls='--',
                       label='measured mean')
        axs[0].fill_between([-0.4, len(drvs) - 0.6], o.mean() - o.std(),
                            o.mean() + o.std(), color='k', alpha=0.08)
    axs[0].set_xticks(range(len(drvs)))
    axs[0].set_xticklabels(drvs, rotation=30, fontsize=7)
    axs[0].set_ylabel('mean simulated drawdown (psi)')
    axs[0].set_title('(a) which producer gauge drives the fractures', fontsize=9)
    axs[0].legend(fontsize=7)
    axs[0].grid(alpha=0.25)
    for sel, c in [('dss_k4', C_B), ('frachit_all', C_B2)]:
        xs, ys, sp = [], [], []
        for D in [1.4, 14.0, 140.0, 550.0, 1150.0]:
            r = _get(runs, selection=sel, barrier_mode='at_source',
                     ratio=1e-05, pad=5000.0, D=D, driver='prod_g1', dt=60.0)
            if not r:
                continue
            xs.append(D)
            ys.append(r[0]['profile']['sim_mean_psi'])
            sp.append(r[0]['profile']['sim_spread_psi'])
        if xs:
            axs[1].semilogx(xs, ys, '-o', color=c, label=f'{sel} mean')
            axs[1].semilogx(xs, sp, '--s', color=c, alpha=0.6,
                            label=f'{sel} max-min')
    axs[1].set_xlabel('annulus diffusivity D (ft$^2$/s)')
    axs[1].set_ylabel('psi')
    axs[1].set_title('(b) annulus diffusivity', fontsize=9)
    axs[1].legend(fontsize=7)
    axs[1].grid(alpha=0.25)
    fig.tight_layout()
    return _dpi_save(fig, os.path.join(outdir, f'fig05_e4_level_{ver}.png'))


def aggregate(cfg, runs, outdir, ver='v1'):
    """The quotable table, and every sensitivity spread, as one JSON."""
    base = dict(pad=5000.0, D=140.0, driver='prod_g1', dt=60.0,
                barrier_mode='at_source')

    def one(r):
        p = r['profile']
        return {'tag': r['tag'], 'selection': r['selection'],
                'n_fractures': r['n_fractures'], 'ratio': r['ratio'],
                'pad_ft': r['pad_ft'], 'D_ft2_s': r['D_ft2_s'],
                'driver': r['driver'], 'dt_s': r['dt_s'],
                'barrier_mode': r['barrier_mode'],
                'sim_mean_psi': p['sim_mean_psi'],
                'sim_spread_psi': p['sim_spread_psi'],
                'sim_std_psi': p['sim_std_psi'],
                'rmse_psi': p['rmse_psi'], 'bias_psi': p['bias_psi'],
                'rmse_after_bias_removal_psi': p['rmse_after_bias_removal_psi'],
                'corr_with_obs': p['corr_with_obs'],
                'manifest': r.get('manifest')}

    out = {'n_runs': len(runs),
           'observed': None, 'headline': {}, 'sensitivities': {},
           'all_runs': [one(r) for r in sorted(runs, key=lambda r: r['tag'])]}
    if runs:
        pg = runs[0]['per_gauge']
        o = np.array([p['obs_drawdown_psi'] for p in pg])
        out['observed'] = {
            'gauges': [p['gauge'] for p in pg],
            'md_ft': [p['md_ft'] for p in pg],
            'drawdown_psi': [p['obs_drawdown_psi'] for p in pg],
            'mean_psi': float(o.mean()), 'std_psi': float(o.std(ddof=0)),
            'spread_psi': float(o.max() - o.min()),
            'note': 'rev2_data.production_drawdown, NUMERIC gauge order'}
    for name, kw in [
            ('hypothesis_B_dss', dict(selection='dss_k4', ratio=1e-05, **base)),
            ('hypothesis_B_frachit', dict(selection='frachit_all', ratio=1e-05,
                                          **base)),
            ('counterexample_no_barrier',
             dict(selection='fig7b_single', barrier_mode='frachit_all',
                  ratio=1.0, pad=5000.0, D=140.0, driver='prod_g1', dt=60.0)),
            ('counterexample_with_barrier',
             dict(selection='fig7b_single', barrier_mode='frachit_all',
                  ratio=0.001, pad=5000.0, D=140.0, driver='prod_g1',
                  dt=60.0))]:
        r = _get(runs, **kw)
        if r:
            out['headline'][name] = one(r[0])
            out['headline'][name]['per_gauge'] = r[0]['per_gauge']

    def spread(label, subset, key='sim_mean_psi'):
        if not subset:
            return
        v = [s['profile'][key] for s in subset]
        out['sensitivities'][label] = {
            'n': len(subset), 'key': key,
            'min': float(min(v)), 'max': float(max(v)),
            'range': float(max(v) - min(v)),
            'by_run': {s['tag']: float(s['profile'][key]) for s in subset}}

    sels = ['dss_k3', 'dss_k4', 'dss_k5', 'dss_k6', 'dss_k4_eofcal',
            'frachit_all', 'frachit_stage7']
    spread('fracture_selection_at_ratio1em5',
           [r for r in runs if r['selection'] in sels and r['ratio'] == 1e-05
            and r['barrier_mode'] == 'at_source' and r['pad_ft'] == 5000.0
            and r['D_ft2_s'] == 140.0 and r['driver'] == 'prod_g1'
            and r['dt_s'] == 60.0])
    spread('fracture_selection_at_ratio1em5_rmse',
           [r for r in runs if r['selection'] in sels and r['ratio'] == 1e-05
            and r['barrier_mode'] == 'at_source' and r['pad_ft'] == 5000.0
            and r['D_ft2_s'] == 140.0 and r['driver'] == 'prod_g1'
            and r['dt_s'] == 60.0], key='rmse_psi')
    spread('barrier_ratio_dss_k4',
           [r for r in runs if r['selection'] == 'dss_k4'
            and r['barrier_mode'] == 'at_source' and r['pad_ft'] == 5000.0
            and r['D_ft2_s'] == 140.0 and r['driver'] == 'prod_g1'
            and r['dt_s'] == 60.0])
    spread('driver_gauge_dss_k4',
           [r for r in runs if r['selection'] == 'dss_k4'
            and r['barrier_mode'] == 'at_source' and r['ratio'] == 1e-05
            and r['pad_ft'] == 5000.0 and r['D_ft2_s'] == 140.0
            and r['dt_s'] == 60.0])
    spread('padding_dss_k4',
           [r for r in runs if r['selection'] == 'dss_k4'
            and r['barrier_mode'] == 'at_source' and r['ratio'] == 1e-05
            and r['D_ft2_s'] == 140.0 and r['driver'] == 'prod_g1'
            and r['dt_s'] == 60.0])
    spread('padding_single_fracture_barrier',
           [r for r in runs if r['selection'] == 'fig7b_single'
            and r['barrier_mode'] == 'frachit_all' and r['ratio'] == 0.001
            and r['D_ft2_s'] == 140.0 and r['driver'] == 'prod_g1'
            and r['dt_s'] == 60.0])
    spread('diffusivity_dss_k4',
           [r for r in runs if r['selection'] == 'dss_k4'
            and r['barrier_mode'] == 'at_source' and r['ratio'] == 1e-05
            and r['pad_ft'] == 5000.0 and r['driver'] == 'prod_g1'
            and r['dt_s'] == 60.0])
    dtl = sorted([r for r in runs if r['selection'] == 'dss_k4'
                  and r['barrier_mode'] == 'at_source' and r['ratio'] == 1e-05
                  and r['pad_ft'] == 5000.0 and r['D_ft2_s'] == 140.0
                  and r['driver'] == 'prod_g1'], key=lambda r: r['dt_s'])
    if len(dtl) > 1:
        ref = [r for r in dtl if r['dt_s'] == 1.0]
        ref = ref[0] if ref else dtl[0]
        rv = np.array([p['sim_drawdown_psi'] for p in ref['per_gauge']])
        out['sensitivities']['dt_ladder_vs_dt1s'] = {
            'reference_dt_s': ref['dt_s'],
            'per_dt': {str(r['dt_s']): {
                'max_abs_gauge_diff_psi': float(np.max(np.abs(
                    np.array([p['sim_drawdown_psi']
                              for p in r['per_gauge']]) - rv))),
                'mean_psi': r['profile']['sim_mean_psi'],
                'wall_s': r.get('wall_s')} for r in dtl}}
    path = os.path.join(outdir, f'e4_summary_{ver}.json')
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists; version the filename')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'  wrote {path}')
    return out


def stage_figures(cfg, outdir, results_name, ver='v1'):
    rpath = os.path.join(outdir, results_name)
    with open(rpath) as fh:
        runs = json.load(fh)['runs']
    runs = [r for r in runs if 'profile' in r]
    picks = dss_picks(cfg, outdir)
    made = []
    for fn in (fig_dss_picks,):
        made.append(fn(cfg, picks, outdir, ver))
    for fn in (fig_profile, fig_selection_spread, fig_padding, fig_level):
        made.append(fn(cfg, runs, outdir, ver))
    agg = aggregate(cfg, runs, outdir, ver)
    return made, agg


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def expand(cfg):
    specs = []
    for block in cfg['runs']:
        for sel in block['selections']:
            for ratio in block['ratios']:
                for pad in block['pads']:
                    for drv in block['drivers']:
                        for D in block['D']:
                            for dt in block['dts']:
                                for bm in block['barrier_modes']:
                                    specs.append({
                                        'block': block['block'],
                                        'selection': sel, 'ratio': float(ratio),
                                        'pad': float(pad), 'driver': drv,
                                        'D': float(D), 'dt': float(dt),
                                        'barrier_mode': bm})
    seen, uniq = set(), []
    for s in specs:
        t = spec_tag(s)
        if t in seen:
            continue
        seen.add(t)
        uniq.append(s)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=CONFIG)
    ap.add_argument('--outdir', default=OUTDIR)
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--only', default=None,
                    help='fnmatch pattern on the run tag')
    ap.add_argument('--block', default=None, help='fnmatch pattern on the block')
    ap.add_argument('--stage', default='all',
                    choices=['picks', 'runs', 'figures', 'all'])
    ap.add_argument('--fig-version', default='v1')
    ap.add_argument('--run-subdir', default='runs',
                    help='subdirectory of --outdir for the per-run folders; '
                         'version it to re-run without overwriting')
    ap.add_argument('--results-name', default='e4_results_v1.json')
    a = ap.parse_args()

    with open(a.config) as fh:
        cfg = json.load(fh)
    os.makedirs(a.outdir, exist_ok=True)
    t_start = time.time()

    picks = dss_picks(cfg, a.outdir)
    print(f"DSS picks: " + ', '.join(
        f"{k}={len(v['md_ft'])}" for k, v in picks['sets'].items()))
    for r in picks['manuscript_handpick_recovery']:
        print(f"  manuscript hand-pick {r['manuscript_md_ft']:.0f} ft -> "
              f"automatic {r['nearest_automatic_pick_md_ft']:.1f} ft "
              f"({r['offset_ft']:+.1f} ft)")
    if a.stage == 'picks':
        return
    if a.stage == 'figures':
        stage_figures(cfg, a.outdir, a.results_name, a.fig_version)
        return

    specs = expand(cfg)
    if a.only:
        specs = [s for s in specs if fnmatch.fnmatch(spec_tag(s), a.only)]
    if a.block:
        specs = [s for s in specs if fnmatch.fnmatch(s['block'], a.block)]
    print(f"{len(specs)} run(s) to do on {a.procs} process(es)")

    for s in specs:
        s['run_subdir'] = a.run_subdir
    args = [(s, cfg, a.outdir, a.config) for s in specs]
    if a.procs <= 1:
        _init(cfg, picks)
        results = [run_one(x) for x in args]
    else:
        import multiprocessing as mp
        ctx = mp.get_context('fork')
        with ctx.Pool(a.procs, initializer=_init, initargs=(cfg, picks)) as p:
            results = p.map(run_one, args, chunksize=1)

    rpath = os.path.join(a.outdir, a.results_name)
    if os.path.exists(rpath):
        with open(rpath) as fh:
            old = json.load(fh)
        by = {r['tag']: r for r in old['runs']}
    else:
        by = {}
    for r in results:
        by[r['tag']] = r
    with open(rpath, 'w') as fh:
        json.dump({'study_id': cfg['study_id'], 'task_id': TASK,
                   'config': a.config,
                   'written_utc': datetime.datetime.utcnow().isoformat() + 'Z',
                   'wall_s': time.time() - t_start,
                   'dss_picks': picks,
                   'runs': sorted(by.values(), key=lambda r: r['tag'])}, fh,
                  indent=1)
    print(f"wrote {rpath} ({len(by)} runs), {time.time() - t_start:.1f} s")
    if a.stage == 'all':
        stage_figures(cfg, a.outdir, a.results_name, a.fig_version)


if __name__ == '__main__':
    main()
