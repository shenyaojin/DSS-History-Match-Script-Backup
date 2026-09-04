"""C3 AMEND -- re-run the parts of C3 whose reported numbers did not survive review.

Run from the repository root:

    python3 scripts/manuscript_well_leakage/rev2/c3_amend.py \
        --config configs/rev2/c3_precursor_amend.json

This is ADDITIVE. It never touches the products of `c3_precursor.py` or
`c3_precursor_addendum.py`; every file it writes carries a new versioned name and
its own manifest (`manifest_amend.json`, written by `rev2_manifest`). The published
config `configs/rev2/c3_precursor.json` is deliberately left byte-identical so the
config sha256 pinned in the two earlier manifests still verifies.

What it recomputes, and why (defect list:
`output/rev2_20260901/A4/challenge_defects/C3_defects.json`):

1. THE QUIESCENT-START TEST AT BOTH NORMS (blocker + major duplicate). The original
   report quoted the explained fraction of the observed precursor minimum at a
   single D (1150) under a single norm. The two norms disagree by a factor of 2 in
   D and the answer at the far gauges follows D, so the run is re-evaluated at
   D = 550 (the house-rules amplitude-normalised optimum), at the extended run's own
   normalised-norm optimum, at D = 1150, and at the extended run's own absolute-norm
   optimum.

2. THE PER-GAUGE SPREAD UNDER THE CORRECT INITIAL CONDITION (major). The original
   answered "how much of the 47x spread is precursor contamination?" by masking the
   early part of a run that still starts from a zero IC at the window edge. The
   per-gauge single fits are redone here with the quiescent-start forward model.
   The grid runs down to 5 ft^2/s, not 100, because the quiescent-start misfit at g7
   has a SECOND basin below 100 that a grid starting at 100 reports as a censored
   edge optimum.

3. GRID NOISE (minor). The published optima come from one grid. They are re-derived
   here on an independently offset grid of 1.15% spacing so that the size of the
   published "47.0x -> 46.5x" contraction can be compared with the grid resolution.

4. THE MISFIT PARTITION UNDER THREE AVERAGING CONVENTIONS (major duplicate). The
   published 4.6% / 7.1% is the pooled sum-of-squares share; every other headline in
   the report is in the gauge-mean norm.

5. THE R2 UNIFORM ROW AT ITS OWN OPTIMUM (minor). The house-rules R2 uniform row
   (82.33 psi, normalised 0.539) is a Nelder-Mead fit at log10_D = 3.0565024;
   reproducing it at the grid point 1127 leaves a 1.3% gap in the normalised value
   that the original report presented as "the match".

6. THE ONSET MARGIN AGAINST THE PUMPING RESTART (major). The +-1 psi plateau of each
   onset is compared with its own lead over the 11:18:57 restart, which is what
   decides whether the ordering is resolved gauge by gauge.

Plus three figures that carry the corrections: v3 of the timing figure (its
annotation contradicted every data file), v2 of the extended-window figure (its
legend D disagreed with the table it sits beside) and v2 of the refit figure (its
panel (d) put two different models' misfits side by side under one label).

The forward solver is NOT reimplemented: `rev2_core.solve_forward` at theta = 1 /
harmonic / lambda = 0 is bitwise identical to the verified R1 kernel. Data loading
goes through `rev2_data`; the manifest through `rev2_manifest`. The pumping-event
detection is imported verbatim from `c3_precursor` so the amend figure's 11:18:57
and 10:50:34 are the same numbers the published manifest carries.
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.join(os.path.dirname(_HERE), 'baseline_calibration')
sys.path.insert(0, _HERE)
sys.path.insert(0, _BASE)

import rev2_core as rc          # noqa: E402
import rev2_data as rd          # noqa: E402
import rev2_manifest as rm      # noqa: E402
from c3_precursor import load_pumping, pumping_events, moving_average  # noqa: E402

REPO = os.getcwd()
EXT_T0 = datetime.datetime(2020, 3, 16, 10, 20, 0)
EXT_T1 = datetime.datetime(2020, 3, 16, 11, 45, 0)

# The published C3 run's own numbers, quoted here so the amend can be compared with
# them without re-deriving them. Both come from output/rev2_20260901/C3/manifest.json.
PUBLISHED_D_FULL = 1126.748911504546          # absolute-norm (gauge-mean) optimum
PUBLISHED_D_NORM = 549.4426717225856          # amplitude-normalised optimum
# house-rules R2 uniform row, output/r2_diffusivity_profile/r2_manifest.json
R2_UNIFORM_LOG10_D = 3.0565024417549544

_G = {}


def log(m):
    print(f"[c3-amend] {m}", flush=True)


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def first_crossing(t, y, level, i0):
    """First time at or after index i0 at which y reaches level from below."""
    for i in range(i0, len(y)):
        if y[i] >= level:
            if i == i0 or y[i - 1] >= level:
                return float(t[i])
            f = (level - y[i - 1]) / (y[i] - y[i - 1])
            return float(t[i - 1] + f * (t[i] - t[i - 1]))
    return float('nan')


def build(cfg):
    """Window gauges, extended gauges, mesh, source, targets and the two masks."""
    w = cfg['window']
    win = rd.Window(md_min_ft=float(w['md_min_ft']), md_max_ft=float(w['md_max_ft']),
                    t_start=datetime.datetime.fromisoformat(w['time_start']),
                    t_end=datetime.datetime.fromisoformat(w['time_end']))
    gw = rd.load_window_gauges(win)
    m = cfg['mesh']
    mesh = rd.build_mesh(win, float(m['domain_pad_low_md_ft']),
                         float(m['domain_pad_high_md_ft']), float(m['dx_ft']))
    frac_hits = rd.load_frac_hits(int(cfg['window']['stage']))
    src, fh_centroid = rd.pick_source_gauge(gw, frac_hits,
                                            cfg['source']['selection_rule'])

    e = cfg['extended_window']
    ext_win = rd.Window(md_min_ft=float(w['md_min_ft']), md_max_ft=float(w['md_max_ft']),
                        t_start=datetime.datetime.fromisoformat(e['time_start']),
                        t_end=datetime.datetime.fromisoformat(e['time_end']))
    ew = rd.load_window_gauges(ext_win)
    ext_t0 = ext_win.t_start
    ext = {}
    for n, s in ew.series.items():
        off = (s.t0_abs - ext_t0).total_seconds()
        ext[int(n)] = {'t': s.taxis_s + off, 'raw': s.raw_psi,
                       'd': s.raw_psi - float(s.raw_psi[0]),
                       't0_abs': s.t0_abs}

    tgt_gauges = [n for n in sorted(gw.series) if n != src]
    targets = []
    for n in tgt_gauges:
        s = gw.series[n]
        t, d = s.taxis_s, s.delta_psi
        amp = float(d.max())
        i = int(np.argmin(d))
        tA = first_crossing(t, d, 0.0, i)
        tB = first_crossing(t, d, 0.1 * amp, i)
        targets.append({
            'g': int(n), 'md_ft': float(s.md_ft),
            'distance_ft': float(abs(s.md_ft - gw.series[src].md_ft)),
            't': t, 'd': d, 'amp': amp,
            'full': np.ones(len(t), bool), 'A': t >= tA, 'B': t >= tB,
            't_start_A': tA, 't_start_B': tB,
            'obs_min_psi': float(d[i]), 'obs_t_min_s': float(t[i]),
            'w0': (s.t0_abs - ext_t0).total_seconds()})

    _G.update(dict(cfg=cfg, win=win, gw=gw, mesh=mesh, src=int(src),
                   fh_centroid=float(fh_centroid), ext=ext, ext_t0=ext_t0,
                   targets=targets, tgt_gauges=tgt_gauges,
                   sidx=mesh.index_of(gw.series[src].md_ft),
                   ridx=[mesh.index_of(t['md_ft']) for t in targets],
                   t_total=float(gw.series[src].taxis_s[-1]),
                   ext_t_total=float(max(ext[n]['t'][-1] for n in ext)),
                   dt=float(cfg['solver']['dt_s'])))
    return _G


def _init(cfg):
    build(cfg)


# ---------------------------------------------------------------------------
# forward evaluation
# ---------------------------------------------------------------------------

def _profile(D):
    return np.full(_G['mesh'].nx, float(D))


def _sim_window(D):
    s = _G['gw'].series[_G['src']]
    ta, rec = rc.solve_forward(_G['mesh'].x, _profile(D), _G['dt'], _G['t_total'],
                               s.taxis_s, s.delta_psi, _G['sidx'],
                               record_idx=_G['ridx'])
    return [np.interp(t['t'], ta, rec[:, k]) for k, t in enumerate(_G['targets'])]


def _sim_ext(D):
    """Quiescent (10:20) start, then re-referenced to the comparison window start."""
    ext, src = _G['ext'], _G['src']
    ta, rec = rc.solve_forward(_G['mesh'].x, _profile(D), _G['dt'], _G['ext_t_total'],
                               ext[src]['t'], ext[src]['d'], _G['sidx'],
                               record_idx=_G['ridx'])
    out = []
    for k, t in enumerate(_G['targets']):
        s = np.interp(t['w0'] + t['t'], ta, rec[:, k])
        out.append(s - np.interp(t['w0'], ta, rec[:, k]))
    return out


def _metrics(sim_list):
    out = {}
    for lab in ('full', 'A', 'B'):
        mse, nrm = [], []
        for k, t in enumerate(_G['targets']):
            msk = t[lab]
            r = sim_list[k][msk] - t['d'][msk]
            v = float(np.mean(r ** 2))
            mse.append(v)
            nrm.append(v / t['amp'] ** 2)
        out[lab] = {'per_gauge_mse': mse,
                    'rmse_gauge_mean_psi': float(np.sqrt(np.mean(mse))),
                    'rmse_normalised': float(np.sqrt(np.mean(nrm)))}
    out['sim_min_psi'] = [float(s.min()) for s in sim_list]
    out['sim_t_min_s'] = [float(_G['targets'][k]['t'][int(np.argmin(s))])
                          for k, s in enumerate(sim_list)]
    return out


def run_window(D):
    return float(D), _metrics(_sim_window(D))


def run_ext(D):
    return float(D), _metrics(_sim_ext(D))


# ---------------------------------------------------------------------------

def collect(results, grid):
    order = np.argsort([r[0] for r in results])
    g = np.array([results[i][0] for i in order])
    out = {'grid': g}
    for lab in ('full', 'A', 'B'):
        out[lab] = {
            'rmse_gauge_mean': np.array([results[i][1][lab]['rmse_gauge_mean_psi']
                                         for i in order]),
            'rmse_normalised': np.array([results[i][1][lab]['rmse_normalised']
                                         for i in order]),
            'per_gauge_mse': np.array([results[i][1][lab]['per_gauge_mse']
                                       for i in order]),
        }
    return out


def optimum(g, curve):
    i = int(np.argmin(curve))
    return {'D_ft2_s': float(g[i]), 'value': float(curve[i]),
            'at_grid_edge': bool(i == 0 or i == len(g) - 1),
            'grid_index': i}


def local_minima(g, curve):
    """Every interior local minimum of a 1-D misfit curve, as (D, value)."""
    out = []
    for i in range(1, len(curve) - 1):
        if curve[i] <= curve[i - 1] and curve[i] < curve[i + 1]:
            if out and abs(np.log10(g[i] / out[-1][0])) < 1e-9:
                continue
            out.append((float(g[i]), float(curve[i])))
    # collapse ties on a flat floor: keep one representative per basin
    dedup = []
    for D, v in out:
        if dedup and D / dedup[-1][0] < 1.5 and abs(v - dedup[-1][1]) < 1e-9:
            continue
        dedup.append((D, v))
    return dedup


def summarise(A, d_high_min, label_set=('full', 'A', 'B')):
    """Per-gauge and uniform optima.

    Reported twice per gauge, because the quiescent-start curves are bimodal:
    `global` is the unrestricted argmin, `high_basin` the argmin over
    D >= d_high_min, which is the branch continuous with the window-start fits and
    the only one comparable with them.
    """
    g = A['grid']
    hi = g >= float(d_high_min)
    out = {}
    for lab in label_set:
        rows = []
        pg = A[lab]['per_gauge_mse']
        for k, t in enumerate(_G['targets']):
            c = np.sqrt(pg[:, k])
            i = int(np.argmin(c))
            j = int(np.where(hi)[0][int(np.argmin(c[hi]))])
            lm = local_minima(g, c)
            best_two = sorted(lm, key=lambda z: z[1])[:2]
            gap = (float(100.0 * (best_two[1][1] / best_two[0][1] - 1.0))
                   if len(best_two) > 1 else float('inf'))
            rows.append({'gauge': t['g'], 'distance_ft': t['distance_ft'],
                         'D_ft2_s': float(g[i]), 'rmse_psi': float(c[i]),
                         'at_grid_edge': bool(i == 0 or i == len(g) - 1),
                         'D_high_basin_ft2_s': float(g[j]),
                         'rmse_high_basin_psi': float(c[j]),
                         'high_basin_is_global': bool(i == j),
                         'n_local_minima': len(lm),
                         'local_minima': lm,
                         'second_basin_penalty_pct': gap})
        for key, ratio_name in (('D_ft2_s', 'global'),
                                ('D_high_basin_ft2_s', 'high_basin')):
            Ds = [r[key] for r in rows]
            out.setdefault(lab, {})[f'spread_{ratio_name}'] = {
                'D_min': float(min(Ds)), 'D_max': float(max(Ds)),
                'ratio': float(max(Ds) / min(Ds)),
                'log10_std': float(np.std(np.log10(Ds), ddof=1))}
        out[lab].update({
            'uniform_absolute_norm': optimum(g, A[lab]['rmse_gauge_mean']),
            'uniform_normalised_norm': optimum(g, A[lab]['rmse_normalised']),
            'per_gauge': rows,
            'any_gauge_at_grid_edge': bool(any(r['at_grid_edge'] for r in rows)),
            'any_gauge_bimodal_within_20pct': bool(
                any(r['second_basin_penalty_pct'] < 20.0 for r in rows)),
        })
        out[lab]['spread'] = out[lab]['spread_high_basin']
    return out


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    t_wall = time.time()
    with open(args.config) as fh:
        cfg = json.load(fh)
    outdir = cfg['outputs']['dir']
    os.makedirs(outdir, exist_ok=True)
    dpi = int(cfg['outputs']['figure_dpi'])
    n_solves = 0

    build(cfg)
    G = _G
    log(f"source g{G['src']} MD {G['gw'].series[G['src']].md_ft:.0f}; targets "
        f"{G['tgt_gauges']}; mesh nx={G['mesh'].nx} "
        f"[{G['mesh'].x[0]:.0f}, {G['mesh'].x[-1]:.0f}] ft")
    log(f"window t_total {G['t_total']:.1f} s; extended t_total "
        f"{G['ext_t_total']:.1f} s")
    log("mask starts A: " + ", ".join(f"g{t['g']}={t['t_start_A']:.1f}"
                                      for t in G['targets']))
    log("mask starts B: " + ", ".join(f"g{t['g']}={t['t_start_B']:.1f}"
                                      for t in G['targets']))

    # ---- quiescence, measured (defect 12: the config rationale overstated it) --
    q_end = (datetime.datetime.fromisoformat(
        cfg['extended_window']['quiescence_check_end']) - G['ext_t0']).total_seconds()
    quiescence = {f"g{n}": float(np.ptp(G['ext'][n]['raw'][G['ext'][n]['t'] <= q_end]))
                  for n in sorted(G['ext'])}
    log("extended-record quiescence ptp psi over 10:20-10:28: " +
        ", ".join(f"{k}={v:.3f}" for k, v in quiescence.items()))

    # ---- 1/2/3. the two sweeps ---------------------------------------------
    gs = cfg['amend']['grid']
    grid = np.logspace(np.log10(gs['min']), np.log10(gs['max']), int(gs['n_points']))
    nproc = int(cfg['search']['processes'])
    log(f"sweeps: {len(grid)} D points ({(grid[1]/grid[0]-1)*100:.3f}% spacing) "
        f"x 2 start conditions, {nproc} workers")
    t0 = time.time()
    with Pool(nproc, initializer=_init, initargs=(cfg,)) as pool:
        res_w = pool.map(run_window, grid, chunksize=4)
        res_e = pool.map(run_ext, grid, chunksize=2)
    n_solves += 2 * len(grid)
    log(f"sweeps done in {time.time()-t0:.0f} s")

    d_high = float(cfg['amend']['high_basin_D_min_ft2_s'])
    AW = collect(res_w, grid)
    AE = collect(res_e, grid)
    SW = summarise(AW, d_high)
    SE = summarise(AE, d_high)
    for nmz, S in (('window-start', SW), ('quiescent-start', SE)):
        for lab in ('full', 'A', 'B'):
            u = S[lab]['uniform_absolute_norm']
            v = S[lab]['uniform_normalised_norm']
            sh = S[lab]['spread_high_basin']
            sg = S[lab]['spread_global']
            log(f"[{nmz}/{lab}] abs D={u['D_ft2_s']:.0f} ({u['value']:.2f} psi); "
                f"norm D={v['D_ft2_s']:.0f} ({v['value']:.4f}); per-gauge span "
                f"high-basin {sh['D_min']:.0f}-{sh['D_max']:.0f} = {sh['ratio']:.1f}x, "
                f"global {sg['D_min']:.0f}-{sg['D_max']:.0f} = {sg['ratio']:.1f}x"
                + (' [SOME GAUGE IS BIMODAL WITHIN 20%]'
                   if S[lab]['any_gauge_bimodal_within_20pct'] else ''))
            for r in S[lab]['per_gauge']:
                if not r['high_basin_is_global']:
                    log(f"    g{r['gauge']}: GLOBAL minimum is the low-D basin "
                        f"D={r['D_ft2_s']:.0f} ({r['rmse_psi']:.2f} psi); the "
                        f"high-D basin is D={r['D_high_basin_ft2_s']:.0f} "
                        f"({r['rmse_high_basin_psi']:.2f} psi) -- not identified")

    # ---- 4. explained fraction of the observed minimum, at four D ------------
    report_D = sorted(set(
        [float(x) for x in cfg['amend']['extended_window_report_D']] +
        [SE['full']['uniform_absolute_norm']['D_ft2_s'],
         SE['full']['uniform_normalised_norm']['D_ft2_s']]))
    with Pool(min(nproc, len(report_D)), initializer=_init, initargs=(cfg,)) as pool:
        rep = pool.map(run_ext, report_D)
    n_solves += len(report_D)
    rep = {d: r for d, r in rep}

    which = {}
    for d in report_D:
        tags = []
        if abs(d - SE['full']['uniform_absolute_norm']['D_ft2_s']) < 1e-9:
            tags.append('quiescent-start absolute-norm optimum')
        if abs(d - SE['full']['uniform_normalised_norm']['D_ft2_s']) < 1e-9:
            tags.append('quiescent-start normalised-norm optimum')
        if abs(d - 550.0) < 1e-9:
            tags.append('house-rules amplitude-normalised baseline (band 457-647, '
                        'contains the manuscript D=480)')
        if abs(d - 1150.0) < 1e-9:
            tags.append('house-rules absolute-norm baseline')
        which[d] = '; '.join(tags) if tags else 'reported for continuity'

    ext_rows = []
    for d in report_D:
        r = rep[d]
        for k, t in enumerate(G['targets']):
            ext_rows.append({
                'run': 'extended_quiescent', 'D_ft2_s': d, 'D_role': which[d],
                'gauge': t['g'], 'distance_ft': t['distance_ft'],
                'obs_min_psi': t['obs_min_psi'], 'obs_t_min_s': t['obs_t_min_s'],
                'sim_min_psi': r['sim_min_psi'][k],
                'sim_t_min_s': r['sim_t_min_s'][k],
                'explained_fraction_of_min': float(r['sim_min_psi'][k]
                                                   / t['obs_min_psi']),
                'sim_t_min_minus_obs_t_min_s': float(r['sim_t_min_s'][k]
                                                     - t['obs_t_min_s']),
                'rmse_gauge_mean_psi_full_window': r['full']['rmse_gauge_mean_psi'],
                'rmse_normalised_full_window': r['full']['rmse_normalised']})
        log(f"  quiescent D={d:7.1f} ({r['full']['rmse_gauge_mean_psi']:6.2f} psi / "
            f"{r['full']['rmse_normalised']:.4f}): explained " + ", ".join(
                f"g{t['g']}={100*r['sim_min_psi'][k]/t['obs_min_psi']:.0f}%"
                for k, t in enumerate(G['targets'])))

    # the window-start (zero-IC) run at the published optimum, for the same table
    sim_w_full = _sim_window(PUBLISHED_D_FULL)
    n_solves += 1
    for k, t in enumerate(G['targets']):
        ext_rows.append({
            'run': 'window_start_zero_IC', 'D_ft2_s': PUBLISHED_D_FULL,
            'D_role': 'published C3 full-window absolute-norm optimum',
            'gauge': t['g'], 'distance_ft': t['distance_ft'],
            'obs_min_psi': t['obs_min_psi'], 'obs_t_min_s': t['obs_t_min_s'],
            'sim_min_psi': float(sim_w_full[k].min()),
            'sim_t_min_s': float(t['t'][int(np.argmin(sim_w_full[k]))]),
            'explained_fraction_of_min': float(sim_w_full[k].min() / t['obs_min_psi']),
            'sim_t_min_minus_obs_t_min_s': float(
                t['t'][int(np.argmin(sim_w_full[k]))] - t['obs_t_min_s']),
            'rmse_gauge_mean_psi_full_window': '',
            'rmse_normalised_full_window': ''})

    # ---- 5. misfit partition under three conventions ------------------------
    part_rows = []
    partition = {}
    for lab in ('A', 'B'):
        ss_ex, ss_in, n_ex, n_in = [], [], [], []
        for k, t in enumerate(G['targets']):
            msk = t[lab]
            r = sim_w_full[k] - t['d']
            ss_ex.append(float(np.sum(r[~msk] ** 2)))
            ss_in.append(float(np.sum(r[msk] ** 2)))
            n_ex.append(int((~msk).sum()))
            n_in.append(int(msk.sum()))
        ss_ex = np.array(ss_ex); ss_in = np.array(ss_in)
        n_ex = np.array(n_ex); n_in = np.array(n_in)
        ntot = n_ex + n_in
        share = 100.0 * ss_ex / (ss_ex + ss_in)
        pooled = float(100.0 * ss_ex.sum() / (ss_ex.sum() + ss_in.sum()))
        mean_share = float(share.mean())
        gm_share = float(100.0 * np.mean(ss_ex / ntot)
                         / np.mean((ss_ex + ss_in) / ntot))
        partition[lab] = {'pooled_sum_of_squares_pct': pooled,
                          'mean_of_per_gauge_shares_pct': mean_share,
                          'share_of_gauge_mean_mse_pct': gm_share,
                          'per_gauge_pct': share.tolist(),
                          'D_ft2_s': PUBLISHED_D_FULL}
        for k, t in enumerate(G['targets']):
            part_rows.append({
                'rule': lab, 'gauge': t['g'], 'distance_ft': t['distance_ft'],
                'D_ft2_s': PUBLISHED_D_FULL,
                't_start_s': t[f't_start_{lab}'],
                'n_excluded': int(n_ex[k]), 'n_retained': int(n_in[k]),
                'ss_excluded': float(ss_ex[k]), 'ss_retained': float(ss_in[k]),
                'share_pct_this_gauge': float(share[k]),
                'convention': 'per_gauge'})
        part_rows.append({'rule': lab, 'gauge': 'ALL', 'distance_ft': '',
                          'D_ft2_s': PUBLISHED_D_FULL, 't_start_s': '',
                          'n_excluded': int(n_ex.sum()), 'n_retained': int(n_in.sum()),
                          'ss_excluded': float(ss_ex.sum()),
                          'ss_retained': float(ss_in.sum()),
                          'share_pct_this_gauge': pooled,
                          'convention': 'pooled_sum_of_squares_over_all_residuals'})
        part_rows.append({'rule': lab, 'gauge': 'ALL', 'distance_ft': '',
                          'D_ft2_s': PUBLISHED_D_FULL, 't_start_s': '',
                          'n_excluded': '', 'n_retained': '', 'ss_excluded': '',
                          'ss_retained': '', 'share_pct_this_gauge': mean_share,
                          'convention': 'mean_over_gauges_of_per_gauge_shares'})
        part_rows.append({'rule': lab, 'gauge': 'ALL', 'distance_ft': '',
                          'D_ft2_s': PUBLISHED_D_FULL, 't_start_s': '',
                          'n_excluded': '', 'n_retained': '', 'ss_excluded': '',
                          'ss_retained': '', 'share_pct_this_gauge': gm_share,
                          'convention': 'excluded_share_of_the_gauge_mean_MSE'})
        log(f"partition rule {lab}: pooled {pooled:.2f}%, mean-over-gauges "
            f"{mean_share:.2f}%, share of gauge-mean MSE {gm_share:.2f}%, "
            f"per gauge " + "/".join(f"{s:.1f}" for s in share))

    # ---- 6. the R2 uniform row at its own optimum ---------------------------
    D_r2 = float(10.0 ** cfg['amend']['r2_uniform_reproduction_log10_D'])
    sim_r2 = _sim_window(D_r2)
    n_solves += 1
    mse, nrm = [], []
    for k, t in enumerate(G['targets']):
        r = sim_r2[k] - t['d']
        v = float(np.mean(r ** 2)); mse.append(v); nrm.append(v / t['amp'] ** 2)
    r2_repro = {'D_ft2_s': D_r2,
                'rmse_gauge_mean_psi': float(np.sqrt(np.mean(mse))),
                'rmse_normalised': float(np.sqrt(np.mean(nrm))),
                'r2_manifest_rmse_psi': 82.33033019786637,
                'r2_manifest_rmse_normalised': 0.5391837549765316}
    # ... and at the published C3 grid point, for the comparison the README makes
    mse, nrm = [], []
    for k, t in enumerate(G['targets']):
        r = sim_w_full[k] - t['d']
        v = float(np.mean(r ** 2)); mse.append(v); nrm.append(v / t['amp'] ** 2)
    r2_repro['at_published_C3_grid_point'] = {
        'D_ft2_s': PUBLISHED_D_FULL,
        'rmse_gauge_mean_psi': float(np.sqrt(np.mean(mse))),
        'rmse_normalised': float(np.sqrt(np.mean(nrm))),
        'D_offset_pct': float(100.0 * (PUBLISHED_D_FULL / D_r2 - 1.0))}
    log(f"R2 uniform row reproduced at D={D_r2:.4f}: "
        f"{r2_repro['rmse_gauge_mean_psi']:.6f} psi / "
        f"{r2_repro['rmse_normalised']:.6f} (R2 published "
        f"{r2_repro['r2_manifest_rmse_psi']:.5f} / "
        f"{r2_repro['r2_manifest_rmse_normalised']:.5f})")

    # ---- 7. pumping events, onsets, and the margin against the restart -------
    pump = load_pumping(cfg)
    events = pumping_events(pump, cfg)
    on_thr = float(cfg['pumping_events']['on_threshold_bpm'])
    win_t0_abs = G['gw'].series[1].t0_abs
    prior = [r for r in events[str(on_thr)]
             if datetime.datetime.fromisoformat(r['start_utc']) < win_t0_abs]
    pump_start = datetime.datetime.fromisoformat(prior[-1]['start_utc'])
    prev_major = max(prior[:-1], key=lambda r: r['max_rate_bpm'])
    prev_stop = datetime.datetime.fromisoformat(prev_major['stop_utc'])
    log(f"pumping start {pump_start.isoformat()}; preceding cycle shut-in "
        f"{prev_stop.isoformat()} ({prev_major['max_rate_bpm']:.1f} bpm)")

    onset_rows = []
    for n in sorted(G['ext']):
        te, de = G['ext'][n]['t'], G['ext'][n]['raw']
        s = G['gw'].series[n]
        i = int(np.argmin(s.delta_psi))
        t_min_abs = s.t0_abs + datetime.timedelta(seconds=float(s.taxis_s[i]))
        tmin = (t_min_abs - G['ext_t0']).total_seconds()
        k = int(np.searchsorted(te, tmin, side='right')) - 1
        sm = moving_average(te, de, 60.0)
        j = int(np.argmax(sm[:k + 1]))
        near = np.where(sm[:k + 1] >= sm[j] - 1.0)[0]
        t_on = G['ext_t0'] + datetime.timedelta(seconds=float(te[j]))
        lo = G['ext_t0'] + datetime.timedelta(seconds=float(te[near[0]]))
        hi = G['ext_t0'] + datetime.timedelta(seconds=float(te[near[-1]]))
        ip = int(np.searchsorted(te, (pump_start - G['ext_t0']).total_seconds(),
                                 side='left'))
        lead = (pump_start - t_on).total_seconds()
        hi_minus_pump = (hi - pump_start).total_seconds()
        onset_rows.append({
            'gauge': n, 'distance_ft': float(abs(s.md_ft
                                                 - G['gw'].series[G['src']].md_ft)),
            'onset_utc_60s': t_on.isoformat(),
            'plateau_lo_utc': lo.isoformat(), 'plateau_hi_utc': hi.isoformat(),
            'plateau_width_s': float(te[near[-1]] - te[near[0]]),
            'lead_over_pump_restart_s': lead,
            'plateau_hi_minus_pump_restart_s': hi_minus_pump,
            'p_smoothed_max_psi': float(sm[j]),
            'p_smoothed_at_restart_psi': float(sm[ip]),
            'drop_from_max_at_restart_psi': float(sm[j] - sm[ip]),
            'ordering_resolved': bool(hi_minus_pump < 0.0)})
        log(f"g{n}: onset {t_on.strftime('%H:%M:%S')} leads the restart by "
            f"{lead:.0f} s; plateau ends {hi.strftime('%H:%M:%S')} "
            f"({hi_minus_pump:+.0f} s vs restart) -> "
            f"{'RESOLVED' if hi_minus_pump < 0 else 'NOT RESOLVED'}")

    n_resolved = sum(1 for r in onset_rows if r['ordering_resolved'])
    log(f"ordering resolved at {n_resolved} of {len(onset_rows)} gauges")

    # ---- 8. grid sensitivity table -----------------------------------------
    published = json.load(open(cfg['amend']['published_run_manifests'][0]))
    pub_u = published['results']['uniform_optima']
    pub_pg = published['results']['per_gauge_single_fits']
    pub_sp = published['results']['per_gauge_spread']
    lab_map = {'full': 'full', 'A': 'A_zero_crossing', 'B': 'B_ten_percent'}
    grid_rows = []
    for lab in ('full', 'A', 'B'):
        pl = lab_map[lab]
        grid_rows.append({
            'quantity': f'uniform D, absolute norm [{pl}]',
            'published_grid': pub_u[pl]['absolute_norm_gauge_mean']['D_ft2_s'],
            'amend_grid': SW[lab]['uniform_absolute_norm']['D_ft2_s'],
            'difference_pct': 100.0 * (SW[lab]['uniform_absolute_norm']['D_ft2_s']
                                       / pub_u[pl]['absolute_norm_gauge_mean']['D_ft2_s'] - 1.0)})
        grid_rows.append({
            'quantity': f'uniform D, normalised norm [{pl}]',
            'published_grid': pub_u[pl]['normalised_norm']['D_ft2_s'],
            'amend_grid': SW[lab]['uniform_normalised_norm']['D_ft2_s'],
            'difference_pct': 100.0 * (SW[lab]['uniform_normalised_norm']['D_ft2_s']
                                       / pub_u[pl]['normalised_norm']['D_ft2_s'] - 1.0)})
        grid_rows.append({
            'quantity': f'per-gauge D span (x) [{pl}]',
            'published_grid': pub_sp[pl]['ratio'],
            'amend_grid': SW[lab]['spread']['ratio'],
            'difference_pct': 100.0 * (SW[lab]['spread']['ratio']
                                       / pub_sp[pl]['ratio'] - 1.0)})
        grid_rows.append({
            'quantity': f'per-gauge sd(log10 D) [{pl}]',
            'published_grid': pub_sp[pl]['log10_std'],
            'amend_grid': SW[lab]['spread']['log10_std'],
            'difference_pct': 100.0 * (SW[lab]['spread']['log10_std']
                                       / pub_sp[pl]['log10_std'] - 1.0)})
        for k, t in enumerate(G['targets']):
            pv = [r for r in pub_pg[pl] if int(r['gauge']) == t['g']][0]['D_ft2_s']
            av = SW[lab]['per_gauge'][k]['D_high_basin_ft2_s']
            grid_rows.append({'quantity': f"per-gauge D g{t['g']} [{pl}]",
                              'published_grid': pv, 'amend_grid': av,
                              'difference_pct': 100.0 * (av / pv - 1.0)})
    log("grid sensitivity: span published %.2fx/%.2fx/%.2fx -> amend %.2fx/%.2fx/%.2fx"
        % (pub_sp['full']['ratio'], pub_sp['A_zero_crossing']['ratio'],
           pub_sp['B_ten_percent']['ratio'], SW['full']['spread']['ratio'],
           SW['A']['spread']['ratio'], SW['B']['spread']['ratio']))

    # ---- 9. figures ---------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    written = []

    cmap = plt.get_cmap('viridis')
    gall = sorted(G['ext'])
    gcol = {n: cmap(0.08 + 0.82 * i / max(len(gall) - 1, 1))
            for i, n in enumerate(gall)}

    def mins(d):
        return (d - G['ext_t0']).total_seconds() / 60.0

    # --- figure v3 of the timing panel: the annotation was wrong -------------
    fig, ax = plt.subplots(3, 1, figsize=(11.5, 12.0),
                           gridspec_kw={'height_ratios': [0.85, 1.3, 1.3]})
    ax[0].sharex(ax[1])
    p = pump[cfg['data']['pumping_start_channel']]
    tp = np.array([(p['start'] + datetime.timedelta(seconds=float(x))
                    - G['ext_t0']).total_seconds() for x in p['taxis']])
    s0 = (tp >= 0) & (tp <= G['ext_t_total'])
    ax[0].plot(tp[s0] / 60.0, p['data'][s0], color='#1f77b4', lw=1.5)
    ax[0].set_ylabel('Slurry rate (bpm)', color='#1f77b4')
    a0b = ax[0].twinx()
    q = pump['Treating Pressure']
    tq = np.array([(q['start'] + datetime.timedelta(seconds=float(x))
                    - G['ext_t0']).total_seconds() for x in q['taxis']])
    s1 = (tq >= 0) & (tq <= G['ext_t_total'])
    a0b.plot(tq[s1] / 60.0, q['data'][s1], color='#999999', lw=1.0)
    a0b.set_ylabel('Treating pressure (psi)', color='#777777')
    ax[0].set_title('C3 (v3)  The "precursor" is the falloff of the PRECEDING '
                    'injection cycle, not a response to this one', fontsize=11.5)

    for a in ax:
        a.axvspan(mins(win_t0_abs), mins(win_t0_abs) + G['t_total'] / 60.0,
                  color='#ffe9b0', alpha=0.6, zorder=0)
        a.axvline(mins(pump_start), color='#d62728', lw=1.7, ls='--', zorder=1)
        a.axvline(mins(prev_stop), color='#2ca02c', lw=1.5, ls=':', zorder=1)
        a.grid(alpha=0.25, lw=0.5)

    for n in gall:
        ax[1].plot(G['ext'][n]['t'] / 60.0, G['ext'][n]['raw'],
                   color=gcol[n], lw=1.3, label=f'g{n}')
    for r in onset_rows:
        n = r['gauge']
        t_on = datetime.datetime.fromisoformat(r['onset_utc_60s'])
        lo = datetime.datetime.fromisoformat(r['plateau_lo_utc'])
        hi = datetime.datetime.fromisoformat(r['plateau_hi_utc'])
        y = float(np.interp(mins(t_on) * 60.0, G['ext'][n]['t'], G['ext'][n]['raw']))
        ax[1].plot([mins(lo), mins(hi)], [y, y], color=gcol[n], lw=2.6,
                   alpha=0.55, solid_capstyle='butt', zorder=4)
        ax[1].plot(mins(t_on), y, marker='v', ms=9, mfc='white',
                   mec=gcol[n], mew=1.9, zorder=5)
    ax[1].set_ylabel('Gauge pressure (psi)')
    ax[1].set_xlim(0.0, G['ext_t_total'] / 60.0)
    ax[1].set_xlabel('minutes after 2020-03-16 10:20:00 UTC')
    ax[1].legend(ncol=7, fontsize=8.5, loc='lower right', framealpha=0.92)
    g7 = [r for r in onset_rows if r['gauge'] == 7][0]
    ax[1].text(0.015, 0.97,
               'v = onset: maximum of the 60 s-smoothed record before the precursor '
               'minimum;\nthe bar is the +-1 psi plateau of that maximum.  The onsets '
               'migrate outward\nfrom the 10:50:34 shut-in of the preceding cycle.  '
               'All seven PRECEDE the\n11:18:57 restart, but only six are RESOLVED: '
               f"g7 leads it by {abs(g7['lead_over_pump_restart_s']):.0f} s, less than\n"
               f"its own {g7['plateau_width_s']:.0f} s plateau, which ends "
               f"{g7['plateau_hi_minus_pump_restart_s']:.0f} s AFTER the restart.",
               transform=ax[1].transAxes, fontsize=8.6, va='top',
               bbox=dict(fc='white', ec='0.7', alpha=0.88, boxstyle='round,pad=0.35'))

    for n in gall:
        s = G['gw'].series[n]
        w0 = mins(s.t0_abs)
        t, d = s.taxis_s, s.delta_psi
        keep = t <= 900.0
        ax[2].plot(w0 + t[keep] / 60.0, d[keep], color=gcol[n], lw=1.6, label=f'g{n}')
        i = int(np.argmin(d))
        ax[2].plot(w0 + t[i] / 60.0, d[i], marker='o', ms=7.5, mfc='white',
                   mec=gcol[n], mew=1.9, zorder=5)
    ax[2].axhline(0.0, color='k', lw=0.9)
    ax[2].set_ylabel('Window-referenced $\\Delta P$ (psi)')
    ax[2].set_xlabel('minutes after 2020-03-16 10:20:00 UTC   '
                     '(shaded = R1 comparison window, opens 11:24:04.8)')
    ax[2].set_xlim(mins(pump_start) - 1.5, mins(win_t0_abs) + 900.0 / 60.0)
    ax[2].set_ylim(-36, 46)
    ax[2].legend(ncol=7, fontsize=8.5, loc='upper left', framealpha=0.92)
    ax[2].annotate('pumping start\n11:18:57', xy=(mins(pump_start), 30),
                   xytext=(mins(pump_start) + 0.5, 34), fontsize=8.8, color='#d62728')
    ax[2].text(0.985, 0.06,
               'o = precursor minimum.  The minima lag pumping start by 399-794 s '
               'and migrate with distance,\nso they are not locked to the start of '
               'pumping.', transform=ax[2].transAxes, fontsize=8.8, va='bottom',
               ha='right',
               bbox=dict(fc='white', ec='0.7', alpha=0.88, boxstyle='round,pad=0.35'))
    fig.tight_layout()
    f_v3 = os.path.join(outdir, 'fig_c3_precursor_vs_pumping_v3.png')
    fig.savefig(f_v3, dpi=dpi)
    plt.close(fig)
    written.append((f_v3, 'figure_png',
                    'v3 of the timing figure: v2 annotated "six of seven PRECEDE the '
                    'restart", which contradicts every data file (all seven onsets '
                    'lead it). v3 states all seven and separates lead from '
                    'resolution using each onset\'s +-1 psi plateau.'))

    # --- figure v2 of the extended-window test: name the D, show both norms ---
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.6), sharex=True)
    D_abs = SE['full']['uniform_absolute_norm']['D_ft2_s']
    D_nrm = SE['full']['uniform_normalised_norm']['D_ft2_s']
    curves = [(550.0, '#7b3294', ':', 2.0,
               f'quiescent, D={550:.0f} (house-rules normalised opt.)'),
              (D_nrm, '#e08214', '-', 1.6,
               f'quiescent, D={D_nrm:.0f} (this run\'s normalised opt.)'),
              (1150.0, '#2ca02c', '--', 1.4,
               'quiescent, D=1150 (house-rules absolute opt.)'),
              (D_abs, '#1f77b4', '-', 1.8,
               f'quiescent, D={D_abs:.0f} (this run\'s absolute opt.)')]
    # the sweeps kept only summaries, so re-solve for the plotted traces
    plot_sims = {}
    for d, *_ in curves:
        plot_sims[d] = _sim_ext(d)
        n_solves += 1
    for k, t in enumerate(G['targets']):
        a = axes.flat[k]
        keep = t['t'] <= 900.0
        a.plot(t['t'][keep], t['d'][keep], color='k', lw=2.0, label='observed',
               zorder=6)
        a.plot(t['t'][keep], sim_w_full[k][keep], color='#d62728', lw=1.3, ls='--',
               label=f'window start, zero IC, D={PUBLISHED_D_FULL:.0f}')
        for d, col, ls, lw, nmm in curves:
            a.plot(t['t'][keep], plot_sims[d][k][keep], color=col, ls=ls, lw=lw,
                   label=nmm)
        a.axhline(0, color='k', lw=0.6)
        a.set_title(f"g{t['g']}  {t['distance_ft']:.0f} ft from source   "
                    f"(observed min {t['obs_min_psi']:.1f} psi)", fontsize=9.5)
        a.grid(alpha=0.25, lw=0.5)
        a.set_ylim(-32, 30)
        if k >= 3:
            a.set_xlabel('s after window start (11:24:04.8)')
        if k % 3 == 0:
            a.set_ylabel('$\\Delta P$ (psi)')
    axes.flat[0].legend(fontsize=7.2, loc='lower right', framealpha=0.93)
    fig.suptitle('C3 (v2)  Pure diffusion from a quiescent start reproduces the '
                 'precursor -- but HOW MUCH depends on which norm picks $D$\n'
                 'at the amplitude-normalised optimum the far field is barely '
                 'reproduced (g6 34-55 %, g7 0-92 %); at the absolute-norm optimum '
                 'g7 is over-produced 2.2-2.7x', fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    f_ext = os.path.join(outdir, 'fig_c3_extended_window_v2.png')
    fig.savefig(f_ext, dpi=dpi)
    plt.close(fig)
    written.append((f_ext, 'figure_png',
                    'v2 of the decisive-test figure: v1 plotted one quiescent curve '
                    'at D=1581 while the README table beside it quoted D=1150, with '
                    'no D in either caption. v2 names every D and shows both norms.'))

    # --- figure v2 of the refit panel: (c) gains the quiescent-start fits, ----
    # --- (d) stops putting two different models in one group -----------------
    fig, ax = plt.subplots(2, 2, figsize=(13.0, 9.4))
    style = {'full': ('#333333', '-', 'full window (before)'),
             'A': ('#1f77b4', '--', 'rule A: after zero crossing'),
             'B': ('#d62728', '-.', 'rule B: after 10% of window max')}
    for lab in ('full', 'A', 'B'):
        c, ls, nmm = style[lab]
        ax[0, 0].loglog(AW['grid'], AW[lab]['rmse_gauge_mean'], color=c, ls=ls,
                        lw=1.6, label=nmm)
        o = SW[lab]['uniform_absolute_norm']
        ax[0, 0].plot(o['D_ft2_s'], o['value'], 'o', color=c, ms=7)
        ax[0, 1].loglog(AW['grid'], AW[lab]['rmse_normalised'], color=c, ls=ls,
                        lw=1.6, label=nmm)
        o = SW[lab]['uniform_normalised_norm']
        ax[0, 1].plot(o['D_ft2_s'], o['value'], 'o', color=c, ms=7)
    ax[0, 0].loglog(AE['grid'], AE['full']['rmse_gauge_mean'], color='#1a9850',
                    ls='-', lw=1.6, label='full window, QUIESCENT start')
    o = SE['full']['uniform_absolute_norm']
    ax[0, 0].plot(o['D_ft2_s'], o['value'], 's', color='#1a9850', ms=7)
    ax[0, 1].loglog(AE['grid'], AE['full']['rmse_normalised'], color='#1a9850',
                    ls='-', lw=1.6, label='full window, QUIESCENT start')
    o = SE['full']['uniform_normalised_norm']
    ax[0, 1].plot(o['D_ft2_s'], o['value'], 's', color='#1a9850', ms=7)
    ax[0, 0].set_xlabel('uniform $D$ (ft$^2$/s)')
    ax[0, 0].set_ylabel('gauge-mean RMSE (psi)')
    ax[0, 0].set_title('(a) absolute norm (gauge-mean RMSE)', fontsize=10)
    ax[0, 1].set_xlabel('uniform $D$ (ft$^2$/s)')
    ax[0, 1].set_ylabel('amplitude-normalised RMSE')
    ax[0, 1].set_title('(b) amplitude-normalised norm', fontsize=10)
    for a in (ax[0, 0], ax[0, 1]):
        a.grid(alpha=0.25, which='both', lw=0.5)
        a.legend(fontsize=8.0)

    for lab in ('full', 'A', 'B'):
        c, ls, nmm = style[lab]
        rows = SW[lab]['per_gauge']
        ax[1, 0].semilogy([r['distance_ft'] for r in rows],
                          [r['D_high_basin_ft2_s'] for r in rows], marker='o',
                          color=c, ls=ls, lw=1.6, ms=6,
                          label='window start, ' + nmm)
    for lab, mk, ls2, col in (('full', 's', '-', '#1a9850'),
                              ('B', '^', '-.', '#8c564b')):
        rows = SE[lab]['per_gauge']
        ax[1, 0].semilogy([r['distance_ft'] for r in rows],
                          [r['D_high_basin_ft2_s'] for r in rows], marker=mk,
                          color=col, ls=ls2, lw=1.8, ms=7,
                          label='QUIESCENT start, ' + style[lab][2])
        for r in rows:
            if not r['high_basin_is_global']:
                ax[1, 0].plot([r['distance_ft']], [r['D_ft2_s']], marker='x',
                              color=col, ms=9, mew=2.0)
                ax[1, 0].plot([r['distance_ft'], r['distance_ft']],
                              [r['D_ft2_s'], r['D_high_basin_ft2_s']],
                              color=col, lw=0.9, ls=':')
    bim = [r for r in SE['full']['per_gauge'] if not r['high_basin_is_global']]
    if bim:
        r = bim[0]
        ax[1, 0].annotate(
            f"x = the SECOND basin.  Under the quiescent start g{r['gauge']}'s "
            f"misfit\nis bimodal: D={r['D_ft2_s']:.0f} ({r['rmse_psi']:.1f} psi) vs "
            f"D={r['D_high_basin_ft2_s']:.0f} "
            f"({r['rmse_high_basin_psi']:.1f} psi),\nonly "
            f"{r['second_basin_penalty_pct']:.0f} % apart -- so its path-averaged "
            "$D$ is NOT identified.",
            (0.02, 0.04), xycoords='axes fraction', fontsize=7.4, va='bottom',
            bbox=dict(fc='white', ec='0.75', alpha=0.9, boxstyle='round,pad=0.3'))
    ax[1, 0].set_xlabel('distance from source (ft)')
    ax[1, 0].set_ylabel('single-gauge path-averaged $D$ (ft$^2$/s)')
    ax[1, 0].set_title('(c) per-gauge single fits: correcting the initial condition '
                       'WIDENS the spread', fontsize=10)
    ax[1, 0].grid(alpha=0.25, which='both', lw=0.5)
    ax[1, 0].legend(fontsize=7.0, loc='upper right')

    labels = ['full', 'A', 'B']
    x = np.arange(len(labels))
    v_abs = [SW[l]['uniform_absolute_norm']['value'] for l in labels]
    D_absl = [SW[l]['uniform_absolute_norm']['D_ft2_s'] for l in labels]
    v_nrm = [SW[l]['uniform_normalised_norm']['value'] for l in labels]
    D_nrml = [SW[l]['uniform_normalised_norm']['D_ft2_s'] for l in labels]
    i_abs = [int(np.argmin(np.abs(AW['grid'] - d))) for d in D_absl]
    v_nrm_same = [float(AW[l]['rmse_normalised'][i])
                  for l, i in zip(labels, i_abs)]
    ax[1, 1].bar(x - 0.27, v_abs, 0.26, color='#4c72b0',
                 label='gauge-mean RMSE at its own optimum')
    axb = ax[1, 1].twinx()
    axb.bar(x + 0.0, v_nrm_same, 0.26, color='#dd8452', hatch='//',
            edgecolor='white',
            label='normalised RMSE of the SAME model (at the absolute optimum)')
    axb.bar(x + 0.27, v_nrm, 0.26, color='#dd8452',
            label='normalised RMSE at ITS OWN optimum (a different $D$)')
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels([style[l][2].replace(': ', ':\n') for l in labels],
                             fontsize=8)
    ax[1, 1].set_ylabel('gauge-mean RMSE (psi)', color='#4c72b0')
    axb.set_ylabel('amplitude-normalised RMSE', color='#dd8452')
    ax[1, 1].set_title('(d) misfit at each mask\'s optimum -- each bar labelled '
                       'with the $D$ it is evaluated at', fontsize=9.5)
    for xi in range(len(labels)):
        ax[1, 1].text(xi - 0.27, v_abs[xi], f'{v_abs[xi]:.1f}\n@{D_absl[xi]:.0f}',
                      ha='center', va='bottom', fontsize=7.4)
        axb.text(xi + 0.0, v_nrm_same[xi],
                 f'{v_nrm_same[xi]:.3f}\n@{D_absl[xi]:.0f}', ha='center',
                 va='bottom', fontsize=7.4)
        axb.text(xi + 0.27, v_nrm[xi], f'{v_nrm[xi]:.3f}\n@{D_nrml[xi]:.0f}',
                 ha='center', va='bottom', fontsize=7.4)
    axb.set_ylim(0, max(v_nrm_same) * 1.45)
    ax[1, 1].set_ylim(0, max(v_abs) * 1.35)
    h1, l1 = ax[1, 1].get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax[1, 1].legend(h1 + h2, l1 + l2, fontsize=6.9, loc='upper left')
    fig.suptitle('C3 (v2)  Recalibration with the precursor excluded, on an '
                 'independently offset 1.15 %-spaced grid', fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    f_refit = os.path.join(outdir, 'fig_c3_refit_before_after_v2.png')
    fig.savefig(f_refit, dpi=dpi)
    plt.close(fig)
    written.append((f_refit, 'figure_png',
                    'v2 of the refit figure: (c) adds the quiescent-start per-gauge '
                    'fits, (d) labels every bar with the D it is evaluated at and '
                    'adds the same-model normalised RMSE, because v1 put two models '
                    '2.05x apart in D side by side under one label.'))

    # ---- 10. csv + arrays ---------------------------------------------------
    def write_csv(path, rows, fields, role, note):
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow(r)
        written.append((path, role, note))

    write_csv(os.path.join(outdir, 'c3_amend_extended_window_norms_v2.csv'),
              ext_rows,
              ['run', 'D_ft2_s', 'D_role', 'gauge', 'distance_ft', 'obs_min_psi',
               'obs_t_min_s', 'sim_min_psi', 'sim_t_min_s',
               'explained_fraction_of_min', 'sim_t_min_minus_obs_t_min_s',
               'rmse_gauge_mean_psi_full_window', 'rmse_normalised_full_window'],
              'csv', 'explained fraction of each observed precursor minimum at BOTH '
                     'norms\' optima, superseding the single-D table in '
                     'c3_extended_window_test.csv')

    pg_rows = []
    for start, S in (('window_start_zero_IC', SW), ('quiescent_start_10:20', SE)):
        for lab in ('full', 'A', 'B'):
            for r in S[lab]['per_gauge']:
                pg_rows.append(dict(
                    r, start_condition=start, scoring=lab_map[lab],
                    local_minima=';'.join(f'{d:.0f}@{v:.2f}'
                                          for d, v in r['local_minima']),
                    spread_ratio_high_basin=S[lab]['spread_high_basin']['ratio'],
                    spread_ratio_global=S[lab]['spread_global']['ratio'],
                    spread_log10_std_high_basin=S[lab]['spread_high_basin']['log10_std']))
    write_csv(os.path.join(outdir, 'c3_amend_per_gauge_D_v2.csv'), pg_rows,
              ['start_condition', 'scoring', 'gauge', 'distance_ft',
               'D_high_basin_ft2_s', 'rmse_high_basin_psi', 'D_ft2_s', 'rmse_psi',
               'high_basin_is_global', 'n_local_minima', 'local_minima',
               'second_basin_penalty_pct', 'at_grid_edge',
               'spread_ratio_high_basin', 'spread_ratio_global',
               'spread_log10_std_high_basin'],
              'csv', 'per-gauge single fits under BOTH start conditions on the '
                     'amend grid (5-60000 ft^2/s, 1.15% spacing). D_high_basin is '
                     'the branch comparable with the window-start fits; D_ft2_s is '
                     'the unrestricted global argmin, which at g7 under the '
                     'quiescent start falls into a second, low-D basin.')

    write_csv(os.path.join(outdir, 'c3_amend_misfit_partition_norms_v2.csv'),
              part_rows,
              ['rule', 'gauge', 'convention', 'distance_ft', 'D_ft2_s', 't_start_s',
               'n_excluded', 'n_retained', 'ss_excluded', 'ss_retained',
               'share_pct_this_gauge'],
              'csv', 'the precursor share of the squared misfit under three '
                     'averaging conventions')

    write_csv(os.path.join(outdir, 'c3_amend_onset_margin_v2.csv'), onset_rows,
              ['gauge', 'distance_ft', 'onset_utc_60s', 'plateau_lo_utc',
               'plateau_hi_utc', 'plateau_width_s', 'lead_over_pump_restart_s',
               'plateau_hi_minus_pump_restart_s', 'p_smoothed_max_psi',
               'p_smoothed_at_restart_psi', 'drop_from_max_at_restart_psi',
               'ordering_resolved'],
              'csv', 'each onset\'s lead over the 11:18:57 restart compared with its '
                     'own +-1 psi plateau; ordering_resolved is False where the '
                     'plateau extends past the restart')

    write_csv(os.path.join(outdir, 'c3_amend_grid_sensitivity_v2.csv'), grid_rows,
              ['quantity', 'published_grid', 'amend_grid', 'difference_pct'],
              'csv', 'published optima vs the same optima on an independently offset '
                     'grid of the same resolution')

    arrays = os.path.join(outdir, 'c3_amend_arrays_v2.npz')
    np.savez(arrays, grid=AW['grid'],
             **{f'window_rmse_gauge_mean_{l}': AW[l]['rmse_gauge_mean']
                for l in ('full', 'A', 'B')},
             **{f'window_rmse_normalised_{l}': AW[l]['rmse_normalised']
                for l in ('full', 'A', 'B')},
             **{f'window_per_gauge_mse_{l}': AW[l]['per_gauge_mse']
                for l in ('full', 'A', 'B')},
             **{f'ext_rmse_gauge_mean_{l}': AE[l]['rmse_gauge_mean']
                for l in ('full', 'A', 'B')},
             **{f'ext_rmse_normalised_{l}': AE[l]['rmse_normalised']
                for l in ('full', 'A', 'B')},
             **{f'ext_per_gauge_mse_{l}': AE[l]['per_gauge_mse']
                for l in ('full', 'A', 'B')},
             target_gauges=np.array(G['tgt_gauges']))
    written.append((arrays, 'arrays_npz', 'both sweeps, full resolution'))

    results = {
        'amends_manifests': cfg['amend']['published_run_manifests'],
        'defect_list': cfg['amend']['defect_list'],
        'grid': {'min': float(grid[0]), 'max': float(grid[-1]),
                 'n_points': int(len(grid)),
                 'spacing_pct': float((grid[1] / grid[0] - 1) * 100)},
        'quiescence_ptp_psi': quiescence,
        'window_start_optima': SW,
        'quiescent_start_optima': SE,
        'extended_window_explained': ext_rows,
        'misfit_partition_conventions': partition,
        'r2_uniform_reproduction': r2_repro,
        'pumping_start_utc': pump_start.isoformat(),
        'preceding_cycle_shutin_utc': prev_stop.isoformat(),
        'preceding_cycle': prev_major,
        'onset_margin': onset_rows,
        'n_onsets_ordering_resolved': int(n_resolved),
        'grid_sensitivity': grid_rows,
        'n_forward_solves': int(n_solves),
    }
    rjson = os.path.join(outdir, 'c3_amend_results.json')
    with open(rjson, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    written.append((rjson, 'json', 'every number the amended README quotes'))

    # ---- 11. manifest -------------------------------------------------------
    s = G['gw'].series[G['src']]
    drv_win = rm.driver_record(
        kind='gauge_series', baseline_removal='subtract_first_sample',
        value_units='delta_psi',
        series_path=os.path.join('data/fiberis_format/s_well/gauges',
                                 f"gauge{G['src']}_data_swell.npz"),
        gauge_number=G['src'], gauge_md_ft=s.md_ft, taxis=s.taxis_s,
        values=s.delta_psi, time_start=cfg['window']['time_start'],
        time_end=cfg['window']['time_end'])
    drv_ext = rm.driver_record(
        kind='gauge_series', baseline_removal='subtract_first_sample',
        value_units='delta_psi',
        series_path=os.path.join('data/fiberis_format/s_well/gauges',
                                 f"gauge{G['src']}_data_swell.npz"),
        gauge_number=G['src'], gauge_md_ft=s.md_ft,
        taxis=G['ext'][G['src']]['t'], values=G['ext'][G['src']]['d'],
        time_start=cfg['extended_window']['time_start'],
        time_end=cfg['extended_window']['time_end'])
    sp = rm.source_protocol(
        application='dirichlet_node',
        solver_class='rev2_core.solve_forward (theta=1, harmonic, lambda=0; '
                     'bitwise identical to r1_calibration_core.solve_forward)',
        placement_rule=cfg['source']['selection_rule'],
        sources=[rm.source_record(G['mesh'].x, md_requested_ft=s.md_ft,
                                  mesh_idx=G['sidx'], driver=drv_win,
                                  label=f"g{G['src']} window-start run",
                                  index_in_source_list=0),
                 rm.source_record(G['mesh'].x, md_requested_ft=s.md_ft,
                                  mesh_idx=G['sidx'], driver=drv_ext,
                                  label=f"g{G['src']} quiescent-start run",
                                  index_in_source_list=1)],
        targets=[{'gauge': t['g'], 'md_ft': t['md_ft'],
                  'distance_ft': t['distance_ft'], 'mesh_idx': int(G['ridx'][k]),
                  'n_samples': int(t['t'].size),
                  'mask_t_start_A_s': t['t_start_A'],
                  'mask_t_start_B_s': t['t_start_B']}
                 for k, t in enumerate(G['targets'])],
        time_level='n', phase_chaining=rm.NONE_DECLARED,
        boundary_conditions={'lbc': cfg['solver']['lbc'],
                             'rbc': cfg['solver']['rbc']})

    taxis_w = np.arange(0.0, np.ceil(G['t_total'] / G['dt']) * G['dt'] + G['dt'] / 2,
                        G['dt'])
    taxis_e = np.arange(0.0,
                        np.ceil(G['ext_t_total'] / G['dt']) * G['dt'] + G['dt'] / 2,
                        G['dt'])
    nm = rm.numerics(
        time=[rm.time_record(taxis_w, mode='fixed', theta=1.0,
                             t_total_requested_s=G['t_total'],
                             dt_requested_s=G['dt'], source_time_level='n',
                             label='comparison window 11:24-11:45'),
              rm.time_record(taxis_e, mode='fixed', theta=1.0,
                             t_total_requested_s=G['ext_t_total'],
                             dt_requested_s=G['dt'], source_time_level='n',
                             label='extended record from 10:20 quiescence')],
        mesh=rm.mesh_record(G['mesh'].x, dx_requested_ft=float(cfg['mesh']['dx_ft']),
                            window_md_ft=(cfg['window']['md_min_ft'],
                                          cfg['window']['md_max_ft']),
                            pad_low_ft=float(cfg['mesh']['domain_pad_low_md_ft']),
                            pad_high_ft=float(cfg['mesh']['domain_pad_high_md_ft']),
                            refinement=rm.NONE_DECLARED),
        interface_avg=cfg['solver']['interface_avg'],
        boundary={'lbc': cfg['solver']['lbc'], 'rbc': cfg['solver']['rbc'],
                  'pml_thickness': 0.0, 'sigma_max': 0.0},
        diffusivity={'profile_family': 'uniform', 'param_names': ['D'],
                     'params': ['swept'], 'D_min': float(grid[0]),
                     'D_max': float(grid[-1]),
                     'n_grid': int(len(grid)),
                     'grid_sha256': rm.sha256_array(grid),
                     'profile_anchor': 'uniform_everywhere'},
        barriers=rm.NONE_DECLARED, leakage=rm.NONE_DECLARED,
        kernel={'name': 'rev2_core.solve_forward', 'banded': True,
                'equivalence_reference':
                    'output/rev2_20260901/A4/selftest_output.txt (T1: max|diff| = '
                    '0.000e+00 psi vs r1_calibration_core.solve_forward)'},
        rng=rm.NONE_DECLARED,
        parallel={'backend': 'multiprocessing.Pool', 'processes': nproc,
                  'deterministic': True})
    nm['n_forward_solves'] = int(n_solves)

    inputs = [('configs/rev2/c3_precursor.json', 'config',
               'the published C3 config, left byte-identical'),
              (cfg['amend']['published_run_manifests'][0], 'prior_run_output',
               'published C3 main-run manifest'),
              (cfg['amend']['published_run_manifests'][1], 'prior_run_output',
               'published C3 addendum manifest'),
              ('output/r2_diffusivity_profile/r2_manifest.json', 'prior_run_output',
               'the R2 uniform row this run reproduces'),
              (cfg['amend']['defect_list'], 'other', 'the reviewers\' defect list')]
    for n in sorted(G['gw'].series):
        inputs.append((f'data/fiberis_format/s_well/gauges/gauge{n}_data_swell.npz',
                       'gauge_series', f'g{n}'))

    outputs = [rm.output_decl(p, role=role,
                              dpi=(dpi if role.startswith('figure') else None),
                              note=note) for p, role, note in written]

    mpath = os.path.join(outdir, 'manifest_amend.json')
    rm.write_manifest(
        mpath, study_id=cfg['study_id'], task_id='C3-amend', config=cfg,
        config_path=args.config, inputs=inputs, source=sp, numerics=nm,
        outputs=outputs, results=results,
        started_utc=datetime.datetime.fromtimestamp(
            t_wall, datetime.timezone.utc).isoformat(),
        run_label='C3 amend against the reproduced defect list',
        allow_undeclared_outputs=True,
        notes=[
            'AMEND run. It adds files to output/rev2_20260901/C3/ and overwrites '
            'nothing: the products of c3_precursor.py and c3_precursor_addendum.py '
            'and their two manifests are untouched, so their hashes still verify.',
            'The undeclared files listed in outputs.undeclared_files_in_output_dir '
            'are those earlier products plus the two reviewers\' own script '
            'directories (challenge_numbers/, challenge_stats/).',
            'configs/rev2/c3_precursor.json asserts in extended_window.rationale '
            'that all seven gauges are "flat to <0.1 psi over 10:20-10:28". The '
            'measured peak-to-peak values are 0.017-0.463 psi (g3 0.463, g4 0.224), '
            'so that string is wrong and it is embedded in config_resolved inside '
            'manifest.json and manifest_addendum.json. The file is deliberately NOT '
            'edited (its sha256 is pinned in both); the corrected wording is in '
            'configs/rev2/c3_precursor_amend.json, and the tolerance both runs '
            'actually applied is quiescence_tol_psi = 0.5, which the data meet.',
            'Forward-solve counts: the published main run did 423 (181 coarse + 176 '
            'refine + 64 extended + 2), not the 421 its README said or the 422 the '
            'progress doc said; the addendum did 1. This amend adds '
            f'{n_solves}.',
        ])
    log(f"wrote {mpath}; {len(written)} product files; {n_solves} forward solves; "
        f"wall {time.time()-t_wall:.0f} s")


if __name__ == '__main__':
    main()
