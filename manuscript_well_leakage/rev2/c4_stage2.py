#!/usr/bin/env python3
"""C4 stage 2 -- the CONVERGED-DOMAIN completion of the (w, ratio) identifiability study.

Round `rev2_20260901`. Task C4. Written as a SEPARATE file from
`c4_identifiability.py` on purpose: that script's sha256 is recorded as
`code.main_script` in the manifests of every C4 solver run already on disk
(sweep/, padcheck/, sweep_pad20000/, dcurve_pad20000/, padprobe40000/), so
editing it would put all of them into code drift. Nothing here is edited into
it; the solver runs this stage needs were produced by the UNMODIFIED script
driven by new config files.

WHAT STAGE 1 LEFT OPEN. Stage 1 mapped the (w, ratio) plane at three background
D on a 5000 ft pad (B2's rule for this geometry) and then discovered, in its own
pad check, that 5000 ft is NOT enough once D0 is raised above about 140: the
shielded-gauge misfit moves -25.6 % at D0 = 1150. Its sections 6 (the planes at
D0 = 550 and 1150 on a converged domain) and 7 (the one-dimensional curve in D0)
were therefore left PENDING. This script closes both, and adds the two things a
referee would ask for next:

  * whether the identifiability conclusion (exponent, censoring, band widths) is
    an artifact of the contaminated domain -- it is not, and the numbers move by
    less than their own standard errors;
  * what the constrained combination actually is once the domain is converged.
    It is the ABSOLUTE series resistance W_tot / D_barrier, in s/ft, and it is
    the same number at every background D0 -- so not even the reduction RATIO is
    identified, and quoting one without both w and D0 is quoting nothing.

MODES
  --mode analyse   solve-free; reads the five cell JSONs, writes one JSON
  --mode figures   solve-free; three 300 dpi figures from that JSON

Every input is hashed into the product. `rev2_manifest` is imported for its
hashing and path helpers only; no solve happens here, so no manifest is written
(the same convention `c4_identifiability.py --mode analyse/figures` follows).
"""

import os
import sys
import json
import argparse
import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rev2_manifest as rm                     # noqa: E402

STUDY_ID = "C4_w_ratio_identifiability_stage2"
TASK_ID = "C4"
MISFIT_KEY = 'shielded_mean_rmse_delta_psi'
N_BARRIERS = 6.0                    # the six stage-7 frac hits
OUT_ROOT = 'output/rev2_20260901/C4'

RUNS = {
    'pad5000':   f"{OUT_ROOT}/sweep/c4_cells_v1.json",
    'pad20000':  f"{OUT_ROOT}/sweep_pad20000/c4_cells_pad20000_v1.json",
    'padcheck20000': f"{OUT_ROOT}/padcheck/c4_cells_pad20000_v1.json",
    'dcurve20000': f"{OUT_ROOT}/dcurve_pad20000/c4_cells_dcurve_pad20000_v1.json",
    'padprobe40000': f"{OUT_ROOT}/padprobe40000/c4_cells_padprobe40000_v1.json",
}
PAD_OF_RUN = {'pad5000': 5000.0, 'pad20000': 20000.0, 'padcheck20000': 20000.0,
              'dcurve20000': 20000.0, 'padprobe40000': 40000.0}


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_runs():
    runs, src = {}, {}
    for k, p in RUNS.items():
        ap = os.path.join(_ROOT, p)
        if not os.path.exists(ap):
            src[k] = {'path': p, 'present': False}
            continue
        with open(ap) as fh:
            doc = json.load(fh)
        runs[k] = doc
        man = os.path.join(os.path.dirname(ap), 'manifest.json')
        src[k] = {'path': p, 'present': True, 'sha256': rm.sha256_file(ap),
                  'n_cells': len(doc['cells']), 'tag': doc.get('tag'),
                  'pad_ft': PAD_OF_RUN[k],
                  'manifest': rm._rel(man) if os.path.exists(man) else None,
                  'manifest_sha256': (rm.sha256_file(man)
                                      if os.path.exists(man) else None)}
    return runs, src


def cells_of(doc, kinds=None):
    return [c for c in doc['cells'] if kinds is None or c['kind'] in kinds]


def key(c):
    return (round(float(c['D0']), 9), round(float(c['w_ft']), 9),
            round(float(np.log10(c['ratio'])), 6))


def plane_matrix(doc, D0):
    cs = [c for c in doc['cells']
          if c['kind'] == 'plane' and abs(c['D0'] - D0) < 1e-9]
    if not cs:
        return None, None, None
    ws = sorted({c['w_ft'] for c in cs})
    lrs = sorted({round(float(np.log10(c['ratio'])), 6) for c in cs},
                 reverse=True)
    A = np.full((len(ws), len(lrs)), np.nan)
    for c in cs:
        A[ws.index(c['w_ft']),
          lrs.index(round(float(np.log10(c['ratio'])), 6))] = c['misfit'][MISFIT_KEY]
    return np.asarray(ws, float), np.asarray(lrs, float), A


def parab_min(x, y):
    """Vertex of the parabola through three points, or (None, None)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    a, b, c = np.polyfit(x, y, 2)
    if a <= 0:
        return None, None
    xv = -b / (2 * a)
    return float(xv), float(a * xv * xv + b * xv + c)


def band(xg, yg, level):
    """Contiguous <= level interval containing the argmin, with censoring."""
    xg = np.asarray(xg, float)
    yg = np.asarray(yg, float)
    if xg[0] > xg[-1]:
        xg, yg = xg[::-1], yg[::-1]
    k = int(np.argmin(yg))
    lo_i = k
    while lo_i > 0 and yg[lo_i - 1] <= level:
        lo_i -= 1
    hi_i = k
    while hi_i < len(xg) - 1 and yg[hi_i + 1] <= level:
        hi_i += 1
    if lo_i == 0:
        lo, lo_c = float(xg[0]), True
    else:
        y0, y1 = yg[lo_i - 1], yg[lo_i]
        lo = float(xg[lo_i - 1] + (level - y0) * (xg[lo_i] - xg[lo_i - 1]) / (y1 - y0))
        lo_c = False
    if hi_i == len(xg) - 1:
        hi, hi_c = float(xg[-1]), True
    else:
        y0, y1 = yg[hi_i], yg[hi_i + 1]
        hi = float(xg[hi_i] + (level - y0) * (xg[hi_i + 1] - xg[hi_i]) / (y1 - y0))
        hi_c = False
    return {'lo': lo, 'hi': hi, 'lo_censored': lo_c, 'hi_censored': hi_c,
            'width': hi - lo, 'argmin_at_grid_edge': bool(k in (0, len(xg) - 1))}


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


def valley(ws, lrs, A, D0):
    """Per-w optimum, the exponent, the resistance and the +10% geometry."""
    per_w = []
    for i, w in enumerate(ws):
        y = A[i]
        j = int(np.argmin(y))
        edge = j in (0, len(y) - 1)
        if edge:
            xv, yv = float(lrs[j]), float(y[j])
        else:
            xv, yv = parab_min(lrs[j - 1:j + 2], y[j - 1:j + 2])
            if xv is None:
                xv, yv = float(lrs[j]), float(y[j])
        Db = float(D0 * 10.0 ** xv)
        Wtot = float(N_BARRIERS * 2.0 * w)
        per_w.append({'w_ft': float(w), 'log10_ratio_opt': xv,
                      'ratio_opt': float(10.0 ** xv),
                      'D_barrier_opt_ft2_s': Db,
                      'misfit_at_opt_psi': yv,
                      'grid_min_psi': float(y[j]),
                      'argmin_on_ratio_grid_edge': bool(edge),
                      'total_width_ft': Wtot,
                      'resistance_s_per_ft': Wtot / Db,
                      'crossing_time_s': Wtot * Wtot / Db,
                      'band_10pct_in_log10_ratio_at_this_w':
                          band(lrs, y, 1.1 * float(y[j]))})
    good = [p for p in per_w if not p['argmin_on_ratio_grid_edge']]
    fit = linfit([np.log10(p['w_ft']) for p in good],
                 [np.log10(p['D_barrier_opt_ft2_s']) for p in good]) \
        if len(good) > 2 else None
    R = np.array([p['resistance_s_per_ft'] for p in per_w])
    T = np.array([p['crossing_time_s'] for p in per_w])
    floor = np.array([p['misfit_at_opt_psi'] for p in per_w])
    Mmin = float(np.nanmin(A))
    lvl = 1.1 * Mmin
    prof_w = np.nanmin(A, axis=1)
    prof_r = np.nanmin(A, axis=0)
    i_min, j_min = np.unravel_index(int(np.nanargmin(A)), A.shape)
    # combination coordinate, lower envelope in 0.1-decade bins
    kap = np.log10(np.asarray(ws)[:, None] * N_BARRIERS * 2.0
                   / (D0 * 10.0 ** np.asarray(lrs)[None, :]))
    bw = 0.10
    e0 = np.floor(kap.min() / bw) * bw
    idx = ((kap - e0) // bw).astype(int)
    ctr, env = [], []
    for k in np.unique(idx):
        ctr.append(float(e0 + (k + 0.5) * bw))
        env.append(float(A[idx == k].min()))
    ctr = np.asarray(ctr)
    env = np.asarray(env)
    kb = band(ctr, env, 1.1 * float(env.min()))
    return {
        'D0_ft2_s': float(D0),
        'global_min_psi': Mmin,
        'global_min_at': {'w_ft': float(ws[i_min]),
                          'ratio': float(10.0 ** lrs[j_min]),
                          'D_barrier_ft2_s': float(D0 * 10.0 ** lrs[j_min])},
        'global_min_on_w_grid_edge': bool(i_min in (0, len(ws) - 1)),
        'global_min_on_ratio_grid_edge': bool(j_min in (0, len(lrs) - 1)),
        'per_w': per_w,
        'exponent_fit_log10_Dbarrier_vs_log10_w': fit,
        'resistance_s_per_ft': {'min': float(R.min()), 'max': float(R.max()),
                                'mean': float(R.mean()),
                                'spread_pct_of_mean':
                                    float(100.0 * (R.max() - R.min()) / R.mean())},
        'crossing_time_s': {'min': float(T.min()), 'max': float(T.max()),
                            'range_factor': float(T.max() / T.min())},
        'floor_variation_psi': float(floor.max() - floor.min()),
        'floor_variation_pct': float(100.0 * (floor.max() - floor.min())
                                     / floor.min()),
        'band_10pct_in_log10_w_profiled': band(np.log10(ws), prof_w, lvl),
        'band_10pct_in_log10_ratio_profiled': band(lrs, prof_r, lvl),
        'band_10pct_in_log10_resistance': {
            'coordinate': 'log10(W_tot / D_barrier), units log10(s/ft)',
            'bin_width_decades': bw,
            'envelope_min_psi': float(env.min()),
            'at_min_s_per_ft': float(10.0 ** ctr[int(np.argmin(env))]),
            'band': kb,
            'band_s_per_ft': [float(10.0 ** kb['lo']), float(10.0 ** kb['hi'])]},
        'w_grid_span_decades': float(np.log10(ws[-1] / ws[0])),
    }


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def run_analyse(outroot, tag):
    runs, src = load_runs()
    out = {'kind': 'derived_view_no_solve', 'study_id': STUDY_ID,
           'task_id': TASK_ID, 'tag': tag, 'generated_utc': utcnow(),
           'code_sha256': rm.sha256_file(os.path.abspath(__file__)),
           'stage1_code_sha256': rm.sha256_file(
               os.path.join(_HERE, 'c4_identifiability.py')),
           'misfit': ('gauge-mean RMSE over the shielded gauges {5, 6} of the '
                      'phase-3 dP referenced to each series own first '
                      'in-window sample; gauge-mean, not sample-pooled (C1)'),
           'sources': src,
           'conventions': {
               'n_barriers': N_BARRIERS,
               'total_width_ft': 'W_tot = 6 * 2w, the six stage-7 barriers in series',
               'resistance': 'R = W_tot / D_barrier, s/ft',
               'band': '+10 % of the minimum, the R1 convention; grid-edge '
                       'endpoints are flagged CENSORED and never quoted as '
                       'estimates'}}

    # ---- A. the planes, both pads ----------------------------------------
    planes = {}
    for run in ('pad5000', 'pad20000'):
        if run not in runs:
            continue
        doc = runs[run]
        for D0 in sorted({c['D0'] for c in doc['cells'] if c['kind'] == 'plane'}):
            ws, lrs, A = plane_matrix(doc, D0)
            if not np.isfinite(A).all():
                raise RuntimeError(
                    f"plane {run}:{D0:g} has {int((~np.isfinite(A)).sum())} "
                    f"missing cells of {A.size}; a partially filled plane must "
                    f"not be summarised as if it were complete")
            v = valley(ws, lrs, A, D0)
            v['grid_complete'] = {'n_w': int(len(ws)), 'n_ratio': int(len(lrs)),
                                  'n_cells': int(A.size), 'n_missing': 0}
            planes[f"{run}:{D0:g}"] = v
    out['planes'] = planes

    # ---- B. cross-pad contamination of the planes ------------------------
    cross = {}
    if 'pad5000' in runs and 'pad20000' in runs:
        m5 = {key(c): c for c in runs['pad5000']['cells']}
        for D0 in sorted({c['D0'] for c in runs['pad20000']['cells']
                          if c['kind'] == 'plane'}):
            rows = []
            for c in runs['pad20000']['cells']:
                if c['kind'] != 'plane' or abs(c['D0'] - D0) > 1e-9:
                    continue
                b = m5.get(key(c))
                if b is None:
                    continue
                a5v = b['misfit'][MISFIT_KEY]
                a20 = c['misfit'][MISFIT_KEY]
                rows.append({'w_ft': c['w_ft'], 'ratio': c['ratio'],
                             'pad5000_psi': a5v, 'pad20000_psi': a20,
                             'difference_psi': a20 - a5v,
                             'difference_pct': 100.0 * (a20 - a5v) / a5v})
            d = np.array([abs(r['difference_pct']) for r in rows])
            worst = rows[int(np.argmax(d))]
            k5 = planes.get(f"pad5000:{D0:g}")
            k20 = planes.get(f"pad20000:{D0:g}")
            cross[f"{D0:g}"] = {
                'n_cells_matched': len(rows),
                'abs_difference_pct': {'median': float(np.median(d)),
                                       'max': float(d.max()),
                                       'min': float(d.min())},
                'worst_cell': worst,
                'derived_quantities': {
                    'min_misfit_psi': [k5['global_min_psi'],
                                       k20['global_min_psi']],
                    'exponent_p': [
                        k5['exponent_fit_log10_Dbarrier_vs_log10_w']['slope'],
                        k20['exponent_fit_log10_Dbarrier_vs_log10_w']['slope']],
                    'exponent_stderr': [
                        k5['exponent_fit_log10_Dbarrier_vs_log10_w']['slope_stderr'],
                        k20['exponent_fit_log10_Dbarrier_vs_log10_w']['slope_stderr']],
                    'resistance_mean_s_per_ft': [
                        k5['resistance_s_per_ft']['mean'],
                        k20['resistance_s_per_ft']['mean']],
                    'ratio_band_decades': [
                        k5['band_10pct_in_log10_ratio_profiled']['width'],
                        k20['band_10pct_in_log10_ratio_profiled']['width']],
                    'resistance_band_decades': [
                        k5['band_10pct_in_log10_resistance']['band']['width'],
                        k20['band_10pct_in_log10_resistance']['band']['width']],
                    'w_band_censored_both_ends': [
                        bool(k5['band_10pct_in_log10_w_profiled']['lo_censored']
                             and k5['band_10pct_in_log10_w_profiled']['hi_censored']),
                        bool(k20['band_10pct_in_log10_w_profiled']['lo_censored']
                             and k20['band_10pct_in_log10_w_profiled']['hi_censored'])],
                    'order': ['pad5000', 'pad20000']},
            }
    out['plane_cross_pad'] = cross

    # ---- C. the invariant across D0 --------------------------------------
    # Converged planes: D0 = 140 at the 5000 ft pad (its own pad check says
    # <= 2.9e-4 psi there) and D0 = 550, 1150 at the 20000 ft pad.
    conv = {'140': 'pad5000:140', '550': 'pad20000:550', '1150': 'pad20000:1150'}
    inv = {}
    for label, sel in (('converged', conv),
                       ('pad5000_all_three',
                        {'140': 'pad5000:140', '550': 'pad5000:550',
                         '1150': 'pad5000:1150'})):
        rowsR, D0s = [], []
        for k, v in sel.items():
            if v not in planes:
                continue
            p = planes[v]
            rowsR.append(p['resistance_s_per_ft']['mean'])
            D0s.append(float(k))
        if len(rowsR) < 2:
            continue
        R = np.array(rowsR)
        inv[label] = {
            'source_planes': {k: v for k, v in sel.items() if v in planes},
            'D0_ft2_s': D0s,
            'mean_resistance_s_per_ft': [float(v) for v in R],
            'spread_pct_of_mean': float(100.0 * (R.max() - R.min()) / R.mean()),
            'fit_log10_R_vs_log10_D0': linfit(np.log10(D0s), np.log10(R))
            if len(D0s) > 2 else None}
    out['resistance_invariance'] = inv

    # ---- D. the converged D curve ----------------------------------------
    for run, name in (('dcurve20000', 'd_curve_pad20000'),
                      ('pad5000', 'd_curve_pad5000')):
        if run not in runs:
            continue
        doc = runs[run]
        dcs = [c for c in doc['cells'] if c['kind'] == 'dcurve']
        if not dcs:
            continue
        D0s = sorted({c['D0'] for c in dcs})
        rows = []
        for D0 in D0s:
            sel = [c for c in dcs if abs(c['D0'] - D0) < 1e-9]
            uni = [c for c in sel if c['ratio'] >= 1.0]
            seal = [c for c in sel if c['ratio'] <= 1e-8]
            bar = sorted([c for c in sel if 1e-8 < c['ratio'] < 1.0],
                         key=lambda c: -c['ratio'])
            if len(bar) < 3:
                continue
            lr = np.array([np.log10(c['ratio']) for c in bar])
            y = np.array([c['misfit'][MISFIT_KEY] for c in bar])
            j = int(np.argmin(y))
            edge = j in (0, len(y) - 1)
            if edge:
                xv, yv = float(lr[j]), float(y[j])
            else:
                xv, yv = parab_min(lr[j - 1:j + 2], y[j - 1:j + 2])
                if xv is None:
                    xv, yv = float(lr[j]), float(y[j])
            w = float(bar[0]['w_ft'])
            Wtot = N_BARRIERS * 2.0 * w
            Db = float(D0 * 10.0 ** xv)
            rows.append({
                'D0_ft2_s': float(D0), 'w_ft': w,
                'log10_ratios': [float(v) for v in lr],
                'misfit_by_ratio_psi': [float(v) for v in y],
                'uniform_misfit_psi': (uni[0]['misfit'][MISFIT_KEY]
                                       if uni else None),
                'sealed_misfit_psi': (seal[0]['misfit'][MISFIT_KEY]
                                      if seal else None),
                'profiled_min_psi': yv, 'ratio_opt': float(10.0 ** xv),
                'D_barrier_opt_ft2_s': Db,
                'resistance_s_per_ft': Wtot / Db,
                'argmin_on_ratio_grid_edge': bool(edge),
                'band_10pct_in_log10_ratio': band(lr, y, 1.1 * float(y[j]))})
        prof = np.array([r['profiled_min_psi'] for r in rows])
        Ds = np.array([r['D0_ft2_s'] for r in rows])
        good = [r for r in rows if not r['argmin_on_ratio_grid_edge']]
        kmin = int(np.argmin(prof))
        out[name] = {
            'pad_ft': PAD_OF_RUN[run], 'w_ft': rows[0]['w_ft'],
            'rows': rows,
            'profiled_over_ratio_min_psi': [float(v) for v in prof],
            'argmin_D0_ft2_s': float(Ds[kmin]),
            'argmin_on_D_grid_edge': bool(kmin in (0, len(Ds) - 1)),
            'band_10pct_in_log10_D0': band(np.log10(Ds), prof,
                                           1.1 * float(prof.min())),
            'fit_log10_Dbarrier_opt_vs_log10_D0': linfit(
                np.log10([r['D0_ft2_s'] for r in good]),
                np.log10([r['D_barrier_opt_ft2_s'] for r in good]))
            if len(good) > 2 else None,
            'fit_log10_ratio_opt_vs_log10_D0': linfit(
                np.log10([r['D0_ft2_s'] for r in good]),
                np.log10([r['ratio_opt'] for r in good]))
            if len(good) > 2 else None,
            'fit_log10_resistance_vs_log10_D0': linfit(
                np.log10([r['D0_ft2_s'] for r in good]),
                np.log10([r['resistance_s_per_ft'] for r in good]))
            if len(good) > 2 else None,
            'resistance_s_per_ft': {
                'values': [r['resistance_s_per_ft'] for r in rows],
                'min': float(min(r['resistance_s_per_ft'] for r in rows)),
                'max': float(max(r['resistance_s_per_ft'] for r in rows))},
        }

    # contamination of the D curve, cell by cell
    if 'dcurve20000' in runs and 'pad5000' in runs:
        m5 = {key(c): c for c in runs['pad5000']['cells'] if c['kind'] == 'dcurve'}
        rows = []
        for c in runs['dcurve20000']['cells']:
            if c['kind'] != 'dcurve':
                continue
            b = m5.get(key(c))
            if b is None:
                continue
            a, d = b['misfit'][MISFIT_KEY], c['misfit'][MISFIT_KEY]
            rows.append({'D0': c['D0'], 'ratio': c['ratio'],
                         'pad5000_psi': a, 'pad20000_psi': d,
                         'difference_psi': d - a,
                         'difference_pct': 100.0 * (d - a) / a})
        by_D = {}
        for r in rows:
            by_D.setdefault(f"{r['D0']:g}", []).append(abs(r['difference_pct']))
        out['d_curve_cross_pad'] = {
            'n_cells_matched': len(rows),
            'worst_abs_pct_by_D0': {k: float(max(v)) for k, v in by_D.items()},
            'median_abs_pct_by_D0': {k: float(np.median(v))
                                     for k, v in by_D.items()},
            'rows': rows}

    # ---- E. the padding ladder 5000 -> 20000 -> 40000 --------------------
    if 'padprobe40000' in runs:
        pool20 = {}
        for run in ('pad20000', 'padcheck20000', 'dcurve20000'):
            if run in runs:
                for c in runs[run]['cells']:
                    pool20.setdefault(key(c), (run, c))
        pool5 = {key(c): c for c in runs.get('pad5000', {'cells': []})['cells']}
        rows = []
        for c in runs['padprobe40000']['cells']:
            k = key(c)
            m40 = c['misfit'][MISFIT_KEY]
            r20 = pool20.get(k)
            m20 = None if r20 is None else r20[1]['misfit'][MISFIT_KEY]
            b5 = pool5.get(k)
            m5v = None if b5 is None else b5['misfit'][MISFIT_KEY]
            rows.append({
                'D0': c['D0'], 'w_ft': c['w_ft'], 'ratio': c['ratio'],
                'pad5000_psi': m5v,
                'pad20000_psi': m20,
                'pad20000_from_run': None if r20 is None else r20[0],
                'pad40000_psi': m40,
                'change_5000_to_20000_pct': (None if (m5v is None or m20 is None)
                                             else 100.0 * (m20 - m5v) / m5v),
                'change_20000_to_40000_pct': (None if m20 is None else
                                              100.0 * (m40 - m20) / m20),
                'change_20000_to_40000_psi': (None if m20 is None
                                              else m40 - m20)})
        by_D = {}
        for r in rows:
            if r['change_20000_to_40000_pct'] is None:
                continue
            by_D.setdefault(f"{r['D0']:g}", []).append(r)
        verdict = {}
        for k, v in by_D.items():
            last = max(abs(q['change_20000_to_40000_pct']) for q in v)
            first = max(abs(q['change_5000_to_20000_pct']) for q in v
                        if q['change_5000_to_20000_pct'] is not None) \
                if any(q['change_5000_to_20000_pct'] is not None for q in v) else None
            verdict[k] = {
                'worst_abs_change_5000_to_20000_pct': first,
                'worst_abs_change_20000_to_40000_pct': last,
                'worst_abs_change_20000_to_40000_psi':
                    float(max(abs(q['change_20000_to_40000_psi']) for q in v)),
                'converged_at_20000': bool(last < 1.0),
                'criterion': ('converged = worst |change| from the 20000 ft to '
                              'the 40000 ft pad is below 1 % of the misfit')}
        out['pad_ladder'] = {
            'note': ('the same five cells (uniform control, ratios 1e-3/1e-4/'
                     '1e-5, sealed control) at w = 1 ft on three domains'),
            'rows': rows, 'verdict_by_D0': verdict}

    # ---- F. between-run reproducibility ----------------------------------
    # Cells solved in two different runs on the SAME mesh must be bit-identical.
    same_mesh = [r for r in ('pad20000', 'padcheck20000', 'dcurve20000')
                 if r in runs]
    seen, dup = {}, []
    for run in same_mesh:
        for c in runs[run]['cells']:
            k = key(c)
            if k in seen:
                o_run, o = seen[k]
                dup.append({'runs': [o_run, run], 'D0': c['D0'],
                            'w_ft': c['w_ft'], 'ratio': c['ratio'],
                            'trace_sha256_equal':
                                bool(o['trace_sha256'] == c['trace_sha256']),
                            'misfit_difference_psi':
                                float(c['misfit'][MISFIT_KEY]
                                      - o['misfit'][MISFIT_KEY])})
            else:
                seen[k] = (run, c)
    out['between_run_reproducibility'] = {
        'note': ('cells solved independently in two runs on the same 20000 ft '
                 'mesh; the traces are hashed, so identity is exact, not '
                 'approximate'),
        'n_duplicate_cells': len(dup),
        'n_bit_identical': int(sum(1 for d in dup if d['trace_sha256_equal'])),
        'max_abs_misfit_difference_psi':
            float(max((abs(d['misfit_difference_psi']) for d in dup),
                      default=0.0)),
        'duplicates': dup}

    # ---- G. barrier-report sanity across every run ------------------------
    wr = []
    for run, doc in runs.items():
        bad_w, nfb, ovl, n = 0, 0, 0, 0
        worst = 0.0
        worst_R = 0.0
        for c in doc['cells']:
            b = c.get('barrier')
            if not b:
                continue
            n += 1
            nfb += int(b['n_fallback'])
            ovl += int(b['n_overlapping_pairs'])
            want = 2.0 * c['w_ft']
            rw = b['realised_full_width_ft']
            e = max(abs(float(rw['min']) - want), abs(float(rw['max']) - want)) / want
            worst = max(worst, e)
            bad_w += int(e > 1e-12)
            # rev2_core's own excess resistance must equal W_tot*(1/D_b - 1/D0)
            Db = float(c['D0']) * float(c['ratio'])
            pred = N_BARRIERS * want * (1.0 / Db - 1.0 / float(c['D0']))
            worst_R = max(worst_R, abs(pred - float(b['excess_resistance_s_per_ft']))
                          / abs(pred))
        wr.append({'run': run, 'n_barrier_cells': n, 'n_fallback_total': nfb,
                   'n_overlapping_total': ovl,
                   'n_cells_with_width_error': bad_w,
                   'worst_relative_width_error': worst,
                   'worst_relative_excess_resistance_error': worst_R})
    out['barrier_width_check'] = {
        'note': ('batch 5 reported that build_barrier_profile can realise '
                 '2w + dx with n_fallback still 0; every C4 cell is checked '
                 'against its own requested width'),
        'by_run': wr}

    # ---- H. does the BAND hold at the level of the simulated traces? ------
    # The valley-floor exponent is a statement about the misfit. The stronger
    # statement is that two barriers with the same series resistance produce the
    # SAME shielded-gauge trace, not merely the same misfit. The ratio ladder is
    # 0.25 decade, so pairs (w, 10w) at (ratio, 10*ratio) have EXACTLY the same
    # resistance and are directly comparable; the trace separation is then read
    # against the manuscript's own panel resolution (E1: 1 typographic point on
    # 104's ax3 = 21.98 psi).
    PT_PSI = 21.97971781305115
    R_opt_of = {k: v['band_10pct_in_log10_resistance']['at_min_s_per_ft']
                for k, v in planes.items()}
    trace_eq = {}
    for run, D0 in (('pad5000', 140.0), ('pad20000', 550.0),
                    ('pad20000', 1150.0)):
        if run not in runs:
            continue
        npz = os.path.join(_ROOT, os.path.dirname(RUNS[run]),
                           os.path.basename(RUNS[run]).replace('.json', '.npz'))
        if not os.path.exists(npz):
            continue
        z = np.load(npz)
        gnum = [int(v) for v in np.asarray(z['gauge_numbers']).ravel()]
        if gnum[:2] != [5, 6]:
            raise RuntimeError(
                f"{npz}: gauge columns are {gnum}, so [:, :2] is not gauges "
                f"5 and 6; the trace comparison would silently use the wrong "
                f"gauges")
        cs = [c for c in runs[run]['cells']
              if c['kind'] == 'plane' and abs(c['D0'] - D0) < 1e-9]
        if not cs:
            continue

        def tag_of(c):
            # c4_identifiability.cell_tag, replicated rather than imported so
            # that this file never imports the stage-1 module
            return (f"D{c['D0']:g}_w{c['w_ft']:g}_r{c['ratio']:g}"
                    .replace('.', 'p').replace('-', 'm'))

        T = {}
        for c in cs:
            k = 'dp_' + tag_of(c)
            if k in z:
                T[key(c)] = np.asarray(z[k][:, :2], float)   # g5, g6 only
        lrs = sorted({round(float(np.log10(c['ratio'])), 6) for c in cs},
                     reverse=True)
        ws = sorted({c['w_ft'] for c in cs})
        pairs, across = [], []
        for w in ws:
            w10 = 10.0 * w
            if w10 not in ws:
                continue
            for lr in lrs:
                k1 = (round(D0, 9), round(w, 9), lr)
                k2 = (round(D0, 9), round(w10, 9), round(lr + 1.0, 6))
                if k1 not in T or k2 not in T:
                    continue
                d = np.abs(T[k1] - T[k2])
                pairs.append({
                    'w_ft': [w, w10], 'log10_ratio': [lr, lr + 1.0],
                    'resistance_s_per_ft': float(N_BARRIERS * 2 * w
                                                 / (D0 * 10.0 ** lr)),
                    'max_abs_trace_difference_psi': float(d.max()),
                    'max_in_typographic_points': float(d.max() / PT_PSI),
                    'peak_dp_psi': float(np.abs(T[k1]).max()),
                    'pct_of_peak': float(100.0 * d.max()
                                         / max(np.abs(T[k1]).max(), 1e-30))})
        # the contrast: ONE grid step in ratio at fixed w, i.e. 0.25 decade of
        # resistance and nothing else
        for w in ws:
            for a, b in zip(lrs[:-1], lrs[1:]):
                k1 = (round(D0, 9), round(w, 9), a)
                k2 = (round(D0, 9), round(w, 9), b)
                if k1 not in T or k2 not in T:
                    continue
                d = np.abs(T[k1] - T[k2])
                across.append({'w_ft': w, 'log10_ratio': [a, b],
                               'd_log10_resistance': float(abs(b - a)),
                               'max_abs_trace_difference_psi': float(d.max()),
                               'max_in_typographic_points': float(d.max() / PT_PSI)})
        # all-pairs view: is the separation a function of the resistance
        # mismatch ALONE?
        ks = sorted(T)
        scat = []
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b = ks[i], ks[j]
                dR = abs((np.log10(N_BARRIERS * 2 * a[1]) - a[2])
                         - (np.log10(N_BARRIERS * 2 * b[1]) - b[2]))
                dw = abs(np.log10(a[1] / b[1]))
                sep = float(np.abs(T[a] - T[b]).max())
                scat.append((float(dR), float(dw), sep))
        scat = np.asarray(scat)
        onband = scat[scat[:, 0] < 1e-9]
        trace_eq[f"{run}:{D0:g}"] = {
            'psi_per_typographic_point': PT_PSI,
            'note': ('pairs (w, 10w) at (ratio, 10*ratio) have IDENTICAL series '
                     'resistance; the ladder is 0.25 decade so the pairing is '
                     'exact, not interpolated. Traces are the stored 10 s '
                     'decimated dP at gauges 5 and 6.'),
            'n_matched_pairs': len(pairs),
            'along_band_10x_width': {
                'max_abs_trace_difference_psi':
                    float(max(p['max_abs_trace_difference_psi'] for p in pairs))
                    if pairs else None,
                'median_abs_trace_difference_psi':
                    float(np.median([p['max_abs_trace_difference_psi']
                                     for p in pairs])) if pairs else None,
                'max_in_typographic_points':
                    float(max(p['max_in_typographic_points'] for p in pairs))
                    if pairs else None,
                'max_pct_of_peak':
                    float(max(p['pct_of_peak'] for p in pairs)) if pairs else None,
                'pairs': pairs},
            'across_band_one_grid_step': {
                'd_log10_resistance': 0.25,
                'max_abs_trace_difference_psi':
                    float(max(q['max_abs_trace_difference_psi'] for q in across))
                    if across else None,
                'median_abs_trace_difference_psi':
                    float(np.median([q['max_abs_trace_difference_psi']
                                     for q in across])) if across else None,
                'median_in_typographic_points':
                    float(np.median([q['max_in_typographic_points']
                                     for q in across])) if across else None},
            'at_the_valley_floor': (lambda: (
                (lambda pr, aq: {
                    'note': ('the matched pair whose common resistance is '
                             'closest to this plane optimum, against one '
                             '0.25-decade ratio step at the same place'),
                    'pair': pr, 'across_step': aq,
                    'ratio_across_over_along': (
                        None if pr is None or aq is None or
                        pr['max_abs_trace_difference_psi'] <= 0 else
                        aq['max_abs_trace_difference_psi']
                        / pr['max_abs_trace_difference_psi'])})(
                    min(pairs, key=lambda q: abs(np.log10(
                        q['resistance_s_per_ft']
                        / R_opt_of[f"{run}:{D0:g}"]))) if pairs else None,
                    min(across, key=lambda q: abs(
                        np.log10(N_BARRIERS * 2 * q['w_ft']
                                 / (D0 * 10.0 ** q['log10_ratio'][0]))
                        - np.log10(R_opt_of[f"{run}:{D0:g}"])))
                    if across else None)))(),
            'all_pairs': {
                'n_pairs': int(scat.shape[0]),
                'n_exactly_equal_resistance': int(onband.shape[0]),
                'separation_at_equal_resistance_psi': {
                    'max': float(onband[:, 2].max()) if onband.size else None,
                    'median': float(np.median(onband[:, 2]))
                    if onband.size else None},
                'max_log10_w_span_at_equal_resistance':
                    float(onband[:, 1].max()) if onband.size else None},
        }
        z.close()
    out['trace_equivalence'] = trace_eq

    # ---- I. how much of the D-curve/plane disagreement is the ratio grid? --
    # The plane ladder is 0.25 decade and the D-curve ladder 0.5 decade, and the
    # optimum is taken as a parabola vertex, so the coarser ladder biases it. The
    # bias is MEASURED, not argued: the plane's own w = 1 ft row is subsampled to
    # the half-decade ladder and re-minimised. If that reproduces the D curve,
    # the disagreement is grid resolution and nothing else.
    gridbias = {}
    for run, D0s in (('pad20000', (550.0, 1150.0)), ('pad5000', (140.0,))):
        if run not in runs:
            continue
        for D0 in D0s:
            cs = [c for c in runs[run]['cells']
                  if c['kind'] == 'plane' and abs(c['D0'] - D0) < 1e-9
                  and abs(c['w_ft'] - 1.0) < 1e-9]
            if len(cs) < 5:
                continue
            cs.sort(key=lambda c: -c['ratio'])
            lr = np.array([np.log10(c['ratio']) for c in cs])
            y = np.array([c['misfit'][MISFIT_KEY] for c in cs])
            j = int(np.argmin(y))
            xf, _ = parab_min(lr[j - 1:j + 2], y[j - 1:j + 2])
            m = np.abs(np.round(lr * 2) - lr * 2) < 1e-9
            lr2, y2 = lr[m], y[m]
            j2 = int(np.argmin(y2))
            xh, _ = parab_min(lr2[j2 - 1:j2 + 2], y2[j2 - 1:j2 + 2])
            Wt = N_BARRIERS * 2.0
            dcv = None
            for nm in ('d_curve_pad20000', 'd_curve_pad5000'):
                if nm in out and PAD_OF_RUN[run] == out[nm]['pad_ft']:
                    for r in out[nm]['rows']:
                        if abs(r['D0_ft2_s'] - D0) < 1e-9:
                            dcv = r['resistance_s_per_ft']
            gridbias[f"{run}:{D0:g}"] = {
                'quarter_decade_ratio_opt': float(10.0 ** xf),
                'half_decade_ratio_opt': float(10.0 ** xh),
                'shift_decades': float(xh - xf),
                'resistance_quarter_decade_s_per_ft': float(Wt / (D0 * 10.0 ** xf)),
                'resistance_half_decade_s_per_ft': float(Wt / (D0 * 10.0 ** xh)),
                'D_curve_resistance_s_per_ft': dcv,
                'reproduces_D_curve': (None if dcv is None else
                                       bool(abs(Wt / (D0 * 10.0 ** xh) - dcv)
                                            < 1e-6 * max(dcv, 1.0))),
                'note': ('the D curve runs a half-decade ladder, so its '
                         'per-D0 resistance carries this bias; quote the '
                         'planes for the VALUE and the D curve for the SLOPE')}
    out['ratio_grid_resolution_bias'] = gridbias

    p = os.path.join(_ROOT, outroot, f"c4_stage2_analysis_{tag}.json")
    rm.assert_absent([p])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=float)
    print(f"[C4.2] analysis -> {p}")

    # console summary
    for k in sorted(planes):
        v = planes[k]
        f = v['exponent_fit_log10_Dbarrier_vs_log10_w']
        print(f"  plane {k:>16}: min {v['global_min_psi']:8.3f} psi | p = "
              f"{f['slope']:.4f} +- {f['slope_stderr']:.4f} (R2 {f['r2']:.5f}) | "
              f"R = {v['resistance_s_per_ft']['min']:.0f}-"
              f"{v['resistance_s_per_ft']['max']:.0f} s/ft "
              f"(mean {v['resistance_s_per_ft']['mean']:.0f})")
    for k, v in inv.items():
        print(f"  invariance[{k}]: R = "
              + ', '.join(f"{a:.0f}" for a in v['mean_resistance_s_per_ft'])
              + f" s/ft at D0 = " + ', '.join(f"{a:g}" for a in v['D0_ft2_s'])
              + f"  spread {v['spread_pct_of_mean']:.1f}%")
    if 'd_curve_pad20000' in out:
        dc = out['d_curve_pad20000']
        print(f"  D curve (20000 ft pad): R = {dc['resistance_s_per_ft']['min']:.0f}"
              f"-{dc['resistance_s_per_ft']['max']:.0f} s/ft; "
              f"fit log10 R vs log10 D0 slope "
              f"{dc['fit_log10_resistance_vs_log10_D0']['slope']:+.4f} +- "
              f"{dc['fit_log10_resistance_vs_log10_D0']['slope_stderr']:.4f}")
    if 'pad_ladder' in out:
        for k, v in sorted(out['pad_ladder']['verdict_by_D0'].items(),
                           key=lambda q: float(q[0])):
            print(f"  pad ladder D0={k:>5}: 5000->20000 "
                  f"{v['worst_abs_change_5000_to_20000_pct']}, 20000->40000 "
                  f"{v['worst_abs_change_20000_to_40000_pct']:.3f}% -> "
                  f"converged={v['converged_at_20000']}")
    b = out['between_run_reproducibility']
    print(f"  between-run duplicates: {b['n_bit_identical']}/"
          f"{b['n_duplicate_cells']} bit-identical, max |dmisfit| "
          f"{b['max_abs_misfit_difference_psi']:.3e} psi")
    return 0


# ---------------------------------------------------------------------------
# figures (solve-free)
# ---------------------------------------------------------------------------
# Same encoding discipline as stage 1: one perceptually uniform sequential ramp
# for a magnitude, ordered ramps for ordered families, direct labels on the
# curves, no dual axes, recessive grid and spines.

INK = '#1a1a1a'
INK2 = '#555555'
ACC = '#3b7ea1'
RED = '#d43d51'

# the three CONVERGED planes and where each comes from
CONVERGED = [(140.0, 'pad5000', 'output/rev2_20260901/C4/c4_analysis_v2.json',
              '5000 ft pad (converged here: its own pad check moves the misfit '
              'by <= 2.9e-4 psi)'),
             (550.0, 'pad20000',
              'output/rev2_20260901/C4/c4_analysis_pad20000_v1.json',
              '20000 ft pad'),
             (1150.0, 'pad20000',
              'output/rev2_20260901/C4/c4_analysis_pad20000_v1.json',
              '20000 ft pad')]


def _style(ax):
    ax.tick_params(colors=INK2, labelsize=8, width=0.8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(INK2)
        ax.spines[sp].set_linewidth(0.8)
    ax.set_axisbelow(True)


def run_figures(outroot, tag, an_tag, amend_tag=None):
    """The stage-2 figures.

    AMEND 1 (2026-09-02).  Two things were wrong here and are fixed:

    * both D-curve panels fitted all NINE backgrounds, including `D0` = 4600,
      which the same report flags NOT CONVERGED and says must not be quoted.
      The fits are now over the eight padding-converged rows and the 4600 point
      is drawn in a distinct open grey marker labelled as excluded.  The third
      panel's title said "no trend with $D_0$ over 131x" -- 131x is the span
      only when the excluded row is in -- while printing a slope that is 2.3
      standard errors from zero on its own arithmetic.  The title is now built
      from the fit that is actually drawn.
    * the per-`w` optimum was a parabola vertex on a 0.25-decade ratio ladder
      and was not converged in that step.  Where `c4_stage2_amend1_<tag>.json`
      is present, the CONVERGED (0.05-decade) optimum is what the figures plot
      and quote; the published 0.25-decade points are kept alongside, marked as
      such, so the size of the estimator's bias is visible rather than hidden.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    runs, _ = load_runs()
    ap = os.path.join(_ROOT, outroot, f"c4_stage2_analysis_{an_tag}.json")
    with open(ap) as fh:
        an = json.load(fh)
    am = None
    am_path = None
    if amend_tag:
        am_path = os.path.join(_ROOT, outroot,
                               f"c4_stage2_amend1_{amend_tag}.json")
        if os.path.exists(am_path):
            with open(am_path) as fh:
                am = json.load(fh)
        else:
            raise SystemExit(f"amend analysis not found: {am_path}")

    def am_plane(key):
        return None if am is None else am.get('planes', {}).get(key)

    def am_dcurve():
        return None if am is None else am.get('d_curve_converged_ladder')
    figdir = os.path.join(_ROOT, outroot, 'figs')
    os.makedirs(figdir, exist_ok=True)
    written = []
    dpi = 300
    stage1 = {}
    for p in {q[2] for q in CONVERGED}:
        with open(os.path.join(_ROOT, p)) as fh:
            stage1[p] = json.load(fh)

    # ---- figure 1: the converged plane ------------------------------------
    Ms = {}
    for D0, run, _p, _n in CONVERGED:
        ws, lrs, A = plane_matrix(runs[run], D0)
        Ms[D0] = (ws, lrs, A)
    vmin = min(float(np.nanmin(A)) for _w, _l, A in Ms.values())
    vmax = max(float(np.nanmax(A)) for _w, _l, A in Ms.values())
    levels = np.linspace(vmin, vmax, 41)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), sharey=True)
    for k, (D0, run, apath, note) in enumerate(CONVERGED):
        ax = axes[k]
        ws, lrs, A = Ms[D0]
        lw_ = np.log10(ws)
        cf = ax.contourf(lw_, lrs, A.T, levels=levels, cmap='viridis_r')
        v = an['planes'][f"{run}:{D0:g}"]
        vam = am_plane(f"{run}:{D0:g}")
        ax.contour(lw_, lrs, A.T, levels=[1.1 * v['global_min_psi']],
                   colors='#ffffff', linewidths=1.6)
        if vam:
            # the converged (0.05-decade) optimum, and the published
            # 0.25-decade one kept alongside so the bias is visible
            ax.plot([np.log10(p['w_ft']) for p in vam['per_w']],
                    [p['coarse_log10_ratio_opt'] for p in vam['per_w']], 'x',
                    ms=4.5, color='#c9c9c9', mew=1.0, zorder=5)
            ax.plot([np.log10(p['w_ft']) for p in vam['per_w']],
                    [p['fine_log10_ratio_opt'] for p in vam['per_w']], 'o',
                    ms=4.5, mfc='#ffffff', mec=INK, mew=0.9, zorder=6)
            f = vam['exponent_fit_log10_Dbarrier_vs_log10_w_fine']
        else:
            ax.plot([np.log10(p['w_ft']) for p in v['per_w']],
                    [p['log10_ratio_opt'] for p in v['per_w']], 'o', ms=4.5,
                    mfc='#ffffff', mec=INK, mew=0.9, zorder=5)
            f = v['exponent_fit_log10_Dbarrier_vs_log10_w']
        xx = np.array([lw_[0] - 0.08, lw_[-1] + 0.08])
        ax.plot(xx, f['slope'] * xx + f['intercept'] - np.log10(D0), '-',
                color='#ffffff', lw=1.2, zorder=4)
        # the region indistinguishable from a perfect seal (stage 1's regimes)
        reg = stage1[apath]['regimes'][f"{D0:g}"]
        S = np.full(A.shape, np.nan)
        for q in reg['cells']:
            i = int(np.argmin(np.abs(ws - q['w_ft'])))
            j = int(np.argmin(np.abs(lrs - np.log10(q['ratio']))))
            S[i, j] = q['sep_from_sealed_pt']
        ax.contour(lw_, lrs, S.T, levels=[1.0], colors=RED, linewidths=1.3,
                   linestyles='--')
        ax.plot([0.0], [-5.0], marker='*', ms=13, mfc='#f2c14e', mec=INK,
                mew=0.8, ls='none', zorder=6)
        Rmean = (vam['resistance_s_per_ft']['converged_mean'] if vam
                 else v['resistance_s_per_ft']['mean'])
        ax.text(0.03, 0.05,
                f"valley: $D_b \\propto w^{{{f['slope']:.2f}}}$\n"
                f"$W_{{tot}}/D_b$ = {Rmean:.0f} s/ft",
                transform=ax.transAxes, fontsize=8, color='#ffffff')
        ax.set_xlim(lw_[0], lw_[-1])
        ax.set_ylim(lrs.min(), lrs.max())
        ax.set_xticks(lw_)
        ax.set_xticklabels([f"{w:g}" for w in ws], fontsize=8)
        ax.set_xlabel('barrier half-width $w$  (ft)', fontsize=9, color=INK)
        ax.set_title(f"$D_0$ = {D0:g} ft$^2$/s   ({note.split('(')[0].strip()})",
                     fontsize=9.5, color=INK)
        if k == 0:
            ax.set_ylabel('reduction ratio  $D_{barrier}/D_0$', fontsize=9,
                          color=INK)
            tk = np.arange(np.ceil(lrs.min()), np.floor(lrs.max()) + 0.1, 1.0)
            ax.set_yticks(tk)
            ax.set_yticklabels([f"$10^{{{int(t)}}}$" for t in tk], fontsize=8)
        _style(ax)
        ax.grid(False)
    cb = fig.colorbar(cf, ax=list(axes), fraction=0.028, pad=0.02)
    cb.set_label('gauge-mean RMSE at gauges 5, 6  (psi)', fontsize=8.5, color=INK)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    handles = [Line2D([], [], color='#ffffff', lw=1.6,
                      label='+10 % of the plane minimum'),
               Line2D([], [], marker='o', ls='none', mfc='#ffffff', mec=INK,
                      label=('best ratio at each $w$ (0.05-decade ladder)'
                             if am else 'best ratio at each $w$')),
               ] + ([Line2D([], [], marker='x', ls='none', color='#c9c9c9',
                            label='the same on the 0.25-decade ladder')]
                    if am else []) + [
               Line2D([], [], color='#ffffff', lw=1.2,
                      label='fitted valley floor'),
               Line2D([], [], color=RED, lw=1.3, ls='--',
                      label='below this the trace is a perfect seal'),
               Line2D([], [], marker='*', ls='none', ms=11, mfc='#f2c14e',
                      mec=INK, label="the manuscript's $w$ = 1 ft, ratio = $10^{-5}$")]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               frameon=False, fontsize=8, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle('The misfit valley is a BAND along constant $w/D_{barrier}$, on a '
                 'converged domain and at every background $D_0$',
                 fontsize=10.5, color=INK, y=1.02)
    p1 = os.path.join(figdir, f"fig_c4_plane_converged_{tag}.png")
    rm.assert_absent([p1])
    fig.savefig(p1, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    written.append((p1, 'the (w, ratio) misfit plane on the converged domain, '
                        'three background D'))

    # ---- figure 2: the invariant ------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.0))
    cmap = plt.get_cmap('cividis')
    ax = axs[0]
    for k, (D0, run, _p, _n) in enumerate(CONVERGED):
        c = cmap(k / 2 * 0.75)
        key = f"{run}:{D0:g}"
        v = an['planes'][key]
        vam = am_plane(key)
        if vam:
            ax.plot([p['w_ft'] for p in vam['per_w']],
                    [p['fine_resistance_s_per_ft'] for p in vam['per_w']],
                    '-o', ms=4.5, lw=1.7, color=c,
                    label=f"$D_0$ = {D0:g}  (0.05-decade ratio ladder)")
            ax.plot([p['w_ft'] for p in vam['per_w']],
                    [p['coarse_resistance_s_per_ft'] for p in vam['per_w']],
                    ':x', ms=5, lw=1.0, color=c, alpha=0.75,
                    label=f"$D_0$ = {D0:g}  (0.25-decade, as published)")
        else:
            ax.plot([p['w_ft'] for p in v['per_w']],
                    [p['resistance_s_per_ft'] for p in v['per_w']], '-o',
                    ms=4.5, lw=1.7, color=c,
                    label=f"$D_0$ = {D0:g}  (converged)")
    if am and am.get('resistance_invariance_converged'):
        inv = am['resistance_invariance_converged']
        Rm = float(np.mean(inv['converged_mean_resistance_s_per_ft']))
        lab = (f"mean over the three planes:\n"
               f"{Rm:.0f} s/ft (0.05-decade ladder)\n"
               f"{np.mean(inv['coarse_mean_resistance_s_per_ft']):.0f} s/ft on "
               "the 0.25-decade one")
    else:
        inv = an['resistance_invariance']['converged']
        Rm = float(np.mean(inv['mean_resistance_s_per_ft']))
        lab = f"mean over the three planes:\n{Rm:.0f} s/ft"
    ax.axhline(Rm, color=INK2, lw=1.0, ls=':')
    ax.set_xscale('log')
    ax.set_xlabel('barrier half-width $w$  (ft)', fontsize=9, color=INK)
    ax.set_ylabel('$W_{tot}/D_{barrier}$ at the best fit  (s/ft)', fontsize=9,
                  color=INK)
    ax.set_title('what the data actually pin', fontsize=9.5, color=INK)
    # AMEND 1b: the annotation used to sit at x = 10.2 in DATA coordinates,
    # i.e. outside the axes, where it collided with the next panel's y-axis and
    # was clipped; the six-entry legend sat on top of the curves.  Both now live
    # inside the axes, in headroom made by the ylim below.  Content unchanged.
    ax.text(0.985, 0.975, lab, transform=ax.transAxes, fontsize=7.5,
            color=INK2, ha='right', va='top')
    ax.legend(fontsize=7, frameon=False, labelcolor=INK, loc='lower left')
    _yvals = [float(v) for ln in ax.get_lines()
              for v in np.asarray(ln.get_ydata(orig=True), dtype=float).ravel()
              if np.isfinite(v)]
    ax.set_ylim(0, 1.55 * max(_yvals + [Rm]))
    ax.grid(True, color='#e6e6e6', lw=0.6)
    _style(ax)

    ax = axs[1]
    dc = an.get('d_curve_pad20000')
    dcam = am_dcurve()
    if dcam:
        # AMEND 1: fit only the padding-converged rows, and draw the excluded
        # one in its own marker. The excluded row is D0 = 4600, which this same
        # report flags NOT CONVERGED even at a 20000 ft pad.
        conv = [r for r in dcam['rows'] if r['padding_converged']]
        bad = [r for r in dcam['rows'] if not r['padding_converged']]
        ax.plot([np.log10(r['D0']) for r in dcam['rows']],
                [r['coarse_log10_ratio_opt'] for r in dcam['rows']], ':x',
                ms=5, lw=1.0, color=INK2, alpha=0.8,
                label='0.5-decade ladder (as published)')
        ax.plot([np.log10(r['D0']) for r in conv],
                [r['fine_log10_ratio_opt'] for r in conv], '-o', ms=4.5,
                lw=1.7, color=ACC, label='0.05-decade ladder, $w$ = 1 ft')
        for r in bad:
            ax.plot([np.log10(r['D0'])], [r['fine_log10_ratio_opt']], 's',
                    ms=6.5, mfc='none', mec='#9a9a9a', mew=1.4, zorder=5)
        if bad:
            ax.plot([], [], 's', ms=6.5, mfc='none', mec='#9a9a9a', mew=1.4,
                    label=(f"$D_0$ = {bad[0]['D0']:g}: not converged\n"
                           "(excluded from the fit)"))
        f = dcam['fit_log10_ratio_vs_log10_D0_8rows_fine']
        xx = np.array([np.log10(min(r['D0'] for r in conv)),
                       np.log10(max(r['D0'] for r in conv))])
        ax.plot(xx, f['slope'] * xx + f['intercept'], '-', color=INK2, lw=1.0)
        ax.text(0.04, 0.10,
                f"slope {f['slope']:.3f} $\\pm$ {f['slope_stderr']:.3f}  "
                f"($n$ = {f['n']} converged rows)\n"
                "(slope $-1$ = the ratio is not a parameter,\n"
                "only $D_{barrier}$ is)", transform=ax.transAxes, fontsize=8,
                color=INK, va='bottom')
    elif dc:
        D0s = np.array([r['D0_ft2_s'] for r in dc['rows']])
        ro = np.array([r['ratio_opt'] for r in dc['rows']])
        ax.plot(np.log10(D0s), np.log10(ro), '-o', ms=4.5, lw=1.7, color=ACC,
                label='$D$ curve, $w$ = 1 ft (0.5-decade ladder)')
        f = dc['fit_log10_ratio_opt_vs_log10_D0']
        xx = np.array([np.log10(D0s).min(), np.log10(D0s).max()])
        ax.plot(xx, f['slope'] * xx + f['intercept'], '-', color=INK2, lw=1.0)
        ax.text(0.04, 0.10,
                f"slope {f['slope']:.3f} $\\pm$ {f['slope_stderr']:.3f}\n"
                "(slope $-1$ = the ratio is not a parameter,\n"
                "only $D_{barrier}$ is)", transform=ax.transAxes, fontsize=8,
                color=INK, va='bottom')
    for k, (D0, run, _p, _n) in enumerate(CONVERGED):
        key = f"{run}:{D0:g}"
        vam = am_plane(key)
        if vam:
            p1 = [p for p in vam['per_w'] if abs(p['w_ft'] - 1.0) < 1e-9]
            if p1:
                ax.plot([np.log10(D0)], [p1[0]['fine_log10_ratio_opt']], 'D',
                        ms=6, mfc='none', mec=RED, mew=1.3, zorder=5)
            continue
        v = an['planes'][key]
        p1w = [p for p in v['per_w'] if abs(p['w_ft'] - 1.0) < 1e-9]
        if p1w:
            ax.plot([np.log10(D0)], [np.log10(p1w[0]['ratio_opt'])], 'D',
                    ms=6, mfc='none', mec=RED, mew=1.3, zorder=5)
    ax.plot([], [], 'D', ms=6, mfc='none', mec=RED, mew=1.3,
            label=('planes, $w$ = 1 ft (same ladder)' if dcam
                   else 'planes, $w$ = 1 ft (0.25-decade ladder)'))
    ax.set_xlabel('$\\log_{10}$  background $D_0$  (ft$^2$/s)', fontsize=9,
                  color=INK)
    ax.set_ylabel('$\\log_{10}$  best-fitting reduction ratio', fontsize=9,
                  color=INK)
    ax.set_title('the "ratio" tracks $1/D_0$', fontsize=9.5, color=INK)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK, loc='upper right')
    ax.grid(True, color='#e6e6e6', lw=0.6)
    _style(ax)

    ax = axs[2]
    panel3_title = 'no trend with $D_0$ over 131x'
    if dcam:
        conv = [r for r in dcam['rows'] if r['padding_converged']]
        bad = [r for r in dcam['rows'] if not r['padding_converged']]
        ax.plot([np.log10(r['D0']) for r in dcam['rows']],
                [r['coarse_resistance_s_per_ft'] for r in dcam['rows']], ':x',
                ms=5, lw=1.0, color=INK2, alpha=0.8,
                label='0.5-decade ladder (as published)')
        ax.plot([np.log10(r['D0']) for r in conv],
                [r['fine_resistance_s_per_ft'] for r in conv], '-o', ms=4.5,
                lw=1.7, color=ACC, label='0.05-decade ladder, $w$ = 1 ft')
        for r in bad:
            ax.plot([np.log10(r['D0'])], [r['fine_resistance_s_per_ft']], 's',
                    ms=6.5, mfc='none', mec='#9a9a9a', mew=1.4, zorder=5)
        if bad:
            ax.plot([], [], 's', ms=6.5, mfc='none', mec='#9a9a9a', mew=1.4,
                    label=(f"$D_0$ = {bad[0]['D0']:g}: not converged\n"
                           "(excluded from the fit)"))
        f = dcam['fit_log10_R_vs_log10_D0_8rows_fine']
        span = dcam['D0_span_factor_8rows']
        sig = abs(f['slope']) / f['slope_stderr'] if f['slope_stderr'] else 0.0
        ax.text(0.04, 0.90,
                f"slope in $\\log_{{10}}$: {f['slope']:+.3f} $\\pm$ "
                f"{f['slope_stderr']:.3f}\n({sig:.1f}$\\sigma$, $n$ = "
                f"{f['n']} converged rows)", transform=ax.transAxes,
                fontsize=8, color=INK, va='top')
        panel3_title = (f"no trend with $D_0$ over {span:.0f}x "
                        f"({f['n']} converged rows)" if sig < 2.0 else
                        f"$W_{{tot}}/D_b$ vs $D_0$ over {span:.0f}x "
                        f"({f['n']} converged rows)")
    elif dc:
        D0s = np.array([r['D0_ft2_s'] for r in dc['rows']])
        R = np.array([r['resistance_s_per_ft'] for r in dc['rows']])
        ax.plot(np.log10(D0s), R, '-o', ms=4.5, lw=1.7, color=ACC,
                label='$D$ curve, $w$ = 1 ft (0.5-decade ladder)')
        dc5 = an.get('d_curve_pad5000')
        if dc5:
            D5 = np.array([r['D0_ft2_s'] for r in dc5['rows']])
            R5 = np.array([r['resistance_s_per_ft'] for r in dc5['rows']])
            ax.plot(np.log10(D5), R5, '--s', ms=3.6, lw=1.1, color=INK2,
                    mfc='white', label='same curve, 5000 ft pad')
        f = dc['fit_log10_resistance_vs_log10_D0']
        ax.text(0.04, 0.90,
                f"slope in $\\log_{{10}}$: {f['slope']:+.3f} $\\pm$ "
                f"{f['slope_stderr']:.3f}", transform=ax.transAxes,
                fontsize=8, color=INK, va='top')
    for k, (D0, run, _p, _n) in enumerate(CONVERGED):
        key = f"{run}:{D0:g}"
        vam = am_plane(key)
        R_ = (vam['resistance_s_per_ft']['converged_mean'] if vam
              else an['planes'][key]['resistance_s_per_ft']['mean'])
        ax.plot([np.log10(D0)], [R_], 'D', ms=6,
                mfc='none', mec=RED, mew=1.3, zorder=5)
    ax.plot([], [], 'D', ms=6, mfc='none', mec=RED, mew=1.3,
            label='plane mean over all $w$')
    ax.set_xlabel('$\\log_{10}$  background $D_0$  (ft$^2$/s)', fontsize=9,
                  color=INK)
    ax.set_ylabel('$W_{tot}/D_{barrier}$ at the best fit  (s/ft)', fontsize=9,
                  color=INK)
    ax.set_title(panel3_title, fontsize=9.5, color=INK)
    ax.set_ylim(0, None)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK, loc='lower right')
    ax.grid(True, color='#e6e6e6', lw=0.6)
    _style(ax)
    fig.suptitle('Neither $w$, nor the reduction ratio, nor $D_{barrier}$ is '
                 'identified on its own -- the series resistance $W_{tot}/D_{barrier}$ is',
                 fontsize=10.5, color=INK, y=1.03)
    p2 = os.path.join(figdir, f"fig_c4_resistance_{tag}.png")
    rm.assert_absent([p2])
    fig.savefig(p2, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    written.append((p2, 'the constrained combination: W/D_barrier against w and '
                        'against D0, converged versus contaminated'))

    # ---- figure 3: the one-dimensional curves and the padding ladder ------
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.0))
    ax = axs[0]
    if dc:
        rows = dc['rows']
        norm = plt.Normalize(np.log10(min(r['D0_ft2_s'] for r in rows)),
                             np.log10(max(r['D0_ft2_s'] for r in rows)))
        cm = plt.get_cmap('cividis')
        for r in rows:
            c = cm(norm(np.log10(r['D0_ft2_s'])) * 0.85)
            ax.plot(r['log10_ratios'], r['misfit_by_ratio_psi'], '-', lw=1.6,
                    color=c)
            j = int(np.argmin(r['misfit_by_ratio_psi']))
            ax.plot([r['log10_ratios'][j]], [r['misfit_by_ratio_psi'][j]], 'o',
                    ms=4, mfc='white', mec=c, mew=1.2)
            if r['D0_ft2_s'] in (35.0, 140.0, 4600.0):
                ax.annotate(f"$D_0$ = {r['D0_ft2_s']:g}",
                            (r['log10_ratios'][0], r['misfit_by_ratio_psi'][0]),
                            textcoords='offset points', xytext=(-5, 3),
                            ha='right', fontsize=7.5, color=INK, va='bottom')
        ax.set_xlabel('$\\log_{10}$  reduction ratio', fontsize=9, color=INK)
        ax.set_ylabel('gauge-mean RMSE at gauges 5, 6  (psi)', fontsize=9,
                      color=INK)
        ax.set_title('misfit vs ratio, one curve per $D_0$\n(20000 ft pad, '
                     '$w$ = 1 ft)', fontsize=9.5, color=INK)
        ax.grid(True, color='#e6e6e6', lw=0.6)
    _style(ax)

    ax = axs[1]
    if dc:
        D0s = np.array([r['D0_ft2_s'] for r in dc['rows']])
        pr = np.array(dc['profiled_over_ratio_min_psi'])
        ax.plot(np.log10(D0s), pr, '-o', ms=4.5, lw=1.8, color=ACC,
                label='ratio re-optimised (20000 ft pad)')
        if dcam:
            for r in dcam['rows']:
                if r['padding_converged']:
                    continue
                j = int(np.argmin(np.abs(D0s - r['D0'])))
                ax.plot([np.log10(D0s[j])], [pr[j]], 's', ms=7.5, mfc='none',
                        mec='#9a9a9a', mew=1.4, zorder=6,
                        label=f"$D_0$ = {r['D0']:g}: not converged, not quoted")
        dc5 = an.get('d_curve_pad5000')
        if dc5:
            ax.plot(np.log10([r['D0_ft2_s'] for r in dc5['rows']]),
                    dc5['profiled_over_ratio_min_psi'], '--s', ms=3.6, lw=1.1,
                    color=INK2, mfc='white', label='the same, 5000 ft pad')
        ax.axhline(1.1 * pr.min(), color=RED, lw=1.0, ls=':')
        ax.text(np.log10(D0s[0]), 1.1 * pr.min(), '  +10 % of the minimum',
                fontsize=7.5, color=RED, ha='left', va='bottom')
        b = dc['band_10pct_in_log10_D0']
        ax.axvspan(b['lo'], b['hi'], color=ACC, alpha=0.10, lw=0)
        ax.set_xlabel('$\\log_{10}$  background $D_0$  (ft$^2$/s)', fontsize=9,
                      color=INK)
        ax.set_ylabel('gauge-mean RMSE at gauges 5, 6  (psi)', fontsize=9,
                      color=INK)
        ax.set_title('misfit vs $D_0$, barrier re-fitted at each $D_0$',
                     fontsize=9.5, color=INK)
        ax.legend(fontsize=7.5, frameon=False, labelcolor=INK, loc='upper left')
        ax.grid(True, color='#e6e6e6', lw=0.6)
    _style(ax)

    ax = axs[2]
    pl = an.get('pad_ladder')
    if pl:
        by = {}
        for r in pl['rows']:
            if r['pad20000_psi'] is None or r['pad5000_psi'] is None:
                continue
            by.setdefault(r['D0'], []).append(r)
        norm = plt.Normalize(np.log10(min(by)), np.log10(max(by)))
        cm = plt.get_cmap('cividis')
        for D0, rr in sorted(by.items()):
            c = cm(norm(np.log10(D0)) * 0.85)
            bar = [r for r in rr if 1e-8 < r['ratio'] < 1.0]
            for r in bar:
                ax.plot([5000, 20000, 40000],
                        [r['pad5000_psi'], r['pad20000_psi'], r['pad40000_psi']],
                        '-o', ms=4, lw=1.4, color=c, alpha=0.9)
            if bar:
                top = max(bar, key=lambda q: q['pad40000_psi'])
                ax.annotate(f"$D_0$ = {D0:g}", (40000, top['pad40000_psi']),
                            textcoords='offset points', xytext=(4, 0),
                            fontsize=7.5, color=INK, va='center')
        ax.set_xscale('log')
        ax.set_xticks([5000, 20000, 40000])
        ax.set_xticklabels(['5000', '20000', '40000'], fontsize=8)
        ax.set_xticks([], minor=True)          # the log locator's minor labels
        ax.set_xlim(4300, 52000)               # collide with the three majors
        ax.set_xlabel('padding at each end of the domain  (ft)', fontsize=9,
                      color=INK)
        ax.set_ylabel('gauge-mean RMSE at gauges 5, 6  (psi)', fontsize=9,
                      color=INK)
        ax.set_title('the padding ladder\n(ratios $10^{-3}$, $10^{-4}$, '
                     '$10^{-5}$ at $w$ = 1 ft)', fontsize=9.5, color=INK)
        ax.grid(True, color='#e6e6e6', lw=0.6)
    _style(ax)
    fig.suptitle('The one-dimensional curves, and why the 5000 ft domain B2 '
                 'settled is not enough above $D_0 \\approx 140$',
                 fontsize=10.5, color=INK, y=1.03)
    p3 = os.path.join(figdir, f"fig_c4_dcurve_converged_{tag}.png")
    rm.assert_absent([p3])
    fig.savefig(p3, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    written.append((p3, '1-D misfit curves in ratio and in D0 on the converged '
                        'domain, and the 5000/20000/40000 ft padding ladder'))

    prov = {'kind': 'figures_no_solve', 'study_id': STUDY_ID, 'tag': tag,
            'generated_utc': utcnow(), 'dpi': dpi,
            'code_sha256': rm.sha256_file(os.path.abspath(__file__)),
            'analysis_json': rm._rel(ap),
            'analysis_sha256': rm.sha256_file(ap),
            'amend1_analysis_json': rm._rel(am_path) if am_path else None,
            'amend1_analysis_sha256':
                rm.sha256_file(am_path) if am_path else None,
            'amend1_changes': ([
                'both D-curve panels fit only the padding-converged rows; the '
                'excluded D0 = 4600 point is drawn in an open grey marker and '
                'labelled',
                'the third panel title is built from the fit that is drawn, so '
                'the span and the printed uncertainty agree',
                'per-w optima, the resistance and the plane annotations are the '
                'CONVERGED 0.05-decade values; the published 0.25-decade ones '
                'are kept alongside as light crosses',
                'the 5000 ft pad per-w curves were dropped from panel 1 of the '
                'resistance figure to make room for that comparison; the pad '
                'contamination is still shown in the D-curve figure and in the '
                'report',
                'AMEND 1b (layout only, no number changes): panel 1 of the '
                'resistance figure drew its mean-resistance annotation at '
                'x = 10.2 in DATA coordinates, outside the axes, where it was '
                'clipped and overprinted the next panel s y-axis, and its '
                'six-entry legend covered the curves. Both are now inside the '
                'axes and the y limit carries headroom for them.',
            ] if am else None),
            'stage1_analysis_sha256': {p: rm.sha256_file(
                os.path.join(_ROOT, p)) for p in stage1},
            'figures': [{'path': rm._rel(q[0]), 'sha256': rm.sha256_file(q[0]),
                         'note': q[1]} for q in written]}
    pp = os.path.join(figdir, f"figures_provenance_stage2_{tag}.json")
    rm.assert_absent([pp])
    with open(pp, 'w') as fh:
        json.dump(prov, fh, indent=1, sort_keys=True)
    for q in written:
        print(f"[C4.2] figure -> {q[0]}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', default='analyse', choices=('analyse', 'figures'))
    ap.add_argument('--outdir', default=OUT_ROOT)
    ap.add_argument('--tag', default='v1')
    ap.add_argument('--analysis-tag', default=None)
    ap.add_argument('--amend-tag', default=None,
                    help='tag of c4_stage2_amend1_<tag>.json; when given, the '
                         'figures plot the CONVERGED 0.05-decade optima and '
                         'fit only the padding-converged D-curve rows')
    a = ap.parse_args(argv)
    if a.mode == 'analyse':
        return run_analyse(a.outdir, a.tag)
    return run_figures(a.outdir, a.tag, a.analysis_tag or a.tag,
                       amend_tag=a.amend_tag)


if __name__ == '__main__':
    raise SystemExit(main())
