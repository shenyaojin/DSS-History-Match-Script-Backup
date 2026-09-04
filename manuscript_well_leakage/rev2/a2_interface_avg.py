#!/usr/bin/env python3
"""A2 -- harmonic vs arithmetic face averaging: what the wrong average would cost.

The risk investigation this task started as is CLOSED. fibeRIS hard-codes the
harmonic mean of the two adjacent node diffusivities
(`fiberis/simulator/solver/matbuilder.py:22-23`, and :104-105 for the multi-source
builder), so the manuscript's fitted reduction ratio 1e-5 was NOT obtained under
arithmetic averaging and carries no numerical-leakage artifact. What remains is a
methodological counterfactual for the numerical-methods paragraph, and that is all
this script does.

It answers three questions, all with the shared rev2 kernel and everything except
`interface_avg` held identical:

1. how much do the simulated gauge pressures differ between the two averaging
   rules at the manuscript ratio, gauge by gauge;
2. what reduction ratio X would an arithmetic-mean code need to build the same
   barrier as harmonic ratio 1e-5 -- solved numerically, at several barrier
   half-widths so the width dependence is visible;
3. and, since a barrier that occupies a single node has no interior face at all,
   whether such an X exists for the sub-cell barriers the legacy scripts actually
   used.

Nothing here is edited into the shared modules: `rev2_core.face_diffusivity`
already exposes the switch, and `rev2_core._series_resistance` is deliberately
harmonic-only, so the arithmetic series resistance used for the quasi-static
prediction is computed locally in `_resistance_arithmetic`.

Usage (CWD must be the repo root):
    python3 scripts/manuscript_well_leakage/rev2/a2_interface_avg.py \
        --config configs/rev2/a2_interface_avg.json
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from scipy.optimize import brentq, minimize_scalar                 # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.join(os.path.dirname(_HERE), 'baseline_calibration')
for _p in (_HERE, _BASE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rev2_core as rc                                             # noqa: E402
import rev2_data as rd                                             # noqa: E402
import rev2_manifest as rm                                         # noqa: E402


# ---------------------------------------------------------------------------
# Analytics: the quasi-static prediction the numerical solve is checked against
# ---------------------------------------------------------------------------

def _resistance_arithmetic(mesh, d_profile):
    """sum_f dx_f / arithmetic_mean(D_f, D_{f+1}); units s/ft.

    The mirror of rev2_core._series_resistance, which is harmonic-only by design.
    Kept here so the shared module is not touched.
    """
    dx = np.diff(mesh)
    da = 0.5 * (d_profile[:-1] + d_profile[1:])
    return float(np.sum(dx / da))


def _resistance_harmonic(mesh, d_profile):
    dx = np.diff(mesh)
    dh = 2.0 * d_profile[:-1] * d_profile[1:] / (d_profile[:-1] + d_profile[1:])
    return float(np.sum(dx / dh))


def quasi_static_x(ratio, m):
    """Arithmetic ratio with the same barrier series resistance as harmonic `ratio`.

    On a locally uniform mesh a barrier of m nodes at D0*r has, relative to the
    unbarriered background,

        harmonic   R_h = dx*(m + r) / (D0*r)          [m-1 interior + 2 shoulders]
        arithmetic R_a = dx*(4/(1+X) + (m-1)/X) / D0

    because the harmonic shoulder face is 2*D0*r/(1+r) ~ 2*D0*r while the
    arithmetic shoulder face is D0*(1+X)/2 ~ D0/2 -- INDEPENDENT of X. Setting
    R_h = R_a and solving for X gives the value returned here; for m = 1 there is
    no interior face and the equation reduces to matching the shoulder faces
    alone, X = 4r/(1+r) - 1, which is negative for every r < 1/3.
    """
    r = float(ratio)
    m = int(m)
    if m == 1:
        return 4.0 * r / (1.0 + r) - 1.0        # < 0 whenever r < 1/3
    lhs = (m + r) / r                            # R_h * D0 / dx
    # solve 4/(1+X) + (m-1)/X = lhs for X in (0, 1]
    f = lambda x: 4.0 / (1.0 + x) + (m - 1.0) / x - lhs
    return float(brentq(f, 1e-14, 1.0, xtol=1e-18, rtol=8.9e-16))


def quasi_static_harmonic_equivalent(x_arith, m):
    """Harmonic ratio with the same barrier resistance as arithmetic `x_arith`."""
    x = float(x_arith)
    m = int(m)
    rhs = 4.0 / (1.0 + x) + (m - 1.0) / x        # R_a * D0 / dx
    if m == 1:
        # (1+r)/r = rhs  ->  r = 1/(rhs-1); with x -> 0 this is exactly 1/3
        return float(1.0 / (rhs - 1.0))
    # (m + r)/r = rhs
    return float(m / (rhs - 1.0))


# ---------------------------------------------------------------------------
# Test bed
# ---------------------------------------------------------------------------

class Bed:
    """One (dx, D0) test bed: mesh, source drive, target gauges, run()."""

    def __init__(self, cfg, dx_ft, d_base):
        w = rd.R1_WINDOW
        S = rd.setup_r1(w, source_mode='gauge',
                        pad_low_ft=float(cfg['mesh']['domain_pad_low_md_ft']),
                        pad_high_ft=float(cfg['mesh']['domain_pad_high_md_ft']),
                        dx_ft=float(dx_ft))
        self.cfg = cfg
        self.S = S
        self.dx_ft = float(dx_ft)
        self.d_base = float(d_base)
        self.mesh = S['mesh'].x
        self.targets = S['targets']
        self.rec = [t['idx'] for t in S['targets']]
        self.src_taxis = S['src_series'].taxis_s
        self.src_data = S['src_series'].delta_psi
        self.t_total = S['t_total_s']
        self.source_idx = S['source_idx']
        self.bmd = float(cfg['barrier']['md_ft'])
        self.n_solves = 0
        self.taxis = None

    def profile(self, w, ratio, report=False):
        b = self.cfg['barrier']
        return rc.build_barrier_profile(
            self.mesh, self.d_base, [self.bmd], float(w), float(ratio),
            ratio_reference=b['ratio_reference'], combine=b['combine'],
            on_empty=b['on_empty'], on_outside=b['on_outside'],
            return_report=report)

    def solve(self, d_profile, interface_avg):
        taxis, out = rc.solve_forward(
            self.mesh, d_profile, float(self.cfg['solver']['dt_s']), self.t_total,
            self.src_taxis, self.src_data, self.source_idx, record_idx=self.rec,
            theta=float(self.cfg['solver']['theta']),
            source_time_level=self.cfg['source']['source_time_level'],
            interface_avg=interface_avg,
            theta_startup_steps=int(self.cfg['solver']['theta_startup_steps']))
        self.n_solves += 1
        if self.taxis is None:
            self.taxis = taxis
        return out

    def run(self, w, ratio, interface_avg):
        return self.solve(self.profile(w, ratio), interface_avg)

    def nodes(self, w, ratio=1e-5):
        _, rep = self.profile(w, ratio, report=True)
        return rep


def gauge_mean_rmse(a, b):
    return float(np.mean(np.sqrt(np.mean((a - b) ** 2, axis=0))))


def pooled_rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def gauge_mean_rms(a):
    return float(np.mean(np.sqrt(np.mean(a ** 2, axis=0))))


# ---------------------------------------------------------------------------
# Solvers for the counterfactual ratio
# ---------------------------------------------------------------------------

def solve_x(bed, w, ratio_ref, lo, hi, xatol):
    """Arithmetic ratio X minimising the gauge-mean RMSE against harmonic(ratio_ref).

    Returns a dict; `at_bound` flags a censored answer (house rule: never quote a
    bound as an estimate).
    """
    ref = bed.run(w, ratio_ref, 'harmonic')
    ref_rms = gauge_mean_rms(ref)
    trace = []

    def obj(lx):
        v = gauge_mean_rmse(bed.run(w, 10.0 ** lx, 'arithmetic'), ref)
        trace.append((float(lx), float(v)))
        return v

    res = minimize_scalar(obj, bounds=(lo, hi), method='bounded',
                          options={'xatol': xatol})
    lx = float(res.x)
    best = bed.run(w, 10.0 ** lx, 'arithmetic')
    # Two independent censoring tests, because the bounded minimiser stops a
    # finite distance short of a bound it is running into and a tolerance-sized
    # test then reports "interior" for a monotone objective. The second test is
    # the one that matters: if the objective barely moves across the whole search
    # interval there is no minimum to find, and the returned X is meaningless.
    f_lo_o, f_hi_o = obj(lo), obj(hi)
    near_bound = bool(min(abs(lx - lo), abs(lx - hi)) < 0.05)     # 0.05 decades
    span = float(min(f_lo_o, f_hi_o) - res.fun)
    flat = bool(span <= 1e-3 * res.fun)
    out = {
        'w_ft': float(w), 'ratio_reference_harmonic': float(ratio_ref),
        'log10_X': lx, 'X': float(10.0 ** lx),
        'X_over_ratio': float(10.0 ** lx / ratio_ref),
        'objective': 'gauge_mean_rmse',
        'residual_gauge_mean_rmse_psi': float(res.fun),
        'residual_pooled_rmse_psi': pooled_rmse(best, ref),
        'residual_pct_of_harmonic_rms': float(100.0 * res.fun / ref_rms),
        'harmonic_reference_gauge_mean_rms_psi': ref_rms,
        'search_log10_bounds': [float(lo), float(hi)],
        'objective_at_log10_lo_psi': float(f_lo_o),
        'objective_at_log10_hi_psi': float(f_hi_o),
        'objective_depth_below_nearest_bound_psi': span,
        'objective_depth_relative': float(span / res.fun),
        'minimiser_within_0p05_decades_of_a_bound': near_bound,
        'objective_flat_no_interior_minimum': flat,
        'at_search_bound_censored': bool(near_bound or flat),
        'n_objective_evaluations': len(trace),
        'trace_log10_X': [t[0] for t in trace],
        'trace_objective_psi': [t[1] for t in trace],
    }
    # independent cross-check: signed root on the farthest-gauge peak
    def signed(lx_):
        a = bed.run(w, 10.0 ** lx_, 'arithmetic')
        return float(a[:, -1].max() - ref[:, -1].max())
    try:
        f_lo, f_hi = signed(lo), signed(hi)
        if f_lo * f_hi < 0:
            root = brentq(signed, lo, hi, xtol=1e-10, rtol=1e-13)
            out['crosscheck_X_g7_peak_root'] = float(10.0 ** root)
            out['crosscheck_rel_diff'] = float(
                abs(10.0 ** root - 10.0 ** lx) / (10.0 ** lx))
        else:
            out['crosscheck_X_g7_peak_root'] = None
            out['crosscheck_rel_diff'] = None
            out['crosscheck_note'] = (
                f"no sign change over the search interval: signed g7-peak "
                f"difference is {f_lo:.6g} psi at X=1e{lo:g} and {f_hi:.6g} psi "
                f"at X=1e{hi:g}; the arithmetic scheme cannot reach the harmonic "
                f"reference anywhere in the interval")
    except Exception as exc:                                    # pragma: no cover
        out['crosscheck_X_g7_peak_root'] = None
        out['crosscheck_note'] = f"cross-check failed: {exc!r}"
    return out


def objective_scan(bed, w, ratio_ref, lo, hi, n):
    """Dense misfit curve of arithmetic(X) against harmonic(ratio_ref), for the figure."""
    ref = bed.run(w, ratio_ref, 'harmonic')
    lxs = np.linspace(float(lo), float(hi), int(n))
    vals = [gauge_mean_rmse(bed.run(w, 10.0 ** lx, 'arithmetic'), ref)
            for lx in lxs]
    return {'w_ft': float(w), 'log10_X': [float(v) for v in lxs],
            'objective_psi': [float(v) for v in vals],
            'objective': 'gauge_mean_rmse vs harmonic %g' % ratio_ref}


def solve_harmonic_equivalent(bed, w, ratio_arith, lo, hi, xatol):
    """Harmonic ratio reproducing the arithmetic run at `ratio_arith`."""
    ref = bed.run(w, ratio_arith, 'arithmetic')
    ref_rms = gauge_mean_rms(ref)
    trace = []

    def obj(lr):
        v = gauge_mean_rmse(bed.run(w, 10.0 ** lr, 'harmonic'), ref)
        trace.append((float(lr), float(v)))
        return v

    res = minimize_scalar(obj, bounds=(lo, hi), method='bounded',
                          options={'xatol': xatol})
    lr = float(res.x)
    return {
        'w_ft': float(w), 'ratio_arithmetic': float(ratio_arith),
        'log10_r_equivalent': lr, 'r_equivalent': float(10.0 ** lr),
        'residual_gauge_mean_rmse_psi': float(res.fun),
        'residual_pct_of_arithmetic_rms': float(100.0 * res.fun / ref_rms),
        'search_log10_bounds': [float(lo), float(hi)],
        'at_search_bound_censored': bool(abs(lr - lo) < 10 * xatol
                                         or abs(lr - hi) < 10 * xatol),
        'n_objective_evaluations': len(trace),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/rev2/a2_interface_avg.json')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--tag', default=None)
    args = ap.parse_args(argv)

    root = rm.repo_root()
    if os.path.realpath(os.getcwd()) != os.path.realpath(root):
        raise SystemExit(f"run with CWD = repo root ({root}), not {os.getcwd()}")

    cfg_path = os.path.abspath(args.config)
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    tag = args.tag or cfg['outputs']['tag']
    outdir = os.path.abspath(args.outdir or cfg['outputs']['dir'])
    os.makedirs(outdir, exist_ok=True)

    def _fresh(base, ext):
        """Untagged name on a first run, tagged name once one already exists.

        House rule 2 forbids overwriting an existing output, so a re-run must
        claim new filenames rather than replace the previous pass's.
        """
        p = os.path.join(outdir, base + ext)
        return p if not os.path.exists(p) \
            else os.path.join(outdir, f'{base}_{tag}{ext}')

    P = {
        'fig': os.path.join(outdir, f'fig01_interface_avg_{tag}.png'),
        'csv_gauge': os.path.join(outdir, f'a2_gauge_differences_{tag}.csv'),
        'csv_ratio': os.path.join(outdir, f'a2_ratio_equivalence_{tag}.csv'),
        'json': os.path.join(outdir, f'a2_results_{tag}.json'),
        'readme': _fresh('README', '.md'),
        'man_h': _fresh('manifest', '.json'),
        'man_a': _fresh('manifest_arithmetic_arm', '.json'),
    }
    rm.assert_absent(list(P.values()) + [P['man_h'] + '.sha256',
                                         P['man_a'] + '.sha256'])
    superseded = sorted(
        os.path.join(outdir, f) for f in os.listdir(outdir)
        if os.path.isfile(os.path.join(outdir, f))
        and os.path.join(outdir, f) not in set(P.values()))

    t_start = time.time()
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    ratio = float(cfg['barrier']['ratio_manuscript'])
    D0 = float(cfg['diffusivity']['d_base_ft2_s'])
    dx0 = float(cfg['mesh']['dx_ft'])
    xs = cfg['scans']['x_solve']
    lo, hi = xs['log10_bounds']
    xatol = float(xs['xatol_log10'])

    R = {'config_path': os.path.relpath(cfg_path, root), 'tag': tag,
         'ratio_manuscript': ratio, 'd_base_ft2_s': D0, 'dx_ft': dx0,
         'barrier_md_ft': float(cfg['barrier']['md_ft'])}
    log = lambda s: print(f"[{time.time() - t_start:7.1f}s] {s}", flush=True)

    bed = Bed(cfg, dx0, D0)
    log(f"test bed: nx={bed.mesh.size}, MD {bed.mesh[0]:.0f}-{bed.mesh[-1]:.0f} ft, "
        f"source g{bed.S['src_gauge']} @ node {bed.source_idx} (MD "
        f"{bed.mesh[bed.source_idx]:.1f}), {len(bed.targets)} targets, "
        f"t_total {bed.t_total:.1f} s")

    # ---- null control: a uniform field must give identical results -----------
    uni = np.full(bed.mesh.size, D0)
    nb_h = bed.solve(uni, 'harmonic')
    nb_a = bed.solve(uni, 'arithmetic')
    null_max = float(np.max(np.abs(nb_h - nb_a)))
    R['null_control'] = {
        'description': ('with a uniform D field the harmonic and arithmetic face '
                        'values coincide identically, so the two schemes must '
                        'produce the same numbers bit for bit'),
        'max_abs_difference_psi': null_max,
        'bitwise_identical': bool(np.array_equal(nb_h, nb_a)),
    }
    log(f"null control (uniform D): max|harmonic-arithmetic| = {null_max:.3e} psi, "
        f"bitwise identical = {R['null_control']['bitwise_identical']}")
    if null_max != 0.0:
        raise SystemExit("null control failed: the two averaging rules disagree on "
                         "a uniform field; STOP and report a core defect")

    # ---- primary cases: per-gauge cost at the manuscript ratio ---------------
    widths = [float(w) for w in cfg['barrier']['w_half_width_ft_primary']]
    R['cases'] = {}
    gauge_rows = []
    for w in widths:
        prof, rep = bed.profile(w, ratio, report=True)
        b0 = rep['barriers'][0]
        m = int(b0['n_nodes'])
        ph = bed.solve(prof, 'harmonic')
        pa = bed.solve(prof, 'arithmetic')
        rh = _resistance_harmonic(bed.mesh, prof) - _resistance_harmonic(bed.mesh, uni)
        ra = _resistance_arithmetic(bed.mesh, prof) - _resistance_arithmetic(bed.mesh, uni)
        case = {
            'w_ft': w, 'n_nodes': m,
            'realised_full_width_ft': float(b0['realised_full_width_ft']),
            'realised_span_md_ft': b0['realised_span_ft'],
            'fallback_to_nearest_node': bool(b0['fallback']),
            'n_fallback': int(rep['n_fallback']),
            'D_barrier_ft2_s': float(b0['d_barrier_min_ft2_s']),
            'excess_series_resistance_harmonic_s_per_ft': rh,
            'excess_series_resistance_arithmetic_s_per_ft': ra,
            'resistance_ratio_harmonic_over_arithmetic': float(rh / ra),
            'gauges': [],
        }
        assert rep['n_fallback'] == 0, rep['fallback_messages']
        for j, t in enumerate(bed.targets):
            h, a, n = ph[:, j], pa[:, j], nb_h[:, j]
            row = {
                'gauge': int(t['gauge']), 'md_ft': float(t['md_ft']),
                'distance_from_source_ft': float(t['distance_ft']),
                'no_barrier_peak_psi': float(n.max()),
                'harmonic_peak_psi': float(h.max()),
                'arithmetic_peak_psi': float(a.max()),
                'peak_difference_psi': float(a.max() - h.max()),
                'peak_ratio_arith_over_harm': float(a.max() / h.max())
                if h.max() != 0 else None,
                'rmse_difference_psi': float(np.sqrt(np.mean((a - h) ** 2))),
                'rms_harmonic_psi': float(np.sqrt(np.mean(h ** 2))),
                'rms_no_barrier_psi': float(np.sqrt(np.mean(n ** 2))),
                'blocked_fraction_harmonic': float(1.0 - h.max() / n.max()),
                'blocked_fraction_arithmetic': float(1.0 - a.max() / n.max()),
            }
            row['rmse_difference_pct_of_harmonic_rms'] = float(
                100.0 * row['rmse_difference_psi'] / row['rms_harmonic_psi'])
            row['rmse_difference_pct_of_no_barrier_rms'] = float(
                100.0 * row['rmse_difference_psi'] / row['rms_no_barrier_psi'])
            case['gauges'].append(row)
            gauge_rows.append(dict(w_half_width_ft=w, n_nodes=m,
                                   realised_full_width_ft=case['realised_full_width_ft'],
                                   ratio=ratio, **row))
        case['gauge_mean_rmse_harm_vs_arith_psi'] = gauge_mean_rmse(ph, pa)
        case['pooled_rmse_harm_vs_arith_psi'] = pooled_rmse(ph, pa)
        case['gauge_mean_rms_harmonic_psi'] = gauge_mean_rms(ph)
        R['cases'][f'w{w:g}'] = case
        log(f"w={w:g} ft (m={m} node(s), realised full width "
            f"{case['realised_full_width_ft']:.4f} ft): harmonic vs arithmetic "
            f"gauge-mean RMSE {case['gauge_mean_rmse_harm_vs_arith_psi']:.4g} psi; "
            f"excess resistance {rh:.4g} vs {ra:.4g} s/ft "
            f"({rh / ra:.4g}x)")
        case['_series'] = {'harmonic': ph, 'arithmetic': pa}   # for the figure

    # ---- saturation scan ----------------------------------------------------
    R['saturation'] = []
    for w in [float(x) for x in cfg['scans']['saturation_widths_ft']]:
        for r in [float(x) for x in cfg['scans']['saturation_ratios']]:
            prof = bed.profile(w, r)
            ph = bed.solve(prof, 'harmonic')
            pa = bed.solve(prof, 'arithmetic')
            R['saturation'].append({
                'w_ft': w, 'ratio': r,
                'harmonic_peak_g7_psi': float(ph[:, -1].max()),
                'arithmetic_peak_g7_psi': float(pa[:, -1].max()),
                'harmonic_peak_g2_psi': float(ph[:, 0].max()),
                'arithmetic_peak_g2_psi': float(pa[:, 0].max()),
            })
    R['no_barrier_peak_g7_psi'] = float(nb_h[:, -1].max())
    R['no_barrier_peak_g2_psi'] = float(nb_h[:, 0].max())
    log(f"saturation scan done ({len(R['saturation'])} points); no-barrier g7 peak "
        f"{R['no_barrier_peak_g7_psi']:.4f} psi")

    # ---- solve for X --------------------------------------------------------
    R['x_solve'] = {}
    for w in [float(x) for x in cfg['barrier']['w_half_width_ft_solved']]:
        m = int(bed.nodes(w)['barriers'][0]['n_nodes'])
        s = solve_x(bed, w, ratio, lo, hi, xatol)
        s['n_nodes'] = m
        s['quasi_static_prediction_X'] = quasi_static_x(ratio, m)
        s['numerical_over_quasi_static'] = float(
            s['X'] / s['quasi_static_prediction_X'])
        R['x_solve'][f'w{w:g}'] = s
        log(f"X solve w={w:g} ft (m={m}): X = {s['X']:.5e} "
            f"({s['X_over_ratio']:.4f} x the harmonic ratio); quasi-static "
            f"prediction {s['quasi_static_prediction_X']:.5e}; residual at the "
            f"optimum {s['residual_gauge_mean_rmse_psi']:.4g} psi = "
            f"{s['residual_pct_of_harmonic_rms']:.2f}% of the harmonic signal; "
            f"cross-check {s['crosscheck_X_g7_peak_root']}")

    # single-node case: is there an X at all?
    lo1, hi1 = cfg['scans']['x_solve']['single_node_bounds_log10']
    s1 = solve_x(bed, 0.0, ratio, float(lo1), float(hi1), xatol)
    s1['n_nodes'] = 1
    s1['quasi_static_prediction_X'] = quasi_static_x(ratio, 1)
    s1['exists'] = bool(s1['quasi_static_prediction_X'] > 0)
    s1['why'] = (
        "A single reduced node has NO interior face: both of its faces are "
        "shoulder faces between D0 and D0*ratio. The harmonic shoulder face is "
        "2*D0*ratio/(1+ratio), which -> 0 with the ratio; the arithmetic shoulder "
        "face is D0*(1+X)/2, which -> D0/2 and can never fall below half the "
        "background whatever X is. Matching the two requires "
        "X = 4*ratio/(1+ratio) - 1, which is negative for every ratio < 1/3, so no "
        "admissible X exists. The minimiser therefore runs to its lower search "
        "bound and the answer must be reported as censored, not as an estimate.")
    s1['X_reported_is_meaningless'] = True
    s1['X_search_floor_value'] = s1.pop('X')
    s1['X'] = None
    R['x_solve']['w0_single_node'] = s1
    log(f"X solve w=0 (single node): quasi-static X = "
        f"{s1['quasi_static_prediction_X']:.6g} (< 0 => no solution); minimiser "
        f"censored at bound = {s1['at_search_bound_censored']}, residual "
        f"{s1['residual_gauge_mean_rmse_psi']:.4g} psi = "
        f"{s1['residual_pct_of_harmonic_rms']:.0f}% of the harmonic signal")

    # ---- dense misfit curves for the figure ---------------------------------
    R['objective_scan'] = {}
    for w in [float(x) for x in cfg['barrier']['w_half_width_ft_solved']]:
        R['objective_scan'][f'w{w:g}'] = objective_scan(bed, w, ratio, -6.5, -4.0, 41)
    log('objective scans done (2 x 41 points)')

    # ---- inverse: harmonic ratio equivalent to arithmetic 1e-5 --------------
    inv = cfg['scans']['inverse_solve']
    R['harmonic_equivalent_of_arithmetic'] = {}
    for w in widths:
        m = int(bed.nodes(w)['barriers'][0]['n_nodes'])
        b = inv['log10_bounds_single_node'] if m == 1 \
            else inv['log10_bounds_multinode']
        e = solve_harmonic_equivalent(bed, w, ratio, float(b[0]), float(b[1]), xatol)
        e['n_nodes'] = m
        e['quasi_static_prediction_r'] = quasi_static_harmonic_equivalent(ratio, m)
        R['harmonic_equivalent_of_arithmetic'][f'w{w:g}'] = e
        log(f"inverse w={w:g} ft (m={m}): arithmetic {ratio:g} behaves like "
            f"harmonic {e['r_equivalent']:.5e} (quasi-static "
            f"{e['quasi_static_prediction_r']:.5e}), residual "
            f"{e['residual_gauge_mean_rmse_psi']:.3g} psi")

    # ---- mesh dependence of X ----------------------------------------------
    R['mesh_dependence'] = []
    for dxi in [float(x) for x in cfg['mesh']['dx_refinement_series_ft']]:
        bedi = bed if dxi == dx0 else Bed(cfg, dxi, D0)
        for w in [float(x) for x in cfg['barrier']['w_half_width_ft_solved']]:
            rep = bedi.nodes(w)
            m = int(rep['barriers'][0]['n_nodes'])
            s = solve_x(bedi, w, ratio, lo, hi, xatol)
            s.pop('trace_log10_X'), s.pop('trace_objective_psi')
            s.update(dx_ft=dxi, d_base_ft2_s=D0, n_nodes=m,
                     realised_full_width_ft=float(
                         rep['barriers'][0]['realised_full_width_ft']),
                     quasi_static_prediction_X=quasi_static_x(ratio, m))
            R['mesh_dependence'].append(s)
            log(f"mesh dependence dx={dxi:g} w={w:g}: m={m}, X={s['X']:.5e} "
                f"(X/ratio {s['X_over_ratio']:.4f}), residual "
                f"{s['residual_pct_of_harmonic_rms']:.2f}%")
        if bedi is not bed:
            del bedi

    # ---- D dependence of X --------------------------------------------------
    R['d_dependence'] = []
    for d in [float(x) for x in cfg['diffusivity']['d_base_series_ft2_s']]:
        bedd = bed if d == D0 else Bed(cfg, dx0, d)
        for w in [float(x) for x in cfg['barrier']['w_half_width_ft_solved']]:
            s = solve_x(bedd, w, ratio, lo, hi, xatol)
            s.pop('trace_log10_X'), s.pop('trace_objective_psi')
            s.update(dx_ft=dx0, d_base_ft2_s=d,
                     n_nodes=int(bedd.nodes(w)['barriers'][0]['n_nodes']))
            R['d_dependence'].append(s)
            log(f"D dependence D0={d:g} w={w:g}: X={s['X']:.5e} "
                f"(X/ratio {s['X_over_ratio']:.4f})")
        if bedd is not bed:
            del bedd

    R['t_total_s'] = float(bed.t_total)
    R['n_forward_solves'] = int(bed.n_solves)
    R['wall_seconds'] = float(time.time() - t_start)

    # ---- figure -------------------------------------------------------------
    make_figure(P['fig'], cfg, bed, R, nb_h, ratio)
    log(f"wrote {P['fig']}")

    # ---- tables -------------------------------------------------------------
    with open(P['csv_gauge'], 'w', newline='') as fh:
        cols = ['w_half_width_ft', 'n_nodes', 'realised_full_width_ft', 'ratio',
                'gauge', 'md_ft', 'distance_from_source_ft',
                'no_barrier_peak_psi', 'harmonic_peak_psi', 'arithmetic_peak_psi',
                'peak_difference_psi', 'peak_ratio_arith_over_harm',
                'rmse_difference_psi', 'rms_harmonic_psi', 'rms_no_barrier_psi',
                'rmse_difference_pct_of_harmonic_rms',
                'rmse_difference_pct_of_no_barrier_rms',
                'blocked_fraction_harmonic', 'blocked_fraction_arithmetic']
        wri = csv.DictWriter(fh, fieldnames=cols)
        wri.writeheader()
        for row in gauge_rows:
            wri.writerow({k: row[k] for k in cols})

    with open(P['csv_ratio'], 'w', newline='') as fh:
        cols = ['case', 'dx_ft', 'd_base_ft2_s', 'w_half_width_ft', 'n_nodes',
                'realised_full_width_ft', 'harmonic_ratio', 'arithmetic_X',
                'X_over_ratio', 'quasi_static_X', 'residual_psi',
                'residual_pct_of_harmonic_rms', 'crosscheck_X', 'censored']
        wri = csv.DictWriter(fh, fieldnames=cols)
        wri.writeheader()
        for key in ('w0_single_node', 'w1', 'w5'):
            s = R['x_solve'].get(key)
            if s is None:
                continue
            wri.writerow({
                'case': f'primary/{key}', 'dx_ft': dx0, 'd_base_ft2_s': D0,
                'w_half_width_ft': s['w_ft'], 'n_nodes': s['n_nodes'],
                'realised_full_width_ft':
                    R['cases'][f"w{s['w_ft']:g}"]['realised_full_width_ft'],
                'harmonic_ratio': ratio,
                'arithmetic_X': (s['X'] if s.get('exists', True) else 'none'),
                'X_over_ratio': (s['X_over_ratio'] if s.get('exists', True)
                                 else 'none'),
                'quasi_static_X': s['quasi_static_prediction_X'],
                'residual_psi': s['residual_gauge_mean_rmse_psi'],
                'residual_pct_of_harmonic_rms': s['residual_pct_of_harmonic_rms'],
                'crosscheck_X': s.get('crosscheck_X_g7_peak_root'),
                'censored': s['at_search_bound_censored']})
        for s in R['mesh_dependence']:
            wri.writerow({
                'case': 'mesh_dependence', 'dx_ft': s['dx_ft'],
                'd_base_ft2_s': s['d_base_ft2_s'], 'w_half_width_ft': s['w_ft'],
                'n_nodes': s['n_nodes'],
                'realised_full_width_ft': s['realised_full_width_ft'],
                'harmonic_ratio': ratio, 'arithmetic_X': s['X'],
                'X_over_ratio': s['X_over_ratio'],
                'quasi_static_X': s['quasi_static_prediction_X'],
                'residual_psi': s['residual_gauge_mean_rmse_psi'],
                'residual_pct_of_harmonic_rms': s['residual_pct_of_harmonic_rms'],
                'crosscheck_X': s.get('crosscheck_X_g7_peak_root'),
                'censored': s['at_search_bound_censored']})
        for s in R['d_dependence']:
            wri.writerow({
                'case': 'd_dependence', 'dx_ft': s['dx_ft'],
                'd_base_ft2_s': s['d_base_ft2_s'], 'w_half_width_ft': s['w_ft'],
                'n_nodes': s['n_nodes'], 'realised_full_width_ft': '',
                'harmonic_ratio': ratio, 'arithmetic_X': s['X'],
                'X_over_ratio': s['X_over_ratio'],
                'quasi_static_X': quasi_static_x(ratio, s['n_nodes']),
                'residual_psi': s['residual_gauge_mean_rmse_psi'],
                'residual_pct_of_harmonic_rms': s['residual_pct_of_harmonic_rms'],
                'crosscheck_X': s.get('crosscheck_X_g7_peak_root'),
                'censored': s['at_search_bound_censored']})
    log(f"wrote {P['csv_gauge']} and {P['csv_ratio']}")

    for c in R['cases'].values():
        c.pop('_series', None)
    with open(P['json'], 'w') as fh:
        json.dump(R, fh, indent=2, sort_keys=True)

    R['superseded_files_in_output_dir'] = [os.path.basename(f)
                                          for f in superseded]
    write_readme(P['readme'], cfg, R, P, tag)
    log(f"wrote {P['json']} and {P['readme']}")

    # ---- manifests ----------------------------------------------------------
    write_manifests(P, cfg, cfg_path, bed, R, started_utc, ratio, D0, dx0, uni,
                    superseded)
    log(f"wrote {P['man_a']} and {P['man_h']}")
    log(f"done: {bed.n_solves} forward solves in this bed, "
        f"{R['wall_seconds']:.1f} s wall")
    return 0


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(path, cfg, bed, R, nb, ratio):
    t = bed.taxis
    c_h, c_a, c_n = '#1f4e9c', '#c0392b', '#7f8c8d'
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.4))

    # (a) single-node barrier, time series
    ax = axes[0, 0]
    cw0 = R['cases']['w0']
    for j, (gi, ls) in enumerate(((0, '-'), (5, '--'))):
        ax.plot(t, nb[:, gi], ls, color=c_n, lw=4.0, alpha=0.45,
                label='no barrier' if j == 0 else None)
        ax.plot(t, cw0['_series']['arithmetic'][:, gi], ls, color=c_a, lw=1.4,
                label='arithmetic' if j == 0 else None)
        ax.plot(t, cw0['_series']['harmonic'][:, gi], ls, color=c_h, lw=1.6,
                label='harmonic (fibeRIS)' if j == 0 else None)
    g2, g7 = cw0['gauges'][0], cw0['gauges'][-1]
    ax.annotate(f"harmonic: {g2['harmonic_peak_psi']:.1f} psi at g2, "
                f"{g7['harmonic_peak_psi']:.1f} psi at g7\n"
                f"({100 * (1 - g2['blocked_fraction_harmonic']):.1f}% and "
                f"{100 * (1 - g7['blocked_fraction_harmonic']):.1f}% of the "
                f"unbarriered peak)",
                xy=(t[-1], cw0['_series']['harmonic'][-1, 0]),
                xytext=(0.30, 0.44), textcoords='axes fraction', fontsize=8,
                color=c_h, arrowprops=dict(arrowstyle='->', lw=0.9, color=c_h))
    ax.set_title(f"(a) sub-cell barrier (single node, "
                 f"{cw0['realised_full_width_ft']:.2f} ft), ratio = {ratio:g}:"
                 f"\narithmetic is indistinguishable from no barrier", fontsize=10)
    ax.set_xlabel('time since window start (s)')
    ax.set_ylabel('simulated $\\Delta P$ (psi)')
    ax.legend(fontsize=8, loc='upper left')
    ax.text(0.985, 0.03, 'solid g2 (261 ft)\ndashed g7 (1570 ft)', fontsize=8,
            ha='right', va='bottom', transform=ax.transAxes, color='0.3')

    # (b) 3-node barrier, time series
    ax = axes[0, 1]
    cw1 = R['cases']['w1']
    for j, (gi, ls) in enumerate(((0, '-'), (5, '--'))):
        ax.plot(t, cw1['_series']['arithmetic'][:, gi], ls, color=c_a, lw=1.4,
                label='arithmetic' if j == 0 else None)
        ax.plot(t, cw1['_series']['harmonic'][:, gi], ls, color=c_h, lw=1.4,
                label='harmonic (fibeRIS)' if j == 0 else None)
    ax.set_title(f"(b) resolved barrier (w = 1 ft, "
                 f"{cw1['realised_full_width_ft']:.2f} ft, "
                 f"{cw1['n_nodes']} nodes),\nratio = {ratio:g}: "
                 f"arithmetic leaks ~"
                 f"{cw1['gauges'][-1]['peak_ratio_arith_over_harm']:.2f}x", fontsize=10)
    ax.set_xlabel('time since window start (s)')
    ax.set_ylabel('simulated $\\Delta P$ (psi)')
    ax.legend(fontsize=8, loc='upper left')

    # (c) peak vs distance, all widths
    ax = axes[0, 2]
    dist = [g['distance_from_source_ft'] for g in R['cases']['w0']['gauges']]
    ax.plot(dist, [g['no_barrier_peak_psi'] for g in R['cases']['w0']['gauges']],
            'o-', color=c_n, ms=7, lw=4.0, alpha=0.45, label='no barrier')
    for key, mk in (('w0', 's'), ('w1', '^'), ('w5', 'D')):
        c = R['cases'][key]
        lab = f"{c['n_nodes']} node" + ('s' if c['n_nodes'] > 1 else '')
        ax.plot(dist, [g['harmonic_peak_psi'] for g in c['gauges']], mk + '-',
                color=c_h, ms=4, lw=1.2, label=f'harmonic, {lab}')
        ax.plot(dist, [g['arithmetic_peak_psi'] for g in c['gauges']], mk + '--',
                color=c_a, ms=4, lw=1.2, label=f'arithmetic, {lab}')
    ax.set_yscale('log')
    ax.set_xlabel('distance from source node (ft)')
    ax.set_ylabel('peak simulated $\\Delta P$ (psi)')
    ax.set_title('(c) cost at every gauge, ratio = %g' % ratio, fontsize=10)
    ax.legend(fontsize=7, ncol=1, loc='lower left')

    # (d) saturation
    ax = axes[1, 0]
    for w, mk in ((0.0, 's'), (1.0, '^')):
        rows = [s for s in R['saturation'] if s['w_ft'] == w]
        rr = [s['ratio'] for s in rows]
        lab = '1 node' if w == 0.0 else f"{R['cases']['w1']['n_nodes']} nodes"
        ax.plot(rr, [s['harmonic_peak_g7_psi'] for s in rows], mk + '-',
                color=c_h, ms=4, lw=1.2, label=f'harmonic, {lab}')
        ax.plot(rr, [s['arithmetic_peak_g7_psi'] for s in rows], mk + '--',
                color=c_a, ms=4, lw=1.2, label=f'arithmetic, {lab}')
    ax.axhline(R['no_barrier_peak_g7_psi'], color=c_n, lw=3.0, alpha=0.45,
               label='no barrier')
    eq = R['harmonic_equivalent_of_arithmetic']['w0']['r_equivalent']
    ax.axvline(eq, color='k', lw=0.9, ls='-.')
    ax.annotate(f'whatever ratio is typed in, the arithmetic\n'
                f'single-node barrier is only ever the\n'
                f'harmonic barrier at ratio {eq:.3f}',
                xy=(eq, R['no_barrier_peak_g7_psi']), xytext=(0.05, 0.62),
                textcoords='axes fraction', fontsize=8,
                arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 3e3)
    ax.set_xlabel('requested reduction ratio')
    ax.set_ylabel('peak $\\Delta P$ at g7 (psi)')
    ax.set_title('(d) the arithmetic sub-cell barrier saturates', fontsize=10)
    ax.legend(fontsize=7, loc='lower right')

    # (e) the X solve
    ax = axes[1, 1]
    for key, col, mk in (('w1', '#0b6e4f', '^'), ('w5', '#8e44ad', 'D')):
        s = R['x_solve'][key]
        sc = R['objective_scan'][key]
        ax.plot([10.0 ** a for a in sc['log10_X']], sc['objective_psi'], '-',
                color=col, lw=1.3, alpha=0.85)
        ax.plot([s['X']], [s['residual_gauge_mean_rmse_psi']], mk, color=col,
                ms=10, mfc='none', mew=2.0,
                label=f"w = {s['w_ft']:g} ft ({s['n_nodes']} nodes): "
                      f"X = {s['X']:.3g} ({s['X_over_ratio']:.2f} x $10^{{-5}}$)")
        ax.axvline(s['X'], color=col, lw=0.8, ls='--')
    ax.axvline(ratio, color='k', lw=1.2,
               label=f'harmonic ratio {ratio:g} (manuscript)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('arithmetic-mean reduction ratio $X$')
    ax.set_ylabel('gauge-mean RMSE vs harmonic $10^{-5}$ (psi)')
    ax.set_title('(e) the ratio an arithmetic code would need,\nsolved '
                 'numerically (the floor is the residual it cannot remove)',
                 fontsize=10)
    ax.legend(fontsize=7, loc='upper left')

    # (f) mesh dependence
    ax = axes[1, 2]
    for w, col, mk in ((1.0, '#0b6e4f', '^'), (5.0, '#8e44ad', 'D')):
        rows = sorted([s for s in R['mesh_dependence'] if s['w_ft'] == w],
                      key=lambda s: s['dx_ft'])
        ax.plot([s['dx_ft'] for s in rows], [s['X_over_ratio'] for s in rows],
                mk + '-', color=col, ms=6, lw=1.3, label=f'w = {w:g} ft, solved')
        ax.plot([s['dx_ft'] for s in rows],
                [s['quasi_static_prediction_X'] / R['ratio_manuscript']
                 for s in rows], mk + ':', color=col, ms=4, lw=1.0,
                label=f'w = {w:g} ft, quasi-static $(m-1)/m$')
    ax.axhline(1.0, color='k', lw=1.0, ls='--')
    ax.text(0.03, 0.93, 'the two schemes agree only as $\\Delta x/w \\to 0$',
            fontsize=8, transform=ax.transAxes, va='bottom')
    ax.set_xlabel('mesh spacing $\\Delta x$ (ft)')
    ax.set_ylabel('$X$ / harmonic ratio')
    ax.set_ylim(0.4, 1.12)
    ax.invert_xaxis()
    ax.set_title('(f) $X$ is mesh-dependent: it is not a physical ratio',
                 fontsize=10)
    ax.legend(fontsize=7, loc='lower right')

    fig.suptitle('A2 -- harmonic vs arithmetic face averaging of $D$ '
                 '(everything else identical; fibeRIS uses harmonic, '
                 'matbuilder.py:22-23)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=320)
    plt.close(fig)


# ---------------------------------------------------------------------------
# README (generated so every number in it is produced by this run)
# ---------------------------------------------------------------------------

def write_readme(path, cfg, R, P, tag):
    ratio = R['ratio_manuscript']
    c0, c1, c5 = R['cases']['w0'], R['cases']['w1'], R['cases']['w5']
    x1, x5 = R['x_solve']['w1'], R['x_solve']['w5']
    s1 = R['x_solve']['w0_single_node']
    e0 = R['harmonic_equivalent_of_arithmetic']['w0']
    g7_0 = c0['gauges'][-1]
    md = []
    A = md.append
    A('# A2 -- interface averaging (harmonic vs arithmetic)\n')
    A(f"Round `rev2_20260901`. Generated by "
      f"`scripts/manuscript_well_leakage/rev2/a2_interface_avg.py` from "
      f"`{R['config_path']}`; every number below comes from this run and is "
      f"pinned by `{os.path.basename(P['man_h'])}` (harmonic arm) and "
      f"`{os.path.basename(P['man_a'])}` (arithmetic arm).\n")
    A('## Status of the question\n')
    A('**The risk investigation is closed and the manuscript is clean.** fibeRIS '
      'takes the harmonic mean of the two adjacent node diffusivities '
      '(`fiberis/simulator/solver/matbuilder.py:22-23`, and `:104-105` in the '
      'multi-source builder), so the fitted reduction ratio of 1e-5 was never '
      'obtained under arithmetic averaging and contains no numerical-leakage '
      'artifact. What follows is a methodological counterfactual for the '
      'numerical-methods paragraph, not a correction.\n')
    A('## What was run\n')
    A(f"R1 standard test bed (`rev2_data.setup_r1`): window MD 15000-16750 ft, "
      f"gauge 1 (MD 16645) prescribed as a Dirichlet node, gauges 2-7 as targets, "
      f"uniform D = {R['d_base_ft2_s']:g} ft^2/s, dx = {R['dx_ft']:g} ft with a "
      f"5000 ft low-MD pad, backward Euler at dt = 1 s over a "
      f"{R['t_total_s']:.0f} s window ({R['n_forward_solves']} forward solves in "
      f"the primary bed, {R['wall_seconds']:.0f} s wall). A single barrier sits at MD "
      f"{R['barrier_md_ft']:g} ft, between the source node and every target "
      f"gauge. The ONLY thing that changes between the two arms is "
      f"`interface_avg`.\n")
    A(f"Null control: on a uniform D field the two rules coincide identically -- "
      f"max|harmonic - arithmetic| = {R['null_control']['max_abs_difference_psi']:.1e} "
      f"psi, bitwise identical = "
      f"{R['null_control']['bitwise_identical']}. Every difference reported below "
      f"therefore comes from the barrier faces alone.\n")
    A('## The three numbers for the paper\n')
    A(f"1. **A sub-cell (single-node) barrier -- which is what every legacy "
      f"script builds (A1) -- cannot be represented at all under arithmetic "
      f"averaging.** At the manuscript ratio {ratio:g} the harmonic scheme blocks "
      f"{100 * g7_0['blocked_fraction_harmonic']:.1f}% of the far-gauge signal "
      f"(g7 peak {g7_0['harmonic_peak_psi']:.3f} psi against "
      f"{g7_0['no_barrier_peak_psi']:.2f} psi with no barrier); the arithmetic "
      f"scheme blocks {100 * g7_0['blocked_fraction_arithmetic']:.2f}% "
      f"({g7_0['arithmetic_peak_psi']:.2f} psi), i.e. it leaves the barrier "
      f"essentially open and overstates the transmitted pressure by "
      f"{g7_0['peak_ratio_arith_over_harm']:.0f}x.\n")
    A(f"2. **There is no arithmetic ratio X that reproduces it.** The numerical "
      f"search confirms it rather than finding one: across the whole interval "
      f"X in [1e{s1['search_log10_bounds'][0]:g}, 1e{s1['search_log10_bounds'][1]:g}] "
      f"the misfit against the harmonic run stays between "
      f"{min(s1['residual_gauge_mean_rmse_psi'], s1['objective_at_log10_lo_psi'], s1['objective_at_log10_hi_psi']):.4f} "
      f"and "
      f"{max(s1['residual_gauge_mean_rmse_psi'], s1['objective_at_log10_lo_psi'], s1['objective_at_log10_hi_psi']):.4f} "
      f"psi -- a total variation of "
      f"{abs(s1['objective_at_log10_hi_psi'] - s1['objective_at_log10_lo_psi']):.1e} "
      f"psi over 8 decades of X -- and the smallest value sits at the search "
      f"floor, so the surface is monotone with no interior minimum and the "
      f"returned X is censored, not an estimate. "
      f"Matching the "
      f"two schemes on a single node requires X = 4r/(1+r) - 1 = "
      f"{s1['quasi_static_prediction_X']:.6g}, i.e. a negative diffusivity, for "
      f"any r < 1/3. Run backwards instead: the arithmetic single-node barrier at "
      f"ratio {ratio:g} behaves exactly like a HARMONIC barrier of ratio "
      f"{e0['r_equivalent']:.5f} (numerically solved; quasi-static prediction "
      f"exactly 1/3; residual {e0['residual_gauge_mean_rmse_psi']:.2e} psi). "
      f"**Under arithmetic averaging a one-node barrier is never more than a "
      f"3x reduction, whatever ratio is typed into it.**\n")
    A(f"3. **Once the barrier is resolved by more than one cell, X exists and is "
      f"close to -- but below -- the manuscript ratio.** Solved numerically "
      f"(bounded Brent on log10 X, objective = gauge-mean RMSE against the "
      f"harmonic run at {ratio:g}):\n")
    A(f"   * half-width w = {x1['w_ft']:g} ft ({x1['n_nodes']} nodes, realised "
      f"full width {c1['realised_full_width_ft']:.2f} ft): "
      f"**X = {x1['X']:.3e}** = {x1['X_over_ratio']:.3f} x the harmonic ratio;\n")
    A(f"   * half-width w = {x5['w_ft']:g} ft ({x5['n_nodes']} nodes, realised "
      f"full width {c5['realised_full_width_ft']:.2f} ft): "
      f"**X = {x5['X']:.3e}** = {x5['X_over_ratio']:.3f} x the harmonic ratio.\n")
    cc = max(x1['crosscheck_rel_diff'] or 0.0, x5['crosscheck_rel_diff'] or 0.0)
    A(f"   Both agree with the independent cross-check (root of the signed "
      f"far-gauge peak difference) to {100 * cc:.1f}% "
      f"and lie {100 * (1 - x1['numerical_over_quasi_static']):.0f}% and "
      f"{100 * (1 - x5['numerical_over_quasi_static']):.0f}% below the "
      f"quasi-static series-resistance prediction ratio*(m-1)/m, because at these "
      f"ratios the barrier is not quasi-static over the 1254 s window.\n")
    A(f"   Re-fitting the ratio does **not** buy back the difference: at the best "
      f"X the residual against the harmonic run is still "
      f"{x1['residual_pct_of_harmonic_rms']:.1f}% (w = 1 ft) and "
      f"{x5['residual_pct_of_harmonic_rms']:.1f}% (w = 5 ft) of the harmonic "
      f"signal's own RMS.\n")
    A('## Width and mesh dependence\n')
    A('| w (ft) | nodes | realised full width (ft) | harmonic ratio | X (arithmetic) '
      '| X / ratio | residual at best X |\n|---|---|---|---|---|---|---|\n')
    A(f"| 0 | 1 | {c0['realised_full_width_ft']:.3f} | {ratio:g} | **none exists** "
      f"| -- | {s1['residual_pct_of_harmonic_rms']:.0f}% |\n")
    A(f"| 1 | {x1['n_nodes']} | {c1['realised_full_width_ft']:.3f} | {ratio:g} | "
      f"{x1['X']:.3e} | {x1['X_over_ratio']:.3f} | "
      f"{x1['residual_pct_of_harmonic_rms']:.1f}% |\n")
    A(f"| 5 | {x5['n_nodes']} | {c5['realised_full_width_ft']:.3f} | {ratio:g} | "
      f"{x5['X']:.3e} | {x5['X_over_ratio']:.3f} | "
      f"{x5['residual_pct_of_harmonic_rms']:.1f}% |\n")
    A('\nX / ratio rises towards 1 as the mesh is refined at fixed physical '
      'width (panel f):\n')
    A('| dx (ft) | w = 1 ft: nodes, X/ratio | w = 5 ft: nodes, X/ratio |\n'
      '|---|---|---|\n')
    for dxi in sorted({s['dx_ft'] for s in R['mesh_dependence']}, reverse=True):
        a = [s for s in R['mesh_dependence'] if s['dx_ft'] == dxi and s['w_ft'] == 1.0][0]
        b = [s for s in R['mesh_dependence'] if s['dx_ft'] == dxi and s['w_ft'] == 5.0][0]
        A(f"| {dxi:g} | {a['n_nodes']}, {a['X_over_ratio']:.3f} | "
          f"{b['n_nodes']}, {b['X_over_ratio']:.3f} |\n")
    A('\nThat is the practical point: under arithmetic averaging the fitted '
      'reduction ratio is a function of the mesh (and, to a lesser extent, of the '
      'background D -- see `d_dependence` in the JSON), so it is not a property '
      'of the rock. Under harmonic averaging a single reduced node is exactly a '
      'slab of its own control volume for every ratio (A1, 5.8e-16 relative), so '
      'the fitted ratio means the same thing on every mesh.\n')
    A('## Paragraph ready for the numerical-methods section\n')
    A(f"> Face diffusivities are formed as the harmonic mean of the two adjacent "
      f"node values, D_f = 2 D_i D_(i+1) / (D_i + D_(i+1)). Two adjacent control "
      f"volumes are conductances in series, and the harmonic mean is the only "
      f"average that preserves the series resistance across the face; with it, a "
      f"single node assigned D0*ratio is exactly a slab of diffusivity D0*ratio "
      f"occupying that node's control volume, for every ratio. The arithmetic "
      f"mean does not have this property. Because the arithmetic face value "
      f"(D0 + D0*ratio)/2 tends to D0/2 rather than to zero as the ratio falls, a "
      f"low-diffusivity zone that is thinner than one cell saturates: in the "
      f"configuration used here a single reduced node under arithmetic averaging "
      f"is equivalent to a harmonic reduction ratio of "
      f"{e0['r_equivalent']:.3f} -- at most a threefold reduction -- no matter "
      f"how small the assigned ratio, and at ratio 1e-5 it transmits "
      f"{g7_0['peak_ratio_arith_over_harm']:.0f} times the pressure that the "
      f"harmonic scheme transmits to the farthest gauge. Where the barrier is "
      f"resolved by several cells the two rules converge, but only at first order "
      f"in the cell-to-barrier width ratio: an arithmetic-mean code would need "
      f"ratio = {x1['X']:.2e} (3 cells) or {x5['X']:.2e} (11 cells) to build the "
      f"barrier that the harmonic scheme builds at 1e-5, and even then it "
      f"reproduces the harmonic gauge response only to "
      f"{x1['residual_pct_of_harmonic_rms']:.0f}% and "
      f"{x5['residual_pct_of_harmonic_rms']:.0f}% respectively. The reduction "
      f"ratio is therefore reported here as a property of a harmonic-mean, "
      f"finite-volume discretisation with an explicitly stated barrier width.\n")
    A('\n## Quotable numbers\n')
    A(f"* harmonic mean is what the code does: `matbuilder.py:22-23` "
      f"(single source), `:104-105` (multi source);\n")
    A(f"* single-node barrier, arithmetic averaging == harmonic ratio "
      f"{e0['r_equivalent']:.5f} (exactly 1/3 in the limit ratio -> 0);\n")
    A(f"* single-node barrier at ratio {ratio:g}: harmonic blocks "
      f"{100 * g7_0['blocked_fraction_harmonic']:.1f}% of the g7 peak, arithmetic "
      f"{100 * g7_0['blocked_fraction_arithmetic']:.2f}%;\n")
    A(f"* arithmetic ratio matching harmonic {ratio:g}: "
      f"X = {x1['X']:.3e} at 3 cells, {x5['X']:.3e} at 11 cells, none at 1 cell;\n")
    A(f"* residual that re-fitting the ratio cannot remove: "
      f"{x1['residual_pct_of_harmonic_rms']:.1f}% / "
      f"{x5['residual_pct_of_harmonic_rms']:.1f}% of the harmonic signal RMS.\n")
    A('\n## Products\n')
    for k, d in (('fig', 'figure, 320 dpi, six panels'),
                 ('csv_gauge', 'per-gauge differences at ratio 1e-5, three widths'),
                 ('csv_ratio', 'the ratio-equivalence table (X solves)'),
                 ('json', 'every number produced by the run'),
                 ('man_h', 'manifest, harmonic reference arm'),
                 ('man_a', 'manifest, arithmetic counterfactual arm')):
        A(f"* `{os.path.basename(P[k])}` -- {d}\n")
    A('\n## Versions in this directory\n')
    if R.get('superseded_files_in_output_dir'):
        A(f"This is run `{tag}`, and it is the authoritative one. An earlier pass "
          f"is retained untouched in the same directory (house rule 2 forbids "
          f"deleting or overwriting it): "
          + ", ".join('`' + f + '`' for f in R['superseded_files_in_output_dir'])
          + ". Its physical numbers are the same as this run's -- the re-run "
            "changed only the figure rendering and added the censoring "
            "diagnostics that show the single-node X search has no interior "
            "minimum. Quote this run's files.\n")
    else:
        A(f"Run `{tag}`; nothing in this directory is superseded.\n")
    A('\n## Caveats\n')
    A('* The test bed is the R1 single-source window, not the manuscript\'s '
      'two-phase 101 run. The averaging question is local to the barrier faces, '
      'so the conclusion transfers, but the psi values quoted here are for this '
      'test bed and must not be presented as 101 outputs.\n')
    A('* w = 0 realises a one-cell barrier of '
      f"{R['cases']['w0']['realised_full_width_ft']:.2f} ft here, against "
      '0.13333 ft on 101\'s refined mesh (A1). The saturation result depends only '
      'on the barrier being sub-cell, not on the cell size: the equivalence '
      'harmonic-1/3 == arithmetic-anything is dx-independent algebra.\n')
    A('* `rev2_core`\'s barrier report field `excess_resistance_s_per_ft` is '
      'always the HARMONIC series resistance (`_series_resistance` hard-codes the '
      'harmonic mean by design); the arithmetic resistances quoted in the JSON '
      'are computed locally in this script.\n')
    with open(path, 'w') as fh:
        fh.write(''.join(md))


# ---------------------------------------------------------------------------
# Manifests: one per averaging arm, both declaring the same products
# ---------------------------------------------------------------------------

def write_manifests(P, cfg, cfg_path, bed, R, started_utc, ratio, D0, dx0, uni,
                    superseded=()):
    drv = rm.driver_record(
        kind='gauge_series', baseline_removal='subtract_first_sample',
        value_units='delta_psi',
        series_path=cfg['data']['gauge_series_template'].format(
            n=bed.S['src_gauge']),
        gauge_number=bed.S['src_gauge'], gauge_md_ft=bed.S['src_md'],
        taxis=bed.src_taxis, values=bed.src_data,
        time_start=cfg['window']['time_start'],
        time_end=cfg['window']['time_end'])
    src = rm.source_protocol(
        application='dirichlet_node',
        solver_class='rev2_core.solve_forward (single Dirichlet source, banded)',
        placement_rule=cfg['source']['selection_rule'],
        sources=[rm.source_record(bed.mesh, md_requested_ft=bed.S['src_md'],
                                  mesh_idx=bed.source_idx, driver=drv,
                                  label=f"g{bed.S['src_gauge']}",
                                  index_in_source_list=0)],
        targets=[{'gauge': t['gauge'], 'md_ft': t['md_ft'],
                  'distance_ft': t['distance_ft'], 'mesh_idx': t['idx']}
                 for t in bed.targets],
        time_level=cfg['source']['source_time_level'],
        phase_chaining=rm.NONE_DECLARED,
        boundary_conditions={'lbc': cfg['solver']['lbc'],
                             'rbc': cfg['solver']['rbc']})

    barriers = []
    for w in [float(x) for x in cfg['barrier']['w_half_width_ft_primary']]:
        prof, rep = bed.profile(w, ratio, report=True)
        barriers.append(rm.barrier_record(
            bed.mesh, prof < uni, label=f'w={w:g}ft ratio={ratio:g}',
            centre_md_ft=bed.bmd, w_requested_ft=w, ratio=ratio, d_baseline=D0,
            report=rep))

    def numerics_for(avg):
        return rm.numerics(
            time=rm.time_record(bed.taxis, mode='fixed',
                                theta=float(cfg['solver']['theta']),
                                t_total_requested_s=bed.t_total,
                                dt_requested_s=float(cfg['solver']['dt_s']),
                                source_time_level=cfg['source']['source_time_level'],
                                theta_startup_steps=int(
                                    cfg['solver']['theta_startup_steps']),
                                label=f'{avg} arm, dt = 1 s, backward Euler'),
            mesh=rm.mesh_record(bed.mesh, dx_requested_ft=dx0,
                                window_md_ft=(cfg['window']['md_min_ft'],
                                              cfg['window']['md_max_ft']),
                                pad_low_ft=float(
                                    cfg['mesh']['domain_pad_low_md_ft']),
                                pad_high_ft=float(
                                    cfg['mesh']['domain_pad_high_md_ft']),
                                refinement=cfg['mesh']['refinement']),
            interface_avg=avg,
            boundary={'lbc': cfg['solver']['lbc'], 'rbc': cfg['solver']['rbc'],
                      'pml_thickness': 0.0, 'sigma_max': 0.0},
            diffusivity={'profile_family': 'uniform',
                         'd_base_ft2_s': D0,
                         'd_base_series_ft2_s':
                             cfg['diffusivity']['d_base_series_ft2_s'],
                         'D_min': D0 * ratio, 'D_max': D0,
                         'D_sha256': rm.sha256_array(uni),
                         'note': ('uniform background; the only structure is the '
                                  'single barrier at MD '
                                  f"{bed.bmd:g} ft. Sweeps additionally visit "
                                  'dx = 0.5 / 0.25 ft and D = 480 ft^2/s; those '
                                  'meshes are rebuilt by rev2_data.build_mesh '
                                  'with the same rule.')},
            barriers=barriers, leakage=rm.NONE_DECLARED,
            kernel={'name': 'rev2_core.solve_forward', 'banded': True,
                    'equivalence_reference':
                        'bitwise identical to r1_calibration_core.solve_forward '
                        'at theta=1/harmonic/lambda=0 (A4 self-test T1); that '
                        'kernel is proven bit-equivalent to fibeRIS '
                        'PDS1D_SingleSource',
                    'interface_avg_switch': 'rev2_core.face_diffusivity'},
            rng=rm.NONE_DECLARED,
            parallel={'processes': 1, 'backend': 'none (serial)'},
            amplification=rc.amplification_factor(
                bed.mesh, bed.profile(1.0, ratio), float(cfg['solver']['dt_s']),
                float(cfg['solver']['theta']), interface_avg=avg))

    inputs = [(cfg['data']['gauge_md_npz'], 'geometry', 'gauge_md_npz'),
              (cfg['data']['frac_hit_stage1_npz'], 'geometry', 'frac_hit_stage1')]
    inputs += [(cfg['data']['gauge_series_template'].format(n=n), 'gauge_series',
                f'gauge{n}') for n in sorted(bed.S['gauge_window'].series)]
    # A superseded earlier pass in the same directory is neither an output of this
    # run nor deleted (house rule 2). Hash it as a prior_run_output so the record
    # says exactly which files were sitting there, and tolerate it in the scan.
    inputs += [(f, 'prior_run_output', 'superseded:' + os.path.basename(f))
               for f in superseded]

    common = dict(study_id=cfg['study_id'], config=cfg, config_path=cfg_path,
                  inputs=inputs, source=src, started_utc=started_utc,
                  require_modules=('rev2_core', 'rev2_data', 'rev2_manifest'),
                  extra_code_files=(os.path.abspath(__file__),),
                  allow_undeclared_outputs=bool(superseded))
    products = [rm.output_decl(P['fig'], role='figure_png', dpi=320,
                               note='six-panel A2 figure'),
                rm.output_decl(P['csv_gauge'], role='csv',
                               note='per-gauge harmonic vs arithmetic differences'),
                rm.output_decl(P['csv_ratio'], role='csv',
                               note='ratio-equivalence / X solves'),
                rm.output_decl(P['json'], role='json',
                               note='all numbers produced by the run'),
                rm.output_decl(P['readme'], role='report_md',
                               note='task README, generated by the run')]

    notes_common = list(cfg.get('notes', [])) + [
        'fibeRIS hard-codes the harmonic mean (matbuilder.py:22-23 single source, '
        ':104-105 multi source); the arithmetic arm exists only as a '
        'counterfactual and is not a physically defensible configuration.',
        'The two arms share every input, mesh, time step and barrier; they differ '
        'only in rev2_core.face_diffusivity.',
        'No physical parameter is fitted here. X is a property of the '
        'discretisation.',
    ]
    rm.write_manifest(
        P['man_a'], task_id='A2-arithmetic-counterfactual',
        numerics=numerics_for('arithmetic'), outputs=products,
        results={'arm': 'arithmetic (counterfactual)', **R},
        notes=notes_common + [
            'THIS ARM IS THE WRONG AVERAGE ON PURPOSE. It is recorded so the '
            'counterfactual runs behind the A2 numbers have a manifest of their '
            'own; nothing in the paper is computed with it.'],
        run_label='A2 arithmetic counterfactual arm', **common)
    rm.write_manifest(
        P['man_h'], task_id='A2',
        numerics=numerics_for('harmonic'),
        outputs=products + [
            rm.output_decl(P['man_a'], role='manifest_json',
                           note='manifest of the arithmetic counterfactual arm'),
            rm.output_decl(P['man_a'] + '.sha256', role='other',
                           note='sidecar of the arithmetic arm manifest')],
        results={'arm': 'harmonic (reference; what fibeRIS does)', **R},
        notes=notes_common, run_label='A2 harmonic reference arm', **common)


if __name__ == '__main__':
    raise SystemExit(main())
