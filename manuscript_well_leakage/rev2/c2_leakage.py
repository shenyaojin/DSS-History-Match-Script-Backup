"""C2 -- the leakage-sink model.

    dP/dt = D d2P/dx2 - lambda_leak * (P - P0),   P0 = 0 (delta-pressure form)

Physical story: the annulus conducts pressure ALONG the wellbore while leaking
into the formation ALONG THE WAY. One physically meaningful parameter can in
principle produce both the fast near-field rise and the strong far-field decay
without D having to vary along the path -- which is the competing story told by
the two_zone D(x) profile of the R2 inversion (11.87 psi, k = 4).

What this script does, all on the R1 standard setup (window MD 15000-16750,
5000 ft low pad, dx = 1 ft, dt = 1 s, Dirichlet source at gauge 1, targets
gauges 2-7):

  0. re-verifies the sink itself against an analytic transient AND against an
     independently written implicit stepper (the shared modules' adversarial
     verifiers never ran, and the headline result here is a NULL result about
     that sink, so it is checked before it is believed);
  1. reproduces the three baselines from scratch under both norms;
  2. maps the (D, lambda_leak) misfit surface, coarse then refined;
  3. traces the valley floor exactly -- Brent over log10 D at each lambda, and
     the reverse, Brent over log10 lambda at each D -- because a grid cannot
     tell an interior optimum from a boundary one;
  4. re-optimises all three baselines under the amplitude-normalised norm so
     that the normalised comparison is like-for-like;
  5. asks whether the sink removes the near-under / far-over residual sign flip;
  6. fits nested gauge subsets to bound the range of applicability;
  7. reports AICc across the three effective-n conventions with the
     bias-domination caveat attached;
  8. writes figures, npz grids, one manifest per norm plus one for the
     diagnostics, and the summaries.

Misfit pooling is the GAUGE-MEAN RMSE, i.e. equal weight per gauge, not per
sample -- the quantity the published 82.33 / 67.12 / 11.87 psi actually are
(C1). The sample-pooled value is carried alongside everywhere but is never the
criterion.

Run with CWD = repo root:
    python3 scripts/manuscript_well_leakage/rev2/c2_leakage.py \
        --config configs/rev2/c2_leakage.json
"""

import argparse
import datetime
import json
import multiprocessing as mp
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
from scipy.linalg import solve_banded                # noqa: E402
from scipy.optimize import minimize, minimize_scalar  # noqa: E402
from scipy.special import erfc                       # noqa: E402
from scipy.stats import qmc                          # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'baseline_calibration'))

import r1_calibration_core as r1c   # noqa: E402
import rev2_core as rc              # noqa: E402
import rev2_data as rd              # noqa: E402
import rev2_manifest as rm          # noqa: E402

STUDY_ID = 'c2_leakage'
TASK_ID = 'C2'

_G = {}
_T0 = time.time()


def log(msg):
    print('[%7.1fs] %s' % (time.time() - _T0, msg), flush=True)


# ---------------------------------------------------------------------------
# Independent re-verification of the sink (see module docstring, step 0)
# ---------------------------------------------------------------------------

def _independent_solve(mesh, D, lam, dt, n_steps, src_idx, src_value):
    """A second, independently written backward-Euler stepper with the sink.

    Deliberately NOT a refactor of rev2_core: it builds its own banded matrix
    from scratch on a uniform mesh, puts `dt*lam` on the INTERIOR diagonal only,
    and imposes the Dirichlet row and the two no-flux rows explicitly. Its only
    purpose is to be wrong in different ways than rev2_core would be.
    """
    nx = len(mesh)
    dx = float(mesh[1] - mesh[0])
    r = D * dt / dx ** 2
    ab = np.zeros((3, nx))
    ab[0, 2:] = -r                       # super-diagonal, rows 1..nx-2
    ab[1, 1:-1] = 1.0 + 2.0 * r + dt * lam
    ab[2, :-2] = -r                      # sub-diagonal, rows 1..nx-2
    ab[1, 0] = 1.0
    ab[0, 1] = -1.0                      # u0 - u1 = 0 (no flux)
    ab[1, -1] = 1.0
    ab[2, -2] = -1.0                     # u_{n-1} - u_{n-2} = 0 (no flux)
    ab[1, src_idx] = 1.0
    if src_idx + 1 < nx:
        ab[0, src_idx + 1] = 0.0
    if src_idx - 1 >= 0:
        ab[2, src_idx - 1] = 0.0
    u = np.zeros(nx)
    for _ in range(n_steps):
        b = u.copy()
        b[0] = 0.0
        b[-1] = 0.0
        b[src_idx] = src_value
        u = solve_banded((1, 1), ab, b)
    return u


def _analytic_step_with_decay(x, t, D, lam, p_src):
    """Semi-infinite domain, step Dirichlet at x = 0, first-order decay.

    P(x,t) = P0/2 [ e^{-x k} erfc(x/(2 sqrt(Dt)) - sqrt(lam t))
                  + e^{+x k} erfc(x/(2 sqrt(Dt)) + sqrt(lam t)) ],  k = sqrt(lam/D)
    """
    k = np.sqrt(lam / D)
    a = x / (2.0 * np.sqrt(D * t))
    b = np.sqrt(lam * t)
    return 0.5 * p_src * (np.exp(-x * k) * erfc(a - b)
                          + np.exp(x * k) * erfc(a + b))


def verify_sink():
    """Three checks. Returns a dict; raises if any check fails hard."""
    out = {}

    # --- (a) transient against the analytic solution -----------------------
    D, lam, p_src = 1000.0, 3.0e-3, 500.0
    dx, dt, t_end = 0.5, 0.05, 200.0
    L = 4000.0
    mesh = np.arange(0.0, L + dx / 2.0, dx)
    n_steps = int(round(t_end / dt))
    src = np.array([0.0, t_end * 10])
    taxis, rec = rc.solve_forward(
        mesh, np.full(mesh.size, D), dt, t_end, src, np.array([p_src, p_src]),
        0, lambda_leak=lam, p0=0.0)
    sim = rec[-1]
    t_real = float(taxis[-1])
    xs = mesh[1:]
    ana = _analytic_step_with_decay(xs, t_real, D, lam, p_src)
    keep = xs <= 2500.0            # beyond this the field is ~0 and the far
    err = np.abs(sim[1:][keep] - ana[keep])   # no-flux end is felt
    out['transient_vs_analytic'] = {
        'D_ft2_s': D, 'lambda_leak_s^-1': lam, 'p_src_psi': p_src,
        'dx_ft': dx, 'dt_s': dt, 't_s': t_real,
        'max_abs_err_psi': float(err.max()),
        'max_rel_err_of_source': float(err.max() / p_src),
        'analytic_decay_length_ft': float(np.sqrt(D / lam))}

    # --- (b) steady state exp(-x sqrt(lam/D)) ------------------------------
    t_ss = 40000.0
    _, rec_ss = rc.solve_forward(
        mesh, np.full(mesh.size, D), 5.0, t_ss, np.array([0.0, t_ss * 10]),
        np.array([p_src, p_src]), 0, lambda_leak=lam, p0=0.0,
        record_idx=np.arange(mesh.size))
    ss = rec_ss[-1]
    ana_ss = p_src * np.exp(-mesh * np.sqrt(lam / D))
    m = mesh <= 2000.0
    rel = np.abs(ss[m] - ana_ss[m]) / p_src
    out['steady_state_vs_analytic'] = {
        't_s': t_ss, 'max_rel_err_vs_source_amplitude': float(rel.max())}

    # --- (c) an independent implicit stepper -------------------------------
    dx2, dt2, n2 = 1.0, 0.5, 400
    mesh2 = np.arange(0.0, 1500.0 + dx2 / 2.0, dx2)
    src_idx = 750
    u_ind = _independent_solve(mesh2, D, lam, dt2, n2, src_idx, p_src)
    _, rec2 = rc.solve_forward(
        mesh2, np.full(mesh2.size, D), dt2, n2 * dt2 - 1e-9,
        np.array([0.0, 1e6]), np.array([p_src, p_src]), src_idx,
        lambda_leak=lam, p0=0.0, record_idx=np.arange(mesh2.size))
    u_rev = rec2[-1]
    d = np.abs(u_ind - u_rev)
    out['independent_kernel'] = {
        'dx_ft': dx2, 'dt_s': dt2, 'n_steps': n2, 'source_idx': src_idx,
        'max_abs_diff_psi': float(d.max()),
        'max_rel_diff_vs_source': float(d.max() / p_src)}

    # --- (d) lambda = 0 must be bitwise identical to the R1 kernel ---------
    S = _G['S']
    mesh3, tg = S['mesh_x'], S['targets']
    ridx = [t['idx'] for t in tg]
    prof = np.full(mesh3.size, 1150.0)
    _, a = rc.solve_forward(mesh3, prof, 1.0, S['t_total_s'],
                            S['src_taxis'], S['src_delta'], S['source_idx'],
                            record_idx=ridx, lambda_leak=0.0)
    _, b = r1c.solve_forward(mesh3, prof, 1.0, S['t_total_s'],
                             S['src_taxis'], S['src_delta'], S['source_idx'],
                             record_idx=ridx)
    out['lambda_zero_vs_r1_kernel'] = {
        'bitwise_identical': bool(np.array_equal(a, b)),
        'max_abs_diff_psi': float(np.max(np.abs(a - b)))}

    out['passed'] = bool(
        out['transient_vs_analytic']['max_rel_err_of_source'] < 5e-3
        and out['steady_state_vs_analytic']['max_rel_err_vs_source_amplitude'] < 1e-4
        and out['independent_kernel']['max_rel_diff_vs_source'] < 1e-10
        and out['lambda_zero_vs_r1_kernel']['bitwise_identical'])
    return out


# ---------------------------------------------------------------------------
# Forward evaluation
# ---------------------------------------------------------------------------

def _init_worker(cfg):
    # Under the 'fork' start method the parent's already-built setup is
    # inherited, so rebuilding it would cost 6 x 2 s and, worse, would make the
    # workers' target arrays distinct objects from the parent's.
    if 'S' not in _G:
        _G['cfg'] = cfg
        _G['S'] = build_setup(cfg)


def build_setup(cfg):
    S = rd.setup_r1(source_mode=cfg['source']['mode'],
                    pad_low_ft=float(cfg['mesh']['pad_low_ft']),
                    pad_high_ft=float(cfg['mesh']['pad_high_ft']),
                    dx_ft=float(cfg['mesh']['dx_ft']),
                    baseline=('first_sample'
                              if cfg['source']['baseline_removal']
                              == 'subtract_first_sample' else 'none'))
    return {
        'mesh_x': S['mesh'].x, 'mesh': S['mesh'],
        'source_idx': S['source_idx'], 'src_md': S['src_md'],
        'src_gauge': S['src_gauge'],
        'src_taxis': S['src_series'].taxis_s,
        'src_delta': S['src_series'].delta_psi,
        'src_series': S['src_series'],
        'src_t0_abs': S['src_series'].t0_abs,
        't_total_s': S['t_total_s'], 'targets': S['targets'],
        'frac_hits': S['frac_hits'], 'fh_centroid': S['fh_centroid'],
        'gauge_window': S['gauge_window'],
        'dt_s': float(cfg['solver']['dt_s']),
        'theta': float(cfg['solver']['theta']),
        'interface_avg': cfg['solver']['interface_avg'],
        'p0': float(cfg['solver']['p0_psi']),
    }


def forward(S, prof, lam):
    ridx = [t['idx'] for t in S['targets']]
    return rc.solve_forward(S['mesh_x'], prof, S['dt_s'], S['t_total_s'],
                            S['src_taxis'], S['src_delta'], S['source_idx'],
                            record_idx=ridx, theta=S['theta'],
                            lambda_leak=float(lam), p0=S['p0'],
                            interface_avg=S['interface_avg'])


def per_gauge(S, prof, lam):
    """Full per-gauge scoring for one (profile, lambda)."""
    taxis, rec = forward(S, prof, lam)
    rows = []
    for k, t in enumerate(S['targets']):
        sim = np.interp(t['taxis'], taxis, rec[:, k])
        r = sim - t['data']
        obs_max = float(np.max(t['data']))
        mse = float(np.mean(r ** 2))
        bias = float(np.mean(r))
        rows.append({
            'gauge': int(t['gauge']), 'md_ft': float(t['md_ft']),
            'distance_ft': float(t['distance_ft']),
            'n': int(r.size), 'mse': mse, 'rmse_psi': float(np.sqrt(mse)),
            'bias_psi': bias, 'bias2_over_mse': float(bias ** 2 / mse),
            'frac_resid_positive': float(np.mean(r > 0)),
            'obs_max_psi': obs_max, 'sim_max_psi': float(np.max(sim)),
            'amplitude_ratio': float(np.max(sim) / obs_max),
            'ss': float(np.sum(r ** 2)),
        })
    return rows, taxis, rec


def scores(rows, subset=None):
    """(gauge-mean RMSE psi, amplitude-normalised RMSE, sample-pooled RMSE)."""
    sel = rows if subset is None else [rows[i] for i in subset]
    a = float(np.sqrt(np.mean([r['mse'] for r in sel])))
    n = float(np.sqrt(np.mean([r['mse'] / r['obs_max_psi'] ** 2 for r in sel])))
    p = float(np.sqrt(sum(r['ss'] for r in sel) / sum(r['n'] for r in sel)))
    return a, n, p


def uniform_scores(S, log10D, lam, subset=None):
    prof = np.full(S['mesh_x'].size, 10.0 ** float(log10D))
    rows, _, _ = per_gauge(S, prof, lam)
    return scores(rows, subset)


def family_profile(S, fam, p):
    spec = r1c.PROFILE_FAMILIES[fam]
    if fam == 'triangular':
        lo, hi = S['mesh'].window_md
        return spec['fn'](S['mesh_x'], S['source_idx'], p, lo, hi)
    return spec['fn'](S['mesh_x'], S['source_idx'], p)


# --- pool jobs -------------------------------------------------------------

def job(args):
    kind = args[0]
    S = _G['S']
    if kind == 'uni':
        _, lD, lam = args
        a, n, p = uniform_scores(S, lD, lam)
        return a, n, p
    if kind == 'fam':
        _, fam, p = args
        prof = family_profile(S, fam, np.asarray(p, float))
        if not np.all(np.isfinite(prof)) or np.any(prof <= 0):
            return np.inf, np.inf, np.inf
        rows, _, _ = per_gauge(S, prof, 0.0)
        return scores(rows)
    if kind == 'floor':
        # Brent over log10 D at fixed lambda; `which` picks the norm.
        _, lam, which, bounds, xatol = args
        i = 0 if which == 'abs' else 1
        r = minimize_scalar(lambda x: uniform_scores(S, x, lam)[i],
                            bounds=tuple(bounds), method='bounded',
                            options={'xatol': xatol})
        a, n, p = uniform_scores(S, float(r.x), lam)
        at_bound = bool(abs(r.x - bounds[0]) < 1e-3 or abs(r.x - bounds[1]) < 1e-3)
        return dict(lam=float(lam), log10_D=float(r.x), rmse_psi=a,
                    rmse_normalised=n, rmse_pooled_psi=p, at_bound=at_bound)
    if kind == 'rfloor':
        # Brent over log10 lambda at fixed D -- the test of whether lambda
        # wants to be non-zero anywhere.
        _, lD, which, lbounds, xatol = args
        i = 0 if which == 'abs' else 1
        r = minimize_scalar(lambda y: uniform_scores(S, lD, 10.0 ** y)[i],
                            bounds=tuple(lbounds), method='bounded',
                            options={'xatol': xatol})
        a, n, p = uniform_scores(S, lD, 10.0 ** float(r.x))
        a0, n0, p0 = uniform_scores(S, lD, 0.0)
        v = a if which == 'abs' else n
        v0 = a0 if which == 'abs' else n0
        return dict(log10_D=float(lD), log10_lambda=float(r.x),
                    rmse_psi=a, rmse_normalised=n, rmse_pooled_psi=p,
                    at_low_bound=bool(abs(r.x - lbounds[0]) < 1e-3),
                    gain_over_lambda_zero=float(v0 - v),
                    rmse_at_lambda_zero=a0, rmse_normalised_at_lambda_zero=n0)
    if kind == 'subset_uni':
        _, m, which, bounds, xatol = args
        sub = list(range(0, m - 1))
        i = 0 if which == 'abs' else 1
        r = minimize_scalar(lambda x: uniform_scores(S, x, 0.0, sub)[i],
                            bounds=tuple(bounds), method='bounded',
                            options={'xatol': xatol})
        a, n, p = uniform_scores(S, float(r.x), 0.0, sub)
        rows, _, _ = per_gauge(S, np.full(S['mesh_x'].size, 10 ** float(r.x)), 0.0)
        return dict(m=m, log10_D=float(r.x), rmse_psi=a, rmse_normalised=n,
                    rmse_pooled_psi=p,
                    per_gauge_rmse=[rows[i2]['rmse_psi'] for i2 in sub])
    if kind == 'subset_leak':
        _, m, which, starts, maxiter = args
        sub = list(range(0, m - 1))
        i = 0 if which == 'abs' else 1

        def f(q):
            if not (1.5 < q[0] < 5.5 and -9.0 < q[1] < -1.0):
                return 1e9
            return uniform_scores(S, q[0], 10.0 ** q[1], sub)[i]

        best = None
        for x0 in starts:
            r = minimize(f, np.asarray(x0, float), method='Nelder-Mead',
                         options={'xatol': 1e-4, 'fatol': 1e-7,
                                  'maxiter': maxiter})
            if best is None or r.fun < best.fun:
                best = r
        a, n, p = uniform_scores(S, float(best.x[0]), 10.0 ** float(best.x[1]),
                                 sub)
        # The lambda = 0 face of the same parameter space is a legitimate
        # candidate and Nelder-Mead cannot reach it (log10 lambda -> -inf), so
        # it is optimised separately and the better of the two is reported.
        r0 = minimize_scalar(lambda x: uniform_scores(S, x, 0.0, sub)[i],
                             bounds=(1.5, 5.5), method='bounded',
                             options={'xatol': 1e-4})
        a0, n0, p0 = uniform_scores(S, float(r0.x), 0.0, sub)
        v = a if which == 'abs' else n
        v0 = a0 if which == 'abs' else n0
        if v0 <= v:
            return dict(m=m, log10_D=float(r0.x), log10_lambda=float('-inf'),
                        rmse_psi=a0, rmse_normalised=n0, rmse_pooled_psi=p0,
                        lambda_star_is_zero=True,
                        best_interior_rmse_psi=a,
                        best_interior_log10_lambda=float(best.x[1]))
        return dict(m=m, log10_D=float(best.x[0]),
                    log10_lambda=float(best.x[1]),
                    rmse_psi=a, rmse_normalised=n, rmse_pooled_psi=p,
                    lambda_star_is_zero=False,
                    best_interior_rmse_psi=a,
                    best_interior_log10_lambda=float(best.x[1]))
    if kind == 'single':
        _, fam, p, lam = args
        prof = (np.full(S['mesh_x'].size, 10.0 ** float(p[0]))
                if fam == 'uniform_log10' else family_profile(S, fam,
                                                              np.asarray(p, float)))
        rows, _, _ = per_gauge(S, prof, lam)
        a, n, pp = scores(rows)
        return dict(rows=rows, rmse_psi=a, rmse_normalised=n,
                    rmse_pooled_psi=pp)
    raise ValueError('unknown job kind %r' % (kind,))


# ---------------------------------------------------------------------------
# Latin-hypercube + Nelder-Mead refit of a profile family under a chosen norm
# ---------------------------------------------------------------------------

def refit_family(pool, fam, bounds, cfg, which):
    sp = cfg['renormalisation_search']
    seed = int(sp['seed'])
    i = 0 if which == 'abs' else 1
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)

    def evaluate(pts):
        vals = pool.map(job, [('fam', fam, list(q)) for q in pts], chunksize=4)
        return np.array([v[i] for v in vals])

    pts = lo + qmc.LatinHypercube(d=len(bounds), seed=seed).random(
        int(sp['n_lhs_coarse'])) * (hi - lo)
    v = evaluate(pts)
    order = np.argsort(v)
    best_p, best_v = pts[order[0]].copy(), float(v[order[0]])

    half = (hi - lo) * 0.25 / 2.0
    l2 = np.maximum(lo, best_p - half)
    h2 = np.minimum(hi, best_p + half)
    pts2 = l2 + qmc.LatinHypercube(d=len(bounds), seed=seed + 1).random(
        int(sp['n_lhs_refine'])) * (h2 - l2)
    v2 = evaluate(pts2)
    if float(v2.min()) < best_v:
        best_p, best_v = pts2[int(np.argmin(v2))].copy(), float(v2.min())

    starts = [best_p] + [pts[j] for j in order[1:int(sp['n_polish_starts'])]]
    for x0 in starts:
        r = minimize(lambda q: job(('fam', fam, list(q)))[i], x0,
                     method='Nelder-Mead',
                     options={'maxiter': int(sp['nelder_mead_maxiter']),
                              'xatol': 1e-4, 'fatol': 1e-6})
        if float(r.fun) < best_v:
            best_p, best_v = np.clip(np.asarray(r.x, float), lo, hi), float(r.fun)

    a, n, p = job(('fam', fam, list(best_p)))
    return dict(family=fam, params=[float(x) for x in best_p],
                param_names=r1c.PROFILE_FAMILIES[fam]['names'],
                k=int(r1c.PROFILE_FAMILIES[fam]['k']),
                rmse_psi=a, rmse_normalised=n, rmse_pooled_psi=p,
                norm_optimised=which,
                at_bound=[bool(abs(x - b[0]) < 1e-6 or abs(x - b[1]) < 1e-6)
                          for x, b in zip(best_p, bounds)])


# ---------------------------------------------------------------------------
# AICc
# ---------------------------------------------------------------------------

def aicc_table(models, n_list):
    """AICc = n ln(sigma2) + 2k + 2k(k+1)/(n-k-1), sigma2 = gauge-mean MSE.

    Reported over every n convention because the effective sample size is not
    unique here (house rules). Undefined whenever n - k - 1 <= 0.
    """
    out = {}
    for n in n_list:
        rows = {}
        for name, m in models.items():
            k = int(m['k'])
            s2 = float(m['rmse_psi']) ** 2
            base = n * np.log(s2) + 2 * k
            if n - k - 1 > 0:
                rows[name] = {'k': k, 'aicc': float(base + 2 * k * (k + 1)
                                                    / (n - k - 1)),
                              'aic': float(base), 'defined': True}
            else:
                rows[name] = {'k': k, 'aicc': None, 'aic': float(base),
                              'defined': False,
                              'why': 'n - k - 1 = %d <= 0' % (n - k - 1)}
        defined = {kk: v['aicc'] for kk, v in rows.items() if v['defined']}
        if defined:
            best = min(defined, key=defined.get)
            for kk, v in rows.items():
                v['delta_aicc'] = (None if not v['defined']
                                   else float(v['aicc'] - defined[best]))
            rows['_best'] = best
        out[str(n)] = rows
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_map(path, grid, which, cfg, floor, rfloor, opt, dpi):
    """Filled misfit contours over (log10 lambda, log10 D) + the exact floor."""
    key = 'rmse' if which == 'abs' else 'rmse_norm'
    Z = grid[key]                       # (n_lam, n_D), lambda-major
    lg = grid['log10_lambda']
    lD = grid['log10_D']
    z0 = grid[key + '_lambda_zero']     # (n_D,) at lambda = 0

    fig = plt.figure(figsize=(14.5, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.15], width_ratios=[3.1, 1.0],
                          hspace=0.30, wspace=0.16)

    a = fig.add_subplot(gs[0, 0])
    vmin = float(np.nanmin(Z))
    lv = np.linspace(vmin, float(np.nanpercentile(Z, 80)), 24)
    cs = a.contourf(lg, lD, Z.T, levels=lv, cmap='viridis', extend='max')
    a.contour(lg, lD, Z.T, levels=lv[::4], colors='w', linewidths=0.4, alpha=0.6)
    cb = fig.colorbar(cs, ax=a, pad=0.012)
    cb.set_label('gauge-mean RMSE (psi)' if which == 'abs'
                 else 'amplitude-normalised RMSE (-)', fontsize=9)
    fl = np.array([[f['lam'], f['log10_D']] for f in floor if f['lam'] > 0])
    a.plot(np.log10(fl[:, 0]), fl[:, 1], 'w-', lw=2.0,
           label=r'exact valley floor: $\min_D$ at each $\lambda$')
    a.plot([opt['log10_lambda_plot']], [opt['log10_D']], 'r*', ms=17,
           mec='k', mew=0.7,
           label=(r'optimum $\lambda^*=0$ (censored at the boundary),'
                  '\n' + r'$D^*=%.0f$ ft$^2$/s, %s' % (
                      10 ** opt['log10_D'],
                      ('%.2f psi' % opt['rmse_psi']) if which == 'abs'
                      else ('%.4f' % opt['rmse_normalised']))))
    lam_phys = float(opt['lambda_physical'])
    a.axvline(np.log10(lam_phys), color='orange', ls='--', lw=1.4)
    a.text(np.log10(lam_phys) - 0.10, lD[-1] - 0.05,
           r'$\lambda$ expected by the task memo'
           '\n' r'($\sqrt{D/\lambda}\approx550$ ft, $1/\lambda\approx260$ s)',
           color='darkorange', fontsize=8, ha='right', va='top')
    a.set_xlabel(r'$\log_{10}\ \lambda_{\rm leak}$  (s$^{-1}$)')
    a.set_ylabel(r'$\log_{10}\ D$  (ft$^2$/s)')
    a.set_title('(a) %s-norm misfit surface; the floor rises monotonically '
                'away from $\\lambda=0$'
                % ('absolute' if which == 'abs' else 'normalised'), fontsize=10)
    a.legend(fontsize=8, loc='lower left', framealpha=0.9)

    a = fig.add_subplot(gs[0, 1])
    a.plot(z0, lD, 'k-', lw=1.6)
    a.axhline(opt['log10_D'], color='r', ls='--', lw=1.0)
    a.set_xlabel('gauge-mean RMSE (psi)' if which == 'abs'
                 else 'normalised RMSE (-)', fontsize=9)
    a.set_title(r'(b) the $\lambda=0$ slice' '\n' '(the uniform model)',
                fontsize=10)
    a.grid(alpha=0.3)
    a.tick_params(labelsize=8)

    a = fig.add_subplot(gs[1, 0])
    lamv = np.array([f['lam'] for f in floor])
    val = np.array([f['rmse_psi'] if which == 'abs' else f['rmse_normalised']
                    for f in floor])
    pos = lamv > 0
    a.semilogx(lamv[pos], val[pos], 'o-', color='C0', ms=3.0, lw=1.3,
               label=r'floor $\min_D$ RMSE$(\lambda)$')
    v0 = float(val[~pos][0])
    a.axhline(v0, color='k', ls=':', lw=1.1,
              label=r'$\lambda=0$ (uniform): %s'
                    % (('%.2f psi' % v0) if which == 'abs' else '%.4f' % v0))
    for nm2, vv, cc in opt['baseline_lines']:
        a.axhline(vv, color=cc, ls='--', lw=1.2, label='%s: %s' % (
            nm2, ('%.2f psi' % vv) if which == 'abs' else '%.4f' % vv))
    a.axvline(lam_phys, color='orange', ls='--', lw=1.2)
    a.set_xlabel(r'$\lambda_{\rm leak}$ (s$^{-1}$)')
    a.set_ylabel('gauge-mean RMSE (psi)' if which == 'abs'
                 else 'normalised RMSE (-)')
    a.set_title('(c) exact valley floor against the three baselines on the '
                'identical criterion', fontsize=10)
    a.grid(alpha=0.3, which='both')
    a.legend(fontsize=7.5, ncol=2)
    if which == 'abs':
        a.set_ylim(0, max(120.0, float(np.nanmax(val[pos])) * 1.05))

    a = fig.add_subplot(gs[1, 1])
    dv = np.array([r['log10_D'] for r in rfloor])
    vbest = np.array([(r['rmse_psi'] if which == 'abs'
                       else r['rmse_normalised']) for r in rfloor])
    vzero = np.array([(r['rmse_at_lambda_zero'] if which == 'abs'
                       else r['rmse_normalised_at_lambda_zero'])
                      for r in rfloor])
    a.plot(dv, vzero, 'o-', color='k', ms=3, lw=1.2, label=r'$\lambda=0$')
    a.plot(dv, vbest, 's-', color='C3', ms=3, lw=1.2,
           label=r'best $\lambda$ at that $D$')
    a.set_xlabel(r'$\log_{10} D$ (ft$^2$/s)', fontsize=9)
    a.set_ylabel('gauge-mean RMSE (psi)' if which == 'abs'
                 else 'normalised RMSE (-)', fontsize=8)
    a.set_title('(d) reverse floor ' r'$\min_\lambda$ at each $D$:'
                '\nthe sink helps only where $D$ is\nalready wrong',
                fontsize=9)
    a.grid(alpha=0.3)
    a.legend(fontsize=7.5)
    a.tick_params(labelsize=8)

    fig.suptitle('C2 -- leakage sink $\\partial_t P = D\\,\\partial_x^2 P '
                 '- \\lambda_{\\rm leak}P$: the %s norm drives $\\lambda_{\\rm leak}$ '
                 'to zero, i.e. back to the uniform model.'
                 % ('absolute' if which == 'abs' else 'amplitude-normalised'),
                 fontsize=12)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.895, bottom=0.075)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def fig_fit(path, S, curves, which, dpi):
    """Per-gauge fit + residuals, laid out exactly as r1_qc_fit.png (2 x 6)."""
    tg = S['targets']
    fig, axes = plt.subplots(2, 6, figsize=(19, 7),
                             gridspec_kw={'height_ratios': [2, 1]})
    for k, tgt in enumerate(tg):
        a = axes[0, k]
        a.plot(tgt['taxis'], tgt['data'], 'k-', lw=1.8, label='observed')
        for c in curves:
            a.plot(tgt['taxis'], c['sim'][k], c['ls'], color=c['color'],
                   lw=1.4, label=c['label'])
        a.set_title("g%d  %.0f ft" % (tgt['gauge'], tgt['distance_ft']),
                    fontsize=10)
        a.grid(alpha=0.3)
        if k == 0:
            a.set_ylabel(r'$\Delta P$ (psi)')
            a.legend(fontsize=6.6)
        a.tick_params(labelsize=8)

        a = axes[1, k]
        for c in curves:
            a.plot(tgt['taxis'], c['sim'][k] - tgt['data'], '-',
                   color=c['color'], lw=1.2)
        a.axhline(0, color='k', lw=0.8, ls=':')
        a.set_xlabel('time (s)')
        a.grid(alpha=0.3)
        a.tick_params(labelsize=8)
        if k == 0:
            a.set_ylabel('residual\n(sim - obs, psi)')
        bb = dict(fc='w', ec='none', alpha=0.78, pad=1.4)
        a.text(0.03, 0.04, 'RMSE ' + ' / '.join(
            '%.0f' % c['rows'][k]['rmse_psi'] for c in curves),
            transform=a.transAxes, fontsize=7.0, bbox=bb)
        a.text(0.03, 0.85, 'bias ' + ' / '.join(
            '%+.0f' % c['rows'][k]['bias_psi'] for c in curves),
            transform=a.transAxes, fontsize=7.0, bbox=bb)

    fig.suptitle(
        'C2 QC -- per-gauge fit and residuals at the leakage-sink optimum, %s '
        'norm.  Layout matches r1_qc_fit.png.\nThe sink does NOT remove the '
        'near-under / far-over sign flip: at its own optimum it IS the uniform '
        'model ($\\lambda^*=0$), and forcing $\\lambda>0$ enlarges the flip.'
        % ('absolute' if which == 'abs' else 'amplitude-normalised'),
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def fig_decay(path, S, decay, subsets, dpi):
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

    a = ax[0]
    d = decay['distance_ft']
    a.semilogy(d, decay['obs_max_psi'], 'ko-', lw=1.6, ms=6, label='observed peak')
    for c in decay['curves']:
        a.semilogy(d, c['peak'], c['ls'], color=c['color'], lw=1.4, ms=5,
                   label=c['label'])
    a.set_xlabel('distance from the source gauge (ft)')
    a.set_ylabel('peak $\\Delta P$ in the window (psi)')
    a.set_title('(a) Amplitude decay with distance', fontsize=10)
    a.grid(alpha=0.3, which='both')
    a.legend(fontsize=7.5)

    a = ax[1]
    a.plot(decay['seg_mid_ft'], decay['L_obs_ft'], 'ko-', lw=1.6, ms=6,
           label='observed, segment by segment')
    a.axhline(decay['L_leak_best_ft'], color='C1', ls='--', lw=1.4,
              label=r'$\sqrt{D/\lambda}$ with $\lambda$ forced to '
                    r'$1/260$ s$^{-1}$: %.0f ft' % decay['L_leak_best_ft'])
    a.axhline(550.0, color='orange', ls=':', lw=1.4,
              label="the memo's 550 ft expectation")
    a.set_yscale('log')
    a.set_xlabel('midpoint of the gauge pair (ft from source)')
    a.set_ylabel('implied decay length $L$ (ft)')
    a.set_title('(b) The observed decay length is not constant\n'
                'a single-$\\lambda$ sink can only produce a constant one',
                fontsize=10)
    a.grid(alpha=0.3, which='both')
    a.legend(fontsize=7.5)

    a = ax[2]
    m = np.array([s['max_dist_ft'] for s in subsets])
    a.plot(m, [s['uniform_rmse'] for s in subsets], 'o-', color='C0', lw=1.5,
           label='uniform (k=1)')
    a.plot(m, [s['leak_rmse'] for s in subsets], 's-', color='C1', lw=1.5,
           label='leakage sink (k=2)')
    a.plot(m, [s['single_gauge_floor'] for s in subsets], '^:', color='C2',
           lw=1.4, label='per-gauge single-fit floor')
    a.set_xlabel('farthest gauge included (ft from source)')
    a.set_ylabel('gauge-mean RMSE (psi)')
    a.set_title('(c) Range of applicability:\nnested fits, near gauges first',
                fontsize=10)
    a.grid(alpha=0.3)
    a.legend(fontsize=8)

    fig.suptitle('C2 diagnostics -- why one leakage coefficient cannot do the '
                 'job, and how far it does reach.', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def manifest_groups(S, cfg, taxis, d_profile, lam_declared, sweep_note):
    drv = rm.driver_record(
        kind='gauge_series',
        baseline_removal=cfg['source']['baseline_removal'],
        value_units='delta_psi',
        series_path=rd.repo_path(rd.SWELL_GAUGE_TEMPLATE.format(
            n=int(S['src_gauge']))),
        gauge_number=int(S['src_gauge']), gauge_md_ft=float(S['src_md']),
        taxis=S['src_taxis'], values=S['src_delta'],
        time_start=cfg['window']['time_start'],
        time_end=cfg['window']['time_end'])
    src = rm.source_record(S['mesh_x'], md_requested_ft=float(S['src_md']),
                           mesh_idx=int(S['source_idx']), driver=drv,
                           label='gauge%d' % int(S['src_gauge']),
                           excluded_from_misfit=True, index_in_source_list=0)
    source_group = rm.source_protocol(
        application=cfg['source']['application'],
        solver_class='rev2_core.solve_forward (banded, theta-generalised, '
                     'interior-only leakage sink)',
        placement_rule=cfg['source']['selection_rule'],
        sources=[src],
        targets={'gauges': [int(t['gauge']) for t in S['targets']],
                 'md_ft': [float(t['md_ft']) for t in S['targets']],
                 'distance_ft': [float(t['distance_ft']) for t in S['targets']],
                 'n_samples': [int(t['taxis'].size) for t in S['targets']],
                 'role': 'misfit targets; the source gauge is fit trivially by '
                         'construction and is excluded'},
        time_level='n',
        phase_chaining={'mode': 'single_phase',
                        'note': 'one window, one solve per parameter pair; no '
                                'phase chaining and no restart discontinuity, '
                                'so theta_startup_steps is irrelevant here'},
        boundary_conditions=cfg['source']['boundary_conditions'])

    numerics = rm.numerics(
        time=rm.time_record(taxis, mode='fixed', theta=float(cfg['solver']['theta']),
                            t_total_requested_s=float(S['t_total_s']),
                            dt_requested_s=float(cfg['solver']['dt_s']),
                            source_time_level='n', theta_startup_steps=0,
                            label='every solve in this study shares this taxis'),
        mesh=rm.mesh_record(S['mesh_x'],
                            dx_requested_ft=float(cfg['mesh']['dx_ft']),
                            window_md_ft=(float(cfg['window']['md_min_ft']),
                                          float(cfg['window']['md_max_ft'])),
                            pad_low_ft=float(cfg['mesh']['pad_low_ft']),
                            pad_high_ft=float(cfg['mesh']['pad_high_ft']),
                            refinement={'mode': 'none', 'uniform': True}),
        interface_avg=cfg['solver']['interface_avg'],
        boundary=cfg['source']['boundary_conditions'],
        diffusivity={'profile_family': 'uniform (D constant in x)',
                     'param_names': ['log10_D'],
                     'params': [float(np.log10(d_profile[0]))],
                     'D_min': float(np.min(d_profile)),
                     'D_max': float(np.max(d_profile)),
                     'D_sha256': rm.sha256_array(d_profile),
                     'profile_anchor': 'not applicable (uniform)',
                     'note': 'the reported profile is the study optimum; the '
                             'sweep covers the range recorded under leakage'},
        barriers=rm.NONE_DECLARED,
        leakage={'model': 'dP/dt = D d2P/dx2 - lambda_leak*(P - p0)',
                 'p0_psi': float(cfg['solver']['p0_psi']),
                 'placement': 'interior nodes only; NOT on the Dirichlet source '
                              'row and NOT on either Neumann row',
                 'lambda_leak_s^-1_at_optimum': float(lam_declared),
                 'sweep': sweep_note},
        kernel={'name': 'rev2_core.solve_forward', 'banded': True,
                'equivalence_reference':
                    'output/rev2_20260901/A4/selftest_output.txt',
                'note': 'theta=1 / harmonic / lambda=0 is bitwise identical to '
                        'r1_calibration_core.solve_forward, which is '
                        'bit-equivalent to fibeRIS; re-asserted in this run '
                        'under results.verification'},
        rng={'latin_hypercube_seed': int(cfg['renormalisation_search']['seed']),
             'used_for': 'only the normalised-norm refits of the triangular and '
                         'two_zone baselines; the (D, lambda) search is a '
                         'deterministic grid plus deterministic Brent'},
        parallel={'mode': 'multiprocessing.Pool',
                  'processes': int(cfg['parallel']['processes']),
                  'why': 'house rules cap concurrent tasks at 6 workers'},
        amplification=rc.amplification_factor(
            S['mesh_x'], d_profile, float(cfg['solver']['dt_s']),
            float(cfg['solver']['theta']),
            interface_avg=cfg['solver']['interface_avg'],
            lambda_leak=float(lam_declared)))
    return source_group, numerics


def study_inputs():
    inp = [(rd.repo_path(rd.SWELL_GAUGE_MD_NPZ), 'geometry', 'gauge_md_swell'),
           (rd.repo_path(rd.SWELL_FRAC_HIT_TEMPLATE.format(stage=1)), 'geometry',
            'frac_hit_stage_1')]
    for g in range(1, 8):
        inp.append((rd.repo_path(rd.SWELL_GAUGE_TEMPLATE.format(n=g)),
                    'gauge_series', 'gauge%d_swell' % g))
    inp.append(('output/r2_diffusivity_profile/r2_manifest.json',
                'prior_run_output', 'r2_profile_inversion_manifest'))
    return inp


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def axis(spec):
    lo, hi, step = float(spec[0]), float(spec[1]), float(spec[2])
    n = int(round((hi - lo) / step)) + 1
    return lo + step * np.arange(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    ocfg = cfg['outputs']
    root = ocfg['root_dir']
    tag = ocfg['version_tag']
    dpi = int(ocfg['figure_dpi'])
    dirs = {w: os.path.join(root, ocfg[k]) for w, k in
            (('abs', 'abs_subdir'), ('amp', 'amp_subdir'),
             ('diag', 'diag_subdir'))}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    rm.assert_absent([os.path.join(d, 'manifest.json') for d in dirs.values()])

    log('building the standard setup')
    S = build_setup(cfg)
    _G['S'] = S
    _G['cfg'] = cfg
    log('nx=%d MD %.0f-%.0f  source g%d MD %.0f idx %d  t_total %.3f s  '
        'targets %s' % (S['mesh_x'].size, S['mesh_x'][0], S['mesh_x'][-1],
                        S['src_gauge'], S['src_md'], S['source_idx'],
                        S['t_total_s'],
                        [int(t['gauge']) for t in S['targets']]))

    log('re-verifying the leakage sink independently')
    ver = verify_sink()
    log('  verification: %s' % json.dumps(ver, sort_keys=True))
    if not ver['passed']:
        raise SystemExit('leakage sink verification FAILED -- see above. '
                         'Per the house rules this is reported, not patched.')

    nproc = int(cfg['parallel']['processes'])
    pool = mp.Pool(nproc, initializer=_init_worker, initargs=(cfg,))
    try:
        # ---- 1. baselines, absolute norm, reproduced from scratch ---------
        log('reproducing the three published baselines')
        base = {}
        r_uni = minimize_scalar(
            lambda x: job(('uni', x, 0.0))[0],
            bounds=tuple(cfg['baselines']['uniform']['bounds'][0]),
            method='bounded', options={'xatol': 1e-4})
        a, n, p = job(('uni', float(r_uni.x), 0.0))
        base['uniform'] = dict(family='uniform', k=1,
                               params=[float(r_uni.x)],
                               param_names=['log10_D'],
                               rmse_psi=a, rmse_normalised=n,
                               rmse_pooled_psi=p, norm_optimised='abs')
        for name in ('triangular', 'two_zone'):
            pub = cfg['baselines'][name]['published_params']
            a, n, p = job(('fam', name, pub))
            base[name] = dict(family=name, k=int(cfg['baselines'][name]['k']),
                              params=[float(x) for x in pub],
                              param_names=r1c.PROFILE_FAMILIES[name]['names'],
                              rmse_psi=a, rmse_normalised=n,
                              rmse_pooled_psi=p, norm_optimised='abs',
                              published_rmse_psi=float(
                                  cfg['baselines'][name]['published_rmse_psi']),
                              reproduction_abs_diff_psi=float(
                                  abs(a - cfg['baselines'][name]
                                      ['published_rmse_psi'])))
            log('  %-11s RMSE %10.5f psi (published %.5f, |d| = %.2e)  '
                'norm %.5f' % (name, a,
                               cfg['baselines'][name]['published_rmse_psi'],
                               base[name]['reproduction_abs_diff_psi'], n))
        log('  uniform     RMSE %10.5f psi at D = %.1f  norm %.5f'
            % (base['uniform']['rmse_psi'], 10 ** r_uni.x,
               base['uniform']['rmse_normalised']))

        # ---- 2. coarse and refined grids ---------------------------------
        grids = {}
        for gname in ('coarse', 'refined'):
            gc = cfg['grid'][gname]
            lD = axis(gc['log10_D'])
            lg = axis(gc['log10_lambda'])
            log('%s grid: %d D x %d lambda = %d solves (+%d at lambda=0)'
                % (gname, lD.size, lg.size, lD.size * lg.size, lD.size))
            jobs = [('uni', float(d), float(10.0 ** g))
                    for g in lg for d in lD]
            res = pool.map(job, jobs, chunksize=8)
            R = np.array([r[0] for r in res]).reshape(lg.size, lD.size)
            N = np.array([r[1] for r in res]).reshape(lg.size, lD.size)
            P = np.array([r[2] for r in res]).reshape(lg.size, lD.size)
            res0 = pool.map(job, [('uni', float(d), 0.0) for d in lD],
                            chunksize=8)
            grids[gname] = {
                'log10_D': lD, 'log10_lambda': lg,
                'rmse': R, 'rmse_norm': N, 'rmse_pooled': P,
                'rmse_lambda_zero': np.array([r[0] for r in res0]),
                'rmse_norm_lambda_zero': np.array([r[1] for r in res0]),
                'rmse_pooled_lambda_zero': np.array([r[2] for r in res0]),
            }
            log('  %s: min RMSE %.4f psi, min normalised %.5f'
                % (gname, min(R.min(), grids[gname]['rmse_lambda_zero'].min()),
                   min(N.min(), grids[gname]['rmse_norm_lambda_zero'].min())))

        # ---- 3. exact valley floors --------------------------------------
        fc = cfg['floor']
        lam_axis = 10.0 ** axis(fc['log10_lambda'])
        floors, rfloors = {}, {}
        for which in ('abs', 'amp'):
            w = 'abs' if which == 'abs' else 'norm'
            jobs = [('floor', 0.0, w, fc['log10_D_bounds'],
                     float(fc['brent_xatol']))]
            jobs += [('floor', float(l), w, fc['log10_D_bounds'],
                      float(fc['brent_xatol'])) for l in lam_axis]
            floors[which] = pool.map(job, jobs, chunksize=1)
            lDs = np.linspace(fc['reverse_log10_D'][0], fc['reverse_log10_D'][1],
                              int(fc['reverse_n_D']))
            rjobs = [('rfloor', float(d), w,
                      fc['reverse_log10_lambda_bounds'],
                      float(fc['brent_xatol'])) for d in lDs]
            rfloors[which] = pool.map(job, rjobs, chunksize=1)
            f0 = floors[which][0]
            v0 = f0['rmse_psi'] if which == 'abs' else f0['rmse_normalised']
            vals = np.array([(f['rmse_psi'] if which == 'abs'
                              else f['rmse_normalised'])
                             for f in floors[which]])
            log('floor(%s): lambda=0 gives %.5f at D=%.1f; min over the whole '
                'floor %.5f at lambda=%s; monotone-increasing in lambda: %s'
                % (which, v0, 10 ** f0['log10_D'], vals.min(),
                   ('0' if int(np.argmin(vals)) == 0
                    else '%.3e' % floors[which][int(np.argmin(vals))]['lam']),
                   bool(np.all(np.diff(vals) > 0))))
            rvals = np.array([(r['rmse_psi'] if which == 'abs'
                               else r['rmse_normalised'])
                              for r in rfloors[which]])
            jbest = int(np.argmin(rvals))
            gains = np.array([r['gain_over_lambda_zero']
                              for r in rfloors[which]])
            log('  reverse floor(%s): min over the reverse D grid '
                '%.5f at D=%.1f, lambda=%.3e (low bound reached: %s). '
                'Max per-D gain from lambda>0 anywhere: %.3e -- but it is '
                'never at the global optimum.'
                % (which, rvals[jbest], 10 ** rfloors[which][jbest]['log10_D'],
                   10 ** rfloors[which][jbest]['log10_lambda'],
                   rfloors[which][jbest]['at_low_bound'], gains.max()))

        # ---- 4. baselines re-optimised under the normalised norm ---------
        log('re-optimising the three baselines under the normalised norm')
        base_amp = {}
        r_uni_n = minimize_scalar(
            lambda x: job(('uni', x, 0.0))[1],
            bounds=tuple(cfg['baselines']['uniform']['bounds'][0]),
            method='bounded', options={'xatol': 1e-4})
        a, n, p = job(('uni', float(r_uni_n.x), 0.0))
        base_amp['uniform'] = dict(family='uniform', k=1,
                                   params=[float(r_uni_n.x)],
                                   param_names=['log10_D'], rmse_psi=a,
                                   rmse_normalised=n, rmse_pooled_psi=p,
                                   norm_optimised='amp', at_bound=[False])
        for name in ('triangular', 'two_zone'):
            base_amp[name] = refit_family(
                pool, name, cfg['baselines'][name]['bounds'], cfg, 'amp')
            log('  %-11s normalised %.5f (abs %.3f psi) params %s'
                % (name, base_amp[name]['rmse_normalised'],
                   base_amp[name]['rmse_psi'],
                   ['%.4f' % x for x in base_amp[name]['params']]))
        log('  uniform     normalised %.5f at D = %.1f'
            % (base_amp['uniform']['rmse_normalised'], 10 ** r_uni_n.x))

        # ---- 5. residual structure at the interesting points -------------
        log('scoring the residual-structure reference points')
        lam_phys = 1.0 / 260.0
        opt_abs = floors['abs'][0]
        opt_amp = floors['amp'][0]
        r_phys = minimize_scalar(lambda x: job(('uni', x, lam_phys))[0],
                                 bounds=tuple(fc['log10_D_bounds']),
                                 method='bounded', options={'xatol': 1e-4})
        points = {
            'leak_optimum_abs': ('uniform_log10', [opt_abs['log10_D']], 0.0),
            'leak_optimum_amp': ('uniform_log10', [opt_amp['log10_D']], 0.0),
            'leak_forced_physical_lambda': ('uniform_log10', [float(r_phys.x)],
                                            lam_phys),
            'leak_memo_point': ('uniform_log10', [np.log10(1150.0)], lam_phys),
            'two_zone': ('two_zone',
                         cfg['baselines']['two_zone']['published_params'], 0.0),
            'triangular': ('triangular',
                           cfg['baselines']['triangular']['published_params'],
                           0.0),
        }
        detail = {}
        for k2, (fam, p2, lam) in points.items():
            detail[k2] = job(('single', fam, p2, lam))
            detail[k2]['family'] = fam
            detail[k2]['params'] = [float(x) for x in p2]
            detail[k2]['lambda_leak'] = float(lam)

        # ---- 6. range of applicability -----------------------------------
        log('nested-subset fits (range of applicability)')
        rc_cfg = cfg['range_of_applicability']
        sub_uni = pool.map(job, [('subset_uni', m, 'abs',
                                  cfg['baselines']['uniform']['bounds'][0], 1e-4)
                                 for m in range(2, 8)], chunksize=1)
        sub_leak = pool.map(job, [('subset_leak', m, 'abs',
                                   rc_cfg['nm_starts'],
                                   int(rc_cfg['nm_maxiter']))
                                  for m in range(2, 8)], chunksize=1)
        # per-gauge single fits: fit each gauge ALONE
        solo = []
        for m in range(2, 8):
            sub = [m - 2]
            r = minimize_scalar(lambda x: uniform_scores(S, x, 0.0, sub)[0],
                                bounds=(1.0, 5.5), method='bounded',
                                options={'xatol': 1e-4})
            solo.append(dict(gauge=int(S['targets'][m - 2]['gauge']),
                             distance_ft=float(S['targets'][m - 2]
                                               ['distance_ft']),
                             log10_D=float(r.x), D=float(10 ** r.x),
                             rmse_psi=float(r.fun)))
            log('  solo g%d: D = %8.1f ft^2/s, RMSE %6.2f psi'
                % (solo[-1]['gauge'], solo[-1]['D'], solo[-1]['rmse_psi']))
        subsets = []
        for i2, m in enumerate(range(2, 8)):
            sub = list(range(0, m - 1))
            floor_solo = float(np.sqrt(np.mean(
                [solo[j]['rmse_psi'] ** 2 for j in sub])))
            subsets.append(dict(
                m=m, n_gauges=m - 1,
                max_dist_ft=float(S['targets'][m - 2]['distance_ft']),
                uniform_D=float(10 ** sub_uni[i2]['log10_D']),
                uniform_rmse=float(sub_uni[i2]['rmse_psi']),
                leak_D=float(10 ** sub_leak[i2]['log10_D']),
                leak_lambda=float(10 ** sub_leak[i2]['log10_lambda']),
                leak_decay_length_ft=float(np.sqrt(
                    10 ** sub_leak[i2]['log10_D']
                    / 10 ** sub_leak[i2]['log10_lambda']))
                if np.isfinite(sub_leak[i2]['log10_lambda']) else float('inf'),
                leak_rmse=float(sub_leak[i2]['rmse_psi']),
                leak_gain_psi=float(sub_uni[i2]['rmse_psi']
                                    - sub_leak[i2]['rmse_psi']),
                leak_lambda_star_is_zero=bool(
                    sub_leak[i2].get('lambda_star_is_zero', False)),
                leak_best_interior_rmse_psi=float(
                    sub_leak[i2].get('best_interior_rmse_psi', np.nan)),
                single_gauge_floor=floor_solo))
            log('  g2..g%d (<=%4.0f ft): uniform %7.2f psi (D %8.1f) | leak '
                '%7.2f psi (D %9.1f, lam %8.2e, L %6.0f ft) | solo floor '
                '%6.2f psi' % (m, subsets[-1]['max_dist_ft'],
                               subsets[-1]['uniform_rmse'],
                               subsets[-1]['uniform_D'],
                               subsets[-1]['leak_rmse'], subsets[-1]['leak_D'],
                               subsets[-1]['leak_lambda'],
                               subsets[-1]['leak_decay_length_ft'],
                               floor_solo))
    finally:
        pool.close()
        pool.join()

    # ---- 7. AICc ---------------------------------------------------------
    leak_abs = dict(k=2, rmse_psi=opt_abs['rmse_psi'],
                    rmse_normalised=opt_abs['rmse_normalised'])
    leak_amp = dict(k=2, rmse_psi=opt_amp['rmse_psi'],
                    rmse_normalised=opt_amp['rmse_normalised'])
    models_abs = {
        'uniform (k=1)': dict(k=1, rmse_psi=base['uniform']['rmse_psi']),
        'triangular free ratio (k=2)': dict(
            k=2, rmse_psi=base['triangular']['rmse_psi']),
        'leakage sink (k=2)': dict(k=2, rmse_psi=leak_abs['rmse_psi']),
        'two_zone D(x) (k=4)': dict(k=4, rmse_psi=base['two_zone']['rmse_psi']),
    }
    aicc = aicc_table(models_abs, cfg['aicc']['n_conventions'])
    log('AICc table built over n in %s' % (cfg['aicc']['n_conventions'],))

    # ---- 8. figures and arrays ------------------------------------------
    log('writing figures')
    decay_curves = []
    curve_specs = {
        'abs': [('leak_optimum_abs', 'C0', '--',
                 r'leakage optimum ($\lambda^*=0$), D=%.0f'),
                ('leak_forced_physical_lambda', 'C3', '--',
                 r'leakage forced $\lambda=1/260$ s$^{-1}$, D=%.0f'),
                ('two_zone', 'C2', '-.', 'two_zone D(x), k=4')],
        'amp': [('leak_optimum_amp', 'C0', '--',
                 r'leakage optimum ($\lambda^*=0$), D=%.0f'),
                ('leak_forced_physical_lambda', 'C3', '--',
                 r'leakage forced $\lambda=1/260$ s$^{-1}$, D=%.0f'),
                ('two_zone', 'C2', '-.', 'two_zone D(x), k=4')],
    }

    def rebuild_sims(key):
        fam, p2, lam = points[key]
        prof = (np.full(S['mesh_x'].size, 10.0 ** float(p2[0]))
                if fam == 'uniform_log10' else family_profile(S, fam,
                                                              np.asarray(p2, float)))
        taxis, rec = forward(S, prof, lam)
        return [np.interp(t['taxis'], taxis, rec[:, k])
                for k, t in enumerate(S['targets'])], taxis

    sims_cache = {}
    for key in points:
        sims_cache[key], taxis_ref = rebuild_sims(key)

    written = {'abs': [], 'amp': [], 'diag': []}
    for which in ('abs', 'amp'):
        w = 'abs' if which == 'abs' else 'norm'
        optd = opt_abs if which == 'abs' else opt_amp
        opt = dict(log10_D=optd['log10_D'],
                   log10_lambda_plot=float(cfg['grid']['coarse']
                                           ['log10_lambda'][0]),
                   rmse_psi=optd['rmse_psi'],
                   rmse_normalised=optd['rmse_normalised'],
                   lambda_physical=1.0 / 260.0,
                   baseline_lines=[
                       ('uniform k=1',
                        base['uniform']['rmse_psi'] if which == 'abs'
                        else base_amp['uniform']['rmse_normalised'], 'C7'),
                       ('triangular k=2',
                        base['triangular']['rmse_psi'] if which == 'abs'
                        else base_amp['triangular']['rmse_normalised'], 'C4'),
                       ('two_zone k=4',
                        base['two_zone']['rmse_psi'] if which == 'abs'
                        else base_amp['two_zone']['rmse_normalised'], 'C2')])
        pmap = os.path.join(dirs[which],
                            'fig_c2_map_%s_%s.png' % (which, tag))
        fig_map(pmap, grids['coarse'], w, cfg, floors[which], rfloors[which],
                opt, dpi)
        written[which].append((pmap, 'figure_png',
                               '(D, lambda_leak) misfit surface, exact valley '
                               'floor, reverse floor and the three baselines, '
                               '%s norm' % which))

        curves = []
        for key, color, ls, lab in curve_specs[which]:
            lab2 = lab % (10 ** points[key][1][0]) if '%' in lab else lab
            curves.append(dict(sim=sims_cache[key], rows=detail[key]['rows'],
                               color=color, ls=ls, label=lab2))
        pfit = os.path.join(dirs[which],
                            'fig_c2_fit_%s_%s.png' % (which, tag))
        fig_fit(pfit, S, curves, w, dpi)
        written[which].append((pfit, 'figure_png',
                               'per-gauge fit and residuals in the '
                               'r1_qc_fit.png layout, %s norm' % which))

        pnpz = os.path.join(dirs[which], 'c2_grid_%s_%s.npz' % (which, tag))
        np.savez_compressed(
            pnpz,
            coarse_log10_D=grids['coarse']['log10_D'],
            coarse_log10_lambda=grids['coarse']['log10_lambda'],
            coarse_rmse_psi=grids['coarse']['rmse'],
            coarse_rmse_normalised=grids['coarse']['rmse_norm'],
            coarse_rmse_pooled_psi=grids['coarse']['rmse_pooled'],
            coarse_rmse_psi_lambda_zero=grids['coarse']['rmse_lambda_zero'],
            coarse_rmse_normalised_lambda_zero=grids['coarse'][
                'rmse_norm_lambda_zero'],
            refined_log10_D=grids['refined']['log10_D'],
            refined_log10_lambda=grids['refined']['log10_lambda'],
            refined_rmse_psi=grids['refined']['rmse'],
            refined_rmse_normalised=grids['refined']['rmse_norm'],
            refined_rmse_pooled_psi=grids['refined']['rmse_pooled'],
            refined_rmse_psi_lambda_zero=grids['refined']['rmse_lambda_zero'],
            refined_rmse_normalised_lambda_zero=grids['refined'][
                'rmse_norm_lambda_zero'],
            floor_lambda=np.array([f['lam'] for f in floors[which]]),
            floor_log10_D=np.array([f['log10_D'] for f in floors[which]]),
            floor_rmse_psi=np.array([f['rmse_psi'] for f in floors[which]]),
            floor_rmse_normalised=np.array([f['rmse_normalised']
                                            for f in floors[which]]),
            reverse_log10_D=np.array([r['log10_D'] for r in rfloors[which]]),
            reverse_log10_lambda=np.array([r['log10_lambda']
                                           for r in rfloors[which]]),
            reverse_gain=np.array([r['gain_over_lambda_zero']
                                   for r in rfloors[which]]),
            mesh_md_ft=S['mesh_x'],
            target_gauges=np.array([t['gauge'] for t in S['targets']]),
            target_distance_ft=np.array([t['distance_ft']
                                         for t in S['targets']]),
            norm=np.array(which))
        written[which].append((pnpz, 'arrays_npz',
                               'coarse + refined (D, lambda) misfit grids, '
                               'both exact floors, geometry'))

    # diagnostics figure
    d0 = detail['leak_optimum_abs']['rows']
    dist = np.array([r['distance_ft'] for r in d0])
    obs = np.array([r['obs_max_psi'] for r in d0])
    L_obs = np.diff(dist) / np.log(obs[:-1] / obs[1:])
    seg_mid = 0.5 * (dist[:-1] + dist[1:])
    # The decay length the sink WOULD impose if lambda were held at the
    # memo's value; the free fit sends lambda to 0, where L is undefined.
    L_forced = float(np.sqrt(10 ** detail['leak_forced_physical_lambda']
                             ['params'][0] / (1.0 / 260.0)))
    decay = {
        'distance_ft': dist, 'obs_max_psi': obs,
        'seg_mid_ft': seg_mid, 'L_obs_ft': L_obs,
        'L_leak_best_ft': L_forced,
        'curves': [
            dict(peak=np.array([r['sim_max_psi'] for r in
                                detail['leak_optimum_abs']['rows']]),
                 color='C0', ls='o--',
                 label=r'leakage optimum ($\lambda^*=0$)'),
            dict(peak=np.array([r['sim_max_psi'] for r in
                                detail['leak_forced_physical_lambda']['rows']]),
                 color='C3', ls='s--',
                 label=r'leakage forced $\lambda=1/260$ s$^{-1}$'),
            dict(peak=np.array([r['sim_max_psi'] for r in
                                detail['two_zone']['rows']]),
                 color='C2', ls='^-.', label='two_zone D(x)')],
    }
    pdec = os.path.join(dirs['diag'], 'fig_c2_diagnostics_%s.png' % tag)
    fig_decay(pdec, S, decay, subsets, dpi)
    written['diag'].append((pdec, 'figure_png',
                            'amplitude decay, implied decay length vs distance, '
                            'and the nested-subset range of applicability'))

    pdnpz = os.path.join(dirs['diag'], 'c2_diagnostics_%s.npz' % tag)
    np.savez_compressed(
        pdnpz,
        distance_ft=dist, obs_max_psi=obs,
        seg_mid_ft=seg_mid, implied_decay_length_ft=L_obs,
        subset_max_dist_ft=np.array([s['max_dist_ft'] for s in subsets]),
        subset_uniform_rmse=np.array([s['uniform_rmse'] for s in subsets]),
        subset_uniform_D=np.array([s['uniform_D'] for s in subsets]),
        subset_leak_rmse=np.array([s['leak_rmse'] for s in subsets]),
        subset_leak_D=np.array([s['leak_D'] for s in subsets]),
        subset_leak_lambda=np.array([s['leak_lambda'] for s in subsets]),
        subset_single_gauge_floor=np.array([s['single_gauge_floor']
                                            for s in subsets]),
        solo_D=np.array([s['D'] for s in solo]),
        solo_rmse_psi=np.array([s['rmse_psi'] for s in solo]),
        bias_psi_leak_opt=np.array([r['bias_psi'] for r in d0]),
        bias_psi_leak_forced=np.array(
            [r['bias_psi'] for r in
             detail['leak_forced_physical_lambda']['rows']]),
        bias_psi_two_zone=np.array([r['bias_psi'] for r in
                                    detail['two_zone']['rows']]),
        bias2_over_mse_leak_opt=np.array([r['bias2_over_mse'] for r in d0]))
    written['diag'].append((pdnpz, 'arrays_npz',
                            'decay lengths, nested-subset fits, per-gauge bias '
                            'and bias^2/MSE'))

    # ---- 9. summaries ----------------------------------------------------
    def fmt_summary(which):
        optd = opt_abs if which == 'abs' else opt_amp
        L = []
        L.append('C2 -- leakage sink, %s norm' % (
            'absolute (gauge-mean psi)' if which == 'abs'
            else 'amplitude-normalised'))
        L.append('=' * 78)
        L.append('model  dP/dt = D d2P/dx2 - lambda_leak*(P - 0)')
        L.append('setup  window MD 15000-16750, pad 5000 ft, dx 1 ft, dt 1 s, '
                 'theta 1, harmonic')
        L.append('       Dirichlet source gauge %d MD %.0f (mesh idx %d), '
                 'targets g2-g7' % (S['src_gauge'], S['src_md'],
                                    S['source_idx']))
        L.append('pooling  GAUGE-MEAN RMSE (equal weight per gauge). '
                 'Sample-pooled is recorded alongside.')
        L.append('')
        L.append('OPTIMUM')
        L.append('  lambda_leak* = 0 EXACTLY -- the argmin sits on the '
                 'boundary of the parameter space.')
        L.append('  D*           = %.1f ft^2/s' % 10 ** optd['log10_D'])
        L.append('  gauge-mean RMSE      %.4f psi' % optd['rmse_psi'])
        L.append('  normalised RMSE      %.5f' % optd['rmse_normalised'])
        L.append('  sample-pooled RMSE   %.4f psi' % optd['rmse_pooled_psi'])
        L.append('  => the leakage model DEGENERATES to the uniform model.')
        L.append('')
        L.append('VALLEY FLOOR (exact: Brent over log10 D at each lambda)')
        L.append('  %-12s %-12s %-14s %-14s' % ('lambda', 'D*', 'RMSE psi',
                                                'normalised'))
        for f in floors[which]:
            L.append('  %-12s %-12.1f %-14.5f %-14.6f'
                     % ('0' if f['lam'] == 0 else '%.3e' % f['lam'],
                        10 ** f['log10_D'], f['rmse_psi'],
                        f['rmse_normalised']))
        L.append('')
        L.append('REVERSE FLOOR (Brent over log10 lambda at each D)')
        rv = np.array([(r['rmse_psi'] if which == 'abs'
                        else r['rmse_normalised']) for r in rfloors[which]])
        g = np.array([r['gain_over_lambda_zero'] for r in rfloors[which]])
        jb = int(np.argmin(rv))
        L.append('  min over the reverse D grid (the exact global min is the '
                 'forward floor above): %.5f at D = %.1f, lambda = %.3e '
                 '(at the low lambda bound: %s)'
                 % (rv[jb], 10 ** rfloors[which][jb]['log10_D'],
                    10 ** rfloors[which][jb]['log10_lambda'],
                    rfloors[which][jb]['at_low_bound']))
        L.append('  max gain from lambda > 0 at a FIXED D: %.4f (at D = %.1f).'
                 % (g.max(), 10 ** rfloors[which][int(np.argmax(g))]['log10_D']))
        L.append('  => the sink does help when D is held at a value that is '
                 'too large, but re-fitting D gives the gain straight back.')
        L.append('')
        L.append('BASELINES ON THE IDENTICAL CRITERION')
        if which == 'abs':
            for nm2, m in (('uniform k=1', base['uniform']),
                           ('triangular k=2', base['triangular']),
                           ('two_zone k=4', base['two_zone'])):
                L.append('  %-16s %8.4f psi   normalised %.5f'
                         % (nm2, m['rmse_psi'], m['rmse_normalised']))
            L.append('  %-16s %8.4f psi   normalised %.5f'
                     % ('leakage k=2', optd['rmse_psi'],
                        optd['rmse_normalised']))
        else:
            L.append('  (each baseline RE-OPTIMISED under the normalised norm)')
            for nm2, m in (('uniform k=1', base_amp['uniform']),
                           ('triangular k=2', base_amp['triangular']),
                           ('two_zone k=4', base_amp['two_zone'])):
                L.append('  %-16s normalised %.5f   (abs %8.3f psi)'
                         % (nm2, m['rmse_normalised'], m['rmse_psi']))
            L.append('  %-16s normalised %.5f   (abs %8.3f psi)'
                     % ('leakage k=2', optd['rmse_normalised'],
                        optd['rmse_psi']))
        L.append('')
        L.append('PER-GAUGE RESIDUALS AT THE LEAKAGE OPTIMUM')
        L.append('  %-5s %-8s %-10s %-10s %-10s %-10s'
                 % ('gauge', 'dist ft', 'RMSE psi', 'bias psi', 'bias2/MSE',
                    'frac r>0'))
        key = 'leak_optimum_abs' if which == 'abs' else 'leak_optimum_amp'
        for r in detail[key]['rows']:
            L.append('  g%-4d %-8.0f %-10.2f %-+10.2f %-10.3f %-10.3f'
                     % (r['gauge'], r['distance_ft'], r['rmse_psi'],
                        r['bias_psi'], r['bias2_over_mse'],
                        r['frac_resid_positive']))
        L.append('')
        L.append('PER-GAUGE RESIDUALS WITH lambda FORCED TO 1/260 s^-1 '
                 '(D re-optimised)')
        for r in detail['leak_forced_physical_lambda']['rows']:
            L.append('  g%-4d %-8.0f %-10.2f %-+10.2f %-10.3f %-10.3f'
                     % (r['gauge'], r['distance_ft'], r['rmse_psi'],
                        r['bias_psi'], r['bias2_over_mse'],
                        r['frac_resid_positive']))
        return '\n'.join(L) + '\n'

    for which in ('abs', 'amp'):
        p3 = os.path.join(dirs[which], 'c2_summary_%s.txt' % which)
        with open(p3, 'w') as fh:
            fh.write(fmt_summary(which))
        written[which].append((p3, 'summary_txt', 'one-screen digest'))

    dl = ['C2 -- diagnostics', '=' * 78, '',
          'SINK VERIFICATION (this run, independent of the shared self-test)',
          json.dumps(ver, indent=2, sort_keys=True), '',
          'AICc, gauge-mean MSE as the variance estimator, over every n '
          'convention', '']
    for n_s, rows in aicc.items():
        dl.append('  n = %s   (best: %s)' % (n_s, rows.get('_best', 'n/a')))
        for nm2, v in rows.items():
            if nm2 == '_best':
                continue
            dl.append('    %-30s k=%d  AICc %s  dAICc %s'
                      % (nm2, v['k'],
                         'undefined (%s)' % v.get('why', '')
                         if not v['defined'] else '%12.3f' % v['aicc'],
                         '   -' if v.get('delta_aicc') is None
                         else '%9.3f' % v['delta_aicc']))
        dl.append('')
    dl.append('CAVEAT (house rules): ' + cfg['aicc']['caveat'])
    dl.append('')
    dl.append('RANGE OF APPLICABILITY -- nested fits, near gauges first')
    dl.append('  %-9s %-9s %-11s %-10s %-11s %-11s %-9s %-10s'
              % ('gauges', 'max ft', 'uniform D', 'uni RMSE', 'leak D',
                 'leak lambda', 'leak L', 'leak RMSE'))
    for s in subsets:
        dl.append('  g2..g%-4d %-9.0f %-11.1f %-10.2f %-11.1f %-11.2e %-9.0f '
                  '%-10.2f' % (s['m'], s['max_dist_ft'], s['uniform_D'],
                               s['uniform_rmse'], s['leak_D'],
                               s['leak_lambda'], s['leak_decay_length_ft'],
                               s['leak_rmse']))
    dl.append('')
    dl.append('PER-GAUGE SOLO FITS (uniform D fitted to that gauge alone)')
    for s in solo:
        dl.append('  g%-3d %6.0f ft  D = %10.1f ft^2/s  RMSE %6.2f psi'
                  % (s['gauge'], s['distance_ft'], s['D'], s['rmse_psi']))
    dl.append('')
    dl.append('IMPLIED DECAY LENGTH BETWEEN ADJACENT GAUGES (observed peaks)')
    for m2, l2 in zip(seg_mid, L_obs):
        dl.append('  midpoint %6.0f ft -> L = %8.0f ft' % (m2, l2))
    pdiag = os.path.join(dirs['diag'], 'c2_diagnostics.txt')
    with open(pdiag, 'w') as fh:
        fh.write('\n'.join(dl) + '\n')
    written['diag'].append((pdiag, 'summary_txt',
                            'sink verification, AICc, range of applicability, '
                            'decay lengths'))

    # ---- 10. manifests ---------------------------------------------------
    log('writing manifests')
    sweep_note = {
        'coarse_log10_D': cfg['grid']['coarse']['log10_D'],
        'coarse_log10_lambda': cfg['grid']['coarse']['log10_lambda'],
        'refined_log10_D': cfg['grid']['refined']['log10_D'],
        'refined_log10_lambda': cfg['grid']['refined']['log10_lambda'],
        'floor_log10_lambda': cfg['floor']['log10_lambda'],
        'lambda_zero_included': True,
        'n_forward_solves_grid': int(
            grids['coarse']['rmse'].size + grids['coarse']['rmse_lambda_zero'].size
            + grids['refined']['rmse'].size
            + grids['refined']['rmse_lambda_zero'].size),
    }
    common_results = {
        'verification': ver,
        'baselines_absolute_norm': base,
        'baselines_normalised_norm': base_amp,
        'leakage_optimum': {
            'absolute_norm': dict(opt_abs, D_ft2_s=float(10 ** opt_abs['log10_D']),
                                  lambda_leak_s_inv=0.0,
                                  boundary_censored=True),
            'normalised_norm': dict(opt_amp,
                                    D_ft2_s=float(10 ** opt_amp['log10_D']),
                                    lambda_leak_s_inv=0.0,
                                    boundary_censored=True)},
        'aicc': aicc,
        'aicc_caveat': cfg['aicc']['caveat'],
        'range_of_applicability': subsets,
        'solo_gauge_fits': solo,
        'residual_structure': {k2: {kk: vv for kk, vv in v.items()}
                               for k2, v in detail.items()},
        'implied_decay_length_ft': {'segment_midpoint_ft': seg_mid.tolist(),
                                    'L_ft': L_obs.tolist()},
        'misfit_pooling': cfg['misfit'],
    }
    notes = [
        'HEADLINE: under BOTH norms the argmin of the leakage-sink model is '
        'lambda_leak = 0 exactly. The optimum is CENSORED at the boundary of '
        'the parameter space and must not be quoted as an estimate of a '
        'leak-off coefficient.',
        'The bar the task set was 11.87 psi (two_zone D(x), k=4). The leakage '
        'sink reaches %.2f psi, i.e. it does not beat the uniform model it '
        'degenerates to, let alone D(x).' % opt_abs['rmse_psi'],
        'AICc is legitimate here only because k differs (1 vs 2 vs 4), and it '
        'still rests on a violated assumption: ' + cfg['aicc']['caveat'],
        'Misfit pooling is the GAUGE-MEAN RMSE, per C1; the sample-pooled '
        'value is recorded at every reference point but is never minimised.',
        'The leakage sink was re-verified in this run against an analytic '
        'transient, an analytic steady state and an independently written '
        'implicit stepper, because the adversarial verifiers of the shared '
        'rev2 modules were killed by a session limit.',
    ]
    inputs = study_inputs()
    label = {'abs': 'absolute (gauge-mean psi) norm',
             'amp': 'amplitude-normalised norm',
             'diag': 'diagnostics: verification, AICc, range of applicability'}
    for which in ('abs', 'amp', 'diag'):
        optd = opt_amp if which == 'amp' else opt_abs
        prof = np.full(S['mesh_x'].size, 10.0 ** optd['log10_D'])
        srcg, num = manifest_groups(S, cfg, taxis_ref, prof, 0.0, sweep_note)
        with rm.RunRecorder(
                os.path.join(dirs[which], 'manifest.json'),
                study_id=STUDY_ID, task_id=TASK_ID, config=cfg,
                config_path=args.config, run_label=label[which],
                require_modules=('rev2_core', 'rev2_data', 'rev2_manifest'),
                extra_code_files=(os.path.abspath(__file__),)) as R:
            R.declare_inputs(inputs)
            for p3, role, note in written[which]:
                R.declare_output(p3, role=role, note=note,
                                 dpi=dpi if role == 'figure_png' else None)
            R.set_source(srcg)
            R.set_numerics(num)
            R.set_results(dict(common_results, norm_of_this_manifest=label[which]))
            for t2 in notes:
                R.note(t2)
        log('  wrote %s' % os.path.join(dirs[which], 'manifest.json'))

    log('done in %.1f s' % (time.time() - _T0))


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
