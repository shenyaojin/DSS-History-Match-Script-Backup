"""Core numerics for the R1 baseline-diffusivity calibration study.

The forward kernel here reproduces fibeRIS PDS1D_SingleSource exactly: the same
harmonic-mean face diffusivity, the same backward-Euler step, the same
Neumann/Neumann rows, and the same Dirichlet source row evaluated at time level
n. Because that discretisation is tridiagonal apart from the source row, it is
solved with solve_banded instead of a dense np.linalg.solve, which is what makes
a 60-point sweep affordable. verify_against_fiberis() below is the guard: it
must pass before any sweep result is trusted.
"""

import numpy as np
from scipy.linalg import solve_banded


def build_uniform_profile(mesh, d_value):
    return np.full(len(mesh), float(d_value))


def build_triangular_profile(mesh, d_max, d_min_ratio, source_idx,
                             taper_lo_md=None, taper_hi_md=None):
    """Triangular profile: peak d_max at the source, decaying to d_max*d_min_ratio.

    The taper endpoints are PHYSICAL measured depths, not mesh ends. Tapering to
    the mesh ends would make the model mesh-dependent: once the domain is padded
    for boundary reasons, the same d_max would imply a different diffusivity at
    the gauges (a 5000 ft pad moves D at the farthest gauge by ~4x). Pinning the
    taper to fixed MDs keeps the parameter physical and comparable across
    padding choices. Outside [taper_lo_md, taper_hi_md] the profile is clamped
    to d_min.

    104r's own profile tapered to its mesh ends, but its mesh WAS the data
    window, so pinning to the comparison window reproduces its intent.
    """
    mesh = np.asarray(mesh, dtype=float)
    d_max = float(d_max)
    d_min = d_max * float(d_min_ratio)
    src_md = mesh[source_idx]
    lo = float(mesh[0]) if taper_lo_md is None else float(taper_lo_md)
    hi = float(mesh[-1]) if taper_hi_md is None else float(taper_hi_md)

    prof = np.full(len(mesh), d_min, dtype=float)
    left = mesh <= src_md
    if src_md > lo:
        frac = np.clip((mesh[left] - lo) / (src_md - lo), 0.0, 1.0)
        prof[left] = d_min + (d_max - d_min) * frac
    else:
        prof[left] = d_max
    right = mesh > src_md
    if hi > src_md:
        frac = np.clip((hi - mesh[right]) / (hi - src_md), 0.0, 1.0)
        prof[right] = d_min + (d_max - d_min) * frac
    else:
        prof[right] = d_max
    return prof


def _face_diffusivity(diffusivity):
    return (2.0 * diffusivity[:-1] * diffusivity[1:]
            / (diffusivity[:-1] + diffusivity[1:]))


def _alpha(mesh, diffusivity, dt):
    dx = np.diff(mesh)
    d_eff = _face_diffusivity(diffusivity)
    dxm, dxp = dx[:-1], dx[1:]
    alpha_l = d_eff[:-1] * dt / (dxm * (dxm + dxp) / 2.0)
    alpha_r = d_eff[1:] * dt / (dxp * (dxm + dxp) / 2.0)
    return alpha_l, alpha_r


def solve_forward(mesh, diffusivity, dt, t_total, source_taxis, source_data,
                  source_idx, initial=None, t0=0.0, record_idx=None):
    """Backward-Euler diffusion with a Dirichlet source node.

    Returns (taxis, recorded) where recorded is (n_time, len(record_idx)) if
    record_idx is given, else the full (n_time, nx) field.
    """
    nx = len(mesh)
    alpha_l, alpha_r = _alpha(mesh, np.asarray(diffusivity, dtype=float), dt)

    # Banded storage: ab[0]=super-diagonal, ab[1]=diagonal, ab[2]=sub-diagonal.
    ab = np.zeros((3, nx))
    ab[1, 1:nx - 1] = 1.0 + alpha_l + alpha_r
    ab[0, 2:nx] = -alpha_r
    ab[2, 0:nx - 2] = -alpha_l
    # Neumann rows, exactly as fibeRIS matbuilder writes them.
    ab[1, 0] = -1.0
    ab[0, 1] = 1.0
    ab[1, nx - 1] = -1.0
    ab[2, nx - 2] = 1.0
    # Dirichlet source row: zero the row, put 1 on the diagonal.
    ab[1, source_idx] = 1.0
    if source_idx + 1 < nx:
        ab[0, source_idx + 1] = 0.0
    if source_idx - 1 >= 0:
        ab[2, source_idx - 1] = 0.0

    u = np.zeros(nx) if initial is None else np.asarray(initial, dtype=float).copy()
    taxis = [t0]
    keep = np.arange(nx) if record_idx is None else np.asarray(record_idx)
    out = [u[keep].copy()]

    t = t0
    while t < t_total:
        # fibeRIS evaluates the source at the current (old) time level.
        src = float(np.interp(t - t0, source_taxis, source_data))
        b = u.copy()
        b[0] = 0.0
        b[-1] = 0.0
        b[source_idx] = src
        u = solve_banded((1, 1), ab, b)
        t += dt
        taxis.append(t)
        out.append(u[keep].copy())

    return np.asarray(taxis), np.asarray(out)


def verify_against_fiberis(mesh, diffusivity, dt, t_total, gauge_dataframe,
                           source_idx):
    """Run both kernels on identical inputs and compare.

    gauge_dataframe must already carry the series the study uses (baseline
    removed, taxis rebased to 0), because absolute round-off scales with the
    magnitude of the field being solved for. Returns absolute and
    field-normalised max differences plus the dense-matrix residuals, which is
    what actually establishes which solve is the accurate one.
    """
    from fiberis.simulator.core import pds

    sim = pds.PDS1D_SingleSource()
    sim.set_mesh(mesh)
    sim.set_bcs('Neumann', 'Neumann')
    sim.set_t0(0)
    sim.set_initial(np.zeros_like(mesh))
    sim.set_diffusivity(np.asarray(diffusivity, dtype=float))
    sim.set_sourceidx(source_idx)
    sim.set_source(gauge_dataframe)
    sim.solve(optimizer=False, dt=dt, t_total=t_total)

    _, mine = solve_forward(mesh, diffusivity, dt, t_total,
                            np.asarray(gauge_dataframe.taxis, dtype=float),
                            np.asarray(gauge_dataframe.data, dtype=float),
                            source_idx)
    n = min(len(sim.snapshot), len(mine))
    abs_diff = float(np.max(np.abs(sim.snapshot[:n] - mine[:n])))
    scale = float(np.max(np.abs(sim.snapshot[:n]))) or 1.0
    return {
        'max_abs_diff_psi': abs_diff,
        'field_max_abs_psi': scale,
        'max_relative_diff': abs_diff / scale,
        'n_steps_compared': int(n),
    }


def dense_residual_check(mesh, diffusivity, dt, gauge_dataframe, source_idx):
    """One backward-Euler step solved both ways; report ||Au-b|| for each.

    A zero residual for both means the two kernels solve the same linear system
    exactly, so any difference between full runs is accumulated round-off rather
    than a different discretisation.
    """
    from fiberis.simulator.core import pds
    from fiberis.simulator.solver import matbuilder, PDESolver_IMP
    from scipy.linalg import solve_banded as _sb

    sim = pds.PDS1D_SingleSource()
    sim.set_mesh(mesh)
    sim.set_bcs('Neumann', 'Neumann')
    sim.set_t0(0)
    sim.set_diffusivity(np.asarray(diffusivity, dtype=float))
    sim.set_sourceidx(source_idx)
    sim.set_source(gauge_dataframe)
    sim.snapshot = [np.zeros_like(mesh)]
    sim.taxis = [0.0]
    A, b = matbuilder.matrix_builder_1d_single_source(sim, dt)

    nx = len(mesh)
    tri = (np.diag(np.diag(A)) + np.diag(np.diag(A, 1), 1)
           + np.diag(np.diag(A, -1), -1))
    ab = np.zeros((3, nx))
    ab[1] = np.diag(A)
    ab[0, 1:] = np.diag(A, 1)
    ab[2, :-1] = np.diag(A, -1)

    u_dense = PDESolver_IMP.solver_implicit(A, b, solver='numpy')
    u_band = _sb((1, 1), ab, b)
    return {
        'matrix_is_exactly_tridiagonal': bool(np.array_equal(A, tri)),
        'cond_A': float(np.linalg.cond(A)),
        'residual_inf_dense': float(np.max(np.abs(A @ u_dense - b))),
        'residual_inf_banded': float(np.max(np.abs(A @ u_band - b))),
        'max_abs_diff_one_step_psi': float(np.max(np.abs(u_dense - u_band))),
    }


def metric_equivalence_check(mesh, diffusivity, dt, t_total, gauge_dataframe,
                             source_idx, targets, threshold_frac):
    """Compare the scored misfit metric itself between fibeRIS and the fast kernel.

    This is the gate that matters: round-off differences in the field are only a
    problem if they move the number the study reports.
    """
    from fiberis.simulator.core import pds

    sim = pds.PDS1D_SingleSource()
    sim.set_mesh(mesh)
    sim.set_bcs('Neumann', 'Neumann')
    sim.set_t0(0)
    sim.set_initial(np.zeros_like(mesh))
    sim.set_diffusivity(np.asarray(diffusivity, dtype=float))
    sim.set_sourceidx(source_idx)
    sim.set_source(gauge_dataframe)
    sim.solve(optimizer=False, dt=dt, t_total=t_total)
    ref_taxis = np.asarray(sim.taxis, dtype=float)

    def pooled_rmse(taxis, field_at_targets):
        sq, n = 0.0, 0
        for k, tgt in enumerate(targets):
            resid = np.interp(tgt['taxis'], taxis, field_at_targets[:, k]) - tgt['data']
            sq += float(np.sum(resid ** 2))
            n += resid.size
        return float(np.sqrt(sq / n))

    idx = [t['idx'] for t in targets]
    rmse_fiberis = pooled_rmse(ref_taxis, np.asarray(sim.snapshot)[:, idx])
    taxis2, rec2 = solve_forward(mesh, diffusivity, dt, t_total,
                                 np.asarray(gauge_dataframe.taxis, dtype=float),
                                 np.asarray(gauge_dataframe.data, dtype=float),
                                 source_idx, record_idx=idx)
    rmse_fast = pooled_rmse(taxis2, rec2)
    return {
        'dt_s': dt,
        't_total_s': t_total,
        'rmse_fiberis_psi': rmse_fiberis,
        'rmse_fast_psi': rmse_fast,
        'abs_rmse_difference_psi': abs(rmse_fiberis - rmse_fast),
    }


def arrival_time(taxis, series, threshold):
    """First time the series crosses threshold, linearly interpolated. NaN if never."""
    above = np.where(series >= threshold)[0]
    if above.size == 0:
        return np.nan
    i = above[0]
    if i == 0:
        return float(taxis[0])
    y0, y1 = series[i - 1], series[i]
    if y1 == y0:
        return float(taxis[i])
    frac = (threshold - y0) / (y1 - y0)
    return float(taxis[i - 1] + frac * (taxis[i] - taxis[i - 1]))


def evaluate_profile(mesh, diffusivity, dt, t_total, source_taxis, source_data,
                     source_idx, targets, threshold_frac):
    """Forward-model once and score against every target gauge.

    targets: list of dicts with keys 'idx' (mesh index), 'taxis', 'data' (obs delta-P).
    Returns a dict of pooled and per-gauge metrics.
    """
    record_idx = [t['idx'] for t in targets]
    taxis, rec = solve_forward(mesh, diffusivity, dt, t_total, source_taxis,
                               source_data, source_idx, record_idx=record_idx)

    per_gauge = []
    sq_pool, n_pool = 0.0, 0
    for k, tgt in enumerate(targets):
        sim_on_obs = np.interp(tgt['taxis'], taxis, rec[:, k])
        resid = sim_on_obs - tgt['data']
        sq_pool += float(np.sum(resid ** 2))
        n_pool += resid.size

        obs_max = float(np.max(tgt['data']))
        sim_max = float(np.max(sim_on_obs))
        thr = threshold_frac * obs_max
        t_obs = arrival_time(tgt['taxis'], tgt['data'], thr)
        t_sim = arrival_time(tgt['taxis'], sim_on_obs, thr)
        per_gauge.append({
            'gauge': tgt['gauge'],
            'md_ft': tgt['md_ft'],
            'distance_ft': tgt['distance_ft'],
            'rmse_psi': float(np.sqrt(np.mean(resid ** 2))),
            'bias_psi': float(np.mean(resid)),
            'obs_max_psi': obs_max,
            'sim_max_psi': sim_max,
            'amplitude_ratio': sim_max / obs_max if obs_max != 0 else np.nan,
            'arrival_obs_s': t_obs,
            'arrival_sim_s': t_sim,
            'arrival_err_s': t_sim - t_obs,
        })

    arr_err = np.array([g['arrival_err_s'] for g in per_gauge], dtype=float)
    amp = np.array([g['amplitude_ratio'] for g in per_gauge], dtype=float)
    # Normalised misfit: each gauge's RMSE divided by its own observed
    # amplitude, so the high-amplitude near gauges do not dominate. With a
    # misspecified model the two norms need not agree, and the disagreement is
    # itself a reportable result.
    norm = np.array([g['rmse_psi'] / g['obs_max_psi'] if g['obs_max_psi'] else np.nan
                     for g in per_gauge], dtype=float)
    return {
        'rmse_pooled_psi': float(np.sqrt(sq_pool / n_pool)),
        'rmse_normalized': float(np.sqrt(np.nanmean(norm ** 2))),
        'arrival_err_mean_s': float(np.nanmean(arr_err)),
        'arrival_err_absmean_s': float(np.nanmean(np.abs(arr_err))),
        'amplitude_ratio_mean': float(np.nanmean(amp)),
        'per_gauge': per_gauge,
        'n_residuals': int(n_pool),
    }


def erfc_implied_diffusivity(distance_ft, amp_ratio, elapsed_s):
    """D that makes erfc(x / (2 sqrt(D t))) equal the observed amplitude ratio.

    A solver-free cross-check on the single-gauge diagnostic. The erfc solution
    assumes a step change at a semi-infinite boundary whereas the real source
    ramps, so the absolute values are indicative only -- it is the spread across
    gauges that carries the information. Returns NaN when no root exists.
    """
    from scipy.optimize import brentq
    from scipy.special import erfc

    if not (0.0 < amp_ratio < 1.0):
        return float('nan')
    f = lambda d: erfc(distance_ft / (2.0 * np.sqrt(d * elapsed_s))) - amp_ratio
    try:
        return float(brentq(f, 1e-3, 1e9))
    except (ValueError, RuntimeError):
        return float('nan')


def arrival_robustness(taxis_sim, sim_series, taxis_obs, obs_series,
                       absolute_thresholds_psi, relative_fraction):
    """Arrival-time error under a gauge-relative and several absolute thresholds.

    Guards against the sign pattern being an artifact of scaling the threshold
    to each gauge's own amplitude.
    """
    sim_on_obs = np.interp(taxis_obs, taxis_sim, sim_series)
    out = {}
    rel_thr = relative_fraction * float(np.max(obs_series))
    for label, thr in [('relative', rel_thr)] + [
            (f'abs_{t:g}psi', float(t)) for t in absolute_thresholds_psi]:
        t_obs = arrival_time(taxis_obs, obs_series, thr)
        t_sim = arrival_time(taxis_obs, sim_on_obs, thr)
        out[label] = float(t_sim - t_obs)
    return out


def exact_gauge_bootstrap(grid, per_gauge_sq):
    """Exact gauge-level bootstrap distribution of the argmin.

    With n gauges there are only C(2n-1, n) distinct resample multisets, so the
    bootstrap can be enumerated with exact multinomial weights instead of
    sampled. That removes Monte-Carlo error entirely -- which matters because a
    finite-draw estimate can report a probability of exactly 0 or 1 for an event
    that in fact has support (e.g. "100% of argmins lie above D=480").

    per_gauge_sq: (n_grid, n_gauges) squared per-gauge misfit.
    Returns the weighted argmin distribution and its exact quantiles.
    """
    from itertools import combinations_with_replacement
    from math import factorial

    grid = np.asarray(grid, dtype=float)
    n_g = per_gauge_sq.shape[1]
    total = float(n_g) ** n_g

    argmins, weights = [], []
    for combo in combinations_with_replacement(range(n_g), n_g):
        counts = np.bincount(combo, minlength=n_g)
        w = factorial(n_g)
        for c in counts:
            w //= factorial(int(c))
        curve = np.sqrt(per_gauge_sq @ counts / float(n_g))
        argmins.append(float(grid[int(np.argmin(curve))]))
        weights.append(w / total)

    argmins = np.asarray(argmins)
    weights = np.asarray(weights)
    order = np.argsort(argmins)
    a_sorted, w_sorted = argmins[order], weights[order]
    cum = np.cumsum(w_sorted)

    def q(p):
        i = int(np.searchsorted(cum, p, side='left'))
        return float(a_sorted[min(i, len(a_sorted) - 1)])

    return {
        'n_multisets': int(len(argmins)),
        'total_weight': float(weights.sum()),
        'argmins': argmins,
        'weights': weights,
        'ci95': [q(0.025), q(0.975)],
        'ci68': [q(0.16), q(0.84)],
        'median': q(0.5),
        'mass_at_grid_min': float(weights[argmins <= grid[0]].sum()),
        'mass_at_grid_max': float(weights[argmins >= grid[-1]].sum()),
    }


def exact_fraction_above(boot, value):
    """Exact bootstrap probability that the argmin exceeds `value`."""
    return float(boot['weights'][boot['argmins'] > float(value)].sum())


def file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def interval_method_spread(grid, curve, per_gauge_sq, n_conventions,
                           reference_values=()):
    """Confidence intervals for D under several defensible conventions.

    The +10% misfit band is a convention with no stated statistical content: what
    confidence it corresponds to depends entirely on the assumed sample size. So
    report the spread across methods instead of one number, and record where each
    reference value falls under each. p=1 free parameter throughout.

    n_conventions: {label: n} e.g. {'gauges': 6, 'autocorr_neff': 34,
    'pooled': 3925}.
    """
    from scipy.stats import f as f_dist

    grid = np.asarray(grid, dtype=float)
    curve = np.asarray(curve, dtype=float)
    i = int(np.argmin(curve))
    best, best_rmse = float(grid[i]), float(curve[i])
    ssr_min = best_rmse ** 2

    out = {}
    for label, n in n_conventions.items():
        n = int(n)
        if n <= 1:
            continue
        p = 1
        crit = 1.0 + (p / float(n - p)) * float(f_dist.ppf(0.95, p, n - p))
        inside = np.where(curve ** 2 / ssr_min <= crit)[0]
        lo, hi = float(grid[inside[0]]), float(grid[inside[-1]])
        refs = {}
        for rv in reference_values:
            if grid[0] <= rv <= grid[-1]:
                ratio = float(np.interp(rv, grid, curve)) ** 2 / ssr_min
                f_stat = (ratio - 1.0) * (n - p) / p
                refs[str(rv)] = {
                    'inside_95': bool(lo <= rv <= hi),
                    'p_value': float(1.0 - f_dist.cdf(max(f_stat, 0.0), p, n - p)),
                }
        out[label] = {
            'n': n,
            'rmse_ratio_at_95': float(np.sqrt(crit)),
            'percent_rmse_rise_at_95': float((np.sqrt(crit) - 1.0) * 100.0),
            'interval': [lo, hi],
            'open_low': bool(inside[0] == 0),
            'open_high': bool(inside[-1] == len(grid) - 1),
            'reference_values': refs,
        }

    # Jackknife on log10(argmin), leaving out one gauge at a time.
    n_g = per_gauge_sq.shape[1]
    loo = []
    for drop in range(n_g):
        keep = [c for c in range(n_g) if c != drop]
        loo.append(np.log10(grid[int(np.argmin(
            np.sqrt(np.mean(per_gauge_sq[:, keep], axis=1))))]))
    loo = np.asarray(loo)
    full = np.log10(best)
    pseudo = n_g * full - (n_g - 1) * loo
    se = float(np.sqrt(np.var(pseudo, ddof=1) / n_g))
    out['jackknife_log10'] = {
        'n': n_g,
        'se_log10_D': se,
        'interval': [float(10 ** (full - 1.96 * se)),
                     float(10 ** (full + 1.96 * se))],
    }
    out['best'] = best
    out['best_rmse'] = best_rmse
    return out


# ---------------------------------------------------------------------------
# R2: parametric D(x) families.
#
# R1 established that no single D describes the window and that the 104r
# triangular family is too stiff (its free taper ratio runs to the grid edge
# without a turning point). These families let the data choose the SHAPE of the
# decay rather than being forced into a fixed one. All are written as functions
# of s = |MD - MD_source| so the parameters are physical distances, independent
# of mesh and padding.
# ---------------------------------------------------------------------------

def _distance_from_source(mesh, source_idx):
    return np.abs(np.asarray(mesh, dtype=float) - float(mesh[source_idx]))


def profile_uniform(mesh, source_idx, p):
    """p = [log10 D]"""
    return np.full(len(mesh), 10.0 ** p[0])


def profile_exponential(mesh, source_idx, p):
    """D(s) = D_far + (D_src - D_far) * exp(-s / L);  p = [log10 D_src, log10 D_far, log10 L]"""
    d_src, d_far, L = 10.0 ** p[0], 10.0 ** p[1], 10.0 ** p[2]
    s = _distance_from_source(mesh, source_idx)
    return d_far + (d_src - d_far) * np.exp(-s / L)


def profile_powerlaw(mesh, source_idx, p):
    """D(s) = D_src * (1 + s/s0)^(-q), floored at D_far;
    p = [log10 D_src, log10 s0, q, log10 D_far]"""
    d_src, s0, q, d_far = 10.0 ** p[0], 10.0 ** p[1], p[2], 10.0 ** p[3]
    s = _distance_from_source(mesh, source_idx)
    return np.maximum(d_src * (1.0 + s / s0) ** (-abs(q)), d_far)


def profile_two_zone(mesh, source_idx, p):
    """Sharp near/far contrast with a smooth (tanh) transition.
    p = [log10 D_near, log10 D_far, log10 s_c, log10 width]"""
    d_n, d_f, sc, w = 10.0 ** p[0], 10.0 ** p[1], 10.0 ** p[2], 10.0 ** p[3]
    s = _distance_from_source(mesh, source_idx)
    frac = 0.5 * (1.0 + np.tanh((s - sc) / max(w, 1e-6)))
    return d_n * (1.0 - frac) + d_f * frac


def profile_triangular_window(mesh, source_idx, p, taper_lo=None, taper_hi=None):
    """104r form kept for comparison; p = [log10 D_max, log10 ratio]"""
    return build_triangular_profile(mesh, 10.0 ** p[0], 10.0 ** p[1], source_idx,
                                    taper_lo, taper_hi)


PROFILE_FAMILIES = {
    'uniform':      {'fn': profile_uniform,      'k': 1,
                     'names': ['log10_D']},
    'triangular':   {'fn': profile_triangular_window, 'k': 2,
                     'names': ['log10_D_max', 'log10_ratio']},
    'exponential':  {'fn': profile_exponential,  'k': 3,
                     'names': ['log10_D_src', 'log10_D_far', 'log10_L']},
    'powerlaw':     {'fn': profile_powerlaw,     'k': 4,
                     'names': ['log10_D_src', 'log10_s0', 'q', 'log10_D_far']},
    'two_zone':     {'fn': profile_two_zone,     'k': 4,
                     'names': ['log10_D_near', 'log10_D_far', 'log10_s_c',
                               'log10_width']},
}


def misfit_for_profile(prof, mesh, dt, t_total, source_taxis, source_data,
                       source_idx, targets):
    """Gauge-mean RMSE (psi) and amplitude-normalised RMSE for one D(x)."""
    if not np.all(np.isfinite(prof)) or np.any(prof <= 0):
        return np.inf, np.inf
    taxis, rec = solve_forward(mesh, prof, dt, t_total, source_taxis,
                               source_data, source_idx,
                               record_idx=[t['idx'] for t in targets])
    if not np.all(np.isfinite(rec)):
        return np.inf, np.inf
    mse, nrm = [], []
    for k, tgt in enumerate(targets):
        r = np.interp(tgt['taxis'], taxis, rec[:, k]) - tgt['data']
        m = float(np.mean(r ** 2))
        mse.append(m)
        nrm.append(m / float(np.max(tgt['data'])) ** 2)
    return float(np.sqrt(np.mean(mse))), float(np.sqrt(np.mean(nrm)))
