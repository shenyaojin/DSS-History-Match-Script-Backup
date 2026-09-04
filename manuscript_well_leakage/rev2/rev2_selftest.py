"""Acceptance tests for the shared rev2 modules.

    python3 scripts/manuscript_well_leakage/rev2/rev2_selftest.py

Run with CWD = repo root. Prints one PASS/FAIL line per test and exits non-zero on
any failure. T1 is the load-bearing one: it is the identity that carries the
already-established fibeRIS equivalence proof of
`r1_calibration_core.solve_forward` into `rev2_core.solve_forward`, so that no
sweep has to re-run the dense fibeRIS solve.

Everything here is read-only except T7, which needs a code file it is allowed to
mutate; it creates its own fixture under output/rev2_20260901/A4/selftest_scratch/
and restores it. No existing output is overwritten or deleted.
"""

import os

# Pin BLAS to one thread BEFORE numpy is imported. This machine runs a dozen
# concurrent rev2 and MOOSE jobs at load average ~60 on 40 CPUs, and a
# multithreaded scipy.linalg.solve then stalls indefinitely on oversubscription --
# the single fibeRIS dense solve in T8a took over 8 s before hanging and 0.57 s
# after. Also keeps this test to one core, per the house-rules cap.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import shutil   # noqa: E402
import sys      # noqa: E402
import tempfile  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'manuscript_well_leakage',
                                'baseline_calibration'))

import r1_calibration_core as r1core  # noqa: E402
import rev2_core as rc                # noqa: E402
import rev2_data as rd                # noqa: E402
import rev2_layout as rl              # noqa: E402
import rev2_manifest as rm            # noqa: E402

RESULTS = []


def check(name, passed, detail=''):
    RESULTS.append((name, bool(passed)))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}"
          + (f"\n        {detail}" if detail else ''), flush=True)
    return bool(passed)


# ---------------------------------------------------------------------------

def t1_kernel_identity():
    """theta=1 / harmonic / lambda=0 must be BITWISE identical to the r1 kernel."""
    S = rd.setup_r1()
    mesh, sidx = S['mesh'].x, S['source_idx']
    src = S['src_series']
    t_total = S['t_total_s']
    ridx = [t['idx'] for t in S['targets']]

    prof = r1core.build_uniform_profile(mesh, 1150.0)
    ta, ra = r1core.solve_forward(mesh, prof, 1.0, t_total, src.taxis_s,
                                  src.delta_psi, sidx, record_idx=ridx)
    tb, rb = rc.solve_forward(mesh, prof, 1.0, t_total, src.taxis_s,
                              src.delta_psi, sidx, record_idx=ridx)
    exact = np.array_equal(ta, tb) and np.array_equal(ra, rb)
    worst = float(np.max(np.abs(ra - rb)))
    ok = check("T1a production problem (nx=6751, D=1150, dt=1 s): bitwise "
               "identical to r1_calibration_core.solve_forward",
               exact,
               f"np.array_equal on taxis and field; max|diff| = {worst:.3e} psi"
               if exact else
               f"NOT BITWISE IDENTICAL -- max|diff| = {worst:.3e} psi")

    # A single configuration is not enough: refined or padded meshes with large dx
    # contrast can expose an operator-ordering difference a one-point test misses.
    win = rd.build_mesh(rd.R1_WINDOW, 0.0, 0.0, 1.0).x
    s_w = int(np.argmin(np.abs(win - src.md_ft)))
    worst_all, bad = 0.0, []
    for dx in (0.5, 1.0, 2.0, 5.0):
        m = rd.build_mesh(rd.R1_WINDOW, 0.0, 0.0, dx).x
        si = int(np.argmin(np.abs(m - src.md_ft)))
        for D in (140.0, 1150.0, 2000.0):
            for dt in (1.0, 2.0):
                p = np.full(len(m), D)
                x1 = r1core.solve_forward(m, p, dt, 300.0, src.taxis_s,
                                          src.delta_psi, si)
                x2 = rc.solve_forward(m, p, dt, 300.0, src.taxis_s,
                                      src.delta_psi, si)
                if not (np.array_equal(x1[0], x2[0])
                        and np.array_equal(x1[1], x2[1])):
                    bad.append((dx, D, dt))
                worst_all = max(worst_all,
                                float(np.max(np.abs(x1[1] - x2[1]))))
    ok &= check("T1b identity holds at every (dx, D, dt) the study uses "
                "(dx 0.5/1/2/5 ft x D 140/1150/2000 x dt 1/2 s)",
                not bad, f"24 configurations, max|diff| = {worst_all:.3e} psi"
                if not bad else f"failed at {bad}")

    # A graded, non-uniform D(x) on a NON-UNIFORM mesh: the harmonic mean and the
    # alpha index shift are where an ordering slip would show up.
    m = np.sort(np.concatenate([np.arange(15000.0, 16750.1, 5.0),
                                np.arange(16600.0, 16700.01, 0.25)]))
    m = m[np.concatenate([[True], np.diff(m) > 1e-6])]
    si = int(np.argmin(np.abs(m - src.md_ft)))
    p = r1core.profile_two_zone(m, si, [np.log10(3851.0), np.log10(253.5),
                                        np.log10(438.3), np.log10(21.1)])
    y1 = r1core.solve_forward(m, p, 1.0, 300.0, src.taxis_s, src.delta_psi, si)
    y2 = rc.solve_forward(m, p, 1.0, 300.0, src.taxis_s, src.delta_psi, si)
    ok &= check("T1c identity on a non-uniform mesh with the two_zone D(x)",
                np.array_equal(y1[0], y2[0]) and np.array_equal(y1[1], y2[1]),
                f"nx={len(m)}, dx {np.diff(m).min():.3g}-{np.diff(m).max():.3g} ft")
    return ok


def t2_leakage():
    """lambda=0 changes nothing; lambda>0 gives the analytic decay length."""
    S = rd.setup_r1()
    mesh, sidx = S['mesh'].x, S['source_idx']
    src = S['src_series']
    prof = np.full(len(mesh), 1150.0)
    a = rc.solve_forward(mesh, prof, 1.0, 200.0, src.taxis_s, src.delta_psi, sidx)
    b = rc.solve_forward(mesh, prof, 1.0, 200.0, src.taxis_s, src.delta_psi, sidx,
                         lambda_leak=0.0)
    c = rc.solve_forward(mesh, prof, 1.0, 200.0, src.taxis_s, src.delta_psi, sidx,
                         lambda_leak=np.zeros(len(mesh)))
    ok = check("T2a lambda_leak=0 (scalar and all-zero array) is bitwise "
               "identical to lambda_leak absent",
               np.array_equal(a[1], b[1]) and np.array_equal(a[1], c[1]))

    # Steady state of D u'' - lambda u = 0 with u(0) = U is U*exp(-x*sqrt(lam/D)).
    D, L = 1150.0, 550.0
    lam = D / L ** 2
    m = np.arange(0.0, 6000.0 + 1e-9, 1.0)
    p = np.full(len(m), D)
    ta = np.array([0.0, 1e6])
    da = np.array([100.0, 100.0])
    _, rec = rc.solve_forward(m, p, 1.0, 20000.0, ta, da, 0, lambda_leak=lam)
    u = rec[-1]
    sel = (m >= 200.0) & (m <= 2000.0)
    slope = np.polyfit(m[sel], np.log(u[sel]), 1)[0]
    L_num = -1.0 / slope
    rel = abs(L_num - L) / L
    ok &= check("T2b numerical steady state matches the analytic decay length "
                "sqrt(D/lambda) to < 1e-4 relative",
                rel < 1e-4,
                f"analytic {L:.4f} ft, numerical {L_num:.4f} ft, "
                f"relative error {rel:.2e} (D={D}, lambda={lam:.6e} 1/s, "
                f"tau = {1 / lam:.1f} s)")

    # The sink must never touch the constraint rows (fibeRIS's PML loop does).
    _, src_col = rc.solve_forward(m, p, 1.0, 50.0, ta, da, 0, lambda_leak=lam,
                                  record_idx=[0, len(m) - 1])
    dev = float(np.max(np.abs(src_col[1:, 0] - 100.0)))
    pml_dev = abs(100.0 / (1.0 + 1.0 * lam) - 100.0)
    # The source COLUMN is not zeroed (fibeRIS does not zero it either), so the LU
    # leaves ~1e-12 relative round-off on the prescribed value; the PML idiom would
    # leave 0.38%, eight orders of magnitude larger.
    ok &= check("T2c the sink is interior-only: the Dirichlet source node keeps "
                "its prescribed value (fibeRIS's PML idiom would pin it at "
                "s/(1+dt*lambda), 0.38% low)",
                dev / 100.0 < 1e-9,
                f"max deviation {dev:.3e} psi ({dev / 100.0:.1e} relative, pure "
                f"LU round-off) vs {pml_dev:.4f} psi if the sink hit the "
                f"Dirichlet row")

    # Sharper: inspect the assembled matrix rows directly.
    coef = rc._coefficients(m, p, 'harmonic')
    si = len(m) // 2
    st = rc._Stepper(m, coef, [si], 1.0, rc._leak_interior(lam, len(m)), 0.0)
    ab, _ = st.build(1.0)
    a_l, a_r = rc._alphas(coef, 1.0)
    interior_ok = abs(ab[1, 1] - (1.0 + a_l[0] + a_r[0] + lam * 1.0)) < 1e-12
    ok &= check("T2d the assembled matrix keeps the three constraint rows "
                "untouched by lambda while the interior diagonal carries "
                "+dt*lambda (fibeRIS's `for i in range(nx): A[i,i] += "
                "dt*sigma[i]` hits all three)",
                ab[1, 0] == -1.0 and ab[1, len(m) - 1] == -1.0
                and ab[1, si] == 1.0 and interior_ok,
                f"Neumann diagonals {ab[1, 0]} / {ab[1, len(m) - 1]}, Dirichlet "
                f"diagonal {ab[1, si]}, interior {ab[1, 1]:.6f}")
    return ok


def t3_interface_avg():
    """Arithmetic and harmonic differ only where D varies."""
    d_const = np.full(400, 1150.0)
    h = rc.face_diffusivity(d_const, 'harmonic')
    a = rc.face_diffusivity(d_const, 'arithmetic')
    ok = check("T3a face averaging is identical where D is constant",
               np.array_equal(h, a), f"max|diff| = {np.max(np.abs(h - a)):.3e}")

    d_var = np.full(400, 1150.0)
    d_var[:200] = 253.5
    hv = rc.face_diffusivity(d_var, 'harmonic')
    av = rc.face_diffusivity(d_var, 'arithmetic')
    same_where_flat = (np.array_equal(hv[:198], av[:198])
                       and np.array_equal(hv[200:], av[200:]))
    ok &= check("T3b they differ ONLY at the faces spanning the jump",
                same_where_flat and hv[199] != av[199],
                f"at the jump: harmonic {hv[199]:.3f} vs arithmetic "
                f"{av[199]:.3f} ft^2/s")

    S = rd.setup_r1()
    mesh, sidx, src = S['mesh'].x, S['source_idx'], S['src_series']
    p_uni = np.full(len(mesh), 1150.0)
    u1 = rc.solve_forward(mesh, p_uni, 1.0, 300.0, src.taxis_s, src.delta_psi,
                          sidx, record_idx=[t['idx'] for t in S['targets']])
    u2 = rc.solve_forward(mesh, p_uni, 1.0, 300.0, src.taxis_s, src.delta_psi,
                          sidx, record_idx=[t['idx'] for t in S['targets']],
                          interface_avg='arithmetic')
    ok &= check("T3c full solve with uniform D is unchanged by interface_avg",
                np.array_equal(u1[1], u2[1]))

    p_two = r1core.profile_two_zone(mesh, sidx,
                                    [np.log10(3851.0), np.log10(253.5),
                                     np.log10(438.3), np.log10(21.1)])
    v1 = rc.solve_forward(mesh, p_two, 1.0, 600.0, src.taxis_s, src.delta_psi,
                          sidx, record_idx=[t['idx'] for t in S['targets']])
    v2 = rc.solve_forward(mesh, p_two, 1.0, 600.0, src.taxis_s, src.delta_psi,
                          sidx, record_idx=[t['idx'] for t in S['targets']],
                          interface_avg='arithmetic')
    d = float(np.max(np.abs(v1[1] - v2[1])))
    ok &= check("T3d with the two_zone D(x) the two averages disagree "
                "measurably", d > 0.0,
                f"max|harmonic - arithmetic| = {d:.4f} psi at the target gauges")
    return ok


def t4_barriers():
    """Physical half-width, mesh independence, and the sub-dx fallback."""
    ok = True
    rows = []
    for dx in (1.0, 0.5, 0.2, 0.1):
        m = np.arange(0.0, 2000.0 + dx / 2.0, dx)
        _, rep = rc.build_barrier_profile(m, 1150.0, [1000.0], 1.0, 1e-3,
                                          return_report=True)
        full = rep['realised_full_width_ft']['max']
        n = int(rep['n_nodes_captured']['max'])
        rows.append((dx, n, full))
        ok &= (abs(full - 2.0) <= dx + 1e-9) and n == 2 * int(round(1.0 / dx)) + 1
    check("T4a w = 1.0 ft realises 2w to within one cell on dx = 1.0/0.5/0.2/"
          "0.1 ft, with node count 2*round(w/dx)+1",
          ok, "  ".join(f"dx={dx}: {n} nodes, realised full width {f:.3f} ft "
                        f"(+{100 * (f - 2) / 2:.0f}%)" for dx, n, f in rows))

    # Legacy reduction: w = 0 must reproduce the single-node assignment exactly.
    from fiberis.utils import mesh_utils
    fh7 = np.load(os.path.join(ROOT, 'data/fiberis_format/s_well/geometry/'
                                     'frac_hit/frac_hit_stage_7_swell.npz'))['data']
    fh8 = np.load(os.path.join(ROOT, 'data/fiberis_format/s_well/geometry/'
                                     'frac_hit/frac_hit_stage_8_swell.npz'))['data']
    x = np.arange(12500, 12500 + 5500 * 1, 1)
    for f in list(fh7) + list(fh8):
        x = mesh_utils.refine_mesh(x, [round(f) - 1, round(f) + 1], 5)
    legacy = np.ones_like(x, dtype=float) * 140.0
    for f in fh7:
        legacy[mesh_utils.locate(x, f)[0]] = 140.0 * 1e-5
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter('always')
        prof, rep = rc.build_barrier_profile(x, 140.0, fh7, 0.0, 1e-5,
                                             return_report=True)
    o = check("T4b w = 0 reproduces the legacy 101/106:190 single-node "
              "assignment elementwise on the real refined mesh (nx=5656)",
              np.array_equal(prof, legacy) and rep['n_fallback'] == 6
              and abs(rep['realised_full_width_ft']['median'] - 2.0 / 15.0) < 1e-9,
              f"n_fallback={rep['n_fallback']}, realised full width "
              f"{rep['realised_full_width_ft']['median']:.5f} ft (= 2/15, not the "
              f"0.2 ft that '5x over +-1 ft' implies), total equivalent width "
              f"{rep['total_equivalent_width_ft']:.3f} ft, excess resistance "
              f"{rep['excess_resistance_s_per_ft']:.1f} s/ft, "
              f"{len(W)} BarrierWidthWarning(s)")
    ok = ok and o

    # Sub-dx request: must fall back, warn, AND record it.
    m = np.arange(15186.0, 15188.0, 2.0 / 15.0)
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter('always')
        sub, rep_s = rc.build_barrier_profile(m, 140.0, [15186.83], 0.02, 1e-5,
                                              return_report=True)
    o = check("T4c sub-dx request (w = 0.02 ft on a dx = 0.1333 ft mesh) falls "
              "back to the nearest node, warns, and records n_fallback",
              rep_s['n_fallback'] == rep_s['n_barriers'] == 1 and len(W) == 1
              and abs(rep_s['width_inflation']['max'] - 10.0 / 3.0) < 0.05
              and not np.array_equal(sub, np.full(len(m), 140.0)),
              f"width_inflation {rep_s['width_inflation']['max']:.2f}x "
              f"(realised half-width {rep_s['barriers'][0]['realised_half_width_ft']:.4f} ft)")
    ok = ok and o
    try:
        rc.build_barrier_profile(m, 140.0, [15186.83], 0.02, 1e-5,
                                 on_empty='raise')
        o = check("T4d on_empty='raise' refuses the sub-dx request", False)
    except ValueError:
        o = check("T4d on_empty='raise' refuses the sub-dx request", True)
    ok = ok and o

    # Overlap of the two closest real frac hits.
    m = np.arange(0.0, 2000.0 + 0.25, 0.5)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        _, r1_ = rc.build_barrier_profile(m, 1150.0, [1000.0, 1008.114], 5.0,
                                          1e-3, return_report=True)
        _, r2_ = rc.build_barrier_profile(m, 1150.0, [1008.114, 1000.0], 5.0,
                                          1e-3, return_report=True)
    o = check("T4e two barriers 8.114 ft apart with w = 5 ft merge into ONE "
              "group, order-independently, and the total width is the merged "
              "span (not the sum)",
              r1_['n_overlapping_pairs'] == 1 and r1_['n_merged_groups'] == 1
              and r1_['total_equivalent_width_ft']
              == r2_['total_equivalent_width_ft']
              and abs(r1_['total_equivalent_width_ft']
                      - r1_['merged_groups'][0]['equivalent_full_width_ft']) < 1e-12,
              f"total {r1_['total_equivalent_width_ft']:.3f} ft vs a naive sum of "
              f"{sum(b['realised_full_width_ft'] for b in r1_['barriers']):.3f} ft")
    ok = ok and o

    # Null controls and the int64-mesh trap.
    xi = np.arange(12500, 18000, 1)  # int64, exactly as 101:77 builds it
    p_int = rc.build_barrier_profile(xi, 140.0, [15000.0], 0.0, 1e-5)
    p_r1 = rc.build_barrier_profile(xi, 140.0, [15000.0], 5.0, 1.0)
    p_e = rc.build_barrier_profile(xi, 140.0, [], 5.0, 1e-5)
    o = check("T4f int64 mesh gives float64 output (1.4e-3, not 0); ratio = 1 "
              "and an empty MD list both return the baseline bit-for-bit",
              p_int.dtype == np.float64
              and p_int[int(np.argmin(np.abs(xi - 15000.0)))] == 140.0 * 1e-5
              and np.array_equal(p_r1, np.full(len(xi), 140.0))
              and np.array_equal(p_e, np.full(len(xi), 140.0)),
              f"barrier D = {p_int[int(np.argmin(np.abs(xi - 15000.0)))]:.6g} ft^2/s")
    ok = ok and o

    # Array d_base, both ratio references.
    mm = rd.build_mesh(rd.R1_WINDOW, 5000.0, 0.0, 1.0)
    si = mm.index_of(16645.0)
    two = r1core.profile_two_zone(mm.x, si, [np.log10(3851.0), np.log10(253.5),
                                             np.log10(438.3), np.log10(21.1)])
    _, ra_ = rc.build_barrier_profile(mm.x, two, [16200.0], 20.0, 1e-3,
                                      ratio_reference='local', return_report=True)
    _, rb_ = rc.build_barrier_profile(mm.x, two, [16200.0], 20.0, 1e-3,
                                      ratio_reference='at_hit', return_report=True)
    b_a, b_b = ra_['barriers'][0], rb_['barriers'][0]
    o = check("T4g array d_base: ratio_reference='local' varies across the "
              "block, 'at_hit' is constant, and both report d_barrier min/max",
              ra_['d_base_is_array'] and b_a['d_barrier_min_ft2_s']
              != b_a['d_barrier_max_ft2_s']
              and b_b['d_barrier_min_ft2_s'] == b_b['d_barrier_max_ft2_s'],
              f"local {b_a['d_barrier_min_ft2_s']:.4g}-{b_a['d_barrier_max_ft2_s']:.4g}, "
              f"at_hit {b_b['d_barrier_min_ft2_s']:.4g} ft^2/s")
    ok = ok and o

    # The 9e-13 ft near-duplicate that destroyed an A1 reference solve.
    bad = np.unique(np.concatenate([np.arange(0.0, 2001.0, 1.0),
                                    np.arange(994.0, 1006.005, 0.005)]))
    try:
        rc.build_barrier_profile(bad, 1150.0, [1000.0], 1.0, 1e-3)
        o = check("T4h a mesh with a ~9e-13 ft node gap is rejected", False)
    except ValueError as exc:
        o = check("T4h a mesh with a ~9e-13 ft node gap is rejected",
                  'strictly increasing' in str(exc), str(exc)[:150])
    ok = ok and o

    # The exact single-node/slab identity the reported widths rest on.
    rng = np.random.default_rng(20260901)
    worst = 0.0
    for _ in range(2000):
        D0 = 10.0 ** rng.uniform(-1, 4)
        ratio = 10.0 ** rng.uniform(-6, -0.1)
        dxl = 10.0 ** rng.uniform(-2, 1)
        dxr = 10.0 ** rng.uniform(-2, 1)
        mesh = np.array([0.0, dxl, dxl + dxr, dxl + dxr + 1.0])
        prof = np.full(4, D0)
        prof[1] = D0 * ratio
        node = rc._series_resistance(mesh, prof) - rc._series_resistance(
            mesh, np.full(4, D0))
        s = (dxl + dxr) / 2.0
        slab = s * (1.0 / (D0 * ratio) - 1.0 / D0)
        worst = max(worst, abs(node - slab) / abs(slab))
    o = check("T4i a single reduced node is EXACTLY a slab of full width "
              "(dxL+dxR)/2 (the identity the realised widths rest on)",
              worst < 1e-12, f"max relative difference over 2000 random "
                             f"(D0, ratio, dxL, dxR) draws: {worst:.3e}")
    return ok and o


def t5_theta_order():
    """Observed temporal order: theta=1 -> 1, theta=0.5 -> 2.

    The problem is an exact eigenmode of the DISCRETE operator (uniform D, the
    Dirichlet source row held at 0, Neumann on the far end), so its solution
    u(t) = u0*exp(-mu t) is analytic for the semi-discrete system and the
    measured error is purely the time scheme -- no spatial error contaminates the
    slope.
    """
    D, dx, nx = 1150.0, 5.0, 60
    mesh = np.arange(nx) * dx
    prof = np.full(nx, D)
    a = D / dx ** 2
    n = nx - 2
    M = np.zeros((n, n))
    for k in range(n):
        M[k, k] = 2 * a
        if k > 0:
            M[k, k - 1] = -a
        if k < n - 1:
            M[k, k + 1] = -a
    M[n - 1, n - 1] = a  # fold in u_{nx-1} = u_{nx-2} (the Neumann row)
    mu, V = np.linalg.eigh(M)
    mu, v = mu[0], V[:, 0]
    u0 = np.zeros(nx)
    u0[1:nx - 1] = v
    u0[nx - 1] = v[-1]
    u0 /= np.max(np.abs(u0))
    T = 1.0 / mu
    ta, da = np.array([0.0, 1e9]), np.array([0.0, 0.0])

    out = {}
    for th in (1.0, 0.5):
        errs = []
        for dt in (T / 8, T / 16, T / 32, T / 64):
            t, rec = rc.solve_forward(mesh, prof, dt, T, ta, da, 0, initial=u0,
                                      theta=th)
            errs.append(float(np.max(np.abs(rec - np.outer(np.exp(-mu * t), u0)))))
        out[th] = (errs, [float(np.log2(errs[i] / errs[i + 1]))
                          for i in range(len(errs) - 1)])
    o_be = out[1.0][1][-1]
    o_cn = out[0.5][1][-1]
    ok = check("T5a theta = 1 (backward Euler) is first order in dt",
               0.85 < o_be < 1.15,
               f"observed orders {['%.3f' % x for x in out[1.0][1]]}, "
               f"errors {['%.3e' % e for e in out[1.0][0]]}")
    ok &= check("T5b theta = 0.5 (Crank-Nicolson) is second order in dt",
                1.85 < o_cn < 2.15,
                f"observed orders {['%.3f' % x for x in out[0.5][1]]}, "
                f"errors {['%.3e' % e for e in out[0.5][0]]}")

    # theta < 1 with the stale datum silently loses that second order, so it must
    # raise rather than warn.
    try:
        rc.solve_forward(mesh, prof, 1.0, 10.0, ta, da, 0, theta=0.5,
                         source_time_level='n')
        ok &= check("T5c theta < 1 with source_time_level='n' is refused", False)
    except ValueError as exc:
        ok &= check("T5c theta < 1 with source_time_level='n' is refused",
                    'first order' in str(exc), str(exc)[:120])

    amp = rc.amplification_factor(mesh, prof, 1.0, 0.5)
    ok &= check("T5d the amplification factor is computed from the actual "
                "a_l/a_r arrays, not 4*D*dt/dx^2",
                amp['g'] < -0.9,
                f"lam_max = {amp['lam_max']:.1f}, g(CN) = {amp['g']:.6f} "
                f"(near -1: an undamped oscillatory mode, which is why CN rings "
                f"at a restart discontinuity)")
    return ok


def t6_layout():
    """All eight 0211 files, plus a raise on a synthetic square panel."""
    d = os.path.join(ROOT, 'output', '0211_simulation_MULTIstage')
    expect = {'phase1.npz': rl.LAYOUT_TIME_MAJOR,
              'phase2.npz': rl.LAYOUT_TIME_MAJOR,
              'phase3_0.1.npz': rl.LAYOUT_TIME_MAJOR,
              'phase3_0.01.npz': rl.LAYOUT_TIME_MAJOR,
              'phase3_0.001.npz': rl.LAYOUT_TIME_MAJOR,
              'phase3_0.0001.npz': rl.LAYOUT_TIME_MAJOR,
              'phase3_1e-05.npz': rl.LAYOUT_TIME_MAJOR,
              'phase3_test.npz': rl.LAYOUT_DEPTH_MAJOR}
    before = {f: rm.sha256_file(os.path.join(d, f), use_cache=False)
              for f in expect}
    panels = {f: rl.load_panel(os.path.join(d, f)) for f in expect}
    got = {f: p.detected_layout for f, p in panels.items()}
    shapes_ok = all(p.data.shape == (len(p.taxis), len(p.daxis))
                    for p in panels.values())
    ok = check("T6a all 8 files in output/0211_simulation_MULTIstage detected "
               "correctly (7 time-major, phase3_test depth-major)",
               got == expect and shapes_ok,
               "; ".join(p.describe() for p in panels.values()))

    gt = rl.load_panel(os.path.join(ROOT, 'output', '0427_simulation_uniform_D',
                                    'phase1.npz'))
    ok &= check("T6b transpose is correct against ground truth: the 0427 "
                "depth-major re-save of phase1 gives an identical canonical array",
                np.array_equal(panels['phase1.npz'].data, gt.data),
                f"max|diff| = "
                f"{np.max(np.abs(panels['phase1.npz'].data - gt.data)):.3e} psi")

    cont = {f: float(np.max(np.abs(rl.final_profile(panels['phase2.npz'])
                                   - panels[f].data[0, :])))
            for f in expect if f.startswith('phase3')}
    ok &= check("T6c physical continuity a transpose bug cannot fake: "
                "phase2's last profile == every phase3 file's first profile",
                all(v == 0.0 for v in cont.values()),
                f"max over the 6 phase-3 files: {max(cont.values()):.3e} psi")

    try:
        rl.detect_layout((100, 100), 100, 100)
        ok &= check("T6d a square panel RAISES instead of guessing", False)
    except rl.LayoutError as exc:
        ok &= check("T6d a square panel RAISES instead of guessing",
                    'rule 5' in str(exc), str(exc)[:130])
    lay, ev = rl.detect_layout((100, 100), 100, 100,
                               declared=rl.LAYOUT_TIME_MAJOR)
    ok &= check("T6e an explicit layout stamp resolves the square case (rule 2)",
                lay == rl.LAYOUT_TIME_MAJOR and ev['rule'] == 2)
    raises = []
    for shape, nt, nx, decl, rule in (((486, 5656), 443, 5656, None, 8),
                                      ((5656,), 486, 5656, None, 3),
                                      ((486, 5656), 486, 5656, 'depth_major', 1),
                                      ((486, 5656), 486, 5656, 'whatever', 0)):
        try:
            rl.detect_layout(shape, nt, nx, decl)
            raises.append((shape, rule, 'did not raise'))
        except rl.LayoutError as exc:
            if f"rule {rule}" not in str(exc):
                raises.append((shape, rule, str(exc)[:60]))
    ok &= check("T6f rules 0/1/3/8 all raise with the right rule number",
                not raises, str(raises) if raises else "")

    after = {f: rm.sha256_file(os.path.join(d, f), use_cache=False)
             for f in expect}
    ok &= check("T6g output/0211_simulation_MULTIstage is untouched "
                "(sha256 before == after)", before == after)
    try:
        rl.save_panel(panels['phase1.npz'], os.path.join(d, 'x.npz'))
        ok &= check("T6h save_panel refuses to write into the frozen dir", False)
    except rl.LayoutError:
        ok &= check("T6h save_panel refuses to write into the frozen dir", True)

    obj = rl.to_fiberis_data2d(panels['phase2.npz'])
    obj.select_depth(14440.0, 15864.29)
    ok &= check("T6i to_fiberis_data2d yields a working depth-major DSS2D",
                obj.data.shape == (len(obj.daxis), len(obj.taxis))
                and obj.data.shape[0] == 1581,
                f"after select_depth: {obj.data.shape}")
    return ok


def t7_manifest():
    """The writer refuses incomplete input; the checker sees a mutated code file."""
    ok = True
    m = rd.build_mesh(rd.R1_WINDOW, 5000.0, 0.0, 1.0)
    taxis = np.arange(0.0, 101.0, 1.0)
    drv = rm.driver_record(kind='gauge_series', baseline_removal='subtract_first_sample',
                           value_units='delta_psi', gauge_number=1,
                           gauge_md_ft=16645.0, taxis=taxis,
                           values=np.zeros_like(taxis))
    srcs = [rm.source_record(m.x, md_requested_ft=16645.0, mesh_idx=6645,
                             driver=drv, label='g1', index_in_source_list=0)]
    sp = rm.source_protocol(application='dirichlet_node',
                            solver_class='rev2_core.solve_forward',
                            placement_rule='nearest_gauge_to_stage1_frac_hits',
                            sources=srcs, targets=rm.NONE_DECLARED,
                            time_level='n', phase_chaining=rm.NONE_DECLARED,
                            boundary_conditions={'lbc': 'Neumann',
                                                 'rbc': 'Neumann'})
    tr = rm.time_record(taxis, mode='fixed', theta=1.0,
                        t_total_requested_s=100.0, dt_requested_s=1.0,
                        source_time_level='n')
    mr = rm.mesh_record(m.x, dx_requested_ft=1.0,
                        window_md_ft=(15000.0, 16750.0), pad_low_ft=5000.0,
                        pad_high_ft=0.0)

    def make_numerics(barriers=rm.NONE_DECLARED):
        return rm.numerics(
            time=rm.time_record(taxis, mode='fixed', theta=1.0,
                                t_total_requested_s=100.0, dt_requested_s=1.0),
            mesh=rm.mesh_record(m.x, dx_requested_ft=1.0,
                                window_md_ft=(15000.0, 16750.0),
                                pad_low_ft=5000.0, pad_high_ft=0.0),
            interface_avg='harmonic',
            boundary={'lbc': 'Neumann', 'rbc': 'Neumann', 'pml_thickness': 0.0,
                      'sigma_max': 0.0},
            diffusivity={'baseline_D_ft2_s': 1150.0, 'profile_family': 'uniform',
                         'param_names': ['log10_D'], 'params': [np.log10(1150.0)],
                         'D_min': 1150.0, 'D_max': 1150.0,
                         'D_sha256': rm.sha256_array(np.full(m.nx, 1150.0)),
                         'profile_anchor': 'physical_md'},
            barriers=barriers, leakage=rm.NONE_DECLARED,
            kernel={'name': 'rev2_core.solve_forward', 'banded': True,
                    'equivalence_reference':
                        'output/r1_baseline_calibration/r1_run_manifest.json'},
            rng=rm.NONE_DECLARED, parallel=rm.NONE_DECLARED)

    tmp = tempfile.mkdtemp(prefix='rev2_selftest_')
    scratch = os.path.join(ROOT, 'output', rm.ROUND_TAG, 'A4', 'selftest_scratch')
    os.makedirs(scratch, exist_ok=True)
    fixture = os.path.join(scratch, 'selftest_code_fixture.py')
    original = b"# fixture for rev2_selftest T7; safe to regenerate\nVALUE = 1\n"
    with open(fixture, 'wb') as fh:
        fh.write(original)
    try:
        arrays = os.path.join(tmp, 'arrays.npz')
        np.savez(arrays, x=m.x)
        man = os.path.join(tmp, 'manifest.json')

        # L1: omitting a group is a TypeError before any work happens.
        try:
            rm.write_manifest(man, study_id='s', task_id='A0', config={},
                              config_path=None, inputs=[], source=sp,
                              outputs=[])  # numerics= missing
            ok &= check("T7a omitting numerics= raises TypeError at the call site",
                        False)
        except TypeError as exc:
            ok &= check("T7a omitting numerics= raises TypeError at the call site",
                        'numerics' in str(exc), str(exc)[:110])

        # L2: a hand-rolled dict cannot pass for a built group.
        try:
            rm.write_manifest(man, study_id='s', task_id='A0', config={},
                              config_path=None, inputs=[], source=sp,
                              numerics={'time': [], 'mesh': {}}, outputs=[])
            ok &= check("T7b a hand-rolled numerics dict is rejected", False)
        except rm.ManifestIncomplete as exc:
            ok &= check("T7b a hand-rolled numerics dict is rejected",
                        True, str(exc)[:110])

        # barriers=[] vs NONE_DECLARED.
        try:
            make_numerics(barriers=[])
            ok &= check("T7c barriers=[] is refused while barriers=NONE_DECLARED "
                        "succeeds", False)
        except rm.ManifestIncomplete as exc:
            ok &= check("T7c barriers=[] is refused while barriers=NONE_DECLARED "
                        "succeeds", isinstance(make_numerics(), dict),
                        str(exc)[:110])

        # A declared but absent output must abort the manifest.
        try:
            rm.write_manifest(man, study_id='s', task_id='A0', config={},
                              config_path=None, inputs=[], source=sp,
                              numerics=make_numerics(),
                              outputs=[rm.output_decl(os.path.join(tmp, 'gone.png'),
                                                      role='figure_png', dpi=300)])
            ok &= check("T7d a declared-but-missing output raises OutputMissing",
                        False)
        except rm.OutputMissing:
            ok &= check("T7d a declared-but-missing output raises OutputMissing",
                        True)

        # An adaptive time_record without its tolerances is incomplete.
        try:
            rm.time_record(taxis, mode='adaptive', theta=1.0,
                           t_total_requested_s=100.0, dt_init_s=2.0)
            ok &= check("T7e an adaptive run without tol/max_dt/min_dt/"
                        "n_steps_rejected is refused", False)
        except rm.ManifestIncomplete as exc:
            ok &= check("T7e an adaptive run without tol/max_dt/min_dt/"
                        "n_steps_rejected is refused", True, str(exc)[:110])
        ok &= check("T7f count_rejected returns None (not 0) for an unavailable "
                    "history", rm.count_rejected([]) is None
                    and rm.count_rejected(['Dynamic time sampling rejected x']) == 1)

        # The real write, then verify.
        doc = rm.write_manifest(
            man, study_id='rev2_selftest', task_id='A-foundation',
            config={'note': 'self-test'}, config_path=None,
            inputs=[(os.path.join(ROOT, 'data/fiberis_format/s_well/geometry/'
                                        'gauge_md_swell.npz'), 'geometry',
                     'gauge_md_npz')],
            source=sp, numerics=make_numerics(),
            outputs=[rm.output_decl(arrays, role='arrays_npz')],
            results={'rmse_psi': 11.872}, notes=['self-test run'],
            require_modules=('rev2_core', 'rev2_manifest', 'r1_calibration_core'),
            extra_code_files=(fixture,))
        rep = rm.verify(man, repo_root=ROOT)
        ok &= check("T7g a complete manifest writes and verifies clean",
                    rep['status'] == 'clean'
                    and rep['code']['n'] >= 5 and not rep['code']['drift']
                    and rep['inputs']['n'] == 1 and not rep['inputs']['drift']
                    and rep['outputs']['n'] == 1
                    and rep['manifest_self']['status'] == 'ok',
                    f"status={rep['status']}, code closure {rep['code']['n']} "
                    f"files (closure_sha256 "
                    f"{doc['code']['closure_sha256'][:16]}...), inputs "
                    f"{rep['inputs']['n']}, outputs {rep['outputs']['n']}")
        ok &= check("T7h the code closure captured fibeRIS automatically",
                    any('fibeRIS/' in p for p in doc['code']['files']),
                    f"{sum(1 for p in doc['code']['files'] if 'fibeRIS/' in p)} "
                    f"fibeRIS files of {doc['code']['n_files']}")
        ok &= check("T7i NONE_DECLARED is recorded as an explicit assertion",
                    'numerics.barriers' in doc['explicitly_none'],
                    f"explicitly_none = {doc['explicitly_none']}")

        # T7 proper: mutate a code file and confirm verify sees it.
        with open(fixture, 'ab') as fh:
            fh.write(b"# mutated by the self-test\n")
        rep2 = rm.verify(man, repo_root=ROOT)
        drifted = [d['logical_name'] for d in rep2['code']['drift']]
        ok &= check("T7j verify detects a deliberately mutated code file "
                    "(exactly one drift row)",
                    rep2['status'] == 'drift' and len(drifted) == 1
                    and 'selftest_code_fixture.py' in drifted[0],
                    f"drift rows: {drifted}")
        try:
            rm.verify(man, repo_root=ROOT, strict=True)
            ok &= check("T7k strict=True raises ManifestDrift", False)
        except rm.ManifestDrift:
            ok &= check("T7k strict=True raises ManifestDrift", True)
        with open(fixture, 'wb') as fh:
            fh.write(original)

        # A missing required module is a hard error, not a silent smaller closure.
        try:
            rm.code_closure(require=('a_module_that_was_never_imported',))
            ok &= check("T7l require_modules= is a hard error", False)
        except rm.ManifestIncomplete:
            ok &= check("T7l require_modules= is a hard error", True)

        # Legacy manifests must both be readable, and r2's missing inputs visible.
        r1rep = rm.verify(os.path.join(ROOT, 'output/r1_baseline_calibration/'
                                             'r1_run_manifest.json'))
        r2rep = rm.verify(os.path.join(ROOT, 'output/r2_diffusivity_profile/'
                                             'r2_manifest.json'))
        ok &= check("T7m both legacy manifests verify through the adapter: r1's "
                    "11 hashes + config still match; r2 is reported as having NO "
                    "input provenance rather than a false clean",
                    r1rep['code']['n'] == 2 and not r1rep['code']['drift']
                    and r1rep['inputs']['n'] == 9
                    and not r1rep['inputs']['drift']
                    and r1rep['config']['status'] == 'ok'
                    and r2rep['inputs'].get('not_recorded') is True
                    and r1rep['completeness']['missing_groups']
                    == ['source_protocol', 'numerics', 'outputs'],
                    f"r1: {r1rep['code']['n']} code + {r1rep['inputs']['n']} "
                    f"inputs all ok; r2 inputs not_recorded="
                    f"{r2rep['inputs'].get('not_recorded')}")

        # T7n (regression, defect 1): a multi-phase chain declares the same
        # nodes in more than one phase. Before the fix, source_protocol saw one
        # flat list and raised a duplicate_source_mesh_idx ERROR on A5's
        # legitimate 18-source / 3-phase chain, while the real bug it was written
        # to catch is two frac hits snapping to ONE node inside ONE solve.
        mesh10 = np.arange(0.0, 200.0 + 1e-9, 10.0)   # dx = 10 ft, as 102r/103r
        hits = [4, 6, 8, 10, 12, 14]

        def phase_srcs(phase, idx_list):
            d = rm.driver_record(
                kind='gauge_series', baseline_removal='subtract_first_sample',
                value_units='delta_psi', gauge_number=6 if phase != 'phase3' else 7,
                gauge_md_ft=15344.0, taxis=taxis,
                values=np.full_like(taxis, 10.0 * int(phase[-1])))
            return [rm.source_record(mesh10, md_requested_ft=float(mesh10[i]),
                                     mesh_idx=i, driver=d,
                                     label=f"{phase}:frachit{j}")
                    for j, i in enumerate(idx_list)]

        spkw = dict(application='dirichlet_node',
                    solver_class='rev2_core.solve_forward',
                    placement_rule='nearest_node_to_frac_hit_md',
                    targets=rm.NONE_DECLARED, time_level='n',
                    phase_chaining={'order': ['phase1', 'phase2', 'phase3']},
                    boundary_conditions={'lbc': 'Neumann', 'rbc': 'Neumann'})
        chain = rm.source_protocol(
            sources=[phase_srcs('phase1', hits), phase_srcs('phase2', hits),
                     phase_srcs('phase3', hits)],
            phase_labels=['phase1', 'phase2', 'phase3'], **spkw)
        within = rm.source_protocol(
            sources=[phase_srcs('phase1', hits),
                     phase_srcs('phase2', [4, 4, 8, 10, 12, 14])], **spkw)
        flat1 = rm.source_protocol(sources=phase_srcs('phase1', hits), **spkw)
        flat_dup = rm.source_protocol(
            sources=phase_srcs('phase1', [4, 4, 8, 10, 12, 14]), **spkw)
        ok &= check("T7n a multi-phase chain may re-drive the same nodes in "
                    "different phases, but a node repeated WITHIN one phase is "
                    "still an error (defect 1)",
                    chain['n_phases'] == 3 and chain['n_sources'] == 18
                    and chain['_discrepancies'] == []
                    and chain['shared_mesh_idx_across_phases'] == hits
                    and chain['sources'][7]['phase_label'] == 'phase2'
                    and [d['kind'] for d in within['_discrepancies']]
                        == ['duplicate_source_mesh_idx']
                    and within['_discrepancies'][0]['phase_index'] == 1
                    and within['duplicate_mesh_idx'] == [4]
                    and flat1['n_phases'] == 1 and not flat1['multi_phase']
                    and flat1['_discrepancies'] == []
                    and [d['kind'] for d in flat_dup['_discrepancies']]
                        == ['duplicate_source_mesh_idx']
                    and _raises(rm.source_protocol, rm.ManifestIncomplete,
                                sources=[phase_srcs('phase1', hits),
                                         phase_srcs('phase2', hits)[0]], **spkw),
                    f"3 phases x 6 nodes: {chain['n_sources']} declared, "
                    f"{chain['n_unique_mesh_idx']} unique, shared across phases "
                    f"{chain['shared_mesh_idx_across_phases']}, discrepancies "
                    f"{len(chain['_discrepancies'])}; a repeat inside phase 1 of "
                    f"a flat declaration still raises "
                    f"{flat_dup['_discrepancies'][0]['kind']}; a mixed "
                    f"flat/nested declaration is refused")

        # T7o (regression, defect 3): the config auto-scan must not hash a file
        # this run WROTE. Before the fix the declared output came back as an
        # 'auto_from_config' INPUT, i.e. the post-run bytes of a product recorded
        # as pre-run data provenance.
        scan_dir = os.path.join(tmp, 'out_scan')
        os.makedirs(scan_dir)
        prod = os.path.join(scan_dir, 'produced.npz')
        np.savez(prod, x=np.arange(3.0))
        man_scan = os.path.join(scan_dir, 'manifest.json')
        cfg_scan = {'data': {'series': os.path.join(ROOT, 'data/fiberis_format/'
                                                          's_well/geometry/'
                                                          'gauge_md_swell.npz')},
                    'outputs': {'arrays': prod, 'manifest': man_scan}}
        doc_scan = rm.write_manifest(
            man_scan, study_id='rev2_selftest', task_id='A-foundation',
            config=cfg_scan, config_path=None, inputs=[], source=sp,
            numerics=make_numerics(),
            outputs=[rm.output_decl(prod, role='arrays_npz')])
        auto = {k: v['abspath'] for k, v in doc_scan['inputs'].items()}
        excl = doc_scan['outputs']['config_paths_excluded_from_input_scan']
        ok &= check("T7o the config auto-scan hashes a config-named INPUT but "
                    "never a declared OUTPUT of the same run (defect 3)",
                    os.path.realpath(prod) not in
                    {os.path.realpath(p) for p in auto.values()}
                    and 'auto:data.series' in auto
                    and [e['config_path'] for e in excl] == ['outputs.arrays']
                    and excl[0]['reason'] == 'declared_output'
                    and rm.verify(man_scan, repo_root=ROOT)['status'] == 'clean',
                    f"inputs hashed: {sorted(auto)}; excluded from the scan: "
                    f"{[(e['config_path'], e['reason']) for e in excl]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok


def t8_multisource_and_adaptive():
    """The two drivers E1/E4 and B1 need: several Dirichlet nodes, and fibeRIS's
    optimizer=True stepping reproduced in the fast kernel."""
    from fiberis.analyzer.Data1D import Data1D_Gauge
    from fiberis.simulator.core import pds

    # --- multi-source, checked against fibeRIS's own multi-source builder ------
    mesh = np.arange(0.0, 200.0 + 1e-9, 1.0)  # nx = 201; the dense solve is O(nx^3)
    prof = np.full(len(mesh), 300.0)
    idxs = [40, 80, 120]
    ta = np.arange(0.0, 201.0, 1.0)
    frames, series = [], []
    for k, _ in enumerate(idxs):
        vals = (50.0 + 30.0 * k) * (1.0 - np.exp(-ta / 40.0))
        f = Data1D_Gauge.Data1DGauge()
        f.data = vals
        f.taxis = ta
        f.start_time = None
        frames.append(f)
        series.append((ta, vals))

    sim = pds.PDS1D_MultiSource()
    sim.set_mesh(mesh)
    sim.set_bcs('Neumann', 'Neumann')
    sim.set_t0(0)
    sim.set_initial(np.zeros_like(mesh))
    sim.set_diffusivity(prof)
    sim.set_sourceidx(idxs)
    sim.set_source(frames)
    sim.solve(optimizer=False, dt=1.0, t_total=200.0)
    ref = np.asarray(sim.snapshot)

    _, mine = rc.solve_forward_multi(mesh, prof, 1.0, 200.0,
                                     [s[0] for s in series],
                                     [s[1] for s in series], idxs)
    n = min(len(ref), len(mine))
    worst = float(np.max(np.abs(ref[:n] - mine[:n])))
    ok = check("T8a solve_forward_multi matches fibeRIS PDS1D_MultiSource "
               "(3 Dirichlet nodes, nx=201, dt=1 s, 200 steps)",
               worst < 1e-9,
               f"max|fibeRIS - rev2| = {worst:.3e} psi over {n} steps; each "
               f"source node holds its own prescribed series")
    ok &= check("T8b duplicate source mesh indices are refused (two frac hits "
                "can snap to one node on a dx = 10 ft mesh)",
                _raises(rc.solve_forward_multi, ValueError, mesh, prof, 1.0,
                        10.0, [s[0] for s in series], [s[1] for s in series],
                        [80, 80, 240]))

    # --- adaptive driver, against A3's measured fibeRIS trace ------------------
    S = rd.setup_r1()
    src = S['src_series']
    m5 = rd.build_mesh(rd.R1_WINDOW, 0.0, 0.0, 5.0)
    p5 = np.full(m5.nx, 1150.0)
    si = m5.index_of(src.md_ft)
    raw = src.raw_psi
    _, _, tr = rc.solve_forward_adaptive(
        m5.x, p5, src.t_total_s, src.taxis_s, raw, si,
        initial=np.full(m5.nx, raw[0]), dt_init=2.0, tol=1e-3,
        safety_factor=0.9, order_p=2, max_dt=30.0, min_dt=1e-4)
    ok &= check("T8c the adaptive driver reproduces the fibeRIS optimizer=True "
                "trace A3 measured on the manuscript protocol",
                tr['n_attempts'] == 43 and tr['n_accepted'] == 43
                and tr['n_rejected'] == 0
                and abs(tr['t_end_s'] - 1262.0) < 1e-9
                and abs(tr['frac_at_max_dt'] - 0.9767441860465116) < 1e-9
                and abs(tr['flip_margin'] - 0.5216) < 1e-3,
                f"{tr['n_attempts']} attempts / {tr['n_accepted']} accepted / "
                f"{tr['n_rejected']} rejected, dt {tr['dt_min_s']}->"
                f"{tr['dt_max_s']} s, {100 * tr['frac_at_max_dt']:.1f}% of steps "
                f"at max_dt, t_end {tr['t_end_s']:.3f} s (overshoot "
                f"{tr['overshoot_s']:.3f} s past t_total), err "
                f"{tr['err_min']:.2e}-{tr['err_max']:.2e} vs tol 1e-3, "
                f"flip_margin {tr['flip_margin']:.4f}")
    ok &= check("T8d the manuscript's 'adaptive' setting is in effect a fixed "
                "dt = 30 s run: the error control never binds",
                tr['err_max'] < 1e-3,
                f"largest error estimate {tr['err_max']:.3e} is "
                f"{1e-3 / tr['err_max']:.0f}x below tol")

    # The delta-pressure formulation is the case fibeRIS cannot run at all.
    # READ THIS TOGETHER WITH T10i: 'accept' clears the exact-zero FIRST step,
    # which is all fibeRIS needs to hang; whether the run then PROCEEDS depends on
    # the mesh, and on the production dx = 1 ft mesh it does not. This check runs
    # on the dx = 5 ft mesh m5, where the controller escapes by collapsing dt to
    # ~9e-4 s -- three orders of magnitude below the production dt = 1 s. Do not
    # read this PASS as "the delta formulation can be run adaptively".
    _, _, tr2 = rc.solve_forward_adaptive(m5.x, p5, 60.0, src.taxis_s,
                                          src.delta_psi, si)
    ok &= check("T8e zero_field_policy='accept' clears the exact-zero first step, "
                "where fibeRIS live-locks (0/0 -> nan -> reject -> dt pinned at "
                "min_dt forever); 'fiberis' reproduces the hang and is stopped by "
                "max_attempts. It is NOT a licence to run the delta formulation "
                "adaptively -- see T10i",
                tr2['n_accepted'] > 0
                and _raises(rc.solve_forward_adaptive, RuntimeError, m5.x, p5,
                            60.0, src.taxis_s, src.delta_psi, si,
                            zero_field_policy='fiberis', max_attempts=200),
                f"accept policy on the dx = 5 ft mesh: {tr2['n_accepted']} "
                f"accepted steps, t_end {tr2['t_end_s']:.1f} s, dt collapsing to "
                f"{tr2['dt_min_s']:.2e} s. NO fibeRIS reference exists for this "
                f"configuration, and at dx = 1 ft it does not run at all (T10i).")
    return ok


def t9_extras():
    """Rannacher start-up, the P0 forcing, RunRecorder, and the data loaders."""
    import datetime as _dt

    # --- Rannacher: CN rings at a restart discontinuity, BE and Rannacher do not
    m = rd.build_mesh(rd.R1_WINDOW, 5000.0, 0.0, 1.0)
    p = np.full(m.nx, 1150.0)
    si = m.index_of(16645.0)
    ta, da = np.array([0.0, 1e6]), np.array([300.0, 300.0])

    def reversals(theta, n_start):
        _, rec = rc.solve_forward(m.x, p, 1.0, 60.0, ta, da, si, theta=theta,
                                  theta_startup_steps=n_start,
                                  record_idx=[si - 1])
        d = np.diff(rec[:, 0])
        return int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))

    be, cn, rann = reversals(1.0, 0), reversals(0.5, 0), reversals(0.5, 4)
    ok = check("T9a Rannacher start-up removes the Crank-Nicolson ringing at a "
               "restart discontinuity (a 0 -> 300 psi datum jump on a zero field)",
               be == 0 and cn > 20 and rann == 0,
               f"increment sign reversals at the node next to the source in the "
               f"first 60 s: backward Euler {be}, CN {cn}, CN with 4 Rannacher "
               f"steps {rann}. Any multi-stage rerun with theta < 1 must set "
               f"theta_startup_steps.")

    # T9a2 (regression, defect 4): the SAME damping through the adaptive driver.
    # It had no theta_startup_steps at all, so an adaptive CN restart -- the
    # manuscript's own two-phase configuration -- could not be damped. The
    # protocol is absolute pressure (8000 psi baseline), because that is what
    # makes the relative error estimator inert (A3) and the run a de-facto fixed
    # dt = max_dt one; the datum jumps 300 psi at the restart.
    base = 8000.0
    ta_abs = np.array([0.0, 1e6])
    da_abs = np.full(2, base + 300.0)
    init = np.full(m.nx, base)

    def adaptive_restart(theta, n_start):
        _, rec, tr = rc.solve_forward_adaptive(
            m.x, p, 60.0, ta_abs, da_abs, si, initial=init, theta=theta,
            theta_startup_steps=n_start, record_idx=[si - 1], dt_init=1.0,
            max_dt=1.0, tol=1e-3, max_attempts=2000)
        d = np.diff(rec[:, 0])
        return int(np.sum(np.sign(d[1:]) != np.sign(d[:-1]))), tr

    abe, tbe = adaptive_restart(1.0, 0)
    acn, tcn = adaptive_restart(0.5, 0)
    aran, tran = adaptive_restart(0.5, 4)
    ok &= check("T9a2 solve_forward_adaptive takes theta_startup_steps with the "
                "same semantics, and it removes the adaptive CN restart ringing "
                "(defect 4)",
                abe == 0 and acn > 20 and aran == 0
                and tran['n_startup_steps_applied'] == 4
                and tcn['n_rejected'] > 0 and tran['n_rejected'] == 0
                and tbe['theta_startup_steps'] == 0,
                f"increment sign reversals at the node next to the source over "
                f"the first 60 s of an ADAPTIVE restart: backward Euler {abe}, "
                f"CN {acn}, CN with 4 Rannacher steps {aran}. The ringing also "
                f"breaks the step controller -- CN takes {tcn['n_rejected']} "
                f"rejections and collapses dt to {tcn['dt_min_s']:.3g} s "
                f"(err_max {tcn['err_max']:.2e} vs tol 1e-3), while the damped "
                f"run holds dt = {tran['dt_max_s']:g} s with "
                f"{tran['n_rejected']} rejections and err_max "
                f"{tran['err_max']:.2e}.")

    # --- P0: the sink relaxes toward P0, not toward zero
    D, L = 1150.0, 550.0
    lam = D / L ** 2
    mm = np.arange(0.0, 6000.0 + 1e-9, 1.0)
    pp = np.full(len(mm), D)
    worst = {}
    for th in (1.0, 0.5):
        _, r = rc.solve_forward(mm, pp, 1.0, 40000.0, np.array([0.0, 1e6]),
                                np.array([100.0, 100.0]), 0, lambda_leak=lam,
                                p0=20.0, theta=th)
        ana = 20.0 + 80.0 * np.exp(-mm / L)
        worst[th] = float(np.max(np.abs(r[-1][:4000] - ana[:4000])))
    ok &= check("T9b the p0 forcing gives the analytic steady state "
                "P0 + (Ps-P0)exp(-x*sqrt(lambda/D)) at every theta",
                max(worst.values()) < 1e-3,
                f"max|numerical - analytic| over 0-4000 ft: theta=1 "
                f"{worst[1.0]:.2e} psi, theta=0.5 {worst[0.5]:.2e} psi "
                f"(P0 = 0 in the delta formulation, so this path is dormant "
                f"there but available for an absolute-pressure run)")

    # --- RunRecorder writes on success and leaves a FAILED.json on an exception
    tmp = tempfile.mkdtemp(prefix='rev2_selftest_rr_')
    try:
        arrays = os.path.join(tmp, 'a.npz')
        np.savez(arrays, x=np.arange(3.0))
        taxis = np.arange(0.0, 11.0, 1.0)
        drv = rm.driver_record(kind='synthetic', baseline_removal='none_absolute_psi',
                               value_units='psi', taxis=taxis,
                               values=np.zeros_like(taxis))

        def groups(mesh_x):
            sp = rm.source_protocol(
                application='dirichlet_node', solver_class='rev2_core.solve_forward',
                placement_rule='explicit_md_list',
                sources=[rm.source_record(mesh_x, md_requested_ft=float(mesh_x[5]),
                                          mesh_idx=5, driver=drv)],
                targets=rm.NONE_DECLARED, time_level='n',
                phase_chaining=rm.NONE_DECLARED,
                boundary_conditions={'lbc': 'Neumann', 'rbc': 'Neumann'})
            nm = rm.numerics(
                time=rm.time_record(taxis, mode='fixed', theta=1.0,
                                    t_total_requested_s=10.0, dt_requested_s=1.0),
                mesh=rm.mesh_record(mesh_x, dx_requested_ft=1.0,
                                    window_md_ft=(float(mesh_x[0]),
                                                  float(mesh_x[-1])),
                                    pad_low_ft=0.0, pad_high_ft=0.0),
                interface_avg='harmonic',
                boundary={'lbc': 'Neumann', 'rbc': 'Neumann',
                          'pml_thickness': 0.0, 'sigma_max': 0.0},
                diffusivity={'baseline_D_ft2_s': 1150.0,
                             'profile_family': 'uniform', 'param_names': [],
                             'params': [], 'D_min': 1150.0, 'D_max': 1150.0,
                             'D_sha256': rm.sha256_array(np.full(len(mesh_x), 1150.0)),
                             'profile_anchor': 'physical_md'},
                barriers=rm.NONE_DECLARED, leakage=rm.NONE_DECLARED,
                kernel={'name': 'rev2_core.solve_forward', 'banded': True,
                        'equivalence_reference': 'rev2_selftest.py:T1'},
                rng=rm.NONE_DECLARED, parallel=rm.NONE_DECLARED)
            return sp, nm

        mesh_x = np.arange(0.0, 20.0, 1.0)
        man_ok = os.path.join(tmp, 'ok', 'manifest.json')
        os.makedirs(os.path.dirname(man_ok))
        shutil.copy(arrays, os.path.join(tmp, 'ok', 'a.npz'))
        sp, nm = groups(mesh_x)
        gauge_md = os.path.join(ROOT, 'data/fiberis_format/s_well/geometry/'
                                      'gauge_md_swell.npz')
        with rm.RunRecorder(man_ok, study_id='rr', task_id='A0', config={'a': 1},
                            config_path=None) as run:
            run.set_source(sp).set_numerics(nm)
            # A manifest with zero recorded inputs is scored 'incomplete', not
            # 'clean' (see T10d): it has no data provenance at all.
            run.declare_inputs([(gauge_md, 'geometry', 'gauge_md')])
            run.declare_output(os.path.join(tmp, 'ok', 'a.npz'), role='arrays_npz')
        rep = rm.verify(man_ok, repo_root=ROOT)
        good = rep['status'] == 'clean' and run.manifest is not None

        man_bad = os.path.join(tmp, 'bad', 'manifest.json')
        os.makedirs(os.path.dirname(man_bad))
        try:
            with rm.RunRecorder(man_bad, study_id='rr', task_id='A0',
                                config={}, config_path=None) as run2:
                run2.set_source(sp)
                raise RuntimeError("simulated crash mid-run")
        except RuntimeError:
            pass
        ok &= check("T9c RunRecorder writes the manifest on success and leaves a "
                    "<manifest>.FAILED.json with the traceback on a crash (the "
                    "failure mode required arguments alone cannot close)",
                    good and os.path.exists(man_bad + '.FAILED.json')
                    and not os.path.exists(man_bad),
                    f"clean manifest at {os.path.basename(man_ok)}; FAILED sidecar "
                    f"written, real manifest withheld")

        va = rm.verify_all(root=tmp)
        ok &= check("T9d verify_all walks a round directory and rolls up statuses",
                    va['_rollup']['n_clean'] >= 1
                    and len(va['_rollup']['failed_runs']) == 1,
                    f"rollup {va['_rollup']}")

        # T9d2 (regression, defect 2): the discovery globs were
        # <root>/*/manifest.json and <root>/manifest.json, so a descriptively
        # named manifest and anything one directory deeper were invisible -- 5 of
        # the 272 manifests in output/rev2_20260901 were audited, and the rollup
        # still read "clean".
        import glob as _g
        deep = os.path.join(tmp, 'deep', 'nested')
        os.makedirs(deep)
        man_named = os.path.join(tmp, 'deep', 'manifest_v2.json')
        man_deep = os.path.join(deep, 'manifest.json')
        for path in (man_named, man_deep):
            rm.write_manifest(path, study_id='rr', task_id='A0', config={'a': 1},
                              config_path=None,
                              inputs=[(gauge_md, 'geometry', 'gauge_md')],
                              source=sp, numerics=groups(mesh_x)[1], outputs=[],
                              allow_undeclared_outputs=True)
        old_glob = sorted(_g.glob(os.path.join(tmp, '*', 'manifest.json'))
                          + _g.glob(os.path.join(tmp, 'manifest.json')))
        found = rm.find_manifests(tmp)
        va2 = rm.verify_all(root=tmp)
        ok &= check("T9d2 discovery finds descriptively named manifests and "
                    "manifests below the first level, and the rollup reports how "
                    "many it found (defect 2)",
                    man_named in found and man_deep in found
                    and man_named not in old_glob and man_deep not in old_glob
                    and va2['_rollup']['n_found'] == len(found)
                    and len(found) == len(old_glob) + 2
                    and va2['_rollup']['n_clean'] >= 3,
                    f"the old two globs saw {len(old_glob)} manifest(s); "
                    f"find_manifests sees {len(found)} "
                    f"({os.path.basename(man_named)} and deep/nested/"
                    f"{os.path.basename(man_deep)} were both invisible before); "
                    f"rollup n_found={va2['_rollup']['n_found']}, "
                    f"n_clean={va2['_rollup']['n_clean']}")

        # --- layout: normalise into a NEW directory and round-trip
        d0 = os.path.join(ROOT, 'output', '0211_simulation_MULTIstage')
        panel = rl.load_panel(os.path.join(d0, 'phase3_test.npz'))
        dst = os.path.join(tmp, 'norm', 'phase3_test.npz')
        rl.save_panel(panel, dst)
        back = rl.load_panel(dst)
        stamped = rl.detect_layout(back.evidence['data_shape'], back.n_t, back.n_x,
                                   declared=rl.LAYOUT_DEPTH_MAJOR)[1]['rule']
        ok &= check("T9e save_panel stamps the layout and the stamped file "
                    "round-trips to an identical canonical array (rule 2, so a "
                    "future square panel resolves without inference)",
                    np.array_equal(panel.data, back.data)
                    and back.evidence['rule'] == 2 and stamped == 2,
                    f"re-read via rule {back.evidence['rule']} "
                    f"({back.detected_layout})")
        rows = rl.layout_report([os.path.join(d0, f) for f in
                                 ('phase1.npz', 'phase3_test.npz')]
                                + [os.path.join(tmp, 'does_not_exist.npz')])
        ok &= check("T9f layout_report never raises, so one bad file cannot abort "
                    "a directory audit",
                    rows[0]['layout'] == rl.LAYOUT_TIME_MAJOR
                    and rows[1]['layout'] == rl.LAYOUT_DEPTH_MAJOR
                    and 'error' in rows[2],
                    f"third row reports: {rows[2].get('error', '')[:70]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- data loaders against the measured reference values
    n, md, dd = rd.production_drawdown(_dt.datetime(2020, 4, 1),
                                       _dt.datetime(2020, 6, 1))
    ref = np.asarray(rd.DRAWDOWN_REFERENCE[('2020-04-01', '2020-06-01')])
    ok &= check("T9g production_drawdown reproduces the measured Apr-Jun 2020 "
                "profile with gauges in NUMERIC order (the manuscript figure used "
                "os.listdir order; sorted() is wrong too)",
                list(n) == list(range(1, 16))
                and float(np.max(np.abs(dd - ref))) < 1e-6,
                f"max|diff| from the reference = {np.max(np.abs(dd - ref)):.2e} psi; "
                f"g1 {dd[0]:.2f}, g7 {dd[6]:.2f}, g15 {dd[14]:.2f} psi")

    pw = rd.manuscript_phase_windows()
    ok &= check("T9h manuscript_phase_windows reproduces 101's phase boundaries, "
                "which are FILE SPANS (get_start_time ignores its threshold), not "
                "pumping events",
                pw['phase1'][1] == _dt.datetime(2020, 3, 18, 4, 24, 28)
                and pw['phase2'][2] == _dt.datetime(2020, 3, 18, 10, 59, 44)
                and pw['phase3'][2] == _dt.datetime(2020, 3, 18, 14, 39, 26)
                and pw['phase3'][0] == 7,
                f"phase1 {pw['phase1'][1]} - {pw['phase1'][2]} (source gauge 6); "
                f"phase3 source gauge {pw['phase3'][0]}. Stage-7 pumping actually "
                f"starts at {rd.load_pumping(7, curves=('slurry_rate',)).pumping_start()}")

    rec = rd.load_das_stage(1, md_range=(15000, 16750),
                            time_range=(rd.R1_WINDOW.t_start, rd.R1_WINDOW.t_end))
    keep = rec.artifact_mask(rec.default_artifact_spans)
    full = np.sqrt(np.mean(rec.data ** 2, axis=1))
    clean = np.sqrt(np.mean(rec.data[:, keep] ** 2, axis=1))
    infl = float(np.median(full / clean))
    ok &= check("T9i the LF-DAS stays in raw counts and the t~600 s acquisition "
                "artifact is attached but not applied; excluding it changes the "
                "per-channel window RMS by the measured factor",
                rec.units == 'counts' and rec.scale_applied is None
                and abs(infl - 11.17) < 0.05 and int(keep.sum()) == 1239,
                f"median RMS inflation if the artifact is left in: {infl:.2f}x "
                f"({int((~keep).sum())} of {keep.size} samples excluded). A C5 "
                f"threshold built on the un-excluded RMS would be ~11x too high.")
    return ok


def t10_repair2():
    """Regressions for the 2026-09-02 repair of the independently reviewed defects.

    Every check here FAILS on the pre-repair modules (measured: 1/8 passing on the
    frozen copy at output/rev2_20260901/A4_repair2/evidence/rev2_manifest_PRE.py)
    and passes after. The first is the round-blocking one: a manifest edited after
    it was written used to audit CLEAN.
    """
    import json
    import subprocess

    gauge_md = os.path.join(ROOT, 'data/fiberis_format/s_well/geometry/'
                                  'gauge_md_swell.npz')
    tmp = tempfile.mkdtemp(prefix='rev2_selftest_t10_')
    ok = True
    try:
        mesh = np.arange(0.0, 100.0, 1.0)
        ta = np.arange(0.0, 11.0, 1.0)

        def groups(parallel=rm.NONE_DECLARED):
            sp = rm.source_protocol(
                application='dirichlet_node',
                solver_class='rev2_core.solve_forward',
                placement_rule='nearest node',
                sources=[rm.source_record(
                    mesh, md_requested_ft=10.0, mesh_idx=10,
                    driver=rm.driver_record(
                        kind='synthetic', baseline_removal='none_absolute_psi',
                        value_units='psi', taxis=ta, values=np.zeros_like(ta)))],
                targets=rm.NONE_DECLARED, time_level='n',
                phase_chaining=rm.NONE_DECLARED,
                boundary_conditions={'lbc': 'Neumann', 'rbc': 'Neumann'})
            nm = rm.numerics(
                time=rm.time_record(ta, mode='fixed', theta=1.0,
                                    t_total_requested_s=10.0, dt_requested_s=1.0),
                mesh=rm.mesh_record(mesh, dx_requested_ft=1.0,
                                    window_md_ft=(0.0, 99.0), pad_low_ft=0.0,
                                    pad_high_ft=0.0),
                interface_avg='harmonic',
                boundary={'lbc': 'Neumann', 'rbc': 'Neumann',
                          'pml_thickness': 0, 'sigma_max': 0.0},
                diffusivity={'D_ft2_s': 1150.0}, barriers=rm.NONE_DECLARED,
                leakage=rm.NONE_DECLARED,
                kernel={'name': 'rev2_core.solve_forward'},
                rng=rm.NONE_DECLARED, parallel=parallel)
            return sp, nm

        def write(sub, **kw):
            d = os.path.join(tmp, sub)
            os.makedirs(d, exist_ok=True)
            prod = os.path.join(d, 'result.npz')
            np.savez(prod, x=np.arange(3.0))
            man = os.path.join(d, 'manifest.json')
            sp, nm = groups(kw.pop('parallel', rm.NONE_DECLARED))
            doc = rm.write_manifest(
                man, study_id='rev2_selftest', task_id='A-foundation',
                config=kw.pop('config', {'D': 1150.0, 'series': gauge_md}),
                config_path=None,
                inputs=kw.pop('inputs', [(gauge_md, 'geometry', 'gauge_md')]),
                source=sp, numerics=nm,
                outputs=[rm.output_decl(prod, role='arrays_npz')],
                results={'RMSE_psi': 11.872}, **kw)
            return man, doc

        # --- T10a THE MAJOR ONE -------------------------------------------------
        man, _ = write('tamper')
        doc = json.load(open(man))
        doc['results']['RMSE_psi'] = 1.0
        doc['numerics']['diffusivity']['D_ft2_s'] = 480.0
        doc['source_protocol']['sources'][0]['mesh_idx'] = 0
        json.dump(doc, open(man, 'w'), indent=2, sort_keys=True)
        rep = rm.verify(man)
        raised = _raises(rm.verify, rm.ManifestDrift, man, strict=True)
        cli = subprocess.run([sys.executable, rm.__file__, 'verify', man,
                              '--strict'], capture_output=True, text=True)
        ok &= check("T10a a manifest EDITED after it was written is scored drift, "
                    "--strict raises and the CLI exits non-zero (the sidecar loop "
                    "is closed; before the repair all three said 'clean')",
                    rep['status'] == 'drift' and raised
                    and cli.returncode != 0,
                    f"status={rep['status']}, strict raised={raised}, CLI exit="
                    f"{cli.returncode}; edits were results.RMSE_psi 11.872->1.0, "
                    f"numerics.diffusivity.D_ft2_s 1150->480, "
                    f"source_protocol.sources[0].mesh_idx 10->0")

        # --- T10b deleting the sidecar is not a way to silence the check --------
        os.remove(man + '.sha256')
        rep_b = rm.verify(man)
        cli_b = subprocess.run([sys.executable, rm.__file__, 'verify', man],
                               capture_output=True, text=True)
        ok &= check("T10b deleting <manifest>.sha256 leaves a rev2-schema "
                    "manifest at least 'incomplete', never 'clean'",
                    rep_b['status'] != 'clean' and cli_b.returncode != 0
                    and (rep_b['manifest_self'] or {}).get('status')
                    == 'sidecar_missing',
                    f"status={rep_b['status']}, CLI exit={cli_b.returncode}")

        # --- T10c no false positive --------------------------------------------
        man_c, _ = write('clean')
        rep_c = rm.verify(man_c)
        ok &= check("T10c an untouched manifest still verifies clean",
                    rep_c['status'] == 'clean' and not rep_c['lines'],
                    f"status={rep_c['status']}")

        # --- T10d zero inputs is not 'all inputs verified' ----------------------
        man_d, _ = write('hollow', inputs=[],
                         config={'template': gauge_md.replace(
                             'gauge_md_swell', 'gauge{n}_data_swell')})
        rep_d = rm.verify(man_d)
        ok &= check("T10d a rev2-schema manifest with ZERO recorded inputs is "
                    "flagged, not presented as 'inputs: 0 files, all ok' (the "
                    "guard used to live only on the legacy branch)",
                    rep_d['status'] == 'incomplete'
                    and rep_d['inputs'].get('not_recorded') is True,
                    f"status={rep_d['status']}, inputs={rep_d['inputs']}; the "
                    f"config path is a '{{n}}' template, which the auto-scan "
                    f"skips, so nothing was recorded")

        # --- T10e inline config hashes round-trip ------------------------------
        cases = {'plain_control': {'D': 1150.0, 'name': 'x'},
                 'numpy_int64': {'n': np.int64(7)},
                 'numpy_float64': {'x': np.float64(1.5)},
                 'NaN': {'seed': float('nan')},
                 'ndarray_8': {'a': np.arange(8.0)},
                 'ndarray_121': {'a': np.arange(121.0)},
                 'list_121': {'a': [float(i) for i in range(121)]},
                 'NONE_DECLARED': {'s': rm.NONE_DECLARED},
                 'tuple': {'t': (1.0, 2.0)}}
        bad = {}
        for name, cfg in cases.items():
            m, _ = write('cfg_' + name, config=cfg)
            r = rm.verify(m)
            if r['config'].get('match') is not True:
                bad[name] = r['config'].get('match')
        ok &= check("T10e an inline (config_path=None) config hash round-trips "
                    "for every type _jsonify rewrites (np.int64, NaN, ndarray, "
                    "list > max_array_len, NONE_DECLARED)",
                    not bad,
                    f"{len(cases)} config types tested, config-hash mismatches: "
                    f"{bad or 'none'}. The writer now hashes the JSONIFIED "
                    f"payload, which is the object verify() re-hashes.")

        # --- T10f an undeclared PRODUCT named in the config -------------------
        d = os.path.join(tmp, 'product')
        os.makedirs(d)
        started = rm.datetime.datetime.now(rm.datetime.timezone.utc).isoformat()
        log = os.path.join(d, 'run.log')
        open(log, 'w').write('line 1\n')
        derived = os.path.join(d, 'derived.npz')
        np.savez(derived, y=np.arange(4.0))
        prod = os.path.join(d, 'result.npz')
        np.savez(prod, x=np.arange(3.0))
        man_f = os.path.join(d, 'manifest.json')
        sp, nm = groups()
        doc_f = rm.write_manifest(
            man_f, study_id='rev2_selftest', task_id='A-foundation',
            config={'data': {'series': derived}, 'outputs': {'log': log}},
            config_path=None, inputs=[(gauge_md, 'geometry', 'gauge_md')],
            source=sp, numerics=nm,
            outputs=[rm.output_decl(prod, role='arrays_npz')],
            started_utc=started, allow_undeclared_outputs=True, results={})
        open(log, 'a').write('appended after the manifest was written\n')
        rep_f = rm.verify(man_f)
        excl = {e['config_path']: e['reason']
                for e in doc_f['outputs']['config_paths_excluded_from_input_scan']}
        ok &= check("T10f an undeclared product under an 'outputs' config key is "
                    "NOT hashed as an input (D2's 8 drift rows), while a file "
                    "DERIVED in-process under a data key still is",
                    'auto:outputs.log' not in doc_f['inputs']
                    and 'auto:data.series' in doc_f['inputs']
                    and excl.get('outputs.log') == 'written_during_run'
                    and rep_f['status'] == 'clean',
                    f"excluded={excl}; inputs={sorted(doc_f['inputs'])}; status "
                    f"after appending to the log = {rep_f['status']} (it was "
                    f"'drift', on 8 real D2 manifests, before the repair)")

        # --- T10g worker-only imports escape the closure -----------------------
        pool = {'mode': 'multiprocessing.Pool', 'n_workers': 6}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            man_g, _ = write('workers', parallel=pool)
        rep_g = rm.verify(man_g)
        kinds_g = [x['kind'] for x in json.load(open(man_g))['discrepancies']]
        man_h, _ = write('workers_declared', parallel=pool, worker_modules=())
        rep_h = rm.verify(man_h)
        kinds_h = [x['kind'] for x in json.load(open(man_h))['discrepancies']]
        ok &= check("T10g a Pool run that declares no worker modules is recorded "
                    "as incomplete (code_closure scans the PARENT's sys.modules "
                    "only), and worker_modules=() is the explicit assertion",
                    'worker_code_closure_undeclared' in kinds_g
                    and rep_g['status'] == 'incomplete'
                    and any(issubclass(x.category, RuntimeWarning) for x in w)
                    and 'worker_code_closure_undeclared' not in kinds_h
                    and rep_h['status'] == 'clean',
                    f"undeclared: status={rep_g['status']} {kinds_g}; "
                    f"declared: status={rep_h['status']} {kinds_h}")

        # --- T10h the mesh guard now protects the SOLVER -----------------------
        dirty = np.sort(np.concatenate([np.arange(0.0, 120.0, 1.0),
                                        [60.0 + 9e-13]]))
        tad = np.arange(0.0, 201.0, 1.0)
        sdd = np.full_like(tad, 300.0)
        _, u_r1 = r1core.solve_forward(dirty, np.full(len(dirty), 300.0), 1.0,
                                       200.0, tad, sdd, 0)
        peak = float(np.max(np.abs(u_r1)))
        ok &= check("T10h solve_forward now rejects a mesh with a near-duplicate "
                    "node, the hazard _check_mesh's docstring advertises but only "
                    "build_barrier_profile used to catch",
                    _raises(rc.solve_forward, ValueError, dirty,
                            np.full(len(dirty), 300.0), 1.0, 200.0, tad, sdd, 0)
                    and rc.solve_forward(np.arange(0.0, 120.0, 1.0),
                                         np.full(120, 300.0), 1.0, 200.0, tad,
                                         sdd, 0)[1] is not None,
                    f"the same mesh (min gap 9.02e-13 ft) returns a finite field "
                    f"peaking at {peak:.4f} psi from r1_calibration_core against "
                    f"a constant 300 psi Dirichlet datum -- a "
                    f"{100 * (peak / 300 - 1):.2f} % overshoot of its own "
                    f"boundary "
                    f"value. The gap was inherited from r1, not a rev2 "
                    f"regression.")

        # --- T10i the adaptive driver cannot integrate the delta formulation ----
        mesh_i = np.arange(0.0, 500.0 + 1e-9, 1.0)
        D_i = np.full(len(mesh_i), 1150.0)
        ta_i = np.arange(0.0, 121.0, 1.0)
        step = np.where(ta_i <= 0.0, 0.0, 300.0)
        _, _, tr_probe = rc.solve_forward_adaptive(
            mesh_i, D_i, 5.0, ta_i, step, 0, dt_init=2.0, tol=1e9, max_dt=2.0,
            min_dt=2.0, max_attempts=20)
        err2 = [a for a in tr_probe['attempts'] if a['t'] > 0][0]['err']
        locked = _raises(rc.solve_forward_adaptive, RuntimeError, mesh_i, D_i,
                         60.0, ta_i, step, 0, dt_init=2.0, tol=1e-3, max_dt=30.0,
                         min_dt=1e-4, zero_field_policy='accept',
                         max_attempts=800)
        _, _, tr_abs = rc.solve_forward_adaptive(
            mesh_i, D_i, 300.0, ta_i, 8300.0 + step, 0,
            initial=np.full(len(mesh_i), 8300.0), dt_init=2.0, tol=1e-3,
            max_dt=30.0, min_dt=1e-4, max_attempts=5000)
        n_rej = int(sum(not a['accepted'] for a in tr_abs['attempts']))
        ok &= check("T10i zero_field_policy='accept' does NOT make the adaptive "
                    "driver runnable on the delta-pressure formulation at dx = 1 "
                    "ft: it clears the exact-zero first step and then live-locks "
                    "on the next one (fixed dt = 1 s is the only option)",
                    locked and len(tr_abs['attempts']) > 0,
                    f"relative whole-field error estimate at the second attempt "
                    f"= {err2:.4e} against tol = 1e-3, and it clears tol only "
                    f"at dt = 4.171e-05 s, 2.4x below the default min_dt = 1e-4, "
                    f"so dt is pinned there and nothing is accepted -> "
                    f"RuntimeError at max_attempts (B1 measured the same stall on "
                    f"the production mesh: dt <= 3.45e-05 s needed). "
                    f"The same mesh and the same controls on the "
                    f"ABSOLUTE-pressure protocol run to completion "
                    f"({len(tr_abs['attempts'])} attempts, {n_rej} rejections).")
        # --- T10j the 2w + dx capture trap is assertable ------------------------
        hits = [16405.45, 16430.91, 16489.09, 16514.54, 16558.18, 16583.63]
        rounded = [float(np.round(h)) for h in hits]
        true_w, round_w, flags = {}, {}, []
        for dx in (1.0, 0.5, 0.2, 0.1):
            mm = rd.build_mesh((15000.0, 16750.0), 5000.0, 0.0, dx)
            pt, rt = rc.build_barrier_profile(mm.x, 1150.0, hits, 1.0, 1e-2,
                                              return_report=True)
            pr, rr = rc.build_barrier_profile(mm.x, 1150.0, rounded, 1.0, 1e-2,
                                              return_report=True)
            true_w[dx] = (rt['realised_full_width_ft']['min'],
                          rt['realised_full_width_ft']['max'],
                          rc.barrier_equivalent_width(mm.x, pt, 1150.0, 1e-2)
                          / len(hits))
            round_w[dx] = (rr['realised_full_width_ft']['max'],
                           rc.barrier_equivalent_width(mm.x, pr, 1150.0, 1e-2)
                           / len(rounded))
            flags.append((rt['n_fallback'], rt['n_width_inflated'],
                          rr['n_fallback'], rr['n_width_inflated']))
        ok &= check("T10j the realised barrier width is 2w for the true frac-hit "
                    "MDs at every dx, and exactly 2w + dx when the requested edges "
                    "land on nodes; n_fallback cannot see the difference but "
                    "n_width_inflated can",
                    all(abs(v[0] - 2.0) < 1e-9 and abs(v[1] - 2.0) < 1e-9
                        and abs(v[2] - 2.0) < 1e-9 for v in true_w.values())
                    and all(abs(round_w[dx][0] - (2.0 + dx)) < 1e-9
                            and abs(round_w[dx][1] - (2.0 + dx)) < 1e-9
                            for dx in round_w)
                    and all(f == (0, 0, 0, 6) for f in flags),
                    f"true MDs: realised = control-volume span = "
                    f"excess-resistance equivalent = 2.000 ft at dx = "
                    f"1.0/0.5/0.2/0.1; rounded MDs: "
                    f"{[round(round_w[d][0], 3) for d in (1.0, 0.5, 0.2, 0.1)]} ft "
                    f"= 2w + dx, and the excess-resistance equivalent agrees to "
                    f"1e-9. (n_fallback, n_width_inflated) per dx: {flags}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok


def _raises(fn, exc_type, *a, **kw):
    try:
        fn(*a, **kw)
    except exc_type:
        return True
    except Exception:
        return False
    return False


def main():
    print("rev2 foundation self-test")
    print(f"repo root: {ROOT}")
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}")
    print("")
    for fn in (t1_kernel_identity, t2_leakage, t3_interface_avg, t4_barriers,
               t5_theta_order, t6_layout, t7_manifest,
               t8_multisource_and_adaptive, t9_extras, t10_repair2):
        try:
            fn()
        except Exception as exc:  # a crashing test is a failing test
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} raised {type(exc).__name__}", False, str(exc))
        print("")
    n_fail = sum(1 for _, p in RESULTS if not p)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    if n_fail:
        print("FAILED: " + ", ".join(n for n, p in RESULTS if not p))
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
