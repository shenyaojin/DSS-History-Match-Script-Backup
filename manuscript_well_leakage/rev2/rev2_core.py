"""Shared rev2 numerics: physical-width barriers and a theta-generalised kernel.

This module is a strict generalisation of
`baseline_calibration/r1_calibration_core.solve_forward`, which is already proven
bit-equivalent to fibeRIS PDS1D_SingleSource. Everything here is arranged so that
the defaults collapse onto that kernel exactly, because that identity is the only
thing that transfers the fibeRIS equivalence proof to rev2 -- re-verifying against
the dense fibeRIS solve on every configuration is unaffordable and is forbidden by
the house rules. `rev2_selftest.py` asserts the identity with `np.array_equal`.

Three generalisations live here, each of which the revision round needs:

* `build_barrier_profile` defines a barrier by a PHYSICAL half-width `w` in ft
  instead of by mesh index. The legacy scripts set a single node, whose physical
  meaning is whatever the local dx happens to be; measured across the five call
  sites that spans 0.0048-5.01 ft (~1000x), which makes their fitted reduction
  ratios mutually incomparable.
* `theta` selects the time weighting (1.0 = backward Euler, what fibeRIS does;
  0.5 = Crank-Nicolson). The memo's claim that the code was already Crank-Nicolson
  traced to a stale comment in a file containing no discretisation.
* `lambda_leak` adds a linear sink -lambda*(P - P0) for the C2 leakage model, and
  `interface_avg` exposes the face-averaging rule that fibeRIS hard-codes.

Units throughout: D in ft^2/s, MD in ft, pressure in psi, time in s,
lambda_leak in s^-1, w (barrier HALF-width) in ft.
"""

import warnings

import numpy as np
from scipy.linalg import solve_banded

__all__ = [
    'BarrierWidthWarning', 'BarrierOverlapWarning',
    'build_barrier_profile', 'barrier_equivalent_width',
    'face_diffusivity', 'amplification_factor',
    'solve_forward', 'solve_forward_multi', 'solve_forward_adaptive',
]


# ---------------------------------------------------------------------------
# Barriers with a physical width
# ---------------------------------------------------------------------------

class BarrierWidthWarning(UserWarning):
    """Requested half-width captured no node; the nearest node was used instead."""


class BarrierOverlapWarning(UserWarning):
    """Two or more barriers share nodes and were merged into one group."""


def _check_mesh(mesh, name='mesh'):
    """Validate and return a float64 copy of a node array.

    The strict-monotonicity floor of 1e-9 ft is not pedantry. `np.unique` on
    concatenated float grids routinely leaves node pairs ~1e-12 ft apart; one such
    mesh raised the no-barrier fine-vs-coarse RMSE of an A1 reference solve from
    7.5e-7 psi to 22.12 psi while looking entirely plausible. The failure is silent
    and large, so it is caught here rather than downstream.

    Called from `_coefficients`, so `solve_forward`, `solve_forward_multi`,
    `solve_forward_adaptive` and `amplification_factor` all inherit it, as well as
    from `build_barrier_profile` / `barrier_equivalent_width`. Until the
    2026-09-02 repair only the two barrier entry points checked. Measured on
    `np.sort(np.concatenate([np.arange(0, 120, 1.0), [60.0 + 9e-13]]))` (min gap
    9.02e-13 ft) with D = 300 ft^2/s and a constant 300 psi Dirichlet datum at node
    0: the pre-repair `solve_forward` returned a finite field peaking at
    324.3075168 psi -- an 8.10 % overshoot of its own boundary value, silently --
    and `r1_calibration_core.solve_forward` returns the IDENTICAL 324.3075168 psi,
    so the gap was inherited from r1, not introduced by rev2. On the same mesh
    de-duplicated the peak is 300.0000000002 psi. The check is O(nx) once per
    solve (it lives in `_coefficients`, outside the time loop), never per step.
    """
    m = np.asarray(mesh, dtype=float)
    if m.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got {m.ndim}-D")
    if m.size < 3:
        raise ValueError(f"{name} must have at least 3 nodes, got {m.size}")
    if not np.all(np.isfinite(m)):
        bad = int(np.argmin(np.isfinite(m)))
        raise ValueError(f"{name}[{bad}] is not finite ({m[bad]})")
    d = np.diff(m)
    if d.min() <= 1e-9:
        i = int(np.argmin(d))
        raise ValueError(
            f"{name} must be strictly increasing with spacing > 1e-9 ft; "
            f"{name}[{i}]={m[i]!r} and {name}[{i + 1}]={m[i + 1]!r} differ by "
            f"{d[i]:.6e} ft")
    return m


def _broadcast_d_base(d_base, nx):
    """Return (d_new_seed, is_array, scalar_value) as float64 of length nx.

    Never `np.ones_like(mesh)`: the 101/106 meshes start life as an int64
    `np.arange`, so an ones_like would truncate D = 1.4e-3 to 0 and produce a
    perfectly sealed barrier that looks like an excellent fit.
    """
    arr = np.asarray(d_base, dtype=float)
    if arr.ndim == 0:
        val = float(arr)
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f"d_base must be finite and > 0, got {val!r}")
        return np.full(nx, val, dtype=float), False, val
    if arr.ndim != 1 or arr.size != nx:
        raise ValueError(
            f"d_base must be a scalar or shape ({nx},), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        i = int(np.argmin(np.isfinite(arr)))
        raise ValueError(f"d_base[{i}] is not finite ({arr[i]})")
    if np.any(arr <= 0.0):
        i = int(np.argmin(arr))
        raise ValueError(f"d_base[{i}] = {arr[i]!r} is not > 0")
    return arr.astype(float, copy=True), True, None


def _stats(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {'min': None, 'median': None, 'max': None, 'mean': None}
    return {'min': float(np.min(a)), 'median': float(np.median(a)),
            'max': float(np.max(a)), 'mean': float(np.mean(a))}


def _series_resistance(mesh, d_profile):
    """sum_f dx_f / harmonic_mean(D_f, D_{f+1}); units s/ft.

    This is the one barrier metric that is comparable across meshes, which the
    node count and the "N-node window" phrasing are not.
    """
    dx = np.diff(mesh)
    dh = 2.0 * d_profile[:-1] * d_profile[1:] / (d_profile[:-1] + d_profile[1:])
    return float(np.sum(dx / dh))


def barrier_equivalent_width(mesh, d_profile, d_base, ratio):
    """Physical full width of the uniform slab with the same series resistance.

    Reads an ARBITRARY D(x) -- including the legacy single-node and 9-node "tent"
    profiles -- and reports what width of D = d_base*ratio it is equivalent to.
    This is what makes the legacy call sites comparable with each other and with
    the new `w`-based barriers: a tent whose shoulders are only a 4x reduction has
    the resistance of its centre node alone, so its 0.8 ft support is physically a
    0.1 ft barrier.
    """
    mesh = _check_mesh(mesh)
    prof = np.asarray(d_profile, dtype=float)
    base, _, _ = _broadcast_d_base(d_base, len(mesh))
    ratio = float(ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must satisfy 0 < ratio <= 1, got {ratio!r}")
    if ratio == 1.0:
        return 0.0
    excess = _series_resistance(mesh, prof) - _series_resistance(mesh, base)
    # A slab of width W at D*ratio replacing background D adds
    # W*(1/(D*ratio) - 1/D) of resistance. Use the mean background so an array
    # d_base still yields a single number.
    d_ref = float(np.mean(base))
    return float(excess / (1.0 / (d_ref * ratio) - 1.0 / d_ref))


def build_barrier_profile(mesh, d_base, barrier_mds, w, ratio, *,
                          ratio_reference='local',
                          combine='min',
                          select_tol=None,
                          on_empty='nearest',
                          on_outside='raise',
                          return_report=False):
    """Reduce D to `ratio` over a physical half-width `w` around each frac hit.

    fibeRIS is a conservative finite-volume scheme with harmonic face averaging
    (matbuilder.py:22-23, 27-28), so node k owns control volume
    (dx_left + dx_right)/2 and setting node k to D0*ratio is EXACTLY a slab of
    D0*ratio spanning [x_k - dx_left/2, x_k + dx_right/2] -- algebraically exact
    for every ratio, verified to 5.8e-16 relative over 20000 random draws and to
    0.3% in a transient solve against a 200x-refined reference. The realised width
    reported here is therefore the sum of the captured nodes' control volumes,

        realised_full_width = (mesh[i1] - mesh[i0]) + dx_left(i0)/2 + dx_right(i1)/2

    and NOT the node count times dx, and NOT mesh[i1] - mesh[i0] (which is 0 for a
    single node and so cannot reduce to the legacy behaviour).

    `w` is the HALF-width, in ft, matching the house-rules symbol table; the
    requested full width is 2w. Every report field is named _half_ or _full_
    because a factor-of-2 slip here would be absorbed invisibly by the fitted
    ratio while making every width quoted in the paper wrong by 2x.

    Parameters
    ----------
    mesh : (nx,) node MDs, ft, strictly increasing.
    d_base : scalar or (nx,) baseline diffusivity, ft^2/s. The array form is
        required: C2's leakage runs and the two_zone D(x) pass a full profile.
    barrier_mds : frac-hit MDs, ft; any order, duplicates allowed, empty legal.
    w : half-width, ft, >= 0. w = 0 with on_empty='nearest' reproduces the legacy
        single-node assignment exactly.
    ratio : D_barrier / D_baseline, 0 < ratio <= 1. ratio = 1 is the null control
        and returns the baseline bit-for-bit.
    ratio_reference : 'local' scales each captured node by its own d_base;
        'at_hit' uses one D interpolated at the hit MD for the whole barrier, so a
        (w, D_barrier) trade-off has a single well-defined D_barrier even on a
        steep D(x).
    combine : 'min' only. See below.
    select_tol : capture tolerance, ft; None -> 1e-9*max(1, |x_hit|).
    on_empty : 'nearest' (fall back to the nearest node, warn, and record it) or
        'raise'. Sweeps that must not silently change their own model pass 'raise'.
    on_outside : 'raise' | 'clip' | 'skip' for a barrier entirely off the mesh.
    return_report : also return the JSON-serialisable report dict.

    Notes
    -----
    `min` combination is required rather than assignment. It is order-independent
    and idempotent -- the 113 frac hits contain an exact duplicate (MD 16696.914),
    so assignment semantics would make the answer depend on os.listdir order -- and
    it is the only rule that cannot RAISE D. The legacy 9-node tent uses assignment
    with shoulder values of exactly 1.0, so an overlapping window resets its
    neighbour back to baseline; that happens for 69 window pairs in
    103_matching_prod_final_fatalwrong2.py. Never reproduce it.

    THE REPORT IS THE AUTHORITY, NOT THE WARNING. Warnings are silenced by any
    global filter and do not cross a multiprocessing.Pool boundary, so a runner
    must assert on `report['n_fallback'] == 0`, never on a captured warning.

    ASSERT ON `report['n_width_inflated'] == 0` AS WELL. Capture is the CLOSED
    interval [x_hit - w, x_hit + w] (plus `select_tol`), so a hit whose requested
    EDGES land on nodes captures one node more than a generically placed one and
    realises 2w + dx rather than 2w -- +50 % of barrier resistance at dx = 1 ft,
    w = 1 ft -- while `n_fallback` stays 0. Measured 2026-09-02: the true stage-2
    frac-hit MDs (16405.45, 16430.91, ...) realise exactly 2.000 ft at
    dx = 1.0/0.5/0.2/0.1 ft, and the same MDs ROUNDED to integers realise
    3.0/2.5/2.2/2.1 ft, i.e. exactly 2w + dx. For an MD drawn at random the
    coincidence has probability ~0 (measured 0/2000 at dx = 1.0 ft); it is
    constructed placements that hit it -- A1's `physical_w_rounded_md` control arm
    deliberately, and A2's node-centred MD 16515.0 (realised 3.0/2.5/2.25 ft at
    dx = 1.0/0.5/0.25 ft) as a side effect of wanting a symmetric node count. A
    grid-independence study must not choose such an MD, or it measures a
    first-order-in-dx change of the model dressed up as convergence.

    The three ways of naming the realised width -- node count x dx, the sum of the
    captured control volumes (what this reports), and the width of the uniform
    D0*ratio slab with the same excess series resistance
    (`barrier_equivalent_width`) -- are the SAME NUMBER on a uniform mesh, verified
    to machine precision on both configurations above. Disagreement about a
    realised width is therefore always about the input, never about the metric.
    """
    mesh = _check_mesh(mesh)
    nx = len(mesh)
    d_new, d_is_array, d_scalar = _broadcast_d_base(d_base, nx)
    d_ref = d_new.copy()  # untouched baseline, needed for the resistance metric

    if combine != 'min':
        raise ValueError(f"combine must be 'min' in v1, got {combine!r}")
    if ratio_reference not in ('local', 'at_hit'):
        raise ValueError(f"ratio_reference must be 'local' or 'at_hit', "
                         f"got {ratio_reference!r}")
    if on_empty not in ('nearest', 'raise'):
        raise ValueError(f"on_empty must be 'nearest' or 'raise', got {on_empty!r}")
    if on_outside not in ('raise', 'clip', 'skip'):
        raise ValueError(f"on_outside must be 'raise', 'clip' or 'skip', "
                         f"got {on_outside!r}")

    ratio = float(ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must satisfy 0 < ratio <= 1, got {ratio!r}")
    w = float(w)
    if not np.isfinite(w) or w < 0.0:
        raise ValueError(f"w (half-width, ft) must be finite and >= 0, got {w!r}")

    mds = np.asarray(barrier_mds, dtype=float).ravel() if np.size(barrier_mds) \
        else np.zeros(0, dtype=float)
    if mds.size and not np.all(np.isfinite(mds)):
        i = int(np.argmin(np.isfinite(mds)))
        raise ValueError(f"barrier_mds[{i}] is not finite ({mds[i]})")

    dx = np.diff(mesh)
    records = []
    fallback_msgs = []
    n_skipped = 0

    for j, x_hit in enumerate(mds):
        tol = float(select_tol) if select_tol is not None \
            else 1e-9 * max(1.0, abs(float(x_hit)))

        outside = (x_hit + w < mesh[0]) or (x_hit - w > mesh[-1])
        if outside:
            msg = (f"barrier {j} at MD {x_hit:.3f} ft with w={w:g} ft lies entirely "
                   f"outside the mesh [{mesh[0]:.3f}, {mesh[-1]:.3f}] ft")
            if on_outside == 'raise':
                raise ValueError(msg + " (pass on_outside='clip' or 'skip' to allow)")
            if on_outside == 'skip':
                warnings.warn(msg + "; skipped", BarrierWidthWarning, stacklevel=2)
                n_skipped += 1
                continue
            warnings.warn(msg + "; clipped to the nearest end node",
                          BarrierWidthWarning, stacklevel=2)
            k = 0 if x_hit < mesh[0] else nx - 1
            i0 = i1 = k
            fallback = True
        else:
            # Node-based selection: the solver stores D at nodes and averages
            # harmonically at faces, so a cell-based rule would not correspond to
            # the quantity that is actually reduced. O(log nx), never a node loop.
            i0 = int(np.searchsorted(mesh, x_hit - w - tol, side='left'))
            i1 = int(np.searchsorted(mesh, x_hit + w + tol, side='right')) - 1
            fallback = False
            if i1 < i0:
                # On any mesh no barrier can be narrower than one control volume;
                # this is physics, not a code limitation, so report the inflation
                # rather than pretend the request was honoured.
                k = int(np.argmin(np.abs(mesh - x_hit)))  # ties -> lowest index,
                # exactly mesh_utils.locate, so the w=0 case stays bit-compatible
                # with the legacy call sites.
                i0 = i1 = k
                fallback = True
                dxl = mesh[k] - mesh[k - 1] if k > 0 else 0.0
                dxr = mesh[k + 1] - mesh[k] if k < nx - 1 else 0.0
                local_dx = (dxl + dxr) / 2.0 if (dxl and dxr) else max(dxl, dxr)
                realised_half = (dxl / 2.0 + dxr / 2.0) / 2.0
                msg = (f"barrier at MD {x_hit:.6g} ft: requested half-width "
                       f"w={w:g} ft captures no node (local dx {local_dx:.4f} ft); "
                       f"fell back to the nearest node (index {k}, MD "
                       f"{mesh[k]:.6g} ft). Realised half-width "
                       f"{realised_half:.4f} ft"
                       + (f" = {realised_half / w:.2f}x the requested value"
                          if w > 0 else "")
                       + ". Refine the mesh or increase w.")
                if on_empty == 'raise':
                    raise ValueError(msg)
                fallback_msgs.append(msg)
                warnings.warn(msg, BarrierWidthWarning, stacklevel=2)

        dxl = float(mesh[i0] - mesh[i0 - 1]) if i0 > 0 else 0.0
        dxr = float(mesh[i1 + 1] - mesh[i1]) if i1 < nx - 1 else 0.0
        full = float(mesh[i1] - mesh[i0]) + dxl / 2.0 + dxr / 2.0
        span = [float(mesh[i0] - dxl / 2.0), float(mesh[i1] + dxr / 2.0)]

        if ratio_reference == 'local':
            d_target = d_ref[i0:i1 + 1] * ratio
        else:
            d_target = np.full(i1 - i0 + 1,
                               float(np.interp(x_hit, mesh, d_ref)) * ratio)
        d_new[i0:i1 + 1] = np.minimum(d_new[i0:i1 + 1], d_target)

        records.append({
            'md_ft': float(x_hit),
            'i0': int(i0), 'i1': int(i1), 'n_nodes': int(i1 - i0 + 1),
            'realised_full_width_ft': full,
            'realised_half_width_ft': full / 2.0,
            'realised_span_ft': span,
            'center_offset_ft': float(0.5 * (span[0] + span[1]) - x_hit),
            'width_inflation': float(full / (2.0 * w)) if w > 0 else float('nan'),
            'dx_left_ft': dxl, 'dx_right_ft': dxr,
            'd_barrier_min_ft2_s': float(np.min(d_target)),
            'd_barrier_max_ft2_s': float(np.max(d_target)),
            'fallback': bool(fallback),
            'touches_domain_edge': bool(i0 == 0 or i1 == nx - 1),
            'overlaps': [], 'group': -1,
        })

    # ---- overlap detection and merging ------------------------------------
    # Overlap is legitimate (the two closest real frac hits are 8.114 ft apart, so
    # w = 5 ft genuinely merges them) but the merged group, not the barrier list,
    # is what the total equivalent width must be summed over -- otherwise the
    # shared nodes are double-counted and the quoted resistance is wrong.
    n_b = len(records)
    pairs = []
    parent = list(range(n_b))

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(n_b):
        for b in range(a + 1, n_b):
            if records[a]['i0'] <= records[b]['i1'] and \
                    records[b]['i0'] <= records[a]['i1']:
                pairs.append([a, b])
                records[a]['overlaps'].append(b)
                records[b]['overlaps'].append(a)
                ra, rb = _find(a), _find(b)
                if ra != rb:
                    parent[ra] = rb

    groups = {}
    for a in range(n_b):
        groups.setdefault(_find(a), []).append(a)
    merged = []
    for gi, (_, members) in enumerate(sorted(groups.items())):
        for a in members:
            records[a]['group'] = gi
        j0 = min(records[a]['i0'] for a in members)
        j1 = max(records[a]['i1'] for a in members)
        gl = float(mesh[j0] - mesh[j0 - 1]) / 2.0 if j0 > 0 else 0.0
        gr = float(mesh[j1 + 1] - mesh[j1]) / 2.0 if j1 < nx - 1 else 0.0
        merged.append({
            'members': [int(a) for a in members],
            'span_ft': [float(mesh[j0] - gl), float(mesh[j1] + gr)],
            'equivalent_full_width_ft': float(mesh[j1] - mesh[j0]) + gl + gr,
        })

    if pairs:
        warnings.warn(
            f"{len(pairs)} overlapping barrier pair(s) merged into "
            f"{len(merged)} group(s): "
            + ", ".join(f"({p[0]},{p[1]})" for p in pairs[:12])
            + (" ..." if len(pairs) > 12 else ""),
            BarrierOverlapWarning, stacklevel=2)

    if not return_report:
        return d_new

    widths = [r['realised_full_width_ft'] for r in records]
    infl = [r['width_inflation'] for r in records] if w > 0 else []
    report = {
        'w_requested_ft': w,
        'full_width_requested_ft': 2.0 * w,
        'ratio': ratio,
        'ratio_reference': ratio_reference,
        'combine': combine,
        'select_tol_ft': None if select_tol is None else float(select_tol),
        'on_empty': on_empty,
        'on_outside': on_outside,
        'd_base_is_array': bool(d_is_array),
        'd_base_scalar_ft2_s': d_scalar,
        'n_barriers': int(n_b),
        'n_unique_mds': int(np.unique(mds).size),
        'n_fallback': int(sum(1 for r in records if r['fallback'])),
        # Capture is a CLOSED interval, so a hit whose requested edges land on
        # nodes takes one node more than a generically placed one and realises
        # 2w + dx. n_fallback stays 0 in every such case, so a caller that asserts
        # only on n_fallback cannot see it -- assert on this too.
        'n_width_inflated': int(sum(
            1 for r in records
            if w > 0.0 and r['realised_full_width_ft'] > 2.0 * w * (1.0 + 1e-9))),
        'n_overlapping_pairs': int(len(pairs)),
        'n_merged_groups': int(len(merged)),
        'n_touching_domain_edge': int(sum(1 for r in records
                                          if r['touches_domain_edge'])),
        'n_skipped_outside': int(n_skipped),
        'mesh_nx': int(nx),
        'mesh_dx_min_ft': float(dx.min()),
        'mesh_dx_max_ft': float(dx.max()),
        'realised_full_width_ft': _stats(widths),
        'width_inflation': ({'min': None, 'median': None, 'max': None}
                            if not infl else
                            {k: _stats(infl)[k] for k in ('min', 'median', 'max')}),
        'n_nodes_captured': _stats([r['n_nodes'] for r in records]),
        'total_equivalent_width_ft': float(
            sum(g['equivalent_full_width_ft'] for g in merged)),
        'excess_resistance_s_per_ft': float(
            _series_resistance(mesh, d_new) - _series_resistance(mesh, d_ref)),
        'merged_groups': merged,
        'barriers': records,
        'fallback_messages': fallback_msgs,
    }
    return d_new, report


# ---------------------------------------------------------------------------
# Discretisation coefficients
# ---------------------------------------------------------------------------

def face_diffusivity(diffusivity, interface_avg='harmonic'):
    """Face diffusivity between neighbouring nodes.

    The harmonic mean is what fibeRIS hard-codes (matbuilder.py:22-23) and is the
    correct rule for conductances in series, which is exactly what two adjacent
    control volumes are. The arithmetic option exists only so A2 can quantify what
    the wrong choice would have cost; it is not a physically defensible default,
    and it breaks the exact single-node/slab identity that
    `build_barrier_profile`'s reported widths rest on.
    """
    d = np.asarray(diffusivity, dtype=float)
    if interface_avg == 'harmonic':
        return (2.0 * d[:-1] * d[1:] / (d[:-1] + d[1:]))
    if interface_avg == 'arithmetic':
        return 0.5 * (d[:-1] + d[1:])
    raise ValueError(f"interface_avg must be 'harmonic' or 'arithmetic', "
                     f"got {interface_avg!r}")


def _coefficients(mesh, diffusivity, interface_avg):
    """Cache the dt-independent parts of alpha_l / alpha_r.

    Everything in the operator is linear in dt, so the adaptive driver can rebuild
    its matrix in O(nx) per attempt without re-running np.diff or the harmonic
    mean. The split is chosen so that `num * dt / den` reproduces
    r1_calibration_core._alpha's floating-point association EXACTLY
    (`d_eff * dt / (dxm*(dxm+dxp)/2)`), which is what makes the theta=1 path
    bitwise identical rather than merely close.
    """
    m = _check_mesh(mesh)
    d = np.asarray(diffusivity, dtype=float)
    if d.shape != m.shape:
        raise ValueError(f"diffusivity shape {d.shape} != mesh shape {m.shape}")
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("diffusivity must be finite and > 0 at every node")
    dx = np.diff(m)
    d_eff = face_diffusivity(d, interface_avg)
    dxm, dxp = dx[:-1], dx[1:]
    return (d_eff[:-1], dxm * (dxm + dxp) / 2.0,
            d_eff[1:], dxp * (dxm + dxp) / 2.0)


def _alphas(coef, dt):
    num_l, den_l, num_r, den_r = coef
    return num_l * dt / den_l, num_r * dt / den_r


def _leak_interior(lambda_leak, nx):
    """Interior-node leakage coefficients, or None when there is no sink.

    Returns lambda restricted to nodes 1..nx-2. The end and source rows are
    ALGEBRAIC CONSTRAINTS, not evolution equations, so a sink cannot act on them:
    fibeRIS's PML loop (`for i in range(nx): A[i,i] += dt*sigma[i]`,
    matbuilder.py:80-81) does hit those rows, which turns the no-flux condition
    into u0 = u1/(1 - dt*lambda) and pins the Dirichlet node 0.38% low at
    lambda = 3.8e-3 s^-1, dt = 1 s. That idiom must not be copied.
    """
    if lambda_leak is None:
        return None
    lam = np.asarray(lambda_leak, dtype=float)
    if lam.ndim == 0:
        if lam == 0.0:
            return None
        if lam < 0.0 or not np.isfinite(lam):
            raise ValueError(f"lambda_leak must be finite and >= 0, got {lam!r}")
        return np.full(nx - 2, float(lam))
    if lam.shape != (nx,):
        raise ValueError(f"lambda_leak array must have shape ({nx},), "
                         f"got {lam.shape}")
    if not np.all(np.isfinite(lam)) or np.any(lam < 0):
        raise ValueError("lambda_leak array must be finite and >= 0")
    inner = lam[1:nx - 1]
    return None if not np.any(inner) else inner.astype(float, copy=True)


def amplification_factor(mesh, diffusivity, dt, theta, *,
                         interface_avg='harmonic', lambda_leak=0.0):
    """Amplification of the stiffest mode, g = (1-(1-theta)lam)/(1+theta*lam).

    lam_max is taken from the ACTUAL a_l/a_r arrays, not from 4*D*dt/dx^2: that
    closed form assumes a uniform mesh, and the refined production meshes reach
    dx = 0.003 ft, where the true lam_max is orders of magnitude larger and the
    Crank-Nicolson risk correspondingly worse. g near -1 means an undamped
    oscillatory mode, which is why theta = 0.5 rings at a restart discontinuity.
    """
    nx = len(mesh)
    coef = _coefficients(mesh, diffusivity, interface_avg)
    a_l, a_r = _alphas(coef, float(dt))
    lam_in = _leak_interior(lambda_leak, nx)
    tot = a_l + a_r + (lam_in * float(dt) if lam_in is not None else 0.0)
    lam_max = 2.0 * float(np.max(tot))
    theta = float(theta)
    return {
        'lam_max': lam_max,
        'g': float((1.0 - (1.0 - theta) * lam_max) / (1.0 + theta * lam_max)),
        'theta': theta,
    }


def _resolve_source_level(theta, source_time_level):
    """theta < 1 mandates the new-level datum; theta = 1 defaults to fibeRIS's."""
    theta = float(theta)
    if not (0.0 <= theta <= 1.0):
        raise ValueError(f"theta must lie in [0, 1], got {theta!r}")
    if source_time_level is None:
        return 'n' if theta == 1.0 else 'n+1'
    if source_time_level not in ('n', 'n+1'):
        raise ValueError(f"source_time_level must be 'n' or 'n+1', "
                         f"got {source_time_level!r}")
    if theta < 1.0 and source_time_level == 'n':
        # With 'n', step k imposes u^{k+1}_src = s(t^k), so the explicit part of
        # rows src+/-1 uses a datum a whole step stale: an O(dt) boundary error
        # that destroys CN's O(dt^2) while still producing plausible plots. Raise
        # rather than warn.
        raise ValueError(
            "source_time_level='n' is invalid for theta < 1: the explicit part "
            "would use a datum one full step stale, reducing the scheme to first "
            "order. Use source_time_level='n+1' (the default for theta < 1).")
    return source_time_level


class _Stepper:
    """One theta-step of (I + theta*(L + Lam)) u^{n+1} = (I - (1-theta)(L+Lam)) u^n.

    Kept as an object so the fixed-dt loop builds the banded matrix once while the
    adaptive driver can rebuild it for dt and dt/2 without recomputing np.diff or
    the harmonic mean.
    """

    def __init__(self, mesh, coef, source_idx, theta, lam_in, p0):
        self.nx = len(mesh)
        self.coef = coef
        self.src = [int(i) for i in np.atleast_1d(source_idx)]
        self.theta = float(theta)
        self.lam_in = lam_in
        self.p0 = float(p0)
        for i in self.src:
            if not (0 <= i < self.nx):
                raise ValueError(f"source index {i} outside mesh of length {self.nx}")

    def build(self, dt):
        nx = self.nx
        theta = self.theta
        a_l, a_r = _alphas(self.coef, dt)
        lam_dt = self.lam_in * dt if self.lam_in is not None else None

        ab = np.zeros((3, nx))
        if theta == 1.0 and lam_dt is None:
            # Written with r1_calibration_core's exact association so the default
            # path is bitwise identical, not merely equal to round-off.
            ab[1, 1:nx - 1] = 1.0 + a_l + a_r
            ab[0, 2:nx] = -a_r
            ab[2, 0:nx - 2] = -a_l
        else:
            tot = a_l + a_r if lam_dt is None else a_l + a_r + lam_dt
            ab[1, 1:nx - 1] = 1.0 + theta * tot
            ab[0, 2:nx] = -theta * a_r
            ab[2, 0:nx - 2] = -theta * a_l
        # Neumann rows are identical for every theta: they carry no time
        # derivative, so there is nothing for theta to weight.
        ab[1, 0] = -1.0
        ab[0, 1] = 1.0
        ab[1, nx - 1] = -1.0
        ab[2, nx - 2] = 1.0
        # Source rows LAST, mirroring matbuilder's ordering (BCs at :50-66, source
        # at :71-76), so precedence matches when a source sits on a boundary node.
        for i in self.src:
            ab[1, i] = 1.0
            if i + 1 < nx:
                ab[0, i + 1] = 0.0
            if i - 1 >= 0:
                ab[2, i - 1] = 0.0
        return ab, (a_l, a_r, lam_dt)

    def rhs(self, u, parts, s_old, s_new, dt):
        """Assemble b. The three constraint-row assignments are load-bearing.

        Rows 0, nx-1 and every source row are algebraic constraints; their RHS is a
        datum (0, 0, s), never a theta blend. If the explicit operator is applied
        as a full mat-vec and row 0 is left un-overwritten, the left row becomes
        u1^{n+1} - u0^{n+1} = u^n_0 -- an imposed gradient proportional to the
        boundary value, fed back every step. Measured on the production mesh at
        theta = 0.5 and 0.75 that OVERFLOWS to inf; it is not a small bias.
        """
        nx = self.nx
        theta = self.theta
        if theta == 1.0:
            b = u.copy()
        else:
            a_l, a_r, lam_dt = parts
            u_exp = u.copy()
            for i, s in zip(self.src, s_old):
                u_exp[i] = s  # neighbours legitimately see the old datum
            diag = a_l + a_r if lam_dt is None else a_l + a_r + lam_dt
            Lu = np.zeros(nx)  # rows 0 and nx-1 identically zero by construction
            Lu[1:nx - 1] = (-a_l * u_exp[0:nx - 2] + diag * u_exp[1:nx - 1]
                            - a_r * u_exp[2:nx])
            b = u_exp - (1.0 - theta) * Lu
        if self.p0 != 0.0 and parts[2] is not None:
            # + lambda*P0 forcing, interior only. Over one step that is
            # dt*lambda*P0 = Lam*P0, NOT dt*Lam*P0 -- Lam already carries the dt.
            # It applies at EVERY theta; p0 defaults to 0 so the theta=1 bitwise
            # identity is untouched. P0 = 0 in the delta-pressure formulation, but
            # an absolute-pressure (101-style) run needs it.
            b[1:nx - 1] += parts[2] * self.p0
        b[0] = 0.0
        b[nx - 1] = 0.0
        for i, s in zip(self.src, s_new):
            b[i] = s
        return b

    def step(self, u, ab, parts, s_old, s_new, dt):
        return solve_banded((1, 1), ab, self.rhs(u, parts, s_old, s_new, dt))


def _interp_sources(sources, t_rel):
    return [float(np.interp(t_rel, ta, da)) for ta, da in sources]


def _normalise_sources(source_taxis, source_data, source_idx):
    """Accept either the single-source triple or the multi-source lists."""
    idx = np.atleast_1d(np.asarray(source_idx)).astype(int).tolist()
    if len(idx) == 1 and np.ndim(source_taxis) == 1:
        return [(np.asarray(source_taxis, dtype=float),
                 np.asarray(source_data, dtype=float))], idx
    if len(source_taxis) != len(idx) or len(source_data) != len(idx):
        raise ValueError(
            f"multi-source needs one taxis and one data series per index: "
            f"{len(source_taxis)} taxis, {len(source_data)} data, {len(idx)} indices")
    return [(np.asarray(ta, dtype=float), np.asarray(da, dtype=float))
            for ta, da in zip(source_taxis, source_data)], idx


def solve_forward(mesh, diffusivity, dt, t_total, source_taxis, source_data,
                  source_idx, initial=None, t0=0.0, record_idx=None, *,
                  theta=1.0, source_time_level=None, lambda_leak=0.0, p0=0.0,
                  interface_avg='harmonic', theta_startup_steps=0):
    """theta-weighted diffusion with Dirichlet source node(s) and an optional sink.

    With every keyword at its default this is bitwise identical to
    r1_calibration_core.solve_forward -- same taxis accumulation, same source
    interpolation at time level n, same banded rows, same association in
    `1 + alpha_l + alpha_r`. That identity is asserted in rev2_selftest.T1 and is
    what carries the fibeRIS equivalence proof into rev2.

    Returns (taxis, recorded); recorded is (n_time, len(record_idx)) when
    record_idx is given, else the full (n_time, nx) field, i.e. TIME-MAJOR, the
    same orientation as rev2_layout.Panel.data.
    """
    nx = len(mesh)
    theta = float(theta)
    level = _resolve_source_level(theta, source_time_level)
    coef = _coefficients(mesh, diffusivity, interface_avg)
    lam_in = _leak_interior(lambda_leak, nx)
    sources, idx = _normalise_sources(source_taxis, source_data, source_idx)

    stepper = _Stepper(mesh, coef, idx, theta, lam_in, p0)
    ab, parts = stepper.build(dt)
    n_start = int(theta_startup_steps)
    if n_start > 0:
        # Rannacher start-up: the first few steps run as two backward-Euler
        # half-steps. CN rings at a restart discontinuity (57 increment sign
        # reversals in the first 60 s at the node next to the source, vs 0 for
        # BE); damping the first steps removes it without giving up second order
        # for the rest of the run.
        be = _Stepper(mesh, coef, idx, 1.0, lam_in, p0)
        ab_half, parts_half = be.build(dt / 2.0)

    u = np.zeros(nx) if initial is None else np.asarray(initial, dtype=float).copy()
    taxis = [t0]
    keep = np.arange(nx) if record_idx is None else np.asarray(record_idx)
    out = [u[keep].copy()]

    t = t0
    k = 0
    while t < t_total:
        s_n = _interp_sources(sources, t - t0)
        if k < n_start:
            s_h = _interp_sources(sources, t + dt / 2.0 - t0)
            s_1 = _interp_sources(sources, t + dt - t0)
            if level == 'n':
                u = be.step(u, ab_half, parts_half, s_n, s_n, dt / 2.0)
                u = be.step(u, ab_half, parts_half, s_n, s_h, dt / 2.0)
            else:
                u = be.step(u, ab_half, parts_half, s_n, s_h, dt / 2.0)
                u = be.step(u, ab_half, parts_half, s_h, s_1, dt / 2.0)
        else:
            s_new = s_n if level == 'n' else _interp_sources(sources, t + dt - t0)
            u = stepper.step(u, ab, parts, s_n, s_new, dt)
        t += dt
        k += 1
        taxis.append(t)
        out.append(u[keep].copy())

    return np.asarray(taxis), np.asarray(out)


def solve_forward_multi(mesh, diffusivity, dt, t_total, source_taxis_list,
                        source_data_list, source_idx_list, initial=None, t0=0.0,
                        record_idx=None, **kwargs):
    """Several Dirichlet source nodes, as fibeRIS matrix_builder_1d_multi_source.

    101/106 drive six stage-7 (or stage-8) frac-hit nodes from one gauge frame,
    so the common case is one series replicated across indices; pass it explicitly
    per index rather than broadcasting, because the manifest must record one
    driver per applied node (two frac hits can snap to the same node on a coarse
    mesh, which silently reduces the number of independent sources).
    """
    idx = [int(i) for i in source_idx_list]
    if len(set(idx)) != len(idx):
        dup = sorted({i for i in idx if idx.count(i) > 1})
        raise ValueError(f"duplicate source mesh indices {dup}: two sources on one "
                         f"node would silently overwrite each other")
    return solve_forward(mesh, diffusivity, dt, t_total, source_taxis_list,
                         source_data_list, idx, initial=initial, t0=t0,
                         record_idx=record_idx, **kwargs)


def solve_forward_adaptive(mesh, diffusivity, t_total, source_taxis, source_data,
                           source_idx, initial=None, t0=0.0, record_idx=None, *,
                           theta=1.0, source_time_level=None, lambda_leak=0.0,
                           p0=0.0, interface_avg='harmonic',
                           dt_init=2.0, tol=1e-3, safety_factor=0.9, order_p=2,
                           max_dt=30.0, min_dt=1e-4, controller_tol=1e-3,
                           zero_field_policy='accept', max_attempts=200000,
                           theta_startup_steps=0):
    """Reproduce fibeRIS's optimizer=True stepping in the fast banded kernel.

    This mirrors pds.py:287-336 plus tso.py:18-49 exactly, including three
    behaviours that are defects rather than design and must be reproduced if the
    manuscript's two-phase figures are to be explained:

    * the ACCEPTED state is the single full-dt step, not the two-half-step
      solution (tso.py:26), and no Richardson extrapolation is done;
    * the error estimate is a relative L2 over the WHOLE field and is not divided
      by 2^q-1, so it is sensitive to an additive baseline offset -- which is why
      the manuscript's ~8300 psi absolute protocol never binds the control and
      the run is in effect a fixed dt = 30 s run (measured: 43 attempts, 0
      rejections, dt = max_dt for 97.7% of steps);
    * the loop condition is `while t < t_total`, so the run OVERSHOOTS t_total
      (measured 1262.0 s against 1254.091 s). Comparisons must interpolate onto
      the gauge time axis rather than use the last snapshot.

    `controller_tol` is separate from `tol` on purpose: fibeRIS's
    time_sampling_optimizer binds `tol` to a named parameter and never forwards it
    to adjust_dt (tso.py:4, :21, :40), so the step-size controller always uses
    1e-3 no matter what the caller asks for. Defaulting controller_tol to 1e-3
    keeps that faithful; set controller_tol=tol for the corrected variant. Both
    numbers must appear in the manifest.

    `zero_field_policy='accept'` is required for the delta-pressure formulation:
    with u == 0 and s(0) == 0 fibeRIS evaluates 0/0 = nan, rejects, and pins dt at
    min_dt forever (reproduced: 4001 attempts, 0 accepted). fibeRIS therefore
    CANNOT run that configuration at all, so no fibeRIS reference exists for it --
    say so wherever such a run is reported rather than implying agreement.

    BUT `accept` IS NOT A LICENCE TO RUN THE DELTA FORMULATION ADAPTIVELY, AND ON
    THE PRODUCTION dx = 1 ft MESH IT DOES NOT RUN. `accept` only clears the
    exact-zero FIRST step (t = t0, s(t0) = 0, so u stays 0 at theta = 1 where the
    datum is read at level n). The live-lock strikes one step later, when the datum
    first becomes non-zero: the solution is then a boundary layer resolved by a
    handful of cells, and the RELATIVE whole-field estimate
    ||u_full - u_h2|| / ||u_full|| plateaus instead of falling. Measured on
    D = 1150 ft^2/s, dx = 1 ft, a 0 -> 300 psi datum onto a zero field, err at that
    second attempt is 8.23e-02 at dt = 2 s, 8.19e-02 at 1 s, 7.92e-02 at 0.1 s,
    4.17e-02 at 1e-3 s, 4.24e-03 at 1e-4 s, and first clears tol = 1e-3 only at
    dt = 4.171e-05 s (bisected) -- 2.4x BELOW the default min_dt = 1e-4 -- so
    `dt_next = max(min_dt, ..)` pins dt at min_dt and nothing is ever accepted.
    This driver then raises RuntimeError at max_attempts, which is better than
    fibeRIS (no guard, hangs) but is still not a working run. B1 measured the same
    stall independently on the production R1 mesh for all three of its profiles
    (D1150, D1223, two_zone): live-lock at t = 2 s, dt = 1e-4, requiring
    dt <= 3.45e-05 s, and it is not an artifact of dt_init (0.5 / 1 / 2 / 5 / 10 s
    all live-lock).

    The estimate is INVARIANT to the datum amplitude (the solution is linear in
    it), so a smooth ramp is no safer than a step: 300*(1 - exp(-t/40)) gives the
    same 8.23e-02. What decides the outcome is dx, because the plateau clears near
    D*dt/dx^2 ~ O(1): measured, dx = 5 ft escapes (dt collapses to 9.3e-4 s, above
    min_dt, then recovers; 5 rejections) while dx = 1 ft does not, at L = 100,
    500, 2000 and 6750 ft alike. A dx = 5 ft run that "works" has therefore taken
    dt three orders of magnitude below the production dt = 1 s.

    **Use fixed dt = 1 s for any delta-pressure problem.** On the manuscript's
    ABSOLUTE-pressure protocol (~8300 psi baseline) the estimator behaves normally:
    43 attempts, 0 rejections, dt at max_dt = 30 s for 97.7 % of the run.

    `theta_startup_steps` is Rannacher start-up, with the same semantics as in
    `solve_forward`/`solve_forward_multi`: the first N ACCEPTED steps are taken as
    two backward-Euler half-steps instead of theta-steps, which damps the
    oscillatory mode a restart discontinuity excites at theta < 1 (HOUSE_RULES
    CORRECTION 3 makes it mandatory for any multi-stage rerun with theta < 1). A
    rejected attempt does not consume a start-up step. Inside a start-up step BOTH
    candidates of the error estimate are Rannacher solutions -- the full-dt one is
    two BE dt/2 steps, the comparison one is four BE dt/4 steps -- so the
    controller keeps working, but the run is then deliberately NOT a reproduction
    of fibeRIS, which has no start-up damping. The default 0 leaves the fibeRIS
    reproduction path bit-for-bit unchanged.

    Returns (taxis, recorded, trace) with trace a dict carrying the per-attempt
    records under 'attempts' plus the summary fields A4's manifest requires,
    including `flip_margin` = min|err - tol|/tol. A flip_margin above ~1e-6 means
    no accept/reject decision could have flipped from the ~1e-12 difference
    between the dense LAPACK solve fibeRIS uses and the banded one used here; at
    or below it the run is only the closest faithful variant, not a reproduction.
    """
    nx = len(mesh)
    theta = float(theta)
    level = _resolve_source_level(theta, source_time_level)
    coef = _coefficients(mesh, diffusivity, interface_avg)
    lam_in = _leak_interior(lambda_leak, nx)
    sources, idx = _normalise_sources(source_taxis, source_data, source_idx)
    stepper = _Stepper(mesh, coef, idx, theta, lam_in, p0)
    n_start = int(theta_startup_steps)
    if n_start < 0:
        raise ValueError("theta_startup_steps must be >= 0")
    be = _Stepper(mesh, coef, idx, 1.0, lam_in, p0) if n_start > 0 else None
    if zero_field_policy not in ('accept', 'fiberis', 'raise'):
        raise ValueError("zero_field_policy must be 'accept', 'fiberis' or 'raise'")

    u = np.zeros(nx) if initial is None else np.asarray(initial, dtype=float).copy()
    keep = np.arange(nx) if record_idx is None else np.asarray(record_idx)
    taxis, out, attempts = [t0], [u[keep].copy()], []

    def _run(u0, dt, s_old, s_new):
        ab, parts = stepper.build(dt)
        return stepper.step(u0, ab, parts, s_old, s_new, dt)

    def _run_rannacher(u0, t_a, h):
        """One Rannacher step over [t_a, t_a + h]: two backward-Euler h/2 steps.

        The datum sequence is the one solve_forward uses for its start-up steps,
        so a fixed-dt and an adaptive restart damp identically.
        """
        ab, parts = be.build(h / 2.0)
        s_a = _interp_sources(sources, t_a - t0)
        s_b = _interp_sources(sources, t_a + h / 2.0 - t0)
        if level == 'n':
            u1 = be.step(u0, ab, parts, s_a, s_a, h / 2.0)
            return be.step(u1, ab, parts, s_a, s_b, h / 2.0)
        s_c = _interp_sources(sources, t_a + h - t0)
        u1 = be.step(u0, ab, parts, s_a, s_b, h / 2.0)
        return be.step(u1, ab, parts, s_b, s_c, h / 2.0)

    t, dt = t0, float(dt_init)
    while t < t_total:
        t_start = t
        in_startup = (len(taxis) - 1) < n_start
        s_n = _interp_sources(sources, t - t0)
        if in_startup:
            # Both candidates are Rannacher, so the error estimate still compares
            # like with like; the accepted state stays the single full-dt solve,
            # as in the theta branch below.
            u_full = _run_rannacher(u, t, dt)
            u_h1 = _run_rannacher(u, t, dt / 2.0)
            u_h2 = _run_rannacher(u_h1, t + dt / 2.0, dt / 2.0)
        elif level == 'n':
            # fibeRIS never advances taxis inside the sub-steps (matbuilder.py:73
            # reads taxis[-1], pds.py:318-325 appends nothing), so ALL THREE
            # solves see the same datum s(t^n). Interpolating the half-steps at
            # their own times would be more accurate and would NOT reproduce
            # fibeRIS -- it perturbs the error estimate enough to flip
            # accept/reject decisions (measured: 52 attempts / 4 rejections
            # instead of 43 / 0 on the manuscript protocol).
            u_full = _run(u, dt, s_n, s_n)
            u_h1 = _run(u, dt / 2.0, s_n, s_n)
            u_h2 = _run(u_h1, dt / 2.0, s_n, s_n)
        else:
            s_h = _interp_sources(sources, t + dt / 2.0 - t0)
            s_1 = _interp_sources(sources, t + dt - t0)
            u_full = _run(u, dt, s_n, s_1)
            u_h1 = _run(u, dt / 2.0, s_n, s_h)
            u_h2 = _run(u_h1, dt / 2.0, s_h, s_1)

        nrm = float(np.linalg.norm(u_full))
        if nrm == 0.0:
            if zero_field_policy == 'raise':
                raise ValueError(
                    "adaptive controller: the field is identically zero, so the "
                    "relative error estimate is 0/0. fibeRIS live-locks here; use "
                    "zero_field_policy='accept' and say so in the report.")
            err = float('nan') if zero_field_policy == 'fiberis' else 0.0
        else:
            err = float(np.linalg.norm(u_full - u_h2) / nrm)

        ratio = 1.0 if err < 1e-14 else (controller_tol / err) ** (1.0 / order_p)
        dt_next = max(min_dt, min(dt * safety_factor * ratio, max_dt))
        accepted = bool(err <= tol)
        if accepted:
            t += dt
            u = u_full
            taxis.append(t)
            out.append(u[keep].copy())
        rec = {'t': float(t_start), 'dt': float(dt), 'err': err,
               'accepted': accepted, 'dt_next': float(dt_next)}
        if in_startup:
            rec['rannacher_startup'] = True
        attempts.append(rec)
        dt = dt_next
        if len(attempts) > max_attempts:
            raise RuntimeError(
                f"adaptive stepping exceeded max_attempts={max_attempts} at "
                f"t={t:.6g}, dt={dt:.6g}; fibeRIS has no such guard and would hang")

    errs = np.array([a['err'] for a in attempts], dtype=float)
    finite = errs[np.isfinite(errs)]
    dts = np.diff(np.asarray(taxis, dtype=float))
    trace = {
        'attempts': attempts,
        'n_attempts': int(len(attempts)),
        'n_accepted': int(len(taxis) - 1),
        'n_rejected': int(len(attempts) - (len(taxis) - 1)),
        'dt_init_s': float(dt_init), 'tol': float(tol),
        'controller_tol': float(controller_tol),
        'safety_factor': float(safety_factor), 'order_p': int(order_p),
        'max_dt_s': float(max_dt), 'min_dt_s': float(min_dt),
        'zero_field_policy': zero_field_policy,
        'theta_startup_steps': int(n_start),
        'n_startup_steps_applied': int(min(n_start, len(taxis) - 1)),
        'dt_min_s': float(dts.min()) if dts.size else None,
        'dt_max_s': float(dts.max()) if dts.size else None,
        'dt_mean_s': float(dts.mean()) if dts.size else None,
        'frac_at_max_dt': float(np.mean(np.isclose(dts, max_dt))) if dts.size else None,
        't_end_s': float(taxis[-1]),
        't_total_requested_s': float(t_total),
        'overshoot_s': float(taxis[-1] - t_total),
        'err_min': float(finite.min()) if finite.size else None,
        'err_max': float(finite.max()) if finite.size else None,
        'flip_margin': (float(np.min(np.abs(finite - tol)) / tol)
                        if finite.size else None),
        'error_norm': 'relative_l2_full_vs_two_half_steps',
    }
    return np.asarray(taxis), np.asarray(out), trace
