"""B3 v2 report -- the AUTHORITATIVE product set, and why v1 is superseded.

`b3_grid.py` ran `--stage all --tag v1` from 12:22 to ~16:00 MDT on 2026-09-02.
Three things about that run's REPORT need correcting; none of them touches a fit.

1. **The shared modules were repaired WHILE the run was in flight.** `A4_repair2`
   rewrote `rev2_manifest.py` at 14:56 and `rev2_core.py` at 15:19 MDT. The B3
   worker processes had imported both at 12:22, so every solve used the PRE
   versions

       rev2_core.py     eac79a20297dad8e71b5f1f7c7e23f2aff3a9d27fee5c7153d6a923480cfbe16
       rev2_manifest.py 37bc87715d06fe432ef8875e85ea2c3f24c0f21ebdbc4d476bb9c0fb5f040d93

   while the v1 manifests, written at the end of the run, hash whatever was on
   disk by then -- the POST versions. All 19 of them nevertheless verify `clean`:
   nothing in the record says the code moved, because the hashes were taken after
   it moved. A manifest that names code which did not produce its numbers, and
   audits clean while doing so, is exactly what `write_manifest` exists to prevent,
   so this file re-runs the report with the modules that are on disk NOW and, in
   the same breath, PROVES the swap is numerically null for every code path B3
   uses (stage `modcheck`: 18 geometry realisations + 30 misfit evaluations
   recomputed with the current modules and compared BITWISE against the v1
   checkpoints). The two functional edits in `rev2_core` are `_coefficients`
   calling `_check_mesh` instead of `np.asarray` (a validator; B3's meshes are
   uniform and exact) and a new `n_width_inflated` key in the barrier report
   (an added field, no change to the returned profile), so the expected result is
   bitwise equality -- but it is measured, not assumed.

2. **`derived_scalars` mixed two scales.** It built the source->g7 series
   resistance as `L / D_fitted + excess_from_stage_geom`, and `stage_geom`
   measures the barrier's excess on a REFERENCE profile of D0 = 1150 ft^2/s. The
   barrier's ratio reference here is `at_hit`, i.e. D_barrier = ratio * D(x_hit),
   so the excess scales as 1/D and the two terms sat at different D -- the
   headline arm's excess was understated by 5081/1150 = 4.42x. `derived_scalars_v2`
   integrates the harmonic-face series resistance node by node over the D(x) array
   the fit ACTUALLY used, on the mesh it used, so no scale assumption enters.
   Diagnostic only: no fitted parameter, no misfit and no stability number for
   `D_ft2_s` or `rmse_psi` is affected.

3. **`worker_modules=` was not declared.** `A4_repair2` open issue 8 names `b3`.
   It is NOT a defect in the v1 manifests -- b3_grid.py passes `require_modules`
   and `extra_code_files`, so the repaired verifier scores all 19 of them `clean`
   -- but naming the workers' own imports is what would catch a repo module first
   imported inside a child process, which is the gap `worker_modules` exists for.
   Declared here.

v1's products and manifests are KEPT (house rule 2) and marked SUPERSEDED in the
task README. Nothing in this file re-fits anything: it reads the v1 checkpoints.

Run from the repo root, after `b3_grid.py --stage all` has finished:

    python3 scripts/manuscript_well_leakage/rev2/b3_grid_v2report.py

Owns: the `_v2` products under output/rev2_20260901/B3/ and the `outputs_v2`
block of configs/rev2/b3_grid.json. Imports b3_grid and the shared rev2 modules;
edits neither.
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
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_BC = os.path.join(os.path.dirname(_HERE), 'baseline_calibration')
if _BC not in sys.path:
    sys.path.insert(0, _BC)

import b3_grid as B                 # noqa: E402
import rev2_core as core            # noqa: E402
import rev2_data as rd              # noqa: E402
import rev2_manifest as rm          # noqa: E402
import r1_calibration_core as r1c   # noqa: E402

CONFIG = 'configs/rev2/b3_grid.json'
TAG = 'v1'                          # the checkpoint set this report reads
OUT = 'output/rev2_20260901/B3'
X_SRC, X_G7 = 16645.0, 15075.0
L_PATH = X_SRC - X_G7

PRE_HASHES = {
    'scripts/manuscript_well_leakage/rev2/rev2_core.py':
        'eac79a20297dad8e71b5f1f7c7e23f2aff3a9d27fee5c7153d6a923480cfbe16',
    'scripts/manuscript_well_leakage/rev2/rev2_manifest.py':
        '37bc87715d06fe432ef8875e85ea2c3f24c0f21ebdbc4d476bb9c0fb5f040d93',
    'scripts/manuscript_well_leakage/rev2/rev2_data.py':
        'a4f171fcd58de9c6250213111d075dcb1d43afc0fe0271cabede78b300022f84',
    'scripts/manuscript_well_leakage/rev2/b3_grid.py':
        'c6d62e1cfe5165ee9f59aa7e44e93c807dc9979bb7862da500fac3d25ab59d10',
}
PRE_HASH_SOURCE = ('output/rev2_20260901/A4_repair2/evidence/sha256_PRE.txt, '
                   'cross-checked against the rev2_core hash recorded in '
                   'output/rev2_20260901/B1/manifest.json and '
                   'output/rev2_20260901/D3/manifest.json (both written before '
                   'this run started)')

WORKER_MODULES = ('rev2_core', 'rev2_data', 'r1_calibration_core')

_CFG_DISK = None        # the config EXACTLY as it is on disk, for the manifests
_PROV = None            # code-provenance block, filled in by stage modcheck
_EXTRA_OUTPUTS = []     # v2-only products to append to every outputs= list
_WIDTH_MATCHED = None   # meshes grouped by the barrier they actually realise


def log(msg):
    print('[%s] %s' % (datetime.datetime.now().strftime('%H:%M:%S'), msg),
          flush=True)


# ---------------------------------------------------------------------------
# 2. the scale-consistent path-equivalent diffusivity
# ---------------------------------------------------------------------------

def _series_resistance(mesh, d, x_lo, x_hi):
    """int dx / D(x) between two nodes, s/ft, with HARMONIC face averaging.

    Harmonic faces are what the solver uses, and for conductances in series the
    harmonic face value makes the resistance of a node pair exactly the sum of the
    two half-cell resistances -- so this is the discrete operator's OWN steady
    resistance, not a quadrature of a continuum integral.
    """
    i0 = int(np.argmin(np.abs(mesh - x_lo)))
    i1 = int(np.argmin(np.abs(mesh - x_hi)))
    if i1 < i0:
        i0, i1 = i1, i0
    seg = np.diff(mesh[i0:i1 + 1])
    face = 2.0 / (1.0 / d[i0:i1] + 1.0 / d[i0 + 1:i1 + 1])
    return float(np.sum(seg / face))


_SETUP_CACHE = {}


def _setup(cfg, dx):
    S = _SETUP_CACHE.get(dx)
    if S is None:
        S = rd.setup_r1(dx_ft=float(dx),
                        pad_low_ft=float(cfg['mesh']['domain_pad_low_md_ft']),
                        pad_high_ft=float(cfg['mesh']['domain_pad_high_md_ft']))
        _SETUP_CACHE[dx] = S
    return S


def _profile_on_mesh(cfg, arm, family, params, dx, fh):
    S = _setup(cfg, dx)
    mesh = S['mesh'].x
    base = r1c.PROFILE_FAMILIES[family]['fn'](mesh, S['source_idx'],
                                              np.asarray(params, float))
    if not arm['barrier']:
        return mesh, base, base, None
    b = cfg['barrier']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', core.BarrierWidthWarning)
        prof, rep = core.build_barrier_profile(
            mesh, base, fh, float(arm['w_half_width_ft']), float(arm['ratio']),
            ratio_reference=b['ratio_reference'], combine=b['combine'],
            on_empty=b['on_empty'], on_outside=b['on_outside'],
            return_report=True)
    return mesh, base, prof, rep


def derived_scalars_v2(cfg, uni_rows, tz_rows, geom_rows):
    """Drop-in replacement for b3_grid.derived_scalars, without the scale mix.

    Same signature and same row keys, plus the two resistance components and the
    realised barrier width, so the CSV/JSON consumers downstream are unchanged.
    """
    fh = rd.load_frac_hits(2)
    arms = {a['name']: a for a in cfg['arms']}
    out = []
    for family, rows in (('uniform', uni_rows), ('two_zone', tz_rows)):
        names = (['log10_D'] if family == 'uniform'
                 else cfg['families']['two_zone']['param_names'])
        for r in rows:
            # Both norms are kept for the uniform family: the normalised-norm
            # optimum is a different model and deserves its own resistance.
            p = [r[n] for n in names]
            mesh, base, prof, rep = _profile_on_mesh(
                cfg, arms[r['arm']], family, p, r['dx_ft'], fh)
            R_base = _series_resistance(mesh, base, X_G7, X_SRC)
            R_tot = _series_resistance(mesh, prof, X_G7, X_SRC)
            widths = ([b['realised_full_width_ft'] for b in rep['barriers']]
                      if rep else [])
            out.append(dict(
                family=family, arm=r['arm'],
                norm=(r['norm'] if family == 'uniform' else 'abs'),
                dx_ft=r['dx_ft'],
                # the family's headline scalar: D for uniform, D_far for two_zone
                D_fitted_ft2_s=float(10.0 ** (r['log10_D'] if family == 'uniform'
                                              else r['log10_D_far'])),
                R_total_s_per_ft=R_tot, D_eq_ft2_s=L_PATH / R_tot,
                R_profile_only_s_per_ft=R_base,
                R_barrier_excess_s_per_ft=R_tot - R_base,
                D_eq_profile_only_ft2_s=L_PATH / R_base,
                realised_full_width_total_ft=float(sum(widths)),
                n_barriers_width_inflated=(int(rep['n_width_inflated'])
                                           if rep else 0),
                n_barriers_fallback=int(rep['n_fallback']) if rep else 0,
                rmse_psi=r['rmse_psi'],
                definition='L / int_{g7}^{src} dx/D(x), D(x) = the array the fit '
                           'used on ITS mesh (family profile + frozen barrier), '
                           'harmonic faces'))
    return out


# ---------------------------------------------------------------------------
# 1. the PRE/POST equivalence proof
# ---------------------------------------------------------------------------

def _mc_worker_init(cfg):
    B._init_worker(cfg)


def _mc_task(t):
    return B._misfit(t)


def stage_modcheck(cfg, nproc):
    """Recompute v1 values with the modules on disk NOW and compare BITWISE."""
    ck = B.load_ckpt('modcheck_v2.json')
    if ck is not None:
        return ck
    t0 = time.time()
    fh = rd.load_frac_hits(2)
    arms = {a['name']: a for a in cfg['arms']}

    # (a) geometry: realised widths, excess resistance, fallback counts
    geom_v1 = B.load_ckpt('geom_%s.json' % TAG)
    geom_now = B.stage_geom(cfg, 'modcheck_tmp')
    os.remove(B.ckpt_path('geom_modcheck_tmp.json'))
    geom_diffs = []
    for a, b in zip(geom_v1, geom_now):
        for k in ('nx', 'n_fallback', 'realised_full_width_min_ft',
                  'realised_full_width_med_ft', 'realised_full_width_max_ft',
                  'excess_resistance_s_per_ft', 'max_centre_offset_ft',
                  'snap_source_ft', 'snap_target_max_ft'):
            if a[k] != b[k]:
                geom_diffs.append({'arm': a['arm'], 'dx_ft': a['dx_ft'],
                                   'key': k, 'v1': a[k], 'now': b[k]})

    # (b) misfits at the v1 optima, one per (family, arm, mesh)
    uni = B.load_ckpt('uniform_restarts_%s.json' % TAG)
    tz = B.load_ckpt('two_zone_restarts_%s.json' % TAG)
    best = {}
    for r in uni:
        if r['norm'] != 'abs':
            continue
        k = ('uniform', r['arm'], r['dx_ft'])
        if k not in best or r['objective'] < best[k]['objective']:
            best[k] = r
    for r in tz:
        k = ('two_zone', r['arm'], r['dx_ft'])
        if k not in best or r['rmse_psi'] < best[k]['rmse_psi']:
            best[k] = r
    keys = sorted(best, key=lambda k: -k[2])
    tasks = [(k[1], k[2], k[0],
              ([best[k]['log10_D']] if k[0] == 'uniform' else best[k]['params']))
             for k in keys]
    with mp.Pool(nproc, initializer=_mc_worker_init, initargs=(cfg,)) as pool:
        vals = list(pool.imap(_mc_task, tasks, chunksize=1))
    misfit_rows = []
    for k, v in zip(keys, vals):
        r = best[k]
        misfit_rows.append(dict(
            family=k[0], arm=k[1], dx_ft=k[2],
            rmse_v1=float(r['rmse_psi']), rmse_now=float(v[0]),
            rmse_bitwise_equal=bool(float(v[0]) == float(r['rmse_psi'])),
            rmse_abs_diff=abs(float(v[0]) - float(r['rmse_psi'])),
            nrmse_v1=float(r['rmse_normalised']), nrmse_now=float(v[1]),
            nrmse_bitwise_equal=bool(float(v[1]) == float(r['rmse_normalised'])),
            nrmse_abs_diff=abs(float(v[1]) - float(r['rmse_normalised']))))

    n_bad = sum(1 for m in misfit_rows
                if not (m['rmse_bitwise_equal'] and m['nrmse_bitwise_equal']))
    res = {
        'n_geometry_rows_compared': len(geom_v1),
        'n_geometry_disagreements': len(geom_diffs),
        'geometry_disagreements': geom_diffs,
        'n_misfits_recomputed': len(misfit_rows),
        'n_misfits_not_bitwise_equal': n_bad,
        'max_rmse_abs_diff_psi': max(m['rmse_abs_diff'] for m in misfit_rows),
        'max_nrmse_abs_diff': max(m['nrmse_abs_diff'] for m in misfit_rows),
        'misfits': misfit_rows,
        'verdict': ('the 2026-09-02 A4_repair2 module repair is NUMERICALLY NULL '
                    'for every code path B3 uses: bitwise identical'
                    if (n_bad == 0 and not geom_diffs) else
                    'THE REPAIR CHANGED B3 NUMBERS -- the v1 fits must be re-run'),
        'wall_s': float(time.time() - t0),
    }
    B.save_ckpt('modcheck_v2.json', res)
    log('modcheck: %d geometry rows, %d misfits, %d disagreements, %.0f s'
        % (len(geom_v1), len(misfit_rows), len(geom_diffs) + n_bad,
           res['wall_s']))
    return res


# ---------------------------------------------------------------------------
# the v2-only figure: forward sensitivity vs calibrated sensitivity
# ---------------------------------------------------------------------------

def width_matched_groups(deriv, family='uniform', norm='abs'):
    """Meshes that realise the SAME barrier, so a comparison across them is a
    statement about the DISCRETISATION alone.

    The barrier is captured on a closed interval, so a mesh can realise a total
    width other than the requested 2w*n_hits (`n_fallback` when nothing is inside
    the window, `n_width_inflated` when the requested edges land on nodes). Two
    meshes with different realised widths carry different MODELS, and comparing
    their optima measures that difference, not convergence. This groups by the
    realised total width rounded to 1e-9 ft.
    """
    out = {}
    for r in deriv:
        if r['family'] != family or r['norm'] != norm:
            continue
        key = (r['arm'], round(r['realised_full_width_total_ft'], 9))
        out.setdefault(key, []).append(r)
    groups = []
    for (arm, w), rs in sorted(out.items()):
        rs = sorted(rs, key=lambda r: -r['dx_ft'])
        D = [L_PATH / r['R_total_s_per_ft'] for r in rs]
        Dfit = [r['D_fitted_ft2_s'] for r in rs]
        groups.append(dict(
            arm=arm, realised_full_width_total_ft=w,
            dx_ft=[r['dx_ft'] for r in rs], n_meshes=len(rs),
            D_fitted_ft2_s=Dfit,
            D_fitted_spread_pct=(100.0 * (max(Dfit) / min(Dfit) - 1.0)
                                 if len(Dfit) > 1 else 0.0),
            D_eq_ft2_s=D,
            D_eq_spread_pct=(100.0 * (max(D) / min(D) - 1.0)
                             if len(D) > 1 else 0.0)))
    return groups


def fig_forward_vs_calibrated(cfg, prof, uni_rows, deriv, path, dpi):
    """A1 asks whether the FORWARD solution moves; B3 asks whether the OPTIMUM
    moves. Panel 1 puts them on one axis; panels 2-3 say what drives the gap."""
    grid = np.asarray(prof[list(prof)[0]]['log10_D'], float)
    dxs = list(cfg['families']['uniform']['dx_ft_sweep'])
    dx_ref = min(dxs)
    tol = float(cfg['criteria']['stability_criterion']['tolerance_percent'])
    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.7))
    table = []
    for arm in B.ARM_COLOR:
        sub = sorted([r for r in uni_rows if r['arm'] == arm
                      and r['norm'] == 'abs'], key=lambda r: r['dx_ft'])
        if not sub:
            continue
        # The forward sensitivity is read at THIS arm's own operating point -- the
        # optimum on the finest mesh -- so it is not a statement about some other
        # part of the misfit surface.
        i = int(np.argmin(np.abs(grid - sub[0]['log10_D'])))
        f = {dx: prof['%s|%g' % (arm, dx)]['rmse_psi'][i] for dx in dxs}
        D = {r['dx_ft']: r['D_ft2_s'] for r in sub}
        fp = {dx: 100.0 * (f[dx] / f[dx_ref] - 1.0) for dx in dxs}
        Dp = {dx: 100.0 * (D[dx] / D[dx_ref] - 1.0) for dx in dxs}
        # dx_ref itself is identically zero on both curves; a log axis cannot
        # show it, and plotting it would draw a spurious cliff.
        xs = [dx for dx in dxs if dx != dx_ref]
        ax[0].plot(xs, [abs(fp[dx]) for dx in xs], 'o-', ms=5,
                   color=B.ARM_COLOR[arm], label='forward RMSE, ' + arm)
        ax[0].plot(xs, [abs(Dp[dx]) for dx in xs], 's--', ms=5, mfc='none',
                   color=B.ARM_COLOR[arm], label='calibrated $D$, ' + arm)
        amp = {dx: (abs(Dp[dx]) / abs(fp[dx]) if abs(fp[dx]) > 1e-6 else None)
               for dx in dxs}
        if arm != 'control':        # the control's ratio is 0/0
            ax[1].plot(xs, [amp[dx] for dx in xs], 'o-', ms=5,
                       color=B.ARM_COLOR[arm], label=arm)
        w = {r['dx_ft']: r['realised_full_width_total_ft'] for r in deriv
             if r['family'] == 'uniform' and r['norm'] == 'abs'
             and r['arm'] == arm}
        table.append(dict(
            arm=arm, D_operating_point_ft2_s=float(10.0 ** grid[i]),
            dx_ft=dxs, dx_reference_ft=dx_ref,
            forward_rmse_psi=[f[dx] for dx in dxs],
            forward_pct=[fp[dx] for dx in dxs],
            calibrated_D_ft2_s=[D[dx] for dx in dxs],
            calibrated_pct=[Dp[dx] for dx in dxs],
            amplification=[amp[dx] for dx in dxs],
            realised_full_width_total_ft=[w.get(dx) for dx in dxs],
            forward_max_abs_pct=float(max(abs(v) for v in fp.values())),
            calibrated_max_abs_pct=float(max(abs(v) for v in Dp.values()))))

    ax[0].axhline(tol, color='k', lw=0.7, ls=':',
                  label=r'$\pm$%g%% (what the grid supports)' % tol)
    ax[0].set_xscale('log'), ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$\Delta x$  (ft)')
    ax[0].set_ylabel(r'|change vs $\Delta x$ = %g ft|  (%%)' % dx_ref)
    ax[0].set_title('the optimum moves MORE than the misfit that locates it',
                    fontsize=10)
    ax[0].grid(alpha=0.3, which='both'), ax[0].legend(fontsize=6.2, loc='best')

    ax[1].axhline(1.0, color='k', lw=0.6)
    ax[1].set_xscale('log')
    ax[1].set_xlabel(r'$\Delta x$  (ft)')
    ax[1].set_ylabel(r'|$\Delta D^*$| / |$\Delta$RMSE|   (% per %)')
    ax[1].set_title('amplification by the re-fit\n(control omitted: its ratio is '
                    '0/0)', fontsize=10)
    ax[1].grid(alpha=0.3), ax[1].legend(fontsize=7)

    # Panel 3: the calibrated D moves because the mesh realises a DIFFERENT
    # barrier, and it moves in proportion. x and y are both excesses over the
    # arm's exactly-realised reference (n_fallback = 0 and n_width_inflated = 0),
    # so a point at the origin cannot be drawn on log axes and is reported in the
    # JSON instead.
    wm_pts = []
    for arm in B.ARM_COLOR:
        if arm == 'control':
            continue
        rs = [r for r in deriv if r['family'] == 'uniform' and r['norm'] == 'abs'
              and r['arm'] == arm]
        if not rs:
            continue
        # round before comparing: 0.5*6 and 0.6+0.4+0.4+0.4+0.6+0.6 are the same
        # width but not the same float
        ref_w = round(min(r['realised_full_width_total_ft'] for r in rs), 9)
        ref = [r for r in rs
               if round(r['realised_full_width_total_ft'], 9) == ref_w]
        ref_D = float(np.mean([r['D_fitted_ft2_s'] for r in ref]))
        xs, ys, lab = [], [], []
        for r in sorted(rs, key=lambda r: r['realised_full_width_total_ft']):
            wx = 100.0 * (round(r['realised_full_width_total_ft'], 9)
                          / ref_w - 1.0)
            dy = 100.0 * (r['D_fitted_ft2_s'] / ref_D - 1.0)
            wm_pts.append(dict(arm=arm, dx_ft=r['dx_ft'],
                               width_excess_pct=wx, D_excess_pct=dy,
                               reference_total_width_ft=ref_w,
                               reference_D_ft2_s=ref_D))
            if wx > 1e-9:
                xs.append(wx), ys.append(abs(dy)), lab.append(r['dx_ft'])
        ax[2].plot(xs, ys, 'o', ms=7, color=B.ARM_COLOR[arm], label=arm)
        for x, y, d in zip(xs, ys, lab):
            ax[2].annotate(r'$\Delta x$ = %g' % d, (x, y),
                           textcoords='offset points', xytext=(6, -8),
                           fontsize=6.5, color=B.ARM_COLOR[arm])
    lo = 0.3
    ax[2].plot([lo, 200], [lo, 200], 'k:', lw=0.8, label='1 : 1')
    ax[2].set_xscale('log'), ax[2].set_yscale('log')
    ax[2].set_xlabel('excess of the REALISED total barrier width  (%)')
    ax[2].set_ylabel(r'|excess of the calibrated $D$|  (%)')
    ax[2].set_title('every departure is the mesh realising a different\nbarrier, '
                    'and it is proportional', fontsize=10)
    ax[2].grid(alpha=0.3, which='both'), ax[2].legend(fontsize=7, loc='best')

    fig.suptitle('B3 — a mesh-independent FORWARD solution does not imply a '
                 'mesh-independent CALIBRATED answer', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return table, wm_pts


# ---------------------------------------------------------------------------
# manifest wrapper: correct code provenance, no edit to any shared module
# ---------------------------------------------------------------------------

def install_manifest_wrapper():
    orig = rm.write_manifest

    def patched(manifest_path, **kw):
        kw['config'] = _CFG_DISK
        kw['worker_modules'] = WORKER_MODULES
        kw['extra_code_files'] = tuple(kw.get('extra_code_files') or ()) + (
            os.path.abspath(__file__),)
        kw['outputs'] = list(kw.get('outputs') or []) + list(_EXTRA_OUTPUTS)
        kw['results'] = dict(kw.get('results') or {}, code_provenance=_PROV,
                             width_matched_groups=_WIDTH_MATCHED)
        kw['notes'] = list(kw.get('notes') or []) + [
            'V2 REPORT, AUTHORITATIVE. It supersedes the v1 products and the v1 '
            'manifests in this directory, which are kept unaltered (house rule 2) '
            'and which record the POST-repair module hashes for a run whose solves '
            'used the PRE-repair modules.',
            'The fits themselves were computed by b3_grid.py under the PRE-repair '
            'rev2_core; results.code_provenance carries both hash sets and the '
            'bitwise re-computation that shows the repair changed no B3 number.',
            "Output paths come from the config's `outputs_v2` block; the `outputs` "
            'block still names the superseded v1 products, and the config stored '
            'here is the file on disk verbatim.',
            'derived_path_equivalent is computed by '
            'b3_grid_v2report.derived_scalars_v2, NOT by b3_grid.derived_scalars: '
            'the latter added a barrier excess resistance measured at D0 = 1150 '
            'to a profile resistance at the fitted D, and with ratio_reference = '
            'at_hit those two scales differ by D_fit/1150.']
        return orig(manifest_path, **kw)

    rm.write_manifest = patched
    return orig


def main():
    global _CFG_DISK, _PROV, _EXTRA_OUTPUTS
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=CONFIG)
    ap.add_argument('--stage', default='all',
                    choices=('all', 'modcheck', 'report'))
    args = ap.parse_args()

    t_start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    T0 = time.time()
    with open(args.config) as fhandle:
        _CFG_DISK = json.load(fhandle)
    cfg = json.loads(json.dumps(_CFG_DISK))
    nproc = int(cfg['run']['processes'])

    prov_now = {p: rm.sha256_file(p, use_cache=False) for p in PRE_HASHES}
    changed = sorted(p for p in PRE_HASHES if prov_now[p] != PRE_HASHES[p])

    mc = stage_modcheck(cfg, nproc)
    if args.stage == 'modcheck':
        log('modcheck only: %s' % mc['verdict'])
        return

    _PROV = {
        'modules_at_solve_time_sha256': PRE_HASHES,
        'modules_at_report_time_sha256': prov_now,
        'modules_changed_between': changed,
        'pre_hash_source': PRE_HASH_SOURCE,
        'why': 'A4_repair2 rewrote rev2_manifest.py (14:56 MDT) and rev2_core.py '
               '(15:19 MDT) on 2026-09-02 while this study was running; the '
               'workers had imported both at 12:22 MDT.',
        'functional_diff_rev2_core': [
            '_coefficients: m = np.asarray(mesh, float) -> m = _check_mesh(mesh) '
            '(a validator; B3 meshes are exactly uniform so it returns the same '
            'array)',
            "build_barrier_profile report: new key 'n_width_inflated'; the "
            'returned diffusivity array is unchanged'],
        'equivalence_check': {k: v for k, v in mc.items() if k != 'misfits'},
    }
    if mc['n_misfits_not_bitwise_equal'] or mc['n_geometry_disagreements']:
        raise SystemExit('modcheck FAILED: %s' % mc['verdict'])

    # --- v2 output paths ---------------------------------------------------
    o2 = cfg.get('outputs_v2')
    if o2 is None:
        raise SystemExit("config has no 'outputs_v2' block; add it before "
                         "running the v2 report")
    cfg['outputs'] = o2

    prof = B.load_ckpt('uniform_grid_%s.json' % TAG)
    uni_restarts = B.load_ckpt('uniform_restarts_%s.json' % TAG)
    tz_restarts = B.load_ckpt('two_zone_restarts_%s.json' % TAG)
    if prof is None or uni_restarts is None or tz_restarts is None:
        raise SystemExit('v1 fit checkpoints missing; run b3_grid.py first')
    if B.load_ckpt('crossmesh_%s.json' % TAG) is None:
        # v1 did not get this far (or was interrupted before it): the cross-mesh
        # forward evaluation is cheap and is a declared product, so run it here
        # rather than emit a manifest that declares a file which does not exist.
        with mp.Pool(nproc, initializer=B._init_worker, initargs=(cfg,)) as pool:
            B.stage_crossmesh(cfg, TAG, pool, uni_restarts, tz_restarts)
    uni_rows = B.summarise_uniform(cfg, prof, uni_restarts)
    tz_rows = B.summarise_two_zone(cfg, tz_restarts)
    geom_rows = B.load_ckpt('geom_%s.json' % TAG)
    deriv = derived_scalars_v2(cfg, uni_rows, tz_rows, geom_rows)
    wm = width_matched_groups(deriv)
    global _WIDTH_MATCHED
    _WIDTH_MATCHED = wm
    fig_fvc = o2['fig_forward_vs_calibrated']
    rm.assert_absent([fig_fvc, o2['forward_vs_calibrated_json']])
    fvc, wm_pts = fig_forward_vs_calibrated(cfg, prof, uni_rows, deriv, fig_fvc,
                                            int(o2['figure_dpi']))
    with open(o2['forward_vs_calibrated_json'], 'w') as fhandle:
        json.dump({'note': 'Forward misfit sensitivity to the mesh at each arm\'s '
                           'own operating point, against the sensitivity of the '
                           'optimum that misfit surface implies. Percentages are '
                           'against the finest mesh in the sweep.',
                   'rows': fvc,
                   'width_matched_note':
                       'A mesh that realises a different barrier width carries a '
                       'different MODEL, so comparing its optimum with another '
                       "mesh's measures that difference and not convergence. "
                       'These groups hold the realised total width fixed, so the '
                       'spread inside a group is the discretisation alone.',
                   'width_matched_groups': wm,
                   'width_excess_vs_D_excess': wm_pts}, fhandle, indent=1)
    _EXTRA_OUTPUTS = [
        rm.output_decl(fig_fvc, role='figure_png', dpi=int(o2['figure_dpi'])),
        rm.output_decl(o2['forward_vs_calibrated_json'], role='json')]

    B.derived_scalars = derived_scalars_v2
    install_manifest_wrapper()
    summary = B.main_report(cfg, args.config, TAG, t_start)
    log('v2 report done in %.0f s' % (time.time() - T0))
    return summary


if __name__ == '__main__':
    main()
