#!/usr/bin/env python3
"""B1 -- time-step convergence AT THE WORKING POINT.

The R1 report priced the time step at D = 500 ft^2/s: pooled RMSE
114.261 / 113.832 / 113.616 / 113.509 / 113.455 psi at dt = 4 / 2 / 1 / 0.5 / 0.25 s,
so "0.161 psi (0.142%)" at the adopted dt = 1 s. That number is not the working
point. The calibrated baseline is D = 1150 (absolute-norm uniform optimum) to
D = 1223 (the triangular family's D_max), where r = D*dt/dx^2 is 2.3-2.4x larger,
and the model that actually wins the comparison is two_zone D(x) with
D_near = 3851 ft^2/s, where r is 7.7x larger still. All three are measured here.

Two things distinguish this from the R1 check:

* R1 quoted the change in the MODEL-DATA misfit, which is a difference of two
  large numbers and hides the solution error almost completely (a 1 psi solution
  error that is orthogonal to the residual moves the RMSE by ~0.004 psi). Here the
  primary quantity is the SOLUTION error against a fine-dt reference, per gauge and
  pooled, in psi and as a fraction of that gauge's peak. The misfit change is
  reported too, so the two conventions can be compared directly.
* The manuscript's two-stage figures used the adaptive stepper, so it is tested on
  this same configuration under both drive protocols. A3 established that on the
  absolute-pressure protocol the estimator never binds and dt pins to
  max_dt = 30 s for 97.7% of the run with zero rejections. That is confirmed or
  refuted here, with the realised dt trace reported either way.

Nothing is fitted. Every number produced is a discretisation error of a fixed
physical model.

AMENDMENT v3 (reviewer defect, reproduced and fixed)
----------------------------------------------------
v2 explained the delta-protocol live-lock with "shrinking dt does not shrink the
estimate, because the Dirichlet assignment at the source node is independent of
dt", and labelled that "measured rather than inferred". It was inferred, and the
v2 pass's OWN stall scan refuted it: `b1_adaptive_trace_v2.json` recorded
`any_dt_accepted: true`, a tol crossing at dt = 3.449e-05 s and a floor
shortfall of 2.90x, and the README then quoted that crossing two sentences after
denying it could exist. The claim is withdrawn. Three measurements replace it:

* 5b2 quantifies HOW the estimate depends on dt (log-log slopes over stated
  bands) instead of asserting that it does not;
* 5c2 runs the counterfactual - lower `min_dt`, change nothing else - so the
  "floor artifact" reading is tested rather than argued;
* 5c3 separates the two live-locks: the t = 2 s one is the floor and is curable,
  the t = 0 one (stock fibeRIS's 0/0 = nan) is not curable by any floor.

The physics, the fixed-dt sweep and the absolute-protocol result are unchanged.

AMENDMENT v5 (three reviewer defects, all reproduced, all real)
---------------------------------------------------------------
No physical number of the v3/v4 study changes. Three CLAIMS are weakened to what
the data support, and one new measurement is added.

1. "Observed order = 1 (0.986-1.009 ... both the pooled RMS and the worst-gauge
   maximum)" was false for the second metric. 0.986-1.009 is exactly the
   POOLED-RMS range; the worst-gauge maximum runs 0.950-1.011, and six of its
   eighteen values sit outside the quoted interval (the low end is two_zone
   8->16 s at 0.9502). The README now prints both ranges, computed from the same
   table it tabulates, and names the pair that produces the low end.
2. "at most 0.34 % of any gauge's peak" was true only for the two uniform cases.
   two_zone -- the model the round declares the winner -- costs 0.513 % of g2's
   simulated peak and 0.516 % of its observed peak, 1.5x larger. Every quotable
   is now scoped, and the all-case bound (0.52 %) is given alongside.
3. The absolute-protocol headline (43 attempts, 0 rejections, 97.67 % at max_dt,
   estimator <= 3.3e-4, flip_margin 0.755) was attributed to the ~8300 psi datum
   alone. It also depends on fibeRIS freezing the level-n datum inside the error
   estimator's sub-steps (rev2_core.py:930-936). Section 5e measures the
   counterfactual: correcting ONLY that -- same tol, same bounds, same protocol,
   same accepted-state formula -- multiplies the peak estimate by ~5 and pushes
   it above tol. The verdict ("a fixed 30 s run in all but name") survives; the
   mechanism sentence does not, and the README now says so.

AMENDMENT v6 (one reviewer defect, reproduced, real; it is a HAND-OFF defect)
----------------------------------------------------------------------------
No physical number of the v5 study changes and nothing is re-fitted. The defect
is that the corrections of v5 never reached the file a reader opens first:
`README.md` was written by the v1 pass and then frozen, because this script
mapped only v1 to the canonical name and every later pass to `README_v<n>.md`.
`README.md` therefore still stated, under the canonical name, all three claims
that v3 and v5 withdrew -- "at most 0.34 % of any gauge's peak" unscoped, the
withdrawn "shrinking dt does not shrink the estimate ... measured rather than
inferred" mechanism, and "the same settings do not produce a solution at all"
without the min_dt finding. A referee handed that summary reproduced two false
statements from it. Three changes:

1. the canonical `README.md` is now written by EVERY pass, alongside
   `README_v<n>.md`, and may only be replaced when its exact bytes already
   exist under another name in the same directory (house rule 2 is enforced,
   not assumed). The v1 text is preserved verbatim as `README_v1_superseded.md`,
   which is where `manifest.json`'s recorded sha256 for `README.md` now
   verifies.
2. bounds are printed so that they BOUND. v5 printed the all-case fraction-of-
   peak bound as "0.51 %" from a measured 0.5133 % (round-to-nearest turns a
   bound into a false statement) and the pooled order range as "0.986-1.009"
   from a measured 0.9864-1.0093. Upper bounds are now rounded up, lower bounds
   down, and the order ranges are printed to four decimals everywhere.
3. the observed order is also reported with the reference's OWN finite step
   removed (E_meas = C(dt - dt_ref) => E_corr = E_meas * dt/(dt - dt_ref)),
   for all three cases and both metrics, because the excess above 1 at the fine
   pairs is that finite reference and not a second-order component. The
   corrected ranges were previously stated in prose in README_v4_ADDENDUM.md
   for D1150 only; they are now computed, tabulated and written to the results
   JSON for every case.

Usage (CWD must be the repo root):
    python3 scripts/manuscript_well_leakage/rev2/b1_dt.py \
        --config configs/rev2/b1_dt.json --version v6
"""

import argparse
import datetime
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
_BASE = os.path.join(_ROOT, 'scripts', 'manuscript_well_leakage',
                     'baseline_calibration')
for _p in (_HERE, _BASE, os.path.join(_ROOT, 'fibeRIS', 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rev2_core as rc                                             # noqa: E402
import rev2_data as rd                                             # noqa: E402
import rev2_manifest as rm                                         # noqa: E402
import r1_calibration_core as r1c                                  # noqa: E402


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

class Tee:
    def __init__(self, path):
        self.fh = open(path, 'w')

    def __call__(self, *args):
        line = ' '.join(str(a) for a in args)
        print(line)
        self.fh.write(line + '\n')
        self.fh.flush()

    def close(self):
        self.fh.close()


def _rms(a):
    a = np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float('nan')


def _adaptive_substep(mesh, diffusivity, t_total, source_taxis, source_data,
                      source_idx, *, substep_datum, initial=None,
                      record_idx=None, theta=1.0, interface_avg='harmonic',
                      dt_init=2.0, tol=1e-3, controller_tol=1e-3,
                      safety_factor=0.9, order_p=2, max_dt=30.0, min_dt=1e-4,
                      max_attempts=4000, t0=0.0):
    """`rev2_core.solve_forward_adaptive` with ONE switch: the datum the error
    estimator's second half-step is given.

    Restricted on purpose to theta = 1, source_time_level 'n', no leakage sink,
    no Rannacher start-up -- exactly the configuration B1 runs -- so that the
    only thing that can differ from the shared module is the switch.

    substep_datum
        'frozen'    all three solves see s(t^n). This is fibeRIS
                    (matbuilder.py:73 reads taxis[-1]; pds.py:318-325 appends
                    nothing inside the sub-steps) and is what rev2_core
                    deliberately reproduces, so it is what the B1 headline
                    reports. Asserted below to be BITWISE identical to
                    rev2_core.solve_forward_adaptive.
        'own_time'  each sub-step reads the datum at its own left endpoint,
                    i.e. the level-n rule applied consistently. The ACCEPTED
                    state is unchanged in form (still the single full-dt solve
                    driven by s(t^n)); only the comparison solve, and hence the
                    error estimate and the step sequence, differ.

    Returns (taxis, recorded, trace) with the same trace keys b1 needs.
    """
    if substep_datum not in ('frozen', 'own_time'):
        raise ValueError("substep_datum must be 'frozen' or 'own_time'")
    if float(theta) != 1.0:
        raise ValueError('_adaptive_substep is the theta = 1 path only')
    nx = len(mesh)
    coef = rc._coefficients(mesh, diffusivity, interface_avg)
    sources, idx = rc._normalise_sources(source_taxis, source_data, source_idx)
    stepper = rc._Stepper(mesh, coef, idx, 1.0, None, 0.0)
    u = np.zeros(nx) if initial is None else np.asarray(initial, float).copy()
    keep = np.arange(nx) if record_idx is None else np.asarray(record_idx)
    taxis, out, attempts = [t0], [u[keep].copy()], []

    def _run(u0, dt, s):
        ab, parts = stepper.build(dt)
        return stepper.step(u0, ab, parts, s, s, dt)

    t, dt = t0, float(dt_init)
    while t < t_total:
        t_start = t
        s_n = rc._interp_sources(sources, t - t0)
        u_full = _run(u, dt, s_n)
        u_h1 = _run(u, dt / 2.0, s_n)
        s_h = (s_n if substep_datum == 'frozen'
               else rc._interp_sources(sources, t + dt / 2.0 - t0))
        u_h2 = _run(u_h1, dt / 2.0, s_h)
        nrm = float(np.linalg.norm(u_full))
        err = 0.0 if nrm == 0.0 else float(np.linalg.norm(u_full - u_h2) / nrm)
        ratio = 1.0 if err < 1e-14 else (controller_tol / err) ** (1.0 / order_p)
        dt_next = max(min_dt, min(dt * safety_factor * ratio, max_dt))
        accepted = bool(err <= tol)
        if accepted:
            t += dt
            u = u_full
            taxis.append(t)
            out.append(u[keep].copy())
        attempts.append({'t': float(t_start), 'dt': float(dt), 'err': err,
                         'accepted': accepted, 'dt_next': float(dt_next)})
        dt = dt_next
        if len(attempts) > max_attempts:
            raise RuntimeError(
                f'adaptive stepping exceeded max_attempts={max_attempts} at '
                f't={t:.6g}, dt={dt:.6g}')
    errs = np.array([a['err'] for a in attempts], dtype=float)
    dts = np.diff(np.asarray(taxis, dtype=float))
    trace = {
        'substep_datum': substep_datum,
        'n_attempts': int(len(attempts)), 'n_accepted': int(len(taxis) - 1),
        'n_rejected': int(len(attempts) - (len(taxis) - 1)),
        'dt_init_s': float(dt_init), 'tol': float(tol),
        'controller_tol': float(controller_tol),
        'safety_factor': float(safety_factor), 'order_p': int(order_p),
        'max_dt_s': float(max_dt), 'min_dt_s': float(min_dt),
        'zero_field_policy': 'accept',
        'dt_min_s': float(dts.min()), 'dt_max_s': float(dts.max()),
        'dt_mean_s': float(dts.mean()), 'dt_median_s': float(np.median(dts)),
        'frac_at_max_dt': float(np.mean(np.isclose(dts, max_dt))),
        't_end_s': float(taxis[-1]), 't_total_requested_s': float(t_total),
        'overshoot_s': float(taxis[-1] - t_total),
        'err_min': float(errs.min()), 'err_max': float(errs.max()),
        'flip_margin': float(np.min(np.abs(errs - tol)) / tol),
        'error_norm': 'relative_l2_full_vs_two_half_steps',
        'attempts': attempts,
    }
    return np.asarray(taxis), np.asarray(out), trace


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

_W = {}


def _init_worker(x, profiles, src_taxis, src_data, src_idx, rec_idx,
                 obs_taxes, grid, t_total, theta, interface_avg):
    _W.update(x=x, profiles=profiles, src_taxis=src_taxis, src_data=src_data,
              src_idx=int(src_idx), rec_idx=list(rec_idx),
              obs_taxes=[np.asarray(t, dtype=float) for t in obs_taxes],
              grid=np.asarray(grid, dtype=float), t_total=float(t_total),
              theta=float(theta), interface_avg=interface_avg)


def _solve_one(job):
    """One fixed-dt solve; returns only the small projections, never the field."""
    case_key, dt = job
    t0 = time.time()
    taxis, rec = rc.solve_forward(
        _W['x'], _W['profiles'][case_key], float(dt), _W['t_total'],
        _W['src_taxis'], _W['src_data'], _W['src_idx'],
        record_idx=_W['rec_idx'], theta=_W['theta'],
        interface_avg=_W['interface_avg'])
    n_steps = int(taxis.size - 1)
    # The accumulated axis is compared with the closed form so the parent can
    # rebuild it for the manifest hash without shipping 642 049 floats back.
    rebuilt = float(dt) * np.arange(n_steps + 1, dtype=float)
    exact = bool(np.array_equal(taxis, rebuilt))
    on_obs = [np.interp(tq, taxis, rec[:, k])
              for k, tq in enumerate(_W['obs_taxes'])]
    on_grid = np.column_stack([np.interp(_W['grid'], taxis, rec[:, k])
                               for k in range(rec.shape[1])])
    return {'case': case_key, 'dt': float(dt), 'n_steps': n_steps,
            't_end': float(taxis[-1]), 'taxis_is_exact_arange': exact,
            'taxis_sha256': rm.sha256_array(taxis),
            'wall_s': time.time() - t0,
            'on_obs': on_obs, 'on_grid': on_grid}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/rev2/b1_dt.json')
    ap.add_argument('--version', default=None,
                    help='output filename version tag; overrides '
                         'outputs.version in the config. The config FILE is '
                         'never edited for a re-run, so an earlier pass keeps '
                         'its recorded config hash.')
    args = ap.parse_args(argv)

    t_wall = time.time()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cfg_path = os.path.abspath(args.config)
    with open(cfg_path) as fh:
        cfg = json.load(fh)

    outdir = cfg['outputs']['dir']
    dpi = int(cfg['outputs']['figure_dpi'])
    ver = str(args.version or cfg['outputs'].get('version', 'v1'))
    cfg['outputs']['version'] = ver     # recorded in config.resolved
    os.makedirs(outdir, exist_ok=True)
    P = lambda name: os.path.join(outdir, name)                    # noqa: E731
    OUT = {
        'log': P(f'b1_dt_{ver}.log'),
        'json': P(f'b1_results_{ver}.json'),
        'csv_gauge': P(f'b1_error_per_gauge_{ver}.csv'),
        'csv_pooled': P(f'b1_pooled_{ver}.csv'),
        'csv_adaptive': P(f'b1_adaptive_{ver}.csv'),
        'trace': P(f'b1_adaptive_trace_{ver}.json'),
        'npz': P(f'b1_panels_{ver}.npz'),
        'fig1': P(f'fig01_b1_convergence_{ver}.png'),
        'fig2': P(f'fig02_b1_error_traces_{ver}.png'),
        'fig3': P(f'fig03_b1_adaptive_{ver}.png'),
        'readme': P('README.md' if ver == 'v1' else f'README_{ver}.md'),
        'manifest': P('manifest.json' if ver == 'v1'
                      else f'manifest_{ver}.json'),
    }
    # AMENDMENT v6. The canonical name is always the CURRENT pass. Freezing
    # README.md at v1 while the corrections went to README_v3/v5.md is what let
    # three withdrawn claims survive under the name a reader opens first.
    OUT['readme_canonical'] = P('README.md')
    _absent = [v for k, v in OUT.items() if k != 'readme_canonical']
    rm.assert_absent(_absent + [OUT['manifest'] + '.sha256'])
    _canon_twin = None
    if os.path.exists(OUT['readme_canonical']):
        # House rule 2: it may be replaced only because its exact bytes are
        # already preserved under another name here, so no content is lost.
        _h = rm.sha256_file(OUT['readme_canonical'], use_cache=False)
        _canon_twin = [f for f in sorted(os.listdir(outdir))
                       if f.startswith('README') and f != 'README.md'
                       and rm.sha256_file(os.path.join(outdir, f),
                                          use_cache=False) == _h]
        if not _canon_twin:
            raise SystemExit(
                'README.md is not byte-identical to any other README* file in '
                + outdir + '; refusing to replace it (house rule 2). Copy it '
                'to README_<tag>_superseded.md first.')
    # Everything already sitting in the directory belongs to a superseded pass.
    # House rule 2 forbids deleting it, so it is hashed as prior_run_output and
    # the undeclared-file scan is told to expect it.
    # README.md is excluded: it is a DECLARED OUTPUT of this pass, and its old
    # bytes are retained under the twin name found above, which is in the list.
    # Dotfiles are excluded too: an NFS silly-rename (.nfs*) is a live handle on
    # a deleted file, so it can vanish between this scan and the hashing in
    # write_manifest, which aborts the whole run after the solves are done.
    superseded = sorted(
        os.path.join(outdir, f) for f in os.listdir(outdir)
        if os.path.isfile(os.path.join(outdir, f))
        and f != 'README.md' and not f.startswith('.'))
    log = Tee(OUT['log'])
    log(f"# B1 time-step convergence at the working point -- started {started}")
    log(f"# config {args.config}")

    # -----------------------------------------------------------------------
    # 1. the R1 bed, verbatim
    # -----------------------------------------------------------------------
    S = rd.setup_r1(source_mode=cfg['source']['mode'],
                    pad_low_ft=float(cfg['mesh']['domain_pad_low_md_ft']),
                    pad_high_ft=float(cfg['mesh']['domain_pad_high_md_ft']),
                    dx_ft=float(cfg['mesh']['dx_ft']),
                    stage=int(cfg['window']['stage']))
    mesh, src = S['mesh'], S['src_series']
    x = mesh.x
    tgts = S['targets']
    rec_idx = [t['idx'] for t in tgts]
    t_total = float(S['t_total_s'])
    dx = float(cfg['mesh']['dx_ft'])
    theta = float(cfg['solver']['theta'])
    iavg = cfg['solver']['interface_avg']

    assert abs(mesh.x[0] - (cfg['window']['md_min_ft']
                            - cfg['mesh']['domain_pad_low_md_ft'])) < 1e-9
    log(f"\n## bed: nx={x.size}, MD {x[0]:.1f}-{x[-1]:.1f} ft, dx={dx:g} ft, "
        f"source g{S['src_gauge']} @ MD {S['src_md']:.1f} (node {S['source_idx']}), "
        f"t_total={t_total:.3f} s")
    log("## targets: " + ", ".join(f"g{t['gauge']}@{t['distance_ft']:.0f}ft"
                                   f"(n={t['data'].size})" for t in tgts))

    # -----------------------------------------------------------------------
    # 2. the three D profiles
    # -----------------------------------------------------------------------
    profiles, case_meta = {}, {}
    for c in cfg['cases']:
        if c['family'] == 'uniform':
            prof = np.full(x.size, float(c['D_ft2_s']))
        elif c['family'] == 'two_zone':
            prof = r1c.profile_two_zone(x, S['source_idx'],
                                        np.asarray(c['params_log10'], float))
        else:
            raise ValueError(f"unknown family {c['family']!r}")
        profiles[c['key']] = prof
        case_meta[c['key']] = dict(
            c, D_min_ft2_s=float(prof.min()), D_max_ft2_s=float(prof.max()),
            D_sha256=rm.sha256_array(prof),
            r_at_dt1=float(prof.max()) * 1.0 / dx ** 2)
        log(f"## case {c['key']:9s} D in [{prof.min():.1f}, {prof.max():.1f}] "
            f"ft^2/s, r = D_max*dt/dx^2 = {case_meta[c['key']]['r_at_dt1']:.1f} "
            f"at dt = 1 s -- {c['label']}")

    # -----------------------------------------------------------------------
    # 3. fixed-dt sweep, in parallel
    # -----------------------------------------------------------------------
    ts = cfg['time_step']
    dt_test = [float(v) for v in ts['dt_test_s']]
    dt_extra = [float(v) for v in ts['dt_extra_s']]
    dt_ref = float(ts['dt_reference_s'])
    dt_ref2 = float(ts['dt_reference_check_s'])
    dt_all = sorted(set(dt_test + dt_extra + [dt_ref, dt_ref2]))

    grid = np.arange(0.0, float(cfg['analysis']['dense_grid_t_end_s'])
                     + float(cfg['analysis']['dense_grid_dt_s']) / 2,
                     float(cfg['analysis']['dense_grid_dt_s']))
    obs_taxes = [t['taxis'] for t in tgts]

    jobs = [(ck, dt) for ck in profiles for dt in dt_all]
    nproc = min(int(cfg['parallel']['processes']), 6, len(jobs))
    log(f"\n## fixed-dt sweep: {len(jobs)} solves "
        f"({len(profiles)} cases x {len(dt_all)} steps), {nproc} processes")
    log("## dt values: " + ", ".join(f"{v:g}" for v in dt_all))

    t_sweep = time.time()
    with mp.Pool(nproc, initializer=_init_worker,
                 initargs=(x, profiles, src.taxis_s, src.delta_psi,
                           S['source_idx'], rec_idx, obs_taxes, grid, t_total,
                           theta, iavg)) as pool:
        raw = list(pool.imap_unordered(_solve_one, jobs, chunksize=1))
    log(f"## sweep wall {time.time() - t_sweep:.1f} s")

    SOL = {(r['case'], r['dt']): r for r in raw}
    for r in sorted(raw, key=lambda r: (r['case'], r['dt'])):
        log(f"   {r['case']:9s} dt={r['dt']:<10g} steps={r['n_steps']:<7d} "
            f"t_end={r['t_end']:9.4f} exact_axis={r['taxis_is_exact_arange']} "
            f"wall={r['wall_s']:6.2f} s")

    # -----------------------------------------------------------------------
    # 4. errors against the fine-dt reference
    # -----------------------------------------------------------------------
    obs_peak = [float(np.max(t['data'])) for t in tgts]
    gnames = [f"g{t['gauge']}" for t in tgts]

    def _metrics(sol, ref):
        """Per-gauge and pooled error of `sol` against `ref`."""
        per, sq, n = [], 0.0, 0
        for k in range(len(tgts)):
            e = sol['on_obs'][k] - ref['on_obs'][k]
            pk = float(np.max(ref['on_obs'][k]))
            pka = float(np.max(np.abs(ref['on_obs'][k])))
            eg = sol['on_grid'][:, k] - ref['on_grid'][:, k]
            per.append({
                'gauge': tgts[k]['gauge'], 'md_ft': tgts[k]['md_ft'],
                'distance_ft': tgts[k]['distance_ft'],
                'n_samples': int(e.size),
                'max_abs_psi': float(np.max(np.abs(e))),
                'rms_psi': _rms(e),
                'sim_peak_psi': pk, 'sim_peak_abs_psi': pka,
                'obs_peak_psi': obs_peak[k],
                'max_abs_frac_of_sim_peak': float(np.max(np.abs(e))) / pk,
                'rms_frac_of_sim_peak': _rms(e) / pk,
                'max_abs_frac_of_obs_peak': float(np.max(np.abs(e)))
                / obs_peak[k],
                'max_abs_dense_grid_psi': float(np.max(np.abs(eg))),
                'rms_dense_grid_psi': _rms(eg),
            })
            sq += float(np.sum(e ** 2))
            n += int(e.size)
        mse = [p['rms_psi'] ** 2 for p in per]
        return {
            'per_gauge': per,
            'pooled_sample_rms_psi': float(np.sqrt(sq / n)),
            'gauge_mean_rms_psi': float(np.sqrt(np.mean(mse))),
            'worst_gauge_max_abs_psi': max(p['max_abs_psi'] for p in per),
            'worst_gauge_max_abs_frac_of_sim_peak':
                max(p['max_abs_frac_of_sim_peak'] for p in per),
            'worst_gauge_rms_frac_of_sim_peak':
                max(p['rms_frac_of_sim_peak'] for p in per),
            'worst_gauge': gnames[int(np.argmax([p['max_abs_psi']
                                                 for p in per]))],
            'n_pooled_samples': int(n),
        }

    def _misfit(sol):
        """Model-data misfit, both conventions (R1's quantity)."""
        mse, sq, n = [], 0.0, 0
        per = []
        for k, t in enumerate(tgts):
            r = sol['on_obs'][k] - t['data']
            mse.append(float(np.mean(r ** 2)))
            sq += float(np.sum(r ** 2))
            n += int(r.size)
            per.append({'gauge': t['gauge'], 'rmse_psi': _rms(r)})
        return {'gauge_mean_rmse_psi': float(np.sqrt(np.mean(mse))),
                'pooled_rmse_psi': float(np.sqrt(sq / n)),
                'normalised_rmse': float(np.sqrt(np.mean(
                    [m / p ** 2 for m, p in zip(mse, obs_peak)]))),
                'per_gauge': per}

    RES = {'cases': {}}
    for ck in profiles:
        ref = SOL[(ck, dt_ref)]
        ref2 = SOL[(ck, dt_ref2)]
        # Richardson-extrapolated reference: backward Euler with a level-n datum
        # is first order, so 2*P(h/2) - P(h) removes the leading term.
        ext = {'on_obs': [2.0 * a - b for a, b in zip(ref2['on_obs'],
                                                      ref['on_obs'])],
               'on_grid': 2.0 * ref2['on_grid'] - ref['on_grid']}
        ref_self = _metrics(ref, ext)      # the reference's OWN residual error
        ref2_self = _metrics(ref2, ext)

        rows = {}
        for dt in dt_test + dt_extra:
            m = _metrics(SOL[(ck, dt)], ref)
            m['vs_richardson_pooled_sample_rms_psi'] = \
                _metrics(SOL[(ck, dt)], ext)['pooled_sample_rms_psi']
            m['misfit'] = _misfit(SOL[(ck, dt)])
            m['r_number'] = float(profiles[ck].max()) * dt / dx ** 2
            m['n_steps'] = SOL[(ck, dt)]['n_steps']
            m['wall_s'] = SOL[(ck, dt)]['wall_s']
            rows[f"{dt:g}"] = m
        ref_misfit = _misfit(ref)
        for key, m in rows.items():
            m['misfit']['gauge_mean_rmse_change_psi'] = (
                m['misfit']['gauge_mean_rmse_psi']
                - ref_misfit['gauge_mean_rmse_psi'])
            m['misfit']['pooled_rmse_change_psi'] = (
                m['misfit']['pooled_rmse_psi'] - ref_misfit['pooled_rmse_psi'])

        # observed order between consecutive (doubling) steps of the test grid
        order = []
        seq = sorted(dt_test + dt_extra)
        for a, b in zip(seq[:-1], seq[1:]):
            if abs(b / a - 2.0) > 1e-9:
                continue
            Ea = rows[f"{a:g}"]['pooled_sample_rms_psi']
            Eb = rows[f"{b:g}"]['pooled_sample_rms_psi']
            Wa = rows[f"{a:g}"]['worst_gauge_max_abs_psi']
            Wb = rows[f"{b:g}"]['worst_gauge_max_abs_psi']
            # AMENDMENT v6. The reference is finite, so what is measured is
            # E(dt) = C (dt - dt_ref), not C dt: exact first order then gives
            # p > 1 at the fine pairs (1.0114 for 0.25->0.5 at dt_ref = 1/256).
            # Rescaling each error by dt/(dt - dt_ref) removes that, and the
            # residual is the observed order of the scheme itself.
            ca, cb = a / (a - dt_ref), b / (b - dt_ref)
            order.append({'dt_fine_s': a, 'dt_coarse_s': b,
                          'p_pooled_rms': float(np.log2(Eb / Ea)),
                          'p_worst_max_abs': float(np.log2(Wb / Wa)),
                          'p_pooled_rms_ref_corrected':
                              float(np.log2((cb * Eb) / (ca * Ea))),
                          'p_worst_max_abs_ref_corrected':
                              float(np.log2((cb * Wb) / (ca * Wa))),
                          'ref_correction_factors': [ca, cb]})

        RES['cases'][ck] = {
            'meta': case_meta[ck], 'rows': rows, 'order': order,
            'reference': {
                'dt_ref_s': dt_ref, 'dt_ref_check_s': dt_ref2,
                'n_steps_ref': ref['n_steps'], 'n_steps_ref_check': ref2['n_steps'],
                'ref_residual_vs_richardson_pooled_rms_psi':
                    ref_self['pooled_sample_rms_psi'],
                'ref_check_residual_vs_richardson_pooled_rms_psi':
                    ref2_self['pooled_sample_rms_psi'],
                'ref_residual_worst_gauge_max_abs_psi':
                    ref_self['worst_gauge_max_abs_psi'],
                'misfit_at_reference': ref_misfit,
            },
        }
        log(f"\n## {ck}: reference dt = {dt_ref:g} s ({ref['n_steps']} steps); "
            f"its own residual vs the Richardson extrapolant is "
            f"{ref_self['pooled_sample_rms_psi']:.3e} psi pooled "
            f"(worst gauge max {ref_self['worst_gauge_max_abs_psi']:.3e} psi)")
        log(f"   {'dt':>6}  {'r':>9}  {'pooled RMS':>11}  {'gmean RMS':>10}  "
            f"{'worst max':>10}  {'worst max %pk':>13}  {'misfit dRMSE':>12}")
        for dt in sorted(dt_test + dt_extra):
            m = rows[f"{dt:g}"]
            log(f"   {dt:6g}  {m['r_number']:9.1f}  "
                f"{m['pooled_sample_rms_psi']:11.5f}  "
                f"{m['gauge_mean_rms_psi']:10.5f}  "
                f"{m['worst_gauge_max_abs_psi']:10.5f}  "
                f"{100 * m['worst_gauge_max_abs_frac_of_sim_peak']:12.4f}%  "
                f"{m['misfit']['pooled_rmse_change_psi']:12.5f}")
        log("   observed order p (pooled RMS): "
            + ", ".join(f"{o['dt_fine_s']:g}->{o['dt_coarse_s']:g}: "
                        f"{o['p_pooled_rms']:.3f}" for o in order))
        log(f"   per gauge at the production step dt = 1 s ({ck}):")
        log(f"      {'gauge':>5} {'dist ft':>8} {'sim peak':>9} {'max|e| psi':>11} "
            f"{'%peak':>8} {'rms psi':>9} {'%peak':>8}")
        for p_ in rows['1']['per_gauge']:
            log(f"      g{p_['gauge']:<4d} {p_['distance_ft']:8.0f} "
                f"{p_['sim_peak_psi']:9.2f} {p_['max_abs_psi']:11.4f} "
                f"{100 * p_['max_abs_frac_of_sim_peak']:7.3f}% "
                f"{p_['rms_psi']:9.4f} "
                f"{100 * p_['rms_frac_of_sim_peak']:7.3f}%")
        # R1's own convention, reproduced: the CHANGE IN THE MISFIT between the
        # production step and the finest step of the required grid. R1 reported
        # 0.161 psi (0.142%) that way at D = 500.
        kfine = f"{min(dt_test):g}"
        r1conv = {
            'finest_dt_of_required_grid_s': float(min(dt_test)),
            'pooled_rmse_psi_by_dt': {f"{d:g}": rows[f"{d:g}"]['misfit']
                                      ['pooled_rmse_psi'] for d in dt_test},
            'gauge_mean_rmse_psi_by_dt': {f"{d:g}": rows[f"{d:g}"]['misfit']
                                          ['gauge_mean_rmse_psi']
                                          for d in dt_test},
            'dt1_minus_dtfine_pooled_psi':
                rows['1']['misfit']['pooled_rmse_psi']
                - rows[kfine]['misfit']['pooled_rmse_psi'],
            'dt1_minus_dtfine_pooled_pct':
                100 * (rows['1']['misfit']['pooled_rmse_psi']
                       - rows[kfine]['misfit']['pooled_rmse_psi'])
                / rows['1']['misfit']['pooled_rmse_psi'],
        }
        RES['cases'][ck]['r1_convention'] = r1conv
        log(f"   R1 convention (change in the model-data misfit, dt = 1 s vs "
            f"dt = {kfine} s): {r1conv['dt1_minus_dtfine_pooled_psi']:.4f} psi "
            f"({r1conv['dt1_minus_dtfine_pooled_pct']:.4f}% of "
            f"{rows['1']['misfit']['pooled_rmse_psi']:.3f} psi pooled misfit) "
            f"-- R1 quoted 0.161 psi (0.142%) at D = 500")

    # -----------------------------------------------------------------------
    # 5. the adaptive scheme on this configuration
    # -----------------------------------------------------------------------
    ad = cfg['adaptive']
    akw = dict(dt_init=float(ad['dt_init']), tol=float(ad['tol']),
               safety_factor=float(ad['safety_factor']),
               order_p=int(ad['order_p']), max_dt=float(ad['max_dt']),
               min_dt=float(ad['min_dt']),
               controller_tol=float(ad['controller_tol']),
               zero_field_policy=ad['zero_field_policy'])
    # same controller settings, minus the argument the theta=1 local variant of
    # section 5e does not take (it has no zero field to police: the absolute
    # protocol starts from a uniform ~8300 psi field)
    akw_ss = {k: v for k, v in akw.items() if k != 'zero_field_policy'}
    keep = ('n_attempts', 'n_accepted', 'n_rejected', 'dt_min_s', 'dt_max_s',
            'dt_mean_s', 'frac_at_max_dt', 't_end_s', 't_total_requested_s',
            'overshoot_s', 'err_min', 'err_max', 'flip_margin', 'tol',
            'controller_tol', 'dt_init_s', 'max_dt_s', 'min_dt_s',
            'safety_factor', 'order_p', 'zero_field_policy', 'error_norm')
    ADAPT = {'settings': {k: (float(v) if isinstance(v, (int, float)) else v)
                          for k, v in akw.items()}}
    log("\n## adaptive scheme, manuscript settings "
        f"(tol={akw['tol']:g}, dt_init={akw['dt_init']:g}, "
        f"max_dt={akw['max_dt']:g}, min_dt={akw['min_dt']:g}, "
        f"safety={akw['safety_factor']:g}, p={akw['order_p']})")

    # --- 5a. delta-pressure protocol (what THIS working point actually uses) --
    ADAPT['delta_protocol'] = {}
    for ck in profiles:
        probe = {'case': ck}
        try:
            tax, rec, tr = rc.solve_forward_adaptive(
                x, profiles[ck], t_total, src.taxis_s, src.delta_psi,
                S['source_idx'], record_idx=rec_idx, theta=theta,
                interface_avg=iavg,
                max_attempts=int(ad['max_attempts_probe']), **akw)
            probe.update(completed=True, **{k: tr[k] for k in keep})
        except RuntimeError as exc:
            probe.update(completed=False, runtime_error=str(exc),
                         max_attempts_probe=int(ad['max_attempts_probe']))
        ADAPT['delta_protocol'][ck] = probe
        log(f"   delta protocol, {ck}: "
            + ("COMPLETED" if probe['completed']
               else f"DID NOT COMPLETE -- {probe['runtime_error']}"))

    # --- 5b. where and why it stalls, with the public kernel only -------------
    # After the first accepted step the field is identically zero, because the
    # delta-pressure datum is exactly 0 at t = 0 and a zero Dirichlet value on a
    # zero field returns a zero field. The controller then imposes a NEW datum on
    # a field whose norm is still ~0.
    #
    # AMENDMENT v3. v2 asserted here that "the relative estimate cannot be
    # reduced by shrinking dt because the Dirichlet assignment itself does not
    # shrink". That is WRONG, and this scan is what refutes it: the estimate is
    # flat only for dt above ~0.1 s, then falls at second order and crosses tol.
    # v2's own JSON already recorded any_dt_accepted = true and the crossing
    # step, so the claim contradicted the file that carried it. The stall is a
    # min_dt FLOOR artifact, quantified in 5b2 (the measured slopes) and 5c2
    # (the counterfactual run with the floor lowered). Do not restore the old
    # sentence.
    t_stall = float(akw['dt_init'])
    tax0, u0field = rc.solve_forward(
        x, profiles['D1150'], t_stall, t_stall * 0.5, src.taxis_s,
        src.delta_psi, S['source_idx'], theta=theta, interface_avg=iavg)
    field_is_zero = bool(np.all(u0field[-1] == 0.0))
    s_stall = float(np.interp(t_stall, src.taxis_s, src.delta_psi))
    log(f"\n## stall diagnostic: after the first accepted step (t = {t_stall:g} s) "
        f"the field is identically zero: {field_is_zero}; the datum then jumps to "
        f"{s_stall:.6f} psi")
    const_t = np.array([0.0, 1.0e9])
    ADAPT['stall_scan'] = {'t_stall_s': t_stall, 'datum_psi': s_stall,
                           'field_is_identically_zero': field_is_zero,
                           'tol': float(akw['tol']),
                           'min_dt_s': float(akw['min_dt']), 'cases': {}}
    for ck in profiles:
        const_v = np.array([s_stall, s_stall])
        rows = []
        hs = [akw['min_dt']] + [1.0 / 2 ** j for j in range(-2, 23)]
        for h in sorted(set(float(v) for v in hs)):
            _, uf = rc.solve_forward(x, profiles[ck], h, h * 0.5, const_t,
                                     const_v, S['source_idx'], theta=theta,
                                     interface_avg=iavg)
            _, uh = rc.solve_forward(x, profiles[ck], h / 2.0, h * 0.75, const_t,
                                     const_v, S['source_idx'], theta=theta,
                                     interface_avg=iavg)
            nrm = float(np.linalg.norm(uf[-1]))
            err = float(np.linalg.norm(uf[-1] - uh[-1]) / nrm) if nrm else 0.0
            rows.append({'dt_s': h, 'err': err, 'field_norm_psi': nrm,
                         'accepted': bool(err <= akw['tol'])})
        # the step floor the tolerance would actually require, by log-log
        # interpolation of the measured estimate through tol
        dts_s = np.array([r['dt_s'] for r in rows])
        errs_s = np.array([r['err'] for r in rows])
        o = np.argsort(dts_s)
        dts_s, errs_s = dts_s[o], errs_s[o]
        below = np.nonzero(errs_s <= akw['tol'])[0]
        if below.size:
            i = int(below[-1])
            dt_needed = float(dts_s[i])
            if i + 1 < dts_s.size:
                lx = np.log10([dts_s[i], dts_s[i + 1]])
                ly = np.log10([errs_s[i], errs_s[i + 1]])
                dt_needed = float(10 ** np.interp(np.log10(akw['tol']), ly, lx))
        else:
            dt_needed = None
        # --- 5b2. HOW the estimate depends on dt, measured, not asserted -----
        # Local log-log slopes plus least-squares slopes over two stated bands.
        # This is the evidence that decides whether the live-lock is intrinsic
        # (slope 0 everywhere) or a floor artifact (slope -> p as dt -> 0).
        slopes = []
        for i in range(1, dts_s.size):
            if errs_s[i] > 0 and errs_s[i - 1] > 0:
                slopes.append({
                    'dt_lo_s': float(dts_s[i - 1]), 'dt_hi_s': float(dts_s[i]),
                    'slope': float(np.log(errs_s[i] / errs_s[i - 1])
                                   / np.log(dts_s[i] / dts_s[i - 1]))})

        def _fit(lo, hi):
            m = (dts_s >= lo) & (dts_s <= hi) & (errs_s > 0)
            if int(m.sum()) < 2:
                return None
            return float(np.polyfit(np.log10(dts_s[m]),
                                    np.log10(errs_s[m]), 1)[0])

        coarse_lo, coarse_hi = 0.25, 4.0
        fine_hi = float(dts_s.min()) * 10.0
        flat_band = [r['err'] for r in rows if coarse_lo <= r['dt_s'] <= coarse_hi]
        ADAPT['stall_scan']['cases'][ck] = {
            'rows': rows,
            'err_at_min_dt': [r['err'] for r in rows
                              if r['dt_s'] == akw['min_dt']][0],
            'any_dt_accepted_at_or_above_min_dt': any(
                r['accepted'] for r in rows if r['dt_s'] >= akw['min_dt']),
            'any_dt_accepted': any(r['accepted'] for r in rows),
            'dt_that_would_meet_tol_s': dt_needed,
            'steps_to_1254s_at_that_dt': (None if dt_needed is None
                                          else int(round(t_total / dt_needed))),
            'min_dt_floor_shortfall_factor': (None if dt_needed is None
                                              else akw['min_dt'] / dt_needed),
            'local_slopes': slopes,
            'slope_fit_coarse_band': {'dt_lo_s': coarse_lo, 'dt_hi_s': coarse_hi,
                                      'slope': _fit(coarse_lo, coarse_hi),
                                      'err_min': float(min(flat_band)),
                                      'err_max': float(max(flat_band)),
                                      'err_spread_pct': 100.0 *
                                      (max(flat_band) / min(flat_band) - 1.0)},
            'slope_fit_finest_decade': {'dt_lo_s': float(dts_s.min()),
                                        'dt_hi_s': fine_hi,
                                        'slope': _fit(dts_s.min(), fine_hi)},
            'slope_at_min_dt': next(
                (s['slope'] for s in slopes
                 if s['dt_hi_s'] == akw['min_dt']), None),
        }
        e = ADAPT['stall_scan']['cases'][ck]['err_at_min_dt']
        sc_ = ADAPT['stall_scan']['cases'][ck]
        log(f"   {ck}: error estimate at min_dt = {akw['min_dt']:g} s is "
            f"{e:.4e} vs tol {akw['tol']:g} "
            f"({e / akw['tol']:.1f}x too large); accepted at any dt >= min_dt: "
            f"{sc_['any_dt_accepted_at_or_above_min_dt']}; accepted at SOME dt: "
            f"{sc_['any_dt_accepted']}; the tolerance would "
            f"need dt <= {sc_['dt_that_would_meet_tol_s']:.3e} s "
            f"({sc_['min_dt_floor_shortfall_factor']:.2f}x below the min_dt "
            f"floor, {sc_['steps_to_1254s_at_that_dt']:,d} steps for the window)")
        log(f"      dt-dependence of the estimate: slope "
            f"{sc_['slope_fit_coarse_band']['slope']:+.3f} over "
            f"{coarse_lo:g}-{coarse_hi:g} s (spread "
            f"{sc_['slope_fit_coarse_band']['err_spread_pct']:.1f}%), slope "
            f"{sc_['slope_fit_finest_decade']['slope']:+.3f} over the finest "
            f"decade {dts_s.min():.3g}-{fine_hi:.3g} s -- the estimate DOES "
            f"shrink with dt")

    # --- 5c. is the stall an artifact of dt_init? ---------------------------
    ADAPT['dt_init_robustness'] = []
    for di in [float(v) for v in ad['dt_init_robustness_s']]:
        kw = dict(akw); kw['dt_init'] = di
        try:
            _, _, tr = rc.solve_forward_adaptive(
                x, profiles['D1150'], t_total, src.taxis_s, src.delta_psi,
                S['source_idx'], record_idx=rec_idx, theta=theta,
                interface_avg=iavg, max_attempts=2000, **kw)
            ADAPT['dt_init_robustness'].append(
                {'dt_init_s': di, 'completed': True,
                 'n_attempts': tr['n_attempts']})
        except RuntimeError as exc:
            ADAPT['dt_init_robustness'].append(
                {'dt_init_s': di, 'completed': False, 'runtime_error': str(exc)})
    log("   dt_init robustness (delta protocol, D1150): "
        + "; ".join(f"dt_init={r['dt_init_s']:g} -> "
                    + ('completed' if r['completed'] else 'live-locked')
                    for r in ADAPT['dt_init_robustness']))

    # --- 5c2. THE COUNTERFACTUAL: lower only min_dt, change nothing else -----
    # AMENDMENT v3. The stall scan says the estimate crosses tol at a dt BELOW
    # the published floor, so the prediction is that lowering the floor alone
    # lets the run complete. That prediction is tested here rather than argued.
    delta_traces = {}
    ADAPT['min_dt_counterfactual'] = {'published_min_dt_s': float(akw['min_dt']),
                                      'note': cfg['adaptive']
                                      ['min_dt_counterfactual_rationale'],
                                      'cases': {}}
    for ck in profiles:
        ADAPT['min_dt_counterfactual']['cases'][ck] = []
        for md in [float(v) for v in ad['min_dt_counterfactual_s']]:
            kw = dict(akw); kw['min_dt'] = md
            row = {'min_dt_s': md,
                   'is_published_setting': bool(md == akw['min_dt'])}
            try:
                tax, rec, tr = rc.solve_forward_adaptive(
                    x, profiles[ck], t_total, src.taxis_s, src.delta_psi,
                    S['source_idx'], record_idx=rec_idx, theta=theta,
                    interface_avg=iavg,
                    max_attempts=int(ad['max_attempts_probe']), **kw)
                on_obs = [np.interp(tq, tax, rec[:, k])
                          for k, tq in enumerate(obs_taxes)]
                on_grid = np.column_stack([np.interp(grid, tax, rec[:, k])
                                           for k in range(rec.shape[1])])
                sol = {'on_obs': on_obs, 'on_grid': on_grid}
                m = _metrics(sol, SOL[(ck, dt_ref)])
                m['misfit'] = _misfit(sol)
                dts_r = np.diff(tax)
                rej_t = sorted({round(float(a['t']), 6) for a in tr['attempts']
                                if not a['accepted']})
                row.update(
                    completed=True, **{k: tr[k] for k in keep},
                    dt_median_s=float(np.median(dts_r)),
                    n_steps_below_1s=int((dts_r < 1.0).sum()),
                    floor_is_active=bool(np.isclose(tr['dt_min_s'], md,
                                                    rtol=1e-9)),
                    n_rejection_times=len(rej_t),
                    n_rejections_at_first_stall=sum(
                        1 for a in tr['attempts']
                        if not a['accepted']
                        and abs(a['t'] - t_stall) < 1e-9),
                    max_err_of_accepted_steps=max(
                        a['err'] for a in tr['attempts'] if a['accepted']),
                    error_vs_fine_reference=m)
                delta_traces[(ck, md)] = {
                    't': np.asarray(tax, float), 'dt': dts_r,
                    'attempt_t': np.array([a['t'] for a in tr['attempts']]),
                    'attempt_dt': np.array([a['dt'] for a in tr['attempts']]),
                    'attempt_err': np.array([a['err'] for a in tr['attempts']]),
                    'attempt_acc': np.array([a['accepted']
                                             for a in tr['attempts']])}
                log(f"   min_dt counterfactual, {ck}, min_dt={md:g}: COMPLETED "
                    f"attempts={tr['n_attempts']}, accepted={tr['n_accepted']}, "
                    f"rejected={tr['n_rejected']} at {len(rej_t)} distinct "
                    f"times, dt {tr['dt_min_s']:.4g}-{tr['dt_max_s']:g} s "
                    f"(median {np.median(dts_r):.4g}), "
                    f"{100 * tr['frac_at_max_dt']:.2f}% at max_dt, floor "
                    f"active={row['floor_is_active']}, pooled RMS vs the "
                    f"dt={dt_ref:g} s reference "
                    f"{m['pooled_sample_rms_psi']:.4f} psi")
            except RuntimeError as exc:
                row.update(completed=False, runtime_error=str(exc),
                           max_attempts_probe=int(ad['max_attempts_probe']))
                log(f"   min_dt counterfactual, {ck}, min_dt={md:g}: "
                    f"LIVE-LOCKED -- {exc}")
            ADAPT['min_dt_counterfactual']['cases'][ck].append(row)

    # --- 5c3. the counterfactual does NOT rescue stock fibeRIS --------------
    # rev2_core's zero_field_policy='accept' is a work-around for a SECOND,
    # independent live-lock: fibeRIS evaluates 0/0 = nan on the identically-zero
    # first step, rejects, and pins dt at min_dt forever. That one sits at t = 0
    # and no floor can cure it, because the field is zero whatever dt is taken.
    ADAPT['zero_field_policy_probe'] = []
    for pol in list(ad['zero_field_policy_probe']):
        for md in [float(v) for v in ad['min_dt_counterfactual_s']]:
            kw = dict(akw); kw['min_dt'] = md; kw['zero_field_policy'] = pol
            r_ = {'zero_field_policy': pol, 'min_dt_s': md}
            try:
                _, _, tr = rc.solve_forward_adaptive(
                    x, profiles['D1150'], t_total, src.taxis_s, src.delta_psi,
                    S['source_idx'], record_idx=rec_idx, theta=theta,
                    interface_avg=iavg, max_attempts=2000, **kw)
                r_.update(completed=True, n_attempts=tr['n_attempts'],
                          n_rejected=tr['n_rejected'])
            except RuntimeError as exc:
                r_.update(completed=False, runtime_error=str(exc),
                          stall_t_s=float(str(exc).split('at t=')[1]
                                          .split(',')[0]))
            ADAPT['zero_field_policy_probe'].append(r_)
    log("   zero_field_policy probe (D1150): "
        + "; ".join(f"{r['zero_field_policy']}/min_dt={r['min_dt_s']:g} -> "
                    + ('completed' if r['completed']
                       else f"live-locked at t={r['stall_t_s']:g}")
                    for r in ADAPT['zero_field_policy_probe']))

    # --- 5d. absolute-pressure protocol (what 101 does) ---------------------
    ADAPT['absolute_protocol'] = {}
    abs_traces = {}
    abs_rec = {}                 # recorded gauge series, for the 5e bitwise check
    p_abs0 = float(src.raw_psi[0])
    for ck in profiles:
        u_init = np.full(x.size, p_abs0)
        tax, rec, tr = rc.solve_forward_adaptive(
            x, profiles[ck], t_total, src.taxis_s, src.raw_psi,
            S['source_idx'], initial=u_init, record_idx=rec_idx, theta=theta,
            interface_avg=iavg, **akw)
        # linearity: the delta-pressure solution is this minus the uniform datum
        on_obs = [np.interp(tq, tax, rec[:, k] - p_abs0)
                  for k, tq in enumerate(obs_taxes)]
        on_grid = np.column_stack([np.interp(grid, tax, rec[:, k] - p_abs0)
                                   for k in range(rec.shape[1])])
        sol = {'on_obs': on_obs, 'on_grid': on_grid}
        m = _metrics(sol, SOL[(ck, dt_ref)])
        m['misfit'] = _misfit(sol)
        m['misfit']['pooled_rmse_change_psi'] = (
            m['misfit']['pooled_rmse_psi']
            - RES['cases'][ck]['reference']['misfit_at_reference']
            ['pooled_rmse_psi'])
        m['misfit']['gauge_mean_rmse_change_psi'] = (
            m['misfit']['gauge_mean_rmse_psi']
            - RES['cases'][ck]['reference']['misfit_at_reference']
            ['gauge_mean_rmse_psi'])
        dts = np.diff(tax)
        ADAPT['absolute_protocol'][ck] = {
            'trace': {k: tr[k] for k in keep},
            'initial_condition_psi': p_abs0,
            'error_vs_fine_reference': m,
            'dt_histogram': {f"{v:g}": int(c) for v, c in
                             zip(*np.unique(np.round(dts, 9),
                                            return_counts=True))},
        }
        abs_rec[ck] = np.asarray(rec, float)
        abs_traces[ck] = {'t': np.asarray(tax, float), 'dt': dts,
                          'attempt_t': np.array([a['t'] for a in tr['attempts']]),
                          'attempt_dt': np.array([a['dt'] for a in tr['attempts']]),
                          'attempt_err': np.array([a['err'] for a in tr['attempts']]),
                          'attempt_acc': np.array([a['accepted']
                                                   for a in tr['attempts']])}
        log(f"   absolute protocol, {ck}: attempts={tr['n_attempts']}, "
            f"accepted={tr['n_accepted']}, rejected={tr['n_rejected']}, "
            f"dt {tr['dt_min_s']:g}-{tr['dt_max_s']:g} s "
            f"(mean {tr['dt_mean_s']:.3f}), {100 * tr['frac_at_max_dt']:.2f}% at "
            f"max_dt, err_max={tr['err_max']:.3e} vs tol {akw['tol']:g}, "
            f"flip_margin={tr['flip_margin']:.3f}, overshoot={tr['overshoot_s']:.3f} s")
        log(f"      cost vs the dt = {dt_ref:g} s reference: pooled RMS "
            f"{m['pooled_sample_rms_psi']:.3f} psi, worst gauge "
            f"{m['worst_gauge_max_abs_psi']:.3f} psi = "
            f"{100 * m['worst_gauge_max_abs_frac_of_sim_peak']:.2f}% of that "
            f"gauge's peak ({m['worst_gauge']})")

    # is the adaptive run just the fixed dt = 30 s run?
    for ck in profiles:
        a = ADAPT['absolute_protocol'][ck]['error_vs_fine_reference']
        f30 = RES['cases'][ck]['rows']['30']
        ADAPT['absolute_protocol'][ck]['vs_fixed_dt30'] = {
            'adaptive_pooled_rms_psi': a['pooled_sample_rms_psi'],
            'fixed_dt30_pooled_rms_psi': f30['pooled_sample_rms_psi'],
            'ratio': a['pooled_sample_rms_psi'] / f30['pooled_sample_rms_psi'],
        }

    # --- 5e. AMENDMENT v5: how much of the absolute-protocol headline is the
    # 8300 psi normalisation, and how much is fibeRIS's frozen sub-step datum?
    # The headline (0 rejections, 97.67% at max_dt, estimator never above
    # 3.3e-04, flip_margin 0.755) was attributed to the datum alone. It is not.
    # rev2_core.py:930-936 reproduces fibeRIS's OTHER habit: all three solves in
    # the estimator see s(t^n). Correct only that -- each sub-step reads the
    # datum at its own left endpoint, the level-n rule applied consistently --
    # and nothing else, and re-measure. `_adaptive_substep` is asserted below to
    # reproduce rev2_core BITWISE on the 'frozen' setting, so the difference
    # between the two rows is the switch and nothing else.
    ADAPT['substep_datum_counterfactual'] = {
        'why': ('the reported absolute-protocol trace inherits fibeRIS\'s '
                'frozen level-n datum inside the error estimator\'s sub-steps '
                '(rev2_core.py:930-936). This measures how much of "0 '
                'rejections / 97.67% at max_dt" depends on that defect, with '
                'tol, bounds, protocol and accepted-state formula unchanged.'),
        'equivalence_check': {}, 'cases': {}}
    substep_traces = {}
    log("\n## 5e. sub-step-datum counterfactual (AMENDMENT v5): the absolute "
        "protocol re-run with the error estimator's sub-steps reading the datum "
        "at their own times, tol and bounds unchanged")
    for ck in profiles:
        u_init = np.full(x.size, p_abs0)
        # (i) equivalence: the local controller on 'frozen' must BE rev2_core
        taxF, recF, trF = _adaptive_substep(
            x, profiles[ck], t_total, src.taxis_s, src.raw_psi,
            S['source_idx'], substep_datum='frozen', initial=u_init,
            record_idx=rec_idx, theta=theta, interface_avg=iavg, **akw_ss)
        ref_tax = np.asarray(abs_traces[ck]['t'], float)
        d_tax = float(np.max(np.abs(taxF - ref_tax))) if taxF.shape == ref_tax.shape else float('nan')
        d_rec = float(np.max(np.abs(recF - abs_rec[ck]))) if recF.shape == abs_rec[ck].shape else float('nan')
        ADAPT['substep_datum_counterfactual']['equivalence_check'][ck] = {
            'same_shape': bool(taxF.shape == ref_tax.shape
                               and recF.shape == abs_rec[ck].shape),
            'max_abs_taxis_diff_s': d_tax,
            'max_abs_recorded_diff_psi': d_rec,
            'n_attempts_local': trF['n_attempts'],
            'n_attempts_rev2_core':
                ADAPT['absolute_protocol'][ck]['trace']['n_attempts'],
            'err_max_local': trF['err_max'],
            'err_max_rev2_core':
                ADAPT['absolute_protocol'][ck]['trace']['err_max'],
        }
        assert d_tax == 0.0 and d_rec == 0.0, (ck, d_tax, d_rec)
        # (ii) the counterfactual
        taxO, recO, trO = _adaptive_substep(
            x, profiles[ck], t_total, src.taxis_s, src.raw_psi,
            S['source_idx'], substep_datum='own_time', initial=u_init,
            record_idx=rec_idx, theta=theta, interface_avg=iavg, **akw_ss)
        on_obs = [np.interp(tq, taxO, recO[:, k] - p_abs0)
                  for k, tq in enumerate(obs_taxes)]
        on_grid = np.column_stack([np.interp(grid, taxO, recO[:, k] - p_abs0)
                                   for k in range(recO.shape[1])])
        mO = _metrics({'on_obs': on_obs, 'on_grid': on_grid},
                      SOL[(ck, dt_ref)])
        base = ADAPT['absolute_protocol'][ck]
        ADAPT['substep_datum_counterfactual']['cases'][ck] = {
            'frozen': {'trace': {k: base['trace'][k] for k in keep},
                       'error_vs_fine_reference': {
                           kk: base['error_vs_fine_reference'][kk]
                           for kk in ('pooled_sample_rms_psi',
                                      'gauge_mean_rms_psi', 'worst_gauge',
                                      'worst_gauge_max_abs_psi',
                                      'worst_gauge_max_abs_frac_of_sim_peak')}},
            'own_time': {'trace': trO,
                         'error_vs_fine_reference': {
                             kk: mO[kk]
                             for kk in ('pooled_sample_rms_psi',
                                        'gauge_mean_rms_psi', 'worst_gauge',
                                        'worst_gauge_max_abs_psi',
                                        'worst_gauge_max_abs_frac_of_sim_peak')}},
            'err_max_ratio': trO['err_max'] / base['trace']['err_max'],
            'estimate_above_tol_when_corrected': bool(trO['err_max']
                                                      > float(akw['tol'])),
        }
        att_o = trO.pop('attempts')
        substep_traces[ck] = {
            't': np.asarray(taxO, float),
            'attempt_t': np.array([a['t'] for a in att_o]),
            'attempt_dt': np.array([a['dt'] for a in att_o]),
            'attempt_err': np.array([a['err'] for a in att_o]),
            'attempt_acc': np.array([a['accepted'] for a in att_o]),
        }
        cf = ADAPT['substep_datum_counterfactual']['cases'][ck]
        log(f"   sub-step datum counterfactual, {ck}: frozen (fibeRIS, "
            f"reported) {base['trace']['n_attempts']} attempts / "
            f"{base['trace']['n_rejected']} rejected / "
            f"{100 * base['trace']['frac_at_max_dt']:.2f}% at max_dt / "
            f"err_max {base['trace']['err_max']:.4e}  ->  own-time sub-steps "
            f"{trO['n_attempts']} / {trO['n_rejected']} / "
            f"{100 * trO['frac_at_max_dt']:.2f}% / {trO['err_max']:.4e} "
            f"({cf['err_max_ratio']:.2f}x, "
            f"{'ABOVE' if cf['estimate_above_tol_when_corrected'] else 'below'}"
            f" tol = {akw['tol']:g})")
        log(f"      cost vs the dt = {dt_ref:g} s reference: "
            f"{base['error_vs_fine_reference']['pooled_sample_rms_psi']:.4f} "
            f"psi pooled frozen -> {mO['pooled_sample_rms_psi']:.4f} psi "
            f"own-time; worst gauge "
            f"{base['error_vs_fine_reference']['worst_gauge_max_abs_psi']:.3f}"
            f" -> {mO['worst_gauge_max_abs_psi']:.3f} psi "
            f"({100 * mO['worst_gauge_max_abs_frac_of_sim_peak']:.2f}% of peak)"
            f"; local 'frozen' controller reproduces rev2_core bitwise "
            f"(max|dtaxis| = {d_tax:g} s, max|dP| = {d_rec:g} psi)")

    # -----------------------------------------------------------------------
    # 6. products
    # -----------------------------------------------------------------------
    with open(OUT['csv_gauge'], 'w') as fh:
        fh.write('case,dt_s,r_number,gauge,md_ft,distance_ft,n_samples,'
                 'max_abs_psi,rms_psi,sim_peak_psi,obs_peak_psi,'
                 'max_abs_pct_of_sim_peak,rms_pct_of_sim_peak,'
                 'max_abs_dense_grid_psi\n')
        for ck in profiles:
            for dt in sorted(dt_test + dt_extra):
                m = RES['cases'][ck]['rows'][f"{dt:g}"]
                for p in m['per_gauge']:
                    fh.write(f"{ck},{dt:g},{m['r_number']:.4f},{p['gauge']},"
                             f"{p['md_ft']:.1f},{p['distance_ft']:.1f},"
                             f"{p['n_samples']},{p['max_abs_psi']:.6f},"
                             f"{p['rms_psi']:.6f},{p['sim_peak_psi']:.4f},"
                             f"{p['obs_peak_psi']:.4f},"
                             f"{100 * p['max_abs_frac_of_sim_peak']:.5f},"
                             f"{100 * p['rms_frac_of_sim_peak']:.5f},"
                             f"{p['max_abs_dense_grid_psi']:.6f}\n")

    with open(OUT['csv_pooled'], 'w') as fh:
        fh.write('case,D_max_ft2_s,dt_s,r_number,n_steps,'
                 'pooled_sample_rms_psi,gauge_mean_rms_psi,'
                 'worst_gauge,worst_gauge_max_abs_psi,'
                 'worst_gauge_max_abs_pct_of_peak,'
                 'misfit_gauge_mean_rmse_psi,misfit_pooled_rmse_psi,'
                 'misfit_pooled_rmse_change_psi\n')
        for ck in profiles:
            for dt in sorted(dt_test + dt_extra):
                m = RES['cases'][ck]['rows'][f"{dt:g}"]
                fh.write(f"{ck},{case_meta[ck]['D_max_ft2_s']:.1f},{dt:g},"
                         f"{m['r_number']:.4f},{m['n_steps']},"
                         f"{m['pooled_sample_rms_psi']:.6f},"
                         f"{m['gauge_mean_rms_psi']:.6f},{m['worst_gauge']},"
                         f"{m['worst_gauge_max_abs_psi']:.6f},"
                         f"{100 * m['worst_gauge_max_abs_frac_of_sim_peak']:.5f},"
                         f"{m['misfit']['gauge_mean_rmse_psi']:.6f},"
                         f"{m['misfit']['pooled_rmse_psi']:.6f},"
                         f"{m['misfit']['pooled_rmse_change_psi']:.6f}\n")

    # One flat row per adaptive run, so every adaptive number in the README can
    # be traced to a csv cell without opening the JSON.
    with open(OUT['csv_adaptive'], 'w') as fh:
        fh.write('case,protocol,min_dt_s,is_published_setting,completed,'
                 'n_attempts,n_accepted,n_rejected,n_rejection_times,'
                 'dt_min_s,dt_median_s,dt_mean_s,dt_max_s,frac_at_max_dt,'
                 'floor_is_active,err_max,tol,t_end_s,'
                 'pooled_sample_rms_psi,gauge_mean_rms_psi,worst_gauge,'
                 'worst_gauge_max_abs_psi,worst_gauge_max_abs_pct_of_peak,'
                 'runtime_error\n')

        def _arow(case, proto, md, pub, r, m):
            f = [case, proto, f"{md:g}", str(int(pub))]
            if r is None or not r.get('completed', False):
                f += ['0'] + [''] * 11 + [f"{akw['tol']:g}"] + [''] * 6
                f += ['"' + (r or {}).get('runtime_error', ''
                                          ).replace('"', "'") + '"']
            else:
                f += ['1', str(r['n_attempts']), str(r['n_accepted']),
                      str(r['n_rejected']), str(r.get('n_rejection_times', '')),
                      f"{r['dt_min_s']:.6g}",
                      f"{r['dt_median_s']:.6g}" if 'dt_median_s' in r else '',
                      f"{r['dt_mean_s']:.6g}", f"{r['dt_max_s']:.6g}",
                      f"{r['frac_at_max_dt']:.6f}",
                      str(int(r['floor_is_active']))
                      if 'floor_is_active' in r else '',
                      f"{r['err_max']:.6e}", f"{r['tol']:g}",
                      f"{r['t_end_s']:.4f}",
                      f"{m['pooled_sample_rms_psi']:.6f}",
                      f"{m['gauge_mean_rms_psi']:.6f}", m['worst_gauge'],
                      f"{m['worst_gauge_max_abs_psi']:.6f}",
                      f"{100 * m['worst_gauge_max_abs_frac_of_sim_peak']:.5f}",
                      '']
            assert len(f) == 24, len(f)
            fh.write(','.join(f) + '\n')

        for ck in profiles:
            for r in ADAPT['min_dt_counterfactual']['cases'][ck]:
                _arow(ck, 'delta', r['min_dt_s'], r['is_published_setting'], r,
                      r.get('error_vs_fine_reference'))
            a = ADAPT['absolute_protocol'][ck]
            _arow(ck, 'absolute', akw['min_dt'], True,
                  dict(a['trace'], completed=True),
                  a['error_vs_fine_reference'])
            # AMENDMENT v5: the same run with the estimator's sub-step datum
            # corrected. is_published_setting = 0 because fibeRIS freezes it.
            o = ADAPT['substep_datum_counterfactual']['cases'][ck]['own_time']
            _arow(ck, 'absolute_own_time_substeps', akw['min_dt'], False,
                  dict(o['trace'], completed=True),
                  o['error_vs_fine_reference'])

    npz = {'dense_grid_s': grid, 'gauge_numbers': np.array(
        [t['gauge'] for t in tgts]), 'gauge_md_ft': np.array(
        [t['md_ft'] for t in tgts]), 'mesh_md_ft': x}
    for ck in profiles:
        npz[f'D_profile__{ck}'] = profiles[ck]
        npz[f'ref_on_grid__{ck}'] = SOL[(ck, dt_ref)]['on_grid']
        for dt in dt_test + dt_extra:
            npz[f'err_on_grid__{ck}__dt{dt:g}'] = (
                SOL[(ck, dt)]['on_grid'] - SOL[(ck, dt_ref)]['on_grid'])
        tr = abs_traces[ck]
        npz[f'adaptive_abs_attempt_t__{ck}'] = tr['attempt_t']
        npz[f'adaptive_abs_attempt_dt__{ck}'] = tr['attempt_dt']
        npz[f'adaptive_abs_attempt_err__{ck}'] = tr['attempt_err']
        so = substep_traces[ck]
        npz[f'adaptive_abs_owntime_t__{ck}'] = so['t']
        npz[f'adaptive_abs_owntime_attempt_t__{ck}'] = so['attempt_t']
        npz[f'adaptive_abs_owntime_attempt_dt__{ck}'] = so['attempt_dt']
        npz[f'adaptive_abs_owntime_attempt_err__{ck}'] = so['attempt_err']
        npz[f'adaptive_abs_owntime_attempt_acc__{ck}'] = so['attempt_acc']
        npz[f'stall_dt__{ck}'] = np.array(
            [r['dt_s'] for r in ADAPT['stall_scan']['cases'][ck]['rows']])
        npz[f'stall_err__{ck}'] = np.array(
            [r['err'] for r in ADAPT['stall_scan']['cases'][ck]['rows']])
    for (ck, md), tr in delta_traces.items():
        tag = f"{ck}__mindt{md:g}"
        npz[f'adaptive_delta_t__{tag}'] = tr['t']
        npz[f'adaptive_delta_attempt_t__{tag}'] = tr['attempt_t']
        npz[f'adaptive_delta_attempt_dt__{tag}'] = tr['attempt_dt']
        npz[f'adaptive_delta_attempt_err__{tag}'] = tr['attempt_err']
        npz[f'adaptive_delta_attempt_acc__{tag}'] = tr['attempt_acc']
    for k, t in enumerate(tgts):
        npz[f'obs_taxis__g{t["gauge"]}'] = t['taxis']
        npz[f'obs_data__g{t["gauge"]}'] = t['data']
    np.savez_compressed(OUT['npz'], **npz)
    log(f"\n## wrote {OUT['npz']} ({len(npz)} arrays)")

    with open(OUT['trace'], 'w') as fh:
        json.dump(ADAPT, fh, indent=2, default=str)

    # ---- figures ----------------------------------------------------------
    colors = {'D1150': '#1f77b4', 'D1223': '#d62728', 'two_zone': '#2ca02c'}
    seq = sorted(dt_test + dt_extra)

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
    for ck in profiles:
        E = [RES['cases'][ck]['rows'][f"{d:g}"]['pooled_sample_rms_psi']
             for d in seq]
        M = [RES['cases'][ck]['rows'][f"{d:g}"]['worst_gauge_max_abs_psi']
             for d in seq]
        ax[0].loglog(seq, E, 'o-', color=colors[ck], label=case_meta[ck]['key'])
        ax[1].loglog(seq, M, 'o-', color=colors[ck], label=case_meta[ck]['key'])
    ref_line = np.array(seq, float)
    base = RES['cases']['D1150']['rows']['1']['pooled_sample_rms_psi']
    ax[0].loglog(ref_line, base * ref_line / 1.0, 'k--', lw=1,
                 label='slope 1 (first order)')
    for a, ttl, yl in ((ax[0], 'pooled sample RMS error', 'psi'),
                       (ax[1], 'worst-gauge maximum error', 'psi')):
        a.axvline(1.0, color='0.5', lw=1, ls=':')
        a.set_xlabel(r'$\Delta t$  (s)')
        a.set_ylabel(f'error vs $\\Delta t_{{ref}}$ = 1/256 s  ({yl})')
        a.set_title(ttl)
        a.grid(True, which='both', alpha=0.3)
        a.legend(fontsize=8)
    for ck in profiles:
        for k in range(len(tgts)):
            F = [100 * RES['cases'][ck]['rows'][f"{d:g}"]['per_gauge'][k]
                 ['max_abs_frac_of_sim_peak'] for d in seq]
            ax[2].loglog(seq, F, '-', color=colors[ck], alpha=0.35 + 0.1 * k,
                         lw=1.2,
                         label=(case_meta[ck]['key'] if k == 0 else None))
    ax[2].axvline(1.0, color='0.5', lw=1, ls=':')
    ax[2].axhline(1.0, color='k', lw=0.8, ls='--')
    ax[2].set_xlabel(r'$\Delta t$  (s)')
    ax[2].set_ylabel('max error / simulated peak  (%)')
    ax[2].set_title('per gauge, all six targets')
    ax[2].grid(True, which='both', alpha=0.3)
    ax[2].legend(fontsize=8)
    fig.suptitle('B1  time-step convergence at the working point '
                 f'(R1 bed, dx = {dx:g} ft, backward Euler, level-n datum)')
    fig.tight_layout()
    fig.savefig(OUT['fig1'], dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 8.0), sharex=True)
    for j, ck in enumerate(profiles):
        for k, t in enumerate(tgts):
            e = (SOL[(ck, 1.0)]['on_grid'][:, k]
                 - SOL[(ck, dt_ref)]['on_grid'][:, k])
            ax[0, j].plot(grid, e, lw=1.0, label=f"g{t['gauge']}")
            ax[1, j].plot(grid, SOL[(ck, dt_ref)]['on_grid'][:, k], lw=1.0)
        ax[0, j].set_title(f"{ck}: error at $\\Delta t$ = 1 s")
        ax[0, j].grid(alpha=0.3)
        ax[1, j].set_title(f"{ck}: reference solution")
        ax[1, j].grid(alpha=0.3)
        ax[1, j].set_xlabel('t  (s)')
    ax[0, 0].set_ylabel(r'$P_{\Delta t=1} - P_{ref}$  (psi)')
    ax[1, 0].set_ylabel(r'$P_{ref}$  (psi)')
    ax[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle('B1  where the dt = 1 s error lives in time '
                 '(negative = the coarse step lags the reference)')
    fig.tight_layout()
    fig.savefig(OUT['fig2'], dpi=dpi)
    plt.close(fig)

    md_cf = sorted({md for (_c, md) in delta_traces})
    md_free = min(md_cf) if md_cf else None      # the lowest floor that ran
    fig, AX = plt.subplots(2, 2, figsize=(13.5, 9.4))
    ax = AX.ravel()
    for ck in profiles:
        tr = abs_traces[ck]
        ax[0].step(tr['attempt_t'], tr['attempt_dt'], where='post',
                   color=colors[ck], label=ck)
        ax[1].semilogy(tr['attempt_t'], np.maximum(tr['attempt_err'], 1e-14),
                       'o-', ms=3, color=colors[ck], label=f'{ck}, as fibeRIS')
        so = substep_traces[ck]
        ax[1].semilogy(so['attempt_t'],
                       np.maximum(so['attempt_err'], 1e-14), 's--', ms=3,
                       lw=1.0, color=colors[ck], alpha=0.75,
                       label=f'{ck}, sub-steps at their own times')
    ax[0].axhline(akw['max_dt'], color='k', ls='--', lw=1,
                  label=f"max_dt = {akw['max_dt']:g} s")
    ax[0].set_xlabel('t  (s)'); ax[0].set_ylabel(r'$\Delta t$  (s)')
    ax[0].set_title('absolute-pressure protocol, published settings:\n'
                    'realised step pins at max_dt, 0 rejections')
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].axhline(akw['tol'], color='k', ls='--', lw=1,
                  label=f"tol = {akw['tol']:g}")
    ax[1].set_xlabel('t  (s)'); ax[1].set_ylabel('relative error estimate')
    _cf0 = ADAPT['substep_datum_counterfactual']['cases']['D1150']
    ax[1].set_title(
        'absolute protocol: the estimator never binds -- but only with\n'
        "fibeRIS's frozen level-n datum inside the sub-steps (solid).\n"
        'Correcting only that (dashed) raises the D1150 peak '
        f"{_cf0['err_max_ratio']:.1f}x, above tol")
    ax[1].grid(alpha=0.3, which='both'); ax[1].legend(fontsize=7, ncol=2)
    for ck in profiles:
        sc_ = ADAPT['stall_scan']['cases'][ck]
        d = np.array([r['dt_s'] for r in sc_['rows']])
        e = np.array([r['err'] for r in sc_['rows']])
        ax[2].loglog(d, e, 'o-', ms=3, color=colors[ck], label=ck)
        if sc_['dt_that_would_meet_tol_s'] is not None:
            ax[2].plot([sc_['dt_that_would_meet_tol_s']], [akw['tol']], '*',
                       ms=13, color=colors[ck], mec='k', mew=0.5, zorder=5)
    sc0_ = ADAPT['stall_scan']['cases']['D1150']
    ax[2].axhline(akw['tol'], color='k', ls='--', lw=1,
                  label=f"tol = {akw['tol']:g}")
    ax[2].axvline(akw['min_dt'], color='0.4', ls=':', lw=1.4,
                  label=f"published min_dt = {akw['min_dt']:g} s")
    if md_free is not None and md_free != akw['min_dt']:
        ax[2].axvline(md_free, color='#8c564b', ls='-.', lw=1.2,
                      label=f"lowered min_dt = {md_free:g} s")
    ax[2].plot([], [], 'k*', ms=11, label='estimate crosses tol')
    ax[2].set_xlabel(r'trial $\Delta t$  (s)')
    ax[2].set_ylabel('relative error estimate')
    ax[2].set_title(
        f"delta protocol at the stall (t = {t_stall:g} s): the estimate DOES\n"
        f"shrink -- flat above ~0.1 s, slope "
        f"{sc0_['slope_fit_finest_decade']['slope']:+.2f} at the fine end, "
        f"crossing at {sc0_['dt_that_would_meet_tol_s']:.2e} s")
    ax[2].grid(alpha=0.3, which='both'); ax[2].legend(fontsize=7.5)
    if md_free is not None:
        for ck in profiles:
            tr = delta_traces.get((ck, md_free))
            if tr is None:
                continue
            ax[3].step(tr['t'][:-1], tr['dt'], where='post', color=colors[ck],
                       lw=1.0, label=ck)
            rj = ~tr['attempt_acc'].astype(bool)
            ax[3].plot(tr['attempt_t'][rj], tr['attempt_dt'][rj], 'x',
                       ms=5, color=colors[ck], alpha=0.85,
                       label=f"{ck}: {int(rj.sum())} rejected")
        ax[3].axhline(akw['max_dt'], color='k', ls='--', lw=1,
                      label=f"max_dt = {akw['max_dt']:g} s")
        ax[3].axhline(md_free, color='#8c564b', ls='-.', lw=1.0,
                      label=f"min_dt = {md_free:g} s (never reached)")
        ax[3].set_yscale('log')
        cf0 = [r for r in ADAPT['min_dt_counterfactual']['cases']['D1150']
               if r['min_dt_s'] == md_free][0]
        ax[3].set_title(
            f"COUNTERFACTUAL: same settings, min_dt = {md_free:g} s only.\n"
            f"D1150 completes: {cf0['n_accepted']} accepted / "
            f"{cf0['n_rejected']} rejected, median "
            f"$\\Delta t$ = {cf0['dt_median_s']:.2f} s")
    else:
        ax[3].set_title('counterfactual not run')
    ax[3].set_xlabel('t  (s)'); ax[3].set_ylabel(r'$\Delta t$  (s)')
    ax[3].grid(alpha=0.3, which='both'); ax[3].legend(fontsize=7, ncol=2)
    fig.suptitle('B1  the manuscript adaptive settings on this configuration '
                 f"(tol = {akw['tol']:g}, dt_init = {akw['dt_init']:g} s, "
                 f"min_dt = {akw['min_dt']:g} s, max_dt = {akw['max_dt']:g} s, "
                 f"safety = {akw['safety_factor']:g}, p = {akw['order_p']:d})")
    fig.tight_layout()
    fig.savefig(OUT['fig3'], dpi=dpi)
    plt.close(fig)

    # ---- results json -----------------------------------------------------
    headline = {}
    for ck in profiles:
        m = RES['cases'][ck]['rows']['1']
        headline[ck] = {
            'dt_1s_pooled_sample_rms_psi': m['pooled_sample_rms_psi'],
            'dt_1s_gauge_mean_rms_psi': m['gauge_mean_rms_psi'],
            'dt_1s_worst_gauge': m['worst_gauge'],
            'dt_1s_worst_gauge_max_abs_psi': m['worst_gauge_max_abs_psi'],
            'dt_1s_worst_gauge_max_abs_pct_of_peak':
                100 * m['worst_gauge_max_abs_frac_of_sim_peak'],
            'dt_1s_worst_gauge_rms_pct_of_peak':
                100 * m['worst_gauge_rms_frac_of_sim_peak'],
            'dt_1s_misfit_pooled_rmse_change_psi':
                m['misfit']['pooled_rmse_change_psi'],
            'dt_1s_misfit_gauge_mean_rmse_change_psi':
                m['misfit']['gauge_mean_rmse_change_psi'],
            'observed_order_pooled_rms': [o['p_pooled_rms']
                                          for o in RES['cases'][ck]['order']],
        }
    RES['headline'] = headline
    RES['adaptive'] = ADAPT
    RES['bed'] = {
        'nx': int(x.size), 'dx_ft': dx, 'md_range_ft': [float(x[0]),
                                                        float(x[-1])],
        'source_gauge': int(S['src_gauge']), 'source_md_ft': float(S['src_md']),
        'source_node': int(S['source_idx']), 't_total_s': t_total,
        'targets': [{'gauge': t['gauge'], 'md_ft': t['md_ft'],
                     'distance_ft': t['distance_ft'], 'n_samples':
                         int(t['data'].size), 'obs_peak_psi': obs_peak[k]}
                    for k, t in enumerate(tgts)],
    }
    RES['r1_prior'] = {
        'D_ft2_s': 500.0,
        'pooled_rmse_psi': {'4': 114.261, '2': 113.832, '1': 113.616,
                            '0.5': 113.509, '0.25': 113.455},
        'quoted_dt1_error_psi': 0.161, 'quoted_dt1_error_pct': 0.142,
        'note': ('R1 measured the CHANGE IN THE MODEL-DATA MISFIT between '
                 'dt = 1 s and dt = 0.25 s, not the solution error. Both '
                 'conventions are reported here so the two are comparable.'),
    }
    with open(OUT['json'], 'w') as fh:
        json.dump(RES, fh, indent=2, default=str)

    wall = time.time() - t_wall
    log(f"\n## total wall {wall:.1f} s")

    # ---- README -----------------------------------------------------------
    _readme_text = _readme(cfg, RES, ADAPT, dt_ref, dt_ref2, seq, profiles,
                           case_meta, OUT, wall, t_stall, akw)
    with open(OUT['readme'], 'w') as fh:
        fh.write(_readme_text)
    log(f"## wrote {OUT['readme']}")
    if OUT['readme_canonical'] != OUT['readme']:
        with open(OUT['readme_canonical'], 'w') as fh:
            fh.write(_readme_text)
        log(f"## wrote {OUT['readme_canonical']} (byte-identical copy of "
            f"{os.path.basename(OUT['readme'])}; the canonical name always "
            f"carries the current pass"
            + (f", and the bytes it held are preserved as "
               f"{', '.join(_canon_twin)}" if _canon_twin else '')
            + ")")
    if superseded:
        log("## superseded files already in this directory, declared as "
            "prior_run_output inputs: "
            + ", ".join(os.path.basename(f) for f in superseded))
    # The log is a DECLARED OUTPUT, so it must be complete and closed before
    # write_manifest hashes it. A single further log line after that point makes
    # the manifest report its own product as drifted.
    log("## log closed here; the manifest is written next and its path is "
        "printed to stdout")
    log.close()

    # -----------------------------------------------------------------------
    # 7. manifest
    # -----------------------------------------------------------------------
    drv = rm.driver_record(
        kind='gauge_series',
        baseline_removal=cfg['source']['baseline_removal'],
        value_units='delta_psi',
        series_path=cfg['data']['gauge_series_template'].format(
            n=S['src_gauge']),
        gauge_number=S['src_gauge'], gauge_md_ft=S['src_md'],
        taxis=src.taxis_s, values=src.delta_psi,
        time_start=cfg['window']['time_start'],
        time_end=cfg['window']['time_end'])
    drv_abs = rm.driver_record(
        kind='gauge_series', baseline_removal='none_absolute_psi',
        value_units='psi',
        series_path=cfg['data']['gauge_series_template'].format(
            n=S['src_gauge']),
        gauge_number=S['src_gauge'], gauge_md_ft=S['src_md'],
        taxis=src.taxis_s, values=src.raw_psi,
        time_start=cfg['window']['time_start'],
        time_end=cfg['window']['time_end'])
    srcp = rm.source_protocol(
        application='dirichlet_node',
        solver_class=cfg['solver']['class'],
        placement_rule=cfg['source']['selection_rule'],
        sources=[[rm.source_record(x, md_requested_ft=S['src_md'],
                                   mesh_idx=S['source_idx'], driver=drv,
                                   label=f"g{S['src_gauge']} delta-psi",
                                   index_in_source_list=0)],
                 [rm.source_record(x, md_requested_ft=S['src_md'],
                                   mesh_idx=S['source_idx'], driver=drv_abs,
                                   label=f"g{S['src_gauge']} absolute-psi",
                                   index_in_source_list=0)]],
        phase_labels=['delta-pressure protocol (fixed-dt sweep and the '
                      'adaptive arm that live-locks)',
                      'absolute-pressure protocol (adaptive arm that runs)'],
        targets=[{'gauge': t['gauge'], 'md_ft': t['md_ft'],
                  'distance_ft': t['distance_ft'], 'mesh_idx': t['idx']}
                 for t in tgts],
        time_level=cfg['source']['source_time_level'],
        phase_chaining='none: the two protocols are independent runs of the '
                       'same physical problem, related by linearity '
                       '(P_abs - P_abs(0) = P_delta), not chained in time',
        boundary_conditions={'lbc': cfg['solver']['lbc'],
                             'rbc': cfg['solver']['rbc']})

    times = []
    for ck in profiles:
        for dt in dt_all:
            n = SOL[(ck, dt)]['n_steps']
            tax = float(dt) * np.arange(n + 1, dtype=float)
            assert rm.sha256_array(tax) == SOL[(ck, dt)]['taxis_sha256'], (ck, dt)
            times.append(rm.time_record(
                tax, mode='fixed', theta=theta, t_total_requested_s=t_total,
                dt_requested_s=dt,
                source_time_level=cfg['source']['source_time_level'],
                theta_startup_steps=0,
                label=(f"{ck}, fixed dt = {dt:g} s"
                       + (' [REFERENCE]' if dt == dt_ref else '')
                       + (' [reference check]' if dt == dt_ref2 else ''))))
    for ck in profiles:
        tr = abs_traces[ck]
        a = ADAPT['absolute_protocol'][ck]['trace']
        times.append(rm.time_record(
            tr['t'], mode='adaptive', theta=theta,
            t_total_requested_s=t_total, dt_init_s=a['dt_init_s'],
            tol=a['tol'], controller_tol=a['controller_tol'],
            max_dt_s=a['max_dt_s'], min_dt_s=a['min_dt_s'],
            safety_factor=a['safety_factor'], order_p=a['order_p'],
            n_steps_rejected=a['n_rejected'],
            source_time_level=cfg['source']['source_time_level'],
            zero_field_policy=a['zero_field_policy'],
            flip_margin=a['flip_margin'],
            label=f"{ck}, manuscript adaptive settings, absolute-pressure "
                  f"protocol"))
    for ck in profiles:
        o = ADAPT['substep_datum_counterfactual']['cases'][ck]['own_time']['trace']
        times.append(rm.time_record(
            substep_traces[ck]['t'], mode='adaptive', theta=theta,
            t_total_requested_s=t_total, dt_init_s=o['dt_init_s'],
            tol=o['tol'], controller_tol=o['controller_tol'],
            max_dt_s=o['max_dt_s'], min_dt_s=o['min_dt_s'],
            safety_factor=o['safety_factor'], order_p=o['order_p'],
            n_steps_rejected=o['n_rejected'],
            source_time_level=cfg['source']['source_time_level'],
            zero_field_policy=o['zero_field_policy'],
            flip_margin=o['flip_margin'],
            label=f"{ck}, absolute-pressure protocol, SUB-STEP DATUM "
                  f"COUNTERFACTUAL (each estimator sub-step reads the datum at "
                  f"its own left endpoint instead of fibeRIS's frozen s(t^n); "
                  f"tol, bounds and accepted-state formula unchanged) -- NOT a "
                  f"fibeRIS reproduction and not the reported trace"))
    for (ck, md) in sorted(delta_traces, key=lambda k: (k[0], -k[1])):
        r = [q for q in ADAPT['min_dt_counterfactual']['cases'][ck]
             if q['min_dt_s'] == md][0]
        times.append(rm.time_record(
            delta_traces[(ck, md)]['t'], mode='adaptive', theta=theta,
            t_total_requested_s=t_total, dt_init_s=r['dt_init_s'],
            tol=r['tol'], controller_tol=r['controller_tol'],
            max_dt_s=r['max_dt_s'], min_dt_s=r['min_dt_s'],
            safety_factor=r['safety_factor'], order_p=r['order_p'],
            n_steps_rejected=r['n_rejected'],
            source_time_level=cfg['source']['source_time_level'],
            zero_field_policy=r['zero_field_policy'],
            flip_margin=r['flip_margin'],
            label=f"{ck}, delta-pressure protocol, min_dt COUNTERFACTUAL "
                  f"{md:g} s (manuscript settings otherwise unchanged; the "
                  f"published min_dt = {akw['min_dt']:g} s live-locks)"))

    num = rm.numerics(
        time=times,
        mesh=rm.mesh_record(x, dx_requested_ft=dx,
                            window_md_ft=(cfg['window']['md_min_ft'],
                                          cfg['window']['md_max_ft']),
                            pad_low_ft=float(cfg['mesh']
                                             ['domain_pad_low_md_ft']),
                            pad_high_ft=float(cfg['mesh']
                                              ['domain_pad_high_md_ft']),
                            refinement=cfg['mesh']['refinement']),
        interface_avg=iavg,
        boundary={'lbc': cfg['solver']['lbc'], 'rbc': cfg['solver']['rbc'],
                  'pml_thickness': 0.0, 'sigma_max': 0.0},
        diffusivity={
            'cases': {ck: {'family': case_meta[ck]['family'],
                           'label': case_meta[ck]['label'],
                           'D_min_ft2_s': case_meta[ck]['D_min_ft2_s'],
                           'D_max_ft2_s': case_meta[ck]['D_max_ft2_s'],
                           'D_sha256': case_meta[ck]['D_sha256'],
                           'params_log10': case_meta[ck].get('params_log10')}
                      for ck in profiles},
            'note': 'three fixed profiles; nothing is fitted in this study'},
        barriers=rm.NONE_DECLARED, leakage=rm.NONE_DECLARED,
        kernel={'name': 'rev2_core.solve_forward / solve_forward_adaptive',
                'banded': True,
                'equivalence_reference':
                    'bitwise identical to r1_calibration_core.solve_forward at '
                    'theta=1/harmonic/lambda=0 (A4 self-test T1); that kernel is '
                    'proven bit-equivalent to fibeRIS PDS1D_SingleSource',
                'adaptive_reference':
                    'solve_forward_adaptive reproduces fibeRIS pds.py:287-336 + '
                    'tso.py:18-49, including the frozen level-n datum inside the '
                    'sub-steps and the un-extrapolated accepted state',
                'adaptive_counterfactual':
                    'section 5e also runs b1_dt._adaptive_substep, a theta=1 '
                    'copy of that controller whose only difference is the datum '
                    'given to the estimator sub-steps. On substep_datum=frozen '
                    'it is asserted BITWISE identical to '
                    'rev2_core.solve_forward_adaptive (max|dt| = max|dP| = 0); '
                    'the substep_datum=own_time rows are the counterfactual and '
                    'are NOT a fibeRIS reproduction'},
        rng=rm.NONE_DECLARED,
        parallel={'processes': nproc,
                  'backend': 'multiprocessing.Pool over (case, dt) solves'},
        amplification={ck: rc.amplification_factor(x, profiles[ck], 1.0, theta,
                                                   interface_avg=iavg)
                       for ck in profiles})

    inputs = [(cfg['data']['gauge_md_npz'], 'geometry', 'gauge_md_npz'),
              (cfg['data']['frac_hit_stage1_npz'], 'geometry',
               'frac_hit_stage1')]
    inputs += [(cfg['data']['gauge_series_template'].format(n=n),
                'gauge_series', f'gauge{n}')
               for n in sorted(S['gauge_window'].series)]
    inputs += [(f, 'prior_run_output', 'superseded:' + os.path.basename(f))
               for f in superseded]
    # The independent re-implementations that reproduced the v2 defect (v3) and
    # the three v4 defects this pass acts on (challenge/referee/). They are
    # hashed so the amendments' provenance is checkable; nothing in there is
    # imported and no number in this manifest comes from it -- every number
    # comes from rev2_core, or from section 5e's local controller, through this
    # script. os.walk, so the referee subdirectory is covered too.
    chal = os.path.join(outdir, 'challenge')
    if os.path.isdir(chal):
        for dirpath, _dn, fns in sorted(os.walk(chal)):
            inputs += [(os.path.join(dirpath, f), 'other',
                        'reviewer_challenge:'
                        + os.path.relpath(os.path.join(dirpath, f), chal))
                       for f in sorted(fns)]
    # The barrier addendum (its own manifested run); hashed as a prior output so
    # this manifest records the state of the whole task directory.
    add = os.path.join(outdir, 'addendum_barrier')
    if os.path.isdir(add):
        inputs += [(os.path.join(add, f), 'prior_run_output',
                    'addendum_barrier:' + f)
                   for f in sorted(os.listdir(add))
                   if os.path.isfile(os.path.join(add, f))]

    products = [
        rm.output_decl(OUT['fig1'], role='figure_png', dpi=dpi,
                       note='convergence curves: pooled RMS, worst-gauge max, '
                            'per-gauge fraction of peak'),
        rm.output_decl(OUT['fig2'], role='figure_png', dpi=dpi,
                       note='error traces in time at dt = 1 s'),
        rm.output_decl(OUT['fig3'], role='figure_png', dpi=dpi,
                       note='adaptive scheme: realised dt, estimator vs tol '
                            '(both the fibeRIS frozen-sub-step estimator and '
                            'the own-time counterfactual), and the '
                            'delta-protocol stall'),
        rm.output_decl(OUT['csv_gauge'], role='csv',
                       note='per-gauge discretisation error, every case and dt'),
        rm.output_decl(OUT['csv_pooled'], role='csv',
                       note='pooled error and misfit change, every case and dt'),
        rm.output_decl(OUT['csv_adaptive'], role='csv',
                       note='one row per adaptive run: both protocols, the '
                            'min_dt counterfactual and the v5 sub-step-datum '
                            'counterfactual, with the realised step statistics '
                            'and the error against the fine reference'),
        rm.output_decl(OUT['json'], role='json',
                       note='every number produced by the run'),
        rm.output_decl(OUT['trace'], role='json',
                       note='adaptive traces, stall scan, dt_init robustness, '
                            'sub-step-datum counterfactual'),
        rm.output_decl(OUT['npz'], role='arrays_npz',
                       note='error fields on the dense grid, dt traces '
                            '(including the own-time sub-step counterfactual), '
                            'stall scans, observed series'),
        rm.output_decl(OUT['log'], role='log', note='full run log'),
        rm.output_decl(OUT['readme'], role='report_md',
                       note='task README, generated by the run'),
        rm.output_decl(OUT['readme_canonical'], role='report_md',
                       note='canonical task README (house rule 7): a '
                            'byte-identical copy of the versioned README of '
                            'this pass, so the name a reader opens first is '
                            'never a superseded pass (AMENDMENT v6)'),
    ]

    rm.write_manifest(
        OUT['manifest'], study_id=cfg['study_id'], task_id='B1',
        config=cfg, config_path=cfg_path, inputs=inputs, source=srcp,
        numerics=num, outputs=products, started_utc=started,
        run_label='B1 time-step convergence at the working point',
        require_modules=('rev2_core', 'rev2_data', 'rev2_manifest',
                         'r1_calibration_core'),
        extra_code_files=(os.path.abspath(__file__),),
        allow_undeclared_outputs=bool(superseded),
        results={
            'headline': headline,
            'reference': {ck: RES['cases'][ck]['reference'] for ck in profiles},
            'order_of_convergence': {ck: RES['cases'][ck]['order']
                                     for ck in profiles},
            'r1_convention_misfit_change': {ck: RES['cases'][ck]
                                            ['r1_convention']
                                            for ck in profiles},
            'adaptive_delta_protocol_completed': {
                ck: ADAPT['delta_protocol'][ck]['completed'] for ck in profiles},
            'adaptive_delta_stall_is_a_min_dt_floor_artifact': {
                ck: {'err_at_published_min_dt':
                     ADAPT['stall_scan']['cases'][ck]['err_at_min_dt'],
                     'estimate_crosses_tol_at_dt_s':
                     ADAPT['stall_scan']['cases'][ck]
                     ['dt_that_would_meet_tol_s'],
                     'published_min_dt_too_coarse_by':
                     ADAPT['stall_scan']['cases'][ck]
                     ['min_dt_floor_shortfall_factor'],
                     'estimator_slope_dt_0p25_to_4s':
                     ADAPT['stall_scan']['cases'][ck]
                     ['slope_fit_coarse_band']['slope'],
                     'estimator_slope_finest_decade':
                     ADAPT['stall_scan']['cases'][ck]
                     ['slope_fit_finest_decade']['slope']}
                for ck in profiles},
            'adaptive_delta_min_dt_counterfactual': {
                ck: [{'min_dt_s': r['min_dt_s'],
                      'completed': r['completed'],
                      'n_accepted': r.get('n_accepted'),
                      'n_rejected': r.get('n_rejected'),
                      'dt_min_s': r.get('dt_min_s'),
                      'dt_median_s': r.get('dt_median_s'),
                      'frac_at_max_dt': r.get('frac_at_max_dt'),
                      'floor_is_active': r.get('floor_is_active'),
                      'pooled_rms_error_psi':
                          (r['error_vs_fine_reference']
                           ['pooled_sample_rms_psi'] if r['completed']
                           else None)}
                     for r in ADAPT['min_dt_counterfactual']['cases'][ck]]
                for ck in profiles},
            'adaptive_delta_zero_field_policy_probe':
                ADAPT['zero_field_policy_probe'],
            'adaptive_absolute_protocol': {
                ck: {'n_attempts': ADAPT['absolute_protocol'][ck]['trace']
                     ['n_attempts'],
                     'n_rejected': ADAPT['absolute_protocol'][ck]['trace']
                     ['n_rejected'],
                     'frac_at_max_dt': ADAPT['absolute_protocol'][ck]['trace']
                     ['frac_at_max_dt'],
                     'err_max': ADAPT['absolute_protocol'][ck]['trace']
                     ['err_max'],
                     'pooled_rms_error_psi': ADAPT['absolute_protocol'][ck]
                     ['error_vs_fine_reference']['pooled_sample_rms_psi'],
                     'worst_gauge_max_abs_psi': ADAPT['absolute_protocol'][ck]
                     ['error_vs_fine_reference']['worst_gauge_max_abs_psi'],
                     'worst_gauge_max_abs_pct_of_peak':
                         100 * ADAPT['absolute_protocol'][ck]
                         ['error_vs_fine_reference']
                         ['worst_gauge_max_abs_frac_of_sim_peak']}
                for ck in profiles},
            'wall_seconds': wall,
        },
        notes=list(cfg.get('notes', [])) + [
            'The manuscript adaptive settings DO NOT RUN on the delta-pressure '
            'protocol at this working point: the controller live-locks at '
            f't = {t_stall:g} s with dt pinned at min_dt = {akw["min_dt"]:g} s '
            'and every attempt rejected. rev2_core raises RuntimeError there; '
            'fibeRIS has no such guard and would hang.',
            'AMENDMENT v3, retraction. The v2 manifest and README explained '
            'that live-lock as intrinsic ("shrinking dt does not shrink the '
            'estimate, because the Dirichlet assignment at the source node is '
            'independent of dt"), and called it measured. It was inferred, and '
            "v2's own stall scan already refuted it. Measured here: the "
            'estimator is flat only for dt >~ 0.1 s (slope '
            f"{ADAPT['stall_scan']['cases']['D1150']['slope_fit_coarse_band']['slope']:+.3f} "
            'over 0.25-4 s), falls at essentially second order below ~1e-3 s '
            'and crosses tol at '
            + ' / '.join(
                f"{ADAPT['stall_scan']['cases'][ck]['dt_that_would_meet_tol_s']:.3e}"
                for ck in profiles)
            + ' s for D1150 / D1223 / two_zone. The published floor min_dt = '
            f'{akw["min_dt"]:g} s is '
            + ' / '.join(
                f"{ADAPT['stall_scan']['cases'][ck]['min_dt_floor_shortfall_factor']:.2f}x"
                for ck in profiles)
            + ' above those crossings, so the live-lock is a FLOOR artifact: '
            'the published settings are unusable on this formulation, the '
            'formulation is not unusable.',
            'AMENDMENT v3, counterfactual (this is the quotable new result). '
            'With min_dt lowered and nothing else changed, the same controller '
            'completes the delta protocol and the estimator BINDS - the '
            'opposite of the absolute protocol, where it never binds: '
            + '; '.join(
                f"{ck} min_dt={r['min_dt_s']:g} -> "
                + ('live-locked' if not r['completed'] else
                   f"{r['n_accepted']} accepted / {r['n_rejected']} rejected, "
                   f"median dt {r['dt_median_s']:.3f} s, "
                   f"{100 * r['frac_at_max_dt']:.2f}% at max_dt, pooled RMS "
                   f"{r['error_vs_fine_reference']['pooled_sample_rms_psi']:.4f}"
                   f" psi")
                for ck in profiles
                for r in ADAPT['min_dt_counterfactual']['cases'][ck])
            + '.',
            'AMENDMENT v3, what min_dt does NOT fix. Stock fibeRIS still cannot '
            'run optimizer=True on the delta formulation, but for a separate '
            'defect: with u == 0 and s(0) == 0 it evaluates 0/0 = nan on the '
            'first step and pins dt at min_dt forever. Measured at min_dt = '
            + ' / '.join(f"{r['min_dt_s']:g}" for r in
                         ADAPT['zero_field_policy_probe']
                         if r['zero_field_policy'] == 'fiberis')
            + ' s: every one live-locks at t = 0. Do not conflate that defect '
              'with the t = 2 s floor artifact above.',
            'On the absolute-pressure protocol the same settings run but the '
            'estimator never binds, so dt pins to max_dt = 30 s. The "adaptive" '
            'run is a fixed dt = 30 s run.',
            'Errors are quoted against a fixed-dt reference at '
            f'dt = {dt_ref:g} s, whose own residual is bounded by Richardson '
            f'extrapolation from dt = {dt_ref2:g} s.',
        ] + ([
            'This pass supersedes the files listed as prior_run_output inputs. '
            'The earlier pass produced identical physics but declared its run '
            'log as an output and then appended two more lines to it after the '
            'manifest had hashed it, so that manifest reports its own log as '
            'drifted. House rule 2 forbids deleting it, so it is retained, '
            'hashed here, and superseded. Quote this manifest, not that one.'
        ] if superseded else []))
    print(f"## wrote {OUT['manifest']}")
    return 0


# ---------------------------------------------------------------------------
# README text
# ---------------------------------------------------------------------------

def _readme(cfg, RES, ADAPT, dt_ref, dt_ref2, seq, profiles, case_meta, OUT,
            wall, t_stall, akw):
    L = []
    A = L.append
    A('# B1 - time-step convergence at the working point\n')
    A(f"Round `rev2_20260901`. Run by `scripts/manuscript_well_leakage/rev2/"
      f"b1_dt.py --config configs/rev2/b1_dt.json`"
      + (f" --version {cfg['outputs']['version']}"
         if cfg['outputs'].get('version', 'v1') != 'v1' else '')
      + f". Wall {wall:.0f} s.\n")
    ver = cfg['outputs'].get('version', 'v1')
    if ver == 'v2':
        A('> **Which files to quote.** The `_v2` products and `manifest_v2.json` '
          'are authoritative. The `_v1` products are a superseded first pass '
          'with identical physics; that pass declared its run log as an output '
          'and then wrote two further lines to it after the manifest had hashed '
          'it, so `manifest.json` reports its own log as drifted and no longer '
          'verifies. House rule 2 forbids deleting it, so it is retained and is '
          'hashed as a `prior_run_output` input of this pass.\n')
    elif ver != 'v1':
        A(f'> **Which files to quote.** This file (`README.md`) and '
          f'`README_{ver}.md` are byte-identical, and the `_{ver}` products '
          f'with `manifest_{ver}.json` are authoritative. Every earlier `_v*` '
          'set is superseded and retained under house rule 2; they are hashed '
          'as `prior_run_output` inputs of this pass. The hand-written '
          'companions are `README_v4_ADDENDUM.md` (v4/v5) and '
          '`amend_v6/README.md` (v6).\n'
          '>\n'
          '> **What changed in v6 (amendment).** No number moves; this pass is '
          'the v5 study re-run, and the leaf-by-leaf diff against '
          '`b1_results_v5.json` is `amend_v6/v5_vs_v6_leafdiff.json`. The '
          'defect was in the HAND-OFF, not in the deliverable: `README.md` was '
          'written by the v1 pass and then frozen, because this script mapped '
          'only v1 to the canonical name, so the corrections v3 and v5 made '
          'never reached the file a reader opens first. Under that name '
          '`README.md` still stated (i) "at most 0.34 % of any gauge\'s peak" '
          'unscoped, (ii) the withdrawn v2 mechanism "shrinking dt does not '
          'shrink the estimate ... measured rather than inferred", and (iii) '
          'the delta-protocol live-lock without the min_dt finding that makes '
          'it a settings failure. A referee handed that summary reproduced two '
          'false statements from it. Every pass now writes the canonical name '
          'as well; the v1 text is preserved verbatim as '
          '`README_v1_superseded.md`, where `manifest.json`\'s recorded '
          'sha256 for `README.md` still verifies. Two presentation defects '
          'found in the same re-read are fixed here: bounds are rounded '
          'OUTWARD (v5 printed the all-case fraction-of-peak bound as 0.51 % '
          'from a measured 0.5133 %), and the observed order is reported with '
          'the finite reference\'s own step removed, for all three cases '
          'rather than in prose for one. The independent recomputation that '
          'reproduced the referee\'s two findings from scratch - own loader, '
          'own mesh, own assembly, own solver, no `rev2_*` and no `b1_dt` - is '
          '`amend_v6/b1_v6_indep_check.py` -> `amend_v6/indep_v6.json`; it '
          'agrees with every error norm and every observed order this pass '
          'publishes to better than 1e-6 relative (the measured worst case is '
          'recorded in that file, under `agreement_with_v6`).\n'
          '>\n'
          '> **What changed in v5 (amendment).** No number moves. Three claims '
          'are weakened to what the data support, after an independent referee '
          'reproduced all three: (a) the observed-order interval 0.986-1.009 is '
          'the POOLED-RMS range only - the worst-gauge maximum runs 0.950-1.011 '
          '- so the phrase "both metrics" is withdrawn; (b) "at most 0.34 % of '
          "any gauge's peak\" covers the two uniform cases only - two_zone, the "
          'winning model, costs 0.513 % simulated / 0.516 % observed - so every '
          'quotable is now scoped; (c) the absolute-protocol headline (0 '
          'rejections, 97.7 % at max_dt) also depends on fibeRIS freezing the '
          "level-n datum inside the estimator's sub-steps, not on the ~8300 psi "
          'normalisation alone - section 5e measures that counterfactual. The '
          'fixed-dt sweep, the reference, the delta-protocol analysis and the '
          'barrier addendum are untouched.\n'
          '>\n'
          '> **What changed in v3 (amendment).** The physics, the fixed-dt '
          'sweep and the absolute-protocol result are unchanged and reproduce '
          'v2 exactly. One CLAIM is withdrawn: v2 explained the '
          'delta-protocol live-lock by "shrinking dt does not shrink the '
          'estimate ... measured rather than inferred". It was inferred, and '
          "v2's own `b1_adaptive_trace_v2.json` contradicted it "
          '(`any_dt_accepted: true`, tol crossing at 3.449e-05 s, floor '
          'shortfall 2.90x). The live-lock is a `min_dt` FLOOR artifact. v3 '
          'measures the dt-dependence of the estimator and runs the '
          'counterfactual with the floor lowered; the new products are '
          "`b1_adaptive_v3.csv` and the fourth panel of `fig03`. Nothing "
          "quoted from v2 outside the 'Delta-pressure protocol' section "
          'changes.\n')
    A('## What was run\n')
    A('The R1 bed verbatim: S-well stage 1, comparison window MD 15000-16750, '
      'domain MD 10000-16750 (5000 ft low-end pad), dx = 1 ft (nx = 6751), '
      'gauge 1 (MD 16645) imposed as a Dirichlet node with the datum read at '
      'time level *n*, targets g2-g7, 1254.091 s of record, delta-pressure with '
      'a zero initial condition, backward Euler (theta = 1), harmonic face '
      'diffusivity, no barrier, no leakage sink.\n')
    A('Three diffusivity cases:\n')
    for ck in profiles:
        m = case_meta[ck]
        A(f"- `{ck}` - {m['label']}; D in "
          f"[{m['D_min_ft2_s']:.1f}, {m['D_max_ft2_s']:.1f}] ft^2/s, so "
          f"r = D_max*dt/dx^2 = {m['r_at_dt1']:.0f} at dt = 1 s.")
    A('')
    A(f"Time steps 0.25 / 0.5 / 1 / 2 / 4 s (the required grid) plus 8 / 16 / "
      f"30 s, because 30 s is the step the manuscript's adaptive scheme "
      f"actually realises.\n")
    A('## The reference, stated explicitly\n')
    A(f"Errors are quoted against a **fixed-step backward-Euler solution at "
      f"dt_ref = {dt_ref:g} s = 1/256 s** on the identical mesh, domain, source "
      f"series and time scheme - 256x finer than the production step and 64x "
      f"finer than the finest step under test. Every dt under test is an exact "
      f"INTEGER multiple of dt_ref (a power of two for 0.25-16 s; 30 s is "
      f"{30.0 / dt_ref:.0f} dt_ref), so the reference's time nodes contain the "
      f"coarse runs' nodes exactly and the reference is never extrapolated to a "
      f"time it did not compute. Each solution is evaluated where the misfit is "
      f"actually formed: interpolated onto each target gauge's own data time "
      f"axis, the same linear interpolation for both, so what is measured is "
      f"the interpolant of the difference.\n")
    A(f"The reference's own residual error is bounded by Richardson "
      f"extrapolation from a second solution at dt = {dt_ref2:g} s = 1/512 s "
      f"(the scheme is first order, so 2*P(h/2) - P(h) removes the leading "
      f"term):\n")
    A('| case | reference residual, pooled RMS (psi) | worst-gauge max (psi) |')
    A('|---|---|---|')
    for ck in profiles:
        r = RES['cases'][ck]['reference']
        A(f"| {ck} | {r['ref_residual_vs_richardson_pooled_rms_psi']:.3e} | "
          f"{r['ref_residual_worst_gauge_max_abs_psi']:.3e} |")
    A('')
    A('## Headline: the error at the production step dt = 1 s\n')
    A('| case | pooled sample RMS (psi) | gauge-mean RMS (psi) | worst gauge | '
      'worst-gauge max (psi) | as % of that gauge SIMULATED peak | misfit '
      'change, pooled (psi) |')
    A('|---|---|---|---|---|---|---|')
    for ck in profiles:
        h = RES['headline'][ck]
        A(f"| {ck} | {h['dt_1s_pooled_sample_rms_psi']:.4f} | "
          f"{h['dt_1s_gauge_mean_rms_psi']:.4f} | {h['dt_1s_worst_gauge']} | "
          f"{h['dt_1s_worst_gauge_max_abs_psi']:.4f} | "
          f"{h['dt_1s_worst_gauge_max_abs_pct_of_peak']:.3f}% | "
          f"{h['dt_1s_misfit_pooled_rmse_change_psi']:+.4f} |")
    A('')
    A('## Per gauge at the production step dt = 1 s\n')
    A('| case | gauge | distance (ft) | simulated peak (psi) | max abs error '
      '(psi) | % of peak | RMS error (psi) | % of peak |')
    A('|---|---|---|---|---|---|---|---|')
    for ck in profiles:
        for p_ in RES['cases'][ck]['rows']['1']['per_gauge']:
            A(f"| {ck} | g{p_['gauge']} | {p_['distance_ft']:.0f} | "
              f"{p_['sim_peak_psi']:.2f} | {p_['max_abs_psi']:.4f} | "
              f"{100 * p_['max_abs_frac_of_sim_peak']:.3f}% | "
              f"{p_['rms_psi']:.4f} | "
              f"{100 * p_['rms_frac_of_sim_peak']:.3f}% |")
    A('')
    A("## The R1 convention, reproduced at the working point\n")
    A('R1 quoted **0.161 psi (0.142%)** at D = 500 ft^2/s, and that number is '
      'the change in the MODEL-DATA MISFIT between dt = 1 s and dt = 0.25 s, '
      'not a solution error. The misfit is a difference of two large numbers, so '
      'it hides most of the discretisation error; both conventions are given '
      'here so the comparison is like for like.\n')
    A('| case | pooled misfit at dt = 1 s (psi) | at dt = 0.25 s (psi) | change '
      '(psi) | change (%) | solution error at dt = 1 s, pooled RMS (psi) |')
    A('|---|---|---|---|---|---|')
    for ck in profiles:
        r = RES['cases'][ck]['r1_convention']
        kf = f"{r['finest_dt_of_required_grid_s']:g}"
        A(f"| {ck} | {r['pooled_rmse_psi_by_dt']['1']:.3f} | "
          f"{r['pooled_rmse_psi_by_dt'][kf]:.3f} | "
          f"{r['dt1_minus_dtfine_pooled_psi']:.4f} | "
          f"{r['dt1_minus_dtfine_pooled_pct']:.4f}% | "
          f"{RES['headline'][ck]['dt_1s_pooled_sample_rms_psi']:.4f} |")
    A('')
    A('## Full sweep\n')
    A('| case | dt (s) | r = D_max dt/dx^2 | pooled RMS (psi) | gauge-mean RMS '
      '(psi) | worst-gauge max (psi) | worst-gauge max, % of peak | '
      'misfit change, pooled (psi) |')
    A('|---|---|---|---|---|---|---|---|')
    for ck in profiles:
        for dt in seq:
            m = RES['cases'][ck]['rows'][f"{dt:g}"]
            A(f"| {ck} | {dt:g} | {m['r_number']:.0f} | "
              f"{m['pooled_sample_rms_psi']:.4f} | {m['gauge_mean_rms_psi']:.4f} "
              f"| {m['worst_gauge_max_abs_psi']:.4f} | "
              f"{100 * m['worst_gauge_max_abs_frac_of_sim_peak']:.3f}% | "
              f"{m['misfit']['pooled_rmse_change_psi']:+.4f} |")
    A('')
    A('## Observed order of convergence\n')
    A('| case | dt pair (s) | p from pooled RMS | p from worst-gauge max | '
      'p corrected, pooled | p corrected, worst-gauge max |')
    A('|---|---|---|---|---|---|')
    _allp, _allw = [], []
    _corp, _corw = [], []
    for ck in profiles:
        for o in RES['cases'][ck]['order']:
            A(f"| {ck} | {o['dt_fine_s']:g} -> {o['dt_coarse_s']:g} | "
              f"{o['p_pooled_rms']:.4f} | {o['p_worst_max_abs']:.4f} | "
              f"{o['p_pooled_rms_ref_corrected']:.4f} | "
              f"{o['p_worst_max_abs_ref_corrected']:.4f} |")
            _allp.append((o['p_pooled_rms'], ck, o))
            _allw.append((o['p_worst_max_abs'], ck, o))
            _corp.append((o['p_pooled_rms_ref_corrected'], ck, o))
            _corw.append((o['p_worst_max_abs_ref_corrected'], ck, o))
    A('')
    # AMENDMENT v5. The v3/v4 text said "0.986-1.009 ... both the pooled RMS and
    # the worst-gauge maximum". 0.986-1.009 is the POOLED range only. Both are
    # now computed from the table above rather than quoted from memory.
    _k0 = lambda t: t[0]                                            # noqa: E731
    _pl, _ph = min(_allp, key=_k0), max(_allp, key=_k0)
    _wl, _wh = min(_allw, key=_k0), max(_allw, key=_k0)
    _out = [t for t in _allw if t[0] < _pl[0] or t[0] > _ph[0]]
    _pairs = RES['cases']['D1150']['order']
    A(f"Over every doubling from {min(o['dt_fine_s'] for o in _pairs):g} to "
      f"{max(o['dt_coarse_s'] for o in _pairs):g} s and all three cases the "
      f"order is one, but **the two metrics do not share an interval**: the "
      f"pooled RMS gives **{_pl[0]:.4f}-{_ph[0]:.4f}**, the worst-gauge maximum "
      f"gives **{_wl[0]:.4f}-{_wh[0]:.4f}**. The low end of the second is "
      f"{_wl[1]} {_wl[2]['dt_fine_s']:g} -> {_wl[2]['dt_coarse_s']:g} s, the "
      f"coarsest pair, where the peak error saturates. "
      f"{len(_out)} of the {len(_allw)} worst-gauge values fall outside the "
      f"pooled interval: "
      + ", ".join(f"{t[1]} {t[2]['dt_fine_s']:g}->{t[2]['dt_coarse_s']:g} "
                  f"{t[0]:.4f}" for t in sorted(_out, key=_k0))
      + ". Quote the pooled range as the pooled range: it is **not** valid for "
        "both metrics, and the v3/v4 text that said it was is withdrawn "
        "(AMENDMENT v5).\n")
    # AMENDMENT v6. The excess above 1 at the fine pairs is the reference's own
    # finite step, not a second-order component. Previously argued in prose in
    # README_v4_ADDENDUM.md for D1150 only; computed here for every case, both
    # metrics, and written to b1_results_*.json.
    _sub = lambda L, hi: [t for t in L if t[2]['dt_coarse_s'] <= hi]  # noqa: E731
    _cp4, _cw4 = _sub(_corp, 4.0), _sub(_corw, 4.0)
    _cpl, _cph = min(_cp4, key=_k0), max(_cp4, key=_k0)
    _cwl, _cwh = min(_cw4, key=_k0), max(_cw4, key=_k0)
    _cpF, _cwF = min(_corp, key=_k0), max(_corw, key=_k0)
    _c05 = RES['cases']['D1150']['order'][0]['ref_correction_factors']
    A(f"The excess above 1 at the finest pairs is not a second-order component: "
      f"it is what a FINITE reference must produce. What is measured is "
      f"E(dt) = C (dt - dt_ref) with dt_ref = {dt_ref:g} s, so exact first "
      f"order predicts p = "
      f"{np.log2((0.5 - dt_ref) / (0.25 - dt_ref)):.4f} for the "
      f"0.25 -> 0.5 pair. The last two columns rescale each error by "
      f"dt/(dt - dt_ref) (factor {_c05[0]:.4f} at dt = 0.25 s, {_c05[1]:.4f} at "
      f"0.5 s) and refit. Over dt = 0.25-4 s, where the correction is the "
      f"relevant one and the peak error has not yet saturated, the corrected "
      f"order is **{_cpl[0]:.4f}-{_cph[0]:.4f}** from the pooled RMS and "
      f"**{_cwl[0]:.4f}-{_cwh[0]:.4f}** from the worst-gauge maximum (low end "
      f"{_cwl[1]} {_cwl[2]['dt_fine_s']:g} -> {_cwl[2]['dt_coarse_s']:g} s), "
      f"over all three cases; including the 4 -> 8 and 8 -> 16 pairs widens it "
      f"to {_cpF[0]:.4f}-{max(_corp, key=_k0)[0]:.4f} and "
      f"{min(_corw, key=_k0)[0]:.4f}-{_cwF[0]:.4f}. First order under either "
      f"metric, as the level-n Dirichlet datum forces; nothing here is second "
      f"order.\n")
    A('## The adaptive scheme\n')
    A(f"Settings as recovered by A3 and as used by the manuscript's two-stage "
      f"figures: tol = {akw['tol']:g}, dt_init = {akw['dt_init']:g} s, "
      f"max_dt = {akw['max_dt']:g} s, min_dt = {akw['min_dt']:g} s, "
      f"safety factor {akw['safety_factor']:g}, controller order p = "
      f"{akw['order_p']:d}, controller_tol = {akw['controller_tol']:g}. The "
      f"error estimate is a relative L2 of the full-step solution against two "
      f"half-steps, over the WHOLE field, not divided by 2^p - 1.\n")
    A('### Delta-pressure protocol (what this working point uses): the '
      'published settings do not run\n')
    for ck in profiles:
        d = ADAPT['delta_protocol'][ck]
        A(f"- `{ck}`: " + ('completed' if d['completed']
                           else '**live-locks** - ' + d['runtime_error']))
    A('')
    A(f"What happens: the delta-pressure datum is exactly 0 at t = 0, so the "
      f"first step is accepted trivially and leaves the field identically zero. "
      f"At t = {t_stall:g} s the controller then imposes a datum of "
      f"{ADAPT['stall_scan']['datum_psi']:.4f} psi on a field whose norm is "
      f"still ~0, and the relative estimate at the floor min_dt = "
      f"{akw['min_dt']:g} s is:\n")
    A('| case | error estimate at min_dt | tol | ratio | dt that meets tol (s) '
      '| min_dt floor is too coarse by |')
    A('|---|---|---|---|---|---|')
    for ck in profiles:
        s_ = ADAPT['stall_scan']['cases'][ck]
        e = s_['err_at_min_dt']
        A(f"| {ck} | {e:.4e} | {akw['tol']:g} | {e / akw['tol']:.1f}x | "
          f"{s_['dt_that_would_meet_tol_s']:.3e} | "
          f"{s_['min_dt_floor_shortfall_factor']:.2f}x |")
    A('')
    sc0 = ADAPT['stall_scan']['cases']['D1150']
    A('**Mechanism (v3 correction).** v2 wrote here that "shrinking dt does not '
      'shrink the estimate, because the Dirichlet assignment at the source node '
      'is independent of dt". That sentence was inferred, not measured, and the '
      "v2 pass's own stall scan refutes it "
      '(`b1_adaptive_trace_v2.json` already recorded `any_dt_accepted: true` '
      'and a crossing step). It is withdrawn. What the scan actually measures '
      'is:\n')
    A('| case | slope over dt = 0.25-4 s | spread of the estimate there | slope '
      'over the finest decade | estimate crosses tol at (s) |')
    A('|---|---|---|---|---|')
    for ck in profiles:
        s_ = ADAPT['stall_scan']['cases'][ck]
        cb, fb = s_['slope_fit_coarse_band'], s_['slope_fit_finest_decade']
        A(f"| {ck} | {cb['slope']:+.3f} | {cb['err_spread_pct']:.1f}% | "
          f"{fb['slope']:+.3f} ({fb['dt_lo_s']:.2e}-{fb['dt_hi_s']:.2e} s) | "
          f"{s_['dt_that_would_meet_tol_s']:.3e} |")
    A('')
    A(f"So the estimate is insensitive to dt only in the band it starts in "
      f"(dt >~ 0.1 s, where it varies by "
      f"{sc0['slope_fit_coarse_band']['err_spread_pct']:.1f}% over a factor of "
      f"16 in dt); below ~1e-3 s it falls at essentially second order and "
      f"crosses tol at {sc0['dt_that_would_meet_tol_s']:.2e} / "
      f"{ADAPT['stall_scan']['cases']['D1223']['dt_that_would_meet_tol_s']:.2e}"
      f" / "
      f"{ADAPT['stall_scan']['cases']['two_zone']['dt_that_would_meet_tol_s']:.2e}"
      f" s. The controller live-locks because the floor min_dt = "
      f"{akw['min_dt']:g} s sits "
      f"{sc0['min_dt_floor_shortfall_factor']:.2f}x / "
      f"{ADAPT['stall_scan']['cases']['D1223']['min_dt_floor_shortfall_factor']:.2f}x"
      f" / "
      f"{ADAPT['stall_scan']['cases']['two_zone']['min_dt_floor_shortfall_factor']:.2f}x"
      f" above that crossing, not because the estimate is irreducible. **The "
      f"published SETTINGS are unusable on this formulation; the formulation "
      f"itself is not.** The stall is not an artifact of dt_init either: "
      + "; ".join(f"dt_init = {r['dt_init_s']:g} s -> "
                  + ('completed' if r['completed'] else 'live-locked')
                  for r in ADAPT['dt_init_robustness']) + '.\n')
    cfc = ADAPT['min_dt_counterfactual']['cases']
    A(f"**The counterfactual, measured.** Lower `min_dt` and change nothing "
      f"else:\n")
    A('| case | min_dt (s) | outcome | attempts | accepted | rejected | '
      'rejections at distinct times | dt min / median / max (s) | % of steps at '
      'max_dt | floor reached? | pooled RMS vs the 1/256 s reference (psi) |')
    A('|---|---|---|---|---|---|---|---|---|---|---|')
    for ck in profiles:
        for r in cfc[ck]:
            tag = ' (published)' if r['is_published_setting'] else ''
            if not r['completed']:
                A(f"| {ck} | {r['min_dt_s']:g}{tag} | **live-locks** | "
                  f"> {r['max_attempts_probe']} | - | - | - | - | - | - | - |")
                continue
            m = r['error_vs_fine_reference']
            A(f"| {ck} | {r['min_dt_s']:g}{tag} | completes | "
              f"{r['n_attempts']} | {r['n_accepted']} | {r['n_rejected']} | "
              f"{r['n_rejection_times']} | {r['dt_min_s']:.3e} / "
              f"{r['dt_median_s']:.3f} / {r['dt_max_s']:g} | "
              f"{100 * r['frac_at_max_dt']:.2f}% | "
              f"{'yes' if r['floor_is_active'] else 'no'} | "
              f"{m['pooled_sample_rms_psi']:.4f} |")
    A('')
    still = [(ck, r['min_dt_s']) for ck in profiles for r in cfc[ck]
             if r['completed'] and r['floor_is_active']]
    if still:
        A('Where "floor reached?" is yes the run completed but still touched '
          'its floor at least once - '
          + ", ".join(f"{ck} at min_dt = {md:g} s" for ck, md in still)
          + ' - so for those the floor is marginal rather than inactive; the '
            'next floor down is the one that is demonstrably not binding.\n')
    ok0 = [r for r in cfc['D1150'] if r['completed']]
    if ok0:
        r0 = ok0[0]
        m0 = r0['error_vs_fine_reference']
        a0 = ADAPT['absolute_protocol']['D1150']['trace']
        same = (len({(r['n_accepted'], r['n_rejected']) for r in ok0}) == 1)
        A(f"At min_dt = {r0['min_dt_s']:g} s the D1150 run completes with "
          f"{r0['n_accepted']} accepted and {r0['n_rejected']} rejected steps, "
          f"dt returns to max_dt = {akw['max_dt']:g} s, and the solution costs "
          f"{m0['pooled_sample_rms_psi']:.4f} psi pooled RMS against the "
          f"1/256 s reference"
          + (f" (against {ADAPT['absolute_protocol']['D1150']['error_vs_fine_reference']['pooled_sample_rms_psi']:.4f} psi "
             f"for the absolute protocol's 30 s pinning)" ) + '. '
          + ("For D1150 lowering the floor further changes nothing - every "
             "completed floor gives the same attempt counts and the same "
             f"dt_min {r0['dt_min_s']:.3e} s, i.e. the floor has gone INACTIVE "
             "and the controller is choosing the step itself. " if same else '')
          + f"That is the substantive finding, and it is the opposite of the "
          f"absolute protocol: there the estimator never binds "
          f"({a0['n_rejected']} rejections, "
          f"{100 * a0['frac_at_max_dt']:.2f}% of steps at max_dt); here it "
          f"binds throughout - only "
          f"{r0['n_rejections_at_first_stall']} of the "
          f"{r0['n_rejected']} rejections belong to the initial descent at "
          f"t = {t_stall:g} s, the rest are spread over "
          f"{r0['n_rejection_times'] - 1} later instants, the median accepted "
          f"step is {r0['dt_median_s']:.3f} s, and the largest estimate among "
          f"accepted steps is {r0['max_err_of_accepted_steps']:.4e} against "
          f"tol = {akw['tol']:g}.\n")
    zp = ADAPT['zero_field_policy_probe']
    fib = [r for r in zp if r['zero_field_policy'] == 'fiberis']
    if fib:
        A(f"**What min_dt does NOT fix.** All of the above is on `rev2_core`'s "
          f"`zero_field_policy='accept'` work-around. Stock fibeRIS hits a "
          f"SECOND, independent live-lock first: with u == 0 and s(0) == 0 it "
          f"evaluates 0/0 = nan, rejects, and pins dt at min_dt forever. That "
          f"one sits at t = 0 and no floor cures it - measured at min_dt = "
          + " / ".join(f"{r['min_dt_s']:g}" for r in fib)
          + " s, every one live-locks at t = "
          + " / ".join(f"{r['stall_t_s']:g}" for r in fib)
          + ". So `optimizer=True` on the delta formulation is unavailable in "
            "fibeRIS as shipped for a reason that is not the step floor; the "
            "step floor is what breaks the corrected controller.\n")
    A('### Absolute-pressure protocol (what `101_fiberis_matching.py` does): '
      'it runs, and it is a fixed 30 s run\n')
    A('| case | attempts | accepted | rejected | dt min-max (s) | % of steps at '
      'max_dt | max error estimate | tol | pooled RMS error vs reference (psi) | '
      'worst gauge max (psi) | % of that peak |')
    A('|---|---|---|---|---|---|---|---|---|---|---|')
    for ck in profiles:
        a = ADAPT['absolute_protocol'][ck]
        t = a['trace']; e = a['error_vs_fine_reference']
        A(f"| {ck} | {t['n_attempts']} | {t['n_accepted']} | {t['n_rejected']} | "
          f"{t['dt_min_s']:g}-{t['dt_max_s']:g} | "
          f"{100 * t['frac_at_max_dt']:.2f}% | {t['err_max']:.3e} | "
          f"{t['tol']:g} | {e['pooled_sample_rms_psi']:.3f} | "
          f"{e['worst_gauge_max_abs_psi']:.3f} | "
          f"{100 * e['worst_gauge_max_abs_frac_of_sim_peak']:.2f}% |")
    A('')
    A('The two protocols are the same physical problem: by linearity '
      'P_abs(t) - P_abs(0) = P_delta(t) for any fixed step. Only the error '
      'ESTIMATOR differs, because it is normalised by the norm of whatever '
      'field it is handed - about 8300 psi of absolute datum in one case and '
      'nothing at all in the other.\n')
    # AMENDMENT v5: the normalisation is necessary but NOT sufficient.
    SS = ADAPT['substep_datum_counterfactual']
    A("#### The 8300 psi normalisation is not the whole mechanism "
      "(AMENDMENT v5)\n")
    A("The row above is a faithful reproduction of fibeRIS, and that includes a "
      "second habit: `matbuilder.py:73` reads `taxis[-1]` and `pds.py:318-325` "
      "appends nothing inside the sub-steps, so **all three solves of the error "
      "estimate see the same datum s(t^n)** - the two half-steps are driven by "
      "a source that is frozen where the full step's is (`rev2_core.py:930-936` "
      "documents this and reproduces it on purpose). A frozen datum removes "
      "most of what the full step and the half-steps could disagree about. "
      "Correcting ONLY that - each sub-step reads the datum at its own left "
      "endpoint, the level-*n* rule applied consistently, with the same tol, "
      "the same bounds, the same protocol and the same accepted-state formula "
      "- gives:\n")
    A('| case | variant | attempts | rejected | % of steps at max_dt | max '
      'error estimate | flip margin | pooled RMS vs reference (psi) | worst '
      'gauge max (psi) | % of that peak |')
    A('|---|---|---|---|---|---|---|---|---|---|')
    for ck in profiles:
        for lab, tag in (('fibeRIS, as reported above', 'frozen'),
                         ('sub-steps at their own times', 'own_time')):
            r = SS['cases'][ck][tag]
            t_, e_ = r['trace'], r['error_vs_fine_reference']
            A(f"| {ck} | {lab} | {t_['n_attempts']} | {t_['n_rejected']} | "
              f"{100 * t_['frac_at_max_dt']:.2f}% | {t_['err_max']:.4e} | "
              f"{t_['flip_margin']:.4f} | "
              f"{e_['pooled_sample_rms_psi']:.4f} | "
              f"{e_['worst_gauge_max_abs_psi']:.3f} | "
              f"{100 * e_['worst_gauge_max_abs_frac_of_sim_peak']:.2f}% |")
    A('')
    _eq = SS['equivalence_check']
    A(f"The two rows of each pair differ in one line of code. The `frozen` rows "
      f"are produced by a LOCAL copy of the controller (`b1_dt._adaptive_substep`) "
      f"that is asserted bitwise identical to `rev2_core.solve_forward_adaptive` "
      f"on this problem - max|dt| = "
      f"{max(v['max_abs_taxis_diff_s'] for v in _eq.values()):g} s and max|dP| = "
      f"{max(v['max_abs_recorded_diff_psi'] for v in _eq.values()):g} psi across "
      f"all three cases - so the difference between the rows is the switch and "
      f"nothing else.\n")
    _r = [SS['cases'][ck]['err_max_ratio'] for ck in profiles]
    _above = [ck for ck in profiles
              if SS['cases'][ck]['estimate_above_tol_when_corrected']]
    A(f"**What this does and does not overturn.** The peak estimate rises "
      f"{min(_r):.1f}-{max(_r):.1f}x and crosses tol in "
      f"{len(_above)} of {len(list(profiles))} cases, so 'zero rejections', "
      f"'97.67% at max_dt' and the flip margin of "
      f"{ADAPT['absolute_protocol']['D1150']['trace']['flip_margin']:.3f} are "
      f"properties of the fibeRIS estimator, not of the absolute-pressure "
      f"normalisation alone: the ~8300 psi datum is a NECESSARY but not a "
      f"SUFFICIENT explanation of why the control never engaged. The "
      f"substantive verdict is unaffected - the corrected controller still "
      f"takes a median step of "
      f"{SS['cases']['D1150']['own_time']['trace']['dt_median_s']:.1f} s, still "
      f"spends "
      f"{100 * SS['cases']['D1150']['own_time']['trace']['frac_at_max_dt']:.1f}% "
      f"of its steps at max_dt, and still costs "
      f"{SS['cases']['D1150']['own_time']['error_vs_fine_reference']['pooled_sample_rms_psi']:.1f}"
      f"-{SS['cases']['two_zone']['own_time']['error_vs_fine_reference']['pooled_sample_rms_psi']:.1f} psi "
      f"pooled, so it is still a 30 s run in all but name. The own-time rows "
      f"are a COUNTERFACTUAL and must never be quoted as what the manuscript "
      f"ran.\n")
    A('## Quotable in the paper\n')
    h = RES['headline']
    a = ADAPT['absolute_protocol']['D1150']
    # AMENDMENT v5: the bound below was quoted unscoped, over the two uniform
    # cases only, while two_zone - the model the round declares the winner -
    # costs 1.5x more. Both are now stated, and the scope is explicit.
    _unif = max(h['D1150']['dt_1s_worst_gauge_max_abs_pct_of_peak'],
                h['D1223']['dt_1s_worst_gauge_max_abs_pct_of_peak'])
    _all = max(h[ck]['dt_1s_worst_gauge_max_abs_pct_of_peak'] for ck in profiles)
    _obs = {ck: max(100 * p['max_abs_psi'] / p['obs_peak_psi']
                    for p in RES['cases'][ck]['rows']['1']['per_gauge'])
            for ck in profiles}
    # AMENDMENT v6: a bound must BOUND. Upper bounds are rounded UP and lower
    # bounds DOWN, so no printed digit is ever smaller than what was measured
    # (v5 printed the all-case bound as 0.51% from a measured 0.5133%).
    _up2 = lambda v: float(np.ceil(v * 100 - 1e-9) / 100)            # noqa: E731
    _unif_obs = max(_obs['D1150'], _obs['D1223'])
    A(f"1. At the working point the fixed step dt = 1 s costs "
      f"{h['D1150']['dt_1s_pooled_sample_rms_psi']:.2f} psi pooled RMS at "
      f"D = 1150 and {h['D1223']['dt_1s_pooled_sample_rms_psi']:.2f} psi at "
      f"D = 1223 - {_unif:.4f}% of any gauge's simulated peak and "
      f"{_unif_obs:.4f}% of any gauge's observed peak, i.e. <= "
      f"{_up2(max(_unif, _unif_obs)):.2f}% under either convention, **for those "
      f"two uniform cases and no others**. The two_zone D(x) profile, which is "
      f"the model that wins the comparison, is stiffer (r = D_max*dt/dx^2 = "
      f"{RES['cases']['two_zone']['rows']['1']['r_number']:.0f} at dt = 1 s) "
      f"and costs {h['two_zone']['dt_1s_pooled_sample_rms_psi']:.2f} psi pooled "
      f"and {h['two_zone']['dt_1s_worst_gauge_max_abs_pct_of_peak']:.4f}% of "
      f"g2's simulated peak ({_obs['two_zone']:.4f}% of its observed peak) - "
      f"{h['two_zone']['dt_1s_worst_gauge_max_abs_pct_of_peak'] / _unif:.1f}x "
      f"the uniform bound. **Over all three cases the bound is "
      f"{_up2(max(_all, max(_obs.values()))):.2f}% under either convention** "
      f"(measured maxima {_all:.4f}% simulated, {max(_obs.values()):.4f}% "
      f"observed). Never quote the {_up2(max(_unif, _unif_obs)):.2f}% figure "
      f"without naming the two cases it covers.")
    A(f"2. Convergence is first order in dt, as it must be with the Dirichlet "
      f"datum read at time level n. Over the 18 doublings from "
      f"{min(o['dt_fine_s'] for o in _pairs):g} to "
      f"{max(o['dt_coarse_s'] for o in _pairs):g} s the observed order is "
      f"{_pl[0]:.4f}-{_ph[0]:.4f} from the pooled RMS and "
      f"{_wl[0]:.4f}-{_wh[0]:.4f} from the worst-gauge maximum "
      f"(low end: {_wl[1]} {_wl[2]['dt_fine_s']:g} -> "
      f"{_wl[2]['dt_coarse_s']:g} s, where the peak error saturates); "
      f"{len(_out)} of the {len(_allw)} worst-gauge values fall outside the "
      f"pooled interval, so the two intervals are different and each must be "
      f"attributed to its own metric. Removing the finite reference's own step "
      f"(dt/(dt - dt_ref)) gives {_cpl[0]:.4f}-{_cph[0]:.4f} pooled and "
      f"{_cwl[0]:.4f}-{_cwh[0]:.4f} worst-gauge over dt = 0.25-4 s, all three "
      f"cases: the excess above 1 is the reference, not second-order "
      f"behaviour.")
    _ae = [ADAPT['absolute_protocol'][ck]['trace']['err_max'] for ck in profiles]
    _af = {100 * ADAPT['absolute_protocol'][ck]['trace']['frac_at_max_dt']
           for ck in profiles}
    _ar = {ADAPT['absolute_protocol'][ck]['trace']['n_rejected']
           for ck in profiles}
    _sh = [ADAPT['stall_scan']['cases'][ck]['min_dt_floor_shortfall_factor']
           for ck in profiles]
    _cr = [ADAPT['stall_scan']['cases'][ck]['dt_that_would_meet_tol_s']
           for ck in profiles]
    A(f"3. The manuscript's adaptive settings are not an accuracy control on "
      f"this configuration - all three cases, not one: on the absolute-pressure "
      f"protocol the estimator peaks at {min(_ae):.1e}-{max(_ae):.1e} against "
      f"tol = {a['trace']['tol']:g}, so dt pins to max_dt = 30 s for "
      + (f"{list(_af)[0]:.2f}% of the run in every case with "
         f"{list(_ar)[0]} rejections"
         if len(_af) == 1 and len(_ar) == 1 else
         f"{min(_af):.2f}-{max(_af):.2f}% of the run with "
         f"{min(_ar)}-{max(_ar)} rejections")
      + f"; on the delta-pressure protocol the same settings produce no "
        f"solution at all, because min_dt = {akw['min_dt']:g} s is "
        f"{min(_sh):.1f}x (D1150) to {max(_sh):.1f}x (two_zone) too coarse for "
        f"the estimator's own tol crossing at {max(_cr):.2e}-{min(_cr):.2e} s.")
    if ok0:
        _cf_ok = {ck: [r for r in cfc[ck] if r['completed']] for ck in profiles}
        _cf_all = all(_cf_ok[ck] for ck in profiles)
        A(f"4. That second failure is a SETTINGS failure, not a property of the "
          f"delta formulation. Lowering only min_dt to {r0['min_dt_s']:g} s "
          f"makes the same controller complete the window in "
          + ('all three cases' if _cf_all else
             ', '.join(ck for ck in profiles if _cf_ok[ck]))
          + f" (D1150: {r0['n_accepted']} accepted / {r0['n_rejected']} "
            f"rejected steps, median dt {r0['dt_median_s']:.2f} s, "
            f"{m0['pooled_sample_rms_psi']:.2f} psi pooled RMS against the "
            f"1/256 s reference"
          + (('; ' + ', '.join(
              ck for ck in profiles
              for r in cfc[ck]
              if r['min_dt_s'] == r0['min_dt_s'] and r['completed']
              and r['floor_is_active'])
              + ' completes at this floor but still touches it, so there the '
                'demonstrably inactive floor is the next one down')
             if any(r['min_dt_s'] == r0['min_dt_s'] and r['completed']
                    and r['floor_is_active']
                    for ck in profiles for r in cfc[ck]) else '')
          + f"), and there the estimator DOES bind. The "
          f"manuscript's adaptive scheme is therefore uncontrolled on the "
          f"protocol it ran (absolute) and misconfigured on the protocol the "
          f"calibration uses (delta); neither is a statement about the physics. "
          f"Stock fibeRIS still cannot run the delta protocol adaptively, but "
          f"for the separate 0/0 defect at t = 0, which min_dt does not touch.")
    _ss0 = SS['cases']['D1150']
    A(f"5. (AMENDMENT v5) The absolute-protocol numbers in item 3 belong to "
      f"fibeRIS's estimator, not to the absolute-pressure normalisation on its "
      f"own. All three solves of that estimate are given the same datum "
      f"s(t^n); correcting only that, with tol and both bounds unchanged, "
      f"moves D1150 from "
      f"{_ss0['frozen']['trace']['n_attempts']} attempts / "
      f"{_ss0['frozen']['trace']['n_rejected']} rejections / "
      f"{100 * _ss0['frozen']['trace']['frac_at_max_dt']:.2f}% at max_dt to "
      f"{_ss0['own_time']['trace']['n_attempts']} / "
      f"{_ss0['own_time']['trace']['n_rejected']} / "
      f"{100 * _ss0['own_time']['trace']['frac_at_max_dt']:.2f}%, with the peak "
      f"estimate rising {_ss0['err_max_ratio']:.1f}x to "
      f"{_ss0['own_time']['trace']['err_max']:.2e}, i.e. ABOVE tol. The "
      f"conclusion 'a fixed 30 s run in all but name' survives (median step "
      f"still {_ss0['own_time']['trace']['dt_median_s']:.0f} s, cost "
      f"{_ss0['own_time']['error_vs_fine_reference']['pooled_sample_rms_psi']:.1f}"
      f" psi against "
      f"{_ss0['frozen']['error_vs_fine_reference']['pooled_sample_rms_psi']:.1f}"
      f" psi); the MECHANISM sentence does not. Say 'the ~8300 psi datum plus "
      f"the frozen sub-step datum', never the datum alone.")
    A('')
    A('## Files\n')
    for k in ('log', 'json', 'csv_gauge', 'csv_pooled', 'csv_adaptive',
              'trace', 'npz', 'fig1', 'fig2', 'fig3', 'readme', 'manifest'):
        A(f"- `{os.path.basename(OUT[k])}`")
    A('- `README.md` (byte-identical copy of '
      f"`{os.path.basename(OUT['readme'])}`; the canonical name always carries "
      'the current pass)')
    A('- `README_v1_superseded.md` (the v1 text, verbatim: the bytes '
      '`manifest.json` hashed as `README.md`)')
    A('- `amend_v6/` (v6 amendment record: independent recomputation, '
      'v5-vs-v6 leaf diff, its own manifest)')
    A('')
    return "\n".join(L)


if __name__ == '__main__':
    raise SystemExit(main())
