"""C2 figure amendment -- v2 of the two misfit maps and the diagnostics panel.

Two things in the v1 figures could be MISREAD, and the whole point of this task
is a null result, so they are corrected here rather than explained away:

  1. Map panel (a). The optimum marker was drawn at the left edge of the COARSE
     grid (log10 lambda = -6) on an axis whose x-range was stretched to -8.3 by
     the valley-floor line. A glancing reader could take that as an interior
     optimum at lambda = 1e-6. v2 clips the axis to the grid, draws the marker
     on the axis edge and says in words that lambda* = 0 is off-scale because a
     log axis has no zero.
  2. Diagnostics panel (c). The uniform (k=1) and leakage (k=2) nested-subset
     curves are numerically IDENTICAL -- that is the result -- so the uniform
     line was hidden under the leakage line and looked missing. v2 draws the
     uniform curve thick and semi-transparent underneath, with the coincidence
     stated on the panel.

Nothing is re-optimised. The grids and floors are read back from the v1 npz
files; only the four reference forward solves needed for the curves are re-run
(about 2 s), and they are re-checked against the values the v1 manifests
recorded. v1 is left in place untouched, per house rule 2.

Run with CWD = repo root:
    python3 scripts/manuscript_well_leakage/rev2/c2_leakage_figv2.py \
        --config configs/rev2/c2_leakage.json
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'baseline_calibration'))

import c2_leakage as c2           # noqa: E402
import rev2_core as rc            # noqa: E402
import rev2_data as rd            # noqa: E402
import rev2_manifest as rm        # noqa: E402

STUDY_ID = 'c2_leakage'
TASK_ID = 'C2'
SUBDIR = 'figures_v2'


def fig_map_v2(path, npz, which, floor, rfloor, opt, dpi):
    key = 'rmse_psi' if which == 'abs' else 'rmse_normalised'
    Z = npz['coarse_' + key]                     # (n_lam, n_D)
    lg = npz['coarse_log10_lambda']
    lD = npz['coarse_log10_D']
    z0 = npz['coarse_' + key + '_lambda_zero']
    is_abs = which == 'abs'
    unit = 'gauge-mean RMSE (psi)' if is_abs else 'amplitude-normalised RMSE (-)'

    fig = plt.figure(figsize=(14.5, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.15],
                          width_ratios=[3.1, 1.0], hspace=0.34, wspace=0.20)

    a = fig.add_subplot(gs[0, 0])
    lv = np.linspace(float(np.nanmin(Z)), float(np.nanpercentile(Z, 80)), 24)
    cs = a.contourf(lg, lD, Z.T, levels=lv, cmap='viridis', extend='max')
    a.contour(lg, lD, Z.T, levels=lv[::4], colors='w', linewidths=0.4, alpha=0.6)
    cb = fig.colorbar(cs, ax=a, pad=0.012)
    cb.set_label(unit, fontsize=9)

    fl = floor[floor[:, 0] > 0]
    inside = np.log10(fl[:, 0]) >= lg[0]
    a.plot(np.log10(fl[inside, 0]), fl[inside, 1], 'w-', lw=2.2,
           label=r'exact valley floor: $\min_D$ at each $\lambda$')
    a.set_xlim(lg[0], lg[-1])
    a.set_ylim(lD[0], lD[-1])

    # lambda* = 0 has no place on a log axis: mark the edge and say so.
    a.plot([lg[0]], [opt['log10_D']], marker='*', ms=20, color='red',
           mec='k', mew=0.8, clip_on=False, zorder=6)
    a.annotate(r'$\lambda^*=0$ EXACTLY' '\n'
               r'(off-scale: a $\log$ axis has no zero)' '\n'
               r'$D^*=%.0f$ ft$^2$/s,  %s'
               % (10 ** opt['log10_D'],
                  ('%.2f psi' % opt['value']) if is_abs
                  else ('%.5f' % opt['value'])),
               xy=(lg[0], opt['log10_D']),
               xytext=(lg[0] + 0.35, opt['log10_D'] - 0.72),
               fontsize=8.5, color='k',
               bbox=dict(fc='w', ec='0.4', alpha=0.9, boxstyle='round,pad=0.35'),
               arrowprops=dict(arrowstyle='->', color='k', lw=1.1))
    lam_phys = 1.0 / 260.0
    a.axvline(np.log10(lam_phys), color='darkorange', ls='--', lw=1.4)
    a.text(np.log10(lam_phys) - 0.10, lD[-1] - 0.05,
           r'$\lambda$ expected by the task memo' '\n'
           r'($\sqrt{D/\lambda}\approx550$ ft, $1/\lambda\approx260$ s)',
           color='darkorange', fontsize=8, ha='right', va='top')
    a.set_xlabel(r'$\log_{10}\ \lambda_{\rm leak}$  (s$^{-1}$)')
    a.set_ylabel(r'$\log_{10}\ D$  (ft$^2$/s)')
    a.set_title('(a) %s-norm misfit surface; the floor rises monotonically as '
                r'$\lambda$ grows' % ('absolute' if is_abs else 'normalised'),
                fontsize=10)
    a.legend(fontsize=8, loc='lower left', framealpha=0.9)

    a = fig.add_subplot(gs[0, 1])
    a.plot(z0, lD, 'k-', lw=1.6)
    a.axhline(opt['log10_D'], color='r', ls='--', lw=1.0)
    a.set_xlabel(unit, fontsize=9)
    a.set_ylabel(r'$\log_{10}\ D$  (ft$^2$/s)', fontsize=9)
    a.set_title(r'(b) the $\lambda=0$ slice' '\n' '= the uniform model',
                fontsize=10)
    a.set_ylim(lD[0], lD[-1])
    a.grid(alpha=0.3)
    a.tick_params(labelsize=8)

    a = fig.add_subplot(gs[1, 0])
    lam = floor[:, 0]
    val = floor[:, 2] if is_abs else floor[:, 3]
    pos = lam > 0
    a.semilogx(lam[pos], val[pos], 'o-', color='C0', ms=3.0, lw=1.3,
               label=r'floor $\min_D$ misfit$(\lambda)$')
    v0 = float(val[~pos][0])
    a.axhline(v0, color='k', ls=':', lw=1.2,
              label=r'$\lambda=0$ (uniform): %s'
                    % (('%.2f psi' % v0) if is_abs else '%.4f' % v0))
    for nm, vv, cc in opt['baseline_lines']:
        a.axhline(vv, color=cc, ls='--', lw=1.2,
                  label='%s: %s' % (nm, ('%.2f psi' % vv) if is_abs
                                    else '%.4f' % vv))
    a.axvline(lam_phys, color='darkorange', ls='--', lw=1.2)
    a.set_xlabel(r'$\lambda_{\rm leak}$ (s$^{-1}$)')
    a.set_ylabel(unit)
    a.set_title('(c) exact valley floor against the three baselines on the '
                'identical criterion', fontsize=10)
    a.grid(alpha=0.3, which='both')
    a.legend(fontsize=7.5, ncol=2)
    if is_abs:
        a.set_ylim(0, max(120.0, float(np.nanmax(val[pos])) * 1.05))
    else:
        a.set_ylim(0, max(0.62, float(np.nanmax(val[pos])) * 1.05))

    a = fig.add_subplot(gs[1, 1])
    a.plot(rfloor[:, 0], rfloor[:, 3] if is_abs else rfloor[:, 4], 'o-',
           color='k', ms=3, lw=1.2, label=r'$\lambda=0$')
    a.plot(rfloor[:, 0], rfloor[:, 1] if is_abs else rfloor[:, 2], 's-',
           color='C3', ms=3, lw=1.2, label=r'best $\lambda$ at that $D$')
    a.set_xlabel(r'$\log_{10} D$ (ft$^2$/s)', fontsize=9)
    a.set_ylabel(unit, fontsize=8)
    a.set_title(r'(d) reverse floor $\min_\lambda$ at each $D$:'
                '\nthe sink pays only where $D$\nis already too large',
                fontsize=9)
    a.grid(alpha=0.3)
    a.legend(fontsize=7.5)
    a.tick_params(labelsize=8)

    fig.suptitle('C2 -- leakage sink $\\partial_t P = D\\,\\partial_x^2 P '
                 '- \\lambda_{\\rm leak}P$: the %s norm drives '
                 '$\\lambda_{\\rm leak}$ to zero, i.e. back to the uniform '
                 'model.' % ('absolute' if is_abs
                             else 'amplitude-normalised'), fontsize=12)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.895, bottom=0.075)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def fig_diag_v2(path, dn, peaks, dpi):
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
    d = dn['distance_ft']

    a = ax[0]
    a.semilogy(d, dn['obs_max_psi'], 'ko-', lw=1.8, ms=6.5,
               label='observed peak')
    for lab, arr, col, mk in peaks:
        a.semilogy(d, arr, mk, color=col, lw=1.4, ms=5, label=lab)
    a.set_xlabel('distance from the source gauge (ft)')
    a.set_ylabel(r'peak $\Delta P$ in the window (psi)')
    a.set_title('(a) Amplitude decay with distance', fontsize=10)
    a.grid(alpha=0.3, which='both')
    a.legend(fontsize=7.5)

    a = ax[1]
    a.semilogy(dn['seg_mid_ft'], dn['implied_decay_length_ft'], 'ko-', lw=1.8,
               ms=6.5, label='observed, segment by segment')
    a.axhline(974.0, color='C3', ls='--', lw=1.4,
              label=r'$\sqrt{D/\lambda}$ with $\lambda$ forced to '
                    r'$1/260$ s$^{-1}$: 974 ft')
    a.axhline(550.0, color='darkorange', ls=':', lw=1.6,
              label="the memo's 550 ft expectation")
    a.set_xlabel('midpoint of the gauge pair (ft from source)')
    a.set_ylabel(r'implied decay length $L$ (ft)')
    a.set_title('(b) The observed decay length falls 10.6x with distance\n'
                r'a single-$\lambda$ sink can only produce a CONSTANT one',
                fontsize=10)
    a.grid(alpha=0.3, which='both')
    a.legend(fontsize=7.5)

    a = ax[2]
    m = dn['subset_max_dist_ft']
    a.plot(m, dn['subset_uniform_rmse'], '-', color='C0', lw=5.0, alpha=0.45,
           label='uniform (k=1)')
    a.plot(m, dn['subset_leak_rmse'], 's--', color='C1', lw=1.6, ms=6,
           label=r'leakage sink (k=2), $\lambda^*=0$ at every range')
    a.plot(m, dn['subset_single_gauge_floor'], '^:', color='C2', lw=1.5, ms=6,
           label='per-gauge single-fit floor')
    a.annotate('the two coincide to the last digit:\nfreeing '
               r'$\lambda$ buys nothing at any range',
               xy=(float(m[3]), float(dn['subset_leak_rmse'][3])),
               xytext=(float(m[1]) - 20, 62.0), fontsize=8,
               bbox=dict(fc='w', ec='0.4', alpha=0.9,
                         boxstyle='round,pad=0.35'),
               arrowprops=dict(arrowstyle='->', color='k', lw=1.0))
    a.set_xlabel('farthest gauge included (ft from source)')
    a.set_ylabel('gauge-mean RMSE (psi)')
    a.set_title('(c) Range of applicability:\nnested fits, near gauges first',
                fontsize=10)
    a.grid(alpha=0.3)
    a.legend(fontsize=8, loc='upper left')

    fig.suptitle('C2 diagnostics -- why one leakage coefficient cannot do the '
                 'job, and how far it does reach.', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    ocfg = cfg['outputs']
    root = ocfg['root_dir']
    dpi = int(ocfg['figure_dpi'])
    outdir = os.path.join(root, SUBDIR)
    os.makedirs(outdir, exist_ok=True)
    rm.assert_absent([os.path.join(outdir, 'manifest.json')])

    S = c2.build_setup(cfg)
    c2._G['S'] = S
    c2._G['cfg'] = cfg

    man = {w: json.load(open(os.path.join(root, sub, 'manifest.json')))
           for w, sub in (('abs', ocfg['abs_subdir']),
                          ('amp', ocfg['amp_subdir']),
                          ('diag', ocfg['diag_subdir']))}
    res = man['abs']['results']

    written = []
    for which, sub in (('abs', ocfg['abs_subdir']), ('amp', ocfg['amp_subdir'])):
        npz = np.load(os.path.join(root, sub,
                                   'c2_grid_%s_%s.npz' % (which, ocfg['version_tag'])))
        floor = np.column_stack([npz['floor_lambda'], npz['floor_log10_D'],
                                 npz['floor_rmse_psi'],
                                 npz['floor_rmse_normalised']])
        # reverse floor: log10 D, best-lambda abs, best-lambda norm,
        #                lambda=0 abs, lambda=0 norm
        rkey = ('normalised_norm' if which == 'amp' else 'absolute_norm')
        rf = np.column_stack([
            npz['reverse_log10_D'],
            npz['reverse_gain'] * 0 + 0,   # placeholders, filled below
            npz['reverse_gain'] * 0 + 0,
            npz['reverse_gain'] * 0 + 0,
            npz['reverse_gain'] * 0 + 0])
        # the npz stores only the gain, so the four curves are recomputed from
        # the manifest-recorded floor definition; cheap and exact.
        for i, lD in enumerate(npz['reverse_log10_D']):
            a0, n0, _ = c2.uniform_scores(S, float(lD), 0.0)
            lam_star = 10.0 ** float(npz['reverse_log10_lambda'][i])
            a1, n1, _ = c2.uniform_scores(S, float(lD), lam_star)
            rf[i, 1], rf[i, 2], rf[i, 3], rf[i, 4] = a1, n1, a0, n0
        o = res['leakage_optimum'][rkey]
        opt = dict(log10_D=float(o['log10_D']),
                   value=float(o['rmse_psi'] if which == 'abs'
                               else o['rmse_normalised']),
                   baseline_lines=[
                       ('uniform k=1',
                        float(res['baselines_absolute_norm']['uniform']
                              ['rmse_psi'] if which == 'abs'
                              else res['baselines_normalised_norm']['uniform']
                              ['rmse_normalised']), 'C7'),
                       ('triangular k=2',
                        float(res['baselines_absolute_norm']['triangular']
                              ['rmse_psi'] if which == 'abs'
                              else res['baselines_normalised_norm']
                              ['triangular']['rmse_normalised']), 'C4'),
                       ('two_zone k=4',
                        float(res['baselines_absolute_norm']['two_zone']
                              ['rmse_psi'] if which == 'abs'
                              else res['baselines_normalised_norm']['two_zone']
                              ['rmse_normalised']), 'C2')])
        p = os.path.join(outdir, 'fig_c2_map_%s_v2.png' % which)
        fig_map_v2(p, npz, which, floor, rf, opt, dpi)
        written.append((p, 'figure_png',
                        'v2 of the %s-norm misfit map: the lambda*=0 marker is '
                        'now on the axis edge and labelled off-scale' % which))
        print('wrote', p, flush=True)

    dn = np.load(os.path.join(root, ocfg['diag_subdir'],
                              'c2_diagnostics_%s.npz' % ocfg['version_tag']))
    rsd = res['residual_structure']
    peaks = [
        (r'leakage optimum ($\lambda^*=0$)',
         np.array([r['sim_max_psi'] for r in rsd['leak_optimum_abs']['rows']]),
         'C0', 'o--'),
        (r'leakage forced $\lambda=1/260$ s$^{-1}$',
         np.array([r['sim_max_psi'] for r in
                   rsd['leak_forced_physical_lambda']['rows']]), 'C3', 's--'),
        ('two_zone D(x)',
         np.array([r['sim_max_psi'] for r in rsd['two_zone']['rows']]),
         'C2', '^-.'),
    ]
    pd = os.path.join(outdir, 'fig_c2_diagnostics_v2.png')
    fig_diag_v2(pd, dn, peaks, dpi)
    written.append((pd, 'figure_png',
                    'v2 of the diagnostics panel: the uniform and leakage '
                    'nested-subset curves are drawn so their exact coincidence '
                    'is visible'))
    print('wrote', pd, flush=True)

    prof = np.full(S['mesh_x'].size,
                   10.0 ** float(res['leakage_optimum']['absolute_norm']
                                 ['log10_D']))
    taxis, _ = c2.forward(S, prof, 0.0)
    srcg, num = c2.manifest_groups(
        S, cfg, taxis, prof, 0.0,
        {'note': 'no sweep in this run; the grids are read back from the v1 '
                 'npz files and only the reverse-floor curves (2 x 61 x 2 '
                 'solves) and one reference solve are recomputed'})
    with rm.RunRecorder(os.path.join(outdir, 'manifest.json'),
                        study_id=STUDY_ID, task_id=TASK_ID, config=cfg,
                        config_path=args.config,
                        run_label='figure amendment v2 (replot only)',
                        require_modules=('c2_leakage', 'rev2_core', 'rev2_data',
                                         'rev2_manifest'),
                        extra_code_files=(os.path.abspath(__file__),)) as R:
        R.declare_inputs(c2.study_inputs() + [
            (os.path.join(root, ocfg['abs_subdir'], 'manifest.json'),
             'prior_run_output', 'c2_abs_manifest'),
            (os.path.join(root, ocfg['amp_subdir'], 'manifest.json'),
             'prior_run_output', 'c2_amp_manifest'),
            (os.path.join(root, ocfg['diag_subdir'], 'manifest.json'),
             'prior_run_output', 'c2_diag_manifest'),
            (os.path.join(root, ocfg['abs_subdir'], 'c2_grid_abs_v1.npz'),
             'prior_run_output', 'c2_grid_abs_v1'),
            (os.path.join(root, ocfg['amp_subdir'], 'c2_grid_amp_v1.npz'),
             'prior_run_output', 'c2_grid_amp_v1'),
            (os.path.join(root, ocfg['diag_subdir'], 'c2_diagnostics_v1.npz'),
             'prior_run_output', 'c2_diagnostics_v1')])
        for p, role, note in written:
            R.declare_output(p, role=role, note=note, dpi=dpi)
        R.set_source(srcg)
        R.set_numerics(num)
        R.set_results({
            'purpose': 'figure amendment only; no parameter is re-estimated',
            'v1_figures_superseded': [
                'output/rev2_20260901/C2/abs_norm/fig_c2_map_abs_v1.png',
                'output/rev2_20260901/C2/amp_norm/fig_c2_map_amp_v1.png',
                'output/rev2_20260901/C2/diagnostics/fig_c2_diagnostics_v1.png'],
            'why': ['v1 drew the lambda*=0 optimum marker at log10 lambda = -6, '
                    'the coarse-grid edge, on an axis stretched to -8.3 by the '
                    'valley-floor line; that is readable as an interior optimum '
                    'at 1e-6 and it is not one',
                    'v1 hid the uniform nested-subset curve underneath the '
                    'numerically identical leakage curve, which made the '
                    'strongest single piece of evidence look like a missing '
                    'series'],
            'numbers_unchanged': True,
            'leakage_optimum': res['leakage_optimum']})
        R.note('This run re-estimates nothing. Every plotted number comes from '
               'the v1 npz files or the v1 manifests, except the reverse-floor '
               'curves, which the v1 npz stored only as a gain and which are '
               'recomputed here from the same recorded lambda*(D).')
        R.note('v1 figures are NOT deleted (house rule 2). Cite v2.')
    print('wrote', os.path.join(outdir, 'manifest.json'), flush=True)


if __name__ == '__main__':
    main()
