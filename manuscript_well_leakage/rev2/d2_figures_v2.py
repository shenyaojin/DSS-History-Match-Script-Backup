"""Figures for D2 v2 (source-gauge leave-one-out). Imported by d2_source_loo_v2.py.

Three products:
  fig_d2_source_loo_v2      what each boundary substitution does to the waveforms
  fig_d2_degradation_v2     per-gauge degradation and the calibrated-D shift,
                            under BOTH norms side by side
  fig_d2_negative_control_v2  the gauge-1 prediction of variant (b): restart
                            envelope, naive no-solver predictors, and the
                            domain-top dependence that the answer inherits
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

NORM_LABEL = {'abs': 'absolute norm (gauge-mean RMSE, psi)',
              'norm': 'amplitude-normalised norm'}
VAR_ORDER = ['baseline', 'base_frachit', 'baseline_far', 'extrap_g1md',
             'extrap_frachit', 'src_g2', 'src_g3']
SHORT = {'baseline': 'baseline\n(g1 @16645)',
         'base_frachit': 'g1 series\n@frac hit',
         'baseline_far': 'baseline\ncalib{4-7}',
         'extrap_g1md': '(a) extrap\n->16645',
         'extrap_frachit': "(a') extrap\n->frac hit",
         'src_g2': '(b) g2 @16384',
         'src_g3': '(c) g3 @16122'}


def _g(ev, n):
    return next(x for x in ev['per_gauge'] if x['gauge'] == n)


def _sim_on(taxis_obs, ta, rec, k):
    return np.interp(taxis_obs, ta, rec[:, k])


def fig_waveforms(cfg, variants, series, sims, scores, path, dpi):
    show_v = [v for v in ('baseline_far', 'extrap_frachit', 'src_g2', 'src_g3')
              if any(V['name'] == v for V in variants)]
    show_g = [1, 3, 4, 7]
    fig, axes = plt.subplots(len(show_g), len(show_v),
                             figsize=(4.0 * len(show_v), 2.5 * len(show_g)),
                             sharex=True)
    byname = {V['name']: V for V in variants}
    for j, vn in enumerate(show_v):
        V = byname[vn]
        gpos = {t['gauge']: k for k, t in enumerate(V['targets'])}
        for i, gn in enumerate(show_g):
            ax = axes[i, j]
            obs = series[gn]
            ax.plot(obs['taxis'], obs['delta_psi'], color='k', lw=1.6,
                    label='observed', zorder=5)
            for (model, which), (ta, rec) in sims[vn].items():
                if model == 'two_zone_bc':
                    continue
                style = '-' if which == 'abs' else '--'
                col = 'C0' if model == 'uniform' else 'C3'
                ax.plot(obs['taxis'], _sim_on(obs['taxis'], ta, rec, gpos[gn]),
                        style, color=col, lw=1.2, alpha=0.9,
                        label=f"{model} [{which}]")
            role = V['roles'][gn]
            ev = scores[vn][('two_zone_frachit', 'abs')]
            r = _g(ev, gn)
            ax.set_title(f"g{gn} (MD {obs['md_ft']:.0f}) - {role}\n"
                         f"two_zone[abs] RMSE {r['rmse_psi']:.1f} psi",
                         fontsize=8)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel('$\\Delta P$ (psi)', fontsize=9)
            if i == len(show_g) - 1:
                ax.set_xlabel('time (s)', fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=6.5, loc='upper left')
        axes[0, j].text(0.5, 1.42, SHORT[vn].replace('\n', ' '),
                        transform=axes[0, j].transAxes, ha='center',
                        fontsize=10, fontweight='bold')
    fig.suptitle('D2 v2 - what removing the source gauge does to the waveforms.\n'
                 'Black: observed. Blue: uniform. Red: two_zone anchored at the '
                 'frac-hit centroid. Solid: absolute norm, dashed: normalised norm.',
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def fig_degradation(cfg, variants, series, scores, fits, path, dpi):
    names = [V['name'] for V in VAR_ORDER_present(variants)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for row, which in enumerate(('abs', 'norm')):
        # (a) per-gauge RMSE by variant, two_zone_frachit
        ax = axes[row, 0]
        for vn in names:
            ev = scores[vn].get(('two_zone_frachit', which))
            if ev is None:
                continue
            gs = [g['gauge'] for g in ev['per_gauge']]
            ys = [(g['rmse_psi'] if which == 'abs' else g['rmse_normalised'])
                  for g in ev['per_gauge']]
            ax.plot(gs, ys, 'o-', lw=1.3, ms=4, label=SHORT[vn].replace('\n', ' '))
        ax.set_yscale('log')
        ax.set_xlabel('gauge')
        ax.set_ylabel('RMSE (psi)' if which == 'abs' else 'RMSE / obs peak')
        ax.set_title(f'(a{row + 1}) per-gauge misfit, two_zone\n{NORM_LABEL[which]}',
                     fontsize=10)
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=7)

        # (b) same for uniform
        ax = axes[row, 1]
        for vn in names:
            ev = scores[vn].get(('uniform', which))
            if ev is None:
                continue
            gs = [g['gauge'] for g in ev['per_gauge']]
            ys = [(g['rmse_psi'] if which == 'abs' else g['rmse_normalised'])
                  for g in ev['per_gauge']]
            ax.plot(gs, ys, 's--', lw=1.3, ms=4, label=SHORT[vn].replace('\n', ' '))
        ax.set_yscale('log')
        ax.set_xlabel('gauge')
        ax.set_ylabel('RMSE (psi)' if which == 'abs' else 'RMSE / obs peak')
        ax.set_title(f'(b{row + 1}) per-gauge misfit, uniform\n{NORM_LABEL[which]}',
                     fontsize=10)
        ax.grid(alpha=0.3, which='both')

        # (c) calibrated D / D_near / D_far by variant
        ax = axes[row, 2]
        x = np.arange(len(names))
        du, dn, df = [], [], []
        for vn in names:
            u = fits[vn].get('uniform')
            du.append(u['per_norm'][which]['D'] if u else np.nan)
            tz = fits[vn].get('two_zone_frachit')
            dn.append(tz['per_norm'][which]['D_near'] if tz else np.nan)
            df.append(tz['per_norm'][which]['D_far'] if tz else np.nan)
        ax.plot(x, du, 'o-', label='uniform $D$')
        ax.plot(x, dn, '^-', label='two_zone $D_{near}$')
        ax.plot(x, df, 'v-', label='two_zone $D_{far}$')
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[n] for n in names], fontsize=7, rotation=30,
                           ha='right')
        ax.set_ylabel('$D$ (ft$^2$/s)')
        ax.set_title(f'(c{row + 1}) calibrated diffusivity\n{NORM_LABEL[which]}',
                     fontsize=10)
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=7)
    fig.suptitle('D2 v2 - degradation per gauge and the calibrated $D$ it buys, '
                 'under both norms', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def VAR_ORDER_present(variants):
    have = {V['name']: V for V in variants}
    return [have[n] for n in VAR_ORDER if n in have]


def fig_negative_control(cfg, variants, series, sims, scores, fits, summary,
                         path, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    obs1 = series[1]
    o1 = obs1['delta_psi']

    # (a) the gauge-1 waveform under every substitution
    ax = axes[0, 0]
    ax.plot(obs1['taxis'], o1, 'k', lw=2.0, label='gauge 1 observed', zorder=6)
    byname = {V['name']: V for V in variants}
    for vn, col in (('src_g2', 'C3'), ('src_g3', 'C1'),
                    ('extrap_frachit', 'C2')):
        if vn not in byname:
            continue
        V = byname[vn]
        k = [t['gauge'] for t in V['targets']].index(1)
        for which, ls in (('abs', '-'), ('norm', '--')):
            key = ('two_zone_frachit', which)
            if key not in sims[vn]:
                continue
            ta, rec = sims[vn][key]
            ax.plot(obs1['taxis'], _sim_on(obs1['taxis'], ta, rec, k), ls,
                    color=col, lw=1.4,
                    label=f"{SHORT[vn].replace(chr(10), ' ')} two_zone[{which}]")
    for r in summary['naive_gauge1_predictors']:
        if r['predictor'] == 'extrap_g2g3_to_16645':
            ax.plot([], [], ' ',
                    label=f"naive g2,g3 line: {r['rmse_psi']:.0f} psi")
    ax.set_xlabel('time (s)')
    ax.set_ylabel('$\\Delta P$ (psi)')
    ax.set_title('(a) blind prediction of gauge 1 (MD 16645)', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # (b) restart envelope of the gauge-1 blind RMSE
    ax = axes[0, 1]
    labels, lo, hi, best = [], [], [], []
    for vn in ('src_g2', 'src_g3'):
        if vn not in fits:
            continue
        for model in ('uniform', 'two_zone_frachit'):
            f = fits[vn].get(model)
            if not f or 'restarts' not in f:
                continue
            for which in ('abs', 'norm'):
                vals = [r['blind_scores']['g1']['rmse_psi']
                        for r in f['restarts'][which]
                        if 'g1' in r.get('blind_scores', {})]
                if not vals:
                    continue
                labels.append(f"{vn}\n{model}\n[{which}]")
                lo.append(min(vals))
                hi.append(max(vals))
                best.append(_g(scores[vn][(model, which)], 1)['rmse_psi'])
    if labels:
        x = np.arange(len(labels))
        ax.vlines(x, lo, hi, color='C0', lw=6, alpha=0.4,
                  label='cold-restart envelope')
        ax.plot(x, best, 'ko', ms=6, label='reported (best restart)')
        for r in summary['naive_gauge1_predictors']:
            ls = {'copy_g2': ':', 'copy_g3': '-.',
                  'extrap_g2g3_to_16645': '--'}.get(r['predictor'])
            if ls:
                ax.axhline(r['rmse_psi'], ls=ls, color='C3', lw=1.2,
                           label=f"naive {r['predictor']}: {r['rmse_psi']:.0f} psi")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel('gauge-1 RMSE (psi)')
        ax.set_title('(b) how much of the answer is the optimiser?\n'
                     'restart envelope vs no-solver predictors', fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    # (c) domain-top dependence
    ax = axes[1, 0]
    rows = summary['domain_top_sensitivity']
    for vn, col in (('src_g2', 'C3'), ('src_g3', 'C1')):
        for model, ls in (('uniform', '--'), ('two_zone_frachit', '-')):
            sel = [r for r in rows if r['variant'] == vn and r['model'] == model
                   and r['norm'] == 'abs']
            if not sel:
                continue
            ax.plot([r['top_md_ft'] for r in sel],
                    [r['g1_rmse_psi'] for r in sel], ls, color=col, marker='o',
                    ms=3, label=f"{vn} {model}")
    ax.set_xlabel('domain top MD (ft)')
    ax.set_ylabel('gauge-1 RMSE (psi)')
    ax.set_title('(c) the blind answer inherits the far-end boundary\n'
                 '(the calibration objective does not move at all)', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # (d) amplitude ratio and arrival error of gauge 1 vs distance from the BC
    ax = axes[1, 1]
    for which, mk in (('abs', 'o'), ('norm', 's')):
        xs, ys, ann = [], [], []
        for V in VAR_ORDER_present(variants):
            ev = scores[V['name']].get(('two_zone_frachit', which))
            if ev is None:
                continue
            g1 = _g(ev, 1)
            if V['roles'][1] in ('source_dirichlet', 'imposed_extrapolation'):
                continue
            xs.append(g1['distance_ft'])
            ys.append(g1['amplitude_ratio'])
            ann.append(f"{V['name']} [{g1['role'][:5]}]")
        ax.plot(xs, ys, mk, ms=8, label=f'two_zone [{which}]')
        for x, y, a in zip(xs, ys, ann):
            ax.annotate(a, (x, y), fontsize=7, xytext=(4, 4),
                        textcoords='offset points')
    ax.axhline(1.0, color='k', lw=0.8)
    ax.set_xlabel('distance of gauge 1 from the Dirichlet node (ft)')
    ax.set_ylabel('gauge-1 amplitude ratio (sim / obs)')
    ax.set_title('(d) near-field amplitude of the blind target', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle('D2 v2 - the negative control: how well does a model driven from '
                 'gauge 2 predict gauge 1, 261 ft away and next to the injection?',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def make_all(cfg, variants, series, sims, scores, fits, summary):
    out = cfg['outputs']
    dpi = int(out['figure_dpi'])
    fig_waveforms(cfg, variants, series, sims, scores, out['figure_main_png'], dpi)
    fig_degradation(cfg, variants, series, scores, fits,
                    out['figure_degradation_png'], dpi)
    fig_negative_control(cfg, variants, series, sims, scores, fits, summary,
                         out['figure_control_png'], dpi)
