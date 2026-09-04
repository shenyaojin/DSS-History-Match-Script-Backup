"""D2 presentation figures, re-plotted from the run's saved arrays.

Kept separate from d2_source_loo.py on purpose. The run script hashes every .py
file it imports into its manifest, so editing it after a run would invalidate
that manifest; this reads only d2_arrays_v1.npz and d2_summary_v1.json and
touches no solver. The v1 figures the run script emits are the raw product; the
v2 figures here are the same numbers with shared per-row y axes (so the two
model columns are actually comparable), honest legend proxies, and an explicit
role annotation per panel.

    python scripts/manuscript_well_leakage/rev2/d2_figures.py \
        --config configs/rev2/d2_source_loo.json
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'manuscript_well_leakage',
                                'baseline_calibration'))
import r1_calibration_core as core  # noqa: E402

ALL_GAUGES = [1, 2, 3, 4, 5, 6, 7]
ORDER = ['baseline_far', 'extrap_g1md', 'extrap_frachit', 'src_g2', 'src_g3']
COLOR = {'baseline_far': '#444444', 'extrap_g1md': '#1b9e77',
         'extrap_frachit': '#7570b3', 'src_g2': '#d95f02', 'src_g3': '#e7298a'}
SHORT = {'baseline_far': 'baseline: g1 @ MD 16645 (the removed gauge)',
         'extrap_g1md': "(a) extrapolate g2,g3 -> MD 16645",
         'extrap_frachit': "(a') extrapolate g2,g3 -> MD 16683 (frac hits)",
         'src_g2': '(b) g2 @ its own MD 16384',
         'src_g3': '(c) g3 @ its own MD 16122'}
TAG = {'source_dirichlet': 'imposed', 'imposed_extrapolation': 'imposed',
       'bc_constituent': 'in BC', 'calibration': 'calib', 'blind': 'BLIND'}
MODELS = ['uniform', 'two_zone_frachit']
MTITLE = {'uniform': 'uniform $D$  (k = 1)',
          'two_zone_frachit': 'two-zone $D(x)$ anchored at the frac hits  (k = 4)'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    out = cfg['outputs']
    dpi = int(out['figure_dpi'])
    A = np.load(out['arrays_npz'])
    S = json.load(open(out['summary_json']))
    V = S['variants']
    gmd = {n: float(m) for n, m in zip(A['gauge_numbers'], A['gauge_md_ft'])}

    def sim(v, m):
        return A[f'sim_taxis__{v}__{m}'], A[f'sim__{v}__{m}']

    def rec(v, m, g):
        return next(x for x in V[v]['scores'][m]['per_gauge'] if x['gauge'] == g)

    # ---------------- main figure: one panel per gauge, two model columns ----
    fig, axes = plt.subplots(7, 2, figsize=(11.6, 17.0), sharex=True, sharey='row')
    for r, n in enumerate(ALL_GAUGES):
        for c, m in enumerate(MODELS):
            ax = axes[r, c]
            ax.axhline(0, color='#bbbbbb', lw=0.6, zorder=0)
            ax.plot(A[f'obs_taxis__g{n}'], A[f'obs_data__g{n}'], color='k', lw=2.2,
                    zorder=6)
            for v in ORDER:
                g = rec(v, m, n)
                ta, sm = sim(v, m)
                ax.plot(ta, sm[:, n - 1], color=COLOR[v], lw=1.35,
                        ls='-' if g['is_prediction'] else ':',
                        zorder=5 if g['is_prediction'] else 4)
            if r == 0:
                ax.set_title(MTITLE[m], fontsize=11, pad=8)
            ax.set_ylabel(f"gauge {n}   MD {gmd[n]:.0f} ft\n$\\Delta P$ [psi]", fontsize=8.5)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.22, lw=0.4)
            txt = "  ".join(f"{v.split('_')[0][:4] if v != 'extrap_frachit' else 'extF'}"
                            f":{TAG[V[v]['roles'][str(n)]]}" for v in ORDER)
            ax.text(0.012, 0.965, txt, transform=ax.transAxes, fontsize=6.2,
                    va='top', color='#777777', family='monospace')
            blind = [v for v in ORDER if V[v]['roles'][str(n)] == 'blind']
            if blind:
                lab = "  ".join(f"{v}: RMSE {rec(v, m, n)['rmse_psi']:.0f} psi, "
                                f"amp {rec(v, m, n)['amplitude_ratio']:.2f}" for v in blind)
                ax.text(0.012, 0.055, lab, transform=ax.transAxes, fontsize=6.2,
                        va='bottom', color='#333333')
    for c in range(2):
        axes[-1, c].set_xlabel('time since 2020-03-16 11:24:00 [s]', fontsize=9.5)

    handles = [Line2D([], [], color='k', lw=2.2, label='observed')]
    handles += [Line2D([], [], color=COLOR[v], lw=1.5, label=SHORT[v]) for v in ORDER]
    handles += [Line2D([], [], color='#888888', lw=1.5, ls=':',
                       label='dotted: Dirichlet node sits on this gauge (imposed, not a prediction)')]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=8.6,
               bbox_to_anchor=(0.5, 0.998), frameon=False)
    fig.suptitle('D2  -  remove the source gauge: does the model still predict?\n'
                 'gauge 1 (MD 16645) drives the published boundary condition; every curve '
                 'below except the grey one is computed without it',
                 fontsize=12, y=1.030)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    p = out['figure_png'].replace('_v1.png', '_v2.png')
    fig.savefig(p, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('wrote', p)

    # ---------------- BC / summary figure -----------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    ax = axes[0]
    ax.plot(A['obs_taxis__g1'], A['obs_data__g1'], color='k', lw=2.4,
            label='gauge 1 observed (MD 16645) - REMOVED', zorder=6)
    for v in ORDER:
        ax.plot(A[f'bc_taxis__{v}'], A[f'bc_data__{v}'], color=COLOR[v], lw=1.4,
                ls='--' if v == 'baseline_far' else '-', label=SHORT[v])
    ax.set_xlabel('time since 11:24:00 [s]')
    ax.set_ylabel('$\\Delta P$ [psi]')
    ax.set_title('a. the driving series each variant imposes', fontsize=10)
    ax.legend(fontsize=7.0, frameon=False, loc='upper left')
    ax.grid(alpha=0.22, lw=0.4)

    ax = axes[1]
    for m, ls, mk in (('uniform', '-', 'o'), ('two_zone_frachit', '--', 's')):
        for v in ORDER:
            gs = [x for x in V[v]['scores'][m]['per_gauge'] if x['is_prediction']]
            ax.plot([x['md_ft'] for x in gs], [x['rmse_psi'] for x in gs],
                    marker=mk, ms=4.5, lw=1.2, color=COLOR[v], ls=ls, alpha=0.9)
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.set_xlabel('gauge MD [ft]  (injection is to the right)')
    ax.set_ylabel('per-gauge RMSE [psi]')
    ax.set_title('b. per-gauge RMSE\nsolid = uniform, dashed = two-zone $D(x)$', fontsize=10)
    ax.grid(alpha=0.22, lw=0.4, which='both')
    ax.legend(handles=[Line2D([], [], color=COLOR[v], lw=1.4, label=SHORT[v].split(':')[0])
                       for v in ORDER], fontsize=7.0, frameon=False)

    ax = axes[2]
    x = np.arange(len(ORDER))
    for m, off, mk in (('uniform', -0.12, 'o'), ('two_zone_frachit', 0.12, 's')):
        d = []
        for v in ORDER:
            f = V[v]['fits'][m]
            d.append(f['D'] if m == 'uniform' else f['D_far'])
        ax.plot(x + off, d, marker=mk, ls='none', ms=8,
                color='#1f4e79' if m == 'uniform' else '#a6611a',
                label='uniform $D$' if m == 'uniform' else 'two-zone $D_{far}$')
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[v].split(':')[0].replace('baseline', 'baseline')
                        for v in ORDER], rotation=30, ha='right', fontsize=7.5)
    ax.set_yscale('log')
    ax.set_ylabel('calibrated $D$ [ft$^2$/s]')
    ax.set_title('c. the calibrated diffusivity moves with the\nboundary, the far-field value does not',
                 fontsize=10)
    ax.grid(alpha=0.22, lw=0.4, which='both', axis='y')
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    p = out['figure_bc_png'].replace('_v1.png', '_v2.png')
    fig.savefig(p, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('wrote', p)

    # append the two v2 files to the run manifest's output inventory
    mp = out['manifest_json']
    man = json.load(open(mp))
    have = {o['path'] for o in man['outputs']}
    for q in (out['figure_png'].replace('_v1.png', '_v2.png'),
              out['figure_bc_png'].replace('_v1.png', '_v2.png'),
              os.path.relpath(os.path.abspath(__file__), REPO)):
        if q not in have and os.path.exists(q):
            man['outputs'].append({'path': q, 'bytes': os.path.getsize(q),
                                   'sha256': core.file_sha256(q)})
    man.setdefault('post_run_figures', {})['d2_figures_py_sha256'] = \
        core.file_sha256(os.path.abspath(__file__))
    man['post_run_figures']['note'] = (
        'fig_*_v2.png were re-plotted from d2_arrays_v1.npz and d2_summary_v1.json by '
        'scripts/manuscript_well_leakage/rev2/d2_figures.py AFTER the solver run. No '
        'solver was re-executed; the v1 figures emitted by the run itself are unchanged.')
    json.dump(man, open(mp, 'w'), indent=2)
    print('updated', mp)


if __name__ == '__main__':
    main()
