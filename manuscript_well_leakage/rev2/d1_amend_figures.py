"""D1 AMEND figures.

Two figures, both new versioned files (nothing is overwritten):

`fig_d1_refit_shift_v2.png` -- the v1 figure plotted only D_near, D_far and s_c
and was titled "how far the calibrated parameters move when one gauge is
removed" (`d1_loo_blind.py:759`). It omitted the ONE parameter that moves: the
transition width w, which ranges 3.2-41.2 ft against an all-gauge 21.1 ft under
the absolute norm, and whose drop-g2 value sits 0.148% above its lower search
bound. v2 adds the w panel with the search bounds drawn, and a sixth panel
showing the cold-start restart ensemble's spread of the blind RMSE against the
published warm-started value.

`fig_d1_acceptance_margin_v2.png` -- the "first failure at 523 ft" headline turns
on a 3.9% margin under the absolute norm, and under the amplitude-normalised
norm (the criterion the manuscript's own "timing and shape" wording implies) the
uniform model passes at 1046 and 1301 ft. Neither fact was visible in text. This
figure states both, per gauge, against the pre-registered thresholds.

    python scripts/manuscript_well_leakage/rev2/d1_amend_figures.py \
        --config configs/rev2/d1_amend.json
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'manuscript_well_leakage',
                                'baseline_calibration'))
sys.path.insert(0, HERE)

import d1_loo_blind as d1        # noqa: E402
import rev2_manifest as rm       # noqa: E402
import r1_calibration_core as core  # noqa: E402
from r1_run_calibration import load_config  # noqa: E402
from d1_amend_coldstart import write_amend_manifest  # noqa: E402

SUMMARY_CSV = 'output/rev2_20260901/D1/d1_summary_v1.csv'
COLDSTART_JSON = 'output/rev2_20260901/D1/amend/d1_coldstart_ensemble_v2.json'
FINAL_REFIT = 'output/rev2_20260901/D1/amend/fig_d1_refit_shift_v3.png'
FINAL_ACC = 'output/rev2_20260901/D1/amend/fig_d1_acceptance_margin_v4.png'
FINAL_MANIFEST = 'output/rev2_20260901/D1/amend/manifest_figures_v3.json'
GNUMS = [2, 3, 4, 5, 6, 7]
PNAMES = ['D_near', 'D_far', 's_c', 'width']
PUNITS = ['ft$^2$/s', 'ft$^2$/s', 'ft', 'ft']


def log(m):
    print(f"[d1-amend-fig] {m}", flush=True)


def load_summary():
    with open(SUMMARY_CSV) as fh:
        return list(csv.DictReader(fh))


def fig_refit_shift(rows, cold, bounds, path, dpi):
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.6))
    x = np.arange(len(GNUMS))
    a = ax[0, 0]
    for norm, mk in (('absolute', 'o-'), ('normalised', 's--')):
        du = [float(r['params_refit_5gauge']) for r in rows
              if r['model'] == 'uniform' and r['norm'] == norm]
        dall = float([r['params_all_gauge'] for r in rows
                      if r['model'] == 'uniform' and r['norm'] == norm][0])
        a.plot(x, du, mk, label=f'{norm} norm refit')
        a.axhline(dall, ls=':', color='0.4')
        a.annotate(f'all-gauge {dall:.0f}', (0, dall), fontsize=7, color='0.3')
    a.set_yscale('log')
    a.set_ylabel('refit uniform D (ft$^2$/s)')
    a.set_title('(a) uniform: refit D with gauge j removed', fontsize=9.5)

    panels = [ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1]]
    for pi, (pname, unit, aa) in enumerate(zip(PNAMES, PUNITS, panels)):
        for norm, mk in (('absolute', 'o-'), ('normalised', 's--')):
            vals = [float(r['params_refit_5gauge'].split(';')[pi]) for r in rows
                    if r['model'] == 'two_zone' and r['norm'] == norm]
            allv = float([r['params_all_gauge'] for r in rows
                          if r['model'] == 'two_zone' and r['norm'] == norm
                          ][0].split(';')[pi])
            aa.plot(x, vals, mk, label=f'{norm} norm refit')
            aa.axhline(allv, ls=':', color='0.4')
        blo, bhi = 10 ** bounds[pi][0], 10 ** bounds[pi][1]
        aa.axhline(blo, color='#d62728', lw=1.2)
        aa.axhline(bhi, color='#d62728', lw=1.2)
        aa.set_yscale('log')
        aa.set_ylabel(f'two_zone {pname} ({unit})')
        ttl = f'({"bcde"[pi]}) two_zone: refit {pname}'
        if pname == 'width':
            ttl += '\nOMITTED FROM v1 - the one parameter that moves'
        aa.set_title(ttl, fontsize=9.5)
        aa.set_ylim(min(blo * 0.7, aa.get_ylim()[0]),
                    max(bhi * 1.4, aa.get_ylim()[1]))
        aa.annotate('search bounds (red)', (0.02, 0.03), xycoords='axes fraction',
                    fontsize=6.5, color='#d62728')
    panels[3].annotate('drop g2 (absolute): w = 3.167 ft,\n0.148% above the lower '
                       'search bound\n-> CENSORED, not an estimate',
                       (0.03, 0.62), xycoords='axes fraction', fontsize=6.8,
                       color='#d62728',
                       bbox=dict(fc='white', ec='#d62728', alpha=0.9, pad=2.5))

    a = ax[1, 2]
    pub = [float(r['blind_rmse_psi']) for r in rows
           if r['model'] == 'two_zone' and r['norm'] == 'absolute']
    rec = {int(r['held_out_gauge']): r
           for r in cold['reconciled_per_gauge'] if r['norm'] == 'absolute'}
    lo = np.array([rec[g]['blind_rmse_psi_min_within_1pct_of_best'] for g in GNUMS])
    hi = np.array([rec[g]['blind_rmse_psi_max_within_1pct_of_best'] for g in GNUMS])
    bst = np.array([rec[g]['best_coldstart_blind_rmse_psi'] for g in GNUMS])
    a.fill_between(x, lo, hi, color='#d62728', alpha=0.18,
                   label='cold-start fits within 1% of the best\nfive-gauge criterion')
    a.plot(x, bst, 'D-', color='#d62728', ms=5,
           label='cold start, best five-gauge fit')
    a.plot(x, pub, 'o-', color='#1f77b4', label='published (warm-started)')
    a.set_yscale('log')
    a.set_ylabel('BLIND RMSE at the held-out gauge (psi)')
    a.set_title('(f) absolute norm: the blind number is search-dependent.\n'
                'At g3 fits that tie on the five calibration gauges\n'
                'predict the withheld gauge over 14-43 psi', fontsize=9.5)

    for aa in ax.ravel():
        aa.set_xticks(x)
        aa.set_xticklabels([f'drop g{g}' for g in GNUMS], fontsize=8)
        aa.grid(alpha=0.25)
        aa.legend(fontsize=6.8)
    fig.suptitle('D1 AMEND  how far the calibrated parameters move when one gauge is removed '
                 '- now including the transition width w, which v1 omitted', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    log(f"wrote {path}")


def fig_acceptance(rows, acc, path, dpi):
    thr = float(acc['max_normalised_rmse'])
    alo, ahi = acc['amplitude_ratio_band']
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    C = {'absolute': '#1f77b4', 'normalised': '#2ca02c'}
    off = {'absolute': (0, -16), 'normalised': (0, 13)}
    for norm in ('absolute', 'normalised'):
        rs = [r for r in rows if r['model'] == 'uniform' and r['norm'] == norm]
        d = np.array([float(r['distance_ft']) for r in rs])
        y = np.array([float(r['blind_rmse_norm']) for r in rs])
        amp = np.array([float(r['blind_amp_ratio']) for r in rs])
        ok = np.array([r['usable_blind_prediction'] == 'True' for r in rs])
        ax[0].plot(d, y, 'o-', color=C[norm], label=f'uniform, {norm} norm')
        ax[1].plot(d, amp, 'o-', color=C[norm], label=f'uniform, {norm} norm')
        for aa, v in ((ax[0], y), (ax[1], amp)):
            aa.plot(d[~ok], v[~ok], 'x', ms=13, mew=2.2, color='#d62728')
        for di, yi in zip(d, y):
            m = 100 * (yi - thr) / thr
            ax[0].annotate(f'{m:+.0f}%', (di, yi), textcoords='offset points',
                           xytext=off[norm], fontsize=8, fontweight='bold',
                           color=C[norm], ha='center',
                           bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.9))
    ax[0].axhline(thr, color='0.3', ls='--')
    ax[0].annotate(f'pre-registered acceptance threshold {thr}',
                   (0.42, 0.55), xycoords='axes fraction', fontsize=8, color='0.3')
    ax[0].annotate('g3 sets the reported 523 ft onset.\n'
                   'It fails the absolute-norm test by only 4%\n'
                   '(0.260 vs 0.250), but by 40% under the\n'
                   'normalised norm - which is why 523 ft,\n'
                   'and not 1046 ft, is the reported onset.',
                   (0.03, 0.70), xycoords='axes fraction', fontsize=7.5,
                   color='#d62728',
                   bbox=dict(fc='white', ec='#d62728', alpha=0.92, pad=3.0))
    ax[0].set_yscale('log')
    ax[0].set_ylabel('BLIND RMSE / observed peak')
    ax[0].set_title('(a) the 523 ft headline rests on a 3.9% margin\n'
                    'labels = margin against the threshold; red x = fails',
                    fontsize=9.5)
    for lim in (alo, ahi):
        ax[1].axhline(lim, color='0.3', ls='--')
    ax[1].axhline(1.0, color='k', lw=0.6)
    ax[1].set_yscale('log')
    ax[1].set_ylabel('blind amplitude ratio sim/obs')
    ax[1].set_title(f'(b) amplitude band [{alo}, {ahi}]: g7 overshoots 3.38x '
                    '(absolute norm)\nbut only 2.09x under the normalised norm',
                    fontsize=9.5)
    for aa in ax:
        aa.set_xlabel('distance from source gauge 1 (ft)')
        aa.grid(alpha=0.25)
        aa.legend(fontsize=8, loc='best')
        for g, dd in zip(GNUMS, [261, 523, 777, 1046, 1301, 1570]):
            aa.annotate(f'g{g}', (dd, aa.get_ylim()[0]), fontsize=7, color='0.45',
                        ha='center', va='bottom')
    fig.suptitle('D1 AMEND  the uniform-model range of applicability is norm-dependent, and its '
                 'first failure is marginal.\nAbsolute norm: fails at 523/1046/1301/1570 ft (2/6 usable). '
                 'Normalised norm: fails at 523 and 1570 ft only (4/6 usable).', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    log(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--only', choices=('all', 'final'), default='all',
                    help="'final' re-emits BOTH figures under the frozen "
                         "post-review code so their manifest carries no code "
                         "drift. The earlier drafts are left in place (house "
                         "rule 2 forbids overwriting them); their file names "
                         "and the reason each was superseded are listed in the "
                         "D1 README. Paths are hard-coded here, not read from "
                         "the config, so re-emitting cannot change the config "
                         "hash recorded by the runs that came before.")
    args = ap.parse_args()
    cfg, _ = load_config(args.config)
    out = cfg['outputs']
    rows = load_summary()
    cold = json.load(open(COLDSTART_JSON))
    dpi = int(out['figure_dpi'])
    bounds = cfg['families']['two_zone']['bounds']
    if args.only == 'final':
        rm.assert_absent([FINAL_REFIT, FINAL_ACC, FINAL_MANIFEST])
        fig_refit_shift(rows, cold, bounds, FINAL_REFIT, dpi)
        fig_acceptance(rows, cfg['acceptance'], FINAL_ACC, dpi)
        S = d1.build_setup(cfg)
        taxis, _ = core.solve_forward(
            S['mesh'], d1._profile(S, 'two_zone',
                                   np.array(cfg['families']['two_zone']['warm_start'], float)),
            S['dt'], S['t_total'], S['src']['taxis'], S['src']['delta_psi'],
            S['source_idx'], record_idx=[0])
        write_amend_manifest(
            FINAL_MANIFEST, cfg, args.config, S, taxis,
            study_id='d1_amend_figures_final',
            results={'figures': [FINAL_REFIT, FINAL_ACC],
                     'supersedes': ['fig_d1_refit_shift_v2.png (same content; its '
                                    'manifest predates a later edit to this script)',
                                    'fig_d1_acceptance_margin_v2.png (draft: the g3 '
                                    'margin labels overlapped)',
                                    'fig_d1_acceptance_margin_v3.png (same content '
                                    'as v4; its manifest predates the code freeze)'],
                     'note': ('No inversion: both figures are re-plots of '
                              'd1_summary_v1.csv plus the cold-start ensemble. One '
                              'forward solve is run only so the manifest records a '
                              'realised taxis.')},
            outputs=[(FINAL_REFIT, 'figure_png', dpi),
                     (FINAL_ACC, 'figure_png', dpi)],
            notes=['Amends major defects 4 (w omitted from the refit-shift '
                   'figure) and 5/12 (523 ft margin, norm dependence) of '
                   'output/rev2_20260901/A4/challenge_defects/D1_defects.json.',
                   'Definitive figure pair for the D1 amend.'],
            extra_inputs=[(COLDSTART_JSON, 'prior_run_output', 'coldstart_ensemble')])
        log(f"wrote {FINAL_MANIFEST}")
        return
    rm.assert_absent([out['fig_refit_shift'], out['fig_acceptance_margin'],
                      out['manifest_figures']])
    fig_refit_shift(rows, cold, bounds, out['fig_refit_shift'], dpi)
    fig_acceptance(rows, cfg['acceptance'], out['fig_acceptance_margin'], dpi)

    S = d1.build_setup(cfg)
    taxis, _ = core.solve_forward(
        S['mesh'], d1._profile(S, 'two_zone',
                               np.array(cfg['families']['two_zone']['warm_start'], float)),
        S['dt'], S['t_total'], S['src']['taxis'], S['src']['delta_psi'],
        S['source_idx'], record_idx=[0])
    write_amend_manifest(
        out['manifest_figures'], cfg, args.config, S, taxis,
        study_id='d1_amend_figures',
        results={'figures': ['fig_d1_refit_shift_v2.png',
                             'fig_d1_acceptance_margin_v2.png'],
                 'note': ('No new inversion: both figures are re-plots of '
                          'd1_summary_v1.csv plus the cold-start ensemble. One '
                          'forward solve is run only to record the realised '
                          'taxis in the manifest.')},
        outputs=[(out['fig_refit_shift'], 'figure_png', dpi),
                 (out['fig_acceptance_margin'], 'figure_png', dpi)],
        notes=['Amends major defects 4 (w omitted from the refit-shift figure) '
               'and 5/12 (523 ft margin, norm dependence) of '
               'output/rev2_20260901/A4/challenge_defects/D1_defects.json.'],
        extra_inputs=[(COLDSTART_JSON, 'prior_run_output', 'coldstart_ensemble')])
    log(f"wrote {out['manifest_figures']}")


if __name__ == '__main__':
    main()
