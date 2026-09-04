"""D1 follow-up: is the blind prediction of gauge 2 under two_zone actually blind?

The two_zone winner puts its transition at s_c = 438 ft. Only gauge 2 (261 ft)
lies inside the near zone; gauges 3-7 (523-1570 ft) are all in the far zone. So
when gauge 2 is the held-out gauge, the five calibration gauges may carry no
information at all about log10_D_near (and little about s_c) -- in which case
the "blind prediction" at gauge 2 is not a prediction from data but whatever the
optimiser happened to leave those coordinates at, i.e. the warm start.

This probe scans log10_D_near and log10_s_c one at a time about each fitted
optimum and reports, for the drop-g2 subset, how far each can move before the
five-gauge criterion rises by 1% and by 10%, and what the blind RMSE at gauge 2
does over that same interval. A flat criterion with a strongly varying blind
error is the signature of an unidentified prediction, and must be reported as
such.

    python scripts/manuscript_well_leakage/rev2/d1_nearzone_identifiability.py \
        --config configs/rev2/d1_loo_blind.json
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d1_loo_blind as d1  # noqa: E402
import r1_calibration_core as core  # noqa: E402
from r1_run_calibration import load_config  # noqa: E402

OUT_JSON = 'output/rev2_20260901/D1/d1_identifiability_v1.json'
OUT_MANIFEST = 'output/rev2_20260901/D1/manifest_identifiability.json'
OUT_FIG = 'output/rev2_20260901/D1/fig_d1_identifiability_v1.png'


def _scan_point(args):
    """(params) -> (subset criterion pieces) for the shared worker setup."""
    p = args
    S = d1._G['S']
    return d1.per_gauge_misfit(S, d1._profile(S, 'two_zone', p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg, cfg_sha = load_config(args.config)
    d1._guard([OUT_JSON, OUT_MANIFEST, OUT_FIG])

    S = d1.build_setup(cfg)
    gnums = [t['gauge'] for t in S['targets']]
    fits = {}
    for fam, path in (('two_zone', 'output/rev2_20260901/D1/manifest_two_zone.json'),):
        m = json.load(open(path))
        for key, r in m['results']['calibrations'].items():
            norm, label = key.split('|')
            fits[(norm, label)] = r['params']

    bounds = cfg['families']['two_zone']['bounds']
    names = cfg['families']['two_zone']['param_names']
    scan_idx = {0: 'log10_D_near', 2: 'log10_s_c'}
    n_scan = 41

    tasks, meta = [], []
    for norm in ('absolute', 'normalised'):
        for label in ('all', 'drop_g2'):
            base = np.array(fits[(norm, label)], float)
            for pi, pname in scan_idx.items():
                grid = np.linspace(bounds[pi][0], bounds[pi][1], n_scan)
                for v in grid:
                    q = base.copy()
                    q[pi] = v
                    tasks.append(q)
                    meta.append((norm, label, pname, float(v)))

    nproc = int(cfg['search']['processes'])
    with Pool(nproc, initializer=d1._init_worker, initargs=(cfg,)) as pool:
        res = pool.map(_scan_point, tasks, chunksize=4)

    g2col = gnums.index(2)
    keep5 = [c for c in range(len(gnums)) if c != g2col]
    out = {}
    for (norm, label, pname, v), (mse, n2) in zip(meta, res):
        key = f'{norm}|{label}|{pname}'
        d = out.setdefault(key, {'param': pname, 'norm': norm, 'subset': label,
                                 'grid_log10': [], 'calib_criterion': [],
                                 'blind_rmse_g2_psi': [], 'blind_amp_g2': []})
        cols = keep5 if label == 'drop_g2' else list(range(len(gnums)))
        vec = n2 if norm == 'normalised' else mse
        d['grid_log10'].append(v)
        d['calib_criterion'].append(float(np.sqrt(np.mean(np.asarray(vec)[cols]))))
        d['blind_rmse_g2_psi'].append(float(np.sqrt(mse[g2col])))
        d['blind_amp_g2'].append(float('nan'))

    summary = {}
    for key, d in out.items():
        g = np.array(d['grid_log10'])
        c = np.array(d['calib_criterion'])
        b = np.array(d['blind_rmse_g2_psi'])
        i = int(np.argmin(c))
        for tag, tol in (('1pct', 0.01), ('10pct', 0.10)):
            ok = np.where(c <= c[i] * (1.0 + tol))[0]
            lo, hi = float(g[ok[0]]), float(g[ok[-1]])
            summary.setdefault(key, {})[tag] = {
                'log10_interval': [lo, hi],
                'physical_interval': [float(10 ** lo), float(10 ** hi)],
                'decades': float(hi - lo),
                'censored_low': bool(ok[0] == 0),
                'censored_high': bool(ok[-1] == len(g) - 1),
                'blind_rmse_g2_range_psi': [float(b[ok].min()), float(b[ok].max())],
                'blind_rmse_g2_ratio_over_interval': float(b[ok].max() / b[ok].min()),
            }
        summary[key]['argmin_log10'] = float(g[i])
        summary[key]['argmin_physical'] = float(10 ** g[i])
        summary[key]['criterion_at_argmin'] = float(c[i])
        summary[key]['blind_rmse_g2_at_argmin_psi'] = float(b[i])
    with open(OUT_JSON, 'w') as fh:
        json.dump({'scans': out, 'summary': summary,
                   'note': ('For subset drop_g2 the five calibration gauges are all '
                            'in the far zone (523-1570 ft) while the two_zone '
                            'transition sits near 438 ft, so log10_D_near is scanned '
                            'to test whether it is constrained at all.')},
                  fh, indent=2)
    print(f"wrote {OUT_JSON}")

    for key in sorted(summary):
        s = summary[key]['1pct']
        print(f"{key:34s} argmin {summary[key]['argmin_physical']:9.1f}  "
              f"+1% interval {s['physical_interval'][0]:8.1f}..{s['physical_interval'][1]:9.1f} "
              f"({s['decades']:.2f} decades{', CENSORED' if s['censored_low'] or s['censored_high'] else ''})  "
              f"blind g2 RMSE over that interval "
              f"{s['blind_rmse_g2_range_psi'][0]:.1f}..{s['blind_rmse_g2_range_psi'][1]:.1f} psi")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    for r, norm in enumerate(('absolute', 'normalised')):
        for c_, pname in enumerate(('log10_D_near', 'log10_s_c')):
            a = ax[r, c_]
            a2 = a.twinx()
            for label, col in (('all', '0.35'), ('drop_g2', '#d62728')):
                d = out[f'{norm}|{label}|{pname}']
                g = np.array(d['grid_log10'])
                cc = np.array(d['calib_criterion'])
                a.plot(10 ** g, cc / cc.min(), '-', color=col,
                       label=f'{label}: calibration criterion / its min')
                if label == 'drop_g2':
                    a2.plot(10 ** g, d['blind_rmse_g2_psi'], ':', color='#1f77b4',
                            label='blind RMSE at gauge 2 (right axis)')
            a.set_xscale('log')
            a.set_yscale('log')
            a2.set_yscale('log')
            a.axhline(1.01, color='0.6', ls='--', lw=0.8)
            a.set_xlabel(pname.replace('log10_', '') +
                         (' (ft)' if 's_c' in pname else ' (ft$^2$/s)'))
            a.set_ylabel('criterion / minimum')
            a2.set_ylabel('blind RMSE at g2 (psi)', color='#1f77b4')
            a.set_title(f'{norm} norm, scanning {pname}', fontsize=9)
            a.grid(alpha=0.25)
            h1, l1 = a.get_legend_handles_labels()
            h2, l2 = a2.get_legend_handles_labels()
            a.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper center')
    fig.suptitle('D1 identifiability probe: when gauge 2 is withheld, do the remaining '
                 'five gauges constrain the near-zone parameters at all?', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_FIG, dpi=int(cfg['outputs']['figure_dpi']))
    plt.close(fig)
    print(f"wrote {OUT_FIG}")

    d1.write_manifest(OUT_MANIFEST, cfg, cfg_sha, args.config, S,
                      {'model_family': 'two_zone',
                       'probe': 'one-at-a-time scan of log10_D_near and log10_s_c '
                                'about each fitted optimum',
                       'n_forward_solves': int(len(tasks)),
                       'n_scan_points_per_parameter': n_scan,
                       'summary': summary},
                      [OUT_JSON, OUT_FIG], 1255, 'identifiability')
    print(f"wrote {OUT_MANIFEST}")


if __name__ == '__main__':
    main()
