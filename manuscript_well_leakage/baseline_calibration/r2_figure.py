"""R2 figure: the inverted D(x) profile and what it buys over a single D."""
import json, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_calibration_core as core
import r2_profile_inversion as r2

CFG = 'configs/r2_diffusivity_profile.json'
OUT = 'figs/manuscript/baseline_calibration/r2_profile.png'
# Best fits (from the R2 manifest; powerlaw refit with corrected bounds).
FITS = {
    'uniform':     ([3.057], 82.330, 'C7'),
    'triangular':  ([3.088, -2.339], 67.124, 'C1'),
    'exponential': ([4.508, 2.343, 1.995], 16.015, 'C0'),
    'powerlaw':    ([np.log10(38113), np.log10(139), 2.85, np.log10(237)], 15.561, 'C4'),
    'two_zone':    ([3.586, 2.404, 2.642, 1.325], 11.872, 'C2'),
}
SINGLE = {2: 20308, 3: 6752, 4: 1231, 5: 675, 6: 611, 7: 409}


def main():
    cfg, _ = r2.load_config(CFG)
    S = r2._setup(cfg, 'gauge')
    mesh, si, tg = S['mesh'], S['source_idx'], S['targets']
    dt, tt = cfg['solver']['dt_s'], float(S['src']['taxis'][-1])

    profs = {}
    for f, (p, _r, _c) in FITS.items():
        fn = core.PROFILE_FAMILIES[f]['fn']
        profs[f] = (fn(mesh, si, np.array(p), S['win_lo'], S['win_hi'])
                    if f == 'triangular' else fn(mesh, si, np.array(p)))

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))

    # (a) misfit by family
    a = ax[0]
    fams = list(FITS)
    r = [FITS[f][1] for f in fams]
    ks = [core.PROFILE_FAMILIES[f]['k'] for f in fams]
    bars = a.bar(range(len(fams)), r, color=[FITS[f][2] for f in fams])
    for i, (v, k) in enumerate(zip(r, ks)):
        a.text(i, v + 2, f'{v:.1f}\nk={k}', ha='center', fontsize=9)
    a.set_xticks(range(len(fams)))
    a.set_xticklabels(fams, rotation=20, fontsize=9)
    a.axhspan(6.2, 18.3, color='k', alpha=0.10)
    a.text(0.02, 20, 'band: what each gauge achieves when fitted ALONE',
           fontsize=8, transform=a.get_yaxis_transform() if False else a.transData)
    a.set_ylabel('gauge-mean RMSE (psi)')
    a.set_title('(a) Letting the data choose the shape of $D(x)$\n'
                'drops misfit 7x, to the single-gauge floor', fontsize=10)
    a.grid(alpha=0.3, axis='y')

    # (b) the profiles
    a = ax[1]
    s = np.abs(mesh - mesh[si])
    for f in fams:
        a.loglog(np.maximum(s, 1), profs[f], color=FITS[f][2], lw=1.9,
                 label=f'{f} ({FITS[f][1]:.1f} psi)')
    a.plot([t['distance_ft'] for t in tg], [SINGLE[t['gauge']] for t in tg],
           'kv', ms=9, label='single-gauge fit (path-AVERAGED)')
    for t in tg:
        a.axvline(t['distance_ft'], color='0.85', lw=0.8, zorder=0)
        a.annotate(f"g{t['gauge']}", (t['distance_ft'], 130),
                   fontsize=7.5, ha='center')
    a.axvspan(1, 100, color='r', alpha=0.07)
    a.text(3, 6e4, 'no gauge here:\nextrapolation', fontsize=7.5, color='r')
    a.set_xlabel('distance from source (ft)')
    a.set_ylabel(r'local $D$ (ft$^2$/s)')
    a.set_xlim(20, 2000)
    a.set_ylim(100, 1e5)
    a.set_title('(b) Inverted profiles agree on the far field (~220-255)\n'
                'and disagree where no gauge constrains them', fontsize=10)
    a.legend(fontsize=7.5, loc='lower left')
    a.grid(alpha=0.3, which='both')

    # (c) fit at the winner vs uniform
    a = ax[2]
    idx = [t['idx'] for t in tg]
    tx_u, fu = core.solve_forward(mesh, profs['uniform'], dt, tt,
                                  S['src']['taxis'], S['src']['delta_psi'], si,
                                  record_idx=idx)
    tx_w, fw = core.solve_forward(mesh, profs['two_zone'], dt, tt,
                                  S['src']['taxis'], S['src']['delta_psi'], si,
                                  record_idx=idx)
    cols = plt.cm.viridis(np.linspace(0, 0.88, len(tg)))
    for k, t in enumerate(tg):
        a.plot(t['taxis'], t['data'], '-', color=cols[k], lw=2.0)
        a.plot(tx_u, fu[:, k], ':', color=cols[k], lw=1.3)
        a.plot(tx_w, fw[:, k], '--', color=cols[k], lw=1.3)
    a.plot([], [], 'k-', lw=2, label='observed')
    a.plot([], [], 'k:', lw=1.3, label='uniform D (82.3 psi)')
    a.plot([], [], 'k--', lw=1.3, label='two-zone D(x) (11.9 psi)')
    a.set_xlabel('time since window start (s)')
    a.set_ylabel(r'$\Delta P$ (psi)')
    a.set_title('(c) Same data, both models\n'
                'the profile tracks every gauge', fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    fig.suptitle('R2 — inverting the diffusivity PROFILE. S well stage 1, '
                 'MD 15000-16750 ft. Source = gauge 1; targets = gauges 2-7.',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
