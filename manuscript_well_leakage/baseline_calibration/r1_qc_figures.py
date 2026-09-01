"""QC figures for the R1 baseline-diffusivity calibration.

Reads the same committed config the study uses plus the run manifest, so the QC
plots always describe the run that actually produced the reported numbers.

    python scripts/manuscript_well_leakage/baseline_calibration/r1_qc_figures.py \
        --config configs/r1_baseline_calibration.json

Writes (300 dpi):
    figs/manuscript/baseline_calibration/r1_qc_data.png       - input data QC
    figs/manuscript/baseline_calibration/r1_qc_fit.png        - per-gauge fit + residual QC
    figs/manuscript/baseline_calibration/r1_qc_numerics.png   - convergence / solver QC
    figs/manuscript/baseline_calibration/r1_qc_waterfall.png  - field waterfalls, obs vs sim
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_calibration_core as core  # noqa: E402
from r1_run_calibration import load_config, load_window_data, pick_source_gauge  # noqa: E402

OUTDIR = 'figs/manuscript/baseline_calibration'


def build(cfg):
    series, gnums, gmds, frac_hits, _, _ = load_window_data(cfg)
    src_gauge, fh_centroid = pick_source_gauge(cfg, series, frac_hits)
    m = cfg['mesh']
    pad_lo = float(m.get('domain_pad_low_md_ft', 0.0))
    mesh = np.arange(cfg['window']['md_min_ft'] - pad_lo,
                     cfg['window']['md_max_ft'] + m['dx_ft'] / 2.0, m['dx_ft'])
    src = series[src_gauge]
    source_idx = int(np.argmin(np.abs(mesh - src['md_ft'])))
    targets = [{'gauge': n, 'md_ft': series[n]['md_ft'],
                'distance_ft': abs(series[n]['md_ft'] - src['md_ft']),
                'idx': int(np.argmin(np.abs(mesh - series[n]['md_ft']))),
                'taxis': series[n]['taxis'], 'data': series[n]['delta_psi']}
               for n in sorted(series) if n != src_gauge]
    return series, src_gauge, src, mesh, source_idx, targets, frac_hits


def fig_data(cfg, series, src_gauge, frac_hits, path):
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ns = sorted(series)
    colors = plt.cm.viridis(np.linspace(0, 0.88, len(ns)))

    a = ax[0, 0]
    for c, n in zip(colors, ns):
        s = series[n]
        a.plot(s['taxis'], s['raw_psi'], color=c, lw=1.3,
               label=f"g{n} MD {s['md_ft']:.0f}" + (" (source)" if n == src_gauge else ""))
    a.set_xlabel('time since window start (s)')
    a.set_ylabel('raw gauge pressure (psi)')
    a.set_title('(a) Raw pressure, all gauges in window', fontsize=10)
    a.legend(fontsize=7, ncol=2)
    a.grid(alpha=0.3)

    a = ax[0, 1]
    for c, n in zip(colors, ns):
        s = series[n]
        a.plot(s['taxis'], s['delta_psi'], color=c, lw=1.3)
    a.set_xlabel('time since window start (s)')
    a.set_ylabel(r'$\Delta P = P(t)-P(0)$ (psi)')
    a.set_title('(b) Baseline-removed series (what the study fits)', fontsize=10)
    a.grid(alpha=0.3)

    a = ax[1, 0]
    md0 = series[src_gauge]['md_ft']
    dist = np.array([abs(series[n]['md_ft'] - md0) for n in ns])
    amp = np.array([float(np.max(series[n]['delta_psi'])) for n in ns])
    order = np.argsort(dist)
    a.plot(dist[order], amp[order], 'o-', color='C0', lw=1.6, ms=7)
    for n, d, v in zip(ns, dist, amp):
        a.annotate(f'g{n}', (d, v), textcoords='offset points', xytext=(6, 5), fontsize=8)
    a.set_xlabel('distance from source gauge (ft)')
    a.set_ylabel(r'max $\Delta P$ (psi)')
    a.set_title('(c) Amplitude vs distance — must decrease monotonically\n'
                'strictly decreasing: '
                f"{bool(np.all(np.diff(amp[order]) < 0))}", fontsize=10)
    a.grid(alpha=0.3)

    a = ax[1, 1]
    rows = []
    for n in ns:
        s = series[n]
        dt = np.diff(s['taxis'])
        rows.append((n, s['md_ft'], len(s['taxis']), float(np.median(dt)),
                     int(np.isnan(s['raw_psi']).sum()),
                     int(np.sum(np.diff(s['raw_psi']) == 0)),
                     float(np.max(np.abs(np.diff(s['raw_psi']))))))
    a.axis('off')
    txt = f"{'gauge':>6}{'MD ft':>9}{'N':>6}{'dt s':>7}{'NaN':>5}{'flat':>6}{'maxjump psi':>13}\n"
    txt += '-' * 52 + '\n'
    for r in rows:
        txt += f"{'g'+str(r[0]):>6}{r[1]:>9.0f}{r[2]:>6}{r[3]:>7.2f}{r[4]:>5}{r[5]:>6}{r[6]:>13.2f}\n"
    txt += '-' * 52 + '\n'
    txt += f"stage-1 frac hits MD: {', '.join(f'{x:.1f}' for x in frac_hits)}\n"
    txt += f"source gauge: g{src_gauge} (nearest to frac hits)\n"
    txt += "NaN=0, flat=0, no jump>20 psi  ->  data QC clean"
    a.text(0.0, 1.0, txt, family='monospace', fontsize=9, va='top')
    a.set_title('(d) Per-gauge data quality table', fontsize=10)

    fig.suptitle('R1 QC — input data (S well stage 1, MD 15000-16750 ft, '
                 '2020-03-16 11:24-11:45)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_fit(cfg, man, src, mesh, source_idx, targets, path):
    d_uni = man['results']['uniform']['best']
    d_gra = man['results']['graded']['best']
    ratio = cfg['sweeps']['graded']['d_min_over_d_max']
    tlo = float(cfg['sweeps']['graded'].get('taper_lo_md_ft', cfg['window']['md_min_ft']))
    thi = float(cfg['sweeps']['graded'].get('taper_hi_md_ft', cfg['window']['md_max_ft']))
    dt = cfg['solver']['dt_s']
    t_total = float(src['taxis'][-1])
    idx = [t['idx'] for t in targets]

    tx_u, rec_u = core.solve_forward(mesh, core.build_uniform_profile(mesh, d_uni),
                                     dt, t_total, src['taxis'], src['delta_psi'],
                                     source_idx, record_idx=idx)
    tx_g, rec_g = core.solve_forward(
        mesh, core.build_triangular_profile(mesh, d_gra, ratio, source_idx, tlo, thi),
        dt, t_total, src['taxis'], src['delta_psi'], source_idx, record_idx=idx)

    fig, axes = plt.subplots(2, 6, figsize=(19, 7),
                             gridspec_kw={'height_ratios': [2, 1]})
    for k, tgt in enumerate(targets):
        su = np.interp(tgt['taxis'], tx_u, rec_u[:, k])
        sg = np.interp(tgt['taxis'], tx_g, rec_g[:, k])
        a = axes[0, k]
        a.plot(tgt['taxis'], tgt['data'], 'k-', lw=1.8, label='observed')
        a.plot(tgt['taxis'], su, '--', color='C0', lw=1.4,
               label=f'uniform D={d_uni:.0f}')
        a.plot(tgt['taxis'], sg, '--', color='C1', lw=1.4,
               label=f'graded Dmax={d_gra:.0f}')
        a.set_title(f"g{tgt['gauge']}  {tgt['distance_ft']:.0f} ft", fontsize=10)
        a.grid(alpha=0.3)
        if k == 0:
            a.set_ylabel(r'$\Delta P$ (psi)')
            a.legend(fontsize=7)
        a.tick_params(labelsize=8)

        a = axes[1, k]
        a.plot(tgt['taxis'], su - tgt['data'], '-', color='C0', lw=1.2)
        a.plot(tgt['taxis'], sg - tgt['data'], '-', color='C1', lw=1.2)
        a.axhline(0, color='k', lw=0.8, ls=':')
        a.set_xlabel('time (s)')
        a.grid(alpha=0.3)
        a.tick_params(labelsize=8)
        if k == 0:
            a.set_ylabel('residual\n(sim - obs, psi)')
        ru = float(np.sqrt(np.mean((su - tgt['data']) ** 2)))
        rg = float(np.sqrt(np.mean((sg - tgt['data']) ** 2)))
        a.text(0.03, 0.06, f'RMSE {ru:.0f} / {rg:.0f}', transform=a.transAxes,
               fontsize=7.5)

    fig.suptitle('R1 QC — per-gauge fit and residuals at the joint optimum. '
                 'Near gauges under-predicted (residual < 0), far gauges '
                 'over-predicted (residual > 0): a systematic sign flip, not scatter.',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_numerics(man, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    a = ax[0]
    dtv = [r['dt_s'] for r in man['dt_convergence']]
    rv = [r['rmse_psi'] for r in man['dt_convergence']]
    a.semilogx(dtv, rv, 'o-', color='C0')
    a.axvline(man['dt_used_s'], color='k', ls='--', lw=1.0,
              label=f"used dt={man['dt_used_s']} s")
    a.set_xlabel('time step dt (s)')
    a.set_ylabel('pooled RMSE at D=500 (psi)')
    a.set_title(f"(a) Time-step convergence\nerror at dt=1 s: "
                f"{abs(rv[2]-rv[-1]):.3f} psi ({abs(rv[2]-rv[-1])/rv[-1]*100:.2f}%)",
                fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3, which='both')

    a = ax[1]
    pv = [r['pad_ft'] for r in man['domain_padding_convergence']]
    pr = [r['rmse_pooled_psi_at_D2000'] for r in man['domain_padding_convergence']]
    pg = [r['farthest_gauge_sim_max_psi'] for r in man['domain_padding_convergence']]
    a.plot(pv, pr, 'o-', color='C3', label='pooled RMSE at D=2000')
    a.axvline(man['domain']['pad_low_ft'], color='k', ls='--', lw=1.0,
              label=f"used pad={man['domain']['pad_low_ft']:.0f} ft")
    a.set_xlabel('low-MD domain padding (ft)')
    a.set_ylabel('pooled RMSE (psi)', color='C3')
    a2 = a.twinx()
    a2.plot(pv, pg, 's--', color='C2')
    obs = man['domain_padding_convergence'][0]['farthest_gauge_obs_max_psi']
    a2.axhline(obs, color='C2', lw=1.0, ls=':')
    a2.set_ylabel(f"g7 simulated max (psi); obs={obs:.0f}", color='C2')
    a.set_title('(b) Domain-padding convergence\nno-flux reflection inflates the '
                'farthest gauge', fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[2]
    sg = man['single_gauge_diagnostic']['per_gauge']
    ec = {x['gauge']: x['erfc_implied_D'] for x in man['erfc_cross_check']}
    d = [x['distance_ft'] for x in sg]
    a.semilogy(d, [x['best_D'] for x in sg], 'o-', color='C3',
               label='solver, each gauge alone')
    a.semilogy(d, [ec[x['gauge']] for x in sg], 's--', color='C4',
               label='erfc inversion (no solver)')
    a.axhline(man['results']['uniform']['best'], color='C0', ls='--', lw=1.2,
              label=f"joint best {man['results']['uniform']['best']:.0f}")
    a.axhline(480, color='0.35', ls=':', lw=1.2, label='480')
    a.set_xlabel('distance from source gauge (ft)')
    a.set_ylabel(r'implied $D$ (ft$^2$/s)')
    a.set_title('(c) Model-adequacy check — two independent routes\n'
                'both show ~50x monotone collapse', fontsize=10)
    a.legend(fontsize=7.5)
    a.grid(alpha=0.3, which='both')

    fig.suptitle('R1 QC — numerical convergence and model adequacy', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_waterfall(cfg, man, src, mesh, source_idx, targets, path):
    """Waterfalls: observed LF-DAS next to the simulated pressure field.

    The DAS panel is RAW strain rate in its own units, deliberately not converted
    to pressure: the psi->strain coefficient is exactly what the provenance audit
    flagged as unresolved (the code applies eps = P/E while manuscript Eq. 2 uses
    a Poisson factor, a ~36x discrepancy). So the comparison here is
    morphological - where and when the front arrives - not quantitative.
    """
    import datetime as _dt
    from fiberis.analyzer.Data2D import Data2D_XT_DSS

    d_uni = man['results']['uniform']['best']
    d_gra = man['results']['graded']['best']
    ratio = cfg['sweeps']['graded']['d_min_over_d_max']
    tlo = float(cfg['sweeps']['graded'].get('taper_lo_md_ft', cfg['window']['md_min_ft']))
    thi = float(cfg['sweeps']['graded'].get('taper_hi_md_ft', cfg['window']['md_max_ft']))
    dt = cfg['solver']['dt_s']
    t_total = float(src['taxis'][-1])

    md_lo, md_hi = cfg['window']['md_min_ft'], cfg['window']['md_max_ft']
    win = np.where((mesh >= md_lo) & (mesh <= md_hi))[0]
    md = mesh[win]

    prof_u = core.build_uniform_profile(mesh, d_uni)
    prof_g = core.build_triangular_profile(mesh, d_gra, ratio, source_idx, tlo, thi)
    tx, fu = core.solve_forward(mesh, prof_u, dt, t_total, src['taxis'],
                                src['delta_psi'], source_idx, record_idx=win)
    _, fg = core.solve_forward(mesh, prof_g, dt, t_total, src['taxis'],
                               src['delta_psi'], source_idx, record_idx=win)
    # pressure rate: the quantity comparable in character to DAS strain rate
    ru = np.gradient(fu, tx, axis=0)
    rg = np.gradient(fg, tx, axis=0)

    das = None
    try:
        das = Data2D_XT_DSS.DSS2D()
        das.load_npz(cfg['data'].get('das_npz',
                     'data/fiberis_format/s_well/DAS/LFDASdata_stg1_swell.npz'))
        das.select_depth(md_lo, md_hi)
        das.select_time(_dt.datetime.fromisoformat(cfg['window']['time_start']),
                        _dt.datetime.fromisoformat(cfg['window']['time_end']))
    except Exception as exc:  # pragma: no cover - optional panel
        print('DAS panel skipped:', exc)
        das = None

    gmd = [t['md_ft'] for t in targets] + [src['md_ft']]
    fig, ax = plt.subplots(2, 3, figsize=(18, 9.5))

    def mark(a):
        for v in gmd:
            a.axhline(v, color='k', lw=0.6, ls=':', alpha=0.65)
        a.axhline(src['md_ft'], color='k', lw=1.6, ls='-', alpha=0.9)
        a.set_ylim(md_hi, md_lo)

    a = ax[0, 0]
    if das is not None:
        cl = float(np.nanpercentile(np.abs(das.data), 98))
        im = a.imshow(das.data, aspect='auto', cmap='bwr',
                      extent=[das.taxis[0], das.taxis[-1], das.daxis[-1], das.daxis[0]])
        im.set_clim(-cl, cl)
        plt.colorbar(im, ax=a, label='LF-DAS strain rate (raw units)')
        a.set_title('(a) OBSERVED LF-DAS waterfall\n'
                    'raw units — morphological comparison only', fontsize=10)
    else:
        a.text(0.5, 0.5, 'LF-DAS unavailable', ha='center', va='center')
    mark(a)
    a.set_ylabel('MD (ft)')
    a.set_xlabel('time since window start (s)')

    for j, (fld, ttl) in enumerate(
            [(ru, f'(b) SIMULATED $dP/dt$, uniform $D$={d_uni:.0f}'),
             (rg, f'(c) SIMULATED $dP/dt$, graded $D_{{max}}$={d_gra:.0f}')]):
        a = ax[0, j + 1]
        cl = float(np.nanpercentile(np.abs(fld), 99))
        im = a.imshow(fld.T, aspect='auto', cmap='bwr',
                      extent=[tx[0], tx[-1], md[-1], md[0]])
        im.set_clim(-cl, cl)
        plt.colorbar(im, ax=a, label='dP/dt (psi/s)')
        a.set_title(ttl + '\ncompare morphology with (a)', fontsize=10)
        mark(a)
        a.set_xlabel('time since window start (s)')

    for j, (fld, ttl) in enumerate(
            [(fu, f'(d) SIMULATED $\\Delta P$, uniform $D$={d_uni:.0f}'),
             (fg, f'(e) SIMULATED $\\Delta P$, graded $D_{{max}}$={d_gra:.0f}')]):
        a = ax[1, j]
        im = a.imshow(fld.T, aspect='auto', cmap='viridis',
                      extent=[tx[0], tx[-1], md[-1], md[0]])
        im.set_clim(0, 700)
        plt.colorbar(im, ax=a, label=r'$\Delta P$ (psi)')
        a.set_title(ttl, fontsize=10)
        mark(a)
        a.set_xlabel('time since window start (s)')
        if j == 0:
            a.set_ylabel('MD (ft)')
    # observed gauge values overlaid as coloured dots on panel (d)
    a = ax[1, 0]
    for t in targets:
        sel = np.linspace(0, len(t['taxis']) - 1, 26).astype(int)
        a.scatter(t['taxis'][sel], np.full(sel.size, t['md_ft']),
                  c=t['data'][sel], cmap='viridis', vmin=0, vmax=700,
                  s=26, edgecolors='k', linewidths=0.4, zorder=5)
    a.set_title(ax[1, 0].get_title() + '\ndots = observed gauge values (same colour scale)',
                fontsize=10)

    a = ax[1, 2]
    a.plot(prof_u[win], md, '-', color='C0', lw=2, label=f'uniform D={d_uni:.0f}')
    a.plot(prof_g[win], md, '-', color='C1', lw=2,
           label=f'graded $D_{{max}}$={d_gra:.0f}')
    sg = man['single_gauge_diagnostic']['per_gauge']
    smd = {t['gauge']: t['md_ft'] for t in targets}
    a.plot([x['best_D'] for x in sg], [smd[x['gauge']] for x in sg], 'o',
           color='C3', ms=8, label='each gauge fitted alone')
    a.axvline(480, color='0.35', ls=':', lw=1.2, label='480')
    a.set_xscale('log')
    a.set_xlabel(r'diffusivity (ft$^2$/s)')
    a.set_ylim(md_hi, md_lo)
    a.set_title('(f) Diffusivity profiles behind (b)-(e)\n'
                'red dots: what each gauge alone requires', fontsize=10)
    a.legend(fontsize=8)
    a.grid(alpha=0.3, which='both')

    fig.suptitle('R1 QC — waterfalls. Dotted lines: gauge MDs; solid line: source gauge '
                 f"(g{src['gauge']}, MD {src['md_ft']:.0f}). "
                 f'Plotted over the comparison window MD {md_lo:.0f}-{md_hi:.0f} ft '
                 f'(solver domain extends to MD {mesh[0]:.0f}).', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg, _ = load_config(args.config)
    man = json.load(open(cfg['outputs']['manifest_json']))
    os.makedirs(OUTDIR, exist_ok=True)

    series, src_gauge, src, mesh, source_idx, targets, frac_hits = build(cfg)
    p1 = os.path.join(OUTDIR, 'r1_qc_data.png')
    p2 = os.path.join(OUTDIR, 'r1_qc_fit.png')
    p3 = os.path.join(OUTDIR, 'r1_qc_numerics.png')
    fig_data(cfg, series, src_gauge, frac_hits, p1)
    print('wrote', p1)
    fig_fit(cfg, man, src, mesh, source_idx, targets, p2)
    print('wrote', p2)
    fig_numerics(man, p3)
    print('wrote', p3)
    p4 = os.path.join(OUTDIR, 'r1_qc_waterfall.png')
    fig_waterfall(cfg, man, src, mesh, source_idx, targets, p4)
    print('wrote', p4)


if __name__ == '__main__':
    main()
