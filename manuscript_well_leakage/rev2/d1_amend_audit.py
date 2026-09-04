"""Traceability audit for the amended D1 README.

Every number the amended `output/rev2_20260901/D1/README.md` quotes is
recomputed here from the file that backs it, and printed next to that file's
path. No solver is run: this reads CSV/JSON only, so it needs no manifest and
can be re-run by a reviewer in a second.

    python scripts/manuscript_well_leakage/rev2/d1_amend_audit.py
"""

import csv
import json
import os

import numpy as np

D1 = 'output/rev2_20260901/D1'
AM = os.path.join(D1, 'amend')
SUMMARY = os.path.join(D1, 'd1_summary_v1.csv')
CALIB = os.path.join(D1, 'd1_calibrations_v1.csv')
COLD = os.path.join(AM, 'd1_coldstart_ensemble_v2.json')
BEST = os.path.join(AM, 'd1_twozone_bestfit_blind_v2.csv')
PROF = os.path.join(AM, 'd1_profile_identifiability_v2.json')
IDV1 = os.path.join(D1, 'd1_identifiability_v1.json')
MAN_U = os.path.join(D1, 'manifest_uniform.json')
MAN_T = os.path.join(D1, 'manifest_two_zone.json')
GNUMS = [2, 3, 4, 5, 6, 7]


def line(claim, value, src):
    print(f"  {claim:<62s} {value:<34s} <- {src}")


def main():
    rows = list(csv.DictReader(open(SUMMARY)))
    best = list(csv.DictReader(open(BEST)))
    cold = json.load(open(COLD))
    mt = json.load(open(MAN_T))
    agg = dict(mt['results']['aggregate_leave_one_out'])
    agg.update(mt['results']['aggregate_leave_one_out_other_family'])

    print("=" * 118)
    print("A. UNIFORM BASELINE (deterministic grid search: no optimiser dependence)")
    for norm in ('absolute', 'normalised'):
        a = agg[f'uniform|{norm}']
        line(f"uniform {norm}: LOO blind gauge-mean RMSE",
             f"{a['loo_blind_rmse_gaugemean_psi']:.3f} psi", MAN_U)
        line(f"uniform {norm}: in-calibration gauge-mean RMSE",
             f"{a['incal_rmse_gaugemean_psi']:.3f} psi", MAN_U)
        line(f"uniform {norm}: usable blind predictions / first failure",
             f"{a['n_usable_blind']}/6, {a['first_failing_distance_ft']} ft", MAN_U)
    for norm in ('absolute', 'normalised'):
        rs = [r for r in rows if r['model'] == 'uniform' and r['norm'] == norm]
        thr = 0.25
        marg = [(int(r['held_out_gauge']), float(r['blind_rmse_norm']),
                 100 * (float(r['blind_rmse_norm']) - thr) / thr) for r in rs]
        line(f"uniform {norm}: g3 blind RMSE/peak and margin vs 0.25",
             f"{marg[1][1]:.5f} ({marg[1][2]:+.1f}%)", SUMMARY)
        fails = [m for m in marg if m[1] > thr]
        line(f"uniform {norm}: gauges failing the RMSE/peak test",
             ",".join(f"g{m[0]}" for m in fails), SUMMARY)
        amps = {int(r['held_out_gauge']): float(r['blind_amp_ratio']) for r in rs}
        line(f"uniform {norm}: g7 amplitude ratio", f"{amps[7]:.3f}", SUMMARY)
        f = [float(r['blind_over_floor']) for r in rs]
        line(f"uniform {norm}: blind / single-gauge floor range",
             f"{min(f):.2f}x - {max(f):.2f}x", SUMMARY)
    fl = {int(r['held_out_gauge']): float(r['single_gauge_floor_rmse_psi'])
          for r in rows if r['model'] == 'uniform' and r['norm'] == 'absolute'}
    line("single-gauge uniform floor range",
         f"{min(fl.values()):.2f} - {max(fl.values()):.2f} psi", SUMMARY)
    cal_u = [r for r in csv.DictReader(open(CALIB))
             if r['model'] == 'uniform' and r['norm'] == 'absolute'
             and r['subset'] != 'all']
    du = sorted(float(r['params_physical']) for r in cal_u)
    line("uniform absolute LOO refit-D range",
         f"{du[0]:.1f} - {du[-1]:.1f} ft^2/s", CALIB)

    print("=" * 118)
    print("B. TWO_ZONE LEAVE-ONE-OUT (stochastic search: quote ranges, not points)")
    for norm in ('absolute', 'normalised'):
        a = cold['aggregate'][norm]
        rs = [r for r in best if r['norm'] == norm]
        b = np.array([float(r['blind_rmse_psi']) for r in rs])
        mn = np.array([float(r['blind_rmse_psi_min_within_1pct']) for r in rs])
        mx = np.array([float(r['blind_rmse_psi_max_within_1pct']) for r in rs])
        env = (float(np.sqrt(np.mean(mn ** 2))), float(np.sqrt(np.mean(mx ** 2))))
        line(f"two_zone {norm}: published warm-started LOO blind",
             f"{a['published_warmstart_value_psi']:.3f} psi", MAN_T)
        line(f"two_zone {norm}: best-of-search LOO blind",
             f"{float(np.sqrt(np.mean(b**2))):.3f} psi", BEST)
        line(f"two_zone {norm}: envelope over fits within 1% of best",
             f"{env[0]:.2f} - {env[1]:.2f} psi", BEST)
        line(f"two_zone {norm}: per-restart LOO blind (5 cold restarts)",
             f"{min(a['loo_blind_rmse_gaugemean_psi_per_restart']):.2f} - "
             f"{max(a['loo_blind_rmse_gaugemean_psi_per_restart']):.2f} psi", COLD)
        u = agg[f'uniform|{norm}']['loo_blind_rmse_gaugemean_psi']
        line(f"two_zone {norm}: improvement factor over uniform ({u:.2f} psi)",
             f"{u/env[1]:.1f}x - {u/env[0]:.1f}x "
             f"(best-of-search {u/float(np.sqrt(np.mean(b**2))):.1f}x)", BEST)
        line(f"two_zone {norm}: usable blind predictions at best fits",
             f"{sum(r['usable_blind_prediction']=='True' for r in rs)}/6", BEST)
        line(f"two_zone {norm}: blind RMSE range at best fits",
             f"{b.min():.2f} - {b.max():.2f} psi", BEST)
        pk = [float(r['blind_rmse_norm']) for r in rs]
        line(f"two_zone {norm}: blind RMSE as % of peak at best fits",
             f"{100*min(pk):.1f}% - {100*max(pk):.1f}%", BEST)
        amp = [float(r['blind_amp_ratio']) for r in rs]
        line(f"two_zone {norm}: blind amplitude ratio range at best fits",
             f"{min(amp):.2f} - {max(amp):.2f}", BEST)
        fo = [float(r['blind_over_floor']) for r in rs]
        line(f"two_zone {norm}: blind / single-gauge floor at best fits",
             f"{min(fo):.2f}x - {max(fo):.2f}x", BEST)
    g3 = [r for r in best if r['norm'] == 'absolute'
          and r['held_out_gauge'] == '3'][0]
    line("drop_g3 absolute: published vs best five-gauge criterion",
         f"{float(g3['published_warmstart_criterion']):.5f} vs "
         f"{float(g3['best_coldstart_criterion']):.5f} psi", BEST)
    line("drop_g3 absolute: blind RMSE at those two fits",
         f"{float(g3['published_warmstart_blind_rmse_psi']):.2f} vs "
         f"{float(g3['blind_rmse_psi']):.2f} psi", BEST)
    line("drop_g3 absolute: blind spread within 1% of best criterion",
         f"{float(g3['blind_rmse_psi_min_within_1pct']):.2f} - "
         f"{float(g3['blind_rmse_psi_max_within_1pct']):.2f} psi", BEST)

    print("=" * 118)
    print("C. COLD-START ANCHOR (replaces the circular warm-started 'reproduction')")
    ca = cold['best_of_search']['absolute|all']
    line("cold-start all-six-gauge two_zone criterion",
         f"{ca['criterion']:.5f} psi (published 11.87195)", COLD)
    line("cold-start all-six-gauge two_zone parameters",
         ";".join(f"{x:.4g}" for x in ca['params_physical']), COLD)
    cal = {(r['norm'], r['subset']): r for r in csv.DictReader(open(CALIB))
           if r['model'] == 'two_zone'}
    ws = json.load(open('configs/rev2/d1_loo_blind.json'
                        ))['families']['two_zone']['warm_start']
    pub = json.load(open(MAN_T))['results']['calibrations']['absolute|all']['params']
    line("published absolute|all params == config warm_start",
         str(all(a == b for a, b in zip(ws, pub))),
         'configs/rev2/d1_loo_blind.json + ' + MAN_T)

    print("=" * 118)
    print("D. IDENTIFIABILITY: published one-at-a-time slice vs profile likelihood")
    v1 = json.load(open(IDV1))['summary']
    for k in ('absolute|drop_g2|log10_D_near', 'normalised|drop_g2|log10_D_near',
              'absolute|drop_g2|log10_s_c', 'normalised|drop_g2|log10_s_c'):
        s = v1[k]
        line(f"v1 FROZEN {k}: +10% band / +1% band",
             f"{s['10pct']['decades']:.4f} dec / {s['1pct']['decades']:.4f} dec", IDV1)
    grid = json.load(open(IDV1))['scans']['absolute|drop_g2|log10_D_near']['grid_log10']
    line("v1 scan step (log10_D_near)",
         f"{grid[1]-grid[0]:.4f} decades, {len(grid)} points over the full bounds",
         IDV1)
    line("v1 reference for absolute|all|log10_s_c vs the fitted optimum",
         f"{v1['absolute|all|log10_s_c']['criterion_at_argmin']:.4f} vs 11.8719 psi",
         IDV1)
    if os.path.exists(PROF):
        p = json.load(open(PROF))['summary']
        for k in sorted(p):
            s = p[k]
            b10, b1 = s['10pct'], s['1pct']
            cens = ('CENSORED' if b10['censored_low'] or b10['censored_high']
                    else 'not censored')
            line(f"PROFILED {k}: +10% band",
                 f"{b10['physical_interval'][0]:.0f}-{b10['physical_interval'][1]:.0f}"
                 f" ({b10['decades']:.2f} dec, {cens})", PROF)
            line(f"PROFILED {k}: +1% band",
                 f"{b1['physical_interval'][0]:.0f}-{b1['physical_interval'][1]:.0f}"
                 f" ({b1['decades']:.2f} dec)", PROF)
            line(f"PROFILED {k}: blind g2 RMSE over the +10% band",
                 f"{b10['blind_rmse_g_range_psi'][0]:.1f}-"
                 f"{b10['blind_rmse_g_range_psi'][1]:.1f} psi", PROF)
    else:
        print(f"  (profile output {PROF} not present)")

    print("=" * 118)
    print("E. WIDTH INSTABILITY (v1 negative result 4)")
    for norm in ('absolute',):
        w = [(int(r['held_out_gauge']),
              float(r['params_refit_5gauge'].split(';')[3])) for r in rows
             if r['model'] == 'two_zone' and r['norm'] == norm]
        wall = float([r['params_all_gauge'] for r in rows
                      if r['model'] == 'two_zone' and r['norm'] == norm
                      ][0].split(';')[3])
        line(f"two_zone {norm}: refit w range vs all-gauge {wall:.2f} ft",
             f"{min(x for _, x in w):.2f} - {max(x for _, x in w):.2f} ft", SUMMARY)
        sh = [r['param_shift_pct'].split(';')[3] for r in rows
              if r['model'] == 'two_zone' and r['norm'] == norm]
        line(f"two_zone {norm}: per-subset w shift (%)", ";".join(sh), SUMMARY)
    lw = float(cal[('absolute', 'drop_g2')]['params'].split(';')[3])
    line("drop_g2 absolute log10_w vs its lower search bound 0.5",
         f"{lw:.10f} (gap {lw-0.5:.2e} dec, w is "
         f"{100*(10**lw/10**0.5-1):.3f}% above the bound)", CALIB)

    print("=" * 118)
    print("F. RUN PROVENANCE")
    for f in ('manifest_uniform.json', 'manifest_two_zone.json',
              'manifest_identifiability.json'):
        line(f"v1 {f} run_utc",
             json.load(open(os.path.join(D1, f)))['run_utc'], os.path.join(D1, f))
    for f in ('d1_loo_blind.py', 'd1_nearzone_identifiability.py'):
        p = os.path.join('scripts/manuscript_well_leakage/rev2', f)
        import datetime
        line(f"{f} mtime (UTC)",
             datetime.datetime.fromtimestamp(
                 os.path.getmtime(p), datetime.timezone.utc).isoformat(), p)
    line("amend cold-start ensemble wall time",
         f"{cold['wall_seconds']:.0f} s", COLD)
    # The one known provenance blemish, made checkable rather than asserted:
    # manifest_profile_identifiability.json records the config FILE-BYTES hash as
    # it stood while two inert output-path keys were temporarily present. The
    # config it actually resolved is embedded in the manifest and is identical to
    # the file on disk today.
    pm = os.path.join(AM, 'manifest_profile_identifiability.json')
    if os.path.exists(pm):
        m = json.load(open(pm))
        cur = json.load(open('configs/rev2/d1_amend.json'))
        line("profile manifest: embedded resolved config == config on disk",
             str(m['config']['resolved'] == cur), pm)
        line("profile manifest: recorded config file-bytes sha vs on-disk sha",
             f"{m['config']['sha256'][:12]} vs "
             f"{__import__('hashlib').sha256(open('configs/rev2/d1_amend.json','rb').read()).hexdigest()[:12]}",
             pm)
    if os.path.exists(PROF):
        line("amend profile scan wall time",
             f"{json.load(open(PROF))['wall_seconds']:.0f} s", PROF)
    print("=" * 118)


if __name__ == '__main__':
    main()
