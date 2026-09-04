"""E2 -- aggregation, figures and the rebuilt Fig. 7b.

Separate from `e2_fig7b.py` on purpose: that file is hashed into all 70 run
manifests (`rev2_manifest.code_closure` includes `__main__`), so editing it after
the runs would make every one of them report code drift. This module only reads
what those runs produced.

Every legend string on every figure comes from `e2_fig7b.label_from_manifest`,
i.e. out of the run's own manifest. Nothing is hard-coded and nothing is ordered
by `os.listdir` -- those are defects 1 and 2 of the figure being rebuilt.

Run:
    python3 scripts/manuscript_well_leakage/rev2/e2_fig7b_figures.py \
        --config configs/rev2/e2_fig7b.json
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.join(os.path.dirname(_HERE), 'baseline_calibration')
for _p in (_HERE, _BASE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rev2_core as rc          # noqa: E402
import rev2_data as rd          # noqa: E402
import rev2_manifest as rm      # noqa: E402
import e2_fig7b as e2           # noqa: E402

DPI = 300
D2 = '2020-06-01'      # what Fig. 7b actually plots (select_time to 2020-06-01)
D15 = '2021-07-01'     # what the runs are integrated to

C_FIELD = '#111111'
C_UNIF = '#0072B2'
C_R3 = '#D55E00'
C_R5 = '#009E73'
C_LEG = '#7f7f7f'


def load_runs(outdir):
    with open(os.path.join(outdir, 'e2_runs.json')) as fh:
        blob = json.load(fh)
    for r in blob['runs']:
        r['manifest_abs'] = os.path.join(rd.REPO_ROOT, r['manifest'])
        r['legend'] = e2.label_from_manifest(r['manifest_abs'])
    return blob


def pick(runs, **kw):
    out = [r for r in runs
           if all(abs(r[k] - v) <= 1e-12 * max(1.0, abs(v))
                  if isinstance(v, (int, float)) and not isinstance(v, bool)
                  else r[k] == v
                  for k, v in kw.items())]
    return out


def one(runs, **kw):
    got = pick(runs, **kw)
    if len(got) != 1:
        raise KeyError(f"expected exactly one run for {kw}, got {len(got)}")
    return got[0]


def dd(run, date=D2):
    return np.asarray(run['drawdown'][date]['sim_drawdown_psi'], dtype=float)


# ---------------------------------------------------------------------------
# convergence analysis
# ---------------------------------------------------------------------------

def pad_convergence(runs, date, tol_psi=1.0, tol_frac=0.005):
    """Smallest tested pad at which every gauge and every barrier strength has
    stopped moving, judged against the largest pad tested.

    Two tolerances because the quantity spans 0.02-3300 psi across the sweep: a
    change counts as settled if it is below `tol_psi` OR below `tol_frac` of the
    reference value, whichever is the easier bar at that gauge.
    """
    sel = [r for r in runs if r['sweep'] == 'S3_pad']
    pads = sorted({r['pad_ft'] for r in sel})
    ratios = sorted({r['ratio'] for r in sel})
    ref_pad = pads[-1]
    table, settled = {}, {}
    for ratio in ratios:
        ref = dd(one(sel, pad_ft=ref_pad, ratio=ratio), date)
        denom = np.maximum(np.abs(ref), 1.0)
        rows = []
        for p in pads:
            v = dd(one(sel, pad_ft=p, ratio=ratio), date)
            d = np.abs(v - ref)
            rows.append({'pad_ft': p,
                         'max_abs_change_psi': float(np.max(d)),
                         'max_rel_change': float(np.max(d / denom)),
                         'gauge_of_max': int(np.argmax(d)) + 1,
                         'drawdown_psi': v.tolist()})
        table['%g' % ratio] = rows
        settled['%g' % ratio] = {r['pad_ft']: (r['max_abs_change_psi'] < tol_psi
                                               or r['max_rel_change'] < tol_frac)
                                 for r in rows}
    verdict = None
    for p in pads:
        if all(settled['%g' % r][p] for r in ratios):
            verdict = p
            break
    return {'reference_pad_ft': ref_pad, 'tol_psi': tol_psi,
            'tol_frac': tol_frac, 'converged_pad_ft': verdict,
            'pads_tested_ft': pads, 'per_ratio': table}


def step_convergence(runs, sweep, key, date=D2):
    """Change in the plotted profile against the FINEST setting of `key`."""
    sel = [r for r in runs if r['sweep'] == sweep]
    out = {}
    for ratio in sorted({r['ratio'] for r in sel}, reverse=True):
        rows = sorted([r for r in sel if r['ratio'] == ratio],
                      key=lambda r: r[key])
        base = dd(rows[0], date)          # rows[0] is the finest
        denom = np.maximum(np.abs(base), 1.0)
        out['%g' % ratio] = [
            {key: r[key], 'nx': r['nx'], 'n_steps': r['n_steps'],
             'profile_rmse_psi': r['drawdown'][date]['profile_rmse_psi'],
             'max_abs_change_vs_finest_psi':
                 float(np.max(np.abs(dd(r, date) - base))),
             'max_rel_change_vs_finest':
                 float(np.max(np.abs(dd(r, date) - base) / denom))}
            for r in rows]
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _annotate_gauges(ax, md, numbers, y, every=1):
    for k, (m, n) in enumerate(zip(md, numbers)):
        if k % every == 0:
            ax.annotate(str(n), (m, y[k]), textcoords='offset points',
                        xytext=(0, 7), ha='center', fontsize=6, color=C_LEG)


def fig01(blob, runs, md, numbers, obs, pad, outpath):
    """The rebuilt Fig. 7b."""
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(md, obs, 'o-', color=C_FIELD, lw=1.8, ms=5, zorder=5,
            label='Field data (S-well gauges, numeric order)')
    for ratio, color in ((1.0, C_UNIF), (1e-3, C_R3), (1e-5, C_R5)):
        r = one(runs, sweep='S4_ratio', pad_ft=pad, ratio=ratio)
        lab = r['legend']['label']
        rmse = r['drawdown'][D2]['profile_rmse_psi']
        ax.plot(md, dd(r), 's--' if ratio == 1.0 else '^-', color=color,
                lw=1.5, ms=4, label=f"{lab}\n    profile RMSE {rmse:.0f} psi")
    _annotate_gauges(ax, md, numbers, obs)
    ax.set_xlabel('Measured depth along the S well (ft)')
    ax.set_ylabel('Pressure drawdown, 2020-04-01 to 2020-06-01 (psi)')
    ax.set_title('Fig. 7b rebuilt: production-period drawdown profile\n'
                 f'converged domain, {int(pad)} ft pad at both ends '
                 f'(nx = {one(runs, sweep="S4_ratio", pad_ft=pad, ratio=1.0)["nx"]})',
                 fontsize=10)
    ax.invert_xaxis()
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.55 * (hi - lo), hi)
    ax.axhline(0.0, color=C_LEG, lw=0.6)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc='lower left', framealpha=0.95)
    ax.text(0.99, 0.02, 'gauge numbers annotated above the field curve',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6,
            color=C_LEG)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def fig02(blob, runs, md, numbers, obs, pad, outpath):
    """The same on the manuscript's 'Gauge Number' axis, with the ordering
    defect drawn."""
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    axis = np.arange(1, len(numbers) + 1)
    ax.plot(axis, obs, 'o-', color=C_FIELD, lw=1.8, ms=5, zorder=5,
            label='Field data, NUMERIC gauge order (correct)')
    order = blob['listdir_order_today']
    field_by_gauge = dict(zip(numbers, obs))
    wrong = [field_by_gauge[g] for g in order]
    ax.plot(axis, wrong, 'x:', color='#CC79A7', lw=1.4, ms=6,
            label=('Field data, os.listdir order (what the manuscript plotted;\n'
                   '    this filesystem today: ' + ', '.join(map(str, order))
                   + ')'))
    for ratio, color in ((1.0, C_UNIF), (1e-3, C_R3), (1e-5, C_R5)):
        r = one(runs, sweep='S4_ratio', pad_ft=pad, ratio=ratio)
        ax.plot(axis, dd(r), 's--' if ratio == 1.0 else '^-', color=color,
                lw=1.4, ms=4, label=r['legend']['label'])
    ax.set_xlabel('Gauge Number')
    ax.set_ylabel('Pressure drawdown, 2020-04-01 to 2020-06-01 (psi)')
    ax.set_title('Defect 2: the field and model curves were built on different\n'
                 'orderings of one "Gauge Number" axis', fontsize=10)
    ax.set_xticks(axis)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.75 * (hi - lo), hi)
    ax.axhline(0.0, color=C_LEG, lw=0.6)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc='lower left', framealpha=0.95)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def fig03(runs, numbers, conv2, conv15, outpath):
    sel = [r for r in runs if r['sweep'] == 'S3_pad']
    pads = sorted({r['pad_ft'] for r in sel})
    ratios = sorted({r['ratio'] for r in sel}, reverse=True)
    show = [(0, 'g1 (MD 16645)'), (7, 'g8 (MD 14821)'), (14, 'g15 (MD 12098)')]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4), sharex=True)
    for j, (date, conv, rowlab) in enumerate(
            [(D2, conv2, 'two-month drawdown (what Fig. 7b plots)'),
             (D15, conv15, 'fifteen-month drawdown')]):
        for i, (gi, glab) in enumerate(show):
            ax = axes[j, i]
            for ratio, color in zip(ratios, (C_UNIF, C_R3, C_R5)):
                y = [dd(one(sel, pad_ft=p, ratio=ratio), date)[gi] for p in pads]
                ax.plot([max(p, 500.0) for p in pads], y, 'o-', color=color,
                        ms=3.5, lw=1.3, label=f'ratio {ratio:g}')
            ax.set_xscale('log')
            ax.grid(alpha=0.25)
            if conv['converged_pad_ft'] is not None:
                ax.axvline(conv['converged_pad_ft'], color=C_LEG, ls='--', lw=1)
            if j == 1:
                ax.set_xlabel('symmetric pad (ft; 0 plotted at 500)')
            if i == 0:
                ax.set_ylabel(f'{rowlab}\n(psi)', fontsize=8)
            ax.set_title(glab, fontsize=9)
            if i == 2 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle('Defect 4: the legacy 103r box is 5000 ft wide; the answer stops '
                 'moving near the diffusion length', fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def fig04(runs, md, obs, pad, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    sel = [r for r in runs if r['sweep'] == 'S3_pad']
    for ax, ratio, color in zip(axes, (1.0, 1e-3), (C_UNIF, C_R3)):
        ax.plot(md, obs, 'o-', color=C_FIELD, lw=1.6, ms=4, label='Field data')
        for p, ls, alpha in ((0.0, ':', 0.55), (5000.0, '-.', 0.75),
                             (pad, '-', 1.0)):
            r = one(sel, pad_ft=p, ratio=ratio)
            ax.plot(md, dd(r), ls, color=color, lw=1.6, alpha=alpha,
                    label=(f"pad {int(p)} ft, RMSE "
                           f"{r['drawdown'][D2]['profile_rmse_psi']:.0f} psi"))
        ax.set_title(one(sel, pad_ft=pad, ratio=ratio)['legend']['label'],
                     fontsize=9)
        ax.set_xlabel('Measured depth (ft)')
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('Two-month drawdown (psi)')
    fig.suptitle('The two curves Fig. 7b plots, in the legacy box and on the '
                 'converged domain', fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def fig05(runs, pad, obs, outpath):
    sel = sorted([r for r in runs if r['sweep'] == 'S4_ratio'],
                 key=lambda r: r['ratio'])
    ratios = [r['ratio'] for r in sel]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    for date, color, lab in ((D2, C_R3, 'two-month (Fig. 7b)'),
                             (D15, C_R5, 'fifteen-month')):
        ax.plot(ratios, [r['drawdown'][date]['profile_rmse_psi'] for r in sel],
                'o-', color=color, ms=4, label=lab)
    unif = one(sel, ratio=1.0)
    ax.axhline(unif['drawdown'][D2]['profile_rmse_psi'], color=C_R3, ls='--',
               lw=1.1, label='uniform, two-month')
    ax.axhline(unif['drawdown'][D15]['profile_rmse_psi'], color=C_R5, ls='--',
               lw=1.1, label='uniform, fifteen-month')
    ax.axhline(float(np.std(obs)), color=C_FIELD, ls=':', lw=1.6,
               label=('best CONSTANT fitted to the field profile\n'
                      '    (%.0f psi) -- every model is worse'
                      % float(np.std(obs))))
    ax.set_xscale('log')
    ax.set_xlabel('barrier reduction ratio $D_b/D_0$')
    ax.set_ylabel('drawdown-profile RMSE over the 15 gauges (psi)')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    ax.set_title('No barrier strength beats the uniform model', fontsize=9)
    ax = axes[1]
    ax.plot(ratios, [r['pooled_timeseries']['sample_pooled_rmse_psi']
                     for r in sel], 'o-', color=C_UNIF, ms=4,
            label='sample-pooled time-series RMSE')
    ax.plot(ratios, [r['pooled_timeseries']['gauge_mean_rmse_psi'] for r in sel],
            's--', color=C_R3, ms=4, label='gauge-mean time-series RMSE')
    ax.set_xscale('log')
    ax.set_xlabel('barrier reduction ratio $D_b/D_0$')
    ax.set_ylabel('time-series RMSE over the whole record (psi)')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    ax.set_title('Same verdict on the full time series', fontsize=9)
    fig.suptitle(f'Barrier-strength sweep on the converged domain '
                 f'({int(pad)} ft pad)', fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def fig06(runs, dtconv, dxconv, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    for ratio, color in zip(sorted(dtconv, key=float, reverse=True), (C_UNIF, C_R3)):
        rows = dtconv[ratio]
        ax.plot([r['dt_s'] for r in rows],
                [max(r['max_abs_change_vs_finest_psi'], 1e-4) for r in rows],
                'o-', color=color, ms=4, label=f'ratio {ratio}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('time step (s)')
    ax.set_ylabel('max |change| in two-month drawdown\nvs the finest dt (psi)')
    ax.axvline(3600.0, color=C_LEG, ls='--', lw=1)
    ax.axvline(360000.0, color='#CC79A7', ls=':', lw=1.2)
    ax.grid(alpha=0.25, which='both')
    ax.legend(fontsize=7)
    ax.set_title('dt: legacy 360000 s (dotted) vs the 3600 s used here (dashed)',
                 fontsize=9)
    ax = axes[1]
    for ratio, color in zip(sorted(dxconv, key=float, reverse=True), (C_UNIF, C_R3, C_R5)):
        rows = dxconv[ratio]
        ax.plot([r['dx_bg_ft'] for r in rows],
                [max(r['max_abs_change_vs_finest_psi'], 1e-4) for r in rows],
                'o-', color=color, ms=4, label=f'ratio {ratio}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('background dx (ft); dx_fine = dx/100')
    ax.set_ylabel('max |change| in two-month drawdown\nvs the finest mesh (psi)')
    ax.grid(alpha=0.25, which='both')
    ax.legend(fontsize=7)
    ax.set_title('mesh', fontsize=9)
    fig.suptitle('Discretisation convergence at the converged domain', fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


def fig07(runs, md, obs, pad, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    for ax, ratio, color in zip(axes, (1.0, 1e-3), (C_UNIF, C_R3)):
        ax.plot(md, obs, 'o-', color=C_FIELD, lw=1.6, ms=4, label='Field data')
        base = one(runs, sweep='S4_ratio', pad_ft=pad, ratio=ratio)
        ax.plot(md, dd(base), '-', color=color, lw=1.7,
                label=(f"driver prod g1 (MD 16196), source MD 16196\n"
                       f"    RMSE {base['drawdown'][D2]['profile_rmse_psi']:.0f} psi"))
        for r in [x for x in runs if x['sweep'] == 'S5_source'
                  and abs(x['ratio'] - ratio) < 1e-12]:
            ax.plot(md, dd(r), '--', lw=1.3,
                    label=(f"driver prod g{r['driver_gauge']} "
                           f"(MD {r['driver_md_ft']:.0f}), source MD "
                           f"{r['source_md_requested_ft']:.0f}\n"
                           f"    RMSE "
                           f"{r['drawdown'][D2]['profile_rmse_psi']:.0f} psi"))
        ax.set_title(base['legend']['label'], fontsize=9)
        ax.set_xlabel('Measured depth (ft)')
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5)
    axes[0].set_ylabel('Two-month drawdown (psi)')
    fig.suptitle('Defect 3 controls: driver identity and source placement',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# aggregate manifest
# ---------------------------------------------------------------------------

def write_figure_manifest(cfg, config_path, figdir, products, primary, notes):
    """One aggregate manifest for the figure set.

    Its `source_protocol` and `numerics` groups are REBUILT for the primary
    converged run (`primary`) from that run's own summary.npz, and are labelled as
    such: they describe that solve, not the figures, exactly as B2's case-level
    manifests do.
    """
    z = np.load(os.path.join(rd.REPO_ROOT, primary['summary_npz']))
    x = np.asarray(z['mesh_md_ft'], dtype=float)
    taxis = np.asarray(z['taxis_s'], dtype=float)
    dt_ax = np.asarray(z['driver_taxis_s'], dtype=float)
    dt_v = np.asarray(z['driver_psi'], dtype=float)
    gnum = [int(v) for v in z['gauge_numbers']]
    gmd = [float(v) for v in z['gauge_md_ft']]
    gidx = [int(v) for v in z['gauge_mesh_idx']]
    sidx = int(z['source_mesh_idx'][0])
    m = cfg['model']
    drv = m['drivers'][primary['driver']]

    mpath = os.path.join(figdir, 'manifest.json')
    rm.assert_absent([mpath])
    with rm.RunRecorder(mpath, study_id=cfg['study_id'], task_id=cfg['task_id'],
                        config=cfg, config_path=config_path,
                        run_label='figures_and_aggregate',
                        require_modules=('rev2_core', 'rev2_data',
                                         'e2_fig7b')) as rec:
        drec = rm.driver_record(
            kind='gauge_series', baseline_removal='none_absolute_psi',
            value_units='psi', series_path=rd.repo_path(drv['path']),
            gauge_number=int(drv['gauge']), gauge_md_ft=float(drv['md_ft']),
            taxis=dt_ax, values=dt_v,
            time_start=m['window']['t_start'], time_end=m['window']['t_end'])
        items = [(rd.repo_path(rd.SWELL_GAUGE_MD_NPZ), 'geometry',
                  'gauge_md_swell'),
                 (rd.repo_path(rd.PROD_GAUGE_MD_NPZ), 'geometry',
                  'gauge_md_prod'),
                 (os.path.join(rd.REPO_ROOT, primary['summary_npz']),
                  'prior_run_output', 'primary_run_summary'),
                 (os.path.join(rd.REPO_ROOT, primary['manifest']),
                  'prior_run_output', 'primary_run_manifest')]
        for f in sorted(os.listdir(rd.repo_path(e2.FRAC_HIT_DIR))):
            items.append((rd.repo_path(e2.FRAC_HIT_DIR, f), 'geometry',
                          os.path.splitext(f)[0]))
        for g in gnum:
            items.append((rd.repo_path(e2.SWELL_GAUGE.format(n=g)),
                          'gauge_series', f's_well_gauge{g}'))
        items.append((rd.repo_path(drv['path']), 'gauge_series',
                      f"driver_{primary['driver']}"))
        rec.declare_inputs(items)
        for path, role, dpi, note in products:
            rec.declare_output(path, role=role, dpi=dpi, note=note)
        rec.set_source(rm.source_protocol(
            application='dirichlet_node',
            solver_class='rev2_core.solve_forward',
            placement_rule=(f"copied from the primary converged run "
                            f"{primary['sweep']}/{primary['tag']}: pinned to "
                            f"physical MD {primary['source_md_requested_ft']:.1f} ft"),
            sources=[rm.source_record(x,
                                      md_requested_ft=primary['source_md_requested_ft'],
                                      mesh_idx=sidx, driver=drec,
                                      label='prod_dirichlet',
                                      index_in_source_list=0)],
            targets=[{'gauge': g, 'md_ft': md, 'mesh_idx': i, 'well': 's_well'}
                     for g, md, i in zip(gnum, gmd, gidx)],
            time_level='n', phase_chaining='none',
            boundary_conditions=e2.boundary_group()))
        rec.set_numerics(rm.numerics(
            time=rm.time_record(taxis, mode='fixed', theta=1.0,
                                t_total_requested_s=float(dt_ax[-1]),
                                dt_requested_s=primary['dt_s'],
                                source_time_level='n',
                                label='primary_converged_run'),
            mesh=rm.mesh_record(x, dx_requested_ft=primary['dx_bg_ft'],
                                window_md_ft=[float(v)
                                              for v in m['core_md_ft']],
                                pad_low_ft=primary['pad_ft'],
                                pad_high_ft=primary['pad_ft'],
                                refinement={'mode': 'see the primary run manifest',
                                            'n_degenerate_calls': 0}),
            interface_avg='harmonic', boundary=e2.boundary_group(),
            diffusivity={'family': 'uniform_with_frac_hit_barriers',
                         'D_ft2_s': primary['D_ft2_s']},
            barriers=rm.NONE_DECLARED,
            leakage={'lambda_leak': 0.0, 'p0_psi': 0.0},
            kernel=e2.kernel_group(), rng=rm.NONE_DECLARED,
            parallel={'processes': 1, 'note': 'figures drawn serially'}))
        rec.set_results({'role': 'aggregate figure manifest',
                         'primary_run': f"{primary['sweep']}/{primary['tag']}",
                         'n_figures': sum(1 for p in products
                                          if p[1].startswith('figure'))})
        for n in notes:
            rec.note(n)
    return mpath


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/rev2/e2_fig7b.json')
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args(argv)
    config_path = os.path.abspath(args.config)
    with open(config_path) as fh:
        cfg = json.load(fh)
    outdir = args.outdir or rd.repo_path(cfg['output_root'])
    t0 = time.time()

    blob = load_runs(outdir)
    runs = blob['runs']
    numbers = blob['field_drawdown'][D2]['gauge']
    md = np.asarray(blob['field_drawdown'][D2]['md_ft'], dtype=float)
    obs2 = np.asarray(blob['field_drawdown'][D2]['drawdown_psi'], dtype=float)
    obs15 = np.asarray(blob['field_drawdown'][D15]['drawdown_psi'], dtype=float)

    conv2 = pad_convergence(runs, D2)
    conv15 = pad_convergence(runs, D15)
    dtconv = step_convergence(runs, 'S1_dt', 'dt_s')
    dxconv = step_convergence(runs, 'S2_dx', 'dx_bg_ft')
    pad = float(cfg['sweeps']['S4_ratio']['pad_ft'][0])

    figdir = os.path.join(outdir, 'figures')
    os.makedirs(figdir, exist_ok=True)
    P = lambda n: os.path.join(figdir, n)  # noqa: E731
    products = []

    fig01(blob, runs, md, numbers, obs2, pad, P('fig01_e2_fig7b_rebuilt_v1.png'))
    products.append((P('fig01_e2_fig7b_rebuilt_v1.png'), 'figure_png', DPI,
                     'THE deliverable: rebuilt Fig. 7b on the converged domain, '
                     'field profile in numeric gauge order, legends read from '
                     'each run manifest'))
    fig02(blob, runs, md, numbers, obs2, pad, P('fig02_e2_axis_defect_v1.png'))
    products.append((P('fig02_e2_axis_defect_v1.png'), 'figure_png', DPI,
                     'defect 2: os.listdir-ordered vs numeric-ordered field curve '
                     'on one "Gauge Number" axis'))
    fig03(runs, numbers, conv2, conv15, P('fig03_e2_pad_convergence_v1.png'))
    products.append((P('fig03_e2_pad_convergence_v1.png'), 'figure_png', DPI,
                     'defect 4: drawdown vs symmetric pad at three gauges, two '
                     'evaluation horizons, three barrier strengths'))
    fig04(runs, md, obs2, pad, P('fig04_e2_legacy_vs_converged_v1.png'))
    products.append((P('fig04_e2_legacy_vs_converged_v1.png'), 'figure_png', DPI,
                     'the two curves Fig. 7b plots, legacy box vs converged '
                     'domain'))
    fig05(runs, pad, obs2, P('fig05_e2_ratio_sweep_v1.png'))
    products.append((P('fig05_e2_ratio_sweep_v1.png'), 'figure_png', DPI,
                     'profile and time-series misfit against barrier strength'))
    fig06(runs, dtconv, dxconv, P('fig06_e2_discretisation_v1.png'))
    products.append((P('fig06_e2_discretisation_v1.png'), 'figure_png', DPI,
                     'dt and dx convergence'))
    fig07(runs, md, obs2, pad, P('fig07_e2_source_controls_v1.png'))
    products.append((P('fig07_e2_source_controls_v1.png'), 'figure_png', DPI,
                     'defect 3: driver identity and source placement controls'))

    # ---- the profile table ------------------------------------------------
    csv_path = P('e2_fig7b_profile_v1.csv')
    cols = [('gauge', numbers), ('md_ft', md.tolist()),
            ('field_drawdown_2mo_psi', obs2.tolist()),
            ('field_drawdown_15mo_psi', obs15.tolist())]
    for ratio in sorted({r['ratio'] for r in runs if r['sweep'] == 'S4_ratio'},
                        reverse=True):
        r = one(runs, sweep='S4_ratio', pad_ft=pad, ratio=ratio)
        cols.append((f'sim_2mo_ratio_{ratio:g}_psi', dd(r, D2).tolist()))
    for p in sorted({r['pad_ft'] for r in runs if r['sweep'] == 'S3_pad'}):
        for ratio in (1.0, 1e-3, 1e-5):
            r = one(runs, sweep='S3_pad', pad_ft=p, ratio=ratio)
            cols.append((f'sim_2mo_pad{int(p)}_ratio_{ratio:g}_psi',
                         dd(r, D2).tolist()))
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow([c[0] for c in cols])
        for i in range(len(numbers)):
            w.writerow([c[1][i] for c in cols])
    products.append((csv_path, 'csv', None,
                     'every profile drawn or discussed, one row per gauge in '
                     'numeric order'))

    # ---- summary ----------------------------------------------------------
    def row(r):
        return {'sweep': r['sweep'], 'tag': r['tag'], 'pad_ft': r['pad_ft'],
                'ratio': r['ratio'], 'D_ft2_s': r['D_ft2_s'],
                'dt_s': r['dt_s'], 'dx_bg_ft': r['dx_bg_ft'], 'nx': r['nx'],
                'driver_gauge': r['driver_gauge'],
                'source_md_ft': r['source_md_requested_ft'],
                'legend_from_manifest': r['legend']['label'],
                'profile_rmse_2mo_psi': r['drawdown'][D2]['profile_rmse_psi'],
                'profile_rmse_15mo_psi': r['drawdown'][D15]['profile_rmse_psi'],
                'profile_bias_2mo_psi': r['drawdown'][D2]['profile_bias_psi'],
                'sim_range_2mo_psi': r['drawdown'][D2]['sim_profile_range_psi'],
                'timeseries_pooled_rmse_psi':
                    r['pooled_timeseries']['sample_pooled_rmse_psi'],
                'timeseries_gauge_mean_rmse_psi':
                    r['pooled_timeseries']['gauge_mean_rmse_psi'],
                'manifest': r['manifest']}

    unif = one(runs, sweep='S4_ratio', pad_ft=pad, ratio=1.0)
    b3 = one(runs, sweep='S4_ratio', pad_ft=pad, ratio=1e-3)
    b5 = one(runs, sweep='S4_ratio', pad_ft=pad, ratio=1e-5)
    unif0 = one(runs, sweep='S3_pad', pad_ft=0.0, ratio=1.0)
    b30 = one(runs, sweep='S3_pad', pad_ft=0.0, ratio=1e-3)
    b50 = one(runs, sweep='S3_pad', pad_ft=0.0, ratio=1e-5)
    best = min([r for r in runs if r['sweep'] == 'S4_ratio'],
               key=lambda r: r['drawdown'][D2]['profile_rmse_psi'])

    summary = {
        'study_id': cfg['study_id'], 'task_id': cfg['task_id'],
        'generated_utc': datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        'n_runs': len(runs),
        'converged_domain': {
            'two_month': conv2, 'fifteen_month': conv15,
            'diffusion_length_ft': {
                'two_month_sqrt_4Dt': float(np.sqrt(
                    4 * 140.0 * (datetime.datetime(2020, 6, 1)
                                 - datetime.datetime(2020, 4, 1)).total_seconds())),
                'fifteen_month_sqrt_4Dt': float(np.sqrt(
                    4 * 140.0 * (datetime.datetime(2021, 7, 1)
                                 - datetime.datetime(2020, 4, 1)).total_seconds()))},
            'pad_used_for_headline_ft': pad},
        'dt_convergence': dtconv,
        'dx_convergence': dxconv,
        'headline': {
            'legacy_box_pad0': {
                'uniform_profile_rmse_2mo_psi':
                    unif0['drawdown'][D2]['profile_rmse_psi'],
                'ratio1e-3_profile_rmse_2mo_psi':
                    b30['drawdown'][D2]['profile_rmse_psi'],
                'ratio1e-5_profile_rmse_2mo_psi':
                    b50['drawdown'][D2]['profile_rmse_psi']},
            'converged': {
                'uniform_profile_rmse_2mo_psi':
                    unif['drawdown'][D2]['profile_rmse_psi'],
                'ratio1e-3_profile_rmse_2mo_psi':
                    b3['drawdown'][D2]['profile_rmse_psi'],
                'ratio1e-5_profile_rmse_2mo_psi':
                    b5['drawdown'][D2]['profile_rmse_psi']},
            'best_fitting_ratio_on_converged_domain': {
                'ratio': best['ratio'],
                'profile_rmse_2mo_psi':
                    best['drawdown'][D2]['profile_rmse_psi']},
            'verdict': (
                'On the converged domain the uniform (no-barrier) model fits the '
                'measured drawdown profile better than every barrier strength '
                'tested, and padding widens the gap. Fig. 7b as captioned is not '
                'supported by these data.')},
        'skill_against_the_simplest_baseline': {
            'field_profile_mean_psi': float(obs2.mean()),
            'field_profile_std_psi': float(obs2.std()),
            'rmse_of_best_constant_psi': float(obs2.std()),
            '_note': ('A constant equal to the field profile\'s own mean is the '
                      'weakest model that can be written down. Its RMSE is the '
                      'profile standard deviation. Every simulated profile on the '
                      'converged domain scores WORSE than that, so on this metric '
                      'Fig. 7b does not discriminate between barrier strengths -- '
                      'it rejects all of them.'),
            'per_model': [
                {'ratio': r['ratio'],
                 'profile_rmse_2mo_psi': r['drawdown'][D2]['profile_rmse_psi'],
                 'r2_vs_best_constant': float(
                     1 - np.sum((np.asarray(r['drawdown'][D2]['sim_drawdown_psi'])
                                 - obs2) ** 2)
                     / np.sum((obs2 - obs2.mean()) ** 2)),
                 'shape_correlation': float(np.corrcoef(
                     np.asarray(r['drawdown'][D2]['sim_drawdown_psi']), obs2)[0, 1]),
                 'demeaned_shape_rmse_psi': float(np.sqrt(np.mean((
                     (np.asarray(r['drawdown'][D2]['sim_drawdown_psi'])
                      - np.mean(r['drawdown'][D2]['sim_drawdown_psi']))
                     - (obs2 - obs2.mean())) ** 2)))}
                for r in sorted([x for x in runs if x['sweep'] == 'S4_ratio'],
                                key=lambda x: -x['ratio'])]},
        'field_drawdown': blob['field_drawdown'],
        'listdir_order_today': blob['listdir_order_today'],
        'runs': [row(r) for r in runs],
    }
    summary_path = P('e2_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    products.append((summary_path, 'other',
                     None, 'aggregated numbers behind every claim in the README'))

    mpath = write_figure_manifest(
        cfg, config_path, figdir, products, unif,
        notes=[('source_protocol and numerics describe the primary converged '
                'run S4_ratio/%s, not the figures; every other run has its own '
                'manifest under output/rev2_20260901/E2/<sweep>/<tag>/'
                % unif['tag']),
               ('barriers is NONE_DECLARED here because the primary run copied '
                'into this manifest is the ratio 1.0 uniform control'),
               ('every legend string on every figure was produced by '
                'e2_fig7b.label_from_manifest reading that run manifest')])
    print(f"figures + aggregate manifest in {figdir}")
    print(f"converged pad (two-month)  : {conv2['converged_pad_ft']}")
    print(f"converged pad (15-month)   : {conv15['converged_pad_ft']}")
    print(f"profile RMSE, converged    : uniform "
          f"{unif['drawdown'][D2]['profile_rmse_psi']:.1f} | 1e-3 "
          f"{b3['drawdown'][D2]['profile_rmse_psi']:.1f} | 1e-5 "
          f"{b5['drawdown'][D2]['profile_rmse_psi']:.1f} psi")
    print(f"profile RMSE, legacy box   : uniform "
          f"{unif0['drawdown'][D2]['profile_rmse_psi']:.1f} | 1e-3 "
          f"{b30['drawdown'][D2]['profile_rmse_psi']:.1f} | 1e-5 "
          f"{b50['drawdown'][D2]['profile_rmse_psi']:.1f} psi")
    print(f"wall {time.time() - t0:.1f} s  -> {mpath}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
