"""Fig. 6 - two-stage figure, D0 = 480 chain, refractive-index-corrected Gamma.

    python3 scripts/manuscript_well_leakage/revision_figs/fig6_d480_two_stage.py
    python3 scripts/manuscript_well_leakage/revision_figs/fig6_d480_two_stage.py --verify-gamma

Run with CWD = repo root.  Supersedes `fig6_history_matching.py` (the live
generator until now) and the standalone `Fig6_panelb_D480_no_scale.*`.

WHAT CHANGED FROM `fig6_history_matching.py`
--------------------------------------------
1.  TRANSFER COEFFICIENT.  The old processing chain computed
    1.55e-6 / (4*pi*0.79*4.09) with 4.09 sitting in the n*L_G slot, i.e. with
    the refractive index omitted.  Equation 1 prints n = 1.4682 and
    L_G = 4.08 m separately and is correct as printed, so the code was short by
    a factor n:

        Gamma  8.94e-9  ->  8.94e-9 / 1.4682  =  6.089088680016346e-09  1/psi

    This is a pure rescaling of the synthetic strain-rate field: no spatial
    pattern, no timing, no sign structure changes.

2.  COLOUR LIMIT, divided by the SAME 1.4682 so the rendering does not move:

        clim  3.889910598773561e-09  ->  2.6494419008129416e-09  strain/s

    `--verify-gamma` renders panel (b) both ways and requires the PNGs to be
    byte-identical.  If they were not, the conversion would not be a pure
    scaling and this script would be wrong.

3.  PANELS (b) AND (d) now come from the D0 = 480 chain (pad 10 000 ft both
    ends, ratio 1e-5, dt 1 s, backward Euler) instead of the D0 = 140 archive.
    Both read the SAME merged chain, which is why (d) has to move with (b).
    Panel (d) still plots Gauge D (gauge 8, MD 14821) - the gauge withheld
    entirely from calibration.  No gauge was switched.

4.  PANELS (a) AND (c) are byte-for-byte the previous ones: the construction
    below is `fig6_history_matching.build` verbatim, on the same figsize, the
    same 4x6 grid and the same frozen layout rect, so they land on the same
    pixels.  `--check-panel-a` asserts that against the previous render.

The chain is NOT re-solved here.  It is read from the arrays that
`d480_two_stage.py --export-chain` persisted from the frozen D0 = 480
configuration; dP/dt is stored Gamma-free precisely so that a change of
transfer coefficient never has to touch the solver.
"""

import argparse
import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import revfig_common as rc                                    # noqa: E402
rc.install_legacy_fiberis_plotting()

import matplotlib.pyplot as plt                               # noqa: E402
from fiberis.analyzer.Data1D import Data1D_PumpingCurve, Data1D_Gauge   # noqa: E402
from fiberis.analyzer.Data2D import Data2D_XT_DSS             # noqa: E402

STEM = 'Fig6_history_matching'
DEFAULT_OUTDIR = 'figs/manuscript_revision_v2'
PREVIOUS_RENDER = 'figs/manuscript_revision/Fig6_history_matching_no_scale.png'
CHAIN_NPZ = 'output/revision_d480/d480_chain_arrays.npz'

# ---- Task 1: derived in code, never typed -------------------------------
N_REFRACTIVE = 1.4682                     # Eq. 1 of the manuscript
GAMMA_OLD = 8.94e-9                       # 1/psi, refractive index omitted
GAMMA_NEW = GAMMA_OLD / N_REFRACTIVE      # 6.089088680016346e-09
K_LEGACY = 6894.76 / 30e9                 # the P/E coefficient the clim descends from
CLIM_LEGACY = 1e-7
CLIM_OLD = CLIM_LEGACY * GAMMA_OLD / K_LEGACY     # 3.889910598773561e-09
CLIM_NEW = CLIM_LEGACY * GAMMA_NEW / K_LEGACY     # 2.6494419008129416e-09

LAYOUT_RECT = (0.035, 0.055, 0.90, 0.955)         # identical to fig6_history_matching

#: The axes rectangles the PREVIOUS generator (`fig6_history_matching.py`)
#: produced, captured from it directly.  They are pinned rather than recomputed
#: because `tight_layout` is not a pure function of the grid: it measures the
#: artists, and panels (b)/(d) now carry a different time axis, which moved the
#: whole axes block by ~3 % of the figure height.  That would have resized
#: panels (a) and (c) even though nothing in them changed.  Pinning makes
#: "panels (a) and (c) unchanged" exact instead of approximate.
#: Order is figure-axes order: ax1 (a), ax2 (c), ax22 (c twin), ax3 (b), ax4 (d).
PINNED_AXES = (
    (0.04571428571428572, 0.30208333333333337, 0.4164285714285715, 0.5929166666666666),
    (0.04571428571428572, 0.09819444444444447, 0.4164285714285715, 0.18513888888888885),
    (0.04571428571428572, 0.09819444444444447, 0.4164285714285715, 0.18513888888888885),
    (0.4728571428571429, 0.30208333333333337, 0.4164285714285714, 0.5929166666666666),
    (0.4728571428571429, 0.09819444444444447, 0.4164285714285714, 0.18513888888888885),
)


def load_chain():
    if not os.path.exists(CHAIN_NPZ):
        raise SystemExit(
            f"{CHAIN_NPZ} is missing. Regenerate it once with:\n"
            f"  python3 scripts/manuscript_well_leakage/revision_figs/"
            f"d480_two_stage.py --export-chain {CHAIN_NPZ}")
    z = np.load(CHAIN_NPZ, allow_pickle=False)
    return {k: z[k] for k in z.files}


def pin_axes(fig, ds):
    """Force the previous generator's axes rectangles, in both scale modes."""
    if len(fig.axes) != len(PINNED_AXES):
        raise RuntimeError(f"expected {len(PINNED_AXES)} axes, got {len(fig.axes)}")
    for ax, rect in zip(fig.axes, PINNED_AXES):
        ax.set_position(rect)
    ds._positions = [(ax, ax.get_position().frozen()) for ax in fig.axes]


def build(mode, outdir, dpi, gamma_new=True, return_panelb=False):
    ds = rc.DualScale(mode)
    inputs = [CHAIN_NPZ]
    ch = load_chain()

    gamma = GAMMA_NEW if gamma_new else GAMMA_OLD
    clim_b = CLIM_NEW if gamma_new else CLIM_OLD

    # ---- geometry (legacy format, exactly as before) ------------------------
    fhp = "data/legacy/s_well/geometry/frac_hit/"
    for p in [fhp + "frac_hit_stage_7_swell.npz", fhp + "frac_hit_stage_8_swell.npz",
              "data/legacy/s_well/geometry/gauge_md_swell.npz"]:
        inputs.append(p)
    frac_hit_stg7 = np.load(fhp + "frac_hit_stage_7_swell.npz")['data']
    frac_hit_stg8 = np.load(fhp + "frac_hit_stage_8_swell.npz")['data']
    gauge_md_all = np.load("data/legacy/s_well/geometry/gauge_md_swell.npz")['data']
    ind = np.array(np.where(np.logical_and(
        gauge_md_all <= np.max(frac_hit_stg7) + 500,
        gauge_md_all >= np.min(frac_hit_stg8) - 500))).flatten()
    gauge_md = gauge_md_all[ind]

    # ---- field data for (a) and (c) -----------------------------------------
    datapath = "data/fiberis_format/"

    def pc(stage, name):
        p = f"{datapath}prod/pumping_data/stage{stage}/{name}.npz"
        inputs.append(p)
        o = Data1D_PumpingCurve.Data1DPumpingCurve()
        o.load_npz(p)
        return o

    pc_stg7_slurry_rate = pc(7, "Slurry Rate")
    pc_stg7_pressure = pc(7, "Treating Pressure")
    pc_stg8_slurry_rate = pc(8, "Slurry Rate")
    pc_stg8_pressure = pc(8, "Treating Pressure")
    stg7_bgtime = pc_stg7_slurry_rate.get_start_time()
    stg8_edtime = pc_stg8_slurry_rate.get_end_time()

    DASdata = Data2D_XT_DSS.DSS2D()
    first = True
    for p in [datapath + "s_well/DAS/LFDASdata_stg7_swell.npz",
              datapath + "s_well/DAS/LFDASdata_stg7_interval_swell.npz",
              datapath + "s_well/DAS/LFDASdata_stg8_swell.npz"]:
        inputs.append(p)
        tmp = Data2D_XT_DSS.DSS2D()
        tmp.load_npz(p)
        if first:
            DASdata.load_npz(p)
            first = False
        else:
            DASdata.right_merge(tmp)
    DASdata.select_depth(np.min(frac_hit_stg8) - 500, np.max(frac_hit_stg7) + 500)

    gauge_dataframe_all = []
    for it in ind:
        p = f'data/fiberis_format/s_well/gauges/gauge{it + 1}_data_swell.npz'
        inputs.append(p)
        g = Data1D_Gauge.Data1DGauge()
        g.load_npz(p)
        g.crop(stg7_bgtime, stg8_edtime)
        gauge_dataframe_all.append(g)

    # ---- the D0 = 480 chain, on absolute time -------------------------------
    t0_abs = datetime.datetime.fromisoformat(str(ch['t0_abs']))
    panel_t = [t0_abs + datetime.timedelta(seconds=float(s))
               for s in ch['panel_taxis_s']]
    strain_rate = ch['panel_dpdt_psi_per_s'] * gamma            # (n_t, n_depth)
    trace_t = [t0_abs + datetime.timedelta(seconds=float(s))
               for s in ch['trace_taxis_s']]
    gnum = list(int(v) for v in ch['gauge_numbers'])

    # ---- figure --------------------------------------------------------------
    scalar_value = 500
    coeff = 0.2
    scalar_taxis = np.repeat(gauge_dataframe_all[0].start_time
                             + datetime.timedelta(minutes=30), 2)
    scalar_tmp_value = np.array([gauge_md[0] + 140,
                                 scalar_value * -coeff + gauge_md[0] + 140])
    selected_gauge_num = 3                    # Gauge D = gauge 8, unchanged
    cmap_a = rc.paled_bwr()
    cmap_b = 'bwr'
    trace_c = rc.GAUGE_COLOR
    trace_lw = rc.GAUGE_LW

    fig = plt.figure(figsize=(14, 8))
    cx = np.array([-1, 1])

    # (a) measured - UNCHANGED
    ax1 = plt.subplot2grid((4, 6), (0, 0), colspan=3, rowspan=3)
    for i in range(len(gauge_md)):
        ax1.axhline(y=gauge_md[i], color='black', linestyle='--')
        t = gauge_dataframe_all[i].calculate_time()
        y = (gauge_dataframe_all[i].data - gauge_dataframe_all[i].data[0]) * -coeff + gauge_md[i]
        ax1.plot(t, y, color=('black' if i == selected_gauge_num else trace_c),
                 linewidth=trace_lw, zorder=4)
    ax1.plot(scalar_taxis, scalar_tmp_value, color=trace_c,
             linewidth=rc.GAUGE_SCALEBAR_LW, zorder=5)
    ax1.text(scalar_taxis[0] + datetime.timedelta(minutes=8),
             scalar_tmp_value[0] - scalar_value / 18, f"{scalar_value} psi",
             fontsize=12, color='black', zorder=6)
    img1 = DASdata.plot(ax=ax1, useTimeStamp=True, cmap=cmap_a, aspect='auto')
    img1.set_clim(cx * 3e2)

    # (c) treating pressure - UNCHANGED
    ax2 = plt.subplot2grid((4, 6), (3, 0), colspan=3, rowspan=1, sharex=ax1)
    pc_stg8_pressure.plot(ax=ax2, useTimeStamp=True, title=None, color='black')
    ax2.set_ylabel("Pressure/psi", color='black')
    ax2.set_xlim(stg7_bgtime, stg8_edtime)
    ax22 = ax2.twinx()
    pc_stg7_pressure.plot(ax=ax22, useTimeStamp=True, title=None, color='black')
    ax22.set_ylabel("Treating Pressure/psi", color='black')
    ax22.legend(["Treating Pressure"], loc='lower right')

    # (b) synthetic - D0 = 480 chain, corrected Gamma and clim
    ax3 = plt.subplot2grid((4, 6), (0, 3), colspan=3, rowspan=3, sharey=ax1)
    for i in range(len(gauge_md)):
        ax3.axhline(y=gauge_md[i], color='black', linestyle='--')
        g = int(ind[i]) + 1              # gauge number = index into gauge_md + 1
        if g not in gnum:
            continue
        col = gnum.index(g)
        y = (ch['trace_psi'][:, col] - ch['trace_psi'][0, col]) * -0.15 + gauge_md[i]
        ax3.plot(trace_t, y, color=('black' if i == selected_gauge_num else trace_c),
                 linewidth=trace_lw, zorder=4)
    img3 = ax3.pcolormesh(panel_t, ch['panel_md_ft'], strain_rate.T,
                          cmap=cmap_b, shading='auto')
    img3.set_clim(cx * clim_b)
    # Pin the time axis to the chain span.  Without this the panel ends with a
    # ~4.8 % blank strip: pcolormesh sets sticky edges that suppress the axis
    # margin, but the gauge traces run 1 s past the mesh's last cell edge, which
    # breaks the sticky clamp on the RIGHT only, so matplotlib applies its
    # default 5 % margin there and nowhere else.  The archive version did not
    # show it because its mesh and its traces came off the same time array.
    ax3.set_xlim(stg7_bgtime, stg8_edtime)

    if return_panelb:
        return fig, ax3, img3, strain_rate, clim_b

    # (d) Gauge D, field vs synthetic - same gauge, D0 = 480 synthetic
    ax4 = plt.subplot2grid((4, 6), (3, 3), colspan=3, rowspan=1)
    t = gauge_dataframe_all[selected_gauge_num].calculate_time()
    ax4.plot(t, gauge_dataframe_all[selected_gauge_num].data, color='black',
             linewidth=1, label='Field data')
    col_d = gnum.index(8)
    ax4.plot(trace_t, ch['trace_psi'][:, col_d], color='red', linewidth=1,
             linestyle='--', label='Synthetic data')
    ax4.legend()

    plt.suptitle("LF-DAS data coplot with history matching result")

    # ---- dual output ---------------------------------------------------------
    ds.strip(ax2, keep_ylabel="Pressure/psi", keep_xlabel="Time")
    ds.strip(ax1, keep_ylabel="Measured depth (ft)", keep_xlabel="Time",
             restore_xlabels=False)
    ds.strip(ax22, x=False, keep_ylabel="Treating Pressure/psi")
    ds.strip(ax3, keep_ylabel="Measured depth (ft)", keep_xlabel="Time",
             restore_ylabels=False)
    ds.strip(ax4, keep_ylabel="Pressure/psi", keep_xlabel="Time")

    ds.freeze(fig, rect=LAYOUT_RECT)
    pin_axes(fig, ds)
    ds.colorbar(fig, img1, rect=(0.912, 0.60, 0.011, 0.28),
                label='(a) LF-DAS strain rate (counts)')
    ds.colorbar(fig, img3, rect=(0.960, 0.60, 0.011, 0.28),
                label=r'(b) synthetic strain rate (strain s$^{-1}$)')
    ds.restore()
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position('right')

    out = rc.save_outputs(fig, outdir, STEM, mode, dpi=dpi)
    boxes = rc.axes_pixel_boxes(fig, [ax1, ax2, ax3, ax4], dpi)
    plt.close(fig)
    params = {
        'chain_arrays': CHAIN_NPZ,
        'chain': {'D0_ft2_s': float(ch['D0']), 'pad_ft': float(ch['pad_ft']),
                  'barrier_ratio': float(ch['ratio']), 'dt_s': float(ch['dt_s']),
                  'panel_time_stride': int(ch['field_stride']),
                  'depth_window_ft': [float(v) for v in ch['panel_depth_window_ft']]},
        'transfer_coefficient': {
            'refractive_index_n': N_REFRACTIVE,
            'gamma_old_per_psi': GAMMA_OLD, 'gamma_new_per_psi': GAMMA_NEW,
            'clim_old_strain_per_s': CLIM_OLD, 'clim_new_strain_per_s': CLIM_NEW,
            'clim_used_strain_per_s': clim_b},
        'clim_panel_a_counts': (cx * 3e2).tolist(),
        'panel_d_gauge': {'number': 8, 'letter': 'D', 'md_ft': float(gauge_md[selected_gauge_num]),
                          'role': 'withheld entirely from calibration'},
        'gauges_in_window': [int(i) + 1 for i in ind],
    }
    return out, boxes, inputs, params


def verify_gamma(outdir, dpi):
    """Task 1 acceptance: new Gamma + new clim must not move a pixel."""
    from PIL import Image
    import hashlib
    tmp = os.path.join(outdir, '_gamma_check')
    os.makedirs(tmp, exist_ok=True)
    paths, mx = {}, {}
    for tag, new in (('gamma_old_8.94e-9', False), ('gamma_new_6.089e-9', True)):
        fig, ax3, img3, sr, clim = build('no_scale', tmp, dpi, gamma_new=new,
                                         return_panelb=True)
        p = os.path.join(tmp, f'panelb_{tag}.png')
        fig.savefig(p, dpi=dpi, bbox_inches=ax3.get_window_extent()
                    .transformed(fig.dpi_scale_trans.inverted()))
        plt.close(fig)
        paths[tag] = p
        mx[tag] = float(np.abs(sr).max())
        print(f"  {tag:20s} clim=+/-{clim:.15e}  max|strain rate|={mx[tag]:.9e}")
    print(f"  strain-rate ratio old/new = {mx['gamma_old_8.94e-9'] / mx['gamma_new_6.089e-9']!r}"
          f"   (expected {N_REFRACTIVE!r})")
    a = np.asarray(Image.open(paths['gamma_old_8.94e-9']).convert('RGB'), int)
    b = np.asarray(Image.open(paths['gamma_new_6.089e-9']).convert('RGB'), int)
    if a.shape != b.shape:
        raise RuntimeError(f"panel (b) canvas differs: {a.shape} vs {b.shape}")
    d = np.abs(a - b)
    n = int((d.sum(2) > 0).sum())
    sha = {k: hashlib.sha256(open(v, 'rb').read()).hexdigest() for k, v in paths.items()}
    print(f"  panel (b) {a.shape[0]}x{a.shape[1]} px   differing={n}   "
          f"max channel diff={int(d.max())}")
    for k, v in sha.items():
        print(f"    sha256 {k:20s} {v}")
    return {'differing_pixels': n, 'max_channel_diff': int(d.max()),
            'total_pixels': int(a.shape[0] * a.shape[1]),
            'sha256': sha, 'byte_identical': len(set(sha.values())) == 1,
            'bit_identical': n == 0}


def check_panel_a(outdir, dpi):
    """Panels (a) and (c) must be unchanged from the previous render."""
    from PIL import Image
    if not os.path.exists(PREVIOUS_RENDER):
        return {'checked': False, 'reason': f'{PREVIOUS_RENDER} not present'}
    a = np.asarray(Image.open(PREVIOUS_RENDER).convert('RGB'), int)
    b = np.asarray(Image.open(os.path.join(outdir, f'{STEM}_no_scale.png')
                              ).convert('RGB'), int)
    if a.shape != b.shape:
        return {'checked': False, 'reason': f'canvas {a.shape} vs {b.shape}'}
    out = {}
    for name, (y0, y1, x0, x1) in {'panel_a': (337, 2222, 258, 2587),
                                   'panel_c': (2295, 2884, 258, 2587)}.items():
        d = np.abs(a[y0:y1, x0:x1] - b[y0:y1, x0:x1])
        out[name] = {'differing_pixels': int((d.sum(2) > 0).sum()),
                     'max_channel_diff': int(d.max())}
    out['checked'] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', choices=list(rc.MODES) + ['both'], default='both')
    ap.add_argument('--outdir', default=DEFAULT_OUTDIR)
    ap.add_argument('--dpi', type=int, default=400)
    ap.add_argument('--verify-gamma', action='store_true')
    a = ap.parse_args()

    if a.verify_gamma:
        print("[Task 1] Gamma 8.94e-9 + clim 3.88991e-9  vs  "
              "Gamma 6.089089e-9 + clim 2.649442e-9, panel (b) only")
        r = verify_gamma(a.outdir, a.dpi)
        print("  VERDICT:", "IDENTICAL" if r['bit_identical'] else "NOT identical")
        return

    modes = list(rc.MODES) if a.scale == 'both' else [a.scale]
    outs, boxes = {}, None
    for m in modes:
        print(f"[{STEM}] rendering {m}")
        out, boxes, inputs, params = build(m, a.outdir, a.dpi)
        outs[m] = out
    if len(modes) == 2:
        rc.assert_data_area_identical(a.outdir, STEM, boxes)
        print(f"[{STEM}] data-area pixel identity: PASS")
        pa = check_panel_a(a.outdir, a.dpi)
        print(f"[{STEM}] panels (a)/(c) vs previous render: {pa}")
        gv = verify_gamma(a.outdir, a.dpi)
        rc.write_manifest(
            a.outdir, STEM,
            figure='Figure 6 - two-stage field vs synthetic comparison, D0 = 480, '
                   'refractive-index-corrected transfer coefficient',
            source_script=os.path.relpath(os.path.abspath(__file__), os.getcwd()),
            inputs=inputs, outputs=outs,
            parameters={**params, 'gamma_pixel_identity_check': gv,
                        'panels_a_c_unchanged_check': pa},
            changes=[
                f'Transfer coefficient Gamma {GAMMA_OLD!r} -> {GAMMA_NEW!r} /psi '
                f'(divided by the refractive index n = {N_REFRACTIVE!r}, which the '
                f'old processing chain omitted from the n*L_G slot).',
                f'Panel (b) colour limit {CLIM_OLD!r} -> {CLIM_NEW!r} strain/s, '
                f'divided by the SAME {N_REFRACTIVE!r}; rendered pixels unchanged.',
                'Panels (b) and (d) now read the D0 = 480 chain (pad 10 000 ft, '
                'ratio 1e-5, dt 1 s) instead of the D0 = 140 archive. Panel (d) '
                'plots the same gauge as before, Gauge D = gauge 8, MD 14821.',
                'Panels (a) and (c) unchanged - same code, same grid, same layout '
                'rect; verified against the previous render.',
                'Dual output (no_scale / with_scale) from one script via a flag.',
            ],
            notes=[
                'The chain was NOT re-solved by this script: it reads the arrays '
                f'persisted at {CHAIN_NPZ}, where dP/dt is stored Gamma-free so a '
                'change of transfer coefficient never touches the solver.',
                'Annotations the caption depends on that NO script in this repo '
                'draws, and which must still be added downstream: the panel letters '
                '(a)-(d), the dashed FDI box, the "stage #1"/"stage #2" labels, the '
                '200 ft bar in the upper panels and the 200 psi bar in (d). The '
                'submitted figs/manuscript/history_matching_res.png does not contain '
                'them either. The 500 psi bar in panel (a) IS drawn and is preserved.',
                'Panel (a) carries no frac-hit markers in this generator - those are '
                'a Figure 3 element - so there was nothing of that kind to preserve.',
            ])
    print(f"[{STEM}] done -> {a.outdir}")


if __name__ == '__main__':
    main()
