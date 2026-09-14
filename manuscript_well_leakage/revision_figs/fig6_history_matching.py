"""Fig. 6 - two-stage field vs synthetic comparison.

    python3 scripts/manuscript_well_leakage/revision_figs/fig6_history_matching.py --scale both
    python3 scripts/manuscript_well_leakage/revision_figs/fig6_history_matching.py --verify-gamma
    python3 scripts/manuscript_well_leakage/revision_figs/fig6_history_matching.py --legacy

Run with CWD = repo root.  Descended from
`scripts/well_leakage_history_matching/DAS_history_matching_visualization/
104_full_history_matching_manuscript.py`, verified to reproduce the submitted
`figs/manuscript/history_matching_res.png` once four post-2025-06 fibeRIS API
changes are reverted (see `revfig_common.install_legacy_fiberis_plotting`):
mean |delta| per 4x4 pixel block was 0.07/255 in panel (a), 0.16/255 in panel (d)
and 0.85/255 in panel (b) - the residual is pcolormesh/imshow edge rasterisation,
not content.

WHAT CHANGES
------------
6.1  Panel (a): Reviewer 1's items 1 and 2 from Fig. 3, with the SAME numbers
     (`revfig_common`), so the two figures match.  Panel (b)'s traces get the
     same colour treatment.  There is no frac-hit scatter in this script, so
     Fig. 3's item 3 does not apply.

6.2  Panel (b): the phase-3 input changes from `phase3_test.npz` to
     `phase3_1e-05.npz`.  `phase3_test.npz` was regenerated 2025-04-28 with
     unrecorded parameters and carries NO barrier: measured against a control
     node one refined cell away, the excess pressure step at the six stage-7
     frac hits is -0.12 to -0.89 psi, i.e. zero to within the background
     gradient, where the 1e-5 run gives +3 to +162 psi.  Run
     `--audit-phase3` to reproduce that table.  `phase3_1e-05.npz` is from the
     original 2025-02-11 sweep and continues from the SAME `phase2.npz`
     snapshot (max|first profile - phase2 last profile| = 0.0000 psi).

6.3  Panel (b) colorbar: `data * 6894.76 / 30e9` (= P/E) is replaced by the
     empirical transfer coefficient Gamma = 8.94e-9 /psi, and the clim is
     divided by the SAME factor so the rendering does not move.  The factor is
     computed from the constants, never typed in.

Panels (c) and (d) are not edited.  Panel (d) still plots the same gauge.  Its
red dashed synthetic curve nevertheless changes in the phase-3 segment, because
it reads the same merged chain panel (b) does - that is a consequence of 6.2,
not a separate edit.
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
from fiberis.utils import mesh_utils                          # noqa: E402

STEM = 'Fig6_history_matching'
DEFAULT_OUTDIR = 'figs/manuscript_revision'

SIMDIR = "output/0211_simulation_MULTIstage/"
PHASE3_REVISION = SIMDIR + "phase3_1e-05.npz"      # 6.2
PHASE3_SUBMITTED = SIMDIR + "phase3_test.npz"

# 6.3 - derived, not typed.  E = 30 GPa, 1 psi = 6894.76 Pa.
PA_PER_PSI = 6894.76
E_PA = 30e9
K_LEGACY = PA_PER_PSI / E_PA          # 2.2982533333e-07 strain/psi  (= P/E)
GAMMA = 8.94e-9                       # 1/psi, empirical transfer coefficient
GAMMA_FACTOR = K_LEGACY / GAMMA       # 25.707531692766597
CLIM_LEGACY = 1e-7
CLIM_GAMMA = CLIM_LEGACY / GAMMA_FACTOR   # 3.88991059877356e-09


def _load_sim(path, inputs):
    """Load a chain panel, orienting it from its own axis lengths.

    `output/0211_simulation_MULTIstage/` straddles fibeRIS commit aabffe2, so it
    holds both layouts: phase1/phase2/phase3_{ratio} are TIME-major on disk and
    phase3_test alone is DEPTH-major.  The submitted script hard-codes one
    transpose for phase1/2 (:24-25) and a literal no-op for phase3 (:37), which
    is right for `phase3_test` and wrong for every other phase-3 file.  Deciding
    from the shape is right for all of them.
    """
    inputs.append(path)
    d = Data2D_XT_DSS.DSS2D()
    d.load_npz(path)
    n_t, n_x = len(d.taxis), len(d.daxis)
    if d.data.shape == (n_t, n_x) and n_t != n_x:
        d.data = d.data.T          # on-disk time-major -> canonical (depth, time)
        layout = 'time_major'
    elif d.data.shape == (n_x, n_t):
        layout = 'depth_major'
    else:
        raise ValueError(f"{path}: cannot orient data{d.data.shape} "
                         f"against (n_t={n_t}, n_x={n_x})")
    return d, layout


def audit_phase3():
    """The measurement behind 6.2, printed so a reviewer can check it."""
    inputs = []
    fh7 = np.load("data/legacy/s_well/geometry/frac_hit/frac_hit_stage_7_swell.npz")['data']
    files = ['phase3_0.1.npz', 'phase3_0.01.npz', 'phase3_0.001.npz',
             'phase3_0.0001.npz', 'phase3_1e-05.npz', 'phase3_test.npz']
    ph2, _ = _load_sim(SIMDIR + 'phase2.npz', inputs)
    panels = {f: _load_sim(SIMDIR + f, inputs)[0] for f in files}
    ref = panels['phase3_test.npz']
    idx = [mesh_utils.locate(ref.daxis, m)[0] for m in fh7]
    ctrl = [i + 25 for i in idx]     # same refined region, no barrier there

    print("continuity with phase2 (max |first profile - phase2 last profile|, psi)")
    for f, p in panels.items():
        print(f"  {f:20s} {np.abs(p.data[:, 0] - ph2.data[:, -1]).max():9.4f}")

    print("\nbarrier EXCESS at the six stage-7 frac hits, final profile (psi)")
    print("  = |P[i-1]-P[i+1]| at the frac-hit node  -  the same measure at a "
          "control node 25 cells away")
    print(f"  {'file':20s}" + ''.join(f"{i:>9d}" for i in range(len(idx))))
    for f, p in panels.items():
        prof = p.data[:, -1]
        ex = [abs(prof[i - 1] - prof[i + 1]) - abs(prof[j - 1] - prof[j + 1])
              for i, j in zip(idx, ctrl)]
        print(f"  {f:20s}" + ''.join(f"{v:9.3f}" for v in ex))

    print("\nmax |panel - phase3_test| over the whole field (psi)")
    for f, p in panels.items():
        if f != 'phase3_test.npz':
            print(f"  {f:20s} {np.abs(p.data - ref.data).max():10.3f}")


def build(mode, outdir, dpi, legacy=False, gamma=True, return_panelb=False):
    """legacy=True reproduces the SUBMITTED figure; gamma=False keeps P/E."""
    ds = rc.DualScale(mode)
    inputs = []

    phase1, _ = _load_sim(SIMDIR + "phase1.npz", inputs)
    phase2, _ = _load_sim(SIMDIR + "phase2.npz", inputs)
    phase2.select_time(30.0, phase2.get_end_time('seconds'))
    phase1.right_merge(phase2)

    p3_path = PHASE3_SUBMITTED if legacy else PHASE3_REVISION
    phase3, p3_layout = _load_sim(p3_path, inputs)
    phase3.select_time(30.0, phase3.get_end_time('seconds'))
    phase1.right_merge(phase3)

    # ---- geometry (legacy format, as submitted) -----------------------------
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

    pf_dataframe = phase1.copy()
    pf_dataframe.select_depth(np.min(frac_hit_stg8) - 500, np.max(frac_hit_stg7) + 500)
    phase1 = pf_dataframe.copy()          # pressure, psi - panels (b) overlay and (d)

    # ---- pumping + LF-DAS ---------------------------------------------------
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

    # ---- 6.3 pressure -> strain --------------------------------------------
    if gamma and not legacy:
        pf_dataframe.data = pf_dataframe.data * GAMMA          # strain = Gamma * P
        clim_b = CLIM_GAMMA
        conv_name, conv_value = 'Gamma', GAMMA
    else:
        pf_dataframe.data = pf_dataframe.data * K_LEGACY       # strain = P / E
        clim_b = CLIM_LEGACY
        conv_name, conv_value = 'P/E', K_LEGACY

    tmp_strain = np.zeros_like(pf_dataframe.data)
    for i in range(pf_dataframe.data.shape[0]):
        tmp_strain[i, :] = (np.gradient(pf_dataframe.data[i, :], axis=0)
                            / np.gradient(pf_dataframe.taxis))

    # ---- figure -------------------------------------------------------------
    scalar_value = 500
    coeff = 0.2
    scalar_taxis = np.repeat(gauge_dataframe_all[0].start_time
                             + datetime.timedelta(minutes=30), 2)
    scalar_tmp_value = np.array([gauge_md[0] + 140,
                                 scalar_value * -coeff + gauge_md[0] + 140])

    selected_gauge_num = 3
    # 6.1 is explicit: item 1 (lighten the waterfall) is PANEL (a) ONLY.  Panel
    # (b) keeps `bwr` and only its trace colour follows panel (a).
    cmap_a = 'bwr' if legacy else rc.paled_bwr()
    cmap_b = 'bwr'
    trace_c = 'cyan' if legacy else rc.GAUGE_COLOR
    trace_lw = 2 if legacy else rc.GAUGE_LW

    fig = plt.figure(figsize=(14, 8))
    cx = np.array([-1, 1])

    # (a) measured
    ax1 = plt.subplot2grid((4, 6), (0, 0), colspan=3, rowspan=3)
    for i in range(len(gauge_md)):
        ax1.axhline(y=gauge_md[i], color='black', linestyle='--')
        t = gauge_dataframe_all[i].calculate_time()
        y = (gauge_dataframe_all[i].data - gauge_dataframe_all[i].data[0]) * -coeff + gauge_md[i]
        if i == selected_gauge_num:
            ax1.plot(t, y, color='black', linewidth=trace_lw, zorder=4)
        else:
            ax1.plot(t, y, color=trace_c, linewidth=trace_lw, zorder=4)
    ax1.plot(scalar_taxis, scalar_tmp_value, color=trace_c,
             linewidth=5 if legacy else rc.GAUGE_SCALEBAR_LW, zorder=5)
    ax1.text(scalar_taxis[0] + datetime.timedelta(minutes=8),
             scalar_tmp_value[0] - scalar_value / 18, f"{scalar_value} psi",
             fontsize=12, color='black', zorder=6)
    img1 = DASdata.plot(ax=ax1, useTimeStamp=True, cmap=cmap_a, aspect='auto')
    img1.set_clim(cx * 3e2)

    # (c) treating pressure
    ax2 = plt.subplot2grid((4, 6), (3, 0), colspan=3, rowspan=1, sharex=ax1)
    pc_stg8_pressure.plot(ax=ax2, useTimeStamp=True, title=None, color='black')
    ax2.set_ylabel("Pressure/psi", color='black')
    ax2.set_xlim(stg7_bgtime, stg8_edtime)
    ax22 = ax2.twinx()
    pc_stg7_pressure.plot(ax=ax22, useTimeStamp=True, title=None, color='black')
    ax22.set_ylabel("Treating Pressure/psi", color='black')
    ax22.legend(["Treating Pressure"], loc='lower right')

    # (b) synthetic
    ax3 = plt.subplot2grid((4, 6), (0, 3), colspan=3, rowspan=3, sharey=ax1)
    real_time_taxis = pf_dataframe.calculate_time()
    for i in range(len(gauge_md)):
        ax3.axhline(y=gauge_md[i], color='black', linestyle='--')
        gi, _ = mesh_utils.locate(phase1.daxis, gauge_md[i])
        y = (phase1.data[gi, :] - phase1.data[gi, 0]) * -0.15 + gauge_md[i]
        if i == selected_gauge_num:
            ax3.plot(real_time_taxis, y, color='black', linewidth=trace_lw, zorder=4)
        else:
            ax3.plot(real_time_taxis, y, color=trace_c, linewidth=trace_lw, zorder=4)
    img3 = ax3.pcolormesh(real_time_taxis, pf_dataframe.daxis, tmp_strain, cmap=cmap_b)
    img3.set_clim(cx * clim_b)

    # (d) single-gauge comparison
    ax4 = plt.subplot2grid((4, 6), (3, 3), colspan=3, rowspan=1)
    t = gauge_dataframe_all[selected_gauge_num].calculate_time()
    ax4.plot(t, gauge_dataframe_all[selected_gauge_num].data, color='black',
             linewidth=1, label='Field data')
    gi, _ = mesh_utils.locate(phase1.daxis, gauge_md[selected_gauge_num])
    ax4.plot(real_time_taxis, phase1.data[gi, :], color='red', linewidth=1,
             linestyle='--', label='Synthetic data')
    ax4.legend()

    plt.suptitle("LF-DAS data coplot with history matching result")

    if return_panelb:
        return fig, ax3, img3, tmp_strain, clim_b

    # ---- dual output --------------------------------------------------------
    for ax in (ax1, ax2, ax22, ax3, ax4):
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
    # Order matters twice over, because `sharex`/`sharey` make the two axes hold
    # the SAME Ticker object: whichever is stripped first is the only one that
    # records the real locator/formatter.
    #   x: ax2 shares with ax1  -> strip ax2 first, ax1 only hides its labels
    #   y: ax1 shares with ax3  -> strip ax1 before ax3
    ds.strip(ax2, keep_ylabel="Pressure/psi", keep_xlabel="Time")
    ds.strip(ax1, keep_ylabel="Measured depth (ft)", keep_xlabel="Time",
             restore_xlabels=False)
    ds.strip(ax22, x=False, keep_ylabel="Treating Pressure/psi")
    ds.strip(ax3, keep_ylabel="Measured depth (ft)", keep_xlabel="Time",
             restore_ylabels=False)   # shares y with ax1; its labels land in ax1
    ds.strip(ax4, keep_ylabel="Pressure/psi", keep_xlabel="Time")

    ds.freeze(fig, rect=(0.035, 0.055, 0.90, 0.955))
    ds.colorbar(fig, img1, rect=(0.912, 0.60, 0.011, 0.28),
                label='(a) LF-DAS strain rate (counts)')
    ds.colorbar(fig, img3, rect=(0.960, 0.60, 0.011, 0.28),
                label=r'(b) synthetic strain rate (strain s$^{-1}$)')
    ds.restore()
    # ax4 is the RIGHT-hand column: left-side tick labels would be drawn back
    # into panel (c)'s data area.  Ticks are absent in no_scale, so moving the
    # side is a no-op there and the two modes stay identical.
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position('right')

    out = rc.save_outputs(fig, outdir, STEM, mode, dpi=dpi)
    boxes = rc.axes_pixel_boxes(fig, [ax1, ax2, ax3, ax4], dpi)
    plt.close(fig)
    params = {
        'phase3_input': p3_path, 'phase3_on_disk_layout': p3_layout,
        'pressure_to_strain': {'name': conv_name, 'value': conv_value,
                               'legacy_P_over_E': K_LEGACY, 'gamma': GAMMA,
                               'factor_legacy_over_gamma': GAMMA_FACTOR},
        'clim_panel_a_counts': (cx * 3e2).tolist(),
        'clim_panel_b_strain_per_s': (cx * clim_b).tolist(),
        'selected_gauge_index_into_window': selected_gauge_num,
        'selected_gauge_number': int(ind[selected_gauge_num]) + 1,
        'selected_gauge_md_ft': float(gauge_md[selected_gauge_num]),
        'gauges_in_window': [int(i) + 1 for i in ind],
        'gauge_md_ft': [float(v) for v in gauge_md],
        'scale_bar_psi': scalar_value, 'gauge_amplitude_coeff': coeff,
    }
    return out, boxes, inputs, params


def verify_gamma(outdir, dpi):
    """6.3's acceptance test: Gamma + rescaled clim must not move a pixel."""
    import matplotlib.pyplot as plt
    from PIL import Image
    tmpdir = os.path.join(outdir, '_gamma_check')
    os.makedirs(tmpdir, exist_ok=True)
    paths = {}
    strains = {}
    for tag, use_gamma in (('P_over_E', False), ('gamma', True)):
        fig, ax3, img3, strain, clim = build('no_scale', tmpdir, dpi,
                                             gamma=use_gamma, return_panelb=True)
        strains[tag] = strain
        # render ONLY panel (b), at its own axes extent, so nothing else can differ
        p = os.path.join(tmpdir, f'panelb_{tag}.png')
        fig.savefig(p, dpi=dpi, bbox_inches=ax3.get_window_extent()
                    .transformed(fig.dpi_scale_trans.inverted()))
        plt.close(fig)
        paths[tag] = p
        print(f"  {tag:9s} clim=+/-{clim:.6e}  max|strain|={np.abs(strain).max():.6e}")

    r = np.abs(strains['P_over_E']).max() / np.abs(strains['gamma']).max()
    print(f"  strain-array ratio P_over_E / gamma = {r!r}  (expected {GAMMA_FACTOR!r})")

    a = np.asarray(Image.open(paths['P_over_E']).convert('RGB'), int)
    b = np.asarray(Image.open(paths['gamma']).convert('RGB'), int)
    if a.shape != b.shape:
        raise RuntimeError(f"panel (b) canvas differs: {a.shape} vs {b.shape}")
    d = np.abs(a - b)
    n = int((d.sum(2) > 0).sum())
    print(f"  panel (b) pixels: {a.shape[0]}x{a.shape[1]}  differing={n}  "
          f"max channel diff={int(d.max())}")
    return {'differing_pixels': n, 'max_channel_diff': int(d.max()),
            'total_pixels': int(a.shape[0] * a.shape[1]),
            'strain_ratio': float(r), 'expected_ratio': GAMMA_FACTOR,
            'bit_identical': n == 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', choices=list(rc.MODES) + ['both'], default='both')
    ap.add_argument('--outdir', default=DEFAULT_OUTDIR)
    ap.add_argument('--dpi', type=int, default=400)
    ap.add_argument('--legacy', action='store_true',
                    help='reproduce the SUBMITTED figure (phase3_test, bwr, cyan, P/E)')
    ap.add_argument('--verify-gamma', action='store_true')
    ap.add_argument('--audit-phase3', action='store_true')
    a = ap.parse_args()

    if a.audit_phase3:
        audit_phase3()
        return
    if a.verify_gamma:
        print("[6.3] Gamma + rescaled clim vs P/E + legacy clim, panel (b) only")
        res = verify_gamma(a.outdir, a.dpi)
        print("  VERDICT:", "IDENTICAL" if res['bit_identical'] else "NOT identical")
        return

    stem = STEM + ('_legacy' if a.legacy else '')
    modes = list(rc.MODES) if a.scale == 'both' else [a.scale]
    outs, boxes = {}, None
    for m in modes:
        print(f"[{stem}] rendering {m}")
        out, boxes, inputs, params = build(m, a.outdir, a.dpi, legacy=a.legacy)
        outs[m] = out
    if len(modes) == 2:
        rep = rc.assert_data_area_identical(a.outdir, stem, boxes)
        print(f"[{stem}] data-area pixel identity: PASS")
        rc.write_manifest(
            a.outdir, stem,
            figure='Figure 6 - two-stage field vs synthetic comparison',
            source_script=os.path.relpath(os.path.abspath(__file__), os.getcwd()),
            inputs=inputs, outputs=outs, parameters=params,
            changes=[
                '6.1 panel (a): waterfall lightened (bwr blended 0.45 to white), '
                'clim +/-3e2 counts unchanged; cyan traces -> #2e2160 at lw 3.0. '
                'The highlighted gauge stays black, as submitted.',
                '6.1 panel (b): its traces take the same colour and width as panel (a); its '
                'colormap stays bwr - item 1 is panel (a) only.',
                '6.2 panel (b): phase-3 input phase3_test.npz -> phase3_1e-05.npz.',
                f'6.3 panel (b): P/E = {K_LEGACY!r} strain/psi -> Gamma = {GAMMA!r} /psi; '
                f'clim +/-{CLIM_LEGACY!r} -> +/-{CLIM_GAMMA!r} strain/s '
                f'(divided by the same factor {GAMMA_FACTOR!r}).',
                'phase-3 orientation is now decided from the file\'s own axis lengths '
                'instead of the hard-coded no-op transpose at 104_full:37, which was '
                'correct only for phase3_test.npz.',
                'Panels (c) and (d) are not edited; panel (d) plots the same gauge. '
                'Its synthetic curve moves in the phase-3 segment because it reads the '
                'same merged chain as panel (b).',
                'Layout: a 10% right-hand strip is reserved in BOTH modes for the two '
                'with_scale colorbars. Panel arrangement (4x6 grid) unchanged.',
            ],
            notes=[
                'pixel_identity_check_6_3: run --verify-gamma.',
                'phase3_test audit: run --audit-phase3.',
            ])
    print(f"[{stem}] done -> {a.outdir}")


if __name__ == '__main__':
    main()
