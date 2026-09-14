"""Fig. 3 (submitted as Fig. 2) - LF-DAS waterfall + pumping curves, S well stages 8/9.

    python3 scripts/manuscript_well_leakage/revision_figs/fig3_waterfall.py --scale no_scale
    python3 scripts/manuscript_well_leakage/revision_figs/fig3_waterfall.py --scale with_scale

Run with CWD = repo root.  Descended from
`scripts/manuscript_well_leakage/well_gauge_coplot/coplot_without_scalar.py`,
which was verified to reproduce the submitted PNG
(`figs/manuscript/DAS_gauge_coplot/Swell_no_scalar.png`) with the upper panel's
data area identical pixel for pixel; the only differences were text-glyph
antialiasing.  `coplot_with_scalar.py` is NOT a with/without-scale sibling of
it - it is an earlier state of the script (no frac-hit scatter, no low-pass
filter, no stage-9 pumping overlay), so it is superseded here rather than
folded in, and the scale flag is implemented on the newer code.

REVIEWER 1: "Figure 2 is not clear.  The cyan lines are barely visible and
there appears to be an x in the legend indicating a frac hit that I cannot see.
Perhaps paling the waterfall plot and/or changing colours might help."

Three changes, UPPER PANEL ONLY (`revfig_common` holds the numbers):

  1. the waterfall is lightened - still `bwr`, still clim = +/-3e2 counts;
  2. the gauge traces go cyan -> dark indigo and 2.0 -> 3.0 pt (the scale bar
     with them, so the "800 psi" bar still matches the traces it measures);
  3. the frac-hit marks go lightgray `x` s=40 -> filled `X` s=150 in near-black
     with a white edge.

The lower pumping panel is untouched.
"""

import argparse
import datetime
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import revfig_common as rc                                   # noqa: E402
import matplotlib.pyplot as plt                              # noqa: E402
from fiberis.analyzer.Data1D import Data1D_PumpingCurve, Data1D_Gauge   # noqa: E402
from fiberis.analyzer.Data2D import Data2D_XT_DSS            # noqa: E402
from fiberis.analyzer.Geometry3D import DataG3D_md           # noqa: E402

STEM = 'Fig3_waterfall'
DEFAULT_OUTDIR = 'figs/manuscript_revision'


def build(mode, outdir, dpi):
    ds = rc.DualScale(mode)
    inputs = []

    # ---- parameters, verbatim from coplot_without_scalar.py -----------------
    stage1, stage2 = 8, 9
    coeff = 0.14                      # amplitude of the PG response, ft/psi
    clim = np.array([-1, 1]) * 3e2    # counts - UNCHANGED
    scalar_value = 800                # psi, the scale bar
    lowpass_hz = 0.01

    datapath = "data/fiberis_format/"

    # ---- pumping ------------------------------------------------------------
    def pc(stage, name):
        p = f"{datapath}prod/pumping_data/stage{stage}/{name}.npz"
        inputs.append(p)
        o = Data1D_PumpingCurve.Data1DPumpingCurve()
        o.load_npz(p)
        return o

    pc_stg7_prop = pc(stage1, "Proppant Concentration")
    pc_stg7_slurry_rate = pc(stage1, "Slurry Rate")
    pc_stg7_pressure = pc(stage1, "Treating Pressure")
    pc_stg8_prop = pc(stage2, "Proppant Concentration")
    pc_stg8_slurry_rate = pc(stage2, "Slurry Rate")
    pc_stg8_pressure = pc(stage2, "Treating Pressure")

    stg7_bgtime = pc_stg7_slurry_rate.get_start_time()
    stg8_edtime = pc_stg8_slurry_rate.get_end_time()
    stg8_bgtime = pc_stg8_slurry_rate.get_start_time()

    # ---- LF-DAS -------------------------------------------------------------
    DASdata = Data2D_XT_DSS.DSS2D()
    for p in [f"{datapath}s_well/DAS/LFDASdata_stg{stage1}_swell.npz",
              f"{datapath}s_well/DAS/LFDASdata_stg{stage1}_interval_swell.npz",
              f"{datapath}s_well/DAS/LFDASdata_stg{stage2}_swell.npz"]:
        inputs.append(p)
        tmp = Data2D_XT_DSS.DSS2D()
        tmp.load_npz(p)
        if DASdata.data is None:
            DASdata.load_npz(p)
        else:
            DASdata.right_merge(tmp)

    # ---- geometry -----------------------------------------------------------
    def geom(p):
        inputs.append(p)
        o = DataG3D_md.G3DMeasuredDepth()
        o.load_npz(p)
        return o

    frachit_stg7_dataframe = geom(f"{datapath}s_well/geometry/frac_hit/frac_hit_stage_{stage1}_swell.npz")
    frachit_stg8_dataframe = geom(f"{datapath}s_well/geometry/frac_hit/frac_hit_stage_{stage2}_swell.npz")
    pg_md_dataframe = geom("data/fiberis_format/s_well/geometry/gauge_md_swell.npz")

    depth_range_min = np.min(frachit_stg8_dataframe.data) - 500
    depth_range_max = np.max(frachit_stg7_dataframe.data) + 700

    ind = np.array(np.where(np.logical_and(
        pg_md_dataframe.data > depth_range_min,
        pg_md_dataframe.data < depth_range_max))).flatten()

    gauge_dataframe_all = []
    for gauge_iter in tqdm(ind, disable=True):
        p = f'data/fiberis_format/s_well/gauges/gauge{gauge_iter + 1}_data_swell.npz'
        inputs.append(p)
        g = Data1D_Gauge.Data1DGauge()
        g.load_npz(p)
        g.crop(stg7_bgtime, stg8_edtime)
        gauge_dataframe_all.append(g)

    scalar_taxis = np.repeat(stg7_bgtime + datetime.timedelta(minutes=30), 2)
    scalar_tmp_value = np.array([pg_md_dataframe.data[ind][0] - 50,
                                 scalar_value * -coeff + pg_md_dataframe.data[ind][0] - 50])

    DASdata.select_depth(depth_range_min, depth_range_max)
    DASdata.apply_lowpass_filter(lowpass_hz)

    # ---- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(7, 5))
    ax1 = plt.subplot2grid((6, 4), (0, 0), colspan=4, rowspan=4)

    flag = 0
    for i in range(len(pg_md_dataframe.data[ind])):
        ax1.axhline(y=pg_md_dataframe.data[ind][i], color='black', linestyle='--')
        datetime_taxis = gauge_dataframe_all[i].calculate_time()
        y = (gauge_dataframe_all[i].data - gauge_dataframe_all[i].data[0]) * -coeff \
            + pg_md_dataframe.data[ind][i]
        # CHANGE 2: cyan lw=2 -> dark indigo lw=3
        ax1.plot(datetime_taxis, y, color=rc.GAUGE_COLOR, linewidth=rc.GAUGE_LW,
                 label='Pressure gauge' if flag == 0 else None, zorder=4)
        flag = 1

    # CHANGE 2 (cont.): the "800 psi" scale bar tracks the trace colour
    ax1.plot(scalar_taxis, scalar_tmp_value, color=rc.GAUGE_COLOR,
             linewidth=rc.GAUGE_SCALEBAR_LW, zorder=5)

    # CHANGE 1: lightened diverging map, SAME symmetric limits
    img1 = DASdata.plot(ax=ax1, use_timestamp=True, cmap=rc.paled_bwr(), aspect='auto')
    img1.set_clim(clim)

    # CHANGE 3: frac hits become findable
    fh7_t = np.repeat(stg7_bgtime + datetime.timedelta(minutes=30),
                      len(frachit_stg7_dataframe.data))
    fh8_t = np.repeat(stg8_bgtime + datetime.timedelta(minutes=30),
                      len(frachit_stg8_dataframe.data))
    for k, (t, dpts) in enumerate([(fh7_t, frachit_stg7_dataframe.data),
                                   (fh8_t, frachit_stg8_dataframe.data)]):
        ax1.scatter(t, dpts, marker=rc.FRACHIT_MARKER, s=rc.FRACHIT_SIZE,
                    c=rc.FRACHIT_COLOR, edgecolors=rc.FRACHIT_EDGE,
                    linewidths=rc.FRACHIT_EDGE_LW, zorder=6,
                    label='Frac Hit' if k == 0 else None)

    # every frac hit must be inside the rendered axes, and we say so out loud
    ylo, yhi = sorted(ax1.get_ylim())
    allfh = np.r_[frachit_stg7_dataframe.data, frachit_stg8_dataframe.data]
    outside = allfh[(allfh < ylo) | (allfh > yhi)]
    print(f"  frac hits: {len(allfh)} total, depth axis [{ylo:.1f}, {yhi:.1f}] ft, "
          f"{len(outside)} outside")
    if len(outside):
        print(f"  !! OUTSIDE THE AXES, NOT MOVED: {np.round(outside, 2)}")

    ax1.legend(loc='lower right')

    # ---- lower pumping panel: untouched -------------------------------------
    ax2 = plt.subplot2grid((6, 4), (4, 0), colspan=4, rowspan=2, sharex=ax1)
    pc_stg7_prop.rename("Prop. Conc.")
    pc_stg7_prop.plot(ax=ax2, use_timestamp=True, title=None, color='blue')
    ax2.set_ylabel(r"Prop. Conc./lb$\cdot$gal$^{-1}$", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_xlim(stg7_bgtime, stg8_edtime)

    ax21 = ax2.twinx()
    pc_stg7_slurry_rate.rename("Slurry Rate")
    pc_stg7_slurry_rate.plot(ax=ax21, use_timestamp=True, title=None, color='green')
    ax21.tick_params(axis='y', labelcolor='green')

    ax22 = ax2.twinx()
    pc_stg7_pressure.rename("Treating Pressure")
    pc_stg7_pressure.plot(ax=ax22, use_timestamp=True, title=None, color='red')
    ax22.tick_params(axis='y', labelcolor='red')

    ax23 = ax2.twinx()
    pc_stg8_pressure.plot(ax=ax23, use_timestamp=True, title=None, color='red')
    ax23.yaxis.set_visible(False)

    h2, l2 = ax2.get_legend_handles_labels()
    h21, l21 = ax21.get_legend_handles_labels()
    h22, l22 = ax22.get_legend_handles_labels()
    ax2.legend(h2 + h21 + h22, l2 + l21 + l22, loc='upper center',
               bbox_to_anchor=(0.52, 1.05))

    pc_stg8_prop.rename("Prop. Conc. Stage 9")
    pc_stg8_prop.plot(ax=ax2, use_timestamp=True, title=None, color='blue')
    pc_stg8_slurry_rate.plot(ax=ax21, use_timestamp=True, title=None, color='green')
    pc_stg8_pressure.rename("Treating Pressure Stage 9")
    pc_stg8_pressure.plot(ax=ax22, use_timestamp=True, title=None, color='red')

    # ---- dual output --------------------------------------------------------
    ax1.set_xlabel("")
    # ax2 FIRST: it shares the x Ticker object with ax1, so whichever is
    # stripped first is the only one that records the real DateFormatter.
    ds.strip(ax2, keep_xlabel="Time", keep_ylabel=r"Prop. Conc./lb$\cdot$gal$^{-1}$")
    ds.strip(ax1, keep_ylabel="Measured depth (ft)", restore_xlabels=False)
    ds.strip(ax21, x=False, keep_ylabel="Slurry rate (bpm)")
    ds.strip(ax22, x=False, keep_ylabel="Treating pressure (psi)")
    ds.strip(ax23, x=False, y=True)
    ax23.yaxis.set_visible(False)

    ds.freeze(fig, rect=(0.045, 0.085, 0.895, 0.985))
    ds.colorbar(fig, img1, rect=(0.915, 0.44, 0.017, 0.40),
                label='LF-DAS strain rate (counts)')
    ds.restore()

    out = rc.save_outputs(fig, outdir, STEM, mode, dpi=dpi)
    boxes = rc.axes_pixel_boxes(fig, [ax1, ax2], dpi)
    plt.close(fig)
    return out, boxes, inputs, {
        'stages': [stage1, stage2], 'coeff_ft_per_psi': coeff,
        'clim_counts': clim.tolist(), 'lowpass_hz': lowpass_hz,
        'scale_bar_psi': scalar_value,
        'depth_window_ft': [float(depth_range_min), float(depth_range_max)],
        'gauges_plotted': [int(i) + 1 for i in ind],
        'frac_hits_outside_axes': outside.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', choices=list(rc.MODES) + ['both'], default='both')
    ap.add_argument('--outdir', default=DEFAULT_OUTDIR)
    ap.add_argument('--dpi', type=int, default=400)
    a = ap.parse_args()

    modes = list(rc.MODES) if a.scale == 'both' else [a.scale]
    outs, boxes = {}, None
    for m in modes:
        print(f"[{STEM}] rendering {m}")
        out, boxes, inputs, params = build(m, a.outdir, a.dpi)
        outs[m] = out

    if len(modes) == 2:
        rep = rc.assert_data_area_identical(a.outdir, STEM, boxes)
        print(f"[{STEM}] data-area pixel identity: PASS {rep}")
        rc.write_manifest(
            a.outdir, STEM,
            figure='Figure 3 (submitted as Figure 2) - LF-DAS waterfall + pumping curves',
            source_script=os.path.relpath(os.path.abspath(__file__), os.getcwd()),
            inputs=inputs, outputs=outs, parameters=params,
            changes=[
                'Upper panel: waterfall lightened (bwr blended 0.45 toward white); '
                'colormap family and clim +/-3e2 counts unchanged.',
                'Upper panel: pressure-gauge traces cyan -> #2e2160, linewidth 2.0 -> 3.0; '
                'the 800 psi scale bar recoloured to match (linewidth 5 -> 6).',
                'Upper panel: frac-hit marks lightgray "x" s=40 -> near-black filled "X" '
                's=150 with a white edge.',
                'Lower pumping panel: unchanged.',
                'Layout: a 10.5% right-hand strip is reserved in BOTH modes so the '
                'with_scale colorbar cannot move the data axes. Panel arrangement '
                '(6x4 grid, rowspans 4 and 2) is unchanged.',
            ],
            notes=[
                'The submitted figure carries annotations no script in this repo draws: '
                '"Pressure Gauge F"/"Pressure Gauge A" labels, the black propagation box, '
                'the arrows, and the "200 ft" and "1 hr" scale bars. They were added '
                'downstream of matplotlib and are NOT reproduced here.',
                'The "800 psi" text label is commented out at coplot_without_scalar.py:145-146; '
                'the bar is drawn, the text is not. Left as-is.',
            ])
    print(f"[{STEM}] done -> {a.outdir}")


if __name__ == '__main__':
    main()
