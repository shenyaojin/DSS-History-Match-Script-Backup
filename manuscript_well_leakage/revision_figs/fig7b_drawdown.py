"""Fig. 7 panel (b) - production pressure drawdown vs gauge number.

    python3 scripts/manuscript_well_leakage/revision_figs/fig7b_drawdown.py --scale both
    python3 scripts/manuscript_well_leakage/revision_figs/fig7b_drawdown.py --forensics

Run with CWD = repo root.  Descended from
`scripts/sponsor_meeting_report_2025/production_sim/
103p_forward_modeling_viz_without_scalar.py` (with-scale sibling
`103p_forward_modeling_viz.py`; the two differ only in the label strings, the
legend call and the y-tick removal, so both are folded into `--scale` here).

Panel (a) of Fig. 7 comes from a DIFFERENT script and is not touched.

WHY THE SUBMITTED PANEL DOES NOT REPRODUCE
------------------------------------------
Re-running the submitted script today gives a visibly different figure: the
field series has a different shape and the orange and green curves have
swapped scenarios.  Nothing on disk changed - `os.listdir` order did.  That is
the defect, demonstrated rather than argued.  `--forensics` recovers what the
submitted PNG actually plots by fitting each candidate series to the marker
pixel positions in `figs/manuscript/production/dropdown_to202051_noscalar.png`.

THE THREE DEFECTS, ALL FIXED
----------------------------
(A) Labels decoupled from data.  `label_tmp` was a hard-coded list of three
    applied by loop position to `pressure_drop_down[0, 2, 4]`, while entries
    1..N were appended in `os.listdir` order; and `dataframe_full[:-1]` silently
    dropped the last file.  Here every run is addressed by name, the ratio is
    parsed from that name, and the label is built from the ratio.
    `pack_result` (fibeRIS `pds.py:415`) stores only daxis/taxis/data/start_time,
    so the FILENAME is the only record of the ratio that exists; the labelling
    is therefore cross-checked against physics - drawdown at the far gauge must
    fall monotonically as the ratio falls - and the check is asserted, not
    assumed.  The manifest records both.

(B) Two different indexings on one "Gauge Number" axis.  The field series was
    built by iterating `os.listdir(pg_data_folder)` (unsorted), the simulated
    series by sampling at `gauge_idx` derived from the gauge-MD geometry file.
    Here the field series is keyed by the gauge number parsed out of each
    filename and placed at that number, and every plotted point is printed.

(C) Inconsistent start index: the simulation used `data_chan[1]` (second time
    sample) and the field used `data[0]`.  Both now use index 0.
"""

import argparse
import datetime
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import revfig_common as rc                                    # noqa: E402
import matplotlib.pyplot as plt                               # noqa: E402
from fiberis.analyzer.Data2D import Data2D_XT_DSS             # noqa: E402
from fiberis.analyzer.Data1D import Data1D_Gauge              # noqa: E402
from fiberis.utils import mesh_utils                          # noqa: E402
from DSS_analyzer_Mariner import Data3D_geometry              # noqa: E402

STEM = 'Fig7b_drawdown'
DEFAULT_OUTDIR = 'figs/manuscript_revision'

SIMDIR = "output/0324_forward_simulator/"
GAUGEDIR = "data/fiberis_format/s_well/gauges/"
GAUGE_MD = "data/legacy/s_well/geometry/gauge_md_swell.npz"
START = datetime.datetime(2020, 4, 1)
END = datetime.datetime(2020, 6, 1)

#: the two scenarios the caption names, in caption order, addressed BY NAME.
SCENARIOS = [
    ('1.0_gauge3.npz',   1.0,  'Simulated drawdown\n (min D = 1 × baseline)'),
    ('1e-05_gauge3.npz', 1e-5, 'Simulated drawdown\n (min D = 1e-5 × baseline)'),
]
SCALE_BAR_PSI = 500


def ratio_from_name(fname):
    """The ratio is only recorded in the filename - pack_result stores no
    parameters - so parse it, and never infer identity from list position."""
    m = re.match(r'^([0-9.eE+-]+)_gauge\d+\.npz$', fname)
    if not m:
        raise ValueError(f"{fname}: cannot parse a ratio out of the filename")
    return float(m.group(1))


def load_all(inputs):
    """Every simulated run, keyed by filename.  Sorted, never listdir order."""
    files = sorted(os.listdir(SIMDIR), key=ratio_from_name)
    panels = {}
    for f in files:
        p = SIMDIR + f
        inputs.append(p)
        d = Data2D_XT_DSS.DSS2D()
        d.load_npz(p)
        d.select_time(START, END)
        panels[f] = d
    return panels


def field_by_gauge(inputs):
    """Field drawdown keyed by the gauge NUMBER parsed from each filename."""
    out = {}
    for fn in sorted(os.listdir(GAUGEDIR),
                     key=lambda s: int(re.search(r'gauge(\d+)_', s).group(1))):
        n = int(re.search(r'gauge(\d+)_', fn).group(1))
        p = GAUGEDIR + fn
        inputs.append(p)
        g = Data1D_Gauge.Data1DGauge()
        g.load_npz(p)
        g.crop(START, END)
        out[n] = float(g.data[0] - g.data[-1])       # (C) index 0, like the sims
    return out


def sim_by_gauge(panel, gauge_idx):
    """Simulated drawdown at each gauge node, gauge number 1..N in MD order."""
    chan = panel.data.T[:, gauge_idx]                # (n_t, n_gauges)
    dd = chan[0] - chan[-1]                          # (C) index 0, was index 1
    return {i + 1: float(v) for i, v in enumerate(dd)}


def monotonicity_check(panels, gauge_idx):
    """(A) cross-check: as the ratio falls, the barrier gets stronger, so the
    drawdown reaching the FAR gauge must fall.  If a filename were mislabelled
    this ordering would break."""
    rows = []
    for f in sorted(panels, key=ratio_from_name):
        s = sim_by_gauge(panels[f], gauge_idx)
        rows.append((ratio_from_name(f), f, s[1], max(s.values())))
    far = [r[2] for r in rows]
    ok = all(far[i] <= far[i + 1] + 1e-9 for i in range(len(far) - 1))
    return rows, ok


def forensics():
    """What does the SUBMITTED PNG actually plot?  Fit, do not guess."""
    from PIL import Image
    from scipy.optimize import linear_sum_assignment
    png = "figs/manuscript/production/dropdown_to202051_noscalar.png"
    im = np.asarray(Image.open(png).convert('RGB'), int)
    g = np.asarray(Image.open(png).convert('L'), int)
    dark = g < 100
    rows = np.nonzero(dark.sum(1) > 300)[0]
    cols = np.nonzero(dark.sum(0) > 200)[0]
    x0, x1 = cols.min(), cols.max()
    W = x1 - x0
    xs = [x0 + 0.05 * W + (n - 1) / 14 * 0.9 * W for n in range(1, 16)]

    marker_y = {}
    for name, c in {'C0': (31, 119, 180), 'C1': (255, 127, 14),
                    'C2': (44, 160, 44)}.items():
        m = (np.abs(im - np.array(c)).sum(2) < 40)
        ys = []
        for xc in xs:
            sl = m[:, int(round(xc)) - 2:int(round(xc)) + 3]
            yy = np.nonzero(sl.any(1))[0]
            ys.append(yy.mean() if len(yy) else np.nan)
        marker_y[name] = np.array(ys)

    inputs = []
    panels = load_all(inputs)
    gauge_md = Data3D_geometry.Data3D_geometry(GAUGE_MD).data
    ref = panels[next(iter(panels))]
    gidx = [mesh_utils.locate(ref.daxis, m)[0] for m in gauge_md]
    sims = {f: np.array([sim_by_gauge(p, gidx)[k] for k in range(1, len(gauge_md) + 1)])
            for f, p in panels.items()}

    keep = np.arange(1, 15)          # gauge 1 sits under the legend box
    best = None
    for n1, v1 in sims.items():
        for n2, v2 in sims.items():
            if n1 == n2:
                continue
            V = np.concatenate([v1[keep], v2[keep]])
            P = np.concatenate([marker_y['C1'][keep], marker_y['C2'][keep]])
            A = np.vstack([V, np.ones_like(V)]).T
            coef, *_ = np.linalg.lstsq(A, P, rcond=None)
            r = float(np.sqrt(((A @ coef - P) ** 2).mean()))
            if best is None or r < best[0]:
                best = (r, n1, n2, coef)
    r, n1, n2, (a, b) = best
    print(f"submitted PNG, best fit over 28 marker positions (rms {r:.3f} px):")
    print(f"  orange  C1 = {n1}   (ratio {ratio_from_name(n1):g})")
    print(f"  green   C2 = {n2}   (ratio {ratio_from_name(n2):g})")

    fld = field_by_gauge([])
    fs = np.array([fld[k] for k in sorted(fld)])
    vals = (marker_y['C0'] - b) / a
    cost = np.abs(vals[1:, None] - fs[None, :])
    ri, ci = linear_sum_assignment(cost)
    perm = [None] * 15
    for i, j in zip(ri, ci):
        perm[i + 1] = j + 1
    perm[0] = (set(range(1, 16)) - set(ci + 1)).pop()
    print(f"  blue    C0 = field data, but at x = 1..15 it plots gauges {perm}")
    print(f"          (gauge 1 is under the legend; |residual| psi = "
          f"{np.round([abs(vals[k] - fs[perm[k]-1]) for k in range(15)], 1).tolist()})")
    listdir_order = [int(re.search(r'gauge(\d+)_', s).group(1))
                     for s in os.listdir(GAUGEDIR)
                     if re.search(r'gauge(\d+)_', s)]
    print(f"  today's os.listdir(gauges/) order = {listdir_order}")
    return {'orange': n1, 'green': n2, 'rms_px': r, 'field_gauge_order': perm}


def build(mode, outdir, dpi, scale_bar=True):
    ds = rc.DualScale(mode)
    inputs = [GAUGE_MD]

    panels = load_all(inputs)
    gauge_md = Data3D_geometry.Data3D_geometry(GAUGE_MD).data
    ref = panels[next(iter(panels))]
    gauge_idx = [mesh_utils.locate(ref.daxis, m)[0] for m in gauge_md]
    n_gauge = len(gauge_md)

    field = field_by_gauge(inputs)
    missing = [k for k in range(1, n_gauge + 1) if k not in field]
    if missing:
        raise RuntimeError(f"no field series for gauge(s) {missing}")

    rows, mono_ok = monotonicity_check(panels, gauge_idx)
    print("  ratio ladder (from filenames) vs drawdown, psi:")
    for ratio, f, far, peak in rows:
        print(f"    {f:20s} ratio={ratio:<9g} gauge1(far)={far:9.1f}  peak={peak:9.1f}")
    print(f"  far-gauge drawdown monotone in ratio: {mono_ok}")
    if not mono_ok:
        raise RuntimeError("filename ratios are not consistent with the physics; "
                           "the labels cannot be trusted")

    for f, _, _ in SCENARIOS:
        if f not in panels:
            raise RuntimeError(f"{f} is missing from {SIMDIR} - the run the "
                               f"caption names is not on disk")

    axis = np.arange(1, n_gauge + 1)
    series = [('Field data', np.array([field[k] for k in axis]), 'field')]
    for fname, ratio, label in SCENARIOS:
        series.append((label, np.array([sim_by_gauge(panels[fname], gauge_idx)[k]
                                        for k in axis]), fname))

    print("  plotted points (gauge number -> psi):")
    for label, y, src in series:
        pretty = ' '.join(f"g{n}={v:.0f}" for n, v in zip(axis, y))
        print(f"    [{src}] {pretty}")

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    colors = []
    for label, y, _src in series:
        ln, = ax.plot(axis, y, label=label, marker='o', linestyle='-')
        colors.append(ln.get_color())
    ax.set_xlabel('Gauge Number')
    ax.legend(loc='center left')

    # Scale bars are drawn in BOTH modes - they are the thing the no_scale
    # version conveys scale WITH, and drawing them identically is what keeps the
    # two renders pixel-identical inside the data area.  Fig. 3's "800 psi" and
    # Fig. 6's "500 psi" bars work the same way.
    span = np.concatenate([s[1] for s in series])
    lo, hi = float(span.min()), float(span.max())
    pad = 0.06 * (hi - lo)
    ax.set_ylim(lo - 2.2 * pad, hi + pad)
    if scale_bar:
        # NOTE: the submitted script draws no scale bar at all - see the manifest.
        xbar = axis[0] + 0.35
        ybar = lo - 1.7 * pad
        ax.plot([xbar, xbar], [ybar, ybar + SCALE_BAR_PSI], color='black', linewidth=4,
                solid_capstyle='butt', zorder=5)
        ax.text(xbar + 0.30, ybar + SCALE_BAR_PSI / 2, f"{SCALE_BAR_PSI} psi",
                fontsize=10, va='center', color='black', zorder=5)

    # gauge NUMBER is an identifier, not a physical scale, so it stays in both
    # modes exactly as the submitted figure has it; only the psi axis is stripped.
    ds.strip(ax, x=False, y=True, keep_ylabel='Pressure Drop Down (psi)')
    ds.freeze(fig, rect=(0.105, 0.035, 0.985, 0.985))
    ds.restore()
    ax.set_ylabel('Pressure Drop Down (psi)', labelpad=26)

    out = rc.save_outputs(fig, outdir, STEM, mode, dpi=dpi)
    boxes = rc.axes_pixel_boxes(fig, [ax], dpi)
    plt.close(fig)

    colour_map = {c: s[0].replace('\n', ' ') for c, s in zip(colors, series)}
    params = {
        'window': [START.isoformat(), END.isoformat()],
        'n_gauges': int(n_gauge),
        'gauge_md_ft': [float(v) for v in gauge_md],
        'field_drawdown_psi_by_gauge': {int(k): field[k] for k in axis},
        'simulated_runs_plotted': [
            {'file': f, 'ratio': r, 'label': lbl.replace('\n', ' '),
             'drawdown_psi_by_gauge':
                 {int(k): sim_by_gauge(panels[f], gauge_idx)[k] for k in axis}}
            for f, r, lbl in SCENARIOS],
        'runs_available_but_not_plotted':
            [f for f in sorted(panels, key=ratio_from_name)
             if f not in {s[0] for s in SCENARIOS}],
        'ratio_ladder_consistency_check': {
            'far_gauge_drawdown_monotone_in_ratio': bool(mono_ok),
            'ladder': [{'file': f, 'ratio': r, 'gauge1_psi': far, 'peak_psi': pk}
                       for r, f, far, pk in rows]},
        'colour_to_scenario': colour_map,
        'scale_bar_psi': SCALE_BAR_PSI if scale_bar else None,
        'drawdown_start_index': 0,
    }
    return out, boxes, inputs, params, colour_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', choices=list(rc.MODES) + ['both'], default='both')
    ap.add_argument('--outdir', default=DEFAULT_OUTDIR)
    ap.add_argument('--dpi', type=int, default=400)
    ap.add_argument('--no-scale-bar', action='store_true',
                    help='omit the 500 psi bar (the submitted script drew none)')
    ap.add_argument('--forensics', action='store_true')
    a = ap.parse_args()

    if a.forensics:
        forensics()
        return

    modes = list(rc.MODES) if a.scale == 'both' else [a.scale]
    outs, boxes, cmap = {}, None, None
    for m in modes:
        print(f"[{STEM}] rendering {m}")
        out, boxes, inputs, params, cmap = build(m, a.outdir, a.dpi,
                                                 scale_bar=not a.no_scale_bar)
        outs[m] = out
    if len(modes) == 2:
        rc.assert_data_area_identical(a.outdir, STEM, boxes)
        print(f"[{STEM}] data-area pixel identity: PASS")
        rc.write_manifest(
            a.outdir, STEM,
            figure='Figure 7 panel (b) - production drawdown vs gauge number',
            source_script=os.path.relpath(os.path.abspath(__file__), os.getcwd()),
            inputs=inputs, outputs=outs, parameters=params,
            changes=[
                '(A) every simulated run is addressed by filename, not by position in '
                'os.listdir; the ratio is parsed from the filename and the label built '
                'from it; no file is dropped.',
                '(A) the run the caption calls "1e-5" is now literally 1e-05_gauge3.npz. '
                'Under today\'s directory order the submitted code would have plotted '
                '0.001_gauge3.npz there and dropped 1e-05_gauge3.npz entirely.',
                '(B) the field series is keyed by the gauge number parsed from each '
                'gauge filename and placed at that number, so both series share one '
                '"Gauge Number" axis.',
                '(C) simulated drawdown now uses time index 0 (was index 1), matching '
                'the field series.',
            ],
            notes=[
                'pack_result (fibeRIS pds.py:415) writes only daxis/taxis/data/'
                'start_time, so no output file records its own diffusivity ratio. The '
                'filename is the only record; it is cross-checked against the physics '
                'by ratio_ladder_consistency_check above.',
                'SCALE BAR: no script in this repo draws a scale bar on this panel. '
                'The 500 psi bar is added here so the no_scale version conveys a '
                'vertical scale at all; pass --no-scale-bar to omit it.',
                'Fig. 7 panel (a) is produced by a different script and was not '
                'touched, opened or regenerated.',
            ])
        print(f"[{STEM}] colour -> scenario: {cmap}")
    print(f"[{STEM}] done -> {a.outdir}")


if __name__ == '__main__':
    main()
