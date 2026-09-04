"""E5 -- Reviewer-1 legibility redraw of Fig. 2 and Fig. 6a.

Round `rev2_20260901`, task E5 of `docs/cc_compute_batch_r2.md` (line 455).
Owner of `output/rev2_20260901/E5/` and of this file.

NO SOLVER IS RUN HERE. Both panels are observation panels: LF-DAS + pressure
gauges + pumping curves, all read straight from `data/fiberis_format/`. House
rule 3 (a manifest per solver run) therefore does not bite; a provenance JSON
with the same content minus the numerics block is written instead
(`e5_provenance.json`), following what F1 did for the same reason.

WHAT REVIEWER 1 ASKED FOR, AND WHERE IT IS IMPLEMENTED
------------------------------------------------------
1. "lighten the waterfall background"          -> `pale_diverging()`; measured
   OKLab chroma drop is written to `e5_legibility_metrics.json`.
2. "replace the cyan traces, thicken them"     -> `ROLE_STYLE`; cyan #00ffff has
   a WCAG contrast of 1.25:1 against white, i.e. it is invisible wherever the
   waterfall is near zero, which is most of the panel.
3. "the frac-hit x markers must actually render" -> `_draw_frac_hits()`. The
   legacy call is `color='lightgray', marker='x', s=40` (coplot_without_scalar
   .py:154,157) drawn on a `bwr` field whose midpoint is pure white: contrast
   1.27:1. Replaced by filled ink markers with a white surface ring.
4. "separate the black dashed gauge lines from the data" -> the gauge locations
   stop being an overlay and become AXIS FURNITURE: a hairline dotted guide in
   chrome grey plus a right-hand tick per gauge. Data is saturated, thick and
   solid; chrome is grey, thin and dotted. Nothing black-dashed is left.

TWO D4 DEFECTS FIXED WHILE HERE
-------------------------------
* `102_IMAGE25abstract.py:144-149`: the `if i == selected_gauge_num:` branch and
  its `else:` are byte-identical (verified: both hash to ed2717bfef3c70f6), so
  the highlighted gauge is not highlighted. Its `ax3` loop (:186-191) has no
  branch at all. `102r:147` draws green. The redraw gives gauge 8 (MD 14821 ft)
  its own colour, its own weight and a legend entry naming it the held-out
  validation point.
* Gauges 6 and 7 are the Dirichlet SOURCE series for phases 1-2 and phase 3
  respectively (`101_fiberis_matching.py:13,16,50-60`) and are then drawn among
  the "measured" curves. They are labelled model INPUTS here, not validation.

COLOUR POLICY (dataviz skill, `references/palette.md`)
-----------------------------------------------------
Categorical slots used: violet `#4a3aa7` (slot 7) and green `#008300` (slot 6).
Validated with the skill's own checker, all-pairs, light surface:

    node scripts/validate_palette.js "#4a3aa7,#008300" --mode light --pairs all
    -> ALL CHECKS PASS; CVD worst-pair dE 27.9 (deutan), normal-vision 34.1

Neither is red nor green-vs-red: the pair is violet/green, which separates under
protanopia, deuteranopia and tritanopia and in greyscale (relative luminance
0.076 vs 0.202). Context gauges take the neutral "Other" grey, and every trace
additionally carries a direct right-hand label, so identity is never colour
alone. The frac-hit markers are drawn in annotation ink rather than in a series
colour: they are geometry, not a series.

Usage
-----
    python3 scripts/manuscript_well_leakage/rev2/e5_redraw.py            # both, both styles
    python3 scripts/manuscript_well_leakage/rev2/e5_redraw.py --figures fig02
    python3 scripts/manuscript_well_leakage/rev2/e5_redraw.py --styles v2

Run from the repo root. Single process, ~2 min, ~1.5 GB peak.
"""

import argparse
import datetime
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.signal import butter, filtfilt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))
_FIBERIS = os.path.join(_REPO, 'fibeRIS', 'src')
if os.path.isdir(_FIBERIS) and _FIBERIS not in sys.path:
    sys.path.insert(0, _FIBERIS)

import rev2_data as rd          # noqa: E402  shared module -- import only
import rev2_manifest as rm      # noqa: E402  shared module -- import only

OUT_DIR = os.path.join(_REPO, 'output', 'rev2_20260901', 'E5')
DPI = 400

# --------------------------------------------------------------------------
# Palette (dataviz skill reference instance; see module docstring)
# --------------------------------------------------------------------------
INK = '#0b0b0b'          # primary ink -- annotation, frac hits
INK_2 = '#52514e'        # secondary ink -- context series ("Other")
MUTED = '#898781'        # axis/label ink
CHROME = '#c3c2b7'       # baseline / guide hairline
SURFACE = '#fcfcfb'

SLOT_VIOLET = '#4a3aa7'  # categorical slot 7
SLOT_GREEN = '#008300'   # categorical slot 6

DIVERGING_POLE_LOW = '#2a78d6'   # palette diverging pair, cool pole
DIVERGING_POLE_HIGH = '#d03b3b'  # palette diverging pair, warm pole
DIVERGING_MID = '#f2f1ee'        # neutral gray midpoint

# role -> (colour, linewidth, linestyle, zorder, right-tick suffix)
ROLE_STYLE = {
    'input':     (SLOT_VIOLET, 2.1, (0, (5.5, 2.2)), 5, 'input'),
    'heldout':   (SLOT_GREEN,  2.9, 'solid',         6, 'held out'),
    'context':   (INK_2,       1.5, 'solid',         4, ''),
    'observed':  (SLOT_VIOLET, 2.4, 'solid',         5, ''),
}

LEGACY_CYAN = '#00ffff'
LEGACY_LIGHTGRAY = '#d3d3d3'


# --------------------------------------------------------------------------
# Colour maths: OKLab chroma and WCAG contrast, so the "it is more legible"
# claim in the README is a measurement rather than an opinion.
# --------------------------------------------------------------------------
def _srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb_to_oklab(rgb):
    """rgb in [0,1], shape (..., 3) -> OKLab (L, a, b)."""
    lin = _srgb_to_linear(rgb)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.stack([
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_], axis=-1)


def oklab_chroma(rgb):
    lab = srgb_to_oklab(rgb)
    return np.hypot(lab[..., 1], lab[..., 2])


def relative_luminance(rgb):
    lin = _srgb_to_linear(rgb)
    return (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2])


def wcag_contrast(rgb_a, rgb_b):
    la = relative_luminance(np.asarray(rgb_a, dtype=float))
    lb = relative_luminance(np.asarray(rgb_b, dtype=float))
    hi, lo = np.maximum(la, lb), np.minimum(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def hex_to_rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


# --------------------------------------------------------------------------
# The pale diverging colormap (reviewer item 1)
# --------------------------------------------------------------------------
def pale_diverging(name='e5_pale_bwr', gamma=1.60, lighten=0.42, n=256):
    """Low-saturation blue<->red diverging map, neutral-gray midpoint.

    `gamma > 1` widens the near-neutral middle, so the bulk of the field (which
    is near zero) recedes and the overlaid traces sit on an almost-white ground.
    `lighten` then blends the whole map toward white. Both are the "low
    saturation version of the diverging colormap" Reviewer 1 offered as an
    alternative to lowering the opacity; doing it in the colormap rather than
    with alpha keeps the traces fully opaque and prints predictably.
    """
    lo = hex_to_rgb(DIVERGING_POLE_LOW)
    hi = hex_to_rgb(DIVERGING_POLE_HIGH)
    mid = hex_to_rgb(DIVERGING_MID)
    t = np.linspace(0.0, 1.0, n)
    u = 2.0 * t - 1.0
    a = np.abs(u) ** gamma
    pole = np.where(u[:, None] < 0, lo[None, :], hi[None, :])
    cols = mid[None, :] * (1.0 - a[:, None]) + pole * a[:, None]
    cols = cols * (1.0 - lighten) + 1.0 * lighten
    return LinearSegmentedColormap.from_list(name, np.clip(cols, 0, 1), N=n)


def cmap_colors(cmap, n=256):
    return np.asarray(cmap(np.linspace(0, 1, n)))[:, :3]


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------
def merge_das(stage_kinds, md_range):
    """Concatenate LF-DAS panels in time. Returns (daxis, t0_abs, taxis_s, data).

    `Data2D.right_merge` is bypassed on purpose: it is the path that forces the
    whole int32 array through `astype(float64)` first (core2D.py:247, and see
    rev2_data.load_das_stage). Contiguity is asserted rather than assumed.
    """
    recs = [rd.load_das_stage(s, kind=k, md_range=md_range) for s, k in stage_kinds]
    base = recs[0]
    gaps = []
    for r in recs[1:]:
        if not np.array_equal(r.daxis_ft, base.daxis_ft):
            raise ValueError(f"channel axis differs between {base.source_path} "
                             f"and {r.source_path}")
    t0 = base.t0_abs
    axes, blocks = [], []
    prev_end = None
    for r in recs:
        off = (r.t0_abs - t0).total_seconds()
        ax = r.taxis_s + off
        if prev_end is not None:
            gaps.append(float(ax[0] - prev_end))
        prev_end = float(ax[-1])
        axes.append(ax)
        blocks.append(r.data)
    taxis = np.concatenate(axes)
    if np.any(np.diff(taxis) <= 0):
        raise ValueError("merged DAS time axis is not strictly increasing")
    data = np.concatenate(blocks, axis=1)
    return base.daxis_ft, t0, taxis, data, gaps, [r.source_path for r in recs]


def lowpass_rows(data, dt, freqcut, order=5):
    """Zero-phase Butterworth low-pass along time; matches
    `Data2D.apply_lowpass_filter` -> `signal_utils.lpfilter` (butter + filtfilt),
    vectorised over channels instead of looped."""
    b, a = butter(order, freqcut / (0.5 / dt), btype='low')
    return filtfilt(b, a, data, axis=-1)


def gauge_traces(numbers, t_start, t_end):
    """Delta-p series for the named gauges over [t_start, t_end].

    `rebase='common'` (rev2_data.load_window_gauges) is required here and not
    the r1 default: the panel's x axis is absolute time shared with the LF-DAS,
    and the per-gauge rebase deliberately reproduces a 16 ms asymmetry that only
    makes sense inside the r1 misfit.
    """
    w = rd.Window(md_min_ft=-1e9, md_max_ft=1e9, t_start=t_start, t_end=t_end)
    gw = rd.load_window_gauges(w, gauges=numbers, rebase='common',
                               baseline='first_sample')
    out = {}
    for n in numbers:
        s = gw.series[int(n)]
        abs_t = np.array([gw.t_ref_abs + datetime.timedelta(seconds=float(v))
                          for v in s.taxis_s])
        out[int(n)] = dict(md=s.md_ft, t=mdates.date2num(abs_t),
                           dp=s.delta_psi, t0=s.t0_abs)
    return out


# --------------------------------------------------------------------------
# Drawing primitives
# --------------------------------------------------------------------------
def _halo(lw, fg='white', alpha=0.9):
    return [pe.withStroke(linewidth=lw, foreground=fg, alpha=alpha)]


def _draw_waterfall(ax, taxis_num, daxis, data, cmap, clim):
    img = ax.imshow(data, aspect='auto', cmap=cmap,
                    vmin=clim[0], vmax=clim[1], interpolation='antialiased',
                    extent=[taxis_num[0], taxis_num[-1], daxis[-1], daxis[0]],
                    zorder=1)
    ax.set_ylim(daxis[-1], daxis[0])   # MD increases downward (well-log sense)
    ax.xaxis_date()
    return img


def _draw_gauge_guides(ax, mds, style):
    """Reviewer item 4. Legacy: black dashed, full weight, sitting in the same
    visual register as the data. v2: chrome-grey dotted hairline BELOW the
    traces, backed by a right-hand tick per gauge, so location reads as axis
    furniture and never as a signal."""
    for md in mds:
        if style == 'legacy':
            ax.axhline(y=md, color='black', linestyle='--', zorder=3)
        else:
            ax.axhline(y=md, color=CHROME, linestyle=(0, (1.0, 3.0)),
                       linewidth=0.7, zorder=2, alpha=0.95)


def _draw_frac_hits(ax, entries, style):
    """entries: list of (label, x_datetime, mds, marker)."""
    handles = []
    for label, x, mds, marker in entries:
        xs = np.repeat(mdates.date2num(x), len(mds))
        if style == 'legacy':
            ax.scatter(xs, mds, color=LEGACY_LIGHTGRAY, marker='x', s=40,
                       zorder=7)
            handles.append(Line2D([], [], linestyle='none', marker='x',
                                  color=LEGACY_LIGHTGRAY, markersize=6,
                                  label=label))
        else:
            ax.scatter(xs, mds, c=INK, marker=marker, s=130, zorder=7,
                       edgecolors='white', linewidths=1.3)
            handles.append(Line2D([], [], linestyle='none', marker=marker,
                                  color=INK, markersize=9,
                                  markeredgecolor='white', markeredgewidth=1.2,
                                  label=label))
    return handles


def _right_gauge_axis(ax, mds, labels, colours):
    axr = ax.twinx()
    axr.set_ylim(ax.get_ylim())
    axr.set_yticks(list(mds))
    axr.set_yticklabels(labels, fontsize=7.5)
    for tick, c in zip(axr.get_yticklabels(), colours):
        tick.set_color(c)
    axr.tick_params(axis='y', length=3, width=0.8, colors=MUTED, pad=2)
    for sp in axr.spines.values():
        sp.set_visible(False)
    return axr


def _scale_bar(ax, x_dt, md_top, psi, coeff, colour=INK, label_side='right'):
    """Vertical bar whose length is `psi` psi at the panel's own psi->ft gain.
    The legacy figures drew this in the same cyan as the data, or dropped it
    entirely (`coplot_without_scalar.py:145-146` comments it out), which is why
    no amplitude in Fig. 2 is currently quotable."""
    x = mdates.date2num(x_dt)
    y0, y1 = md_top, md_top - psi * coeff
    ax.plot([x, x], [y0, y1], color=colour, linewidth=4.0,
            solid_capstyle='butt', zorder=8, path_effects=_halo(6.4))
    ax.annotate(f"{psi:g} psi", xy=(x, 0.5 * (y0 + y1)),
                xytext=(11 if label_side == 'right' else -11, 0),
                textcoords='offset points', fontsize=7.5, color=INK,
                va='center', ha='left' if label_side == 'right' else 'right',
                path_effects=_halo(2.6))


def _add_colorbar(fig, ax, img, *, x0, width=0.017):
    """Colour bar as an explicitly placed axes rather than a gridspec column:
    the right-hand gauge labels need the gap, and a gridspec wspace cannot be
    made large enough without shrinking the panel."""
    pos = ax.get_position()
    cax = fig.add_axes([x0, pos.y0, width, pos.height])
    cb = fig.colorbar(img, cax=cax, extend='both')
    cb.set_label('LF-DAS (counts)', fontsize=7.5, color=INK_2)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(CHROME)
    return cb


def _style_axes(ax, ylabel=None, xlabel=None):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(CHROME)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=7.5, width=0.8, length=3)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_color(INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK_2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=INK_2)


def _pumping_panel(ax, ps, key, colour, ylabel):
    t = np.array([ps.t0_abs + datetime.timedelta(seconds=float(v))
                  for v in ps.taxis_s])
    y = np.asarray(ps.curves[key].data, dtype=float)
    ax.plot(mdates.date2num(t), y, color=colour, linewidth=1.3)
    ax.xaxis_date()
    _style_axes(ax, ylabel=ylabel)
    ax.grid(axis='y', color='#e1e0d9', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    return float(np.nanmax(y)) if y.size else np.nan


# --------------------------------------------------------------------------
# Figure 2
# --------------------------------------------------------------------------
FIG02_STAGE_A, FIG02_STAGE_B = 8, 9
FIG02_COEFF = 0.14       # ft per psi, coplot_without_scalar.py:18
FIG02_CLIM = 300.0       # counts, :149
FIG02_LOWPASS_HZ = 0.01  # :104


def build_fig02_data():
    fh_a = rd.load_frac_hits(FIG02_STAGE_A, unique=False)
    fh_b = rd.load_frac_hits(FIG02_STAGE_B, unique=False)
    md_lo = float(np.min(fh_b)) - 500.0      # :80 (stage2 = 9)
    md_hi = float(np.max(fh_a)) + 700.0      # :81 (stage1 = 8)

    ps_a = rd.load_pumping(FIG02_STAGE_A)
    ps_b = rd.load_pumping(FIG02_STAGE_B)
    t_start, t_end = ps_a.file_start_abs, ps_b.file_end_abs

    md_tab = rd.load_gauge_md_table()
    sel = [int(n) for n, md in zip(md_tab.numbers, md_tab.md_ft)
           if md_lo < md < md_hi]                                   # :83-85
    tr = gauge_traces(sel, t_start, t_end)

    daxis, t0, taxis, data, gaps, paths = merge_das(
        [(FIG02_STAGE_A, 'stage'), (FIG02_STAGE_A, 'interval'),
         (FIG02_STAGE_B, 'stage')], (md_lo, md_hi))
    dt = float(np.median(np.diff(taxis)))
    data = lowpass_rows(data, dt, FIG02_LOWPASS_HZ, order=5)
    abs_t = np.array([t0 + datetime.timedelta(seconds=float(v)) for v in taxis])
    return dict(md_lo=md_lo, md_hi=md_hi, frac=dict(a=fh_a, b=fh_b),
                pump=dict(a=ps_a, b=ps_b), t_start=t_start, t_end=t_end,
                gauges=sel, traces=tr, daxis=daxis, taxis_num=mdates.date2num(abs_t),
                data=data, das_paths=paths, das_gaps=gaps, das_dt=dt)


def render_fig02(D, style, out_path):
    cmap = plt.get_cmap('bwr') if style == 'legacy' else pale_diverging()
    legacy = (style == 'legacy')
    fig = plt.figure(figsize=(7.8, 9.1), facecolor='white')
    gs = fig.add_gridspec(4, 1, height_ratios=[4.9, 1.0, 0.85, 0.85],
                          hspace=0.17,
                          left=0.095, right=0.845, top=0.945, bottom=0.155)
    ax = fig.add_subplot(gs[0])
    img = _draw_waterfall(ax, D['taxis_num'], D['daxis'], D['data'], cmap,
                          (-FIG02_CLIM, FIG02_CLIM))
    _draw_gauge_guides(ax, [D['traces'][g]['md'] for g in D['gauges']], style)

    col, lw, ls, z, _ = ROLE_STYLE['observed']
    trace_colour = LEGACY_CYAN if legacy else col
    trace_lw = 2.0 if legacy else lw
    for g in D['gauges']:
        s = D['traces'][g]
        y = -s['dp'] * FIG02_COEFF + s['md']
        ax.plot(s['t'], y, color=trace_colour, linewidth=trace_lw,
                linestyle='solid', zorder=z,
                path_effects=None if legacy else _halo(trace_lw + 1.6))

    fh_x_a = D['pump']['a'].file_start_abs + datetime.timedelta(minutes=32)
    fh_x_b = D['pump']['b'].file_start_abs + datetime.timedelta(minutes=32)
    fh_handles = _draw_frac_hits(
        ax, [(f"Frac hit, stage {FIG02_STAGE_A}", fh_x_a, D['frac']['a'], 'X'),
             (f"Frac hit, stage {FIG02_STAGE_B}", fh_x_b, D['frac']['b'], 'P')],
        style)

    _style_axes(ax, ylabel='Measured depth (ft)')
    ax.set_xlim(mdates.date2num(D['t_start']), mdates.date2num(D['t_end']))
    plt.setp(ax.get_xticklabels(), visible=False)

    if legacy:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(handles=[Line2D([], [], color=LEGACY_CYAN, linewidth=2.0,
                                  label='Pressure gauge')] + fh_handles,
                  loc='lower right', fontsize=8)
    else:
        _scale_bar(ax, D['t_start'] + datetime.timedelta(minutes=48),
                   D['md_lo'] + 160.0, 800, FIG02_COEFF)
        mds = [D['traces'][g]['md'] for g in D['gauges']]
        _right_gauge_axis(ax, mds, [f"g{g}" for g in D['gauges']],
                          [trace_colour] * len(mds))
        handles = [Line2D([], [], color=trace_colour, linewidth=trace_lw,
                          label=r'Gauge $\Delta p$ (gain: scale bar)')] \
            + fh_handles \
            + [Line2D([], [], color=CHROME, linewidth=0.9,
                      linestyle=(0, (1.0, 3.0)),
                      label='Gauge location (axis guide)')]
        leg = fig.legend(handles=handles, loc='lower center', ncol=2,
                         bbox_to_anchor=(0.5, 0.052), fontsize=7.6,
                         frameon=False, handlelength=2.4, columnspacing=2.2)
        ax.set_title(
            "(a)  LF-DAS waterfall, S well, stages "
            f"{FIG02_STAGE_A} and {FIG02_STAGE_B}, with the co-located gauge "
            r"$\Delta p$ overlaid",
            fontsize=9.5, color=INK, loc='left', pad=6)

    _add_colorbar(fig, ax, img, x0=0.900)

    axes_p = []
    specs = [('treating_pressure', INK, 'Treating\npressure (psi)', '(b)'),
             ('slurry_rate', INK_2, 'Slurry rate\n(bbl/min)', '(c)'),
             ('proppant_concentration', MUTED, 'Proppant\n(lb/gal)', '(d)')]
    for row, (key, colour, ylab, tag) in enumerate(specs, start=1):
        axp = fig.add_subplot(gs[row], sharex=ax)
        for ps in (D['pump']['a'], D['pump']['b']):
            _pumping_panel(axp, ps, key, colour, ylab)
        axes_p.append(axp)
        if legacy:
            axp.set_xticks([])
            axp.set_yticks([])
            axp.set_ylabel('')
        else:
            axp.annotate(tag, xy=(0.004, 0.90), xycoords='axes fraction',
                         fontsize=8.5, color=INK, ha='left', va='top')
    for axp in axes_p[:-1]:
        plt.setp(axp.get_xticklabels(), visible=False)
    axes_p[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    if not legacy:
        axes_p[-1].set_xlabel('Time on 2020-03-18 (UTC)', fontsize=8, color=INK_2)
        fig.text(0.5, 0.014,
                 "Panels (b)-(d): one pumping channel per axis. The submitted "
                 "figure stacked four y-scales on a single panel.\n"
                 "Colour bar is clipped at " + u"±" + f"{FIG02_CLIM:g}"
                 " counts (arrows); LF-DAS is left in raw counts (F1).",
                 fontsize=6.9, color=MUTED, ha='center', va='bottom')

    fig.savefig(out_path, dpi=DPI, facecolor='white')
    plt.close(fig)
    return out_path

# --------------------------------------------------------------------------
# Figure 6a
# --------------------------------------------------------------------------
FIG06_STAGE_A, FIG06_STAGE_B = 7, 8
FIG06_COEFF = 0.2        # ft per psi, 104_full:131
FIG06_CLIM = 300.0       # counts, 104_full:159
HELD_OUT_GAUGE = 8       # D4: MD 14821 ft, the sole held-out comparison
BC_GAUGES = {6: 'phases 1-2', 7: 'phase 3'}   # 101_fiberis_matching.py:13,16
BC_SHORT = {6: '1-2', 7: '3'}                 # right-hand tick labels


def build_fig06a_data():
    fh_a = rd.load_frac_hits(FIG06_STAGE_A, unique=False)
    fh_b = rd.load_frac_hits(FIG06_STAGE_B, unique=False)
    md_lo = float(np.min(fh_b)) - 500.0      # 104_full:62,65
    md_hi = float(np.max(fh_a)) + 500.0

    ps_a = rd.load_pumping(FIG06_STAGE_A)
    ps_b = rd.load_pumping(FIG06_STAGE_B)
    t_start, t_end = ps_a.file_start_abs, ps_b.file_end_abs
    phases = rd.manuscript_phase_windows()

    md_tab = rd.load_gauge_md_table()
    sel = [int(n) for n, md in zip(md_tab.numbers, md_tab.md_ft)
           if md_lo <= md <= md_hi]
    tr = gauge_traces(sel, t_start, t_end)

    daxis, t0, taxis, data, gaps, paths = merge_das(
        [(FIG06_STAGE_A, 'stage'), (FIG06_STAGE_A, 'interval'),
         (FIG06_STAGE_B, 'stage')], (md_lo, md_hi))
    abs_t = np.array([t0 + datetime.timedelta(seconds=float(v)) for v in taxis])
    return dict(md_lo=md_lo, md_hi=md_hi, frac=dict(a=fh_a, b=fh_b),
                pump=dict(a=ps_a, b=ps_b), t_start=t_start, t_end=t_end,
                phases=phases, gauges=sel, traces=tr, daxis=daxis,
                taxis_num=mdates.date2num(abs_t), data=data, das_paths=paths,
                das_gaps=gaps, das_dt=float(np.median(np.diff(taxis))))


def _fig06_role(g):
    if g == HELD_OUT_GAUGE:
        return 'heldout'
    if g in BC_GAUGES:
        return 'input'
    return 'context'


def render_fig06a(D, style, out_path):
    cmap = plt.get_cmap('bwr') if style == 'legacy' else pale_diverging()
    legacy = (style == 'legacy')
    fig = plt.figure(figsize=(7.8, 7.5), facecolor='white')
    gs = fig.add_gridspec(2, 1, height_ratios=[4.6, 1.05], hspace=0.15,
                          left=0.10, right=0.805, top=0.935, bottom=0.205)
    ax = fig.add_subplot(gs[0])
    img = _draw_waterfall(ax, D['taxis_num'], D['daxis'], D['data'], cmap,
                          (-FIG06_CLIM, FIG06_CLIM))
    _draw_gauge_guides(ax, [D['traces'][g]['md'] for g in D['gauges']], style)

    right_labels, right_colours, mds = [], [], []
    for g in D['gauges']:
        s = D['traces'][g]
        role = _fig06_role(g)
        col, lw, ls, z, _ = ROLE_STYLE[role]
        if legacy:
            # 104_full:144-149 -- selected gauge black, the rest cyan.
            # (102_IMAGE25abstract.py:144-149 draws BOTH branches cyan.)
            col = 'black' if g == HELD_OUT_GAUGE else LEGACY_CYAN
            lw, ls, z = 2.0, 'solid', 5
        y = -s['dp'] * FIG06_COEFF + s['md']
        ax.plot(s['t'], y, color=col, linewidth=lw, linestyle=ls, zorder=z,
                path_effects=None if legacy else _halo(lw + 1.6))
        mds.append(s['md'])
        if g == HELD_OUT_GAUGE:
            right_labels.append(f"g{g}\nheld out")
        elif g in BC_GAUGES:
            right_labels.append(f"g{g}\ninput {BC_SHORT[g]}")
        else:
            right_labels.append(f"g{g}")
        right_colours.append(col)

    fh_x_a = D['pump']['a'].file_start_abs + datetime.timedelta(minutes=28)
    fh_x_b = D['pump']['b'].file_start_abs + datetime.timedelta(minutes=28)
    fh_handles = _draw_frac_hits(
        ax, [(f"Frac hit, stage {FIG06_STAGE_A}", fh_x_a, D['frac']['a'], 'X'),
             (f"Frac hit, stage {FIG06_STAGE_B}", fh_x_b, D['frac']['b'], 'P')],
        style)

    _style_axes(ax, ylabel='Measured depth (ft)')
    ax.set_xlim(mdates.date2num(D['t_start']), mdates.date2num(D['t_end']))
    plt.setp(ax.get_xticklabels(), visible=False)

    if legacy:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # Phase boundaries. These are FILE spans, not pumping events
        # (rev2_data.manuscript_phase_windows / PumpingStage.pumping_start).
        b1 = D['phases']['phase1'][2]
        b2 = D['phases']['phase3'][1]
        for b in (b1, b2):
            ax.axvline(mdates.date2num(b), color=INK_2, linewidth=0.9,
                       linestyle=(0, (2.5, 2.0)), zorder=3, alpha=0.75)
        blend = ax.get_xaxis_transform()
        edges = [mdates.date2num(D['t_start']), mdates.date2num(b1),
                 mdates.date2num(b2), mdates.date2num(D['t_end'])]
        for k, name in enumerate(('phase 1', 'phase 2', 'phase 3')):
            ax.text(0.5 * (edges[k] + edges[k + 1]), 0.012, name,
                    transform=blend, ha='center', va='bottom', fontsize=7.4,
                    color=INK, path_effects=_halo(3.0), zorder=9)
        _scale_bar(ax, D['t_start'] + datetime.timedelta(minutes=72),
                   D['md_lo'] + 200.0, 500, FIG06_COEFF)
        _right_gauge_axis(ax, mds, right_labels, right_colours)

        handles = [
            Line2D([], [], color=ROLE_STYLE['heldout'][0],
                   linewidth=ROLE_STYLE['heldout'][1],
                   label='Held-out validation gauge: g8, MD 14821 ft'),
            Line2D([], [], color=ROLE_STYLE['input'][0],
                   linewidth=ROLE_STYLE['input'][1],
                   linestyle=ROLE_STYLE['input'][2],
                   label='Model INPUT, Dirichlet source: g6, g7'),
            Line2D([], [], color=ROLE_STYLE['context'][0],
                   linewidth=ROLE_STYLE['context'][1],
                   label='Measured, not used by the model: g5, g9'),
        ] + fh_handles + [
            Line2D([], [], color=CHROME, linewidth=0.9,
                   linestyle=(0, (1.0, 3.0)),
                   label='Gauge location (axis guide)')]
        fig.legend(handles=handles, loc='lower center', ncol=2,
                   bbox_to_anchor=(0.5, 0.068), fontsize=7.5, frameon=False,
                   handlelength=2.6, columnspacing=2.2)
        ax.set_title(
            "(a)  Observed LF-DAS and gauge response, S well, stages "
            f"{FIG06_STAGE_A} and {FIG06_STAGE_B}",
            fontsize=9.5, color=INK, loc='left', pad=6)

    _add_colorbar(fig, ax, img, x0=0.900)

    axp = fig.add_subplot(gs[1], sharex=ax)
    for ps in (D['pump']['a'], D['pump']['b']):
        _pumping_panel(axp, ps, 'treating_pressure', INK,
                       'Treating\npressure (psi)')
    if legacy:
        axp.set_xticks([])
        axp.set_yticks([])
        axp.set_ylabel('')
    else:
        for b in (D['phases']['phase1'][2], D['phases']['phase3'][1]):
            axp.axvline(mdates.date2num(b), color=INK_2, linewidth=0.9,
                        linestyle=(0, (2.5, 2.0)), alpha=0.75)
        axp.annotate('(c)', xy=(0.004, 0.90), xycoords='axes fraction',
                     fontsize=8.5, color=INK, ha='left', va='top')
        axp.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axp.set_xlabel('Time on 2020-03-18 (UTC)', fontsize=8, color=INK_2)
        fig.text(0.5, 0.014,
                 "Gauges 6 and 7 impose the Dirichlet boundary condition "
                 "(phases 1-2 and phase 3) and are model inputs, not "
                 "validation data;\ngauge 8 takes no part in the model. Phase "
                 "bounds are FILE spans, not pumping events. Colour bar clipped "
                 "at " + u"±" + f"{FIG06_CLIM:g} counts (arrows).",
                 fontsize=6.9, color=MUTED, ha='center', va='bottom')

    fig.savefig(out_path, dpi=DPI, facecolor='white')
    plt.close(fig)
    return out_path

# --------------------------------------------------------------------------
# Legibility metrics -- the before/after claim, measured
# --------------------------------------------------------------------------
def legibility_metrics():
    legacy_cmap = plt.get_cmap('bwr')
    v2_cmap = pale_diverging()
    out = {}
    for name, cm in (('bwr_legacy', legacy_cmap), ('e5_pale_diverging', v2_cmap)):
        cols = cmap_colors(cm)
        out[name] = dict(
            mean_oklab_chroma=float(np.mean(oklab_chroma(cols))),
            max_oklab_chroma=float(np.max(oklab_chroma(cols))),
            mean_relative_luminance=float(np.mean(relative_luminance(cols))),
            min_relative_luminance=float(np.min(relative_luminance(cols))))
    out['chroma_reduction_pct'] = 100.0 * (
        1.0 - out['e5_pale_diverging']['mean_oklab_chroma']
        / out['bwr_legacy']['mean_oklab_chroma'])

    def worst_contrast(hexcol, cm):
        c = hex_to_rgb(hexcol)
        cols = cmap_colors(cm)
        v = wcag_contrast(np.broadcast_to(c, cols.shape), cols)
        return float(np.min(v)), float(np.max(v))

    marks = {}
    marks['legacy_cyan_trace_on_bwr'] = worst_contrast(LEGACY_CYAN, legacy_cmap)
    marks['legacy_lightgray_frachit_on_bwr'] = worst_contrast(LEGACY_LIGHTGRAY,
                                                              legacy_cmap)
    marks['v2_violet_trace_on_pale'] = worst_contrast(SLOT_VIOLET, v2_cmap)
    marks['v2_green_heldout_on_pale'] = worst_contrast(SLOT_GREEN, v2_cmap)
    marks['v2_grey_context_on_pale'] = worst_contrast(INK_2, v2_cmap)
    marks['v2_ink_frachit_on_pale'] = worst_contrast(INK, v2_cmap)
    out['worst_best_wcag_contrast_vs_colormap'] = {
        k: dict(worst=v[0], best=v[1]) for k, v in marks.items()}

    out['contrast_vs_white'] = {
        'legacy_cyan': float(wcag_contrast(hex_to_rgb(LEGACY_CYAN),
                                           np.ones(3))),
        'legacy_lightgray': float(wcag_contrast(hex_to_rgb(LEGACY_LIGHTGRAY),
                                                np.ones(3))),
        'v2_violet': float(wcag_contrast(hex_to_rgb(SLOT_VIOLET), np.ones(3))),
        'v2_green': float(wcag_contrast(hex_to_rgb(SLOT_GREEN), np.ones(3))),
        'v2_grey_context': float(wcag_contrast(hex_to_rgb(INK_2), np.ones(3))),
        'v2_ink': float(wcag_contrast(hex_to_rgb(INK), np.ones(3)))}

    # Greyscale-print separability of the three Fig. 6a roles.
    out['greyscale_relative_luminance'] = {
        'violet_input': float(relative_luminance(hex_to_rgb(SLOT_VIOLET))),
        'green_heldout': float(relative_luminance(hex_to_rgb(SLOT_GREEN))),
        'grey_context': float(relative_luminance(hex_to_rgb(INK_2))),
        'legacy_cyan': float(relative_luminance(hex_to_rgb(LEGACY_CYAN)))}

    out['line_widths_pt'] = {
        'legacy_all_traces': 2.0,
        'v2_observed': ROLE_STYLE['observed'][1],
        'v2_heldout': ROLE_STYLE['heldout'][1],
        'v2_input': ROLE_STYLE['input'][1],
        'v2_context': ROLE_STYLE['context'][1]}
    out['frac_hit_marker_area_pt2'] = {'legacy': 40, 'v2': 130}
    return out


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def write_provenance(path, products, inputs, metrics, notes):
    doc = {
        'study_id': 'rev2_20260901',
        'task_id': 'E5',
        'kind': 'figure_redraw_no_solver',
        'why_no_manifest': (
            'House rule 3 requires a manifest per SOLVER run. E5 runs no '
            'solver: both panels are observation panels read from '
            'data/fiberis_format/. This document carries the same provenance '
            'content minus the numerics block.'),
        'written_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'cwd': os.getcwd(),
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'matplotlib': matplotlib.__version__,
        'code_closure': rm.code_closure(extra=(os.path.abspath(__file__),)),
        'inputs': [rm.file_record(p, role=r)
                   for p, r in sorted(set(inputs))],
        'outputs': [rm.output_decl(p, role='figure_png', dpi=DPI)
                    for p in products],
        'legibility_metrics': metrics,
        'notes': notes,
    }
    # rev2_manifest._jsonify returns a 3-TUPLE (doc, none_paths, nonfinite_paths),
    # not a document. Dumping its return value directly writes a 3-element list
    # with the document at index 0 -- which is what e5_provenance_v3.json is.
    payload, none_paths, nonfinite_paths = rm._jsonify(doc)
    payload['jsonify_none_declared_paths'] = none_paths
    payload['jsonify_nonfinite_paths'] = nonfinite_paths
    with open(path, 'w') as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    return path


def _input_records(gauges, das_paths, stages):
    recs = [(p, 'das') for p in das_paths]
    recs += [(rd.repo_path(rd.SWELL_GAUGE_TEMPLATE.format(n=g)), 'gauge_series')
             for g in gauges]
    recs += [(rd.repo_path(rd.SWELL_FRAC_HIT_TEMPLATE.format(stage=s)), 'geometry')
             for s in stages]
    recs += [(os.path.join(rd.repo_path(rd.PUMPING_DIR_TEMPLATE.format(stage=s)), f),
              'pumping')
             for s in stages for f in rd.PUMPING_CURVE_FILES.values()]
    recs.append((rd.repo_path(rd.SWELL_GAUGE_MD_NPZ), 'geometry'))
    return recs


def _selected_gauges(md_lo, md_hi, inclusive):
    """The gauge-selection rule of the legacy scripts, without loading any DAS."""
    md_tab = rd.load_gauge_md_table()
    if inclusive:
        return [int(n) for n, md in zip(md_tab.numbers, md_tab.md_ft)
                if md_lo <= md <= md_hi]
    return [int(n) for n, md in zip(md_tab.numbers, md_tab.md_ft)
            if md_lo < md < md_hi]


def _provenance_only(args):
    """Rebuild the input/output inventory for an existing render, cheaply.

    Everything here is deterministic given `--version`: the gauge selection is a
    pure function of the frac-hit and gauge-MD files, and the DAS file list is a
    template. No array is read, so this runs in well under a second.
    """
    fh8, fh9 = rd.load_frac_hits(8, unique=False), rd.load_frac_hits(9, unique=False)
    fh7 = rd.load_frac_hits(7, unique=False)
    inputs, products, notes = [], [], []

    g02 = _selected_gauges(float(np.min(fh9)) - 500.0,
                           float(np.max(fh8)) + 700.0, inclusive=False)
    das02 = [rd.repo_path(rd.SWELL_DAS_TEMPLATE.format(stage=s, kind=k))
             for s, k in ((8, ''), (8, '_interval'), (9, ''))]
    inputs += _input_records(g02, das02, (FIG02_STAGE_A, FIG02_STAGE_B))
    notes.append(f"fig02 gauges plotted: {g02}")

    g06 = _selected_gauges(float(np.min(fh8)) - 500.0,
                           float(np.max(fh7)) + 500.0, inclusive=True)
    das06 = [rd.repo_path(rd.SWELL_DAS_TEMPLATE.format(stage=s, kind=k))
             for s, k in ((7, ''), (7, '_interval'), (8, ''))]
    inputs += _input_records(g06, das06, (FIG06_STAGE_A, FIG06_STAGE_B))
    notes.append(f"fig06a gauges plotted: {g06}; held out g{HELD_OUT_GAUGE}; "
                 f"BC inputs {sorted(BC_GAUGES)}")

    for stem in (f'fig02_das_gauge_coplot_{args.version}.png',
                 f'fig02_das_gauge_coplot_legacy_{args.version}.png',
                 f'fig06a_observed_das_gauges_{args.version}.png',
                 f'fig06a_observed_das_gauges_legacy_{args.version}.png'):
        p = os.path.join(args.outdir, stem)
        if os.path.exists(p):
            products.append(p)
    notes.append('provenance re-emitted with --provenance-only; the figures '
                 'themselves were not re-rendered.')

    ppath = os.path.join(args.outdir,
                         f'e5_provenance_{args.version}_fixed.json')
    if os.path.exists(ppath):
        raise SystemExit(f"refusing to overwrite {ppath} (house rule 2)")
    write_provenance(ppath, products, inputs, legibility_metrics(), notes)
    print('wrote', ppath)
    for n in notes:
        print('  note:', n)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--figures', nargs='+', default=['fig02', 'fig06a'],
                    choices=['fig02', 'fig06a'])
    ap.add_argument('--styles', nargs='+', default=['legacy', 'v2'],
                    choices=['legacy', 'v2'])
    ap.add_argument('--outdir', default=OUT_DIR)
    ap.add_argument('--version', default='v2',
                    help='suffix for the redrawn products')
    ap.add_argument('--provenance-only', action='store_true',
                    help='re-emit the provenance record for an already-rendered '
                         'version, as e5_provenance_<version>_fixed.json. Exists '
                         'because the first v3 emission dumped _jsonify\'s '
                         '3-tuple; house rule 2 forbids overwriting that file.')
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    products, inputs, notes = [], [], []

    if args.provenance_only:
        return _provenance_only(args)

    def _target(stem):
        p = os.path.join(args.outdir, stem)
        if os.path.exists(p):
            raise SystemExit(
                f"refusing to overwrite {p} (house rule 2). Bump --version.")
        return p

    if 'fig02' in args.figures:
        D = build_fig02_data()
        inputs += _input_records(D['gauges'], D['das_paths'],
                                 (FIG02_STAGE_A, FIG02_STAGE_B))
        notes.append(f"fig02 DAS merge gaps (s): {D['das_gaps']}; dt {D['das_dt']}")
        notes.append(f"fig02 gauges plotted: {D['gauges']} at MD "
                     f"{[round(D['traces'][g]['md'], 1) for g in D['gauges']]}")
        for st in args.styles:
            stem = (f'fig02_das_gauge_coplot_legacy_{args.version}.png'
                    if st == 'legacy'
                    else f'fig02_das_gauge_coplot_{args.version}.png')
            products.append(render_fig02(D, st, _target(stem)))
            print('wrote', products[-1])
        del D

    if 'fig06a' in args.figures:
        D = build_fig06a_data()
        inputs += _input_records(D['gauges'], D['das_paths'],
                                 (FIG06_STAGE_A, FIG06_STAGE_B))
        notes.append(f"fig06a DAS merge gaps (s): {D['das_gaps']}")
        notes.append(f"fig06a gauges plotted: {D['gauges']} at MD "
                     f"{[round(D['traces'][g]['md'], 1) for g in D['gauges']]}; "
                     f"held out g{HELD_OUT_GAUGE}; BC inputs {sorted(BC_GAUGES)}")
        for st in args.styles:
            stem = (f'fig06a_observed_das_gauges_legacy_{args.version}.png'
                    if st == 'legacy'
                    else f'fig06a_observed_das_gauges_{args.version}.png')
            products.append(render_fig06a(D, st, _target(stem)))
            print('wrote', products[-1])
        del D

    metrics = legibility_metrics()
    mpath = os.path.join(args.outdir, f'e5_legibility_metrics_{args.version}.json')
    if not os.path.exists(mpath):
        with open(mpath, 'w') as fh:
            json.dump(metrics, fh, indent=2)
        print('wrote', mpath)
    ppath = os.path.join(args.outdir, f'e5_provenance_{args.version}.json')
    if not os.path.exists(ppath):
        write_provenance(ppath, products, inputs, metrics, notes)
        print('wrote', ppath)
    for n in notes:
        print('  note:', n)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
