"""Shared machinery for the SJ-0626-0151 revision figures.

Three things live here and nothing else:

1. **The dual output.** Every revision figure is rendered twice from ONE code
   path, selected by ``--scale {no_scale,with_scale}``:

       no_scale    no tick labels, no numeric axis values, no colorbar numbers.
                   Scale bars stay. THIS is the manuscript version.
       with_scale  ticks, tick labels and a numeric colorbar. Internal use.

   The two must differ *only* in that furniture, so the axes rectangles are
   frozen before the furniture is added: the layout pass always runs with the
   ticks stripped, and ``with_scale`` re-enables them afterwards and hangs the
   colorbar in a reserved right-hand strip that is present (and empty) in
   ``no_scale`` too.  `assert_data_area_identical` checks the result.

2. **The three Reviewer-1 legibility settings**, defined once so Fig. 3 and
   Fig. 6a are literally the same numbers.

3. **A run manifest** - input paths + SHA-256, parameters, environment - written
   beside every figure, per the round's house rule 3.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap   # noqa: E402


# ---------------------------------------------------------------- appearance

#: Item 1 - the waterfall is lightened, not made transparent: overlays stay
#: fully opaque and the figure prints predictably.  The map is still `bwr` and
#: the clim is still symmetric +/-3e2; every colour is blended this far toward
#: white.  0.45 lifts the darkest colour in the map from relative luminance
#: 0.0722 (saturated `bwr` blue) to 0.2305, which is what buys the overlays
#: their contrast.  See `contrast_report()`.
WHITE_BLEND = 0.45

#: Item 2 - the gauge traces.  Cyan #00ffff has a worst-case WCAG contrast of
#: EXACTLY 1.00:1 against `bwr` *and against every lightened version of it*
#: (there is a step in the map whose luminance equals cyan's 0.787), so
#: lightening alone cannot rescue it - the colour has to change.  This deep
#: indigo reads as "dark purple", scores 3.73:1 worst-case on the lightened map
#: and 13.98:1 on white.
GAUGE_COLOR = '#2e2160'
GAUGE_LW = 3.0            # was 2.0
GAUGE_SCALEBAR_LW = 6.0   # was 5.0

#: Item 3 - the frac-hit markers.  `lightgray` + `marker='x'` + `s=40` was
#: 1.01:1 against the map's white midpoint, which is why Reviewer 1 could not
#: find them.  A filled `X` takes a face colour AND an edge, so the mark carries
#: its own contrast wherever it lands.
FRACHIT_COLOR = '#0b0b0b'
FRACHIT_EDGE = 'white'
FRACHIT_EDGE_LW = 0.9
FRACHIT_MARKER = 'X'
FRACHIT_SIZE = 150        # was 40

#: reserved right-hand strip: the colorbar goes here in with_scale, and the
#: strip is left empty in no_scale so both modes share one axes rectangle.
LAYOUT_RECT = (0.045, 0.02, 0.895, 0.985)
CBAR_RECT_UPPER = (0.915, 0.42, 0.017, 0.40)
CBAR_RECT_RIGHT = (0.915, 0.42, 0.017, 0.40)

MODES = ('no_scale', 'with_scale')


def paled_bwr(blend: float = WHITE_BLEND, n: int = 256) -> LinearSegmentedColormap:
    """`bwr`, every colour blended `blend` of the way to white.

    Still diverging, still white at the midpoint, still symmetric - only the
    saturation of the two poles changes, which is exactly what Reviewer 1 asked
    for ("perhaps paling the waterfall plot").
    """
    base = matplotlib.colormaps['bwr'](np.linspace(0, 1, n))[:, :3]
    return LinearSegmentedColormap.from_list(
        f'bwr_pale{blend:.2f}', base * (1.0 - blend) + blend, N=n)


def _rel_lum(rgb):
    c = np.asarray(rgb, float)
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(c @ np.array([0.2126, 0.7152, 0.0722]))


def _contrast(a, b):
    la, lb = _rel_lum(a), _rel_lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def contrast_report(blend: float = WHITE_BLEND) -> dict:
    """Worst-case WCAG contrast of each mark against every colour in the map."""
    import matplotlib.colors as mcolors
    legacy = matplotlib.colormaps['bwr'](np.linspace(0, 1, 256))[:, :3]
    pale = legacy * (1 - blend) + blend
    out = {'white_blend': blend,
           'darkest_relative_luminance': {'legacy_bwr': min(_rel_lum(c) for c in legacy),
                                          'paled_bwr': min(_rel_lum(c) for c in pale)}}
    marks = {'legacy cyan #00ffff': 'cyan',
             'legacy frac-hit lightgray': 'lightgray',
             f'revision gauge {GAUGE_COLOR}': GAUGE_COLOR,
             f'revision frac-hit {FRACHIT_COLOR}': FRACHIT_COLOR}
    out['worst_case_contrast'] = {}
    for name, col in marks.items():
        rgb = mcolors.to_rgb(col)
        out['worst_case_contrast'][name] = {
            'vs_legacy_bwr': round(min(_contrast(c, rgb) for c in legacy), 3),
            'vs_paled_bwr': round(min(_contrast(c, rgb) for c in pale), 3),
            'vs_white': round(_contrast(rgb, (1, 1, 1)), 3)}
    return out


# ------------------------------------------------------------------- layout

class DualScale:
    """Freeze the layout once, then add the with_scale furniture on top.

    Usage::

        ds = DualScale(mode)
        ...  build every artist ...
        ds.strip(ax1, x=True, y=True)      # what the manuscript version shows
        ds.freeze(fig)                     # tight_layout, identical in both modes
        ds.colorbar(fig, img, label=...)   # no-op in no_scale
        ds.restore()                       # re-enables ticks in with_scale only
    """

    def __init__(self, mode: str):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.with_scale = (mode == 'with_scale')
        self._stripped = []

    def strip(self, ax, x=True, y=True, keep_xlabel=None, keep_ylabel=None,
              restore_xlabels=True, restore_ylabels=True):
        """Hide ticks for the layout pass; remember how to put them back.

        The locator and formatter objects already on the axis are SAVED and
        reinstalled by `restore`, rather than replaced with fresh defaults.
        That matters: `Data2D.plot(use_timestamp=True)` installs a
        `DateFormatter`, and handing the axis a `ScalarFormatter` instead would
        label a time axis with raw matplotlib date numbers.  Nothing touches
        which SIDE a tick sits on either, so a `twinx` axis stays on the right.
        """
        rec = {'ax': ax, 'x': x, 'y': y, 'restore_xlabels': restore_xlabels,
               'restore_ylabels': restore_ylabels,
               'xlabel': ax.get_xlabel() if keep_xlabel is None else keep_xlabel,
               'ylabel': ax.get_ylabel() if keep_ylabel is None else keep_ylabel,
               'xloc': ax.xaxis.get_major_locator(),
               'xfmt': ax.xaxis.get_major_formatter(),
               'yloc': ax.yaxis.get_major_locator(),
               'yfmt': ax.yaxis.get_major_formatter()}
        self._stripped.append(rec)
        if x:
            ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
            ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel("")
        if y:
            ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
            ax.set_ylabel("")
        return ax

    def freeze(self, fig, rect=LAYOUT_RECT):
        """The one layout call.  Runs with every tick hidden in BOTH modes, so
        the axes rectangles cannot depend on the mode."""
        fig.tight_layout(rect=rect)
        self._positions = [(ax, ax.get_position().frozen()) for ax in fig.axes]
        return fig

    def restore(self):
        """Put the ticks back - with_scale only.  Called AFTER freeze(), so it
        cannot move an axes."""
        if not self.with_scale:
            return
        null = matplotlib.ticker.NullLocator
        for r in self._stripped:
            ax = r['ax']
            if r['x']:
                # `Axes.sharex` makes both axes hold the SAME Ticker object, so
                # stripping one strips the other and the second axes records a
                # NullLocator.  Reinstalling that would undo the first axes'
                # restore, hence the guard; and an axes that only wants its
                # labels hidden uses tick_params, which is per-axes.
                if not isinstance(r['xloc'], null):
                    ax.xaxis.set_major_locator(r['xloc'])
                    ax.xaxis.set_major_formatter(r['xfmt'])
                if r['restore_xlabels']:
                    ax.tick_params(axis='x', labelsize=7)
                    ax.set_xlabel(r['xlabel'], fontsize=8)
                else:
                    ax.tick_params(axis='x', labelbottom=False, labeltop=False)
                    ax.set_xlabel("")
            if r['y']:
                if not isinstance(r['yloc'], null):
                    ax.yaxis.set_major_locator(r['yloc'])
                    ax.yaxis.set_major_formatter(r['yfmt'])
                if r['restore_ylabels']:
                    ax.tick_params(axis='y', labelsize=7)
                    ax.set_ylabel(r['ylabel'], fontsize=8)
                else:
                    # a `sharey` partner: its labels would be drawn into the
                    # NEIGHBOURING panel's data area, so only the owner shows them
                    ax.tick_params(axis='y', labelleft=False, labelright=False)
                    ax.set_ylabel("")
        # re-pin every axes to the frozen rectangle, belt and braces
        for ax, pos in self._positions:
            ax.set_position(pos)

    def colorbar(self, fig, mappable, rect=CBAR_RECT_UPPER, label=None):
        """Numeric colorbar in the reserved strip - with_scale only.

        The strip is outside every data axes, and `fig.add_axes` (not
        `fig.colorbar(ax=...)`) is used on purpose: the latter steals space from
        the parent axes and would move the data.
        """
        if not self.with_scale:
            return None
        cax = fig.add_axes(rect)
        cb = fig.colorbar(mappable, cax=cax)
        cb.ax.tick_params(labelsize=6)
        if label:
            cb.set_label(label, fontsize=7)
        return cb


# --------------------------------------------------------------- saving / QA

def save_outputs(fig, outdir, stem, mode, dpi=400):
    """PNG (quick view) + TIFF (submission), same pixels, >= 300 dpi."""
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f"{stem}_{mode}.png")
    tif = os.path.join(outdir, f"{stem}_{mode}.tiff")
    fig.savefig(png, dpi=dpi)
    fig.savefig(tif, dpi=dpi, pil_kwargs={'compression': 'tiff_lzw'})
    from PIL import Image
    with Image.open(tif) as im:
        w, h = im.size
    if min(w, h) < 600:
        raise RuntimeError(f"{tif}: short side {min(w, h)} px < 600 px floor")
    if dpi < 300:
        raise RuntimeError(f"{tif}: {dpi} dpi < 300 dpi floor")
    return {'png': png, 'tiff': tif, 'pixels': [w, h], 'dpi': dpi}


def assert_data_area_identical(outdir, stem, axes_boxes_px, tol_px=0):
    """The two modes must agree pixel for pixel inside every data axes.

    `axes_boxes_px` is a list of (y0, y1, x0, x1) in the saved image's pixel
    coordinates, produced by `axes_pixel_boxes`.
    """
    from PIL import Image
    a = np.asarray(Image.open(os.path.join(outdir, f"{stem}_no_scale.png")).convert('RGB'), int)
    b = np.asarray(Image.open(os.path.join(outdir, f"{stem}_with_scale.png")).convert('RGB'), int)
    if a.shape != b.shape:
        raise RuntimeError(f"{stem}: canvas differs {a.shape} vs {b.shape}")
    report = []
    for (y0, y1, x0, x1) in axes_boxes_px:
        d = np.abs(a[y0:y1, x0:x1] - b[y0:y1, x0:x1])
        n = int((d.sum(2) > tol_px).sum())
        report.append({'box': [y0, y1, x0, x1], 'differing_pixels': n,
                       'max_channel_diff': int(d.max())})
        if n:
            raise RuntimeError(
                f"{stem}: data area {y0}:{y1},{x0}:{x1} differs between modes "
                f"in {n} px (max channel diff {int(d.max())})")
    return report


def axes_pixel_boxes(fig, axes, dpi, inset=1):
    """Axes rectangles in saved-image pixel coordinates (top-down)."""
    h_px = fig.get_size_inches()[1] * dpi
    boxes = []
    for ax in axes:
        p = ax.get_position()
        x0 = int(np.ceil(p.x0 * fig.get_size_inches()[0] * dpi)) + inset
        x1 = int(np.floor(p.x1 * fig.get_size_inches()[0] * dpi)) - inset
        y0 = int(np.ceil(h_px - p.y1 * h_px)) + inset
        y1 = int(np.floor(h_px - p.y0 * h_px)) - inset
        boxes.append((y0, y1, x0, x1))
    return boxes


# ----------------------------------------------------------------- manifest

def sha256(path, _cache={}):
    key = (path, os.path.getmtime(path))
    if key not in _cache:
        h = hashlib.sha256()
        with open(path, 'rb') as fh:
            for blk in iter(lambda: fh.read(1 << 20), b''):
                h.update(blk)
        _cache[key] = h.hexdigest()
    return _cache[key]


def _fiberis_commit():
    try:
        return subprocess.run(['git', '-C', 'fibeRIS', 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip() or None
    except Exception:
        return None


def write_manifest(outdir, stem, *, figure, source_script, inputs, outputs,
                   parameters, changes, notes=None):
    """House rule 3: every run records what it read, with hashes, and why."""
    os.makedirs(outdir, exist_ok=True)
    doc = {
        'figure': figure,
        'manuscript': 'SJ-0626-0151 (SPE Journal), revision 1',
        'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'generating_script': {'path': source_script, 'sha256': sha256(source_script)},
        'shared_module': {'path': __file__.replace(os.getcwd() + '/', ''),
                          'sha256': sha256(__file__)},
        'inputs': [{'path': p, 'sha256': sha256(p), 'bytes': os.path.getsize(p)}
                   for p in sorted(set(inputs))],
        'outputs': outputs,
        'parameters': parameters,
        'changes_from_submitted_version': changes,
        'appearance_settings': {
            'white_blend': WHITE_BLEND,
            'gauge_color': GAUGE_COLOR, 'gauge_linewidth': GAUGE_LW,
            'frachit_color': FRACHIT_COLOR, 'frachit_marker': FRACHIT_MARKER,
            'frachit_size': FRACHIT_SIZE, 'frachit_edge': FRACHIT_EDGE,
        },
        'contrast': contrast_report(),
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'numpy': np.__version__,
            'matplotlib': matplotlib.__version__,
            'fiberis_commit': _fiberis_commit(),
            'cwd': os.getcwd(),
        },
    }
    if notes:
        doc['notes'] = notes
    path = os.path.join(outdir, f"{stem}_manifest.json")
    with open(path, 'w') as fh:
        json.dump(doc, fh, indent=2, default=str)
    return path


# ------------------------------------------------------- fibeRIS API compat

def install_legacy_fiberis_plotting():
    """Make a 2025-04-era script render the pixels it rendered then.

    fibeRIS commit a78bf16 (2025-06-03) changed four things that alter loading
    or pixels.  Reverting them is what makes the submitted Fig. 6 reproducible,
    and is required before any judgement about what changed in the revision:

      1. `load_npz` gained shape validation, which rejects the pre-`aabffe2`
         TIME-major `phase1.npz` / `phase2.npz`.
      2. `Data2D.plot`: kwarg `useTimeStamp` -> `use_timestamp`.
      3. `Data2D.plot`: default method `imshow` -> `pcolormesh`.  imshow
         resamples onto the pixel grid; pcolormesh draws every cell.
      4. `Data2D.plot`/`Data1D.plot` now call `fig.autofmt_xdate()`, which sets
         subplot bottom=0.2 and so moves every axes in the figure;
         `Data1D.plot`'s `useLegend=False` became `use_legend=True`.
    """
    import matplotlib.figure as mfig
    from fiberis.analyzer.Data2D import core2D as c2
    from fiberis.analyzer.Data1D import core1D as c1

    def legacy_load_npz(self, filename):
        fn = filename if filename.endswith('.npz') else filename + '.npz'
        ds = np.load(fn, allow_pickle=True)
        self.data = ds['data'].astype(float)
        self.taxis = ds['taxis'].astype(float)
        self.daxis = ds['daxis'].astype(float)
        st = ds['start_time']
        if isinstance(st, np.ndarray) and st.size == 1:
            st = st.item()
        if isinstance(st, np.datetime64):
            st = st.astype('datetime64[ms]').astype(datetime.datetime)
        elif isinstance(st, str):
            st = datetime.datetime.fromisoformat(st)
        self.start_time = st
        return self

    c2.Data2D.load_npz = legacy_load_npz
    mfig.Figure.autofmt_xdate = lambda self, *a, **k: None

    orig2 = c2.Data2D.plot

    def plot2(self, *a, **kw):
        if 'useTimeStamp' in kw:
            kw['use_timestamp'] = kw.pop('useTimeStamp')
        kw.setdefault('method', 'imshow')
        return orig2(self, *a, **kw)
    c2.Data2D.plot = plot2

    orig1 = c1.Data1D.plot

    def plot1(self, *a, **kw):
        if 'useTimeStamp' in kw:
            kw['use_timestamp'] = kw.pop('useTimeStamp')
        if 'useLegend' in kw:
            kw['use_legend'] = kw.pop('useLegend')
        kw.setdefault('use_legend', False)
        return orig1(self, *a, **kw)
    c1.Data1D.plot = plot1
