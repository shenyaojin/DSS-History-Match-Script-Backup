# scripts/tensile_fault/data_transfer/106_lfdas_viz.py
"""
Visualize the converted LF-DAS Data2D .npz files (Gold 4-PB) as waterfall plots.

Input  : data_fervo/fiberis_format/LFDAS/LFDAS_G4-PB_YYYYMM.npz  (from 105_lfdas_npy_extractor.py)
Output : figs/tensile_fault_qc/lfdas_waterfall/*.png   (folder created if absent)

For each .npz two single-panel depth-vs-time waterfalls are produced:

  1. *_overview.png            -- full-fiber raw strain rate, robust symmetric
                                  color scale (|clim| = P98 of |amplitude|). QC view.
  2. *_stimulated_interval.png -- notebook-style processed view: median filter
                                  (kernel 3x7) + de-mean by subtracting, per time
                                  column, the median of the quiet MD 1800-5000 ft
                                  band; cropped to the stimulated interval
                                  9500-12250 ft; bwr color scale +/- 2 nanostrain/s.

Rendering is done through ``Data2D.plot(method='imshow', ...)``. imshow (not the
class default pcolormesh) is required: a 2304 x ~34000 array would otherwise build
~79M QuadMesh cells. The time axis is decimated with a NaN-safe max-abs block
reduction before plotting so short-lived strain-rate transients are preserved
rather than aliased away.

Processing constants follow the reference notebook
``from_pc/LFDAS_NPY_Vis_07-13-2026-1.ipynb`` (the corrected reference).

Author: Shenyao Jin, shenyaojin@mines.edu
"""

import os

# Headless rendering (repo convention for the tensile_fault plotting scripts).
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage import median_filter

# Data2D.plot calls fig.autofmt_xdate() for timestamped x-axes, which is benign
# but warns under constrained_layout. The layout still renders correctly.
warnings.filterwarnings("ignore", message=".*incompatible with subplots_adjust.*")

from fiberis.analyzer.Data2D.core2D import Data2D

# --------------------------------------------------------------------------- #
# Paths  (this script lives in scripts/tensile_fault/data_transfer/ -> parents[3] is repo root)
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data_fervo" / "fiberis_format" / "LFDAS"
FIG_DIR = REPO_ROOT / "figs" / "tensile_fault_qc" / "lfdas_waterfall"

NPZ_FILES = [
    "LFDAS_G4-PB_202501.npz",
    "LFDAS_G4-PB_202502.npz",
]

# --------------------------------------------------------------------------- #
# Processing / style constants (from the 07-13-2026 reference notebook)
# --------------------------------------------------------------------------- #
MEDFILT_KERNEL = (3, 7)                    # (depth, time) median-filter kernel
DEMEAN_MD_LOW, DEMEAN_MD_HIGH = 1800.0, 5000.0   # quiet reference band [ft]
STIM_MD_TOP, STIM_MD_BOTTOM = 9500.0, 12250.0    # stimulated display window [ft]
STIM_CLIM = 2.0                            # +/- color limit for processed view [nanostrain/s]

TARGET_COLS = 2600                         # time-decimation target (~fig width in px)
ROBUST_PCT = 98.0                          # overview color-limit percentile

CMAP = "bwr"
DPI = 150
FIGSIZE = (16, 6.5)
TIME_FMT = "%m/%d\n%H:%M"
CLABEL = "Strain rate (nanostrain/s)"
YLABEL = "Gold 4-PB Measured Depth [ft]"
XLABEL = "Time [UTC-7]"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def maxabs_decimate(data: np.ndarray, taxis: np.ndarray, target_cols: int):
    """Decimate along time by keeping, in each contiguous block of columns, the
    sample with the largest absolute value (NaN-safe). Preserves transients that
    naive striding would drop. Returns (decimated_data, decimated_taxis)."""
    n_time = data.shape[1]
    factor = max(1, n_time // target_cols)
    if factor == 1:
        return data, taxis

    n_keep = (n_time // factor) * factor
    blk = data[:, :n_keep].reshape(data.shape[0], -1, factor)
    abs_blk = np.where(np.isnan(blk), -np.inf, np.abs(blk))
    idx = abs_blk.argmax(axis=2)                       # (n_depth, n_blocks)
    dec = np.take_along_axis(blk, idx[:, :, None], axis=2)[:, :, 0]
    tdec = taxis[:n_keep].reshape(-1, factor)[:, 0]
    return dec, tdec


def _waterfall(ax, das: Data2D, clim, title):
    """Render one Data2D as an imshow waterfall on ``ax`` with shared styling."""
    das.plot(
        ax=ax,
        method="imshow",              # MUST be explicit: class default is pcolormesh
        use_timestamp=True,
        cmap=CMAP,
        clim=clim,
        colorbar=True,
        clabel=CLABEL,
        ylabel=YLABEL,
        interpolation="nearest",
    )
    ax.set_xlabel(XLABEL)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(TIME_FMT))


def plot_overview(das: Data2D, label: str, out_path: Path):
    """Full-fiber raw strain-rate waterfall with a robust symmetric color scale."""
    dec, tdec = maxabs_decimate(das.data, das.taxis, TARGET_COLS)
    view = Data2D(data=dec, taxis=tdec.astype(float), daxis=das.daxis,
                  start_time=das.start_time, name=label)

    clim_val = float(np.nanpercentile(np.abs(dec), ROBUST_PCT))
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    _waterfall(
        ax, view, (-clim_val, clim_val),
        f"LF-DAS strain-rate overview (full fiber) — {label}\n"
        f"|clim| = P{ROBUST_PCT:.0f} = {clim_val:.0f} nanostrain/s",
    )
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return clim_val


def plot_stimulated_interval(das: Data2D, label: str, out_path: Path):
    """Notebook-style processed waterfall over the stimulated interval."""
    # Median filter over the full section (needed for both the reference band
    # and the display window), then de-mean by the quiet-band median per column.
    med = median_filter(das.data, size=MEDFILT_KERNEL, mode="nearest")
    band = (das.daxis >= DEMEAN_MD_LOW) & (das.daxis <= DEMEAN_MD_HIGH)
    ref = np.nanmedian(med[band, :], axis=0)
    demeaned = med - ref[None, :]

    # Crop to the stimulated display window, then decimate time.
    dmask = (das.daxis >= STIM_MD_TOP) & (das.daxis <= STIM_MD_BOTTOM)
    dec, tdec = maxabs_decimate(demeaned[dmask, :], das.taxis, TARGET_COLS)
    view = Data2D(data=dec, taxis=tdec.astype(float), daxis=das.daxis[dmask],
                  start_time=das.start_time, name=label)

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    _waterfall(
        ax, view, (-STIM_CLIM, STIM_CLIM),
        f"LF-DAS strain rate — {label}\n"
        f"medfilt{list(MEDFILT_KERNEL)} + de-mean over {int(DEMEAN_MD_LOW)}-{int(DEMEAN_MD_HIGH)} ft, "
        f"stimulated interval {int(STIM_MD_TOP)}-{int(STIM_MD_BOTTOM)} ft",
    )
    ax.set_ylim(STIM_MD_BOTTOM, STIM_MD_TOP)   # depth increases downward
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Figure directory: {FIG_DIR}")

    for fname in NPZ_FILES:
        npz_path = DATA_DIR / fname
        if not npz_path.exists():
            print(f"Warning: not found, skipping: {npz_path}")
            continue

        label = npz_path.stem
        print(f"\n=== {label} ===")

        das = Data2D()
        das.load_npz(str(npz_path))
        print(f"  Loaded {das.data.shape[0]} channels x {das.data.shape[1]} samples; "
              f"start {das.start_time}, depth {das.daxis.min():.0f}..{das.daxis.max():.0f} ft")

        overview_path = FIG_DIR / f"{label}_overview.png"
        clim_val = plot_overview(das, label, overview_path)
        print(f"  Saved -> {overview_path.name}  (|clim|={clim_val:.0f} nanostrain/s)")

        stim_path = FIG_DIR / f"{label}_stimulated_interval.png"
        plot_stimulated_interval(das, label, stim_path)
        print(f"  Saved -> {stim_path.name}  (clim +/-{STIM_CLIM:.0f} nanostrain/s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
