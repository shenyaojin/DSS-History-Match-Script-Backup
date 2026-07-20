# scripts/tensile_fault/data_transfer/107_lfdas_reproduce_strain_rate_waterfall.py
"""
Reproduce the TOP panel of ``data_fervo/legacy/image.png``:

    "DASCore Strain-rate Waterfall with Overlaid 4-hour Mean Profiles"
    (Gold 4-PB, depth 10200-10500 ft, time T1 -> T3, UTC-7)

using the converted fibeRIS Data2D .npz files instead of the raw .npy. The
recipe follows Cell 5 + Cell 17 of the reference notebook
``from_pc/LFDAS_NPY_Vis_07-13-2026-1.ipynb``:

  * data     : LFDAS_G4-PB_202502_late.npz (+) LFDAS_G4-PB_202503.npz, right-merged
               into one continuous Feb 24 -> Mar 7 record (strain rate, nanostrain/s).
  * process  : median filter (3 x 7) -> de-mean by subtracting, per time column,
               the median over the quiet MD 1800-5000 ft band -> interpolate across
               the interrogator-vibration artifact window -> crop to 10200-10500 ft
               and T1-T3.
  * overlay  : one 4-hour mean strain-rate depth profile per window, drawn as a black
               curve anchored at the window start, amplitude mapped to a horizontal
               time offset (half-width 28 h; scale reference = P95(|profiles|) x 10).
  * markers  : T1/T2/T3 green dashed vertical lines; yellow-star peak annotation of
               the 4-hour profile nearest 2025-03-03 00:00.

Output: figs/tensile_fault_qc/lfdas_waterfall/strain_rate_waterfall_T1-T3_10200-10500ft.png

Author: Shenyao Jin, shenyaojin@mines.edu
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage import median_filter

from fiberis.analyzer.Data2D.core2D import Data2D

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data_fervo" / "fiberis_format" / "LFDAS"
FIG_DIR = REPO_ROOT / "figs" / "tensile_fault_qc" / "lfdas_waterfall"
OUT_PNG = FIG_DIR / "strain_rate_waterfall_T1-T3_10200-10500ft.png"

# Two batches that together span the T1->T3 window (chronological order).
SEGMENTS = ["LFDAS_G4-PB_202502_late.npz", "LFDAS_G4-PB_202503.npz"]

# --------------------------------------------------------------------------- #
# Parameters (Cell 17 of the reference notebook, all times UTC-7 / local)
# --------------------------------------------------------------------------- #
T1 = pd.Timestamp("2025-02-24 11:00")
T2 = pd.Timestamp("2025-02-28 00:00")
T3 = pd.Timestamp("2025-03-03 22:00")
TIME_MARKS = [(T1, "T1"), (T2, "T2"), (T3, "T3")]

PLOT_MD_TOP, PLOT_MD_BOTTOM = 10200.0, 10500.0     # display depth window [ft]
DEMEAN_MD_LOW, DEMEAN_MD_HIGH = 1800.0, 5000.0     # quiet reference band [ft]
MEDFILT_KERNEL = (3, 7)                             # (depth, time)

PROFILE_INTERVAL_H = 4                              # 4-hour mean profiles
OVERLAY_HALF_WIDTH_H = 28.0                         # amplitude -> +/- this many hours
PROFILE_SCALE_MULTIPLIER = 10.0                     # scale ref = P95(|profiles|) * this
REFERENCE_PROFILE_TIME = pd.Timestamp("2025-03-03 00:00")  # star annotation target

# Interrogator-vibration artifact interpolated out (Cell 5).
ARTIFACT_WINDOWS = [(pd.Timestamp("2025-02-25 19:53:50"),
                     pd.Timestamp("2025-02-25 19:54:30"))]

RATE_CLIM = (-0.3, 0.3)                             # nanostrain/s
CMAP = "bwr"
CLABEL = "Strain rate (nanostrain/s)"
TITLE = "DASCore Strain-rate Waterfall with Overlaid 4-hour Mean Profiles"
DPI = 150

SECONDS_PER_DAY = 86400.0


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
def load_merged():
    """Right-merge the segment .npz into one continuous Data2D (full depth)."""
    merged = Data2D()
    merged.load_npz(str(DATA_DIR / SEGMENTS[0]))
    for seg in SEGMENTS[1:]:
        nxt = Data2D()
        nxt.load_npz(str(DATA_DIR / seg))
        merged.right_merge(nxt)
    return merged


def abs_times(das: Data2D) -> pd.DatetimeIndex:
    """Absolute sample times as a pandas DatetimeIndex (start_time + taxis)."""
    return pd.to_datetime(das.start_time) + pd.to_timedelta(das.taxis, unit="s")


def interp_rows_over_mask(data: np.ndarray, col_mask: np.ndarray) -> None:
    """In place: per depth row, linearly interpolate the masked time columns from
    the surrounding valid samples (used for NaN fill and artifact removal)."""
    if not col_mask.any():
        return
    x = np.arange(data.shape[1])
    for r in range(data.shape[0]):
        y = data[r]
        valid = np.isfinite(y) & ~col_mask
        if valid.sum() >= 2:
            data[r, col_mask] = np.interp(x[col_mask], x[valid], y[valid])


def build_rate_field(das: Data2D):
    """Median-filter + de-mean + artifact-clean, then crop to the display window.
    Returns (LP_rate_plot, Depth_plot[ft], Time_plot[DatetimeIndex])."""
    times = abs_times(das)

    # Crop time to T1..T3 with a small buffer so the median filter is exact inside.
    buf = pd.Timedelta(minutes=10)
    tbuf = (times >= T1 - buf) & (times <= T3 + buf)
    LP = das.data[:, tbuf].astype(float)
    times = times[tbuf]
    daxis = das.daxis

    # Preserve full-column acquisition gaps as NaN (rendered gray); interpolate the
    # rest away so the median filter is not NaN-contaminated.
    gap_col = np.isnan(LP).all(axis=0)
    isolated = np.isnan(LP) & ~gap_col[None, :]
    if isolated.any():
        x = np.arange(LP.shape[1])
        for r in range(LP.shape[0]):
            y = LP[r]
            bad = np.isnan(y) & ~gap_col
            good = np.isfinite(y) & ~gap_col
            if bad.any() and good.sum() >= 2:
                y[bad] = np.interp(x[bad], x[good], y[good])
    LP[:, gap_col] = 0.0  # placeholder for the filter; re-masked to NaN below

    # Median filter (3 depth x 7 time) then de-mean by the quiet-band column median.
    med = median_filter(LP, size=MEDFILT_KERNEL, mode="nearest")
    band = (daxis >= DEMEAN_MD_LOW) & (daxis <= DEMEAN_MD_HIGH)
    demeaned = med - np.nanmedian(med[band, :], axis=0)[None, :]
    demeaned[:, gap_col] = np.nan

    # Remove the interrogator-vibration artifact (interpolate across those columns).
    art = np.zeros(len(times), dtype=bool)
    for a, b in ARTIFACT_WINDOWS:
        art |= (times >= a) & (times <= b)
    interp_rows_over_mask(demeaned, art & ~gap_col)

    # Crop to the display depth window and exact T1..T3.
    dmask = (daxis >= PLOT_MD_TOP) & (daxis <= PLOT_MD_BOTTOM)
    tmask = (times >= T1) & (times <= T3)
    return demeaned[dmask][:, tmask], daxis[dmask], times[tmask]


# --------------------------------------------------------------------------- #
# 4-hour mean profiles + overlay
# --------------------------------------------------------------------------- #
def compute_mean_profiles(data, times):
    """One depth profile (nanmean over time) per 4-hour window, anchored at the
    window start. Returns (profile_matrix[n_depth, n_win], centers[DatetimeIndex])."""
    window_starts = pd.date_range(T1, T3, freq=f"{PROFILE_INTERVAL_H}h")
    profiles, centers = [], []
    for ws in window_starts:
        we = min(ws + pd.Timedelta(hours=PROFILE_INTERVAL_H), T3)
        if ws >= T3:
            continue
        m = (times >= ws) & (times < we)
        if not m.any():
            continue
        profiles.append(np.nanmean(data[:, m], axis=1))
        centers.append(ws)
    return np.column_stack(profiles), pd.DatetimeIndex(centers)


def scale_reference(profiles):
    """P95(|profiles|) * multiplier -> the amplitude that maps to the half-width."""
    p95 = np.nanpercentile(np.abs(profiles), 95)
    if not np.isfinite(p95) or p95 == 0:
        p95 = np.nanmax(np.abs(profiles))
    return p95 * PROFILE_SCALE_MULTIPLIER


def overlay_profiles(ax, profiles, centers, depth, scale_ref):
    """Draw each profile as a black curve, amplitude -> horizontal time offset."""
    sec_per_unit = OVERLAY_HALF_WIDTH_H * 3600.0 / scale_ref
    for j, center in enumerate(centers):
        prof = profiles[:, j]
        fin = np.isfinite(prof)
        if not fin.any():
            continue
        xnum = mdates.date2num(center) + prof[fin] * sec_per_unit / SECONDS_PER_DAY
        ax.plot(xnum, depth[fin], color="black", linewidth=0.8, alpha=0.9, zorder=5)


def annotate_peak(ax, profiles, centers, depth, scale_ref):
    """Yellow star at the peak |amplitude| of the profile nearest the target time."""
    sec_per_unit = OVERLAY_HALF_WIDTH_H * 3600.0 / scale_ref
    j = int(np.argmin(np.abs(centers - REFERENCE_PROFILE_TIME)))
    prof = profiles[:, j]
    fin = np.flatnonzero(np.isfinite(prof))
    pidx = fin[int(np.nanargmax(np.abs(prof[fin])))]
    pval, pdep = float(prof[pidx]), float(depth[pidx])
    xnum = mdates.date2num(centers[j]) + pval * sec_per_unit / SECONDS_PER_DAY
    ax.scatter([xnum], [pdep], marker="*", color="yellow", edgecolor="black",
               linewidth=0.8, s=150, zorder=10)
    ax.text(xnum, pdep + 8, f"{pval:+.3g} nanostrain/s", color="black", fontsize=9,
            ha="center", va="top", zorder=11,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.88, pad=2))
    return pval, pdep, centers[j]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    das = load_merged()
    print(f"Merged: {das.data.shape[0]} ch x {das.data.shape[1]} samples, "
          f"{das.start_time} .. {das.get_end_time()}")

    LP_rate, depth, times = build_rate_field(das)
    print(f"Display field: {LP_rate.shape[0]} ch ({depth.min():.0f}-{depth.max():.0f} ft) "
          f"x {LP_rate.shape[1]} samples ({times[0]} .. {times[-1]})")

    profiles, centers = compute_mean_profiles(LP_rate, times)
    scale_ref = scale_reference(profiles)
    print(f"{profiles.shape[1]} 4-hour profiles; scale_ref = {scale_ref:.4g} nanostrain/s")

    # --- Plot the strain-rate waterfall (imshow, depth increases downward) --- #
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad("0.82")

    fig, ax = plt.subplots(figsize=(16, 5.2), constrained_layout=True)
    extent = [mdates.date2num(times[0]), mdates.date2num(times[-1]),
              depth[-1], depth[0]]                        # bottom=deepest, top=shallowest
    im = ax.imshow(LP_rate, aspect="auto", cmap=cmap, origin="upper",
                   vmin=RATE_CLIM[0], vmax=RATE_CLIM[1], extent=extent,
                   interpolation="nearest")

    overlay_profiles(ax, profiles, centers, depth, scale_ref)
    pval, pdep, pcenter = annotate_peak(ax, profiles, centers, depth, scale_ref)
    print(f"Peak (profile nearest {REFERENCE_PROFILE_TIME}): "
          f"{pval:+.4g} nanostrain/s at {pdep:.1f} ft (center {pcenter}).")

    # T1/T2/T3 green dashed reference lines + labels.
    for mt, label in TIME_MARKS:
        xn = mdates.date2num(mt)
        ax.axvline(xn, color="green", linestyle="--", linewidth=1.8, alpha=0.95, zorder=4)
        ax.text(xn, 1.01, label, color="green", fontsize=12, fontweight="bold",
                ha="center", va="bottom", transform=ax.get_xaxis_transform(), clip_on=False)

    ax.set_ylim(PLOT_MD_BOTTOM, PLOT_MD_TOP)              # depth increases downward
    ax.set_xlim(mdates.date2num(T1), mdates.date2num(T3))
    ax.set_ylabel("Gold 4-PB Measured Depth [ft]")
    ax.set_xlabel("Time [UTC-7]")
    ax.set_title(TITLE)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))

    cb = fig.colorbar(im, ax=ax, pad=0.01, extend="both")
    cb.set_label(CLABEL)

    fig.savefig(OUT_PNG, dpi=DPI)
    plt.close(fig)
    print(f"Saved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
