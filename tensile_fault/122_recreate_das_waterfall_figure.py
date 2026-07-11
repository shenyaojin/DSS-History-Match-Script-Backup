"""
Recreate an image.png-style DASCore waterfall figure from the 4-hour-mean profile
CSVs (the only Gold 4-PB DAS data in the repo; the original full-resolution
waterfall lives in an external DASCore pipeline and is not stored here).

The overlaid black "4-hour mean profiles" ARE these CSVs, so they reproduce
exactly; the background variable-density map is the mean-profile field
(gouraud-interpolated), i.e. a lower-resolution stand-in for the true waterfall.

Layout mirrors data_fervo/legacy/image.png:
  - top    : strain-rate waterfall + overlaid mean-profile wiggles, +/-0.3 nanostrain/s
  - bottom : strain waterfall (T1 referenced) + overlaid wiggles, +/-0.1 millistrain
  - T1/T2/T3 markers, T2 (2/28) green dashed line, yellow-star peak annotations.

Run from the repository root:
    python scripts/tensile_fault/122_recreate_das_waterfall_figure.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DIR = REPO_ROOT / "data_fervo" / "legacy"
STRAIN_CSV = LEGACY_DIR / (
    "strain_4h_mean_profiles_20250224_1500_to_20250303_2200_"
    "10200_10500ft_4h_mean_T1_ref.csv"
)
STRAIN_RATE_CSV = LEGACY_DIR / (
    "strain_rate_4h_mean_profiles_20250224_1500_to_20250303_2200_"
    "10200_10500ft_4h_mean_T1_ref.csv"
)
FIG_DIR = REPO_ROOT / "figs" / "tensile_fault_qc" / "das_waterfall_recreation"

T2 = dt.datetime(2025, 2, 28, 0, 0, 0)   # green dashed boundary


def load(csv_path):
    df = pd.read_csv(csv_path)
    depths = df["measured_depth_ft"].to_numpy(float)
    tcols = df.columns[1:]
    times = np.array([dt.datetime.fromisoformat(c) for c in tcols])
    matrix = df[tcols].to_numpy(float)  # (depth, time)
    return depths, times, matrix


def draw_panel(ax, depths, times, matrix, cmap, vlim, cbar_label, title,
               unit, wiggle_gain=1.6):
    tnum = mdates.date2num(times)
    mesh = ax.pcolormesh(tnum, depths, matrix, cmap=cmap, vmin=-vlim, vmax=vlim,
                         shading="gouraud")

    # overlaid 4-hour mean profiles as wiggles (deflection in x by value)
    spacing = np.median(np.diff(tnum))
    peak = np.nanmax(np.abs(matrix))
    gain = wiggle_gain * spacing / peak if peak > 0 else 0.0
    for j, t in enumerate(tnum):
        ax.plot(t + gain * matrix[:, j], depths, color="black", lw=0.7, alpha=0.9)

    # T1/T2/T3 markers, just inside the top edge to avoid colliding with the title
    ax.axvline(mdates.date2num(T2), color="green", ls="--", lw=1.6)
    tr = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    ax.text(mdates.date2num(times[0]), 0.98, "T1", color="green", transform=tr,
            fontweight="bold", va="top", ha="left", fontsize=12)
    ax.text(mdates.date2num(T2), 0.98, "T2", color="green", transform=tr,
            fontweight="bold", va="top", ha="center", fontsize=12)
    ax.text(mdates.date2num(times[-1]), 0.98, "T3", color="green", transform=tr,
            fontweight="bold", va="top", ha="right", fontsize=12)

    # yellow star at peak response
    ip, jt = np.unravel_index(int(np.nanargmax(np.abs(matrix))), matrix.shape)
    ax.plot(tnum[jt], depths[ip], marker="*", color="yellow", markersize=18,
            markeredgecolor="black", markeredgewidth=0.7, zorder=5)
    ax.annotate(f"{matrix[ip, jt]:+.3g} {unit}", (tnum[jt], depths[ip]),
                xytext=(-6, 14), textcoords="offset points", fontsize=9,
                ha="right", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.4"))

    ax.set_ylim(depths[-1], depths[0])  # depth increases downward
    ax.set_ylabel("Gold 4-PB Measured Depth [ft]")
    ax.set_title(title)
    cb = ax.figure.colorbar(mesh, ax=ax, pad=0.01)
    cb.set_label(cbar_label)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    s_depths, s_times, s_mat = load(STRAIN_CSV)
    r_depths, r_times, r_mat = load(STRAIN_RATE_CSV)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                         constrained_layout=True)

    draw_panel(ax_top, r_depths, r_times, r_mat, cmap="bwr", vlim=0.3,
               cbar_label="Strain rate (nanostrain/s)",
               title="DASCore Strain-rate Waterfall with Overlaid 4-hour Mean Profiles",
               unit="nanostrain/s")
    draw_panel(ax_bot, s_depths, s_times, s_mat, cmap="RdBu_r", vlim=0.1,
               cbar_label="Strain (millistrain)",
               title="DASCore Strain Waterfall with Overlaid 4-hour Mean Profiles (T1 Referenced)",
               unit="millistrain")

    ax_bot.set_xlabel("Time [UTC-7]")
    ax_bot.xaxis.set_major_locator(mdates.DayLocator())
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))

    out = FIG_DIR / "das_strain_waterfall_recreation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")
    print(f"strain-rate peak: {np.nanmax(np.abs(r_mat)):.3f} nanostrain/s")
    print(f"strain peak     : {np.nanmax(np.abs(s_mat)):.4f} millistrain")


if __name__ == "__main__":
    main()
