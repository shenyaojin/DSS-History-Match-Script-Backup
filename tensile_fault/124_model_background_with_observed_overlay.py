"""
Strain waterfall with OUR simulation as the background and the observed DAS
4-hour-mean profiles overlaid as black wiggles on top.

Background (color) = simulated monitor strain from the new-geometry two-SRV run
                     (0711_new_geometry_two_srv), mapped from monitor position (m)
                     to measured depth (ft) with the fracture crossing at the
                     yellow-star depth.
Black wiggles       = observed DAS 4-hour-mean strain profiles (T1 referenced).

Run from the repository root:
    python scripts/tensile_fault/124_model_background_with_observed_overlay.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DIR = REPO_ROOT / "data_fervo" / "legacy"
STRAIN_CSV = LEGACY_DIR / (
    "strain_4h_mean_profiles_20250224_1500_to_20250303_2200_"
    "10200_10500ft_4h_mean_T1_ref.csv"
)
MODEL_STRAIN_NPZ = (
    REPO_ROOT / "output" / "0711_new_geometry_two_srv" / "postprocessor_npz"
    / "monitor_normal_strain_no_rotation.npz"
)
FIG_DIR = REPO_ROOT / "figs" / "tensile_fault_qc" / "das_waterfall_recreation"

T2 = dt.datetime(2025, 2, 28, 0, 0, 0)
STAR_DEPTH_FT = 10373.4
FT = 0.3048


def load_observed(csv_path):
    df = pd.read_csv(csv_path)
    depths = df["measured_depth_ft"].to_numpy(float)
    tcols = df.columns[1:]
    times = np.array([dt.datetime.fromisoformat(c) for c in tcols])
    return depths, times, df[tcols].to_numpy(float)  # millistrain


def load_model():
    d = np.load(MODEL_STRAIN_NPZ, allow_pickle=True)
    data = d["data"].astype(float)
    dax = d["daxis"].astype(float)
    tax = d["taxis"].astype(float)
    start = d["start_time"].item()
    if isinstance(start, str):
        start = dt.datetime.fromisoformat(start)
    times = np.array([start + dt.timedelta(seconds=float(s)) for s in tax])
    # Anchor to the FRACTURE crossing = monitor-line midpoint (the perpendicular
    # monitor is symmetric about the fracture, so the strain band is centered
    # there). Using the off-center argmax would push the whole band too deep.
    frac_pos = 0.5 * (dax.min() + dax.max())
    depth_ft = STAR_DEPTH_FT + (dax - frac_pos) / FT
    return depth_ft, times, data * 1e3  # (pos, time) millistrain


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    o_depths, o_times, o_mat = load_observed(STRAIN_CSV)
    m_depths, m_times, m_mat = load_model()

    vlim = 0.1
    fig, ax = plt.subplots(figsize=(16, 6.5), constrained_layout=True)

    # background = simulated strain waterfall
    mesh = ax.pcolormesh(mdates.date2num(m_times), m_depths, m_mat,
                         cmap="bwr", vmin=-vlim, vmax=vlim, shading="gouraud")

    # overlay = observed 4h-mean profiles (black wiggles)
    tnum = mdates.date2num(o_times)
    spacing = np.median(np.diff(tnum))
    gain = 1.6 * spacing / np.nanmax(np.abs(o_mat))
    for j, t in enumerate(tnum):
        ax.plot(t + gain * o_mat[:, j], o_depths, color="black", lw=0.8, alpha=0.9)

    ax.axvline(mdates.date2num(T2), color="green", ls="--", lw=1.6)
    tr = ax.get_xaxis_transform()
    for x, lab, ha in [(tnum[0], "T1", "left"), (mdates.date2num(T2), "T2", "center"),
                       (tnum[-1], "T3", "right")]:
        ax.text(x, 0.98, lab, color="green", transform=tr, fontweight="bold",
                va="top", ha=ha, fontsize=12)

    ip, jt = np.unravel_index(int(np.nanargmax(np.abs(o_mat))), o_mat.shape)
    ax.plot(tnum[jt], o_depths[ip], marker="*", color="yellow", markersize=18,
            markeredgecolor="black", markeredgewidth=0.7, zorder=6)

    ax.set_ylim(o_depths[-1], o_depths[0])   # crop view to observed depth window
    ax.set_xlim(tnum[0], tnum[-1])
    ax.set_ylabel("Gold 4-PB Measured Depth [ft]")
    ax.set_xlabel("Time [UTC-7]")
    ax.set_title("Simulated Strain Waterfall (background) with Observed 4-hour Mean "
                 "Profiles (black) — new-geometry two-SRV, C*=1.63e7")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    cb = fig.colorbar(mesh, ax=ax, pad=0.01)
    cb.set_label("Strain (millistrain)  —  background: simulated")

    ax.legend(handles=[mlines.Line2D([], [], color="black", lw=1.2,
                                     label="Observed 4h-mean profile")],
              loc="lower left", framealpha=0.9)

    out = FIG_DIR / "das_strain_model_background_observed_overlay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
