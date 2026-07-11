"""
Recreate the DASCore strain waterfall (bottom panel of image.png) and overlay
OUR simulated strain result on it, so observed vs simulated can be read together.

Background color + black wiggles = observed DAS 4-hour-mean strain (T1 referenced).
Red wiggles                     = simulated monitor strain from the new-geometry
                                  two-SRV run (0711_new_geometry_two_srv), sampled
                                  along the perpendicular monitor, mapped from
                                  monitor position (m) to measured depth (ft) so the
                                  fracture crossing sits at the yellow-star depth.

Run from the repository root:
    python scripts/tensile_fault/123_das_waterfall_with_model_overlay.py
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
STAR_DEPTH_FT = 10373.4   # yellow-star channel (fracture crossing)
FT = 0.3048


def load_observed(csv_path):
    df = pd.read_csv(csv_path)
    depths = df["measured_depth_ft"].to_numpy(float)
    tcols = df.columns[1:]
    times = np.array([dt.datetime.fromisoformat(c) for c in tcols])
    matrix = df[tcols].to_numpy(float)  # (depth, time), millistrain
    return depths, times, matrix


def load_model():
    d = np.load(MODEL_STRAIN_NPZ, allow_pickle=True)
    data = d["data"].astype(float)          # (position, time), strain
    dax = d["daxis"].astype(float)          # position along monitor (m)
    tax = d["taxis"].astype(float)          # seconds
    start = d["start_time"].item()
    if isinstance(start, str):
        start = dt.datetime.fromisoformat(start)
    times = np.array([start + dt.timedelta(seconds=float(s)) for s in tax])

    # map monitor position -> measured depth: fracture crossing = monitor midpoint
    # (symmetric perpendicular monitor) -> star depth
    frac_pos = 0.5 * (dax.min() + dax.max())
    depth_ft = STAR_DEPTH_FT + (dax - frac_pos) / FT
    return depth_ft, times, data * 1e3     # -> millistrain


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    o_depths, o_times, o_mat = load_observed(STRAIN_CSV)
    m_depths, m_times, m_mat = load_model()

    vlim = 0.1
    fig, ax = plt.subplots(figsize=(16, 6.5), constrained_layout=True)
    tnum = mdates.date2num(o_times)
    mesh = ax.pcolormesh(tnum, o_depths, o_mat, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                         shading="gouraud")

    # shared wiggle gain from the observed field so amplitudes are comparable
    spacing = np.median(np.diff(tnum))
    peak = np.nanmax(np.abs(o_mat))
    gain = 1.6 * spacing / peak

    # observed 4h-mean profiles (black)
    for j, t in enumerate(tnum):
        ax.plot(t + gain * o_mat[:, j], o_depths, color="black", lw=0.7, alpha=0.9)

    # simulated profiles (red), cropped to the observed depth window
    win = (m_depths >= o_depths[0]) & (m_depths <= o_depths[-1])
    md, mm = m_depths[win], m_mat[win, :]
    mtnum = mdates.date2num(m_times)
    for j, t in enumerate(mtnum):
        ax.plot(t + gain * mm[:, j], md, color="red", lw=1.0, alpha=0.85)

    # T markers + boundary
    ax.axvline(mdates.date2num(T2), color="green", ls="--", lw=1.6)
    tr = ax.get_xaxis_transform()
    for x, lab, ha in [(tnum[0], "T1", "left"), (mdates.date2num(T2), "T2", "center"),
                       (tnum[-1], "T3", "right")]:
        ax.text(x, 0.98, lab, color="green", transform=tr, fontweight="bold",
                va="top", ha=ha, fontsize=12)

    # yellow star at observed peak
    ip, jt = np.unravel_index(int(np.nanargmax(np.abs(o_mat))), o_mat.shape)
    ax.plot(tnum[jt], o_depths[ip], marker="*", color="yellow", markersize=18,
            markeredgecolor="black", markeredgewidth=0.7, zorder=6)

    ax.set_ylim(o_depths[-1], o_depths[0])
    ax.set_ylabel("Gold 4-PB Measured Depth [ft]")
    ax.set_xlabel("Time [UTC-7]")
    ax.set_title("DASCore Strain Waterfall (T1 Referenced): Observed vs Simulated "
                 "(new-geometry two-SRV, C*=1.63e7)")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    cb = fig.colorbar(mesh, ax=ax, pad=0.01)
    cb.set_label("Strain (millistrain)")

    handles = [
        mlines.Line2D([], [], color="black", lw=1.2, label="Observed 4h-mean profile"),
        mlines.Line2D([], [], color="red", lw=1.4, label="Simulated (this study)"),
    ]
    ax.legend(handles=handles, loc="lower left", framealpha=0.9)

    out = FIG_DIR / "das_strain_waterfall_observed_vs_simulated.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")
    print(f"observed strain peak : {np.nanmax(np.abs(o_mat)):.4f} millistrain")
    print(f"simulated strain peak: {np.nanmax(np.abs(m_mat)):.4f} millistrain")


if __name__ == "__main__":
    main()
