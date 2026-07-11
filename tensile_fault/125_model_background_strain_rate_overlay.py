"""
Strain-RATE waterfall with OUR simulation as the background and the observed DAS
4-hour-mean strain-rate profiles overlaid as black wiggles (companion to 124).

Background (color) = simulated monitor strain-rate from the new-geometry two-SRV
                     run, 1/s -> nanostrain/s, mapped from monitor position (m) to
                     measured depth (ft) with the fracture crossing at the star.
Black wiggles       = observed DAS 4-hour-mean strain-rate profiles (nanostrain/s).

Run from the repository root:
    python scripts/tensile_fault/125_model_background_strain_rate_overlay.py
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
STRAIN_RATE_CSV = LEGACY_DIR / (
    "strain_rate_4h_mean_profiles_20250224_1500_to_20250303_2200_"
    "10200_10500ft_4h_mean_T1_ref.csv"
)
MODEL_RATE_NPZ = (
    REPO_ROOT / "output" / "0711_new_geometry_two_srv" / "postprocessor_npz"
    / "monitor_normal_strain_rate_no_rotation.npz"
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
    return depths, times, df[tcols].to_numpy(float)  # nanostrain/s


def load_model():
    d = np.load(MODEL_RATE_NPZ, allow_pickle=True)
    data = d["data"].astype(float)          # (position, time) 1/s
    dax = d["daxis"].astype(float)
    tax = d["taxis"].astype(float)
    start = d["start_time"].item()
    if isinstance(start, str):
        start = dt.datetime.fromisoformat(start)
    times = np.array([start + dt.timedelta(seconds=float(s)) for s in tax])
    # drop time columns that are all-NaN (strain-rate is NaN at t=0)
    good = np.isfinite(data).any(axis=0)
    data, times = data[:, good], times[good]
    frac_pos = 0.5 * (dax.min() + dax.max())     # fracture crossing = monitor midpoint
    depth_ft = STAR_DEPTH_FT + (dax - frac_pos) / FT
    return depth_ft, times, data * 1e9           # -> nanostrain/s


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    o_depths, o_times, o_mat = load_observed(STRAIN_RATE_CSV)
    m_depths, m_times, m_mat = load_model()

    vlim = 0.3
    fig, ax = plt.subplots(figsize=(16, 6.5), constrained_layout=True)
    mesh = ax.pcolormesh(mdates.date2num(m_times), m_depths, m_mat,
                         cmap="bwr", vmin=-vlim, vmax=vlim, shading="gouraud")

    tnum = mdates.date2num(o_times)
    gain = 1.6 * np.median(np.diff(tnum)) / np.nanmax(np.abs(o_mat))
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

    ax.set_ylim(o_depths[-1], o_depths[0])
    ax.set_xlim(tnum[0], tnum[-1])
    ax.set_ylabel("Gold 4-PB Measured Depth [ft]")
    ax.set_xlabel("Time [UTC-7]")
    ax.set_title("Simulated Strain-rate Waterfall (background) with Observed 4-hour Mean "
                 "Profiles (black) — new-geometry two-SRV, C*=1.63e7")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    cb = fig.colorbar(mesh, ax=ax, pad=0.01)
    cb.set_label("Strain rate (nanostrain/s)  —  background: simulated")

    ax.legend(handles=[mlines.Line2D([], [], color="black", lw=1.2,
                                     label="Observed 4h-mean profile")],
              loc="lower left", framealpha=0.9)

    out = FIG_DIR / "das_strain_rate_model_background_observed_overlay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")
    print(f"observed strain-rate peak : {np.nanmax(np.abs(o_mat)):.3f} nanostrain/s")
    print(f"simulated strain-rate peak: {np.nanmax(np.abs(m_mat)):.3f} nanostrain/s")


if __name__ == "__main__":
    main()
