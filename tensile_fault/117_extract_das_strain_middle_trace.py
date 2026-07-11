"""
Visualize the legacy DASCore 4-hour-mean strain / strain-rate waterfalls and
extract the middle-depth trace from each into fiberis Data1D NPZ files.

Inputs (data_fervo/legacy/):
    strain_4h_mean_profiles_..._10200_10500ft_4h_mean_T1_ref.csv
    strain_rate_4h_mean_profiles_..._10200_10500ft_4h_mean_T1_ref.csv

Each CSV is a depth (rows) x time (columns) matrix:
    - column 0        : measured_depth_ft
    - columns 1..N    : 4-hour-mean profile timestamps (header) -> values

Outputs:
    figs/tensile_fault_qc/das_middle_trace/
        das_strain_waterfalls.png          (2-panel QC waterfall, like image.png)
        das_middle_trace_timeseries.png    (the two extracted middle traces)
    data_fervo/fiberis_format/post_processing/
        strain_das_middle_trace_<depth>ft.npz
        strain_rate_das_middle_trace_<depth>ft.npz

Run from the repository root:
    python scripts/tensile_fault/117_extract_das_strain_middle_trace.py
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
FIBERIS_SRC = REPO_ROOT / "fibeRIS" / "src"
if str(FIBERIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIBERIS_SRC))

from fiberis.analyzer.Data1D.core1D import Data1D


LEGACY_DIR = REPO_ROOT / "data_fervo" / "legacy"
STRAIN_CSV = LEGACY_DIR / (
    "strain_4h_mean_profiles_20250224_1500_to_20250303_2200_"
    "10200_10500ft_4h_mean_T1_ref.csv"
)
STRAIN_RATE_CSV = LEGACY_DIR / (
    "strain_rate_4h_mean_profiles_20250224_1500_to_20250303_2200_"
    "10200_10500ft_4h_mean_T1_ref.csv"
)

FIG_DIR = REPO_ROOT / "figs" / "tensile_fault_qc" / "das_middle_trace"
NPZ_DIR = REPO_ROOT / "data_fervo" / "fiberis_format" / "post_processing"


def load_waterfall(csv_path: Path):
    """Load a depth x time CSV into (depths_ft, times[datetime], matrix[depth, time])."""
    df = pd.read_csv(csv_path)
    depths = df["measured_depth_ft"].to_numpy(dtype=float)
    time_cols = df.columns[1:]
    times = np.array([dt.datetime.fromisoformat(c) for c in time_cols])
    matrix = df[time_cols].to_numpy(dtype=float)  # shape: (n_depth, n_time)
    return depths, times, matrix


def plot_waterfalls(strain, strain_rate, out_path: Path) -> None:
    """Reproduce the 2-panel QC waterfall (strain-rate on top, strain on bottom)."""
    sr_depths, sr_times, sr_mat = strain_rate
    s_depths, s_times, s_mat = strain

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    # Top: strain rate (nanostrain/s)
    m1 = ax_top.pcolormesh(
        sr_times, sr_depths, sr_mat, cmap="RdBu_r",
        vmin=-0.3, vmax=0.3, shading="nearest",
    )
    ax_top.set_ylabel("Gold 4-PB Measured Depth [ft]")
    ax_top.set_title("DASCore Strain-rate Waterfall (4-hour Mean Profiles)")
    ax_top.invert_yaxis()
    cb1 = fig.colorbar(m1, ax=ax_top, pad=0.01)
    cb1.set_label("Strain rate (nanostrain/s)")

    # Bottom: strain (millistrain, T1 referenced)
    m2 = ax_bot.pcolormesh(
        s_times, s_depths, s_mat, cmap="RdBu_r",
        vmin=-0.1, vmax=0.1, shading="nearest",
    )
    ax_bot.set_ylabel("Gold 4-PB Measured Depth [ft]")
    ax_bot.set_title("DASCore Strain Waterfall (4-hour Mean Profiles, T1 Referenced)")
    ax_bot.set_xlabel("Time [UTC-7]")
    ax_bot.invert_yaxis()
    cb2 = fig.colorbar(m2, ax=ax_bot, pad=0.01)
    cb2.set_label("Strain (millistrain)")

    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved waterfall QC figure -> {out_path}")


def extract_middle_trace(depths, times, matrix, name: str) -> tuple[Data1D, float]:
    """Extract the middle-depth trace as a fiberis Data1D time series."""
    mid_idx = len(depths) // 2
    mid_depth = float(depths[mid_idx])
    trace = matrix[mid_idx, :].astype(float)

    start_time = times[0]
    taxis = np.array([(t - start_time).total_seconds() for t in times], dtype=float)

    d1d = Data1D(
        data=trace,
        taxis=taxis,
        start_time=start_time,
        name=name,
    )
    return d1d, mid_depth


def plot_middle_traces(strain_d1d, s_depth, rate_d1d, r_depth, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    strain_d1d.plot(ax=ax1, use_timestamp=True)
    ax1.set_title(f"Middle-depth strain trace @ {s_depth:.1f} ft")
    ax1.set_ylabel("Strain (millistrain)")

    rate_d1d.plot(ax=ax2, use_timestamp=True)
    ax2.set_title(f"Middle-depth strain-rate trace @ {r_depth:.1f} ft")
    ax2.set_ylabel("Strain rate (nanostrain/s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved middle-trace figure -> {out_path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading legacy DAS waterfalls...")
    strain = load_waterfall(STRAIN_CSV)
    strain_rate = load_waterfall(STRAIN_RATE_CSV)
    print(f"  strain      : {strain[2].shape[0]} depths x {strain[2].shape[1]} times")
    print(f"  strain_rate : {strain_rate[2].shape[0]} depths x {strain_rate[2].shape[1]} times")
    print(f"  depth range : {strain[0][0]:.1f} - {strain[0][-1]:.1f} ft")
    print(f"  time range  : {strain[1][0]} -> {strain[1][-1]}")

    # 1. QC waterfall (should look like data_fervo/legacy/image.png)
    plot_waterfalls(strain, strain_rate, FIG_DIR / "das_strain_waterfalls.png")

    # 2. Extract the middle-depth trace from each
    strain_d1d, s_depth = extract_middle_trace(*strain, name="das_strain_middle_trace")
    rate_d1d, r_depth = extract_middle_trace(*strain_rate, name="das_strain_rate_middle_trace")
    print(f"Middle strain trace      @ {s_depth:.2f} ft ({strain_d1d.data.size} pts)")
    print(f"Middle strain-rate trace @ {r_depth:.2f} ft ({rate_d1d.data.size} pts)")

    plot_middle_traces(strain_d1d, s_depth, rate_d1d, r_depth,
                       FIG_DIR / "das_middle_trace_timeseries.png")

    # 3. Save fiberis Data1D NPZ files
    strain_npz = NPZ_DIR / f"strain_das_middle_trace_{int(round(s_depth))}ft.npz"
    rate_npz = NPZ_DIR / f"strain_rate_das_middle_trace_{int(round(r_depth))}ft.npz"
    strain_d1d.savez(str(strain_npz))
    rate_d1d.savez(str(rate_npz))
    print(f"Saved -> {strain_npz}")
    print(f"Saved -> {rate_npz}")

    # 4. Round-trip sanity check
    check = Data1D()
    check.load_npz(str(strain_npz))
    print(f"Round-trip OK: {check.name}, start={check.start_time}, "
          f"n={check.data.size}, taxis[-1]={check.taxis[-1]:.0f}s")


if __name__ == "__main__":
    main()
