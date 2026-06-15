"""
Crop the three Bearskin full-pressure records and average them for simulation.

Input:
    data_fervo/fiberis_format/pressure_data/Bearskin_*_Pressure.npz

Processing:
    Crop all traces to 2025-02-24 11:00:00 through 2025-02-28 00:00:00,
    align them on a common time axis, average the three pressure curves.

Output:
    data_fervo/fiberis_format/post_processing/synthetic_data_simulation.npz

The saved pressure values remain in psi. The current MOOSE baseline builder
converts these source gauge values from psi to Pa when building the input file.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
FIBERIS_SRC = REPO_ROOT / "fibeRIS" / "src"
if str(FIBERIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIBERIS_SRC))

from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge


PRESSURE_DIR = REPO_ROOT / "data_fervo" / "fiberis_format" / "pressure_data"
OUTPUT_DIR = REPO_ROOT / "data_fervo" / "fiberis_format" / "post_processing"
FIG_DIR = REPO_ROOT / "figs" / "tensile_fault_pressure_data"
OUTPUT_PATH = OUTPUT_DIR / "synthetic_data_simulation.npz"
SUMMARY_PATH = OUTPUT_DIR / "synthetic_data_simulation_summary.txt"
QC_FIG_PATH = FIG_DIR / "synthetic_data_simulation_qc.png"

CROP_START = dt.datetime(2025, 2, 24, 11, 0, 0)
CROP_END = dt.datetime(2025, 2, 28, 0, 0, 0)


def pretty_name(path: Path) -> str:
    return path.stem.replace("Bearskin_", "Bearskin ").replace("_Pressure", "")


def load_and_crop_gauges() -> list[Data1DGauge]:
    paths = sorted(PRESSURE_DIR.glob("Bearskin_*_Pressure.npz"))
    if len(paths) != 3:
        raise FileNotFoundError(
            f"Expected 3 full-pressure files in {PRESSURE_DIR}, found {len(paths)}"
        )

    gauges = []
    for path in paths:
        gauge = Data1DGauge()
        gauge.load_npz(str(path))
        gauge.name = pretty_name(path)
        gauge.select_time(CROP_START, CROP_END)
        gauges.append(gauge)
    return gauges


def common_time_axis(gauges: list[Data1DGauge]) -> tuple[dt.datetime, np.ndarray]:
    common_start = max(gauge.start_time for gauge in gauges)
    common_end = min(gauge.get_end_time(use_timestamp=True) for gauge in gauges)
    if common_start >= common_end:
        raise ValueError(f"Invalid overlap window: {common_start} to {common_end}")

    dts = []
    for gauge in gauges:
        diffs = np.diff(gauge.taxis)
        diffs = diffs[diffs > 0]
        if diffs.size:
            dts.append(float(np.median(diffs)))
    if not dts:
        raise ValueError("Cannot infer time step from cropped pressure gauges.")

    dt_seconds = float(np.median(dts))
    duration_seconds = (common_end - common_start).total_seconds()
    n_steps = int(np.floor(duration_seconds / dt_seconds)) + 1
    taxis = np.arange(n_steps, dtype=float) * dt_seconds
    return common_start, taxis


def average_gauges(gauges: list[Data1DGauge]) -> tuple[Data1DGauge, list[Data1DGauge]]:
    start_time, taxis = common_time_axis(gauges)
    aligned = []
    for gauge in gauges:
        aligned_gauge = gauge.copy()
        aligned_gauge.interpolate(taxis, new_start_time=start_time)
        aligned.append(aligned_gauge)

    stack = np.vstack([gauge.data for gauge in aligned])
    averaged = Data1DGauge(
        data=np.nanmean(stack, axis=0),
        taxis=taxis,
        start_time=start_time,
        name="synthetic_data_simulation",
    )
    averaged.history.add_record(
        "Created by averaging cropped Bearskin 1-IA, 3-PA, and 4-PB pressure traces.",
        level="INFO",
    )
    return averaged, aligned


def save_outputs(averaged: Data1DGauge, aligned: list[Data1DGauge]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    averaged.savez(str(OUTPUT_PATH))

    lines = [
        "synthetic_data_simulation.npz",
        f"crop_start: {CROP_START.isoformat()}",
        f"crop_end_requested: {CROP_END.isoformat()}",
        f"output_start_time: {averaged.start_time.isoformat()}",
        f"output_end_time: {averaged.get_end_time(use_timestamp=True).isoformat()}",
        f"n_points: {len(averaged.data)}",
        f"time_step_seconds: {np.median(np.diff(averaged.taxis)):.6g}",
        "pressure_unit: psi",
        f"pressure_min_psi: {np.nanmin(averaged.data):.6g}",
        f"pressure_max_psi: {np.nanmax(averaged.data):.6g}",
        f"pressure_mean_psi: {np.nanmean(averaged.data):.6g}",
        "source_curves:",
    ]
    for gauge in aligned:
        lines.append(
            f"  - {gauge.name}: min={np.nanmin(gauge.data):.6g} psi, "
            f"max={np.nanmax(gauge.data):.6g} psi, mean={np.nanmean(gauge.data):.6g} psi"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_qc(averaged: Data1DGauge, aligned: list[Data1DGauge]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12.5, 5.8))

    colors = {
        "Bearskin 1-IA": "#1f77b4",
        "Bearskin 3-PA": "#d62728",
        "Bearskin 4-PB": "#2ca02c",
    }
    for gauge in aligned:
        ax.plot(
            gauge.calculate_time(),
            gauge.data,
            linewidth=0.9,
            alpha=0.42,
            color=colors.get(gauge.name),
            label=gauge.name,
        )

    ax.plot(
        averaged.calculate_time(),
        averaged.data,
        linewidth=2.4,
        color="#111111",
        label="Average: synthetic_data_simulation",
        zorder=5,
    )

    ax.set_title("Cropped Pressure Curves and Averaged Simulation Source", fontsize=14)
    ax.set_xlabel("Date/time")
    ax.set_ylabel("Pressure (psi)")
    ax.set_ylim(bottom=0)
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.16)
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:,.0f}")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    ax.legend(loc="upper right", frameon=True, framealpha=0.94)
    fig.tight_layout()
    fig.savefig(QC_FIG_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    gauges = load_and_crop_gauges()
    averaged, aligned = average_gauges(gauges)
    save_outputs(averaged, aligned)
    plot_qc(averaged, aligned)

    print(f"Saved averaged simulation data: {OUTPUT_PATH}")
    print(f"Saved summary:                 {SUMMARY_PATH}")
    print(f"Saved QC figure:               {QC_FIG_PATH}")


if __name__ == "__main__":
    main()
