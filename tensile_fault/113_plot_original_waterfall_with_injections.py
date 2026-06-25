"""
Plot the original-injection perpendicular-monitor waterfall together with
the three cropped well pressure curves.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FIBERIS_SRC = REPO_ROOT / "fibeRIS" / "src"
if str(FIBERIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIBERIS_SRC))

from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge


PSI_TO_PA = 6894.76
PROJECT_NAME = "0625_perpendicular_monitor_5sixth"

FIBERIS_DATA_DIR = REPO_ROOT / "data_fervo" / "fiberis_format"
POST_PROCESSING_DIR = FIBERIS_DATA_DIR / "post_processing"
PRESSURE_DATA_DIR = FIBERIS_DATA_DIR / "pressure_data"
SOURCE_PRESSURE_PATH = POST_PROCESSING_DIR / "synthetic_data_simulation.npz"
CROPPED_PRESSURE_PATH = (
    POST_PROCESSING_DIR / "synthetic_data_simulation_20250224_1500_to_20250228_0000.npz"
)
WELL_PRESSURE_PATHS = [
    (PRESSURE_DATA_DIR / "Bearskin_1-IA_Pressure.npz", "Bearskin 1-IA"),
    (PRESSURE_DATA_DIR / "Bearskin_3-PA_Pressure.npz", "Bearskin 3-PA"),
    (PRESSURE_DATA_DIR / "Bearskin_4-PB_Pressure.npz", "Bearskin 4-PB"),
]

EXPORT_DIR = REPO_ROOT / "output" / PROJECT_NAME / "postprocessor_npz"
FIG_DIR = REPO_ROOT / "figs" / "tensile_fault_qc" / PROJECT_NAME


def crop_original_pressure() -> Data1DGauge:
    pressure = Data1DGauge()
    pressure.load_npz(str(SOURCE_PRESSURE_PATH))
    crop_start = dt.datetime(pressure.start_time.year, 2, 24, 15, 0, 0)
    crop_end = dt.datetime(pressure.start_time.year, 2, 28, 0, 0, 0)
    pressure.crop(crop_start, crop_end)
    CROPPED_PRESSURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pressure.savez(str(CROPPED_PRESSURE_PATH))
    return pressure


def crop_pressure(path: Path, name: str) -> Data1DGauge:
    pressure = Data1DGauge()
    pressure.load_npz(str(path))
    crop_start = dt.datetime(pressure.start_time.year, 2, 24, 15, 0, 0)
    crop_end = dt.datetime(pressure.start_time.year, 2, 28, 0, 0, 0)
    pressure.crop(crop_start, crop_end)
    pressure.name = name
    return pressure


def load_data2d(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing exported waterfall data: {path}")
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def finite_symmetric_limits(data: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return -1.0, 1.0
    value = np.nanpercentile(np.abs(finite), percentile)
    return (-value, value) if value > 0 else (-1.0, 1.0)


def pressure_mpa(pressure: Data1DGauge) -> np.ndarray:
    return pressure.data * PSI_TO_PA / 1.0e6


def plot_combined_waterfall(
    waterfall_npz: Path,
    output_filename: str,
    title: str,
    cbar_label: str,
) -> Path:
    crop_original_pressure()
    well_pressures = [
        crop_pressure(path, label)
        for path, label in WELL_PRESSURE_PATHS
    ]

    waterfall = load_data2d(waterfall_npz)
    waterfall_data = waterfall["data"]
    time_hours = waterfall["taxis"] / 3600.0
    distance_m = waterfall["daxis"]
    clim = finite_symmetric_limits(waterfall_data)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax_map, ax_curve) = plt.subplots(
        2,
        1,
        figsize=(12.5, 8.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.7, 1.0]},
    )

    mesh = ax_map.pcolormesh(
        time_hours,
        distance_m,
        waterfall_data,
        shading="auto",
        cmap="bwr",
        vmin=clim[0],
        vmax=clim[1],
    )
    ax_map.set_title(title)
    ax_map.set_ylabel("Position along perpendicular monitor (m)")
    ax_map.grid(True, linestyle=":", alpha=0.18)
    cbar = fig.colorbar(mesh, ax=ax_map)
    cbar.set_label(cbar_label)

    for pressure, color, linewidth in [
        (well_pressures[0], "black", 1.9),
        (well_pressures[1], "#1f77b4", 1.7),
        (well_pressures[2], "#d62728", 1.7),
    ]:
        ax_curve.plot(
            pressure.taxis / 3600.0,
            pressure_mpa(pressure),
            color=color,
            linewidth=linewidth,
            label=pressure.name,
        )

    ax_curve.set_xlabel("Simulation time since crop start (hours)")
    ax_curve.set_ylabel("Well pressure (MPa)")
    ax_curve.grid(True, linestyle=":", alpha=0.35)
    ax_curve.legend(loc="upper right", frameon=False, ncols=3)

    path = FIG_DIR / output_filename
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> None:
    paths = [
        plot_combined_waterfall(
            EXPORT_DIR / "monitor_normal_strain_no_rotation.npz",
            "original_strain_waterfall_with_three_well_pressure_curves.png",
            "Original-Case Strain Waterfall With Cropped Well Pressure Curves",
            "strain_yy, baseline corrected",
        ),
        plot_combined_waterfall(
            EXPORT_DIR / "monitor_normal_strain_rate_no_rotation.npz",
            "original_strain_rate_waterfall_with_three_well_pressure_curves.png",
            "Original-Case Strain-Rate Waterfall With Cropped Well Pressure Curves",
            "strain_rate_yy (1/s)",
        ),
    ]
    for path in paths:
        print(f"Saved combined waterfall/pressure figure to: {path}")


if __name__ == "__main__":
    main()
