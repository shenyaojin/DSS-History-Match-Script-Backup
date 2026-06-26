"""
Run a synthetic log-ramp pressure injection case with the perpendicular monitor.

The curve keeps the same start/end time and sampling as the cropped real
simulation pressure, but replaces pressure with a monotonic log-style ramp.
"""

from __future__ import annotations

import importlib.util
import datetime as dt
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FIBERIS_SRC = REPO_ROOT / "fibeRIS" / "src"
if str(FIBERIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIBERIS_SRC))

from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.moose.templates.baseline_model_generator_fervo import build_baseline_model


SCRIPT_111 = REPO_ROOT / "scripts" / "tensile_fault" / "111_run_perpendicular_monitor_5sixth.py"
spec = importlib.util.spec_from_file_location("perpendicular_case", SCRIPT_111)
case = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(case)


PROJECT_NAME = "0625_perpendicular_monitor_5sixth_log_pressure_synthetic"
MATRIX_PERM = 1e-18
CURVE_STYLE_NAME = "Log-Style"
CURVE_SHORT_NAME = "log"
PRESSURE_LINE_LABEL = "Synthetic log injection"
PLOT_TITLE_PREFIX = "Synthetic Log-Pressure Case"
REFERENCE_PRESSURE_PATH = (
    REPO_ROOT
    / "data_fervo"
    / "fiberis_format"
    / "post_processing"
    / "synthetic_data_simulation_20250224_1500_to_20250228_0000.npz"
)
SYNTHETIC_PRESSURE_PATH = (
    REPO_ROOT
    / "data_fervo"
    / "fiberis_format"
    / "post_processing"
    / "synthetic_log_pressure_20250224_1500_to_20250228_0000.npz"
)
OUTPUT_DIR = REPO_ROOT / "output" / PROJECT_NAME
FIG_DIR = REPO_ROOT / "figs" / "tensile_fault_qc" / PROJECT_NAME


def set_case_paths() -> None:
    case.PROJECT_NAME = PROJECT_NAME
    case.MATRIX_PERM = MATRIX_PERM
    case.CROPPED_PRESSURE_PATH = SYNTHETIC_PRESSURE_PATH
    case.OUTPUT_DIR = OUTPUT_DIR
    case.EXPORT_DIR = OUTPUT_DIR / "postprocessor_npz"
    case.FIG_DIR = FIG_DIR


def create_log_pressure_curve() -> Data1DGauge:
    reference = Data1DGauge()
    reference.load_npz(str(REFERENCE_PRESSURE_PATH))

    tau = reference.taxis / float(reference.taxis[-1])
    ramp = np.log1p(14.0 * tau) / np.log1p(14.0)

    start_psi = 3000.0
    end_psi = 8000.0
    pressure = Data1DGauge(
        data=start_psi + (end_psi - start_psi) * ramp,
        taxis=reference.taxis.copy(),
        start_time=reference.start_time,
        name=f"synthetic_{CURVE_SHORT_NAME}_pressure",
    )
    SYNTHETIC_PRESSURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pressure.savez(str(SYNTHETIC_PRESSURE_PATH))
    return pressure


def save_geometry() -> Path:
    builder = build_baseline_model(
        project_name=PROJECT_NAME,
        pressure_profile_path=str(SYNTHETIC_PRESSURE_PATH),
        hf_length_ft=case.HF_LENGTH_FT,
        shift_list_ft=np.array([case.MONITOR_SHIFT_FT]),
        angle=case.MONITOR_ANGLE_DEG,
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    geometry_path = FIG_DIR / "geometry.png"
    builder.plot_geometry(save_path=str(geometry_path), hide_legend=True, equal_aspect=True)
    return geometry_path


def plot_synthetic_pressure(pressure: Data1DGauge) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    ax.plot(
        pressure.taxis / 3600.0,
        pressure.data,
        color="black",
        linewidth=2.2,
    )
    ax.set_title(f"Synthetic {CURVE_STYLE_NAME} Injection Pressure")
    ax.set_xlabel("Simulation time (hours)")
    ax.set_ylabel("Pressure (psi)")
    ax.grid(True, linestyle=":", alpha=0.35)
    path = FIG_DIR / f"synthetic_{CURVE_SHORT_NAME}_injection_curve.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def parse_start_time(value) -> dt.datetime:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, dt.datetime):
        return value
    return dt.datetime.fromisoformat(str(value))


def datetime_axis(start_time: dt.datetime, taxis: np.ndarray) -> list[dt.datetime]:
    return [start_time + dt.timedelta(seconds=float(seconds)) for seconds in taxis]


def finite_symmetric_limits(data: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return -1.0, 1.0
    value = np.nanpercentile(np.abs(finite), percentile)
    return (-value, value) if value > 0 else (-1.0, 1.0)


def load_waterfall_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing exported waterfall file: {path}")
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def plot_waterfall_with_log_pressure(
    pressure: Data1DGauge,
    waterfall_path: Path,
    output_filename: str,
    title: str,
    cbar_label: str,
) -> Path:
    waterfall = load_waterfall_npz(waterfall_path)
    start_time = parse_start_time(waterfall.get("start_time", pressure.start_time))
    x_dates = datetime_axis(start_time, waterfall["taxis"])
    pressure_dates = datetime_axis(pressure.start_time, pressure.taxis)
    clim = finite_symmetric_limits(waterfall["data"])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax_map, ax_pressure) = plt.subplots(
        2,
        1,
        figsize=(12.8, 8.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.8, 1.0]},
    )

    mesh = ax_map.pcolormesh(
        x_dates,
        waterfall["daxis"],
        waterfall["data"],
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

    ax_pressure.plot(
        pressure_dates,
        pressure.data,
        color="black",
        linewidth=2.1,
        label=PRESSURE_LINE_LABEL,
    )
    ax_pressure.set_ylabel("Pressure (psi)")
    ax_pressure.set_xlabel("Datetime")
    ax_pressure.grid(True, linestyle=":", alpha=0.35)
    ax_pressure.legend(loc="upper left", frameon=False)
    ax_pressure.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax_pressure.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    fig.autofmt_xdate(rotation=25, ha="right")

    path = FIG_DIR / output_filename
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_coplots(pressure: Data1DGauge) -> list[Path]:
    return [
        plot_waterfall_with_log_pressure(
            pressure,
            case.EXPORT_DIR / "monitor_normal_strain_no_rotation.npz",
            f"synthetic_{CURVE_SHORT_NAME}_strain_waterfall_with_pressure_datetime.png",
            f"{PLOT_TITLE_PREFIX}: Strain Waterfall",
            "strain_yy, baseline corrected",
        ),
        plot_waterfall_with_log_pressure(
            pressure,
            case.EXPORT_DIR / "monitor_normal_strain_rate_no_rotation.npz",
            f"synthetic_{CURVE_SHORT_NAME}_strain_rate_waterfall_with_pressure_datetime.png",
            f"{PLOT_TITLE_PREFIX}: Strain-Rate Waterfall",
            "strain_rate_yy (1/s)",
        ),
    ]


def main() -> None:
    set_case_paths()
    pressure = create_log_pressure_curve()
    curve_path = plot_synthetic_pressure(pressure)
    geometry_path = save_geometry()

    print(f"Synthetic pressure saved to: {SYNTHETIC_PRESSURE_PATH}")
    print(f"Synthetic pressure range: {np.nanmin(pressure.data):.1f} to {np.nanmax(pressure.data):.1f} psi")
    print(f"Synthetic curve figure: {curve_path}")
    print(f"Geometry figure: {geometry_path}")

    case.build_and_run_model()
    case.export_results(pressure)
    for path in plot_coplots(pressure):
        print(f"Saved waterfall/pressure datetime coplot: {path}")


if __name__ == "__main__":
    main()
