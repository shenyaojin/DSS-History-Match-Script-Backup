"""
Run a perpendicular-monitor tensile-fault MOOSE case.

This case uses the averaged pressure curve cropped from Feb 24 15:00 to
Feb 28 00:00, places the monitor/fracture intersection at 5/6 of the fracture
length, and exports pressure, no-rotation strain_yy, and no-rotation
strain_rate_yy deliverables.
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
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
FIBERIS_SRC = REPO_ROOT / "fibeRIS" / "src"
if str(FIBERIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIBERIS_SRC))

from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.moose.runner import MooseRunner
from fiberis.moose.templates.baseline_model_generator_fervo import (
    all_line_post_processor_info_extractor,
    build_baseline_model,
)


PROJECT_NAME = "0625_perpendicular_monitor_5sixth"
HF_LENGTH_FT = 250.0
MONITOR_SHIFT_FT = round(HF_LENGTH_FT / 3.0, 3)
MONITOR_ANGLE_DEG = 0.0
NUM_PROCESSORS = 20
PSI_TO_PA = 6894.76
MATRIX_PERM = 1e-18

SOURCE_PRESSURE_PATH = (
    REPO_ROOT
    / "data_fervo"
    / "fiberis_format"
    / "post_processing"
    / "synthetic_data_simulation.npz"
)
CROPPED_PRESSURE_PATH = (
    REPO_ROOT
    / "data_fervo"
    / "fiberis_format"
    / "post_processing"
    / "synthetic_data_simulation_20250224_1500_to_20250228_0000.npz"
)
OUTPUT_DIR = REPO_ROOT / "output" / PROJECT_NAME
EXPORT_DIR = OUTPUT_DIR / "postprocessor_npz"
FIG_DIR = REPO_ROOT / "figs" / "tensile_fault_qc" / PROJECT_NAME


def crop_pressure_curve() -> Data1DGauge:
    pressure = Data1DGauge()
    pressure.load_npz(str(SOURCE_PRESSURE_PATH))

    crop_start = dt.datetime(pressure.start_time.year, 2, 24, 15, 0, 0)
    crop_end = dt.datetime(pressure.start_time.year, 2, 28, 0, 0, 0)
    pressure.crop(crop_start, crop_end)
    CROPPED_PRESSURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pressure.savez(str(CROPPED_PRESSURE_PATH))
    return pressure


def finite_limits(data, percentile=99.0, symmetric=True):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return (-1.0, 1.0)
    if symmetric:
        value = np.nanpercentile(np.abs(finite), percentile)
        return (-value, value) if value > 0 else (-1.0, 1.0)
    return (
        np.nanpercentile(finite, 100.0 - percentile),
        np.nanpercentile(finite, percentile),
    )


def baseline_correct(dataframe):
    corrected = dataframe.copy()
    data = corrected.data.copy()
    finite_mask = np.isfinite(data)
    valid_rows = finite_mask.any(axis=1)
    baseline = np.full(data.shape[0], np.nan)
    first_finite_cols = np.argmax(finite_mask[valid_rows], axis=1)
    baseline[valid_rows] = data[valid_rows, first_finite_cols]
    data[valid_rows, :] -= baseline[valid_rows, np.newaxis]
    corrected.data = data
    return corrected


def build_and_run_model() -> None:
    builder = build_baseline_model(
        project_name=PROJECT_NAME,
        pressure_profile_path=str(CROPPED_PRESSURE_PATH),
        hf_length_ft=HF_LENGTH_FT,
        shift_list_ft=np.array([MONITOR_SHIFT_FT]),
        angle=MONITOR_ANGLE_DEG,
        matrix_perm=MATRIX_PERM,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_file_path = OUTPUT_DIR / f"{PROJECT_NAME}_input.i"
    builder.generate_input_file(output_filepath=str(input_file_path))

    runner = MooseRunner(
        moose_executable_path=str(
            REPO_ROOT / "moose_env" / "moose" / "modules" / "porous_flow" / "porous_flow-opt"
        ),
        mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec",
    )
    success, _, stderr = runner.run(
        input_file_path=str(input_file_path),
        output_directory=str(OUTPUT_DIR),
        num_processors=NUM_PROCESSORS,
        log_file_name="simulation.log",
        stream_output=True,
    )
    if not success:
        raise RuntimeError(f"MOOSE simulation failed. stderr={stderr}")


def combine_records(records):
    combined = {}
    for record in records:
        shift_ft = record["shift_ft"]
        combined.setdefault(
            shift_ft,
            {
                "sampler_name": f"fiber_line_{shift_ft:g}ft",
                "sampler_type": "combined",
                "shift_ft": shift_ft,
            },
        ).update(record)
    return [combined[key] for key in sorted(combined)]


def save_data2d(dataframe, filename):
    path = EXPORT_DIR / filename
    dataframe.savez(str(path))
    return path


def plot_map(dataframe, filename, title, cbar_label, cmap="bwr", symmetric=True):
    clim = finite_limits(dataframe.data, symmetric=symmetric)
    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    mesh = ax.pcolormesh(
        dataframe.taxis / 3600.0,
        dataframe.daxis,
        dataframe.data,
        shading="auto",
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
    )
    ax.set_title(title)
    ax.set_xlabel("Simulation time (hours)")
    ax.set_ylabel("Position along perpendicular monitor (m)")
    ax.grid(True, linestyle=":", alpha=0.2)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(cbar_label)
    path = FIG_DIR / filename
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_pressure_source_mean_max(pressure_dataframe, source_pressure):
    pressure_mpa = pressure_dataframe.data / 1.0e6
    finite_counts = np.isfinite(pressure_mpa).sum(axis=0)
    valid_cols = finite_counts > 0
    mean_pressure = np.full(pressure_mpa.shape[1], np.nan)
    max_pressure = np.full(pressure_mpa.shape[1], np.nan)
    mean_pressure[valid_cols] = (
        np.nansum(pressure_mpa[:, valid_cols], axis=0) / finite_counts[valid_cols]
    )
    max_pressure[valid_cols] = np.nanmax(pressure_mpa[:, valid_cols], axis=0)

    fig, ax = plt.subplots(figsize=(11.5, 5.5), constrained_layout=True)
    ax.plot(
        source_pressure.taxis / 3600.0,
        source_pressure.data * PSI_TO_PA / 1.0e6,
        color="black",
        linewidth=2.0,
        label="Injection/source pressure",
    )
    ax.plot(
        pressure_dataframe.taxis / 3600.0,
        mean_pressure,
        color="#1f77b4",
        linewidth=1.6,
        label="Mean sampled pressure",
    )
    ax.plot(
        pressure_dataframe.taxis / 3600.0,
        max_pressure,
        color="#d62728",
        linewidth=1.4,
        label="Max sampled pressure",
    )
    ax.set_title("Injection Curve vs Sampled Model Pressure")
    ax.set_xlabel("Simulation time (hours)")
    ax.set_ylabel("Pressure (MPa)")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend()
    path = FIG_DIR / "pressure_source_mean_max.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def export_point_postprocessors():
    csv_path = OUTPUT_DIR / f"{PROJECT_NAME}_input_csv.csv"
    if not csv_path.exists():
        return []
    point_df = pd.read_csv(csv_path)
    npz_path = EXPORT_DIR / "point_postprocessors.npz"
    np.savez(
        npz_path,
        time=point_df["time"].to_numpy(),
        columns=np.array(point_df.columns.to_list()),
        data=point_df.to_numpy(),
    )

    paths = [npz_path]
    for token, title, ylabel, filename in [
        ("dispx", "Point Postprocessor X Displacement", "disp_x (m)", "point_postprocessor_dispx.png"),
        ("dispy", "Point Postprocessor Y Displacement", "disp_y (m)", "point_postprocessor_dispy.png"),
    ]:
        columns = [column for column in point_df.columns if token in column]
        if not columns:
            continue
        fig, ax = plt.subplots(figsize=(11.5, 5.5), constrained_layout=True)
        for column in columns:
            ax.plot(point_df["time"] / 3600.0, point_df[column], linewidth=1.5, label=column)
        ax.set_title(title)
        ax.set_xlabel("Simulation time (hours)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(fontsize=8)
        fig_path = FIG_DIR / filename
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        paths.append(fig_path)
    return paths


def write_summary(record, paths, source_pressure):
    summary_path = FIG_DIR / "perpendicular_monitor_summary.txt"
    lines = [
        f"{PROJECT_NAME} summary",
        f"pressure source: {SOURCE_PRESSURE_PATH}",
        f"cropped pressure: {CROPPED_PRESSURE_PATH}",
        f"crop window: {source_pressure.start_time} to {source_pressure.start_time + dt.timedelta(seconds=float(source_pressure.taxis[-1]))}",
        f"hf_length_ft: {HF_LENGTH_FT:g}",
        f"monitor_shift_ft_from_center: {MONITOR_SHIFT_FT:g}",
        "normalized intersection: 5/6 of fracture length from left tip",
        f"monitor_angle_deg_from_vertical: {MONITOR_ANGLE_DEG:g}",
        f"matrix_perm_m2: {MATRIX_PERM:.6g}",
        "tensor_rotation: none; monitor-normal strain is global strain_yy",
        "",
        "Data ranges:",
    ]
    for key, value in record.items():
        if not hasattr(value, "data"):
            continue
        finite = np.isfinite(value.data)
        lines.append(
            f"- {value.name}: shape={value.data.shape}, "
            f"finite={finite.sum()}/{finite.size}, "
            f"range=[{np.nanmin(value.data):.6g}, {np.nanmax(value.data):.6g}]"
        )
    lines.extend(["", "Deliverables:"])
    for path in paths:
        lines.append(f"- {path}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def export_results(source_pressure):
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    records = combine_records(all_line_post_processor_info_extractor(output_dir=str(OUTPUT_DIR)))
    if len(records) != 1:
        raise RuntimeError(f"Expected one perpendicular monitor line, found {len(records)}.")
    record = records[0]

    for value in record.values():
        if hasattr(value, "start_time"):
            value.start_time = source_pressure.start_time

    pressure = record["pressure"]
    strain_xx = baseline_correct(record["strain_xx"])
    strain_yy = baseline_correct(record["strain_yy"])
    strain_xy = baseline_correct(record["strain_xy"])
    strain_rate_xx = record["strain_rate_xx"]
    strain_rate_yy = record["strain_rate_yy"]
    strain_rate_xy = record["strain_rate_xy"]

    strain_yy.name = "monitor_normal_strain_no_rotation"
    strain_rate_yy.name = "monitor_normal_strain_rate_no_rotation"
    record.update({
        "pressure": pressure,
        "strain_xx": strain_xx,
        "strain_yy": strain_yy,
        "strain_xy": strain_xy,
        "strain_rate_xx": strain_rate_xx,
        "strain_rate_yy": strain_rate_yy,
        "strain_rate_xy": strain_rate_xy,
    })

    paths = [
        save_data2d(pressure, "pressure.npz"),
        save_data2d(strain_xx, "strain_xx.npz"),
        save_data2d(strain_yy, "monitor_normal_strain_no_rotation.npz"),
        save_data2d(strain_xy, "strain_xy.npz"),
        save_data2d(strain_rate_xx, "strain_rate_xx.npz"),
        save_data2d(strain_rate_yy, "monitor_normal_strain_rate_no_rotation.npz"),
        save_data2d(strain_rate_xy, "strain_rate_xy.npz"),
        plot_pressure_source_mean_max(pressure, source_pressure),
        plot_map(pressure, "pressure_map.png", "Sampled Pressure Along Perpendicular Monitor", "pressure (Pa)", cmap="viridis", symmetric=False),
        plot_map(strain_yy, "monitor_normal_strain_no_rotation.png", "Monitor-Normal Strain, No Tensor Rotation", "strain"),
        plot_map(strain_rate_yy, "monitor_normal_strain_rate_no_rotation.png", "Monitor-Normal Strain Rate, No Tensor Rotation", "strain rate (1/s)"),
    ]
    paths.extend(export_point_postprocessors())
    summary_path = write_summary(record, paths, source_pressure)
    print(f"Saved exported data to: {EXPORT_DIR}")
    print(f"Saved QC figures to: {FIG_DIR}")
    print(summary_path)


def main():
    source_pressure = crop_pressure_curve()
    print(f"Cropped averaged pressure curve saved to: {CROPPED_PRESSURE_PATH}")
    print(f"Monitor shift from fracture center: {MONITOR_SHIFT_FT:g} ft")
    print("Monitor/fracture intersection: 5/6 of fracture length from left tip")
    print("Monitor is perpendicular to fracture; tensor rotation disabled.")
    build_and_run_model()
    export_results(source_pressure)


if __name__ == "__main__":
    main()
