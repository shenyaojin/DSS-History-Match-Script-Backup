"""
Export and QC every fiber line VectorPostprocessor from the tensile-fault MOOSE run.
"""

from __future__ import annotations

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
from fiberis.analyzer.TensorProcessor.coreT2D import Tensor2D
from fiberis.moose.templates.baseline_model_generator_fervo import (
    all_line_post_processor_info_extractor,
)


PROJECT_NAME = "1124_misfit_func"
OUTPUT_DIR = REPO_ROOT / "output" / PROJECT_NAME
EXPORT_DIR = OUTPUT_DIR / "postprocessor_npz"
FIG_DIR = REPO_ROOT / "figs" / "tensile_fault_qc" / PROJECT_NAME / "all_line_postprocessors"
SOURCE_PRESSURE_PATH = (
    REPO_ROOT
    / "data_fervo"
    / "fiberis_format"
    / "post_processing"
    / "synthetic_data_simulation.npz"
)
FIBER_ANGLE_FROM_Y_DEG = 30.0
PSI_TO_PA = 6894.76


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


def tensor2d_from_components(component_xx, component_yy, component_xy, name):
    tensor = np.zeros((component_xx.data.shape[0], component_xx.data.shape[1], 2, 2))
    tensor[:, :, 0, 0] = component_xx.data
    tensor[:, :, 1, 1] = component_yy.data
    tensor[:, :, 0, 1] = component_xy.data
    tensor[:, :, 1, 0] = component_xy.data
    return Tensor2D(
        data=tensor,
        taxis=component_xx.taxis,
        daxis=component_xx.daxis,
        dim=2,
        start_time=component_xx.start_time,
        name=name,
    )


def fiber_aligned_component(component_xx, component_yy, component_xy, name):
    tensor = tensor2d_from_components(component_xx, component_yy, component_xy, name)
    return tensor.get_directional_component(
        FIBER_ANGLE_FROM_Y_DEG,
        reference_axis="y",
        clockwise=True,
        name=f"{name}_fiber_aligned",
    )


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


def save_data2d(dataframe, filename):
    path = EXPORT_DIR / filename
    dataframe.savez(str(path))
    return path


def pcolormesh(ax, dataframe, data, title, cmap, clim, cbar_label):
    mesh = ax.pcolormesh(
        dataframe.taxis / 3600.0,
        dataframe.daxis,
        data,
        shading="auto",
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
    )
    ax.set_title(title)
    ax.set_ylabel("Position along line (m)")
    ax.grid(True, linestyle=":", alpha=0.2)
    cbar = ax.figure.colorbar(mesh, ax=ax)
    cbar.set_label(cbar_label)
    return mesh


def plot_stacked_maps(records, key, filename, title, cbar_label, cmap="bwr", symmetric=True):
    selected = [record for record in records if key in record]
    if not selected:
        return None

    clim = finite_limits(
        np.concatenate([record[key].data.ravel() for record in selected]),
        symmetric=symmetric,
    )
    fig, axes = plt.subplots(
        len(selected),
        1,
        figsize=(12, max(3.0 * len(selected), 4.0)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, record in zip(axes, selected):
        shift = record["shift_ft"]
        pcolormesh(
            ax,
            record[key],
            record[key].data,
            f"{title}, offset={shift:g} ft",
            cmap,
            clim,
            cbar_label,
        )
    axes[-1].set_xlabel("Simulation time (hours)")
    path = FIG_DIR / filename
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_pressure_means(records, source_pressure):
    selected = [record for record in records if "pressure" in record]
    if not selected:
        return None
    fig, ax = plt.subplots(figsize=(11.5, 5.5), constrained_layout=True)
    ax.plot(
        source_pressure.taxis / 3600.0,
        source_pressure.data * PSI_TO_PA / 1.0e6,
        color="black",
        linewidth=2.0,
        label="Injection/source",
    )
    for record in selected:
        pressure_mpa = record["pressure"].data / 1.0e6
        finite_counts = np.isfinite(pressure_mpa).sum(axis=0)
        mean_pressure = np.full(pressure_mpa.shape[1], np.nan)
        valid_cols = finite_counts > 0
        mean_pressure[valid_cols] = (
            np.nansum(pressure_mpa[:, valid_cols], axis=0) / finite_counts[valid_cols]
        )
        ax.plot(
            record["pressure"].taxis / 3600.0,
            mean_pressure,
            linewidth=1.25,
            label=f"{record['shift_ft']:g} ft mean",
        )
    ax.set_title("Pressure Line-Sampler Means")
    ax.set_xlabel("Simulation time (hours)")
    ax.set_ylabel("Pressure (MPa)")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(ncol=2, fontsize=8)
    path = FIG_DIR / "all_pressure_line_means.png"
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

    figures = []
    for token, title, ylabel, filename in [
        ("dispx", "Point Postprocessor X Displacement", "disp_x (m)", "point_postprocessor_dispx.png"),
        ("dispy", "Point Postprocessor Y Displacement", "disp_y (m)", "point_postprocessor_dispy.png"),
    ]:
        columns = [column for column in point_df.columns if token in column]
        if not columns:
            continue
        fig, ax = plt.subplots(figsize=(11.5, 5.5), constrained_layout=True)
        for column in columns:
            ax.plot(point_df["time"] / 3600.0, point_df[column], linewidth=1.2, label=column)
        ax.set_title(title)
        ax.set_xlabel("Simulation time (hours)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(ncol=2, fontsize=7)
        fig_path = FIG_DIR / filename
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        figures.append(fig_path)

    return [npz_path, *figures]


def write_summary(records, paths):
    summary_path = FIG_DIR / "all_line_postprocessors_summary.txt"
    lines = [
        f"{PROJECT_NAME} all line postprocessor summary",
        f"output_dir: {OUTPUT_DIR}",
        "",
        "Records:",
    ]
    for record in records:
        lines.append(
            f"- {record['sampler_type']}: shift_ft={record['shift_ft']}, sampler={record['sampler_name']}"
        )
        for key, value in record.items():
            if not hasattr(value, "data"):
                continue
            finite = np.isfinite(value.data)
            lines.append(
                f"  {value.name}: shape={value.data.shape}, "
                f"finite={finite.sum()}/{finite.size}, "
                f"range=[{np.nanmin(value.data):.6g}, {np.nanmax(value.data):.6g}]"
            )
    lines.extend(["", "Figures/data:"])
    for path in paths:
        if path is not None:
            lines.append(f"- {path}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    source_pressure = Data1DGauge()
    source_pressure.load_npz(str(SOURCE_PRESSURE_PATH))
    records = all_line_post_processor_info_extractor(output_dir=str(OUTPUT_DIR))

    by_shift = {}
    for record in records:
        record_shift = record["shift_ft"]
        by_shift.setdefault(record_shift, {}).update(record)
        for value in record.values():
            if hasattr(value, "start_time"):
                value.start_time = source_pressure.start_time

    combined_records = []
    paths = []
    for shift_ft in sorted(by_shift):
        record = by_shift[shift_ft]
        output_record = {
            "sampler_name": f"fiber_line_{shift_ft:g}ft",
            "sampler_type": "combined",
            "shift_ft": shift_ft,
        }
        if "pressure" in record:
            output_record["pressure"] = record["pressure"]
            paths.append(save_data2d(record["pressure"], f"pressure_{shift_ft:g}ft.npz"))
        if {"strain_xx", "strain_yy", "strain_xy"}.issubset(record):
            strain_xx = baseline_correct(record["strain_xx"])
            strain_yy = baseline_correct(record["strain_yy"])
            strain_xy = baseline_correct(record["strain_xy"])
            fiber_strain = fiber_aligned_component(strain_xx, strain_yy, strain_xy, f"strain_{shift_ft:g}ft")
            output_record.update({
                "strain_xx": strain_xx,
                "strain_yy": strain_yy,
                "strain_xy": strain_xy,
                "strain_fiber_aligned": fiber_strain,
            })
            for data_obj in [strain_xx, strain_yy, strain_xy, fiber_strain]:
                paths.append(save_data2d(data_obj, f"{data_obj.name}.npz"))
        if {"strain_rate_xx", "strain_rate_yy", "strain_rate_xy"}.issubset(record):
            rate_xx = record["strain_rate_xx"]
            rate_yy = record["strain_rate_yy"]
            rate_xy = record["strain_rate_xy"]
            fiber_rate = fiber_aligned_component(rate_xx, rate_yy, rate_xy, f"strain_rate_{shift_ft:g}ft")
            output_record.update({
                "strain_rate_xx": rate_xx,
                "strain_rate_yy": rate_yy,
                "strain_rate_xy": rate_xy,
                "strain_rate_fiber_aligned": fiber_rate,
            })
            for data_obj in [rate_xx, rate_yy, rate_xy, fiber_rate]:
                paths.append(save_data2d(data_obj, f"{data_obj.name}.npz"))
        combined_records.append(output_record)

    paths.extend([
        plot_pressure_means(combined_records, source_pressure),
        plot_stacked_maps(
            combined_records,
            "strain_fiber_aligned",
            "all_fiber_aligned_strain.png",
            "Fiber-Aligned Strain",
            "strain",
        ),
        plot_stacked_maps(
            combined_records,
            "strain_rate_fiber_aligned",
            "all_fiber_aligned_strain_rate.png",
            "Fiber-Aligned Strain Rate",
            "strain rate (1/s)",
        ),
        plot_stacked_maps(
            combined_records,
            "pressure",
            "all_pressure_maps.png",
            "Pressure",
            "pressure (Pa)",
            cmap="viridis",
            symmetric=False,
        ),
    ])
    paths.extend(export_point_postprocessors())
    summary_path = write_summary(combined_records, paths)

    print(f"Exported {len(combined_records)} fiber line offsets to: {EXPORT_DIR}")
    print(f"Saved all-line QC figures to: {FIG_DIR}")
    print(summary_path)


if __name__ == "__main__":
    main()
