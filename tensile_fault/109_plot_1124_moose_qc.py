"""
Generate QC plots from the completed 1124_misfit_func MOOSE output.

This script does not run MOOSE. It reads the vector postprocessor CSV files in
output/1124_misfit_func and saves figures under figs/tensile_fault_qc/1124_misfit_func.
"""

from __future__ import annotations

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
from fiberis.analyzer.TensorProcessor.coreT2D import Tensor2D
from fiberis.io.reader_moose_tensor_from_data2d import MOOSETensorFromData2D
from fiberis.moose.templates.baseline_model_generator_fervo import (
    post_processor_info_extractor,
    strain_rate_info_extractor,
)


OUTPUT_DIR = REPO_ROOT / "output" / "1124_misfit_func"
FIG_DIR = REPO_ROOT / "figs" / "tensile_fault_qc" / "1124_misfit_func"
SOURCE_PRESSURE_PATH = (
    REPO_ROOT
    / "data_fervo"
    / "fiberis_format"
    / "post_processing"
    / "synthetic_data_simulation.npz"
)
PSI_TO_PA = 6894.76
FIBER_ANGLE_FROM_Y_DEG = 30.0


def set_sim_start_time(dataframes, start_time):
    for dataframe in dataframes:
        dataframe.start_time = start_time


def baseline_correct(dataframe):
    data = dataframe.data.copy()
    finite_mask = np.isfinite(data)
    valid_rows = finite_mask.any(axis=1)
    baseline = np.full(data.shape[0], np.nan)
    first_finite_cols = np.argmax(finite_mask[valid_rows], axis=1)
    baseline[valid_rows] = data[valid_rows, first_finite_cols]
    data[valid_rows, :] -= baseline[valid_rows, np.newaxis]
    dataframe.data = data
    return dataframe


def clone_like(dataframe, data, name):
    new_dataframe = dataframe.copy()
    new_dataframe.data = data
    new_dataframe.name = name
    return new_dataframe


def compute_strain_rate_from_strain(strain_xx, strain_yy, strain_xy):
    strain_rates = []
    for dataframe, name in [
        (strain_xx, "strain_rate_xx"),
        (strain_yy, "strain_rate_yy"),
        (strain_xy, "strain_rate_xy"),
    ]:
        rate_data = np.gradient(dataframe.data, dataframe.taxis, axis=1)
        strain_rates.append(clone_like(dataframe, rate_data, name))
    return strain_rates


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


def load_or_compute_strain_rate(output_dir, strain_xx, strain_yy, strain_xy, start_time):
    try:
        strain_rate_xx, strain_rate_yy, strain_rate_xy = strain_rate_info_extractor(
            output_dir=str(output_dir)
        )
        set_sim_start_time([strain_rate_xx, strain_rate_yy, strain_rate_xy], start_time)
        return strain_rate_xx, strain_rate_yy, strain_rate_xy, "MOOSE TimeDerivativeAux output"
    except FileNotFoundError:
        strain_rates = compute_strain_rate_from_strain(strain_xx, strain_yy, strain_xy)
        return (*strain_rates, "computed from strain gradient; rerun MOOSE for direct postprocessor output")


def finite_limits(data, percentile=99.0, symmetric=False):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return (-1.0, 1.0)
    if symmetric:
        value = np.nanpercentile(np.abs(finite), percentile)
        if value == 0:
            value = 1.0
        return (-value, value)
    return (
        np.nanpercentile(finite, 100.0 - percentile),
        np.nanpercentile(finite, percentile),
    )


def plot_data2d(ax, dataframe, data, title, cmap, clim, clabel):
    time_hours = dataframe.taxis / 3600.0
    mesh = ax.pcolormesh(
        time_hours,
        dataframe.daxis,
        data,
        shading="auto",
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
    )
    ax.set_title(title)
    ax.set_xlabel("Simulation time (hours)")
    ax.set_ylabel("Position along sampler (m)")
    ax.grid(True, linestyle=":", alpha=0.2)
    cbar = ax.figure.colorbar(mesh, ax=ax)
    cbar.set_label(clabel)
    return mesh


def save_strain_components(strain_xx, strain_yy, strain_xy):
    components = [
        (strain_xx, "Strain XX"),
        (strain_yy, "Strain YY"),
        (strain_xy, "Strain XY"),
    ]
    max_abs = max(
        finite_limits(component.data, percentile=99.0, symmetric=True)[1]
        for component, _ in components
    )
    clim = (-max_abs, max_abs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    for ax, (component, title) in zip(axes, components):
        plot_data2d(ax, component, component.data, title, "bwr", clim, "strain")
    fig.suptitle("Simulated Strain Components, Baseline Corrected")
    path = FIG_DIR / "strain_components.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_strain_rate_components(strain_rate_xx, strain_rate_yy, strain_rate_xy, source_label):
    components = [
        (strain_rate_xx, "Strain Rate XX"),
        (strain_rate_yy, "Strain Rate YY"),
        (strain_rate_xy, "Strain Rate XY"),
    ]
    max_abs = max(
        finite_limits(component.data, percentile=99.0, symmetric=True)[1]
        for component, _ in components
    )
    clim = (-max_abs, max_abs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    for ax, (component, title) in zip(axes, components):
        plot_data2d(ax, component, component.data, title, "bwr", clim, "strain rate (1/s)")
    fig.suptitle(f"Simulated Strain-Rate Components ({source_label})")
    path = FIG_DIR / "strain_rate_components.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_fiber_aligned_map(dataframe, filename, title, clabel):
    clim = finite_limits(dataframe.data, percentile=99.0, symmetric=True)
    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    plot_data2d(ax, dataframe, dataframe.data, title, "bwr", clim, clabel)
    path = FIG_DIR / filename
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_pressure_map(pressure_dataframe):
    pressure_mpa = pressure_dataframe.data / 1.0e6
    clim = finite_limits(pressure_mpa, percentile=99.0, symmetric=False)
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    plot_data2d(
        ax,
        pressure_dataframe,
        pressure_mpa,
        "Simulated Pressure Along Fiber Sampler",
        "viridis",
        clim,
        "pressure (MPa)",
    )
    path = FIG_DIR / "pressure_sampler_map.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_pressure_source_comparison(pressure_dataframe, source_pressure):
    fig, ax = plt.subplots(figsize=(11.5, 5.5), constrained_layout=True)
    sim_time_hours = pressure_dataframe.taxis / 3600.0
    source_time_hours = source_pressure.taxis / 3600.0

    pressure_mpa = pressure_dataframe.data / 1.0e6
    ax.plot(
        source_time_hours,
        source_pressure.data * PSI_TO_PA / 1.0e6,
        color="black",
        linewidth=2.0,
        label="Injection/source pressure",
    )
    finite_counts = np.sum(np.isfinite(pressure_mpa), axis=0)
    mean_pressure = np.full(pressure_mpa.shape[1], np.nan)
    max_pressure = np.full(pressure_mpa.shape[1], np.nan)
    valid_cols = finite_counts > 0
    mean_pressure[valid_cols] = (
        np.nansum(pressure_mpa[:, valid_cols], axis=0) / finite_counts[valid_cols]
    )
    max_pressure[valid_cols] = np.nanmax(pressure_mpa[:, valid_cols], axis=0)
    ax.plot(
        sim_time_hours,
        mean_pressure,
        color="#1f77b4",
        linewidth=1.6,
        label="Mean sampled pressure",
    )
    ax.plot(
        sim_time_hours,
        max_pressure,
        color="#d62728",
        linewidth=1.2,
        alpha=0.8,
        label="Max sampled pressure",
    )
    ax.set_title("Injection Curve vs Sampled Model Pressure")
    ax.set_xlabel("Simulation time (hours)")
    ax.set_ylabel("Pressure (MPa)")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(frameon=True)
    path = FIG_DIR / "pressure_source_comparison.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_centerline_traces(strain_xx, strain_yy, strain_xy):
    center_idx = len(strain_xx.daxis) // 2
    time_hours = strain_xx.taxis / 3600.0
    fig, ax = plt.subplots(figsize=(11.5, 5.5), constrained_layout=True)
    for dataframe, label in [
        (strain_xx, "strain_xx"),
        (strain_yy, "strain_yy"),
        (strain_xy, "strain_xy"),
    ]:
        ax.plot(time_hours, dataframe.data[center_idx, :], linewidth=1.5, label=label)
    ax.set_title(f"Center Sampler Strain Traces, position={strain_xx.daxis[center_idx]:.2f} m")
    ax.set_xlabel("Simulation time (hours)")
    ax.set_ylabel("Strain")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(frameon=True)
    path = FIG_DIR / "center_strain_traces.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_centerline_strain_rate_traces(strain_rate_xx, strain_rate_yy, strain_rate_xy):
    center_idx = len(strain_rate_xx.daxis) // 2
    time_hours = strain_rate_xx.taxis / 3600.0
    fig, ax = plt.subplots(figsize=(11.5, 5.5), constrained_layout=True)
    for dataframe, label in [
        (strain_rate_xx, "strain_rate_xx"),
        (strain_rate_yy, "strain_rate_yy"),
        (strain_rate_xy, "strain_rate_xy"),
    ]:
        ax.plot(time_hours, dataframe.data[center_idx, :], linewidth=1.5, label=label)
    ax.set_title(f"Center Sampler Strain-Rate Traces, position={strain_rate_xx.daxis[center_idx]:.2f} m")
    ax.set_xlabel("Simulation time (hours)")
    ax.set_ylabel("Strain rate (1/s)")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(frameon=True)
    path = FIG_DIR / "center_strain_rate_traces.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_tensor_qc(strain_xx, strain_yy, strain_xy):
    tensor_reader = MOOSETensorFromData2D()
    tensor_reader.read(strain_xx, strain_yy, strain_xy)
    tensor_list = tensor_reader.to_analyzer()
    center_tensor = tensor_list[len(tensor_list) // 2]

    finite_time_mask = np.all(np.isfinite(center_tensor.data), axis=(0, 1))
    if not np.any(finite_time_mask):
        return None

    tensor_data = center_tensor.data[:, :, finite_time_mask]
    time_hours = center_tensor.taxis[finite_time_mask] / 3600.0
    principal_strains = np.zeros((2, len(time_hours)))
    orientations = np.zeros(len(time_hours))

    for i in range(len(time_hours)):
        eigenvalues, eigenvectors = np.linalg.eig(tensor_data[:, :, i])
        principal_strains[:, i] = eigenvalues
        orientations[i] = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 8), sharex=True, constrained_layout=True)
    ax1.plot(time_hours, principal_strains[0, :], label="principal strain 1")
    ax1.plot(time_hours, principal_strains[1, :], label="principal strain 2")
    ax1.set_title("Center Tensor Principal Strains")
    ax1.set_ylabel("Strain")
    ax1.grid(True, linestyle=":", alpha=0.35)
    ax1.legend(frameon=True)

    ax2.plot(time_hours, orientations, color="#2ca02c", label="orientation")
    ax2.set_xlabel("Simulation time (hours)")
    ax2.set_ylabel("Orientation (deg)")
    ax2.grid(True, linestyle=":", alpha=0.35)
    ax2.legend(frameon=True)

    path = FIG_DIR / "center_tensor_principal_strains.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def write_summary(paths, dataframes):
    summary_path = FIG_DIR / "qc_summary.txt"
    lines = ["1124_misfit_func QC summary", ""]
    for dataframe in dataframes:
        finite = np.isfinite(dataframe.data)
        lines.append(
            f"{dataframe.name}: shape={dataframe.data.shape}, "
            f"finite={finite.sum()}/{finite.size}, "
            f"range=[{np.nanmin(dataframe.data):.6g}, {np.nanmax(dataframe.data):.6g}]"
        )
    lines.append("")
    lines.append("Figures:")
    for path in paths:
        if path is not None:
            lines.append(f"- {path}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pressure_dataframe, strain_xx, strain_yy, strain_xy = post_processor_info_extractor(
        output_dir=str(OUTPUT_DIR)
    )
    source_pressure = Data1DGauge()
    source_pressure.load_npz(str(SOURCE_PRESSURE_PATH))

    set_sim_start_time(
        [pressure_dataframe, strain_xx, strain_yy, strain_xy],
        source_pressure.start_time,
    )
    strain_xx = baseline_correct(strain_xx)
    strain_yy = baseline_correct(strain_yy)
    strain_xy = baseline_correct(strain_xy)
    strain_rate_xx, strain_rate_yy, strain_rate_xy, strain_rate_source = load_or_compute_strain_rate(
        OUTPUT_DIR,
        strain_xx,
        strain_yy,
        strain_xy,
        source_pressure.start_time,
    )
    fiber_strain = fiber_aligned_component(strain_xx, strain_yy, strain_xy, "strain")
    fiber_strain_rate = fiber_aligned_component(
        strain_rate_xx,
        strain_rate_yy,
        strain_rate_xy,
        "strain_rate",
    )

    paths = [
        save_strain_components(strain_xx, strain_yy, strain_xy),
        save_strain_rate_components(strain_rate_xx, strain_rate_yy, strain_rate_xy, strain_rate_source),
        save_fiber_aligned_map(
            fiber_strain,
            "fiber_aligned_strain.png",
            f"Fiber-Aligned Strain ({FIBER_ANGLE_FROM_Y_DEG:g} deg clockwise from +y)",
            "strain",
        ),
        save_fiber_aligned_map(
            fiber_strain_rate,
            "fiber_aligned_strain_rate.png",
            f"Fiber-Aligned Strain Rate ({FIBER_ANGLE_FROM_Y_DEG:g} deg clockwise from +y)",
            "strain rate (1/s)",
        ),
        save_pressure_map(pressure_dataframe),
        save_pressure_source_comparison(pressure_dataframe, source_pressure),
        save_centerline_traces(strain_xx, strain_yy, strain_xy),
        save_centerline_strain_rate_traces(strain_rate_xx, strain_rate_yy, strain_rate_xy),
        save_tensor_qc(strain_xx, strain_yy, strain_xy),
    ]
    summary_path = write_summary(
        paths,
        [
            pressure_dataframe,
            strain_xx,
            strain_yy,
            strain_xy,
            fiber_strain,
            strain_rate_xx,
            strain_rate_yy,
            strain_rate_xy,
            fiber_strain_rate,
        ],
    )

    print(f"Saved QC figures to: {FIG_DIR}")
    for path in paths:
        if path is not None:
            print(path)
    print(summary_path)


if __name__ == "__main__":
    main()
