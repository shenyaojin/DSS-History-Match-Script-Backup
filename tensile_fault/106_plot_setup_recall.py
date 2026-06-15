"""
Plot the remembered MOOSE setup for the tensile-fault work.

This script does not run MOOSE. It rebuilds the Python-side ModelBuilder setup,
then saves:
  1. the injection pressure curve,
  2. the model geometry,
  3. a short text summary of the pressure curve.

Run from the repository root:
    python scripts/tensile_fault/106_plot_setup_recall.py

Useful variants:
    python scripts/tensile_fault/106_plot_setup_recall.py --model baseline
    python scripts/tensile_fault/106_plot_setup_recall.py --model prototype
"""

from __future__ import annotations

import argparse
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
from fiberis.moose.config import (
    HydraulicFractureConfig,
    MatrixConfig,
    SimpleFluidPropertiesConfig,
    ZoneMaterialProperties,
)
from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.templates.baseline_model_generator_fervo import build_baseline_model


PSI_TO_PA = 6894.76
FT_TO_M = 0.3048


def load_baseline_injection_curve() -> Data1DGauge:
    """Load the Fervo/synthetic-fault pressure profile used by build_baseline_model."""
    pressure = Data1DGauge()
    pressure.load_npz(
        REPO_ROOT
        / "data_fervo"
        / "fiberis_format"
        / "post_processing"
        / "synthetic_data_simulation.npz"
    )
    pressure.data = PSI_TO_PA * pressure.data
    pressure.name = "baseline_injection_pressure"
    return pressure


def make_prototype_injection_curve() -> Data1DGauge:
    """Recreate the synthetic pulse train from 101_prototype_tensile_fault.py."""
    pressure = Data1DGauge()
    pressure.taxis = np.linspace(0.0, 400.0, 401)
    pressure.data = np.zeros_like(pressure.taxis, dtype=float)
    pressure.data[(pressure.taxis >= 50) & (pressure.taxis <= 125)] = 5.0e7
    pressure.data[(pressure.taxis >= 185) & (pressure.taxis <= 265)] = 5.0e7
    pressure.data[(pressure.taxis >= 300) & (pressure.taxis <= 350)] = 5.0e7
    pressure.name = "prototype_injection_pressure"
    return pressure


def build_baseline_recall_model() -> ModelBuilder:
    """Match the setup in 104_test_full.py."""
    hf_length_ft = 250
    return build_baseline_model(
        project_name="1203_rotated_monitor_well_recall",
        hf_length_ft=hf_length_ft,
        srv_height_ft=5,
        shift_list_ft=np.array(
            [
                0.8 * hf_length_ft / 2,
                0.9 * hf_length_ft / 2,
                1.1 * hf_length_ft / 2,
                1.3 * hf_length_ft / 2,
            ]
        ),
        angle=30,
        srv_perm=1e-15,
        fracture_perm=1e-13,
        matrix_perm=1e-20,
    )


def build_prototype_recall_model() -> ModelBuilder:
    """Rebuild the geometry/material part of 101_prototype_tensile_fault.py."""
    builder = ModelBuilder(project_name="1111_test_tensile_fault_recall")
    fault_y = 0.0
    domain_bounds = (-200.0 * FT_TO_M, 200.0 * FT_TO_M)
    domain_length = 400.0 * FT_TO_M

    builder.build_stitched_mesh_for_fractures(
        fracture_y_coords=fault_y,
        domain_bounds=domain_bounds,
        domain_length=domain_length,
        nx=200,
        ny_per_layer_half=80,
        bias_y=1.1,
    )

    matrix_perm = 1e-18
    fault_perm_x = 1e-15
    fault_perm_y = 1e-17
    matrix_mats = ZoneMaterialProperties(
        porosity=0.01,
        permeability=f"{matrix_perm} 0 0 0 {matrix_perm} 0 0 0 {matrix_perm}",
    )
    fault_mats = ZoneMaterialProperties(
        porosity=0.15,
        permeability=f"{fault_perm_x} 0 0 0 {fault_perm_y} 0 0 0 {matrix_perm}",
    )
    builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))

    center_x = domain_length / 2.0
    fault = HydraulicFractureConfig(
        name="fault",
        length=200.0 * FT_TO_M,
        height=2.0 * FT_TO_M,
        center_x=center_x,
        center_y=fault_y,
        materials=fault_mats,
    )
    builder.add_fracture_config(fault)
    builder.add_hydraulic_fracture_2d(fault, target_block_id=1)
    builder.add_nodeset_by_coord(
        nodeset_op_name="injection",
        new_boundary_name="injection",
        coordinates=(center_x, fault_y, 0),
    )

    builder.add_variables(
        [
            {"name": "pp", "params": {"initial_condition": 0}},
            {"name": "disp_x", "params": {"initial_condition": 0}},
            {"name": "disp_y", "params": {"initial_condition": 0}},
        ]
    )
    builder.set_porous_flow_dictator(
        dictator_name="dictator", porous_flow_variables="pp"
    )
    builder.add_global_params(
        {"PorousFlowDictator": "dictator", "displacements": "'disp_x disp_y'"}
    )
    builder.add_fluid_properties_config(
        SimpleFluidPropertiesConfig(
            name="water", bulk_modulus=2.2e9, viscosity=1.0e-3, density0=1000.0
        )
    )
    builder.add_poromechanics_materials(
        fluid_properties_name="water",
        biot_coefficient=0.7,
        solid_bulk_compliance=2e-11,
    )
    builder.add_standard_tensor_aux_vars_and_kernels(
        {"stress": "stress", "total_strain": "strain"}
    )
    return builder


def plot_injection_curve(pressure: Data1DGauge, output_path: Path, title: str) -> None:
    time_s = np.asarray(pressure.taxis, dtype=float)
    pressure_pa = np.asarray(pressure.data, dtype=float)
    time_hr = time_s / 3600.0
    pressure_mpa = pressure_pa / 1e6
    pressure_psi = pressure_pa / PSI_TO_PA

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.plot(time_hr, pressure_mpa, color="tab:blue", linewidth=1.8)
    ax1.set_xlabel("Time since simulation start (hours)")
    ax1.set_ylabel("Pressure (MPa)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(time_hr, pressure_psi, color="tab:orange", alpha=0.0)
    ax2.set_ylabel("Pressure (psi)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    max_pressure_mpa = float(np.nanmax(pressure_mpa))
    max_pressure_psi = float(np.nanmax(pressure_psi))
    duration_hr = float((time_s[-1] - time_s[0]) / 3600.0)
    ax1.set_title(
        f"{title}\nmax={max_pressure_mpa:.2f} MPa ({max_pressure_psi:.0f} psi), "
        f"duration={duration_hr:.2f} hr"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_injection_summary(
    pressure: Data1DGauge, output_path: Path, model_name: str
) -> None:
    time_s = np.asarray(pressure.taxis, dtype=float)
    pressure_pa = np.asarray(pressure.data, dtype=float)
    active = pressure_pa > 0
    total_duration_s = float(time_s[-1] - time_s[0])
    active_duration_s = 0.0
    if len(time_s) > 1:
        dt = np.diff(time_s)
        active_duration_s = float(np.sum(dt[active[:-1]]))

    lines = [
        f"model: {model_name}",
        f"number_of_points: {len(time_s)}",
        f"time_start_s: {time_s[0]:.6g}",
        f"time_end_s: {time_s[-1]:.6g}",
        f"duration_s: {total_duration_s:.6g}",
        f"duration_hr: {total_duration_s / 3600.0:.6g}",
        f"pressure_min_pa: {np.nanmin(pressure_pa):.6g}",
        f"pressure_max_pa: {np.nanmax(pressure_pa):.6g}",
        f"pressure_min_psi: {np.nanmin(pressure_pa) / PSI_TO_PA:.6g}",
        f"pressure_max_psi: {np.nanmax(pressure_pa) / PSI_TO_PA:.6g}",
        f"active_pressure_duration_s: {active_duration_s:.6g}",
        f"active_pressure_duration_hr: {active_duration_s / 3600.0:.6g}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save injection curve and geometry plots for tensile_fault MOOSE setup."
    )
    parser.add_argument(
        "--model",
        choices=("baseline", "prototype"),
        default="baseline",
        help="baseline matches 104_test_full.py; prototype matches 101_prototype_tensile_fault.py.",
    )
    parser.add_argument(
        "--fig-dir",
        default=str(REPO_ROOT / "figs" / "tensile_fault_recall"),
        help="Directory where figures and summary text will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "baseline":
        pressure = load_baseline_injection_curve()
        builder = build_baseline_recall_model()
        title = "Injection pressure used by baseline/Fervo synthetic-fault setup"
    else:
        pressure = make_prototype_injection_curve()
        builder = build_prototype_recall_model()
        title = "Synthetic pulse injection pressure used by prototype tensile-fault setup"

    injection_plot = fig_dir / f"{args.model}_injection_curve.png"
    geometry_plot = fig_dir / f"{args.model}_model_geometry.png"
    summary_path = fig_dir / f"{args.model}_injection_curve_summary.txt"

    plot_injection_curve(pressure, injection_plot, title)
    builder.plot_geometry(
        save_path=str(geometry_plot),
        hide_legend=True,
        equal_aspect=True,
    )
    plt.close("all")
    write_injection_summary(pressure, summary_path, args.model)

    print(f"Saved injection curve: {injection_plot}")
    print(f"Saved model geometry:  {geometry_plot}")
    print(f"Saved curve summary:   {summary_path}")


if __name__ == "__main__":
    main()
