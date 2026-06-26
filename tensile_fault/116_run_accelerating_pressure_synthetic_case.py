"""
Run a synthetic accelerating pressure injection case.

This uses the same timing, geometry, monitor, and post-processing workflow as
the log-pressure case, but replaces the pressure history with a convex-up ramp
so the injection pressure increases faster near the end than near the start.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_114 = Path(__file__).resolve().with_name("114_run_log_pressure_synthetic_case.py")
spec = importlib.util.spec_from_file_location("synthetic_pressure_case", SCRIPT_114)
synthetic_case = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(synthetic_case)


synthetic_case.PROJECT_NAME = "0625_perpendicular_monitor_5sixth_accelerating_pressure_matrix_perm_1e-20"
synthetic_case.MATRIX_PERM = 1e-20
synthetic_case.CURVE_STYLE_NAME = "Accelerating Quadratic"
synthetic_case.CURVE_SHORT_NAME = "accelerating"
synthetic_case.PRESSURE_LINE_LABEL = "Synthetic accelerating injection"
synthetic_case.PLOT_TITLE_PREFIX = "Synthetic Accelerating-Pressure Case"
synthetic_case.SYNTHETIC_PRESSURE_PATH = (
    synthetic_case.REPO_ROOT
    / "data_fervo"
    / "fiberis_format"
    / "post_processing"
    / "synthetic_accelerating_pressure_20250224_1500_to_20250228_0000.npz"
)
synthetic_case.OUTPUT_DIR = synthetic_case.REPO_ROOT / "output" / synthetic_case.PROJECT_NAME
synthetic_case.FIG_DIR = (
    synthetic_case.REPO_ROOT / "figs" / "tensile_fault_qc" / synthetic_case.PROJECT_NAME
)


def create_accelerating_pressure_curve():
    reference = synthetic_case.Data1DGauge()
    reference.load_npz(str(synthetic_case.REFERENCE_PRESSURE_PATH))

    tau = reference.taxis / float(reference.taxis[-1])
    ramp = tau**2.0

    start_psi = 3000.0
    end_psi = 8000.0
    pressure = synthetic_case.Data1DGauge(
        data=start_psi + (end_psi - start_psi) * ramp,
        taxis=reference.taxis.copy(),
        start_time=reference.start_time,
        name="synthetic_accelerating_pressure",
    )
    synthetic_case.SYNTHETIC_PRESSURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pressure.savez(str(synthetic_case.SYNTHETIC_PRESSURE_PATH))
    return pressure


synthetic_case.create_log_pressure_curve = create_accelerating_pressure_curve


if __name__ == "__main__":
    synthetic_case.main()
