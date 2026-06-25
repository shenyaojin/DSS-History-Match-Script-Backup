"""
Run the synthetic log-pressure case with matrix permeability reduced to 1/100.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_114 = Path(__file__).resolve().with_name("114_run_log_pressure_synthetic_case.py")
spec = importlib.util.spec_from_file_location("log_pressure_case", SCRIPT_114)
log_case = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(log_case)


log_case.PROJECT_NAME = "0625_perpendicular_monitor_5sixth_log_pressure_matrix_perm_1e-20"
log_case.MATRIX_PERM = 1e-20
log_case.OUTPUT_DIR = log_case.REPO_ROOT / "output" / log_case.PROJECT_NAME
log_case.FIG_DIR = log_case.REPO_ROOT / "figs" / "tensile_fault_qc" / log_case.PROJECT_NAME


if __name__ == "__main__":
    log_case.main()
