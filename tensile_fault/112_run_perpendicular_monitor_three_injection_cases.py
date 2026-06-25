"""
Run/export the three perpendicular-monitor injection cases:

1. Cropped averaged injection curve.
2. Mean sampled pressure from case 1 used as injection.
3. Max sampled pressure from case 1 used as injection.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FIBERIS_SRC = REPO_ROOT / "fibeRIS" / "src"
if str(FIBERIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIBERIS_SRC))

from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.moose.templates.baseline_model_generator_fervo import (
    all_line_post_processor_info_extractor,
    build_baseline_model,
)


SCRIPT_111 = REPO_ROOT / "scripts" / "tensile_fault" / "111_run_perpendicular_monitor_5sixth.py"
spec = importlib.util.spec_from_file_location("perpendicular_case", SCRIPT_111)
case = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(case)


def set_case_paths(project_name: str, pressure_path: Path) -> None:
    case.PROJECT_NAME = project_name
    case.CROPPED_PRESSURE_PATH = pressure_path
    case.OUTPUT_DIR = case.REPO_ROOT / "output" / project_name
    case.EXPORT_DIR = case.OUTPUT_DIR / "postprocessor_npz"
    case.FIG_DIR = case.REPO_ROOT / "figs" / "tensile_fault_qc" / project_name


def load_pressure_curve(path: Path) -> Data1DGauge:
    pressure = Data1DGauge()
    pressure.load_npz(str(path))
    return pressure


def save_geometry(project_name: str, pressure_path: Path) -> Path:
    builder = build_baseline_model(
        project_name=project_name,
        pressure_profile_path=str(pressure_path),
        hf_length_ft=case.HF_LENGTH_FT,
        shift_list_ft=np.array([case.MONITOR_SHIFT_FT]),
        angle=case.MONITOR_ANGLE_DEG,
    )
    case.FIG_DIR.mkdir(parents=True, exist_ok=True)
    geometry_path = case.FIG_DIR / "geometry.png"
    builder.plot_geometry(save_path=str(geometry_path), hide_legend=True, equal_aspect=True)
    return geometry_path


def pressure_record_from_average_case():
    records = case.combine_records(
        all_line_post_processor_info_extractor(output_dir=str(case.OUTPUT_DIR))
    )
    if len(records) != 1 or "pressure" not in records[0]:
        raise RuntimeError("Could not find exactly one pressure line sampler in the averaged case.")
    return records[0]["pressure"]


def sampled_pressure_curves_from_average_case(source_pressure: Data1DGauge) -> tuple[Path, Path]:
    pressure = pressure_record_from_average_case()
    pressure.start_time = source_pressure.start_time
    pressure_pa = pressure.data
    finite_counts = np.isfinite(pressure_pa).sum(axis=0)
    valid_cols = finite_counts > 0

    mean_pa = np.full(pressure_pa.shape[1], np.nan)
    max_pa = np.full(pressure_pa.shape[1], np.nan)
    mean_pa[valid_cols] = np.nansum(pressure_pa[:, valid_cols], axis=0) / finite_counts[valid_cols]
    max_pa[valid_cols] = np.nanmax(pressure_pa[:, valid_cols], axis=0)

    if np.any(~np.isfinite(mean_pa)) or np.any(~np.isfinite(max_pa)):
        good = np.isfinite(mean_pa)
        mean_pa = np.interp(pressure.taxis, pressure.taxis[good], mean_pa[good])
        good = np.isfinite(max_pa)
        max_pa = np.interp(pressure.taxis, pressure.taxis[good], max_pa[good])

    mean_curve = Data1DGauge(
        data=mean_pa / case.PSI_TO_PA,
        taxis=pressure.taxis.copy(),
        start_time=source_pressure.start_time,
        name="mean_sampled_pressure_injection",
    )
    max_curve = Data1DGauge(
        data=max_pa / case.PSI_TO_PA,
        taxis=pressure.taxis.copy(),
        start_time=source_pressure.start_time,
        name="max_sampled_pressure_injection",
    )

    mean_path = (
        case.REPO_ROOT
        / "data_fervo"
        / "fiberis_format"
        / "post_processing"
        / "synthetic_data_simulation_20250224_1500_to_20250228_0000_mean_sampled_pressure.npz"
    )
    max_path = (
        case.REPO_ROOT
        / "data_fervo"
        / "fiberis_format"
        / "post_processing"
        / "synthetic_data_simulation_20250224_1500_to_20250228_0000_max_sampled_pressure.npz"
    )
    mean_curve.savez(str(mean_path))
    max_curve.savez(str(max_path))
    return mean_path, max_path


def ensure_average_case() -> Data1DGauge:
    set_case_paths("0625_perpendicular_monitor_5sixth", case.CROPPED_PRESSURE_PATH)
    source_pressure = case.crop_pressure_curve()
    save_geometry(case.PROJECT_NAME, case.CROPPED_PRESSURE_PATH)
    if not case.OUTPUT_DIR.exists() or not any(case.OUTPUT_DIR.glob("*fiber_pressure_sampler*0000.csv")):
        case.build_and_run_model()
    case.export_results(source_pressure)
    return source_pressure


def run_case(project_name: str, pressure_path: Path) -> None:
    set_case_paths(project_name, pressure_path)
    source_pressure = load_pressure_curve(pressure_path)
    save_geometry(project_name, pressure_path)
    case.build_and_run_model()
    case.export_results(source_pressure)


def main() -> None:
    averaged_pressure = ensure_average_case()
    mean_path, max_path = sampled_pressure_curves_from_average_case(averaged_pressure)

    run_case("0625_perpendicular_monitor_5sixth_mean_sampled_injection", mean_path)
    run_case("0625_perpendicular_monitor_5sixth_max_sampled_injection", max_path)

    print("Finished three perpendicular-monitor injection cases.")
    for project_name in [
        "0625_perpendicular_monitor_5sixth",
        "0625_perpendicular_monitor_5sixth_mean_sampled_injection",
        "0625_perpendicular_monitor_5sixth_max_sampled_injection",
    ]:
        print(case.REPO_ROOT / "figs" / "tensile_fault_qc" / project_name)


if __name__ == "__main__":
    main()
