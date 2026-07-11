"""
Run the tensile-fault model on an enlarged geometry with two nested SRV zones,
using the calibrated DAS-derived injection pressure (C*=1.63e7 psi/strain), and
compare the modeled monitor strain response against the observed DAS.

Geometry changes vs the baseline:
  - Square 200 m x 200 m domain (MOOSE units are meters).
  - Fracture at the domain center (250 ft long, as before).
  - Two nested SRV zones centered on the fracture:
      * wide   : 20 m tall, lower  perm (1e-15)
      * narrow : 10 m tall, higher perm (1e-14)
  - Postprocessor line kept at 5/6 of the fracture length (perpendicular monitor),
    identical to the calibration run.

The injection curve is the calibrated history-match pressure. Because we only
enlarged the domain and refined the SRV structure, the strain response is
expected to stay close to the calibration case.

Run from the repository root:
    python scripts/tensile_fault/121_run_new_geometry_two_srv.py
"""

from __future__ import annotations

import datetime as dt
import importlib.util
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
from fiberis.moose.runner import MooseRunner
from fiberis.moose.templates.baseline_model_generator_fervo import build_baseline_model


# reuse the 111 case (build/export machinery) and 119 helpers (observed DAS loader)
SCRIPT_119 = REPO_ROOT / "scripts" / "tensile_fault" / "119_run_das_pressure_case.py"
spec = importlib.util.spec_from_file_location("das_pressure_case", SCRIPT_119)
dc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(dc)
case = dc.case  # the 111 module instance owned by 119


PROJECT_NAME = "0711_new_geometry_two_srv"
FT = 0.3048
NUM_PROCESSORS = 20
CUTOFF = dt.datetime(2025, 2, 28, 0, 0, 0)

# --- new geometry ---
MODEL_WIDTH_M = 100.0        # y half-extent -> 200 m total (fiber direction)
MODEL_LENGTH_M = 200.0       # x-extent -> 200 m (fracture direction)
HF_LENGTH_FT = 250.0
MONITOR_SHIFT_FT = round(HF_LENGTH_FT / 3.0, 3)   # 5/6 of fracture length
MONITOR_ANGLE_DEG = 0.0
MATRIX_PERM = 1e-18
FRACTURE_PERM = 1e-13
SRV_LENGTH_M = 280.0 * FT     # keep the baseline SRV length (85.3 m)
SRV_SPECS = [
    {"name": "srv_wide", "length_m": SRV_LENGTH_M, "height_m": 20.0, "perm": 1e-15, "porosity": 0.10},
    {"name": "srv_narrow", "length_m": SRV_LENGTH_M, "height_m": 10.0, "perm": 1e-14, "porosity": 0.12},
]

PRESSURE_NPZ = (
    REPO_ROOT / "data_fervo" / "fiberis_format" / "post_processing"
    / "das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz"
)

OUTPUT_DIR = REPO_ROOT / "output" / PROJECT_NAME
EXPORT_DIR = OUTPUT_DIR / "postprocessor_npz"
FIG_DIR = REPO_ROOT / "figs" / "tensile_fault_qc" / PROJECT_NAME


def build_and_run():
    builder = build_baseline_model(
        project_name=PROJECT_NAME,
        pressure_profile_path=str(PRESSURE_NPZ),
        model_width=MODEL_WIDTH_M,
        model_length=MODEL_LENGTH_M,
        hf_length_ft=HF_LENGTH_FT,
        shift_list_ft=np.array([MONITOR_SHIFT_FT]),
        angle=MONITOR_ANGLE_DEG,
        matrix_perm=MATRIX_PERM,
        fracture_perm=FRACTURE_PERM,
        srv_specs=SRV_SPECS,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    builder.plot_geometry(save_path=str(FIG_DIR / "geometry.png"), hide_legend=False, equal_aspect=True)

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


def score_vs_das(depth_star, das_times, das_rate_nano):
    d = np.load(EXPORT_DIR / "monitor_normal_strain_rate_no_rotation.npz", allow_pickle=True)
    m_taxis = d["taxis"].astype(float)
    m_daxis = d["daxis"].astype(float)
    m_data = d["data"].astype(float)

    ipos = int(np.nanargmax(np.nanmax(np.abs(m_data), axis=1)))
    model_rate_nano = m_data[ipos, :] * 1e9
    fin = np.isfinite(model_rate_nano)
    das_rel = np.array([(t - das_times[0]).total_seconds() for t in das_times])
    model_on_das = np.interp(das_rel, m_taxis[fin], model_rate_nano[fin])

    pre = das_times < CUTOFF
    pair = pre & np.isfinite(das_rate_nano) & np.isfinite(model_on_das)
    o, m = das_rate_nano[pair], model_on_das[pair]
    rms = float(np.sqrt(np.mean((m - o) ** 2)))
    scale = float(np.dot(m, o) / np.dot(m, m)) if np.dot(m, m) > 0 else np.nan
    denom = np.sqrt(np.mean((o - np.mean(o)) ** 2)) or np.nan
    nrms = rms / denom if denom else np.nan
    corr = float(np.corrcoef(m, o)[0, 1]) if m.size > 1 else np.nan

    fig, ax = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    ax.plot(das_times, das_rate_nano, color="k", lw=1.8, label=f"Observed DAS @ {depth_star:.0f} ft")
    ax.plot(das_times, model_on_das, color="tab:red", lw=1.8,
            label=f"Model monitor (pos={m_daxis[ipos]:.1f} m)")
    ax.axvline(CUTOFF, color="0.4", ls="--", lw=1.1, label="2/28 (HM boundary)")
    ax.set_title(f"{PROJECT_NAME}: model vs observed strain-rate "
                 f"(pre-2/28 RMS={rms:.3f}, NRMS={nrms:.2f}, scale={scale:.3f}, r={corr:.2f})")
    ax.set_ylabel("Strain rate (nanostrain/s)")
    ax.set_xlabel("Datetime")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.savefig(FIG_DIR / "das_vs_model_strain_rate.png", dpi=150)
    plt.close(fig)
    return {"rms": rms, "nrms": nrms, "scale": scale, "corr": corr}


def main():
    # point the 111 export machinery at this project
    case.PROJECT_NAME = PROJECT_NAME
    case.OUTPUT_DIR = OUTPUT_DIR
    case.EXPORT_DIR = EXPORT_DIR
    case.FIG_DIR = FIG_DIR
    case.SOURCE_PRESSURE_PATH = PRESSURE_NPZ
    case.CROPPED_PRESSURE_PATH = PRESSURE_NPZ

    pressure = Data1DGauge()
    pressure.load_npz(str(PRESSURE_NPZ))

    build_and_run()
    case.export_results(pressure)

    depth_star, das_times, _ts, das_rate_nano, _S = dc.load_star_das()
    score = score_vs_das(depth_star, das_times, das_rate_nano)
    print(f"[{PROJECT_NAME}] pre-2/28  RMS={score['rms']:.4f}  NRMS={score['nrms']:.3f}  "
          f"scale={score['scale']:.3f}  r={score['corr']:.3f}")
    print(f"[{PROJECT_NAME}] figures in {FIG_DIR}")
    return score


if __name__ == "__main__":
    main()
