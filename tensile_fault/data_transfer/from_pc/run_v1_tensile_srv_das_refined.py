"""V1 tensile-SRV, REFINED geometry (option a): round the tensile lobe + de-tip the fiber.

Changes vs v1_tensile_srv_das:
  - SRV = 4 nested layers with GRADED permeability (smooth transition) instead of 2
    sharp blocks -> rounds the boxy tensile lobe.
  - Fracture lengthened 250 -> 400 ft and fiber moved from 83 ft to 40 ft offset, so the
    perpendicular monitor samples the plane-strain interior, not near the fracture tip.
Same smooth DAS pressure (prepended at T1=IC). Generation + optional --run.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "fibeRIS" / "src"))
from fiberis.moose.templates.baseline_model_generator_fervo import build_baseline_model
from fiberis.moose.runner import MooseRunner

FT = 0.3048
PROJECT = "v1_tensile_srv_das_refined"
T1 = pd.Timestamp("2025-02-24 11:00")
DAS_PRESSURE = REPO / "data_fervo" / "fiberis_format" / "post_processing" / \
    "das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz"
OUT_DIR = REPO / "output" / PROJECT
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPO / "figs" / "tensile_fault_qc" / PROJECT
FIG_DIR.mkdir(parents=True, exist_ok=True)
PREPPED = OUT_DIR / "das_pressure_T1_prepended.npz"
I_FILE = OUT_DIR / f"{PROJECT}_input.i"

# ---- prep smooth DAS pressure (prepend T1=11:00 at IC) -----------------------
d = np.load(DAS_PRESSURE, allow_pickle=True)
p = np.asarray(d["data"], float); t = np.asarray(d["taxis"], float)
t0 = pd.Timestamp(str(d["start_time"].item() if hasattr(d["start_time"], "item") else d["start_time"]))
das_times = t0 + pd.to_timedelta(t, unit="s"); ic = float(p[0])
times = pd.DatetimeIndex([T1]).append(das_times); vals = np.concatenate([[ic], p])
taxis_s = (times - T1).total_seconds().to_numpy(); assert np.all(np.diff(taxis_s) > 0)
np.savez(PREPPED, data=vals, taxis=taxis_s, start_time=str(T1))

# ---- refined geometry -------------------------------------------------------
FRAC_LEN_FT = 400.0
SRV_LEN_FT = 440.0
FIBER_SHIFT_FT = 40.0
# graded nested SRV: widest+lowest-perm outside -> narrowest+highest-perm at centre
SRV_SPECS = [
    {"name": "srv_l1", "length_m": SRV_LEN_FT * FT, "height_m": 100 * FT, "perm": 3e-16, "porosity": 0.08},
    {"name": "srv_l2", "length_m": SRV_LEN_FT * FT, "height_m": 75 * FT,  "perm": 1e-15, "porosity": 0.10},
    {"name": "srv_l3", "length_m": SRV_LEN_FT * FT, "height_m": 50 * FT,  "perm": 3e-15, "porosity": 0.12},
    {"name": "srv_l4", "length_m": SRV_LEN_FT * FT, "height_m": 25 * FT,  "perm": 1e-14, "porosity": 0.14},
]
builder = build_baseline_model(
    project_name=PROJECT, pressure_profile_path=str(PREPPED),
    model_width=100.0, model_length=200.0, hf_length_ft=FRAC_LEN_FT,
    shift_list_ft=np.array([FIBER_SHIFT_FT]), angle=0.0,
    matrix_perm=1e-18, fracture_perm=1e-13, srv_specs=SRV_SPECS,
)
builder.plot_geometry(save_path=str(FIG_DIR / "geometry.png"), hide_legend=False, equal_aspect=True)
builder.generate_input_file(output_filepath=str(I_FILE))
print(f"Refined geometry: frac {FRAC_LEN_FT:.0f} ft, fiber offset {FIBER_SHIFT_FT:.0f} ft, "
      f"graded SRV heights {[s['height_m']/FT for s in SRV_SPECS]} ft")
print("Generated .i:", I_FILE)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", action="store_true"); ap.add_argument("--np", type=int, default=20)
    a = ap.parse_args()
    if a.run:
        runner = MooseRunner(
            moose_executable_path=str(REPO / "moose_env/moose/modules/porous_flow/porous_flow-opt"),
            mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec")
        ok, _o, err = runner.run(input_file_path=str(I_FILE), output_directory=str(OUT_DIR),
                                 num_processors=a.np, log_file_name="simulation.log",
                                 stream_output=True, clean_output_dir=False)
        print("MOOSE success:", ok)
        if not ok:
            print((err or "")[-1500:]); sys.exit(1)
