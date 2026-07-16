"""V1 tensile-SRV MOOSE run driven by the SMOOTH DAS-derived pressure (T1->T3).

Physical rationale (Pengchao): the formation between the injection well and the fiber
acts as a low-pass filter, so the fracture pressure is smooth -- not the fluctuating
wellhead injection. We use the DAS-derived history-match pressure
(das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz; monotonic, 4056->6077 psi),
prepended with T1=11:00 at the IC so the MOOSE time grid aligns with the 11:00-based
observed / DDM-shear 4h grids. Poroelastic strain is linear in dp, so a scale sweep on
the resulting tensile strain finds the pressure magnitude that matches the profile.

Run with --run to launch MOOSE.
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
PROJECT = "v1_tensile_srv_das"
OBSERVED_WIDTH_FT = 90.0
T1 = pd.Timestamp("2025-02-24 11:00")

DAS_PRESSURE = REPO / "data_fervo" / "fiberis_format" / "post_processing" / \
    "das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz"
OUT_DIR = REPO / "output" / PROJECT
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPO / "figs" / "tensile_fault_qc" / PROJECT
FIG_DIR.mkdir(parents=True, exist_ok=True)
PREPPED = OUT_DIR / "das_pressure_T1_prepended.npz"
I_FILE = OUT_DIR / f"{PROJECT}_input.i"

# ---- prep: prepend T1=11:00 at IC so the run starts at T1 (dp=0) --------------
d = np.load(DAS_PRESSURE, allow_pickle=True)
p = np.asarray(d["data"], float)
t = np.asarray(d["taxis"], float)
t0 = pd.Timestamp(str(d["start_time"].item() if hasattr(d["start_time"], "item") else d["start_time"]))
das_times = t0 + pd.to_timedelta(t, unit="s")
ic = float(p[0])
times = pd.DatetimeIndex([T1]).append(das_times)     # 11:00 + 17:00,21:00,...
vals = np.concatenate([[ic], p])
taxis_s = (times - T1).total_seconds().to_numpy()
assert np.all(np.diff(taxis_s) > 0)
np.savez(PREPPED, data=vals, taxis=taxis_s, start_time=str(T1))
print(f"DAS pressure prepped: {len(vals)} steps, {taxis_s[-1]/86400:.1f} days from T1, "
      f"psi {vals.min():.0f}..{vals.max():.0f} (IC={ic:.0f}), monotonic={np.all(np.diff(vals)>=0)}")

SRV_SPECS = [
    {"name": "srv_wide",   "length_m": 280 * FT, "height_m": OBSERVED_WIDTH_FT * FT,     "perm": 1e-15, "porosity": 0.10},
    {"name": "srv_narrow", "length_m": 280 * FT, "height_m": OBSERVED_WIDTH_FT / 2 * FT, "perm": 1e-14, "porosity": 0.12},
]
builder = build_baseline_model(
    project_name=PROJECT, pressure_profile_path=str(PREPPED),
    model_width=100.0, model_length=200.0, hf_length_ft=250.0,
    shift_list_ft=np.array([round(250.0 / 3.0, 3)]), angle=0.0,
    matrix_perm=1e-18, fracture_perm=1e-13, srv_specs=SRV_SPECS,
)
builder.generate_input_file(output_filepath=str(I_FILE))
print("Generated .i:", I_FILE)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--np", type=int, default=20)
    args = ap.parse_args()
    if args.run:
        runner = MooseRunner(
            moose_executable_path=str(REPO / "moose_env/moose/modules/porous_flow/porous_flow-opt"),
            mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec",
        )
        ok, _o, err = runner.run(input_file_path=str(I_FILE), output_directory=str(OUT_DIR),
                                 num_processors=args.np, log_file_name="simulation.log",
                                 stream_output=True, clean_output_dir=False)
        print("MOOSE success:", ok)
        if not ok:
            print((err or "")[-1500:]); sys.exit(1)
