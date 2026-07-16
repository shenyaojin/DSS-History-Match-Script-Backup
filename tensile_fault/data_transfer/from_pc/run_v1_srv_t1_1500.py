"""V1 tensile-SRV MOOSE run with T1 = 2025-02-24 15:00 (tensile onset).

Same refined geometry as run_v1_tensile_srv_das_refined (graded 4-layer SRV, 400 ft
fracture, fiber 40 ft off centre) and the same smooth DAS-derived pressure, but the
run and reference are anchored at T1=15:00 (the observed tensile onset / DAS-pressure
convention) instead of 11:00. The DAS pressure natively starts at 17:00; we prepend
T1=15:00 at the initial pressure so the run begins at T1 with dp=0.
"""
import argparse, os, sys
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "fibeRIS" / "src"))
from fiberis.moose.templates.baseline_model_generator_fervo import build_baseline_model
from fiberis.moose.runner import MooseRunner

FT = 0.3048
PROJECT = "v1_srv_t1_1500"
T1 = pd.Timestamp("2025-02-24 15:00")           # <-- tensile onset
DAS_PRESSURE = REPO / "data_fervo" / "fiberis_format" / "post_processing" / \
    "das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz"
OUT_DIR = REPO / "output" / PROJECT; OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPO / "figs" / "tensile_fault_qc" / PROJECT; FIG_DIR.mkdir(parents=True, exist_ok=True)
PREPPED = OUT_DIR / "das_pressure_T1_1500_prepended.npz"
I_FILE = OUT_DIR / f"{PROJECT}_input.i"

# prep DAS pressure, prepend T1=15:00 at IC
d = np.load(DAS_PRESSURE, allow_pickle=True)
p = np.asarray(d["data"], float); t = np.asarray(d["taxis"], float)
t0 = pd.Timestamp(str(d["start_time"].item() if hasattr(d["start_time"], "item") else d["start_time"]))
das_times = t0 + pd.to_timedelta(t, unit="s"); ic = float(p[0])
times = pd.DatetimeIndex([T1]).append(das_times); vals = np.concatenate([[ic], p])
taxis_s = (times - T1).total_seconds().to_numpy(); assert np.all(np.diff(taxis_s) > 0)
np.savez(PREPPED, data=vals, taxis=taxis_s, start_time=str(T1))
print(f"pressure: {len(vals)} steps from T1=15:00, {taxis_s[-1]/86400:.1f} days, "
      f"{vals.min():.0f}..{vals.max():.0f} psi (IC={ic:.0f})")

# refined geometry (identical to the T1=11:00 refined run)
SRV_SPECS = [
    {"name": "srv_l1", "length_m": 440*FT, "height_m": 100*FT, "perm": 3e-16, "porosity": 0.08},
    {"name": "srv_l2", "length_m": 440*FT, "height_m": 75*FT,  "perm": 1e-15, "porosity": 0.10},
    {"name": "srv_l3", "length_m": 440*FT, "height_m": 50*FT,  "perm": 3e-15, "porosity": 0.12},
    {"name": "srv_l4", "length_m": 440*FT, "height_m": 25*FT,  "perm": 1e-14, "porosity": 0.14},
]
builder = build_baseline_model(
    project_name=PROJECT, pressure_profile_path=str(PREPPED),
    model_width=100.0, model_length=200.0, hf_length_ft=400.0,
    shift_list_ft=np.array([40.0]), angle=0.0,
    matrix_perm=1e-18, fracture_perm=1e-13, srv_specs=SRV_SPECS)
builder.generate_input_file(output_filepath=str(I_FILE))
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
