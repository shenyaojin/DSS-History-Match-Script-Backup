"""Prep injection pressure (T1->T2) + generate the tensile-SRV .i for the pressure-scale study.

The averaged Bearskin injection pressure (synthetic_data_simulation.npz) spans exactly
T1 (2025-02-24 11:00) -> T2 (2025-02-28 00:00), which is the PURE-TENSILE stage (shear=0),
so the tensile target is the observed strain directly (no DDM shear needed here).

We feed the injection curve at scale s=1; poroelastic strain is linear in the pressure
perturbation, so a later scale sweep multiplies the modelled strain by s (no re-run).

Generation only unless --run is passed.
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
PROJECT = "v1_tensile_srv_injection"
OBSERVED_WIDTH_FT = 90.0
T1 = pd.Timestamp("2025-02-24 11:00")
T2 = pd.Timestamp("2025-02-28 00:00")
DOWNSAMPLE = "2h"

SRC_PRESSURE = REPO / "data_fervo" / "fiberis_format" / "post_processing" / "synthetic_data_simulation.npz"
OUT_DIR = REPO / "output" / PROJECT
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPO / "figs" / "tensile_fault_qc" / PROJECT
FIG_DIR.mkdir(parents=True, exist_ok=True)
PREPPED_PRESSURE = OUT_DIR / "injection_pressure_T1_T2_2h.npz"
I_FILE = OUT_DIR / f"{PROJECT}_input.i"

# ---- prep pressure: crop T1->T2, downsample to 2h means -----------------------
d = np.load(SRC_PRESSURE, allow_pickle=True)
p = np.asarray(d["data"], float)
t = np.asarray(d["taxis"], float)
t0 = pd.Timestamp(str(d["start_time"].item() if hasattr(d["start_time"], "item") else d["start_time"]))
times = t0 + pd.to_timedelta(t, unit="s")
s = pd.Series(p, index=times)
s = s[(s.index >= T1) & (s.index <= T2)]
# origin="start" aligns 2h bins to T1 so the time axis starts at 0 and is strictly increasing
s2 = s.resample(DOWNSAMPLE, origin="start").mean().dropna()
taxis_s = (s2.index - s2.index[0]).total_seconds().to_numpy()
assert np.all(np.diff(taxis_s) > 0), "pressure taxis must be strictly increasing"
np.savez(PREPPED_PRESSURE, data=s2.to_numpy(float), taxis=taxis_s,
         start_time=str(s2.index[0]))
print(f"Prepped injection pressure: {len(s2)} steps over "
      f"{taxis_s[-1]/3600:.1f} h, psi {s2.min():.0f}..{s2.max():.0f} "
      f"(IC/T1={s2.iloc[0]:.0f} psi)")

# ---- geometry: SRV widened to observed ~90 ft along MD ------------------------
SRV_SPECS = [
    {"name": "srv_wide",   "length_m": 280 * FT, "height_m": OBSERVED_WIDTH_FT * FT,     "perm": 1e-15, "porosity": 0.10},
    {"name": "srv_narrow", "length_m": 280 * FT, "height_m": OBSERVED_WIDTH_FT / 2 * FT, "perm": 1e-14, "porosity": 0.12},
]
builder = build_baseline_model(
    project_name=PROJECT,
    pressure_profile_path=str(PREPPED_PRESSURE),
    model_width=100.0,
    model_length=200.0,
    hf_length_ft=250.0,
    shift_list_ft=np.array([round(250.0 / 3.0, 3)]),
    angle=0.0,
    matrix_perm=1e-18,
    fracture_perm=1e-13,
    srv_specs=SRV_SPECS,
)
builder.plot_geometry(save_path=str(FIG_DIR / "geometry.png"), hide_legend=False, equal_aspect=True)
builder.generate_input_file(output_filepath=str(I_FILE))
print("Generated .i:", I_FILE)

# ---- run MOOSE (only with --run) ---------------------------------------------
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
        ok, _out, err = runner.run(
            input_file_path=str(I_FILE),
            output_directory=str(OUT_DIR),
            num_processors=args.np,
            log_file_name="simulation.log",
            stream_output=True,
            clean_output_dir=False,   # keep the prepped .i/npz; don't rmtree the dir
        )
        print("MOOSE success:", ok)
        if not ok:
            print("stderr tail:", (err or "")[-2000:])
            sys.exit(1)
