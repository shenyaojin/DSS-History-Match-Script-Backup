"""V2 tensile-SRV MOOSE run: GRADED LOW-PERMEABILITY SRV (kills the pressure plateau).

Why
---
In V1 the SRV permeabilities (srv_narrow 1e-14, srv_wide 1e-15) give a hydraulic
diffusivity D = k*M/mu ~ 1.8e-2 m^2/s, so sqrt(D*t) ~ 100 m over the 7-day window --
far larger than the ~14 m SRV. Pressure therefore equilibrates across the whole SRV and
the modelled tensile lobe comes out FLAT-TOPPED (the "plateau" feature). Dropping the SRV
permeability to ~1e-17 puts sqrt(D*t) at ~10 m, i.e. comparable to the zone thickness, so
a real pressure GRADIENT develops inside the SRV and the lobe becomes peaked.

Changes vs V1 (output/v1_tensile_srv_das):
  * permeabilities lowered and graded outward   1e-16 -> 3e-17 -> 1e-17   (matrix 1e-18)
  * NEW outermost low-perm zone `srv_outer`, 130 ft tall (MD 10308-10438), so the SRV
    reaches and CONTACTS the shear plane at MD ~10319 (the DDM fault2 doublet centre,
    midway between its -0.039@10301 and +0.036@10337 lobes).

Everything else (mesh, driving DAS pressure, fiber sampler) is identical to V1 so the two
runs are directly comparable.  V1 outputs are NOT touched.

Run with --run to launch MOOSE.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "fibeRIS" / "src"))
from fiberis.moose.templates.baseline_model_generator_fervo import build_baseline_model  # noqa: E402
from fiberis.moose.runner import MooseRunner  # noqa: E402

FT = 0.3048
PROJECT = "v2_srv_graded"
T1 = pd.Timestamp("2025-02-24 11:00")
STAR_DEPTH_FT = 10373.4
SHEAR_PLANE_MD = 10319.0          # DDM fault2 doublet centre (see module docstring)

DAS_PRESSURE = REPO / "data_fervo" / "fiberis_format" / "post_processing" / \
    "das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz"
OUT_DIR = REPO / "output" / PROJECT
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREPPED = OUT_DIR / "das_pressure_T1_prepended.npz"
I_FILE = OUT_DIR / f"{PROJECT}_input.i"

# ---- prep: prepend T1=11:00 at the IC so the run starts at T1 (dp=0) ----------
d = np.load(DAS_PRESSURE, allow_pickle=True)
p = np.asarray(d["data"], float)
t = np.asarray(d["taxis"], float)
t0 = pd.Timestamp(str(d["start_time"].item() if hasattr(d["start_time"], "item") else d["start_time"]))
das_times = t0 + pd.to_timedelta(t, unit="s")
ic = float(p[0])
times = pd.DatetimeIndex([T1]).append(das_times)
vals = np.concatenate([[ic], p])
taxis_s = (times - T1).total_seconds().to_numpy()
assert np.all(np.diff(taxis_s) > 0)
np.savez(PREPPED, data=vals, taxis=taxis_s, start_time=str(T1))
print(f"DAS pressure prepped: {len(vals)} steps, {taxis_s[-1]/86400:.1f} days, "
      f"psi {vals.min():.0f}..{vals.max():.0f} (IC={ic:.0f})")

# ---- graded, LOW-perm SRV stack (outermost first; the template sorts by height) ----
#   height_ft   MD span                  perm      role
#   130         10308.4 .. 10438.4       1e-17     NEW outer zone, contacts the shear plane
#    90         10328.4 .. 10418.4       3e-17     mid
#    45         10350.9 .. 10395.9       1e-16     inner
SRV_SPECS = [
    {"name": "srv_outer",  "length_m": 280 * FT, "height_m": 130.0 * FT, "perm": 1e-17, "porosity": 0.08},
    {"name": "srv_wide",   "length_m": 280 * FT, "height_m":  90.0 * FT, "perm": 3e-17, "porosity": 0.10},
    {"name": "srv_narrow", "length_m": 280 * FT, "height_m":  45.0 * FT, "perm": 1e-16, "porosity": 0.12},
]
for s in SRV_SPECS:
    half = s["height_m"] / FT / 2.0
    print(f"  {s['name']:11s} h={s['height_m']/FT:6.1f} ft  MD {STAR_DEPTH_FT-half:.1f}..{STAR_DEPTH_FT+half:.1f}  k={s['perm']:.0e}")
print(f"  shear plane at MD {SHEAR_PLANE_MD:.0f}  ->  inside srv_outer: "
      f"{STAR_DEPTH_FT - 130.0/2 <= SHEAR_PLANE_MD <= STAR_DEPTH_FT + 130.0/2}")

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
