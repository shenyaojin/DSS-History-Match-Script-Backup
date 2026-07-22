"""V3 tensile-SRV MOOSE run: ASYMMETRIC, graded low-permeability SRV.

V2 kept the SRV symmetric about the fracture (MD 10373.4), so extending it upward to reach
the shear plane also extended it downward to MD 10438 — which over-produced the deep
poroelastic COMPRESSION lobe (the modelled strain went negative below ~10430 where the
observation is still slightly positive).

V3 drops the symmetry requirement: the SRV reaches UP to the shear plane but stops at
MD ~10400 below the fracture. Zones are specified directly by (top_md, bottom_md).

  zone         MD span            k        note
  srv_outer    10308 .. 10400     1e-17    top contacts the DDM shear plane (MD ~10319)
  srv_wide     10328 .. 10396     3e-17
  srv_narrow   10351 .. 10390     1e-16
  hf           10373.4            1e-13    (matrix 1e-18)

Permeabilities are the V2 graded values (which removed the pressure plateau); only the
geometry changes. V1/V2 outputs are NOT touched.

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
PROJECT = "v3_srv_asym"
T1 = pd.Timestamp("2025-02-24 11:00")
STAR = 10373.4                       # MD at model y = 0
SHEAR_PLANE_MD = 10319.0

DAS_PRESSURE = REPO / "data_fervo" / "fiberis_format" / "post_processing" / \
    "das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz"
OUT_DIR = REPO / "output" / PROJECT
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREPPED = OUT_DIR / "das_pressure_T1_prepended.npz"
I_FILE = OUT_DIR / f"{PROJECT}_input.i"

# ---- prep driving pressure (identical to V1/V2) ------------------------------
d = np.load(DAS_PRESSURE, allow_pickle=True)
p = np.asarray(d["data"], float)
t = np.asarray(d["taxis"], float)
t0 = pd.Timestamp(str(d["start_time"].item() if hasattr(d["start_time"], "item") else d["start_time"]))
ic = float(p[0])
times = pd.DatetimeIndex([T1]).append(t0 + pd.to_timedelta(t, unit="s"))
vals = np.concatenate([[ic], p])
taxis_s = (times - T1).total_seconds().to_numpy()
assert np.all(np.diff(taxis_s) > 0)
np.savez(PREPPED, data=vals, taxis=taxis_s, start_time=str(T1))
print(f"DAS pressure prepped: {len(vals)} steps, {taxis_s[-1]/86400:.1f} days, IC={ic:.0f} psi")


def zone(name, top_md, bot_md, perm, porosity):
    """Build an SRV spec from an ASYMMETRIC MD span (top = shallower, bot = deeper)."""
    centre_md = 0.5 * (top_md + bot_md)
    return {"name": name, "length_m": 280 * FT,
            "height_m": (bot_md - top_md) * FT,
            "center_y": (centre_md - STAR) * FT,          # per-zone centre -> asymmetric
            "perm": perm, "porosity": porosity}


SRV_SPECS = [
    zone("srv_outer",  10308.0, 10400.0, 1e-17, 0.08),
    zone("srv_wide",   10328.0, 10396.0, 3e-17, 0.10),
    zone("srv_narrow", 10351.0, 10390.0, 1e-16, 0.12),
]
for s in SRV_SPECS:
    half = s["height_m"] / FT / 2.0
    c = STAR + s["center_y"] / FT
    print(f"  {s['name']:11s} MD {c-half:7.1f} .. {c+half:7.1f}  h={s['height_m']/FT:5.1f} ft  k={s['perm']:.0e}")
print(f"  fracture at MD {STAR}; shear plane MD {SHEAR_PLANE_MD:.0f} inside srv_outer: "
      f"{10308.0 <= SHEAR_PLANE_MD <= 10400.0}")
print(f"  asymmetry: {STAR-10308.0:.0f} ft above the fracture vs {10400.0-STAR:.0f} ft below")

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
