"""V5: the fracture/SRV extended along strike to the real fault length.

V1-V4 all used a 250 ft fracture in a 656 ft domain, so the model truncated ~1900 ft of
fracture that physically exists. The fault frame says:

    fiber crossing      along-strike    0 ft   (the fault centre)
    Bearskin 3-PA stage along-strike +1934 ft  (36 ft off the fault plane)
    DDM fault1 half-length              1920 ft

so the fracture must run ~1935 ft each way from the fiber crossing: one tip reaches the
injecting stage, and the monitoring well sits at the centre, crossed by the full SRV.

Only the along-strike extent changes. The MD cross-section is V3's exactly, the driving
DAS pressure and its Dirichlet node stay at the fiber crossing, and the fiber sampler keeps
its 83.333 ft offset -- so the hitting channel on the monitoring well is unchanged and V3 vs
V5 isolates the length effect alone.

    domain     656 ft  ->  5400 ft
    fracture   250 ft  ->  3870 ft   (half-length 1935 ft, just past the stage at 1934 ft)
    SRV zones  280 ft  ->  3870 ft
    nx         200     ->  650       (2.53 m elements along strike)

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
PROJECT = "v5_full_length"
T1 = pd.Timestamp("2025-02-24 11:00")
STAR = 10373.4                  # MD at model y = 0; fiber crossing = fault centre
INJ_STRIKE_FT = 1934.0          # Bearskin 3-PA stage MD 9726, along strike from the crossing
HF_LENGTH_FT = 3870.0           # half = 1935 ft -> the tip just reaches the stage
DOMAIN_FT = 5400.0              # leaves ~765 ft of matrix beyond each fracture tip
NX = 650

DAS_PRESSURE = REPO / "data_fervo" / "fiberis_format" / "post_processing" / \
    "das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz"
OUT_DIR = REPO / "output" / PROJECT
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREPPED = OUT_DIR / "das_pressure_T1_prepended.npz"
I_FILE = OUT_DIR / f"{PROJECT}_input.i"

# ---- driving pressure: identical to V1-V4 ------------------------------------
d = np.load(DAS_PRESSURE, allow_pickle=True)
p = np.asarray(d["data"], float)
t = np.asarray(d["taxis"], float)
t0 = pd.Timestamp(str(d["start_time"].item() if hasattr(d["start_time"], "item")
                      else d["start_time"]))
ic = float(p[0])
times = pd.DatetimeIndex([T1]).append(t0 + pd.to_timedelta(t, unit="s"))
vals = np.concatenate([[ic], p])
taxis_s = (times - T1).total_seconds().to_numpy()
assert np.all(np.diff(taxis_s) > 0)
np.savez(PREPPED, data=vals, taxis=taxis_s, start_time=str(T1))
print(f"[{PROJECT}] DAS pressure prepped: {len(vals)} steps, {taxis_s[-1]/86400:.1f} days, "
      f"IC={ic:.0f} psi")

# ---- V3's MD cross-section, stretched along strike ---------------------------
ZONES_MD = [("srv_outer", 10308.0, 10400.0, 1e-17, 0.08),
            ("srv_wide", 10328.0, 10396.0, 3e-17, 0.10),
            ("srv_narrow", 10351.0, 10390.0, 1e-16, 0.12)]
SRV_SPECS = [{"name": n, "length_m": HF_LENGTH_FT * FT, "height_m": (b - a) * FT,
              "center_y": (0.5 * (a + b) - STAR) * FT, "perm": k, "porosity": por}
             for n, a, b, k, por in ZONES_MD]
for n, a, b, k, _por in ZONES_MD:
    print(f"  {n:11s} MD {a:8.1f} .. {b:8.1f}   k={k:.0e}   length {HF_LENGTH_FT:.0f} ft")
print(f"  fracture half-length {HF_LENGTH_FT/2:.0f} ft vs stage at {INJ_STRIKE_FT:.0f} ft "
      f"-> tip is {HF_LENGTH_FT/2 - INJ_STRIKE_FT:+.0f} ft past the stage")
print(f"  monitoring well sits at the fracture CENTRE, so the SRV crosses it by "
      f"{HF_LENGTH_FT/2:.0f} ft each way")
print(f"  domain {DOMAIN_FT:.0f} ft, nx={NX} -> {DOMAIN_FT*FT/NX:.2f} m elements along strike")

builder = build_baseline_model(
    project_name=PROJECT, pressure_profile_path=str(PREPPED),
    model_width=100.0, model_length=DOMAIN_FT * FT, hf_length_ft=HF_LENGTH_FT,
    shift_list_ft=np.array([round(250.0 / 3.0, 3)]), angle=0.0, nx=NX,
    matrix_perm=1e-18, fracture_perm=1e-13, srv_specs=SRV_SPECS,
)
builder.generate_input_file(output_filepath=str(I_FILE))
print("Generated .i:", I_FILE)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--np", type=int, default=24)
    args = ap.parse_args()
    if args.run:
        runner = MooseRunner(
            moose_executable_path=str(REPO / "moose_env/moose/modules/porous_flow/porous_flow-opt"),
            mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec",
        )
        ok, _o, err = runner.run(input_file_path=str(I_FILE), output_directory=str(OUT_DIR),
                                 num_processors=args.np, log_file_name="simulation.log",
                                 stream_output=False, clean_output_dir=False)
        print("MOOSE success:", ok)
        if not ok:
            print((err or "")[-2000:])
            sys.exit(1)
