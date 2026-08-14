"""V8: SRV rebuilt to the updated two-PARALLEL-fault DDM geometry, BC on the monitoring well.

Geometry re-derived from data_fervo/legacy/07152026/two_fault_histories_<new suffix>.csv by
projecting the Gold 4-PB survey onto each fault plane (both strike 0 deg, dip 90):

    fault1 TENSILE  centre x = 281.75 ft  ->  pierces the well at MD 10375.0
    fault2 SHEAR    centre x = 234.48 ft  ->  pierces the well at MD 10327.8
    separation along the fault normal      =  47.2 ft

So the SRV half-width is pinned by the data, not chosen: making srv_outer's outer edge land on
the shear plane while staying symmetric about the tensile fracture forces +/- 47.2 ft
(V6/V7 used +/- 54.4 ft against the old MD 10319 shear plane).

    srv_outer   10327.8 - 10422.2   (+/- 47.2 ft)   k=1e-17   <- outer edge = shear plane
    srv_wide    10342.3 - 10407.7   (+/- 32.7 ft)   k=3e-17
    srv_narrow  10358.7 - 10391.3   (+/- 16.3 ft)   k=1e-16
    hf core     10375.0             (0.2 ft)        k=1e-13

The pressure BC sits ON the monitoring well (shift = 0, so the injection nodeset and the fiber
sampler share the same x), because the DAS-derived curve was back-computed from strain in the
V1 tensile fault at that channel.

Along strike the fracture is 3840 ft (fault1_L from the notebook) centred on the fiber. The
real fault tip stands off 14.3 ft from the well, which is far below the 2.5 m element size and
irrelevant at this scale -- and V6b/V7b showed along-strike length is saturated anyway.

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
PROJECT = "v8_parallel_faults"
T1 = pd.Timestamp("2025-02-24 12:00")          # new reference window start
STAR = 10375.0                                  # tensile fault pierces the well here
SHEAR_MD = 10327.8                              # shear fault pierces the well here
HALF_FT = STAR - SHEAR_MD                       # 47.2 ft -> srv_outer half-width
SRV_LEN_FT = 3840.0                             # fault1_L from the notebook
DOMAIN_FT = 5400.0
NX = 650

DAS_PRESSURE = REPO / "data_fervo" / "fiberis_format" / "post_processing" / \
    "das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz"
OUT_DIR = REPO / "output" / PROJECT
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREPPED = OUT_DIR / "das_pressure_T1_prepended.npz"
I_FILE = OUT_DIR / f"{PROJECT}_input.i"

# ---- driving pressure, re-based on the NEW T1 = 12:00 ------------------------
d = np.load(DAS_PRESSURE, allow_pickle=True)
p = np.asarray(d["data"], float)
t = np.asarray(d["taxis"], float)
t0 = pd.Timestamp(str(d["start_time"].item() if hasattr(d["start_time"], "item")
                      else d["start_time"]))
ic = float(p[0])
times = pd.DatetimeIndex([T1]).append(t0 + pd.to_timedelta(t, unit="s"))
vals = np.concatenate([[ic], p])
taxis_s = (times - T1).total_seconds().to_numpy()
assert np.all(np.diff(taxis_s) > 0), "pressure curve starts before the new T1"
np.savez(PREPPED, data=vals, taxis=taxis_s, start_time=str(T1))
print(f"[{PROJECT}] DAS pressure: {len(vals)} steps, T1={T1}, curve starts "
      f"{(t0 - T1).total_seconds()/3600:.1f} h after T1, ends "
      f"{taxis_s[-1]/86400:.2f} d after T1, IC={ic:.0f} psi (uniform in every block)")

# ---- SRV pinned by the fault separation --------------------------------------
HALF_HEIGHTS = [("srv_outer", HALF_FT, 1e-17, 0.08),
                ("srv_wide", HALF_FT * 90.0 / 130.0, 3e-17, 0.10),
                ("srv_narrow", HALF_FT * 45.0 / 130.0, 1e-16, 0.12)]
SRV_SPECS = [{"name": n, "length_m": SRV_LEN_FT * FT, "height_m": 2 * h * FT,
              "center_y": 0.0, "perm": k, "porosity": por}
             for n, h, k, por in HALF_HEIGHTS]
for n, h, k, _por in HALF_HEIGHTS:
    print(f"  {n:11s} MD {STAR - h:8.1f} .. {STAR + h:8.1f}   +/-{h:5.1f} ft   k={k:.0e} m2")
print(f"  fracture MD {STAR}, shear plane MD {SHEAR_MD} -> srv_outer edge is ON the shear "
      f"plane, SRV symmetric about the monitoring well")
print(f"  BC at the monitoring well (shift = 0): injection node and fiber sampler share x")
print(f"  SRV length {SRV_LEN_FT:.0f} ft; domain {DOMAIN_FT:.0f} ft, nx={NX} -> "
      f"{DOMAIN_FT*FT/NX:.2f} m; {4*NX*110:,} elements")

builder = build_baseline_model(
    project_name=PROJECT, pressure_profile_path=str(PREPPED),
    model_width=100.0, model_length=DOMAIN_FT * FT, hf_length_ft=SRV_LEN_FT,
    shift_list_ft=np.array([0.0]), angle=0.0, nx=NX,
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
        n = len(list(OUT_DIR.glob(f"{PROJECT}_input_csv_fiber_strain_sampler_*ft_*.csv")))
        print(f"sampler files written: {n}")
        if not ok or n == 0:
            print((err or "")[-2000:])
            sys.exit(1)
