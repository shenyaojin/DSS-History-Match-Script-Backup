"""V9: SRV spans injector -> monitoring well -> 1800 ft beyond, BC on the monitoring well.

Along strike, measured from the monitoring well (positive = toward Bearskin 3-PA):

      -1800 ft ............ 0 ............ +1934 ft
      SRV far tip     monitoring well     SRV near tip = the injecting stage

Total 3734 ft. The SRV therefore STARTS at the injector rather than running past it (V7
showed running past adds nothing) and continues 1800 ft on the far side of the well.

MD cross-section is V8's, pinned by the parallel-fault geometry: symmetric +/- 47.2 ft about
the tensile fracture at MD 10375.0, so srv_outer's edge lands exactly on the shear plane at
MD 10327.8.

Because the builder pins the injection nodeset to the SRV centre -- which this asymmetric
layout puts 67 ft from the fiber -- the BC is moved back onto the monitoring well and snapped
to a mesh node (an off-node coord makes ExtraNodesetGenerator MPI_Abort during setup).

Run with --run to launch MOOSE.
"""
import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fault_geometry as fg

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "fibeRIS" / "src"))
from fiberis.moose.templates.baseline_model_generator_fervo import build_baseline_model  # noqa: E402
from fiberis.moose.runner import MooseRunner  # noqa: E402

FT = 0.3048
PROJECT = os.environ.get("V9_PROJECT", "v9_srv_to_injector")
T1 = pd.Timestamp("2025-02-24 12:00")
STAR = fg.STAR_1200                                  # 10365.107 ft, notebook projection
SHEAR_MD = fg.SHEAR_MD_1200                          # 10317.837 ft
HALF_FT = fg.HALF_1200                               # 47.270 ft, pinned by the fault separation

TOWARD_INJ_FT = 1934.0                               # SRV tip sits at the injecting stage
BEYOND_FT = 1800.0                                   # and runs 1800 ft past the well
SRV_LEN_FT = TOWARD_INJ_FT + BEYOND_FT               # 3734 ft
WELL_OFFSET_FT = -(TOWARD_INJ_FT - BEYOND_FT) / 2.0  # -67 ft
DOMAIN_FT = 5600.0
NX = 680

DAS_PRESSURE = REPO / "data_fervo" / "fiberis_format" / "post_processing" / \
    "das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz"
OUT_DIR = REPO / "output" / PROJECT
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREPPED = OUT_DIR / "das_pressure_T1_prepended.npz"
I_FILE = OUT_DIR / f"{PROJECT}_input.i"

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
print(f"[{PROJECT}] DAS pressure: {len(vals)} steps, T1={T1}, IC={ic:.0f} psi")

HALF_HEIGHTS = fg.SRV_ZONES_1200
SRV_SPECS = [{"name": n, "length_m": SRV_LEN_FT * FT, "height_m": 2 * h * FT,
              "center_y": 0.0, "perm": k, "porosity": por}
             for n, h, k, por in HALF_HEIGHTS]
for n, h, k, _por in HALF_HEIGHTS:
    print(f"  {n:11s} MD {STAR - h:9.3f} .. {STAR + h:9.3f}   +/-{h:6.3f} ft   k={k:.0e} m2")
print(f"  along strike: -{BEYOND_FT:.0f} ft (past the well) .. +{TOWARD_INJ_FT:.0f} ft "
      f"(the injecting stage);  total {SRV_LEN_FT:.0f} ft")
print(f"  domain {DOMAIN_FT:.0f} ft, nx={NX} -> {DOMAIN_FT*FT/NX:.2f} m; {4*NX*110:,} elements")

builder = build_baseline_model(
    project_name=PROJECT, pressure_profile_path=str(PREPPED),
    model_width=100.0, model_length=DOMAIN_FT * FT, hf_length_ft=SRV_LEN_FT,
    shift_list_ft=np.array([WELL_OFFSET_FT]), angle=0.0, nx=NX,
    matrix_perm=1e-18, fracture_perm=1e-13, srv_specs=SRV_SPECS,
)
builder.generate_input_file(output_filepath=str(I_FILE))

# ---- put the BC back on the monitoring well, snapped to a node ---------------
centre_x = DOMAIN_FT * FT / 2.0
fiber_x = centre_x + WELL_OFFSET_FT * FT
dx = DOMAIN_FT * FT / NX
snapped = round(fiber_x / dx) * dx
txt = I_FILE.read_text()
new = re.sub(r"(\[injection\](?:.|\n)*?coord = ')[\d.]+",
             lambda m: f"{m.group(1)}{snapped:.6f}", txt, count=1)
assert new != txt, "injection coord not found"
I_FILE.write_text(new)
print(f"  BC on the monitoring well: {centre_x:.2f} -> {snapped:.4f} m "
      f"({(snapped - fiber_x) / FT:+.2f} ft from the fiber sampler)")
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
        n = len(list(OUT_DIR.glob(f"{PROJECT}_input_csv_fiber_strain_sampler_*ft_*.csv")))
        print(f"MOOSE success: {ok}   sampler files: {n}")
        if not ok or n == 0:
            print((err or "")[-2000:])
            sys.exit(1)
