"""V11: SRV runs 1500 ft past the monitoring well and 1800 ft past the injecting stage.

Along strike from the monitoring well (positive = toward Bearskin 3-PA):

      -1500 ft ......... 0 ......... +1934 ft ......... +3734 ft
      SRV far tip   monitoring well   injecting stage   SRV near tip
                                      (yellow star)
      |<-- 1500 -->|<--- 1934 --->|<---- 1800 ---->|

Total 5234 ft. Note the fault strikes 0 deg, i.e. the SRV runs roughly NORTH-SOUTH; "west of
the well" and "east of the star" are read here as the two along-strike directions, with the
star (the stage) on the positive side.

MD cross-section is unchanged and pinned by the parallel-fault geometry (fault_geometry.py):
symmetric +/-47.270 ft about the tensile fracture at MD 10365.107, so srv_outer's edge lands
on the shear plane at MD 10317.837.

BC stays on the monitoring well, snapped to a mesh node.

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
PROJECT = os.environ.get("V11_PROJECT", "v11_srv_extended")
T1 = pd.Timestamp(fg.T1_1200)
STAR, SHEAR_MD = fg.STAR_1200, fg.SHEAR_MD_1200

STAGE_FT = 1934.0                                    # the yellow star, along strike
PAST_STAGE_FT = 1800.0                               # SRV continues this far beyond it
PAST_WELL_FT = 1500.0                                # and this far the other way
TOWARD_INJ_FT = STAGE_FT + PAST_STAGE_FT             # +3734 ft
SRV_LEN_FT = TOWARD_INJ_FT + PAST_WELL_FT            # 5234 ft
WELL_OFFSET_FT = -(TOWARD_INJ_FT - PAST_WELL_FT) / 2.0   # -1117 ft
DOMAIN_FT = 7200.0
NX = 880

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

SRV_SPECS = [{"name": n, "length_m": SRV_LEN_FT * FT, "height_m": 2 * h * FT,
              "center_y": 0.0, "perm": k, "porosity": por}
             for n, h, k, por in fg.SRV_ZONES_1200]
for n, h, k, _por in fg.SRV_ZONES_1200:
    print(f"  {n:11s} MD {STAR - h:9.3f} .. {STAR + h:9.3f}   +/-{h:6.3f} ft   k={k:.0e} m2")
print(f"  along strike: -{PAST_WELL_FT:.0f} ft (past the well) .. +{TOWARD_INJ_FT:.0f} ft "
      f"({PAST_STAGE_FT:.0f} ft past the stage at {STAGE_FT:.0f} ft);  total {SRV_LEN_FT:.0f} ft")
print(f"  hf core k = {float(os.environ.get('FRAC_PERM', 1e-13)):.0e} m2 "
      f"(1e-16 == same as srv_narrow, i.e. NO distinct core)")
print(f"  domain {DOMAIN_FT:.0f} ft, nx={NX} -> {DOMAIN_FT*FT/NX:.2f} m; {4*NX*110:,} elements")

builder = build_baseline_model(
    project_name=PROJECT, pressure_profile_path=str(PREPPED),
    model_width=100.0, model_length=DOMAIN_FT * FT, hf_length_ft=SRV_LEN_FT,
    shift_list_ft=np.array([WELL_OFFSET_FT]), angle=0.0, nx=NX,
    matrix_perm=1e-18, fracture_perm=float(os.environ.get('FRAC_PERM', 1e-13)),
    srv_specs=SRV_SPECS,
)
builder.generate_input_file(output_filepath=str(I_FILE))

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
