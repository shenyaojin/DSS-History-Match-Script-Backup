"""V7: SRV extended 1600 ft PAST the injecting stage, not stopped short of it.

V6 capped the SRV at +1800 ft, which left it 134 ft short of the Bearskin 3-PA stage at
+1934 ft. V7 carries it 1600 ft beyond the stage, giving the same run-out past the injector
that V6 already had past the monitoring well:

      -1600 ft ......... 0 ......... +1934 ft ......... +3534 ft
      SRV far tip    monitoring       stage            SRV near tip
                        well        MD 9726
      |<-- 1600 -->|<--- 1934 --->|<--- 1600 --->|

Total 5134 ft. The MD cross-section is V6's, unchanged: symmetric +/-54.4 ft about the
hitting channel MD 10373.4, with srv_outer's top on the DDM shear plane at MD 10319.
Driving DAS pressure and its Dirichlet node are unchanged from V1-V6.

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

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "fibeRIS" / "src"))
from fiberis.moose.templates.baseline_model_generator_fervo import build_baseline_model  # noqa: E402
from fiberis.moose.runner import MooseRunner  # noqa: E402

FT = 0.3048
PROJECT = os.environ.get("V7_PROJECT", "v7_past_injector")
T1 = pd.Timestamp("2025-02-24 11:00")
STAR = 10373.4
SHEAR_MD = 10319.0
INJ_STRIKE_FT = 1934.0          # Bearskin 3-PA stage MD 9726, along strike from the well

RUNOUT_FT = 1600.0                                   # run-out past each well
TOWARD_INJ_FT = INJ_STRIKE_FT + RUNOUT_FT            # +3534 ft
BEYOND_FT = RUNOUT_FT                                # -1600 ft
SRV_LEN_FT = TOWARD_INJ_FT + BEYOND_FT               # 5134 ft
WELL_OFFSET_FT = -(TOWARD_INJ_FT - BEYOND_FT) / 2.0  # -967 ft
DOMAIN_FT = 7000.0
NX = 850

DAS_PRESSURE = REPO / "data_fervo" / "fiberis_format" / "post_processing" / \
    "das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz"
OUT_DIR = REPO / "output" / PROJECT
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREPPED = OUT_DIR / "das_pressure_T1_prepended.npz"
I_FILE = OUT_DIR / f"{PROJECT}_input.i"

# ---- driving pressure: unchanged from V1-V6 ----------------------------------
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
      f"IC={ic:.0f} psi (uniform in every block)")

# ---- V6's symmetric MD cross-section, unchanged ------------------------------
HALF_HEIGHTS = [("srv_outer", STAR - SHEAR_MD, 1e-17, 0.08),
                ("srv_wide", (STAR - SHEAR_MD) * 90.0 / 130.0, 3e-17, 0.10),
                ("srv_narrow", (STAR - SHEAR_MD) * 45.0 / 130.0, 1e-16, 0.12)]
SRV_SPECS = [{"name": n, "length_m": SRV_LEN_FT * FT, "height_m": 2 * h * FT,
              "center_y": 0.0, "perm": k, "porosity": por}
             for n, h, k, por in HALF_HEIGHTS]
for n, h, k, _por in HALF_HEIGHTS:
    print(f"  {n:11s} MD {STAR - h:8.1f} .. {STAR + h:8.1f}   +/-{h:5.1f} ft   k={k:.0e} m2")
print(f"  along strike: -{BEYOND_FT:.0f} ft (past the well) .. +{TOWARD_INJ_FT:.0f} ft "
      f"({RUNOUT_FT:.0f} ft past the stage at {INJ_STRIKE_FT:.0f} ft)")
print(f"  SRV length {SRV_LEN_FT:.0f} ft (V6 was 3400 ft); domain {DOMAIN_FT:.0f} ft, "
      f"nx={NX} -> {DOMAIN_FT*FT/NX:.2f} m; {4*NX*110:,} elements")

builder = build_baseline_model(
    project_name=PROJECT, pressure_profile_path=str(PREPPED),
    model_width=100.0, model_length=DOMAIN_FT * FT, hf_length_ft=SRV_LEN_FT,
    shift_list_ft=np.array([WELL_OFFSET_FT]), angle=0.0, nx=NX,
    matrix_perm=1e-18, fracture_perm=1e-13, srv_specs=SRV_SPECS,
)
builder.generate_input_file(output_filepath=str(I_FILE))
print("Generated .i:", I_FILE)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--np", type=int, default=24)
    ap.add_argument("--bc-at-fiber", action="store_true",
                    help="move the injection nodeset onto the fiber x. The builder pins it to "
                         "the SRV centre, which for this asymmetric layout is 967 ft from the "
                         "fiber -- far enough that leak-off kills the signal before it arrives.")
    args = ap.parse_args()
    if args.bc_at_fiber:
        centre_x = DOMAIN_FT * FT / 2.0
        fiber_x = centre_x + WELL_OFFSET_FT * FT
        # ExtraNodesetGenerator needs the coordinate to land ON a mesh node -- an off-node
        # coord makes MOOSE MPI_Abort during setup. Every earlier run got this for free by
        # sitting at the domain centre with an even nx; an off-centre BC does not.
        dx = DOMAIN_FT * FT / NX
        snapped = round(fiber_x / dx) * dx
        txt = I_FILE.read_text()
        new = re.sub(r"(\[injection\](?:.|\n)*?coord = ')[\d.]+",
                     lambda m: f"{m.group(1)}{snapped:.6f}", txt, count=1)
        assert new != txt, "injection coord not found"
        I_FILE.write_text(new)
        print(f"BC moved: injection node {centre_x:.1f} -> {snapped:.4f} m, snapped to mesh "
              f"node {round(fiber_x / dx)} ({(snapped - fiber_x) / FT:+.2f} ft from the fiber)")
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
