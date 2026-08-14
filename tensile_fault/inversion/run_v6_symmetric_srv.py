"""V6: SRV symmetric about the monitoring well in MD, and trimmed along strike.

Two changes from V5:

1. MD cross-section is now SYMMETRIC about the hitting channel (MD 10373.4) instead of
   carrying V3's asymmetry (65.4 ft up / 26.6 ft down). Zone heights are the ones that also
   put srv_outer's top exactly on the DDM shear plane at MD 10319, so both constraints hold:

       srv_outer   10319.0 - 10427.8   (+/- 54.4 ft)   k=1e-17
       srv_wide    10335.7 - 10411.1   (+/- 37.7 ft)   k=3e-17
       srv_narrow  10354.6 - 10392.2   (+/- 18.8 ft)   k=1e-16

2. Along strike the SRV is trimmed and made asymmetric about the well: it reaches +1800 ft
   toward the Bearskin 3-PA stage (which sits at +1934 ft -- close enough, since it is
   ~1900 ft from the fiber and a hundred feet there is invisible at the monitoring well) and
   continues 1600 ft past the well on the far side. Total 3400 ft, down from V5's 3870 ft.

Because the builder centres the fracture, SRV and injection node on the domain centre, the
SRV centre sits at well +100 ft and the fiber sampler is offset by -100 ft to land on the
well. The pressure BC therefore stays ~100 ft from the hitting channel, essentially the same
83 ft offset V1-V5 used, and the driving DAS curve is unchanged.

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
PROJECT = os.environ.get("V6_PROJECT", "v6_symmetric_srv")
T1 = pd.Timestamp("2025-02-24 11:00")
STAR = 10373.4                  # hitting channel; SRV is symmetric about this in MD
SHEAR_MD = 10319.0
INJ_STRIKE_FT = 1934.0          # Bearskin 3-PA stage MD 9726, along strike from the well

TOWARD_INJ_FT = 1800.0          # SRV reach on the injection side of the well
BEYOND_FT = 1600.0              # SRV reach past the well on the far side
SRV_LEN_FT = TOWARD_INJ_FT + BEYOND_FT              # 3400 ft
WELL_OFFSET_FT = -(TOWARD_INJ_FT - BEYOND_FT) / 2.0  # -100 ft: well sits 100 ft off centre
DOMAIN_FT = 5000.0
NX = 600

DAS_PRESSURE = REPO / "data_fervo" / "fiberis_format" / "post_processing" / \
    "das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz"
OUT_DIR = REPO / "output" / PROJECT
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREPPED = OUT_DIR / "das_pressure_T1_prepended.npz"
I_FILE = OUT_DIR / f"{PROJECT}_input.i"

# ---- driving pressure: unchanged from V1-V5 ----------------------------------
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
      f"IC={ic:.0f} psi (uniform across every block)")

# ---- symmetric MD cross-section ----------------------------------------------
HALF_HEIGHTS = [("srv_outer", STAR - SHEAR_MD, 1e-17, 0.08),      # 54.4 ft -> top on shear plane
                ("srv_wide", (STAR - SHEAR_MD) * 90.0 / 130.0, 3e-17, 0.10),
                ("srv_narrow", (STAR - SHEAR_MD) * 45.0 / 130.0, 1e-16, 0.12)]
SRV_SPECS = [{"name": n, "length_m": SRV_LEN_FT * FT, "height_m": 2 * h * FT,
              "center_y": 0.0, "perm": k, "porosity": por}
             for n, h, k, por in HALF_HEIGHTS]
for n, h, k, _por in HALF_HEIGHTS:
    print(f"  {n:11s} MD {STAR - h:8.1f} .. {STAR + h:8.1f}   +/-{h:5.1f} ft   k={k:.0e}")
print(f"  symmetric about the hitting channel MD {STAR}; srv_outer top on the shear "
      f"plane MD {SHEAR_MD:.0f}")
print(f"  along strike: {TOWARD_INJ_FT:.0f} ft toward the stage (at {INJ_STRIKE_FT:.0f} ft, "
      f"{INJ_STRIKE_FT - TOWARD_INJ_FT:.0f} ft short) and {BEYOND_FT:.0f} ft past the well")
print(f"  SRV length {SRV_LEN_FT:.0f} ft (V5 was 3870 ft); domain {DOMAIN_FT:.0f} ft, "
      f"nx={NX} -> {DOMAIN_FT*FT/NX:.2f} m")

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
                    help="move the injection nodeset onto the fiber x, so this can act as the "
                         "BC-distance control for V7b. Snapped to a mesh node: an off-node "
                         "coord makes ExtraNodesetGenerator MPI_Abort during setup.")
    args = ap.parse_args()
    if args.bc_at_fiber:
        centre_x = DOMAIN_FT * FT / 2.0
        fiber_x = centre_x + WELL_OFFSET_FT * FT
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
