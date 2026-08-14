"""V4 tensile-SRV MOOSE runs: two ways of honouring the SRV design intent.

Design intent (Shenyao): the MOOSE fracture marks the CENTRE of the SRV (the poroelastic
tensile response fills the whole SRV); the DDM shear plane rides the SRV/matrix BOUNDARY.
V3 honours neither -- it is asymmetric (65.4 ft up / 26.6 ft down) and its outer boundary
sits 11 ft past the shear plane at MD 10319.

Two variants, so the two changes can be told apart:

  v4_srv_centred   BOTH constraints. V2's symmetric stack rescaled x0.8369 so srv_outer's
                   top lands on the shear plane and the fracture sits at the centre.
                   srv_outer 10319.0-10427.8.  Risk: the base is only 10.6 ft shallower
                   than V2's 10438.4, and that depth is what over-produced the deep
                   compression lobe V3 was built to remove.

  v4_srv_boundary  BOUNDARY ONLY. Top moved to the shear plane, V3's shallow base at
                   10400 kept. srv_outer 10319.0-10400.0 (asymmetric).  Tests "boundary
                   on the shear plane" without reintroducing the deep lobe.

Permeabilities, mesh, driving DAS pressure and fiber sampler are identical to V2/V3, so
geometry is the only variable. V1/V2/V3 outputs are NOT touched.

Usage:
    python run_v4_srv_variants.py --variant centred  [--run] [--np 20]
    python run_v4_srv_variants.py --variant boundary [--run] [--np 20]
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
T1 = pd.Timestamp("2025-02-24 11:00")
STAR = 10373.4                       # MD at model y = 0 (fracture)
SHEAR_PLANE_MD = 10319.0             # intended SRV/matrix boundary

DAS_PRESSURE = REPO / "data_fervo" / "fiberis_format" / "post_processing" / \
    "das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz"

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["centred", "boundary"], required=True)
ap.add_argument("--run", action="store_true")
ap.add_argument("--np", type=int, default=20)
args = ap.parse_args()

# --- zone spans, in MD ---------------------------------------------------------
# V2's symmetric heights (130 / 90 / 45 ft) are the grading template for both variants.
V2_HEIGHTS = [("srv_outer", 130.0, 1e-17, 0.08),
              ("srv_wide", 90.0, 3e-17, 0.10),
              ("srv_narrow", 45.0, 1e-16, 0.12)]

if args.variant == "centred":
    PROJECT = "v4_srv_centred"
    scale = 2 * (STAR - SHEAR_PLANE_MD) / V2_HEIGHTS[0][1]      # outer top -> shear plane
    ZONES_MD = [(n, STAR - h * scale / 2, STAR + h * scale / 2, k, p)
                for n, h, k, p in V2_HEIGHTS]
else:
    PROJECT = "v4_srv_boundary"
    # Keep V3's proportions but move every top up so srv_outer starts on the shear plane.
    V3_MD = [("srv_outer", 10308.0, 10400.0), ("srv_wide", 10328.0, 10396.0),
             ("srv_narrow", 10351.0, 10390.0)]
    shift = SHEAR_PLANE_MD - V3_MD[0][1]                        # +11.0 ft
    up_scale = (V3_MD[0][1] + shift - STAR) / (V3_MD[0][1] - STAR)
    ZONES_MD = [(n, STAR + (t - STAR) * up_scale, b, k, p)
                for (n, t, b), (_n, _h, k, p) in zip(V3_MD, V2_HEIGHTS)]

OUT_DIR = REPO / "output" / PROJECT
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREPPED = OUT_DIR / "das_pressure_T1_prepended.npz"
I_FILE = OUT_DIR / f"{PROJECT}_input.i"

# ---- prep driving pressure (identical to V1/V2/V3) ---------------------------
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

SRV_SPECS = [{"name": n, "length_m": 280 * FT, "height_m": (b - t) * FT,
              "center_y": (0.5 * (t + b) - STAR) * FT, "perm": k, "porosity": por}
             for n, t, b, k, por in ZONES_MD]
for (n, t, b, k, _por) in ZONES_MD:
    print(f"  {n:11s} MD {t:8.1f} .. {b:8.1f}   h={b - t:6.1f} ft   k={k:.0e}")
outer_t, outer_b = ZONES_MD[0][1], ZONES_MD[0][2]
print(f"  fracture MD {STAR}; SRV centre MD {0.5 * (outer_t + outer_b):.1f} "
      f"(fracture offset {STAR - 0.5 * (outer_t + outer_b):+.1f} ft); "
      f"outer top vs shear plane {outer_t - SHEAR_PLANE_MD:+.1f} ft")

builder = build_baseline_model(
    project_name=PROJECT, pressure_profile_path=str(PREPPED),
    model_width=100.0, model_length=200.0, hf_length_ft=250.0,
    shift_list_ft=np.array([round(250.0 / 3.0, 3)]), angle=0.0,
    matrix_perm=1e-18, fracture_perm=1e-13, srv_specs=SRV_SPECS,
)
builder.generate_input_file(output_filepath=str(I_FILE))
print("Generated .i:", I_FILE)

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
