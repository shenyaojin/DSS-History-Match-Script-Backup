"""Draw the 2D MOOSE tensile-SRV geometry with fiberis (geometry only, no MOOSE run).

Reproduces the plane where the fiber meets the fracture:
  Y (model, m) = fiber direction = along measured depth (MD)
  X (model, m) = fracture direction (in-plane); plane strain in the third direction
  fracture / SRV = blocks (hf + nested SRV zones), thin in Y, long in X
  fiber sampler runs along Y at a monitor shift in X, samples strain_yy = the DAS quantity

Geometry priorities (per Pengchao): keep the real fiber/fault position; take the DDM
geometry as primary; the SRV *width along MD* is set from the OBSERVED ~90 ft feature
(the DDM plane cannot give this width -> that is why we use a MOOSE SRV zone).

MD <-> Y mapping (from script 124):  MD_ft = 10373.4 + (Y_model - Y_frac)/0.3048
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "fibeRIS" / "src"))
from fiberis.moose.templates.baseline_model_generator_fervo import build_baseline_model

FT = 0.3048
STAR_DEPTH_FT = 10373.4          # fracture crossing on the fiber (yellow-star MD)
OBSERVED_WIDTH_FT = 90.0         # observed along-MD feature width -> widest SRV

PRESSURE_NPZ = (REPO / "data_fervo" / "fiberis_format" / "post_processing"
                / "das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz")
FIG = REPO / "figs" / "tensile_fault_qc" / "v1_geometry" / "moose_tensile_srv_geometry.png"
FIG.parent.mkdir(parents=True, exist_ok=True)

# SRV widened so the along-MD (Y) thickness matches the observed ~90 ft feature.
SRV_SPECS = [
    {"name": "srv_wide",   "length_m": 280 * FT, "height_m": OBSERVED_WIDTH_FT * FT,      "perm": 1e-15, "porosity": 0.10},
    {"name": "srv_narrow", "length_m": 280 * FT, "height_m": OBSERVED_WIDTH_FT / 2 * FT,  "perm": 1e-14, "porosity": 0.12},
]

builder = build_baseline_model(
    project_name="v1_moose_tensile_geometry_preview",
    pressure_profile_path=str(PRESSURE_NPZ),
    model_width=100.0,          # Y half-extent (m) -> +/-100 m = MD 10373 +/-328 ft
    model_length=200.0,         # X extent (m), fracture direction
    hf_length_ft=250.0,
    shift_list_ft=np.array([round(250.0 / 3.0, 3)]),   # fiber offset along X
    angle=0.0,
    matrix_perm=1e-18,
    fracture_perm=1e-13,
    srv_specs=SRV_SPECS,
)
builder.plot_geometry(save_path=str(FIG), hide_legend=False, equal_aspect=True)

# report the geometry in MD terms
def y_to_md(y_m, y_frac=0.0):
    return STAR_DEPTH_FT + (y_m - y_frac) / FT

print("Saved geometry:", FIG)
print(f"Fracture crossing: Y=0 m  <->  MD {STAR_DEPTH_FT:.1f} ft")
for s in SRV_SPECS:
    half = s["height_m"] / 2
    print(f"{s['name']:10s}: Y=+/-{half:5.2f} m  ->  MD {y_to_md(-half):.0f} - {y_to_md(half):.0f} ft "
          f"(along-MD width {s['height_m']/FT:.0f} ft)")
print(f"Fiber sampler spans Y=+/-100 m -> MD {y_to_md(-100):.0f} - {y_to_md(100):.0f} ft "
      f"(observed window 10200-10500 is well inside)")
