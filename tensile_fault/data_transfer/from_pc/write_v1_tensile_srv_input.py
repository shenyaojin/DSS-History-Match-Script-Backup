"""Generate the MOOSE input file (.i) for the V1 tensile-SRV model via fiberis.

Generation only -- this does NOT run MOOSE. Pattern mirrors script 121
(build_baseline_model -> plot_geometry -> generate_input_file); the run step
(MooseRunner) is intentionally left out until the geometry/params are settled.

Geometry (settled):
  Y = fiber = MD ; X = fracture direction ; fracture/SRV = blocks thin in Y, long in X
  fracture crossing Y=0  <->  MD 10373.4 ft   (MD = 10373.4 + (Y-Yfrac)/0.3048)
  SRV along-MD width = observed ~90 ft (wide) / ~45 ft (narrow)
Tunable MOOSE params (to reproduce observation later): SRV perm/porosity, elastic
modulus / Biot, SRV widths, and the injection pressure (the inversion unknown).
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
PROJECT = "v1_tensile_srv"
OBSERVED_WIDTH_FT = 90.0

PRESSURE_NPZ = (REPO / "data_fervo" / "fiberis_format" / "post_processing"
                / "das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz")
OUT_DIR = REPO / "output" / PROJECT
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPO / "figs" / "tensile_fault_qc" / PROJECT
FIG_DIR.mkdir(parents=True, exist_ok=True)
I_FILE = OUT_DIR / f"{PROJECT}_input.i"

# SRV widened so the along-MD (Y) thickness matches the observed ~90 ft feature.
SRV_SPECS = [
    {"name": "srv_wide",   "length_m": 280 * FT, "height_m": OBSERVED_WIDTH_FT * FT,     "perm": 1e-15, "porosity": 0.10},
    {"name": "srv_narrow", "length_m": 280 * FT, "height_m": OBSERVED_WIDTH_FT / 2 * FT, "perm": 1e-14, "porosity": 0.12},
]

builder = build_baseline_model(
    project_name=PROJECT,
    pressure_profile_path=str(PRESSURE_NPZ),   # placeholder pressure; swap for unit-ΔP in the g(z) run
    model_width=100.0,                         # Y half-extent (m) -> MD 10373 +/-328 ft
    model_length=200.0,                        # X extent (m), fracture direction
    hf_length_ft=250.0,
    shift_list_ft=np.array([round(250.0 / 3.0, 3)]),   # fiber crosses 83 ft off the injection point
    angle=0.0,
    matrix_perm=1e-18,
    fracture_perm=1e-13,
    srv_specs=SRV_SPECS,
)

builder.plot_geometry(save_path=str(FIG_DIR / "geometry.png"), hide_legend=False, equal_aspect=True)
builder.generate_input_file(output_filepath=str(I_FILE))

print("\n=== input file generated (NOT run) ===")
print("i-file :", I_FILE)
print("geometry:", FIG_DIR / "geometry.png")
print("MOOSE binary (for when we run):",
      REPO / "moose_env/moose/modules/porous_flow/porous_flow-opt")
