"""Verify the physical origin of the observed DAS shear band.

Hypothesis (from the well geometry): the observed antisymmetric shear doublet is a
STRIKE-SLIP dislocation projected onto the fiber axis. Because the Gold 4-PB lateral
crosses the ~N-S fault while running ~E-W but with a small (~1 deg) along-strike tilt
that FLIPS SIGN around MD 10400, the along-well projection d(u.t_hat)/ds picks up the
slip discontinuity -> the axial shear band. A PERFECTLY PERPENDICULAR fiber (no
along-strike component) should see almost nothing -> which is why our 2D perpendicular
MOOSE model (strike-slip goes into strain_xy, not the axial strain_yy) misses it.

This runs DDM fault2 (shear only) through:
  (A) the REAL well path            -> expect the 0.039 me antisymmetric band
  (B) a straightened PERP fiber     -> expect ~0
and overlays the observed shear (v1_ddm_shear_strain_4h ... which is itself the
real-well DDM shear, used here as the target shape) for a consistency check.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "tensile_fault" / "data_transfer" / "from_pc"))
from DDMpy_log import Well, DynamicFracture  # noqa: E402

WELL_CSV = REPO / "data_fervo" / "legacy" / "Gold_4_PB_Well_Geometry.csv"
SHEAR_CSV = (REPO / "data_fervo" / "legacy" / "07152026_decomposed"
             / "v1_ddm_shear_strain_4h_20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref.csv")
OUT_FIG = REPO / "figs" / "tensile_fault_qc" / "inversion" / "verify_shear_projection.png"
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

# ---- fault2 (shear) geometry / history  (verbatim from export_ddm_shear_component.py)
MD_CENTER = 10340.0
fault2_strike, dip = 0.6, 90
fault2_L, fracture_height = 3800, 4000
S1_max = -0.03

T0 = datetime(2025, 2, 24, 0, 0)
T1 = datetime(2025, 2, 24, 11, 0)
T2 = datetime(2025, 2, 28, 0, 0)
T3 = datetime(2025, 3, 3, 22, 0)
mins = lambda dt: (dt - T0).total_seconds() / 60.0
t1, t2, t3 = mins(T1), mins(T2), mins(T3)

N = 144
delta_t = t3 / N
N1 = int(t1 // delta_t)
N2 = int((t2 - t1) // delta_t + 1)
N3 = int((t3 - t2 + 1) // delta_t + 1)
taxis = np.concatenate([np.linspace(0, t1 - delta_t, N1),
                        np.linspace(t1, t2 - delta_t, N2),
                        np.linspace(t2, t3, N3)])
length = np.full_like(taxis, fault2_L)
height = np.full_like(taxis, fracture_height)
width = np.zeros_like(taxis)                                   # no opening
shear = np.concatenate([np.zeros(N1 + N2), np.linspace(0, S1_max, N3)])  # slip, T2->T3

# ---- well geometry -----------------------------------------------------------
df = pd.read_csv(WELL_CSV).sort_values("MD")
md_csv = df["MD"].to_numpy(float)
xg, yg, zg = (df["x_gold"].to_numpy(float), df["y_gold"].to_numpy(float),
              df["z_gold"].to_numpy(float))
xc = float(np.interp(MD_CENTER, md_csv, xg))
yc = float(np.interp(MD_CENTER, md_csv, yg))
zc = float(np.interp(MD_CENTER, md_csv, zg))
print(f"fault centre @ MD {MD_CENTER:.0f}: ({xc:.1f},{yc:.1f},{zc:.1f}) ft")

# (A) real well path
well_real = Well.set_well_by_points(np.c_[xg, yg, zg], N=len(xg) * 10, smooth=31)
well_real.gauge_length = 30.48

# (B) perpendicular straightened fiber: keep the REAL x (E-W progression) and z (the
#     vertical descent that builds MD), but REMOVE the small along-strike wiggle by
#     holding y constant at the fault centre. In the lateral (z~const, x varying) the
#     tangent becomes pure E-W with no along-strike component -> the ideal fiber
#     perpendicular to the fault strike. dy<<dx,dz so MD is essentially unchanged.
well_perp = Well.set_well_by_points(np.c_[xg, np.full_like(yg, yc), zg],
                                    N=len(xg) * 10, smooth=31)
well_perp.gauge_length = 30.48


def run(well):
    fr = DynamicFracture.GlobalRectangularFracture()
    fr.set_global_coors(fault2_strike, dip, xc, yc, zc)
    fr.define_LHW(taxis=taxis, length=length, height=height, width=width, S1=shear, S2=None)
    fr.set_monitor_wells(well)
    fr.calculate()
    sd = fr.gather_strain_data()[0]
    return np.asarray(sd.daxis, float), np.asarray(sd.taxis, float), np.asarray(sd.data, float)


md_r, tt_r, str_r = run(well_real)   # strain [MD, time], units strain
md_p, tt_p, str_p = run(well_perp)
print("real/perp strain shapes:", str_r.shape, str_p.shape)


def profile_at(md, tt, data, when):
    """T1-referenced strain profile (millistrain) at the time nearest `when`, cropped 10200-10500."""
    j = int(np.argmin(np.abs(tt - mins(when))))
    j1 = int(np.argmin(np.abs(tt - mins(T1))))
    prof = (data[:, j] - data[:, j1]) * 1e3
    m = (md >= 10200) & (md <= 10500)
    return md[m], prof[m]


mdw_r, prof_r = profile_at(md_r, tt_r, str_r, T3)
mdw_p, prof_p = profile_at(md_p, tt_p, str_p, T3)

# observed shear (real-well DDM shear export) at the last 4h column
obs = pd.read_csv(SHEAR_CSV)
md_o = obs["measured_depth_ft"].to_numpy(float)
prof_o = obs[obs.columns[-1]].to_numpy(float)

print(f"\n(A) REAL well  : peak |strain| = {np.nanmax(np.abs(prof_r)):.4f} me  "
      f"(+{np.nanmax(prof_r):.4f}@{mdw_r[np.nanargmax(prof_r)]:.0f}, {np.nanmin(prof_r):+.4f}@{mdw_r[np.nanargmin(prof_r)]:.0f})")
print(f"(B) PERP fiber : peak |strain| = {np.nanmax(np.abs(prof_p)):.4f} me")
print(f"    -> PERP/REAL peak ratio = {np.nanmax(np.abs(prof_p))/np.nanmax(np.abs(prof_r)):.2%}")
print(f"(obs shear     : peak |strain| = {np.nanmax(np.abs(prof_o)):.4f} me)")

# ---- figure ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 7))
ax.plot(prof_o, md_o, "k-", lw=2.4, label="observed shear (target)")
ax.plot(prof_r, mdw_r, "-", color="#c0392b", lw=2.0, label="DDM shear → REAL oblique well")
ax.plot(prof_p, mdw_p, "--", color="#2980b9", lw=2.0, label="DDM shear → PERPENDICULAR fiber")
ax.axvline(0, color="0.6", lw=0.8)
ax.set_ylim(10500, 10200)
ax.set_xlabel("T1-referenced axial strain at T3 (mε)")
ax.set_ylabel("Measured depth (ft)")
ax.set_title("Shear band = strike-slip projected onto the well's ~1° obliquity\n"
             "(perpendicular fiber sees almost nothing)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=140)
print(f"\nsaved {OUT_FIG}")
