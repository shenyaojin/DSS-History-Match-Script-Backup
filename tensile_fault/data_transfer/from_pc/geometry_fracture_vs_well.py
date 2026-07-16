"""Fracture-vs-well geometry for the V1 fault (Gold 4-PB monitoring well).

Reconstructs the DDM two-fault geometry in the SAME local frame as the well
survey (x_gold, y_gold, z_gold, feet) and answers: where is the fracture
relative to the monitoring well / DAS fiber?

Geometry facts pulled from the modeling notebook (cell 15):
  fault1: strike=-0.8 deg, dip=90, length=3840 ft, height=4000 ft  (tensile, T1-T3)
  fault2: strike=+0.6 deg, dip=90, length=3800 ft, height=4000 ft  (shear,   T2-T3)
  shared centre = pumping stage coordinate chosen to sit at well MD ~= 10340 ft
      (the DAS "hitting channel"; the exact stage csv is not in this repo, so we
       place the centre at the well point at MD 10340 -- Pengchao's stated intent).
Rotation convention reimplemented verbatim from DDMpy_log/Fracture.py so no DDM
import (which needs JIN_pylib) is required.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
WELL_CSV = REPO / "data_fervo" / "legacy" / "Gold_4_PB_Well_Geometry.csv"
PROJ_NPZ = REPO / "data_fervo" / "fiberis_format" / "projection_data_gold4pb.npz"
STIM_NPZ = REPO / "data_fervo" / "fiberis_format" / "stimulation_loc_bearskin.npz"
OUT_DIR = REPO / "figs" / "tensile_fault_qc" / "v1_geometry"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MD_CENTER = 10340.0          # fault centre pinned to this fiber MD
MD_TOP, MD_BOT = 10200.0, 10500.0   # DAS analysis window


# --- DDM rotation, reimplemented from DDMpy_log/Fracture.py --------------------
def _rotate_matrix(strike_rad, dip_rad):
    if dip_rad == np.pi / 2:
        dip_rad = np.pi / 2 - 1e-3
    strike = -strike_rad + np.pi
    return (np.array([[1, 0, 0],
                      [0, np.cos(dip_rad), np.sin(dip_rad)],
                      [0, -np.sin(dip_rad), np.cos(dip_rad)]])
            @ np.array([[np.sin(strike), -np.cos(strike), 0],
                        [np.cos(strike), np.sin(strike), 0],
                        [0, 0, 1]]))


def local_to_global(local_xyz, strike_deg, dip_deg):
    R = _rotate_matrix(np.deg2rad(strike_deg), np.deg2rad(dip_deg))
    return np.linalg.inv(R) @ np.asarray(local_xyz, dtype=float)


def fault_axes(strike_deg, dip_deg):
    """Return unit vectors (length_dir x1, height_dir x2, normal x3) in global frame."""
    x1 = local_to_global([1, 0, 0], strike_deg, dip_deg)
    x2 = local_to_global([0, 1, 0], strike_deg, dip_deg)
    x3 = local_to_global([0, 0, 1], strike_deg, dip_deg)
    return x1 / np.linalg.norm(x1), x2 / np.linalg.norm(x2), x3 / np.linalg.norm(x3)


def fault_rectangle(center, strike_deg, dip_deg, length, height):
    corners_local = np.array([[-length / 2, -height / 2, 0],
                              [+length / 2, -height / 2, 0],
                              [+length / 2, +height / 2, 0],
                              [-length / 2, +height / 2, 0],
                              [-length / 2, -height / 2, 0]])
    return np.array([center + local_to_global(c, strike_deg, dip_deg) for c in corners_local])


# --- load well ----------------------------------------------------------------
w = pd.read_csv(WELL_CSV).sort_values("MD").reset_index(drop=True)
md = w["MD"].to_numpy(float)
X = w["x_gold"].to_numpy(float)
Y = w["y_gold"].to_numpy(float)
Z = w["z_gold"].to_numpy(float)       # negative down
TVD = w["TVDrkb"].to_numpy(float)     # positive down


def well_at(md_query):
    return np.array([np.interp(md_query, md, X),
                     np.interp(md_query, md, Y),
                     np.interp(md_query, md, Z)])


center = well_at(MD_CENTER)
tvd_center = np.interp(MD_CENTER, md, TVD)

# well tangent (unit) at the centre via finite difference in MD
p_minus, p_plus = well_at(MD_CENTER - 10), well_at(MD_CENTER + 10)
tangent = (p_plus - p_minus)
tangent /= np.linalg.norm(tangent)

# --- fault geometry -----------------------------------------------------------
FAULTS = {
    "fault1 (tensile T1-T3)": dict(strike=-0.8, dip=90, L=3840, H=4000, color="tab:red"),
    "fault2 (shear T2-T3)":  dict(strike=+0.6, dip=90, L=3800, H=4000, color="tab:blue"),
}

print("=" * 74)
print(f"Fault centre pinned at fiber MD {MD_CENTER:.0f} ft")
print(f"  global (x,y,z)_gold = ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) ft")
print(f"  TVD = {tvd_center:.1f} ft")
print(f"Well tangent at centre (unit, x,y,z) = "
      f"({tangent[0]:.3f}, {tangent[1]:.3f}, {tangent[2]:.3f})")
well_incl = np.degrees(np.arccos(abs(tangent[2])))  # from vertical
well_incl_from_horiz = 90 - well_incl
print(f"  well inclination from horizontal ~ {well_incl_from_horiz:.1f} deg "
      f"(azimuth in X-Y ~ {np.degrees(np.arctan2(tangent[1], tangent[0])):.1f} deg from +X)")

rects = {}
for name, f in FAULTS.items():
    x1, x2, x3 = fault_axes(f["strike"], f["dip"])
    rects[name] = fault_rectangle(center, f["strike"], f["dip"], f["L"], f["H"])
    ang_tan_normal = np.degrees(np.arccos(np.clip(abs(np.dot(tangent, x3)), 0, 1)))
    ang_well_to_plane = 90 - ang_tan_normal
    # signed distance of every well node to the fault plane; find MD of piercing
    d = (np.column_stack([X, Y, Z]) - center) @ x3
    sign_change = np.where(np.diff(np.sign(d)) != 0)[0]
    pierce_md = [float(np.interp(0, [d[i], d[i + 1]], [md[i], md[i + 1]])) for i in sign_change]
    print("-" * 74)
    print(f"{name}: strike={f['strike']:+.1f} deg, dip={f['dip']}, "
          f"L={f['L']} ft, H={f['H']} ft")
    print(f"  fault normal (x,y,z) = ({x3[0]:.3f}, {x3[1]:.3f}, {x3[2]:.3f})")
    print(f"  angle between well axis and fault PLANE = {ang_well_to_plane:.1f} deg "
          f"(90 deg = fiber perpendicular to plane)")
    print(f"  fiber pierces fault plane at MD = "
          f"{', '.join(f'{m:.0f}' for m in pierce_md) or 'n/a'} ft")

# --- optional overlays --------------------------------------------------------
proj = np.load(PROJ_NPZ)
proj_xyz = np.column_stack([proj["xaxis"], proj["yaxis"], -proj["zaxis"]])  # z->negative down
proj_md = proj["data"]
print("-" * 74)
print(f"projection_data_gold4pb: {len(proj_md)} points, MD {proj_md.min():.0f}-{proj_md.max():.0f} ft "
      f"(same X-Y frame as well)")

# -----------------------------------------------------------------------------
# Figure: 4 panels
fig = plt.figure(figsize=(17, 11))

# (A) plan view, full well
axA = fig.add_subplot(2, 2, 1)
axA.plot(X, Y, "k-", lw=1, label="Gold 4-PB well")
for name, f in FAULTS.items():
    r = rects[name]
    axA.plot(r[:, 0], r[:, 1], "-", color=f["color"], lw=2, label=name)
axA.plot(*center[:2], "y*", ms=16, mec="k", label=f"fault centre (MD {MD_CENTER:.0f})")
axA.scatter(proj_xyz[:, 0], proj_xyz[:, 1], c="green", marker="^", s=40,
            label="projected frac hits", zorder=5)
axA.set(xlabel="x_gold (ft, ~East)", ylabel="y_gold (ft, ~North)",
        title="(A) Plan view - full well")
axA.axis("equal"); axA.grid(alpha=0.3); axA.legend(fontsize=8, loc="best")

# (B) plan view, zoom around fault
axB = fig.add_subplot(2, 2, 2)
w_win = (md >= MD_TOP - 400) & (md <= MD_BOT + 400)
axB.plot(X[w_win], Y[w_win], "k-", lw=2.2, label="well (MD 9800-10900)")
for md_tick in [MD_TOP, MD_CENTER, MD_BOT]:
    p = well_at(md_tick)
    axB.plot(p[0], p[1], "ko", ms=4)
    axB.annotate(f"MD {md_tick:.0f}", (p[0], p[1]), fontsize=8,
                 textcoords="offset points", xytext=(4, 4))
for name, f in FAULTS.items():
    r = rects[name]
    axB.plot(r[:, 0], r[:, 1], "-", color=f["color"], lw=2, label=name)
axB.plot(*center[:2], "y*", ms=18, mec="k")
axB.set(xlabel="x_gold (ft)", ylabel="y_gold (ft)",
        title="(B) Plan view - zoom on fault / DAS window")
axB.axis("equal"); axB.grid(alpha=0.3); axB.legend(fontsize=8)

# (C) depth section: x_gold vs TVD near fault
axC = fig.add_subplot(2, 2, 3)
axC.plot(X, TVD, "k-", lw=1, label="Gold 4-PB well")
w_win2 = (md >= MD_TOP) & (md <= MD_BOT)
axC.plot(X[w_win2], TVD[w_win2], "r-", lw=3, label="DAS window MD 10200-10500")
# vertical fault (dip 90): draw its TVD extent at the fault's x, centred on tvd_center
for name, f in FAULTS.items():
    fx = center[0]
    axC.plot([fx, fx], [tvd_center - f["H"] / 2, tvd_center + f["H"] / 2],
             color=f["color"], lw=2, alpha=0.6, label=f"{name} (H={f['H']} ft)")
axC.plot(center[0], tvd_center, "y*", ms=18, mec="k")
axC.set(xlabel="x_gold (ft)", ylabel="TVD (ft)", title="(C) Depth section (X vs TVD)")
axC.set_ylim(tvd_center + 2300, tvd_center - 2300)
axC.grid(alpha=0.3); axC.legend(fontsize=8)

# (D) 3D view
axD = fig.add_subplot(2, 2, 4, projection="3d")
axD.plot(X, Y, TVD, "k-", lw=1, label="well")
axD.plot(X[w_win2], Y[w_win2], TVD[w_win2], "r-", lw=3, label="DAS window")
for name, f in FAULTS.items():
    r = rects[name].copy()
    r_tvd = tvd_center + (r[:, 2] - center[2]) * -1  # z_gold(neg down) -> TVD(pos down) about centre
    axD.plot(r[:, 0], r[:, 1], r_tvd, "-", color=f["color"], lw=1.5, label=name)
axD.scatter(*center[:2], tvd_center, c="y", marker="*", s=120, edgecolor="k")
axD.set(xlabel="x_gold", ylabel="y_gold", zlabel="TVD (ft)", title="(D) 3D")
axD.invert_zaxis(); axD.legend(fontsize=7); axD.view_init(elev=18, azim=-60)

fig.suptitle("V1 fault vs Gold 4-PB monitoring well - reconstructed DDM geometry",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.98))
out = OUT_DIR / "v1_fracture_vs_well_geometry.png"
fig.savefig(out, dpi=130)
print("=" * 74)
print("Saved:", out)
