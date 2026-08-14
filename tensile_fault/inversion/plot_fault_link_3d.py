"""Does the V1 fault actually connect Bearskin 3-PA to the fiber crossing at MD 10373?

The question matters because the whole "injection -> fault -> fiber" picture rests on it: if
the stimulated stage does not lie within the fault's footprint, pressure has no modelled path
to the monitoring well and extending the MOOSE domain along strike buys nothing.

Everything is drawn in the DDM fault frame built from Pengchao's fault1 parameters
(strike -0.8 deg, dip 90, L = 3840 ft, H = 4000 ft, centre pinned at well MD 10340).

Panels
  (A) 3-D view: Gold 4-PB lateral, the fault plane, the Bearskin 3-PA stages
  (B) the fault plane seen FACE-ON (along-strike vs TVD) -- the deciding panel
  (C) plan view
  (D) how far each stage sits off the fault plane
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[3]
WELL_CSV = REPO / "data_fervo" / "legacy" / "Gold_4_PB_Well_Geometry.csv"
STIM_NPZ = REPO / "data_fervo" / "fiberis_format" / "stimulation_loc_bearskin.npz"
FIG = REPO / "figs" / "tensile_fault_qc" / "v5_injector_scale"
FIG.mkdir(parents=True, exist_ok=True)

STAR = 10373.4              # fiber pierces the fault here
DDM_CENTRE_MD = 10340.0     # where DDMpy pins the fault centre, on the well
STRIKE1, DIP = -0.8, 90.0
FAULT_L, FAULT_H = 3840.0, 4000.0
MD_LO, MD_HI = 10200.0, 10500.0

C_WELL, C_CUR, C_NEW, C_INJ, C_FAULT = "#1a7f5a", "#c0392b", "#1a6ea8", "#b8860b", "#8e44ad"
MUTED, INK = "#7f8c8d", "#2c3e50"


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


R_inv = np.linalg.inv(_rotate_matrix(np.deg2rad(STRIKE1), np.deg2rad(DIP)))
e_strike, e_dip, e_norm = [(R_inv @ e) / np.linalg.norm(R_inv @ e)
                           for e in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]

w = pd.read_csv(WELL_CSV).sort_values("MD")
md_s = w["MD"].to_numpy(float)
XYZ = w[["x_gold", "y_gold", "z_gold"]].to_numpy(float)     # z negative down
TVD = w["TVDrkb"].to_numpy(float)
at = lambda q: np.array([np.interp(q, md_s, XYZ[:, i]) for i in range(3)])  # noqa: E731

centre = at(DDM_CENTRE_MD)                 # fault centre (DDM pin)
cross = at(STAR)                           # fiber piercing point
tvd_centre = float(np.interp(DDM_CENTRE_MD, md_s, TVD))

stim = np.load(STIM_NPZ, allow_pickle=True)
S = np.column_stack([stim["xaxis"], stim["yaxis"], -stim["zaxis"].astype(float)])
stage_md = np.asarray(stim["data"], float)


def in_fault_frame(P):
    """(along-strike, along-dip, fault-normal) of P relative to the fault centre, ft."""
    r = np.atleast_2d(P) - centre
    return r @ e_strike, r @ e_dip, r @ e_norm


s_s, s_d, s_n = in_fault_frame(S)
c_s, c_d, c_n = in_fault_frame(cross)
half_L, half_H = FAULT_L / 2, FAULT_H / 2
on_plane = int(np.argmin(np.abs(s_n)))

print(f"fault centre  = well MD {DDM_CENTRE_MD:.0f}, TVD {tvd_centre:.0f} ft")
print(f"fault extent  = +/-{half_L:.0f} ft along strike, +/-{half_H:.0f} ft along dip\n")
print(f"fiber crossing (MD {STAR}):  along-strike {c_s[0]:+8.1f}  dip {c_d[0]:+8.1f}  "
      f"normal {c_n[0]:+7.1f}   inside: {abs(c_s[0]) <= half_L and abs(c_d[0]) <= half_H}")
print()
print(f"{'stage MD':>9s} {'strike':>9s} {'dip':>9s} {'normal':>9s}  {'inside footprint?':>18s}")
for i in range(len(S)):
    inside = abs(s_s[i]) <= half_L and abs(s_d[i]) <= half_H
    print(f"{stage_md[i]:9.1f} {s_s[i]:9.1f} {s_d[i]:9.1f} {s_n[i]:9.1f}  {str(inside):>18s}")

gap = abs(s_s[on_plane]) - half_L
print(f"\nDECIDING NUMBER: the on-plane stage (MD {stage_md[on_plane]:.0f}) sits at "
      f"{s_s[on_plane]:+.0f} ft along strike;")
print(f"the fault tip is at {half_L:+.0f} ft -> it is {abs(gap):.0f} ft "
      f"{'BEYOND the tip' if gap > 0 else 'inside the footprint'}.")
print(f"along-strike distance stage -> fiber crossing = {abs(s_s[on_plane] - c_s[0]):.0f} ft")
print(f"fault half-length is {half_L:.0f} ft, i.e. the 3840 ft length was very nearly "
      f"the stage-to-fiber span")

fig = plt.figure(figsize=(17, 12))

# --- (A) 3-D ------------------------------------------------------------------
axA = fig.add_subplot(2, 2, 1, projection="3d")
corners = np.array([centre + a * e_strike * half_L + b * e_dip * half_H
                    for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
ctv = tvd_centre - (corners[:, 2] - centre[2])
axA.plot_trisurf(corners[:, 0], corners[:, 1], ctv, color=C_FAULT, alpha=0.18,
                 linewidth=0, shade=False)
for k in range(4):
    p, q = corners[k], corners[(k + 1) % 4]
    axA.plot([p[0], q[0]], [p[1], q[1]],
             [tvd_centre - (p[2] - centre[2]), tvd_centre - (q[2] - centre[2])],
             color=C_FAULT, lw=1.5)
axA.plot(XYZ[:, 0], XYZ[:, 1], TVD, "-", color=MUTED, lw=1.2, label="Gold 4-PB")
win = (md_s >= MD_LO) & (md_s <= MD_HI)
axA.plot(XYZ[win, 0], XYZ[win, 1], TVD[win], "-", color=C_WELL, lw=4, label="DAS window")
stage_tvd = tvd_centre - (S[:, 2] - centre[2])
axA.scatter(S[:, 0], S[:, 1], stage_tvd, c=C_INJ, marker="^", s=55, depthshade=False,
            label="Bearskin 3-PA stages")
axA.scatter(*cross[:2], np.interp(STAR, md_s, TVD), c=C_NEW, marker="o", s=90,
            depthshade=False, edgecolor="white", label="fiber pierces fault")
axA.set(xlabel="x_gold [ft]", ylabel="y_gold [ft]", zlabel="TVD [ft]")
axA.invert_zaxis()
axA.view_init(elev=20, azim=-72)
axA.set_title("(A) 3-D — fault plane, lateral, stages", fontweight="bold", fontsize=11)
axA.legend(fontsize=7.5, loc="upper left")

# --- (B) fault plane face-on --------------------------------------------------
axB = fig.add_subplot(2, 2, 2)
axB.add_patch(Rectangle((-half_L, -half_H), FAULT_L, FAULT_H, facecolor=C_FAULT,
                        alpha=0.13, edgecolor=C_FAULT, lw=2.2))
axB.text(-half_L + 60, half_H - 160, "V1 fault plane   3840 x 4000 ft", color=C_FAULT,
         fontsize=9.5, fontweight="bold")
sc = axB.scatter(s_s, s_d, c=np.abs(s_n), cmap="YlOrBr_r", s=130, marker="^",
                 edgecolor=INK, lw=.7, vmin=0, vmax=950, zorder=5)
cb = plt.colorbar(sc, ax=axB, pad=0.02)
cb.set_label("distance off the fault plane [ft]", fontsize=9)
axB.plot(c_s[0], c_d[0], "o", ms=13, color=C_NEW, mec="white", mew=1.6, zorder=6)
axB.annotate(f"fiber crosses here\nMD {DDM_CENTRE_MD:.0f} lies ON the plane;\n"
             f"MOOSE fracture MD {STAR} is {c_n[0]:.0f} ft off it",
             (c_s[0], c_d[0]), color=C_NEW, fontsize=9,
             fontweight="bold", ha="left", textcoords="offset points", xytext=(14, -46))
axB.annotate(f"stage MD {stage_md[on_plane]:.0f}\n(on the plane)",
             (s_s[on_plane], s_d[on_plane]), color=C_INJ, fontsize=9, fontweight="bold",
             ha="right", textcoords="offset points", xytext=(-14, 26))
axB.annotate("", xy=(s_s[on_plane], -1450), xytext=(c_s[0], -1450),
             arrowprops=dict(arrowstyle="<->", color=INK, lw=1.7))
axB.text((s_s[on_plane] + c_s[0]) / 2, -1380, f"{abs(s_s[on_plane] - c_s[0]):.0f} ft",
         ha="center", fontsize=10, fontweight="bold", color=INK)
axB.axvline(half_L, color=C_CUR, ls="--", lw=1.8)
axB.text(half_L - 40, 1500, f"fault tip\n{'stage is ' + str(round(abs(gap))) + ' ft past it' if gap > 0 else 'stage inside'}",
         color=C_CUR, fontsize=9, fontweight="bold", ha="right")
axB.set_xlim(-2350, 2350)
axB.set_ylim(-2300, 2300)
axB.set_xlabel("along fault strike, from the fault centre  [ft]")
axB.set_ylabel("along dip (up is shallower)  [ft]")
axB.set_title("(B) The fault plane, face-on — does the stage fall inside?",
              fontweight="bold", fontsize=11)
axB.grid(alpha=0.22)
axB.set_aspect("equal")
for s in ("top", "right"):
    axB.spines[s].set_visible(False)

# --- (C) plan view ------------------------------------------------------------
axC = fig.add_subplot(2, 2, 3)
axC.plot(XYZ[:, 0], XYZ[:, 1], "-", color=MUTED, lw=1.4, label="Gold 4-PB")
axC.plot(XYZ[win, 0], XYZ[win, 1], "-", color=C_WELL, lw=4.5, label="DAS window")
tr = np.array([centre + s * e_strike * half_L for s in (-1, 1)])
axC.plot(tr[:, 0], tr[:, 1], "-", color=C_FAULT, lw=2.6, label="fault trace (3840 ft)")
axC.plot(S[:, 0], S[:, 1], "^", ms=10, color=C_INJ, mec="white", ls="none",
         label="Bearskin 3-PA stages")
axC.plot(S[on_plane, 0], S[on_plane, 1], "*", ms=22, color=C_INJ, mec=INK, mew=1.3,
         ls="none", label=f"stage MD {stage_md[on_plane]:.0f} (on plane)")
axC.plot(*cross[:2], "o", ms=11, color=C_NEW, mec="white", mew=1.3, label="fiber pierces")
axC.set_xlabel("x_gold  [ft]  (~East)")
axC.set_ylabel("y_gold  [ft]  (~North)")
axC.set_title("(C) Plan view", fontweight="bold", fontsize=11)
axC.legend(fontsize=8, loc="lower right", frameon=False)
axC.grid(alpha=0.22)
axC.set_aspect("equal")
axC.set_xlim(-500, 1600)
axC.set_ylim(200, 2950)
for s in ("top", "right"):
    axC.spines[s].set_visible(False)

# --- (D) off-plane distance ---------------------------------------------------
axD = fig.add_subplot(2, 2, 4)
axD.barh(np.arange(len(S)), s_n, color=C_INJ, edgecolor="white", height=0.62)
axD.axvline(0, color=C_FAULT, lw=2.4, ls="--")
axD.text(22, len(S) - 0.35, "fault plane", color=C_FAULT, fontsize=9.5, fontweight="bold")
axD.set_yticks(np.arange(len(S)))
axD.set_yticklabels([f"MD {m:.0f}" for m in stage_md], fontsize=9)
for i in range(len(S)):
    axD.text(1020, i, f"strike {s_s[i]:+.0f} ft", va="center", ha="right", fontsize=8.5,
             color=INK)
axD.set_xlim(-330, 1050)
axD.set_xlabel("distance off the fault plane  [ft]")
axD.set_title("(D) Only one stage is on the plane", fontweight="bold", fontsize=11)
axD.grid(alpha=0.22, axis="x")
for s in ("top", "right", "left"):
    axD.spines[s].set_visible(False)

fig.suptitle("Does the V1 fault connect Bearskin 3-PA to the fiber crossing?",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = FIG / "fault_link_3d.png"
fig.savefig(out, dpi=140)
print("\nsaved", out)
