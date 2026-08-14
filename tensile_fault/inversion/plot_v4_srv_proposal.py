"""V4 SRV proposal: SRV centred on the fracture, outer boundary ON the DDM shear plane.

Design intent (Shenyao): the MOOSE fracture marks the CENTRE of the SRV, because the
poroelastic tensile response fills the whole SRV; the DDM shear plane rides the
SRV/matrix BOUNDARY, because that is where shear localises. V3 honours neither -- it is
deliberately asymmetric (65.4 ft up / 26.6 ft down) and its boundary sits 11 ft past the
shear plane.

V4 = V2's symmetric graded stack, rescaled by 108.8/130 so srv_outer's top lands exactly
on the shear plane at MD 10319 and the fracture sits at its centre. Permeabilities are
unchanged from V2/V3, so geometry is the only variable.

NOTE the lineage before deciding: V2 was already symmetric about the fracture
(10308.4-10438.4) and was replaced by V3 precisely because that symmetric extent
over-produced the deep poroelastic compression lobe below ~MD 10430. V4 is 10.6 ft
shallower at the base than V2, so it is a partial -- not a full -- retreat from V3.

This script only DRAWS the setup; it does not run MOOSE.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[3]
WELL_CSV = REPO / "data_fervo" / "legacy" / "Gold_4_PB_Well_Geometry.csv"
STIM_NPZ = REPO / "data_fervo" / "fiberis_format" / "stimulation_loc_bearskin.npz"
FIG = REPO / "figs" / "tensile_fault_qc" / "v4_srv_centred"
FIG.mkdir(parents=True, exist_ok=True)

FT = 0.3048
STAR = 10373.4          # fracture / model y = 0 / intended SRV centre
SHEAR_MD = 10319.0      # DDM fault2 doublet centre -> intended SRV/matrix boundary
STRIKE1 = -0.8
MD_LO, MD_HI = 10200.0, 10500.0
SRV_X0, SRV_X1 = 57.328, 142.672
HF_X0, HF_X1 = 61.9, 138.1
FIBER_X, INJ_X = 125.3998984, 100.0

# --- the three SRV generations, as (name, top_MD, bottom_MD, perm) ------------
V3 = [("srv_outer", 10308.0, 10400.0, 1e-17),
      ("srv_wide", 10328.0, 10396.0, 3e-17),
      ("srv_narrow", 10351.0, 10390.0, 1e-16)]
V2_HEIGHTS = [("srv_outer", 130.0, 1e-17), ("srv_wide", 90.0, 3e-17),
              ("srv_narrow", 45.0, 1e-16)]          # symmetric about STAR
V2 = [(n, STAR - h / 2, STAR + h / 2, k) for n, h, k in V2_HEIGHTS]

# V4: rescale V2 so srv_outer's top lands exactly on the shear plane, still centred.
SCALE = 2 * (STAR - SHEAR_MD) / V2_HEIGHTS[0][1]
V4 = [(n, STAR - h * SCALE / 2, STAR + h * SCALE / 2, k) for n, h, k in V2_HEIGHTS]

FILL = {"srv_outer": "#aed6f1", "srv_wide": "#5dade2", "srv_narrow": "#2e86c1"}
C_V3, C_V4, C_V2 = "#c0392b", "#1a6ea8", "#b8860b"
C_WELL, C_SHEAR, MUTED, INK = "#1a7f5a", "#8e44ad", "#7f8c8d", "#2c3e50"

print(f"V4 rescale factor vs V2 = {SCALE:.4f}\n")
print(f"{'zone':12s} {'V2 (symmetric)':>22s} {'V3 (current)':>22s} {'V4 (proposed)':>22s}")
for (n, t2, b2, k), (_, t3, b3, _k3), (_, t4, b4, _k4) in zip(V2, V3, V4):
    print(f"{n:12s} {t2:9.1f}-{b2:<9.1f} {t3:9.1f}-{b3:<9.1f} {t4:9.1f}-{b4:<9.1f}")
for tag, Z in (("V2", V2), ("V3", V3), ("V4", V4)):
    t, b = Z[0][1], Z[0][2]
    print(f"{tag}: srv_outer {t:.1f}-{b:.1f}  centre {0.5*(t+b):.1f}  "
          f"fracture offset {STAR - 0.5*(t+b):+.1f} ft  "
          f"top vs shear plane {t - SHEAR_MD:+.1f} ft")

# --- well geometry in the fault frame -----------------------------------------
w = pd.read_csv(WELL_CSV).sort_values("MD")
md_s = w["MD"].to_numpy(float)
XYZ = w[["x_gold", "y_gold", "z_gold"]].to_numpy(float)
TVD = w["TVDrkb"].to_numpy(float)


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


R_inv = np.linalg.inv(_rotate_matrix(np.deg2rad(STRIKE1), np.deg2rad(90.0)))
e_strike, _e_dip, e_norm = [(R_inv @ e) / np.linalg.norm(R_inv @ e)
                            for e in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]
at = lambda q: np.array([np.interp(q, md_s, XYZ[:, i]) for i in range(3)])  # noqa: E731
pierce = at(STAR)

fig, axs = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
(axA, axB), (axC, axD) = axs
ytop, ybot = -32.0, 32.0


def draw_section(ax, zones, title, tag_colour):
    y_of = lambda q: (q - STAR) * FT                                   # noqa: E731
    ax.add_patch(Rectangle((0, ytop), 200, ybot - ytop, facecolor="#f4f6f7",
                           edgecolor=MUTED, lw=0.8))
    for nm, t, b, k in zones:
        ax.add_patch(Rectangle((SRV_X0, y_of(t)), SRV_X1 - SRV_X0, y_of(b) - y_of(t),
                               facecolor=FILL[nm], edgecolor="white", lw=2.0))
        ax.text(SRV_X0 + 3, y_of(t) + 0.8, f"{nm}  k={k:.0e}  MD {t:.0f}-{b:.0f}",
                fontsize=8, va="top", color=INK)
    ax.add_patch(Rectangle((HF_X0, -0.35), HF_X1 - HF_X0, 0.7, facecolor="#154360"))
    ax.text(HF_X1 - 2, -1.0, "hf core  k=1e-13", fontsize=8.5, color="#154360",
            va="bottom", ha="right", fontweight="bold")
    ax.plot(INJ_X, 0.0, "o", ms=9, color="#154360", mec="white", mew=1.5, zorder=6)

    top = zones[0][1]
    ax.axhline(y_of(SHEAR_MD), color=C_SHEAR, ls="--", lw=2.0)
    ax.text(2, y_of(SHEAR_MD) - 1.0,
            f"DDM shear plane\nMD {SHEAR_MD:.0f}   (SRV edge {top - SHEAR_MD:+.0f} ft)",
            color=C_SHEAR, fontsize=8.5, fontweight="bold", va="bottom")
    centre = 0.5 * (zones[0][1] + zones[0][2])
    ax.axhline(y_of(centre), color=tag_colour, ls=":", lw=2.0)
    lbl = (f"SRV centre = fracture\nMD {centre:.0f}" if abs(STAR - centre) < 0.5
           else f"SRV centre  MD {centre:.0f}\n(fracture {STAR - centre:+.0f} ft below)")
    ax.text(2, y_of(centre) + 1.2, lbl, color=tag_colour, fontsize=8.5,
            fontweight="bold", va="top")

    ax.plot([FIBER_X, FIBER_X], [ytop, ybot], color=C_WELL, lw=2.5, zorder=5)
    ax.text(FIBER_X + 3, ytop + 2.5, "DAS fiber", color=C_WELL, fontsize=9.5,
            va="top", fontweight="bold")
    ax.set_xlim(0, 200)
    ax.set_ylim(ybot, ytop)
    ax.set_xlabel("x  [m]   (along fault strike)")
    ax.set_ylabel("model y  [m]   (fault normal)")
    ax.set_title(title, fontweight="bold", fontsize=11, color=tag_colour)
    sec = ax.secondary_yaxis("right", functions=(lambda y: STAR + y / FT,
                                                 lambda m: (m - STAR) * FT))
    sec.set_ylabel("Measured depth  [ft]")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


draw_section(axA, V3, "(A) V3 — CURRENT (asymmetric: 65 ft up / 27 ft down)", C_V3)
draw_section(axB, V4, "(B) V4 — PROPOSED (centred on the fracture, boundary on the shear plane)", C_V4)

# --- (C) permeability vs MD, all three generations ----------------------------
mdq = np.linspace(MD_LO, MD_HI, 4000)
for zones, colour, tag, ls, lw in ((V2, C_V2, "V2 (symmetric, superseded)", "--", 1.8),
                                   (V3, C_V3, "V3 (current)", "-", 2.4),
                                   (V4, C_V4, "V4 (proposed)", "-", 2.4)):
    kk = np.full_like(mdq, 1e-18)
    for _n, t, b, k in zones:
        kk[(mdq >= t) & (mdq <= b)] = k
    axC.step(mdq, kk, where="mid", color=colour, lw=lw, ls=ls, label=tag)
axC.axvline(SHEAR_MD, color=C_SHEAR, ls="--", lw=1.8)
axC.text(SHEAR_MD + 2, 1.4e-16, "shear plane", color=C_SHEAR, fontsize=9, fontweight="bold")
axC.axvline(STAR, color=INK, ls=":", lw=1.5)
axC.text(STAR + 2, 3e-18, "fracture", color=INK, fontsize=9, fontweight="bold")
axC.set_yscale("log")
axC.set_ylim(3e-19, 4e-16)
axC.set_xlim(10280, 10460)
axC.set_xlabel("Measured depth  [ft]")
axC.set_ylabel("permeability  [m$^2$]")
axC.set_title("(C) Permeability grading — V4 vs the two it descends from", fontweight="bold",
              fontsize=11)
axC.legend(loc="upper left", fontsize=9, frameon=False)
axC.grid(alpha=0.25, which="both")
for s in ("top", "right"):
    axC.spines[s].set_visible(False)

# --- (D) well geometry, plan view ---------------------------------------------
axD.plot(XYZ[:, 0], XYZ[:, 1], "-", color=MUTED, lw=1.4, label="Gold 4-PB (monitoring)")
win = (md_s >= MD_LO) & (md_s <= MD_HI)
axD.plot(XYZ[win, 0], XYZ[win, 1], "-", color=C_WELL, lw=4,
         label=f"DAS window MD {MD_LO:.0f}-{MD_HI:.0f}")
half_L = 3840 / 2
trace = np.array([pierce + s * e_strike * half_L for s in (-1, 1)])
axD.plot(trace[:, 0], trace[:, 1], "--", color=C_SHEAR, lw=2.2,
         label="V1 fault trace (strike -0.8$\\degree$, L=3840 ft)")
axD.plot(*pierce[:2], "*", ms=20, color=C_V4, mec="white", mew=1.2, zorder=6,
         label=f"fiber pierces fault, MD {STAR:.0f}")
stim = np.load(STIM_NPZ, allow_pickle=True)
axD.plot(stim["xaxis"], stim["yaxis"], "^", ms=9, color=C_V2, mec="white", mew=1.0,
         ls="none", label="Bearskin stimulation points")
for q, dy in ((MD_LO, -16), (MD_HI, 8)):
    p = at(q)
    axD.annotate(f"MD {q:.0f}", p[:2], fontsize=8.5, color=INK,
                 textcoords="offset points", xytext=(6, dy))
sep = float(np.mean(stim["yaxis"])) - pierce[1]
axD.annotate("", xy=(pierce[0], float(np.mean(stim["yaxis"]))), xytext=tuple(pierce[:2]),
             arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.4))
axD.text(pierce[0] - 60, pierce[1] + sep / 2, f"{sep:.0f} ft", fontsize=9, color=MUTED,
         rotation=90, va="center", ha="right", fontweight="bold")
axD.set_xlim(-600, 1600)
axD.set_ylim(200, 2900)
axD.set_xlabel("x_gold  [ft]  (~East)")
axD.set_ylabel("y_gold  [ft]  (~North)")
axD.set_title("(D) Well geometry — plan view, zoomed on the fault crossing", fontweight="bold",
              fontsize=11)
axD.legend(fontsize=8.5, loc="upper left", frameon=False)
axD.grid(alpha=0.25)
axD.set_aspect("equal")
for s in ("top", "right"):
    axD.spines[s].set_visible(False)

fig.suptitle("V4 SRV proposal — fracture-centred, boundary on the shear plane",
             fontsize=14, fontweight="bold")
out = FIG / "v4_srv_proposal.png"
fig.savefig(out, dpi=140)
print("\nsaved", out)
