"""Model geometry figure for the V2 graded low-perm SRV MOOSE model.

Left  : 2-D section (plane strain) with the nested zones coloured by permeability, the DAS
        fiber sampling line, and the DDM shear plane.
Right : permeability vs MD (step profile) — makes the outward grading explicit.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[3]
FIG = REPO / "figs" / "tensile_fault_qc" / "v3_srv_asym"
FIG.mkdir(parents=True, exist_ok=True)

FT = 0.3048
STAR = 10373.4                       # MD at model y = 0
md = lambda y: STAR + y / FT         # noqa: E731
ytop, ybot = -30.0, 30.0             # plotted y window (m)  -> MD 10275 .. 10472

SRV_X0, SRV_X1 = 57.328, 142.672     # SRV zones x-extent (m)
HF_X0, HF_X1 = 61.9, 138.1           # hydraulic fracture x-extent (m)
FIBER_X = 125.3998984                # fiber sampling line (m)  = 83.33 ft from the fracture
SHEAR_MD = 10319.0                   # DDM fault2 doublet centre
SHEAR_Y = (SHEAR_MD - STAR) * FT

# name, half-height (m), perm, colour
# ASYMMETRIC zones: (name, top_MD, bottom_MD, perm, colour)
ZONES_MD = [
    ("srv_outer",  10308.0, 10400.0, 1e-17, "#aed6f1"),
    ("srv_wide",   10328.0, 10396.0, 3e-17, "#5dade2"),
    ("srv_narrow", 10351.0, 10390.0, 1e-16, "#2e86c1"),
]
y_of = lambda m: (m - STAR) * FT

fig, (ax, axp) = plt.subplots(1, 2, figsize=(15, 7.5), width_ratios=[2.4, 1],
                              constrained_layout=True)

# ---------------- left: 2-D section ----------------
ax.add_patch(Rectangle((0, ytop), 200, ybot - ytop, facecolor="#ecf0f1", edgecolor="k", lw=0.8))
for name, t_md, b_md, k, c in ZONES_MD:
    y0, y1 = y_of(t_md), y_of(b_md)
    ax.add_patch(Rectangle((SRV_X0, y0), SRV_X1 - SRV_X0, y1 - y0,
                           facecolor=c, edgecolor="k", lw=0.9))
    ax.text(SRV_X0 + 3, y0 + 1.1, f"{name}  k={k:.0e}   MD {t_md:.0f}–{b_md:.0f}",
            fontsize=8.5, va="bottom")
# DDM shear plane
ax.axhline(SHEAR_Y, color="#8e44ad", ls="--", lw=2.2)
ax.text(4, SHEAR_Y - 1.2, f"DDM shear plane  MD {SHEAR_MD:.0f}  →  now INSIDE srv_outer",
        color="#8e44ad", fontsize=9.5, fontweight="bold")

# fiber
ax.plot([FIBER_X, FIBER_X], [ytop, ybot], color="#117864", lw=2.4)
ax.text(FIBER_X + 2.5, ytop + 2.5, "DAS fiber\n(x = 125.4 m,\n83.3 ft offset)",
        color="#117864", fontsize=9, va="top")

ax.set_xlim(0, 200); ax.set_ylim(ybot, ytop)          # invert: larger MD downward
ax.set_xlabel("x  [m]  (along fracture / strike)")
ax.set_ylabel("model y  [m]")
ax.set_title("SRV geometry — asymmetric, outward-graded permeability (2-D plane strain)", fontweight="bold")
sec = ax.secondary_yaxis("right", functions=(md, lambda m: (m - STAR) * FT))
sec.set_ylabel("Measured depth  [ft]")

# ---------------- right: permeability vs MD ----------------
yy = np.linspace(ytop, ybot, 2000)
kk = np.full_like(yy, 1e-18)
for _n, t_md, b_md, k, _c in ZONES_MD:
    kk[(yy >= y_of(t_md)) & (yy <= y_of(b_md))] = k
axp.step(kk, md(yy), where="mid", color="#2c3e50", lw=2)
axp.set_xscale("log"); axp.set_xlim(3e-19, 1e-15)
axp.set_ylim(md(ybot), md(ytop))
axp.axhline(SHEAR_MD, color="#8e44ad", ls="--", lw=2)
axp.text(4e-19, SHEAR_MD - 2, "shear plane", color="#8e44ad", fontsize=9, fontweight="bold")
axp.set_xlabel("permeability  [m$^2$]"); axp.set_ylabel("Measured depth  [ft]")
axp.set_title("permeability grading\n(asymmetric: 65 ft up / 27 ft down)", fontweight="bold")
axp.grid(alpha=0.3, which="both")
for nm, t_md, b_md, k, _c in ZONES_MD:
    axp.text(k * 1.25, t_md + 3, nm, fontsize=8, color="#2c3e50")

fig.suptitle("SRV zones and their permeability", fontsize=13, fontweight="bold")
fig.savefig(FIG / "v3_srv_geometry.png", dpi=150)
print("saved", FIG / "v3_srv_geometry.png")
