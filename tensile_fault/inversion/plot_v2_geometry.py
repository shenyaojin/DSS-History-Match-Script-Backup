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
FIG = REPO / "figs" / "tensile_fault_qc" / "v2_srv_graded"
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
ZONES = [
    ("matrix",     None,   1e-18, "#ecf0f1"),
    ("srv_outer",  19.812, 1e-17, "#aed6f1"),
    ("srv_wide",   13.716, 3e-17, "#5dade2"),
    ("srv_narrow",  6.858, 1e-16, "#2e86c1"),
]

fig, (ax, axp) = plt.subplots(1, 2, figsize=(15, 7.5), width_ratios=[2.4, 1],
                              constrained_layout=True)

# ---------------- left: 2-D section ----------------
ax.add_patch(Rectangle((0, ytop), 200, ybot - ytop, facecolor=ZONES[0][3], edgecolor="k", lw=0.8))
for name, hh, k, c in ZONES[1:]:
    ax.add_patch(Rectangle((SRV_X0, -hh), SRV_X1 - SRV_X0, 2 * hh,
                           facecolor=c, edgecolor="k", lw=0.9))
    ax.text(SRV_X0 + 3, -hh + 1.1,
            f"{name}  k={k:.0e}   MD {md(-hh):.0f}–{md(hh):.0f}", fontsize=8.5, va="bottom")
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
ax.set_title("SRV geometry — symmetric, outward-graded permeability (2-D plane strain)", fontweight="bold")
sec = ax.secondary_yaxis("right", functions=(md, lambda m: (m - STAR) * FT))
sec.set_ylabel("Measured depth  [ft]")

# ---------------- right: permeability vs MD ----------------
edges, perms = [], []
for name, hh, k, c in ZONES[1:][::-1]:                 # inner -> outer
    edges.append(hh); perms.append(k)
yy = np.linspace(ytop, ybot, 2000)
kk = np.full_like(yy, 1e-18)
for hh, k in zip([19.812, 13.716, 6.858], [1e-17, 3e-17, 1e-16]):
    kk[np.abs(yy) <= hh] = k
axp.step(kk, md(yy), where="mid", color="#2c3e50", lw=2)
axp.set_xscale("log"); axp.set_xlim(3e-19, 1e-15)
axp.set_ylim(md(ybot), md(ytop))
axp.axhline(SHEAR_MD, color="#8e44ad", ls="--", lw=2)
axp.text(4e-19, SHEAR_MD - 2, "shear plane", color="#8e44ad", fontsize=9, fontweight="bold")
axp.set_xlabel("permeability  [m$^2$]"); axp.set_ylabel("Measured depth  [ft]")
axp.set_title("permeability grading\n(symmetric: 65 ft up / 65 ft down)", fontweight="bold")
axp.grid(alpha=0.3, which="both")
for hh, k, nm in zip([6.858, 13.716, 19.812], [1e-16, 3e-17, 1e-17],
                     ["srv_narrow", "srv_wide", "srv_outer"]):
    axp.text(k * 1.25, md(-hh) + 3, nm, fontsize=8, color="#2c3e50")

fig.suptitle("SRV zones and their permeability", fontsize=13, fontweight="bold")
fig.savefig(FIG / "v2_srv_geometry.png", dpi=150)
print("saved", FIG / "v2_srv_geometry.png")
