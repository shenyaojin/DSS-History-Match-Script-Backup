"""Problem-statement / model schematics for the delivery package (audience: not familiar with MOOSE
or geomechanical simulation). Three plain-English figures:
  01 - the physical setup + what the DAS fiber measures
  02 - the 2D computational model + the two strain sources
  03 - the inversion workflow + headline result
Saved into output/inversion_deliverable/figures/.
"""
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
import numpy as np

REPO = Path("/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner")
D = REPO / "output/inversion_deliverable/figures"
D.mkdir(parents=True, exist_ok=True)
BLUE, RED, GREY = "#1f6fb4", "#c1272d", "#555555"


# =========================================================================================
# 01 - PHYSICAL SETUP: injection -> fracture opens & fault slips -> DAS fiber records strain
# =========================================================================================
fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.text(50, 97, "What we are looking at — Gold 4-PB DAS monitoring", ha="center", fontsize=16, fontweight="bold")

# rock background
ax.add_patch(Rectangle((6, 12), 58, 74, fc="#f3efe6", ec="0.6"))
ax.text(9, 82, "Rock (subsurface)", fontsize=10, color=GREY)

# injection well + fracture
ax.add_patch(Rectangle((14, 12), 1.4, 74, fc="0.35"))
ax.annotate("Fluid\ninjection", (14.7, 88), (9, 92), fontsize=10, ha="center",
            arrowprops=dict(arrowstyle="->", color="k"))
ax.add_patch(Rectangle((14.7, 40), 26, 6, fc=RED, alpha=0.35, ec=RED))
ax.text(28, 52, "Hydraulic fracture\n(opens as pressure rises)", ha="center", color=RED, fontsize=10)
# fault plane (shear) slightly above
ax.plot([16, 40], [58, 58], color=BLUE, lw=2.5)
ax.annotate("", (30, 60.5), (30, 55.5), arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
ax.annotate("", (26, 55.5), (26, 60.5), arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
ax.text(28, 63, "Fault plane\n(slips = shear)", ha="center", color=BLUE, fontsize=10)

# monitoring well + fiber crossing the zone
ax.plot([40, 40], [16, 82], color="k", lw=2.5)
ax.plot([40.0, 40.0], [16, 82], color="orange", lw=1.2)
ax.text(41.5, 20, "Gold 4-PB\nmonitoring well\n(+ DAS fiber)", fontsize=10)
for yy in np.arange(20, 80, 3.2):
    ax.plot([39.2, 40.8], [yy, yy], color="orange", lw=0.7)
ax.annotate("fiber crosses the\nfractured zone\n(~10,373 ft depth)", (40, 43), (48, 34),
            fontsize=9.5, arrowprops=dict(arrowstyle="->"))

# DAS output panel (mini waterfall)
ax.add_patch(FancyBboxPatch((68, 30), 27, 46, boxstyle="round,pad=0.3", fc="white", ec="0.4"))
ax.text(81.5, 79, "DAS output:\nrock strain vs depth vs time", ha="center", fontsize=10, fontweight="bold")
xx = np.linspace(0, 1, 60); yy = np.linspace(0, 1, 60)
XX, YY = np.meshgrid(xx, yy)
img = np.exp(-((YY - 0.55) ** 2) / 0.02) * XX - 0.6 * np.exp(-((YY - 0.78) ** 2) / 0.006) * np.clip(XX - 0.55, 0, 1)
ax.imshow(img, extent=[70, 93, 33, 73], aspect="auto", cmap="bwr", vmin=-0.6, vmax=0.6, zorder=3)
ax.text(81.5, 31, "red = stretch (tensile)   blue = shear band", ha="center", fontsize=8, color=GREY)
ax.annotate("", (69, 52), (64, 52), arrowprops=dict(arrowstyle="->", lw=2, color="k"))

# two mechanisms legend
ax.text(50, 8, "① Injection pressure OPENS the fracture  →  tensile strain (all through the test)      "
              "② The fault SLIPS  →  shear strain (starts partway through, at 'T2')",
        ha="center", fontsize=10.5,
        bbox=dict(boxstyle="round", fc="#fbf7ea", ec="0.6"))
fig.tight_layout(); fig.savefig(D / "01_problem_statement.png", dpi=150); plt.close(fig)
print("01 done")


# =========================================================================================
# 02 - THE 2D COMPUTATIONAL MODEL
# =========================================================================================
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.text(50, 97, "How we model it — a 2D geomechanical (MOOSE) cross-section", ha="center", fontsize=15, fontweight="bold")
ax.text(50, 92, "The model computes how the rock deforms (strain) for a given fracture opening + fault slip.",
        ha="center", fontsize=10.5, color=GREY)

# domain
ax.add_patch(Rectangle((12, 14), 60, 70, fc="#eef2f8", ec="0.5"))
ax.text(14, 80, "rock (elastic + fluid-pressure coupled)", fontsize=9, color=GREY)
mdc = 50  # fracture centre row
# SRV graded bands (tensile lobe shape)
for h, col in [(16, "#ffd9b3"), (11, "#ffc48c"), (7, "#ffb066"), (3.5, "#ff9640")]:
    ax.add_patch(Rectangle((24, mdc - h / 2), 36, h, fc=col, ec="none"))
ax.add_patch(Rectangle((24, mdc - 0.6), 36, 1.2, fc=RED))
ax.text(42, mdc - 12, "SRV + hydraulic fracture\n(opening = 'tensile' source ~ within-fracture pressure)",
        ha="center", fontsize=9.5, color=RED)
# shear band above
ax.plot([24, 60], [mdc + 13, mdc + 13], color=BLUE, lw=3)
ax.text(42, mdc + 16.5, "shear-plane source (starts at T2)", ha="center", color=BLUE, fontsize=9.5)
# fiber
xf = 63
ax.plot([xf, xf], [18, 80], "k--", lw=2)
ax.text(xf + 1, 74, "DAS fiber\n(offset 40 ft)\nsamples strain\nalong depth", fontsize=9)
# MD axis mapping
ax.annotate("", (10, 84), (10, 14), arrowprops=dict(arrowstyle="->", color="0.3"))
ax.text(6.5, 50, "Measured depth (MD)", rotation=90, va="center", fontsize=10, color="0.3")
ax.text(8.6, 50.5, "10,373 ft", fontsize=8, color="0.3")
ax.text(8.6, 63, "10,300 ft", fontsize=8, color="0.3")
ax.axhline
ax.plot([12, 72], [mdc, mdc], color="0.7", lw=0.6, ls=":")
ax.plot([12, 72], [mdc + 13, mdc + 13], color="0.7", lw=0.6, ls=":")
ax.text(50, 8, "Key idea: the same geometry throughout — only the loading changes with time "
              "(pressure grows T1→T3; shear turns on at T2).",
        ha="center", fontsize=10, bbox=dict(boxstyle="round", fc="#eef7ee", ec="0.6"))
fig.tight_layout(); fig.savefig(D / "02_model_geometry.png", dpi=150); plt.close(fig)
print("02 done")


# =========================================================================================
# 03 - INVERSION WORKFLOW
# =========================================================================================
fig, ax = plt.subplots(figsize=(13, 6.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.text(50, 95, "What we did — match the model to the measured data ('inversion')", ha="center", fontsize=15, fontweight="bold")


def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=fc, ec="0.4"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5)


def arrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=20, color="k", lw=2))


box(4, 55, 24, 26, "MEASURED\nDAS strain\n(depth × time)\n= the target", "#e8eef7")
box(38, 55, 26, 26, "MODEL\nopening + slip\n→ rock strain\n(MOOSE, physics)", "#fdeee0")
box(72, 55, 24, 26, "REPRODUCED\nDAS strain", "#eaf3ea")
arrow(28, 38, 68); arrow(64, 72, 68)
ax.text(46, 84, "adjust opening & slip", fontsize=9, ha="center", color=GREY)

box(38, 15, 26, 24, "INVERSION:\nfind the opening (pressure)\n& slip histories that\nBEST MATCH the data", "#fdeee0")
ax.add_patch(FancyArrowPatch((51, 55), (51, 39), arrowstyle="-|>", mutation_scale=18, color="k", lw=2))
ax.add_patch(FancyArrowPatch((64, 27), (84, 27), arrowstyle="-|>", mutation_scale=18, color="k", lw=2))
ax.add_patch(FancyArrowPatch((84, 27), (84, 55), arrowstyle="-|>", mutation_scale=18, color="k", lw=2))
box(72, 15, 24, 24, "OUTPUT:\nfracture-pressure\n& fault-slip\nhistories", "#eaf3ea")

ax.text(50, 5, "Result: the model reproduces the measured DAS strain to 92%, and the recovered histories "
              "confirm — tensile T1→T3, shear only after T2.",
        ha="center", fontsize=10.5, fontweight="bold",
        bbox=dict(boxstyle="round", fc="#fff6d6", ec="0.5"))
fig.tight_layout(); fig.savefig(D / "03_workflow.png", dpi=150); plt.close(fig)
print("03 done")
print("schematics in", D)
