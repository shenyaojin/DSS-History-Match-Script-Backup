"""Schematics for the MOOSE-DDM vs DDMpy explainer doc.
Generates 3 portable PNGs into output/inversion_deliverable/MOOSE_DDM_figs/.
  s1  DD (displacement jump)  ==  eigenstrain band  (the core equivalence)
  s2  pipeline comparison (DDMpy analytic vs MOOSE FEM)
  s3  the 3D well-projection issue (why 2D MOOSE can't render the shear band)
Language-neutral labels (English + a few symbols) so it drops into both README versions.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, FancyArrowPatch, FancyBboxPatch

OUT = Path("/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/output/inversion_deliverable/MOOSE_DDM_figs")
OUT.mkdir(parents=True, exist_ok=True)
BLUE, RED, ORANGE, GREY = "#2c3e50", "#c0392b", "#e67e22", "#95a5a6"


# ============================================================ s1: DD == eigenstrain band
fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
for ax in axs:
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

# --- left: displacement discontinuity (DDMpy) ---
ax = axs[0]
ax.set_title("DDMpy: displacement discontinuity (dislocation)", fontsize=12, fontweight="bold")
ax.add_patch(Rectangle((1.0, 1.4), 7.0, 3.3, facecolor="#d5dbdb", edgecolor="k"))    # bottom block
ax.add_patch(Rectangle((1.4, 5.2), 7.0, 3.3, facecolor="#d5dbdb", edgecolor="k"))    # top block (opened + slipped)
ax.plot([1.0, 8.0], [4.9, 4.9], color=BLUE, lw=2.2)                                  # lower fault face
ax.plot([1.4, 8.4], [5.2, 5.2], color=BLUE, lw=2.2)                                  # upper fault face
ax.annotate("", xy=(1.2, 5.2), xytext=(1.2, 4.9), arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
ax.text(0.9, 5.05, "W", color=RED, fontsize=11, ha="right", va="center")
ax.text(0.9, 4.55, "(opening)", color=RED, fontsize=8, ha="right", va="center")
ax.annotate("", xy=(5.6, 5.05), xytext=(4.6, 5.05), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.5))
ax.text(5.1, 5.5, "S (slip)", color=ORANGE, fontsize=10, ha="center")
ax.text(4.7, 3.0, "displacement jump\n$u(0^+)-u(0^-)$", ha="center", fontsize=10, style="italic")
ax.text(4.7, 0.6, "analytic Okada element", ha="center", fontsize=9, color=GREY)

# --- right: eigenstrain band (MOOSE) ---
ax = axs[1]
ax.set_title("MOOSE: eigenstrain band (thickness h)", fontsize=12, fontweight="bold")
ax.add_patch(Rectangle((1, 1.4), 8, 7.35, facecolor="#eef2f3", edgecolor="k"))      # solid domain
ax.add_patch(Rectangle((1, 4.6), 8, 0.9, facecolor="#f9e79f", edgecolor=RED, hatch="///", lw=1.8))
ax.annotate("", xy=(9.2, 5.5), xytext=(9.2, 4.6), arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
ax.text(9.35, 5.05, "h", color=RED, fontsize=11, va="center")
ax.text(5, 5.05, r"$\varepsilon^*$  (prescribed)", ha="center", fontsize=12, color=RED, fontweight="bold")
ax.text(5, 3.1, r"$\varepsilon^*_{nn}=W/h$" + "\n" + r"$\varepsilon^*_{nt}=S/(2h)$",
        ha="center", fontsize=11)
ax.text(5, 0.7, "ComputeEigenstrain in a thin band", ha="center", fontsize=9, color=GREY)

# equivalence arrow between panels
fig.text(0.5, 0.52, "≡", ha="center", va="center", fontsize=30, color=BLUE, fontweight="bold")
fig.text(0.5, 0.40, "same elastic field\n($h\\!\\to\\!0$ → Okada)", ha="center", va="center",
         fontsize=9, color=BLUE)
fig.suptitle("A DD element  ≡  a thin eigenstrain band (Mura equivalent inclusion)",
             fontsize=13, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(OUT / "s1_dd_equals_eigenstrain.png", dpi=150)
plt.close(fig)


# ============================================================ s2: pipeline comparison
fig, ax = plt.subplots(figsize=(12, 6.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 11); ax.axis("off")


def box(x, y, w, h, text, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                facecolor=fc, edgecolor=ec, lw=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)


def down(x, y0, y1):
    ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6))


ax.text(3.0, 10.5, "DDMpy  (analytic, full-space)", ha="center", fontsize=12, fontweight="bold", color=BLUE)
ax.text(9.0, 10.5, "MOOSE  (FEM, finite domain)", ha="center", fontsize=12, fontweight="bold", color=RED)
L = ["Element(L,H,W,S1,S2)\nconstant-DD source", "analytic Okada  u(x)\n(J / Chinnery integrals)",
     "project onto REAL 3D\nwell tangent  d(u·t̂)/ds", "gauge-length boxcar\n→ DAS axial strain"]
R = ["add_prescribed_dd_band\n→ eigenstrain band ε*", "FEM solve  ∇·σ = 0\nσ = C:(ε−ε*)",
     "sample strain_yy (2D)\nOR project on t̂ (3D)", "gauge-length boxcar\n→ DAS axial strain"]
ys = [8.4, 6.2, 4.0, 1.8]
for i, (l, r) in enumerate(zip(L, R)):
    box(1.1, ys[i], 3.8, 1.4, l, "#eaf2f8", BLUE)
    box(7.1, ys[i], 3.8, 1.4, r, "#fdedec", RED)
    if i < 3:
        down(3.0, ys[i], ys[i + 1] + 1.4); down(9.0, ys[i], ys[i + 1] + 1.4)
# shared linearity note
ax.text(6.0, 7.0, "both\nlinear\n→ super-\npose", ha="center", va="center", fontsize=8.5,
        color=GREY, style="italic")
# highlight the divergence at the projection step
ax.add_patch(Rectangle((0.7, ys[2] - 0.15), 10.6, 1.7, fill=False, edgecolor=ORANGE, lw=2.2, ls="--"))
ax.text(6.0, ys[2] + 0.7, "↕", ha="center", va="center", fontsize=14, color=ORANGE)
ax.text(6.0, ys[2] - 0.55, "the ONLY place they differ:\n2D MOOSE puts strike-slip in strain_xy (axial≈0)",
        ha="center", va="center", fontsize=8.5, color=ORANGE, fontweight="bold")
fig.suptitle("Same mechanism, two engines — identical to <1% (see verification)",
             fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "s2_pipeline.png", dpi=150); plt.close(fig)


# ============================================================ s3: the 3D projection issue
fig, ax = plt.subplots(figsize=(11, 6.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("Why the SHEAR band needs the 3D well projection (map view, looking down)",
             fontsize=12.5, fontweight="bold")
# fault plane (~N-S vertical), shown as a shaded strip
ax.add_patch(Rectangle((5.7, 1.0), 0.6, 8.0, facecolor="#d5dbdb", edgecolor=BLUE, lw=1.5))
ax.text(6.0, 9.4, "fault (~N–S)", ha="center", color=BLUE, fontsize=10)
# strike-slip arrows on the fault
ax.annotate("", xy=(5.9, 8.2), xytext=(5.9, 6.6), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.5))
ax.annotate("", xy=(6.1, 2.0), xytext=(6.1, 3.6), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.5))
ax.text(4.7, 7.6, "strike-slip S", color=ORANGE, fontsize=10, ha="right")
# real well: ~E-W with a small tilt, crossing the fault
ax.annotate("", xy=(11.0, 5.35), xytext=(1.0, 4.75), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.6))
ax.text(1.0, 4.2, "real well (~E–W, ~1° tilt)", color=RED, fontsize=10)
ax.text(9.6, 5.6, r"$\hat{t}$", color=RED, fontsize=13)
# perpendicular idealization
ax.annotate("", xy=(11.0, 3.0), xytext=(1.0, 3.0), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2.0, ls=(0, (5, 3))))
ax.text(1.0, 2.5, "2D idealization (exactly ⊥)", color=GREY, fontsize=10)
# annotations
ax.text(9.0, 8.3, "REAL oblique well:\nslip has a small component\nalong t̂ → AXIAL shear band ✓",
        fontsize=9.5, color=RED, ha="center",
        bbox=dict(boxstyle="round", fc="#fdedec", ec=RED))
ax.text(9.0, 1.2, "2D ⊥ fiber:\nstrike-slip → strain_xy,\naxial strain_yy ≈ 0  ✗",
        fontsize=9.5, color=GREY, ha="center",
        bbox=dict(boxstyle="round", fc="#f4f6f6", ec=GREY))
ax.text(6.0, 0.3, "axial DAS strain = t̂ · ε · t̂  — depends on the well's 3D orientation vs the fault",
        ha="center", fontsize=9.5, style="italic")
fig.tight_layout(); fig.savefig(OUT / "s3_projection.png", dpi=150); plt.close(fig)

print("saved 3 schematics to", OUT)
