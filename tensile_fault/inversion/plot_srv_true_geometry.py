"""Does the TRUE Gold 4-PB well geometry change the V3 SRV setup?

The MOOSE model is 2-D plane strain with y = fiber/MD direction and x = along fault
strike, and it maps MD to model y with the flat assumption

    y(MD) = (MD - 10373.4) * 0.3048        # 1 ft of MD == 1 ft of fault-NORMAL distance

which is exact only if the fiber is perpendicular to the fault plane. This script drops
that assumption: it rebuilds the fault frame from the DDM rotation convention, projects
the real well survey onto it, and re-derives every SRV zone boundary as a TRUE
fault-normal distance.

Panels
  (A) current SRV setup   — zones from MD spans, straight fiber at x = 125.4 m
  (B) true-geometry setup — zones from true fault-normal distance, true well path
  (C) the offsets, in ft  — true y - assumed y, plus along-strike / along-dip drift
  (D) obliquity factor cos^2(theta) — how much of the fault-normal strain the axial
      DAS channel actually sees

The two fracture depths in play are NOT a contradiction: MOOSE's fracture sits deeper
because it marks the CENTRE of the SRV (the poroelastic tensile response fills the whole
SRV), while the DDM shear plane sits shallower because it rides the SRV/matrix BOUNDARY.
The sections mark both, and report how far the current V3 zones sit from that intent.
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
FIG = REPO / "figs" / "tensile_fault_qc" / "v3_srv_asym"
FIG.mkdir(parents=True, exist_ok=True)

FT = 0.3048
STAR = 10373.4                  # MD at model y = 0 (MOOSE fracture == intended SRV centre)
SHEAR_MD = 10319.0              # DDM fault2 doublet centre (rides the SRV/matrix boundary)
STRIKE1 = -0.8                  # fault1 (tensile) strike, deg
SRV_X0, SRV_X1 = 57.328, 142.672
HF_X0, HF_X1 = 61.9, 138.1
FIBER_X = 125.3998984           # fiber sampling line, m (83.33 ft from the injection node)
INJ_X = 100.0
MD_LO, MD_HI = 10200.0, 10500.0

# name, top_MD, bottom_MD, perm, colour  (sequential blue: darker = more permeable)
ZONES = [
    ("srv_outer",  10308.0, 10400.0, 1e-17, "#aed6f1"),
    ("srv_wide",   10328.0, 10396.0, 3e-17, "#5dade2"),
    ("srv_narrow", 10351.0, 10390.0, 1e-16, "#2e86c1"),
]
SRV_TOP, SRV_BOT = ZONES[0][1], ZONES[0][2]     # outermost SRV = the SRV/matrix boundary
SRV_CENTRE = 0.5 * (SRV_TOP + SRV_BOT)

C_WELL, C_CUR, C_SHEAR = "#1a7f5a", "#c0392b", "#8e44ad"
INK, MUTED = "#2c3e50", "#7f8c8d"


# --- fault frame, reimplemented from DDMpy_log/Fracture.py --------------------
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


def fault_axes(strike_deg, dip_deg=90.0):
    """Unit vectors (strike, dip, normal) of the fault, in the well's global frame."""
    R = np.linalg.inv(_rotate_matrix(np.deg2rad(strike_deg), np.deg2rad(dip_deg)))
    return [(R @ e) / np.linalg.norm(R @ e)
            for e in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]


# --- project the real survey onto the fault frame ----------------------------
w = pd.read_csv(WELL_CSV).sort_values("MD")
md_s = w["MD"].to_numpy(float)
XYZ = w[["x_gold", "y_gold", "z_gold"]].to_numpy(float)
at = lambda q: np.array([np.interp(q, md_s, XYZ[:, i]) for i in range(3)])  # noqa: E731

e_strike, e_dip, e_norm = fault_axes(STRIKE1)
anchor = at(STAR)                       # pin the plane at the DAS-picked pierce depth

mdq = np.arange(MD_LO, MD_HI + 0.01, 0.25)
P = np.column_stack([np.interp(mdq, md_s, XYZ[:, i]) for i in range(3)])
R = P - anchor
d_norm = R @ e_norm                     # TRUE fault-normal distance, ft  -> model y
d_strike = R @ e_strike                 # along-strike drift, ft          -> model x
d_dip = R @ e_dip                       # out-of-plane (vertical) drift, ft
d_flat = mdq - STAR                     # what the current model assumes

tan = np.gradient(P, mdq, axis=0)
tan /= np.linalg.norm(tan, axis=1)[:, None]
cos_th = np.abs(tan @ e_norm)           # axial-vs-normal projection

true_y = lambda q: float(np.interp(q, mdq, d_norm))   # noqa: E731
flat_y = lambda q: q - STAR                           # noqa: E731

print(f"{'zone':12s} {'MD span':>18s} | {'current y [ft]':>17s} | {'true y [ft]':>17s} | {'shift [ft]':>16s}")
for nm, t, b, k, _c in ZONES:
    ct, cb, tt, tb = flat_y(t), flat_y(b), true_y(t), true_y(b)
    print(f"{nm:12s} {t:8.1f}-{b:8.1f} | {ct:8.2f} {cb:8.2f} | {tt:8.2f} {tb:8.2f} | "
          f"{tt - ct:+7.3f} {tb - cb:+7.3f}")
max_off = np.abs(d_norm - d_flat).max()
print(f"\nmax |true y - assumed y| over MD {MD_LO:.0f}-{MD_HI:.0f} : {max_off:.3f} ft")
print(f"along-strike drift : {d_strike.ptp():.2f} ft   along-dip drift : {d_dip.ptp():.2f} ft")
print(f"obliquity cos^2    : {cos_th.min()**2:.5f} .. {cos_th.max()**2:.5f}  "
      f"(max angle {np.degrees(np.arccos(cos_th.min())):.2f} deg)")
print(f"survey nodes inside the window: {int(((md_s >= MD_LO) & (md_s <= MD_HI)).sum())}  "
      f"-> {md_s[(md_s >= MD_LO) & (md_s <= MD_HI)]}")

print(f"\ndesign intent: fracture at the SRV centre, shear plane on the SRV/matrix boundary")
for nm, t, b, _k, _c in ZONES:
    print(f"  {nm:12s} centre MD {0.5 * (t + b):8.1f}   fracture is {STAR - 0.5 * (t + b):+6.1f} ft "
          f"off it   ({STAR - t:.1f} ft up / {b - STAR:.1f} ft down)")
print(f"  shear plane MD {SHEAR_MD:.0f} is {SHEAR_MD - SRV_TOP:+.0f} ft INSIDE the "
      f"srv_outer/matrix boundary at MD {SRV_TOP:.0f}")

fig, axs = plt.subplots(2, 2, figsize=(16.5, 11.5), constrained_layout=True)
(axA, axB), (axC, axD) = axs
ytop, ybot = -32.0, 32.0        # plotted model-y window, m


def draw_section(ax, y_of, title, well_x=None, well_y=None):
    """One 2-D plane-strain section; y_of maps MD -> model y in metres."""
    ax.add_patch(Rectangle((0, ytop), 200, ybot - ytop, facecolor="#f4f6f7",
                           edgecolor=MUTED, lw=0.8))
    for nm, t, b, k, c in ZONES:
        y0, y1 = y_of(t), y_of(b)
        ax.add_patch(Rectangle((SRV_X0, y0), SRV_X1 - SRV_X0, y1 - y0,
                               facecolor=c, edgecolor="white", lw=2.0))  # 2px surface gap
        ax.text(SRV_X1 + 4, y0 + 1.0, f"{nm}   k={k:.0e}", fontsize=8.5, va="bottom", color=INK)
    ax.add_patch(Rectangle((HF_X0, y_of(STAR) - 0.35), HF_X1 - HF_X0, 0.7,
                           facecolor="#154360", edgecolor="none"))
    ax.text(HF_X0 + 2, y_of(STAR) - 1.0, "hf core  k=1e-13", fontsize=8.5, color="#154360",
            va="bottom", fontweight="bold")

    ax.axhline(y_of(SRV_TOP), color=C_SHEAR, ls="-", lw=1.4, alpha=0.55)
    ax.text(3, y_of(SRV_TOP) - 1.0, f"SRV / matrix boundary   MD {SRV_TOP:.0f}",
            color=C_SHEAR, fontsize=8.5, va="bottom")
    ax.axhline(y_of(SHEAR_MD), color=C_SHEAR, ls="--", lw=2.0)
    ax.text(3, y_of(SHEAR_MD) + 1.2,
            f"DDM shear plane   MD {SHEAR_MD:.0f}   ({SHEAR_MD - SRV_TOP:+.0f} ft inside the boundary)",
            color=C_SHEAR, fontsize=9, fontweight="bold", va="top")
    ax.axhline(y_of(SRV_CENTRE), color=C_CUR, ls=":", lw=2.0)
    ax.text(3, y_of(SRV_CENTRE) + 1.2,
            f"SRV centre   MD {SRV_CENTRE:.0f}   (fracture is {STAR - SRV_CENTRE:+.0f} ft below it)",
            color=C_CUR, fontsize=9, fontweight="bold", va="top")

    ax.plot(INJ_X, y_of(STAR), "o", ms=9, color="#154360", mec="white", mew=1.5, zorder=6)
    ax.annotate("injection node", (INJ_X, y_of(STAR)), textcoords="offset points",
                xytext=(-16, -16), fontsize=8.5, color="#154360")
    if well_x is None:
        ax.plot([FIBER_X, FIBER_X], [ytop, ybot], color=C_WELL, lw=2.5, zorder=5)
    else:
        ax.plot(well_x, well_y, color=C_WELL, lw=2.5, zorder=5)
    ax.text(FIBER_X + 3, ytop + 2.5, "DAS fiber", color=C_WELL, fontsize=9.5,
            va="top", fontweight="bold")

    ax.set_xlim(0, 200)
    ax.set_ylim(ybot, ytop)
    ax.set_xlabel("x  [m]   (along fault strike)")
    ax.set_ylabel("model y  [m]   (fault normal)")
    ax.set_title(title, fontweight="bold", fontsize=11)
    sec = ax.secondary_yaxis("right", functions=(lambda y: STAR + y / FT,
                                                 lambda m: (m - STAR) * FT))
    sec.set_ylabel("Measured depth  [ft]")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


draw_section(axA, lambda q: flat_y(q) * FT,
             "(A) CURRENT — MD mapped flat:  y = (MD - 10373.4) x 0.3048")
wx = FIBER_X + (d_strike - float(np.interp(STAR, mdq, d_strike))) * FT
draw_section(axB, lambda q: true_y(q) * FT,
             "(B) TRUE WELL GEOMETRY — y = survey projected on the fault normal",
             well_x=wx, well_y=d_norm * FT)
axB.text(0.985, 0.03, f"largest zone-boundary shift: {max_off:.3f} ft",
         transform=axB.transAxes, ha="right", fontsize=9.5, color=C_CUR, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_CUR, lw=1.2))

# --- (C) the offsets, all in ft ----------------------------------------------
axC.axhline(0, color=MUTED, lw=1)
axC.plot(mdq, d_norm - d_flat, color=C_CUR, lw=2.2)
axC.plot(mdq, d_strike - float(np.interp(STAR, mdq, d_strike)), color=C_WELL, lw=2.2)
axC.plot(mdq, d_dip - float(np.interp(STAR, mdq, d_dip)), color=C_SHEAR, lw=2.2, ls="--")
axC.annotate("fault-normal error  (true y - assumed y)\n"
             f"max {max_off:.2f} ft over the whole window",
             (10240, 0.18), color=C_CUR, fontsize=9, fontweight="bold", ha="left")
axC.annotate("along-strike drift (in plane)", (10470, -2.0), color=C_WELL, fontsize=9,
             fontweight="bold", ha="right")
axC.annotate("along-dip drift (out of plane)", (10470, -1.35), color=C_SHEAR, fontsize=9,
             fontweight="bold", ha="right")
for nd in md_s[(md_s >= MD_LO) & (md_s <= MD_HI)]:
    axC.axvline(nd, color=MUTED, lw=1, ls=":", zorder=0)
axC.text(10206, -2.35, "dotted = the only survey nodes in the window; the path between\n"
                       "them is a straight-line interpolation, so there is no finer\n"
                       "geometry to recover here",
         fontsize=8.5, color=MUTED, va="bottom")
axC.set_ylim(-2.6, 0.75)
for nm, t, b, _k, _c in ZONES:
    axC.axvspan(t, b, color="#aed6f1", alpha=0.18, zorder=0)
axC.set_xlim(MD_LO, MD_HI)
axC.set_xlabel("Measured depth  [ft]")
axC.set_ylabel("offset from the flat assumption  [ft]")
axC.set_title("(C) What the true geometry actually moves — shaded = SRV extent",
              fontweight="bold", fontsize=11)
axC.grid(alpha=0.25)
for s in ("top", "right"):
    axC.spines[s].set_visible(False)

# --- (D) obliquity -----------------------------------------------------------
axD.plot(mdq, cos_th ** 2, color=C_CUR, lw=2.4)
axD.axhline(1.0, color=MUTED, lw=1, ls="--")
axD.text(MD_LO + 6, 0.9955, "1.0 = fiber exactly perpendicular to the fault", fontsize=9,
         color=MUTED, va="top")
axD.text(MD_LO + 6, 0.9915,
         f"whole-window range {cos_th.min()**2:.4f}-1.0000, i.e. at most "
         f"{100 * (1 - cos_th.min()**2):.2f}% —\nplotted on a 1% axis so the flatness is honest",
         fontsize=9, color=C_CUR, va="top", fontweight="bold")
for nm, t, b, _k, _c in ZONES:
    axD.axvspan(t, b, color="#aed6f1", alpha=0.18, zorder=0)
axD.set_ylim(0.99, 1.002)
axD.set_xlim(MD_LO, MD_HI)
axD.set_xlabel("Measured depth  [ft]")
axD.set_ylabel(r"$\cos^2\theta$   (axial DAS / fault-normal strain)")
axD.set_title(r"(D) Obliquity factor — how much of $\varepsilon_{nn}$ the fiber sees",
              fontweight="bold", fontsize=11)
axD.grid(alpha=0.25)
for s in ("top", "right"):
    axD.spines[s].set_visible(False)

fig.suptitle("V3 SRV setup — flat MD mapping vs. the true Gold 4-PB well geometry",
             fontsize=14, fontweight="bold")
out = FIG / "srv_true_geometry_check.png"
fig.savefig(out, dpi=140)
print("\nsaved", out)
