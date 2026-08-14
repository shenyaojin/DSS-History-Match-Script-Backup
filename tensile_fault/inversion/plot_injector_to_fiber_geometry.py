"""Where the Bearskin injectors actually sit relative to the fiber -- and how badly the
current MOOSE domain under-covers that distance.

The MOOSE section is the HORIZONTAL plane containing the fault: x = along fault strike,
y = fault normal, vertical handled by plane strain. So the along-strike separation between
the injection point and the fiber/fault crossing is a distance the model has to span.

It does not. Projecting the Bearskin stimulation points onto the fault frame puts them
1921-1981 ft along strike from the crossing, while the model domain is 656 ft long, the
fracture 250 ft, and the injection node sits 83 ft from the fiber.

Panels
  (A) current model footprint, drawn to true scale against the injector->fiber distance
  (B) proposed footprint spanning injector -> fiber
  (C) plan view: Gold 4-PB, the fault trace, the Bearskin stimulation points, both footprints
  (D) per-stage decomposition -- which stage actually sits on the fault plane
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
FIG = REPO / "figs" / "tensile_fault_qc" / "v5_injector_scale"
FIG.mkdir(parents=True, exist_ok=True)

FT = 0.3048
STAR = 10373.4
STRIKE1 = -0.8
C_WELL, C_CUR, C_NEW, C_INJ, C_SHEAR = "#1a7f5a", "#c0392b", "#1a6ea8", "#b8860b", "#8e44ad"
MUTED, INK = "#7f8c8d", "#2c3e50"

# current model, in feet along strike
CUR_DOMAIN_FT = 200.0 / FT          # 656 ft
CUR_HF_FT = 250.0
CUR_FIBER_OFFSET_FT = 250.0 / 3.0   # 83.3 ft


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
e_strike, e_dip, e_norm = [(R_inv @ e) / np.linalg.norm(R_inv @ e)
                           for e in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]

w = pd.read_csv(WELL_CSV).sort_values("MD")
md_s = w["MD"].to_numpy(float)
XYZ = w[["x_gold", "y_gold", "z_gold"]].to_numpy(float)
at = lambda q: np.array([np.interp(q, md_s, XYZ[:, i]) for i in range(3)])  # noqa: E731
cross = at(STAR)                                    # fiber/fault crossing

stim = np.load(STIM_NPZ, allow_pickle=True)
S = np.column_stack([stim["xaxis"], stim["yaxis"], -stim["zaxis"].astype(float)])
stage_md = np.asarray(stim["data"], float)
rel = S - cross
s_strike, s_norm, s_dip = rel @ e_strike, rel @ e_norm, rel @ e_dip

print(f"{'stage':>9s} {'along-strike':>13s} {'fault-normal':>13s} {'vertical':>10s}")
for i in range(len(S)):
    print(f"{stage_md[i]:9.1f} {s_strike[i]:13.1f} {s_norm[i]:13.1f} {s_dip[i]:10.1f}")
on_plane = int(np.argmin(np.abs(s_norm)))
INJ_STRIKE_FT = float(s_strike[on_plane])
print(f"\nstage MD {stage_md[on_plane]:.0f} sits {s_norm[on_plane]:+.1f} ft off the fault "
      f"plane -> the one that can feed it")
print(f"along-strike injector -> fiber = {INJ_STRIKE_FT:.0f} ft")
print(f"current domain {CUR_DOMAIN_FT:.0f} ft, fracture {CUR_HF_FT:.0f} ft, "
      f"injection {CUR_FIBER_OFFSET_FT:.0f} ft from fiber")
print(f"shortfall factor: {INJ_STRIKE_FT / CUR_FIBER_OFFSET_FT:.0f}x on the injection offset, "
      f"{INJ_STRIKE_FT / CUR_HF_FT:.1f}x on the fracture length")

# proposed: injection at domain centre, fiber at +INJ_STRIKE, symmetric domain
NEW_HALF_FT = INJ_STRIKE_FT + 550.0
NEW_DOMAIN_FT = 2 * NEW_HALF_FT
NEW_HF_FT = 2 * (INJ_STRIKE_FT + 70.0)
print(f"proposed domain {NEW_DOMAIN_FT:.0f} ft, fracture {NEW_HF_FT:.0f} ft "
      f"(fault1 in DDM is 3840 ft)")

fig, axs = plt.subplots(2, 2, figsize=(17, 11.5), constrained_layout=True)
(axA, axB), (axC, axD) = axs
XLIM = (-350, 2650)


def strike_section(ax, domain_ft, hf_ft, inj_ft, fiber_ft, title, colour):
    """Along-strike view: distance along strike (ft) vs fault-normal (ft)."""
    ax.add_patch(Rectangle((inj_ft - domain_ft / 2, -330), domain_ft, 660,
                           facecolor="#f4f6f7", edgecolor=colour, lw=1.8))
    ax.add_patch(Rectangle((inj_ft - hf_ft / 2, -55), hf_ft, 110,
                           facecolor="#aed6f1", edgecolor="none"))
    ax.plot([inj_ft - hf_ft / 2, inj_ft + hf_ft / 2], [0, 0], color="#154360", lw=3)
    ax.plot(inj_ft, 0, "*", ms=22, color=C_INJ, mec="white", mew=1.4, zorder=6)
    ax.annotate("injection", (inj_ft, 0), color=C_INJ, fontsize=9.5, fontweight="bold",
                ha="center", textcoords="offset points", xytext=(0, 16))
    ax.plot([fiber_ft, fiber_ft], [-330, 330], color=C_WELL, lw=3, zorder=5)
    ax.annotate("DAS fiber", (fiber_ft, -250), color=C_WELL, fontsize=9.5,
                fontweight="bold", ha="center", textcoords="offset points", xytext=(0, 8))
    ax.annotate("", xy=(fiber_ft, 210), xytext=(inj_ft, 210),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6))
    ax.text((inj_ft + fiber_ft) / 2, 235, f"{abs(fiber_ft - inj_ft):.0f} ft", ha="center",
            fontsize=10, fontweight="bold", color=INK)
    ax.text(0.015, 0.05, f"model domain {domain_ft:.0f} ft", transform=ax.transAxes,
            fontsize=9, color=colour, fontweight="bold")
    ax.set_xlim(*XLIM)
    ax.set_ylim(-340, 340)
    ax.set_xlabel("along fault strike, from the injector  [ft]")
    ax.set_ylabel("fault normal  [ft]")
    ax.set_title(title, fontweight="bold", fontsize=11, color=colour)
    ax.grid(alpha=0.22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


strike_section(axA, CUR_DOMAIN_FT, CUR_HF_FT, 0.0, CUR_FIBER_OFFSET_FT,
               "(A) CURRENT — fiber only 83 ft from injection, fracture 250 ft", C_CUR)
axA.annotate(f"the real fiber is out here,\n{INJ_STRIKE_FT:.0f} ft away",
             (INJ_STRIKE_FT, 0), color=C_WELL, fontsize=10, fontweight="bold", ha="center",
             textcoords="offset points", xytext=(0, 60),
             arrowprops=dict(arrowstyle="->", color=C_WELL, lw=1.8))
strike_section(axB, NEW_DOMAIN_FT, NEW_HF_FT, 0.0, INJ_STRIKE_FT,
               f"(B) PROPOSED — domain spans injector to fiber ({NEW_DOMAIN_FT:.0f} ft)", C_NEW)

# --- (C) plan view -------------------------------------------------------------
axC.plot(XYZ[:, 0], XYZ[:, 1], "-", color=MUTED, lw=1.4, label="Gold 4-PB (monitoring)")
win = (md_s >= 10200) & (md_s <= 10500)
axC.plot(XYZ[win, 0], XYZ[win, 1], "-", color=C_WELL, lw=4.5, label="DAS window 10200-10500")
half_L = 3840 / 2
tr = np.array([cross + s * e_strike * half_L for s in (-1, 1)])
axC.plot(tr[:, 0], tr[:, 1], "--", color=C_SHEAR, lw=2, label="V1 fault trace (3840 ft)")
axC.plot(S[:, 0], S[:, 1], "^", ms=10, color=C_INJ, mec="white", mew=1.0, ls="none",
         label="Bearskin stimulation points")
axC.plot(S[on_plane, 0], S[on_plane, 1], "*", ms=22, color=C_INJ, mec=INK, mew=1.4,
         ls="none", label=f"stage MD {stage_md[on_plane]:.0f} (on the fault plane)")
axC.plot(*cross[:2], "o", ms=11, color=C_NEW, mec="white", mew=1.4,
         label="fiber pierces fault")
# the span the model actually has to cover, drawn along the fault instead of as a big box
span = np.array([S[on_plane], S[on_plane] - e_strike * INJ_STRIKE_FT])
axC.plot(span[:, 0], span[:, 1], "-", color=C_NEW, lw=6, alpha=0.35, solid_capstyle="butt",
         label=f"span the model must cover ({INJ_STRIKE_FT:.0f} ft)")
corners = np.array([cross + a * e_strike * CUR_DOMAIN_FT / 2 + b * e_norm * 328.0
                    for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]])
axC.plot(corners[:, 0], corners[:, 1], "-", color=C_CUR, lw=2.2, alpha=0.95,
         label=f"current domain ({CUR_DOMAIN_FT:.0f} ft) — covers only this")
axC.set_xlabel("x_gold  [ft]  (~East)")
axC.set_ylabel("y_gold  [ft]  (~North)")
axC.set_title("(C) Plan view — the model footprint against the real well pair",
              fontweight="bold", fontsize=11)
axC.legend(fontsize=7.8, loc="upper right", frameon=False)
axC.grid(alpha=0.22)
axC.set_aspect("equal")
axC.set_xlim(-500, 1600)
axC.set_ylim(250, 2950)
for s in ("top", "right"):
    axC.spines[s].set_visible(False)

# --- (D) per-stage decomposition ----------------------------------------------
axD.barh(np.arange(len(S)), s_norm, color=C_INJ, edgecolor="white", height=0.62)
axD.axvline(0, color=C_SHEAR, lw=2.4, ls="--")
axD.text(20, len(S) - 0.4, "fault plane", color=C_SHEAR, fontsize=9.5, fontweight="bold")
axD.set_yticks(np.arange(len(S)))
axD.set_yticklabels([f"MD {m:.0f}" for m in stage_md], fontsize=9)
for i in range(len(S)):
    axD.text(1000, i, f"{s_strike[i]:.0f} ft along strike", va="center", ha="right",
             fontsize=8.5, color=INK)
axD.text(1000, len(S) - 0.4, "distance TO the fiber", va="center", ha="right",
         fontsize=8.5, color=MUTED, style="italic")
axD.set_xlim(-330, 1030)
axD.set_xlabel("distance off the fault plane  [ft]   (fault-normal)")
axD.set_title("(D) Which Bearskin stage can actually feed the fault",
              fontweight="bold", fontsize=11)
axD.grid(alpha=0.22, axis="x")
for s in ("top", "right", "left"):
    axD.spines[s].set_visible(False)

fig.suptitle("Injector-to-fiber geometry — the MOOSE domain is ~24x too short along strike",
             fontsize=14, fontweight="bold")
out = FIG / "injector_to_fiber_geometry.png"
fig.savefig(out, dpi=140)
print("\nsaved", out)
