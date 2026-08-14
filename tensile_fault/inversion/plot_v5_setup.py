"""V5 model setup: the SRV at its real along-strike length, drawn against the well geometry.

Reads the geometry straight out of the generated MOOSE input file so the figure cannot drift
from what was actually run.

Panels
  (A) along-strike view of the whole domain -- V5, with V3's footprint drawn to scale inside it
  (B) the MD cross-section through the fiber (unchanged from V3)
  (C) plan view: Gold 4-PB, the fault trace, Bearskin 3-PA, and the SRV footprint
  (D) where the fracture tip lands relative to the injecting stage
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import fault_geometry as fg

_ap = argparse.ArgumentParser(description="Draw a MOOSE SRV setup from its generated .i")
_ap.add_argument("--project", default="v5_full_length")
_ap.add_argument("--label", default=None, help="short name used in titles")
_args = _ap.parse_args()
PROJ = _args.project
LABEL = _args.label or PROJ.split("_")[0].upper()

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[3]
I_FILE = REPO / "output" / PROJ / f"{PROJ}_input.i"
WELL_CSV = REPO / "data_fervo" / "legacy" / "Gold_4_PB_Well_Geometry.csv"
STIM_NPZ = REPO / "data_fervo" / "fiberis_format" / "stimulation_loc_bearskin.npz"
FIG = REPO / "figs" / "tensile_fault_qc" / PROJ
FIG.mkdir(parents=True, exist_ok=True)

FT = 0.3048
STAR = fg.star_for_project(PROJ)
STRIKE1 = -0.8
INJ_STAGE_MD = 9726.5
C_WELL, C_V3, C_V5, C_INJ, C_SHEAR = "#1a7f5a", "#c0392b", "#1a6ea8", "#b8860b", "#8e44ad"
MUTED, INK = "#7f8c8d", "#2c3e50"
FILL = {"srv_outer": "#aed6f1", "srv_wide": "#5dade2", "srv_narrow": "#2e86c1"}
PERM = {"srv_outer": 1e-17, "srv_wide": 3e-17, "srv_narrow": 1e-16}

# ---- read the geometry back out of the .i ------------------------------------
txt = I_FILE.read_text()
blocks = dict(re.findall(r"\[(\w+_bbox)\]\n(.*?)\n  \[\]", txt, re.S))
zones = {}
for name, body in blocks.items():
    bl = [float(v) for v in re.search(r"bottom_left = '([^']+)'", body).group(1).split()]
    tr = [float(v) for v in re.search(r"top_right = '([^']+)'", body).group(1).split()]
    zones[name.replace("_bbox", "")] = dict(x0=bl[0], x1=tr[0], y0=bl[1], y1=tr[1])
dom_x = max(float(v) for v in re.findall(r"xmax = ([\d.]+)", txt))
inj_x = float(re.search(r"\[injection\].*?coord = '([\d.]+)", txt, re.S).group(1))
fib_x = float(re.search(r"fiber_strain_sampler.*?start_point = '([\d.]+)", txt, re.S).group(1))
hf = zones["hf"]
print(f"domain {dom_x:.1f} m ({dom_x/FT:.0f} ft)   injection x={inj_x:.1f} m   fiber x={fib_x:.1f} m")
print(f"fracture x {hf['x0']:.1f}..{hf['x1']:.1f} m  = {(hf['x1']-hf['x0'])/FT:.0f} ft")
for n in ("srv_outer", "srv_wide", "srv_narrow"):
    z = zones[n]
    print(f"  {n:11s} length {(z['x1']-z['x0'])/FT:7.0f} ft   MD "
          f"{STAR + z['y0']/FT:8.1f} .. {STAR + z['y1']/FT:8.1f}")

# strike coordinate measured from the fiber crossing, in ft
s_of = lambda xm: (xm - fib_x) / FT                                   # noqa: E731
HALF_FT = (hf["x1"] - hf["x0"]) / FT / 2
TIP_FT = s_of(hf["x1"])            # fracture tip on the injection side, from the well
FAR_FT = s_of(hf["x0"])            # the opposite tip
INJ_STRIKE_FT = 1934.0
V3_DOMAIN_FT, V3_HF_FT = 200.0 / FT, 250.0

_o = zones["srv_outer"]
_asym = abs(_o["y1"] + _o["y0"]) / FT     # 0 if the cross-section straddles the well evenly
_xsec_note = ("symmetric about the monitoring well" if _asym < 1.0
              else f"asymmetric by {_asym/2:.0f} ft")
print(f"along strike from the well: {FAR_FT:+.0f} ft .. {TIP_FT:+.0f} ft  "
      f"(total {TIP_FT - FAR_FT:.0f} ft)")
print(f"MD cross-section: {_xsec_note}")

fig, axs = plt.subplots(2, 2, figsize=(17, 11.5), constrained_layout=True)
(axA, axB), (axC, axD) = axs

# --- (A) along-strike view -----------------------------------------------------
axA.add_patch(Rectangle((s_of(0.0), -330), dom_x / FT, 660, facecolor="#f4f6f7",
                        edgecolor=C_V5, lw=1.8))
for n in ("srv_outer", "srv_wide", "srv_narrow"):
    z = zones[n]
    axA.add_patch(Rectangle((s_of(z["x0"]), z["y0"] / FT), (z["x1"] - z["x0"]) / FT,
                            (z["y1"] - z["y0"]) / FT, facecolor=FILL[n], edgecolor="none"))
axA.plot([s_of(hf["x0"]), s_of(hf["x1"])], [0, 0], color="#154360", lw=3)
axA.plot([0, 0], [-330, 330], color=C_WELL, lw=3, zorder=6)
axA.annotate(f"monitoring well\n(hitting channel MD {STAR:.0f})", (0, -295), color=C_WELL,
             fontsize=9.5, fontweight="bold", ha="center", va="top")
axA.plot(s_of(inj_x), 0, "o", ms=11, color="#154360", mec="white", mew=1.5, zorder=7)
axA.annotate(f"pressure BC — ONE node\nDirichlet pp = DAS curve\n({s_of(inj_x):+.0f} ft from the fiber)",
             (s_of(inj_x), 105), color="#154360", fontsize=8.5, fontweight="bold",
             ha="center", va="top")
axA.plot(INJ_STRIKE_FT, 0, "*", ms=22, color=C_INJ, mec=INK, mew=1.2, zorder=7)
axA.annotate(f"Bearskin 3-PA\nstage MD {INJ_STAGE_MD:.0f}", (INJ_STRIKE_FT, 0), color=C_INJ,
             fontsize=9.5, fontweight="bold", ha="left", textcoords="offset points",
             xytext=(16, 30))
axA.annotate("", xy=(INJ_STRIKE_FT, 245), xytext=(0, 245),
             arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6))
axA.text(INJ_STRIKE_FT / 2, 235, f"{INJ_STRIKE_FT:.0f} ft", ha="center", fontsize=10,
         fontweight="bold", color=INK, va="bottom")
# DDM shear plane, and a compact permeability key (the zones are only +/-54 ft tall here,
# so labelling them in place would be unreadable at this along-strike scale)
_shear_ft = fg.shear_md_for_project(PROJ) - STAR
axA.axhline(_shear_ft, color=C_SHEAR, ls="--", lw=1.8)
axA.text(-2850, _shear_ft - 10, f"DDM shear plane  MD {STAR + _shear_ft:.1f}", color=C_SHEAR, fontsize=8.5,
         fontweight="bold", va="bottom")
_key = "permeability k  [m^2]\n" + "\n".join(
    f"  {n:11s} {PERM[n]:.0e}   MD {STAR + zones[n]['y0']/FT:.0f}-{STAR + zones[n]['y1']/FT:.0f}"
    for n in ("srv_outer", "srv_wide", "srv_narrow")) + \
    f"\n  hf core     1e-13   MD {STAR:.0f} (0.2 ft)\n  matrix      1e-18   everywhere else"
axA.text(-2850, 120, _key, fontsize=8, color=INK, va="top", family="monospace",
         bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=MUTED, lw=0.9, alpha=0.94))

axA.set_xlim(-2900, max(2900, TIP_FT + 350))
axA.set_ylim(340, -340)   # shallower MD up, deeper MD down
axA.set_xlabel("along fault strike, from the monitoring well  [ft]")
axA.set_ylabel("fault normal  [ft]")
axA.set_title(f"(A) {LABEL} — fracture {2*HALF_FT:.0f} ft, domain {dom_x/FT:.0f} ft",
              fontweight="bold", fontsize=11, color=C_V5)
secA = axA.secondary_yaxis("right", functions=(lambda y: STAR + y, lambda m: m - STAR))
secA.set_ylabel("Measured depth  [ft]")
axA.grid(alpha=0.22)

# --- (B) MD cross-section ------------------------------------------------------
axB.add_patch(Rectangle((-400, -32), 800, 64, facecolor="#f4f6f7", edgecolor=MUTED, lw=0.8))
for n in ("srv_outer", "srv_wide", "srv_narrow"):
    z = zones[n]
    axB.add_patch(Rectangle((-350, z["y0"]), 700, z["y1"] - z["y0"], facecolor=FILL[n],
                            edgecolor="white", lw=2.0))
    axB.text(-340, z["y0"] + 0.8, f"{n}  k={PERM[n]:.0e}  MD "
             f"{STAR + z['y0']/FT:.0f}-{STAR + z['y1']/FT:.0f}", fontsize=8.5, va="top",
             color=INK)
axB.add_patch(Rectangle((-350, -0.35), 700, 0.7, facecolor="#154360"))
axB.text(340, -1.0, "hf core  k=1e-13", fontsize=8.5, color="#154360", va="bottom",
         ha="right", fontweight="bold")
axB.axhline(_shear_ft * FT, color=C_SHEAR, ls="--", lw=2)
axB.text(340, _shear_ft * FT - 1.0, f"DDM shear plane  MD {STAR + _shear_ft:.1f}", color=C_SHEAR,
         fontsize=9, fontweight="bold", va="bottom", ha="right")
axB.plot([0, 0], [-32, 32], color=C_WELL, lw=2.6)
axB.text(6, -29, "DAS fiber", color=C_WELL, fontsize=9.5, fontweight="bold")
axB.set_xlim(-400, 400)
axB.set_ylim(32, -32)
axB.set_xlabel("along fault strike, from the fiber  [ft]   (zoom)")
axB.set_ylabel("model y  [m]  (fault normal)")
axB.set_title(f"(B) MD cross-section — {_xsec_note}", fontweight="bold", fontsize=11)
sec = axB.secondary_yaxis("right", functions=(lambda y: STAR + y / FT,
                                              lambda m: (m - STAR) * FT))
sec.set_ylabel("Measured depth  [ft]")
axB.grid(alpha=0.22)

# --- (C) plan view -------------------------------------------------------------
def _rot(s, d):
    if d == np.pi / 2:
        d = np.pi / 2 - 1e-3
    s = -s + np.pi
    return (np.array([[1, 0, 0], [0, np.cos(d), np.sin(d)], [0, -np.sin(d), np.cos(d)]])
            @ np.array([[np.sin(s), -np.cos(s), 0], [np.cos(s), np.sin(s), 0], [0, 0, 1]]))


R = np.linalg.inv(_rot(np.deg2rad(STRIKE1), np.deg2rad(90.0)))
e_s, _e_d, e_n = [(R @ e) / np.linalg.norm(R @ e) for e in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]
w = pd.read_csv(WELL_CSV).sort_values("MD")
md_s = w["MD"].to_numpy(float)
XYZ = w[["x_gold", "y_gold", "z_gold"]].to_numpy(float)
cross = np.array([np.interp(STAR, md_s, XYZ[:, i]) for i in range(3)])
stim = np.load(STIM_NPZ, allow_pickle=True)
S = np.column_stack([stim["xaxis"], stim["yaxis"], -stim["zaxis"].astype(float)])
i_on = int(np.argmin(np.abs((S - cross) @ e_n)))

axC.plot(XYZ[:, 0], XYZ[:, 1], "-", color=MUTED, lw=1.4, label="Gold 4-PB")
win = (md_s >= 10200) & (md_s <= 10500)
axC.plot(XYZ[win, 0], XYZ[win, 1], "-", color=C_WELL, lw=4.5, label="DAS window")
srvz = zones["srv_outer"]
_hw = (srvz["y1"] - srvz["y0"]) / FT / 2
corners = np.array([cross + a * e_s + b * e_n * _hw
                    for a, b in [(FAR_FT, -1), (TIP_FT, -1), (TIP_FT, 1), (FAR_FT, 1),
                                 (FAR_FT, -1)]])
axC.fill(corners[:, 0], corners[:, 1], color=C_V5, alpha=0.18)
axC.plot(corners[:, 0], corners[:, 1], "-", color=C_V5, lw=2,
         label=f"{LABEL} SRV ({TIP_FT:+.0f} / {FAR_FT:+.0f} ft about the well)")
axC.plot(S[:, 0], S[:, 1], "^", ms=9, color=C_INJ, mec="white", ls="none",
         label="Bearskin 3-PA stages")
axC.plot(S[i_on, 0], S[i_on, 1], "*", ms=20, color=C_INJ, mec=INK, mew=1.2, ls="none",
         label=f"stage MD {INJ_STAGE_MD:.0f} (on plane)")
axC.plot(*cross[:2], "o", ms=10, color=C_V5, mec="white", mew=1.3, label="hitting channel")
axC.set_xlabel("x_gold  [ft]  (~East)")
axC.set_ylabel("y_gold  [ft]  (~North)")
axC.set_title("(C) Plan view — SRV footprint against the well pair",
              fontweight="bold", fontsize=11)
axC.legend(fontsize=7.8, loc="lower right", frameon=False)
axC.grid(alpha=0.22)
axC.set_aspect("equal")
axC.set_xlim(-600, 1500)
axC.set_ylim(200, 3000)

# --- (D) injection-side tip detail ---------------------------------------------
_lo = min(TIP_FT, 1920.0, INJ_STRIKE_FT) - 150
_hi = max(TIP_FT, 1920.0, INJ_STRIKE_FT) + 150
axD.add_patch(Rectangle((_lo, -60), TIP_FT - _lo, 120, facecolor=FILL["srv_outer"],
                        edgecolor="none"))
axD.plot([_lo, TIP_FT], [0, 0], color="#154360", lw=3.5)
axD.axvline(TIP_FT, color=C_V5, ls="-", lw=2.6)
axD.text(TIP_FT - 10, 78, f"{LABEL} SRV tip\n{TIP_FT:.0f} ft", color=C_V5, fontsize=10,
         fontweight="bold", ha="right")
axD.axvline(1920.0, color=C_SHEAR, ls="--", lw=2.2)
axD.text(1920 + 10, -80, "DDM fault tip\n1920 ft", color=C_SHEAR, fontsize=10,
         fontweight="bold", ha="left", va="top")
axD.plot(INJ_STRIKE_FT, 0, "*", ms=26, color=C_INJ, mec=INK, mew=1.3, zorder=6)
axD.annotate(f"stage MD {INJ_STAGE_MD:.0f}\n{INJ_STRIKE_FT:.0f} ft", (INJ_STRIKE_FT, 0),
             color=C_INJ, fontsize=10, fontweight="bold", ha="left",
             textcoords="offset points", xytext=(14, 24))
_d = TIP_FT - INJ_STRIKE_FT
axD.text((_lo + _hi) / 2, -120,
         f"SRV tip is {abs(_d):.0f} ft {'past' if _d >= 0 else 'short of'} the stage "
         f"— at {INJ_STRIKE_FT:.0f} ft from the fiber, invisible there",
         fontsize=9.5, color=INK, ha="center", va="top",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_V5, lw=1.2))
axD.set_xlim(_lo, _hi)
axD.set_ylim(-200, 140)
axD.set_xlabel("along fault strike, from the monitoring well  [ft]")
axD.set_ylabel("fault normal  [ft]")
axD.set_title("(D) The injection-side tip", fontweight="bold", fontsize=11)
axD.grid(alpha=0.22)

for ax in (axA, axB, axC, axD):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle(f"{LABEL} setup — SRV geometry and well configuration",
             fontsize=14, fontweight="bold")
out = FIG / f"{PROJ}_setup_geometry.png"
fig.savefig(out, dpi=140)
print("\nsaved", out)
