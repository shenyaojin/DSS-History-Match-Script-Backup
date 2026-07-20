"""Polished deliverable figures for the V1 MOOSE+DDM report (PPT-ready).

Produces, into output/V1_MOOSE_DDM_deliverable/figures/:
  fig1_geometry_2d.png        - 2D MOOSE tensile-SRV cross-section (MD axis, graded layers, fiber)
  fig2_geometry_consistency.png - the SAME geometry across both stages; only the loading evolves
  fig3_pressure_qc.png        - why smooth pressure (raw injection vs DAS low-pass) + inferred p
"""
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
import pandas as pd

mpl.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
                     "axes.labelsize": 12, "figure.dpi": 150, "savefig.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})

REPO = Path(__file__).resolve().parents[4]
DELIV = REPO / "output" / "V1_MOOSE_DDM_deliverable" / "figures"
DELIV.mkdir(parents=True, exist_ok=True)
FT = 0.3048
STAR = 10373.4
T1 = pd.Timestamp("2025-02-24 15:00"); T2 = pd.Timestamp("2025-02-28 00:00"); T3 = pd.Timestamp("2025-03-03 22:00")


def y_to_md(y_m):
    return STAR + y_m / FT


# graded SRV layers (height in ft along MD, perm) -- current model: outer width 75 ft, perm x0.3
SRV = [(75, 9e-17, "#fff2cc"), (56, 3e-16, "#ffe08a"), (38, 9e-16, "#ffc04d"), (19, 3e-15, "#ff9c1a")]
HF_H = 0.2
SRV_W = 75.0
SHEAR_MD = STAR - SRV_W / 2.0        # DDM shear plane at the (upper) SRV-matrix boundary

# =============================================================================
# FIG 1 - 2D MOOSE tensile-SRV cross-section
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 7))
X0, X1 = 40, 160          # strike extent shown (m); fracture ~400 ft = 122 m centred at 100
ax.add_patch(Rectangle((0, y_to_md(-100)), 200, y_to_md(100) - y_to_md(-100),
                       fc="#eaf2fb", ec="none", zorder=0))
for h, perm, col in SRV:                      # widest first
    ax.add_patch(Rectangle((X0, STAR - h / 2), X1 - X0, h, fc=col, ec="0.5", lw=0.6, zorder=2,
                           label=f"SRV k={perm:.0e} ({h} ft)"))
ax.add_patch(Rectangle((X0, STAR - HF_H / 2), X1 - X0, HF_H, fc="#d62728", ec="none", zorder=4))
ax.plot(100, STAR, "^", ms=13, color="cyan", mec="k", zorder=6)
ax.annotate("injection\n(fracture core)", (100, STAR), xytext=(70, 10250),
            arrowprops=dict(arrowstyle="->"), ha="center", fontsize=10)
# fiber (vertical line along MD) at 40 ft offset
xf = 100 + 40 * FT
ax.plot([xf, xf], [10200, 10500], "k--", lw=2.2, zorder=5)
ax.annotate("DAS fiber $\perp$ fault\n(samples strain_yy along MD)", (xf, 10250), xytext=(140, 10240),
            arrowprops=dict(arrowstyle="->"), fontsize=10)
ax.annotate("", (X0 - 3, STAR - 45), (X0 - 3, STAR + 45), arrowprops=dict(arrowstyle="<->", color="tab:blue"))
ax.text(X0 - 6, STAR, "SRV width\n≈ 75 ft\n(from observation)", color="tab:blue", ha="right", va="center", fontsize=10)
ax.axhline(STAR, color="0.6", lw=0.8, ls=":")
ax.text(198, STAR, "fault centre\nMD 10373 ft", ha="right", va="center", fontsize=9, color="0.3")
ax.set_xlim(0, 200); ax.set_ylim(10500, 10200)
ax.set_xlabel("Along-fault (strike) direction  [m]"); ax.set_ylabel("Measured Depth (MD)  [ft]")
ax.set_title("2D MOOSE poroelastic tensile model — SRV cross-section\n(Y=fiber=MD,  X=strike;  MD = 10373.4 + Y/0.3048)")
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
fig.tight_layout(); fig.savefig(DELIV / "fig1_geometry_2d.png"); plt.close(fig)
print("fig1 done")

# =============================================================================
# FIG 2 - geometry is IDENTICAL across the two stages; only the loading evolves
# =============================================================================
fig, axs = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw=dict(width_ratios=[1, 1.15]))

# left: the single shared geometry (fault plane + SRV + fiber), stages annotated
ax = axs[0]
ax.add_patch(Rectangle((0, y_to_md(-70)), 200, y_to_md(70) - y_to_md(-70), fc="#eaf2fb", ec="none"))
for h, perm, col in SRV:
    ax.add_patch(Rectangle((40, STAR - h / 2), 120, h, fc=col, ec="0.6", lw=0.5))
ax.add_patch(Rectangle((40, STAR - HF_H / 2), 120, HF_H, fc="#d62728", ec="none"))
xf = 100 + 40 * FT
ax.plot([xf, xf], [10250, 10480], "k--", lw=2.2)
ax.text(xf + 3, 10465, "DAS fiber", fontsize=10, rotation=90, va="bottom")
ax.axhline(STAR, color="0.5", ls=":", lw=0.8)
ax.text(100, STAR - 55, "MOOSE tensile SRV  (active T1→T3)", ha="center", color="#b30000", fontweight="bold", fontsize=11)
ax.axhline(SHEAR_MD, color="tab:blue", ls="--", lw=1.8)
ax.text(100, SHEAR_MD - 6, "DDM shear plane @ SRV–matrix boundary (active T2→T3)",
        ha="center", va="bottom", color="tab:blue", fontweight="bold", fontsize=9.5)
ax.set_xlim(0, 200); ax.set_ylim(10500, 10200)
ax.set_xlabel("strike [m]"); ax.set_ylabel("MD [ft]")
ax.set_title("Shared geometry\nfault centre MD 10373 ft, SRV ≈ 75 ft, fiber $\perp$ fault")

# right: loading histories on the SAME geometry (tensile p, shear slip)
ax = axs[1]
hist = pd.read_csv(REPO / "data_fervo/legacy/07152026/two_fault_histories_20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref.csv")
hist["time"] = pd.to_datetime(hist["time"])
pcsv = pd.read_csv(REPO / "output/v1_srv_t1_1500/inferred_fracture_pressure_history.csv")
pcsv["time"] = pd.to_datetime(pcsv["time"])
ax.plot(pcsv["time"], pcsv["fracture_dp_psi"], color="#b30000", lw=2.4, label="tensile: inferred fracture Δp (MOOSE)")
ax.set_ylabel("tensile Δp  [psi]", color="#b30000"); ax.tick_params(axis="y", colors="#b30000")
ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
ax2.plot(hist["time"], np.abs(hist["fault2_shear_ft"]) * 12, color="tab:blue", lw=2.4, ls="--",
         label="shear: DDM slip magnitude")
ax2.set_ylabel("shear slip  [inch]", color="tab:blue"); ax2.tick_params(axis="y", colors="tab:blue")
for tt, lab in zip([T1, T2, T3], ["T1", "T2", "T3"]):
    ax.axvline(tt, color="green", ls="--", lw=1.5)
    ax.text(tt, 1.01, lab, transform=ax.get_xaxis_transform(), color="green", fontweight="bold", ha="center", va="bottom")
ax.axvspan(T1, T2, color="#ffecec", alpha=0.6, zorder=0)
ax.axvspan(T2, T3, color="#eef3ff", alpha=0.6, zorder=0)
ax.text(T1 + (T2 - T1) / 2, ax.get_ylim()[1] * 0.9, "Stage 1\ntensile only", ha="center", fontsize=10, color="#b30000")
ax.text(T2 + (T3 - T2) / 2, ax.get_ylim()[1] * 0.9, "Stage 2\ntensile + shear", ha="center", fontsize=10, color="tab:blue")
ax.set_xlim(T1, T3); ax.set_title("Same geometry, evolving loading\n(tensile grows T1→T3;  shear turns on at T2)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9)
fig.suptitle("Geometry stays consistent across both stages — only the loading changes over time", fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(DELIV / "fig2_geometry_consistency.png"); plt.close(fig)
print("fig2 done")

# =============================================================================
# FIG 3 - pressure QC: raw injection (fluctuating) vs DAS smooth + inferred p
# =============================================================================
fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
inj = np.load(REPO / "data_fervo/fiberis_format/post_processing/synthetic_data_simulation.npz", allow_pickle=True)
it = pd.Timestamp(str(inj["start_time"])) + pd.to_timedelta(np.asarray(inj["taxis"], float), unit="s")
a1.plot(it, np.asarray(inj["data"], float), color="0.6", lw=0.8, label="raw wellhead injection (fluctuating)")
das = np.load(REPO / "output/v1_srv_t1_1500/das_pressure_T1_1500_prepended.npz", allow_pickle=True)
dt = T1 + pd.to_timedelta(np.asarray(das["taxis"], float), unit="s")
a1.plot(dt, np.asarray(das["data"], float), color="tab:orange", lw=2.6, label="DAS-derived smooth pressure (used)")
a1.set_ylabel("pressure [psi]"); a1.set_title("Why a smooth pressure: the formation between well and fiber is a low-pass filter")
a1.legend(loc="upper left"); a1.grid(alpha=0.3)
a1.text(0.5, 0.05, "raw injection scale → 20% var. reduction\nsmooth DAS pressure → 49% var. reduction",
        transform=a1.transAxes, ha="center", fontsize=10, bbox=dict(fc="white", ec="0.6"))

a2.plot(pcsv["time"], pcsv["das_pressure_psi"], color="tab:orange", lw=1.4, ls=":", label="DAS pressure (s=1)")
a2.plot(pcsv["time"], pcsv["fracture_pressure_psi"], color="#b30000", lw=2.8, label="inferred within-fracture pressure (s=0.48)")
a2.axhline(pcsv["das_pressure_psi"].iloc[0], color="k", ls=":", lw=1, label=f"IC {pcsv['das_pressure_psi'].iloc[0]:.0f} psi")
for tt, lab in zip([T1, T2, T3], ["T1", "T2", "T3"]):
    a2.axvline(tt, color="green", ls="--", lw=1.4)
    a2.text(tt, 1.01, lab, transform=a2.get_xaxis_transform(), color="green", fontweight="bold", ha="center", va="bottom")
a2.set_ylabel("pressure [psi]"); a2.set_xlabel("Time [UTC-7]")
a2.set_title("Inferred within-fracture pressure history (deliverable): 4056 → ~5033 psi, Δp peak ~977 psi")
a2.legend(loc="upper left"); a2.grid(alpha=0.3)
a2.set_xlim(T1, T3); a2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
fig.tight_layout(); fig.savefig(DELIV / "fig3_pressure_qc.png"); plt.close(fig)
print("fig3 done")
print("figures in", DELIV)
