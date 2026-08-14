"""All three Bearskin injection pressure records on their own axes, with the inferred
within-fracture pressure for comparison.

Split out of the pressure-detail figure because the injection records span 0-9800 psi while
the inferred fracture pressure only spans 4056-5700 psi -- sharing one axis flattens the
inferred ramp into a nearly straight line.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fault_geometry as fg

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "output" / "inversion"
_ap = argparse.ArgumentParser()
_ap.add_argument("--project", default="v11_srv_extended")
_ap.add_argument("--label", default=None)
args = _ap.parse_args()
PROJ = args.project
LABEL = args.label or PROJ.split("_")[0].upper()
FIG = REPO / "figs" / "tensile_fault_qc" / PROJ
FIG.mkdir(parents=True, exist_ok=True)

T1, T2, T3 = (pd.Timestamp(fg.T1_1200), pd.Timestamp(fg.T2_1200), pd.Timestamp(fg.T3_1200))
WELLS = {"Bearskin 1-IA": "#b8860b", "Bearskin 3-PA": "#c0392b", "Bearskin 4-PB": "#7d3c98"}

det = pd.read_csv(OUT / f"{PROJ}_pressure_detail.csv")
det["window_start"] = pd.to_datetime(det["window_start"])

fig, axs = plt.subplots(4, 1, figsize=(14, 14), constrained_layout=True, sharex=True)
for ax, (well, colour) in zip(axs[:3], WELLS.items()):
    df = pd.read_csv(REPO / "data_fervo" / "legacy" / f"{well} Pressure.csv")
    df["t"] = pd.to_datetime(df["MSTTIMESTAMP"])
    df["p"] = pd.to_numeric(df["PRESSURE_PSI"], errors="coerce")
    m = (df["t"] >= T1) & (df["t"] <= T3) & df["p"].notna()
    df = df.loc[m]
    ax.plot(df["t"], df["p"], lw=.6, color=colour, alpha=.5, label="1-minute record")
    ax.plot(df["t"], df["p"].rolling(240, center=True, min_periods=1).mean(), lw=2.2,
            color=colour, label="4-hour rolling mean")
    stg = df.dropna(subset=["STAGENUMBER"])
    if len(stg):
        ax.set_title(f"{well}   —   stages {int(stg['STAGENUMBER'].min())}"
                     f"–{int(stg['STAGENUMBER'].max())},   "
                     f"peak {df['p'].max():.0f} psi", fontweight="bold", fontsize=11)
    ax.set_ylabel("injection pressure  [psi]")
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    print(f"{well}: {len(df)} samples, {df['p'].min():.0f}..{df['p'].max():.0f} psi")

axs[3].plot(det["window_start"], det["p_fracture_psi"], "o-", ms=4.5, lw=2.6,
            color="#154360", label=f"{LABEL} inferred pressure AT the fracture")
for c, lbl in [("p_plus10ft_psi", "+10 ft into the SRV"),
               ("p_plus25ft_psi", "+25 ft"), ("p_plus40ft_psi", "+40 ft")]:
    if c in det:
        axs[3].plot(det["window_start"], det[c], lw=1.7, ls="--", alpha=.8, label=lbl)
axs[3].axhline(det["p_fracture_psi"].iloc[0], color="#2c3e50", ls=":", lw=1,
               label=f"initial condition {det['p_fracture_psi'].iloc[0]:.0f} psi")
axs[3].set_ylabel("inferred pressure  [psi]")
axs[3].set_xlabel("Time  [UTC-7]")
axs[3].set_title(f"{LABEL} INFERRED within-fracture pressure — own axis, so the ramp is legible",
                 fontweight="bold", fontsize=11)
axs[3].legend(fontsize=9, loc="upper left", frameon=False)

for ax in axs:
    ax.axvline(T2, color="#8e44ad", ls="--", lw=1.4)
    ax.set_xlim(T1, T3)
    ax.grid(alpha=.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs[0].text(T2, 1.02, "T2", transform=axs[0].get_xaxis_transform(), color="#8e44ad",
            ha="center", fontweight="bold")

fig.suptitle("Bearskin injection pressures vs the inferred within-fracture pressure",
             fontsize=14, fontweight="bold")
out = FIG / f"{PROJ}_injection_curves.png"
fig.savefig(out, dpi=140)
print("saved", out)
