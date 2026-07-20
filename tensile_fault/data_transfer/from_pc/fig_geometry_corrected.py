"""Corrected V1 fault-vs-well geometry figure.

The V1 fault does NOT originate at Gold 4-PB: it propagates from the Bearskin
injection well and INTERSECTS the Gold 4-PB monitoring well at MD 10373 ft.
Map view + the MD↔model mapping note.
"""
import os
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

mpl.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold", "savefig.dpi": 150})
REPO = Path(__file__).resolve().parents[4]
DELIV = REPO / "output" / "V1_MOOSE_DDM_deliverable" / "figures"

w = pd.read_csv(REPO / "data_fervo/legacy/Gold_4_PB_Well_Geometry.csv")
xg, yg, md = w["x_gold"].to_numpy(float), w["y_gold"].to_numpy(float), w["MD"].to_numpy(float)
i373 = int(np.argmin(np.abs(md - 10373)))
xc, yc = xg[i373], yg[i373]                       # Gold 4-PB crossing at MD 10373
stim = np.load(REPO / "data_fervo/fiberis_format/stimulation_loc_bearskin.npz", allow_pickle=True)
sx, sy = np.asarray(stim["xaxis"], float), np.asarray(stim["yaxis"], float)   # Bearskin injection well

fig, ax = plt.subplots(figsize=(12, 8))
# Gold 4-PB monitoring well
ax.plot(xg, yg, "-", color="black", lw=2.2, label="Gold 4-PB monitoring well (DAS fiber)")
# Bearskin injection well
ax.plot(sx, sy, "-", color="tab:purple", lw=2.2, label="Bearskin injection well (fault source)")
ax.scatter(sx, sy, c="tab:purple", s=25, zorder=5)
# V1 fault plane trace: from the injection well toward the Gold crossing (fault strikes ~N-S here)
xf_inj = np.interp(yc, sy[np.argsort(sy)], sx[np.argsort(sx)]) if False else float(np.interp(xc, sx, sy))
# draw the fault as the line through the Gold crossing toward the injection well
ax.plot([xc, xc], [yc, sy.mean()], "-", color="tab:red", lw=3, alpha=0.85, zorder=4,
        label="V1 fault plane (propagates injection → monitor)")
ax.plot([xc - 400, xc + 400], [yc, yc], color="tab:red", lw=1, ls=":")  # short strike hint at crossing
ax.scatter([xc], [yc], marker="*", s=420, color="gold", ec="k", lw=1.5, zorder=6)
ax.annotate("V1 fault ∩ Gold 4-PB\nMD 10373 ft\n(monitor CROSSES the fault —\nfault is NOT centred on Gold 4-PB)",
            (xc, yc), xytext=(xc + 900, yc - 700), fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=1.5), bbox=dict(fc="#fff8e1", ec="k", alpha=0.95))
ax.annotate("fault propagates ~1950 ft\nfrom the injection well",
            (xc, (yc + sy.mean()) / 2), xytext=(xc + 900, (yc + sy.mean()) / 2),
            fontsize=10, color="tab:red", arrowprops=dict(arrowstyle="->", color="tab:red"))
ax.text(sx.mean(), sy.mean() + 120, "Bearskin injection well", color="tab:purple", ha="center", fontsize=10)
ax.set_xlabel("x_gold  [ft]  (~East)"); ax.set_ylabel("y_gold  [ft]  (~North)")
ax.set_title("V1 fault geometry (map view): fault propagates from the Bearskin injection well\nand intersects the Gold 4-PB monitoring well at MD 10373 ft")
ax.legend(loc="upper right", fontsize=10); ax.grid(alpha=0.3); ax.set_aspect("equal")
ax.set_xlim(-1000, 3000); ax.set_ylim(200, 2900)
fig.tight_layout(); fig.savefig(DELIV / "fig0_fracture_vs_well.png"); print("Saved corrected fig0:", DELIV / "fig0_fracture_vs_well.png")
