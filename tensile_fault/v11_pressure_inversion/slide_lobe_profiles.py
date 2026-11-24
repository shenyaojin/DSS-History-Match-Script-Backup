"""Slide figure: tensile-only MOOSE fiber strain (end of the +1000 psi ramp, 10-m gauge), normalised to the
peak, for the SRV setups that cross the monitoring well, vs the DDM (07) tensile profile and the observation.
All lengths in ft.  Output: figs/tensile_fault_qc/moose_tensile_ddm_workflow_v4/slide_lobe_profiles.png
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "output" / "v11_pinv"
FIG = REPO / "figs" / "tensile_fault_qc" / "moose_tensile_ddm_workflow_v4"
DDM = REPO / "Pengchao" / "Fervo_Small_Shear_Test" / "output" / "crossing_cutout" / "strain_fit_2h"
FT, STAR = 0.3048, 10365.107
CASES = [("v11_w300_well", "V11 SRV, full height", "#8c8c8c", "-"),
         ("h060_w300_well", "SRV x0.6", "#1f5fbf", "-"),
         ("g060_w300_well", "x0.6 + graded transition", "#7b3294", "--"),
         ("s060p15_w300_well", "x0.6, 15 ft deeper", "#17a2b8", "-"),
         ("a080p9_w300_well", "x0.8 asymmetric", "#e08214", "--"),
         ("e25s15_h060_w300_well", "x0.6, 15 ft deeper, E_srv 25 GPa", "#d7301f", "-"),
         ("e10s15_h060_w300_well", "x0.6, 15 ft deeper, E_srv 10 GPa", "#7f0000", "-")]

fig, ax = plt.subplots(1, 2, figsize=(11, 6.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
yc = np.arange(-160, 160.5, 1.0)
for run, label, col, ls in CASES:
    s = pd.read_csv(sorted(glob.glob(str(RUNS / run / "*fiber_strain_sampler*_0093.csv")))[0]).sort_values("y")
    y, e = s.y.to_numpy() / FT, s.strain_yy.to_numpy()
    g = np.array([np.interp(np.linspace(c - 16.4, c + 16.4, 33), y, e).mean() for c in yc])
    ax[0].plot(g / g.max(), STAR + yc, ls, color=col, lw=2, label=label)
P = pd.read_csv(DDM / "baseline_tensile_profiles_2h.csv")
last = P[P.center_time == P.center_time.max()].sort_values("md_ft")
ax[0].plot(last.strain_millistrain / last.strain_millistrain.max(), last.md_ft, color="k", lw=2.4, ls=":",
           label="DDM tensile (Pengchao 07)")
ax[0].axvline(0, color="0.3", lw=.8)
ax[0].axvspan(-0.2, 0, color="#2166ac", alpha=.07)
ax[0].set(xlim=(-0.2, 1.05), ylim=(STAR + 160, STAR - 160), xlabel="Tensile strain / peak",
          ylabel="Measured depth (ft)", title="(a) Tensile-only profile, every SRV setup crossing the well")
ax[0].legend(fontsize=8.5, loc="lower right")
ax[0].grid(alpha=.3)

# (b) flank budget below the red (>= 10420 ft, last 24 windows), millistrain
rows = [("All-DDM\n(Pengchao)", .0148, -.0111), ("MOOSE\n50 GPa", .0077, -.0111),
        ("MOOSE\n25 GPa", .0019, -.0111), ("MOOSE\n10 GPa", -.0018, -.0111)]
x = np.arange(len(rows))
ax[1].bar(x - .2, [r[1] for r in rows], .38, color="#d7301f", label="tensile part")
ax[1].bar(x + .2, [r[2] for r in rows], .38, color="#2166ac", label="DDM shear part")
ax[1].scatter(x, [r[1] + r[2] for r in rows], color="k", zorder=3, s=40, label="total")
ax[1].axhline(.0103, color="k", ls="--", lw=1.2)
ax[1].text(3.45, .0108, "observed +0.010", ha="right", fontsize=9)
ax[1].axhline(0, color="0.3", lw=.8)
ax[1].set_xticks(x, [r[0] for r in rows], fontsize=9)
ax[1].set(ylabel="Mean strain below the red, >=10420 ft (millistrain)", title="(b) What fills the lower flank")
ax[1].legend(fontsize=9, loc="lower left")
ax[1].grid(alpha=.3, axis="y")
fig.tight_layout()
fig.savefig(FIG / "slide_lobe_profiles.png", dpi=200)
print(FIG / "slide_lobe_profiles.png")
