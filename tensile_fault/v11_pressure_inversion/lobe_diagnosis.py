"""Why the MOOSE tensile profile has strong negative lobes and the DDM tensile does not.

(a) DDM tensile (07 baseline) vs MOOSE tensile for three wall distances, last window, each scaled to its peak,
    on the 44 DAS channels.
(b) MOOSE gauge-averaged strain along the whole sampled fiber (per psi of boundary pressure, end of the ramp
    run), showing the compression plateau that the fixed walls (disp_y = 0) impose.
(c) Analytic 2-D pressurized crack: fiber crossing the crack vs fiber just beyond its tip (DDM geometry).
All lengths in ft.
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import crack_lobe_check as ck

REPO = Path(__file__).resolve().parents[3]
FIG = REPO / "figs" / "tensile_fault_qc" / "moose_tensile_ddm_workflow_v3"
FIG.mkdir(parents=True, exist_ok=True)
DDM = REPO / "Pengchao" / "Fervo_Small_Shear_Test" / "output" / "crossing_cutout" / "strain_fit_2h"
FT, STAR, GAUGE_M = 0.3048, 10365.107, 10.0
RUNS = [("h060", "v2_h060", "MOOSE, walls ±328 ft (V11, v2)", "#888888"),
        ("h060_w300", "v3_h060_w300", "MOOSE, walls ±984 ft", "#1f5fbf"),
        ("h060_w1000", "v3_h060_w1000", "MOOSE, walls ±3281 ft", "#17becf")]

rows = []
fig, ax = plt.subplots(1, 3, figsize=(19, 7.5), gridspec_kw={"width_ratios": [1, 1.25, 1]})
D = pd.read_csv(DDM / "baseline_tensile_profiles_2h.csv")
d = D[D.center_time == D.center_time.max()]
ax[0].plot(d.strain_millistrain / d.strain_millistrain.max(), d.md_ft, "-", color="#d0731f", lw=2.6, label="DDM tensile (07 baseline)")
rows.append(dict(model="DDM tensile (07)", min_over_peak=(d.strain_millistrain.min() / d.strain_millistrain.max())))
for run, tag, label, col in RUNS:
    P = pd.read_csv(REPO / "output" / "v11_pinv" / f"ddm_workflow_{tag}" / "baseline_profiles_2h.csv")
    m = P[P.center_time == P.center_time.max()]
    e = m.moose_tensile_millistrain.to_numpy()
    ax[0].plot(e / e.max(), m.md_ft, "--", color=col, lw=2, label=label)
    # full fiber, raw sampler at the last output, 10-m gauge average, per psi
    f = sorted(glob.glob(str(REPO / "output" / "v11_pinv" / run / "*fiber_strain_sampler_*_0093.csv")))[0]
    s = pd.read_csv(f).sort_values("y")
    yc = np.arange(s.y.min() + 5, s.y.max() - 5 + 1e-9, 0.5)
    g = np.array([np.interp(np.linspace(c - GAUGE_M / 2, c + GAUGE_M / 2, 41), s.y, s.strain_yy).mean() for c in yc])
    g = g / 1000.0 * 1e9                                  # 1000-psi ramp -> nanostrain per psi
    ax[1].plot(g, STAR + yc / FT, color=col, lw=2, label=label)
    w = np.abs(yc / FT) <= 145
    rows.append(dict(model=label, min_over_peak=e.min() / e.max(),
                     far_field_over_peak_295ft=g[np.argmin(abs(yc / FT - 295))] / g.max(),
                     net_over_positive_area_window=np.trapz(g[w], yc[w]) / np.trapz(np.clip(g[w], 0, None), yc[w])))
ax[0].axvline(0, color="k", lw=.6)
ax[0].axhspan(10205, 10494, color="0.93", zorder=0)
ax[0].set(ylim=(10500, 10200), xlabel="Strain / peak (last window)", ylabel="Measured depth (ft)",
          title="(a) Tensile profile shape on the 44 DAS channels")
ax[0].legend(fontsize=9, loc="lower right")
ax[1].axvline(0, color="k", lw=.6)
ax[1].axhspan(10205, 10494, color="0.93", zorder=0, label="DAS window")
ax[1].axhspan(10365.107 - 328, 10365.107 - 3600, color="#f4cccc", alpha=.5, zorder=0)
ax[1].axhspan(10365.107 + 328, 10365.107 + 3600, color="#f4cccc", alpha=.5, zorder=0, label="outside the V11 walls (±328 ft)")
ax[1].set(ylim=(10365.1 + 1000, 10365.1 - 1000), xlabel="Strain per psi of BC pressure (nanostrain/psi)",
          title="(b) MOOSE strain along the whole fiber, end of ramp run")
ax[1].legend(fontsize=9, loc="lower right")
y = np.linspace(-300, 300, 1201)
for label, x0, a, col in ck.cases:
    e = ck.das(x0, a, y)
    ax[2].plot(e / e.max(), STAR + y, color=col, lw=2, label=label.replace(": ", ":\n"))
ax[2].axvline(0, color="k", lw=.6)
ax[2].axhspan(10205, 10494, color="0.93", zorder=0)
ax[2].set(ylim=(STAR + 300, STAR - 300), xlabel="Strain / peak",
          title="(c) Analytic pressurized crack, no walls")
ax[2].legend(fontsize=9, loc="lower right")
for a in ax:
    a.grid(alpha=.3)
fig.tight_layout()
fig.savefig(FIG / "lobe_diagnosis.png", dpi=125)
T = pd.DataFrame(rows)
T.to_csv(FIG / "lobe_diagnosis.csv", index=False)
print(T.to_string(index=False))
