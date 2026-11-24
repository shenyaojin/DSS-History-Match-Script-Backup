"""How much of the BC pressure reaches the fiber: BC on the monitoring well vs BC at the Bearskin 4-PB crossing.

(a) Pressure at the fiber/fracture crossing (MOOSE fiber pressure sampler) during the +1000 psi ramp run,
    as a fraction of the BC step, for both BC locations.
(b) Central-channel strain per psi of BC pressure held from T1 (sum of the hat basis), log scale.
All lengths in ft.
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import invert_pressure as ip

REPO = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO / "output" / "v11_pinv"
FIG = REPO / "figs" / "tensile_fault_qc" / "moose_tensile_ddm_workflow_v3"
FIG.mkdir(parents=True, exist_ok=True)
PSI = 6894.757
P0 = 4056.0
RUNS = [("v11_w300_well", "SRV ×1.0, BC on the monitoring well", "#1f5fbf", "-"),
        ("h060_w300_well", "SRV ×0.6, BC on the monitoring well", "#17becf", "-"),
        ("v11_w300_4pb", "SRV ×1.0, BC at Bearskin 4-PB (+1973 ft)", "#7B3294", "--"),
        ("h060_w300_4pb", "SRV ×0.6, BC at Bearskin 4-PB (+1973 ft)", "#c27ba0", "--")]

rows = []
fig, ax = plt.subplots(1, 2, figsize=(15, 5.8))
JC = int(np.argmin(np.abs(ip.MD - ip.STAR)))
for run, label, col, ls in RUNS:
    t = pd.read_csv(glob.glob(str(RUNS_DIR / run / "*input_csv.csv"))[0]).time.to_numpy()
    files = sorted(glob.glob(str(RUNS_DIR / run / "*fiber_pressure_sampler*_[0-9][0-9][0-9][0-9].csv")))
    frac, keep = [], []
    for i, f in enumerate(files):
        s = pd.read_csv(f)
        if s.empty:                                                         # initial output has no sampled rows
            continue
        keep.append(i)
        frac.append((s.pp.iloc[int(np.argmin(np.abs(s.y)))] / PSI - P0) / 1000.0)
    frac = np.array(frac)
    th = t[-len(files):][keep] / 3600
    ax[0].semilogy(th, np.clip(frac, 1e-8, None), ls, color=col, lw=2, label=label)
    step = ip.basis(run)[0][:, JC, :].sum(1) * 1e6                          # millistrain -> nanostrain per psi
    ax[1].semilogy(np.arange(len(step)) * 2.0, np.clip(step, 1e-6, None), ls, color=col, lw=2, label=label)
    rows.append(dict(run=run, label=label, fiber_dP_fraction_at_24h=float(np.interp(24, th, frac)),
                     fiber_dP_fraction_at_end=float(frac[-1]), end_h=float(th[-1]),
                     center_strain_nanostrain_per_psi_end=float(step[-1])))
ax[0].set(xlabel="Hours after the BC step", ylabel="ΔP at fiber crossing / ΔP at BC",
          title="(a) Pressure reaching the fiber, +1000 psi ramp run")
ax[1].set(xlabel="Hours after T1 (pressure held from T1)", ylabel="Central-channel strain (nanostrain per psi)",
          title=f"(b) Strain response at MD {ip.MD[JC]:.1f} ft")
for a in ax:
    a.grid(alpha=.3, which="both")
    a.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "bc_location_check.png", dpi=125)
T = pd.DataFrame(rows)
T.to_csv(FIG / "bc_location_check.csv", index=False)
print(T.to_string(index=False))
