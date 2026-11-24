"""Compare SRV-height variants of moose_ddm_workflow.py (v1 = V11 height, v2_h060, v2_h040), all in ft."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RUNS = [("v1", "V11 height ×1.0 (v1)", "#888888"), ("v2_h060", "SRV height ×0.6 (v2)", "#1f5fbf"),
        ("v2_h040", "SRV height ×0.4", "#d0731f")]
FIG = REPO / "figs" / "tensile_fault_qc" / "moose_tensile_ddm_workflow_v2"
FIG.mkdir(parents=True, exist_ok=True)

rows = []
for tag, label, _ in RUNS:
    s = json.loads((REPO / "output" / "v11_pinv" / f"ddm_workflow_{tag}" / "summary.json").read_text())
    outer = s["srv"][0]
    for key in ["baseline", "inverted"]:
        d = s[key]
        rb, hm = d["red_blue_boundary_md_ft"], d["red_zone_halfmax_md_ft"]
        rows.append(dict(variant=tag, case=key, srv_half_height_ft=outer["half_height_ft"], srv_md_top_ft=outer["md_top_ft"],
                         srv_md_bottom_ft=outer["md_bottom_ft"], rmse=d["rmse_strain"], ddm_rmse=d["ddm_rmse_strain"],
                         zero_top_obs=rb["observed_top"], zero_top_sim=rb["simulated_top"], zero_bottom_sim=rb["simulated_bottom"],
                         halfmax_top_obs=hm["observed_top"], halfmax_bottom_obs=hm["observed_bottom"],
                         halfmax_top_sim=hm["simulated_top"], halfmax_bottom_sim=hm["simulated_bottom"],
                         peak_ratio=d["peak_ratio_sim_obs_last_window"], **{f"rmse_{p}": v for p, v in d["rmse_by_piece"].items()},
                         dP_last_psi=d["dP_last_window_psi"]))
T = pd.DataFrame(rows)
T.to_csv(FIG / "srv_width_comparison.csv", index=False)
print(T.to_string(index=False))

# last-window and mid-shear profiles, inverted case
fig, axs = plt.subplots(1, 3, figsize=(15, 6.5), sharey=True)
for ax, when in zip(axs, ["2025-02-28 00:00", "2025-03-03 00:00", "2025-03-06 00:00"]):
    for i, (tag, label, col) in enumerate(RUNS):
        P = pd.read_csv(REPO / "output" / "v11_pinv" / f"ddm_workflow_{tag}" / "inverted_profiles_2h.csv", parse_dates=["center_time"])
        t = P.center_time.unique()
        tw = t[np.argmin(np.abs(t - pd.Timestamp(when)))]
        w = P[P.center_time == tw]
        if i == 0:
            ax.plot(w.observed_strain_millistrain, w.md_ft, "k-", lw=2.4, label="Observed")
        ax.plot(w.moose_tensile_millistrain + w.ddm_shear_millistrain, w.md_ft, "--", color=col, lw=1.8, label=label)
    ax.axvline(0, color="k", lw=.6)
    for md, c in [(10365.107, "#B36B00"), (10317.837, "teal")]:
        ax.axhline(md, color=c, ls=":", lw=1)
    ax.set(title=f"Window {pd.Timestamp(tw):%m/%d %H:%M}", xlabel="Strain (millistrain)")
    ax.grid(alpha=.3)
axs[0].set(ylim=(10500, 10200), ylabel="Measured depth (ft)")
axs[0].legend(loc="lower right", fontsize=9)
fig.suptitle("Inverted fit: MOOSE tensile + DDM shear for three SRV heights (orange dotted F1 10365.1 ft, teal dotted F2 shear 10317.8 ft)")
fig.tight_layout()
fig.savefig(FIG / "srv_width_profiles.png", dpi=130)
print(FIG)
