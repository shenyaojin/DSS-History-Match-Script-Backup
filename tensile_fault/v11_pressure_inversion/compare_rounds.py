"""Score MOOSE tensile variants for the one-sided (blue above, red, no blue below) waterfall target.

For each run: tensile-only lobe shape at the end of the ramp run (10-m gauge, per psi), and the fit
metrics of its central-channel workflow (ddm_workflow_ctr_<run>). The lower-flank score is the mean of
the simulated strain over 10420-10494 ft in the last 24 windows, against the observed mean there.
Usage: python compare_rounds.py run1 run2:tag ...   (tag defaults to ctr_<run>; all lengths in ft)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import invert_pressure as ip

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "output" / "v11_pinv"
FIG = REPO / "figs" / "tensile_fault_qc" / "moose_tensile_ddm_workflow_v4"
FIG.mkdir(parents=True, exist_ok=True)
FT = 0.3048


def lobe(run):
    f = sorted(glob.glob(str(RUNS / run / "*fiber_strain_sampler*_0093.csv")))[0]
    s = pd.read_csv(f).sort_values("y")
    y, e = s.y.to_numpy() / FT, s.strain_yy.to_numpy()
    yc = np.arange(-150, 150.5, 1.0)
    g = np.array([np.interp(np.linspace(c - 16.4, c + 16.4, 33), y, e).mean() for c in yc])
    pk = g.max()
    hm = yc[g >= pk / 2]
    return dict(lobe_up=g[yc < 0].min() / pk, lobe_low=g[yc > 0].min() / pk, halfmax_ft=hm.max() - hm.min(),
                peak_offset_ft=yc[g.argmax()], peak_nanostrain_per_psi=pk * 1e6)


def fit(run, tag):
    d = RUNS / f"ddm_workflow_{tag}"
    s = json.loads((d / "summary.json").read_text())
    out = {}
    for k in ["baseline", "inverted"]:
        m = s[k]
        P = pd.read_csv(d / f"{k}_profiles_2h.csv")
        P["sim"] = P.moose_tensile_millistrain + P.ddm_shear_millistrain
        last = P[P.center_time.isin(sorted(P.center_time.unique())[-24:])]
        low = last[(last.md_ft >= 10420)]
        up = last[(last.md_ft >= 10270) & (last.md_ft <= 10310)]
        out |= {f"{k}_rmse": m["rmse_strain"], f"{k}_rmse_T1T2": m["rmse_by_piece"]["T1-T2"],
                f"{k}_rmse_T2mid": m["rmse_by_piece"]["T2-mid"], f"{k}_rmse_midT3": m["rmse_by_piece"]["mid-T3"],
                f"{k}_rb_top": m["red_blue_boundary_md_ft"]["simulated_top"],
                f"{k}_rb_bottom": m["red_blue_boundary_md_ft"]["simulated_bottom"],
                f"{k}_hm_top": m["red_zone_halfmax_md_ft"]["simulated_top"],
                f"{k}_hm_bottom": m["red_zone_halfmax_md_ft"]["simulated_bottom"],
                f"{k}_peak_ratio": m["peak_ratio_sim_obs_last_window"],
                f"{k}_low_flank_sim": low.sim.mean(), "low_flank_obs": low.observed_strain_millistrain.mean(),
                f"{k}_up_flank_sim": up.sim.mean(), "up_flank_obs": up.observed_strain_millistrain.mean(),
                f"{k}_dP_end_psi": m["dP_last_window_psi"]}
    return out


rows = []
for arg in sys.argv[1:]:                                     # run  or  run:tag  (default tag ctr_<run>)
    r, _, tag = arg.partition(":")
    tag = tag or f"ctr_{r}"
    row = dict(run=r, tag=tag)
    try:
        row |= lobe(r)
    except Exception as e:                                   # noqa: BLE001 - report and keep going
        row["note"] = f"lobe: {e}"
    try:
        row |= fit(r, tag)
    except Exception as e:                                   # noqa: BLE001
        row["note"] = row.get("note", "") + f" fit: {e}"
    rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(FIG / "round_scores.csv", mode="a", index=False, header=not (FIG / "round_scores.csv").exists())
pd.set_option("display.width", 250)
cols = ["run", "tag", "lobe_up", "lobe_low", "halfmax_ft", "peak_offset_ft", "inverted_rb_top", "inverted_rb_bottom",
        "inverted_hm_top", "inverted_hm_bottom", "inverted_low_flank_sim", "low_flank_obs", "inverted_up_flank_sim",
        "up_flank_obs", "baseline_rmse", "inverted_rmse", "inverted_rmse_T1T2", "inverted_rmse_midT3", "inverted_peak_ratio"]
print(T[[c for c in cols if c in T]].round(4).T.to_string(header=False))
