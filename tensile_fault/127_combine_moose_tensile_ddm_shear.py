"""
Combine the MOOSE tensile response (script 126) with the DDM shear profiles of the
Fervo_Small_Shear_Test crossing_cutout case, at the peak-tensile channel.

    total_new = MOOSE tensile (pressure-driven) + DDM shear
    total_ddm = DDM tensile + DDM shear          (original notebook decomposition)

Caveat: the MOOSE monitor line spans only ~+-50 ft about the fracture, so the
comparison is made at the single peak channel (the channel whose DDM tensile
strain is largest, the one the pressure was derived from), not over the full fiber.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY = REPO_ROOT / "data_fervo" / "legacy"
FIG_ROOT = REPO_ROOT / "figs" / "tensile_fault_qc" / "crossing_cutout_tensile"
COLORS = {"baseline": "tab:blue", "inverted": "tab:red"}


def channel(name, component):
    d = pd.read_csv(LEGACY / f"{name}_{component}_profiles_2h.csv", parse_dates=["center_time"])
    return (d.pivot(index="md_ft", columns="center_time", values="strain_millistrain"),
            d.pivot(index="md_ft", columns="center_time", values="strain_rate_nanostrain_s"))


def moose_peak_series(name, times):
    out = REPO_ROOT / "output" / f"crossing_cutout_tensile_{name}" / "postprocessor_npz"
    r = np.load(out / "monitor_normal_strain_rate_no_rotation.npz", allow_pickle=True)
    s = np.load(out / "monitor_normal_strain_no_rotation.npz", allow_pickle=True)
    ipos = int(np.nanargmax(np.nanmax(np.abs(r["data"].astype(float)), axis=1)))
    t0 = times[0]  # pressure t=0 was the first window centre
    mt = (times - t0).total_seconds().to_numpy(float)

    def on_windows(npz, scale):
        ta, y = npz["taxis"].astype(float), npz["data"][ipos].astype(float) * scale
        ok = np.isfinite(y)
        return np.interp(mt, ta[ok], y[ok])

    return on_windows(s, 1e3), on_windows(r, 1e9)


def main():
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, constrained_layout=True)
    for j, name in enumerate(COLORS):
        ts, tr = channel(name, "tensile")
        ss, sr = channel(name, "shear")
        md = ts.abs().max(axis=1).idxmax()
        times = ts.columns
        ms, mr = moose_peak_series(name, times)
        ddm_ts, ddm_tr = ts.loc[md].to_numpy(), tr.loc[md].to_numpy()
        sh_s, sh_r = ss.loc[md].to_numpy(), sr.loc[md].to_numpy()
        for i, (ddm_t, moose_t, shear, unit) in enumerate(
            [(ddm_ts, ms, sh_s, "Strain (millistrain)"), (ddm_tr, mr, sh_r, "Strain rate (nanostrain/s)")]
        ):
            ax = axes[i, j]
            ax.plot(times, ddm_t + shear, "k", lw=2, label="DDM tensile + DDM shear")
            ax.plot(times, moose_t + shear, color=COLORS[name], lw=2, label="MOOSE tensile + DDM shear")
            ax.plot(times, shear, color="tab:green", lw=1.3, ls="--", label="DDM shear")
            ax.set_ylabel(unit)
            ax.grid(True, ls=":", alpha=0.4)
            if i == 0:
                ax.set_title(f"{name} @ {md:.0f} ft")
            if i == 0 and j == 0:
                ax.legend(loc="upper left", fontsize=9)
        n = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))
        rows.append(dict(case=name, md_ft=md,
                         rel_rmse_strain=n(ms + sh_s, ddm_ts + sh_s), rel_rmse_rate=n(mr + sh_r, ddm_tr + sh_r),
                         moose_peak_strain=float(np.nanmax(np.abs(ms))), ddm_peak_strain=float(np.nanmax(np.abs(ddm_ts))),
                         shear_peak_strain=float(np.nanmax(np.abs(sh_s)))))
        pd.DataFrame({"center_time": times, "ddm_tensile_strain": ddm_ts, "moose_tensile_strain": ms, "ddm_shear_strain": sh_s,
                      "ddm_tensile_rate": ddm_tr, "moose_tensile_rate": mr, "ddm_shear_rate": sh_r}
                     ).to_csv(FIG_ROOT / f"{name}_peak_channel_combined.csv", index=False)
    fig.savefig(FIG_ROOT / "combined_moose_tensile_ddm_shear.png", dpi=200)
    plt.close(fig)
    summary = pd.DataFrame(rows)
    summary.to_csv(FIG_ROOT / "combined_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
