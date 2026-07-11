"""
Phase 0 (fast, no MOOSE): bracket the coefficient C that turns the yellow-star
DAS strain-rate into a synthetic injection pressure.

Idea:
    S(t)     = integral of the yellow-star strain-rate trace  (~ T1-ref strain)
    P_sim(t) = P0 + C * S(t)                                   (psi)

We bracket C two ways against the observed averaged pressure record (which only
exists up to 2025-02-28):
    1. Least-squares fit of (P0, C) to the observed pressure pre-2/28.
    2. Fixed-P0 fit of C alone.
Then we materialize a small family of target-peak curves (P_sim ends at a chosen
peak psi) and save each as a fiberis Data1D NPZ for the MOOSE forward sweep.

Run from the repository root:
    python scripts/tensile_fault/118_das_pressure_coefficient_bracket.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
FIBERIS_SRC = REPO_ROOT / "fibeRIS" / "src"
if str(FIBERIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIBERIS_SRC))

from fiberis.analyzer.Data1D.core1D import Data1D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge


LEGACY_DIR = REPO_ROOT / "data_fervo" / "legacy"
STRAIN_RATE_CSV = LEGACY_DIR / (
    "strain_rate_4h_mean_profiles_20250224_1500_to_20250303_2200_"
    "10200_10500ft_4h_mean_T1_ref.csv"
)
STRAIN_CSV = LEGACY_DIR / (
    "strain_4h_mean_profiles_20250224_1500_to_20250303_2200_"
    "10200_10500ft_4h_mean_T1_ref.csv"
)

# Observed averaged injection pressure (psi). Exists only up to 2025-02-28.
OBS_PRESSURE_PATH = (
    REPO_ROOT / "data_fervo" / "fiberis_format" / "post_processing"
    / "synthetic_data_simulation.npz"
)

NPZ_DIR = REPO_ROOT / "data_fervo" / "fiberis_format" / "post_processing"
FIG_DIR = REPO_ROOT / "figs" / "tensile_fault_qc" / "das_pressure_bracket"

CUTOFF = dt.datetime(2025, 2, 28, 0, 0, 0)       # history-match boundary
P0_FIXED_PSI = 2700.0                             # reservoir baseline for fixed-P0 fit
TARGET_PEAKS_PSI = [3000.0, 5000.0, 8000.0, 12000.0]


def load_star_strain_rate():
    """Return (depth_ft, times, S_rate_strain_per_s, S_cumulative_strain) for the
    yellow-star channel = the depth with maximum cumulative strain."""
    dr = pd.read_csv(STRAIN_RATE_CSV)
    ds = pd.read_csv(STRAIN_CSV)
    depths = dr["measured_depth_ft"].to_numpy(float)
    tcols = dr.columns[1:]
    times = np.array([dt.datetime.fromisoformat(c) for c in tcols])

    rate = dr[tcols].to_numpy(float)     # nanostrain/s, depth x time
    strain = ds[tcols].to_numpy(float)   # millistrain,  depth x time

    istar = int(np.argmax(strain[:, -1]))       # max final cumulative strain = the star
    depth_star = float(depths[istar])

    tsec = np.array([(t - times[0]).total_seconds() for t in times], dtype=float)
    rate_star = rate[istar] * 1e-9              # nanostrain/s -> strain/s
    cum = np.concatenate(
        [[0.0], np.cumsum(0.5 * (rate_star[1:] + rate_star[:-1]) * np.diff(tsec))]
    )
    return depth_star, times, tsec, rate[istar], cum


def build_pressure(S, taxis, start_time, p0, c, name):
    return Data1DGauge(data=p0 + c * S, taxis=taxis.copy(), start_time=start_time, name=name)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)

    depth_star, times, tsec, rate_star_nano, S = load_star_strain_rate()
    start_time = times[0]
    s_max = float(S[-1])
    pre_mask = times < CUTOFF
    s_at_cut = float(S[np.where(pre_mask)[0][-1]])

    print(f"Yellow-star channel  : {depth_star:.1f} ft")
    print(f"Window               : {times[0]} -> {times[-1]}")
    print(f"Cumulative strain S  : S(2/28)={s_at_cut:.3e}, S(end)={s_max:.3e}")

    # --- Bracket C against the observed pressure record (pre-2/28) ---
    obs = Data1DGauge()
    obs.load_npz(str(OBS_PRESSURE_PATH))
    # Observed pressure interpolated onto the DAS sample times, pre-2/28 only.
    das_dt = np.array([start_time + dt.timedelta(seconds=float(s)) for s in tsec])
    obs_dt = np.array([obs.start_time + dt.timedelta(seconds=float(s)) for s in obs.taxis])
    obs_rel = np.array([(d - start_time).total_seconds() for d in obs_dt])
    das_rel = tsec
    fit_mask = (das_dt < CUTOFF) & (das_dt >= obs_dt[0])
    P_obs_on_das = np.interp(das_rel[fit_mask], obs_rel, obs.data)
    S_fit = S[fit_mask]

    # (1) free (P0, C) least squares:  P_obs ~ P0 + C * S
    A = np.vstack([np.ones_like(S_fit), S_fit]).T
    (p0_free, c_free), *_ = np.linalg.lstsq(A, P_obs_on_das, rcond=None)
    # (2) fixed P0 = P0_FIXED_PSI:  C = <S, P_obs - P0> / <S, S>
    resid = P_obs_on_das - P0_FIXED_PSI
    c_fixed = float(np.dot(S_fit, resid) / np.dot(S_fit, S_fit))

    print("\n--- Coefficient bracket vs observed pressure (pre-2/28) ---")
    print(f"free  fit : P0={p0_free:8.1f} psi, C={c_free:.3e} psi/strain  "
          f"-> P_sim(end)={p0_free + c_free*s_max:8.1f} psi, P_sim(2/28)={p0_free + c_free*s_at_cut:7.1f}")
    print(f"fixedP0   : P0={P0_FIXED_PSI:8.1f} psi, C={c_fixed:.3e} psi/strain  "
          f"-> P_sim(end)={P0_FIXED_PSI + c_fixed*s_max:8.1f} psi, P_sim(2/28)={P0_FIXED_PSI + c_fixed*s_at_cut:7.1f}")

    # --- Target-peak family (for the MOOSE forward sweep) ---
    print("\n--- Target-peak curves (P0 = %.0f psi, peak = value at end of window) ---" % P0_FIXED_PSI)
    print(f"{'peak_psi':>9} {'C(psi/strain)':>14} {'P_sim@2/28':>11} {'P_sim@end':>10}")
    saved = []
    for peak in TARGET_PEAKS_PSI:
        c = (peak - P0_FIXED_PSI) / s_max
        p = build_pressure(S, tsec, start_time, P0_FIXED_PSI, c,
                           name=f"das_injection_pressure_peak{int(peak)}psi_{int(round(depth_star))}ft")
        out = NPZ_DIR / f"{p.name}.npz"
        p.savez(str(out))
        saved.append((peak, c, p, out))
        print(f"{peak:9.0f} {c:14.3e} {P0_FIXED_PSI + c*s_at_cut:11.1f} {P0_FIXED_PSI + c*s_max:10.1f}")

    # Also save the free-fit curve as a reference candidate.
    p_free = build_pressure(S, tsec, start_time, p0_free, c_free,
                            name=f"das_injection_pressure_lsqfit_{int(round(depth_star))}ft")
    p_free.savez(str(NPZ_DIR / f"{p_free.name}.npz"))

    # --- QC figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [1.0, 1.6]})

    ax1.plot(das_dt, S * 1e3, color="tab:green", lw=2, label=f"cumulative strain @ {depth_star:.0f} ft")
    ax1.set_ylabel("Cumulative strain (millistrain)")
    ax1.axvline(CUTOFF, color="k", ls="--", lw=1.2, label="2/28 00:00 (HM boundary)")
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper left")
    ax1.set_title(f"DAS-derived injection pressure — coefficient bracket (star @ {depth_star:.0f} ft)")

    ax2.plot(obs_dt, obs.data, color="0.5", lw=1.3, label="Observed avg pressure (real)")
    ax2.plot(das_dt, p_free.data, color="tab:red", lw=2.2, ls="-",
             label=f"LSQ fit: P0={p0_free:.0f}, C={c_free:.2e}")
    cmap = plt.get_cmap("viridis")
    for i, (peak, c, p, _) in enumerate(saved):
        ax2.plot(das_dt, p.data, lw=1.6, color=cmap(i / max(1, len(saved) - 1)),
                 label=f"peak {peak:.0f} psi (C={c:.2e})")
    ax2.axvline(CUTOFF, color="k", ls="--", lw=1.2)
    ax2.set_ylabel("Pressure (psi)")
    ax2.set_xlabel("Datetime")
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=8, ncol=2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    fig.tight_layout()
    fig_path = FIG_DIR / "das_pressure_coefficient_bracket.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved QC figure -> {fig_path}")
    print(f"Saved {len(saved)} target-peak NPZ + 1 LSQ-fit NPZ to {NPZ_DIR}")


if __name__ == "__main__":
    main()
