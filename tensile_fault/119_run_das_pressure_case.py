"""
Phase 1 base runner: forward-model one DAS-derived injection-pressure case and
score the modeled monitor strain-rate against the observed DAS strain-rate
(yellow-star channel) over the pre-2/28 history-match window.

Injection pressure:
    P_sim(t) = P0 + C * S(t),   S = integral of the yellow-star DAS strain-rate,
    C chosen so P_sim ends at PEAK_PSI at the end of the window.

Reuses the perpendicular-monitor MOOSE plumbing from script 111 (build+run+export)
exactly like 114 does, then adds a DAS-vs-model scoring step.

Run a single case directly:
    python scripts/tensile_fault/119_run_das_pressure_case.py
Or drive a coefficient sweep with 120_sweep_das_pressure_coefficient.py.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
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

from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge


# --- import the perpendicular-monitor MOOSE case (build/run/export) from 111 ---
SCRIPT_111 = REPO_ROOT / "scripts" / "tensile_fault" / "111_run_perpendicular_monitor_5sixth.py"
spec = importlib.util.spec_from_file_location("perpendicular_case", SCRIPT_111)
case = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(case)


# --- tunable case parameters (overridable by the sweep driver) ---
PEAK_PSI = 8000.0
P0_PSI = 2700.0
MATRIX_PERM = 1e-18
CUTOFF = dt.datetime(2025, 2, 28, 0, 0, 0)

LEGACY_DIR = REPO_ROOT / "data_fervo" / "legacy"
STRAIN_RATE_CSV = LEGACY_DIR / (
    "strain_rate_4h_mean_profiles_20250224_1500_to_20250303_2200_"
    "10200_10500ft_4h_mean_T1_ref.csv"
)
STRAIN_CSV = LEGACY_DIR / (
    "strain_4h_mean_profiles_20250224_1500_to_20250303_2200_"
    "10200_10500ft_4h_mean_T1_ref.csv"
)
NPZ_DIR = REPO_ROOT / "data_fervo" / "fiberis_format" / "post_processing"


def _project_name() -> str:
    return f"0711_das_pressure_peak{int(PEAK_PSI)}psi"


def load_star_das():
    """Yellow-star channel: (depth_ft, times, strain_rate_nano, S_cumulative_strain)."""
    dr = pd.read_csv(STRAIN_RATE_CSV)
    ds = pd.read_csv(STRAIN_CSV)
    depths = dr["measured_depth_ft"].to_numpy(float)
    tcols = dr.columns[1:]
    times = np.array([dt.datetime.fromisoformat(c) for c in tcols])
    rate = dr[tcols].to_numpy(float)
    strain = ds[tcols].to_numpy(float)
    istar = int(np.argmax(strain[:, -1]))
    tsec = np.array([(t - times[0]).total_seconds() for t in times], dtype=float)
    rate_star_s = rate[istar] * 1e-9
    cum = np.concatenate(
        [[0.0], np.cumsum(0.5 * (rate_star_s[1:] + rate_star_s[:-1]) * np.diff(tsec))]
    )
    return float(depths[istar]), times, tsec, rate[istar], cum


def build_pressure_npz():
    depth_star, times, tsec, _rate, S = load_star_das()
    s_max = float(S[-1])
    c = (PEAK_PSI - P0_PSI) / s_max
    pressure = Data1DGauge(
        data=P0_PSI + c * S,
        taxis=tsec.copy(),
        start_time=times[0],
        name=f"das_injection_pressure_peak{int(PEAK_PSI)}psi_{int(round(depth_star))}ft",
    )
    out = NPZ_DIR / f"{pressure.name}.npz"
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    pressure.savez(str(out))
    print(f"[{_project_name()}] star @ {depth_star:.0f} ft, C={c:.3e} psi/strain, "
          f"P_sim {pressure.data[0]:.0f} -> {pressure.data[-1]:.0f} psi")
    return out, pressure, c, depth_star


def set_case_paths(pressure_npz: Path):
    case.PROJECT_NAME = _project_name()
    case.MATRIX_PERM = MATRIX_PERM
    case.CROPPED_PRESSURE_PATH = pressure_npz
    case.OUTPUT_DIR = REPO_ROOT / "output" / _project_name()
    case.EXPORT_DIR = case.OUTPUT_DIR / "postprocessor_npz"
    case.FIG_DIR = REPO_ROOT / "figs" / "tensile_fault_qc" / _project_name()


def score_against_das(depth_star, das_times, das_rate_nano):
    """Compare modeled monitor strain-rate to observed DAS strain-rate (pre-2/28)."""
    model_path = case.EXPORT_DIR / "monitor_normal_strain_rate_no_rotation.npz"
    d = np.load(model_path, allow_pickle=True)
    m_taxis = d["taxis"].astype(float)        # seconds
    m_daxis = d["daxis"].astype(float)        # position along monitor (m)
    m_data = d["data"].astype(float)          # (position, time) in 1/s

    # observation point = monitor position with the strongest response (fracture intersection)
    peak_per_pos = np.nanmax(np.abs(m_data), axis=1)
    ipos = int(np.nanargmax(peak_per_pos))
    model_rate_nano = m_data[ipos, :] * 1e9   # 1/s -> nanostrain/s

    # strain-rate has a NaN at t=0 (derivative); interpolate over finite samples only
    fin = np.isfinite(model_rate_nano)
    das_rel = np.array([(t - das_times[0]).total_seconds() for t in das_times])
    model_on_das = np.interp(das_rel, m_taxis[fin], model_rate_nano[fin])

    pre = das_times < CUTOFF
    pair = pre & np.isfinite(das_rate_nano) & np.isfinite(model_on_das)
    o = das_rate_nano[pair]
    m = model_on_das[pair]
    rms = float(np.sqrt(np.mean((m - o) ** 2)))
    scale = float(np.dot(m, o) / np.dot(m, m)) if np.dot(m, m) > 0 else np.nan
    denom = np.sqrt(np.mean((o - np.mean(o)) ** 2)) or np.nan
    nrms = rms / denom if denom else np.nan
    corr = float(np.corrcoef(m, o)[0, 1]) if m.size > 1 else np.nan

    # QC overlay
    case.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    ax.plot(das_times, das_rate_nano, color="k", lw=1.8, label=f"Observed DAS @ {depth_star:.0f} ft")
    ax.plot(das_times, model_on_das, color="tab:red", lw=1.8,
            label=f"Model monitor (pos={m_daxis[ipos]:.1f} m)")
    ax.axvline(CUTOFF, color="0.4", ls="--", lw=1.1, label="2/28 (HM boundary)")
    ax.set_title(f"{_project_name()}: model vs observed strain-rate  "
                 f"(pre-2/28 RMS={rms:.3f}, NRMS={nrms:.2f}, r={corr:.2f})")
    ax.set_ylabel("Strain rate (nanostrain/s)")
    ax.set_xlabel("Datetime")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    path = case.FIG_DIR / "das_vs_model_strain_rate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {"peak_psi": PEAK_PSI, "rms": rms, "nrms": nrms, "scale": scale,
            "corr": corr, "fig": str(path)}


def main():
    pressure_npz, pressure, c, depth_star = build_pressure_npz()
    set_case_paths(pressure_npz)
    case.build_and_run_model()
    case.export_results(pressure)

    _d, das_times, _ts, das_rate_nano, _S = load_star_das()
    score = score_against_das(depth_star, das_times, das_rate_nano)
    score["coefficient"] = c
    print(f"[{_project_name()}] SCORE  C={c:.3e}  RMS={score['rms']:.4f}  "
          f"NRMS={score['nrms']:.3f}  scale={score['scale']:.3f}  r={score['corr']:.3f}")
    print(f"[{_project_name()}] QC: {score['fig']}")
    return score


if __name__ == "__main__":
    main()
