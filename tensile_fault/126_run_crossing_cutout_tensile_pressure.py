"""
Replace the DDM tensile component of the Fervo_Small_Shear_Test crossing_cutout
case (notebooks 03/04, 2-hour rate fit) with the MOOSE tensile-fault model.

Pressure histories (one per case) come from the DDM tensile strain profiles:
    P(t) = P0 + C * S(t),  S = tensile strain at the peak channel (millistrain -> strain)
    baseline -> baseline_tensile_profiles_2h.csv
    inverted -> inverted_tensile_profiles_2h.csv
Each is run through the 111 perpendicular-monitor MOOSE case; modeled monitor
strain / strain-rate are overlaid on the DDM tensile profile that produced the pressure.

    python scripts/tensile_fault/126_run_crossing_cutout_tensile_pressure.py [--skip-run]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FIBERIS_SRC = REPO_ROOT / "fibeRIS" / "src"
if str(FIBERIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIBERIS_SRC))

from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.moose.templates.baseline_model_generator_fervo import build_baseline_model

spec = importlib.util.spec_from_file_location(
    "perpendicular_case", REPO_ROOT / "scripts" / "tensile_fault" / "111_run_perpendicular_monitor_5sixth.py"
)
case = importlib.util.module_from_spec(spec)
spec.loader.exec_module(case)

LEGACY_DIR = REPO_ROOT / "data_fervo" / "legacy"
NPZ_DIR = REPO_ROOT / "data_fervo" / "fiberis_format" / "post_processing"
FIG_ROOT = REPO_ROOT / "figs" / "tensile_fault_qc" / "crossing_cutout_tensile"
CASES = {
    "baseline": LEGACY_DIR / "baseline_tensile_profiles_2h.csv",
    "inverted": LEGACY_DIR / "inverted_tensile_profiles_2h.csv",
}
COLORS = {"baseline": "tab:blue", "inverted": "tab:red"}

P0_PSI = 4056.0     # Phase-0 LSQ intercept (script 118)
C_PSI_PER_STRAIN = 1.63e7  # history-match coefficient (script 119/120)
MATRIX_PERM = 1e-18


def load_profiles(csv):
    d = pd.read_csv(csv, parse_dates=["center_time"])
    strain = d.pivot(index="md_ft", columns="center_time", values="strain_millistrain")
    rate = d.pivot(index="md_ft", columns="center_time", values="strain_rate_nanostrain_s")
    return strain, rate


def pressure_from_profile(name, strain):
    """Pressure Data1DGauge from the peak-|strain| channel (millistrain -> strain)."""
    md = strain.abs().max(axis=1).idxmax()
    times = strain.columns.to_pydatetime()
    tsec = np.array([(t - times[0]).total_seconds() for t in times], float)
    s = strain.loc[md].to_numpy(float) * 1e-3
    s = s - s[0]
    gauge = Data1DGauge(
        data=P0_PSI + C_PSI_PER_STRAIN * s,
        taxis=tsec,
        start_time=times[0],
        name=f"crossing_cutout_{name}_pressure_C{C_PSI_PER_STRAIN:.2e}_{md:.0f}ft",
    )
    path = NPZ_DIR / f"{gauge.name}.npz"
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    gauge.savez(str(path))
    return path, gauge, float(md)


def set_case_paths(name, pressure_npz):
    case.PROJECT_NAME = f"crossing_cutout_tensile_{name}"
    case.MATRIX_PERM = MATRIX_PERM
    case.CROPPED_PRESSURE_PATH = pressure_npz
    case.OUTPUT_DIR = REPO_ROOT / "output" / case.PROJECT_NAME
    case.EXPORT_DIR = case.OUTPUT_DIR / "postprocessor_npz"
    case.FIG_DIR = FIG_ROOT / name


def plot_pressure_curves(gauges, md):
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for name, g in gauges.items():
        t = pd.to_datetime(g.start_time) + pd.to_timedelta(g.taxis, unit="s")
        ax.plot(t, g.data, color=COLORS[name], lw=2, label=f"{name} pressure")
    ax.set(ylabel="Injection pressure (psi)", xlabel="Time (UTC-7)",
           title=f"Pressure from DDM tensile strain @ {md:.0f} ft: P0={P0_PSI:.0f} psi, C={C_PSI_PER_STRAIN:.2e} psi/strain")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend()
    path = FIG_ROOT / "baseline_vs_inverted_pressure.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_geometry():
    builder = build_baseline_model(
        project_name="crossing_cutout_tensile_geometry",
        pressure_profile_path=str(next(iter(NPZ_DIR.glob("crossing_cutout_baseline_pressure_*.npz")))),
        hf_length_ft=case.HF_LENGTH_FT,
        shift_list_ft=np.array([case.MONITOR_SHIFT_FT]),
        angle=case.MONITOR_ANGLE_DEG,
        matrix_perm=MATRIX_PERM,
    )
    path = FIG_ROOT / "model_geometry.png"
    builder.plot_geometry(save_path=str(path), hide_legend=True, equal_aspect=True)
    plt.close("all")
    return path


def plot_comparison(name, ddm_strain, ddm_rate, md, t0):
    d = np.load(case.EXPORT_DIR / "monitor_normal_strain_rate_no_rotation.npz", allow_pickle=True)
    ds = np.load(case.EXPORT_DIR / "monitor_normal_strain_no_rotation.npz", allow_pickle=True)
    ipos = int(np.nanargmax(np.nanmax(np.abs(d["data"].astype(float)), axis=1)))
    mt = t0 + pd.to_timedelta(d["taxis"].astype(float), unit="s")
    rate = d["data"][ipos].astype(float) * 1e9
    mts = t0 + pd.to_timedelta(ds["taxis"].astype(float), unit="s")
    strain = ds["data"][ipos].astype(float) * 1e3
    i = int(np.argmin(np.abs(ddm_strain.index.to_numpy() - md)))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(ddm_strain.columns, ddm_strain.iloc[i], "k", lw=2, label=f"DDM tensile {name} @ {md:.0f} ft")
    axes[0].plot(mts, strain, color=COLORS[name], lw=2, label="MOOSE monitor")
    axes[0].set_ylabel("Strain (millistrain)")
    axes[1].plot(ddm_rate.columns, ddm_rate.iloc[i], "k", lw=2, label="DDM tensile (2h mean)")
    axes[1].plot(mt, rate, color=COLORS[name], lw=1.5, label="MOOSE monitor")
    axes[1].set_ylabel("Strain rate (nanostrain/s)")
    for ax in axes:
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(loc="upper left")
    path = FIG_ROOT / f"{name}_moose_vs_ddm_tensile.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-run", action="store_true", help="reuse existing MOOSE outputs")
    args = ap.parse_args()
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    prof, gauges, npzs, mds = {}, {}, {}, {}
    for name, csv in CASES.items():
        prof[name] = load_profiles(csv)
        npzs[name], gauges[name], mds[name] = pressure_from_profile(name, prof[name][0])
        print(f"[{name}] peak channel {mds[name]:.1f} ft, P {gauges[name].data[0]:.0f} -> "
              f"{gauges[name].data[-1]:.0f} psi (max {gauges[name].data.max():.0f})")
    print("Saved:", plot_pressure_curves(gauges, mds["baseline"]))
    print("Saved:", plot_geometry())

    for name in CASES:
        set_case_paths(name, npzs[name])
        if not args.skip_run:
            case.build_and_run_model()
        case.export_results(gauges[name])
        print("Saved:", plot_comparison(name, *prof[name], mds[name], pd.Timestamp(gauges[name].start_time)))


if __name__ == "__main__":
    main()
