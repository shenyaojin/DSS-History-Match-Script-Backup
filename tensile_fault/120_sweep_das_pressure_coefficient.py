"""
Phase 1 driver: sweep the DAS-pressure coefficient by running the forward
tensile-fault model at several target peaks and scoring each against the
observed DAS strain-rate (pre-2/28). Prints a ranked table and a score-vs-C plot.

Peaks are centered on the Phase-0 least-squares bracket (C ~ 3.1e7 psi/strain,
i.e. a peak near 8000 psi). Adjust PEAKS as needed.

Run from the repository root (this launches multiple MOOSE runs, sequentially):
    python scripts/tensile_fault/120_sweep_das_pressure_coefficient.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_119 = REPO_ROOT / "scripts" / "tensile_fault" / "119_run_das_pressure_case.py"
spec = importlib.util.spec_from_file_location("das_pressure_case", SCRIPT_119)
das_case = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(das_case)


# Target peaks (psi) spanning the Phase-0 bracket. P_sim ends at each peak.
PEAKS = [5000.0, 6500.0, 8000.0, 9500.0]
P0_PSI = 2700.0
MATRIX_PERM = 1e-18

FIG_DIR = REPO_ROOT / "figs" / "tensile_fault_qc" / "das_pressure_sweep"


def main():
    das_case.P0_PSI = P0_PSI
    das_case.MATRIX_PERM = MATRIX_PERM

    results = []
    for peak in PEAKS:
        das_case.PEAK_PSI = peak
        print(f"\n===== running peak {peak:.0f} psi =====")
        try:
            results.append(das_case.main())
        except Exception as exc:  # keep the sweep going if one case fails
            print(f"!! peak {peak:.0f} psi failed: {exc}")

    if not results:
        print("No successful cases.")
        return

    results.sort(key=lambda r: r["rms"])
    print("\n===== ranked by pre-2/28 RMS misfit =====")
    print(f"{'peak_psi':>9} {'C(psi/strain)':>14} {'RMS':>8} {'NRMS':>7} {'scale':>7} {'r':>6}")
    for r in results:
        print(f"{r['peak_psi']:9.0f} {r['coefficient']:14.3e} "
              f"{r['rms']:8.4f} {r['nrms']:7.3f} {r['scale']:7.3f} {r['corr']:6.3f}")
    best = results[0]
    print(f"\nBest: peak={best['peak_psi']:.0f} psi, C={best['coefficient']:.3e} psi/strain "
          f"(RMS={best['rms']:.4f}, r={best['corr']:.3f})")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cs = [r["coefficient"] for r in results]
    rms = [r["rms"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(cs, rms, "o-", color="tab:blue")
    for r in results:
        ax.annotate(f"{r['peak_psi']:.0f}", (r["coefficient"], r["rms"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Coefficient C (psi/strain)")
    ax.set_ylabel("pre-2/28 RMS misfit (nanostrain/s)")
    ax.set_title("DAS-pressure coefficient sweep")
    ax.grid(True, ls=":", alpha=0.4)
    path = FIG_DIR / "coefficient_vs_rms.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved sweep summary -> {path}")


if __name__ == "__main__":
    main()
