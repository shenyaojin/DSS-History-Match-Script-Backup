# Compare L1 inversion results across the realistic RFS-DSS noise levels.
#
# Reads each inv/rfsdss_<tag>/optimized_alphas_L1.txt (plus best-data /
# best-total vectors if present), compares them against the synthetic truth
# alpha, and writes:
#   rfsdss_comparison_summary.csv   one row per (level, which) with zone means
#                                   and relative-L2 errors
#   rfsdss_alpha_overlay.png        alpha profiles of all levels vs truth
#
# The clean (v2) result is included as the noise-free reference. Levels that
# have not been run yet are skipped, so this can be run at any time.
#
# Run as:
#   python compare_rfsdss_results.py

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

TOTAL_LAYERS = 200
LAYER_HEIGHT = 0.5
BACKGROUND_ALPHA = -18.0

# (label, family, run_dir). family is one of: clean, rfsdss.
LEVELS = [
    ("clean (v2)", "clean", os.path.join(HERE, "..", "..", "perm5layer_100_v2strain", "inv")),
    ("rfsdss 0.5%", "rfsdss", os.path.join(HERE, "rfsdss_0p5pct")),
    ("rfsdss 1%", "rfsdss", os.path.join(HERE, "rfsdss_1pct")),
    ("rfsdss 2%", "rfsdss", os.path.join(HERE, "rfsdss_2pct")),
    ("rfsdss 5%", "rfsdss", os.path.join(HERE, "rfsdss_5pct")),
    ("rfsdss 10%", "rfsdss", os.path.join(HERE, "rfsdss_10pct")),
]

FAMILY_STYLE = {"clean": "-", "rfsdss": "-"}


def layer_geometry():
    y_bottom = -50.0 + np.arange(TOTAL_LAYERS) * LAYER_HEIGHT
    y_top = y_bottom + LAYER_HEIGHT
    y_center = 0.5 * (y_bottom + y_top)
    return y_bottom, y_top, y_center


def zone_masks():
    y_bottom, y_top, y_center = layer_geometry()
    free_window = (-25.0 <= y_center) & (y_center <= 25.0)
    low_srv = (y_bottom >= -20.0) & (y_top <= -16.0)
    fracture = (y_bottom >= 14.0) & (y_top <= 20.0)
    return free_window, low_srv, fracture


def build_truth_alpha():
    alpha = np.full(TOTAL_LAYERS, BACKGROUND_ALPHA)
    _, low_srv, fracture = zone_masks()
    alpha[low_srv] = -15.0
    alpha[fracture] = np.log10(3e-15)
    return alpha


def rel_l2(a, truth, mask=None):
    if mask is None:
        num = np.linalg.norm(a - truth)
        den = np.linalg.norm(truth)
    else:
        num = np.linalg.norm(a[mask] - truth[mask])
        den = np.linalg.norm(truth[mask])
    return float(num / den) if den != 0 else float("nan")


def summarize(label, family, which, alpha, truth):
    free_window, low_srv, fracture = zone_masks()
    return {
        "level": label,
        "family": family,
        "which": which,
        "low_srv_mean": float(np.mean(alpha[low_srv])),
        "fracture_mean": float(np.mean(alpha[fracture])),
        "matrix_free_mean": float(np.mean(alpha[free_window & ~low_srv & ~fracture])),
        "max_alpha_err": float(np.max(np.abs(alpha - truth))),
        "rel_l2_all": rel_l2(alpha, truth),
        "rel_l2_free": rel_l2(alpha, truth, free_window),
    }


def main():
    truth = build_truth_alpha()
    _, _, y_center = layer_geometry()

    variants = [
        ("final", "optimized_alphas_L1.txt"),
        ("best_data", "best_data_alpha_L1.txt"),
        ("best_total", "best_total_alpha_L1.txt"),
    ]

    rows = []
    final_profiles = {}
    for label, family, run_dir in LEVELS:
        run_dir = os.path.abspath(run_dir)
        for which, fname in variants:
            path = os.path.join(run_dir, fname)
            if not os.path.exists(path):
                continue
            alpha = np.loadtxt(path)
            if alpha.shape[0] != TOTAL_LAYERS:
                print(f"WARNING: {path} has {alpha.shape[0]} layers, expected {TOTAL_LAYERS}; skipping.")
                continue
            rows.append(summarize(label, family, which, alpha, truth))
            if which == "final":
                final_profiles[label] = (family, alpha)

    truth_row = summarize("truth", "truth", "truth", truth, truth)
    rows.insert(0, truth_row)

    summary = pd.DataFrame(rows)
    out_csv = os.path.join(HERE, "rfsdss_comparison_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote summary: {out_csv}")

    if not final_profiles:
        print("No final alpha vectors found yet; skipping overlay plot.")
        print("Run run_all_rfsdss.sh first, then re-run this script.")
        return

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.plot(truth, y_center, color="k", lw=2.5, label="truth", zorder=5)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(final_profiles)))
    for color, (label, (family, alpha)) in zip(cmap, final_profiles.items()):
        ax.plot(
            alpha,
            y_center,
            lw=1.6,
            color=color,
            linestyle=FAMILY_STYLE.get(family, "-"),
            label=f"L1 final, {label}",
        )
    for y in (-20.0, -16.0, 14.0, 20.0):
        ax.axhline(y, color="k", lw=0.7, alpha=0.3)
    ax.set_xlabel("log10 permeability alpha")
    ax.set_ylabel("Layer center y (m)")
    ax.set_title("L1 inverted alpha vs realistic RFS-DSS noise level")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_png = os.path.join(HERE, "rfsdss_alpha_overlay.png")
    fig.savefig(out_png, dpi=200)
    print(f"Wrote overlay plot: {out_png}")


if __name__ == "__main__":
    main()
