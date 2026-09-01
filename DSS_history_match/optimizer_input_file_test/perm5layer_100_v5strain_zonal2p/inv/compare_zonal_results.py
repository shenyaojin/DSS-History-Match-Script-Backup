# Compare the 2-parameter zonal inversion across all noise cases and both
# zone variants (exact vs perturbed).
#
# The whole point of this study: with the anomaly geometry KNOWN and only two
# scalars (theta_frac, theta_srv) to invert, the inversion should be almost
# completely INSENSITIVE to measurement noise. This script quantifies that by
# reading each run's recovered theta (and its expanded 200-dim alpha) and
# comparing against the synthetic truth, writing:
#   zonal_comparison_summary.csv   one row per (variant, level, which) with the
#                                  recovered theta, zone means, and rel-L2 errors
#   zonal_alpha_overlay.png        alpha profiles of every run vs truth
#   zonal_theta_vs_noise.png       recovered theta_frac/theta_srv vs noise level
#
# Runs that have not finished yet are skipped, so this can be run at any time.
# The truth zones are ALWAYS the exact ones (the synthetic anomalies live at the
# exact windows); under the 'pert' variant the recovered anomaly sits at shifted
# layers, so its zone means / rel-L2 (measured on the exact mask) degrade -- that
# degradation is the zone-misplacement diagnostic, not a noise effect.
#
# Run as:
#   python compare_zonal_results.py

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

# 14 noise cases: (case token, display label). case token == <family>[_<tag>].
CASES = [
    ("clean", "clean"),
    ("median_1pct", "median 1%"), ("median_2pct", "median 2%"),
    ("median_5pct", "median 5%"), ("median_10pct", "median 10%"),
    ("peak_0p5pct", "peak 0.5%"), ("peak_1pct", "peak 1%"),
    ("peak_2pct", "peak 2%"), ("peak_5pct", "peak 5%"),
    ("rfsdss_0p5pct", "rfsdss 0.5%"), ("rfsdss_1pct", "rfsdss 1%"),
    ("rfsdss_2pct", "rfsdss 2%"), ("rfsdss_5pct", "rfsdss 5%"),
    ("rfsdss_10pct", "rfsdss 10%"),
]
VARIANTS = ["exact", "pert"]
VARIANT_STYLE = {"exact": "-", "pert": "-."}


def layer_geometry():
    y_bottom = -50.0 + np.arange(TOTAL_LAYERS) * LAYER_HEIGHT
    y_top = y_bottom + LAYER_HEIGHT
    y_center = 0.5 * (y_bottom + y_top)
    return y_bottom, y_top, y_center


def zone_masks():
    """EXACT truth zones (the synthetic anomalies live here regardless of variant)."""
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


THETA_FRAC_TRUTH = float(np.log10(3e-15))   # ~ -14.5229
THETA_SRV_TRUTH = -15.0


def rel_l2(a, truth, mask=None):
    if mask is None:
        num = np.linalg.norm(a - truth)
        den = np.linalg.norm(truth)
    else:
        num = np.linalg.norm(a[mask] - truth[mask])
        den = np.linalg.norm(truth[mask])
    return float(num / den) if den != 0 else float("nan")


def read_theta(run_dir, which):
    fname = {
        "final": "optimized_theta_zonal.txt",
        "best_data": "best_data_theta_zonal.txt",
        "best_total": "best_total_theta_zonal.txt",
    }[which]
    path = os.path.join(run_dir, fname)
    if not os.path.exists(path):
        return (np.nan, np.nan)
    t = np.loadtxt(path)
    return (float(t[0]), float(t[1]))


def summarize(variant, label, which, alpha, theta, truth):
    free_window, low_srv, fracture = zone_masks()
    return {
        "variant": variant,
        "level": label,
        "which": which,
        "theta_frac": theta[0],
        "theta_srv": theta[1],
        "frac_err": theta[0] - THETA_FRAC_TRUTH,
        "srv_err": theta[1] - THETA_SRV_TRUTH,
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

    variants_which = [
        ("final", "optimized_alphas_L1.txt"),
        ("best_data", "best_data_alpha_L1.txt"),
        ("best_total", "best_total_alpha_L1.txt"),
    ]

    rows = []
    final_profiles = {}   # (variant, label) -> alpha
    theta_track = {v: {"labels": [], "frac": [], "srv": []} for v in VARIANTS}

    for variant in VARIANTS:
        for case, label in CASES:
            run_dir = os.path.abspath(os.path.join(HERE, f"{variant}_{case}"))
            for which, fname in variants_which:
                path = os.path.join(run_dir, fname)
                if not os.path.exists(path):
                    continue
                alpha = np.loadtxt(path)
                if alpha.shape[0] != TOTAL_LAYERS:
                    print(f"WARNING: {path} has {alpha.shape[0]} layers; skipping.")
                    continue
                theta = read_theta(run_dir, which)
                rows.append(summarize(variant, label, which, alpha, theta, truth))
                if which == "final":
                    final_profiles[(variant, label)] = alpha
                    theta_track[variant]["labels"].append(label)
                    theta_track[variant]["frac"].append(theta[0])
                    theta_track[variant]["srv"].append(theta[1])

    truth_row = summarize("truth", "truth", "truth", truth,
                          (THETA_FRAC_TRUTH, THETA_SRV_TRUTH), truth)
    rows.insert(0, truth_row)

    summary = pd.DataFrame(rows)
    out_csv = os.path.join(HERE, "zonal_comparison_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote summary: {out_csv}")

    if not final_profiles:
        print("No final results found yet; skipping plots.")
        print("Run run_all_zonal.sh first, then re-run this script.")
        return

    # --- alpha overlay ---
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.plot(truth, y_center, color="k", lw=2.5, label="truth", zorder=5)
    keys = sorted(final_profiles.keys())
    cmap = plt.cm.viridis(np.linspace(0, 0.9, max(1, len(keys))))
    for color, key in zip(cmap, keys):
        variant, label = key
        ax.plot(final_profiles[key], y_center, lw=1.3, color=color,
                linestyle=VARIANT_STYLE.get(variant, "-"),
                label=f"{variant} {label}")
    for y in (-20.0, -16.0, 14.0, 20.0):
        ax.axhline(y, color="k", lw=0.7, alpha=0.3)
    ax.set_xlabel("log10 permeability alpha")
    ax.set_ylabel("Layer center y (m)")
    ax.set_title("Zonal 2-parameter inverted alpha vs noise\n"
                 "(solid = exact zones, dash-dot = perturbed zones)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=6, ncol=2)
    fig.tight_layout()
    out_png = os.path.join(HERE, "zonal_alpha_overlay.png")
    fig.savefig(out_png, dpi=200)
    print(f"Wrote overlay plot: {out_png}")

    # --- recovered theta vs noise level (the noise-insensitivity money plot) ---
    fig2, ax2 = plt.subplots(1, 2, figsize=(15, 6))
    for variant in VARIANTS:
        tr = theta_track[variant]
        if not tr["labels"]:
            continue
        x = np.arange(len(tr["labels"]))
        style = "o-" if variant == "exact" else "s--"
        ax2[0].plot(x, tr["frac"], style, label=variant)
        ax2[1].plot(x, tr["srv"], style, label=variant)
        if variant == VARIANTS[0]:
            for a in ax2:
                a.set_xticks(x)
                a.set_xticklabels(tr["labels"], rotation=60, ha="right", fontsize=7)
    ax2[0].axhline(THETA_FRAC_TRUTH, color="k", ls=":", label=f"truth {THETA_FRAC_TRUTH:.3f}")
    ax2[1].axhline(THETA_SRV_TRUTH, color="k", ls=":", label=f"truth {THETA_SRV_TRUTH:.1f}")
    ax2[0].set(ylabel="recovered theta_frac", title="Fracture-zone log10 perm vs noise")
    ax2[1].set(ylabel="recovered theta_srv", title="SRV-zone log10 perm vs noise")
    for a in ax2:
        a.grid(True, alpha=0.25)
        a.legend(fontsize=8)
    fig2.suptitle("Zonal 2-parameter recovery vs noise level "
                  "(flat line = noise-insensitive)", fontsize=13)
    fig2.tight_layout(rect=(0, 0, 1, 0.97))
    out_png2 = os.path.join(HERE, "zonal_theta_vs_noise.png")
    fig2.savefig(out_png2, dpi=150)
    print(f"Wrote theta-vs-noise plot: {out_png2}")


if __name__ == "__main__":
    main()
