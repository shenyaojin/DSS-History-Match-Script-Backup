# Compare the 3x3x3 sensitivity grid for the 2-parameter zonal inversion.
#
# Factors: A = real DSS background noise (bg1..3), B = white noise (w1..3),
#          C = zone-position offset (s0/s2/s4 layers).
# For each cell we read the recovered theta = [theta_frac, theta_srv] and compare
# against the synthetic truth (theta_frac=log10(3e-15), theta_srv=-15).
#
# Outputs:
#   grid_comparison_summary.csv   one row per (bg, white, shift) cell
#   grid_sensitivity_heatmaps.png three A x B heatmaps (one per shift) of the
#                                 combined recovery error
#   grid_main_effects.png         error vs each factor (averaged over the others)
#
# Cells that have not finished yet are skipped, so this can be run at any time.
#
# Run as:
#   python compare_grid_results.py

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

THETA_FRAC_TRUTH = float(np.log10(3e-15))   # ~ -14.5229
THETA_SRV_TRUTH = -15.0

BG_LEVELS = [1, 2, 3]
W_LEVELS = [1, 2, 3]
SHIFTS = [0, 2, 4]


def read_theta(run_dir):
    p = os.path.join(run_dir, "optimized_theta_zonal.txt")
    if not os.path.exists(p):
        return None
    t = np.loadtxt(p)
    if t.shape[0] != 2:
        return None
    return float(t[0]), float(t[1])


def main():
    rows = []
    for i in BG_LEVELS:
        for j in W_LEVELS:
            for k in SHIFTS:
                cell = f"bg{i}_w{j}_s{k}"
                th = read_theta(os.path.join(HERE, cell))
                if th is None:
                    continue
                fe = th[0] - THETA_FRAC_TRUTH
                se = th[1] - THETA_SRV_TRUTH
                rows.append({
                    "cell": cell, "bg": i, "white": j, "shift": k,
                    "theta_frac": th[0], "theta_srv": th[1],
                    "frac_err": fe, "srv_err": se,
                    "abs_frac_err": abs(fe), "abs_srv_err": abs(se),
                    "combined_err": float(np.hypot(fe, se)),
                })
    if not rows:
        print("No finished grid cells yet. Run run_all_grid.sh first.")
        return
    df = pd.DataFrame(rows)
    out_csv = os.path.join(HERE, "grid_comparison_summary.csv")
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote summary: {out_csv}  ({len(df)}/27 cells)")

    # --- heatmaps: one A x B panel per shift, colored by combined_err ---
    metric = "combined_err"
    vmax = df[metric].max() if len(df) else 1.0
    fig, axes = plt.subplots(1, len(SHIFTS), figsize=(5.2 * len(SHIFTS), 5), squeeze=False)
    for ax, k in zip(axes[0], SHIFTS):
        grid = np.full((len(BG_LEVELS), len(W_LEVELS)), np.nan)
        for a, i in enumerate(BG_LEVELS):
            for b, j in enumerate(W_LEVELS):
                sub = df[(df.bg == i) & (df.white == j) & (df.shift == k)]
                if len(sub):
                    grid[a, b] = sub[metric].iloc[0]
        im = ax.imshow(grid, origin="lower", cmap="magma_r", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(W_LEVELS))); ax.set_xticklabels([f"w{j}" for j in W_LEVELS])
        ax.set_yticks(range(len(BG_LEVELS))); ax.set_yticklabels([f"bg{i}" for i in BG_LEVELS])
        ax.set(xlabel="white noise level", ylabel="DSS bg noise level",
               title=f"position shift = {k} layers")
        for a in range(len(BG_LEVELS)):
            for b in range(len(W_LEVELS)):
                if not np.isnan(grid[a, b]):
                    ax.text(b, a, f"{grid[a, b]:.2f}", ha="center", va="center",
                            color="w" if grid[a, b] > vmax * 0.5 else "k", fontsize=9)
    fig.suptitle("Zonal 2-param sensitivity grid — combined theta error "
                 "(dex, sqrt(frac_err^2+srv_err^2))", fontsize=13)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="combined error [dex]", shrink=0.8)
    out_png = os.path.join(HERE, "grid_sensitivity_heatmaps.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Wrote heatmaps: {out_png}")

    # --- main effects: mean combined_err vs each factor (averaging the others) ---
    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, (fac, levs, lab) in enumerate([
        ("bg", BG_LEVELS, "DSS bg noise level"),
        ("white", W_LEVELS, "white noise level"),
        ("shift", SHIFTS, "position shift [layers]"),
    ]):
        m = df.groupby(fac)["combined_err"].mean()
        sd = df.groupby(fac)["combined_err"].std()
        ax2[a].errorbar(m.index, m.values, yerr=sd.values, fmt="o-", capsize=4)
        ax2[a].set(xlabel=lab, ylabel="mean combined error [dex]",
                   title=f"main effect: {fac}")
        ax2[a].grid(alpha=0.3)
    fig2.suptitle("Main effects (mean over the other two factors)", fontsize=13)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    out_png2 = os.path.join(HERE, "grid_main_effects.png")
    fig2.savefig(out_png2, dpi=140)
    print(f"Wrote main-effects plot: {out_png2}")


if __name__ == "__main__":
    main()
