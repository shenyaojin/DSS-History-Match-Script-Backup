import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKDIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STRAIN_CSV = os.path.join(WORKDIR, "inv_output", "optimize_temp_out_main_0001.csv")
DEFAULT_OBJECTIVE_CSV = os.path.join(WORKDIR, "inv_output", "optimize_temp_out.csv")
DEFAULT_OUTPUT = os.path.join(WORKDIR, "inversion_qc_strain_and_alpha.png")

ALPHA_FILES = {
    "truth": os.path.join(WORKDIR, "true_alphas_TV.txt"),
    "tikhonov": os.path.join(WORKDIR, "optimized_alphas.txt"),
    "tv": os.path.join(WORKDIR, "optimized_alphas_TV.txt"),
}

TOTAL_LAYERS = 200
LAYER_HEIGHT = 0.5
BACKGROUND_ALPHA = -18.0


def build_synthetic_truth_alpha():
    alpha = np.full(TOTAL_LAYERS, BACKGROUND_ALPHA)
    y_bottom = -50.0 + np.arange(TOTAL_LAYERS) * LAYER_HEIGHT
    y_top = y_bottom + LAYER_HEIGHT
    alpha[(y_bottom >= -20.0) & (y_top <= -16.0)] = -15.0
    alpha[(y_bottom >= 14.0) & (y_top <= 20.0)] = np.log10(3e-15)
    return alpha


def read_objective(objective_csv):
    if not objective_csv or not os.path.exists(objective_csv):
        return None
    df = pd.read_csv(objective_csv)
    col = "OptimizationReporter/objective_value"
    if col not in df.columns or df.empty:
        return None
    return float(df[col].iloc[-1])


def choose_simulation_column(strain_csv, objective_csv):
    df = pd.read_csv(strain_csv, usecols=["measurement_values", "simulation_values"])
    sim = df["simulation_values"].to_numpy()
    objective = read_objective(objective_csv)
    if np.nanmax(np.abs(sim)) == 0.0:
        return None, objective
    return "simulation_values", objective


def pivot_grid(df, value_column):
    table = df.pivot(
        index="measurement_ycoord",
        columns="measurement_time",
        values=value_column,
    )
    table = table.sort_index().sort_index(axis=1)
    return table.index.to_numpy(), table.columns.to_numpy(), table.to_numpy()


def plot_grid(ax, times, depths, values, title, clim):
    im = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=[times.min() / 3600.0, times.max() / 3600.0, depths.min(), depths.max()],
        cmap="bwr",
        vmin=-clim,
        vmax=clim,
    )
    ax.set_title(title)
    ax.set_xlabel("Time (hour)")
    ax.set_ylabel("Location y (m)")
    for y in (-20.0, -16.0, 14.0, 20.0):
        ax.axhline(y, color="k", linewidth=0.8, alpha=0.45)
    return im


def load_alpha_profiles(extra_alpha_file=None, extra_alpha_label=None):
    profiles = {"truth": build_synthetic_truth_alpha()}
    for name, path in ALPHA_FILES.items():
        if os.path.exists(path):
            profiles[name] = np.loadtxt(path)
    if extra_alpha_file:
        profiles[extra_alpha_label or "qc alpha"] = np.loadtxt(extra_alpha_file)
    return profiles


def summarize_alpha(name, alpha):
    active = np.where(np.abs(alpha - BACKGROUND_ALPHA) > 1e-6)[0]
    if len(active) == 0:
        return f"{name}: active_layers=0, min={alpha.min():.6g}, max={alpha.max():.6g}"
    return (
        f"{name}: active_layers={len(active)}, "
        f"active_range={active[0] + 1}..{active[-1] + 1}, "
        f"min={alpha.min():.6g}, max={alpha.max():.6g}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot strain reporter output and saved inversion alpha profiles."
    )
    parser.add_argument("--strain-csv", default=DEFAULT_STRAIN_CSV)
    parser.add_argument("--objective-csv", default=DEFAULT_OBJECTIVE_CSV)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha-file", default=None, help="Alpha vector actually used by this QC run.")
    parser.add_argument("--alpha-label", default=None, help="Legend label for --alpha-file.")
    args = parser.parse_args()

    if not os.path.exists(args.strain_csv):
        raise FileNotFoundError(args.strain_csv)

    sim_column, objective = choose_simulation_column(args.strain_csv, args.objective_csv)
    df = pd.read_csv(args.strain_csv)
    required = {"measurement_time", "measurement_ycoord", "measurement_values"}
    if sim_column is not None:
        required.add(sim_column)
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{args.strain_csv} is missing columns: {sorted(missing)}")

    depths, times, truth = pivot_grid(df, "measurement_values")
    if sim_column is None:
        sim = np.full_like(truth, np.nan)
        residual = np.full_like(truth, np.nan)
    else:
        _, _, sim = pivot_grid(df, sim_column)
        residual = sim - truth

    sim_abs = 0.0 if np.all(np.isnan(sim)) else np.nanmax(np.abs(sim))
    strain_clim = max(np.nanmax(np.abs(truth)), sim_abs)
    residual_clim = 0.0 if np.all(np.isnan(residual)) else np.nanmax(np.abs(residual))
    if residual_clim == 0.0 or np.isnan(residual_clim):
        residual_clim = strain_clim if strain_clim > 0.0 else 1.0

    profiles = load_alpha_profiles(args.alpha_file, args.alpha_label)
    y_centers = -50.0 + (np.arange(TOTAL_LAYERS) + 0.5) * LAYER_HEIGHT

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    im0 = plot_grid(axes[0, 0], times, depths, truth, "Measured strain_yy", strain_clim)
    fig.colorbar(im0, ax=axes[0, 0], label="strain_yy")

    if sim_column is None:
        axes[0, 1].axis("off")
        axes[0, 1].text(
            0.5,
            0.5,
            "simulation_values was not written\nby this OptimizationData CSV",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )
    else:
        im1 = plot_grid(
            axes[0, 1],
            times,
            depths,
            sim,
            f"Simulated strain_yy ({sim_column})",
            strain_clim,
        )
        fig.colorbar(im1, ax=axes[0, 1], label="strain_yy")

    if sim_column is None:
        axes[1, 0].axis("off")
        axes[1, 0].text(
            0.5,
            0.5,
            "strain residual unavailable\nfrom this reporter CSV",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
    else:
        im2 = plot_grid(
            axes[1, 0],
            times,
            depths,
            residual,
            f"Simulation - measurement, max |res| = {np.nanmax(np.abs(residual)):.3e}",
            residual_clim,
        )
        fig.colorbar(im2, ax=axes[1, 0], label="strain_yy residual")

    ax = axes[1, 1]
    for name, alpha in profiles.items():
        ax.plot(alpha, y_centers, label=name)
    for y in (-20.0, -16.0, 14.0, 20.0):
        ax.axhline(y, color="k", linewidth=0.8, alpha=0.35)
    ax.set_title("Alpha profiles")
    ax.set_xlabel("log10 permeability alpha")
    ax.set_ylabel("Layer center y (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    objective_text = "not available" if objective is None else f"{objective:.6e}"
    fig.suptitle(
        f"Inversion QC: {os.path.relpath(args.strain_csv, WORKDIR)} | "
        f"objective={objective_text}",
        fontsize=12,
    )
    fig.savefig(args.output, dpi=220)

    print(f"Saved QC figure to: {args.output}")
    print(f"strain_csv: {args.strain_csv}")
    print(f"simulation column used: {sim_column}")
    print(f"objective: {objective_text}")
    print(f"max |truth|: {np.nanmax(np.abs(truth)):.6e}")
    if sim_column is None:
        print("max |simulation - measurement|: unavailable (simulation_values all zero)")
    else:
        print(f"max |simulation - measurement|: {np.nanmax(np.abs(residual)):.6e}")
    for name, alpha in profiles.items():
        print(summarize_alpha(name, alpha))


if __name__ == "__main__":
    main()
