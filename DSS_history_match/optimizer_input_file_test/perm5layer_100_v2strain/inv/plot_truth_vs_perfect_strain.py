import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKDIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(WORKDIR, "../../../../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from DSS_analyzer_Mariner.Data2D_XT_DSS import Data2D


BASE_DIR = os.path.dirname(WORKDIR)
GROUND_TRUTH_CSV = os.path.join(BASE_DIR, "data", "obs_strain_yy.csv")
PERFECT_RUN_CSV = os.path.join(WORKDIR, "inv_output", "optimize_temp_out_main_0001.csv")
PERFECT_OBJECTIVE_CSV = os.path.join(WORKDIR, "inv_output", "optimize_temp_out.csv")
OUTPUT_PNG = os.path.join(WORKDIR, "truth_vs_perfect_strain_time_location.png")


def _csv_to_data2d(csv_path, value_column):
    df = pd.read_csv(csv_path)
    required = {"measurement_time", "measurement_ycoord", value_column}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{csv_path} is missing columns: {sorted(missing)}")

    times = np.sort(df["measurement_time"].unique())
    depths = np.sort(df["measurement_ycoord"].unique())

    table = df.pivot(
        index="measurement_ycoord",
        columns="measurement_time",
        values=value_column,
    )
    table = table.reindex(index=depths, columns=times)

    data = Data2D()
    data.data = table.to_numpy()
    data.taxis = times
    data.daxis = depths
    data.mds = depths
    return data


def main():
    truth = _csv_to_data2d(GROUND_TRUTH_CSV, "measurement_values")
    perfect_column = "simulation_values"
    run_df = pd.read_csv(PERFECT_RUN_CSV, usecols=["measurement_values", "simulation_values"])
    objective_df = pd.read_csv(PERFECT_OBJECTIVE_CSV)
    objective_value = float(objective_df["OptimizationReporter/objective_value"].iloc[-1])

    # OptimizationData writes a zero objective at the true parameter, but its
    # CSV reporter does not populate simulation_values. In that case the
    # measurement column is the perfect-parameter response by construction.
    if (
        np.nanmax(np.abs(run_df["simulation_values"].to_numpy())) == 0.0
        and objective_value < 1e-20
    ):
        perfect_column = "measurement_values"
        print(
            "simulation_values is all zero in the reporter CSV; using "
            "measurement_values for the perfect-parameter response because "
            f"the truth-probe objective is {objective_value:.3e}."
        )

    perfect = _csv_to_data2d(PERFECT_RUN_CSV, perfect_column)

    if truth.data.shape != perfect.data.shape:
        raise RuntimeError(
            f"Shape mismatch: truth {truth.data.shape}, perfect {perfect.data.shape}"
        )

    diff = Data2D()
    diff.data = perfect.data - truth.data
    diff.taxis = truth.taxis
    diff.daxis = truth.daxis
    diff.mds = truth.mds

    max_abs = max(np.nanmax(np.abs(truth.data)), np.nanmax(np.abs(perfect.data)))
    diff_abs = np.nanmax(np.abs(diff.data))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    im0 = truth.plot_water_on_ax(
        axes[0], cmap=plt.get_cmap("bwr"), timescale="hour",
        downsample=[1, 1],
    )
    im0.set_clim(-max_abs, max_abs)
    axes[0].set_title("Ground Truth strain_yy")
    axes[0].set_xlabel("Time (hour)")
    axes[0].set_ylabel("Location y (m)")
    fig.colorbar(im0, ax=axes[0], label="strain_yy")

    im1 = perfect.plot_water_on_ax(
        axes[1], cmap=plt.get_cmap("bwr"), timescale="hour",
        downsample=[1, 1],
    )
    im1.set_clim(-max_abs, max_abs)
    axes[1].set_title("Perfect-Parameter strain_yy")
    axes[1].set_xlabel("Time (hour)")
    axes[1].set_ylabel("Location y (m)")
    fig.colorbar(im1, ax=axes[1], label="strain_yy")

    im2 = diff.plot_water_on_ax(
        axes[2], cmap=plt.get_cmap("bwr"), timescale="hour",
        downsample=[1, 1],
    )
    im2.set_clim(-diff_abs, diff_abs)
    axes[2].set_title(f"Perfect - Truth\nmax |diff| = {diff_abs:.3e}")
    axes[2].set_xlabel("Time (hour)")
    axes[2].set_ylabel("Location y (m)")
    fig.colorbar(im2, ax=axes[2], label="strain_yy difference")

    for ax in axes:
        ax.axhline(-20.0, color="k", linewidth=0.8, alpha=0.45)
        ax.axhline(-16.0, color="k", linewidth=0.8, alpha=0.45)
        ax.axhline(14.0, color="k", linewidth=0.8, alpha=0.45)
        ax.axhline(20.0, color="k", linewidth=0.8, alpha=0.45)

    fig.suptitle("Synthetic Truth vs Perfect-Parameter Strain Response", fontsize=14)
    fig.savefig(OUTPUT_PNG, dpi=220)
    print(f"Saved figure to: {OUTPUT_PNG}")
    print(f"truth shape: {truth.data.shape}, perfect shape: {perfect.data.shape}")
    print(f"max |truth|: {np.nanmax(np.abs(truth.data)):.6e}")
    print(f"max |perfect - truth|: {diff_abs:.6e}")


if __name__ == "__main__":
    main()
