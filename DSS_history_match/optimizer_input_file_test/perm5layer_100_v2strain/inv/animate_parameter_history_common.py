"""Shared parameter-history animation helper for the inversion runners."""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_parameter_history(csv_name, gif_name, label="Inversion"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_name)
    gif_path = os.path.join(script_dir, gif_name)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Parameter history not found: {csv_path}")

    data = np.loadtxt(csv_path, delimiter=",")
    data = np.atleast_2d(data)
    n_iterations, n_params = data.shape

    # Flip layer order so index 0 corresponds to top y=+50 instead of bottom.
    data = data[:, ::-1]

    truth = np.full(n_params, -18.0)
    truth[132:140] = -15.0
    truth[60:72] = np.log10(3e-15)

    y_min = min(data.min(), truth.min()) - 0.5
    y_max = max(data.max(), truth.max()) + 0.5

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(np.arange(n_params), truth, "r--", lw=1.5, label="Ground truth", zorder=1)
    ax.axvspan(132, 139, color="orange", alpha=0.12, label="Low-perm SRV zone")
    ax.axvspan(60, 71, color="green", alpha=0.12, label="High-perm SRV zone")
    (line,) = ax.plot([], [], lw=1.2, label=label, zorder=2)
    title = ax.set_title("")
    ax.set_xlim(0, n_params - 1)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Flipped Parameter Index")
    ax.set_ylabel("Alpha Value")
    ax.legend(loc="upper right")

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        x = np.arange(n_params)
        line.set_data(x, data[frame])
        title.set_text(f"{label}: iteration {frame + 1} / {n_iterations}")
        return line, title

    anim = FuncAnimation(
        fig, update, frames=n_iterations, init_func=init, interval=500, blit=True
    )
    plt.tight_layout()
    anim.save(gif_path, writer="pillow", fps=2)
    print(f"Saved to {gif_path}")

    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()


__all__ = ["animate_parameter_history"]
