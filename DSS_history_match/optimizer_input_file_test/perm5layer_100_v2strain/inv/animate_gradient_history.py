"""
Animate the gradient history from inversion iterations.
Each frame shows one row (iteration) of the gradient map as a line plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Load data
# script_dir = os.path.dirname(os.path.abspath(__file__))
# csv_path = os.path.join(script_dir, "gradient_history.csv")

csv_path = "scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100_v2strain/inv/gradient_history.csv"

data = np.loadtxt(csv_path, delimiter=",")
n_iterations, n_params = data.shape

# Flip layer order so index 0 corresponds to the TOP (y=+50) instead of bottom.
data = data[:, ::-1]

# Global y-axis limits
y_min = data.min() - 0.05
y_max = data.max() + 0.05

# Ground truth zones (for visual reference — shade SRV layers)
# After flip: original 60..67 → 132..139, original 128..139 → 60..71
fig, ax = plt.subplots(figsize=(12, 5))
ax.axhline(0.0, color="r", ls="--", lw=1.5, label="Zero (truth)", zorder=1)
ax.axvspan(132, 139, color="orange", alpha=0.15, label="Low-perm SRV zone")
ax.axvspan(60, 71, color="green", alpha=0.15, label="High-perm SRV zone")
(line,) = ax.plot([], [], lw=1.2, label="Gradient", zorder=2)
title = ax.set_title("")
ax.set_xlim(0, n_params - 1)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("Parameter Index")
ax.set_ylabel("Gradient Value")
ax.legend(loc="upper right")


def init():
    line.set_data([], [])
    return (line,)


def update(frame):
    x = np.arange(n_params)
    line.set_data(x, data[frame])
    title.set_text(f"Iteration {frame + 1} / {n_iterations}")
    return line, title


anim = FuncAnimation(
    fig, update, frames=n_iterations, init_func=init, interval=500, blit=True
)

plt.tight_layout()

# Save as GIF
gif_path = os.path.join("scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100_v2strain/inv", "gradient_history.gif")
anim.save(gif_path, writer="pillow", fps=2)
print(f"Saved to {gif_path}")

plt.show()
