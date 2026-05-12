"""
Animate the parameter history from inversion iterations.
Each frame shows one row (iteration) of the parameter map as a line plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "parameter_history.csv")

data = np.loadtxt(csv_path, delimiter=",")
n_iterations, n_params = data.shape

# Flip layer order so index 0 corresponds to the TOP (y=+50) instead of bottom.
data = data[:, ::-1]

# Ground truth: caprock α=-18 everywhere, except SRV zones
# After flip: original layer i → new index (n_params-1-i)
# Low-perm SRV:  original 60..67  → flipped 132..139
# High-perm SRV: original 128..139 → flipped 60..71
truth = np.full(n_params, -18.0)
truth[132:140] = -15.0
truth[60:72] = np.log10(3e-15)

# Global y-axis limits (include truth so it's visible)
y_min = min(data.min(), truth.min()) - 0.5
y_max = max(data.max(), truth.max()) + 0.5

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(np.arange(n_params), truth, "r--", lw=1.5, label="Ground truth", zorder=1)
(line,) = ax.plot([], [], lw=1.2, label="Inversion", zorder=2)
title = ax.set_title("")
ax.set_xlim(0, n_params - 1)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("Parameter Index")
ax.set_ylabel("Parameter Value")
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
gif_path = os.path.join(script_dir, "parameter_history.gif")
anim.save(gif_path, writer="pillow", fps=2)
print(f"Saved to {gif_path}")

plt.show()
