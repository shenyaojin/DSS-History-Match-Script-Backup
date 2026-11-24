"""Why MOOSE tensile has strong negative lobes and the DDM tensile does not: analytic check.

2-D plane-strain pressurized crack (Westergaard, infinite medium, no walls), half-length a, face pressure p.
Fiber = line perpendicular to the crack plane at along-strike position x0. DAS strain = 10-m gauge
displacement difference (includes the opening jump if the fiber crosses the crack).
  Z    = p (z / sqrt(z^2 - a^2) - 1),   Zbar = p (sqrt(z^2 - a^2) - z)
  2 mu u_y = 2 (1 - nu) Im Zbar - y Re Z
Cases (lengths in ft):
  DDM-like  : a = 1920 (3840-ft ellipse), fiber 21 ft BEYOND the tip (Pengchao's fixed_plane_geometry.csv)
  MOOSE-like: a = 2617 (5234-ft V11 SRV),  fiber 1500 ft INSIDE the tip (V11 well position)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
FIG = REPO / "figs" / "tensile_fault_qc" / "moose_tensile_ddm_workflow_v2"
NU, GAUGE = 0.2, 10 / 0.3048
mu = 1.0  # shapes only; amplitudes are normalised


def uy(x, y, a, p=1.0):
    z = x + 1j * y
    s = np.sqrt(z - a) * np.sqrt(z + a)          # branch cut on the crack, -a < x < a
    Z, Zb = p * (z / s - 1), p * (s - z)
    return (2 * (1 - NU) * Zb.imag - y * Z.real) / (2 * mu)


def das(x0, a, y):
    eps = 1e-6                                   # keep off the crack face exactly
    return (uy(x0, y + GAUGE / 2 + eps, a) - uy(x0, y - GAUGE / 2 + eps, a)) / GAUGE


y = np.linspace(-300, 300, 1201)                 # fault-normal offset (ft), = MD - 10365.1
cases = [("DDM-like: fiber 21 ft beyond the tip of a 3840-ft crack", 1920 + 21, 1920, "#d0731f"),
         ("MOOSE-like: fiber 1500 ft inside the tip of a 5234-ft crack", 2617 - 1500, 2617, "#1f5fbf")]
fig, ax = plt.subplots(figsize=(7.5, 7))
for label, x0, a, col in cases:
    e = das(x0, a, y)
    e /= e.max()
    w = np.abs(y) <= 145                         # the 44-channel DAS window (10205-10494 ft)
    print(f"{label}: min/peak in DAS window {e[w].min():+.3f}, at +-145 ft {e[np.argmin(abs(y-145))]:+.3f}, "
          f"at +-300 ft {e[-1]:+.3f}")
    ax.plot(e, 10365.1 + y, color=col, lw=2, label=label)
ax.axvline(0, color="k", lw=.6)
ax.axhspan(10205, 10494, color="0.9", zorder=0, label="DAS window")
ax.set(ylim=(10365.1 + 300, 10365.1 - 300), xlabel="Strain / peak", ylabel="Measured depth (ft)",
       title="Analytic 2-D pressurized crack, infinite medium (no walls)")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(FIG / "analytic_crack_lobes.png", dpi=130)
