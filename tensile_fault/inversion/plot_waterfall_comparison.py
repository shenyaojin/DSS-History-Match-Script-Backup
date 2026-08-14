"""Observed vs simulated strain waterfalls, and the two bases that make up the simulation.

All six panels share one diverging colour scale so the residual can be read against the signal
it came from rather than against its own stretched range.

  (A) observed                      DAS, 4-hour means, T1-referenced
  (B) model  a.MOOSE + b.DDM        what the inversion fits
  (C) residual  observed - model
  (D) a.MOOSE tensile               the poroelastic component
  (E) b.DDM shear                   the fault-2 slip component (zero before T2)
  (F) DDM total, as uploaded        two-fault direct strain, for reference

Everything is on the 4-hour window grid, which is the only grid the observed and simulated
products share: the upload carries observed 4h profiles (no observed waterfall) plus DDM
waterfalls at full resolution. If a full-resolution observed waterfall is wanted, it has to be
rebuilt from data_fervo/fiberis_format/LFDAS/*.npz via build_observation.py.
"""
import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fault_geometry as fg

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "output" / "inversion"

_ap = argparse.ArgumentParser(description="Observed vs simulated strain waterfalls")
_ap.add_argument("--project", default="v10_final")
_ap.add_argument("--label", default=None)
args = _ap.parse_args()
PROJ = args.project
LABEL = args.label or PROJ.split("_")[0].upper()

FIG = REPO / "figs" / "tensile_fault_qc" / PROJ
FIG.mkdir(parents=True, exist_ok=True)
FT = 0.3048
STAR, SHEAR_MD = fg.STAR_1200, fg.SHEAR_MD_1200
T1, T2 = pd.Timestamp(fg.T1_1200), pd.Timestamp(fg.T2_1200)
DEC = REPO / "data_fervo" / "legacy" / "07152026_decomposed"
SHEAR_CSV = DEC / f"fault2_shear_strain_4h_{fg.SUFFIX_1200}.csv"
TOTAL_CSV = REPO / "data_fervo" / "legacy" / "07152026" / \
    f"two_fault_direct_strain_4h_{fg.SUFFIX_1200}.csv"

# ---------------------------------------------------------------- observation + bases
z = np.load(OUT / "observation_T1_1200.npz", allow_pickle=True)
O_full = np.asarray(z["strain_4h"], float)
o_md = np.asarray(z["md_ft"], float)
o_win = pd.DatetimeIndex([pd.Timestamp(str(s)) for s in z["window_starts"]])


def on_obs_grid(path):
    df = pd.read_csv(path)
    md = df["measured_depth_ft"].to_numpy(float)
    win = pd.DatetimeIndex(pd.to_datetime(df.columns[1:]))
    M = df.iloc[:, 1:].to_numpy(float)
    return md, win, M


b_md, b_win, B_full = on_obs_grid(SHEAR_CSV)
t_md, t_win, T_full = on_obs_grid(TOTAL_CSV)
common = pd.DatetimeIndex([t for t in o_win if t in set(b_win)])
nt = len(common)
O = O_full[:, [list(o_win).index(t) for t in common]]
B = np.array([np.interp(o_md, b_md, B_full[:, list(b_win).index(t)]) for t in common]).T
DDMTOT = np.array([np.interp(o_md, t_md, T_full[:, list(t_win).index(t)]) for t in common]).T
tc_s = np.array([(t - T1).total_seconds() for t in common])

d = REPO / "output" / PROJ
taxis_s = pd.read_csv(d / f"{PROJ}_input_csv.csv")["time"].to_numpy(float)
vpp = sorted(glob.glob(str(d / f"{PROJ}_input_csv_fiber_strain_sampler_*ft_*.csv")))
n = min(len(vpp), len(taxis_s))
y = None
cols = []
for f in vpp[:n]:
    dd = pd.read_csv(f)
    if len(dd) and y is None:
        y = dd.sort_values("y")["y"].to_numpy(float)
    cols.append(dd.sort_values("y")["strain_yy"].to_numpy(float) if len(dd) else None)
cols = [c if c is not None else np.zeros_like(y) for c in cols]   # t=0 has no deformation
ms = np.column_stack(cols)
ms = (ms - ms[:, [0]]) * 1e3
m_md = STAR + y / FT
order = np.argsort(m_md)
m_md, ms = m_md[order], ms[order]
M_t = np.array([np.interp(tc_s, taxis_s[:n], ms[k, :]) for k in range(ms.shape[0])])
M = np.array([np.interp(o_md, m_md, M_t[:, j]) for j in range(nt)]).T

# ---------------------------------------------------------------- per-window NNLS
a = np.full(nt, np.nan)
b = np.full(nt, np.nan)
bscale = float(np.nanmax(np.abs(B)))
for j in range(nt):
    dj, mj, bj = O[:, j], M[:, j], B[:, j]
    v = np.isfinite(dj) & np.isfinite(mj) & np.isfinite(bj)
    if v.sum() < 3:
        continue
    if float(np.nanmax(np.abs(bj[v]))) > 1e-3 * bscale:
        a[j], b[j] = nnls(np.column_stack([mj[v], bj[v]]), dj[v])[0]
    else:
        den = float(np.dot(mj[v], mj[v]))
        a[j], b[j] = (max(0.0, float(np.dot(mj[v], dj[v]) / den)) if den > 0 else 0.0), 0.0

TEN = M * a                       # a.MOOSE tensile
SHR = B * b                       # b.DDM shear
MODEL = TEN + SHR
RES = O - MODEL
vmask = np.isfinite(O) & np.isfinite(MODEL)
vr = 100 * (1 - np.nanmean(RES[vmask] ** 2) / np.nanmean(O[vmask] ** 2))
print(f"{LABEL}: variance reduction {vr:.2f}%   a final {a[-1]:.3f}   b final {b[-1]:.3f}")
print(f"  observed peak |{np.nanmax(np.abs(O)):.4f}| me, residual peak |{np.nanmax(np.abs(RES)):.4f}| me")

# ---------------------------------------------------------------- figure
lim = float(np.nanpercentile(np.abs(O), 99))
ext = [mdates.date2num(common[0].to_pydatetime()),
       mdates.date2num(common[-1].to_pydatetime()), o_md[-1], o_md[0]]
PANELS = [(O, "(A) OBSERVED   DAS 4-h mean, T1-referenced"),
          (MODEL, f"(B) MODEL   a·MOOSE + b·DDM     VR {vr:.1f}%"),
          (RES, "(C) RESIDUAL   observed − model"),
          (TEN, "(D) a·MOOSE tensile"),
          (SHR, "(E) b·DDM shear   (zero before T2)"),
          (DDMTOT, "(F) DDM total, as uploaded")]

fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True, sharex=True,
                        sharey=True)
for ax, (Mx, ttl) in zip(axs.ravel(), PANELS):
    im = ax.imshow(Mx, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, extent=ext,
                   interpolation="nearest")
    ax.axhline(STAR, color="#2c3e50", ls=":", lw=1.3)
    ax.axhline(SHEAR_MD, color="#4d2d63", ls="--", lw=1.4)
    ax.axvline(mdates.date2num(T2.to_pydatetime()), color="#2c3e50", ls="--", lw=1.4)
    ax.set_title(ttl, fontweight="bold", fontsize=10.5)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
for ax in axs[1, :]:
    ax.set_xlabel("Time  [UTC-7]")
for ax in axs[:, 0]:
    ax.set_ylabel("Measured depth  [ft]")
axs[0, 0].text(ext[0] + 0.15, STAR - 4, f"fracture {STAR:.0f}", fontsize=8, color="#2c3e50")
axs[0, 0].text(ext[0] + 0.15, SHEAR_MD - 4, f"shear plane {SHEAR_MD:.0f}", fontsize=8,
               color="#4d2d63")
axs[0, 0].text(mdates.date2num(T2.to_pydatetime()) + 0.08, 10210, "T2", fontsize=9,
               color="#2c3e50", fontweight="bold")

cb = fig.colorbar(im, ax=axs, location="right", shrink=0.82, pad=0.012)
cb.set_label("strain  [mε]   (T1-referenced, common scale on every panel)", fontsize=10)
fig.suptitle(f"{LABEL} — observed vs simulated strain waterfalls", fontsize=14,
             fontweight="bold")
out = FIG / f"{PROJ}_waterfall_comparison.png"
fig.savefig(out, dpi=140)
print("saved", out)
