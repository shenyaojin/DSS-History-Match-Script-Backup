"""Compare our simulation vs the REAL DAS strain (extracted from LF-DAS NPY), T2->T3.

The raw LF-DAS only covers T2->T3 (Feb 28 onward; a data gap precedes it), so the
model-vs-observation comparison uses the real DAS there. Everything is referenced to
the first real-DAS time (2025-02-28 17:00). Model forward (MOOSE) is unchanged; only
its per-4h coefficients (s_p tensile pressure, s_sh shear) are fit to the real DAS.
"""
import glob, os
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[4]
PROJ = "v1_srv_w75_p01"; W = 75.0
OUT = REPO / "output" / PROJ
FIG = REPO / "figs" / "tensile_fault_qc" / "das_observed"; FIG.mkdir(parents=True, exist_ok=True)
FT = 0.3048; STAR = 10373.4; SHEAR_MD = STAR - W / 2
SHEAR = REPO / "data_fervo/legacy/07152026_decomposed/v1_ddm_shear_strain_4h_20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref.csv"
PROFILE_HOURS = 2.6

# --- real DAS strain waterfall (T2->T3) ---
d = np.load(OUT.parent / "das_observed" / "das_strain_waterfall_T1T3.npz", allow_pickle=True)
das_md = d["md_ft"]; das_t = pd.Timestamp(str(d["start_time"])) + pd.to_timedelta(d["taxis_s"], unit="s"); das = d["data"].astype(float)
REF = das_t[0]                                   # 2025-02-28 17:00 (first real DAS)
T3 = pd.Timestamp("2025-03-03 22:00")
grid = pd.date_range(REF, T3, freq="4h")
o_md = das_md


def dasavg(md_t_mat, tms, target):               # 4h-window mean of the DAS onto grid
    out = []
    for ws in grid:
        m = (tms >= ws) & (tms < ws + pd.Timedelta("4h"))
        out.append(np.nanmean(md_t_mat[:, m], axis=1) if m.any() else np.full(md_t_mat.shape[0], np.nan))
    return np.column_stack(out)


O = dasavg(das, das_t, grid); O -= np.nan_to_num(O[:, [0]])    # real DAS 4h profiles, ref to REF

# --- MOOSE tensile + DDM shear on the same grid, referenced to REF ---
tdf = pd.read_csv(OUT / f"{PROJ}_input_csv.csv"); taxis = tdf["time"].to_numpy(float)
vpp = sorted(glob.glob(str(OUT / f"{PROJ}_input_csv_fiber_strain_sampler_*ft_*.csv")))
n = min(len(vpp), len(taxis)); taxis = taxis[:n]; vpp = vpp[:n]
yv = next(pd.read_csv(f).sort_values("y")["y"].to_numpy(float) for f in vpp if len(pd.read_csv(f)))
cols = [pd.read_csv(vpp[i]).sort_values("y")["strain_yy"].to_numpy(float) if len(pd.read_csv(vpp[i])) else np.zeros_like(yv) for i in range(n)]
m_all = np.column_stack(cols) * 1e3; m_md = STAR + yv / FT; oo = np.argsort(m_md); m_md, m_all = m_md[oo], m_all[oo]
m_t = pd.Timestamp("2025-02-24 15:00") + pd.to_timedelta(taxis, unit="s")
sdf = pd.read_csv(SHEAR); s_md = sdf["measured_depth_ft"].to_numpy(float); s_ct = pd.to_datetime(sdf.columns[1:]); s_mat = sdf.iloc[:, 1:].to_numpy(float)


def to_grid(mat, src, md_src, shift=0.0):
    gt = (grid - pd.Timestamp("2024-01-01")).total_seconds().to_numpy(); st = (src - pd.Timestamp("2024-01-01")).total_seconds().to_numpy()
    tmp = np.array([np.interp(gt, st, mat[k]) for k in range(mat.shape[0])])       # interp in time
    return np.array([np.interp(o_md - shift, md_src, tmp[:, j]) for j in range(len(grid))]).T  # interp in MD (+shift)


M = to_grid(m_all, m_t, m_md); M -= M[:, [0]]
SH = to_grid(s_mat, s_ct, s_md, shift=SHEAR_MD - STAR); SH -= SH[:, [0]]

# --- per-4h 2-param inversion vs REAL DAS ---
s_p = np.zeros(len(grid)); s_sh = np.zeros(len(grid))
for j in range(len(grid)):
    o = O[:, j]; f = np.isfinite(o) & np.isfinite(M[:, j])
    if f.sum() < 3 or np.dot(M[f, j], M[f, j]) == 0:
        continue
    A = np.column_stack([M[f, j], SH[f, j]]); coef, *_ = np.linalg.lstsq(A, o[f], rcond=None); s_p[j], s_sh[j] = coef
MODEL = M * s_p[None, :] + SH * s_sh[None, :]
vv = np.isfinite(O) & np.isfinite(MODEL)
rms0 = np.sqrt(np.nanmean(O[vv]**2)); rmsr = np.sqrt(np.nanmean((O - MODEL)[vv]**2))
print(f"model vs REAL DAS (T2->T3): RMS {rms0:.4f}->{rmsr:.4f} mε ({100*(1-rmsr/rms0):.0f}% var red)")
print(f"  s_p {np.nanmin(s_p[1:]):.2f}..{np.nanmax(s_p[1:]):.2f}, s_sh {np.nanmin(s_sh[1:]):.2f}..{np.nanmax(s_sh[1:]):.2f}")

# --- figures ---
p95 = np.nanpercentile(np.abs(O), 95) or 1.0


def wig(ax, prof, color):
    sec = PROFILE_HOURS * 3600 / p95
    for j, t in enumerate(grid):
        fmask = np.isfinite(prof[:, j])
        ax.plot(t + pd.to_timedelta(prof[fmask, j] * sec, unit="s"), o_md[fmask], color=color, lw=0.8, alpha=0.9, zorder=5)


def wf(ax, prof, title, lim=0.08):
    im = ax.imshow(prof, aspect="auto", cmap="seismic", vmin=-lim, vmax=lim,
                   extent=[mdates.date2num(grid[0]), mdates.date2num(grid[-1]), o_md[-1], o_md[0]], interpolation="bilinear")
    ax.xaxis_date(); ax.set_ylim(10500, 10200); ax.set_ylabel("MD [ft]"); ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M")); plt.colorbar(im, ax=ax, label="mε")


figA, (a1, a2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True, constrained_layout=True)
wf(a1, O, "REAL DAS strain (from LF-DAS NPY) + 4h profiles"); wig(a1, O, "black")
wf(a2, MODEL, f"Model (MOOSE tensile + DDM shear) fit to real DAS + 4h profiles"); wig(a2, MODEL, "black")
a2.set_xlabel("Time [UTC-7]")
figA.suptitle(f"V1 T2→T3: REAL DAS vs model — {100*(1-rmsr/rms0):.0f}% var. reduction  (T1→T2 has no DAS data)", fontweight="bold")
figA.savefig(FIG / "realdas_vs_model_2panel.png", dpi=150); print("Saved:", FIG / "realdas_vs_model_2panel.png")

figB, axB = plt.subplots(figsize=(15, 6), constrained_layout=True)
wf(axB, MODEL, "Model strain (background) + overlaid REAL DAS (black) and model (red) 4h profiles")
wig(axB, O, "black"); wig(axB, MODEL, "red")
axB.plot([], [], "k-", lw=2, label="REAL DAS profiles"); axB.plot([], [], "r-", lw=2, label="model profiles"); axB.legend(loc="lower right")
axB.set_xlabel("Time [UTC-7]")
figB.savefig(FIG / "realdas_vs_model_overlay.png", dpi=150); print("Saved:", FIG / "realdas_vs_model_overlay.png")
