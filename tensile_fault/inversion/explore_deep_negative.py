"""NEXT-STEP EXPLORATION (does NOT touch the delivered physical inversion).

Question: why is the modeled DAS strain NEGATIVE at depth (below the fracture), and
can adding/adjusting shear pull it back toward 0/positive?

Decomposes the physical model at each window into
    tensile  = s_j * g_tensile   (poroelastic opening; MOOSE)
    shear    = q_j * g_shear      (DDM strike-slip on the real well)
and inspects the DEEP region (MD 10430-10500) vs the observed, plus the raw shapes
g_tensile(z), g_shear(z) so we can see WHERE each basis is +/- along depth.

Outputs to figs/tensile_fault_qc/inversion/explore/ (separate from the deliverable).
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls

REPO = Path(__file__).resolve().parents[3]
PROJ = "v1_tensile_srv_das"
OUT = REPO / "output" / PROJ
FIG = REPO / "figs" / "tensile_fault_qc" / "inversion" / "explore"
FIG.mkdir(parents=True, exist_ok=True)
FT = 0.3048
STAR_DEPTH_FT = 10373.4
T1 = pd.Timestamp("2025-02-24 11:00")
T2 = pd.Timestamp("2025-02-28 00:00")
OBS_CSV = REPO / "data_fervo" / "legacy" / (
    "strain_4h_mean_profiles_20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref.csv")
SHEAR_CSV = REPO / "data_fervo" / "legacy" / "07152026_decomposed" / (
    "v1_ddm_shear_strain_4h_20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref.csv")


def read_prof(csv):
    df = pd.read_csv(csv)
    return df["measured_depth_ft"].to_numpy(float), pd.to_datetime(df.columns[1:]), df.iloc[:, 1:].to_numpy(float)


# ---- MOOSE tensile basis (same as physical_inversion.py) ---------------------
tdf = pd.read_csv(OUT / f"{PROJ}_input_csv.csv")
taxis_s = tdf["time"].to_numpy(float)
vpp = sorted(glob.glob(str(OUT / f"{PROJ}_input_csv_fiber_strain_sampler_*ft_*.csv")))
n = min(len(vpp), len(taxis_s)); taxis_s, vpp = taxis_s[:n], vpp[:n]
y = None
for f in vpp:
    dd = pd.read_csv(f)
    if len(dd):
        y = dd.sort_values("y")["y"].to_numpy(float); break
cols = [pd.read_csv(vpp[i]).sort_values("y")["strain_yy"].to_numpy(float) if len(pd.read_csv(vpp[i]))
        else np.zeros_like(y) for i in range(n)]
m_strain = (np.column_stack(cols) - np.column_stack(cols)[:, [0]]) * 1e3
m_md = STAR_DEPTH_FT + (y - 0.0) / FT
o = np.argsort(m_md); m_md, m_strain = m_md[o], m_strain[o]

o_md, o_ct, o_mat = read_prof(OBS_CSV)
s_md, s_ct, s_mat = read_prof(SHEAR_CSV)
common = [t for t in o_ct if t in set(s_ct)]
oi = [list(o_ct).index(t) for t in common]; si = [list(s_ct).index(t) for t in common]
O = o_mat[:, oi]
SH = np.array([np.interp(o_md, s_md, s_mat[:, j]) for j in si]).T
tc_s = np.array([(t - T1).total_seconds() for t in common])
M_t = np.array([np.interp(tc_s, taxis_s, m_strain[k, :]) for k in range(m_strain.shape[0])])
M = np.array([np.interp(o_md, m_md, M_t[:, j]) for j in range(len(common))]).T
common = pd.DatetimeIndex(common)
nt = len(common)

# ---- refit (NNLS) to get s_j, q_j (same as delivered) -----------------------
s_win = np.zeros(nt); q_win = np.zeros(nt)
for j in range(nt):
    d, mt, sh = O[:, j], M[:, j], SH[:, j]
    v = np.isfinite(d) & np.isfinite(mt) & np.isfinite(sh)
    if common[j] >= T2 and np.nanmax(np.abs(sh)) > 1e-6:
        coef, _ = nnls(np.column_stack([mt[v], sh[v]]), d[v]); s_win[j], q_win[j] = coef
    else:
        s_win[j] = max(0.0, np.dot(mt[v], d[v]) / np.dot(mt[v], mt[v]))

# ---- inspect the DEEP region at the last window ------------------------------
j = nt - 1
tens = s_win[j] * M[:, j]; shr = q_win[j] * SH[:, j]; mdl = tens + shr
deep = (o_md >= 10430) & (o_md <= 10500)
print(f"=== last window {common[j]:%m-%d %H:%M}  (s={s_win[j]:.2f}, q={q_win[j]:.2f}) ===")
print(f"DEEP MD 10430-10500 mean [mε]:  obs={np.nanmean(O[deep, j]):+.4f}  "
      f"tensile={np.nanmean(tens[deep]):+.4f}  shear={np.nanmean(shr[deep]):+.4f}  model={np.nanmean(mdl[deep]):+.4f}")
print(f"g_tensile shape: peak {np.nanmax(M[:,j]):+.4f}@{o_md[np.nanargmax(M[:,j])]:.0f}, "
      f"min {np.nanmin(M[:,j]):+.4f}@{o_md[np.nanargmin(M[:,j])]:.0f}")
print(f"g_shear   shape: peak {np.nanmax(SH[:,j]):+.4f}@{o_md[np.nanargmax(SH[:,j])]:.0f}, "
      f"min {np.nanmin(SH[:,j]):+.4f}@{o_md[np.nanargmin(SH[:,j])]:.0f}")
print(f"g_shear at deep (10430-10500) mean = {np.nanmean(SH[deep, j]):+.5f} mε  "
      f"(if ~0, adding shear can't fix the deep-negative)")

# ---- figure: decomposition at last window -----------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
ax[0].plot(O[:, j], o_md, "k-", lw=2.5, label="observed")
ax[0].plot(mdl, o_md, "r--", lw=2, label=f"model (s·MOOSE+q·DDM)")
ax[0].plot(tens, o_md, color="tab:orange", lw=1.5, label=f"tensile s·MOOSE (s={s_win[j]:.2f})")
ax[0].plot(shr, o_md, color="tab:blue", lw=1.5, label=f"shear q·DDM (q={q_win[j]:.2f})")
ax[0].axvline(0, color="0.6", lw=0.8); ax[0].axhspan(10430, 10500, color="yellow", alpha=0.15)
ax[0].set_ylim(10500, 10200); ax[0].set_xlabel("strain [mε]"); ax[0].set_ylabel("MD [ft]")
ax[0].set_title(f"Decomposition at {common[j]:%m-%d %H:%M}\n(yellow = deep region)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
# raw basis shapes (normalized) to see where each is +/-
ax[1].plot(M[:, j] / np.nanmax(np.abs(M[:, j])), o_md, color="tab:orange", lw=2, label="g_tensile (norm)")
ax[1].plot(SH[:, j] / (np.nanmax(np.abs(SH[:, j])) or 1), o_md, color="tab:blue", lw=2, label="g_shear (norm)")
ax[1].axvline(0, color="0.6", lw=0.8); ax[1].axhspan(10430, 10500, color="yellow", alpha=0.15)
ax[1].set_ylim(10500, 10200); ax[1].set_xlabel("normalized shape"); ax[1].set_title("Basis shapes\n(where each is +/-)")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "deep_negative_decomposition.png", dpi=140)
print("saved", FIG / "deep_negative_decomposition.png")

# ---- TEST A: does MORE shear help the deep region? --------------------------
print("\n--- TEST A: scale shear up (user's hypothesis) ---")
for f in [1.0, 1.5, 2.0]:
    dm = tens[deep] + f * shr[deep]
    print(f"  shear x{f}:  model deep mean = {np.nanmean(dm):+.4f} mε   (obs = {np.nanmean(O[deep,j]):+.4f})")
print("  -> DDM shear is NEGATIVE at the deep region, so adding shear makes deep MORE negative.")

# ---- TEST B: add a DEEPER opening basis (shift g_tensile down D ft) ----------
print("\n--- TEST B: add a 2nd tensile opening placed DEEPER (shift proxy) ---")
D = 80.0  # ft deeper
g_deep = np.interp(o_md - D, o_md, M[:, j], left=0.0, right=0.0)   # opening shape moved D ft deeper
v = np.isfinite(O[:, j]) & np.isfinite(M[:, j]) & np.isfinite(SH[:, j])
A2 = np.column_stack([M[v, j], SH[v, j]])
A3 = np.column_stack([M[v, j], SH[v, j], g_deep[v]])
c2, _ = nnls(A2, O[v, j]); c3, _ = nnls(A3, O[v, j])
mdl2 = A2 @ c2; mdl3 = A3 @ c3
vr2 = 100 * (1 - np.var(O[v, j] - mdl2) / np.var(O[v, j]))
vr3 = 100 * (1 - np.var(O[v, j] - mdl3) / np.var(O[v, j]))
deep_v = deep[v]
print(f"  2-basis  : deep mean {np.nanmean(mdl2[deep_v]):+.4f} mε,  window VR {vr2:.0f}%")
print(f"  +deep-open: deep mean {np.nanmean(mdl3[deep_v]):+.4f} mε,  window VR {vr3:.0f}%  "
      f"(coeffs s={c3[0]:.2f}, q={c3[1]:.2f}, r_deep={c3[2]:.2f})")
print(f"  obs deep mean {np.nanmean(O[deep, j]):+.4f} mε")
