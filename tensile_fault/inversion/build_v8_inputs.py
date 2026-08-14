"""Prepare the observation and the DDM shear basis for the updated two-PARALLEL-fault model.

New upload (suffix 20250224_1200_to_20250304_0000_10200_10500ft_4h_mean_T1_ref):
  observed strain / strain-rate 4h profiles   -> data_fervo/legacy/
  two-fault TOTAL strain / rate / waterfalls  -> data_fervo/legacy/07152026/
  histories (strike, centre, width, shear)    -> data_fervo/legacy/07152026/

The two-parallel-fault model is cleanly separable, which the previous (non-parallel) one was
not: the histories show fault1 carries width only (shear identically 0) and fault2 carries
shear only (width identically 0). So

    total(z,t) = width(t) * g(z)  +  shear_contribution(z,t)

with g(z) time-invariant for a single rectangular element. Before T2 the shear is exactly
zero, so g(z) is fitted there and the shear falls out by subtraction -- no approximation.

NOTE: the notebook's cell 28 ("Parallel-fault subcell: fault-2 pure-shear strain and
strain-rate profiles") exports `fault2_shear_strain_4h_<suffix>.csv` directly, but that file
was NOT part of the upload -- only the two-fault TOTAL products were. This script therefore
DERIVES it. When the real export arrives, drop it in the same place under the same name and
this script's output is superseded.

Writes:
  data_fervo/legacy/07152026_decomposed/fault2_shear_strain_4h_<suffix>.csv   [DERIVED]
  data_fervo/legacy/07152026_decomposed/fault1_tensile_strain_4h_<suffix>.csv [QC]
  output/inversion/observation_T1_1200.npz
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SUFFIX = "20250224_1200_to_20250304_0000_10200_10500ft_4h_mean_T1_ref"
LEGACY = REPO / "data_fervo" / "legacy"
DDM = LEGACY / "07152026"
OUTDDM = LEGACY / "07152026_decomposed"
OUTDDM.mkdir(parents=True, exist_ok=True)
OUT = REPO / "output" / "inversion"
OUT.mkdir(parents=True, exist_ok=True)
FIG = REPO / "figs" / "tensile_fault_qc" / "v8_parallel_faults"
FIG.mkdir(parents=True, exist_ok=True)

T1 = pd.Timestamp("2025-02-24 12:00")
T2 = pd.Timestamp("2025-02-28 00:00")
T3 = pd.Timestamp("2025-03-04 00:00")


def read_profiles(path):
    df = pd.read_csv(path)
    md = df["measured_depth_ft"].to_numpy(float)
    win = pd.DatetimeIndex(pd.to_datetime(df.columns[1:]))
    return md, win, df.iloc[:, 1:].to_numpy(float)


o_md, o_win, O = read_profiles(LEGACY / f"strain_4h_mean_profiles_{SUFFIX}.csv")
t_md, t_win, TOT = read_profiles(DDM / f"two_fault_direct_strain_4h_{SUFFIX}.csv")
assert list(o_win) == list(t_win), "observed and DDM window grids differ"
print(f"observed  {O.shape}  MD {o_md.min():.1f}-{o_md.max():.1f}")
print(f"DDM total {TOT.shape}  MD {t_md.min():.1f}-{t_md.max():.1f}")
print(f"windows {len(o_win)}: {o_win[0]} .. {o_win[-1]}   T1={T1}  T2={T2}  T3={T3}")

# ---- fault1 opening history, sampled on the profile windows ------------------
h = pd.read_csv(DDM / f"two_fault_histories_{SUFFIX}.csv")
h_t = pd.to_datetime(h["time"])
h_s = (h_t - T1).dt.total_seconds().to_numpy()
w_hist = h["fault1_width_ft"].to_numpy(float)
assert np.allclose(h["fault1_shear_ft"], 0.0), "fault1 is not pure tensile"
assert np.allclose(h["fault2_width_ft"], 0.0), "fault2 is not pure shear"
win_s = (o_win - T1).total_seconds().to_numpy()
width = np.interp(win_s, h_s, w_hist)
print(f"fault1 width at windows: {width.min():.5f} .. {width.max():.5f} ft")

# ---- fit the time-invariant tensile shape g(z) on the pre-T2 windows ---------
pre = (o_win >= T1) & (o_win < T2)
den = float(np.sum(width[pre] ** 2))
g = (TOT[:, pre] @ width[pre]) / den                     # least-squares g(z)
TEN = np.outer(g, width)                                  # tensile everywhere
resid_pre = TOT[:, pre] - TEN[:, pre]
print(f"tensile fit on {int(pre.sum())} pre-T2 windows: residual "
      f"{np.abs(resid_pre).max():.6f} me vs signal {np.abs(TOT[:, pre]).max():.6f} me "
      f"({100*np.abs(resid_pre).max()/max(np.abs(TOT[:, pre]).max(),1e-12):.2f}%)")

SHEAR_DERIVED = TOT - TEN
SHEAR_DERIVED[:, pre] = 0.0                               # exactly zero before T2

# ---- prefer the notebook's own cell-28 export over the derived one -----------
UPLOADED = OUTDDM / ("fault2_shear_strain_4h_20250228_0000_to_20250304_0000_"
                     "10200_10500ft_4h_mean_T2_ref.csv")
if UPLOADED.exists():
    u = pd.read_csv(UPLOADED)
    u_md = u["measured_depth_ft"].to_numpy(float)
    u_win = pd.DatetimeIndex(pd.to_datetime(u.columns[1:]))
    U = u.iloc[:, 1:].to_numpy(float)
    assert np.allclose(u_md, t_md), "uploaded shear is on a different MD grid"
    # the export is masked to T2..T3 and referenced to T2; shear is identically zero
    # before T2, so T2- and T1-referencing coincide and the gap is a clean zero pad.
    SHEAR = np.zeros_like(TOT)
    idx = {t: j for j, t in enumerate(o_win)}
    missing = [t for t in u_win if t not in idx]
    assert not missing, f"uploaded windows not in the observation grid: {missing[:3]}"
    for j, t in enumerate(u_win):
        SHEAR[:, idx[t]] = U[:, j]
    npad = len(o_win) - len(u_win)
    d = SHEAR - SHEAR_DERIVED
    scale = max(np.abs(SHEAR).max(), 1e-12)
    print(f"USING UPLOADED shear export ({UPLOADED.name})")
    print(f"  {U.shape} over {u_win[0]} .. {u_win[-1]}; zero-padded {npad} pre-T2 windows")
    print(f"  cross-check vs my derived separation: max |diff| {np.abs(d).max():.5f} me "
          f"({100*np.abs(d).max()/scale:.2f}% of peak), rms {np.sqrt(np.mean(d**2)):.6f} me")
    SHEAR_SOURCE = "notebook cell-28 export (T2-referenced, zero-padded before T2)"
else:
    SHEAR = SHEAR_DERIVED
    SHEAR_SOURCE = "DERIVED by separation - cell-28 export not found"
    print("!! " + SHEAR_SOURCE)
print(f"shear basis: peak |{np.abs(SHEAR).max():.4f}| me, "
      f"max |pre-T2| {np.abs(SHEAR[:, pre]).max():.2e} me")


def write_profiles(path, md, win, M, note):
    cols = {"measured_depth_ft": md}
    for j, t in enumerate(win):
        cols[str(t)] = M[:, j]
    pd.DataFrame(cols).to_csv(path, index=False)
    print(f"  wrote {path.name}  [{note}]")


write_profiles(OUTDDM / f"fault2_shear_strain_4h_{SUFFIX}.csv", t_md, o_win, SHEAR,
               SHEAR_SOURCE)
write_profiles(OUTDDM / f"fault1_tensile_strain_4h_{SUFFIX}.csv", t_md, o_win, TEN, "QC")

# ---- observation npz on the new grid ----------------------------------------
np.savez(OUT / "observation_T1_1200.npz", strain_4h=O, md_ft=o_md,
         window_starts=np.array([str(t) for t in o_win]),
         T1=str(T1), T2=str(T2), T3=str(T3))
print(f"  wrote {OUT / 'observation_T1_1200.npz'}")

# ---- QC figure ---------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True, sharey=True)
jl = len(o_win) - 1
for ax, (M, ttl, c) in zip(axs, [(TOT, "DDM total (uploaded)", "#2c3e50"),
                                 (TEN, "fault1 tensile  width(t)·g(z)  [QC only]", "#c0392b"),
                                 (SHEAR, "fault2 shear — UPLOADED export (used)", "#1a6ea8")]):
    for j in range(0, len(o_win), 6):
        ax.plot(M[:, j], t_md, lw=1.1, alpha=.45, color=c)
    ax.plot(M[:, jl], t_md, lw=3, color=c, label=f"{o_win[jl]:%m-%d %H:%M}")
    ax.axvline(0, color="#7f8c8d", lw=1)
    ax.set_title(ttl, fontweight="bold", fontsize=11)
    ax.set_xlabel("strain  [mε]")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, frameon=False)
axs[0].set_ylabel("Measured depth  [ft]")
axs[0].set_ylim(10500, 10200)
fig.suptitle("Two-parallel-fault decomposition — fault1 pure tensile, fault2 pure shear",
             fontsize=13, fontweight="bold")
out = FIG / "ddm_decomposition_qc.png"
fig.savefig(out, dpi=140)
print("saved", out)
