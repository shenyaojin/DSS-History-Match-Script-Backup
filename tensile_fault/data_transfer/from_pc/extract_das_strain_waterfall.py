"""Extract the REAL observed DAS strain waterfall (full resolution) from the raw LF-DAS NPY.

The fiberis LFDAS npz files only cover T2->T3, so we read the raw NPY (which cover the
full window) and replicate the LFDAS-notebook processing without DASCore:
  rate = npy * STRAIN_RATE_SCALE  (nanostrain/s)  -> per-minute mean
  common-mode demean using the MD 1800-5000 ft reference band
  integrate rate over time -> strain (millistrain), T1(15:00)-referenced
MD axis is taken from the fiberis LFDAS npz `daxis` (already measured depth, ft).
Output: output/das_observed/das_strain_waterfall_T1T3.npz  (+ QC figure).
"""
import glob, os
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[4]
NPY_DIRS = ["data_fervo/legacy/LF-DAS NPY/NPY_G4-PB_MM_UTC_202502",
            "data_fervo/legacy/LF-DAS NPY/NPY_G4-PB_MM_UTC_202503"]
MD_NPZ = REPO / "data_fervo/fiberis_format/LFDAS/LFDAS_G4-PB_202503.npz"   # for the MD (daxis)
OUT = REPO / "output" / "das_observed"; OUT.mkdir(parents=True, exist_ok=True)
FIG = REPO / "figs" / "tensile_fault_qc" / "das_observed"; FIG.mkdir(parents=True, exist_ok=True)

fs, GL = 1000, 10
STRAIN_RATE_SCALE = 116 * fs / GL / 8192          # -> nanostrain/s
LOCAL_OFFSET = pd.Timedelta(hours=7)              # UTC -> UTC-7 (local)
T1 = pd.Timestamp("2025-02-24 15:00"); T3 = pd.Timestamp("2025-03-03 22:00")
MD_LO, MD_HI = 10200, 10500
REF_LO, REF_HI = 1800, 5000                        # common-mode reference band

# --- list NPY files in [T1-2h, T3] (local), sorted ---
files = []
for dd in NPY_DIRS:
    files += glob.glob(str(REPO / dd / "*.npy"))
df = pd.DataFrame({"path": files})
df["utc"] = df.path.apply(lambda x: pd.to_datetime(Path(x).stem.split("UTC_")[-1], format="%Y%m%d_%H%M%S"))
df["local"] = df["utc"] - LOCAL_OFFSET
df = df[(df["local"] >= T1 - pd.Timedelta("2h")) & (df["local"] <= T3)].sort_values("local").reset_index(drop=True)
print(f"loading {len(df)} NPY files ({df['local'].min()} -> {df['local'].max()})")

# --- load: per-file mean rate (nanostrain/s) -> (n_ch, n_time) ---
rate = np.empty((2304, len(df)), dtype=np.float32)
for i, p in enumerate(df["path"]):
    rate[:, i] = np.load(p).mean(axis=0) * STRAIN_RATE_SCALE
    if i % 1000 == 0:
        print(f"  {i}/{len(df)}")
times = pd.DatetimeIndex(df["local"].values)
md = np.asarray(np.load(MD_NPZ, allow_pickle=True, mmap_mode="r")["daxis"], float)   # MD (ft) for the 2304 channels
print(f"rate {rate.shape}, MD {md.min():.0f}..{md.max():.0f} ft")

# --- common-mode demean using the shallow reference band ---
refmask = (md >= REF_LO) & (md <= REF_HI)
rate = rate - np.nanmedian(rate[refmask, :], axis=0, keepdims=True)

# --- integrate rate -> strain (millistrain); cap dt so gaps don't over-integrate ---
tsec = (times - times[0]).total_seconds().to_numpy()
dt = np.diff(tsec, prepend=tsec[0]); dt[dt > 120] = 0.0          # skip gaps > 2 min
strain_ms = np.cumsum(rate * dt[None, :], axis=1) / 1e6          # nanostrain -> millistrain

# --- crop MD, T1-reference, crop time ---
dmask = (md >= MD_LO) & (md <= MD_HI)
md_w = md[dmask]; strain_w = strain_ms[dmask, :]
j_t1 = int(np.argmin(np.abs((times - T1).total_seconds())))
strain_w = strain_w - strain_w[:, [j_t1]]                        # T1 referenced
tmask = (times >= T1) & (times <= T3)
times_w = times[tmask]; strain_w = strain_w[:, tmask]
print(f"observed strain waterfall: {strain_w.shape} (MD {md_w.min():.0f}..{md_w.max():.0f}, "
      f"{times_w[0]} -> {times_w[-1]}), peak |eps|={np.nanmax(np.abs(strain_w)):.4f} mε")

np.savez(OUT / "das_strain_waterfall_T1T3.npz", data=strain_w.astype(np.float32),
         md_ft=md_w, taxis_s=(times_w - times_w[0]).total_seconds().to_numpy(),
         start_time=str(times_w[0]))

# --- QC waterfall figure ---
fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
im = ax.imshow(strain_w, aspect="auto", cmap="seismic", vmin=-0.1, vmax=0.1,
               extent=[mdates.date2num(times_w[0]), mdates.date2num(times_w[-1]), md_w[-1], md_w[0]],
               interpolation="none")
ax.xaxis_date(); ax.set_ylim(MD_HI, MD_LO); ax.set_ylabel("Gold 4-PB Measured Depth [ft]"); ax.set_xlabel("Time [UTC-7]")
ax.axvline(pd.Timestamp("2025-02-28"), color="lime", ls="--")
ax.set_title("Observed DAS strain waterfall (from raw LF-DAS NPY, T1=15:00 referenced)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d")); plt.colorbar(im, ax=ax, label="Strain (millistrain)")
fig.savefig(FIG / "das_strain_waterfall_observed.png", dpi=140)
print("Saved:", OUT / "das_strain_waterfall_T1T3.npz", "+", FIG / "das_strain_waterfall_observed.png")
