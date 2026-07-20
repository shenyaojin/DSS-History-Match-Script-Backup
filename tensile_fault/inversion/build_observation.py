"""Phase 0 of the gradient-inversion workflow: build the observed DAS STRAIN target.

Reuses the cleaning pipeline of 107_lfdas_reproduce_strain_rate_waterfall.py (merge the two
gap-filling Data2D segments, median filter, quiet-band demean, artifact interpolation), then
INTEGRATES strain-rate -> strain (T1-referenced) and reduces to 4-hour-mean depth profiles on the
MD 10200-10500 window. Emits:

  output/inversion/observation.npz
      strain_4h [n_md, n_win]  (millistrain, T1-ref, 4-hour-window means)
      md_ft [n_md], window_starts (ISO strings), T1/T2/T3
      strain_wf [n_md, n_time], times (for QC)
  output/inversion/measurement_data.csv
      OptimizationData format (measurement_xcoord/ycoord/zcoord/time/value) mapping each MD channel
      to the MOOSE fiber node (x = fiber offset, y = (MD-STAR)*FT) for the adjoint phase.

T1 = 2025-02-24 15:00 to match the unified forward model (poro_dd_real_ls).
"""
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage import median_filter

import sys
REPO = Path("/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner")
sys.path.insert(0, str(REPO / "fibeRIS" / "src"))
from fiberis.analyzer.Data2D.core2D import Data2D  # noqa: E402

DATA_DIR = REPO / "data_fervo" / "fiberis_format" / "LFDAS"
SEGMENTS = ["LFDAS_G4-PB_202502_late.npz", "LFDAS_G4-PB_202503.npz"]
OUT = REPO / "output" / "inversion"; OUT.mkdir(parents=True, exist_ok=True)
FIG = REPO / "figs" / "tensile_fault_qc" / "inversion"; FIG.mkdir(parents=True, exist_ok=True)

T1 = pd.Timestamp("2025-02-24 15:00")     # match the forward model
T2 = pd.Timestamp("2025-02-28 00:00")
T3 = pd.Timestamp("2025-03-03 22:00")
MD_LO, MD_HI = 10200.0, 10500.0
DEMEAN_LO, DEMEAN_HI = 1800.0, 5000.0
MEDFILT = (3, 7)
ARTIFACT = [(pd.Timestamp("2025-02-25 19:53:50"), pd.Timestamp("2025-02-25 19:54:30"))]
PROFILE_H = 4
FT = 0.3048
STAR = 10373.4
FIBER_X = 100.0 + 40 * FT                  # fervo fiber lateral offset (model metres)


def load_merged():
    m = Data2D(); m.load_npz(str(DATA_DIR / SEGMENTS[0]))
    for s in SEGMENTS[1:]:
        nxt = Data2D(); nxt.load_npz(str(DATA_DIR / s)); m.right_merge(nxt)
    return m


def clean_rate(das):
    """Median-filter + quiet-band demean + artifact interp; return (rate[md,t], daxis, times)."""
    times = pd.to_datetime(das.start_time) + pd.to_timedelta(das.taxis, unit="s")
    buf = pd.Timedelta(minutes=10)
    tb = (times >= T1 - buf) & (times <= T3 + buf)
    LP = das.data[:, tb].astype(float); times = times[tb]; daxis = das.daxis

    gap = np.isnan(LP).all(axis=0)
    x = np.arange(LP.shape[1])
    for r in range(LP.shape[0]):
        y = LP[r]; bad = np.isnan(y) & ~gap; good = np.isfinite(y) & ~gap
        if bad.any() and good.sum() >= 2:
            y[bad] = np.interp(x[bad], x[good], y[good])
    LP[:, gap] = 0.0
    med = median_filter(LP, size=MEDFILT, mode="nearest")
    band = (daxis >= DEMEAN_LO) & (daxis <= DEMEAN_HI)
    rate = med - np.nanmedian(med[band, :], axis=0)[None, :]
    rate[:, gap] = np.nan
    art = np.zeros(len(times), bool)
    for a, b in ARTIFACT:
        art |= (times >= a) & (times <= b)
    am = art & ~gap
    if am.any():
        for r in range(rate.shape[0]):
            y = rate[r]; v = np.isfinite(y) & ~am
            if v.sum() >= 2:
                rate[r, am] = np.interp(x[am], x[v], y[v])
    return rate, daxis, times


def main():
    das = load_merged()
    print(f"merged {das.data.shape}, {das.start_time} .. {das.get_end_time()}")
    rate, daxis, times = clean_rate(das)

    # crop MD, integrate rate -> strain (millistrain), reference to T1
    dm = (daxis >= MD_LO) & (daxis <= MD_HI)
    md = daxis[dm]; rate = rate[dm]
    tsec = (times - times[0]).total_seconds().to_numpy()
    dt = np.diff(tsec, prepend=tsec[0]); dt[dt > 120] = 0.0        # skip gaps
    strain = np.cumsum(np.nan_to_num(rate) * dt[None, :], axis=1) / 1e6   # nanostrain -> millistrain
    j1 = int(np.argmin(np.abs((times - T1).total_seconds())))
    strain = strain - strain[:, [j1]]                              # T1-referenced

    # crop to T1..T3
    tm = (times >= T1) & (times <= T3)
    strain = strain[:, tm]; twf = times[tm]
    print(f"strain waterfall {strain.shape}, MD {md.min():.0f}..{md.max():.0f}, peak |eps|={np.nanmax(np.abs(strain)):.4f} me")

    # 4-hour-mean profiles, anchored at window start
    starts = pd.date_range(T1, T3, freq=f"{PROFILE_H}h")
    profs, centers = [], []
    for ws in starts:
        we = min(ws + pd.Timedelta(hours=PROFILE_H), T3)
        if ws >= T3:
            continue
        mmask = (twf >= ws) & (twf < we)
        if mmask.any():
            profs.append(np.nanmean(strain[:, mmask], axis=1)); centers.append(ws)
    strain_4h = np.column_stack(profs)                             # [md, win]
    centers = pd.DatetimeIndex(centers)
    print(f"{strain_4h.shape[1]} 4-hour profiles; peak profile |eps|={np.nanmax(np.abs(strain_4h)):.4f} me")

    np.savez(OUT / "observation.npz",
             strain_4h=strain_4h.astype(np.float32), md_ft=md,
             window_starts=np.array([str(c) for c in centers]),
             strain_wf=strain.astype(np.float32),
             times=np.array([str(t) for t in twf]),
             T1=str(T1), T2=str(T2), T3=str(T3))

    # OptimizationData measurement CSV (for the adjoint phase): one row per (MD, window center)
    rows = []
    for j, c in enumerate(centers):
        t_rel = (c - T1).total_seconds()
        for i, mdi in enumerate(md):
            y_model = (mdi - STAR) * FT
            rows.append((FIBER_X, y_model, 0.0, t_rel, float(strain_4h[i, j])))
    pd.DataFrame(rows, columns=["measurement_xcoord", "measurement_ycoord",
                                "measurement_zcoord", "measurement_time",
                                "measurement_values"]).to_csv(OUT / "measurement_data.csv", index=False)
    print("wrote", OUT / "observation.npz", "and", OUT / "measurement_data.csv")

    # QC figure: strain waterfall + 4h profile overlay
    fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    lim = np.nanpercentile(np.abs(strain), 98) or 0.05
    im = ax.imshow(strain, aspect="auto", cmap="seismic", vmin=-lim, vmax=lim,
                   extent=[mdates.date2num(twf[0]), mdates.date2num(twf[-1]), md[-1], md[0]],
                   interpolation="bilinear")
    ax.xaxis_date(); ax.set_ylim(MD_HI, MD_LO); ax.set_ylabel("MD [ft]"); ax.set_xlabel("Time [UTC-7]")
    ax.set_title(f"Observed DAS STRAIN (T1={T1:%m/%d %H:%M}-ref) — inversion target, peak {np.nanmax(np.abs(strain)):.3f} me")
    for tt, lb in [(T1, "T1"), (T2, "T2"), (T3, "T3")]:
        ax.axvline(mdates.date2num(tt), color="green", ls="--", lw=1.4)
        ax.text(mdates.date2num(tt), 1.01, lb, transform=ax.get_xaxis_transform(), color="green", ha="center", fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d")); plt.colorbar(im, ax=ax, label="me")
    fig.savefig(FIG / "observation_strain_waterfall.png", dpi=140)
    print("saved", FIG / "observation_strain_waterfall.png")


if __name__ == "__main__":
    main()
