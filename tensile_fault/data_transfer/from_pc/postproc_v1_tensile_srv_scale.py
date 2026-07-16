"""Post-process the V1 tensile-SRV MOOSE run + pressure-scale sweep (T1->T2, pure tensile).

Reads the fiber strain_yy VectorPostprocessor along Y, maps Y->MD, 4h-averages, then
because poroelastic strain is LINEAR in the pressure perturbation, finds the pressure
scale s that best matches the observed 4h strain profiles:
    min_s || eps_obs(MD,t) - s * eps_MOOSE_tensile(MD,t) ||   (shear = 0 in T1->T2)
Outputs: best global s and per-snapshot s, the implied within-fracture pressure history,
and comparison figures.
"""
import glob
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
PROJ = "v1_tensile_srv_injection"
OUT = REPO / "output" / PROJ
FIG = REPO / "figs" / "tensile_fault_qc" / PROJ
FIG.mkdir(parents=True, exist_ok=True)

FT = 0.3048
STAR_DEPTH_FT = 10373.4
Y_FRAC = 0.0
T1 = pd.Timestamp("2025-02-24 11:00")
T2 = pd.Timestamp("2025-02-28 00:00")
OBS_CSV = REPO / "data_fervo" / "legacy" / (
    "strain_4h_mean_profiles_20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref.csv")
PRESSURE_NPZ = OUT / "injection_pressure_T1_T2_2h.npz"

# ---- read MOOSE fiber strain_yy VPP -> strain_yy[y, t] -----------------------
main_csv = OUT / f"{PROJ}_input_csv.csv"
tdf = pd.read_csv(main_csv)
taxis_s = tdf["time"].to_numpy(float)                       # seconds from T1
vpp = sorted(glob.glob(str(OUT / f"{PROJ}_input_csv_fiber_strain_sampler_83.333ft_*.csv")))
assert len(vpp) >= len(taxis_s), f"{len(vpp)} vpp files vs {len(taxis_s)} timesteps"

# find the y-grid from the first non-empty VPP file
y = None
for f in vpp:
    df = pd.read_csv(f)
    if len(df):
        y = df.sort_values("y")["y"].to_numpy(float)
        break
cols = []
for i in range(len(taxis_s)):
    df = pd.read_csv(vpp[i])
    if len(df):                                            # t=0 (INITIAL) file is empty
        cols.append(df.sort_values("y")["strain_yy"].to_numpy(float))
    else:
        cols.append(np.zeros_like(y))
strain = np.column_stack(cols)                              # (n_y, n_t), dimensionless
strain = strain - strain[:, [0]]                            # T1 reference (t=0)
strain_ms = strain * 1e3                                    # -> millistrain

md = STAR_DEPTH_FT + (y - Y_FRAC) / FT                      # Y(m) -> MD(ft)
order = np.argsort(md)
md, strain_ms = md[order], strain_ms[order]
mtimes = T1 + pd.to_timedelta(taxis_s, unit="s")
print(f"MOOSE: {strain_ms.shape[0]} depths (MD {md.min():.0f}-{md.max():.0f}), "
      f"{strain_ms.shape[1]} times ({mtimes[0]} -> {mtimes[-1]}), peak |eps|={np.nanmax(np.abs(strain_ms)):.4f} mε")


def four_hour(mat, times):
    times = pd.DatetimeIndex(times)
    cols, cts = [], []
    for ws in pd.date_range(T1, T2, freq="4h"):
        if ws >= T2:
            continue
        m = (times >= ws) & (times < min(ws + pd.Timedelta("4h"), T2))
        if not m.any():
            continue
        cols.append(np.nanmean(mat[:, m], axis=1)); cts.append(ws)
    return np.column_stack(cols), pd.DatetimeIndex(cts)


moose_prof, moose_ct = four_hour(strain_ms, mtimes)         # (n_md, n_win)

# ---- observed 4h strain profiles (T1->T2 only; shear=0 there) ----------------
odf = pd.read_csv(OBS_CSV)
o_md = odf["measured_depth_ft"].to_numpy(float)
o_ct_all = pd.to_datetime(odf.columns[1:])
o_mat_all = odf.iloc[:, 1:].to_numpy(float)
keep = o_ct_all < T2
o_ct, o_mat = o_ct_all[keep], o_mat_all[:, keep]

# align windows by start time; interpolate MOOSE onto observed MD grid
common = [t for t in moose_ct if t in set(o_ct)]
mi = [list(moose_ct).index(t) for t in common]
oi = [list(o_ct).index(t) for t in common]
M = np.array([np.interp(o_md, md, moose_prof[:, j]) for j in mi]).T   # (n_o_md, n_common)
O = o_mat[:, oi]
valid = np.isfinite(M) & np.isfinite(O)
print(f"Comparing {len(common)} T1->T2 windows on {len(o_md)} obs depths.")

# ---- linear pressure-scale inversion ----------------------------------------
Mv, Ov = M[valid], O[valid]
s_global = float(np.dot(Mv, Ov) / np.dot(Mv, Mv))
resid = O - s_global * M
rms0 = np.sqrt(np.nanmean(O[valid] ** 2))
rms_r = np.sqrt(np.nanmean(resid[valid] ** 2))
# per-snapshot scale
s_snap = np.array([float(np.dot(M[:, j][valid[:, j]], O[:, j][valid[:, j]]) /
                          np.dot(M[:, j][valid[:, j]], M[:, j][valid[:, j]]))
                   for j in range(len(common))])
# misfit vs scale curve
svals = np.linspace(0, 2.5 * s_global, 200)
misfit = [np.sqrt(np.nanmean((O[valid] - sv * M[valid]) ** 2)) for sv in svals]
print(f"BEST GLOBAL pressure scale s = {s_global:.3f}  "
      f"(RMS {rms0:.4f} -> {rms_r:.4f} mε, {100*(1-rms_r/rms0):.0f}% variance reduction)")
print(f"per-snapshot s range: {s_snap.min():.2f} .. {s_snap.max():.2f}")

# ---- implied within-fracture pressure ---------------------------------------
pd_ = np.load(PRESSURE_NPZ, allow_pickle=True)
p_inj = np.asarray(pd_["data"], float)
p_ic = float(p_inj[0])
p_t = T1 + pd.to_timedelta(np.asarray(pd_["taxis"], float), unit="s")
dp_inj = p_inj - p_ic
frac_dp = s_global * dp_inj                                 # scaled perturbation (psi)
print(f"injection Δp peak = {dp_inj.max():.0f} psi;  implied fracture Δp peak (s={s_global:.2f}) "
      f"= {frac_dp.max():.0f} psi;  absolute peak = {p_ic + frac_dp.max():.0f} psi")

# ---- figures -----------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
# (a) misfit vs scale
ax[0,0].plot(svals, misfit, "b-"); ax[0,0].axvline(s_global, color="r", ls="--",
             label=f"best s={s_global:.2f}")
ax[0,0].set(xlabel="pressure scale s", ylabel="RMS misfit (mε)",
            title="(a) misfit vs pressure scale (T1→T2)"); ax[0,0].grid(alpha=0.3); ax[0,0].legend()
# (b) profiles near T2 (last common window)
j = len(common) - 1
ax[0,1].plot(O[:, j], o_md, "k-", lw=2, label=f"observed ({common[j]:%m-%d %H:%M})")
ax[0,1].plot(s_global * M[:, j], o_md, "r--", lw=2, label=f"MOOSE×{s_global:.2f}")
ax[0,1].plot(M[:, j], o_md, "0.7", lw=1, label="MOOSE (s=1)")
ax[0,1].set(xlabel="strain (mε)", ylabel="MD (ft)", title="(b) profile near T2")
ax[0,1].set_ylim(10500, 10200); ax[0,1].grid(alpha=0.3); ax[0,1].legend(fontsize=8)
# (c) per-snapshot scale over time
ax[1,0].plot(common, s_snap, "o-", color="tab:green"); ax[1,0].axhline(s_global, color="r", ls="--",
             label=f"global {s_global:.2f}")
ax[1,0].set(xlabel="time", ylabel="per-snapshot scale s", title="(c) per-snapshot pressure scale")
ax[1,0].grid(alpha=0.3); ax[1,0].legend(); ax[1,0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
# (d) implied fracture pressure vs injection
ax[1,1].plot(p_t, p_inj, "0.6", lw=1.2, label="injection (s=1)")
ax[1,1].plot(p_t, p_ic + frac_dp, "r-", lw=1.8, label=f"implied fracture p (s={s_global:.2f})")
ax[1,1].axhline(p_ic, color="k", ls=":", lw=1, label=f"IC {p_ic:.0f} psi")
ax[1,1].set(xlabel="time", ylabel="pressure (psi)", title="(d) implied within-fracture pressure")
ax[1,1].grid(alpha=0.3); ax[1,1].legend(fontsize=8); ax[1,1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
fig.suptitle(f"V1 tensile-SRV: pressure-scale match to observed 4h profiles (T1→T2)\n"
             f"best s={s_global:.2f}, RMS {rms0:.3f}→{rms_r:.3f} mε", fontweight="bold")
fig.savefig(FIG / "pressure_scale_match.png", dpi=140)
print("Saved:", FIG / "pressure_scale_match.png")

# waterfall comparison
fig2, axs = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True, sharey=True)
ext = [mdates.date2num(common[0].to_pydatetime()), mdates.date2num(common[-1].to_pydatetime()), o_md[-1], o_md[0]]
for a, (mat, ttl, cm, lim) in zip(axs, [
        (O, "observed", "seismic", 0.1), (s_global*M, f"MOOSE×{s_global:.2f}", "seismic", 0.1),
        (O - s_global*M, "residual", "bwr", 0.05)]):
    im = a.imshow(mat, aspect="auto", cmap=cm, vmin=-lim, vmax=lim, extent=ext, interpolation="none")
    a.xaxis_date(); a.set_title(ttl); a.set_ylim(10500, 10200)
    a.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")); plt.colorbar(im, ax=a, label="mε")
axs[0].set_ylabel("MD (ft)")
fig2.suptitle("V1 tensile-SRV T1→T2: observed vs scaled MOOSE tensile vs residual", fontweight="bold")
fig2.savefig(FIG / "pressure_scale_waterfall.png", dpi=140)
print("Saved:", FIG / "pressure_scale_waterfall.png")
