"""Per-4h two-basis inversion:   observed  ≈  a_j · MOOSE_tensile  +  b_j · DDMpy_profiles

The shear signature on the fiber is intrinsically 3-D (a strike-slip dislocation's AXIAL DAS strain
depends on the well's 3-D orientation); our 2-D plane-strain MOOSE puts strike-slip into strain_xy
(≈0 in the axial strain_yy), so the 3-D part must come from DDMpy. The forward is therefore a
two-basis superposition fit once per 4-hour window, giving two histories a_j and b_j.

  observation : built fresh from the raw LFDAS .npz by integrating strain-rate -> strain
                (output/inversion/observation_T1_1100.npz, from build_observation.py --t1 11:00)
  basis a     : MOOSE poroelastic tensile, output/v1_tensile_srv_das  (fiber strain_yy)
  basis b     : data_fervo/legacy/strain_4h_mean_profiles_...csv      (DDMpy 4-hour profiles)

Includes a CSV-identity diagnostic: if the fit returns b≈1, a≈0 with near-perfect variance
reduction, the "basis" is really the observation itself and the result must not be presented as
a physical inversion.
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls

REPO = Path(__file__).resolve().parents[3]
PROJ = "v1_tensile_srv_das"
MOOSE_DIR = REPO / "output" / PROJ
OBS_NPZ = REPO / "output" / "inversion" / "observation_T1_1100.npz"
DDM_DEFAULT = REPO / "data_fervo" / "legacy" / (
    "strain_4h_mean_profiles_20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref.csv")
DDM_SHEAR = REPO / "data_fervo" / "legacy" / "07152026_decomposed" / (
    "v1_ddm_shear_strain_4h_20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref.csv")

_ap = argparse.ArgumentParser(description="Per-4h two-basis inversion: obs ~ a*MOOSE + b*DDMpy")
_ap.add_argument("--basis-b", default=str(DDM_DEFAULT),
                 help="CSV for basis b; use 'shear' for the decomposed DDM shear")
_ap.add_argument("--tag", default="", help="suffix for output filenames")
_ap.add_argument("--moose-proj", default="v1_tensile_srv_das",
                 help="MOOSE project dir under output/ providing basis a")
_args = _ap.parse_args()
PROJ = _args.moose_proj
MOOSE_DIR = REPO / "output" / PROJ
DDM_CSV = DDM_SHEAR if _args.basis_b == "shear" else Path(_args.basis_b)
TAG = ("_" + _args.tag) if _args.tag else ""

OUT = REPO / "output" / "inversion"
FIG = REPO / "figs" / "tensile_fault_qc" / "inversion" / "ab"
FIG.mkdir(parents=True, exist_ok=True)
print(f"basis b file: {DDM_CSV.name}")

FT = 0.3048
STAR_DEPTH_FT = 10373.4
T1 = pd.Timestamp("2025-02-24 11:00")
T2 = pd.Timestamp("2025-02-28 00:00")

# ---------------------------------------------------------------- 1. observation (from npz)
z = np.load(OBS_NPZ, allow_pickle=True)
O_full = np.asarray(z["strain_4h"], float)                 # [md, win] millistrain, T1-ref
o_md = np.asarray(z["md_ft"], float)
o_win = pd.DatetimeIndex([pd.Timestamp(str(s)) for s in z["window_starts"]])
print(f"observation : {O_full.shape}  MD {o_md.min():.0f}..{o_md.max():.0f}  "
      f"windows {o_win[0]:%m-%d %H:%M} .. {o_win[-1]:%m-%d %H:%M}")

# ---------------------------------------------------------------- 2. basis a: MOOSE tensile
tdf = pd.read_csv(MOOSE_DIR / f"{PROJ}_input_csv.csv")
taxis_s = tdf["time"].to_numpy(float)
vpp = sorted(glob.glob(str(MOOSE_DIR / f"{PROJ}_input_csv_fiber_strain_sampler_*ft_*.csv")))
n = min(len(vpp), len(taxis_s))
taxis_s, vpp = taxis_s[:n], vpp[:n]
y = None
for f in vpp:
    dd = pd.read_csv(f)
    if len(dd):
        y = dd.sort_values("y")["y"].to_numpy(float); break
cols = [pd.read_csv(vpp[i]).sort_values("y")["strain_yy"].to_numpy(float) if len(pd.read_csv(vpp[i]))
        else np.zeros_like(y) for i in range(n)]
m_strain = (np.column_stack(cols) - np.column_stack(cols)[:, [0]]) * 1e3      # mε, T1-ref
m_md = STAR_DEPTH_FT + y / FT
order = np.argsort(m_md); m_md, m_strain = m_md[order], m_strain[order]
print(f"basis a MOOSE: {m_strain.shape}  MD {m_md.min():.0f}..{m_md.max():.0f}  "
      f"t 0..{taxis_s.max()/86400:.2f} d")

# ---------------------------------------------------------------- 3. basis b: DDMpy profiles
bdf = pd.read_csv(DDM_CSV)
b_md = bdf["measured_depth_ft"].to_numpy(float)
b_win = pd.DatetimeIndex(pd.to_datetime(bdf.columns[1:]))
B_full = bdf.iloc[:, 1:].to_numpy(float)                    # [md, win] mε
print(f"basis b DDMpy: {B_full.shape}  MD {b_md.min():.0f}..{b_md.max():.0f}  "
      f"windows {b_win[0]:%m-%d %H:%M} .. {b_win[-1]:%m-%d %H:%M}")

# ---------------------------------------------------------------- 4. align (time + space)
common = [t for t in o_win if t in set(b_win)]              # windows present in both
oi = [list(o_win).index(t) for t in common]
bi = [list(b_win).index(t) for t in common]
common = pd.DatetimeIndex(common)
nt = len(common)

O = O_full[:, oi]                                                            # [o_md, nt]
B = np.array([np.interp(o_md, b_md, B_full[:, j]) for j in bi]).T             # -> obs MD grid
tc_s = np.array([(t - T1).total_seconds() for t in common])
M_t = np.array([np.interp(tc_s, taxis_s, m_strain[k, :]) for k in range(m_strain.shape[0])])
M = np.array([np.interp(o_md, m_md, M_t[:, j]) for j in range(nt)]).T         # -> obs MD grid

assert O.shape == M.shape == B.shape, (O.shape, M.shape, B.shape)
print(f"\naligned: O/M/B all {O.shape} on MD {o_md.min():.0f}..{o_md.max():.0f}, "
      f"{nt} windows {common[0]:%m-%d %H:%M} .. {common[-1]:%m-%d %H:%M}")

# ---------------------------------------------------------------- 5. per-window inversion
a_win = np.full(nt, np.nan); b_win_c = np.full(nt, np.nan)
a_un = np.full(nt, np.nan); b_un = np.full(nt, np.nan)
corr_BO = np.full(nt, np.nan)
model = np.zeros_like(O)
B_SCALE = float(np.nanmax(np.abs(B)))          # overall amplitude of basis b (degeneracy guard)
for j in range(nt):
    d, mj, bj = O[:, j], M[:, j], B[:, j]
    v = np.isfinite(d) & np.isfinite(mj) & np.isfinite(bj)
    if v.sum() < 3:
        continue
    # Guard: where basis b carries no signal (e.g. the DDM shear is ~0 before T2) the scale b is
    # degenerate — any value gives the same model — so solve for a alone and report b = 0.
    if float(np.nanmax(np.abs(bj[v]))) > 1e-3 * B_SCALE:
        A = np.column_stack([mj[v], bj[v]])
        coef, _ = nnls(A, d[v])                              # non-negative (scale-up factors)
        a_win[j], b_win_c[j] = coef
        un, *_ = np.linalg.lstsq(A, d[v], rcond=None)         # unconstrained diagnostic
        a_un[j], b_un[j] = un
    else:
        aa = max(0.0, float(np.dot(mj[v], d[v]) / np.dot(mj[v], mj[v]))) if np.dot(mj[v], mj[v]) > 0 else 0.0
        a_win[j], b_win_c[j] = aa, 0.0
        a_un[j], b_un[j] = aa, 0.0
    model[:, j] = a_win[j] * mj + b_win_c[j] * bj
    if np.std(bj[v]) > 0 and np.std(d[v]) > 0:
        corr_BO[j] = float(np.corrcoef(bj[v], d[v])[0, 1])

v = np.isfinite(O) & np.isfinite(model)
rms0 = float(np.sqrt(np.nanmean(O[v] ** 2)))
rms_r = float(np.sqrt(np.nanmean((O - model)[v] ** 2)))
vr = 100 * (1 - (rms_r / rms0) ** 2)
print(f"\nRESULT: RMS {rms0:.4f} -> {rms_r:.4f} mε   |   variance reduction {vr:.1f}%")
print(f"  a (MOOSE scale): {np.nanmin(a_win):.3f} .. {np.nanmax(a_win):.3f}  median {np.nanmedian(a_win):.3f}")
print(f"  b (DDMpy scale): {np.nanmin(b_win_c):.3f} .. {np.nanmax(b_win_c):.3f}  median {np.nanmedian(b_win_c):.3f}")
pre, post = common < T2, common >= T2
print(f"  b before T2: median {np.nanmedian(b_win_c[pre]):.3f} | after T2: median {np.nanmedian(b_win_c[post]):.3f}")
print(f"  corr(B,O) per window: median {np.nanmedian(corr_BO):.3f}  max {np.nanmax(corr_BO):.3f}")
print(f"  unconstrained: a {np.nanmedian(a_un):+.3f}, b {np.nanmedian(b_un):+.3f} (medians)")

# ---------------------------------------------------------------- 6. CSV-identity diagnostic
flag = (vr > 98.0) and (abs(np.nanmedian(b_win_c) - 1.0) < 0.15) and (abs(np.nanmedian(a_win)) < 0.10)
print("\n=== CSV-identity diagnostic ===")
if flag:
    print("  *** WARNING: b≈1, a≈0 and variance reduction >98%.")
    print("  *** The 'DDMpy basis' is reproducing the observation almost exactly — i.e. that CSV is")
    print("  *** effectively the observation itself, not an independent basis. Do NOT report this as")
    print("  *** a physical inversion; pick a genuine DDM output (e.g. the decomposed shear) instead.")
else:
    print(f"  OK — no identity signature (VR {vr:.1f}%, median b {np.nanmedian(b_win_c):.3f}, "
          f"median a {np.nanmedian(a_win):.3f}).")

# ---------------------------------------------------------------- 7. outputs
np.savez(OUT / f"ab_inversion_result{TAG}.npz",
         times=np.array([str(t) for t in common]), a_win=a_win, b_win=b_win_c,
         a_unconstrained=a_un, b_unconstrained=b_un, corr_BO=corr_BO,
         vr=vr, rms0=rms0, rms_r=rms_r, md_ft=o_md)
pd.DataFrame({"window_start": common, "a_moose": a_win, "b_ddmpy": b_win_c,
              "a_unconstrained": a_un, "b_unconstrained": b_un,
              "corr_B_O": corr_BO}).to_csv(OUT / f"ab_inversion_ab{TAG}.csv", index=False)
print("\nwrote", OUT / f"ab_inversion_result{TAG}.npz", "and", OUT / f"ab_inversion_ab{TAG}.csv")

# ---------------------------------------------------------------- 7b. corrected pressure curve
# The MOOSE strain is T1-referenced, so it responds to the pressure PERTURBATION dp = p - IC.
# Scaling the modelled strain by a is therefore equivalent to scaling dp by a:
#     p_corrected(t) = IC + a(t) * (p_baseline(t) - IC)
PREPPED = MOOSE_DIR / "das_pressure_T1_prepended.npz"
if PREPPED.exists():
    pz = np.load(PREPPED, allow_pickle=True)
    p_base = np.asarray(pz["data"], float)
    p_tax = np.asarray(pz["taxis"], float)
    p_ic = float(p_base[0])
    p_base_win = np.interp(tc_s, p_tax, p_base)                 # baseline at window centres
    p_corr_win = p_ic + a_win * (p_base_win - p_ic)             # corrected by a
    pd.DataFrame({"window_start": common, "a_moose": a_win,
                  "p_baseline_psi": p_base_win,
                  "p_corrected_psi": p_corr_win}).to_csv(OUT / f"ab_pressure_curve{TAG}.csv", index=False)
    print(f"pressure: IC {p_ic:.0f} psi | baseline peak {p_base_win.max():.0f} | "
          f"corrected peak {np.nanmax(p_corr_win):.0f} psi (Δp {np.nanmax(p_corr_win) - p_ic:.0f})")

    figp, axp = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    axp.plot(T1 + pd.to_timedelta(p_tax, unit="s"), p_base, color="tab:orange", lw=1.6,
             label="baseline DAS pressure (a = 1)")
    axp.plot(common, p_corr_win, "o-", color="#b30000", lw=2.2, ms=4,
             label="corrected   p = IC + a·(p_base − IC)")
    axp.axhline(p_ic, color="k", ls=":", lw=1, label=f"IC {p_ic:.0f} psi")
    axp.axvline(T2, color="green", ls="--", lw=1.4)
    axp.text(T2, 1.01, "T2", transform=axp.get_xaxis_transform(), color="green",
             ha="center", fontweight="bold")
    axp.set_xlim(common[0], common[-1]); axp.set_xlabel("Time [UTC-7]")
    axp.set_ylabel("pressure [psi]"); axp.legend(loc="upper left"); axp.grid(alpha=0.3)
    axp.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    axp.set_title(f"Within-fracture pressure corrected by a — {p_ic:.0f} → "
                  f"{np.nanmax(p_corr_win):.0f} psi", fontweight="bold")
    figp.savefig(FIG / f"ab_pressure_curve{TAG}.png", dpi=140); plt.close(figp)

# ---- fig 1: waterfalls -------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True, sharey=True)
ext = [mdates.date2num(common[0].to_pydatetime()), mdates.date2num(common[-1].to_pydatetime()),
       o_md[-1], o_md[0]]
lim = float(np.nanpercentile(np.abs(O), 98))
for ax, (mat, ttl, lm) in zip(axs, [(O, "observed (from npz, integrated)", lim),
                                    (model, "model  a·MOOSE + b·DDMpy", lim),
                                    (O - model, "residual", lim / 2)]):
    im = ax.imshow(mat, aspect="auto", cmap="seismic", vmin=-lm, vmax=lm, extent=ext,
                   interpolation="none")
    ax.xaxis_date(); ax.set_title(ttl); ax.set_ylim(10500, 10200)
    ax.axvline(mdates.date2num(T2.to_pydatetime()), color="lime", ls="--", lw=1.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d")); plt.colorbar(im, ax=ax, label="mε")
axs[0].set_ylabel("MD [ft]")
fig.suptitle(f"Per-4h two-basis inversion — variance reduction {vr:.1f}%", fontweight="bold")
fig.savefig(FIG / f"ab_waterfall{TAG}.png", dpi=140); plt.close(fig)

# ---- fig 2: a and b histories ------------------------------------------------
fig2, ax2 = plt.subplots(2, 1, figsize=(11, 7.5), constrained_layout=True, sharex=True)
ax2[0].plot(common, a_win, "o-", color="tab:orange"); ax2[0].axvline(T2, color="0.5", ls=":")
ax2[0].set(ylabel="a  (MOOSE scale)", title="(a) per-window MOOSE tensile scale"); ax2[0].grid(alpha=0.3)
ax2[1].plot(common, b_win_c, "s-", color="tab:blue"); ax2[1].axvline(T2, color="0.5", ls=":")
ax2[1].set(ylabel="b  (DDMpy scale)", xlabel="time",
           title="(b) per-window DDMpy scale  (T2 dotted)"); ax2[1].grid(alpha=0.3)
ax2[1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
fig2.suptitle("Recovered per-4h scales", fontweight="bold")
fig2.savefig(FIG / f"ab_histories{TAG}.png", dpi=140); plt.close(fig2)

# ---- fig 3: 107-style 4-hour-profile overlay ---------------------------------
cnum = mdates.date2num([c.to_pydatetime() for c in common])
sref = (max(np.nanpercentile(np.abs(O), 95), np.nanpercentile(np.abs(model), 95)) or 1.0) * 10.0
sec = 28.0 * 3600.0 / sref
fig3, ax3 = plt.subplots(figsize=(16, 6), constrained_layout=True)
im = ax3.imshow(model, aspect="auto", cmap="bwr", vmin=-lim, vmax=lim, extent=ext,
                interpolation="bilinear")
ax3.xaxis_date(); ax3.set_ylim(10500, 10200); ax3.set_ylabel("Gold 4-PB MD [ft]"); ax3.set_xlabel("Time")
for prof, color in [(O, "black"), (model, "red")]:
    for j in range(nt):
        p = prof[:, j]; fin = np.isfinite(p)
        if fin.any():
            ax3.plot(cnum[j] + p[fin] * sec / 86400.0, o_md[fin], color=color, lw=0.8, alpha=0.9, zorder=5)
ax3.plot([], [], "k-", lw=2, label="observed 4h profiles"); ax3.plot([], [], "r-", lw=2, label="model 4h profiles")
ax3.legend(loc="lower right"); ax3.axvline(mdates.date2num(T2.to_pydatetime()), color="green", ls="--", lw=1.5)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d")); plt.colorbar(im, ax=ax3, label="mε")
ax3.set_title(f"Model background + 4-hour profiles: observed (black) vs model (red) — VR {vr:.1f}%",
              fontweight="bold")
fig3.savefig(FIG / f"ab_profiles_overlay{TAG}.png", dpi=140); plt.close(fig3)


# ---- fig 4: 107-style two-panel comparison (observed top / model bottom) -----
def _wf_ax(ax, dat, title):
    im = ax.imshow(dat, aspect="auto", cmap="bwr", vmin=-lim, vmax=lim, extent=ext,
                   interpolation="bilinear")
    ax.xaxis_date(); ax.set_ylim(10500, 10200)
    ax.set_ylabel("Gold 4-PB MD [ft]"); ax.set_title(title)
    for mt, lb in [(cnum[0], "T1"), (mdates.date2num(T2.to_pydatetime()), "T2"), (cnum[-1], "T3")]:
        ax.axvline(mt, color="green", ls="--", lw=1.6, zorder=4)
        ax.text(mt, 1.01, lb, transform=ax.get_xaxis_transform(), color="green",
                fontweight="bold", ha="center")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.colorbar(im, ax=ax, label="strain [mε]")


def _ovl(ax, profiles, color):
    for j in range(nt):
        p = profiles[:, j]; fin = np.isfinite(p)
        if fin.any():
            ax.plot(cnum[j] + p[fin] * sec / 86400.0, o_md[fin],
                    color=color, lw=0.8, alpha=0.9, zorder=5)


fig4, (p1, p2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, constrained_layout=True)
_wf_ax(p1, O, "Observed DAS strain (from npz, integrated) — 4-hour mean profiles")
_ovl(p1, O, "black")
_wf_ax(p2, model, "Model:  a·MOOSE tensile  +  b·DDMpy shear — 4-hour mean profiles")
_ovl(p2, model, "black")
p2.set_xlabel("Time [UTC-7]")
fig4.suptitle(f"Observed vs model DAS strain — variance reduction {vr:.1f}%",
              fontweight="bold", fontsize=14)
fig4.savefig(FIG / f"ab_compare_obs_vs_model{TAG}.png", dpi=150); plt.close(fig4)
print("saved 4 figures to", FIG)
