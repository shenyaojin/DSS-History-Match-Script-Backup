"""Compare the SRV geometry variants under the same two-basis inversion.

For each MOOSE project it runs the identical per-4h fit used by ab_inversion.py

    observed(z, t_j)  ~  a_j * MOOSE_tensile(z, t_j)  +  b_j * DDM_shear(z, t_j)

(NNLS, with the degeneracy guard that forces b = 0 while the shear basis is still ~0),
then reports variance reduction, the inferred within-fracture pressure, and -- the point
of the exercise -- the late-time depth profile, where the deep poroelastic compression
lobe either does or does not reappear.

Variants:
  v3_srv_asym       current: asymmetric, 65 ft up / 27 ft down, boundary 11 ft past the shear plane
  v4_srv_centred    fracture at the SRV centre AND boundary on the shear plane (10319-10427.8)
  v4_srv_boundary   boundary on the shear plane, V3's shallow base kept  (10319-10400)
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

import fault_geometry as fg

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "output" / "inversion"
VINTAGE = {
    # tag: (observation npz, DDM shear-basis csv, T1, T2)
    "1100": ("observation_T1_1100.npz",
             "v1_ddm_shear_strain_4h_20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref.csv",
             "2025-02-24 11:00", "2025-02-28 00:00"),
    "1200": ("observation_T1_1200.npz",
             "fault2_shear_strain_4h_20250224_1200_to_20250304_0000_10200_10500ft_4h_mean_T1_ref.csv",
             "2025-02-24 12:00", "2025-02-28 00:00"),
}

FT = 0.3048

ALL_VARIANTS = {
    "v3_srv_asym": ("V3  (250 ft frac, asym 10308-10400)", "#c0392b", "-"),
    "v4_srv_centred": ("V4 centred  (10319-10427.8)", "#1a6ea8", "-"),
    "v4_srv_boundary": ("V4 boundary (10319-10400)", "#b8860b", "--"),
    "v5_full_length": ("V5  (3870 ft, asym cross-section)", "#1a6ea8", "-"),
    "v6_symmetric_srv": ("V6  (3400 ft, -1600/+1800)", "#b8860b", "--"),
    "v7_past_injector": ("V7  (5134 ft, BC 967 ft away - INVALID)", "#7f8c8d", ":"),
    "v6b_bc_at_fiber": ("V6b (3400 ft, -1600/+1800, BC at fiber)", "#c0392b", "--"),
    "v7b_bc_at_fiber": ("V7b (5134 ft, -1600/+3534, BC at fiber)", "#1a7f5a", "-"),
    "v8_parallel_faults": ("V8  (SRV -1920/+1920, BC on well)", "#b8860b", "--"),
    "v9_srv_to_injector": ("V9  (SRV -1800/+1934, STAR 10375 - superseded)", "#b8860b", "--"),
    "v10_final": ("V10 (SRV -1800/+1934)", "#b8860b", "--"),
    "v11_srv_extended": ("V11 (with hf core, k=1e-13)", "#1a6ea8", "-"),
    "v12_no_hf_core": ("V12 (NO hf core, k=1e-16 = srv_narrow)", "#c0392b", "--"),
}
_ap = argparse.ArgumentParser(description="Compare SRV geometry variants")
_ap.add_argument("--variants", default="v3_srv_asym,v4_srv_centred,v4_srv_boundary",
                 help="comma-separated MOOSE project names under output/")
_ap.add_argument("--out", default="srv_variant_comparison",
                 help="basename for the figure and csv")
_ap.add_argument("--figdir", default="v4_srv_centred", help="subfolder under figs/tensile_fault_qc")
_ap.add_argument("--vintage", default="1100", choices=sorted(VINTAGE),
                 help="which observation / DDM-shear pair to fit against")
_args = _ap.parse_args()
VARIANTS = [(v, *ALL_VARIANTS[v]) for v in _args.variants.split(",")]
_obs, _shear, _t1, _t2 = VINTAGE[_args.vintage]
OBS_NPZ = OUT / _obs
DDM_SHEAR = REPO / "data_fervo" / "legacy" / "07152026_decomposed" / _shear
FIG = REPO / "figs" / "tensile_fault_qc" / _args.figdir
FIG.mkdir(parents=True, exist_ok=True)
T1, T2 = pd.Timestamp(_t1), pd.Timestamp(_t2)
SHEAR_MD = fg.shear_md(_args.vintage)
# the tensile fault pierces the well at a different MD in the parallel-fault update
STAR = fg.star(_args.vintage)
print(f"vintage {_args.vintage}: T1={T1}  STAR={STAR}  shear basis {DDM_SHEAR.name}")
C_OBS, MUTED, INK = "#2c3e50", "#7f8c8d", "#2c3e50"

# ---------------------------------------------------------------- observation
z = np.load(OBS_NPZ, allow_pickle=True)
O_full = np.asarray(z["strain_4h"], float)
o_md = np.asarray(z["md_ft"], float)
o_win = pd.DatetimeIndex([pd.Timestamp(str(s)) for s in z["window_starts"]])

# ---------------------------------------------------------------- DDM shear basis
bdf = pd.read_csv(DDM_SHEAR)
b_md = bdf["measured_depth_ft"].to_numpy(float)
b_win = pd.DatetimeIndex(pd.to_datetime(bdf.columns[1:]))
B_full = bdf.iloc[:, 1:].to_numpy(float)

common = pd.DatetimeIndex([t for t in o_win if t in set(b_win)])
oi = [list(o_win).index(t) for t in common]
bi = [list(b_win).index(t) for t in common]
nt = len(common)
O = O_full[:, oi]
B = np.array([np.interp(o_md, b_md, B_full[:, j]) for j in bi]).T
tc_s = np.array([(t - T1).total_seconds() for t in common])
print(f"observation {O.shape}, {nt} windows {common[0]:%m-%d %H:%M} .. {common[-1]:%m-%d %H:%M}")


def moose_basis(proj):
    """T1-referenced fiber strain_yy from a MOOSE project, on the observation grid."""
    d = REPO / "output" / proj
    taxis_s = pd.read_csv(d / f"{proj}_input_csv.csv")["time"].to_numpy(float)
    vpp = sorted(glob.glob(str(d / f"{proj}_input_csv_fiber_strain_sampler_*ft_*.csv")))
    n = min(len(vpp), len(taxis_s))
    taxis_s, vpp = taxis_s[:n], vpp[:n]
    y = None
    for f in vpp:
        dd = pd.read_csv(f)
        if len(dd):
            y = dd.sort_values("y")["y"].to_numpy(float)
            break
    cols = []
    for f in vpp:
        dd = pd.read_csv(f)
        cols.append(dd.sort_values("y")["strain_yy"].to_numpy(float) if len(dd)
                    else np.zeros_like(y))
    ms = np.column_stack(cols)
    ms = (ms - ms[:, [0]]) * 1e3                                   # millistrain, T1-ref
    m_md = STAR + y / FT
    order = np.argsort(m_md)
    m_md, ms = m_md[order], ms[order]
    M_t = np.array([np.interp(tc_s, taxis_s, ms[k, :]) for k in range(ms.shape[0])])
    return np.array([np.interp(o_md, m_md, M_t[:, j]) for j in range(nt)]).T


def invert(M):
    """Per-window NNLS on [MOOSE, DDM]; returns a, b, model, variance reduction.

    Also records the UNCONSTRAINED b. Where that is negative, NNLS has clipped b to zero and
    the shear amplitude is simply not identifiable in that window -- the shear basis is still
    tiny and is ~0.6 correlated with the tensile basis. Those windows must not be read as
    'no slip'.
    """
    a = np.full(nt, np.nan)
    b = np.full(nt, np.nan)
    b_unc = np.full(nt, np.nan)
    model = np.zeros_like(O)
    bscale = float(np.nanmax(np.abs(B)))
    for j in range(nt):
        dj, mj, bj = O[:, j], M[:, j], B[:, j]
        v = np.isfinite(dj) & np.isfinite(mj) & np.isfinite(bj)
        if v.sum() < 3:
            continue
        if float(np.nanmax(np.abs(bj[v]))) > 1e-3 * bscale:
            A = np.column_stack([mj[v], bj[v]])
            coef, _ = nnls(A, dj[v])
            a[j], b[j] = coef
            b_unc[j] = np.linalg.lstsq(A, dj[v], rcond=None)[0][1]
        else:
            den = float(np.dot(mj[v], mj[v]))
            a[j] = max(0.0, float(np.dot(mj[v], dj[v]) / den)) if den > 0 else 0.0
            b[j] = 0.0
        model[:, j] = a[j] * mj + b[j] * bj
    v = np.isfinite(O) & np.isfinite(model)
    rms0 = float(np.sqrt(np.nanmean(O[v] ** 2)))
    rmsr = float(np.sqrt(np.nanmean((O - model)[v] ** 2)))
    return a, b, model, 100 * (1 - (rmsr / rms0) ** 2), rms0, rmsr, b_unc


results = {}
for proj, label, colour, ls in VARIANTS:
    M = moose_basis(proj)
    a, b, model, vr, rms0, rmsr, b_unc = invert(M)
    results[proj] = dict(M=M, a=a, b=b, b_unc=b_unc, model=model, vr=vr, rms0=rms0, rmsr=rmsr,
                         label=label, colour=colour, ls=ls)
    print(f"\n{label}")
    print(f"  variance reduction {vr:5.1f}%   RMS {rms0:.4f} -> {rmsr:.4f} me")
    # The first few windows sit at T1 where the modelled strain is still ~0, so `a` is
    # degenerate there and blows up; quote the post-T2 window instead.
    print(f"  a after T2 {np.nanmedian(a[common >= T2]):.3f}   a final {a[-1]:.3f}   "
          f"(all-window median {np.nanmedian(a):.3f} is skewed by {int((a > 3).sum())} "
          f"degenerate T1 windows)")
    print(f"  b median after T2 {np.nanmedian(b[common >= T2]):.3f}   b final {b[-1]:.3f}")

# inferred pressure (a scales the pressure perturbation)
pz = np.load(REPO / "output" / VARIANTS[0][0] / "das_pressure_T1_prepended.npz",
             allow_pickle=True)
p_base = np.asarray(pz["data"], float)
p_ic = float(p_base[0])
p_base_win = np.interp(tc_s, np.asarray(pz["taxis"], float), p_base)

fig, axs = plt.subplots(2, 2, figsize=(16.5, 11), constrained_layout=True)
(axA, axB), (axC, axD) = axs
jlast = nt - 1

# --- (A) late-time depth profile: does the deep lobe come back? ---------------
axA.plot(O[:, jlast], o_md, color=C_OBS, lw=3.2, label="observed", zorder=5)
for proj, r in results.items():
    axA.plot(r["model"][:, jlast], o_md, color=r["colour"], lw=2.1, ls=r["ls"],
             label=f"{r['label']}   VR {r['vr']:.1f}%")
axA.axvline(0, color=MUTED, lw=1)
axA.axhline(STAR, color=INK, ls=":", lw=1.4)
axA.text(axA.get_xlim()[0], STAR - 3, " fracture", fontsize=8.5, color=INK)
axA.axhline(SHEAR_MD, color="#8e44ad", ls="--", lw=1.6)
axA.text(axA.get_xlim()[0], SHEAR_MD - 3, f" shear plane MD {SHEAR_MD:.0f}", fontsize=8.5, color="#8e44ad")
axA.axhspan(10420, 10500, color="#c0392b", alpha=0.07)
axA.text(axA.get_xlim()[1], 10460, "deep-lobe zone  ", fontsize=8.5, color="#c0392b",
         ha="right", fontweight="bold")
axA.set_ylim(10500, 10200)
axA.set_xlabel("strain  [m$\\varepsilon$]  (T1-referenced)")
axA.set_ylabel("Measured depth  [ft]")
axA.set_title(f"(A) Depth profile at {common[jlast]:%m-%d %H:%M} — the deciding panel",
              fontweight="bold", fontsize=11)
axA.legend(fontsize=8.5, loc="lower left", frameon=False)
axA.grid(alpha=0.25)

# --- (B) residual RMS vs depth ------------------------------------------------
for proj, r in results.items():
    res = np.sqrt(np.nanmean((O - r["model"]) ** 2, axis=1))
    axB.plot(res, o_md, color=r["colour"], lw=2.1, ls=r["ls"], label=r["label"])
axB.plot(np.sqrt(np.nanmean(O ** 2, axis=1)), o_md, color=C_OBS, lw=2.4, ls=":",
         label="observation RMS (no model)")
axB.axhline(STAR, color=INK, ls=":", lw=1.4)
axB.axhspan(10420, 10500, color="#c0392b", alpha=0.07)
axB.set_ylim(10500, 10200)
axB.set_xlabel("RMS residual over all windows  [m$\\varepsilon$]")
axB.set_ylabel("Measured depth  [ft]")
axB.set_title("(B) Where each geometry fails", fontweight="bold", fontsize=11)
axB.legend(fontsize=8.5, loc="lower right", frameon=False)
axB.grid(alpha=0.25)

# --- (C) inferred within-fracture pressure ------------------------------------
axC.plot(common, p_base_win, color=MUTED, lw=1.8, ls="--", label="DAS baseline (a = 1)")
for proj, r in results.items():
    axC.plot(common, p_ic + r["a"] * (p_base_win - p_ic), marker="o", color=r["colour"],
             lw=2.1, ms=3.5, ls=r["ls"], label=r["label"])
axC.axhline(p_ic, color=INK, ls=":", lw=1)
axC.axvline(T2, color="#8e44ad", ls="--", lw=1.5)
axC.text(T2, 1.01, "T2", transform=axC.get_xaxis_transform(), color="#8e44ad",
         ha="center", fontweight="bold")
axC.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
axC.set_xlim(common[0], common[-1])
axC.set_xlabel("Time  [UTC-7]")
axC.set_ylabel("within-fracture pressure  [psi]")
axC.set_title("(C) Inferred pressure   p = IC + a$\\cdot$(p$_{DAS}$ - IC)", fontweight="bold",
              fontsize=11)
axC.legend(fontsize=8.5, loc="upper left", frameon=False)
axC.grid(alpha=0.25)

# --- (D) absolute shear slip, against the DDM baseline ------------------------
# b(t) scales the DDM shear BASIS, so the physical quantity it implies is the fault-2 slip:
#   inferred slip(t) = b(t) * |imposed DDM slip(t)|.  Plot that, not the dimensionless b.
_HIST = REPO / "data_fervo" / "legacy" / "07152026" / (
    "two_fault_histories_20250224_1200_to_20250304_0000_10200_10500ft_4h_mean_T1_ref.csv")
if _args.vintage == "1200" and _HIST.exists():
    _h = pd.read_csv(_HIST)
    _hs = (pd.to_datetime(_h["time"]) - T1).dt.total_seconds().to_numpy()
    slip0 = np.interp(tc_s, _hs, np.abs(_h["fault2_shear_ft"].to_numpy(float)))
    axD.plot(common, slip0, lw=2.8, color=MUTED, ls="--",
             label=f"DDM baseline, b = 1  (peak {slip0.max():.4f} ft)")
    axD.fill_between(common, 0, slip0, color=MUTED, alpha=.10)
    for proj, r in results.items():
        sl = r["b"] * slip0
        axD.plot(common, sl, marker="o", color=r["colour"], lw=2.1, ms=3.5, ls=r["ls"],
                 label=f"{r['label']}   peak {np.nanmax(sl):.4f} ft")
    _bu = list(results.values())[0]["b_unc"]
    _clip = np.isfinite(_bu) & (_bu < 0)
    if _clip.any():
        _last = int(np.max(np.where(_clip)[0]))
        axD.axvspan(common[0], common[_last], color="#7f8c8d", alpha=.16, zorder=0)
        axD.text(common[max(_last - 6, 0)], axD.get_ylim()[1] * 0.60,
                 "b NOT identifiable here\n(shear basis still tiny and ~0.6\n"
                 "correlated with the tensile basis;\nunconstrained b < 0, NNLS clips to 0)",
                 fontsize=8.5, color="#4d5656", ha="right", va="top")
    axD.set_ylabel("fault-2 shear slip  [ft]")
    axD.set_title("(D) Absolute shear slip — inferred vs the DDM baseline",
                  fontweight="bold", fontsize=11)
else:
    for proj, r in results.items():
        axD.plot(common, r["b"], marker="o", color=r["colour"], lw=2.1, ms=3.5, ls=r["ls"],
                 label=r["label"])
    axD.axhline(1.0, color=MUTED, lw=1, ls="--")
    axD.set_ylabel("b   (DDM shear scale)")
    axD.set_title("(D) Shear amplitude the fit needs", fontweight="bold", fontsize=11)
axD.axvline(T2, color="#8e44ad", ls="--", lw=1.5)
axD.text(T2, 1.01, "T2", transform=axD.get_xaxis_transform(), color="#8e44ad",
         ha="center", fontweight="bold")
axD.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
axD.set_xlim(common[0], common[-1])
axD.set_xlabel("Time  [UTC-7]")
axD.legend(fontsize=8.5, loc="upper left", frameon=False)
axD.grid(alpha=0.25)

for ax in (axA, axB, axC, axD):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("SRV geometry variants under the same two-basis inversion",
             fontsize=14, fontweight="bold")
out = FIG / f"{_args.out}.png"
fig.savefig(out, dpi=140)
print("\nsaved", out)

summary = pd.DataFrame([{"variant": p, "label": r["label"], "variance_reduction_pct": r["vr"],
                         "rms_obs_me": r["rms0"], "rms_residual_me": r["rmsr"],
                         "a_median_after_T2": float(np.nanmedian(r["a"][common >= T2])),
                         "a_final": float(r["a"][-1]),
                         "b_median_after_T2": float(np.nanmedian(r["b"][common >= T2])),
                         "b_final": float(r["b"][-1]),
                         "p_peak_psi": float(np.nanmax(p_ic + r["a"] * (p_base_win - p_ic)))}
                        for p, r in results.items()])
summary.to_csv(OUT / f"{_args.out}.csv", index=False)
print(summary.to_string(index=False))
print("\nwrote", OUT / f"{_args.out}.csv")
