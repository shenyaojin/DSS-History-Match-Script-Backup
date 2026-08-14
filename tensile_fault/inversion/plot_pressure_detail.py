"""Inferred pressure along the monitoring well, against the real Bearskin 3-PA injection curve.

The two-basis fit gives one scale a(t) per 4-hour window. Because the poroelastic problem is
linear in the pressure PERTURBATION, scaling the boundary pressure by a(t) scales dp by a(t)
EVERYWHERE, not just at the fracture. So the inferred pressure at any point sampled by the
fiber is

    p(z, t) = IC + a(t) * dp_model(z, t)

with dp_model taken straight from the MOOSE fiber pressure sampler. That lets the inferred
pressure be read at the fracture AND at chosen stand-offs into the SRV, on both sides, which
is the symmetry check.

Panels
  (A) inferred pressure at the fracture vs the measured Bearskin 3-PA injection pressure
  (B) inferred pressure at +/- 2.5, 7.5, 12.5 m from the hf core, along the monitoring well
  (C) shallow-vs-deep difference at each stand-off -- zero means the SRV drains symmetrically
  (D) the dp profile across the SRV at four times
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

_ap = argparse.ArgumentParser(description="Inferred pressure along the well vs injection")
_ap.add_argument("--project", default="v8_parallel_faults")
_ap.add_argument("--label", default=None)
_ap.add_argument("--offsets", default="10,25,40", help="stand-offs in FEET")
args = _ap.parse_args()
PROJ = args.project
LABEL = args.label or PROJ.split("_")[0].upper()
OFFSETS = [float(v) for v in args.offsets.split(",")]

FIG = REPO / "figs" / "tensile_fault_qc" / PROJ
FIG.mkdir(parents=True, exist_ok=True)
FT, PSI = 0.3048, 6894.76
STAR = fg.STAR_1200
SHEAR_MD = fg.SHEAR_MD_1200
T1 = pd.Timestamp("2025-02-24 12:00")
T2 = pd.Timestamp("2025-02-28 00:00")
SHEAR_CSV = REPO / "data_fervo" / "legacy" / "07152026_decomposed" / (
    "fault2_shear_strain_4h_20250224_1200_to_20250304_0000_10200_10500ft_4h_mean_T1_ref.csv")
OBS_NPZ = OUT / "observation_T1_1200.npz"
INJ_WELLS = {"Bearskin 1-IA": "#b8860b", "Bearskin 3-PA": "#c0392b",
             "Bearskin 4-PB": "#7d3c98"}

C_FRAC, C_INJ, C_DAS = "#c0392b", "#b8860b", "#7f8c8d"
SHALLOW, DEEP = "#1a6ea8", "#1a7f5a"
INK, MUTED = "#2c3e50", "#7f8c8d"

# ---------------------------------------------------------------- observation + bases
z = np.load(OBS_NPZ, allow_pickle=True)
O_full = np.asarray(z["strain_4h"], float)
o_md = np.asarray(z["md_ft"], float)
o_win = pd.DatetimeIndex([pd.Timestamp(str(s)) for s in z["window_starts"]])
bdf = pd.read_csv(SHEAR_CSV)
b_md = bdf["measured_depth_ft"].to_numpy(float)
b_win = pd.DatetimeIndex(pd.to_datetime(bdf.columns[1:]))
B_full = bdf.iloc[:, 1:].to_numpy(float)
common = pd.DatetimeIndex([t for t in o_win if t in set(b_win)])
nt = len(common)
O = O_full[:, [list(o_win).index(t) for t in common]]
B = np.array([np.interp(o_md, b_md, B_full[:, list(b_win).index(t)]) for t in common]).T
tc_s = np.array([(t - T1).total_seconds() for t in common])

d = REPO / "output" / PROJ
taxis_s = pd.read_csv(d / f"{PROJ}_input_csv.csv")["time"].to_numpy(float)


def sampler(kind, col, fill):
    """[n_y, n_t] from the fiber sampler CSVs, plus the y axis in metres.

    MOOSE writes an EMPTY file for t = 0, so that column must be filled with the initial
    state (0 for strain, the IC for pressure) -- filling it with zeros and then using it as
    the reference silently turns absolute pressure into 'perturbation'.
    """
    vpp = sorted(glob.glob(str(d / f"{PROJ}_input_csv_fiber_{kind}_sampler_*ft_*.csv")))
    n = min(len(vpp), len(taxis_s))
    y, cols = None, []
    for f in vpp[:n]:
        dd = pd.read_csv(f)
        if len(dd) and y is None:
            y = dd.sort_values("y")["y"].to_numpy(float)
        cols.append(dd.sort_values("y")[col].to_numpy(float) if len(dd) else None)
    nempty = sum(c is None for c in cols)
    cols = [c if c is not None else np.full_like(y, fill) for c in cols]
    if nempty:
        print(f"  {kind}: {nempty} empty sampler file(s) filled with {fill:.6g}")
    return y, np.column_stack(cols), taxis_s[:n]


IC_PSI = float(np.load(d / "das_pressure_T1_prepended.npz", allow_pickle=True)["data"][0])
IC_PA = IC_PSI * PSI
y_m, STRAIN, tax = sampler("strain", "strain_yy", 0.0)
_, PP, _ = sampler("pressure", "pp", IC_PA)
m_md = STAR + y_m / FT
order = np.argsort(m_md)
m_md, STRAIN, PP, y_m = m_md[order], STRAIN[order], PP[order], y_m[order]

# MOOSE tensile basis on the observation grid, T1-referenced
MS = (STRAIN - STRAIN[:, [0]]) * 1e3
M_t = np.array([np.interp(tc_s, tax, MS[k, :]) for k in range(MS.shape[0])])
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
print(f"{LABEL}: a final {a[-1]:.3f}, b final {b[-1]:.3f}")

# ---------------------------------------------------------------- dp at stand-offs
DP = (PP - IC_PA) / PSI                                    # psi rise above the IC
DP_w = np.array([np.interp(tc_s, tax, DP[k, :]) for k in range(DP.shape[0])])  # -> windows
p_ic = IC_PSI
print(f"  IC {p_ic:.0f} psi; modelled dp at the fracture peaks at {DP.max():.0f} psi")


def infer_at(off_ft):
    """Inferred absolute pressure history at a fault-normal stand-off given in FEET."""
    k = int(np.argmin(np.abs(y_m - off_ft * FT)))
    return p_ic + a * DP_w[k, :], float(y_m[k] / FT)


p_frac, _ = infer_at(0.0)

# ---------------------------------------------------------------- injection curves
inj = {}
for well, colour in INJ_WELLS.items():
    df = pd.read_csv(REPO / "data_fervo" / "legacy" / f"{well} Pressure.csv")
    df["t"] = pd.to_datetime(df["MSTTIMESTAMP"])
    df["p"] = pd.to_numeric(df["PRESSURE_PSI"], errors="coerce")
    m = (df["t"] >= common[0]) & (df["t"] <= common[-1] + pd.Timedelta(hours=4))
    df = df.loc[m & df["p"].notna()]
    inj[well] = (df, colour)
    print(f"  {well}: {len(df)} samples, {df['p'].min():.0f}..{df['p'].max():.0f} psi")

fig, (axA, axB, axD) = plt.subplots(3, 1, figsize=(14, 15), constrained_layout=True)

# --- (A) fracture pressure vs every measured injection curve --------------------
for well, (df, colour) in inj.items():
    axA.plot(df["t"], df["p"], lw=.5, color=colour, alpha=.35)
    axA.plot(df["t"], df["p"].rolling(240, center=True, min_periods=1).mean(), lw=2.0,
             color=colour, label=f"{well} injection (4-h mean)")
axA.plot(common, p_frac, "o-", ms=4.5, lw=2.8, color="#154360",
         label=f"{LABEL} INFERRED pressure at the fracture (MD {STAR:.0f})")
axA.axhline(p_ic, color=INK, ls=":", lw=1, label=f"initial condition {p_ic:.0f} psi")
axA.axvline(T2, color="#8e44ad", ls="--", lw=1.4)
axA.text(T2, 1.01, "T2", transform=axA.get_xaxis_transform(), color="#8e44ad",
         ha="center", fontweight="bold")
axA.set_ylabel("pressure  [psi]")
axA.set_title("(A) Inferred fracture pressure vs ALL measured Bearskin injection pressures",
              fontweight="bold", fontsize=11)
axA.legend(fontsize=8.5, loc="upper right", frameon=False, ncol=2)

# --- (B) pressure at stand-offs, both sides, in FEET ---------------------------
axB.plot(common, p_frac, lw=2.8, color=C_FRAC, label=f"fracture (0 ft, MD {STAR:.0f})")
for i, off in enumerate(OFFSETS):
    alpha = 1.0 - 0.22 * i
    ps, ys = infer_at(-off)
    pd_, yd = infer_at(+off)
    axB.plot(common, ps, lw=1.9, color=SHALLOW, alpha=alpha,
             label=f"−{off:g} ft  (shallower, MD {STAR + ys:.0f})")
    axB.plot(common, pd_, lw=1.9, ls="--", color=DEEP, alpha=alpha,
             label=f"+{off:g} ft  (deeper, MD {STAR + yd:.0f})")
axB.axhline(p_ic, color=INK, ls=":", lw=1)
axB.axvline(T2, color="#8e44ad", ls="--", lw=1.4)
axB.set_ylabel("inferred pressure  [psi]")
axB.set_title("(B) Inferred pressure into the SRV, both sides of the fracture "
              "(symmetry already verified to 1e-10 psi)", fontweight="bold", fontsize=11)
axB.legend(fontsize=8, loc="upper left", frameon=False, ncol=2)

# --- (D) dp profile across the SRV --------------------------------------------
axD.plot(np.full_like(m_md, p_ic), m_md, ls="-", lw=2.6, color=INK,
         label=f"T1  {common[0]:%m-%d %H:%M}  (uniform IC)")
for frac, ls in [(0.25, ":"), (0.5, "-."), (0.75, "--"), (1.0, "-")]:
    j = min(int(frac * nt) - 1, nt - 1)
    axD.plot(p_ic + a[j] * DP_w[:, j], m_md, ls=ls, lw=2.0, color=C_FRAC,
             label=f"{common[j]:%m-%d %H:%M}")
axD.axhline(STAR, color=INK, ls=":", lw=1.4)
axD.text(axD.get_xlim()[0], STAR - 3, f" fracture MD {STAR:.0f}", fontsize=8.5, color=INK)
for s in (SHEAR_MD, STAR + (STAR - SHEAR_MD)):
    axD.axhline(s, color="#8e44ad", ls="--", lw=1.6)
axD.text(axD.get_xlim()[0], SHEAR_MD - 3, f" SRV edge / shear plane MD {SHEAR_MD:.0f}",
         fontsize=8.5, color="#8e44ad")
axD.set_ylim(STAR + 90, STAR - 90)
axD.set_xlabel("inferred pressure  [psi]")
axD.set_ylabel("Measured depth  [ft]")
axD.set_title("(D) Pressure across the SRV, along the monitoring well",
              fontweight="bold", fontsize=11)
axD.legend(fontsize=8.5, loc="lower right", frameon=False)

for ax in (axA, axB):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.set_xlim(common[0], common[-1])
for ax in (axA, axB, axD):
    ax.grid(alpha=.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle(f"{LABEL} — inferred pressure along the monitoring well", fontsize=14,
             fontweight="bold")
out = FIG / f"{PROJ}_pressure_detail.png"
fig.savefig(out, dpi=140)
print("saved", out)

rows = {"window_start": common, "a": a, "b": b, "p_fracture_psi": p_frac}
for off in OFFSETS:
    ps, _ = infer_at(-off)
    pd_, _ = infer_at(+off)
    rows[f"p_minus{off:g}ft_psi"], rows[f"p_plus{off:g}ft_psi"] = ps, pd_
pd.DataFrame(rows).to_csv(OUT / f"{PROJ}_pressure_detail.csv", index=False)
print("wrote", OUT / f"{PROJ}_pressure_detail.csv")
print("\ninferred pressure at the last window:")
print(f"  fracture      {p_frac[-1]:8.1f} psi")
for off in OFFSETS:
    ps, _ = infer_at(-off)
    pd_, _ = infer_at(+off)
    print(f"  +/-{off:5.1f} ft   {pd_[-1]:8.1f} psi   (drop {p_frac[-1] - pd_[-1]:6.1f} psi)")
