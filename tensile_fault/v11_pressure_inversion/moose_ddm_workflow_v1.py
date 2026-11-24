"""Notebooks 07/08 (crossing_cutout, strain fit, 2 h) with the DDM tensile source replaced by MOOSE V11.

Everything else is Pengchao's workflow: same observed 2-h strain profiles, same time laws, same
DDM shear profiles (07 baseline / 08 inverted, used as supplied), same per-window scaling.

Baseline (07):  tensile opening law  A0*x^2 (T1-T2) + A1*y (T2-T3)  ->  BC pressure law
                dP_base(t) = c0*x^2 + c1*y ;  observed - DDM shear(07) ~ MOOSE(dP_base), c >= 0
Inverted (08):  per window j:  observed_j - DDM shear(08)_j ~ alpha_j * MOOSE_base_j, alpha_j >= 0
                dP_inv (window mean) = alpha_j * dP_base (window mean)   [08: slip = scale * base slip]
Check:          exact MOOSE superposition driven by dP_inv vs the scaled profiles.
Strain only; no rate fitting.  MOOSE response = one V11 ramp run (output/v11_pinv/base).
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).parent))
import invert_pressure as ip  # noqa: E402

REPO = ip.REPO
DDM = REPO / "Pengchao" / "Fervo_Small_Shear_Test" / "output" / "crossing_cutout" / "strain_fit_2h"
OUT = REPO / "output" / "v11_pinv" / "ddm_workflow"
FIG = REPO / "figs" / "tensile_fault_qc" / "moose_tensile_ddm_workflow"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
T1, T2, T3 = map(pd.Timestamp, ["2025-02-24 12:00", "2025-02-28 00:00", "2025-03-06 00:00"])
EDGES = pd.date_range(T1, T3, freq="2h")
CENTERS = EDGES[:-1] + pd.Timedelta("1h")
P0, PSI_MPA = ip.P0_PSI, ip.PSI_MPA
plt.rcParams.update({"font.size": 15, "axes.titlesize": 18, "axes.labelsize": 15})


def pivot(name, col="strain_millistrain"):
    return pd.read_csv(DDM / name).pivot(index="md_ft", columns="center_time", values=col).to_numpy(float)


# ----------------------------------------------------------------------------- data (07/08 outputs)
base_npz = np.load(DDM / "baseline_profiles_2h.npz")
MD = base_npz["depth_ft"]
OBS = base_npz["observed_profiles"][1]                                   # [44, 114] millistrain
SHEAR = {"baseline": pivot("baseline_shear_profiles_2h.csv"), "inverted": pivot("inverted_shear_profiles_2h.csv")}
DDM_T = {"baseline": pivot("baseline_tensile_profiles_2h.csv"), "inverted": pivot("inverted_tensile_profiles_2h.csv")}
SLIP = {k: {c: pd.read_csv(DDM / f"{k}_{c}_slip_2h.csv")["mean_absolute_slip_mm"].to_numpy() for c in ["tensile", "shear"]}
        for k in ["baseline", "inverted"]}
DDM_RMSE = {k: pd.read_csv(DDM / f"{k}_metrics.csv").set_index("quantity").loc["strain", "rmse"] for k in ["baseline", "inverted"]}
assert np.allclose(MD, ip.MD)

# ----------------------------------------------------------------------------- MOOSE response
Gs = ip.basis("base")[0]                                                 # [window, channel, knot 1..114], per psi
tk = (EDGES - T1).total_seconds().to_numpy()
x = np.clip(tk / (T2 - T1).total_seconds(), 0, 1)
y = np.clip((tk - (T2 - T1).total_seconds()) / (T3 - T2).total_seconds(), 0, 1)
LAWS = np.stack([x ** 2, y])                                             # pressure laws at the 115 knots
M = np.stack([(Gs @ law[1:]).T for law in LAWS])                       # [2, 44, 114] millistrain per psi
valid = np.ones_like(OBS, bool)
valid[:, 0] = False                                                      # first window = reference (as 07)


def rmse(model, target=OBS):
    return float(np.sqrt(np.mean((model - target)[valid] ** 2)))


# ----------------------------------------------------------------------------- baseline (07 analog)
resid = OBS - SHEAR["baseline"]
A = np.column_stack([m[valid] for m in M])
c, _ = nnls(A, resid[valid])
dp_base = c @ LAWS                                                       # psi at the knots
t_base = np.tensordot(c, M, 1)                                           # MOOSE tensile strain [44, 114]
dp_base_mean = 0.5 * (dp_base[:-1] + dp_base[1:])                        # window means (piecewise linear)

# ----------------------------------------------------------------------------- inverted (08 analog)
alpha = np.full(114, np.nan)
for j in range(1, 114):
    m, r = t_base[:, j], OBS[:, j] - SHEAR["inverted"][:, j]
    alpha[j] = max(0.0, float(m @ r) / float(m @ m))
dp_inv_mean = alpha * dp_base_mean
dp_inv_mean[0] = dp_base_mean[0]                                         # reference window retained (as 08)
t_inv = alpha[None, :] * t_base
t_inv[:, 0] = t_base[:, 0]
# exact MOOSE response to the inverted pressure (knots from window means, 0 at T1)
dp_inv_knots = np.r_[0.0, np.interp(tk[1:], (CENTERS - T1).total_seconds().to_numpy(), dp_inv_mean)]
t_inv_exact = (Gs @ dp_inv_knots[1:]).T

MODELS = {"baseline": dict(dp=dp_base_mean, tensile=t_base, shear=SHEAR["baseline"]),
          "inverted": dict(dp=dp_inv_mean, tensile=t_inv, shear=SHEAR["inverted"])}
summary = {"c_psi": dict(T1_T2_x2=float(c[0]), T2_T3_linear=float(c[1])), "P0_psi": P0,
           "dP_base_T2_psi": float(dp_base[42]), "dP_base_T3_psi": float(dp_base[114])}
for k, m in MODELS.items():
    total = m["tensile"] + m["shear"]
    summary[k] = dict(rmse_strain=rmse(total), rel_rmse=rmse(total) / float(np.sqrt(np.mean(OBS[valid] ** 2))),
                      ddm_rmse_strain=float(DDM_RMSE[k]), rmse_vs_ddm_tensile=rmse(m["tensile"], DDM_T[k]),
                      dP_last_window_before_T2_psi=float(m["dp"][41]), dP_last_window_psi=float(m["dp"][113]))
summary["inverted"]["exact_vs_scaled_rel_diff"] = float(np.linalg.norm(t_inv_exact - t_inv) / np.linalg.norm(t_inv))
summary["inverted"]["rmse_strain_exact_forward"] = rmse(t_inv_exact + SHEAR["inverted"])
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
pd.DataFrame({"center_time_utc_minus_7": CENTERS.strftime("%Y-%m-%d %H:%M:%S"),
              "elapsed_s_from_T1": (CENTERS - T1).total_seconds(),
              "baseline_pressure_psi": P0 + dp_base_mean, "baseline_dP_psi": dp_base_mean,
              "inverted_scale_alpha": alpha, "inverted_pressure_psi": P0 + dp_inv_mean, "inverted_dP_psi": dp_inv_mean,
              "inverted_pressure_MPa": (P0 + dp_inv_mean) * PSI_MPA,
              "ddm_baseline_tensile_mm": SLIP["baseline"]["tensile"], "ddm_inverted_tensile_mm": SLIP["inverted"]["tensile"],
              }).to_csv(OUT / "moose_pressure_2h.csv", index=False)
for k, m in MODELS.items():
    pd.DataFrame({"center_time": np.repeat(CENTERS.strftime("%Y-%m-%d %H:%M:%S"), len(MD)), "md_ft": np.tile(MD, 114),
                  "observed_strain_millistrain": OBS.T.ravel(), "moose_tensile_millistrain": m["tensile"].T.ravel(),
                  "ddm_shear_millistrain": m["shear"].T.ravel(),
                  "residual_millistrain": (m["tensile"] + m["shear"] - OBS).T.ravel()}).to_csv(OUT / f"{k}_profiles_2h.csv", index=False)
print(json.dumps(summary, indent=2))


# ----------------------------------------------------------------------------- figures (notebook layout)
def tmarks(ax, label=False):
    for t, s in [(T1, "T1"), (T2, "T2"), (T3, "T3")]:
        ax.axvline(t, color="#B36B00", ls="--", lw=1.2)
        if label:
            ax.text(t, 1.012, s, transform=ax.get_xaxis_transform(), color="#996000", fontsize=13, ha="center", va="bottom")


def injection_panel(ax):
    inj = pd.read_csv(DDM / "injection_pressure.csv", parse_dates=["time"])
    for well, f in inj.groupby("well", sort=False):
        ax.plot(f.time, f.pressure_psi, lw=1.1, color=f.color.iloc[0], label=well)
    st = inj[inj.well == inj.well.iloc[0]].set_index("time").stage_number
    runs = [(p.index[0], p.index[-1], int(p.iloc[0])) for _, p in st.groupby(st.ne(st.shift()).cumsum()) if p.notna().any()]
    top = inj.pressure_psi.max()
    for i, (lo, hi, n) in enumerate(runs):
        ax.text(lo + (hi - lo) / 2, top * (1.15 + .17 * (i % 2)), str(n), ha="center", va="center", fontsize=12)
    ax.set(ylim=(0, top * 1.45), ylabel="Injection\npressure (psi)")
    ax.set_title("Injection pressure | recorded stage numbers above curves", pad=34)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=13)
    tmarks(ax)


def pressure_panel(ax, key):
    m = MODELS[key]
    ax.plot(CENTERS, P0 + m["dp"], color="royalblue", lw=2.4, label="MOOSE tensile: BC pressure")
    ax.set_ylabel("MOOSE pressure (psi)", color="royalblue")
    tw = ax.twinx()
    tw.plot(CENTERS, SLIP[key]["shear"], color="firebrick", lw=2, label="|Shear| (DDM)")
    tw.set_ylabel("|Shear slip| (mm)", color="firebrick")
    ax.set_title("MOOSE tensile pressure and DDM shear slip history", pad=30)
    ax.legend(handles=[Line2D([], [], color="royalblue", lw=2, label="Tensile pressure (MOOSE)"),
                       Line2D([], [], color="firebrick", lw=2, label="|Shear| slip (DDM)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=13)
    tmarks(ax)


def waterfall(ax, background, sim):
    cmap = plt.get_cmap("seismic")
    md_edges = np.r_[MD[0] - (MD[1] - MD[0]) / 2, (MD[1:] + MD[:-1]) / 2, MD[-1] + (MD[-1] - MD[-2]) / 2]
    im = ax.pcolormesh(EDGES, md_edges, background, cmap=cmap, vmin=-.15, vmax=.15, shading="flat", rasterized=True)
    scale = max(np.nanpercentile(np.abs(np.r_[OBS.ravel(), sim.ravel()]), 98), 1e-12)
    w = .0275 / scale
    xc = mdates.date2num(CENTERS)
    for j, a in enumerate(xc):
        ax.plot(a + OBS[:, j] * w, MD, color="black", lw=1.1, zorder=5,
                path_effects=[pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])
        ax.plot(a + sim[:, j] * w, MD, color="#00852B", ls="--", lw=1.6, zorder=6,
                path_effects=[pe.Stroke(linewidth=2.8, foreground="white"), pe.Normal()])
    for md, col, lbl, xa in [(10365.107, "#B36B00", "F1 (tensile, MOOSE)", .13), (10317.837, "teal", "F2 (shear, DDM)", .56)]:
        ax.axhline(md, color=col, ls="--", lw=1.1)
        ax.annotate(f"{lbl}: {md:.1f} ft", xy=(xa, 1), xycoords="axes fraction", xytext=(0, 6),
                    textcoords="offset points", fontsize=13, color=col)
    ax.set(ylim=(10500, 10200), ylabel="Measured depth (ft)")
    tmarks(ax, True)
    return im


def summary_figure(key):
    m = MODELS[key]
    sim = m["tensile"] + m["shear"]
    fig = plt.figure(figsize=(24, 19))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.1, 1, 2.2, 2.2], width_ratios=[1, .015],
                          left=.075, right=.9, bottom=.06, top=.9, hspace=.55, wspace=.05)
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[i, 0], sharex=axes[0]) for i in range(1, 4)]
    injection_panel(axes[0])
    pressure_panel(axes[1], key)
    for row, bg, name in [(2, OBS, "Observed"), (3, sim, "Simulated (MOOSE tensile + DDM shear)")]:
        im = waterfall(axes[row], bg, sim)
        axes[row].set_title(f"{name} strain waterfall | 2-hour mean profiles", pad=30)
        fig.colorbar(im, cax=fig.add_subplot(gs[row, 1]), label="Strain (millistrain)")
    fig.legend(handles=[Line2D([], [], color="black", lw=1.4, label="Observed profile"),
                        Line2D([], [], color="#00852B", ls="--", lw=2, label="Simulated profile")],
               loc="upper center", bbox_to_anchor=(.5, .985), ncol=2, frameon=False)
    fig.suptitle(f"{key.capitalize()}: MOOSE V11 tensile (pressure) + DDM small shear, strain fit "
                 f"(RMSE {summary[key]['rmse_strain']:.4f} millistrain; all-DDM {summary[key]['ddm_rmse_strain']:.4f})",
                 y=.955, fontsize=19)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%y\n%H:%M"))
    axes[-1].set(xlim=(T1, T3), xlabel="Time (fixed UTC−7)")
    fig.savefig(FIG / f"{key}_summary.png", dpi=130)
    plt.close(fig)


for key in MODELS:
    summary_figure(key)

# profile comparison at six times
fig, axes = plt.subplots(2, 6, figsize=(24, 13), sharey=True, constrained_layout=True)
for row, key in enumerate(MODELS):
    m = MODELS[key]
    for col, j in enumerate([20, 41, 60, 80, 100, 113]):
        a = axes[row, col]
        a.plot(OBS[:, j], MD, "k", lw=2.5, label="Observed")
        a.plot(m["tensile"][:, j] + m["shear"][:, j], MD, color="#00852B", ls="--", lw=2.5, label="MOOSE tensile + DDM shear")
        a.plot(m["tensile"][:, j], MD, color="royalblue", lw=1.5, label="MOOSE tensile")
        a.plot(DDM_T[key][:, j], MD, color="royalblue", lw=1.2, ls=":", label="DDM tensile (07/08)")
        a.plot(m["shear"][:, j], MD, color="firebrick", lw=1.5, label="DDM shear")
        a.axvline(0, color="0.5", lw=.8)
        a.set_title(f"{key} | {CENTERS[j]:%m/%d %H:%M}", fontsize=14)
        a.set_xlabel("strain (millistrain)")
axes[0, 0].set_ylim(10500, 10200)
axes[0, 0].set_ylabel("Measured depth (ft)")
axes[1, 0].set_ylabel("Measured depth (ft)")
axes[0, 0].legend(fontsize=11, loc="lower right")
fig.savefig(FIG / "profiles_fitted_vs_observed.png", dpi=110)
plt.close(fig)

# pressure history vs DDM tensile opening
fig, ax = plt.subplots(2, 1, figsize=(18, 11), sharex=True, constrained_layout=True)
ax[0].plot(CENTERS, P0 + dp_base_mean, color="0.4", lw=2, ls="--", label="Baseline pressure (07 time law)")
ax[0].plot(CENTERS, P0 + dp_inv_mean, color="royalblue", lw=2.5, label="Inverted pressure (08 per-window scaling)")
ax[0].set_ylabel("MOOSE BC pressure (psi)")
sec = ax[0].secondary_yaxis("right", functions=(lambda p: p * PSI_MPA, lambda q: q / PSI_MPA))
sec.set_ylabel("MPa")
ax[0].legend(loc="upper left")
ax[0].set_title("MOOSE tensile pressure (P(T1) = 4056 psi assumed)")
for k, ls in [("baseline", "--"), ("inverted", "-")]:
    dp = MODELS[k]["dp"]
    ax[1].plot(CENTERS, dp / dp[113], color="royalblue", ls=ls, lw=2, label=f"MOOSE ΔP, {k} (normalized)")
    ax[1].plot(CENTERS, SLIP[k]["tensile"] / SLIP[k]["tensile"][113], color="k", ls=ls, lw=1.6, label=f"DDM tensile opening, {k} (normalized)")
ax[1].set_ylabel("normalized to last window")
ax[1].legend(loc="upper left", fontsize=12)
ax[1].set_title("Pattern comparison: MOOSE pressure vs DDM tensile opening")
for a in ax:
    tmarks(a, True)
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
fig.savefig(FIG / "pressure_vs_ddm_opening.png", dpi=120)
plt.close(fig)

# scaled vs exact MOOSE response for the inverted pressure
fig, ax = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
i = int(np.argmin(np.abs(MD - 10366.7)))
ax[0].plot(CENTERS, OBS[i] - SHEAR["inverted"][i], "k", lw=2, label="observed − DDM shear (target)")
ax[0].plot(CENTERS, t_inv[i], color="royalblue", lw=2, label="α × MOOSE baseline (08-style scaling)")
ax[0].plot(CENTERS, t_inv_exact[i], color="#f39c12", lw=2, ls="--", label="exact MOOSE response to inverted pressure")
ax[0].set(ylabel="strain (millistrain)", title=f"MD {MD[i]:.1f} ft")
ax[0].legend(fontsize=12)
ax[1].plot(CENTERS, alpha, color="royalblue", lw=2)
ax[1].axhline(1, color="0.5", lw=1)
ax[1].set(ylabel="α (inverted / baseline)", title="Per-window scale α")
for a in ax:
    tmarks(a, True)
    a.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
fig.savefig(FIG / "scaling_check.png", dpi=120)
plt.close(fig)
print("figures ->", FIG)
