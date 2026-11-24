"""Figures for the V11 pressure inversion (reads output/v11_pinv/results/)."""
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "output" / "v11_pinv" / "results"
FIG = REPO / "figs" / "tensile_fault_qc" / "v11_pressure_inversion"
FIG.mkdir(parents=True, exist_ok=True)
Z = np.load(RES / "inversion_results.npz")
S = json.loads((RES / "summary.json").read_text())
RUNS = S["runs"]
MD = Z["md"]
CEN = pd.to_datetime(Z["centers"])
T1, T2, T3 = map(pd.Timestamp, ["2025-02-24 12:00", "2025-02-28 00:00", "2025-03-06 00:00"])
KN = T1 + pd.to_timedelta(np.arange(115) * 7200, unit="s")
P0, K_T2, K_T3, PSI_MPA = S["P0_psi"], 42, 114, 0.00689476
C = {"inverted": "#c0392b", "baseline": "#1a6ea8", "handoff_ratefit": "#7f8c8d", "obs": "black", "shear": "#1a7f5a"}
LBL = {"inverted": "Inverted tensile (strain fit, primary)", "baseline": "Baseline tensile",
       "handoff_ratefit": "Inverted tensile (handoff, rate fit)"}
LS = ["-", "--", "-.", ":", (0, (1, 1)), (0, (5, 1, 1, 1))]
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":"})


def tmarks(ax, label=True):
    for t, s in [(T1, "T1"), (T2, "T2"), (T3, "T3")]:
        ax.axvline(t, color="#b36b00", ls="--", lw=1)
        if label:
            ax.text(t, 1.01, s, transform=ax.get_xaxis_transform(), ha="center", va="bottom", color="#996000")


def datefmt(ax):
    # AutoDateLocator: a fixed DayLocator explodes (>20k ticks) on shared date axes
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))


def save(fig, name):
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print("saved", name)


def heat(ax, data, vlim, title, cmap="seismic", label=""):
    im = ax.pcolormesh(CEN, MD, data, cmap=cmap, vmin=-vlim, vmax=vlim, shading="nearest", rasterized=True)
    ax.set_ylim(MD.max(), MD.min())
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("MD (ft)")
    for md, c in [(10365.107, "k"), (10317.837, "#8e44ad")]:
        ax.axhline(md, color=c, lw=0.8, ls=":")
    tmarks(ax, False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))   # DayLocator explodes on shared pcolormesh axes
    plt.colorbar(im, ax=ax, label=label, pad=0.01)


# --------------------------------------------------------------------------- 1 geometry
src = REPO / "figs" / "tensile_fault_qc" / "v11_srv_extended" / "v11_srv_extended_setup_geometry.png"
shutil.copy(src, FIG / "01_v11_geometry.png")

# --------------------------------------------------------------------------- 2 ramp response + dt convergence
tg, Rs = Z["base/tg"], Z["base/Rs"]
ipk = int(np.argmax(np.abs(Rs[:, -1])))
fig, ax = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
for tau_h in [2, 12, 24, 72, 228]:
    j = int(np.argmin(np.abs(tg - tau_h * 3600)))
    ax[0].plot(Rs[:, j] * 1000, MD, lw=2, label=f"{tau_h} h")
ax[0].set(ylim=(MD.max(), MD.min()), xlabel="Strain per 1000 psi (millistrain)", ylabel="MD (ft)",
          title="Unit ramp response along the fiber (10-m gauge)")
ax[0].axhline(10365.107, color="k", ls=":")
ax[0].legend(title="time after ramp")
for run, ls in zip(RUNS, LS):
    t, R = Z[f"{run}/tg"], Z[f"{run}/Rs"]
    ax[1].plot(t / 3600, R[ipk] * 1000, ls=ls, lw=2, label=run)
ax[1].set(xlabel="Hours after ramp start", ylabel="Strain per 1000 psi (millistrain)",
          title=f"Ramp response at MD {MD[ipk]:.0f} ft")
ax[1].legend()
if "fine" in RUNS:
    tb, Rb, tf, Rf = Z["base/tg"], Z["base/Rs"], Z["fine/tg"], Z["fine/Rs"]
    Rfi = np.array([np.interp(tb, tf, r) for r in Rf])
    err = np.linalg.norm(Rb - Rfi, axis=0) / np.maximum(np.linalg.norm(Rfi, axis=0), 1e-30)
    ax[2].semilogy(tb[1:] / 3600, err[1:], color="k")
    ax[2].set(xlabel="Hours after ramp start", ylabel="relative L2 difference (44 channels)",
              title="Time-step convergence: base vs fine")
else:
    ax[2].text(0.5, 0.5, "fine run not finished", ha="center", transform=ax[2].transAxes)
save(fig, "02_ramp_response_convergence.png")

# --------------------------------------------------------------------------- 3 GCV
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
for key in ["inverted", "baseline", "handoff_ratefit"]:
    sc = Z[f"base/{key}/scan"]
    lam = S[f"A/base/{key}"]["lam"]
    ax[0].loglog(sc[:, 0], sc[:, 1], color=C[key], label=LBL[key])
    ax[0].axvline(lam, color=C[key], ls=":")
    ax[1].loglog(sc[:, 3], sc[:, 2], ".-", color=C[key], ms=3)
    i = int(np.argmin(np.abs(sc[:, 0] - lam)))
    ax[1].plot(sc[i, 3], sc[i, 2], "o", color=C[key], ms=9, mfc="none", mew=2)
ax[0].set(xlabel="regularization weight lambda", ylabel="GCV", title="GCV choice of smoothing weight")
ax[0].legend(fontsize=9)
ax[1].set(xlabel="||second difference of dP|| (psi)", ylabel="strain RMSE (millistrain)", title="L-curve (circle = GCV choice)")
save(fig, "03_gcv_lcurve.png")

# --------------------------------------------------------------------------- 4 pressure, approach A
fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
for key in ["inverted", "baseline", "handoff_ratefit"]:
    dp, sg = Z[f"base/{key}/dp"], Z[f"base/{key}/dp_sigma"]
    lw = 2.5 if key == "inverted" else 1.6
    ax.plot(KN, P0 + dp, color=C[key], lw=lw, label=LBL[key])
    if key == "inverted":
        ax.fill_between(KN, P0 + dp - 2 * sg, P0 + dp + 2 * sg, color=C[key], alpha=0.15, label="formal ±2σ (white-noise)")
m = S["A/base/inverted"]
ax.annotate(f"ΔP(T1→T2) = {m['dP_T2_psi']:.0f} psi ({m['dP_T2_psi']*PSI_MPA:.2f} MPa)",
            xy=(T2, P0 + m["dP_T2_psi"]), xytext=(T2 - pd.Timedelta("2.6D"), P0 + m["dP_T3_psi"] * 0.8),
            arrowprops=dict(arrowstyle="->"), fontsize=11)
tmarks(ax)
datefmt(ax)
ax.set(ylabel="BC pressure on the monitoring well (psi)", xlabel="Date (fixed UTC−7, 2025)",
       title=f"Approach A: inverted pressure history (P(T1) = {P0:.0f} psi assumed)")
sec = ax.secondary_yaxis("right", functions=(lambda p: p * PSI_MPA, lambda m: m / PSI_MPA))
sec.set_ylabel("MPa")
ax.legend(loc="upper left")
save(fig, "04_pressure_approachA.png")

# --------------------------------------------------------------------------- 5 profiles target vs model
for key in ["inverted", "baseline"]:
    tgt_s, tgt_r = Z[f"target/{key}/strain"], Z[f"target/{key}/rate"]
    mod_s, mod_r = Z[f"base/{key}/model_strain"], Z[f"base/{key}/model_rate"]
    fig, axes = plt.subplots(2, 6, figsize=(18, 8.5), sharey=True, constrained_layout=True)
    picks = np.linspace(6, 113, 6).astype(int)
    for j, w in enumerate(picks):
        for i, (t, mo, u) in enumerate([(tgt_s, mod_s, "strain (millistrain)"), (tgt_r, mod_r, "rate (nanostrain/s)")]):
            a = axes[i, j]
            a.plot(t[:, w], MD, "k", lw=2, label="DDM tensile target")
            a.plot(mo[:, w], MD, color=C[key], lw=2, ls="--", label="MOOSE V11")
            a.set_xlabel(u)
            if i == 0:
                a.set_title(CEN[w].strftime("%m/%d %H:%M"))
    axes[0, 0].set_ylim(MD.max(), MD.min())
    axes[0, 0].set_ylabel("MD (ft)")
    axes[1, 0].set_ylabel("MD (ft)")
    axes[0, 0].legend(fontsize=9, loc="lower right")
    fig.suptitle(f"Approach A ({LBL[key]}): target vs modeled profiles (top strain = fitted, bottom rate = diagnostic)")
    save(fig, f"05_profiles_{key}.png")

    fig, axes = plt.subplots(3, 2, figsize=(18, 11), sharex=True, constrained_layout=True)
    vs, vr = np.nanmax(np.abs(tgt_s)), np.nanpercentile(np.abs(tgt_r), 99)
    for i, (d, nm) in enumerate([(tgt_s, "DDM tensile target"), (mod_s, "MOOSE V11 model"), (mod_s - tgt_s, "residual (model − target)")]):
        heat(axes[i, 0], d, vs if i < 2 else vs / 4, f"Strain: {nm}", label="millistrain")
    for i, (d, nm) in enumerate([(tgt_r, "DDM tensile target"), (mod_r, "MOOSE V11 model"), (mod_r - tgt_r, "residual (model − target)")]):
        heat(axes[i, 1], d, vr if i < 2 else vr / 2, f"Rate: {nm}", label="nanostrain/s")
    fig.suptitle(f"Approach A ({LBL[key]}): 2-hour waterfalls (residual color scale 4× / 2× tighter)")
    save(fig, f"06_waterfall_{key}.png")

# --------------------------------------------------------------------------- 7 pressure vs opening
dp = Z["base/inverted/dp"]
op = Z["open_knots"]
vo = S["A/base/inverted"]["vs_opening"]
fig, ax = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
a = ax[0, 0]
a.plot(KN, dp, color=C["inverted"], lw=2, label="ΔP (psi)")
b = a.twinx()
b.plot(KN, op, color="k", lw=2, ls="--", label="DDM tensile opening (mm)")
a.set_ylabel("ΔP from T1 (psi)", color=C["inverted"])
b.set_ylabel("opening (mm)")
b.grid(False)
tmarks(a)
datefmt(a)
a.set_title("Raw curves")
a.legend(loc="upper left")
b.legend(loc="lower right")
a = ax[0, 1]
a.plot(KN, dp / dp[K_T3], color=C["inverted"], lw=2, label="ΔP / ΔP(T3)")
a.plot(KN, op / op[K_T3], color="k", lw=2, ls="--", label="opening / opening(T3)")
tmarks(a)
datefmt(a)
a.set_title(f"Normalized to T3: shape RMSE = {vo['shape_rmse_normalized']:.3f}")
a.legend()
a = ax[1, 0]
sl = slice(0, K_T2 + 1)
a.plot(KN[sl], dp[sl] / dp[K_T2], color=C["inverted"], lw=2, label="ΔP / ΔP(T2)")
a.plot(KN[sl], op[sl] / op[K_T2], color="k", lw=2, ls="--", label="opening / opening(T2)")
a.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
a.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
a.set_title(f"T1–T2 zoom, normalized to T2: shape RMSE = {vo['shape_rmse_normalized_T1T2']:.3f}")
a.legend()
a = ax[1, 1]
sc = a.scatter(op, dp, c=(KN - T1).total_seconds() / 86400, cmap="viridis", s=18)
a.plot(op, vo["alpha_psi_per_mm"] * op + vo["beta_psi"], "k:", label=f"linear: {vo['alpha_psi_per_mm']:.0f} psi/mm, R²={vo['r2']:.3f}")
a.axvline(op[K_T2], color="#b36b00", ls="--", lw=1)
a.text(op[K_T2], a.get_ylim()[0], " T2", color="#996000", va="bottom")
plt.colorbar(sc, ax=a, label="days after T1")
a.set(xlabel="DDM tensile opening (mm)", ylabel="ΔP (psi)", title="ΔP vs opening")
a.legend(loc="upper left")
fig.suptitle("Inverted pressure (approach A, primary) vs DDM tensile opening (handoff knots)")
save(fig, "07_pressure_vs_opening.png")

# --------------------------------------------------------------------------- 8 approach B
fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
ax.plot(KN, P0 + Z["base/baseline/dp"], color=C["baseline"], lw=1.5, ls=":", label="baseline pressure (A, unscaled)")
for fit, ls in [("strain", "-"), ("rate", "--")]:
    bb = S[f"B/base/{fit}"]
    ax.plot(KN, Z[f"base/B_{fit}/pressure_psi"], color="#6c3483", ls=ls, lw=2.2,
            label=f"B, fitted to observed {fit}: a = {bb['a']:.3f}, b(shear) = {bb['b']:.3f}")
ax.plot(KN, P0 + Z["base/inverted/dp"], color=C["inverted"], lw=1.5, alpha=0.8, label="A, inverted tensile (for comparison)")
tmarks(ax)
datefmt(ax)
ax.set(ylabel="BC pressure (psi)", title="Approach B: scaled baseline pressure  P0 + a·ΔP_baseline(t)")
ax.legend(loc="upper left", fontsize=10)
save(fig, "08_pressure_approachB.png")

obs_s, obs_r = Z["obs_strain"], Z["obs_rate"]
bs, br = Z["base/B_strain/model_strain"], Z["base/B_rate/model_rate"]
fig, axes = plt.subplots(3, 2, figsize=(18, 11), sharex=True, constrained_layout=True)
vs, vr = np.nanmax(np.abs(obs_s)), np.nanpercentile(np.abs(obs_r), 99)
for i, (d, nm) in enumerate([(obs_s, "observed"), (bs, "B model (strain fit)"), (bs - obs_s, "residual")]):
    heat(axes[i, 0], d, vs if i < 2 else vs / 4, f"Strain: {nm}", label="millistrain")
for i, (d, nm) in enumerate([(obs_r, "observed"), (br, "B model (rate fit)"), (br - obs_r, "residual")]):
    heat(axes[i, 1], d, vr if i < 2 else vr / 2, f"Rate: {nm}", label="nanostrain/s")
fig.suptitle("Approach B: observed vs  a·MOOSE(baseline pressure) + b·DDM baseline shear")
save(fig, "09_waterfall_approachB.png")

base_tot_s = Z["target/baseline/strain"] + Z["target/baseline/shear_strain"]
inv_tot_s = Z["target/inverted/strain"] + Z["target/inverted/shear_strain"]
a_tot_s = Z["base/inverted/model_strain"] + Z["target/inverted/shear_strain"]
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, constrained_layout=True)
for a, md in zip(axes, [10317.837, 10365.107, 10420.0]):
    i = int(np.argmin(np.abs(MD - md)))
    a.plot(CEN, obs_s[i], "k", lw=2.2, label="observed")
    a.plot(CEN, bs[i], color="#6c3483", lw=2, label="B (strain fit)")
    a.plot(CEN, a_tot_s[i], color=C["inverted"], lw=1.6, ls="--", label="A inverted + DDM inv. shear")
    a.plot(CEN, base_tot_s[i], color=C["baseline"], lw=1.2, ls=":", label="DDM baseline total (nb 03)")
    a.plot(CEN, inv_tot_s[i], color="#7f8c8d", lw=1.2, ls="-.", label="DDM inverted total (nb 04)")
    a.set_title(f"MD {MD[i]:.1f} ft")
    tmarks(a, False)
    datefmt(a)
axes[0].set_ylabel("strain (millistrain)")
axes[0].legend(fontsize=9)
save(fig, "10_channel_timeseries.png")

# --------------------------------------------------------------------------- 11 sensitivity
fig, ax = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
for run, ls in zip(RUNS, LS):
    ax[0].plot(KN, Z[f"{run}/inverted/dp"], ls=ls, lw=2, label=f"{run}: ΔP(T2) = {S[f'A/{run}/inverted']['dP_T2_psi']:.0f} psi")
ax[0].plot(KN, Z["base/inverted/dp_unreg"], color="0.6", lw=0.8, label="base, no regularization")
tmarks(ax[0])
datefmt(ax[0])
ax[0].set(ylabel="ΔP from T1 (psi)", title="Approach A sensitivity (inverted tensile target)")
ax[0].legend(fontsize=9)
for run, ls in zip(RUNS, LS):
    ax[1].plot(KN, Z[f"{run}/inverted/dp"] / Z[f"{run}/inverted/dp"][K_T3], ls=ls, lw=2, label=run)
ax[1].plot(KN, op / op[K_T3], "k", lw=1.5, label="opening (normalized)")
tmarks(ax[1])
datefmt(ax[1])
ax[1].set(ylabel="normalized to T3", title="Normalized pressure shape vs DDM opening")
ax[1].legend(fontsize=9)
save(fig, "11_sensitivity.png")

# --------------------------------------------------------------------------- 12 forward validation
fv = RES / "forward_validation.npz"
if fv.exists():
    V = np.load(fv)
    fig, ax = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    i = int(np.argmax(np.abs(V["super_strain"][:, -1])))
    ax[0].plot(CEN, V["target_strain"][i], "k", lw=2, label="DDM tensile target")
    ax[0].plot(CEN, V["super_strain"][i], color=C["inverted"], lw=2, label="superposition (inversion)")
    ax[0].plot(CEN, V["direct_strain"][i], color="#f39c12", lw=1.5, ls="--", label="direct MOOSE run")
    ax[0].set(ylabel="strain (millistrain)", title=f"MD {MD[i]:.0f} ft")
    ax[0].legend()
    tmarks(ax[0], False)
    datefmt(ax[0])
    ax[1].plot(CEN, V["target_rate"][i], "k", lw=2, label="DDM tensile target")
    ax[1].plot(CEN, V["super_rate"][i], color=C["inverted"], lw=2, label="superposition")
    ax[1].plot(CEN, V["direct_rate"][i], color="#f39c12", lw=1.5, ls="--", label="direct MOOSE run")
    ax[1].set(ylabel="rate (nanostrain/s)", title=f"rel. diff strain {float(V['rel_diff_strain']):.2%}, rate {float(V['rel_diff_rate']):.2%}")
    ax[1].legend()
    tmarks(ax[1], False)
    datefmt(ax[1])
    fig.suptitle("Forward check: direct MOOSE run with the inverted pressure vs the superposition prediction")
    save(fig, "12_forward_validation.png")

# --------------------------------------------------------------------------- 13 spatial shape limit
fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True, constrained_layout=True)
tgt = Z["target/inverted/strain"]
for a, w, ttl in [(ax[0], K_T2 - 1, "last window before T2"), (ax[1], K_T3 - 1, "last window (T3)")]:
    a.plot(tgt[:, w] / tgt[:, w].max(), MD, "k", lw=2.5, label="DDM inverted tensile")
    for run, ls in zip(RUNS, LS):
        m = Z[f"{run}/inverted/model_strain"][:, w]
        a.plot(m / m.max(), MD, ls=ls, lw=1.8, label=f"V11 {run} ({S[f'A/{run}/inverted']['rel_rmse_strain']:.0%} misfit)")
    a.axvline(0, color="0.4", lw=0.8)
    a.set(xlabel="strain / peak", title=f"{ttl}: {CEN[w].strftime('%m/%d %H:%M')}")
ax[0].set_ylim(MD.max(), MD.min())
ax[0].set_ylabel("MD (ft)")
ax[0].legend(fontsize=9, loc="lower left")
fig.suptitle("Spatial shape: V11 compresses the flanks (negative), the DDM tensile target stays positive")
save(fig, "13_spatial_shape.png")
