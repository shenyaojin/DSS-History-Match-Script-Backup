"""Paper figures from completed runs only; no simulation imports or subprocesses.

Run with the existing moose Python environment. All input paths are fixed below.
Creates figures, derived data, and provenance. Companion Markdown files contain
the English captions and Chinese interpretation guide.
"""
from pathlib import Path
import hashlib
import json
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "scripts/DSS_history_match/optimizer_input_file_test"
V2 = BASE / "perm5layer_100_v2strain"
V4 = BASE / "perm5layer_100_v4strain_rfsdss"
V5 = BASE / "perm5layer_100_v5strain_zonal2p"
FIG = HERE / "figures"
DATA = HERE / "data"
for folder in (FIG, DATA):
    folder.mkdir(parents=True, exist_ok=True)

BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#777777"
GREEN = "#009E73"
YEDGE = np.linspace(-50, 50, 201)
YCENTER = (YEDGE[:-1] + YEDGE[1:]) / 2
FREE = (YCENTER >= -25) & (YCENTER <= 25)
LOW = (YCENTER >= -20) & (YCENTER < -16)
HIGH = (YCENTER >= 14) & (YCENTER < 20)
TRUTH = np.full(200, -18.)
TRUTH[LOW] = -15.
TRUTH[HIGH] = np.log10(3e-15)
ALABEL = r"$\log_{10}(k_{xx}\,/\,\mathrm{m}^2)$"
source_files = {}
audits = []
metrics = []
figmeta = {}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": .6, "lines.linewidth": 1.4,
    "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "mathtext.fontset": "dejavusans", "axes.unicode_minus": True,
})


def source(path, role):
    path = Path(path)
    key = str(path.relative_to(ROOT))
    if key not in source_files:
        source_files[key] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "bytes": path.stat().st_size, "roles": []}
    if role not in source_files[key]["roles"]:
        source_files[key]["roles"].append(role)
    return path


def alpha(path, role):
    a = np.loadtxt(source(path, role))
    assert a.shape == (200,) and np.isfinite(a).all(), path
    return a


def frame(path, role):
    df = pd.read_csv(source(path, role))
    keys = ["measurement_ycoord", "measurement_time"]
    assert not df.duplicated(keys).any(), path
    df = df.sort_values(keys).reset_index(drop=True)
    assert len(df) == 65000 and df.measurement_time.nunique() == 130
    assert df.measurement_ycoord.nunique() == 500
    assert np.isfinite(df.select_dtypes("number").to_numpy()).all()
    return df


clean = frame(V2 / "inv/measurement_data.csv", "clean synthetic observation")
clean_grid = clean.measurement_values.to_numpy().reshape(500, 130)
REF = float(np.median(np.max(np.abs(clean_grid), axis=1)))
POS = clean.measurement_time.to_numpy() > 0


def aligned(df):
    for c in ["measurement_ycoord", "measurement_time", "measurement_xcoord"]:
        np.testing.assert_allclose(df[c], clean[c], rtol=0, atol=1e-10)


def model_rmse(a):
    return float(np.sqrt(np.mean((a[FREE] - TRUTH[FREE])**2)))


def case(folder, name, zonal=False, qc=False):
    a = alpha(folder / "optimized_alphas_L1.txt", name + ": final alpha")
    obs = frame(folder / "measurement_data.csv", name + ": observations")
    aligned(obs)
    hname = "objective_history_zonal.csv" if zonal else "objective_history_L1.csv"
    hist = pd.read_csv(source(folder / hname, name + ": objective evaluations"))
    last_a = pd.read_csv(source(folder / "parameter_history_L1.csv", name + ": evaluated alpha"),
                         header=None).iloc[-1].to_numpy()
    np.testing.assert_allclose(a, last_a, atol=6e-9, rtol=0)
    row = {"case": name, "parameterization": "zonal" if zonal else "layerwise",
           "J_data": float(hist.obj_raw.iloc[-1]), "model_rmse_dex": model_rmse(a),
           "low_zone_mean_alpha": float(a[LOW].mean()),
           "high_zone_mean_alpha": float(a[HIGH].mean()),
           "noise_rms_microstrain": float(np.sqrt(np.mean((obs.measurement_values.to_numpy()[POS]
                                                           - clean.measurement_values.to_numpy()[POS])**2))*1e6),
           "reference_strain": REF, "n_objective_evaluations": len(hist)}
    row["relative_noise_rms_pct"] = row["noise_rms_microstrain"] * 1e-6 / REF * 100
    df = None
    if zonal:
        theta = np.loadtxt(source(folder / "optimized_theta_zonal.txt", name + ": final theta"))
        row.update(theta_high=float(theta[0]), theta_low=float(theta[1]))
    if qc:
        tag = "zonal_final" if zonal else "L1_final"
        qa = alpha(folder / f"qc_alpha_{tag}.txt", name + ": actual QC alpha")
        np.testing.assert_allclose(qa, a, atol=6e-9, rtol=0)
        df = frame(folder / f"qc_output_{tag}/qc_opt_{tag}_out_forward0_csv_data_0129.csv",
                   name + ": complete final forward response")
        aligned(df)
        np.testing.assert_allclose(df.measurement_values, obs.measurement_values, atol=1e-16, rtol=1e-10)
        assert np.max(np.abs(df.simulation_values)) > 0, name
        residual = df.simulation_values.to_numpy() - df.measurement_values.to_numpy()
        j = float(.5 * np.sum(residual[POS]**2))
        np.testing.assert_allclose(j, row["J_data"], rtol=2e-7, atol=1e-20)
        np.testing.assert_allclose(residual[POS], df.misfit_values.to_numpy()[POS], rtol=1e-7, atol=1e-16)
        row.update(J_data_recomputed=j, residual_rms_microstrain=float(np.sqrt(np.mean(residual[POS]**2))*1e6))
        audits.append({"case": name, "qc_alpha_max_abs_difference": float(np.max(np.abs(qa-a))),
                       "objective_relative_difference": abs(j-row["J_data"])/row["J_data"],
                       "n_samples": int(POS.sum()), "all_zero_response": False})
        df.to_csv(DATA / f"{name}_forward.csv.gz", index=False, compression="gzip")
    metrics.append(row)
    return {"alpha": a, "obs": obs, "forward": df, "metrics": row}


clean_case = case(V2 / "inv", "clean_layerwise", qc=True)
clean_zonal = case(V5 / "inv/exact_clean", "clean_zonal", zonal=True)
initial = alpha(V2 / "inv/initial_alpha_L1.txt", "clean initial model")
layer_cases, zone_cases = {}, {}
for tag in ["1pct", "2pct", "5pct", "10pct"]:
    layer_cases[tag] = case(V4 / f"inv/rfsdss_{tag}", "layerwise_" + tag, qc=(tag == "10pct"))
    zone_cases[tag] = case(V5 / f"inv/exact_rfsdss_{tag}", "zonal_" + tag, zonal=True, qc=(tag == "10pct"))
    np.testing.assert_array_equal(layer_cases[tag]["obs"].measurement_values,
                                  zone_cases[tag]["obs"].measurement_values)
    li = alpha(V4 / f"inv/rfsdss_{tag}/initial_alpha_L1.txt", f"{tag}: layerwise initial model")
    zi = np.loadtxt(source(V5 / f"inv/exact_rfsdss_{tag}/initial_theta_zonal.txt", f"{tag}: zonal initial model"))
    expanded = np.full(200, -18.)
    expanded[HIGH], expanded[LOW] = zi
    np.testing.assert_allclose(li, expanded, rtol=0, atol=1e-10)
pert = case(V5 / "inv/pert_rfsdss_10pct", "shifted_zonal_10pct", zonal=True, qc=True)
np.testing.assert_array_equal(pert["obs"].measurement_values, zone_cases["10pct"]["obs"].measurement_values)
pd.DataFrame(metrics).to_csv(DATA / "case_metrics.csv", index=False)
pd.DataFrame({"y_bottom_m": YEDGE[:-1], "y_top_m": YEDGE[1:], "y_center_m": YCENTER,
              "truth_alpha": TRUTH, "initial_alpha": initial,
              "clean_final_alpha": clean_case["alpha"],
              "layerwise_10pct_alpha": layer_cases["10pct"]["alpha"],
              "zonal_10pct_alpha": zone_cases["10pct"]["alpha"],
              "shifted_zonal_10pct_alpha": pert["alpha"]}).to_csv(DATA / "alpha_profiles.csv", index=False)


def panel(ax, letter, title):
    ax.set_title(f"({letter})  {title}", loc="left", pad=9, fontweight="medium")


def style(ax, grid=False):
    ax.tick_params(direction="out")
    if grid:
        ax.grid(axis="both", color="#dddddd", lw=.45, zorder=0)
    ax.set_axisbelow(True)


def save(fig, stem, width_mm, height_mm, metadata=None):
    fig.set_size_inches(width_mm/25.4, height_mm/25.4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    check_texts = list(fig.texts)
    for ax in fig.axes:
        check_texts += [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label]
        check_texts += list(ax.texts)
        if ax.get_legend() is not None:
            check_texts += ax.get_legend().get_texts()
    for text in check_texts:
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= width+1 and box.y1 <= height+1, (stem, text.get_text(), box.bounds)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(FIG / f"{stem}.{ext}", dpi=600)
    fig.savefig(FIG / f"{stem}_preview.png", dpi=170)
    figmeta[stem] = {"width_mm": width_mm, "height_mm": height_mm, **(metadata or {})}
    plt.close(fig)


def profile(ax, profiles, full=False):
    for values, label, color, ls, lw in profiles:
        ax.stairs(values, YEDGE, orientation="horizontal", baseline=None,
                  label=label, color=color, linestyle=ls, linewidth=lw)
    ax.set_ylim((-50, 50) if full else (-25, 25))
    ax.set_xlabel(ALABEL)
    ax.set_ylabel(r"Vertical coordinate $y$ (m)")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    style(ax, grid=True)


def grid_values(df, column):
    p = df.pivot(index="measurement_ycoord", columns="measurement_time", values=column)
    return p.columns.to_numpy()/3600, p.index.to_numpy(), p.to_numpy()*1e6


def heat(ax, df, kind, limit, title, letter, clip=False, ylabel=True):
    d = df.copy()
    if kind == "residual":
        d["plot_value"] = d.simulation_values - d.measurement_values
    else:
        d["plot_value"] = d[kind]
    # Baseline frame is retained in Fig. 2 observations/predictions; residual maps
    # omit t=0 consistently with the objective. No temporal resampling is used.
    if kind == "residual":
        d = d.loc[d.measurement_time > 0]
    t, y, z = grid_values(d, "plot_value")
    im = ax.pcolormesh(t, y, z, cmap="RdBu_r", vmin=-limit, vmax=limit,
                       shading="nearest", rasterized=True)
    ax.set_xlim(0, clean.measurement_time.max()/3600)
    ax.set_ylim(-25, 25)
    ax.set_xticks([0, 24, 48, 72, 96])
    ax.set_xlabel("Time (h)")
    if ylabel:
        ax.set_ylabel(r"Vertical coordinate $y$ (m)")
    panel(ax, letter, title)
    style(ax)
    return im, float(np.mean(np.abs(z) > limit)*100)


def figure1():
    raw = source(DATA / "adjoint_console_record.txt", "completed discrete validation: rounded console transcription").read_text()
    rows = re.findall(r"^(1e-[0-9]+)\s+(-[0-9.eE+-]+)\s+([0-9.eE+-]+)$", raw, re.M)
    fd = pd.DataFrame(rows, columns=["epsilon", "central_fd", "relative_error"]).astype(float)
    fd.to_csv(DATA / "discrete_adjoint_verification.csv", index=False)
    logfile = source(V2 / "inv/104_gradient_fd_check_eps01.stdout", "completed MOOSE finite-difference verification")
    text = logfile.read_text(errors="replace")
    records = []
    for match in re.finditer(r"^\s*(65|135)\s+(-[\d.eE+-]+)\s+(-[\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$", text, re.M):
        idx, f, a, ratio, err = match.groups()
        records.append({"zero_based_index": int(idx), "finite_difference": float(f),
                        "adjoint": float(a), "relative_error": float(err), "epsilon": .1})
    moose = pd.DataFrame(records)
    assert len(moose) == 2
    moose.to_csv(DATA / "moose_gradient_check.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(7.09, 2.91))
    fig.subplots_adjust(left=.095, right=.985, bottom=.22, top=.82, wspace=.40)
    ax = axs[0]
    ax.loglog(fd.epsilon, fd.relative_error, "o-", color=BLUE, ms=4, label="Central FD vs. adjoint")
    eps = np.array([1e-4, 1e-1])
    ax.loglog(eps, 1.18*eps**2, "--", color=GRAY, lw=1, label=r"$O(h^2)$ reference")
    ax.set_xlabel(r"Perturbation amplitude $h$")
    ax.set_ylabel("Relative derivative error")
    ax.set_ylim(5e-10, .05)
    ax.set_xlim(6e-7, .16)
    ax.text(.04, .96, "Dot-product test: max. error\n" + r"$5.4\times10^{-14}$ (5 directions)",
            transform=ax.transAxes, va="top", fontsize=7)
    ax.legend(loc="lower right", frameon=False, fontsize=6.5)
    panel(ax, "a", "Discrete adjoint verification")
    style(ax, True)
    ax = axs[1]
    for j, r in moose.iterrows():
        ax.plot([r.finite_difference*1e8, r.adjoint*1e8], [j, j], color=GRAY, lw=1)
        ax.text(-3.35, j, f"{r.relative_error*100:.2f}%", va="center", fontsize=7)
    ax.scatter(moose.finite_difference*1e8, [0, 1], color=BLUE, s=27, marker="o", label="Central FD", zorder=3)
    ax.scatter(moose.adjoint*1e8, [0, 1], color=ORANGE, s=30, marker="x", label="Adjoint", zorder=4)
    ax.set_yticks([0, 1], ["Index 65", "Index 135"])
    ax.set_ylim(-.55, 1.55)
    ax.set_xlim(-4.35, -3.05)
    ax.set_xlabel(r"$\partial J_{\mathrm{data}}/\partial\alpha_i$ ($\times10^{-8}$)")
    ax.text(.97, .95, r"$h=0.1$ in $\alpha=\log_{10} k$", ha="right", va="top", transform=ax.transAxes, fontsize=7)
    ax.legend(frameon=False, loc="lower right", ncol=2, handletextpad=.4, columnspacing=.8)
    panel(ax, "b", "MOOSE gradient check")
    style(ax, True)
    save(fig, "fig01_adjoint_verification", 180, 78,
         {"discrete_record_precision": "printed values; console transcription", "moose_indices": "zero-based"})


def figure2():
    fig = plt.figure(figsize=(7.09, 5.7))
    gs = fig.add_gridspec(2, 3, left=.085, right=.98, bottom=.10, top=.94,
                          wspace=.55, hspace=.67, height_ratios=[1, 1.05])
    ax = fig.add_subplot(gs[0, :2])
    ax.add_patch(Rectangle((0, -50), 100, 100, facecolor="#f4f4f4", edgecolor="#444444", lw=.7))
    for lo, hi, color in [(-20, -16, "#b8dce7"), (14, 20, "#f2c3a4")]:
        ax.add_patch(Rectangle((0, lo), 100, hi-lo, facecolor=color, edgecolor="none"))
    ax.plot([45, 45], [-50, 50], color=ORANGE, lw=2)
    ax.plot([60, 60], [-25, 25], color=BLUE, lw=2)
    ax.annotate("Pressure boundary\n" + r"$x=45$ m", xy=(45, 37), xytext=(4, 34),
                arrowprops={"arrowstyle": "-", "lw": .6}, fontsize=7, va="center")
    ax.annotate("DSS observation line\n" + r"$x=60$ m; 500 channels", xy=(60, 5), xytext=(64, -8),
                arrowprops={"arrowstyle": "-", "lw": .6}, fontsize=7, va="center")
    ax.text(3, -35, r"Background: $10^{-18}$ m$^2$", fontsize=7)
    ax.text(3, -13, r"$10^{-15}$ m$^2$; 4 m thick", fontsize=7)
    ax.text(3, 23, r"$3\times10^{-15}$ m$^2$; 6 m thick", fontsize=7)
    ax.set_xlim(0, 100)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("Horizontal coordinate x (m)")
    ax.set_ylabel("Vertical coordinate y (m)")
    panel(ax, "a", "Model and acquisition (schematic)")
    ax = fig.add_subplot(gs[0, 2])
    profile(ax, [(TRUTH, "Truth", "black", "-", 1.7),
                 (initial, "Initial", GRAY, "--", 1.1),
                 (clean_case["alpha"], "Recovered", BLUE, "-", 1.2)])
    ax.set_xlim(-18.3, -14.1)
    ax.legend(frameon=False, loc="center left", fontsize=6.5)
    panel(ax, "b", "Permeability recovery")
    df = clean_case["forward"]
    limit = float(max(abs(df.measurement_values).max(), abs(df.simulation_values).max())*1e6)
    residual_limit = float(np.max(np.abs((df.simulation_values-df.measurement_values).to_numpy()[POS]))*1e6)
    axes = [fig.add_subplot(gs[1, j]) for j in range(3)]
    im, _ = heat(axes[0], df, "measurement_values", limit, "Synthetic observation", "c")
    heat(axes[1], df, "simulation_values", limit, "Recovered prediction", "d", ylabel=False)
    imr, _ = heat(axes[2], df, "residual", residual_limit, "Strain residual", "e", ylabel=False)
    # Horizontal colorbars avoid squeezing the three time–position panels.
    cb = fig.colorbar(im, ax=axes[:2], orientation="horizontal", fraction=.07, pad=.27, aspect=35)
    cb.set_label(r"Axial strain $\epsilon_{yy}$ ($\mu\epsilon$)", labelpad=2)
    cb = fig.colorbar(imr, ax=axes[2], orientation="horizontal", fraction=.07, pad=.27, aspect=17)
    cb.set_label(r"Residual ($\mu\epsilon$)", labelpad=2)
    axes[2].text(.03, .97, f"RMS = {clean_case['metrics']['residual_rms_microstrain']:.4f} " + r"$\mu\epsilon$",
                 transform=axes[2].transAxes, va="top", fontsize=6.7,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 2})
    save(fig, "fig02_clean_reconstruction", 180, 155,
         {"strain_color_limit_microstrain": limit, "residual_color_limit_microstrain": residual_limit,
          "time_coordinates": "native nonuniform; no resampling", "residual_clipped_pct": 0})


def figure3():
    lc, zc = layer_cases["10pct"], zone_cases["10pct"]
    fig, axs = plt.subplots(1, 3, gridspec_kw={"width_ratios": [.92, 1.18, 1.18]})
    fig.subplots_adjust(left=.085, right=.97, bottom=.28, top=.82, wspace=.47)
    profile(axs[0], [(TRUTH, "Truth", "black", "-", 1.7),
                     (lc["alpha"], "Layerwise", BLUE, "-", 1.1),
                     (zc["alpha"], "Zonal", ORANGE, "--", 1.5)])
    axs[0].set_xlim(min(lc["alpha"].min(), zc["alpha"].min())-.25,
                    max(lc["alpha"].max(), zc["alpha"].max())+.25)
    axs[0].legend(frameon=False, loc="center left", fontsize=6.5)
    panel(axs[0], "a", "Model recovery")
    values = np.concatenate([(c["forward"].simulation_values-c["forward"].measurement_values).to_numpy()[POS]*1e6
                             for c in [lc, zc]])
    lim = float(np.quantile(abs(values), .99))
    clip = {}
    for ax, c, letter, title in [(axs[1], lc, "b", "Layerwise residual"), (axs[2], zc, "c", "Zonal residual")]:
        im, clipped = heat(ax, c["forward"], "residual", lim, title, letter, ylabel=False)
        clip[c["metrics"]["case"]] = clipped
        ax.text(.03, .97, f"Data RMS = {c['metrics']['residual_rms_microstrain']:.3f} " + r"$\mu\epsilon$",
                transform=ax.transAxes, va="top", fontsize=6.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})
    cb = fig.colorbar(im, ax=axs[1:], orientation="horizontal", fraction=.10, pad=.31,
                      aspect=32, extend="both")
    cb.set_label(r"Prediction − noisy observation ($\mu\epsilon$); common 99th-percentile limits", fontsize=7)
    axs[0].text(0, -.30, "Model RMSE (dex)\n" +
                f"Layerwise: {lc['metrics']['model_rmse_dex']:.3f}\nZonal: {zc['metrics']['model_rmse_dex']:.3f}",
                transform=axs[0].transAxes, fontsize=7, va="top")
    save(fig, "fig03_fit_vs_recovery", 180, 102,
         {"nominal_floor_pct": 10, "relative_noise_rms_pct": lc["metrics"]["relative_noise_rms_pct"],
          "common_residual_limit_microstrain": lim, "clipped_pct_by_case": clip})


def figure4():
    fig, axs = plt.subplots(1, 3)
    fig.subplots_adjust(left=.09, right=.98, bottom=.27, top=.80, wspace=.47)
    series = [[clean_case] + list(layer_cases.values()), [clean_zonal] + list(zone_cases.values())]
    for cases, color, marker, label in zip(series, [BLUE, ORANGE], ["o", "s"], ["Layerwise", "Zonal"]):
        x = [c["metrics"]["relative_noise_rms_pct"] for c in cases]
        for ax, key in zip(axs, ["low_zone_mean_alpha", "high_zone_mean_alpha", "model_rmse_dex"]):
            y = [c["metrics"][key] for c in cases]
            ax.plot(x, y, marker=marker, color=color, ms=4, lw=1.2, label=label,
                    markerfacecolor="white" if label == "Zonal" else color)
    for ax, truth in zip(axs[:2], [-15., np.log10(3e-15)]):
        ax.axhline(truth, color="black", linestyle="--", lw=1, label="Truth")
        ax.set_ylabel(ALABEL)
    axs[2].set_ylabel("Model RMSE (dex)")
    axs[2].set_ylim(bottom=0)
    for ax, letter, title in zip(axs, "abc", ["Lower zone", "Upper zone", "Whole free window"]):
        ax.set_xlabel("Relative noise RMS (%)")
        ax.set_xlim(-.7, 26)
        ax.set_xticks([0, 5, 10, 15, 20, 25])
        style(ax, True)
        panel(ax, letter, title)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .01), frameon=False, ncol=3)
    save(fig, "fig04_noise_sensitivity", 180, 83,
         {"nominal_floor_pct": [0, 1, 2, 5, 10], "replicates_per_nonzero_level": 1,
          "uncertainty_intervals": "none: no ensemble available"})


def figureS1():
    exact = zone_cases["10pct"]
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, left=.095, right=.98, bottom=.15, top=.92, wspace=.40, hspace=.70)
    ax = fig.add_subplot(gs[0, 0])
    for x, shift, color, label in [(0, 0, "black", "True / exact"), (1, 1, ORANGE, "Shifted +1 m")]:
        ax.plot([x, x], [-25, 25], color="#bbbbbb", lw=1)
        for low, high in [(-20, -16), (14, 20)]:
            ax.add_patch(Rectangle((x-.14, low+shift), .28, high-low, facecolor=color, alpha=.85))
    ax.annotate("+1 m", xy=(1, 20.5), xytext=(.45, 22), fontsize=7,
                arrowprops={"arrowstyle": "->", "lw": .6})
    ax.set_xticks([0, 1], ["True / exact", "Shifted prior"])
    ax.set_xlim(-.45, 1.55)
    ax.set_ylim(-25, 25)
    ax.set_ylabel("Vertical coordinate y (m)")
    panel(ax, "a", "Zone geometry")
    ax = fig.add_subplot(gs[0, 1])
    profile(ax, [(TRUTH, "Truth", "black", "-", 1.7),
                 (exact["alpha"], "Exact zones", BLUE, "-", 1.2),
                 (pert["alpha"], "Shifted zones", ORANGE, "--", 1.4)])
    ax.set_xlim(-18.4, -12.8)
    ax.legend(frameon=False, loc="center left")
    panel(ax, "b", "Recovered permeability")
    axes = [fig.add_subplot(gs[1, j]) for j in range(2)]
    # Remove the shared noise realization for this diagnostic only: compare
    # recovered predictions with the available clean synthetic response.
    values = []
    for c in [exact, pert]:
        values.append((c["forward"].simulation_values.to_numpy()-clean.measurement_values.to_numpy())[POS]*1e6)
    lim = float(np.max(np.abs(np.concatenate(values))))
    clean_rms = {}
    for ax, c, letter, title, err in zip(axes, [exact, pert], "cd",
            ["Exact zones: prediction error", "Shifted zones: prediction error"], values):
        df = c["forward"].copy()
        df["measurement_values"] = clean.measurement_values.to_numpy()
        im, _ = heat(ax, df, "residual", lim, title, letter, ylabel=ax is axes[0])
        clean_rms[c["metrics"]["case"]] = float(np.sqrt(np.mean(err**2)))
        ax.text(.03, .97, f"RMS = {np.sqrt(np.mean(err**2)):.3f} " + r"$\mu\epsilon$",
                transform=ax.transAxes, va="top", fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=.075, pad=.28, aspect=40)
    cb.set_label(r"Prediction − clean synthetic response ($\mu\epsilon$)")
    save(fig, "figS01_zone_mislocation", 180, 150,
         {"zone_shift_m": 1, "common_prediction_error_limit_microstrain": lim,
          "clean_response_rms_microstrain": clean_rms,
          "bottom_panels": "error relative to clean truth; NOT the fitted noisy-data residual"})


figure1()
figure2()
figure3()
figure4()
figureS1()

# Physical setup and parameterization sources behind the schematic/truth arrays.
for path, role in [
    (BASE / "perm5layer_100layer/fwd/101_ground_truth.py", "synthetic truth, geometry, and acquisition"),
    (V2 / "inv/106_optimization_runner_L1.py", "layerwise parameterization, initialization, and L1 settings"),
    (V2 / "inv/104_gradient_fd_check_eps01.py", "zero-based indexing and finite-difference setup"),
    (V5 / "inv/exact_rfsdss_10pct/107_optimization_runner_zonal_L1.py", "zonal expansion and gradient reduction"),
    (V5 / "inv/pert_rfsdss_10pct/zonal.env", "shifted-zone mode"),
    (V4 / "noise_adding/add_noise_rfsdss.py", "structured synthetic noise design"),
]:
    source(path, role)

# Convenience review document; use the individual figure PDFs for publication.
caption_path = HERE / "captions_en.md"
if caption_path.exists():
    sections = re.findall(r"^## (Figure [^\n]+)\n\n(.*?)(?=\n## |\Z)",
                          caption_path.read_text(), re.M | re.S)
    assert len(sections) == len(figmeta)
    with PdfPages(HERE / "figure_review.pdf") as pdf:
        for (stem, meta), (heading, caption) in zip(figmeta.items(), sections):
            page = plt.figure(figsize=(210/25.4, 297/25.4))
            page.text(.07, .96, heading, fontsize=11, va="top", weight="bold")
            ax = page.add_axes([.07, .37, .86, .55])
            ax.imshow(plt.imread(FIG / f"{stem}.png"))
            ax.axis("off")
            caption = textwrap.fill(" ".join(caption.split()), width=105)
            page.text(.07, .34, caption, fontsize=9, va="top", linespacing=1.45)
            page.text(.07, .04, "Review copy · individual figure PDFs retain publication-size vector text and lines",
                      fontsize=7, color=GRAY)
            pdf.savefig(page, dpi=180)
            plt.close(page)

(DATA / "validation.json").write_text(json.dumps({"checks": audits,
    "time_steps": {"minimum_s": float(np.diff(np.sort(clean.measurement_time.unique())).min()),
                   "maximum_s": float(np.diff(np.sort(clean.measurement_time.unique())).max())},
    "n_free_layers": int(FREE.sum()), "objective_mask": "measurement_time > 0",
    "observation_pair_equality": "verified for all four noise levels and exact/shifted 10pct",
    "initial_model_equality": "verified for all four paired noise levels",
    "figure_title_label_canvas_bounds": "passed at final figure dimensions",
    "no_simulation_executed": True}, indent=2) + "\n")
(DATA / "figure_metadata.json").write_text(json.dumps(figmeta, indent=2) + "\n")
source(Path(__file__), "reproducible plotting and metric computation")
(DATA / "source_manifest.json").write_text(json.dumps(source_files, indent=2) + "\n")
print(pd.DataFrame(metrics)[["case", "J_data", "model_rmse_dex", "relative_noise_rms_pct"]].to_string(index=False))
print(f"Saved {len(figmeta)} figures to {FIG}")
