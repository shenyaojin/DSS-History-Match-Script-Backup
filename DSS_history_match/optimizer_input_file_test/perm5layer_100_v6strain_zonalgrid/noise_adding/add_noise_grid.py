# Generate the 9 noisy datasets for the 3x3x3 zonal sensitivity grid.
#
# Two DATA factors (the third factor, zone-position offset, lives in the
# inversion, not the data):
#   A = real DSS background noise   (levels bg1<bg2<bg3)  -- REAL texture
#   B = white instrument noise      (levels w1 <w2 <w3 )  -- pure i.i.d. Gaussian
#
# Factor A uses the *character* of real field DSS background noise, extracted
# from the pre-injection window of an observed strain waterfall, then resampled
# onto the synthetic (130 time x 500 channel) grid and scaled to a target level.
# The real background's native amplitude (~4e-8 strain) is far too small to
# affect the inversion (< 0.2% of signal), so we keep its temporal structure
# (the slow drift / wander that distinguishes it from white noise) but scale it
# to meaningful levels anchored to REF, matched to the white-noise levels so the
# grid isolates "noise character (real vs white) x amplitude".
#
#   d_noisy(c,t) = d_clean(c,t) + sigma_A(i) * realTexture(c,t)
#                                + sigma_B(j) * N(0,1)
#   sigma_A(i) = LEVELS_A[i] * REF   ,   sigma_B(j) = LEVELS_B[j] * REF
#   REF = median over channels of each channel's peak |strain|  (= 2.262e-5)
#
# Real DSS source (from the Explore characterization):
#   output/das_observed/das_strain_waterfall_T1T3.npz
#     data (45 ch x 4620 t), millistrain; pre-injection background = t index 1..41.
#     -> *1e-3 to dimensionless strain, per-channel detrended residual as texture.
#
# Outputs (in noise_data/):
#   measurement_data_bg<i>_w<j>.csv (+ .meta)   the 9 datasets
#   grid_noise_summary.csv
#   grid_noise_qc.png                            noise-character QC figure

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V6 = os.path.dirname(HERE)
CLEAN_CSV = os.path.join(V6, "data", "obs_strain_yy.csv")
CLEAN_META = os.path.join(V6, "data", "obs_strain_yy.meta")
NOISE_DIR = os.path.join(V6, "noise_data")
REPO = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner"
DAS_NPZ = os.path.join(REPO, "output", "das_observed", "das_strain_waterfall_T1T3.npz")

VALUE_COL = "measurement_values"
CHANNEL_COL = "measurement_ycoord"
TIME_COL = "measurement_time"

# level fractions of REF (A and B matched so the grid isolates character)
LEVELS_A = [0.03, 0.06, 0.12]   # real-textured DSS bg
LEVELS_B = [0.03, 0.06, 0.12]   # white
BASE_SEED = 20260901

# real background extraction window (Explore report)
DAS_BG_TCOLS = (1, 42)          # pre-injection samples (exclude the all-zero t=0 col)


def read_clean_meta():
    meta = {}
    if os.path.exists(CLEAN_META):
        for line in open(CLEAN_META):
            line = line.strip()
            if line and "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    return meta


def median_channel_peak(df):
    a = df[VALUE_COL].abs()
    return float(np.median(a.groupby(df[CHANNEL_COL]).max().to_numpy()))


def load_real_bg_residual():
    """Return first differences, excluding the integration startup increment."""
    with np.load(DAS_NPZ, allow_pickle=True) as d:
        raw = np.asarray(d["data"], dtype=float)
    block = raw[:, 1:42] * 1e-3
    return np.diff(block, axis=1)[:, 1:]


def real_texture(rng, n_time, n_chan, bg_res):
    """Bootstrap independent standardized increments with real channel scales."""
    s_real = bg_res.std(axis=1)
    if np.any(s_real <= 0):
        raise ValueError("Real DSS channel has zero increment variance")
    pool = ((bg_res - bg_res.mean(axis=1, keepdims=True)) /
            s_real[:, None]).ravel()
    real_channels = rng.integers(len(s_real), size=n_chan)
    indices = rng.integers(pool.size, size=(n_time, n_chan))
    T = pool[indices] * s_real[real_channels][None, :] / np.median(s_real)
    T -= T.mean()
    T /= T.std()
    return T


def mean_lag1(field):
    """Mean of per-channel Pearson lag-1 correlations, including t=0."""
    x = field[:-1] - field[:-1].mean(axis=0)
    y = field[1:] - field[1:].mean(axis=0)
    return float(np.mean(np.sum(x*y, axis=0) /
                         np.sqrt(np.sum(x*x, axis=0)*np.sum(y*y, axis=0))))


def build_extraction_qc(diag, bg_res):
    import matplotlib.pyplot as plt
    with np.load(DAS_NPZ, allow_pickle=True) as d:
        raw = np.asarray(d["data"], dtype=float)
    block = raw[:, 1:42] * 1e-3
    centered = bg_res - bg_res.mean(axis=1, keepdims=True)
    native_std = bg_res.std(axis=1)
    pool = (centered / native_std[:, None]).ravel()
    d = diag[(3, 3)]
    A, B = d["noise_a"], d["noise_b"]
    ac = mean_lag1(A)
    fig, axes = plt.subplots(2, 3, figsize=(19, 10), constrained_layout=True)
    ax = axes.ravel()
    lim = np.percentile(np.abs(raw), 99)
    im = ax[0].imshow(raw, aspect="auto", origin="lower", cmap="RdBu_r",
                      vmin=-lim, vmax=lim, extent=(-0.5, 4619.5, -0.5, 44.5))
    ax[0].axvspan(1, 41, color="lime", alpha=0.8)
    ax[0].annotate("Extract indices 1–41 (1–41 min)", xy=(21, 35), xytext=(700, 36),
                   arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
    ax[0].set(title="(a) Full cumulative DSS waterfall", xlabel="Time index (60 s/sample)", ylabel="Real channel")
    fig.colorbar(im, ax=ax[0], label="mstrain (color clipped at 99th percentile)")
    for c in [0, 11, 22, 33, 44]:
        ax[1].plot(np.arange(1,42), block[c]*1e9, label=f"ch {c}")
        ax[2].plot(np.arange(3,42), centered[c]*1e9, label=f"ch {c}", lw=1)
    ax[1].set(title="(b) Extracted cumulative strain: drift", xlabel="Time index / minutes", ylabel="Cumulative strain [nε]")
    ax[1].legend(fontsize=8, ncol=2)
    ax[2].axhline(0, color="black", lw=0.6)
    ax[2].set(title=f"(c) First differences, channel-centered\nMedian native std = {np.median(native_std):.2e} strain", xlabel="Increment ending at time index (startup dropped)", ylabel="Increment [nε]")
    ax[3].hist(pool, bins=65, density=True, alpha=0.65, label=f"DSS empirical pool (n={pool.size})")
    xx=np.linspace(min(pool.min(), -4), max(pool.max(),4), 500)
    ax[3].plot(xx, np.exp(-xx**2/2)/np.sqrt(2*np.pi), label="Gaussian N(0,1)")
    ax[3].set_yscale("log")
    ax[3].set(title="(d) Standardized increments: distribution / tails", xlabel="Increment / channel std", ylabel="Probability density (log scale)")
    ax[3].legend(fontsize=8)
    lim=np.percentile(np.abs(A*1e9),99)
    im=ax[4].imshow(A*1e9, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim,vmax=lim)
    ax[4].set(title="(e) bg3 A: independent bootstrap, 130 × 500", xlabel="Output channel", ylabel="Output time index")
    fig.colorbar(im, ax=ax[4], label="Noise [nε], 99th-percentile color limits")
    ax[5].plot(A[:,0]*1e9, label="A: DSS bootstrap", lw=1)
    ax[5].plot(B[:,0]*1e9, label="B: Gaussian white", lw=1, alpha=0.7)
    ax[5].set(title=f"(f) A lag1 autocorr = {ac:.5f} (channel mean)\nOld ≈ 0.97; one output channel shown", xlabel="Output time index", ylabel="Noise [nε]")
    ax[5].legend(fontsize=8)
    fig.suptitle("DSS noise extraction: cumulative strain → increments → independent bootstrap", fontsize=16)
    out=os.path.join(NOISE_DIR,"dss_noise_extraction.png")
    fig.savefig(out,dpi=170)
    plt.close(fig)
    print(f"Wrote DSS extraction figure: {out}")


def main():
    os.makedirs(NOISE_DIR, exist_ok=True)
    df_clean = pd.read_csv(os.path.abspath(CLEAN_CSV))
    ref = median_channel_peak(df_clean)
    base_meta = read_clean_meta()

    times = np.sort(df_clean[TIME_COL].unique())
    chans = np.sort(df_clean[CHANNEL_COL].unique())
    n_time, n_chan = times.size, chans.size
    ti = {t: k for k, t in enumerate(times)}
    ci = {c: k for k, c in enumerate(chans)}
    row_t = df_clean[TIME_COL].map(ti).to_numpy()
    row_c = df_clean[CHANNEL_COL].map(ci).to_numpy()
    t0_mask = (row_t == 0)

    bg_res = load_real_bg_residual()
    print(f"REF (median channel peak): {ref:.6e}  ({ref*1e9:.0f} nε)")
    print(f"real DSS first-difference block: {bg_res.shape}  std={bg_res.std():.3e} strain "
          f"(native, before scaling)")
    print(f"LEVELS_A (real texture) x REF: {[f'{p*100:.0f}%={p*ref*1e9:.0f}nε' for p in LEVELS_A]}")
    print(f"LEVELS_B (white)        x REF: {[f'{p*100:.0f}%={p*ref*1e9:.0f}nε' for p in LEVELS_B]}\n")

    diag = {}
    summary = []
    for i, fa in enumerate(LEVELS_A, start=1):
        for j, fb in enumerate(LEVELS_B, start=1):
            rng = np.random.default_rng(BASE_SEED + 100 * i + j)
            sigma_a = fa * ref
            sigma_b = fb * ref
            noise_a = sigma_a * real_texture(rng, n_time, n_chan, bg_res)
            noise_a[0, :] = 0.0
            ac = mean_lag1(noise_a)
            print(f"bg{i}_w{j}: A mean channel lag1 autocorr = {ac:.6f}")
            if abs(ac) >= 0.03:
                raise ValueError(f"A lag1 acceptance failed: {ac}")
            noise_b = sigma_b * rng.standard_normal((n_time, n_chan))
            noise = noise_a + noise_b
            noise[0, :] = 0.0                          # keep t=0 reference frame

            flat = noise[row_t, row_c]
            flat[t0_mask] = 0.0
            dfn = df_clean.copy()
            dfn[VALUE_COL] = df_clean[VALUE_COL].to_numpy() + flat
            for col in ("misfit_values", "simulation_values"):
                if col in dfn.columns:
                    dfn[col] = 0.0

            stem = f"measurement_data_bg{i}_w{j}"
            dfn.to_csv(os.path.join(NOISE_DIR, f"{stem}.csv"), index=False)
            meta = dict(base_meta)
            meta.update({
                "noise_model": "grid_realbg_plus_white",
                "ref_value": f"{ref:.10e}",
                "bg_level": i, "white_level": j,
                "sigma_realbg": f"{sigma_a:.10e}", "sigma_white": f"{sigma_b:.10e}",
                "realbg_frac_ref": f"{fa}", "white_frac_ref": f"{fb}",
                "das_source": os.path.basename(DAS_NPZ),
                "seed": f"{BASE_SEED + 100 * i + j}",
            })
            with open(os.path.join(NOISE_DIR, f"{stem}.meta"), "w") as f:
                for k, v in meta.items():
                    f.write(f"{k}={v}\n")

            tot = float(noise.std())
            summary.append({
                "bg_level": i, "white_level": j,
                "sigma_realbg": sigma_a, "sigma_white": sigma_b,
                "realbg_pct": fa * 100, "white_pct": fb * 100,
                "total_noise_std": tot, "snr_ref": ref / tot if tot else np.inf,
                "file": f"{stem}.csv",
            })
            diag[(i, j)] = {"noise_a": noise_a, "noise_b": noise_b, "noise": noise,
                            "times": times, "chans": chans}
            print(f"bg{i}_w{j}: sigma_A={sigma_a*1e9:5.0f}nε (real) + sigma_B={sigma_b*1e9:5.0f}nε "
                  f"(white) -> total_std={tot*1e9:5.0f}nε")

    pd.DataFrame(summary).to_csv(os.path.join(NOISE_DIR, "grid_noise_summary.csv"), index=False)
    build_qc(diag, ref, chans, times)
    build_extraction_qc(diag, bg_res)
    print(f"\nWrote 9 datasets + summary to {NOISE_DIR}")


def build_qc(diag, ref, chans, times):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    # (a) real-texture vs white time series at one edge channel, mid level (2,2)
    d = diag[(2, 2)]
    ci = int(np.argmin(np.abs(chans - (-25.0))))
    ax[0, 0].plot(times, d["noise_a"][:, ci], label="real DSS texture (A)", lw=2)
    ax[0, 0].plot(times, d["noise_b"][:, ci], label="white (B)", lw=1, alpha=0.8)
    ax[0, 0].set(title="(a) A vs B character @ edge ch (bg2_w2)", xlabel="time", ylabel="strain")
    ax[0, 0].legend(fontsize=8)
    # (b) real-texture heatmap (shows temporal wander structure)
    vmax = np.percentile(np.abs(d["noise_a"]), 99)
    im = ax[0, 1].pcolormesh(times, chans, d["noise_a"].T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ax[0, 1].set(title="(b) real DSS texture field (A, bg2_w2)", xlabel="time", ylabel="ycoord")
    fig.colorbar(im, ax=ax[0, 1])
    # (c) white heatmap
    vmax = np.percentile(np.abs(d["noise_b"]), 99)
    im = ax[0, 2].pcolormesh(times, chans, d["noise_b"].T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ax[0, 2].set(title="(c) white field (B, bg2_w2)", xlabel="time", ylabel="ycoord")
    fig.colorbar(im, ax=ax[0, 2])
    # (d) total noise std across the 9 cells
    tot = np.array([[float(diag[(i, j)]["noise"].std()) for j in (1, 2, 3)] for i in (1, 2, 3)])
    im = ax[1, 0].imshow(tot * 1e9, origin="lower", cmap="magma_r")
    ax[1, 0].set(title="(d) total noise std [nε] over 9 datasets", xlabel="white level", ylabel="bg level")
    ax[1, 0].set_xticks([0, 1, 2]); ax[1, 0].set_xticklabels(["w1", "w2", "w3"])
    ax[1, 0].set_yticks([0, 1, 2]); ax[1, 0].set_yticklabels(["bg1", "bg2", "bg3"])
    for a in range(3):
        for b in range(3):
            ax[1, 0].text(b, a, f"{tot[a,b]*1e9:.0f}", ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im, ax=ax[1, 0])
    # (e) power spectrum: real texture vs white (temporal), to show A is not white
    from numpy.fft import rfft, rfftfreq
    a_ps = np.abs(rfft(d["noise_a"] - d["noise_a"].mean(0), axis=0)).mean(1)
    b_ps = np.abs(rfft(d["noise_b"] - d["noise_b"].mean(0), axis=0)).mean(1)
    fr = rfftfreq(len(times))
    ax[1, 1].loglog(fr[1:], a_ps[1:], label="real texture (A)", lw=2)
    ax[1, 1].loglog(fr[1:], b_ps[1:], label="white (B)", lw=1)
    ax[1, 1].set(title="(e) temporal power spectrum (A is red/low-freq, B flat)",
                 xlabel="freq (1/step)", ylabel="|FFT| mean over ch")
    ax[1, 1].legend(fontsize=8)
    # (f) example noisy vs clean at a lobe channel (bg3_w3, worst)
    d3 = diag[(3, 3)]
    df_clean = pd.read_csv(os.path.abspath(CLEAN_CSV))
    cl = int(np.argmin(np.abs(chans - 18.0)))
    cval = chans[cl]
    sub = df_clean[df_clean[CHANNEL_COL] == cval].sort_values(TIME_COL)
    ax[1, 2].plot(sub[TIME_COL], sub[VALUE_COL], "k-", lw=2, label="clean")
    ax[1, 2].plot(times, sub[VALUE_COL].to_numpy() + d3["noise"][:, cl], "r.", ms=4, alpha=0.6,
                  label="bg3_w3 noisy")
    ax[1, 2].set(title=f"(f) clean vs worst-cell noisy @ lobe y={cval:.1f}", xlabel="time", ylabel="strain")
    ax[1, 2].legend(fontsize=8)

    fig.suptitle(f"Grid noise QC — real DSS texture (A) + white (B), REF={ref:.3e}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(NOISE_DIR, "grid_noise_qc.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote QC figure: {out}")


if __name__ == "__main__":
    main()
