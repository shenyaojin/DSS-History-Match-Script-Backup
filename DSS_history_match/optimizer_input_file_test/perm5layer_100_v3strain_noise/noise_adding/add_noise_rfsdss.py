# Generate *realistic* RFS-DSS noise realizations for the strain_yy observation,
# at several controllable noise LEVELS.
#
# Unlike add_noise.py (which adds a single spatially/temporally uniform i.i.d.
# Gaussian std to every point), this generator reproduces the qualitative
# structure actually seen in Rayleigh-frequency-shift DSS strain data by adding
# FOUR distinct components on top of the clean observation:
#
#   d_noisy(c,t) = d_clean(c,t)
#                + floor_c * N(0,1)      # (1) per-channel white noise floor
#                + drift_c(t)            # (2) per-channel low-frequency drift
#                + common(t)             # (3) shared common-mode drift
#                + spike(c,t)            # (4) sparse heavy-tailed spikes
#
# Design choices (confirmed with the user):
#   * Absolute magnitudes are ANCHORED to the median-channel-peak reference
#     REF = median over receivers of each channel's peak |strain| over time
#     (= the same "typical channel amplitude" used by the mednoise family),
#     so these datasets are directly comparable to the mednoise_*pct sweep.
#   * The noise LEVEL is a single knob: `floor_frac` = floor mean / REF. The
#     other three components scale with the floor by FIXED RATIOS (below), so
#     raising the level scales every component's amplitude together while the
#     noise *shape* (per-channel spread, drift spectrum, spike pattern) is fixed.
#         floor  mean  = floor_frac        * REF   (per-channel, lognormal, NOT
#                                                    scaled by local signal level)
#         drift  std   = 0.75 * floor_frac * REF   (per channel, low-freq in time)
#         common std   = 0.50 * floor_frac * REF   (single series, all channels)
#         spike  scale = 7.5  * floor_frac * REF   (heavy-tailed, random sign)
#   * The per-channel noise floor varies channel-to-channel (lognormal) and is
#     an ABSOLUTE strain level -- it does NOT follow the signal peak.
#   * SPIKES are non-Gaussian, sparse, and DRIVEN BY THE FIBER STRETCHING. Real
#     RFS-DSS spikes are rare while the strain is low (early time / quiet zones)
#     and become frequent where/when the fibre is pulled hard. So the spike
#     probability is proportional to the local strain magnitude, times an extra
#     spatial weight that grows away from the low-strain array centre:
#
#         w(c,t)  = |d_clean(c,t)| / max|d_clean|                      # strain drive
#                   * [ w_min + (1-w_min) * (|y_c - y0| / d_max)^gamma ] # spatial
#         p(c,t)  = clip( scale * w(c,t), 0, p_cap )
#
#     `scale` is auto-set each run so the mean spike probability hits
#     SPIKE_TARGET_FRAC. Because p depends only on the clean signal + geometry
#     (not the level), the spike *pattern in space and time* is identical across
#     levels; only the spike amplitude scales with the level. Consequences,
#     matching the observed data:
#       - early low-strain samples get very few spikes;
#       - the low-strain array centre (y~0) is quietest, spikes grow toward the
#         high-strain lobes (y~+/-18) and stay elevated toward the edges;
#       - a channel gets most of its spikes while it is being stretched.
#
# Each level uses its own deterministic seed, so datasets are reproducible and
# their realizations are independent.
#
# Run as:
#   python add_noise_rfsdss.py
#
# Outputs (in noise_adding/):
#   measurement_data_rfsdss_<tag>.csv   clean + realistic RFS-DSS noise per level
#   measurement_data_rfsdss_<tag>.meta  metadata (component stds + params)
#   rfsdss_noise_qc_<tag>.png           per-level QC figure (see build_qc_figure)
#   rfsdss_noise_levels_summary.png     cross-level comparison figure
#   rfsdss_noise_summary.csv            requested vs realized statistics per level

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN_CSV = os.path.join(HERE, "..", "data", "obs_strain_yy.csv")
CLEAN_META = os.path.join(HERE, "..", "data", "obs_strain_yy.meta")

VALUE_COL = "measurement_values"
CHANNEL_COL = "measurement_ycoord"
TIME_COL = "measurement_time"

# --- fracture geometry (from inv/106_optimization_runner_L1.py zone masks) ---
FRACTURE_Y_LO = 14.0
FRACTURE_Y_HI = 20.0
FRACTURE_CENTER = 0.5 * (FRACTURE_Y_LO + FRACTURE_Y_HI)  # y = 17.0 (drawn for ref)

# --- component ratios relative to the floor (fixed across levels) ---
RATIO_DRIFT = 0.75    # drift std   / floor mean
RATIO_COMMON = 0.50   # common std  / floor mean
RATIO_SPIKE = 7.5     # spike scale / floor mean

# --- shape parameters (fixed across levels) ---
FLOOR_LOGSTD = 0.40      # lognormal spread of the per-channel floor
FLOOR_CLIP = (0.4, 3.0)  # clip floor_c to [lo, hi] * mean
DRIFT_MODES = 3          # low-frequency cosine modes per channel
COMMON_MODES = 2         # low-frequency modes for the common-mode series

# --- spike occurrence model: strain-driven x spatial weight ---
SPIKE_CENTER_Y = 0.0     # low-strain array centre; spikes grow away from it
SPIKE_GAMMA = 2.0        # spatial growth exponent (|y - y0|/d_max)^gamma
SPIKE_SPATIAL_WMIN = 0.30  # spatial weight floor at the centre (0 = none there)
SPIKE_TARGET_FRAC = 0.012  # mean spike probability (fraction of points spiked)
SPIKE_P_CAP = 0.25       # per-point probability cap

# --- noise levels: (tag, floor_frac = floor mean / REF, seed) ---
LEVELS = [
    ("0p5pct", 0.005, 20260709),
    ("1pct", 0.010, 20260710),
    ("2pct", 0.020, 20260711),
    ("5pct", 0.050, 20260712),
    ("10pct", 0.100, 20260713),
]


def read_clean_meta():
    meta = {}
    if os.path.exists(CLEAN_META):
        with open(CLEAN_META, "r") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                meta[key.strip()] = val.strip()
    return meta


def median_channel_peak(df):
    """Median over receivers of each channel's peak |strain| over time."""
    abs_values = df[VALUE_COL].abs()
    per_channel_peak = abs_values.groupby(df[CHANNEL_COL]).max()
    return float(np.median(per_channel_peak.to_numpy()))


def clean_matrix(df, times, chans, ti, ci):
    """Reshape the clean measurement column into a (n_time, n_chan) matrix."""
    mat = np.zeros((times.size, chans.size))
    rt = df[TIME_COL].map(ti).to_numpy()
    rc = df[CHANNEL_COL].map(ci).to_numpy()
    mat[rt, rc] = df[VALUE_COL].to_numpy()
    return mat


def spike_probability(clean_mat, chans):
    """Strain-driven spike probability p(t, c); shared by all levels.

    p grows with the local strain magnitude (fibre being stretched) and with an
    extra spatial weight that increases away from the low-strain centre y0.
    `scale` is chosen so the mean probability equals SPIKE_TARGET_FRAC.
    """
    strain_drive = np.abs(clean_mat)
    peak = strain_drive.max()
    if peak > 0:
        strain_drive = strain_drive / peak  # -> [0, 1]

    dist = np.abs(chans - SPIKE_CENTER_Y)
    d_max = dist.max() if dist.max() > 0 else 1.0
    spatial = SPIKE_SPATIAL_WMIN + (1.0 - SPIKE_SPATIAL_WMIN) * (dist / d_max) ** SPIKE_GAMMA

    w = strain_drive * spatial[None, :]
    mean_w = w.mean()
    scale = SPIKE_TARGET_FRAC / mean_w if mean_w > 0 else 0.0
    p = np.clip(scale * w, 0.0, SPIKE_P_CAP)
    return p, spatial


def lowfreq_series(rng, tn, n_series, n_modes, target_std):
    """Sum of low-frequency cosine modes with random amplitude/phase.

    Returns an (n_time, n_series) array whose per-series std == target_std.
    `tn` is the time axis normalised to [0, 1] in *real* time so the drift is
    smooth with respect to the physical elapsed time, not the sample index.
    """
    out = np.zeros((tn.size, n_series))
    for k in range(1, n_modes + 1):
        amp = rng.standard_normal(n_series)
        phase = rng.uniform(0.0, 2.0 * np.pi, n_series)
        out += amp[None, :] * np.cos(2.0 * np.pi * k * tn[:, None] + phase[None, :])
    std = out.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return out / std * target_std


def build_noise(clean_mat, times, chans, p_spike, ref, floor_frac, seed):
    """Build the (n_time, n_chan) noise matrix and per-channel diagnostics.

    floor_frac sets the overall level; drift/common/spike scale with it via the
    fixed RATIO_* constants, so the noise shape is level-independent. p_spike is
    precomputed once (level-independent) so the spike pattern is shared.
    """
    n_time, n_chan = times.size, chans.size
    rng = np.random.default_rng(seed)

    floor_mean = floor_frac * ref
    drift_std = RATIO_DRIFT * floor_mean
    common_std = RATIO_COMMON * floor_mean
    spike_amp = RATIO_SPIKE * floor_mean

    # (1) per-channel absolute white noise floor (lognormal, NOT signal-scaled)
    floor_c = floor_mean * np.exp(rng.normal(0.0, FLOOR_LOGSTD, n_chan))
    floor_c = np.clip(floor_c, FLOOR_CLIP[0] * floor_mean, FLOOR_CLIP[1] * floor_mean)
    white = rng.standard_normal((n_time, n_chan)) * floor_c[None, :]

    # normalised real-time axis for the low-frequency components
    span = times.max() - times.min()
    tn = (times - times.min()) / span if span > 0 else np.zeros(n_time)

    # (2) per-channel low-frequency drift
    drift = lowfreq_series(rng, tn, n_chan, DRIFT_MODES, drift_std)

    # (3) shared common-mode drift (same series added to every channel)
    common = lowfreq_series(rng, tn, 1, COMMON_MODES, common_std)  # (n_time, 1)

    # (4) strain-driven sparse heavy-tailed spikes (pattern from p_spike)
    spike_mask = rng.random((n_time, n_chan)) < p_spike
    spike_mag = spike_amp * (1.0 + rng.exponential(1.0, (n_time, n_chan)))
    spike_sign = rng.choice((-1.0, 1.0), size=(n_time, n_chan))
    spikes = spike_mask * spike_sign * spike_mag

    noise = white + drift + common + spikes

    diag = {
        "times": times,
        "chans": chans,
        "floor_c": floor_c,
        "floor_mean": floor_mean,
        "drift_std": drift_std,
        "common_std": common_std,
        "spike_amp": spike_amp,
        "spike_mask": spike_mask,
        "spike_count_per_chan": spike_mask.sum(axis=0),
        "spike_count_per_time": spike_mask.sum(axis=1),
        "white": white,
        "drift": drift,
        "common": common[:, 0],
        "spikes": spikes,
        "noise": noise,
    }
    return noise, diag


def build_qc_figure(df_clean, clean_mat, diag, p_spike, spatial, ref, tag, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, chans = diag["times"], diag["chans"]
    noise = diag["noise"]
    ci_center = int(np.argmin(np.abs(chans - SPIKE_CENTER_Y)))
    ci_lobe = int(np.argmin(np.abs(chans - 18.0)))
    ci_edge = int(np.argmin(np.abs(chans - (-25.0))))

    fig, ax = plt.subplots(2, 4, figsize=(23, 10))

    # (a) noise realization heatmap (channel x time)
    vmax = np.percentile(np.abs(noise), 99.5)
    im = ax[0, 0].pcolormesh(times, chans, noise.T, cmap="RdBu_r",
                             vmin=-vmax, vmax=vmax, shading="auto")
    ax[0, 0].set(title="(a) added noise (channel x time)", xlabel="time", ylabel="ycoord")
    fig.colorbar(im, ax=ax[0, 0], label="strain")

    # (b) spike probability heatmap p(t,c): few early/centre, many at stretched lobes
    im2 = ax[0, 1].pcolormesh(times, chans, (p_spike * 100).T, cmap="magma", shading="auto")
    ax[0, 1].axhline(SPIKE_CENTER_Y, color="w", ls="--", lw=1)
    ax[0, 1].set(title="(b) spike prob p(t,c) [%]  (strain-driven)",
                 xlabel="time", ylabel="ycoord")
    fig.colorbar(im2, ax=ax[0, 1], label="prob [%]")

    # (c) spike count per channel (spatial): fewest at low-strain centre y0
    ax[0, 2].bar(chans, diag["spike_count_per_chan"], width=0.09, color="C3")
    ax[0, 2].axvline(SPIKE_CENTER_Y, color="k", ls="--", lw=1, label="centre y0")
    axc2 = ax[0, 2].twinx()
    sig_peak = df_clean.assign(_a=df_clean[VALUE_COL].abs()).groupby(CHANNEL_COL)["_a"].max()
    sig_peak = sig_peak.reindex(chans).to_numpy()
    axc2.plot(chans, sig_peak * 1e9, color="C2", lw=1.5, label="signal peak")
    axc2.set_ylabel("signal peak [nε]", color="C2")
    ax[0, 2].set(xlabel="ycoord", ylabel="# spikes",
                 title="(c) spikes track strain lobes (spatial)")
    ax[0, 2].legend(fontsize=7, loc="upper center")

    # (d) spike count vs TIME (temporal): few early (low strain), many when pulled
    ax[0, 3].plot(times, diag["spike_count_per_time"], color="C3", lw=1.5, label="# spikes")
    axd2 = ax[0, 3].twinx()
    mean_abs_t = np.abs(clean_mat).mean(axis=1)
    axd2.plot(times, mean_abs_t * 1e9, color="C2", lw=1.5, label="mean |strain|")
    axd2.set_ylabel("mean |strain| [nε]", color="C2")
    ax[0, 3].set(xlabel="time", ylabel="# spikes over channels",
                 title="(d) few early, more when stretched (temporal)")

    # (e) per-channel floor and spatial weight
    axe = ax[1, 0]
    axe.plot(chans, diag["floor_c"] * 1e9, color="C0", label="floor_c [nε]")
    axe.axhline(diag["floor_mean"] * 1e9, color="C0", ls=":", lw=1, label="floor mean")
    axe.set(xlabel="ycoord", ylabel="floor [nε]", title="(e) per-channel floor & spatial wt")
    axe2 = axe.twinx()
    axe2.plot(chans, spatial, color="C4", label="spatial weight")
    axe2.set_ylabel("spatial weight", color="C4")
    axe.legend(fontsize=7, loc="upper left")

    # (f) example channel time series: clean vs noisy, centre vs lobe
    for ci, lbl, col in ((ci_center, f"centre y={chans[ci_center]:.1f}", "C0"),
                         (ci_lobe, f"lobe y={chans[ci_lobe]:.1f}", "C1")):
        cval = chans[ci]
        clean_c = df_clean[df_clean[CHANNEL_COL] == cval].sort_values(TIME_COL)
        noisy_v = clean_c[VALUE_COL].to_numpy() + diag["noise"][:, ci]
        ax[1, 1].plot(clean_c[TIME_COL], clean_c[VALUE_COL], col + "-", lw=1.5,
                      label=f"clean {lbl}")
        ax[1, 1].plot(times, noisy_v, col + ".", ms=3, alpha=0.6, label=f"noisy {lbl}")
    ax[1, 1].set(xlabel="time", ylabel="strain", title="(f) clean vs noisy time series")
    ax[1, 1].legend(fontsize=7)

    # (g) component decomposition at the lobe channel (most stretched)
    ci = ci_lobe
    ax[1, 2].plot(times, diag["white"][:, ci], label="white floor", alpha=0.7)
    ax[1, 2].plot(times, diag["drift"][:, ci], label="per-chan drift", lw=2)
    ax[1, 2].plot(times, diag["common"], label="common-mode", lw=2)
    ax[1, 2].plot(times, diag["spikes"][:, ci], "k.", ms=4, label="spikes")
    ax[1, 2].set(xlabel="time", ylabel="strain",
                 title=f"(g) noise components @ lobe y={chans[ci]:.1f}")
    ax[1, 2].legend(fontsize=8)

    # (h) noise floor vs signal amplitude per channel (floor flat -> not signal-scaled)
    ax[1, 3].plot(chans, sig_peak * 1e9, color="C2", label="signal peak |strain| [nε]")
    ax[1, 3].plot(chans, diag["floor_c"] * 1e9, color="C0", label="noise floor_c [nε]")
    ax[1, 3].set(xlabel="ycoord", ylabel="strain [nε]", yscale="log",
                 title="(h) floor flat vs signal (not signal-scaled)")
    ax[1, 3].legend(fontsize=8)

    fig.suptitle(
        f"Realistic RFS-DSS noise QC  [level {tag}]  (REF={ref:.3e}, floor mean="
        f"{diag['floor_mean']*1e9:.0f} nε, spikes strain-driven)", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=125)
    plt.close(fig)


def build_levels_summary_figure(df_clean, per_level, ref, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    tags = [r["tag"] for r in per_level]

    # (a) realized total noise std vs level
    ax[0].bar(tags, [r["total_noise_std"] * 1e9 for r in per_level], color="C0")
    ax[0].set(xlabel="level", ylabel="total noise std [nε]",
              title="(a) noise level scales with floor_frac")
    for i, r in enumerate(per_level):
        ax[0].text(i, r["total_noise_std"] * 1e9, f"{r['total_noise_std']*1e9:.0f}",
                   ha="center", va="bottom", fontsize=8)

    # (b) example lobe channel: clean + noisy overlay across levels
    chans = per_level[0]["diag"]["chans"]
    ci_lobe = int(np.argmin(np.abs(chans - 18.0)))
    cval = chans[ci_lobe]
    clean_c = df_clean[df_clean[CHANNEL_COL] == cval].sort_values(TIME_COL)
    ax[1].plot(clean_c[TIME_COL], clean_c[VALUE_COL], "k-", lw=2, label="clean", zorder=5)
    for r in per_level:
        t = r["diag"]["times"]
        noisy = clean_c[VALUE_COL].to_numpy() + r["diag"]["noise"][:, ci_lobe]
        ax[1].plot(t, noisy, ".", ms=3, alpha=0.5, label=r["tag"])
    ax[1].set(xlabel="time", ylabel="strain",
              title=f"(b) lobe channel y={cval:.1f}: clean vs levels")
    ax[1].legend(fontsize=8)

    # (c) total noise std vs floor mean (linear)
    ax[2].plot([r["floor_mean"] * 1e9 for r in per_level],
               [r["total_noise_std"] * 1e9 for r in per_level], "o-")
    for r in per_level:
        ax[2].annotate(r["tag"], (r["floor_mean"] * 1e9, r["total_noise_std"] * 1e9),
                       fontsize=8, textcoords="offset points", xytext=(4, 4))
    ax[2].set(xlabel="floor mean [nε]", ylabel="total noise std [nε]",
              title="(c) total noise vs floor level")

    fig.suptitle(f"RFS-DSS noise levels summary  (REF={ref:.3e} = {ref*1e9:.0f} nε)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    clean_path = os.path.abspath(CLEAN_CSV)
    if not os.path.exists(clean_path):
        raise FileNotFoundError(clean_path)

    df_clean = pd.read_csv(clean_path)
    if VALUE_COL not in df_clean.columns:
        raise RuntimeError(f"{clean_path} has no '{VALUE_COL}' column.")

    ref = median_channel_peak(df_clean)
    print(f"Clean observation       : {clean_path}")
    print(f"median-channel-peak REF : {ref:.6e}  ({ref * 1e9:.1f} nε)")
    print(f"spike centre y0         : {SPIKE_CENTER_Y}   (strain-driven spikes)")
    print("")

    base_meta = read_clean_meta()
    times = np.sort(df_clean[TIME_COL].unique())
    chans = np.sort(df_clean[CHANNEL_COL].unique())
    ti = {t: i for i, t in enumerate(times)}
    ci = {c: i for i, c in enumerate(chans)}
    row_t = df_clean[TIME_COL].map(ti).to_numpy()
    row_c = df_clean[CHANNEL_COL].map(ci).to_numpy()

    clean_mat = clean_matrix(df_clean, times, chans, ti, ci)
    p_spike, spatial = spike_probability(clean_mat, chans)
    print(f"spike prob: mean={p_spike.mean()*100:.3f}%  max={p_spike.max()*100:.2f}%  "
          f"(target mean={SPIKE_TARGET_FRAC*100:.2f}%)\n")

    per_level = []
    summary_rows = []
    for tag, floor_frac, seed in LEVELS:
        noise, diag = build_noise(clean_mat, times, chans, p_spike, ref, floor_frac, seed)
        noise_flat = noise[row_t, row_c]

        df_noisy = df_clean.copy()
        df_noisy[VALUE_COL] = df_clean[VALUE_COL].to_numpy() + noise_flat
        for col in ("misfit_values", "simulation_values"):
            if col in df_noisy.columns:
                df_noisy[col] = 0.0

        out_csv = os.path.join(HERE, f"measurement_data_rfsdss_{tag}.csv")
        df_noisy.to_csv(out_csv, index=False)

        meta = dict(base_meta)
        meta.update({
            "noise_model": "rfsdss_realistic",
            "noise_reference": "median_channel_peak",
            "ref_value": f"{ref:.10e}",
            "level_tag": tag,
            "floor_frac": f"{floor_frac}",
            "floor_mean": f"{diag['floor_mean']:.10e}",
            "drift_std": f"{diag['drift_std']:.10e}",
            "common_std": f"{diag['common_std']:.10e}",
            "spike_scale": f"{diag['spike_amp']:.10e}",
            "spike_driver": "strain_magnitude_x_spatial",
            "spike_center_y": f"{SPIKE_CENTER_Y}",
            "spike_gamma": f"{SPIKE_GAMMA}",
            "spike_spatial_wmin": f"{SPIKE_SPATIAL_WMIN}",
            "spike_target_frac": f"{SPIKE_TARGET_FRAC}",
            "seed": f"{seed}",
        })
        with open(os.path.join(HERE, f"measurement_data_rfsdss_{tag}.meta"), "w") as f:
            for key, val in meta.items():
                f.write(f"{key}={val}\n")

        total_noise_std = float(noise.std())
        n_spikes = int(diag["spike_mask"].sum())
        r = {"tag": tag, "floor_frac": floor_frac, "floor_mean": diag["floor_mean"],
             "total_noise_std": total_noise_std, "diag": diag}
        per_level.append(r)
        summary_rows.append({
            "tag": tag,
            "floor_pct": floor_frac * 100.0,
            "seed": seed,
            "ref_value": ref,
            "floor_mean": diag["floor_mean"],
            "drift_std_target": diag["drift_std"],
            "common_std_target": diag["common_std"],
            "spike_scale": diag["spike_amp"],
            "white_std_realized": float(diag["white"].std()),
            "drift_std_realized": float(diag["drift"].std()),
            "common_std_realized": float(diag["common"].std()),
            "n_spikes": n_spikes,
            "spike_fraction": float(diag["spike_mask"].mean()),
            "total_noise_std": total_noise_std,
            "snr_medchanpeak": ref / total_noise_std if total_noise_std > 0 else np.inf,
            "file": os.path.basename(out_csv),
        })

        out_png = os.path.join(HERE, f"rfsdss_noise_qc_{tag}.png")
        build_qc_figure(df_clean, clean_mat, diag, p_spike, spatial, ref, tag, out_png)
        print(f"[{tag:>6}] floor={floor_frac*100:>5.1f}%  floor_mean="
              f"{diag['floor_mean']*1e9:6.0f} nε  total_std={total_noise_std*1e9:6.0f} nε  "
              f"n_spikes={n_spikes:5d}  -> {os.path.basename(out_csv)}")

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(HERE, "rfsdss_noise_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote noise summary     : {summary_path}")

    out_png = os.path.join(HERE, "rfsdss_noise_levels_summary.png")
    build_levels_summary_figure(df_clean, per_level, ref, out_png)
    print(f"Wrote levels summary fig: {out_png}")


if __name__ == "__main__":
    main()
