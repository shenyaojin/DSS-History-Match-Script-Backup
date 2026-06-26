# Generate noisy copies of the clean strain_yy observation file for the
# L1 noise-sensitivity study (v3).
#
# Two noise families are produced, both i.i.d. Gaussian with a single absolute
# standard deviation applied to EVERY measurement point (including the t=0
# baseline rows), so the perturbation is independent of the local signal level:
#
#     d_noisy_i = d_i + N(0, std^2),   std = pct * REF
#
#   "peak"  family : REF = max_i |d_i|                     (3.589e-05)
#                    levels 0.5%, 1%, 2%, 5%
#   "median" family: REF = median over receivers of each
#                    channel's peak |strain| over time     (2.262e-05)
#                    levels 1%, 2%, 5%, 10%
#
# The median family uses a robust "typical signal amplitude" (the median of the
# per-receiver peak strain) rather than median|d| over all points, because the
# latter is dominated by tiny early-time values and would give near-zero noise.
#
# Each (family, level) uses its own deterministic seed, so datasets are
# reproducible and noise realizations are independent.
#
# Run as:
#   python add_noise.py
#
# Outputs (written next to this script, in noise_adding/):
#   measurement_data_clean.csv               labeled copy of the clean obs
#   measurement_data_noise_<tag>.csv         peak family   (tag: 0p5pct,1pct,2pct,5pct)
#   measurement_data_mednoise_<tag>.csv      median family (tag: 1pct,2pct,5pct,10pct)
#   *.meta                                   metadata mirroring obs_strain_yy.meta
#   noise_summary.csv                        requested vs realized noise statistics

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN_CSV = os.path.join(HERE, "..", "data", "obs_strain_yy.csv")
CLEAN_META = os.path.join(HERE, "..", "data", "obs_strain_yy.meta")
VALUE_COL = "measurement_values"
CHANNEL_COL = "measurement_ycoord"

# (file_prefix, ref_kind, [(tag, fraction, seed), ...])
# ref_kind selects which amplitude statistic scales the noise std.
NOISE_FAMILIES = [
    (
        "measurement_data_noise",
        "peak",
        [
            ("0p5pct", 0.005, 20260608),
            ("1pct", 0.010, 20260609),
            ("2pct", 0.020, 20260610),
            ("5pct", 0.050, 20260611),
        ],
    ),
    (
        "measurement_data_mednoise",
        "median_channel_peak",
        [
            ("1pct", 0.010, 20260701),
            ("2pct", 0.020, 20260702),
            ("5pct", 0.050, 20260703),
            ("10pct", 0.100, 20260704),
        ],
    ),
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


def write_meta(path, base_meta, noise_std, ref_kind):
    meta = dict(base_meta)
    meta["noise_std"] = f"{noise_std:.10e}"
    meta["noise_reference"] = ref_kind
    with open(path, "w") as f:
        for key, val in meta.items():
            f.write(f"{key}={val}\n")


def compute_references(df):
    values = df[VALUE_COL].to_numpy(dtype=float)
    abs_values = np.abs(values)
    peak = float(abs_values.max())
    # Median over receivers of each channel's peak |strain| over time.
    per_channel_peak = df.assign(_a=abs_values).groupby(CHANNEL_COL)["_a"].max()
    median_channel_peak = float(np.median(per_channel_peak.to_numpy()))
    return {"peak": peak, "median_channel_peak": median_channel_peak}


def main():
    clean_path = os.path.abspath(CLEAN_CSV)
    if not os.path.exists(clean_path):
        raise FileNotFoundError(clean_path)

    df_clean = pd.read_csv(clean_path)
    if VALUE_COL not in df_clean.columns:
        raise RuntimeError(f"{clean_path} has no '{VALUE_COL}' column.")

    clean_values = df_clean[VALUE_COL].to_numpy(dtype=float)
    n = clean_values.size
    refs = compute_references(df_clean)
    peak = refs["peak"]

    print(f"Clean observation     : {clean_path}")
    print(f"n points              : {n}")
    print(f"max|d| (peak)         : {refs['peak']:.6e}")
    print(f"median per-chan peak  : {refs['median_channel_peak']:.6e}")
    print("")

    base_meta = read_clean_meta()

    # Keep a labeled clean copy alongside the noisy files for convenience.
    clean_copy = os.path.join(HERE, "measurement_data_clean.csv")
    df_clean.to_csv(clean_copy, index=False)
    write_meta(os.path.join(HERE, "measurement_data_clean.meta"), base_meta, 0.0, "none")
    print(f"Wrote clean copy      : {clean_copy}")
    print("")

    summary_rows = []
    for prefix, ref_kind, levels in NOISE_FAMILIES:
        ref_value = refs[ref_kind]
        print(f"=== family '{prefix}'  (ref={ref_kind}={ref_value:.4e}) ===")
        for tag, frac, seed in levels:
            target_std = frac * ref_value
            rng = np.random.default_rng(seed)
            noise = rng.normal(loc=0.0, scale=target_std, size=n)

            noisy = df_clean.copy()
            noisy[VALUE_COL] = clean_values + noise
            for col in ("misfit_values", "simulation_values"):
                if col in noisy.columns:
                    noisy[col] = 0.0

            out_csv = os.path.join(HERE, f"{prefix}_{tag}.csv")
            noisy.to_csv(out_csv, index=False)
            write_meta(
                os.path.join(HERE, f"{prefix}_{tag}.meta"),
                base_meta,
                target_std,
                ref_kind,
            )

            realized_std = float(np.std(noise))
            snr_peak = peak / target_std if target_std > 0 else np.inf
            summary_rows.append(
                {
                    "family": ref_kind,
                    "tag": tag,
                    "noise_pct": frac * 100.0,
                    "seed": seed,
                    "ref_value": ref_value,
                    "target_std": target_std,
                    "realized_std": realized_std,
                    "snr_globalpeak": snr_peak,
                    "file": os.path.basename(out_csv),
                }
            )
            print(
                f"  [{tag:>6}] pct={frac*100:>5.1f}%  std={target_std:.4e}  "
                f"realized_std={realized_std:.4e}  SNR_peak={snr_peak:6.1f}  "
                f"-> {os.path.basename(out_csv)}"
            )
        print("")

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(HERE, "noise_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote noise summary   : {summary_path}")


if __name__ == "__main__":
    main()
