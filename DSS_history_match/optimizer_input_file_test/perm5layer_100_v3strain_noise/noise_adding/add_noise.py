# Generate noisy copies of the clean strain_yy observation file for the
# L1 noise-sensitivity study (v3).
#
# Noise model (decided 2026-06-08):
#   For each level p in {0.5%, 1%, 2%, 5%}, the added noise is i.i.d. Gaussian
#   with the SAME absolute standard deviation at every measurement point:
#
#       std_p = p * max_i |d_i|
#       d_noisy_i = d_i + N(0, std_p^2)
#
#   where d is the clean `measurement_values` column and max|d| is the peak
#   absolute strain amplitude. Noise is added to ALL points (including the
#   t=0 baseline / zero rows), so the perturbation is independent of the local
#   signal level. Each level uses its own deterministic seed so the datasets
#   are reproducible and the noise realizations are independent across levels.
#
# Run as:
#   python add_noise.py
#
# Outputs (written next to this script, in noise_adding/):
#   measurement_data_clean.csv            copy of the clean observation
#   measurement_data_noise_<tag>.csv      one per noise level
#   measurement_data_noise_<tag>.meta     metadata mirroring obs_strain_yy.meta
#   noise_summary.csv                     requested vs realized noise statistics

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN_CSV = os.path.join(HERE, "..", "data", "obs_strain_yy.csv")
CLEAN_META = os.path.join(HERE, "..", "data", "obs_strain_yy.meta")
VALUE_COL = "measurement_values"

# (tag, fraction, seed). Tags use "p" for the decimal point so they are
# filesystem-friendly (0p5pct == 0.5%).
NOISE_LEVELS = [
    ("0p5pct", 0.005, 20260608),
    ("1pct", 0.010, 20260609),
    ("2pct", 0.020, 20260610),
    ("5pct", 0.050, 20260611),
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


def write_meta(path, base_meta, noise_std):
    meta = dict(base_meta)
    meta["noise_std"] = f"{noise_std:.10e}"
    with open(path, "w") as f:
        for key, val in meta.items():
            f.write(f"{key}={val}\n")


def main():
    clean_path = os.path.abspath(CLEAN_CSV)
    if not os.path.exists(clean_path):
        raise FileNotFoundError(clean_path)

    df_clean = pd.read_csv(clean_path)
    if VALUE_COL not in df_clean.columns:
        raise RuntimeError(f"{clean_path} has no '{VALUE_COL}' column.")

    clean_values = df_clean[VALUE_COL].to_numpy(dtype=float)
    n = clean_values.size
    max_abs = float(np.max(np.abs(clean_values)))
    rms = float(np.sqrt(np.mean(clean_values ** 2)))

    print(f"Clean observation : {clean_path}")
    print(f"n points          : {n}")
    print(f"max|d| (peak)     : {max_abs:.6e}")
    print(f"rms(d)            : {rms:.6e}")
    print("")

    base_meta = read_clean_meta()

    # Keep a labeled clean copy alongside the noisy files for convenience.
    clean_copy = os.path.join(HERE, "measurement_data_clean.csv")
    df_clean.to_csv(clean_copy, index=False)
    write_meta(os.path.join(HERE, "measurement_data_clean.meta"), base_meta, 0.0)
    print(f"Wrote clean copy  : {clean_copy}")
    print("")

    summary_rows = []
    for tag, frac, seed in NOISE_LEVELS:
        target_std = frac * max_abs
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=target_std, size=n)

        noisy = df_clean.copy()
        noisy[VALUE_COL] = clean_values + noise
        # The reporter recomputes misfit/simulation during the solve; keep these
        # columns zeroed so the file only carries the (noisy) observation.
        for col in ("misfit_values", "simulation_values"):
            if col in noisy.columns:
                noisy[col] = 0.0

        out_csv = os.path.join(HERE, f"measurement_data_noise_{tag}.csv")
        noisy.to_csv(out_csv, index=False)
        write_meta(
            os.path.join(HERE, f"measurement_data_noise_{tag}.meta"),
            base_meta,
            target_std,
        )

        realized_std = float(np.std(noise))
        realized_max = float(np.max(np.abs(noise)))
        snr_peak = max_abs / target_std if target_std > 0 else np.inf
        snr_rms = rms / target_std if target_std > 0 else np.inf
        summary_rows.append(
            {
                "tag": tag,
                "noise_pct": frac * 100.0,
                "seed": seed,
                "target_std": target_std,
                "realized_std": realized_std,
                "realized_abs_max": realized_max,
                "snr_peak": snr_peak,
                "snr_rms": snr_rms,
                "file": os.path.basename(out_csv),
            }
        )
        print(
            f"[{tag:>6}] pct={frac*100:>4.1f}%  std={target_std:.4e}  "
            f"realized_std={realized_std:.4e}  SNR_peak={snr_peak:6.1f}  "
            f"-> {os.path.basename(out_csv)}"
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(HERE, "noise_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("")
    print(f"Wrote noise summary: {summary_path}")


if __name__ == "__main__":
    main()
