# scripts/tensile_fault/data_transfer/105_lfdas_npy_extractor.py
"""
Convert legacy LF-DAS ``.npy`` files (Gold 4-PB) to fibeRIS ``Data2D`` ``.npz`` format.

Source : data_fervo/legacy/LF-DAS NPY/NPY_G4-PB_MM_UTC_YYYYMM/*.npy
Output : data_fervo/fiberis_format/LFDAS/LFDAS_G4-PB_YYYYMM.npz

Each source ``.npy`` has shape ``(6, 2304)``:
  * rows    -> the 6 low-frequency samples inside the file's minute, taken at
              offsets [5, 15, 25, 35, 45, 55] s from the UTC timestamp in the
              filename (a uniform 10 s sample interval).
  * columns -> the 2304 fiber channels.

Processing (kept consistent with the reference notebook
``from_pc/LFDAS_NPY_Vis_07-13-2026.ipynb``):
  1. Raw counts are scaled to strain rate in nanostrain/s:  116 * fs / GL / 8192.
  2. The channel index is calibrated to Gold 4-PB measured depth (ft):
        MD = DEPTH_SLOPE * channel + DEPTH_LEAD
  3. UTC file times are shifted to project local time (UTC-7).
  4. Acquisition gaps (> 10 s) are filled with NaN columns so ``taxis`` stays
     on a regular grid (toggle with ``FILL_TIME_GAPS``).

The result is a plain ``fiberis.analyzer.Data2D.core2D.Data2D`` (NOT the DSS
subclass), saved with the standard keys ``data``, ``taxis``, ``daxis`` and
``start_time`` via ``Data2D.savez``.

Author: Shenyao Jin, shenyaojin@mines.edu
"""

import os
import glob
import datetime

import numpy as np
from tqdm import tqdm

from fiberis.analyzer.Data2D.core2D import Data2D

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SOURCE_ROOT = "data_fervo/legacy/LF-DAS NPY"
OUTPUT_DIR = "data_fervo/fiberis_format/LFDAS"
OUTPUT_PREFIX = "LFDAS_G4-PB"

# {output label : source sub-directory under SOURCE_ROOT}. One .npz per entry.
SOURCE_DIRS = {
    "202501": "NPY_G4-PB_MM_UTC_202501",
    "202502": "NPY_G4-PB_MM_UTC_202502",
}

# --- Acquisition constants (from the 07-13-2026 reference notebook) --------- #
TIME_SAMPLE_OFFSETS_S = np.arange(5, 60, 10)   # 6 samples/file, offsets in seconds
EXPECTED_TIME_STEP_S = 10.0                    # nominal LF-DAS sample interval [s]
FILL_TIME_GAPS = True                          # insert NaN columns across gaps > step
LOCAL_TIME_OFFSET_H = 7                         # UTC -> UTC-7 local; set 0 to keep UTC

# --- Strain-rate scale: raw counts -> nanostrain/s -------------------------- #
FS = 1000                                       # interrogator sample rate [Hz]
GL = 10                                         # gauge length [m]
STRAIN_RATE_SCALE = 116 * FS / GL / 8192        # == 116000 / 8192 nanostrain/s per count

# --- Channel index -> measured depth [ft] (Gold 4-PB well survey) ----------- #
BOWF, EOWF, BOWMD, EOWMD = 52, 2255, 15, 14807
DEPTH_SLOPE = (EOWMD - BOWMD) / (EOWF - BOWF)   # ft per channel
DEPTH_LEAD = BOWMD - DEPTH_SLOPE * BOWF         # intercept [ft]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_file_start(file_name: str) -> datetime.datetime:
    """Extract the UTC start time from a '...UTC_YYYYMMDD_HHMMSS.npy' filename."""
    stem = os.path.splitext(os.path.basename(file_name))[0]
    return datetime.datetime.strptime(stem.split("UTC_")[-1], "%Y%m%d_%H%M%S")


def load_directory(dir_path: str):
    """Load every ``.npy`` in ``dir_path`` into a (n_depth, n_time) strain-rate
    array and a matching array of local sample times (np.datetime64[s]).

    Returns
    -------
    data : np.ndarray, shape (n_channels, n_time)
    times_local : np.ndarray[datetime64[s]], shape (n_time,)
    """
    files = sorted(glob.glob(os.path.join(dir_path, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {dir_path}")

    data_blocks = []   # each (n_offsets, n_channels)
    time_blocks = []   # UTC sample times

    n_expected = len(TIME_SAMPLE_OFFSETS_S)
    local_offset = np.timedelta64(LOCAL_TIME_OFFSET_H, "h")

    for f in tqdm(files, desc=f"Loading {os.path.basename(dir_path)}"):
        arr = np.load(f)
        if arr.shape[0] != n_expected:
            print(f"  - Skipping {os.path.basename(f)}: expected {n_expected} "
                  f"time samples, got shape {arr.shape}.")
            continue

        t0 = parse_file_start(f)
        sample_times = [np.datetime64(t0, "s") + np.timedelta64(int(o), "s")
                        for o in TIME_SAMPLE_OFFSETS_S]

        data_blocks.append(arr.astype(float) * STRAIN_RATE_SCALE)
        time_blocks.extend(sample_times)

    # (n_time, n_channels) -> (n_channels, n_time)
    data = np.concatenate(data_blocks, axis=0).T
    times_utc = np.array(time_blocks, dtype="datetime64[s]")

    # Guarantee chronological order before shifting to local time.
    order = np.argsort(times_utc)
    data = data[:, order]
    times_local = times_utc[order] - local_offset

    return data, times_local


def fill_time_gaps(data: np.ndarray, times: np.ndarray,
                   expected_step_s: float = EXPECTED_TIME_STEP_S):
    """Insert NaN columns wherever consecutive samples are more than one
    ``expected_step_s`` apart, so the returned time axis is on a regular grid.

    Returns the gap-filled ``(data, times)`` and the number of inserted columns.
    """
    step = np.timedelta64(int(round(expected_step_s)), "s")
    n_depth = data.shape[0]

    filled_cols = [data[:, 0]]
    filled_times = [times[0]]
    n_inserted = 0

    for i in range(1, data.shape[1]):
        gap_s = (times[i] - times[i - 1]) / np.timedelta64(1, "s")
        n_missing = max(0, int(round(gap_s / expected_step_s)) - 1)
        for k in range(1, n_missing + 1):
            filled_times.append(times[i - 1] + k * step)
            filled_cols.append(np.full(n_depth, np.nan))
            n_inserted += 1
        filled_times.append(times[i])
        filled_cols.append(data[:, i])

    data_filled = np.column_stack(filled_cols)
    times_filled = np.array(filled_times, dtype="datetime64[s]")
    return data_filled, times_filled, n_inserted


def build_data2d(data: np.ndarray, times_local: np.ndarray, name: str) -> Data2D:
    """Assemble a Data2D object with all required fields populated."""
    start_time = times_local[0].astype("datetime64[us]").astype(datetime.datetime)
    taxis = (times_local - times_local[0]) / np.timedelta64(1, "s")
    taxis = taxis.astype(float)
    daxis = np.arange(data.shape[0]) * DEPTH_SLOPE + DEPTH_LEAD

    das = Data2D(
        data=data,
        taxis=taxis,
        daxis=daxis,
        start_time=start_time,
        name=name,
    )
    das.record_log(
        f"Converted from legacy LF-DAS .npy. strain_rate_scale={STRAIN_RATE_SCALE:.6f} "
        f"nanostrain/s per count; depth = {DEPTH_SLOPE:.6f}*ch + {DEPTH_LEAD:.4f} ft; "
        f"time zone = UTC-{LOCAL_TIME_OFFSET_H}.",
        level="INFO",
    )
    return das


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    output_path = os.path.abspath(OUTPUT_DIR)
    os.makedirs(output_path, exist_ok=True)
    print(f"Output directory: {output_path}")
    print(f"Strain-rate scale : {STRAIN_RATE_SCALE:.6f} nanostrain/s per count")
    print(f"Depth calibration : MD = {DEPTH_SLOPE:.6f} * channel + {DEPTH_LEAD:.4f} ft")
    print(f"Time zone         : UTC-{LOCAL_TIME_OFFSET_H}\n")

    for label, sub_dir in SOURCE_DIRS.items():
        dir_path = os.path.abspath(os.path.join(SOURCE_ROOT, sub_dir))
        if not os.path.isdir(dir_path):
            print(f"Warning: source directory not found, skipping: {dir_path}")
            continue

        print(f"=== Processing {label} ({sub_dir}) ===")
        data, times_local = load_directory(dir_path)
        print(f"  Loaded array: {data.shape[0]} channels x {data.shape[1]} samples")

        if FILL_TIME_GAPS:
            data, times_local, n_inserted = fill_time_gaps(data, times_local)
            print(f"  Filled {n_inserted} NaN column(s) across acquisition gaps "
                  f"-> {data.shape[1]} samples total")

        name = f"{OUTPUT_PREFIX}_{label}"
        das = build_data2d(data, times_local, name)

        out_file = os.path.join(output_path, f"{name}.npz")
        das.savez(out_file)
        print(f"  Saved -> {out_file}")

        # --- QC: reload and report -------------------------------------- #
        qc = Data2D()
        qc.load_npz(out_file)
        qc.print_info()
        finite = np.isfinite(qc.data)
        pct_nan = 100.0 * (1.0 - finite.mean())
        print(f"  End time    : {qc.get_end_time()}")
        print(f"  Depth range : {qc.daxis.min():.2f} .. {qc.daxis.max():.2f} ft")
        print(f"  Value range : {np.nanmin(qc.data):.3f} .. {np.nanmax(qc.data):.3f} "
              f"nanostrain/s  ({pct_nan:.2f}% NaN)\n")

    print("Done.")


if __name__ == "__main__":
    main()
