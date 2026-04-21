# Extract strain_yy observations from the ground-truth MOOSE run and write them
# as a measurement CSV for the fibeRIS inversion driver.
#
# The forward run (101_ground_truth.py) emits two line samplers:
#   - observation_disp   -> disp_y     (legacy observation channel)
#   - observation_strain -> strain_yy  (the channel we now use)
#
# Samplers are sorted alphabetically by the MOOSE VPP reader, so:
#   post_processor_id=0 -> observation_disp
#   post_processor_id=1 -> observation_strain
# For observation_strain the CSV columns are [id, strain_yy, x, y, z], so the
# variable column is at index 1.
#
# NOTE: MOOSE's OptimizationReporter reads the column named `measurement_values`
# by name. We therefore keep the on-disk column headers identical to the legacy
# disp_y schema and encode the change of observation type in the file name
# (obs_strain_yy.csv) and in the accompanying provenance sidecar.

import os

import numpy as np
import pandas as pd

from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader


# --- Paths -----------------------------------------------------------------
BASE_DIR = "scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100_v2strain"
GT_OUTPUT_DIR = os.path.join(BASE_DIR, "fwd", "output_gt")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "obs_strain_yy.csv")

# --- Sampler selection -----------------------------------------------------
STRAIN_POST_PROCESSOR_ID = 1  # observation_strain (sorted after observation_disp)
STRAIN_VARIABLE_INDEX = 1     # column `strain_yy` in [id, strain_yy, x, y, z]

# --- Fiber geometry --------------------------------------------------------
FIBER_X = 60.0
FIBER_Z = 0.0
# Sampler start is (60, -25, 0); daxis from the reader is distance-from-start
# along the sampler line, i.e. 0..50. Subtract 25 to recover the true y-coord.
DAXIS_OFFSET = -25.0

# --- Noise model -----------------------------------------------------------
# strain_yy is several orders of magnitude smaller than disp_y, so the disp_y
# noise level is not applicable. Keep this parameterised so it can be tuned.
NOISE_STD = 0.0   # absolute strain units (m/m). Set >0 to inject Gaussian noise.
NOISE_SEED = 0    # set to an int for reproducibility


def _load_strain_yy(directory: str) -> "pd.DataFrame":
    reader = MOOSEVectorPostProcessorReader()
    reader.read(directory,
                post_processor_id=STRAIN_POST_PROCESSOR_ID,
                variable_index=STRAIN_VARIABLE_INDEX)

    if reader.variable_name != "strain_yy":
        raise RuntimeError(
            f"Expected variable 'strain_yy' but reader returned "
            f"'{reader.variable_name}'. Check sampler ordering / column layout."
        )

    analyzer = reader.to_analyzer()
    # Shift daxis so that it equals the physical y-coordinate of each sample.
    analyzer.daxis = analyzer.daxis + DAXIS_OFFSET
    # First time step is the IC (all zeros by construction); keep as zero to
    # mirror the legacy disp_y extractor and avoid fitting transient artefacts.
    analyzer.data[:, 0] = 0.0
    return analyzer


def _add_noise(values: np.ndarray, std: float, seed: int) -> np.ndarray:
    if std <= 0.0:
        return values
    rng = np.random.default_rng(seed)
    return values + rng.normal(loc=0.0, scale=std, size=values.shape)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    strain_data = _load_strain_yy(GT_OUTPUT_DIR)
    print(strain_data)

    n_depth, n_time = strain_data.data.shape  # (depth, time)

    # Flatten in Fortran order so (time, depth) indices line up with the
    # repeated time / tiled depth axes below.
    times = np.repeat(strain_data.taxis, n_depth)
    y_coords = np.tile(strain_data.daxis, n_time)
    x_coords = np.full_like(times, FIBER_X)
    z_coords = np.full_like(times, FIBER_Z)

    values = strain_data.data.flatten(order="F")
    values = _add_noise(values, NOISE_STD, NOISE_SEED)

    df = pd.DataFrame({
        "measurement_time":   times,
        "measurement_values": values,
        "measurement_xcoord": x_coords,
        "measurement_ycoord": y_coords,
        "measurement_zcoord": z_coords,
        "misfit_values":      values,   # Ground truth: misfit == sim == obs at t=0
        "simulation_values":  values,
    })

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Observation CSV saved to: {OUTPUT_CSV}")
    print(f"  shape (time, depth) = ({n_time}, {n_depth})")
    print(f"  variable = strain_yy, noise_std = {NOISE_STD}")

    # Provenance sidecar so downstream readers can confirm which physical
    # quantity is stored in the `measurement_values` column.
    meta_path = os.path.splitext(OUTPUT_CSV)[0] + ".meta"
    with open(meta_path, "w") as f:
        f.write("observation_variable=strain_yy\n")
        f.write(f"n_time={n_time}\n")
        f.write(f"n_receivers={n_depth}\n")
        f.write(f"noise_std={NOISE_STD}\n")
    print(f"Meta file saved to: {meta_path}")


if __name__ == "__main__":
    main()
