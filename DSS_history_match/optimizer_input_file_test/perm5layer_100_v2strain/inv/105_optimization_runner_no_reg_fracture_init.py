# L-BFGS-B outer driver for the strain_yy permeability inversion, no reg.
#
# This case is intended to test the data term alone from a biased initial model.
# Inside the free observation window, the initial model has two explicit SRV
# areas:
#   - matrix between the SRV areas: alpha = -20
#   - low-perm SRV interval: alpha = -18
#   - high-perm SRV / fracture interval: alpha = -16
#
# Outside the free observation window, layers stay fixed at alpha = -18.
#
# The objective returned to SciPy is only the MOOSE OptimizationData misfit.
# No Tikhonov smoothness, no prior anchor, and no TV term are added.
#
# Run as:
#   python 105_optimization_runner_no_reg_fracture_init.py
#
# One-probe check at the initial model:
#   NO_REG_RUN_MODE=init_probe python 105_optimization_runner_no_reg_fracture_init.py

import os
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from fiberis.moose.runner import MooseRunner


# --- Paths --------------------------------------------------------------------
WORKDIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(WORKDIR, "optimize.i")
OUTPUT_DIR = os.path.join(WORKDIR, "inv_output")

# --- Objective / gradient scaling for L-BFGS-B --------------------------------
SCALE_FACTOR = 1e6
BASELINE_OBJ = 0.0

# --- Adjoint gradient correction ---------------------------------------------
# The custom C++ VPP already includes rho/mu in the adjoint inner product.
GRAD_CORRECTION = 1.0

TOTAL_LAYERS = 200
LAYER_HEIGHT = 0.5
FIXED_OUTSIDE_ALPHA = -18.0
MATRIX_INIT_ALPHA = float(os.environ.get("NO_REG_MATRIX_INIT_ALPHA", "-20.0"))
LOW_SRV_INIT_ALPHA = -18.0
FRACTURE_INIT_ALPHA = -16.0

# Run modes:
#   optimize   - run full L-BFGS-B from the seeded initial model.
#   init_probe - evaluate obj+grad once at the seeded initial model and exit.
RUN_MODE = os.environ.get("NO_REG_RUN_MODE", "optimize").strip().lower()
MAXITER = int(os.environ.get("NO_REG_MAXITER", "300"))

print(f"Working directory : {WORKDIR}")
print(f"MOOSE cwd         : {OUTPUT_DIR}")
print("Regularization    : none")
print(f"Matrix init alpha : {MATRIX_INIT_ALPHA}")
print(f"Low SRV init alpha: {LOW_SRV_INIT_ALPHA}")
print(f"Frac init alpha   : {FRACTURE_INIT_ALPHA}")
print(f"Run mode          : {RUN_MODE}")
print(f"Max iterations    : {MAXITER}")

# --- Sanity check -------------------------------------------------------------
with open(INPUT_FILE, "r") as f:
    base_moose_content = f.read()
if "measurement_data.csv" not in base_moose_content:
    raise RuntimeError(f"{INPUT_FILE} does not reference measurement_data.csv.")
fwd_path = os.path.join(WORKDIR, "forward_and_adjoint.i")
with open(fwd_path, "r") as f:
    fwd_content = f.read()
if "variable = 'strain_yy'" not in fwd_content:
    raise RuntimeError(
        f"{fwd_path} does not declare OptimizationData.variable='strain_yy'. "
        f"Refusing to run: this script assumes the strain_yy observation channel."
    )

runner = MooseRunner(
    moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/combined/combined-opt",
    mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec",
)

# --- History files ------------------------------------------------------------
RUN_TAG = "no_reg_fracture_init"
HISTORY_FILE = os.path.join(WORKDIR, f"parameter_history_{RUN_TAG}.csv")
GRADIENT_HISTORY_FILE = os.path.join(WORKDIR, f"gradient_history_{RUN_TAG}.csv")
OBJECTIVE_HISTORY_FILE = os.path.join(WORKDIR, f"objective_history_{RUN_TAG}.csv")
CHECKPOINT_FILE = os.path.join(WORKDIR, f"checkpoint_alpha_{RUN_TAG}.npy")
INITIAL_ALPHA_FILE = os.path.join(WORKDIR, f"initial_alpha_{RUN_TAG}.txt")
INITIAL_ZONE_FILE = os.path.join(WORKDIR, f"initial_zones_{RUN_TAG}.csv")
OPTIMIZED_ALPHA_FILE = os.path.join(WORKDIR, f"optimized_alphas_{RUN_TAG}.txt")

for p in (HISTORY_FILE, GRADIENT_HISTORY_FILE, OBJECTIVE_HISTORY_FILE):
    if os.path.exists(p):
        os.remove(p)
with open(OBJECTIVE_HISTORY_FILE, "w") as f:
    f.write("iter,obj_raw,obj_total,obj_scaled,grad_norm_scaled\n")

iteration_count = 0


def layer_bounds_y():
    y_bottom = -50.0 + np.arange(TOTAL_LAYERS) * LAYER_HEIGHT
    y_top = y_bottom + LAYER_HEIGHT
    y_center = 0.5 * (y_bottom + y_top)
    return y_bottom, y_top, y_center


def initial_zone_masks():
    y_bottom, y_top, y_center = layer_bounds_y()
    free_window = (-25.0 <= y_center) & (y_center <= 25.0)
    low_perm_srv = (y_bottom >= -20.0) & (y_top <= -16.0)
    fracture = (y_bottom >= 14.0) & (y_top <= 20.0)
    return y_bottom, y_top, y_center, free_window, low_perm_srv, fracture


def build_initial_alpha():
    """Initial model with two explicit SRV areas in the free window."""
    alpha = np.full(TOTAL_LAYERS, FIXED_OUTSIDE_ALPHA)
    _, _, _, free_window, low_perm_srv, fracture = initial_zone_masks()

    alpha[free_window] = MATRIX_INIT_ALPHA
    alpha[low_perm_srv] = LOW_SRV_INIT_ALPHA
    alpha[fracture] = FRACTURE_INIT_ALPHA
    return alpha


def make_bounds():
    """Free layers inside the observation window; fixed background outside."""
    _, _, y_center = layer_bounds_y()
    bounds = []
    for yc in y_center:
        if -25.0 <= yc <= 25.0:
            bounds.append((-25.0, -10.0))
        else:
            bounds.append((FIXED_OUTSIDE_ALPHA, FIXED_OUTSIDE_ALPHA))
    return bounds


def summarize_alpha(name, alpha):
    _, _, _, free_window, low_perm_srv, fracture = initial_zone_masks()
    active = np.where(np.abs(alpha - FIXED_OUTSIDE_ALPHA) > 1e-12)[0]
    free_layers = np.where(free_window)[0]
    low_layers = np.where(low_perm_srv)[0]
    frac_layers = np.where(fracture)[0]

    print(f"{name}: min={alpha.min():.6f}, max={alpha.max():.6f}, active_layers={len(active)}")
    print(f"{name}: free window 1-based layer range {free_layers[0] + 1}..{free_layers[-1] + 1}")
    print(
        f"{name}: low SRV 1-based layers {low_layers[0] + 1}..{low_layers[-1] + 1}, "
        f"alpha={LOW_SRV_INIT_ALPHA:.6f}"
    )
    print(
        f"{name}: fracture 1-based layers {frac_layers[0] + 1}..{frac_layers[-1] + 1}, "
        f"alpha={FRACTURE_INIT_ALPHA:.6f}"
    )
    if len(active):
        print(f"{name}: active 1-based layer range {active[0] + 1}..{active[-1] + 1}")


def save_initial_zone_table(alpha):
    y_bottom, y_top, y_center, free_window, low_perm_srv, fracture = initial_zone_masks()
    zone = np.full(TOTAL_LAYERS, "fixed_outside", dtype=object)
    zone[free_window] = "matrix_init"
    zone[low_perm_srv] = "low_srv"
    zone[fracture] = "fracture"

    df = pd.DataFrame({
        "layer_1based": np.arange(1, TOTAL_LAYERS + 1),
        "y_bottom": y_bottom,
        "y_top": y_top,
        "y_center": y_center,
        "zone": zone,
        "alpha_initial": alpha,
    })
    df.to_csv(INITIAL_ZONE_FILE, index=False)


def objective_and_gradient(x):
    global iteration_count
    iteration_count += 1

    print(f"\n--- Iter {iteration_count}: evaluating obj+grad for {len(x)} layers ---")

    with open(HISTORY_FILE, "a") as f:
        f.write(",".join(f"{v:.10e}" for v in x) + "\n")

    param_str = "; ".join(f"{v:.15e}" for v in x)
    new_moose_content = re.sub(
        r"(initial_condition\s*=\s*')[^']*(')",
        r"\1" + param_str + r"\2",
        base_moose_content,
        count=1,
    )

    temp_input_path = os.path.join(WORKDIR, "optimize_temp_no_reg.i")
    with open(temp_input_path, "w") as f:
        f.write(new_moose_content)

    obj_csv = os.path.join(OUTPUT_DIR, "optimize_temp_no_reg_out.csv")
    grad_csv = os.path.join(OUTPUT_DIR, "optimize_temp_no_reg_out_OptimizationReporter_0001.csv")
    log_path = os.path.join(OUTPUT_DIR, "simulation_opt_no_reg.log")
    for p in (obj_csv, grad_csv, log_path):
        if os.path.exists(p):
            os.remove(p)

    success, _, _ = runner.run(
        input_file_path=temp_input_path,
        output_directory=OUTPUT_DIR,
        num_processors=20,
        log_file_name="simulation_opt_no_reg.log",
        stream_output=True,
        clean_output_dir=False,
    )
    if not success:
        print("MOOSE run failed; penalizing this step.")
        return 1e10, np.zeros_like(x)

    try:
        # --- Objective (CSV preferred, log fallback) ---
        obj_val = None
        if os.path.exists(obj_csv):
            obj_df = pd.read_csv(obj_csv)
            if len(obj_df) > 1:
                obj_val = float(obj_df["OptimizationReporter/objective_value"].iloc[-1])
        if obj_val is None or obj_val == 0.0:
            with open(log_path, "r") as lf:
                for line in lf:
                    m = re.search(r"Objective value\s*=\s*([0-9eE.+\-]+)", line)
                    if m:
                        obj_val = float(m.group(1))
        if obj_val is None:
            raise RuntimeError("Could not parse objective value from MOOSE output")

        # --- Raw per-layer gradient ---
        grad_df = pd.read_csv(grad_csv)
        grad_cols = [f"grad_perm_{i+1}" for i in range(TOTAL_LAYERS)]
        grad_array = grad_df[grad_cols].iloc[-1].values.copy()
        grad_array *= GRAD_CORRECTION

        total_obj = float(obj_val)
        total_grad = grad_array

        scaled_obj = (total_obj - BASELINE_OBJ) * SCALE_FACTOR
        scaled_grad = total_grad * SCALE_FACTOR

        print(f"Obj raw: {obj_val:.4e} | Reg: 0.0000e+00 | Total: {total_obj:.4e}")
        print(f"Obj scaled: {scaled_obj:.4f}")
        print(f"||grad|| raw={np.linalg.norm(total_grad):.4e}  "
              f"scaled={np.linalg.norm(scaled_grad):.4e}")

        with open(GRADIENT_HISTORY_FILE, "a") as f:
            f.write(",".join(f"{v:.10e}" for v in scaled_grad) + "\n")

        with open(OBJECTIVE_HISTORY_FILE, "a") as f:
            f.write(
                f"{iteration_count},{obj_val:.10e},{total_obj:.10e},"
                f"{scaled_obj:.10e},{np.linalg.norm(scaled_grad):.10e}\n"
            )

        return scaled_obj, scaled_grad

    except Exception as e:
        print(f"Error reading MOOSE output: {e}")
        return 1e10, np.zeros_like(x)


if __name__ == "__main__":
    if RUN_MODE not in {"optimize", "init_probe"}:
        raise RuntimeError("NO_REG_RUN_MODE must be either 'optimize' or 'init_probe'.")

    x0 = build_initial_alpha()
    np.savetxt(INITIAL_ALPHA_FILE, x0)
    save_initial_zone_table(x0)
    summarize_alpha("Initial alpha", x0)
    print(f"Initial alpha saved to: {INITIAL_ALPHA_FILE}")
    print(f"Initial zone table saved to: {INITIAL_ZONE_FILE}")

    bounds = make_bounds()

    if RUN_MODE == "init_probe":
        print("Evaluating objective/gradient once at the seeded initial model...")
        obj, grad = objective_and_gradient(x0)
        print(f"Initial probe scaled objective: {obj:.10e}")
        print(f"Initial probe scaled grad norm: {np.linalg.norm(grad):.10e}")
        temp_i = os.path.join(WORKDIR, "optimize_temp_no_reg.i")
        if os.path.exists(temp_i):
            os.remove(temp_i)
        raise SystemExit(0)

    print(f"Starting L-BFGS-B optimization with {TOTAL_LAYERS} parameters...")

    def _checkpoint(xk):
        np.save(CHECKPOINT_FILE, xk)

    res = minimize(
        objective_and_gradient,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        callback=_checkpoint,
        options={
            "maxiter": MAXITER,
            "ftol": 1e-12,
            "gtol": 1e-10,
            "iprint": 1,
        },
    )

    print("\n" + "=" * 50)
    print("Optimization Result Summary (No Reg, Fracture Init):")
    print("=" * 50)
    print(res.message)
    print(f"Success       : {res.success}")
    print(f"Final Objective: {res.fun}")

    np.savetxt(OPTIMIZED_ALPHA_FILE, res.x)
    print(f"Optimized alphas saved to: {OPTIMIZED_ALPHA_FILE}")

    temp_i = os.path.join(WORKDIR, "optimize_temp_no_reg.i")
    if os.path.exists(temp_i):
        os.remove(temp_i)
