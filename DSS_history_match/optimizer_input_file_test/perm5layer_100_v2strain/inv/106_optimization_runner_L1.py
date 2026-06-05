# L-BFGS-B outer driver for the strain_yy permeability inversion, L1 variant.
#
# Regularization:
#   J_L1(alpha) = BETA_L1 * Sum_i |alpha_i - ALPHA_REFERENCE|
#
# For L-BFGS-B this is smoothed as:
#   |z| ~= sqrt(z^2 + DELTA_L1^2) - DELTA_L1
#
# Run as:
#   python 106_optimization_runner_L1.py

import os
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from fiberis.moose.runner import MooseRunner


WORKDIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(WORKDIR, "optimize.i")
OUTPUT_DIR = os.path.join(WORKDIR, "inv_output")

SCALE_FACTOR = 1e6
BASELINE_OBJ = 0.0
GRAD_CORRECTION = 1.0

TOTAL_LAYERS = 200
LAYER_HEIGHT = 0.5
FIXED_OUTSIDE_ALPHA = -18.0
MATRIX_INIT_ALPHA = float(os.environ.get("L1_MATRIX_INIT_ALPHA", "-18.0"))
LOW_SRV_INIT_ALPHA = -16.0
FRACTURE_INIT_ALPHA = -16.0

ALPHA_REFERENCE = -18.0
BETA_L1 = float(os.environ.get("BETA_L1", "1e-9"))
DELTA_L1 = float(os.environ.get("DELTA_L1", "0.05"))

RUN_MODE = os.environ.get("L1_RUN_MODE", "optimize").strip().lower()
MAXITER = int(os.environ.get("L1_MAXITER", "300"))
NUM_PROCESSORS = int(os.environ.get("L1_NUM_PROCESSORS", "20"))

print(f"Working directory : {WORKDIR}")
print(f"MOOSE cwd         : {OUTPUT_DIR}")
print(f"Regularization    : smoothed L1, beta = {BETA_L1}, delta = {DELTA_L1}")
print(f"Initial model     : matrix={MATRIX_INIT_ALPHA}, low SRV=-16, fracture=-16")
print(f"Run mode          : {RUN_MODE}")
print(f"Max iterations    : {MAXITER}")

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

RUN_TAG = "L1"
HISTORY_FILE = os.path.join(WORKDIR, "parameter_history_L1.csv")
GRADIENT_HISTORY_FILE = os.path.join(WORKDIR, "gradient_history_L1.csv")
OBJECTIVE_HISTORY_FILE = os.path.join(WORKDIR, "objective_history_L1.csv")
CHECKPOINT_FILE = os.path.join(WORKDIR, "checkpoint_alpha_L1.npy")
INITIAL_ALPHA_FILE = os.path.join(WORKDIR, "initial_alpha_L1.txt")
INITIAL_ZONE_FILE = os.path.join(WORKDIR, "initial_zones_L1.csv")
OPTIMIZED_ALPHA_FILE = os.path.join(WORKDIR, "optimized_alphas_L1.txt")

for p in (HISTORY_FILE, GRADIENT_HISTORY_FILE, OBJECTIVE_HISTORY_FILE):
    if os.path.exists(p):
        os.remove(p)
with open(OBJECTIVE_HISTORY_FILE, "w") as f:
    f.write("iter,obj_raw,reg_l1,obj_total,obj_scaled,grad_norm_scaled\n")

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
    alpha = np.full(TOTAL_LAYERS, FIXED_OUTSIDE_ALPHA)
    _, _, _, free_window, low_perm_srv, fracture = initial_zone_masks()
    alpha[free_window] = MATRIX_INIT_ALPHA
    alpha[low_perm_srv] = LOW_SRV_INIT_ALPHA
    alpha[fracture] = FRACTURE_INIT_ALPHA
    return alpha


def make_bounds():
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
    print(f"{name}: low SRV 1-based layers {low_layers[0] + 1}..{low_layers[-1] + 1}")
    print(f"{name}: fracture 1-based layers {frac_layers[0] + 1}..{frac_layers[-1] + 1}")
    if len(active):
        print(f"{name}: active 1-based layer range {active[0] + 1}..{active[-1] + 1}")


def save_initial_zone_table(alpha):
    y_bottom, y_top, y_center, free_window, low_perm_srv, fracture = initial_zone_masks()
    zone = np.full(TOTAL_LAYERS, "fixed_outside", dtype=object)
    zone[free_window] = "matrix_init"
    zone[low_perm_srv] = "low_srv"
    zone[fracture] = "fracture"
    pd.DataFrame({
        "layer_1based": np.arange(1, TOTAL_LAYERS + 1),
        "y_bottom": y_bottom,
        "y_top": y_top,
        "y_center": y_center,
        "zone": zone,
        "alpha_initial": alpha,
    }).to_csv(INITIAL_ZONE_FILE, index=False)


def l1_obj_and_grad(x):
    deviation = x - ALPHA_REFERENCE
    denom = np.sqrt(deviation * deviation + DELTA_L1 * DELTA_L1)
    obj = BETA_L1 * np.sum(denom - DELTA_L1)
    grad = BETA_L1 * deviation / denom
    return obj, grad


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

    temp_input_path = os.path.join(WORKDIR, "optimize_temp_L1.i")
    with open(temp_input_path, "w") as f:
        f.write(new_moose_content)

    obj_csv = os.path.join(OUTPUT_DIR, "optimize_temp_L1_out.csv")
    grad_csv = os.path.join(OUTPUT_DIR, "optimize_temp_L1_out_OptimizationReporter_0001.csv")
    log_path = os.path.join(OUTPUT_DIR, "simulation_opt_L1.log")
    for p in (obj_csv, grad_csv, log_path):
        if os.path.exists(p):
            os.remove(p)

    success, _, _ = runner.run(
        input_file_path=temp_input_path,
        output_directory=OUTPUT_DIR,
        num_processors=NUM_PROCESSORS,
        log_file_name="simulation_opt_L1.log",
        stream_output=True,
        clean_output_dir=False,
    )
    if not success:
        print("MOOSE run failed; penalizing this step.")
        return 1e10, np.zeros_like(x)

    try:
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

        grad_df = pd.read_csv(grad_csv)
        grad_cols = [f"grad_perm_{i+1}" for i in range(TOTAL_LAYERS)]
        grad_array = grad_df[grad_cols].iloc[-1].values.copy()

        reg_obj, reg_grad = l1_obj_and_grad(x)
        grad_array *= GRAD_CORRECTION

        total_obj = float(obj_val) + reg_obj
        total_grad = grad_array + reg_grad
        scaled_obj = (total_obj - BASELINE_OBJ) * SCALE_FACTOR
        scaled_grad = total_grad * SCALE_FACTOR

        print(f"Obj raw: {obj_val:.4e} | Reg L1: {reg_obj:.4e} | Total: {total_obj:.4e}")
        print(f"Obj scaled: {scaled_obj:.4f}")
        print(f"||grad|| raw={np.linalg.norm(total_grad):.4e}  "
              f"scaled={np.linalg.norm(scaled_grad):.4e}")

        with open(GRADIENT_HISTORY_FILE, "a") as f:
            f.write(",".join(f"{v:.10e}" for v in scaled_grad) + "\n")
        with open(OBJECTIVE_HISTORY_FILE, "a") as f:
            f.write(
                f"{iteration_count},{obj_val:.10e},{reg_obj:.10e},"
                f"{total_obj:.10e},{scaled_obj:.10e},"
                f"{np.linalg.norm(scaled_grad):.10e}\n"
            )
        return scaled_obj, scaled_grad

    except Exception as e:
        print(f"Error reading MOOSE output: {e}")
        return 1e10, np.zeros_like(x)


if __name__ == "__main__":
    if RUN_MODE not in {"optimize", "init_probe"}:
        raise RuntimeError("L1_RUN_MODE must be either 'optimize' or 'init_probe'.")

    x0 = build_initial_alpha()
    np.savetxt(INITIAL_ALPHA_FILE, x0)
    save_initial_zone_table(x0)
    summarize_alpha("Initial alpha", x0)
    print(f"Initial alpha saved to: {INITIAL_ALPHA_FILE}")
    print(f"Initial zone table saved to: {INITIAL_ZONE_FILE}")

    bounds = make_bounds()

    if RUN_MODE == "init_probe":
        obj, grad = objective_and_gradient(x0)
        print(f"Initial probe scaled objective: {obj:.10e}")
        print(f"Initial probe scaled grad norm: {np.linalg.norm(grad):.10e}")
        temp_i = os.path.join(WORKDIR, "optimize_temp_L1.i")
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
    print("Optimization Result Summary (L1):")
    print("=" * 50)
    print(res.message)
    print(f"Success       : {res.success}")
    print(f"Final Objective: {res.fun}")

    np.savetxt(OPTIMIZED_ALPHA_FILE, res.x)
    print(f"Optimized alphas saved to: {OPTIMIZED_ALPHA_FILE}")

    temp_i = os.path.join(WORKDIR, "optimize_temp_L1.i")
    if os.path.exists(temp_i):
        os.remove(temp_i)
