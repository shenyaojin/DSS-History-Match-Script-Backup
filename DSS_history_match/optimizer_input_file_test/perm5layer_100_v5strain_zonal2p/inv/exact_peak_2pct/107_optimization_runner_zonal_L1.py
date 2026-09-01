# L-BFGS-B outer driver for the strain_yy permeability inversion:
# ZONAL 2-PARAMETER variant.
#
# Strategy (decided with advisor): the forward/adjoint MOOSE model is UNCHANGED.
# Instead of inverting ~100 free layers, we assume the anomaly geometry is KNOWN
# (fracture + low-SRV zones, e.g. read from DSS) and invert only TWO scalars:
#
#   theta = [theta_frac, theta_srv]
#
# theta_frac is the single log10-permeability shared by every layer of the
# fracture zone; theta_srv is the single value shared by every layer of the
# low-SRV zone; every other layer is pinned at the background BACKGROUND_ALPHA.
#
# Because the forward model still exposes 200 per-layer parameters, we keep MOOSE
# untouched and bridge in Python:
#   * expand(theta) -> 200-dim alpha  (fracture layers = theta_frac,
#                                       srv layers = theta_srv, rest = -18)
#     is written into optimize.i's initial_condition each iteration;
#   * MOOSE returns 200 per-layer gradients grad_perm_i = dJ_data/d(alpha_i);
#   * reduce_grad sums them within each zone -> 2-dim gradient. This is exact:
#     since alpha_i == theta_zone for all i in a zone, d(alpha_i)/d(theta_zone)=1
#     so dJ/d(theta_zone) = sum_{i in zone} grad_perm_i (chain rule; verified
#     against the MOOSE OptimizationFunctionInnerProduct + ParsedOptimization
#     Function '10^alpha' — the 10^alpha*ln10 factor is applied MOOSE-side, so
#     grad_perm_i is already in alpha space and summing is correct).
#
# Regularization: smoothed L1, J_L1 = BETA_L1 * Sum_i |alpha_i - ALPHA_REFERENCE|
# computed on the expanded 200-vector then reduced to theta (only the 12 frac +
# 8 srv layers deviate from the reference -18, so it is nearly negligible here;
# kept for consistency with the layered L1 study).
#
# Two zone geometries are supported for the noise-robustness study:
#   ZONAL_MODE=exact -> zones exactly match the synthetic truth
#   ZONAL_MODE=pert  -> zone y-windows shifted by ZONAL_PERT_SHIFT_LAYERS layers
#                       (mimics reading the fracture region from DSS with error)
#
# Run as:
#   python 107_optimization_runner_zonal_L1.py

import os
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from fiberis.moose.runner import MooseRunner


WORKDIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(WORKDIR, "optimize.i")
QUIET_FORWARD_FILE = os.path.join(WORKDIR, "forward_and_adjoint_zonal_quiet.i")
OUTPUT_DIR = os.path.join(WORKDIR, "inv_output")

SCALE_FACTOR = float(os.environ.get("ZONAL_SCALE_FACTOR", "1e6"))
BASELINE_OBJ = 0.0
GRAD_CORRECTION = 1.0

TOTAL_LAYERS = 200          # kept: MOOSE still has 200 per-layer parameters
LAYER_HEIGHT = 0.5
BACKGROUND_ALPHA = -18.0    # value pinned on every layer outside the two zones

# --- theta space ---
THETA_FRAC0 = float(os.environ.get("ZONAL_FRAC_INIT", "-16.0"))
THETA_SRV0 = float(os.environ.get("ZONAL_SRV_INIT", "-16.0"))
THETA_LOWER = -25.0         # must stay within MOOSE per-layer bounds (optimize.i)
THETA_UPPER = -10.0

# truth targets (for logging only): low_srv=-15, fracture=log10(3e-15)
THETA_FRAC_TRUTH = float(np.log10(3e-15))
THETA_SRV_TRUTH = -15.0

# --- zone geometry (exact vs perturbed) ---
FRAC_Y = (14.0, 20.0)       # fracture zone y-window (truth)
SRV_Y = (-20.0, -16.0)      # low-SRV zone y-window (truth)
ZONAL_MODE = os.environ.get("ZONAL_MODE", "exact").strip().lower()
PERT_SHIFT_LAYERS = int(os.environ.get("ZONAL_PERT_SHIFT_LAYERS", "2"))
FRAC_SHIFT = int(os.environ.get("ZONAL_FRAC_SHIFT", str(PERT_SHIFT_LAYERS)))
SRV_SHIFT = int(os.environ.get("ZONAL_SRV_SHIFT", str(PERT_SHIFT_LAYERS)))

# --- L1 regularization (values from the layered L1 study; semantics re-derived) ---
ALPHA_REFERENCE = -18.0
BETA_L1 = float(os.environ.get("BETA_L1", "2e-11"))
DELTA_L1 = float(os.environ.get("DELTA_L1", "0.05"))

# --- convergence (2-dim: converges fast) ---
RUN_MODE = os.environ.get("ZONAL_RUN_MODE", "optimize").strip().lower()
MAXITER = int(os.environ.get("ZONAL_MAXITER", "60"))
NUM_PROCESSORS = int(os.environ.get("ZONAL_NP", "20"))
FTOL = float(os.environ.get("ZONAL_FTOL", "1e-7"))
GTOL = float(os.environ.get("ZONAL_GTOL", "1e-10"))

# --- step-movement stop (off by default in 2-D; the L2 rule is redundant here) ---
USE_STEP_STOP = os.environ.get("ZONAL_USE_STEP_STOP", "0") == "1"
STEP_INF_TOL = float(os.environ.get("ZONAL_STEP_INF_TOL", "6e-3"))
STEP_PATIENCE = int(os.environ.get("ZONAL_STEP_PATIENCE", "2"))
STEP_MIN_ACCEPTED = int(os.environ.get("ZONAL_STEP_MIN_ACCEPTED", "3"))


def layer_bounds_y():
    y_bottom = -50.0 + np.arange(TOTAL_LAYERS) * LAYER_HEIGHT
    y_top = y_bottom + LAYER_HEIGHT
    y_center = 0.5 * (y_bottom + y_top)
    return y_bottom, y_top, y_center


def active_zone_windows():
    """Return (frac_lo, frac_hi), (srv_lo, srv_hi) for the current ZONAL_MODE.

    In 'pert' mode the y-windows are shifted by an integer number of layers
    (so the layer COUNT is preserved and exact/pert stay comparable).
    """
    if ZONAL_MODE == "pert":
        fs = FRAC_SHIFT * LAYER_HEIGHT
        ss = SRV_SHIFT * LAYER_HEIGHT
        return (FRAC_Y[0] + fs, FRAC_Y[1] + fs), (SRV_Y[0] + ss, SRV_Y[1] + ss)
    return FRAC_Y, SRV_Y


def zone_indices(y_lo, y_hi):
    y_bottom, y_top, _ = layer_bounds_y()
    return np.where((y_bottom >= y_lo) & (y_top <= y_hi))[0]


(_FR_LO, _FR_HI), (_SR_LO, _SR_HI) = active_zone_windows()
FRAC_IDX = zone_indices(_FR_LO, _FR_HI)
SRV_IDX = zone_indices(_SR_LO, _SR_HI)

if len(FRAC_IDX) == 0 or len(SRV_IDX) == 0:
    raise RuntimeError(
        f"Empty zone(s): fracture y={( _FR_LO,_FR_HI)} -> {len(FRAC_IDX)} layers, "
        f"srv y={(_SR_LO,_SR_HI)} -> {len(SRV_IDX)} layers. Check ZONAL_MODE/shift."
    )

RUN_TAG = "zonal"
# --- self-describing theta outputs ---
THETA_HISTORY_FILE = os.path.join(WORKDIR, "theta_history_zonal.csv")
GRADIENT_HISTORY_FILE = os.path.join(WORKDIR, "gradient_history_zonal.csv")
OBJECTIVE_HISTORY_FILE = os.path.join(WORKDIR, "objective_history_zonal.csv")
BEST_HISTORY_FILE = os.path.join(WORKDIR, "best_history_zonal.csv")
STEP_HISTORY_FILE = os.path.join(WORKDIR, "step_history_zonal.csv")
CHECKPOINT_FILE = os.path.join(WORKDIR, "checkpoint_theta_zonal.npy")
INITIAL_THETA_FILE = os.path.join(WORKDIR, "initial_theta_zonal.txt")
OPTIMIZED_THETA_FILE = os.path.join(WORKDIR, "optimized_theta_zonal.txt")
BEST_DATA_THETA_FILE = os.path.join(WORKDIR, "best_data_theta_zonal.txt")
BEST_TOTAL_THETA_FILE = os.path.join(WORKDIR, "best_total_theta_zonal.txt")
STEP_STOP_THETA_FILE = os.path.join(WORKDIR, "step_stop_theta_zonal.txt")
# --- 200-dim expanded outputs (compat with compare / plot_inversion_qc / QC) ---
HISTORY_FILE = os.path.join(WORKDIR, "parameter_history_L1.csv")
INITIAL_ALPHA_FILE = os.path.join(WORKDIR, "initial_alpha_L1.txt")
INITIAL_ZONE_FILE = os.path.join(WORKDIR, "initial_zones_L1.csv")
OPTIMIZED_ALPHA_FILE = os.path.join(WORKDIR, "optimized_alphas_L1.txt")
BEST_DATA_ALPHA_FILE = os.path.join(WORKDIR, "best_data_alpha_L1.txt")
BEST_TOTAL_ALPHA_FILE = os.path.join(WORKDIR, "best_total_alpha_L1.txt")

print(f"Working directory : {WORKDIR}")
print(f"MOOSE cwd         : {OUTPUT_DIR}")
print(f"Parameterization  : ZONAL 2-parameter [theta_frac, theta_srv]")
print(f"Zone mode         : {ZONAL_MODE}"
      + (f" (shift frac={FRAC_SHIFT}, srv={SRV_SHIFT} layers)" if ZONAL_MODE == "pert" else ""))
print(f"Fracture zone     : y in ({_FR_LO}, {_FR_HI})  -> "
      f"1-based layers {FRAC_IDX[0]+1}..{FRAC_IDX[-1]+1} ({len(FRAC_IDX)} layers)")
print(f"Low-SRV zone      : y in ({_SR_LO}, {_SR_HI})  -> "
      f"1-based layers {SRV_IDX[0]+1}..{SRV_IDX[-1]+1} ({len(SRV_IDX)} layers)")
print(f"theta targets     : frac={THETA_FRAC_TRUTH:.6f}, srv={THETA_SRV_TRUTH:.6f}")
print(f"Regularization    : smoothed L1, beta = {BETA_L1}, delta = {DELTA_L1}")
print(f"Run mode          : {RUN_MODE}   max iterations: {MAXITER}")
print(f"SciPy stop        : ftol = {FTOL}, gtol = {GTOL}")
print(f"Step stop         : {'ON' if USE_STEP_STOP else 'OFF'} "
      f"(Linf <= {STEP_INF_TOL}, patience {STEP_PATIENCE}, min accepted {STEP_MIN_ACCEPTED})")

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

for p in (
    THETA_HISTORY_FILE,
    HISTORY_FILE,
    GRADIENT_HISTORY_FILE,
    OBJECTIVE_HISTORY_FILE,
    BEST_HISTORY_FILE,
    STEP_HISTORY_FILE,
):
    if os.path.exists(p):
        os.remove(p)
with open(OBJECTIVE_HISTORY_FILE, "w") as f:
    f.write("iter,obj_raw,reg_l1,obj_total,obj_scaled,grad_norm_scaled\n")
with open(BEST_HISTORY_FILE, "w") as f:
    f.write("kind,iter,obj_raw,reg_l1,obj_total\n")
with open(STEP_HISTORY_FILE, "w") as f:
    f.write("accepted_iter,step_l2,step_linf,small_step_count\n")

iteration_count = 0
best_data_obj = np.inf
best_total_obj = np.inf
accepted_iteration_count = 0
previous_accepted_theta = None
small_step_count = 0
stopped_by_step_tolerance = False


def expand(theta):
    """theta=[theta_frac, theta_srv] -> 200-dim alpha (zones tied, rest = -18)."""
    alpha = np.full(TOTAL_LAYERS, BACKGROUND_ALPHA)
    alpha[FRAC_IDX] = theta[0]
    alpha[SRV_IDX] = theta[1]
    return alpha


def reduce_grad(grad200):
    """Sum per-layer gradients within each zone (d alpha_i / d theta_zone = 1)."""
    return np.array([grad200[FRAC_IDX].sum(), grad200[SRV_IDX].sum()])


def l1_zonal(theta):
    """Smoothed L1 on the expanded 200-vector, reduced to a 2-dim theta grad."""
    alpha = expand(theta)
    deviation = alpha - ALPHA_REFERENCE
    denom = np.sqrt(deviation * deviation + DELTA_L1 * DELTA_L1)
    reg_obj = BETA_L1 * np.sum(denom - DELTA_L1)
    reg_grad_full = BETA_L1 * deviation / denom
    return reg_obj, reduce_grad(reg_grad_full)


def make_bounds_theta():
    return [(THETA_LOWER, THETA_UPPER), (THETA_LOWER, THETA_UPPER)]


def summarize_theta(name, theta):
    alpha = expand(theta)
    active = np.where(np.abs(alpha - BACKGROUND_ALPHA) > 1e-12)[0]
    print(f"{name}: theta_frac={theta[0]:.6f} (target {THETA_FRAC_TRUTH:.4f}), "
          f"theta_srv={theta[1]:.6f} (target {THETA_SRV_TRUTH:.4f})")
    print(f"{name}: expanded alpha min={alpha.min():.6f}, max={alpha.max():.6f}, "
          f"active_layers={len(active)}")


def save_active_zone_table(alpha):
    y_bottom, y_top, y_center = layer_bounds_y()
    zone = np.full(TOTAL_LAYERS, "background", dtype=object)
    zone[FRAC_IDX] = "fracture"
    zone[SRV_IDX] = "low_srv"
    with open(INITIAL_ZONE_FILE, "w") as f:
        f.write("layer_1based,y_bottom,y_top,y_center,zone,alpha_initial\n")
        for i in range(TOTAL_LAYERS):
            f.write(
                f"{i+1},{y_bottom[i]:.4f},{y_top[i]:.4f},{y_center[i]:.4f},"
                f"{zone[i]},{alpha[i]:.10e}\n"
            )


def quiet_moose_content(content):
    content = re.sub(r"(\bverbose\s*=\s*)true\b", r"\1false", content)
    content = re.sub(r"(\bconsole\s*=\s*)true\b", r"\1false", content)
    return content


def objective_and_gradient(theta):
    global iteration_count, best_data_obj, best_total_obj
    iteration_count += 1
    alpha = expand(theta)
    print(f"\n--- Iter {iteration_count}: theta = [{theta[0]:.6f}, {theta[1]:.6f}] ---")

    # 200-dim expanded alpha history (for run_parameter_history_qc.py reuse)
    with open(HISTORY_FILE, "a") as f:
        f.write(",".join(f"{v:.10e}" for v in alpha) + "\n")
    # self-describing 2-dim theta history
    with open(THETA_HISTORY_FILE, "a") as f:
        f.write(",".join(f"{v:.10e}" for v in theta) + "\n")

    param_str = "; ".join(f"{v:.15e}" for v in alpha)
    new_moose_content = re.sub(
        r"(initial_condition\s*=\s*')[^']*(')",
        r"\1" + param_str + r"\2",
        base_moose_content,
        count=1,
    )
    new_moose_content = new_moose_content.replace(
        "'../forward_and_adjoint.i'",
        "'../forward_and_adjoint_zonal_quiet.i'",
    )
    new_moose_content = quiet_moose_content(new_moose_content)

    temp_input_path = os.path.join(WORKDIR, "optimize_temp_zonal.i")
    with open(temp_input_path, "w") as f:
        f.write(new_moose_content)
    with open(QUIET_FORWARD_FILE, "w") as f:
        f.write(quiet_moose_content(fwd_content))

    obj_csv = os.path.join(OUTPUT_DIR, "optimize_temp_zonal_out.csv")
    grad_csv = os.path.join(OUTPUT_DIR, "optimize_temp_zonal_out_OptimizationReporter_0001.csv")
    log_path = os.path.join(OUTPUT_DIR, "simulation_opt_zonal.log")
    for p in (obj_csv, grad_csv, log_path):
        if os.path.exists(p):
            os.remove(p)

    success, _, _ = runner.run(
        input_file_path=temp_input_path,
        output_directory=OUTPUT_DIR,
        num_processors=NUM_PROCESSORS,
        log_file_name="simulation_opt_zonal.log",
        stream_output=False,
        clean_output_dir=False,
    )
    if not success:
        print("MOOSE run failed; penalizing this step.")
        return 1e10, np.zeros_like(theta)

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
        grad200 = grad_df[grad_cols].iloc[-1].values.copy()
        grad200 *= GRAD_CORRECTION

        grad_data_theta = reduce_grad(grad200)          # data-misfit gradient, 2-dim
        reg_obj, reg_grad_theta = l1_zonal(theta)       # L1 gradient, 2-dim

        total_obj = float(obj_val) + reg_obj
        total_grad = grad_data_theta + reg_grad_theta
        scaled_obj = (total_obj - BASELINE_OBJ) * SCALE_FACTOR
        scaled_grad = total_grad * SCALE_FACTOR

        print(f"Obj raw: {obj_val:.4e} | Reg L1: {reg_obj:.4e} | Total: {total_obj:.4e}")
        print(f"grad_theta (data) = [{grad_data_theta[0]:.4e}, {grad_data_theta[1]:.4e}]  "
              f"scaled ||grad|| = {np.linalg.norm(scaled_grad):.4e}")

        with open(GRADIENT_HISTORY_FILE, "a") as f:
            f.write(",".join(f"{v:.10e}" for v in scaled_grad) + "\n")
        with open(OBJECTIVE_HISTORY_FILE, "a") as f:
            f.write(
                f"{iteration_count},{obj_val:.10e},{reg_obj:.10e},"
                f"{total_obj:.10e},{scaled_obj:.10e},"
                f"{np.linalg.norm(scaled_grad):.10e}\n"
            )

        if obj_val < best_data_obj:
            best_data_obj = obj_val
            np.savetxt(BEST_DATA_THETA_FILE, theta)
            np.savetxt(BEST_DATA_ALPHA_FILE, expand(theta))
            with open(BEST_HISTORY_FILE, "a") as f:
                f.write(f"data,{iteration_count},{obj_val:.10e},{reg_obj:.10e},{total_obj:.10e}\n")
            print(f"New best data-fit theta saved: {theta}")

        if total_obj < best_total_obj:
            best_total_obj = total_obj
            np.savetxt(BEST_TOTAL_THETA_FILE, theta)
            np.savetxt(BEST_TOTAL_ALPHA_FILE, expand(theta))
            with open(BEST_HISTORY_FILE, "a") as f:
                f.write(f"total,{iteration_count},{obj_val:.10e},{reg_obj:.10e},{total_obj:.10e}\n")
            print(f"New best total-objective theta saved: {theta}")

        return scaled_obj, scaled_grad

    except Exception as e:
        print(f"Error reading MOOSE output: {e}")
        return 1e10, np.zeros_like(theta)


if __name__ == "__main__":
    if RUN_MODE not in {"optimize", "init_probe"}:
        raise RuntimeError("ZONAL_RUN_MODE must be either 'optimize' or 'init_probe'.")

    theta0 = np.array([THETA_FRAC0, THETA_SRV0])
    np.savetxt(INITIAL_THETA_FILE, theta0)
    np.savetxt(INITIAL_ALPHA_FILE, expand(theta0))
    save_active_zone_table(expand(theta0))
    summarize_theta("Initial theta", theta0)
    print(f"Initial theta saved to: {INITIAL_THETA_FILE}")
    print(f"Initial zone table saved to: {INITIAL_ZONE_FILE}")

    bounds = make_bounds_theta()

    if RUN_MODE == "init_probe":
        obj, grad = objective_and_gradient(theta0)
        print(f"Initial probe scaled objective: {obj:.10e}")
        print(f"Initial probe scaled grad norm: {np.linalg.norm(grad):.10e}")
        temp_i = os.path.join(WORKDIR, "optimize_temp_zonal.i")
        if os.path.exists(temp_i):
            os.remove(temp_i)
        raise SystemExit(0)

    print("Starting L-BFGS-B optimization with 2 zonal parameters...")

    def _checkpoint(xk):
        global accepted_iteration_count
        global previous_accepted_theta
        global small_step_count
        global stopped_by_step_tolerance

        accepted_iteration_count += 1
        np.save(CHECKPOINT_FILE, xk)

        if previous_accepted_theta is None:
            previous_accepted_theta = xk.copy()
            with open(STEP_HISTORY_FILE, "a") as f:
                f.write(f"{accepted_iteration_count},nan,nan,0\n")
            return

        step = xk - previous_accepted_theta
        step_l2 = float(np.linalg.norm(step))
        step_linf = float(np.max(np.abs(step)))
        previous_accepted_theta = xk.copy()

        # 2-D: the L2 rule is redundant (L2 <= sqrt(2)*Linf), so gate on Linf only.
        if step_linf <= STEP_INF_TOL:
            small_step_count += 1
        else:
            small_step_count = 0

        with open(STEP_HISTORY_FILE, "a") as f:
            f.write(
                f"{accepted_iteration_count},{step_l2:.10e},"
                f"{step_linf:.10e},{small_step_count}\n"
            )

        print(
            f"Accepted step {accepted_iteration_count}: "
            f"||dtheta||={step_l2:.4e}, max|dtheta|={step_linf:.4e}, "
            f"small_step_count={small_step_count}/{STEP_PATIENCE}"
        )

        if (
            USE_STEP_STOP
            and accepted_iteration_count >= STEP_MIN_ACCEPTED
            and small_step_count >= STEP_PATIENCE
        ):
            stopped_by_step_tolerance = True
            np.savetxt(STEP_STOP_THETA_FILE, xk)
            print(
                "Stopping: accepted L-BFGS-B steps below movement tolerance "
                f"for {STEP_PATIENCE} consecutive accepted iterations."
            )
            raise StopIteration

    try:
        res = minimize(
            objective_and_gradient,
            theta0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            callback=_checkpoint,
            options={
                "maxiter": MAXITER,
                "ftol": FTOL,
                "gtol": GTOL,
            },
        )
    except StopIteration:
        class StepStopResult:
            pass

        res = StepStopResult()
        res.x = np.load(CHECKPOINT_FILE)
        res.fun = np.nan
        res.success = True
        res.message = "Stopped by accepted-step movement tolerance."

    print("\n" + "=" * 50)
    print("Optimization Result Summary (zonal 2-parameter L1):")
    print("=" * 50)
    print(res.message)
    print(f"Success       : {res.success}")
    print(f"Stopped by step tolerance: {stopped_by_step_tolerance}")
    print(f"Final Objective: {res.fun}")
    summarize_theta("Final theta", np.asarray(res.x))

    np.savetxt(OPTIMIZED_THETA_FILE, res.x)
    np.savetxt(OPTIMIZED_ALPHA_FILE, expand(np.asarray(res.x)))
    print(f"Optimized theta saved to : {OPTIMIZED_THETA_FILE}")
    print(f"Optimized alphas saved to: {OPTIMIZED_ALPHA_FILE}")

    temp_i = os.path.join(WORKDIR, "optimize_temp_zonal.i")
    if os.path.exists(temp_i):
        os.remove(temp_i)
