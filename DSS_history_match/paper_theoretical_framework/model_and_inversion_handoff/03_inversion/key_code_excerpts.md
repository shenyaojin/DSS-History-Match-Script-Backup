# 原始 Python 代码摘录

所有代码逐行摘自 `original/106_optimization_runner_L1.py`，未执行或导入原脚本。以下片段用于阅读，不是独立程序。原脚本顶层会清理历史文件，阅读时不要通过 import 加载。

## 设置与默认超参数

原文件第 27–50 行。

```python
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
BETA_L1 = float(os.environ.get("BETA_L1", "2e-11"))
DELTA_L1 = float(os.environ.get("DELTA_L1", "0.05"))

RUN_MODE = os.environ.get("L1_RUN_MODE", "optimize").strip().lower()
MAXITER = int(os.environ.get("L1_MAXITER", "300"))
NUM_PROCESSORS = int(os.environ.get("L1_NUM_PROCESSORS", "20"))
FTOL = float(os.environ.get("L1_FTOL", "1e-7"))
GTOL = float(os.environ.get("L1_GTOL", "1e-10"))
STEP_L2_TOL = float(os.environ.get("L1_STEP_L2_TOL", "1.2e-2"))
STEP_INF_TOL = float(os.environ.get("L1_STEP_INF_TOL", "6e-3"))
STEP_PATIENCE = int(os.environ.get("L1_STEP_PATIENCE", "3"))
STEP_MIN_ACCEPTED = int(os.environ.get("L1_STEP_MIN_ACCEPTED", "5"))
```

## layer_bounds_y

原文件第 122–126 行。

```python
def layer_bounds_y():
    y_bottom = -50.0 + np.arange(TOTAL_LAYERS) * LAYER_HEIGHT
    y_top = y_bottom + LAYER_HEIGHT
    y_center = 0.5 * (y_bottom + y_top)
    return y_bottom, y_top, y_center
```

## initial_zone_masks

原文件第 129–134 行。

```python
def initial_zone_masks():
    y_bottom, y_top, y_center = layer_bounds_y()
    free_window = (-25.0 <= y_center) & (y_center <= 25.0)
    low_perm_srv = (y_bottom >= -20.0) & (y_top <= -16.0)
    fracture = (y_bottom >= 14.0) & (y_top <= 20.0)
    return y_bottom, y_top, y_center, free_window, low_perm_srv, fracture
```

## build_initial_alpha

原文件第 137–143 行。

```python
def build_initial_alpha():
    alpha = np.full(TOTAL_LAYERS, FIXED_OUTSIDE_ALPHA)
    _, _, _, free_window, low_perm_srv, fracture = initial_zone_masks()
    alpha[free_window] = MATRIX_INIT_ALPHA
    alpha[low_perm_srv] = LOW_SRV_INIT_ALPHA
    alpha[fracture] = FRACTURE_INIT_ALPHA
    return alpha
```

## make_bounds

原文件第 146–154 行。

```python
def make_bounds():
    _, _, y_center = layer_bounds_y()
    bounds = []
    for yc in y_center:
        if -25.0 <= yc <= 25.0:
            bounds.append((-25.0, -10.0))
        else:
            bounds.append((FIXED_OUTSIDE_ALPHA, FIXED_OUTSIDE_ALPHA))
    return bounds
```

## l1_obj_and_grad

原文件第 187–192 行。

```python
def l1_obj_and_grad(x):
    deviation = x - ALPHA_REFERENCE
    denom = np.sqrt(deviation * deviation + DELTA_L1 * DELTA_L1)
    obj = BETA_L1 * np.sum(denom - DELTA_L1)
    grad = BETA_L1 * deviation / denom
    return obj, grad
```

## objective_and_gradient

原文件第 201–306 行。

```python
def objective_and_gradient(x):
    global iteration_count, best_data_obj, best_total_obj
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
    new_moose_content = new_moose_content.replace(
        "'../forward_and_adjoint.i'",
        "'../forward_and_adjoint_L1_quiet.i'",
    )
    new_moose_content = quiet_moose_content(new_moose_content)

    temp_input_path = os.path.join(WORKDIR, "optimize_temp_L1.i")
    with open(temp_input_path, "w") as f:
        f.write(new_moose_content)
    with open(QUIET_FORWARD_FILE, "w") as f:
        f.write(quiet_moose_content(fwd_content))

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
        stream_output=False,
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

        if obj_val < best_data_obj:
            best_data_obj = obj_val
            np.savetxt(BEST_DATA_ALPHA_FILE, x)
            with open(BEST_HISTORY_FILE, "a") as f:
                f.write(f"data,{iteration_count},{obj_val:.10e},{reg_obj:.10e},{total_obj:.10e}\n")
            print(f"New best data-fit alpha saved to: {BEST_DATA_ALPHA_FILE}")

        if total_obj < best_total_obj:
            best_total_obj = total_obj
            np.savetxt(BEST_TOTAL_ALPHA_FILE, x)
            with open(BEST_HISTORY_FILE, "a") as f:
                f.write(f"total,{iteration_count},{obj_val:.10e},{reg_obj:.10e},{total_obj:.10e}\n")
            print(f"New best total-objective alpha saved to: {BEST_TOTAL_ALPHA_FILE}")

        return scaled_obj, scaled_grad

    except Exception as e:
        print(f"Error reading MOOSE output: {e}")
        return 1e10, np.zeros_like(x)
```

## _checkpoint

原文件第 333–380 行。

```python
    def _checkpoint(xk):
        global accepted_iteration_count
        global previous_accepted_alpha
        global small_step_count
        global stopped_by_step_tolerance

        accepted_iteration_count += 1
        np.save(CHECKPOINT_FILE, xk)

        if previous_accepted_alpha is None:
            previous_accepted_alpha = xk.copy()
            with open(STEP_HISTORY_FILE, "a") as f:
                f.write(f"{accepted_iteration_count},nan,nan,0\n")
            return

        step = xk - previous_accepted_alpha
        step_l2 = float(np.linalg.norm(step))
        step_linf = float(np.max(np.abs(step)))
        previous_accepted_alpha = xk.copy()

        if step_l2 <= STEP_L2_TOL and step_linf <= STEP_INF_TOL:
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
            f"||dx||={step_l2:.4e}, max|dx|={step_linf:.4e}, "
            f"small_step_count={small_step_count}/{STEP_PATIENCE}"
        )

        if (
            accepted_iteration_count >= STEP_MIN_ACCEPTED
            and small_step_count >= STEP_PATIENCE
        ):
            stopped_by_step_tolerance = True
            np.savetxt(STEP_STOP_ALPHA_FILE, xk)
            print(
                "Stopping: accepted L-BFGS-B steps are below movement "
                f"tolerance for {STEP_PATIENCE} consecutive accepted iterations."
            )
            raise StopIteration
```

## L-BFGS-B 调用

原文件第 382–395 行。

```python
    try:
        res = minimize(
            objective_and_gradient,
            x0,
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
```
