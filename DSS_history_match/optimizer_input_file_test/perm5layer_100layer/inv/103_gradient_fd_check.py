"""
Finite-difference gradient verification for the adjoint-based inversion.

Perturbs selected layers one at a time, runs two forward solves per layer
(alpha +/- epsilon), and compares the central-difference gradient against
the adjoint gradient from MOOSE.

Usage:
    python 103_gradient_fd_check.py
"""

import os
import re
import numpy as np
import pandas as pd
from fiberis.moose.runner import MooseRunner

# ── Configuration ──────────────────────────────────────────────────────────
WORKDIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(WORKDIR, "optimize.i")
OUTPUT_DIR = WORKDIR
TOTAL_LAYERS = 200
NUM_PROCESSORS = 20
EPSILON = 0.5  # Perturbation size in alpha (log10) space

# Layers to check — chosen to span different zones:
#   Layer  50 (y=-24.75): inversion region edge, caprock
#   Layer  65 (y=-17.25): inside low-perm SRV zone
#   Layer 100 (y=  0.25): center, caprock
#   Layer 115 (y=  7.75): where adjoint gradient is most negative
#   Layer 135 (y= 17.75): inside high-perm SRV zone
TEST_LAYERS = [50, 65, 100, 115, 135]

# ── MOOSE runner ───────────────────────────────────────────────────────────
runner = MooseRunner(
    moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/combined/combined-opt",
    mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
)

with open(INPUT_FILE, "r") as f:
    base_moose_content = f.read()


def run_moose_and_get_objective_and_gradient(alpha_vector, tag=""):
    """Run MOOSE with the given alpha vector and return (objective, gradient_array)."""
    param_str = "; ".join([f"{v:.15e}" for v in alpha_vector])
    new_content = re.sub(
        r"(initial_condition\s*=\s*')[^']*(')",
        r"\g<1>" + param_str + r"\g<2>",
        base_moose_content,
        count=1
    )

    temp_input = os.path.join(WORKDIR, f"fd_check_{tag}.i")
    with open(temp_input, "w") as f:
        f.write(new_content)

    obj_csv = os.path.join(OUTPUT_DIR, f"fd_check_{tag}_out.csv")
    grad_csv = os.path.join(OUTPUT_DIR, f"fd_check_{tag}_out_OptimizationReporter_0001.csv")
    for p in [obj_csv, grad_csv]:
        if os.path.exists(p):
            os.remove(p)

    success, stdout, stderr = runner.run(
        input_file_path=temp_input,
        output_directory=OUTPUT_DIR,
        num_processors=NUM_PROCESSORS,
        log_file_name=f"fd_check_{tag}.log",
        stream_output=True,
        clean_output_dir=False
    )

    if not success:
        raise RuntimeError(f"MOOSE run failed for tag={tag}")

    obj_df = pd.read_csv(obj_csv)
    obj_val = float(obj_df["OptimizationReporter/objective_value"].iloc[-1])

    grad_array = None
    if os.path.exists(grad_csv):
        grad_df = pd.read_csv(grad_csv)
        grad_cols = [f"grad_perm_{i+1}" for i in range(TOTAL_LAYERS)]
        grad_array = grad_df[grad_cols].iloc[-1].values

    # Clean up temp files
    for p in [temp_input, obj_csv, grad_csv]:
        if os.path.exists(p):
            os.remove(p)

    return obj_val, grad_array


if __name__ == "__main__":
    alpha0 = np.full(TOTAL_LAYERS, -18.0)
    layer_height = 0.5

    # ── Step 1: Get adjoint gradient at the base point ─────────────────────
    print("=" * 60)
    print("Step 1: Computing adjoint objective & gradient at base point")
    print("=" * 60)
    obj0, adjoint_grad = run_moose_and_get_objective_and_gradient(alpha0, tag="base")
    print(f"Base objective: {obj0:.10e}")

    # ── Step 2: Finite-difference gradient for selected layers ─────────────
    print("\n" + "=" * 60)
    print(f"Step 2: Finite-difference check (epsilon = {EPSILON})")
    print("=" * 60)

    results = []
    for layer_idx in TEST_LAYERS:
        y_center = -50.0 + (layer_idx + 0.5) * layer_height
        print(f"\n--- Layer {layer_idx} (y = {y_center:.2f}) ---")

        # Forward perturbation: alpha_i + epsilon
        alpha_plus = alpha0.copy()
        alpha_plus[layer_idx] += EPSILON
        obj_plus, _ = run_moose_and_get_objective_and_gradient(alpha_plus, tag=f"plus_{layer_idx}")
        print(f"  J(alpha + eps) = {obj_plus:.10e}")

        # Backward perturbation: alpha_i - epsilon
        alpha_minus = alpha0.copy()
        alpha_minus[layer_idx] -= EPSILON
        obj_minus, _ = run_moose_and_get_objective_and_gradient(alpha_minus, tag=f"minus_{layer_idx}")
        print(f"  J(alpha - eps) = {obj_minus:.10e}")

        # Central difference
        fd_grad = (obj_plus - obj_minus) / (2.0 * EPSILON)
        adj_grad = adjoint_grad[layer_idx]

        # Relative error (use max of magnitudes to avoid division by near-zero)
        denom = max(abs(fd_grad), abs(adj_grad), 1e-30)
        rel_err = abs(fd_grad - adj_grad) / denom

        results.append({
            "layer": layer_idx,
            "y_center": y_center,
            "fd_grad": fd_grad,
            "adjoint_grad": adj_grad,
            "rel_error": rel_err,
        })
        print(f"  FD gradient:      {fd_grad:.10e}")
        print(f"  Adjoint gradient: {adj_grad:.10e}")
        print(f"  Relative error:   {rel_err:.4f} ({rel_err*100:.2f}%)")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Layer':>6}  {'y':>7}  {'FD grad':>14}  {'Adjoint grad':>14}  {'Rel err':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['layer']:>6}  {r['y_center']:>7.2f}  {r['fd_grad']:>14.6e}  {r['adjoint_grad']:>14.6e}  {r['rel_error']:>9.4f}")

    print("\nInterpretation:")
    print("  Rel error < 0.05 (5%)  : gradient is correct")
    print("  Rel error 0.05 - 0.20  : acceptable, may need smaller epsilon")
    print("  Rel error > 0.20 (20%) : gradient is likely WRONG")
