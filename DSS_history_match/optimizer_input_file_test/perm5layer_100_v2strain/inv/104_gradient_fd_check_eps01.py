"""
Tighter FD verification at eps = 0.1 (alpha space) for two layers.

Goal: disambiguate FD truncation vs remaining bias in the adjoint gradient.
At eps = 0.5 the central FD had ~18% bias against the adjoint (Adj/FD ~ 0.82).
At eps = 0.1, FD truncation O(eps^2) shrinks 25x. If Adj/FD moves toward 1.0,
the previous bias was FD truncation and the adjoint is exact. If it stays ~0.82,
there is a residual bias (likely from the strain-yy dipole approximation).

Adjoint gradient at base is reused from the previous eps=0.5 run, so this only
runs 4 forward-only-style MOOSE invocations (each invocation still does fwd+adj
because the executioner is TransientAndAdjoint; J is what we need from each).

Usage:
    python 104_gradient_fd_check_eps01.py
"""

import os
import re
import numpy as np
import pandas as pd
from fiberis.moose.runner import MooseRunner

WORKDIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(WORKDIR, "optimize.i")
OUTPUT_DIR = WORKDIR
TOTAL_LAYERS = 200
NUM_PROCESSORS = 20
EPSILON = 0.1
TEST_LAYERS = [65, 135]

# Adjoint gradient values at base (alpha = -18 everywhere) recovered from the
# eps=0.5 verification run on 2026-05-03.
ADJOINT_GRAD_AT_BASE = {
    50:  -2.015893e-08,
    65:  -3.673302e-08,
    100:  2.401989e-09,
    115: -1.542103e-08,
    135: -4.047876e-08,
}

runner = MooseRunner(
    moose_executable_path="/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/combined/combined-opt",
    mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec",
)

with open(INPUT_FILE, "r") as f:
    base_moose_content = f.read()


def run_and_get_objective(alpha_vector, tag):
    param_str = "; ".join([f"{v:.15e}" for v in alpha_vector])
    new_content = re.sub(
        r"(initial_condition\s*=\s*')[^']*(')",
        r"\g<1>" + param_str + r"\g<2>",
        base_moose_content,
        count=1,
    )

    # Rewrite '../foo' references to absolute paths (we run from inv/ directly,
    # not from inv/inv_output/ as the production runner does).
    new_content = new_content.replace(
        "'../forward_and_adjoint.i'",
        f"'{os.path.join(WORKDIR, 'forward_and_adjoint.i')}'",
    )
    new_content = new_content.replace(
        "'../measurement_data.csv'",
        f"'{os.path.join(WORKDIR, 'measurement_data.csv')}'",
    )

    temp_input = os.path.join(WORKDIR, f"fd_eps01_{tag}.i")
    with open(temp_input, "w") as f:
        f.write(new_content)

    success, _, _ = runner.run(
        input_file_path=temp_input,
        output_directory=OUTPUT_DIR,
        num_processors=NUM_PROCESSORS,
        log_file_name=f"fd_eps01_{tag}.log",
        stream_output=True,
        clean_output_dir=False,
    )
    if not success:
        raise RuntimeError(f"MOOSE run failed for tag={tag}")

    # Pull J from the TAO log line: 'iteration=0\tf=...'
    log = os.path.join(WORKDIR, f"fd_eps01_{tag}.log")
    with open(log, "r") as f:
        for line in f:
            if "iteration=0" in line and "f=" in line:
                # Format: 'TAO SOLVER:  iteration=0\tf=3.2588e-06\tgnorm=...'
                m = re.search(r"f=([0-9.eE+-]+)", line)
                if m:
                    obj_val = float(m.group(1))
                    break
        else:
            raise RuntimeError(f"Could not find objective in {log}")

    # Cleanup big artifacts to keep disk usage in check; keep log + main CSVs.
    for fname in os.listdir(WORKDIR):
        if fname.startswith(f"fd_eps01_{tag}_out") and (
            fname.endswith(".e") or fname.endswith(".csv")
        ):
            os.remove(os.path.join(WORKDIR, fname))
    if os.path.exists(temp_input):
        os.remove(temp_input)

    return obj_val


if __name__ == "__main__":
    alpha0 = np.full(TOTAL_LAYERS, -18.0)

    print("=" * 70)
    print(f"FD verification at eps = {EPSILON} for layers {TEST_LAYERS}")
    print("Reusing adjoint gradient at base from the eps=0.5 run.")
    print("=" * 70)

    results = []
    for layer_idx in TEST_LAYERS:
        print(f"\n--- Layer {layer_idx} ---")

        alpha_plus = alpha0.copy()
        alpha_plus[layer_idx] += EPSILON
        Jp = run_and_get_objective(alpha_plus, tag=f"plus_{layer_idx}")
        print(f"  J(alpha + eps) = {Jp:.10e}")

        alpha_minus = alpha0.copy()
        alpha_minus[layer_idx] -= EPSILON
        Jm = run_and_get_objective(alpha_minus, tag=f"minus_{layer_idx}")
        print(f"  J(alpha - eps) = {Jm:.10e}")

        fd_grad = (Jp - Jm) / (2.0 * EPSILON)
        adj_grad = ADJOINT_GRAD_AT_BASE[layer_idx]
        ratio = adj_grad / fd_grad if abs(fd_grad) > 1e-30 else float("nan")
        rel_err = abs(fd_grad - adj_grad) / max(abs(fd_grad), abs(adj_grad), 1e-30)

        results.append((layer_idx, Jp, Jm, fd_grad, adj_grad, ratio, rel_err))
        print(f"  FD gradient:      {fd_grad:.10e}")
        print(f"  Adjoint gradient: {adj_grad:.10e}")
        print(f"  Adjoint / FD:     {ratio:.6e}")
        print(f"  Relative error:   {rel_err:.4f}")

    print("\n" + "=" * 70)
    print(f"SUMMARY (eps = {EPSILON})")
    print("=" * 70)
    print(f"{'Layer':>6}  {'FD grad':>14}  {'Adjoint grad':>14}  {'Adj/FD':>12}  {'Rel err':>10}")
    print("-" * 70)
    for L, Jp, Jm, fd, adj, ratio, rel in results:
        print(f"{L:>6}  {fd:>14.6e}  {adj:>14.6e}  {ratio:>12.4e}  {rel:>9.4f}")

    print("\nInterpretation:")
    print(f"  At eps=0.5, Adj/FD was ~0.82 for these layers.")
    print(f"  If Adj/FD now -> 1.0, the previous gap was FD truncation (gradient exact).")
    print(f"  If Adj/FD stays ~0.82, there is a residual bias (likely the strain dipole h=0.5).")
