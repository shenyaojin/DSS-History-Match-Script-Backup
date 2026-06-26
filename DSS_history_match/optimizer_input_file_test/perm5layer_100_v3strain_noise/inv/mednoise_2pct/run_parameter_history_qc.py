import argparse
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from fiberis.moose.runner import MooseRunner


WORKDIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(WORKDIR, "optimize.i")
FORWARD_FILE = os.path.join(WORKDIR, "forward_and_adjoint.i")

MOOSE_EXE = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/combined/combined-opt"
MPIEXEC = "/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"


def read_history_row(history_file, row_index):
    data = np.loadtxt(history_file, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[row_index].astype(float)


def replace_initial_condition(text, alpha):
    param_str = "; ".join(f"{v:.15e}" for v in alpha)
    text, count = re.subn(
        r"(initial_condition\s*=\s*')[^']*(')",
        r"\1" + param_str + r"\2",
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not replace exactly one initial_condition block.")
    return text


def make_parent_input(label, alpha):
    with open(INPUT_FILE, "r") as f:
        text = f.read()
    text = replace_initial_condition(text, alpha)
    text = re.sub(
        r"(input_files\s*=\s*')\.\./forward_and_adjoint\.i(')",
        rf"\1../qc_forward_{label}.i\2",
        text,
        count=1,
    )
    path = os.path.join(WORKDIR, f"qc_opt_{label}.i")
    with open(path, "w") as f:
        f.write(text)
    return path


def make_forward_input(label):
    with open(FORWARD_FILE, "r") as f:
        text = f.read()

    text, count = re.subn(
        r"(\[data\]\s*\n\s*type\s*=\s*OptimizationData(?:.|\n)*?^\s*)outputs\s*=\s*'none'",
        r"\1outputs = 'csv'",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError("Could not enable CSV output for the forward data reporter.")

    # The normal forward run writes a large Exodus file. This one-off QC only
    # needs the reporter CSV, so remove the Exodus output block from the temp input.
    text, _ = re.subn(
        r"\n\s*\[exodus\]\s*\n\s*type\s*=\s*Exodus\s*\n\s*\[\]\s*",
        "\n",
        text,
        count=1,
    )

    path = os.path.join(WORKDIR, f"qc_forward_{label}.i")
    with open(path, "w") as f:
        f.write(text)
    return path


def find_strain_csv(output_dir):
    candidates = []
    for name in os.listdir(output_dir):
        if not name.endswith(".csv"):
            continue
        path = os.path.join(output_dir, name)
        try:
            df = pd.read_csv(path, nrows=10)
        except Exception:
            continue
        cols = set(df.columns)
        if {"measurement_values", "simulation_values", "measurement_time", "measurement_ycoord"} <= cols:
            full = pd.read_csv(path, usecols=["simulation_values", "misfit_values"])
            sim_max = float(np.nanmax(np.abs(full["simulation_values"].to_numpy())))
            misfit_max = float(np.nanmax(np.abs(full["misfit_values"].to_numpy())))
            candidates.append((sim_max, misfit_max, path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def main():
    parser = argparse.ArgumentParser(
        description="Run a one-off strain QC forward/adjoint probe for a parameter-history row."
    )
    parser.add_argument(
        "--history-file",
        default=os.path.join(WORKDIR, "parameter_history.csv"),
        help="CSV history file containing one alpha vector per row.",
    )
    parser.add_argument("--row", type=int, default=-1, help="0-based row index; -1 is last row.")
    parser.add_argument("--label", default=None, help="Output label. Defaults to history stem plus row.")
    parser.add_argument("--np", type=int, default=20, help="MPI process count.")
    parser.add_argument("--keep-inputs", action="store_true", help="Keep temporary qc_opt/qc_forward inputs.")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not run MOOSE; plot from an existing qc_output_<label> directory.",
    )
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Stream MOOSE stdout while the QC run is executing.",
    )
    args = parser.parse_args()

    history_file = os.path.abspath(args.history_file)
    if not os.path.exists(history_file):
        raise FileNotFoundError(history_file)

    history_stem = os.path.splitext(os.path.basename(history_file))[0]
    label = args.label or f"{history_stem}_row{args.row}"
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    output_dir = os.path.join(WORKDIR, f"qc_output_{label}")
    os.makedirs(output_dir, exist_ok=True)

    alpha = read_history_row(history_file, args.row)
    alpha_path = os.path.join(WORKDIR, f"qc_alpha_{label}.txt")
    np.savetxt(alpha_path, alpha)

    parent_input = None
    forward_input = None
    if not args.skip_run:
        parent_input = make_parent_input(label, alpha)
        forward_input = make_forward_input(label)

        runner = MooseRunner(moose_executable_path=MOOSE_EXE, mpiexec_path=MPIEXEC)
        success, _, _ = runner.run(
            input_file_path=parent_input,
            output_directory=output_dir,
            num_processors=args.np,
            log_file_name=f"qc_{label}.log",
            stream_output=args.stream_output,
            clean_output_dir=False,
        )
        if not success:
            raise RuntimeError("MOOSE QC run failed.")

    strain_csv = find_strain_csv(output_dir)
    if strain_csv is None:
        raise RuntimeError(
            f"No reporter CSV with nonempty strain fields found in {output_dir}. "
            "Check the QC log and forward reporter output."
        )

    objective_csv = os.path.join(output_dir, f"qc_opt_{label}_out.csv")
    plot_path = os.path.join(WORKDIR, f"qc_strain_{label}.png")
    plot_cmd = [
        sys.executable,
        os.path.join(WORKDIR, "plot_inversion_qc.py"),
        "--strain-csv",
        strain_csv,
        "--objective-csv",
        objective_csv,
        "--output",
        plot_path,
        "--alpha-file",
        alpha_path,
        "--alpha-label",
        f"qc {label}",
    ]
    subprocess.run(plot_cmd, check=True)

    if not args.keep_inputs and not args.skip_run:
        for path in (parent_input, forward_input):
            if os.path.exists(path):
                os.remove(path)

    print(f"QC output directory: {output_dir}")
    print(f"History file       : {history_file}")
    print(f"History row        : {args.row}")
    print(f"Alpha vector       : {alpha_path}")
    print(f"Strain CSV         : {strain_csv}")
    print(f"QC plot            : {plot_path}")


if __name__ == "__main__":
    main()
