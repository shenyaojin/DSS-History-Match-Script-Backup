"""Get the UNIFIED poroelastic + prescribed-DD model to converge.

Builds the real fervo poroelastic model (flat pressure so only the DD acts), optionally injects a
strike-slip DD band, patches the Executioner with convergence aids, runs, and reports the first-step
nonlinear residual + whether it finished. Parameterized so we can bisect what makes it converge.

  python poro_dd_converge.py --tag base                     # no DD (baseline)
  python poro_dd_converge.py --tag fix --dd --autoscale --linesearch none --adaptive
"""
import argparse
import glob
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np

REPO = Path("/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner")
sys.path.insert(0, str(REPO / "fibeRIS" / "src"))
from fiberis.moose.templates.baseline_model_generator_fervo import build_baseline_model  # noqa: E402
from fiberis.moose.input_generator import MooseBlock  # noqa: E402
from fiberis.moose.runner import MooseRunner  # noqa: E402

FT = 0.3048


def top(mb, name):
    for b in mb._top_level_blocks:
        if b.block_name == name:
            return b
    return None


def sub(block, name):
    for b in block.sub_blocks:
        if b.block_name == name:
            return b
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--dd", action="store_true")
    ap.add_argument("--slip_ft", type=float, default=0.03)
    ap.add_argument("--autoscale", action="store_true")
    ap.add_argument("--linesearch", default=None)         # e.g. 'none'
    ap.add_argument("--adaptive", action="store_true")    # IterationAdaptiveDT
    ap.add_argument("--nsteps", type=int, default=40)
    ap.add_argument("--nl_abs_tol", type=float, default=None)
    ap.add_argument("--nl_rel_tol", type=float, default=None)
    ap.add_argument("--real", action="store_true", help="use the real DAS pressure (unified model)")
    ap.add_argument("--np", type=int, default=8)
    a = ap.parse_args()

    OUT = REPO / "output" / f"poro_dd_{a.tag}"; OUT.mkdir(parents=True, exist_ok=True)
    prep = OUT / "pressure.npz"
    if a.real:
        # real smooth DAS pressure, prepend T1=15:00 as IC (same as run_v1_srv)
        import pandas as pd
        DAS = REPO / "data_fervo/fiberis_format/post_processing/das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz"
        d = np.load(DAS, allow_pickle=True); p = np.asarray(d["data"], float); t = np.asarray(d["taxis"], float)
        t0 = pd.Timestamp(str(d["start_time"].item() if hasattr(d["start_time"], "item") else d["start_time"]))
        T1 = pd.Timestamp("2025-02-24 15:00")
        times = pd.DatetimeIndex([T1]).append(t0 + pd.to_timedelta(t, unit="s"))
        vals = np.concatenate([[float(p[0])], p])
        np.savez(prep, data=vals, taxis=(times - T1).total_seconds().to_numpy(), start_time=str(T1))
        tend = float((times - T1).total_seconds().to_numpy()[-1])
    else:
        tend = 3600.0
        tseq = np.linspace(0.0, tend, a.nsteps + 1)
        np.savez(prep, data=np.full_like(tseq, 2.7965e7), taxis=tseq, start_time="2025-02-24 15:00")

    srv = [{"name": f"srv_l{i+1}", "length_m": 440 * FT, "height_m": (75 * f) * FT,
            "perm": p, "porosity": po}
           for i, (f, p, po) in enumerate(zip([1, .75, .5, .25],
                                              [9e-17, 3e-16, 9e-16, 3e-15], [.08, .10, .12, .14]))]
    mb = build_baseline_model(
        project_name=f"poro_dd_{a.tag}", pressure_profile_path=str(prep),
        model_width=100.0, model_length=200.0, hf_length_ft=400.0,
        shift_list_ft=np.array([40.0]), angle=0.0,
        matrix_perm=1e-18, fracture_perm=1e-13, srv_specs=srv)

    if a.dd:
        mb.add_prescribed_dd_band(name="shear", center_x=100.0, center_y=0.0,
                                  length=60.0, band_thickness=1.0, slip=a.slip_ft * FT,
                                  ramp=(0.0, tend))
        # disp fiber sampler (offset 40 ft) to check the DD actually deforms the medium
        xf = 100.0 + 40 * FT
        s = MooseBlock("dd_disp", block_type="LineValueSampler")
        s.add_param("variable", "disp_x disp_y")
        s.add_param("start_point", f"'{xf} -90 0'")
        s.add_param("end_point", f"'{xf} 90 0'")
        s.add_param("num_points", 361)
        s.add_param("sort_by", "y")
        top(mb, "VectorPostprocessors").add_sub_block(s)

    # --- patch the Executioner with convergence aids ---
    exe = top(mb, "Executioner")
    if a.linesearch:
        exe.add_param("line_search", a.linesearch)
    if a.autoscale:
        exe.add_param("automatic_scaling", True)
        exe.add_param("compute_scaling_once", False)
    if a.nl_abs_tol is not None:
        exe.add_param("nl_abs_tol", a.nl_abs_tol)
    if a.nl_rel_tol is not None:
        exe.add_param("nl_rel_tol", a.nl_rel_tol)
    if a.adaptive:
        ts = sub(exe, "TimeStepper")
        if ts:
            exe.sub_blocks.remove(ts)
        newts = MooseBlock("TimeStepper", block_type="IterationAdaptiveDT")
        newts.add_param("dt", tend / a.nsteps)
        newts.add_param("optimal_iterations", 8)
        newts.add_param("growth_factor", 1.5)
        newts.add_param("cutback_factor", 0.5)
        exe.add_sub_block(newts)

    ifile = OUT / f"poro_dd_{a.tag}_input.i"
    mb.generate_input_file(output_filepath=str(ifile))
    print("generated:", ifile, "| DD:", a.dd, "| autoscale:", a.autoscale,
          "| linesearch:", a.linesearch, "| adaptive:", a.adaptive)

    r = MooseRunner(moose_executable_path=str(REPO / "moose_env/moose/modules/combined/combined-opt"),
                    mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec")
    ok, out, err = r.run(input_file_path=str(ifile), output_directory=str(OUT), num_processors=a.np,
                         log_file_name="sim.log", stream_output=False, clean_output_dir=False)
    log = (out or "") + (err or "")
    r0 = re.findall(r"0 Nonlinear \|R\| = ([\d.eE+\-]+)", log)
    finished = "Finished Executing" in log or "Aborting as solve did not converge" not in log
    print(f"\n=== tag={a.tag} : MOOSE success={ok} ===")
    if r0:
        print(f"  first-step 0th nonlinear |R| = {r0[0]}")
    nsolved = log.count("Solve Converged")
    print(f"  converged solves: {nsolved}")
    if not ok:
        print("  TAIL:", log[-1200:])
        return
    # did the DD actually deform the medium?  (check disp_x jump across the fault)
    if a.dd:
        import pandas as pd
        fc = sorted(glob.glob(str(OUT / "*dd_disp*.csv")))
        if fc:
            df = pd.read_csv(fc[-1]).sort_values("y")
            y = df["y"].to_numpy(float); ux = df["disp_x"].to_numpy(float)
            jump = ux[np.argmin(np.abs(y - 1.0))] - ux[np.argmin(np.abs(y + 1.0))]
            print(f"  disp_x jump across fault = {jump*1e3:.4f} mm  (prescribed {a.slip_ft*FT*1e3:.4f} mm, "
                  f"ratio {jump/(a.slip_ft*FT):.3f})")


if __name__ == "__main__":
    main()
