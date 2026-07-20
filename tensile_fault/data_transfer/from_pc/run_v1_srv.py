"""Parameterized V1 tensile-SRV MOOSE run (T1=15:00, smooth DAS pressure).

Sweep-friendly: SRV outer width and permeability scale are CLI args, so we can
narrow the SRV / lower perm to sharpen the modelled tensile lobe (per Pengchao:
if the modelled strain curve is too wide, narrow the SRV and lower permeability).

  python run_v1_srv.py --srv_width_ft 75 --perm_scale 0.3 --run
"""
import argparse, os, sys
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "fibeRIS" / "src"))
from fiberis.moose.templates.baseline_model_generator_fervo import build_baseline_model
from fiberis.moose.runner import MooseRunner

FT = 0.3048
T1 = pd.Timestamp("2025-02-24 15:00")
DAS = REPO / "data_fervo/fiberis_format/post_processing/das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srv_width_ft", type=float, default=75.0, help="outer SRV width along MD")
    ap.add_argument("--perm_scale", type=float, default=0.3, help="multiply all SRV perms")
    ap.add_argument("--project", default=None)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--np", type=int, default=20)
    a = ap.parse_args()
    W = a.srv_width_ft
    proj = a.project or f"v1_srv_w{int(W)}_p{a.perm_scale:g}".replace(".", "")
    OUT = REPO / "output" / proj; OUT.mkdir(parents=True, exist_ok=True)
    PREP = OUT / "das_pressure_prepended.npz"; IFILE = OUT / f"{proj}_input.i"

    # smooth DAS pressure, prepend T1=15:00 at IC
    d = np.load(DAS, allow_pickle=True); p = np.asarray(d["data"], float); t = np.asarray(d["taxis"], float)
    t0 = pd.Timestamp(str(d["start_time"].item() if hasattr(d["start_time"], "item") else d["start_time"]))
    times = pd.DatetimeIndex([T1]).append(t0 + pd.to_timedelta(t, unit="s")); vals = np.concatenate([[float(p[0])], p])
    np.savez(PREP, data=vals, taxis=(times - T1).total_seconds().to_numpy(), start_time=str(T1))

    # graded 4-layer SRV, proportional to the outer width; perms scaled
    base_perm = [3e-16, 1e-15, 3e-15, 1e-14]
    frac = [1.0, 0.75, 0.5, 0.25]
    por = [0.08, 0.10, 0.12, 0.14]
    srv = [{"name": f"srv_l{i+1}", "length_m": 440 * FT, "height_m": W * frac[i] * FT,
            "perm": base_perm[i] * a.perm_scale, "porosity": por[i]} for i in range(4)]
    builder = build_baseline_model(
        project_name=proj, pressure_profile_path=str(PREP),
        model_width=100.0, model_length=200.0, hf_length_ft=400.0,
        shift_list_ft=np.array([40.0]), angle=0.0,
        matrix_perm=1e-18, fracture_perm=1e-13, srv_specs=srv)
    builder.generate_input_file(output_filepath=str(IFILE))
    print(f"[{proj}] SRV outer width {W:.0f} ft, perm_scale {a.perm_scale}, perms {[f'{bp*a.perm_scale:.1e}' for bp in base_perm]}")
    print("Generated .i:", IFILE)

    if a.run:
        r = MooseRunner(moose_executable_path=str(REPO / "moose_env/moose/modules/porous_flow/porous_flow-opt"),
                        mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec")
        ok, _o, err = r.run(input_file_path=str(IFILE), output_directory=str(OUT), num_processors=a.np,
                            log_file_name="simulation.log", stream_output=True, clean_output_dir=False)
        print("MOOSE success:", ok)
        if not ok:
            print((err or "")[-1500:]); sys.exit(1)


if __name__ == "__main__":
    main()
