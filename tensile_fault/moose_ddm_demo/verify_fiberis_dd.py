"""Verify the fiberis `ModelBuilder.add_prescribed_dd_band` abstraction end-to-end.

Build a clean linear-elastic Steady model THROUGH fiberis (same physics as the standalone demo),
inject a prescribed strike-slip DD via add_prescribed_dd_band, run MOOSE, and confirm the disp_x
jump across the fault == the prescribed slip and the profile matches DDMpy(Okada). This proves the
DD mechanism is now a first-class, Python-callable fiberis primitive (ready to drop into the
poroelastic model for the unified forward operator + gradient-based inversion).
"""
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner")
sys.path.insert(0, str(REPO / "fibeRIS" / "src"))
sys.path.insert(0, str(REPO / "scripts/tensile_fault/data_transfer/from_pc"))
from fiberis.moose.model_builder import ModelBuilder  # noqa: E402
from fiberis.moose.input_generator import MooseBlock  # noqa: E402
from fiberis.moose.runner import MooseRunner  # noqa: E402
from DDMpy_log.Element import Element  # noqa: E402

FT = 0.3048
STAR = 10373.4
OUT = REPO / "output" / "dd_fiberis_verify"; OUT.mkdir(parents=True, exist_ok=True)
FIG = REPO / "figs/tensile_fault_qc/moose_ddm_demo"; FIG.mkdir(parents=True, exist_ok=True)

NU, E = 0.3, 30e9
L, BAND = 200.0, 0.5             # DD length, band thickness (m)
SLIP = 0.03 * FT                 # prescribed strike-slip (m)
DOM = 300.0
OFFSET = 40 * FT                 # fiber lateral offset
SPAN = 150 * FT


def mat(mat_block, name, mtype, **params):
    b = MooseBlock(name, block_type=mtype)
    for k, v in params.items():
        b.add_param(k, v)
    mat_block.add_sub_block(b)


def main():
    mb = ModelBuilder("dd_elastic")

    # --- graded Cartesian mesh (fine centre), centred at origin (same as the demo) ---
    def segs(hf, df, dcoarse):
        nf, nc = int(round(2 * hf / df)), int(round((DOM - hf) / dcoarse))
        return f"{DOM-hf} {2*hf} {DOM-hf}", f"{nc} {nf} {nc}"
    dxw, ixn = segs(140.0, 0.5, 6.0)
    dyw, iyn = segs(30.0, 0.25, 6.0)
    mb._add_generic_mesh_generator("gen", "CartesianMeshGenerator",
                                   {"dim": 2, "dx": f"'{dxw}'", "ix": f"'{ixn}'",
                                    "dy": f"'{dyw}'", "iy": f"'{iyn}'"}, input_op="")
    mb._add_generic_mesh_generator("centre", "TransformGenerator",
                                   {"transform": "TRANSLATE_CENTER_ORIGIN"})

    mb.add_global_params({"displacements": "disp_x disp_y"})
    mb.add_variables(["disp_x", "disp_y"])
    mb.add_stress_divergence_tensor_kernel("sdx", "disp_x", 0)
    mb.add_stress_divergence_tensor_kernel("sdy", "disp_y", 1)

    # --- elastic materials (strain material added FIRST so add_prescribed_dd_band can wire it) ---
    materials = mb._get_or_create_toplevel_moose_block("Materials")
    mat(materials, "elasticity", "ComputeIsotropicElasticityTensor", youngs_modulus=E, poissons_ratio=NU)
    mat(materials, "strain", "ComputeSmallStrain", displacements="disp_x disp_y")
    mat(materials, "stress", "ComputeLinearElasticStress")

    # >>> the abstraction under test <<<  (patches the strain material's eigenstrain_names)
    mb.add_prescribed_dd_band(name="shear", center_x=0.0, center_y=0.0,
                              length=L, band_thickness=BAND, slip=SLIP)

    mb.add_boundary_condition("clamp_x", "DirichletBC", "disp_x", ["left", "right", "top", "bottom"], params={"value": 0})
    mb.add_boundary_condition("clamp_y", "DirichletBC", "disp_y", ["left", "right", "top", "bottom"], params={"value": 0})

    # --- fiber disp sampler ---
    vpp = mb._get_or_create_toplevel_moose_block("VectorPostprocessors")
    s = MooseBlock("fiber", block_type="LineValueSampler")
    s.add_param("variable", "disp_x disp_y")
    s.add_param("start_point", f"'{OFFSET} {-SPAN} 0'")
    s.add_param("end_point", f"'{OFFSET} {SPAN} 0'")
    s.add_param("num_points", 601)
    s.add_param("sort_by", "y")
    vpp.add_sub_block(s)

    exe = mb._get_or_create_toplevel_moose_block("Executioner")
    for k, v in {"type": "Steady", "solve_type": "NEWTON",
                 "petsc_options_iname": "'-pc_type -pc_factor_mat_solver_package'",
                 "petsc_options_value": "'lu mumps'", "line_search": "none"}.items():
        exe.add_param(k, v)
    mb._get_or_create_toplevel_moose_block("Outputs").add_param("csv", True)

    ifile = OUT / "dd_elastic_input.i"
    mb.generate_input_file(output_filepath=str(ifile))
    print("generated:", ifile)

    r = MooseRunner(moose_executable_path=str(REPO / "moose_env/moose/modules/combined/combined-opt"),
                    mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec")
    ok, _o, err = r.run(input_file_path=str(ifile), output_directory=str(OUT), num_processors=8,
                        log_file_name="sim.log", stream_output=False, clean_output_dir=False)
    print("MOOSE success:", ok)
    if not ok:
        print((err or "")[-2500:]); sys.exit(1)

    fcsv = sorted(glob.glob(str(OUT / "dd_elastic_input_out_fiber_*.csv")))
    df = pd.read_csv(fcsv[-1]).sort_values("y")
    y = df["y"].to_numpy(float); ux = df["disp_x"].to_numpy(float)
    up = ux[np.argmin(np.abs(y - BAND))]; dn = ux[np.argmin(np.abs(y + BAND))]
    jump = up - dn

    e = Element(length=L, height=20000.0, width=0.0, S1=SLIP, S2=0.0); e.mu = NU
    e.set_coors(np.full_like(y, OFFSET), np.zeros_like(y), y.copy())
    u1 = e.u1
    ddm_jump = u1.max() - u1.min()
    print(f"\nprescribed slip           = {SLIP*1e3:.4f} mm")
    print(f"fiberis MOOSE disp_x jump = {jump*1e3:.4f} mm  (ratio vs slip {jump/SLIP:.3f})")
    print(f"DDMpy      u1     jump    = {ddm_jump*1e3:.4f} mm  (ratio MOOSE/DDMpy {jump/ddm_jump:.3f})")

    md = STAR + y / FT
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(u1 * 1e3, md, "k-", lw=2.4, label=f"DDMpy u1 (jump {ddm_jump*1e3:.2f} mm)")
    ax.plot(ux * 1e3, md, "r--", lw=1.8, label=f"fiberis MOOSE disp_x (jump {jump*1e3:.2f} mm)")
    ax.set_ylim(md.max(), md.min()); ax.set_xlabel("disp_x [mm]"); ax.set_ylabel("MD [ft]")
    ax.set_title(f"fiberis add_prescribed_dd_band vs DDMpy — strike-slip {SLIP*1e3:.2f} mm\n"
                 f"MOOSE/DDMpy jump ratio {jump/ddm_jump:.3f}")
    ax.grid(alpha=0.3); ax.legend()
    p = FIG / "fiberis_dd_verify.png"; fig.savefig(p, dpi=140); print("saved", p)


if __name__ == "__main__":
    main()
