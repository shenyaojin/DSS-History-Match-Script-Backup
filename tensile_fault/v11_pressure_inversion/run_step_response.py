"""V11 SRV model driven by a unit pressure RAMP (0 -> DP_PSI over the first 2 h, then held).

Because the V11 poroelastic model is linear in (p - p_IC), the ramp response R(tau) gives the
response to any pressure history that is piecewise-linear on the 2-hour knot grid:

    hat basis at knot k  =  ramp(t - t_{k-1}) - ramp(t - t_k),   t_k = T1 + 2h * k

Geometry, materials, BC location and mesh are exactly those of
scripts/tensile_fault/inversion/run_v11_srv_extended.py (V11): SRV 5234 ft along strike,
domain 7200 ft, nx=880, BC = one node on the monitoring well, 3-zone SRV stack symmetric
about the tensile fracture (MD 10365.107), hf core 1e-13 m2, matrix 1e-18 m2.

Only the pressure history and the time-step sequence differ (the template derives its steps
from the pressure curve, which would sample a held step far too coarsely).

    python run_step_response.py --name base --run
    python run_step_response.py --name fine --ramp-dt 150 --ratio 1.04 --run      # dt convergence
    python run_step_response.py --name srvx3 --srv-perm-scale 3 --run           # sensitivity
    python run_step_response.py --name fwd --pressure-knots <A_inverted_pressure_knots.csv> --run
"""
import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "tensile_fault" / "inversion"))
sys.path.insert(0, str(REPO / "fibeRIS" / "src"))
import fault_geometry as fg  # noqa: E402
from fiberis.moose.templates.baseline_model_generator_fervo import build_baseline_model  # noqa: E402
from fiberis.moose.runner import MooseRunner  # noqa: E402

FT = 0.3048
WINDOW_S = 7200.0
END_S = 820800.0                     # T1 2025-02-24 12:00 -> T3 2025-03-06 00:00

# --- V11 geometry (copied from run_v11_srv_extended.py, unchanged) ---
STAGE_FT, PAST_STAGE_FT, PAST_WELL_FT = 1934.0, 1800.0, 1500.0
TOWARD_INJ_FT = STAGE_FT + PAST_STAGE_FT
SRV_LEN_FT = TOWARD_INJ_FT + PAST_WELL_FT
WELL_OFFSET_FT = -(TOWARD_INJ_FT - PAST_WELL_FT) / 2.0
DOMAIN_FT = 7200.0
NX = 880


def srv_zones(variant):
    """(name, half-height ft, perm m2, porosity), outermost first."""
    if variant == "v11":
        return fg.SRV_ZONES_1200
    if variant == "transition":
        # V11 heights (srv_outer top on the DDM shear plane), all SRV layers raised to the srv_narrow
        # permeability, plus a 1e-17 m2 transition band one layer-step (outer - wide = 14.5 ft) outside the
        # shear plane, so the SRV -> matrix drop is 1e-16 -> 1e-17 -> 1e-18 instead of a single step.
        (_, h_out, _, p_out), (_, h_wide, _, p_wide), (_, h_nar, k_nar, p_nar) = fg.SRV_ZONES_1200
        return [("srv_transition", h_out + (h_out - h_wide), 1e-17, 0.05),
                ("srv_outer", h_out, k_nar, p_out), ("srv_wide", h_wide, k_nar, p_wide), ("srv_narrow", h_nar, k_nar, p_nar)]
    if variant == "graded":
        # V11 stack plus four outer shells whose permeability falls log-evenly from the srv_outer value
        # (1e-17) to the matrix (1e-18), so the SRV -> matrix contrast is spread over ~1.8x the outer
        # half-height instead of a single step. Porosity is graded 0.08 -> 0.01 the same way.
        h_out = fg.SRV_ZONES_1200[0][1]
        shells = [(f"srv_g{i}", h_out * (1 + 0.2 * i), 10 ** (-17 - 0.2 * i), round(0.08 - 0.014 * i, 3)) for i in (4, 3, 2, 1)]
        return shells + list(fg.SRV_ZONES_1200)
    raise ValueError(variant)


def time_sequence(ramp_dt, ratio):
    """Uniform steps through the ramp, then geometric growth to END_S."""
    t = list(np.arange(0.0, WINDOW_S + ramp_dt / 2, ramp_dt))
    dt = ramp_dt
    while t[-1] < END_S:
        dt *= ratio
        t.append(min(t[-1] + dt, END_S))
    return np.array(t)


def build(args):
    out = REPO / "output" / "v11_pinv" / args.name
    out.mkdir(parents=True, exist_ok=True)
    project = f"v11_pinv_{args.name}"
    prof = out / "ramp_pressure.npz"
    if args.pressure_knots:
        knots = pd.read_csv(args.pressure_knots)
        taxis, p = knots["elapsed_s_from_T1"].to_numpy(float), knots["pressure_psi"].to_numpy(float)
    else:
        taxis, p = np.array([0.0, WINDOW_S, END_S]), args.ic_psi + args.dp_psi * np.array([0.0, 1.0, 1.0])
    np.savez(prof, data=p, taxis=taxis, start_time="2025-02-24 12:00:00")

    # fracture/SRV extent past the monitoring fiber (V11: 1500 ft; < 0 = it stops short of the fiber)
    srv_len_ft = TOWARD_INJ_FT + args.past_well_ft
    well_offset_ft = -(TOWARD_INJ_FT - args.past_well_ft) / 2.0
    dom_ft = DOMAIN_FT
    if args.align_well_node:
        # V11's 7200 ft / 880 cells puts the well 136.5 cells from the (node) centre, so the BC had to be
        # snapped 3.9 ft off the fiber. Shrink the domain slightly so the well itself is a mesh node.
        k = round(-well_offset_ft / (DOMAIN_FT / NX))
        dom_ft = NX * -well_offset_ft / k
    specs = [{"name": n, "length_m": srv_len_ft * FT, "height_m": 2 * h * args.srv_height_scale * FT,
              "center_y": args.srv_shift_ft * FT,
              "perm": k * args.srv_perm_scale, "porosity": por}
             for n, h, k, por in srv_zones(args.srv_variant)]
    builder = build_baseline_model(
        project_name=project, pressure_profile_path=str(prof),
        model_width=args.model_width, model_length=dom_ft * FT, hf_length_ft=srv_len_ft,
        shift_list_ft=np.array([well_offset_ft]), angle=0.0, nx=NX,
        matrix_perm=1e-18, fracture_perm=args.frac_perm, srv_specs=specs,
    )
    ifile = out / f"{project}_input.i"
    builder.generate_input_file(output_filepath=str(ifile))

    txt = ifile.read_text()
    # BC on the monitoring well, snapped to a mesh node (as in V11)
    centre_x = dom_ft * FT / 2.0
    dx = dom_ft * FT / NX
    x_well = centre_x + well_offset_ft * FT
    x_bc = x_well + args.bc_along_strike_ft * FT            # + = toward the injector
    snapped = round(x_bc / dx) * dx
    new = re.sub(r"(\[injection\](?:.|\n)*?coord = ')[\d.]+", lambda m: f"{m.group(1)}{snapped:.6f}", txt, count=1)
    assert new != txt, "injection coord not found"
    seq = (np.arange(0.0, END_S + 1, WINDOW_S / args.sub) if args.pressure_knots
           else time_sequence(args.ramp_dt, args.ratio))
    seq_str = " ".join(f"{v:.1f}" for v in seq)
    new, n1 = re.subn(r"time_sequence = '[^']*'", f"time_sequence = '{seq_str}'", new)
    new, n2 = re.subn(r"end_time = [\d.eE+]+", f"end_time = {END_S:.1f}", new)
    # the template's default changed after V11 was generated; pin the V11 deck value
    new, n3 = re.subn(r"solid_bulk_compliance = [\d.eE+-]+", "solid_bulk_compliance = 2e-11", new)
    # Solver fix (numerical only). With the template's Newton settings (nl_abs_tol = 1e-3):
    #  - a step whose initial residual is already below 1e-3 is accepted with ZERO iterations, so
    #    pressure diffusion freezes once the BC stops changing (|R0| = 2.7e-4 after the ramp);
    #  - forcing an iteration instead leaves a round-off floor (~4e-2, mixed N / kg/s residual) that
    #    the tolerances cannot see past, so steps "fail" and dt collapses to ~1 s.
    # The model is linear in p (up to the 2.2 GPa fluid compressibility), so one direct (MUMPS LU)
    # linear solve per step with no convergence test is the exact implicit-Euler step.
    new, n4 = re.subn(r"solve_type = 'Newton'", "solve_type = 'LINEAR'", new)
    # keep the fiber sampling at the V11 spacing (~0.9 m) when the walls move out
    new, n5 = re.subn(r"num_points = 200", f"num_points = {int(round(200 * args.model_width / 100.0))}", new)
    if args.free_walls:
        # top/bottom (the walls the fiber runs into) traction-free; disp_y held at the far strike ends instead,
        # so the fiber's integrated strain is no longer forced to zero (closer to the DDM's infinite medium)
        new, n6 = re.subn(r"(\[confiney\](?:.|\n)*?boundary = )'top bottom'", r"\1'left right'", new)
        assert n6 == 1
    if args.srv_youngs_gpa:
        # softer SRV + hf core: the matrix tensor keeps 50 GPa on 'matrix' only; a second tensor covers the rest
        blocks = re.search(r"new_block = '([^']*)'", new).group(1).split()
        soft = " ".join(b for b in blocks if b != "matrix")
        new, n7 = re.subn(r"(\[elasticity_tensor_matrix\]\n(\s*)type = ComputeIsotropicElasticityTensor\n)"
                          r"(\s*youngs_modulus = [\d.eE+]+\n\s*poissons_ratio = ([\d.]+)\n)(\s*)\[\]",
                          lambda m: f"{m.group(1)}{m.group(2)}block = 'matrix'\n{m.group(3)}{m.group(5)}[]\n"
                                    f"{m.group(5)}[elasticity_tensor_srv]\n{m.group(2)}type = ComputeIsotropicElasticityTensor\n"
                                    f"{m.group(2)}block = '{soft}'\n{m.group(2)}youngs_modulus = {args.srv_youngs_gpa * 1e9:.1f}\n"
                                    f"{m.group(2)}poissons_ratio = {m.group(4)}\n{m.group(5)}[]", new)
        assert n7 == 1
    assert n1 == 1 and n2 == 1 and n3 == 1 and n4 == 1 and n5 >= 1
    ifile.write_text(new)
    print(f"[{project}] {len(seq)} steps, IC {args.ic_psi:.0f} psi, ramp +{args.dp_psi:.0f} psi, "
          f"srv perm x{args.srv_perm_scale:g}, srv {args.srv_variant} height x{args.srv_height_scale:g} shift {args.srv_shift_ft:+g} ft, E_srv {args.srv_youngs_gpa or 50:g} GPa, walls +-{args.model_width:g} m{' (free)' if args.free_walls else ''}, past fiber {args.past_well_ft:g} ft, hf core {args.frac_perm:.0e}; "
          f"domain {dom_ft:.2f} ft; well x={x_well:.4f} m, BC x={snapped:.4f} m "
          f"({(snapped - x_well) / FT:+.2f} ft along strike, snap error {(snapped - x_bc) / FT:+.2f} ft)")
    return ifile, out, project


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base")
    ap.add_argument("--ramp-dt", type=float, default=300.0)
    ap.add_argument("--ratio", type=float, default=1.08)
    ap.add_argument("--ic-psi", type=float, default=4056.0)   # V11 prior initial pressure
    ap.add_argument("--dp-psi", type=float, default=1000.0)
    ap.add_argument("--srv-perm-scale", type=float, default=1.0)
    ap.add_argument("--srv-height-scale", type=float, default=1.0,
                    help="scale all SRV zone half-heights about the fracture (1 = V11, top at the shear plane)")
    ap.add_argument("--model-width", type=float, default=100.0,
                    help="half-width of the domain across the fault (m); disp_y = 0 is fixed on these walls (V11: 100 m)")
    ap.add_argument("--free-walls", action="store_true",
                    help="traction-free top/bottom; disp_y fixed on the left/right strike ends instead")
    ap.add_argument("--past-well-ft", type=float, default=PAST_WELL_FT,
                    help="SRV/fracture length beyond the monitoring fiber (ft); negative = tip short of the fiber")
    ap.add_argument("--srv-youngs-gpa", type=float, default=None,
                    help="Young's modulus of the SRV zones + hf core (GPa); default = matrix 50 GPa everywhere")
    ap.add_argument("--srv-shift-ft", type=float, default=0.0,
                    help="move the SRV stack (not the hf core) along the fiber; + = deeper MD")
    ap.add_argument("--srv-variant", default="v11", choices=["v11", "transition", "graded"],
                    help="v11 = V11 layered SRV; transition = V11 heights, SRV raised to 1e-16 m2, plus a 1e-17 m2 band outside")
    ap.add_argument("--align-well-node", action="store_true",
                    help="size the domain so a mesh node sits exactly on the monitoring well (BC on the fiber)")
    ap.add_argument("--bc-along-strike-ft", type=float, default=0.0,
                    help="pressure BC distance from the monitoring well along strike, toward the injector "
                         "(Bearskin 4-PB crosses the F1 plane 1973.8 ft away)")
    ap.add_argument("--frac-perm", type=float, default=1e-13)
    ap.add_argument("--np", type=int, default=16)
    ap.add_argument("--pressure-knots", help="forward run: knot CSV from invert_pressure.py instead of the ramp")
    ap.add_argument("--sub", type=int, default=2, help="time steps per 2-h knot interval (forward run)")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    ifile, out, project = build(args)
    if args.run:
        runner = MooseRunner(
            moose_executable_path=str(REPO / "moose_env/moose/modules/porous_flow/porous_flow-opt"),
            mpiexec_path="/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec",
        )
        ok, _o, err = runner.run(input_file_path=str(ifile), output_directory=str(out),
                                 num_processors=args.np, log_file_name="simulation.log",
                                 stream_output=False, clean_output_dir=False)
        n = len(list(out.glob(f"{project}_input_csv_fiber_strain_sampler_*ft_*.csv")))
        print(f"[{project}] MOOSE success: {ok}   sampler files: {n}")
        if not ok:
            print((err or "")[-2000:])
            sys.exit(1)


if __name__ == "__main__":
    main()
