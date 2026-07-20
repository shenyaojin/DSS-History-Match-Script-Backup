"""Phase 1: finite-difference gradient inversion on the unified MOOSE poroelastic + DD forward.

Wraps the converged unified model (real DAS pressure tensile + prescribed-DD shear, line_search=none)
as a forward operator F(pressure_scale, slip_scale) -> model fiber strain on the observation grid
(MD 10200-10500 x 4-hour windows). Because the forward is LINEAR in the pressure perturbation and in
the slip, we build two BASIS responses (tensile from unit pressure, shear from unit slip) with two
runs and can then evaluate any (a,b) instantly; a finite-difference gradient over the scales is used
to (i) drive scipy L-BFGS-B and (ii) provide the ground-truth gradient for validating the adjoint.

Modes:
  --mode synth_slip   self-consistency: plant a slip scale, recover it (validates the loop)
  --mode real_tensile invert the tensile pressure scale vs the real observed strain (fits red lobe)
"""
import argparse
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
from fiberis.moose.templates.baseline_model_generator_fervo import build_baseline_model  # noqa: E402
from fiberis.moose.input_generator import MooseBlock  # noqa: E402
from fiberis.moose.runner import MooseRunner  # noqa: E402

FT = 0.3048
STAR = 10373.4
T1 = pd.Timestamp("2025-02-24 15:00")
NOMINAL_SLIP = 0.03 * FT
DAS_P = REPO / "data_fervo/fiberis_format/post_processing/das_injection_pressure_HISTORYMATCH_C1p63e7_10373ft.npz"
MOOSE = str(REPO / "moose_env/moose/modules/combined/combined-opt")
MPIEXEC = "/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
OBS = np.load(REPO / "output/inversion/observation.npz", allow_pickle=True)
FIG = REPO / "figs/tensile_fault_qc/inversion"; FIG.mkdir(parents=True, exist_ok=True)


def _pressure_npz(path, pressure_scale):
    """Real DAS pressure with the PERTURBATION above IC scaled; prepend T1 as IC."""
    d = np.load(DAS_P, allow_pickle=True)
    p = np.asarray(d["data"], float); t = np.asarray(d["taxis"], float)
    t0 = pd.Timestamp(str(d["start_time"].item() if hasattr(d["start_time"], "item") else d["start_time"]))
    p0 = float(p[0])
    p_scaled = p0 + pressure_scale * (p - p0)
    times = pd.DatetimeIndex([T1]).append(t0 + pd.to_timedelta(t, unit="s"))
    vals = np.concatenate([[p0], p_scaled])
    np.savez(path, data=vals, taxis=(times - T1).total_seconds().to_numpy(), start_time=str(T1))
    return float((times - T1).total_seconds().to_numpy()[-1])


def run_forward(slip_scale, pressure_scale=1.0, tag="fwd", nprocs=8):
    """Run the unified model; return (md_obs, centers, strain_yy_4h, strain_xy_4h) on the obs grid."""
    out = REPO / "output" / f"fdinv_{tag}"; out.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(str(out / "*fiber_strain_sampler*.csv")):
        os.remove(f)
    prep = out / "pressure.npz"
    tend = _pressure_npz(prep, pressure_scale)

    srv = [{"name": f"srv_l{i+1}", "length_m": 440 * FT, "height_m": (75 * f) * FT, "perm": p, "porosity": po}
           for i, (f, p, po) in enumerate(zip([1, .75, .5, .25], [9e-17, 3e-16, 9e-16, 3e-15], [.08, .10, .12, .14]))]
    mb = build_baseline_model(project_name=f"fdinv_{tag}", pressure_profile_path=str(prep),
                              model_width=100.0, model_length=200.0, hf_length_ft=400.0,
                              shift_list_ft=np.array([40.0]), angle=0.0,
                              matrix_perm=1e-18, fracture_perm=1e-13, srv_specs=srv)
    if slip_scale != 0.0:
        mb.add_prescribed_dd_band(name="shear", center_x=100.0, center_y=0.0, length=60.0,
                                  band_thickness=1.0, slip=slip_scale * NOMINAL_SLIP, ramp=(0.0, tend))
    ifile = out / "in.i"; mb.generate_input_file(output_filepath=str(ifile))
    r = MooseRunner(moose_executable_path=MOOSE, mpiexec_path=MPIEXEC)
    ok, _o, err = r.run(input_file_path=str(ifile), output_directory=str(out), num_processors=nprocs,
                        log_file_name="sim.log", stream_output=False, clean_output_dir=False)
    if not ok:
        print((err or "")[-1500:]); raise SystemExit(f"forward {tag} failed")

    # assemble model strain(MD_fiber, model_time) for yy and xy
    main = pd.read_csv(out / "in_csv.csv"); mtimes = main["time"].to_numpy(float)
    fs = sorted(glob.glob(str(out / "*fiber_strain_sampler*.csv")))
    yy_cols, xy_cols, y0, keep_t = [], [], None, []
    for i, f in enumerate(fs):
        df = pd.read_csv(f)
        if len(df) < 100:
            continue
        df = df.sort_values("y")
        if y0 is None:
            y0 = df["y"].to_numpy(float)
        yy_cols.append(df["strain_yy"].to_numpy(float)); xy_cols.append(df["strain_xy"].to_numpy(float))
        keep_t.append(i)
    yy = np.array(yy_cols).T; xy = np.array(xy_cols).T                      # [md_fiber, time]
    mt = mtimes[keep_t]; md_fiber = STAR + y0 / FT
    yy -= yy[:, [0]]; xy -= xy[:, [0]]                                      # T1-ref

    # reduce onto the observation grid (window centers in time, obs MD in depth)
    md_obs = OBS["md_ft"]; centers = pd.to_datetime(OBS["window_starts"])
    ctsec = (centers - T1).total_seconds().to_numpy()
    def to_grid(mat):
        tmp = np.array([np.interp(ctsec, mt, mat[k]) for k in range(mat.shape[0])])   # interp time
        oo = np.argsort(md_fiber)
        return np.array([np.interp(md_obs, md_fiber[oo], tmp[oo, j]) for j in range(len(centers))]).T
    return md_obs, centers, to_grid(yy) * 1e3, to_grid(xy) * 1e3           # millistrain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="synth_slip", choices=["synth_slip", "real_tensile", "test_forward"])
    ap.add_argument("--scales", default="0.5,1.0,1.5,2.0")
    a = ap.parse_args()

    if a.mode == "test_forward":
        md, c, yy, xy = run_forward(slip_scale=1.0, pressure_scale=1.0, tag="test")
        print(f"model strain on obs grid: yy peak {np.nanmax(np.abs(yy)):.4f} me, xy peak {np.nanmax(np.abs(xy)):.4f} me")
        print(f"obs strain peak {np.nanmax(np.abs(OBS['strain_4h'])):.4f} me; grids md {md.shape} win {len(c)}")
        return

    if a.mode == "synth_slip":
        # Run a scan of slip scales at NOMINAL pressure (=1, avoids the flat-pressure degeneracy).
        # The shear appears in strain_xy; plant a synthetic target = model at slip=TRUE, then verify
        # the objective is minimized there and the FD gradient from real reruns is consistent.
        scales = [float(s) for s in a.scales.split(",")]
        true = 1.6
        runs = {}
        for s in sorted(set(scales + [true, true + 0.1, true - 0.1])):
            _, _, _, xy = run_forward(slip_scale=s, pressure_scale=1.0, tag=f"s{str(s).replace('.','p')}")
            runs[s] = xy
        target = runs[true]                                                # synthetic observation
        objs = [0.5 * np.nansum((runs[s] - target) ** 2) for s in scales]
        # FD gradient at TRUE +/- 0.1 from independent MOOSE reruns (should ~0 at the minimum)
        gp = 0.5 * np.nansum((runs[true + 0.1] - target) ** 2)
        gm = 0.5 * np.nansum((runs[true - 0.1] - target) ** 2)
        fd_grad_min = (gp - gm) / 0.2
        # FD gradient at a scale AWAY from the minimum (scales[0]) vs the linear-model prediction
        s0 = scales[0]
        basis = (runs[max(scales)] - runs[min(scales)]) / (max(scales) - min(scales))   # d(model)/d(slip)
        an_grad0 = np.nansum((runs[s0] - target) * basis)
        print(f"\nSYNTH slip: planted TRUE={true}")
        print(f"  objective(scales={scales}) = {[f'{o:.3e}' for o in objs]}")
        print(f"  min at scale = {scales[int(np.argmin(objs))]} ; FD grad @TRUE = {fd_grad_min:.3e} (want ~0)")
        print(f"  FD-linearity: d(model)/d(slip) basis reproduces runs? "
              f"resid@true = {np.nanmax(np.abs(runs[min(scales)] + (true-min(scales))*basis - target)):.2e} me")
        print(f"  analytic grad @s0={s0}: {an_grad0:.3e}")
        fig, ax = plt.subplots(figsize=(7, 5))
        ss = np.linspace(min(scales + [true]), max(scales + [true]), 50)
        b0 = runs[min(scales)]
        ax.plot(ss, [0.5 * np.nansum((b0 + (s - min(scales)) * basis - target) ** 2) for s in ss], "b-", label="linear model")
        ax.plot(scales, objs, "ko", label="MOOSE runs")
        ax.axvline(true, color="green", ls="--", label=f"planted {true}")
        ax.set_xlabel("slip scale"); ax.set_ylabel("objective"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title("Phase-1a FD self-consistency (synthetic slip scalar)")
        fig.savefig(FIG / "fd_synth_slip.png", dpi=140); print("saved", FIG / "fd_synth_slip.png")

    if a.mode == "real_tensile":
        # Tensile strain is linear in Delta p (=pressure_scale*(p_nom-p0)) through the origin, so one
        # run at pressure_scale=1 (slip=0) gives the tensile basis; the global scale is a 1-D LS. We
        # also rerun at scale=1.5 to CONFIRM linearity (FD path, no assumption).
        md, c, yy1, _xy1 = run_forward(slip_scale=0.0, pressure_scale=1.0, tag="tens1")
        _, _, yy15, _ = run_forward(slip_scale=0.0, pressure_scale=1.5, tag="tens15")
        lin_resid = np.nanmax(np.abs(yy15 - 1.5 * yy1))
        obs = OBS["strain_4h"]                                             # observed axial strain
        v = np.isfinite(obs) & np.isfinite(yy1)
        p_star = np.nansum(obs[v] * yy1[v]) / np.nansum(yy1[v] * yy1[v])   # 1-D LS
        model = p_star * yy1
        rms0 = np.sqrt(np.nanmean(obs[v] ** 2)); rmsr = np.sqrt(np.nanmean((obs - model)[v] ** 2))
        print(f"\nREAL tensile: linearity resid |yy(1.5)-1.5*yy(1)| = {lin_resid:.2e} me")
        print(f"  recovered global pressure scale p* = {p_star:.4f}")
        print(f"  strain RMS {rms0:.4f} -> {rmsr:.4f} me  ({100*(1-rmsr/rms0):.0f}% var. reduction)")
        # residual = obs - tensile model = the unexplained SHEAR band (for Phase 3)
        resid = obs - model
        md_o = OBS["md_ft"]; centers = pd.to_datetime(OBS["window_starts"])
        import matplotlib.dates as mdates
        fig, axs = plt.subplots(1, 3, figsize=(19, 6), sharey=True, constrained_layout=True)
        ext = [mdates.date2num(centers[0]), mdates.date2num(centers[-1]), md_o[-1], md_o[0]]
        for ax, dat, ti in [(axs[0], obs, "observed axial strain"),
                            (axs[1], model, f"model tensile (p*={p_star:.2f})"),
                            (axs[2], resid, "residual = unexplained shear")]:
            im = ax.imshow(dat, aspect="auto", cmap="seismic", vmin=-0.09, vmax=0.09, extent=ext, interpolation="bilinear")
            ax.xaxis_date(); ax.set_ylim(10500, 10200); ax.set_title(ti); ax.set_xlabel("time")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d")); plt.colorbar(im, ax=ax, label="me")
        axs[0].set_ylabel("MD [ft]")
        fig.suptitle(f"Phase-1 FD tensile inversion vs real DAS — p*={p_star:.3f}, {100*(1-rmsr/rms0):.0f}% var. reduction",
                     fontweight="bold")
        fig.savefig(FIG / "fd_real_tensile.png", dpi=140); print("saved", FIG / "fd_real_tensile.png")


if __name__ == "__main__":
    main()
