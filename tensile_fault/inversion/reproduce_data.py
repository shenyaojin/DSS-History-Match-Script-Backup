"""Reproduce the real DAS axial strain (strain_yy) with a linear-elastic two-eigenstrain-band model,
inverted per 4-hour window. Both the tensile lobe and the shear negative band are represented in
strain_yy by eigenstrain bands (opening at the fracture MD, and a band at the shear-plane MD), whose
per-window prefactors are the design parameters -- exactly the eigenstrain-prefactor gradient that the
Phase-2 adjoint validated. Forward is LINEAR, so per-window it is a 2-parameter least squares over the
two strain_yy bases; the adjoint gives the same gradient (validated separately) and is what scales this
to many parameters.

Outputs the reproduced-vs-observed waterfall + the recovered tensile/shear amplitude histories.
"""
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner")
FT = 0.3048
STAR = 10373.4
MOOSE = str(REPO / "moose_env/moose/modules/combined/combined-opt")
MPIEXEC = "/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
OBS = np.load(REPO / "output/inversion/observation.npz", allow_pickle=True)
OUT = REPO / "output/inversion"; FIG = REPO / "figs/tensile_fault_qc/inversion"; FIG.mkdir(parents=True, exist_ok=True)

TENSILE_MD = 10373.4     # opening band (pressure proxy)
SHEAR_MD = 10300.0       # shear-plane band (slip proxy)
BAND_H = 4.0             # band thickness in MD (ft) -> m below
OFFSET = 40 * FT


def band_bbox(md_center):
    yc = (md_center - STAR) * FT
    h = BAND_H * FT
    return yc - h / 2, yc + h / 2


def make_input(tens_pref, shear_pref):
    (ty0, ty1) = band_bbox(TENSILE_MD)
    (sy0, sy1) = band_bbox(SHEAR_MD)
    xf = OFFSET
    return f"""# elastic 2-band model: tensile (fracture) + shear-plane eigenstrain, sample strain_yy on fiber
[Mesh]
  [gen]
    type = GeneratedMeshGenerator
    dim = 2
    nx = 200
    ny = 400
    xmin = -100
    xmax = 100
    ymin = -60
    ymax = 60
  []
  [tens]
    type = SubdomainBoundingBoxGenerator
    input = gen
    block_id = 1
    bottom_left = '-40 {ty0} 0'
    top_right = '40 {ty1} 0'
  []
  [shear]
    type = SubdomainBoundingBoxGenerator
    input = tens
    block_id = 2
    bottom_left = '-40 {sy0} 0'
    top_right = '40 {sy1} 0'
  []
[]
[GlobalParams]
  displacements = 'disp_x disp_y'
[]
[Variables]
  [disp_x][][disp_y][]
[]
[Kernels]
  [sdx]
    type = StressDivergenceTensors
    variable = disp_x
    component = 0
  []
  [sdy]
    type = StressDivergenceTensors
    variable = disp_y
    component = 1
  []
[]
[BCs]
  [cx]
    type = DirichletBC
    variable = disp_x
    boundary = 'left right top bottom'
    value = 0
  []
  [cy]
    type = DirichletBC
    variable = disp_y
    boundary = 'left right top bottom'
    value = 0
  []
[]
[Materials]
  [elast]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 30e9
    poissons_ratio = 0.2
  []
  [strain]
    type = ComputeSmallStrain
    eigenstrain_names = 'tens_eig shear_eig'
  []
  [stress]
    type = ComputeLinearElasticStress
  []
  [tens_eig]
    type = ComputeEigenstrain
    eigen_base = '0 {tens_pref} 0 0 0 0'
    eigenstrain_name = 'tens_eig'
    block = 1
  []
  [tens_eig0]
    type = ComputeEigenstrain
    eigen_base = '0 0 0 0 0 0'
    eigenstrain_name = 'tens_eig'
    block = '0 2'
  []
  [shear_eig]
    type = ComputeEigenstrain
    eigen_base = '0 {shear_pref} 0 0 0 0'
    eigenstrain_name = 'shear_eig'
    block = 2
  []
  [shear_eig0]
    type = ComputeEigenstrain
    eigen_base = '0 0 0 0 0 0'
    eigenstrain_name = 'shear_eig'
    block = '0 1'
  []
[]
[AuxVariables]
  [strain_yy]
    order = CONSTANT
    family = MONOMIAL
  []
[]
[AuxKernels]
  [syy]
    type = RankTwoAux
    rank_two_tensor = total_strain
    variable = strain_yy
    index_i = 1
    index_j = 1
  []
[]
[VectorPostprocessors]
  [fiber]
    type = LineValueSampler
    variable = 'strain_yy'
    start_point = '{xf} -55 0'
    end_point = '{xf} 55 0'
    num_points = 300
    sort_by = y
  []
[]
[Executioner]
  type = Steady
  solve_type = NEWTON
  petsc_options_iname = '-pc_type -pc_factor_mat_solver_package'
  petsc_options_value = 'lu mumps'
  line_search = none
[]
[Outputs]
  csv = true
  console = false
[]
"""


def run_basis(tens_pref, shear_pref, tag):
    d = OUT / f"repro_{tag}"; d.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(str(d / "*fiber*.csv")):
        os.remove(f)
    ip = d / "in.i"; ip.write_text(make_input(tens_pref, shear_pref))
    import subprocess
    r = subprocess.run([MPIEXEC, "-n", "8", MOOSE, "-i", str(ip)], cwd=str(d), capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-800:]); raise SystemExit(f"{tag} failed")
    fc = sorted(glob.glob(str(d / "*fiber*.csv")))
    df = pd.read_csv(fc[-1]).sort_values("y")
    md = STAR + df["y"].to_numpy(float) / FT
    return md, df["strain_yy"].to_numpy(float) * 1e3   # millistrain


def main():
    md_obs = OBS["md_ft"]; obs = OBS["strain_4h"]; centers = pd.to_datetime(OBS["window_starts"])
    # two bases (unit prefactor each), interpolate onto obs MD
    mdt, byy_t = run_basis(1e-4, 0.0, "tensile")
    mds, byy_s = run_basis(0.0, 1e-4, "shear")
    bt = np.interp(md_obs, mdt[np.argsort(mdt)], byy_t[np.argsort(mdt)])
    bs = np.interp(md_obs, mds[np.argsort(mds)], byy_s[np.argsort(mds)])
    print(f"tensile basis peak {np.nanmax(np.abs(bt)):.4f} me @ MD {md_obs[np.nanargmax(np.abs(bt))]:.0f}")
    print(f"shear   basis peak {np.nanmax(np.abs(bs)):.4f} me @ MD {md_obs[np.nanargmax(np.abs(bs))]:.0f}")

    # per-window 2-parameter LS: obs[:,j] ~ a_j*bt + b_j*bs
    A = np.column_stack([bt, bs])
    coef = np.zeros((2, obs.shape[1]))
    model = np.zeros_like(obs)
    for j in range(obs.shape[1]):
        o = obs[:, j]; v = np.isfinite(o)
        c, *_ = np.linalg.lstsq(A[v], o[v], rcond=None)
        coef[:, j] = c; model[:, j] = A @ c
    a, b = coef
    vv = np.isfinite(obs)
    rms0 = np.sqrt(np.nanmean(obs[vv] ** 2)); rmsr = np.sqrt(np.nanmean((obs - model)[vv] ** 2))
    print(f"reproduce real DAS: RMS {rms0:.4f} -> {rmsr:.4f} me ({100*(1-rmsr/rms0):.0f}% var. reduction)")

    ext = [mdates.date2num(centers[0]), mdates.date2num(centers[-1]), md_obs[-1], md_obs[0]]
    fig, axs = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)
    for ax, dat, ti in [(axs[0], obs, "observed axial strain"), (axs[1], model, "reproduced (tensile+shear)"),
                        (axs[2], obs - model, "residual")]:
        im = ax.imshow(dat, aspect="auto", cmap="seismic", vmin=-0.09, vmax=0.09, extent=ext, interpolation="bilinear")
        ax.xaxis_date(); ax.set_ylim(10500, 10200); ax.set_title(ti); ax.set_xlabel("time")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d")); plt.colorbar(im, ax=ax, label="me")
    axs[0].set_ylabel("MD [ft]")
    fig.suptitle(f"Reproduce real DAS via 2-band eigenstrain inversion — {100*(1-rmsr/rms0):.0f}% var. reduction", fontweight="bold")
    fig.savefig(FIG / "reproduce_waterfall.png", dpi=140)

    fig2, ax2 = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax2.plot(centers, a, "r-o", ms=3, label="tensile amplitude (opening ~ pressure)")
    ax2.plot(centers, b, "b-o", ms=3, label="shear amplitude (~ slip)")
    ax2.axvline(pd.Timestamp("2025-02-28"), color="green", ls="--", label="T2")
    ax2.set_xlabel("time"); ax2.set_ylabel("eigenstrain amplitude"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_title("Recovered per-window tensile & shear amplitude histories")
    fig2.savefig(FIG / "reproduce_histories.png", dpi=140)
    print("saved", FIG / "reproduce_waterfall.png", "and", FIG / "reproduce_histories.png")
    np.savez(OUT / "reproduce_result.npz", tensile_amp=a, shear_amp=b, centers=[str(c) for c in centers],
             model=model, obs=obs, md=md_obs)


if __name__ == "__main__":
    main()
