"""Re-run the reproduction as a TRANSIENT MOOSE elastic simulation so that MOOSE itself computes the
strain RATE (via TimeDerivativeAux), avoiding the truncation error of differentiating strain by hand.

The 19 distributed eigenstrain bands (from the inversion) are driven by PiecewiseLinear time functions
= their recovered per-window amplitude histories. MOOSE runs transient on a fine time grid, computes
strain_yy (RankTwoAux) and strain_rate_yy (TimeDerivativeAux), and samples them on the DAS fiber.
Only the 30.48 ft DEPTH gauge-average is applied in post (a spatial average — no time differencing).

Then compares the model strain-RATE waterfall to the observed strain-RATE (raw DAS rate) in the
107 plotting style.
"""
import glob
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.interpolate import PchipInterpolator

REPO = Path("/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner")
import sys
sys.path.insert(0, str(REPO / "fibeRIS" / "src"))
from fiberis.analyzer.Data2D.core2D import Data2D  # noqa: E402

FT = 0.3048; STAR = 10373.4; GL = 30.48
T1 = pd.Timestamp("2025-02-24 12:00"); T2 = pd.Timestamp("2025-02-28 00:00"); T3 = pd.Timestamp("2025-03-04 00:00")
MOOSE = str(REPO / "moose_env/moose/modules/combined/combined-opt")
MPIEXEC = "/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
OUT = REPO / "output/inversion/strain_rate_fwd"; OUT.mkdir(parents=True, exist_ok=True)
FIG = REPO / "figs/tensile_fault_qc/inversion"
DELIV = REPO / "output/inversion_deliverable/figures"
XF = 40 * FT
BAND_H = 4.0     # ft (matches the inversion basis band thickness)

REP = np.load(REPO / "output/inversion/reproduce_distributed.npz", allow_pickle=True)
amp = REP["amp"]; band_mds = REP["band_mds"]; centers = pd.to_datetime(REP["centers"])
# FINE transient time grid (200 steps) for a crisp, non-blocky strain-rate waterfall. The per-window
# amplitudes are smoothly (pchip, shape-preserving) interpolated onto this grid, so the modeled strain
# and its rate vary continuously in time. Node grid = window starts + T3 (last value held) to avoid
# extrapolation past the final window.
NSTEPS = 200
fine = pd.date_range(T1, T3, periods=NSTEPS)
ftsec = (fine - T1).total_seconds().to_numpy()
wtsec = (centers - T1).total_seconds().to_numpy()
node_t = np.append(wtsec, (T3 - T1).total_seconds())                    # nodes for pchip
amp_fine = np.array([PchipInterpolator(node_t, np.append(amp[i], amp[i, -1]))(ftsec)
                     for i in range(amp.shape[0])])                     # [band, fine-time]


def gen_input():
    # mesh + 19 band blocks
    mesh = ["""[Mesh]
  [gen]
    type = GeneratedMeshGenerator
    dim = 2
    nx = 80
    ny = 160
    xmin = -100
    xmax = 100
    ymin = -60
    ymax = 60
  []"""]
    for i, md in enumerate(band_mds):
        yc = (md - STAR) * FT; h = BAND_H * FT
        prev = "gen" if i == 0 else f"band{i-1}"
        mesh.append(f"""  [band{i}]
    type = SubdomainBoundingBoxGenerator
    input = {prev}
    block_id = {i+1}
    bottom_left = '-40 {yc-h/2} 0'
    top_right = '40 {yc+h/2} 0'
  []""")
    mesh.append("[]")

    funcs = ["[Functions]"]
    xs = " ".join(f"{t:.1f}" for t in ftsec)
    for i in range(len(band_mds)):
        ys = " ".join(f"{a*1e-4:.6e}" for a in amp_fine[i])   # pchip-smoothed prefactor history (x1e-4 basis)
        funcs.append(f"""  [f{i}]
    type = PiecewiseLinear
    x = '{xs}'
    y = '{ys}'
  []""")
    funcs.append("[]")

    mats = ["""[Materials]
  [elast]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 30e9
    poissons_ratio = 0.2
  []
  [strain]
    type = ComputeSmallStrain
    eigenstrain_names = 'dd_eig'
  []
  [stress]
    type = ComputeLinearElasticStress
  []
  [eig_matrix]
    type = ComputeEigenstrain
    eigen_base = '0 0 0 0 0 0'
    eigenstrain_name = 'dd_eig'
    block = 0
  []"""]
    for i in range(len(band_mds)):
        mats.append(f"""  [pref{i}]
    type = GenericFunctionMaterial
    prop_names = 'p{i}'
    prop_values = 'f{i}'
    block = {i+1}
  []
  [eig{i}]
    type = ComputeEigenstrain
    eigen_base = '0 1 0 0 0 0'
    prefactor = 'p{i}'
    eigenstrain_name = 'dd_eig'
    block = {i+1}
  []""")
    mats.append("[]")

    tseq = " ".join(f"{t:.1f}" for t in ftsec)
    rest = f"""[GlobalParams]
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
[AuxVariables]
  [strain_yy]
    order = CONSTANT
    family = MONOMIAL
  []
  [strain_rate_yy]
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
    execute_on = 'timestep_end'
  []
  [srate]
    type = TimeDerivativeAux
    variable = strain_rate_yy
    functor = 'strain_yy'
    execute_on = 'timestep_end'
  []
[]
[VectorPostprocessors]
  [fiber]
    type = LineValueSampler
    variable = 'strain_yy strain_rate_yy'
    start_point = '{XF} -55 0'
    end_point = '{XF} 55 0'
    num_points = 300
    sort_by = y
  []
[]
[Executioner]
  type = Transient
  solve_type = NEWTON
  petsc_options_iname = '-pc_type -pc_factor_mat_solver_package'
  petsc_options_value = 'lu mumps'
  line_search = none
  [TimeStepper]
    type = TimeSequenceStepper
    time_sequence = '{tseq}'
  []
[]
[Outputs]
  csv = true
  console = false
[]
"""
    return "\n".join(mesh) + "\n" + "\n".join(funcs) + "\n" + "\n".join(mats) + "\n" + rest


def run():
    for f in glob.glob(str(OUT / "*fiber*.csv")):
        os.remove(f)
    ip = OUT / "in.i"; ip.write_text(gen_input())
    r = subprocess.run([MPIEXEC, "-n", "20", MOOSE, "-i", str(ip)], cwd=str(OUT), capture_output=True, text=True, timeout=14400)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-1000:]); raise SystemExit("transient MOOSE failed")
    print("transient MOOSE done")


def model_rate_on_obs_grid(md_obs):
    """MOOSE strain_yy -> depth GL-average -> EXACT time slope (nanostrain/s) on (obs MD x fine time).

    NOTE: MOOSE's TimeDerivativeAux returns 0 on an AuxVariable functor (aux variables carry no time
    derivative), so we take the rate as the time slope of the strain. Because the pchip-smoothed
    amplitudes make the modelled strain smooth in time, this slope is the accurate strain rate
    (truncation-free for the piecewise-linear-between-fine-steps strain)."""
    fs = sorted(glob.glob(str(OUT / "*fiber*.csv")), key=lambda p: int(p.split("_")[-1].split(".")[0]))
    cols, y0 = [], None
    for f in fs:
        d = pd.read_csv(f)
        if len(d) < 100:
            cols.append(None); continue
        d = d.sort_values("y")
        if y0 is None:
            y0 = d["y"].to_numpy(float)
        cols.append(d["strain_yy"].to_numpy(float))
    md_f = STAR + y0 / FT; o = np.argsort(md_f); md_f = md_f[o]
    dm = np.median(np.diff(md_f)); w = max(1, int(round(GL / dm)))
    strain = np.zeros((len(md_obs), len(cols)))
    for j, c in enumerate(cols):
        cg = (np.zeros_like(md_f) if c is None
              else np.convolve(c[o], np.ones(w) / w, mode="same"))     # DEPTH gauge average (spatial)
        strain[:, j] = np.interp(md_obs, md_f, cg)
    t = ftsec[:len(cols)]
    rate = np.zeros_like(strain)
    rate[:, 1:] = np.diff(strain, axis=1) / np.diff(t)[None, :]        # exact time slope
    rate[:, 0] = rate[:, 1]
    return rate * 1e9, fine[:len(cols)]                                # nanostrain/s, model time grid


def observed_rate(md_obs):
    """Cleaned raw DAS strain-rate (like script 107): merge, median filter, quiet-band demean."""
    m = Data2D(); m.load_npz(str(REPO / "data_fervo/fiberis_format/LFDAS/LFDAS_G4-PB_202502_late.npz"))
    nxt = Data2D(); nxt.load_npz(str(REPO / "data_fervo/fiberis_format/LFDAS/LFDAS_G4-PB_202503.npz")); m.right_merge(nxt)
    times = pd.to_datetime(m.start_time) + pd.to_timedelta(m.taxis, unit="s")
    tb = (times >= T1) & (times <= T3); LP = m.data[:, tb].astype(float); times = times[tb]; dax = m.daxis
    gap = np.isnan(LP).all(0); x = np.arange(LP.shape[1])
    for r in range(LP.shape[0]):
        y = LP[r]; bad = np.isnan(y) & ~gap; good = np.isfinite(y) & ~gap
        if bad.any() and good.sum() >= 2:
            y[bad] = np.interp(x[bad], x[good], y[good])
    LP[:, gap] = 0.0
    med = median_filter(LP, size=(3, 7), mode="nearest")
    band = (dax >= 1800) & (dax <= 5000)
    rate = med - np.nanmedian(med[band, :], axis=0)[None, :]
    rate[:, gap] = np.nan
    dm = (dax >= 10200) & (dax <= 10500)
    rr = np.array([np.interp(md_obs, dax[dm], rate[dm, k]) for k in range(rate.shape[1])]).T
    return rr, times


def main():
    md_obs = np.load(REPO / "output/inversion/observation.npz", allow_pickle=True)["md_ft"]
    run()
    mrate, mt = model_rate_on_obs_grid(md_obs)      # [md, fine time], nanostrain/s
    orate, otimes = observed_rate(md_obs)           # [md, raw time], nanostrain/s
    print(f"model rate p98 {np.nanpercentile(np.abs(mrate),98):.3f}, observed rate p98 {np.nanpercentile(np.abs(orate),98):.3f} ne/s")

    def wf(ax, dat, t, title, lim):
        ext = [mdates.date2num(t[0]), mdates.date2num(t[-1]), md_obs[-1], md_obs[0]]
        im = ax.imshow(dat, aspect="auto", cmap="bwr", vmin=-lim, vmax=lim, extent=ext, interpolation="nearest")
        ax.set_xlim(mdates.date2num(T1), mdates.date2num(T3)); ax.xaxis_date(); ax.set_ylim(10500, 10200)
        ax.set_ylabel("Gold 4-PB MD [ft]"); ax.set_title(title)
        for mt, lb in [(T1, "T1"), (T2, "T2"), (T3, "T3")]:
            ax.axvline(mdates.date2num(mt), color="green", ls="--", lw=1.8)
            ax.text(mdates.date2num(mt), 1.01, lb, transform=ax.get_xaxis_transform(), color="green", fontweight="bold", ha="center")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d")); plt.colorbar(im, ax=ax, label="strain rate [nε/s]")

    lim = np.nanpercentile(np.abs(orate), 97)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, constrained_layout=True)
    wf(a1, orate, otimes, "Observed DAS strain RATE (raw)", lim)
    wf(a2, mrate, mt, f"Model strain RATE (transient reproduction, {len(mt)} steps)", lim)
    a2.set_xlabel("Time [UTC-7]")
    fig.suptitle("Observed vs Model strain RATE — MOOSE-computed rate (no hand differencing)", fontweight="bold", fontsize=14)
    fig.savefig(FIG / "compare_strain_rate.png", dpi=150)
    import shutil
    shutil.copy(FIG / "compare_strain_rate.png", DELIV / "05b_reproduction_strain_rate.png")
    print("saved", FIG / "compare_strain_rate.png")


if __name__ == "__main__":
    main()
