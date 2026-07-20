"""MOOSE-vs-DDMpy demo: can a MOOSE FEM reproduce the DDM (Okada rectangular-dislocation)
mechanism for a PRESCRIBED displacement discontinuity?

Mechanism in MOOSE: a thin fault band (length L, thickness h) carrying a uniform EIGENSTRAIN
is elastically equivalent to a constant displacement-discontinuity element (a dislocation loop
around the band edge):
    opening  W  ->  eigenstrain  eps*_yy = W / h
    slip     S  ->  eigenstrain  eps*_xy = S / (2 h)
As h -> 0 with (eps* . h) fixed this converges to the DDM element. No mesh splitting / contact /
constraints needed -- standard solid_mechanics + ComputeEigenstrain, one linear Steady solve.

Reference: DDMpy_log/Element.py (the Okada-1992 / Yongzan Xue full-space rectangular constant-DD
kernel, nu=0.3, G-independent). We evaluate its FULL in-plane strain tensor at the fiber points by
finite-differencing u1,u3 in the x2=0 (mid-height) plane, so H is taken large -> plane strain, the
same 2D limit MOOSE solves.

Coordinate map  MOOSE (x=strike, y=fault-normal) <-> DDMpy (x1=length/strike, x3=normal, x2=dip):
    x <-> x1 ,  y <-> x3 ,  out-of-plane z <-> x2 .  Fault band at y=0.  Fiber vertical at x=offset.

Usage:  python moose_ddm_demo.py open       # opening/tensile DD
        python moose_ddm_demo.py slip       # strike-slip DD
"""
import glob
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "scripts/tensile_fault/data_transfer/from_pc")
from DDMpy_log.Element import Element  # noqa: E402

REPO = Path("/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner")
MOOSE = REPO / "moose_env/moose/modules/combined/combined-opt"
MPIEXEC = "/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec"
OUT = REPO / "output/moose_ddm_demo"
FIG = REPO / "figs/tensile_fault_qc/moose_ddm_demo"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

FT = 0.3048
NU = 0.3
E = 30e9                      # arbitrary: prescribed-DD strain is G-independent
# --- geometry (metres) ---
L = 200.0                    # fault length along strike (x)
H_DDM = 20000.0              # DDM height -> large so mid-height is plane-strain (matches 2D MOOSE)
HBAND = 2.0                  # eigenstrain-band thickness h
DOM = 300.0                  # half-domain (x,y in [-DOM, DOM])
OFFSET = 40 * FT             # fiber lateral offset from fault centre (40 ft)
SPAN = 150 * FT              # fiber half-length sampled (+/-150 ft)
NPTS = 601
GL = 30.48                   # gauge length (ft) boxcar
# graded mesh: fine centre, coarse outside
FINE_X, DXF = 140.0, 0.5     # x fine half-width, cell size (covers fault tips +-100 & fiber)
FINE_Y, DYF = 30.0, 0.25     # y fine half-width, cell size (near the fault crossing)
DCOARSE = 6.0                # coarse cell size outside the fine zone

W_OPEN = 0.008 * FT          # tensile opening (fault1 W_max)
S_SLIP = 0.03 * FT           # strike slip  (fault2 S1_max)


def gl_boxcar(y, eps, gl_ft=GL):
    dy = np.median(np.diff(y)); w = max(1, int(round(gl_ft * FT / dy)))
    return np.convolve(eps, np.ones(w) / w, mode="same")


def make_input(mode):
    """Return the MOOSE input-file text for 'open' or 'slip'.

    Graded Cartesian mesh, centred at origin: fine cells (DYF) in a central band around
    y=0 (which is a cell EDGE), coarse outside. The band bbox +/-DYF tags exactly the two
    rows adjacent to the fault (centroids +/-DYF/2) -> true thickness h_eff = 2*DYF. Scale
    the eigenstrain by h_eff so (eigenstrain * thickness) == the prescribed DD (opening/slip).
    """
    # graded segments (widths, #elements), symmetric; TRANSLATE_CENTER_ORIGIN re-centres to 0
    def segs(half_fine, dfine, half_dom, dcoarse):
        nf = int(round(2 * half_fine / dfine))          # fine centre
        nc = int(round((half_dom - half_fine) / dcoarse))
        w_out = half_dom - half_fine
        return (f"{w_out} {2*half_fine} {w_out}", f"{nc} {nf} {nc}")
    dxw, ixn = segs(FINE_X, DXF, DOM, DCOARSE)
    dyw, iyn = segs(FINE_Y, DYF, DOM, DCOARSE)
    h_eff = 2 * DYF
    half = DYF                                          # bbox half-height -> tags 2 rows
    if mode == "open":
        eyy, exy = W_OPEN / h_eff, 0.0
    else:
        eyy, exy = 0.0, S_SLIP / (2 * h_eff)
    eig = f"'{0.0} {eyy} {0.0} {0.0} {0.0} {exy}'"     # xx yy zz yz xz xy
    return f"""# MOOSE-DDM demo ({mode}): eigenstrain thin-band dislocation
[Mesh]
  [gen]
    type = CartesianMeshGenerator
    dim = 2
    dx = '{dxw}'
    ix = '{ixn}'
    dy = '{dyw}'
    iy = '{iyn}'
  []
  [centre]
    type = TransformGenerator
    input = gen
    transform = TRANSLATE_CENTER_ORIGIN
  []
  [band]
    type = SubdomainBoundingBoxGenerator
    input = centre
    block_id = 1
    bottom_left = '{-L/2} {-half} 0'
    top_right   = '{ L/2} { half} 0'
  []
[]

[GlobalParams]
  displacements = 'disp_x disp_y'
[]

[Physics/SolidMechanics/QuasiStatic]
  [all]
    strain = SMALL
    planar_formulation = PLANE_STRAIN
    add_variables = true
    eigenstrain_names = 'eig'
    generate_output = 'strain_xx strain_yy strain_xy'
  []
[]

[Materials]
  [elasticity]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = {E}
    poissons_ratio = {NU}
  []
  [stress]
    type = ComputeLinearElasticStress
  []
  [eig_band]
    type = ComputeEigenstrain
    eigen_base = {eig}
    eigenstrain_name = 'eig'
    block = 1
  []
  [eig_matrix]
    type = ComputeEigenstrain
    eigen_base = '0 0 0 0 0 0'
    eigenstrain_name = 'eig'
    block = 0
  []
[]

[BCs]
  [clamp_x]
    type = DirichletBC
    variable = disp_x
    boundary = 'left right top bottom'
    value = 0
  []
  [clamp_y]
    type = DirichletBC
    variable = disp_y
    boundary = 'left right top bottom'
    value = 0
  []
[]

[VectorPostprocessors]
  [fiber]
    type = LineValueSampler
    variable = 'disp_x disp_y strain_xx strain_yy strain_xy'
    start_point = '{OFFSET} {-SPAN} 0'
    end_point   = '{OFFSET} { SPAN} 0'
    num_points = {NPTS}
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
[]
"""


def run_moose(mode, nproc=8):
    ipath = OUT / f"demo_{mode}.i"
    ipath.write_text(make_input(mode))
    # clean old fiber csvs
    for f in glob.glob(str(OUT / f"demo_{mode}_out_fiber_*.csv")):
        os.remove(f)
    cmd = [MPIEXEC, "-n", str(nproc), str(MOOSE), "-i", str(ipath)]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(OUT), capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print("STDERR", r.stderr[-2000:]); raise SystemExit("MOOSE failed")
    fcsv = sorted(glob.glob(str(OUT / f"demo_{mode}_out_fiber_*.csv")))
    if not fcsv:
        fcsv = sorted(glob.glob(str(OUT / f"demo_{mode}_out_*.csv")))
    import pandas as pd
    df = pd.read_csv(fcsv[-1]).sort_values("y")
    return {k: df[k].to_numpy(float) for k in
            ["y", "disp_x", "disp_y", "strain_xx", "strain_yy", "strain_xy"]}


def ddm_disp(mode, y):
    """DDMpy displacements (u1 along strike, u3 along fiber/normal) at the fiber x1=OFFSET, x2=0."""
    W = W_OPEN if mode == "open" else 0.0
    S = S_SLIP if mode == "slip" else 0.0
    e = Element(length=L, height=H_DDM, width=W, S1=S, S2=0.0); e.mu = NU
    e.set_coors(np.full_like(y, OFFSET), np.zeros_like(y), y.astype(float).copy())
    return e.u1, e.u3


def dstrain(u, s):
    """axial strain from displacement by the SAME centred-difference operator DDMpy's Well uses."""
    return gl_boxcar(s, np.gradient(u, s))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "open"
    m = run_moose(mode)
    y = m["y"]; md = 10373.4 + y / FT
    du1, du3 = ddm_disp(mode, y)                        # DDMpy displacements
    # dominant response component: opening -> disp_y(=u3); strike-slip -> disp_x(=u1)
    if mode == "open":
        mu_ax, du_ax, mstrain, clabel = m["disp_y"], du3, m["strain_yy"], "disp_y / eps_yy"
    else:
        mu_ax, du_ax, mstrain, clabel = m["disp_x"], du1, m["strain_xy"], "disp_x / eps_xy"
    # axial strain, IDENTICAL operator (gradient of the disp component + boxcar) for both
    me_ax = dstrain(mu_ax, y) * 1e3; de_ax = dstrain(du_ax, y) * 1e3
    # element strain reported by MOOSE (RankTwoAux), boxcar-averaged, for reference
    me_elem = gl_boxcar(y, mstrain) * 1e3

    def metrics(a, b):
        good = np.abs(b) > 0.02 * max(np.nanmax(np.abs(b)), 1e-30)
        cc = np.corrcoef(a[good], b[good])[0, 1] if good.sum() > 5 else np.nan
        r = np.nanmax(np.abs(a)) / max(np.nanmax(np.abs(b)), 1e-30)
        return r, cc

    print(f"\n=== {mode.upper()} : MOOSE vs DDMpy ===")
    print(f"  axial disp  peak: MOOSE {np.nanmax(np.abs(mu_ax))*1e3:.4f}  DDMpy {np.nanmax(np.abs(du_ax))*1e3:.4f} mm  "
          f"(ratio {np.nanmax(np.abs(mu_ax))/max(np.nanmax(np.abs(du_ax)),1e-30):.3f})")
    r1, c1 = metrics(me_ax, de_ax); print(f"  axial strain (matched d/ds op): ratio {r1:.3f}  corr {c1:.4f}")
    r2, c2 = metrics(me_elem, de_ax); print(f"  axial strain (MOOSE element eps): ratio {r2:.3f}  corr {c2:.4f}")

    fig, axs = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    axs[0].plot(du_ax * 1e3, md, "k-", lw=2.4, label="DDMpy u_axial")
    axs[0].plot(mu_ax * 1e3, md, "r--", lw=1.8, label="MOOSE disp_y")
    axs[0].set_title("axial DISPLACEMENT [mm]"); axs[0].set_xlabel("mm")
    axs[1].plot(de_ax, md, "k-", lw=2.4, label=f"DDMpy (peak {np.nanmax(np.abs(de_ax)):.3f})")
    axs[1].plot(me_ax, md, "r--", lw=1.8, label=f"MOOSE (peak {np.nanmax(np.abs(me_ax)):.3f})")
    axs[1].set_title(f"axial STRAIN, matched operator\nratio {r1:.2f}  corr {c1:.3f}"); axs[1].set_xlabel("me")
    axs[2].plot(de_ax, md, "k-", lw=2.4, label="DDMpy")
    axs[2].plot(me_elem, md, "b--", lw=1.8, label="MOOSE element eps_yy")
    axs[2].set_title(f"MOOSE element strain vs DDMpy\nratio {r2:.2f}  corr {c2:.3f}"); axs[2].set_xlabel("me")
    for ax in axs:
        ax.invert_yaxis(); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axs[0].set_ylabel("MD [ft]")
    fig.suptitle(f"MOOSE eigenstrain-band  vs  DDMpy(Okada)  —  {mode} DD  "
                 f"(L={L:.0f} m, h_band={2*DYF:.2g} m, nu={NU}, GL={GL:.1f} ft)", fontweight="bold")
    fig.tight_layout()
    p = FIG / f"moose_vs_ddm_{mode}.png"; fig.savefig(p, dpi=140); print("saved", p)


if __name__ == "__main__":
    main()
