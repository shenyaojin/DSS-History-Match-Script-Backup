"""Re-run phase 3 of the 0211 two-stage chain at a chosen barrier ratio.

Numerics, mesh, source and initial condition are copied verbatim from
scripts/well_leakage_history_matching/101_fiberis_matching.py (:75-84 mesh,
:93 D=140, :186-190 barrier, :198-202 initial from phase2, :204-209 source,
:212-213 solve). Nothing here is a new modelling choice.
"""
import sys, os, argparse
import numpy as np
sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg')
from fiberis.simulator.core import pds
from fiberis.utils import mesh_utils
from fiberis.analyzer.Data1D import Data1D_Gauge, Data1D_PumpingCurve
from DSS_analyzer_Mariner import Data3D_geometry

ap = argparse.ArgumentParser()
ap.add_argument('--ratio', type=float, required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()

datapath = "data/fiberis_format/"
pg = datapath + "s_well/gauges/"
gauge_next = Data1D_Gauge.Data1DGauge(); gauge_next.load_npz(pg + "gauge7_data_swell.npz")
pc8 = Data1D_PumpingCurve.Data1DPumpingCurve(); pc8.load_npz(datapath + "prod/pumping_data/stage8/Slurry Rate.npz")
stg8_bg, stg8_ed = pc8.get_start_time(), pc8.get_end_time()
src = gauge_next.copy(); src.crop(stg8_bg, stg8_ed); src.rename("Phase 3 Source")

fhp = "data/legacy/s_well/geometry/frac_hit/"
fh7 = Data3D_geometry.Data3D_geometry(fhp + "frac_hit_stage_7_swell.npz").data
fh8 = Data3D_geometry.Data3D_geometry(fhp + "frac_hit_stage_8_swell.npz").data

dx, nx = 1, 5500
x = np.arange(12500, 12500 + nx * dx, dx)
for m in np.round(fh7): x = mesh_utils.refine_mesh(x, [m - 1, m + 1], 5)
for m in np.round(fh8): x = mesh_utils.refine_mesh(x, [m - 1, m + 1], 5)
idx7 = [mesh_utils.locate(x, m)[0] for m in fh7]
idx8 = [mesh_utils.locate(x, m)[0] for m in fh8]
print(f"mesh nodes {len(x)}  stage7 barrier nodes {idx7}  stage8 source nodes {idx8}")

d = 140
d3 = np.ones_like(x, dtype=float) * d
for i in idx7:
    d3[i] = d * a.ratio

f = pds.PDS1D_MultiSource()
f.set_mesh(x); f.set_bcs('Neumann', 'Neumann'); f.set_sourceidx(idx8)
f.set_diffusivity(d3)
prev = np.load("output/0211_simulation_MULTIstage/phase2.npz", allow_pickle=True)
f.set_initial(prev['data'][-1, :])          # phase2.npz is time-major: [-1,:] IS the last profile
f.set_source([src for _ in idx8])
f.solve(optimizer=True, dt_init=2, print_progress=False,
        max_dt=30, min_dt=1e-4, tol=1e-3, safety_factor=0.9, p=2)
f.pack_result(filename=a.out)
print("wrote", a.out)
