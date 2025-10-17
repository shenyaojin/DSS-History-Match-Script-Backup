#%% To test the Victor's input file, and the impact of removing single kernel
import numpy as np
import matplotlib.pyplot as plt

from fiberis.analyzer.Data1D import core1D
from fiberis.analyzer.Data2D import core2D

#%% Define file paths
# What I'm concerned about is 1. pore pressure at production well, 2. strain_yy
base_path = "scripts/fiberis_moose_generator/repeat_vf/saved_npz"

import os
porous_model_filepath_pp = os.path.join(base_path, "porous_kernel/pp_prod.npz")
porous_model_filepath_strain = os.path.join(base_path, "porous_kernel/strain_yy_prod.npz")

single_diffusion_filepath_pp = os.path.join(base_path, "single_diffusion/pp_prod.npz")
single_diffusion_filepath_strain = os.path.join(base_path, "single_diffusion/strain_yy_prod.npz")

with_kernel_filepath_pp = os.path.join(base_path, "with_kernel/pp_prod.npz")
with_kernel_filepath_strain = os.path.join(base_path, "with_kernel/strain_yy_prod.npz")

#%% Load data from DOT npz files
pressure_porous = core1D.Data1D()
strain_porous = core1D.Data1D()
pressure_single_diffusion = core1D.Data1D()
strain_single_diffusion = core1D.Data1D()
pressure_with_kernel = core1D.Data1D()
strain_with_kernel = core1D.Data1D()

pressure_porous.load_npz(porous_model_filepath_pp)
strain_porous.load_npz(porous_model_filepath_strain)
pressure_single_diffusion.load_npz(single_diffusion_filepath_pp)
strain_single_diffusion.load_npz(single_diffusion_filepath_strain)
pressure_with_kernel.load_npz(with_kernel_filepath_pp)
strain_with_kernel.load_npz(with_kernel_filepath_strain)

#%% Pre-processing: remove the first data point
pressure_porous.select_time(50, pressure_porous.taxis[-1].astype(float))
strain_porous.select_time(50, strain_porous.taxis[-1].astype(float))
pressure_single_diffusion.select_time(50, pressure_single_diffusion.taxis[-1].astype(float))
strain_single_diffusion.select_time(50, strain_single_diffusion.taxis[-1].astype(float))
pressure_with_kernel.select_time(50, pressure_with_kernel.taxis[-1].astype(float))
strain_with_kernel.select_time(50, strain_with_kernel.taxis[-1].astype(float))


#%% Plot all three scenarios in one figure for pore pressure at production well

fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

pressure_porous.plot(ax=ax1)
ax1.set_title("Pore Pressure at Production Well - Porous Kernel Model")
ax1.set_ylabel("Pore Pressure (Pa)")

pressure_single_diffusion.plot(ax=ax2)
ax2.set_title("Pore Pressure at Production Well - Single Diffusion Model")
ax2.set_ylabel("Pore Pressure (Pa)")

pressure_with_kernel.plot(ax=ax3)
ax3.set_title("Pore Pressure at Production Well - With dual-kernel Model")
ax3.set_ylabel("Pore Pressure (Pa)")

ax3.set_xlabel("Time (s)")
plt.tight_layout()


#%% Plot all three scenarios in one figure for strain_yy at production well
fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
strain_porous.plot(ax=ax1)
ax1.set_title("Strain_yy at Production Well - Porous Kernel Model")
ax1.set_ylabel("Strain_yy")
strain_single_diffusion.plot(ax=ax2)
ax2.set_title("Strain_yy at Production Well - Single Diffusion Model")
ax2.set_ylabel("Strain_yy")
strain_with_kernel.plot(ax=ax3)
ax3.set_title("Strain_yy at Production Well - With dual-kernel Model")
ax3.set_ylabel("Strain_yy")
ax3.set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("scripts/fiberis_moose_generator/repeat_vf/viz_results/sensitivity_analysis_figs/strain_yy_prod_three_scenarios.png", dpi=300)