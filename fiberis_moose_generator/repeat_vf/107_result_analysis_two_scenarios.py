#%% To test the improved kernel files and compare their results.
import numpy as np
import matplotlib.pyplot as plt
import os
from fiberis.analyzer.Data1D import core1D

# This script visualizes the results from the two "improved kernel" scenarios.
# It is based on the workflow from 104_result_analysis_three_scenarios.py.

# --- Configuration ---
base_path = "scripts/fiberis_moose_generator/repeat_vf/saved_npz"
viz_output_dir = "scripts/fiberis_moose_generator/repeat_vf/viz_results/107_analysis_two_scenarios"

# Define paths for the two scenarios' NPZ files
scenario1_path = os.path.join(base_path, "106_improved_kernel")
scenario2_path = os.path.join(base_path, "106_improved_kernel_porou_DT")

# Check if NPZ directories exist
if not os.path.exists(scenario1_path) or not os.path.exists(scenario2_path):
    raise FileNotFoundError("NPZ directories not found. Please run the simulation script (106) first.")

# --- Define File Paths ---
# What I'm concerned about is 1. pore pressure at production well, 2. strain_yy
scen1_pp_filepath = os.path.join(scenario1_path, "pp_prod.npz")
scen1_strain_filepath = os.path.join(scenario1_path, "strain_yy_prod.npz")

scen2_pp_filepath = os.path.join(scenario2_path, "pp_prod.npz")
scen2_strain_filepath = os.path.join(scenario2_path, "strain_yy_prod.npz")

# --- Load Data from NPZ Files ---
pressure_s1 = core1D.Data1D()
strain_s1 = core1D.Data1D()
pressure_s2 = core1D.Data1D()
strain_s2 = core1D.Data1D()

pressure_s1.load_npz(scen1_pp_filepath)
strain_s1.load_npz(scen1_strain_filepath)
pressure_s2.load_npz(scen2_pp_filepath)
strain_s2.load_npz(scen2_strain_filepath)

# --- Pre-processing: remove the first data point ---
pressure_s1.select_time(50, pressure_s1.taxis[-1].astype(float))
strain_s1.select_time(50, strain_s1.taxis[-1].astype(float))
pressure_s2.select_time(50, pressure_s2.taxis[-1].astype(float))
strain_s2.select_time(50, strain_s2.taxis[-1].astype(float))


# --- Plot Pore Pressure Comparison ---
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

pressure_s1.plot(ax=ax1)
ax1.set_title("Pore Pressure at Production Well - Improved Kernel")
ax1.set_ylabel("Pore Pressure (Pa)")

pressure_s2.plot(ax=ax2)
ax2.set_title("Pore Pressure at Production Well - Improved Kernel w/ Porous DT")
ax2.set_ylabel("Pore Pressure (Pa)")

ax2.set_xlabel("Time (s)")
plt.tight_layout()


# --- Plot Strain_yy Comparison ---
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

strain_s1.plot(ax=ax1)
ax1.set_title("Strain_yy at Production Well - Improved Kernel")
ax1.set_ylabel("Strain_yy")

strain_s2.plot(ax=ax2)
ax2.set_title("Strain_yy at Production Well - Improved Kernel w/ Porous DT")
ax2.set_ylabel("Strain_yy")

ax2.set_xlabel("Time (s)")
plt.tight_layout()

# --- Save Figures ---
os.makedirs(viz_output_dir, exist_ok=True)
fig1_path = os.path.join(viz_output_dir, "pressure_prod_two_scenarios.png")
fig2_path = os.path.join(viz_output_dir, "strain_yy_prod_two_scenarios.png")

fig1.savefig(fig1_path, dpi=300)
fig2.savefig(fig2_path, dpi=300)

print(f"Comparison plots saved to {viz_output_dir}")