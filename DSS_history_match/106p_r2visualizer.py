# scripts/DSS_history_match/106p_r2visualizer.py
# This script visualizes the 2D waterfall plots from the line samplers
# of a MOOSE simulation run, specifically for pressure and strain.
# Shenyao Jin, shenyaojin@mines.edu, 11/24/2025

import os
import matplotlib.pyplot as plt
import datetime
from typing import List, Tuple
import numpy as np

from fiberis.analyzer.Data2D.core2D import Data2D
from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge


def extract_vector_postprocessor_data(output_dir: str) -> Tuple[Data2D, Data2D, datetime.datetime]:
    """
    Extracts pressure and strain data from MOOSE vector postprocessor outputs.

    This function scans the output directory for line sampler CSVs, reads them,
    and converts them into Data2D objects suitable for plotting. It also
    reconstructs the simulation start time from the original gauge data files
    to ensure correct timestamping.

    :param output_dir: The directory containing the MOOSE simulation output files.
    :return: A tuple containing the pressure Data2D object, the strain Data2D object,
             and the simulation start time.
    """
    # --- 1. Extract Data using MOOSEVectorPostProcessorReader ---
    vector_reader = MOOSEVectorPostProcessorReader()
    try:
        max_processor_id, _ = vector_reader.get_max_indices(output_dir)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find simulation output files in '{output_dir}'. "
                                f"Please ensure the simulation has been run successfully.")

    pressure_data2d = None
    strain_data2d = None

    # Loop through all vector postprocessor output files
    for i in range(max_processor_id + 1):
        # variable_index=1 assumes the first variable in each file is the one of interest
        vector_reader.read(directory=output_dir, post_processor_id=i, variable_index=1)

        if "pressure_monitor_well" in vector_reader.sampler_name:
            pressure_data2d = vector_reader.to_analyzer()
            print(f"Found and loaded pressure data from '{vector_reader.sampler_name}'.")
        elif "fiber_strain" in vector_reader.sampler_name:
            strain_data2d = vector_reader.to_analyzer()
            print(f"Found and loaded strain data from '{vector_reader.sampler_name}'.")

    if pressure_data2d is None:
        raise FileNotFoundError("Could not find and extract 'pressure_monitor_well' data.")
    if strain_data2d is None:
        raise FileNotFoundError("Could not find and extract 'fiber_strain' data.")

    # --- 2. Determine the simulation start time for correct timestamping ---
    # This logic is replicated from the simulation script to ensure consistency.
    pressure_gauge_g1_path = "data/fiberis_format/prod/gauges/pressure_g1.npz"
    injection_gauge_pressure_path = "data/fiberis_format/prod/gauges/gauge4_data_prod.npz"

    gauge_data_interference = Data1DGauge()
    gauge_data_interference.load_npz(pressure_gauge_g1_path)
    injection_gauge_pressure = Data1DGauge()
    injection_gauge_pressure.load_npz(injection_gauge_pressure_path)
    injection_gauge_pressure.select_time(injection_gauge_pressure.start_time, gauge_data_interference.start_time)

    sim_start_time = injection_gauge_pressure.start_time
    pressure_data2d.start_time = sim_start_time
    strain_data2d.start_time = sim_start_time

    return pressure_data2d, strain_data2d, sim_start_time


def visualize_waterfall_plots(pressure_data: Data2D, strain_data: Data2D, fig_output_dir: str):
    """
    Generates and saves 2D waterfall plots for pressure and strain data.

    :param pressure_data: Data2D object for pressure.
    :param strain_data: Data2D object for strain.
    :param fig_output_dir: Directory to save the output plot images.
    """
    print(f"Generating waterfall plots and saving to '{fig_output_dir}'...")
    # --- Plot Pressure Data ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    pressure_data.plot(ax=ax1, use_timestamp=False, cmap="bwr", method='pcolormesh')
    ax1.set_title("Pressure Profile Over Time (Waterfall Plot)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Position along monitoring line (m)")
    fig1.tight_layout()
    pressure_fig_path = os.path.join(fig_output_dir, "pressure_waterfall.png")
    fig1.savefig(pressure_fig_path)
    print(f"Saved pressure plot to '{pressure_fig_path}'")
    plt.close(fig1)

    # --- Plot Strain Data ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    strain_data.plot(ax=ax2, use_timestamp=False, cmap="bwr", method='pcolormesh', clim=(-2e-5, 2e-5))
    ax2.set_title("Strain (yy) Profile Over Time (Waterfall Plot)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Position along monitoring line (m)")
    fig2.tight_layout()
    strain_fig_path = os.path.join(fig_output_dir, "strain_waterfall.png")
    fig2.savefig(strain_fig_path)
    print(f"Saved strain plot to '{strain_fig_path}'")
    plt.close(fig2)


if __name__ == "__main__":
    # Define the directory where the simulation results from script 106r2 are stored
    simulation_output_dir = "output/1124_DSS_3xSingleFrac_match"

    # Define a directory to save the figures
    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    figure_output_dir = os.path.join("figs", f"waterfall_plots_{date_time_str}")
    os.makedirs(figure_output_dir, exist_ok=True)

    try:
        # 1. Extract the data from the simulation output files
        pressure_dataframe, strain_dataframe, start_time = extract_vector_postprocessor_data(simulation_output_dir)
        dss_data = Data2D()
        dss_data.load_npz("data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz")
        pressure_dataframe.select_time(dss_data.start_time, dss_data.get_end_time())
        strain_dataframe.select_time(dss_data.start_time, dss_data.get_end_time())

        # Apply baseline correction after time cropping
        if strain_dataframe.data.shape[0] > 0:
            # Subtract the first value of each channel (row) from that entire channel
            strain_dataframe.data -= strain_dataframe.data[:, 0][:, np.newaxis]
            print("Applied baseline correction to each channel in the strain data.")
            print(np.min(strain_dataframe.data), np.max(strain_dataframe.data))

        # 2. Generate and save the waterfall plots
        visualize_waterfall_plots(pressure_dataframe, strain_dataframe, figure_output_dir)

        # 3. Plot single channel comparison
        frac_loc = 14888
        # Get simulated strain at the center of the monitoring line
        sim_strain_1d = strain_dataframe.get_value_by_depth(0.5 * (strain_dataframe.daxis[0] + strain_dataframe.daxis[-1]))
        sim_gauge = Data1DGauge(data=sim_strain_1d, taxis=strain_dataframe.taxis, start_time=strain_dataframe.start_time)

        # Get observed data at the specified depth
        obs_data_array = dss_data.get_value_by_depth(frac_loc)

        if obs_data_array is not None:
            obs_gauge = Data1DGauge(data=obs_data_array * 1e-6, taxis=dss_data.taxis, start_time=dss_data.start_time)
            obs_gauge.name = f"Observed Strain at {frac_loc} ft"

            # Interpolate observation data to simulation time axis
            obs_gauge.interpolate(sim_gauge.taxis, sim_gauge.start_time)

            # Apply baseline correction to the observed data
            obs_gauge.data -= obs_gauge.data[0]

            # Plotting the comparison
            fig, ax = plt.subplots(figsize=(12, 6))
            sim_gauge.plot(ax=ax, use_timestamp=True, label=f'Simulated Strain (yy)')
            obs_gauge.plot(ax=ax, use_timestamp=True, label=f'Measured Strain')
            ax.set_title(f'Strain Comparison at Depth {frac_loc:.2f} ft')
            ax.set_ylabel('Strain (yy)')
            ax.set_xlabel('Time')
            ax.legend()
            plt.tight_layout()

            # Save the figure
            fig_path = os.path.join(figure_output_dir, f"strain_comparison_depth_{int(frac_loc)}.png")
            plt.savefig(fig_path)
            print(f"Single channel comparison plot saved to: {fig_path}")
            plt.close(fig)
        else:
            print(f"Warning: Could not retrieve observation data for depth {frac_loc}. Skipping single channel plot.")

        print("\nVisualization complete.")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure that the simulation has been run and the output directory is correct.")
