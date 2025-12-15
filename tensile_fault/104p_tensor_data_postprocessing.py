import os
import numpy as np
import matplotlib.pyplot as plt
from fiberis.io.reader_moose_tensor_vpp import MOOSETensorVPPReader
from fiberis.analyzer.TensorProcessor.coreT2D import Tensor2D
from fiberis.analyzer.Data2D.core2D import Data2D
from fiberis.analyzer.Data1D.core1D import Data1D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy

def process_tensor_data(output_dir: str, fig_dir: str):
    """
    Reads, processes, and plots strain tensor data from MOOSE VectorPostprocessor outputs.
    """
    # Ensure the figure directory exists
    os.makedirs(fig_dir, exist_ok=True)

    # Load pressure gauge data
    pg_data_orig = Data1DGauge()
    pg_data_orig.load_npz("data_fervo/fiberis_format/post_processing/Bearskin3PA_Stage_28_timestep_profile.npz")

    # Load the real start time
    try:
        time_profile = Data1D()
        time_profile.load_npz("data_fervo/fiberis_format/post_processing/Bearskin3PA_Stage_28_timestep_profile.npz")
        real_start_time = time_profile.start_time
        print(f"Successfully loaded real start time: {real_start_time}")
    except Exception as e:
        print(f"Could not load start time profile. Error: {e}")
        return

    reader = MOOSETensorVPPReader()
    available_samplers = reader.list_available_samplers(output_dir)

    if not available_samplers:
        print(f"No strain tensor samplers found in {output_dir}")
        return

    for sampler_name in available_samplers:
        print(f"\n--- Processing data from sampler: {sampler_name} ---")

        # Read the data for the current sampler
        reader.read(directory=output_dir, sampler_name=sampler_name)

        # Convert to Tensor2D analyzer object
        tensor_data: Tensor2D = reader.to_analyzer()

        # --- Debug Plotting before rotation ---
        pre_rotation_yy = tensor_data.get_component('yy')
        if pre_rotation_yy.data is not None and pre_rotation_yy.daxis is not None:
            plt.figure()
            plt.plot(pre_rotation_yy.daxis, pre_rotation_yy.data[:, -1])
            plt.title("Strain YY Component Before Rotation (Second to Last Timestep)")
            plt.xlabel("Position along well (m)")
            plt.ylabel("Strain")
            plt.grid(True)
            plt.show()
        # ------------------------------------

        # Rotate the tensor field by 30 degrees counter-clockwise
        tensor_data.rotate_tensor(30)
        print("Rotated tensor by 30 degrees.")

        # Extract the yy-component (now aligned with the monitor well)
        strain_yy_data: Data2D = tensor_data.get_component('yy')

        # Replace the simulation start time with the real start time
        strain_yy_data.set_start_time(real_start_time)
        print(f"Updated start time to: {strain_yy_data.start_time}")

        # --- Plot Strain Data ---
        pg_dataframe = deepcopy(pg_data_orig)
        pg_dataframe.select_time(strain_yy_data.start_time, strain_yy_data.get_end_time())

        fig = plt.figure(figsize=(10, 8))
        ax1 = plt.subplot2grid((5, 4), (0, 0), rowspan=4, colspan=4)  # Strain plot
        ax2 = plt.subplot2grid((5, 4), (4, 0), rowspan=1, colspan=4, sharex=ax1)  # Pressure gauge plot

        im1 = strain_yy_data.plot(ax=ax1, cmap='bwr', use_timestamp=True, colorbar=False, vmin=-1.1e-5, vmax=1.1e-5)
        ax1.set_title(f'Strain Along Monitor Well\nSampler: {sampler_name}')
        ax1.set_ylabel('Position along well (m)')
        ax1.tick_params(labelbottom=False)

        # Plot pressure gauge data
        pg_dataframe.plot(ax=ax2, use_timestamp=True)
        ax2.set_ylabel("Pressure (psi)")
        ax2.set_xlabel('Time')

        # add colorbar for strain plot
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
        cbar.set_label("Strain")

        # To align the plots, add a dummy axes for the pressure plot colorbar space
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="2%", pad=0.05)
        cax2.axis('off')

        fig.tight_layout()
        strain_fig_path = os.path.join(fig_dir, f"{sampler_name}_strain_with_pressure.png")
        fig.savefig(strain_fig_path)
        plt.close(fig)
        print(f"Saved strain plot to: {strain_fig_path}")

        # --- Calculate and Plot Strain Rate ---
        if strain_yy_data.data is not None and strain_yy_data.taxis is not None and len(strain_yy_data.taxis) > 1:
            # Ensure the time axis length matches the data's time dimension before differentiation
            time_axis_len = strain_yy_data.data.shape[1]
            taxis_matched = strain_yy_data.taxis[:time_axis_len]
            print(taxis_matched)
            # Calculate strain rate using numpy.gradient for a more robust calculation
            # strain_yy_data.apply_lowpass_filter(0.00005)
            strain_rate_values = np.gradient(strain_yy_data.data, taxis_matched, axis=1)

            # The time axis for the gradient result is the same as the input
            new_taxis = taxis_matched

            # Create a new Data2D object for the strain rate
            strain_rate_data = Data2D(
                data=strain_rate_values,
                taxis=new_taxis,
                daxis=strain_yy_data.daxis,
                start_time=strain_yy_data.start_time,
                name=f"{sampler_name}_strain_rate"
            )

            # Filter the strain rate data
            # strain_rate_data.apply_lowpass_filter(0.001)
            # Plot strain data
            from fiberis.utils.viz_utils import plot_dss_and_gauge_co_plot
            chan_data = strain_yy_data.get_value_by_depth(strain_yy_data.daxis[-1] / 2)
            chan_dataframe = Data1DGauge()
            chan_dataframe.data = chan_data
            chan_dataframe.taxis = strain_yy_data.taxis[1:]
            chan_dataframe.start_time = strain_yy_data.start_time

            print(len(strain_yy_data.data))
            print(len(strain_yy_data.taxis))
            fig, ax = plt.subplots()
            chan_dataframe.plot(ax=ax, use_timestamp=True)
            plt.show()

            # Plot Strain Rate
            fig2 = plt.figure(figsize=(10, 8))
            ax1_rate = plt.subplot2grid((5, 4), (0, 0), rowspan=4, colspan=4)  # Strain rate plot
            ax2_rate = plt.subplot2grid((5, 4), (4, 0), rowspan=1, colspan=4, sharex=ax1_rate)  # Pressure gauge plot

            im2 = strain_rate_data.plot(ax=ax1_rate, cmap='bwr', use_timestamp=True, colorbar=False, vmin=-1e-8, vmax=1e-8)
            ax1_rate.set_title(f'Strain Rate Along Monitor Well\nSampler: {sampler_name}')
            ax1_rate.set_ylabel('Position along well (m)')
            ax1_rate.tick_params(labelbottom=False)

            # Plot pressure gauge data
            pg_dataframe_rate = deepcopy(pg_data_orig)
            pg_dataframe_rate.select_time(strain_rate_data.start_time, strain_rate_data.get_end_time())
            pg_dataframe_rate.plot(ax=ax2_rate, use_timestamp=True)
            ax2_rate.set_ylabel("Pressure (psi)")
            ax2_rate.set_xlabel('Time')

            # add colorbar for strain rate plot
            divider_rate = make_axes_locatable(ax1_rate)
            cax_rate = divider_rate.append_axes("right", size="2%", pad=0.05)
            cbar_rate = fig2.colorbar(im2, cax=cax_rate, orientation='vertical')
            cbar_rate.set_label("Strain Rate (1/s)")

            # To align the plots, add a dummy axes for the pressure plot colorbar space
            divider2_rate = make_axes_locatable(ax2_rate)
            cax2_rate = divider2_rate.append_axes("right", size="2%", pad=0.05)
            cax2_rate.axis('off')

            fig2.tight_layout()
            strain_rate_fig_path = os.path.join(fig_dir, f"{sampler_name}_strain_rate_with_pressure.png")
            fig2.savefig(strain_rate_fig_path)
            plt.close(fig2)
            print(f"Saved strain rate plot to: {strain_rate_fig_path}")
        else:
            print("Skipping strain rate calculation due to insufficient data.")




if __name__ == "__main__":
    # Define the output directory where MOOSE results are stored
    moose_output_directory = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/output/1203_rotated_monitor_well"
    figures_directory = "figs/12032025"

    # Run the processing function
    process_tensor_data(moose_output_directory, figures_directory)

    print("\n--- All processing complete. ---")

