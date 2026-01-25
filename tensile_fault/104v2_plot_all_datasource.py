# scripts/tensile_fault/104v2_plot_all_datasource.py

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge

# --- Configuration ---
INPUT_FILES = [
    "data_fervo/fiberis_format/pressure_data/Bearskin_1-IA_Pressure.npz",
    "data_fervo/fiberis_format/pressure_data/Bearskin_3-PA_Pressure.npz",
    "data_fervo/fiberis_format/pressure_data/Bearskin_4-PB_Pressure.npz",
]
OUTPUT_FILE = "data_fervo/fiberis_format/post_processing/gauge_moose_source_avg_high_resolution.npz"
QC_PLOT_FILE = "figs/01242026/gauge_moose_source_qc.png"

# Time range for cropping
T1 = datetime.datetime(2025, 2, 24, 15, 0, 0)
T2 = datetime.datetime(2025, 2, 28, 0, 0, 0)

# Number of points for interpolation
N_POINTS = 12000

# --- Main Script ---

def main():
    """
    Main function to load, process, and average pressure data.
    """
    # Ensure output directory for the plot exists
    os.makedirs(os.path.dirname(QC_PLOT_FILE), exist_ok=True)

    cropped_and_interpolated_data = []

    # Define the new common time axis
    duration_seconds = (T2 - T1).total_seconds()
    common_taxis = np.linspace(0, duration_seconds, N_POINTS)

    for file_path in INPUT_FILES:
        if not os.path.exists(file_path):
            print(f"Warning: Input file not found, skipping: {file_path}")
            continue

        print(f"Processing file: {os.path.basename(file_path)}")
        
        gauge = Data1DGauge()
        gauge.load_npz(file_path)

        # 1. Crop the data to the specified time range
        gauge.crop(T1, T2)
        
        # 2. Interpolate to the common time axis
        gauge.interpolate(common_taxis, new_start_time=T1)
        
        cropped_and_interpolated_data.append(gauge.data)

    if not cropped_and_interpolated_data:
        print("Error: No data was processed. Exiting.")
        return

    # 3. Average the data
    averaged_data = np.mean(np.array(cropped_and_interpolated_data), axis=0)

    # 4. Create a new Data1DGauge object for the averaged data
    final_gauge = Data1DGauge(
        data=averaged_data,
        taxis=common_taxis,
        start_time=T1,
        name="Averaged_Gauge_Pressure_for_MOOSE"
    )

    # 5. Save to .npz file
    final_gauge.savez(OUTPUT_FILE)
    print(f"\nSuccessfully saved averaged data to: {OUTPUT_FILE}")
    final_gauge.print_info()

    # 6. Quality Control Plot
    print(f"Generating QC plot and saving to: {QC_PLOT_FILE}")
    fig, ax = plt.subplots(figsize=(12, 6))
    final_gauge.plot(ax=ax, use_timestamp=True, title="QC Plot: Averaged Pressure Data")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pressure (PSI)")
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(QC_PLOT_FILE)
    plt.close(fig)
    print("QC plot saved.")


if __name__ == "__main__":
    main()