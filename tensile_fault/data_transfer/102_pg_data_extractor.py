# scripts/tensile_fault/data_transfer/102_pg_data_extractor.py

import os
import matplotlib.pyplot as plt
from fiberis.io.reader_bearskin_pp1d import BearskinPP1D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge

# --- Configuration ---
# List of relative paths to the input CSV files
INPUT_FILES = [
    "data_fervo/legacy/Bearskin 1-IA Pressure.csv",
    "data_fervo/legacy/Bearskin 3-PA Pressure.csv",
    "data_fervo/legacy/Bearskin 4-PB Pressure.csv",
]

# Relative path to the output directory
OUTPUT_DIR = "data_fervo/fiberis_format"

# --- Main Script ---

def main():
    """
    Main function to process the files and save them as .npz.
    """
    # Ensure the output directory exists
    output_path = os.path.abspath(OUTPUT_DIR)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

    # Instantiate the reader
    reader = BearskinPP1D()
    
    # Store the path of the last file created for QC
    last_file_saved = None

    # Loop through each input file
    for relative_path in INPUT_FILES:
        file_path = os.path.abspath(relative_path)
        if not os.path.exists(file_path):
            print(f"Warning: Input file not found, skipping: {file_path}")
            continue
        
        print(f"\nProcessing file: {os.path.basename(file_path)}")

        try:
            # Read the entire data from the file (stage_num is ignored)
            reader.read(file_path)

            # Convert to analyzer object to get metadata
            analyzer = reader.to_analyzer()

            # Construct a descriptive filename from the input file
            base_name = os.path.basename(file_path)
            file_name = os.path.splitext(base_name)[0].replace(' ', '_') + ".npz"
            full_output_path = os.path.join(output_path, file_name)

            # Save the data using the analyzer's savez method
            analyzer.savez(full_output_path)
            print(f"  - Successfully extracted and saved data to {file_name}")
            last_file_saved = full_output_path

        except Exception as e:
            print(f"  - Error processing file {os.path.basename(file_path)}: {e}")

    # --- Quality Control (QC) ---
    if last_file_saved:
        print(f"\n--- Performing QC on last saved file: {os.path.basename(last_file_saved)} ---")
        
        # 1. Instantiate an analyzer object
        qc_analyzer = Data1DGauge()
        
        # 2. Load the data from the .npz file
        qc_analyzer.load_npz(last_file_saved)
        print("File loaded successfully for QC.")
        qc_analyzer.print_info()
        
        # 3. Plot the data
        print("Generating QC plot...")
        fig, ax = plt.subplots(figsize=(12, 6))
        qc_analyzer.plot(ax=ax, use_timestamp=True)
        
        ax.set_title(f"QC Plot: {qc_analyzer.name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Pressure (PSI)")
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        print("QC plot displayed.")
    else:
        print("\nNo files were saved, skipping QC plot.")

if __name__ == "__main__":
    main()