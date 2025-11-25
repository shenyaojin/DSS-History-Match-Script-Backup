# scripts/tensile_fault/data_transfer/103_geometry_extractor.py

import os
import matplotlib.pyplot as plt
from fiberis.io import Gold4PB3D
from fiberis.analyzer.Geometry3D.coreG3D import DataG3D
import numpy as np

# --- Configuration ---
INPUT_FILE = "data_fervo/legacy/Gold_4_PB_Well_Geometry.csv"
OUTPUT_DIR = "data_fervo/fiberis_format"

# --- Main Script ---

def main():
    """
    Main function to process the geometry file, save it as NPZ, and run QC.
    """
    # --- 1. Data Extraction and Saving ---
    
    # Ensure the output directory exists
    output_path = os.path.abspath(OUTPUT_DIR)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

    # Instantiate the reader
    reader = Gold4PB3D()
    
    input_file_path = os.path.abspath(INPUT_FILE)
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        return

    print(f"Processing file: {os.path.basename(input_file_path)}")

    try:
        # Read the CSV data
        reader.read(input_file_path)

        # Convert to analyzer object
        analyzer = reader.to_analyzer()
        analyzer.savez("data_fervo/fiberis_format/Gold_4_PB_Well_Geometry.npz")

        # Construct the output filename
        base_name = os.path.splitext(os.path.basename(input_file_path))[0]
        output_npz_path = os.path.join(output_path, f"{base_name}.npz")

        print(f"Successfully converted and saved data to: {output_npz_path}")

    except Exception as e:
        print(f"An error occurred during file processing: {e}")
        return

    # --- 2. Quality Control (QC) ---
    print(f"\n--- Performing QC on the new file: {os.path.basename(output_npz_path)} ---")
    
    if not os.path.exists(output_npz_path):
        print("QC failed: Output file was not created.")
        return

    # Instantiate a new analyzer for QC
    qc_analyzer = DataG3D()

    try:
        # Load the data from the .npz file
        # We need to load the npz file manually as DataG3D might not have a load_npz method
        with open(output_npz_path, 'rb') as f:
            data_structure = np.load(f, allow_pickle=True)
            qc_analyzer.data = data_structure['data'] # MD
            qc_analyzer.xaxis = data_structure['ew']   # x_gold
            qc_analyzer.yaxis = data_structure['ns']   # y_gold
            qc_analyzer.zaxis = data_structure['tvd']  # TVDrkb
            qc_analyzer.name = os.path.basename(output_npz_path)
        
        print("File loaded successfully for QC.")
        qc_analyzer.print_info()

        # Plot the 3D geometry
        print("Generating QC plot...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Assuming a simple plot method for DataG3D, we'll plot it manually
        ax.plot(qc_analyzer.xaxis, qc_analyzer.yaxis, qc_analyzer.zaxis, label=qc_analyzer.name)
        
        ax.set_xlabel("X (East-West)")
        ax.set_ylabel("Y (North-South)")
        ax.set_zlabel("True Vertical Depth (TVD)")
        ax.set_title(f"QC Plot: {qc_analyzer.name}")
        ax.legend()
        ax.invert_zaxis() # Typically depth is shown increasing downwards
        
        plt.tight_layout()
        plt.show()
        print("QC plot displayed.")

    except Exception as e:
        print(f"An error occurred during QC: {e}")


if __name__ == "__main__":
    main()
