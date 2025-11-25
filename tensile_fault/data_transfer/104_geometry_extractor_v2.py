# This script is to extract other geometry data from csv files.
# Shenyao Jin, 11/25/2025
import numpy as np
import os
import sys

# Add the project root to the python path
# This is necessary to import the fiberis module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


from fiberis.io import Gold4PBProjection, BearskinInjection

def main():
    """
    Main function to extract geometry data from a CSV file and save it in NPZ format.
    """
    # Define file paths
    input_filepath = "data_fervo/legacy/injection_projected_along_strike.csv"
    output_dir = "data_fervo/fiberis_format"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # --- Process Projection Data ---
    print("Processing Gold4PB Projection Data...")
    try:
        # 1. Initialize the reader
        projection_reader = Gold4PBProjection()

        # 2. Read the data from the CSV file
        projection_reader.read(input_filepath)

        # 3. Convert to an analyzer object
        projection_analyzer = projection_reader.to_analyzer()

        # 4. Define the output path and save the data
        projection_output_path = os.path.join(output_dir, "projection_data_gold4pb.npz")
        projection_analyzer.savez(projection_output_path)
        
        print(f"Successfully saved projection data to: {projection_output_path}")
        projection_analyzer.print_info()

    except Exception as e:
        print(f"An error occurred while processing projection data: {e}")


    # --- Process Injection/Stimulation Data ---
    print("\nProcessing Bearskin Injection Data...")
    try:
        # 1. Initialize the reader
        injection_reader = BearskinInjection()

        # 2. Read the data from the same CSV file
        injection_reader.read(input_filepath)

        # 3. Convert to an analyzer object
        injection_analyzer = injection_reader.to_analyzer()

        # 4. Define the output path and save the data
        injection_output_path = os.path.join(output_dir, "stimulation_loc_bearskin.npz")
        injection_analyzer.savez(injection_output_path)

        print(f"Successfully saved injection data to: {injection_output_path}")
        injection_analyzer.print_info()

    except Exception as e:
        print(f"An error occurred while processing injection data: {e}")


if __name__ == "__main__":
    main()
