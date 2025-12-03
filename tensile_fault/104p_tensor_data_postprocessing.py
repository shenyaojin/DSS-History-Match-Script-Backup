import os
from fiberis.io.reader_moose_tensor_vpp import MOOSETensorVPPReader
from fiberis.analyzer.TensorProcessor.coreT2D import Tensor2D

def process_tensor_data(output_dir: str):
    """
    Reads strain tensor data from MOOSE VectorPostprocessor outputs and prints a summary.
    """
    reader = MOOSETensorVPPReader()

    # List available samplers and select the first one by default
    available_samplers = reader.list_available_samplers(output_dir)

    if not available_samplers:
        print(f"No strain tensor samplers found in {output_dir}")
        return

    # Select the first sampler by default
    selected_sampler_name = available_samplers[0]
    print(f"Processing data from sampler: {selected_sampler_name}")

    # Read the data for the selected sampler
    reader.read(directory=output_dir, sampler_name=selected_sampler_name)

    # Convert to Tensor2D analyzer object
    tensor_data: Tensor2D = reader.to_analyzer()

    # Print the summary of the Tensor2D object for QC
    print(tensor_data)
    print(f"Data shape: {tensor_data.data.shape}")
    print(f"Time axis length: {len(tensor_data.taxis)}")
    print(f"Depth axis length: {len(tensor_data.daxis)}")

    return tensor_data

if __name__ == "__main__":
    # Define the output directory where MOOSE results are stored
    moose_output_directory = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/output/1203_rotated_monitor_well"

    # Run the processing function
    tensor_field = process_tensor_data(moose_output_directory)

    if tensor_field:
        print("\nSuccessfully processed tensor data.")
    else:
        print("\nFailed to process tensor data.")
