import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_misfit_2d_scatter(csv_file, output_file):
    """
    Reads misfit data from a CSV file and creates a 2D scatter plot.

    The plot shows 'srv_perm' vs 'fracture_perm', with the color and size
    of the points representing the total misfit (misfit1 + misfit2).
    """
    # Read the data from the CSV file
    df = pd.read_csv(csv_file)

    # Calculate the total misfit
    df['total_misfit'] = df['misfit1'] + df['misfit2']

    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['srv_perm'], df['fracture_perm'], c=df['total_misfit'],
                            s=df['total_misfit'] / 1e5, cmap='viridis', alpha=0.6)

    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Total Misfit')

    # Set plot labels and title
    plt.xlabel('SRV Permeability (m^2)')
    plt.ylabel('Fracture Permeability (m^2)')
    plt.title('2D Misfit Landscape')

    # Use log scale for the axes to better visualize the parameter range
    plt.xscale('log')
    plt.yscale('log')

    # Add grid for better readability
    plt.grid(True, which="both", ls="--")

    # Save the plot to the specified output file
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    # Define the input CSV file and the output image file
    csv_file_path = 'output/1009_loop_para/misfit_results.csv'
    output_plot_path = 'output/1009_loop_para/misfit_plot.png'

    # Generate and save the plot
    plot_misfit_2d_scatter(csv_file_path, output_plot_path)
