# scripts/tensile_fault/data_transfer/101_reader_test_gauge_26.py

import matplotlib.pyplot as plt
from fiberis.io import BearskinPP1D
import os

# Define the relative path to the data file
relative_file_path = "data_fervo/legacy/Bearskin 1-IA Pressure.csv"

# Get the absolute path to the data file
# os.path.abspath will resolve the path based on the current working directory
file_path = os.path.abspath(relative_file_path)

# 1. Instantiate the reader
reader = BearskinPP1D()

# 2. Read the data for stage 26
try:
    print(f"Reading data for stage 26 from: {file_path}")
    reader.read(file_path, stage_num=26)
    print("Data read successfully.")
except (FileNotFoundError, ValueError) as e:
    print(f"Error reading data: {e}")
    exit()

# 3. Convert to an analyzer object
analyzer = reader.to_analyzer()
print("Converted to analyzer object.")
analyzer.print_info()

# 4. Plot the data
print("Plotting data...")
fig, ax = plt.subplots(figsize=(12, 6))
analyzer.plot(ax=ax, use_timestamp=True, label='Pressure (PSI)')

# Customize the plot
ax.set_title(f"Pressure Data for Stage 26 - {analyzer.name}")
ax.set_xlabel("Time (MST)")
ax.set_ylabel("Pressure (PSI)")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()
print("Plot displayed.")
