import numpy as np
import matplotlib.pyplot as plt
from fiberis.analyzer.Geometry3D.coreG3D import DataG3D

# 1. Load data using DataG3D
well_geometry = DataG3D()
well_geometry.load_npz('data_fervo/fiberis_format/Gold_4_PB_Well_Geometry.npz')

projection_data = DataG3D()
projection_data.load_npz('data_fervo/fiberis_format/projection_data_gold4pb.npz')

stimulation_loc = DataG3D()
stimulation_loc.load_npz('data_fervo/fiberis_format/stimulation_loc_bearskin.npz')

# 2. Filter out the horizontal part of the well
horizontal_mask = well_geometry.xaxis > 20
x_well_h = well_geometry.xaxis[horizontal_mask]
y_well_h = well_geometry.yaxis[horizontal_mask]

# 3. Determine the direction of the horizontal well using linear regression
slope, intercept = np.polyfit(x_well_h, y_well_h, 1)
v_well = np.array([1, slope]) # Direction vector of the well

# 4. Calculate the intersection angles
angles = []
for i in range(len(stimulation_loc.xaxis)):
    # Create a vector for the stimulation-projection line
    v_stim_proj = np.array([
        projection_data.xaxis[i] - stimulation_loc.xaxis[i],
        projection_data.yaxis[i] - stimulation_loc.yaxis[i]
    ])

    # Calculate the angle between the well vector and the stim-proj vector
    dot_product = np.dot(v_well, v_stim_proj)
    norm_prod = np.linalg.norm(v_well) * np.linalg.norm(v_stim_proj)
    angle = np.arccos(dot_product / norm_prod)
    angle_deg = np.degrees(angle)

    # Calculate the distance between stimulation and projection points
    distance = np.linalg.norm(v_stim_proj)
    print(f"Pair {i+1}: Distance = {distance:.2f} ft")

    # We care about the acute angle
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    angles.append(angle_deg.item())

# Calculate the average angle
average_angle = np.mean(angles)
print(f"Calculated Angles: {angles}")
print(f"Average Intersection Angle: {average_angle:.2f} degrees")


# 5. Make the visualization better
fig, ax = plt.subplots(figsize=(12, 10))

# Plot horizontal well geometry
ax.plot(x_well_h, y_well_h, 'k-', label='Horizontal Well Geometry', linewidth=3)

# Plot stimulation and projection points
ax.scatter(stimulation_loc.xaxis, stimulation_loc.yaxis, s=100, c='red', marker='o', label='Stimulation Locations', zorder=5)
ax.scatter(projection_data.xaxis, projection_data.yaxis, s=100, c='blue', marker='^', label='Projection Data', zorder=5)

# Plot lines connecting the points
for i in range(len(stimulation_loc.xaxis)):
    ax.plot([stimulation_loc.xaxis[i], projection_data.xaxis[i]],
            [stimulation_loc.yaxis[i], projection_data.yaxis[i]],
            'grey', linestyle='--', linewidth=1)

# Add labels, title, and legend
ax.set_xlabel('X Coordinate (ft)')
ax.set_ylabel('Y Coordinate (ft)')
ax.set_title(f'Well Intersection Analysis\nAverage Angle: {average_angle:.2f}°')
ax.legend()
ax.grid(True)
ax.set_aspect('equal', adjustable='box') # Ensure correct geometric representation

# Show the plot
plt.show()

# Fracture length
fault_length_ft = np.array([3125, 2768, 2693, 3324, 3366, 3542, 3631])