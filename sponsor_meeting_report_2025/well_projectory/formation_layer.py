import numpy as np
import matplotlib.pyplot as plt

# Example x-axis (distance or easting)
x = np.linspace(0, 5000, 100)

# Synthetic formation boundaries (tops and bottoms)
top_UBS   = -300 - 5  * np.sin(x/500)  # Just an example shape
base_UBS  = top_UBS   - 20
top_MB    = base_UBS
base_MB   = top_MB    - 30
top_LBS   = base_MB
base_LBS  = top_LBS   - 20
top_TF1   = base_LBS
base_TF1  = top_TF1   - 15
top_TF2   = base_TF1
base_TF2  = top_TF2   - 20

# Start plotting
fig, ax = plt.subplots(figsize=(8,5))

# Fill each formation interval
ax.fill_between(x, top_UBS,  base_UBS,  color='gray',  alpha=0.4, label='UBS')
ax.fill_between(x, top_MB,   base_MB,   color='blue',  alpha=0.4, label='MB')
ax.fill_between(x, top_LBS,  base_LBS,  color='green', alpha=0.4, label='LBS')
ax.fill_between(x, top_TF1,  base_TF1,  color='yellow',alpha=0.4, label='TF1')
ax.fill_between(x, top_TF2,  base_TF2,  color='brown', alpha=0.4, label='TF2')

# Example well trajectory (just plotting a single line)
well_depth = -300 - 50 * np.exp(-(x-2500)**2/(2*1000**2))  # Synthetic shape
ax.plot(x, well_depth, 'k-', label='Well Trajectory')

# Cosmetics
ax.invert_yaxis()  # Invert if you want depth increasing down
ax.set_xlabel('Easting (ft)')
ax.set_ylabel('TVD (ft)')
ax.legend(loc='best')
ax.set_title('Cross Section with Formation Shading')

plt.show()