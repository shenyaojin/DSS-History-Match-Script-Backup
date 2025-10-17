# Simple script to calculate fluid migration through pressure conduit, considering different cluster spacing
# Shenyao Jin, shenyaojin@mines.edu
# Use the info of cement shrinkage.

import numpy as np
import matplotlib.pyplot as plt

#Define parameters
D = 140 # hydraulic diffusivity (ft^2/s)
phi = 0.1 # porosity
L = np.linspace(10, 20, 20) # cluster spacing, ft.
ct = 1e-5 # compressibility, in psi^-1

# Calculate A (the area of the pressure conduit)
d1 = 0.46 # in feet, 5.5 inch for the casing
d2 = 0.125 # in feet, conduit, 2mm, for each side)

pi = 3.141

r1 = d1 / 2
r2 = (d1 + d2) / 2

A = pi * r2 ** 2 - pi * r1 ** 2

# Apply Darcy's law
# Assume the delta P is 500 psi
dP = 2000

q_array = D * phi * ct * A / L * dP # In ft^3/s
# Convert to bbl/day -> bbl/min by dividing by 24 and 60
q_array = q_array * 15388.5 / 24 / 60

plt.figure()
plt.plot(L, q_array)
plt.xlabel('Cluster spacing (ft)')
plt.ylabel('Fluid migration rate (bpm)')
plt.show()
