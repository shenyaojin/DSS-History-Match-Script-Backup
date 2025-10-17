# Pack the well trajectory data
# Shenyao Jin, Mar 18 2025

from fiberis.io import reader_mariner_3d

# Define the path to geometry data
s_well_geometry = 'data/legacy/s_well/geometry/swell_geometry.npz'
h_well_geometry = 'data/legacy/h_well/geometry/hwell_geometry.npz'
prod_geometry = 'data/legacy/prod/geometry/prod_geometry.npz'

# Destination path
s_destination = 'data/fiberis_format/s_well/geometry/swell_geometry.npz'
h_destination = 'data/fiberis_format/h_well/geometry/hwell_geometry.npz'
prod_destination = 'data/fiberis_format/prod/geometry/prod_geometry.npz'

# Rewrite the geometry data to fiberis format
def tmp_rewrite(path, destination):
    dataframe = reader_mariner_3d.Mariner3D()
    dataframe.read(path)
    dataframe.write(destination)

# rewrite the geometry data
tmp_rewrite(s_well_geometry, s_destination)
tmp_rewrite(h_well_geometry, h_destination)
tmp_rewrite(prod_geometry, prod_destination)