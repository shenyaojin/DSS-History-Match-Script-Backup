# Pack the gauge MD data to fiberis format.
# Shenyao Jin, Mar 18 2025

from fiberis.io import reader_mariner_3d

# Define the path to geometry data
s_well_geometry = 'data/legacy/s_well/geometry/gauge_md_swell.npz'
h_well_geometry = 'data/legacy/h_well/geometry/gauge_md_hwell.npz'
prod_geometry = 'data/legacy/prod/geometry/gauge_md_prod.npz'

# Destination path
s_destination = 'data/fiberis_format/s_well/geometry/gauge_md_swell.npz'
h_destination = 'data/fiberis_format/h_well/geometry/gauge_md_hwell.npz'
prod_destination = 'data/fiberis_format/prod/geometry/gauge_md_prod.npz'

# Rewrite the geometry data to fiberis format
def tmp_rewrite(path, destination):
    dataframe = reader_mariner_3d.Mariner3D()
    dataframe.read(path)
    dataframe.write(destination)

# rewrite the geometry data
tmp_rewrite(s_well_geometry, s_destination)
tmp_rewrite(h_well_geometry, h_destination)
tmp_rewrite(prod_geometry, prod_destination)