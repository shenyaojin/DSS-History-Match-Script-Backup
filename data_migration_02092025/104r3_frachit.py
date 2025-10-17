# Pack the fracture hit data into fiberis format
# Shenyao Jin, Mar 18 2025

from fiberis.io import reader_mariner_3d
from glob import glob

# Define the input and output file paths
input_file_list_swell = glob('data/legacy/s_well/geometry/frac_hit/*.npz')
input_file_list_hwell = glob('data/legacy/h_well/geometry/frac_hit/*.npz')
input_file_list_prod = glob('data/legacy/prod/geometry/perf/*.npz')

def rewrite(path):
    dataframe = reader_mariner_3d.Mariner3D()
    dataframe.read(path)
    dataframe.write(path.replace('data/legacy', 'data/fiberis_format'))

# Process the input files
for file in input_file_list_swell:
    rewrite(file)

for file in input_file_list_hwell:
    rewrite(file)

for file in input_file_list_prod:
    rewrite(file)