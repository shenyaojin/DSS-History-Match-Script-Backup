# Migrate the PG data from legacy data format to fiberis data format
from fiberis.io import reader_mariner_gauge1d
from tqdm import tqdm

# define the path to legacy data
path = 'data/legacy/h_well/gauge_data/'

import os
files = os.listdir(path)

# read the data from the legacy format
for file in tqdm(files):
    filepath = path + file
    reader_io = reader_mariner_gauge1d.MarinerGauge1D()
    reader_io.read(filepath)

    # write the data into fiberis format
    reader_io.write('data/fiberis_format/h_well/gauges/' + file)