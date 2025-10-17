from fiberis.io import reader_mariner_pp1d
from tqdm import tqdm
import re

path = "data/legacy/prod/pumping_curve/"

import os
files = os.listdir(path)

for file in tqdm(files):
    # pumping_curve_stage1.npz
    # Strip the stage number
    match = re.search(r'stage\d+', file)
    if match:
        stage = match.group()  # Extract the matched stage
    os.system("mkdir -p data/fiberis_format/prod/" + stage)
    data = reader_mariner_pp1d.MarinerPP1D()
    data.read(path + file)
    destination_path = "data/fiberis_format/prod/" + stage + "/"
    data.write(destination_path)