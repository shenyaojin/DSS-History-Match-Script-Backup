from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge

dir = "data/fiberis_format/s_well/gauges"
files = sorted(glob(f"{dir}/*.npz"))

for file in files:
    gauge = Data1DGauge()
    gauge.load_npz(file)
    print("Elem in Pa", gauge.data[-200] * 6894.76)  # Convert psi to Pa
