from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
import numpy as np
import matplotlib.pyplot as plt

data_path = "data/fiberis_format/prod/gauges/pressure_g1.npz"

gauge = Data1DGauge()
gauge.load_npz(data_path)

print(gauge.taxis[-1] / 3600, "hours total")

start_hour = 55
length_hours = 84

gauge.select_time(start_hour * 3600, (start_hour + length_hours) * 3600)

fig, ax = plt.subplots()
gauge.plot(ax = ax, use_timestamp=True)
plt.show()