#%% Import libs
from fiberis.analyzer.Data1D import Data1D_Gauge
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


#%% Load the data directory
datapath = "data/fiberis_format/s_well/gauges/*.npz"
files = glob(datapath)
files = sorted(files)

#%% print files
gauge_data_all = []
for file in files:
    gauge_data = Data1D_Gauge.Data1DGauge()
    gauge_data.load_npz(file)
    gauge_data_all.append(gauge_data)

#%% Plot the data
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))

for i, gauge_data in enumerate(gauge_data_all):
    ax.scatter(gauge_data.taxis, gauge_data.data, s=20, label=f'Gauge {i+1}', alpha=0.6)

ax.set_xlabel("Time")
ax.set_ylabel("Value")

ax.legend(title="Gauges", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()

plt.savefig("figs/sponsor_meeting_report_2025/production/s_well_plot.png")
plt.show()