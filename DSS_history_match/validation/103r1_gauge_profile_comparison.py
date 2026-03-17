import numpy as np
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
import matplotlib.pyplot as plt
import os
import datetime

full_path = "scripts/DSS_history_match/validation/data/full.npz"
interf_path = "scripts/DSS_history_match/validation/data/interference.npz"

full_frame = Data1DGauge()
full_frame.load_npz(full_path)

interf_frame = Data1DGauge()
interf_frame.load_npz(interf_path)

# crop and compare
full_frame.select_time(interf_frame.start_time, interf_frame.get_end_time())

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1)

interf_frame.plot(ax=ax1, use_timestamp=False)
full_frame.plot(ax=ax2, use_timestamp=False)

# Add label
ax1.set_title("Simulated Strain (INTERF)")

ax2.set_title("Simulated Strain (FULL)")
ax2.set_xlabel("Time (s)")

plt.tight_layout()
plt.show()