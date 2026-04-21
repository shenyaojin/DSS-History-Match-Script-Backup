import numpy as np
from matplotlib import pyplot as plt

from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D

reader = MOOSEVectorPostProcessorReader()
reader.read("scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100layer/fwd/output_gt", post_processor_id=0, variable_index=1)

disp_y_dataframe = reader.to_analyzer()

disp_y_dataframe.daxis -= 25
disp_y_dataframe.data[:, 0] = np.zeros_like(disp_y_dataframe.data[:, 0]) # padding

# Strain
reader.read("scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100layer/fwd/output_gt", post_processor_id=1, variable_index=1)
disp_y_dataframe = reader.to_analyzer()
strain_yy_dataframe = reader.to_analyzer()
strain_yy_dataframe.daxis -= 25
strain_yy_dataframe.data[:, 0] = np.zeros_like(strain_yy_dataframe.data[:, 0])

# Plot the strain_yy
fig, ax = plt.subplots()
img = strain_yy_dataframe.plot(cmap="bwr", vmax=np.max(strain_yy_dataframe.data), vmin=np.min(strain_yy_dataframe.data),
                         ax=ax, method='pcolormesh')
ax.set_xlabel("Time/s")
ax.set_ylabel("Measured Depth (m)")
ax.set_title("Ground Truth STRAIN_YY")
# Add color bar 
cbar = fig.colorbar(img, ax=ax)
cbar.set_label("Strain (m/m)")

plt.show()
