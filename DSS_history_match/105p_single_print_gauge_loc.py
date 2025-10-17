#%% Load gauge measured depth data from a .npz file and print the data
from fiberis.analyzer.Geometry3D.DataG3D_md import G3DMeasuredDepth
gauge_md_path = "data/fiberis_format/s_well/geometry/gauge_md_swell.npz"

gauge_md_dataframe = G3DMeasuredDepth()
gauge_md_dataframe.load_npz(gauge_md_path)
print(gauge_md_dataframe.data)
