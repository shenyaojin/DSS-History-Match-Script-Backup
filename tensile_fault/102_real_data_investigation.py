from fiberis.analyzer.Geometry3D.DataG3D_md import G3DMeasuredDepth

dataframe = G3DMeasuredDepth()
dataframe.load_npz("data_fervo/fiberis_format/stimulation_loc_bearskin.npz")

dataframe.plot()