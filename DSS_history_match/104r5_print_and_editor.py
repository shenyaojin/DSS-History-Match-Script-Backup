#%% I will try to fix the bug in const DT: what is causing this?
# This version will not include visualization, will start from rendering input file.
# Modify from 103_timestep_bug_investigation.py
# Shenyao Jin, shenyaojin@mines.edu
# Last modified on 10/27/2025 by Gemini
# This script now demonstrates and tests the print functionalities of ModelBuilder and MooseModelEditor.

import numpy as np
import os
import matplotlib.pyplot as plt
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.analyzer.Data1D.core1D import Data1D
from fiberis.analyzer.Geometry3D.DataG3D_md import G3DMeasuredDepth

from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.editor import MooseModelEditor
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties,
    SimpleFluidPropertiesConfig, PointValueSamplerConfig, LineValueSamplerConfig,
    AdaptiveTimeStepperConfig, TimeSequenceStepper, PostprocessorConfig
)

#%% Data handling
datapath = "data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz"

DSSdata = DSS2D()
DSSdata.load_npz(datapath)

mds = DSSdata.daxis
ind = (mds>7500)&(mds<15000)
drift_val = np.median(DSSdata.data[ind,:],axis=0)
DSSdata.data -= drift_val.reshape((1,-1))

DSSdata.select_time(0, 400000)
DSSdata.select_depth(12000, 16360)

DSSdata_copy = DSSdata.copy()
DSSdata_copy.select_depth(14500, 15500)

hf1_md = 14888
hf2_md = 14972
hf3_md = 14992
time_point = 160000

chan1 = DSSdata.get_value_by_depth(hf1_md)
chan2 = DSSdata.get_value_by_depth(hf2_md)
chan3 = DSSdata.get_value_by_depth(hf3_md)

time_slice, _ = DSSdata_copy.get_value_by_time(time_point)
datapath = "data/fiberis_format/prod/gauges/pressure_g1.npz"

gauge_data = Data1DGauge()
gauge_data.load_npz(datapath)

DSSdata_crop = DSSdata.copy()
DSSdata_crop.select_depth(min(hf1_md, hf2_md)-100, max(hf1_md, hf2_md)+100)
DSSdata_crop.select_time(0, 400000)

gauge_md_path = "data/fiberis_format/s_well/geometry/gauge_md_swell.npz"
dataframe_gauge_md = G3DMeasuredDepth()
dataframe_gauge_md.load_npz(gauge_md_path)
print(dataframe_gauge_md)
ind = (dataframe_gauge_md.data > min(hf1_md, hf2_md)-100) & (dataframe_gauge_md.data < max(hf1_md, hf2_md)+100)
gauge_mds = dataframe_gauge_md.data[ind]
print(gauge_mds)

mesh_y_range = DSSdata_crop.daxis[-1] - DSSdata_crop.daxis[0]
hf_1_loc = (hf1_md - DSSdata_crop.daxis[0])
hf_2_loc = (hf2_md - DSSdata_crop.daxis[0])
hf_3_loc = (hf3_md - DSSdata_crop.daxis[0])
print(f"HF1 location in mesh: {hf_1_loc:.3f} ft")
print(f"HF2 location in mesh: {hf_2_loc:.3f} ft")
print(f"HF3 location in mesh: {hf_3_loc:.3f} ft")

gauge_locs = (gauge_mds - DSSdata_crop.daxis[0])
print("Gauge locations in mesh (ft):", gauge_locs)

inj_gauge_pressure_path = "data/fiberis_format/prod/gauges/gauge4_data_prod.npz"
from fiberis.analyzer.Data1D.core1D import Data1D
injection_pressure_dataframe = Data1D()
injection_pressure_dataframe.load_npz(inj_gauge_pressure_path)
print(injection_pressure_dataframe)

injection_pressure_dataframe.remove_abnormal_data(threshold=300, method='mean')

#%% Configure MOOSE model and test print functionalities
print("\n--- Configuring ModelBuilder ---")
# 1. basic parameters
output_dir = "output/1027_print_test"
os.makedirs(output_dir, exist_ok=True)
builder = ModelBuilder(project_name="DSS_Print_Test")

# 2. mesh, HF, SRV and matrix
conversion_factor = 0.3048  # ft to m
shift = - 0.831 # To make coords shift to int
fracture_y1_coords = (hf_1_loc + shift) * conversion_factor
fracture_y2_coords = (hf_2_loc + shift) * conversion_factor
fracture_y3_coords = (hf_3_loc + shift) * conversion_factor

domain_bounds = (-100 * conversion_factor, 400 * conversion_factor)
domain_length = 500 * conversion_factor

fracture_y_coords = [fracture_y1_coords, fracture_y2_coords, fracture_y3_coords]
builder.build_stitched_mesh_for_fractures(
    fracture_y_coords=fracture_y_coords,
    domain_bounds=domain_bounds,
    domain_length=domain_length,
    nx = 200,
    ny_per_layer_half= 25,
    bias_y=1.2
)

matrix_mats = ZoneMaterialProperties(porosity=0.01, permeability="'1E-17 0 0  0 1E-17 0  0 0 1E-17'")
srv_mats = ZoneMaterialProperties(porosity=0.1, permeability="'1E-15 0 0  0 1E-15 0  0 0 1E-15'")
fracture_mats = ZoneMaterialProperties(porosity=0.1, permeability="'1E-12 0 0  0 1E-12 0  0 0 1E-12'")

builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))

center_x_val = domain_length / 2.0
geometries = [
    SRVConfig(name="srv_1", length=300 * conversion_factor, height=50 * conversion_factor, center_x=center_x_val, center_y=fracture_y1_coords, materials=srv_mats),
    HydraulicFractureConfig(name="hf_1", length=250 * conversion_factor, height=0.2 * conversion_factor, center_x=center_x_val, center_y=fracture_y1_coords, materials=fracture_mats),
    SRVConfig(name="srv_2", length=300 * conversion_factor, height=50 * conversion_factor, center_x=center_x_val, center_y=fracture_y2_coords, materials=srv_mats),
    HydraulicFractureConfig(name="hf_2", length=250 * conversion_factor, height=0.2 * conversion_factor, center_x=center_x_val, center_y=fracture_y2_coords, materials=fracture_mats),
    SRVConfig(name="srv_3", length=300 * conversion_factor, height=50 * conversion_factor,  center_x=center_x_val, center_y=fracture_y3_coords, materials=srv_mats),
    HydraulicFractureConfig(name="hf_3", length=250 * conversion_factor, height=0.2 * conversion_factor, center_x=center_x_val, center_y=fracture_y3_coords, materials=fracture_mats)
]

sorted_geometries = sorted(geometries, key=lambda x: x.height, reverse=True)
next_block_id = 1
for geom_config in sorted_geometries:
    if isinstance(geom_config, SRVConfig):
        builder.add_srv_config(geom_config)
        builder.add_srv_zone_2d(geom_config, target_block_id=next_block_id)
    elif isinstance(geom_config, HydraulicFractureConfig):
        builder.add_fracture_config(geom_config)
        builder.add_hydraulic_fracture_2d(geom_config, target_block_id=next_block_id)
    next_block_id += 1

builder.add_nodeset_by_coord(nodeset_op_name="injection_1", new_boundary_name="injection_1", coordinates=(center_x_val, fracture_y1_coords, 0))
builder.add_nodeset_by_coord(nodeset_op_name="injection_2", new_boundary_name="injection_2", coordinates=(center_x_val, fracture_y2_coords, 0))
builder.add_nodeset_by_coord(nodeset_op_name="injection_3", new_boundary_name="injection_3", coordinates=(center_x_val, fracture_y3_coords, 0))

# --- Test Print Functionalities ---
print("\n--- Testing ModelBuilder __str__ method ---")
print(builder)

print("\n--- Testing MooseModelEditor print_model_structure method ---")
editor = MooseModelEditor(builder)
editor.print_model_structure()

print("\n--- Print geometry info ---")
print(builder.extract_geometry())

# {'mesh': {'domain_bounds': (-30.48, 121.92), 'domain_length': 152.4}, 'srv_zones': [{'name': 'srv_1', 'length': 91.44, 'height': 15.24, 'center_x': 76.2, 'center_y': np.float64(30.17521666875), 'materials': <fiberis.moose.config.ZoneMaterialProperties object at 0x7f472ded6e70>, 'mesh_length_param': None, 'mesh_height_param': None}, {'name': 'srv_2', 'length': 91.44, 'height': 15.24, 'center_x': 76.2, 'center_y': np.float64(55.778416668750005), 'materials': <fiberis.moose.config.ZoneMaterialProperties object at 0x7f472ded6e70>, 'mesh_length_param': None, 'mesh_height_param': None}, {'name': 'srv_3', 'length': 91.44, 'height': 15.24, 'center_x': 76.2, 'center_y': np.float64(61.87441666875001), 'materials': <fiberis.moose.config.ZoneMaterialProperties object at 0x7f472ded6e70>, 'mesh_length_param': None, 'mesh_height_param': None}], 'hydraulic_fractures': [{'name': 'hf_1', 'length': 76.2, 'height': 0.06096000000000001, 'center_x': 76.2, 'center_y': np.float64(30.17521666875), 'materials': <fiberis.moose.config.ZoneMaterialProperties object at 0x7f472daa9250>, 'orientation_angle': 0.0, 'mesh_length_param': None, 'mesh_height_param': None}, {'name': 'hf_2', 'length': 76.2, 'height': 0.06096000000000001, 'center_x': 76.2, 'center_y': np.float64(55.778416668750005), 'materials': <fiberis.moose.config.ZoneMaterialProperties object at 0x7f472daa9250>, 'orientation_angle': 0.0, 'mesh_length_param': None, 'mesh_height_param': None}, {'name': 'hf_3', 'length': 76.2, 'height': 0.06096000000000001, 'center_x': 76.2, 'center_y': np.float64(61.87441666875001), 'materials': <fiberis.moose.config.ZoneMaterialProperties object at 0x7f472daa9250>, 'orientation_angle': 0.0, 'mesh_length_param': None, 'mesh_height_param': None}], 'nodesets': [{'name': 'injection_1', 'coordinates': (76.2, np.float64(30.17521666875), 0)}, {'name': 'injection_2', 'coordinates': (76.2, np.float64(55.778416668750005), 0)}, {'name': 'injection_3', 'coordinates': (76.2, np.float64(61.87441666875001), 0)}], 'postprocessors': []}
