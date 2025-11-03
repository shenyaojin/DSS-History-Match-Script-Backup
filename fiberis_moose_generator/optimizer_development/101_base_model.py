# This script will create a base model for DSS history matching, also serving as the initial input of the optimizer
# Scripted by: Shenyao, 11/01/2025
# This base model is a single fracture model with undetermined number of post processors.
# It follows the structure mentioned in my Notion notes.

import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

from fiberis.analyzer.Data1D.Data1D_MOOSEps import Data1D_MOOSEps
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge
from fiberis.moose.model_builder import ModelBuilder
from fiberis.moose.config import (
    MatrixConfig, SRVConfig, HydraulicFractureConfig, ZoneMaterialProperties, SimpleFluidPropertiesConfig,
    PointValueSamplerConfig, LineValueSamplerConfig, TimeSequenceStepper
)

from typing import Union, Dict, Any

def model_builder_single_frac(**kwargs) -> ModelBuilder:
    """
    To build a base model for DSS history matching. -- Shenyao

    :param kwargs: These are optional arguments that can be passed to customize the model building process.
    :return: A model builder object representing the model. It can be adjusted further by passing additional parameters.
    """
    conversion_factor = 0.3048  # ft to m
    biot_coeff = kwargs.get('biot_coefficient', 0.7)
    builder = ModelBuilder(project_name=kwargs.get('project_name', 'base_model_single_frac'))

    # Mesh and Geometry
    frac_coords = 0
    domain_bounds = (- kwargs.get('model_width', 400 * conversion_factor) / 2,
                     + kwargs.get('model_width', 400 * conversion_factor) / 2)

    domain_length = kwargs.get('model_length', 800 * conversion_factor)
    builder.build_stitched_mesh_for_fractures(
            fracture_y_coords=frac_coords,
            domain_bounds=domain_bounds,
            domain_length=domain_length,
            nx=kwargs.get('nx', 200),
            ny_per_layer_half=kwargs.get('ny_per_layer_half', 80),
            bias_y=kwargs.get('bias_y', 1.2)
        )

    matrix_perm = kwargs.get('matrix_perm', 1e-18)
    srv_perm = kwargs.get('srv_perm', 1e-15)
    srv_perm2 = kwargs.get('srv_perm2', 1e-14)
    fracture_perm = kwargs.get('fracture_perm', 1e-13)

    # The tensor format for permeability in fiberis:
    matrix_perm_str = f"{matrix_perm} 0 0 0 {matrix_perm} 0 0 0 {matrix_perm}"
    srv_perm_str = f"{srv_perm} 0 0 0 {srv_perm} 0 0 0 {srv_perm}"
    srv_perm_str2 = f"{srv_perm2} 0 0 0 {srv_perm2} 0 0 0 {srv_perm2}"
    fracture_perm_str = f"{fracture_perm} 0 0 0 {fracture_perm} 0 0 0 {fracture_perm}"

    matrix_mats = ZoneMaterialProperties(porosity=0.01, permeability=matrix_perm_str)
    srv_mats = ZoneMaterialProperties(porosity=0.1, permeability=srv_perm_str)
    srv_mats2 = ZoneMaterialProperties(porosity=0.1, permeability=srv_perm_str2)
    fracture_mats = ZoneMaterialProperties(porosity=0.16, permeability=fracture_perm_str)

    builder.set_matrix_config(MatrixConfig(name="matrix", materials=matrix_mats))

    center_x_val = domain_length / 2.0
    srv_length_ft2 = kwargs.get('srv_length_ft', 400)
    srv_height_ft2 = kwargs.get('srv_height_ft', 50)
    srv_length_ft1 = kwargs.get('srv_length_ft1', 250)
    srv_height_ft1 = kwargs.get('srv_height_ft1', 150)
    hf_length_ft = kwargs.get('hf_length_ft', 250)
    hf_height_ft = kwargs.get('hf_height_ft', 0.2)

    geometries = [
        SRVConfig(name="srv_tall", length=srv_length_ft1 * conversion_factor, height=srv_height_ft1 * conversion_factor,
                  center_x=center_x_val, center_y=frac_coords, materials=srv_mats2),
        SRVConfig(name="srv_wide", length=srv_length_ft2 * conversion_factor, height=srv_height_ft2 * conversion_factor,
                  center_x=center_x_val, center_y=frac_coords, materials=srv_mats),
        HydraulicFractureConfig(name="hf", length=hf_length_ft * conversion_factor,
                                height=hf_height_ft * conversion_factor, center_x=center_x_val,
                                center_y=frac_coords, materials=fracture_mats)
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

    builder.add_nodeset_by_coord(nodeset_op_name="injection", new_boundary_name="injection",
                                 coordinates=(center_x_val, frac_coords, 0))

    builder.add_variables([
        {"name": "pp", "params": {"initial_condition": kwargs.get('initial_pressure', 5.17E7)}},
        {"name": "disp_x", "params": {"initial_condition": 0}},
        {"name": "disp_y", "params": {"initial_condition": 0}}
    ])

    builder.set_porous_flow_dictator(dictator_name="dictator", porous_flow_variables="pp")
    builder.add_global_params({"PorousFlowDictator": "dictator", "displacements": "'disp_x disp_y'"})

    builder.add_porous_flow_mass_time_derivative_kernel(kernel_name="dt", variable="pp")
    builder.add_porous_flow_darcy_base_kernel(kernel_name="flux", variable="pp")
    builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_x", variable="disp_x", component=0)
    builder.add_stress_divergence_tensor_kernel(kernel_name="grad_stress_y", variable="disp_y", component=1)
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_x", variable="disp_x", component=0,
                                                             biot_coefficient=biot_coeff)
    builder.add_porous_flow_effective_stress_coupling_kernel(kernel_name="eff_stress_y", variable="disp_y", component=1,
                                                             biot_coefficient=biot_coeff)
    builder.add_porous_flow_mass_volumetric_expansion_kernel(kernel_name="mass_exp", variable="pp")

    fluid_property = SimpleFluidPropertiesConfig(name="water", bulk_modulus=2.2E9, viscosity=1.0E-3, density0=1000.0)
    builder.add_fluid_properties_config(fluid_property)
    builder.add_poromechanics_materials(
        fluid_properties_name="water",
        biot_coefficient=biot_coeff,
        solid_bulk_compliance=2E-11
    )

    gauge_data_for_moose = Data1DGauge()
    gauge_data_for_moose.load_npz("data/fiberis_format/post_processing/history_matching_pressure_profile_full.npz")

    builder.add_piecewise_function_from_data1d(
        name = "injection_pressure_func",
        source_data1d = gauge_data_for_moose
    )

    builder.set_hydraulic_fracturing_bcs(
        injection_well_boundary_name="injection",
        injection_pressure_function_name="injection_pressure_func",
        confine_disp_x_boundaries="left right",
        confine_disp_y_boundaries="top bottom"
    )

    builder.add_standard_tensor_aux_vars_and_kernels({"stress": "stress", "total_strain": "strain"})
