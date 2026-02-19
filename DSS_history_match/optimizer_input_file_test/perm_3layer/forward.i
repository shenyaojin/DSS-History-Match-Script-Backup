# Ground truth model for forward model testing
# Shenyao Jin, 02092026

[Mesh]
[]

[Variables]
    [pp]
        initial_condition = 0.0
    []
    [disp_x]
        initial_condition = 0.0
    []
    [disp_y]
        initial_condition = 0.0
    []
[]

[UserObjects]
    [dictator]
        type = PorousFlowDictator
        porous_flow_vars = 'pp'
        number_fluid_phases = 1
        number_fluid_components = 1
    []
[]

[GlobalParams]
    PorousFlowDictator = 'dictator'
    displacements = 'disp_x disp_y'
[]

[Kernels]
  [dot_pp]
    type = TimeDerivative
    variable = 'pp'
  []
  [flux]
    type = PorousFlowFullySaturatedDarcyBase
    variable = 'pp'
    gravity = '0 0 0'
  []
  [grad_stress_x]
    type = StressDivergenceTensors
    variable = 'disp_x'
    component = 0
  []
  [grad_stress_y]
    type = StressDivergenceTensors
    variable = 'disp_y'
    component = 1
  []
  [eff_stress_x]
    type = PorousFlowEffectiveStressCoupling
    variable = 'disp_x'
    component = 0
    biot_coefficient = 0.7
  []
  [eff_stress_y]
    type = PorousFlowEffectiveStressCoupling
    variable = 'disp_y'
    component = 1
    biot_coefficient = 0.7
  []
[]

[FluidProperties]
  [oil]
    type = SimpleFluidProperties
    bulk_modulus = 1.2e9
    viscosity = 0.0012
    density0 = 825.0
    thermal_expansion = 0.0008
    cp = 2100.0
    cv = 2000.0
    porepressure_coefficient = 1.0
  []
[]

[Materials]
    [porosity_top]
        type = PorousFlowPorosityConst
        porosity = 0.01
        block = 'matrix_top'
    []

    [permeability_top]
        type = ADGenericFunctionRankTwoTensor
        tensor_name = 'permeability'
        permeability = 'perm_up func_zero func_zero  func_zero func_kyy func_zero  func_zero func_zero func_zero'
        block = 'matrix_top'
    []

    [porosity_bottom]
        type = PorousFlowPorosityConst
        porosity = 0.01
        block = 'matrix_bottom'
    []

    [permeability_bottom]
        type = ADGenericFunctionRankTwoTensor
        tensor_name = 'permeability'
        permeability = 'perm_down func_zero func_zero  func_zero func_kyy func_zero  func_zero func_zero func_zero'
        block = 'matrix_bottom'
    []

    [porosity_srv]
        type = PorousFlowPorosityConst
        porosity = 0.14
        block = 'srv'
    []

    [permeability_srv]
        type = ADGenericFunctionRankTwoTensor
        tensor_name = 'permeability'
        permeability = 'perm_center func_zero func_zero  func_zero func_kyy func_zero  func_zero func_zero func_zero'
        block = 'srv'
    []

    [temperature]
        type = PorousFlowTemperature
    []
    [biot_modulus]
        type = PorousFlowConstantBiotModulus
        biot_coefficient = 0.7
        solid_bulk_compliance = 2e-11
        fluid_bulk_modulus = 2200000000.0
        block = 'matrix_bottom matrix_top srv'
    []
    [massfrac]
        type = PorousFlowMassFraction
    []
    [simple_fluid]
        type = PorousFlowSingleComponentFluid
        fp = 'oil'
        phase = 0
    []
    [PS]
        type = PorousFlow1PhaseFullySaturated
        porepressure = 'pp'
    []
    [relp]
        type = PorousFlowRelativePermeabilityConst
        phase = 0
    []
    [eff_fluid_pressure_qp]
        type = PorousFlowEffectiveFluidPressure
    []
    [elasticity_tensor_matrix]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = 50000000000.0
        poissons_ratio = 0.2
    []
    [strain]
        type = ComputeSmallStrain
        displacements = 'disp_x disp_y'
    []
    [stress]
        type = ComputeLinearElasticStress
    []
    [vol_strain]
        type = PorousFlowVolumetricStrain
    []
[]

[Functions]
    [injection_pressure_func]
        type = PiecewiseConstant
        x = '0.0 7200.0 10800.0 14400.0 18000.0 21600.0 25200.0 28800.0'
        y = '0.0 1380000.0 1380000.0 1380000.0 690000.0 345000.0 345000.0 345000.0'
    []

    [perm_up]
        type = ParsedOptimizationFunction
        expression = 'alpha'
        param_symbol_names = 'alpha'
        params_vector_name = 'params/perm_1'
    []

    [perm_center]
        type = ParsedOptimizationFunction
        expression = 'alpha'
        param_symbol_names = 'alpha'
        params_vector_name = 'params/perm_2'
    []

    [perm_down]
        type = ParsedOptimizationFunction
        expression = 'alpha'
        param_symbol_names = 'alpha'
        params_vector_name = 'params/perm_3'
    []

    [func_kyy]
        type = ParsedFunction
        expression = '1E-20'
    []

    [func_zero]
        type = ParsedFunction
        expression = '0'
    []
[]

[BCs]
  [injection_pressure]
    type = FunctionDirichletBC
    variable = 'pp'
    boundary = 'injection_well'
    function = 'injection_pressure_func'
  []
  [confinex]
    type = DirichletBC
    variable = 'disp_x'
    boundary = 'left right'
    value = 0
  []
  [confiney]
    type = DirichletBC
    variable = 'disp_y'
    boundary = 'top bottom'
    value = 0
  []
[]

[AuxVariables]
  [stress_xx]
    order = 'CONSTANT'
    family = 'MONOMIAL'
  []
  [stress_xy]
    order = 'CONSTANT'
    family = 'MONOMIAL'
  []
  [stress_yx]
    order = 'CONSTANT'
    family = 'MONOMIAL'
  []
  [stress_yy]
    order = 'CONSTANT'
    family = 'MONOMIAL'
  []
  [strain_xx]
    order = 'CONSTANT'
    family = 'MONOMIAL'
  []
  [strain_xy]
    order = 'CONSTANT'
    family = 'MONOMIAL'
  []
  [strain_yx]
    order = 'CONSTANT'
    family = 'MONOMIAL'
  []
  [strain_yy]
    order = 'CONSTANT'
    family = 'MONOMIAL'
  []
[]


[AuxKernels]
  [stress_xx]
    type = RankTwoAux
    rank_two_tensor = 'stress'
    variable = 'stress_xx'
    index_i = 0
    index_j = 0
  []
  [stress_xy]
    type = RankTwoAux
    rank_two_tensor = 'stress'
    variable = 'stress_xy'
    index_i = 0
    index_j = 1
  []
  [stress_yx]
    type = RankTwoAux
    rank_two_tensor = 'stress'
    variable = 'stress_yx'
    index_i = 1
    index_j = 0
  []
  [stress_yy]
    type = RankTwoAux
    rank_two_tensor = 'stress'
    variable = 'stress_yy'
    index_i = 1
    index_j = 1
  []
  [strain_xx]
    type = RankTwoAux
    rank_two_tensor = 'total_strain'
    variable = 'strain_xx'
    index_i = 0
    index_j = 0
  []
  [strain_xy]
    type = RankTwoAux
    rank_two_tensor = 'total_strain'
    variable = 'strain_xy'
    index_i = 0
    index_j = 1
  []
  [strain_yx]
    type = RankTwoAux
    rank_two_tensor = 'total_strain'
    variable = 'strain_yx'
    index_i = 1
    index_j = 0
  []
  [strain_yy]
    type = RankTwoAux
    rank_two_tensor = 'total_strain'
    variable = 'strain_yy'
    index_i = 1
    index_j = 1
  []
[]

[Executioner]
  type = 'Transient'
  solve_type = 'NEWTON'
  end_time = 28800
  verbose = true
  l_tol = 0.001
  l_max_its = 2000
  nl_max_its = 200
  nl_abs_tol = 0.001
  nl_rel_tol = 0.001
  [TimeStepper]
    type = ConstantDT
    dt = 600
  []
[]

[Preconditioning]
  active = 'mumps'
  [mumps]
    type = SMP
    full = true
    petsc_options = '-snes_converged_reason -ksp_diagonal_scale -ksp_diagonal_scale_fix -ksp_gmres_modifiedgramschmidt -snes_linesearch_monitor'
    petsc_options_iname = '-ksp_type -pc_type -pc_factor_mat_solver_package -pc_factor_shift_type'
    petsc_options_value = 'gmres      lu         mumps                     NONZERO'
  []
  [basic]
    type = SMP
    full = true
  []
  [preferred_but_might_not_be_installed]
    type = SMP
    full = true
    petsc_options_iname = '-pc_type -pc_factor_mat_solver_package'
    petsc_options_value = ' lu         mumps'
  []
[]

[Outputs]
  console = false
  file_base = 'forward'
[]

[Problem]
    library_path = '/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/optimization/lib'
[]

[Reporters]
  [measure_data]
    type = OptimizationData
    objective_name = objective_value
    variable = disp_y
  []
  [params]
    type = ConstantReporter
    real_vector_names = 'perm_1 perm_2 perm_3'
    real_vector_values = '1E-20 1E-20 1E-21' # dummy value
  []
[]