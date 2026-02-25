[Mesh]
    [grid]
        type = GeneratedMeshGenerator
        dim = 2
        nx = 200
        ny = 200
        xmin = 0.0
        xmax = 200.0
        ymin = 0.0
        ymax = 100.0
    []

    [matrix_top]
        type = SubdomainBoundingBoxGenerator
        input = 'grid'
        block_id = 1
        bottom_left = '0.0 52.0 0.0'
        top_right = '200.0 100.0 0.0'
    []

    [matrix_bottom]
        type = SubdomainBoundingBoxGenerator
        input = 'matrix_top'
        block_id = 2
        bottom_left = '0.0 0.0 0.0'
        top_right = '200.0 48.0 0.0'
    []

    [srv]
        type = SubdomainBoundingBoxGenerator
        input = 'matrix_bottom'
        block_id = 3
        bottom_left = '0.0 48.0 0.0'
        top_right = '200.0 52.0 0.0'
    []

    [injection]
        type = ExtraNodesetGenerator
        input = 'srv'
        new_boundary = 'injection_well'
        coord = '20.0 50.0 0.0'
    []

    [final_block_rename]
        type = RenameBlockGenerator
        input = 'injection'
        old_block = '1 2 3'
        new_block = 'matrix_top matrix_bottom srv'
    []
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
    [pp_adjoint]
        solver_sys = adjoint
        initial_condition = 0
    []
    [disp_x_adjoint]
        solver_sys = adjoint
        initial_condition = 0
    []
    [disp_y_adjoint]
        solver_sys = adjoint
        initial_condition = 0
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
    displacements = 'disp_x disp_y'
  []
  [grad_stress_y]
    type = StressDivergenceTensors
    variable = 'disp_y'
    component = 1
    displacements = 'disp_x disp_y'
  []
  [eff_stress_x]
    type = PorousFlowEffectiveStressCoupling
    variable = 'disp_x'
    component = 0
    biot_coefficient = 0.7
    displacements = 'disp_x disp_y'
  []
  [eff_stress_y]
    type = PorousFlowEffectiveStressCoupling
    variable = 'disp_y'
    component = 1
    biot_coefficient = 0.7
    displacements = 'disp_x disp_y'
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
        type = PorousFlowPermeabilityConstFromVar
        perm_xx = perm_xx_top
        perm_yy = perm_yy_var
        perm_zz = perm_zz_var
        block = 'matrix_top'
    []

    [porosity_bottom]
        type = PorousFlowPorosityConst
        porosity = 0.01
        block = 'matrix_bottom'
    []

    [permeability_bottom]
        type = PorousFlowPermeabilityConstFromVar
        perm_xx = perm_xx_down
        perm_yy = perm_yy_var
        perm_zz = perm_zz_var
        block = 'matrix_bottom'
    []

    [porosity_srv]
        type = PorousFlowPorosityConst
        porosity = 0.14
        block = 'srv'
    []

    [permeability_srv]
        type = PorousFlowPermeabilityConstFromVar
        perm_xx = perm_xx_center
        perm_yy = perm_yy_var
        perm_zz = perm_zz_var
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
        displacements = 'disp_x disp_y'
    []
[]

[Functions]
    [injection_pressure_func]
    type = PiecewiseConstant
    x = '0.0 725.0 2023.0 3446.0 4010.0 4486.0 6280.0 7401.0 8565.0 9324.0 10004.0 11209.0 12462.0 13293.0 13963.0 15250.0 16729.0 17594.0 18285.0 19779.0 21134.0 21932.0 22626.0 23747.0 24889.0 26189.0 27496.0 28872.0 30958.0 31714.0 32670.0 34576.0 36124.0 36770.0 37868.0 40675.0 41599.0 42350.0 44163.0 44947.0 45879.0 47003.0 47855.0 50000.0 53789.0 55753.0 57158.0 58487.0 60090.0 64075.0 70155.0 70565.0 70784.0 71329.0 73249.0 74569.0 75530.0 77450.0 80570.0 83570.0 84995.0 87650.0 89731.0 97414.0 101177.0 110886.0 116559.0 121001.0 125569.0 137602.0 151970.0 173303.0 195503.0 227259.0 277985.0 315033.0 318138.0 340874.0 341330.0 341570.0 341790.0 342540.0 342847.0 343130.0 343568.0 343817.0 344209.0 344355.0 345472.0 346009.0 346609.0 346947.0 347778.0 348823.0 349380.0 349953.0 350329.0 350970.0 351597.0 351798.0 352607.0 353330.0 353864.0 354794.0 356061.0 357077.0 358293.0 359450.0 360590.0 362930.0 364129.0 366290.0 368086.0 369650.0 371955.0 373263.0 386280.0 389820.0 390821.0 396160.0 401327.0 402975.0 404295.0 405306.0 408952.0 413163.0 415921.0 421390.0 427129.0 428431.0'
    y = '12691246.674523842 12696027.22531744 12685664.883670641 12701609.01617064 12685981.34625988 12682156.905625 12700747.17117064 12685880.34492064 12707413.004454361 12689475.85194448 12685940.94296628 12706605.02131948 12685126.23054564 12694579.59461308 12710261.133283721 12690135.70805552 12711405.77375988 12695825.22953372 12692411.5166468 12709810.00913692 12693313.75804564 12704591.80655756 12705891.31023808 12693892.807569481 12707763.12726192 12692034.45601192 12700652.90601192 12689246.925228199 12690283.83507936 12683483.340238081 12687408.775317442 12679362.64555552 12683436.207658721 12676851.174067441 12681241.19875988 12671882.09985116 12679537.70351192 12671956.16336308 12676716.51250988 12669336.96125 12677329.229146801 12676945.43922616 12669060.90195436 12675309.278204361 12662361.40155756 12669835.21797616 12662415.27031744 12667727.73805552 12660947.43796628 12662805.789523842 12645124.50155756 12674521.49671628 12707096.54875988 12767297.76648808 12936771.50507936 13037041.7860218 13096711.08680552 13200287.35000988 13346861.59805552 13465809.673521802 13512100.17515872 13587073.9539782 13637828.543819482 13774505.04672616 13820681.08155756 13914157.596944481 13964932.38843256 13996867.78601192 14040606.41976192 14127215.112976162 14221371.679226162 14346103.541329361 14449928.9328968 14582807.926101161 14760920.30944448 14875404.30360116 14899434.96226192 14965191.046805521 14901717.50703372 14881827.73780756 14873115.02264884 14891671.62797616 14913722.738988081 14924172.609613081 14932144.675863082 14921135.95046628 14846148.706180522 14832163.91968256 14858120.271021802 14853151.196805522 14799763.939384881 14777355.96938488 14785940.752271801 14708590.163521802 14704711.861021802 14675530.32711308 14649190.1858532 14625334.58514884 14587076.74780756 14580559.048442442 14579172.015783722 14552858.812351162 14549317.16719244 14517482.76405756 14501188.508750001 14496912.95086308 14470970.071884882 14454945.13969244 14453174.317113081 14425999.268680522 14422471.08898808 14396164.61484128 14386098.53413692 14365966.3727282 14354567.128125 14336656.91344244 14213709.339384882 14176784.665942442 14159143.774375001 14119310.37405756 14090741.56368052 14097548.78780756 14090155.777976159 14093993.67718256 14083489.94469244 14059567.00976192 14033873.2591468 14010179.25093256 14004543.59131948 13997130.37984128'
    []


    [perm_up]
        type = ParsedOptimizationFunction
        expression = '10^alpha'
        param_symbol_names = 'alpha'
        param_vector_name = 'params/perm_1'
    []

    [perm_center]
        type = ParsedOptimizationFunction
        expression = '10^alpha'
        param_symbol_names = 'alpha'
        param_vector_name = 'params/perm_2'
    []

    [perm_down]
        type = ParsedOptimizationFunction
        expression = '10^alpha'
        param_symbol_names = 'alpha'
        param_vector_name = 'params/perm_3'
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
  [perm_xx_top]
    order = CONSTANT
    family = MONOMIAL
  []
  [perm_xx_center]
    order = CONSTANT
    family = MONOMIAL
  []
  [perm_xx_down]
    order = CONSTANT
    family = MONOMIAL
  []
  [perm_yy_var]
    order = CONSTANT
    family = MONOMIAL
  []
  [perm_zz_var]
    order = CONSTANT
    family = MONOMIAL
  []
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
  [perm_xx_top_aux]
    type = FunctionAux
    variable = perm_xx_top
    function = perm_up
    block = 'matrix_top'
    execute_on = 'INITIAL TIMESTEP_BEGIN TIMESTEP_END'
  []
  [perm_xx_center_aux]
    type = FunctionAux
    variable = perm_xx_center
    function = perm_center
    block = 'srv'
    execute_on = 'INITIAL TIMESTEP_BEGIN TIMESTEP_END'
  []
  [perm_xx_down_aux]
    type = FunctionAux
    variable = perm_xx_down
    function = perm_down
    block = 'matrix_bottom'
    execute_on = 'INITIAL TIMESTEP_BEGIN TIMESTEP_END'
  []
  [perm_yy_aux]
    type = FunctionAux
    variable = perm_yy_var
    function = func_kyy
    execute_on = 'INITIAL TIMESTEP_BEGIN TIMESTEP_END'
  []
  [perm_zz_aux]
    type = FunctionAux
    variable = perm_zz_var
    function = func_zero
    execute_on = 'INITIAL TIMESTEP_BEGIN TIMESTEP_END'
  []
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
  type = TransientAndAdjoint
  forward_system = nl0
  adjoint_system = adjoint

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

[Problem]
  nl_sys_names = 'nl0 adjoint'
  kernel_coverage_check = false
[]

[Reporters]
  [data]
    type = OptimizationData
    objective_name = objective_value
    variable = disp_y
    outputs = none
  []
  [params]
    type = ConstantReporter
    real_vector_names = 'perm_1 perm_2 perm_3'
    real_vector_values = '1E-15; 1E-15; 1E-15'
    outputs = none
  []
[]

[DiracKernels]
  [misfit]
    type = ReporterTimePointSource
    variable = disp_y_adjoint
    value_name = data/misfit_values
    x_coord_name = data/measurement_xcoord
    y_coord_name = data/measurement_ycoord
    z_coord_name = data/measurement_zcoord
    time_name = data/measurement_time
  []
[]

[VectorPostprocessors]
  [grad_perm_up]
    type = ElementOptimizationDiffusionCoefFunctionInnerProduct
    variable = pp_adjoint
    forward_variable = pp
    function = perm_up
    block = matrix_top
    execute_on = ADJOINT_TIMESTEP_END
    outputs = none
  []
  [grad_perm_center]
    type = ElementOptimizationDiffusionCoefFunctionInnerProduct
    variable = pp_adjoint
    forward_variable = pp
    function = perm_center
    block = srv
    execute_on = ADJOINT_TIMESTEP_END
    outputs = none
  []
  [grad_perm_down]
    type = ElementOptimizationDiffusionCoefFunctionInnerProduct
    variable = pp_adjoint
    forward_variable = pp
    function = perm_down
    block = matrix_bottom
    execute_on = ADJOINT_TIMESTEP_END
    outputs = none
  []
[]

[Outputs]
  exodus = true
  console = true
[]


