"""Extend the existing input with three independent zone kyy reporters."""
from pathlib import Path
import copy
import hashlib
import json
import re
import shutil

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'pilot_5iter'
ROOT = HERE / 'comparison_xx_yy_60'
INITIAL = [-18., -17.5, -19.]
YY_NAMES = ['kyy_A', 'kyy_B', 'kyy_C']


def replace_block(text, name, transform):
    pattern = r'^  \['+re.escape(name)+r'\].*?^  \[\]'
    text, count = re.subn(pattern, lambda m: transform(m[0]), text, flags=re.M | re.S)
    assert count == 1, (name, count)
    return text


def set_value(text, name, value):
    text, count = re.subn(r'(^\s*'+re.escape(name)+r"\s*=\s*')[^']*(')",
                          lambda m: m[1]+value+m[2], text, count=1, flags=re.M)
    assert count == 1, name
    return text


def add_objects(text, section, objects):
    pattern = r'(^\['+re.escape(section)+r'\]\n)(.*?)(^\[\])'
    text, count = re.subn(pattern, lambda m: m[1]+m[2]+objects+m[3], text, flags=re.M | re.S)
    assert count == 1, section
    return text


def main():
    if ROOT.exists():
        raise RuntimeError('Comparison directory exists; refusing to overwrite')
    cfg = json.loads((SOURCE / 'config.json').read_text())
    active = [z for z in cfg['zones'] if z['optimized']]
    assert [z['layer_indices_zero_based'] for z in active] == [[91], [94], [98, 99, 100]]
    alpha = np.full(200, -21.)
    for zone in cfg['zones']:
        alpha[zone['layer_indices_zero_based']] = zone['alpha']
    for zone, value in zip(active, INITIAL):
        zone['alpha'] = value
        zone['initial_kxx_m2'] = 10.**value
        alpha[zone['layer_indices_zero_based']] = value
    full = np.r_[alpha, INITIAL]
    names = [f'perm_{i+1}' for i in range(200)] + YY_NAMES
    master = (SOURCE / 'optimize.i').read_text()
    for key, value in {
        'parameter_names': ' '.join(names), 'num_values': ' '.join(['1']*203),
        'initial_condition': '; '.join(f'{v:.16g}' for v in full),
        'lower_bounds': '; '.join(['-25']*203), 'upper_bounds': '; '.join(['-10']*203),
    }.items():
        master = set_value(master, key, value)
    # Preserve all existing transfer ordering; append only the three yy entries.
    for tail, additions in [
        ("OptimizationReporter/perm_200'", [f'OptimizationReporter/{n}' for n in YY_NAMES]),
        ("params/perm_200'", [f'params/{n}' for n in YY_NAMES]),
        ("grad_perm_layer_200/inner_product'", [f'grad_{n}/inner_product' for n in YY_NAMES]),
        ("OptimizationReporter/grad_perm_200'", [f'OptimizationReporter/grad_{n}' for n in YY_NAMES]),
    ]:
        assert master.count(tail) == 1, tail
        master = master.replace(tail, tail[:-1]+'\n'+'\n'.join(additions)+"'")

    forward = (SOURCE / 'forward_and_adjoint.i').read_text()
    forward = replace_block(forward, 'params', lambda block: set_value(
        set_value(block, 'real_vector_names', ' '.join(names)),
        'real_vector_values', '; '.join(f'{v:.16g}' for v in full)))
    active_layers = {i for z in active for i in z['layer_indices_zero_based']}
    background_blocks = ' '.join(f'layer_{i+1}' for i in range(200) if i not in active_layers)
    forward = replace_block(forward, 'perm_yy_aux', lambda block: block.replace(
        "    function = 'func_kyy'", "    function = 'func_kyy'\n    block = '"+background_blocks+"'"))
    functions, auxkernels, gradients = [], [], []
    for name, zone in zip(YY_NAMES, active):
        blocks = ' '.join(f'layer_{i+1}' for i in zone['layer_indices_zero_based'])
        functions.append(f"""  [{name}_function]
    type = ParsedOptimizationFunction
    expression = '10^alpha'
    param_symbol_names = 'alpha'
    param_vector_name = 'params/{name}'
  []
""")
        auxkernels.append(f"""  [{name}_aux]
    type = FunctionAux
    variable = 'perm_yy_var'
    function = '{name}_function'
    block = '{blocks}'
    execute_on = 'INITIAL TIMESTEP_BEGIN TIMESTEP_END'
  []
""")
        gradients.append(f"""  [grad_{name}]
    type = PorousFlowOptimizationAnisotropicDiffusionInnerProduct
    variable = 'pp_adjoint'
    forward_variable = 'pp'
    function = '{name}_function'
    component = 1
    block = '{blocks}'
    execute_on = 'ADJOINT_TIMESTEP_END'
    outputs = 'none'
  []
""")
    forward = add_objects(forward, 'Functions', ''.join(functions))
    forward = add_objects(forward, 'AuxKernels', ''.join(auxkernels))
    forward = add_objects(forward, 'VectorPostprocessors', ''.join(gradients))
    cfg.update(initial_kxx_theta=INITIAL, initial_kyy_theta=INITIAL,
               transverse_permeability_m2=None,
               kyy_description='Three active zones use independent yy reporters; all other yy is 1e-21',
               background_change='All background tensor components 1e-21; active-zone xx and yy as configured',
               parameter_order=['xx_A', 'xx_B', 'xx_C', 'yy_A', 'yy_B', 'yy_C'],
               optimizer=dict(method='L-BFGS-B', maxiter=60, ftol=1e-7, gtol=1e-8, maxls=20))
    ROOT.mkdir()
    for mode in ['fixed_yy', 'free_yy']:
        run = ROOT / mode
        run.mkdir()
        local = copy.deepcopy(cfg)
        local.update(case_stage='Matched-initial-tensor comparison, at most 60 accepted iterations',
                     solve_kyy=mode == 'free_yy', n_parameters=6 if mode == 'free_yy' else 3,
                     initial_theta=INITIAL*2 if mode == 'free_yy' else INITIAL,
                     regularization_weights=[1.]*(6 if mode == 'free_yy' else 3),
                     regularization_scope='One term per independently optimized zone/component; fixed yy penalty omitted as constant',
                     finite_difference_checks='Two new yy-only directions before free-yy optimization; existing xx checks not repeated')
        local['parameter_order'] = local['parameter_order'][:local['n_parameters']]
        (run / 'config.json').write_text(json.dumps(local, indent=2)+'\n')
        (run / 'optimize.i').write_text(master)
        (run / 'forward_and_adjoint.i').write_text(forward)
        np.savetxt(run / 'initial_alpha.txt', full)
        for name in ['measurement_data.csv', 'solver_times.csv']:
            shutil.copy2(SOURCE / name, run / name)
        inputs = ['config.json', 'optimize.i', 'forward_and_adjoint.i', 'measurement_data.csv',
                  'solver_times.csv', 'initial_alpha.txt']
        audit = dict(input_sha256={n: hashlib.sha256((run/n).read_bytes()).hexdigest() for n in inputs},
                     source_case=str(SOURCE), n_solver_steps=129, max_solver_steps=240,
                     background_yy_layers=195, active_yy_layers=sorted(active_layers),
                     yy_gradient_component=1, mpi_ranks=20)
        (run / 'preflight.json').write_text(json.dumps(audit, indent=2)+'\n')
    for name in ['optimize.i', 'forward_and_adjoint.i', 'measurement_data.csv', 'solver_times.csv', 'initial_alpha.txt']:
        assert (ROOT/'fixed_yy'/name).read_bytes() == (ROOT/'free_yy'/name).read_bytes(), name
    print('Prepared identical PDE inputs for fixed_yy and free_yy; no solves launched.')


if __name__ == '__main__':
    main()
