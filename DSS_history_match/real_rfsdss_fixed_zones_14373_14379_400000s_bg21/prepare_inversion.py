"""Apply the approved 14376-ft proposal to the existing field/V6 input template."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'real_rfsdss_fixed_zones_14710_14725_400000s_bg21'
SOURCE = PREVIOUS/'comparison_xx_yy_60/fixed_yy'
RUN = HERE/'inversion'
sys.path.insert(0,str(PREVIOUS))
from prepare_xy_comparison import replace_block, set_value, add_objects


def main():
    if RUN.exists():
        raise RuntimeError('Refusing to overwrite existing inversion')
    p = json.loads((HERE/'PROPOSED_PARAMETERS.json').read_text())
    edges = np.array(p['mesh']['element_edges_y_m'])
    materials = np.array(p['mesh']['material_edges_y_m'])
    zone = p['zones'][0]
    indices = zone['layer_indices_zero_based']
    alpha = np.full(200,-21.)
    alpha[indices] = zone['initial_alpha_xx']
    full = np.r_[alpha,zone['initial_alpha_yy']]
    values = '; '.join(f'{v:.16g}' for v in full)
    names = [f'perm_{i+1}' for i in range(200)]+['kyy_A']
    original = (SOURCE/'forward_and_adjoint.i').read_text()
    forward = replace_block(original,'base_mesh',lambda b:set_value(b,'dy',' '.join(f'{v:.16g}' for v in np.diff(edges))))
    for i in range(200):
        forward = replace_block(forward,f'layer_{i+1}_bbox',lambda b,i=i:set_value(
            set_value(b,'bottom_left',f'0 {materials[i]:.16g} 0'),'top_right',f'100 {materials[i+1]:.16g} 0'))
    for name in ['kyy_B_function','kyy_C_function','kyy_B_aux','kyy_C_aux','grad_kyy_B','grad_kyy_C']:
        forward = replace_block(forward,name,lambda b:'')
    zone_blocks = ' '.join(f'layer_{i+1}' for i in indices)
    other_blocks = ' '.join(f'layer_{i+1}' for i in range(200) if i not in indices)
    forward = replace_block(forward,'kyy_A_aux',lambda b:set_value(b,'block',zone_blocks))
    forward = replace_block(forward,'grad_kyy_A',lambda b:set_value(b,'block',zone_blocks))
    forward = replace_block(forward,'perm_yy_aux',lambda b:set_value(b,'block',other_blocks))
    forward = replace_block(forward,'params',lambda b:set_value(
        set_value(b,'real_vector_names',' '.join(names)),'real_vector_values',values))
    data = pd.read_csv(HERE/'data/measurement_data.csv')
    ys = data.measurement_ycoord.to_numpy()
    cell = np.searchsorted(edges,ys,side='right')-1
    xcenters = (np.floor(data.measurement_xcoord.to_numpy()/.5)+.5)*.5
    vectors = [edges[cell+1],edges[cell],1/(edges[cell+1]-edges[cell]),xcenters]
    sampling = '; '.join(' '.join(f'{v:.16g}' for v in arr) for arr in vectors)
    forward = replace_block(forward,'sampling_geometry',lambda b:set_value(b,'real_vector_values',sampling))
    forward = forward.replace('# h_fd = 0.5 matches one mesh element in y (200 elements across 100 m).',
        '# sampling_geometry uses the exact local element edges and inverse height.')
    forward = add_objects(forward,'Reporters',"""  [signed_data]
    type = OptimizationData
    variable = 'strain_yy'
    measurement_file = 'measurement_data.csv'
    file_xcoord = 'measurement_xcoord'
    file_ycoord = 'measurement_ycoord'
    file_zcoord = 'measurement_zcoord'
    file_time = 'measurement_time'
    file_value = 'measurement_values'
    outputs = 'csv'
  []
""")
    master = (SOURCE/'optimize.i').read_text()
    for key,value in dict(parameter_names=' '.join(names),num_values=' '.join(['1']*201),
        initial_condition=values,lower_bounds='; '.join(['-21']*201),upper_bounds='; '.join(['-12']*201)).items():
        master=set_value(master,key,value)
    for name in ['B','C']:
        for ref in [f'OptimizationReporter/kyy_{name}',f'params/kyy_{name}',
                    f'grad_kyy_{name}/inner_product',f'OptimizationReporter/grad_kyy_{name}']:
            assert master.count('\n'+ref)==1,ref
            master=master.replace('\n'+ref,'')
    assert 'kyy_B' not in master+forward and 'kyy_C' not in master+forward
    for section in ['Kernels','Materials','BCs','Executioner','Preconditioning','FluidProperties']:
        pattern=r'^\['+section+r'\].*?^\[\]'
        assert re.search(pattern,forward,re.M|re.S)[0]==re.search(pattern,original,re.M|re.S)[0],section
    assert len(data)==18*130 and data.measurement_time.nunique()==130
    assert len(np.unique(cell))==18 and len(edges)-1==216
    assert np.all(np.isin(materials,edges))
    mid=(edges[:-1]+edges[1:])/2
    block=np.searchsorted(materials,mid,side='right')-1
    assert np.sum(np.isin(block,indices))==4
    np.testing.assert_allclose(np.sum(np.diff(edges)[np.isin(block,indices)]),zone['width_m'],atol=1e-12)
    cfg=dict(p, status='USER_APPROVED; prepared for initial evaluation, two FD checks, and L-BFGS-B',
        initial_theta=[zone['initial_alpha_xx'],zone['initial_alpha_yy']],
        background_alpha=-21.,n_parameters=2,zone_layer_indices=[indices],
        parameter_order=['alpha_xx_A','alpha_yy_A'],regularization_reference_alpha=-21.,
        regularization_delta=.05,beta_l1=p['regularization']['beta'],n_time=130,
        end_time=p['actual_time_s'][1],md_center_ft=14376.,y_center_m=50.,
        mesh_edges_m=materials.tolist(),mesh_element_edges_m=edges.tolist(),
        solver_tolerances='Unchanged l_tol=nl_abs_tol=nl_rel_tol=0.001',source_template=str(SOURCE))
    RUN.mkdir()
    for name in ['measurement_data.csv','solver_times.csv']:
        shutil.copy2(HERE/'data'/name,RUN/name)
    (RUN/'forward_and_adjoint.i').write_text(forward)
    (RUN/'optimize.i').write_text(master)
    (RUN/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    np.savetxt(RUN/'initial_alpha.txt',full)
    audit=dict(input_sha256={n:hashlib.sha256((RUN/n).read_bytes()).hexdigest() for n in
        ['forward_and_adjoint.i','optimize.i','config.json','initial_alpha.txt','measurement_data.csv','solver_times.csv']},
        approved_proposal_sha256=hashlib.sha256((HERE/'PROPOSED_PARAMETERS.json').read_bytes()).hexdigest(),
        nx=200,ny=216,material_blocks=200,n_parameters=2,moose_reporter_slots=201,
        zone_layer_indices_zero_based=indices,zone_y_elements=4,distinct_observation_elements=18,
        n_solver_steps=129,max_solver_steps=240,mpi_ranks=20,
        unchanged_sections=['Kernels','Materials','BCs','Executioner','Preconditioning','FluidProperties'])
    (RUN/'preflight.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
