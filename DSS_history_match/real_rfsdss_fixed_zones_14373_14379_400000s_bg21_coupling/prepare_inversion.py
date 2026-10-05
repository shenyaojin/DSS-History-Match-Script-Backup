"""Isolate the approved amplitude-coupling run without changing prior results."""
from pathlib import Path
import hashlib
import json
import re
import shutil

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.with_name(HERE.name.removesuffix('_coupling'))
RUN = HERE / 'inversion'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if RUN.exists():
        raise RuntimeError('Refusing to overwrite an existing run')
    cfg = json.loads((SOURCE / 'inversion/config.json').read_text())
    cfg.update(initial_theta=[-17., -17., 1.], n_parameters=3,
               parameter_order=['alpha_xx_A', 'alpha_yy_A', 'coupling_C'],
               parameter_bounds=[[-21., -12.], [-21., -12.], [1e-6, 1.]],
               coupling_bounds=[1e-6, 1.],
               coupling_initialization='Bootstrap at C=1; initialize C from clipped dot(s,d)/dot(s,s)',
               coupling_lower_bound_reason='Numerical positivity floor for d/C; flag any bound hit in results',
               coupling_regularization='None; original L1 applies only to the two permeability amplitudes',
               status='USER_APPROVED: initial solve, FD checks, then at most 60 accepted iterations',
               validation='Two permeability central FD checks at initialized C plus exact C finite difference',
               source_template=str(SOURCE / 'inversion'),
               initial_rationale='User-approved higher permeability and independent global attenuation factor')
    cfg['zones'][0].update(initial_alpha_xx=-17., initial_alpha_yy=-17., kxx_m2=1e-17, kyy_m2=1e-17)
    full = np.full(201, -21.)
    full[cfg['zone_layer_indices'][0]] = -17.
    full[-1] = -17.
    forward = (SOURCE / 'inversion/forward_and_adjoint.i').read_text()
    master = (SOURCE / 'inversion/optimize.i').read_text()
    # The exact identity C² ||s-d/C||² = ||Cs-d||² lets us reuse the existing
    # positive-half-wave adjoint. Only the private solver target is rescaled.
    forward = forward.replace('measurement_data.csv', 'solver_measurement_data.csv')
    master = master.replace('measurement_data.csv', 'solver_measurement_data.csv')
    values = '; '.join(f'{x:.16g}' for x in full)
    master, count = re.subn(r"(initial_condition\s*=\s*')[^']*(')",
                           lambda m: m[1] + values + m[2], master, count=1)
    assert count == 1
    original = (SOURCE / 'inversion/forward_and_adjoint.i').read_text()
    assert forward.replace('solver_measurement_data.csv', 'measurement_data.csv') == original
    obs = pd.read_csv(SOURCE / 'inversion/measurement_data.csv')
    assert len(obs) == 18 * 130 and obs.measurement_time.nunique() == 130
    assert np.all(obs.measurement_values >= 0)
    assert cfg['n_solver_steps'] == 129 and cfg['max_solver_steps'] == 240
    RUN.mkdir()
    # Copy only small observation/QC caches, never previous solver output.
    shutil.copytree(SOURCE / 'data', HERE / 'data')
    for name in ['measurement_data.csv', 'solver_times.csv']:
        shutil.copy2(SOURCE / 'inversion' / name, RUN / name)
    shutil.copy2(RUN / 'measurement_data.csv', RUN / 'solver_measurement_data.csv')
    (RUN / 'forward_and_adjoint.i').write_text(forward)
    (RUN / 'optimize.i').write_text(master)
    (RUN / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    np.savetxt(RUN / 'initial_alpha.txt', full)
    names = ['forward_and_adjoint.i', 'optimize.i', 'config.json', 'initial_alpha.txt',
             'measurement_data.csv', 'solver_times.csv']
    audit = dict(input_sha256={n: digest(RUN / n) for n in names},
                 source_sha256={n: digest(SOURCE / 'inversion' / n) for n in names},
                 source_driver_sha256=digest(SOURCE / 'run_inversion.py'),
                 nx=200, ny=216, n_solver_steps=129, max_solver_steps=240,
                 mpi_ranks=20, n_parameters=3, moose_reporter_slots=201,
                 changes=['Initial active xx/yy=-17', 'Private solver observation file for d/C',
                          'External coupling objective and chain-rule gradients'],
                 forward_physics_byte_identical=True)
    (RUN / 'preflight.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
