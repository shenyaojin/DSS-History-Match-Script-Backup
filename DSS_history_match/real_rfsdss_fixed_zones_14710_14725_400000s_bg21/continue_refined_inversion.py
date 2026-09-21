"""Wait for the mesh comparison; invert only if J_data decreases and FD passes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

from run_xy_comparison import Comparison, save_json

HERE = Path(__file__).resolve().parent
REFINED = HERE/'mesh_refinement_channel_scale'
COARSE = HERE/'comparison_xx_yy_60/fixed_yy'
CONTROL = HERE/'mesh_refinement_followup'
RUN = HERE/'mesh_refinement_inversion'


def status(stage, **details):
    record = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), **details)
    save_json(CONTROL/'status.json', record)
    print(json.dumps(record), flush=True)


def compare_data(coarse, refined):
    keys = ['measurement_time', 'measurement_xcoord', 'measurement_ycoord',
            'measurement_zcoord', 'measurement_values']
    np.testing.assert_allclose(coarse[keys], refined[keys], rtol=1e-10, atol=1e-18)
    assert len(refined) == 45*130 and refined.measurement_time.nunique() == 130
    values = []
    for frame in [coarse, refined]:
        assert np.isfinite(frame[keys+['simulation_values']].to_numpy()).all()
        assert (frame.simulation_values >= 0).all()
        use = frame.measurement_time > 0
        residual = frame.loc[use, 'simulation_values']-frame.loc[use, 'measurement_values']
        values.append(float(.5*np.sum(residual**2)))
    assert values[0] > 0
    return dict(coarse_J_data=values[0], refined_J_data=values[1],
                reduction_fraction=1-values[1]/values[0],
                passed=values[1] < values[0], criterion='refined_J_data < coarse_J_data')


def verify_inputs():
    audit = json.loads((REFINED/'preflight.json').read_text())
    for name, digest in audit['input_sha256'].items():
        assert hashlib.sha256((REFINED/name).read_bytes()).hexdigest() == digest, name
    assert (REFINED/'INPUT_CHECK_PASSED').exists()
    return audit


def prepare_and_seed(audit):
    if RUN.exists():
        raise RuntimeError('Inversion directory exists; refusing a duplicate or overwrite')
    cfg = json.loads((REFINED/'config.json').read_text())
    assert cfg['solve_kyy'] is False and cfg['n_parameters'] == 3
    cfg.update(case_stage='Refined grid inversion after lower fixed-parameter J_data and two FD checks',
               initial_kxx_theta=cfg['initial_theta'],
               finite_difference_checks='Two xx-only central differences on refined grid before optimization',
               initial_evaluation_source=str(REFINED/'evaluations/0001'))
    RUN.mkdir()
    names = ['optimize.i', 'forward_and_adjoint.i', 'measurement_data.csv', 'solver_times.csv', 'initial_alpha.txt']
    for name in names:
        shutil.copy2(REFINED/name, RUN/name)
        assert hashlib.sha256((RUN/name).read_bytes()).hexdigest() == audit['input_sha256'][name]
    save_json(RUN/'config.json', cfg)
    save_json(RUN/'preflight.json', dict(
        input_sha256={n: hashlib.sha256((RUN/n).read_bytes()).hexdigest() for n in names+['config.json']},
        input_check='Reused from byte-identical MOOSE inputs and data in completed refinement case',
        mpi_ranks=20, n_solver_steps=129, max_solver_steps=240))
    (RUN/'INPUT_CHECK_PASSED').write_text('Inherited check: byte-identical MOOSE inputs and data\n')
    (RUN/'.gitignore').write_text('/solver_output/\n/evaluations/\n/iterations/\n/*.i\n/measurement_data.csv\n/solver_times.csv\n/initial_alpha.txt\n/last_solver_stderr.txt\n/strain_final.csv\n/RUNNING\n/INPUT_CHECK_PASSED\n')
    model = Comparison(RUN)
    np.testing.assert_array_equal(model.expand(model.theta0), np.loadtxt(RUN/'initial_alpha.txt'))
    source = REFINED/'evaluations/0001'
    ev = json.loads((source/'evaluation.json').read_text())
    np.testing.assert_array_equal(ev['theta'], model.theta0)
    assert np.isfinite(ev['gradient_scaled']).all() and len(ev['gradient_scaled']) == 3
    destination = RUN/'evaluations/0001'
    destination.parent.mkdir()
    shutil.copytree(source, destination)
    shutil.copy2(REFINED/'objective_history.csv', RUN/'objective_history.csv')
    model.count = 1
    key = tuple(model.theta0)
    model.artifacts[key] = destination
    model.cache[key] = (float(ev['scaled_objective']), np.asarray(ev['gradient_scaled']))
    save_json(RUN/'initial_reuse.json', dict(source=str(source), evaluation=ev,
        note='Completed refined forward/adjoint evaluation reused without another solve'))
    return model


def check_gradients(model):
    initial = json.loads((model.artifacts[tuple(model.theta0)]/'evaluation.json').read_text())
    checks = []
    for direction in [np.array([-.7, -.9, .6]), np.array([.5, .8, -.6])]:
        direction /= np.linalg.norm(direction)
        h = .02
        objectives = []
        for sign in [1, -1]:
            theta = model.theta0+sign*h*direction
            model.evaluate(theta, purpose='refined_grid_gradient_check')
            ev = json.loads((model.artifacts[tuple(theta)]/'evaluation.json').read_text())
            objectives.append(ev['data_objective']*model.cfg['objective_scale'])
        fd = (objectives[0]-objectives[1])/(2*h)
        adjoint = float(np.asarray(initial['data_gradient_unscaled'])@direction*model.cfg['objective_scale'])
        tolerance = max(1e-7, .05*max(abs(fd), abs(adjoint)))
        checks.append(dict(direction=direction.tolist(), h=h, fd=fd, adjoint=adjoint,
                           tolerance=tolerance, relative_error=abs(fd-adjoint)/max(abs(fd), abs(adjoint), 1e-8),
                           passed=bool(abs(fd-adjoint) <= tolerance)))
        save_json(RUN/'gradient_check.json', checks)
        print('REFINED_GRADIENT_CHECK '+json.dumps(checks[-1]), flush=True)
        if not checks[-1]['passed']:
            raise RuntimeError('Refined-grid directional gradient check failed; optimization not started')


def main():
    os.environ['PATH'] = '/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin:'+os.environ['PATH']
    CONTROL.mkdir(exist_ok=True)
    # Exclusive and persistent: completed/failed controllers cannot silently restart.
    with (CONTROL/'STARTED').open('x') as stream:
        stream.write(f'pid={os.getpid()}\n')
    status('waiting_for_mesh_comparison')
    try:
        audit = verify_inputs()
        while not (REFINED/'COMPLETE').exists():
            if (REFINED/'FAILED').exists():
                raise RuntimeError('Mesh comparison failed: '+(REFINED/'FAILED').read_text())
            marker = REFINED/'RUNNING'
            if not marker.exists():
                raise RuntimeError('Mesh comparison stopped without COMPLETE')
            pid = int(marker.read_text().strip().split('=')[1])
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                if not (REFINED/'COMPLETE').exists():
                    raise RuntimeError('Mesh comparison process disappeared without COMPLETE')
            time.sleep(10)
        verify_inputs()
        gate = compare_data(pd.read_csv(COARSE/'strain_final.csv'), pd.read_csv(REFINED/'strain_refined.csv'))
        ev = json.loads((REFINED/'evaluations/0001/evaluation.json').read_text())
        assert np.isclose(gate['refined_J_data'], ev['data_objective'], rtol=1e-8, atol=1e-22)
        save_json(CONTROL/'decision.json', gate)
        if not gate['passed']:
            status('stopped_no_data_misfit_reduction', **gate)
            return
        status('gradient_checks', **gate)
        model = prepare_and_seed(audit)
        (RUN/'RUNNING').write_text(f'pid={os.getpid()}\n')
        check_gradients(model)
        status('optimizing', max_iterations=60, parameters=['xx_A', 'xx_B', 'xx_C'])
        model.run_comparison()
        status('complete', result=json.loads((RUN/'optimizer_result.json').read_text()))
    except Exception as exc:
        status('failed', reason=f'{type(exc).__name__}: {exc}')
        if RUN.exists():
            (RUN/'FAILED').write_text(f'{type(exc).__name__}: {exc}\n')
        raise
    finally:
        if RUN.exists():
            (RUN/'RUNNING').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
