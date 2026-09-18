"""Five accepted L-BFGS-B iterations, reusing the current initial PDE solve.

The completed baseline driver supplies expansion, MOOSE execution, rectification,
and gradient reduction. This wrapper changes only bounds/reference/run length,
and keeps each evaluation's small observation report for intermediate QC.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from run_inversion import Inversion, plot_qc

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'inversion'
RUN = HERE / 'pilot_5iter'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def regularization(theta, cfg):
    delta = cfg['regularization_delta']
    d = np.asarray(theta)-cfg['regularization_reference_alpha']
    denom = np.sqrt(d*d+delta*delta)
    return cfg['beta_l1']*np.sum(denom-delta), cfg['beta_l1']*d/denom


def prepare():
    if RUN.exists():
        raise RuntimeError('Pilot directory already exists; refusing to overwrite')
    cfg = json.loads((SOURCE / 'config.json').read_text())
    cfg.update(theta_bounds=[-21., -12.], regularization_reference_alpha=-21.,
               case_stage='At most five accepted optimization iterations, then stop for user review',
               optimizer=dict(method='L-BFGS-B', maxiter=5, ftol=1e-7, gtol=1e-8, maxls=20),
               finite_difference_checks='Skipped per prior explicit user instruction')
    assert cfg['background_alpha'] == -21. and cfg['transverse_permeability_m2'] == 1e-21
    RUN.mkdir()
    for name in ['forward_and_adjoint.i', 'optimize.i', 'measurement_data.csv',
                 'solver_times.csv', 'initial_alpha.txt']:
        shutil.copy2(SOURCE / name, RUN / name)
    write_json(RUN / 'config.json', cfg)
    names = ['forward_and_adjoint.i', 'optimize.i', 'measurement_data.csv',
             'solver_times.csv', 'initial_alpha.txt', 'config.json']
    write_json(RUN / 'preflight.json', dict(
        initial_source=str(SOURCE), max_accepted_iterations=5,
        input_sha256={n: hashlib.sha256((RUN/n).read_bytes()).hexdigest() for n in names}))
    print('Prepared five-iteration pilot: lower bound/reference -21; beta and delta unchanged.', flush=True)


class Pilot(Inversion):
    def __init__(self):
        super().__init__(RUN, 20)
        self.artifacts = {}
        (RUN / 'evaluations').mkdir()
        (RUN / 'iterations').mkdir()

    def archive(self, theta, report, value, gradient, data_objective, purpose):
        folder = RUN / 'evaluations' / f'{self.count:04d}'
        folder.mkdir()
        shutil.copy2(report, folder / 'strain.csv')
        np.savetxt(folder / 'gradient_scaled.txt', gradient)
        reg, _ = regularization(theta, self.cfg)
        write_json(folder / 'evaluation.json', dict(
            evaluation=self.count, purpose=purpose, theta=np.asarray(theta).tolist(),
            data_objective=float(data_objective), regularization=float(reg),
            scaled_objective=float(value), gradient_scaled=np.asarray(gradient).tolist()))
        self.artifacts[tuple(theta)] = folder

    def seed_initial(self):
        # Identical PDE/data/initial alpha; only the external regularization and
        # admissible optimization interval differ from the completed source.
        for name in ['forward_and_adjoint.i', 'optimize.i', 'measurement_data.csv', 'initial_alpha.txt']:
            assert (RUN/name).read_bytes() == (SOURCE/name).read_bytes(), name
        np.testing.assert_array_equal(self.expand(self.theta0), np.loadtxt(RUN / 'initial_alpha.txt'))
        source_hist = pd.read_csv(SOURCE / 'objective_history.csv')
        assert len(source_hist) == 1 and source_hist.iloc[0].purpose == 'initial'
        np.testing.assert_array_equal([source_hist.iloc[0][f'theta_{i+1}'] for i in range(3)], self.theta0)
        gf = pd.read_csv(SOURCE / 'solver_output/evaluate_out_OptimizationReporter_0001.csv')
        full = gf[[f'grad_perm_{i+1}' for i in range(200)]].iloc[-1].to_numpy()
        data_gradient = np.array([full[idx].sum() for idx in self.indices])
        data_obj = float(source_hist.iloc[0].data_objective)
        report = SOURCE / 'strain_initial.csv'
        frame = pd.read_csv(report)
        np.testing.assert_allclose(.5*np.sum(frame.misfit_values**2), data_obj, rtol=1e-8, atol=1e-22)
        reg, reg_grad = regularization(self.theta0, self.cfg)
        value = (data_obj+reg)*self.cfg['objective_scale']
        gradient = (data_gradient+reg_grad)*self.cfg['objective_scale']
        assert np.isfinite(value) and np.isfinite(gradient).all()
        self.count = 1
        self.cache[tuple(self.theta0)] = value, gradient
        self.best = value
        row = dict(evaluation=1, purpose='initial_reused_data_new_L1_reference',
                   data_objective=data_obj, regularization=reg, scaled_objective=value,
                   gradient_norm=float(np.linalg.norm(gradient)), wall_seconds=0.)
        row.update({f'theta_{i+1}':v for i,v in enumerate(self.theta0)})
        pd.DataFrame([row]).to_csv(RUN / 'objective_history.csv', index=False)
        self.archive(self.theta0, report, value, gradient, data_obj, row['purpose'])
        shutil.copy2(report, RUN / 'strain_initial.csv')
        np.savetxt(RUN / 'initial_gradient.txt', gradient)
        np.savetxt(RUN / 'initial_data_gradient_unscaled.txt', data_gradient)
        np.savetxt(RUN / 'best_theta.txt', self.theta0)
        write_json(RUN / 'initial_reuse.json', dict(
            source=str(SOURCE), source_regularization_reference=-18., new_reference=-21.,
            data_gradient=data_gradient.tolist(), new_scaled_gradient=gradient.tolist(),
            data_objective=data_obj, new_regularization=float(reg), new_scaled_objective=float(value)))
        print('INITIAL_DATA_AND_ADJOINT_REUSED; L1 recomputed at reference -21; no finite differences.', flush=True)

    def evaluate(self, theta, purpose='optimize', force=False):
        key = tuple(theta)
        if key in self.cache and not force:
            return self.cache[key]
        value, gradient = super().evaluate(theta, purpose, force)
        report = self.find_report()
        reg, _ = regularization(theta, self.cfg)
        self.archive(theta, report, value, gradient, value/self.cfg['objective_scale']-reg, purpose)
        return value, gradient

    def record_iteration(self, theta, iteration):
        folder = self.artifacts[tuple(theta)]
        evaluation = json.loads((folder / 'evaluation.json').read_text())
        target = RUN / 'iterations' / f'{iteration:02d}'
        shutil.copytree(folder, target)
        frame = pd.read_csv(target / 'strain.csv')
        active = frame.measurement_time > 0
        record = dict(iteration=iteration, evaluation=evaluation['evaluation'],
                      scaled_objective=evaluation['scaled_objective'],
                      data_objective=evaluation['data_objective'],
                      regularization=evaluation['regularization'],
                      residual_rms_microstrain=float(np.sqrt(np.mean(frame.loc[active, 'misfit_values']**2))*1e6))
        record.update({f'theta_{i+1}':float(v) for i,v in enumerate(theta)})
        pd.DataFrame([record]).to_csv(RUN / 'accepted_iteration_history.csv', index=False,
                                     mode='w' if iteration == 0 else 'a', header=iteration == 0)
        np.savetxt(RUN / 'checkpoint_theta.txt', theta)
        np.savetxt(target / 'alpha.txt', self.expand(theta))
        plot_qc(RUN, target / 'strain.csv', theta, f'iter_{iteration:02d}')
        print('ACCEPTED_ITERATION '+json.dumps(record), flush=True)

    def run_pilot(self):
        self.seed_initial()
        self.record_iteration(self.theta0, 0)
        accepted = 0

        def callback(theta):
            nonlocal accepted
            accepted += 1
            assert accepted <= 5, 'Exceeded requested pilot limit'
            self.record_iteration(theta, accepted)

        result = minimize(self.evaluate, self.theta0, method='L-BFGS-B', jac=True,
                          bounds=[tuple(self.cfg['theta_bounds'])]*3, callback=callback,
                          options=dict(maxiter=5, ftol=1e-7, gtol=1e-8, maxls=20))
        limited = int(result.status) == 1 and int(result.nit) == 5
        write_json(RUN / 'optimizer_result.json', dict(
            success=bool(result.success), status=int(result.status), message=str(result.message),
            iterations=int(result.nit), evaluations=int(result.nfev), theta=result.x.tolist(),
            objective=float(result.fun), stopped_at_user_limit=limited,
            termination='user_iteration_limit' if limited else 'converged_early' if result.success else 'optimizer_failure'))
        # The matching archived observation report is already complete. No extra
        # final forward/adjoint evaluation is needed to display the pilot result.
        folder = self.artifacts[tuple(result.x)]
        shutil.copy2(folder / 'strain.csv', RUN / 'strain_after_pilot.csv')
        shutil.copy2(folder / 'gradient_scaled.txt', RUN / 'gradient_after_pilot.txt')
        plot_qc(RUN, RUN / 'strain_after_pilot.csv', result.x, 'after_pilot')
        np.savetxt(RUN / 'theta_after_pilot.txt', result.x)
        np.savetxt(RUN / 'alpha_after_pilot.txt', self.expand(result.x))
        active_values = iter(result.x)
        table = []
        for zone in self.cfg['zones']:
            alpha = next(active_values) if zone['optimized'] else zone['alpha']
            table.append(dict(lo_ft=zone['lo_ft'], hi_ft=zone['hi_ft'], optimized=zone['optimized'],
                              initial_alpha=zone['alpha'], alpha_after_pilot=alpha, kxx_m2=10.**alpha))
        pd.DataFrame(table).to_csv(RUN / 'permeability_after_pilot.csv', index=False)
        if not result.success and not limited:
            raise RuntimeError(f'Pilot stopped unexpectedly: {result.message}')
        (RUN / 'PILOT_COMPLETE').write_text(f'{accepted} accepted iterations; {result.message}\n')
        (RUN / 'PAUSED_FOR_REVIEW').write_text('Do not continue beyond this pilot until the user decides.\n')
        print('PILOT_COMPLETE; awaiting user review; no automatic continuation.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    if (RUN / 'RUNNING').exists() or (RUN / 'objective_history.csv').exists():
        raise RuntimeError('Pilot already started; refusing to duplicate or overwrite')
    audit = json.loads((RUN / 'preflight.json').read_text())
    for name, digest in audit['input_sha256'].items():
        assert hashlib.sha256((RUN/name).read_bytes()).hexdigest() == digest, name
    (RUN / 'RUNNING').write_text(f'pid={os.getpid()}\n')
    try:
        while not (SOURCE / 'INITIAL_COMPLETE').exists() or (SOURCE / 'RUNNING').exists():
            if (SOURCE / 'FAILED').exists():
                raise RuntimeError('Initial background-21 solve failed; inspect before continuing')
            print('WAITING_FOR_INITIAL_BACKGROUND21_SOLVE', flush=True)
            time.sleep(30)
        Pilot().run_pilot()
    except Exception as exc:
        (RUN / 'FAILED').write_text(f'{type(exc).__name__}: {exc}\n')
        raise
    finally:
        (RUN / 'RUNNING').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
