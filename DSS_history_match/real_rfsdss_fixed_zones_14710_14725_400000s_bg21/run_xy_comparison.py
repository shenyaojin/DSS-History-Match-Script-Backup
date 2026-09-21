"""Matched initial tensor, fixed/free zone kyy; reuse the field/V6 solver."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from run_inversion import Inversion

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'comparison_xx_yy_60'
YY_NAMES = ['kyy_A', 'kyy_B', 'kyy_C']


def save_json(path, value):
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def components(theta, cfg):
    return np.asarray(theta[:3]), np.asarray(theta[3:] if cfg['solve_kyy'] else cfg['initial_kyy_theta'])


def plot_qc(run, report, theta, label):
    import matplotlib.pyplot as plt
    cfg = json.loads((run/'config.json').read_text())
    df = pd.read_csv(report)
    t = np.sort(df.measurement_time.unique())
    y = np.sort(df.measurement_ycoord.unique())
    md = 14725+(y-50)/.3048
    arrays = [df.pivot(index='measurement_ycoord', columns='measurement_time', values=c)
              .reindex(index=y, columns=t).to_numpy()*1e6 for c in ['measurement_values', 'simulation_values']]
    obs, sim = arrays
    vmax = max(np.max(abs(obs)), np.max(abs(sim)), 1e-12)
    fig, axes = plt.subplots(1, 5, figsize=(21, 7), sharey=True, layout='constrained',
                            gridspec_kw={'width_ratios': [2, 2, 2, 1, 1]})
    for ax, values, title in zip(axes[:3], [obs, sim, sim-obs], ['Observation', 'Simulation', 'Residual: sim − obs']):
        im = ax.pcolormesh(t, md, values, shading='nearest', cmap='bwr', vmin=-vmax, vmax=vmax, rasterized=True)
        ax.axvspan(t[-1], 400000, facecolor='.9', hatch='///', edgecolor='.6')
        ax.set(title=title, xlabel='Time [s]', xlim=(0, 400000), ylim=(14725, 14710))
    fig.colorbar(im, ax=axes[:3], label='Strain [microstrain]', shrink=.65)
    axes[0].set_ylabel('Measured depth [ft]')
    xx, yy = components(theta, cfg)
    active = [z for z in cfg['zones'] if z['optimized']]
    edges = sorted({14710., 14725., *[max(14710, min(14725, z[k])) for z in active for k in ['lo_ft', 'hi_ft']]})
    for ax, values, initial, title in zip(axes[3:], [xx, yy],
                                         [cfg['initial_kxx_theta'], cfg['initial_kyy_theta']],
                                         ['kxx: optimized', 'kyy: optimized' if cfg['solve_kyy'] else 'kyy: fixed']):
        for profile, style, name in [(initial, '--', 'Initial'), (values, '-', 'Current')]:
            xp, yp = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mid = (lo+hi)/2
                alpha = next((profile[i] for i, z in enumerate(active) if z['lo_ft'] <= mid < z['hi_ft']), -21.)
                xp.extend([alpha, alpha]); yp.extend([lo, hi])
            ax.plot(xp, yp, style, label=name)
        ax.set(title=title, xlabel='log10(k / m²)', xlim=(-21.5, -11.5))
        ax.legend(fontsize=8)
    fig.suptitle(f'S-well | {run.name} | {label} | shared initial xx = yy; background = 1e-21 m²')
    fig.savefig(run/f'qc_{label}.png', dpi=150)
    plt.close(fig)


class Comparison(Inversion):
    def __init__(self, run):
        super().__init__(run, 20)
        self.artifacts = {}

    def expand(self, theta):
        xx, yy = components(theta, self.cfg)
        return np.r_[super().expand(xx), yy]

    def regularization(self, theta):
        d = np.asarray(theta)-self.cfg['regularization_reference_alpha']
        delta = self.cfg['regularization_delta']
        denom = np.sqrt(d*d+delta*delta)
        beta = self.cfg['beta_l1']
        return beta*np.sum(denom-delta), beta*d/denom

    def evaluate(self, theta, purpose='optimize', force=False):
        key = tuple(theta)
        if key in self.cache and not force:
            return self.cache[key]
        self.count += 1
        print(f'EVALUATION {self.count} purpose={purpose} theta={theta}', flush=True)
        alpha = self.expand(theta)
        text, count = re.subn(r"(initial_condition\s*=\s*')[^']*(')",
                             lambda m: m[1]+'; '.join(f'{v:.16g}' for v in alpha)+m[2], self.master, count=1)
        assert count == 1 and len(alpha) == 203
        text = text.replace('verbose = true', 'verbose = false').replace('console = true', 'console = false')
        path = self.run/'evaluate.i'
        path.write_text(text)
        # Match the validated driver: clear stale CSV only in this case's solver directory.
        for p in self.output.glob('*.csv'):
            p.unlink()
        started = time.monotonic()
        ok, stdout, stderr = self.runner.run(str(path), str(self.output), num_processors=20,
                           log_file_name='simulation.log', stream_output=False, clean_output_dir=False)
        (self.run/'last_solver_stderr.txt').write_text(stderr)
        if not ok:
            raise RuntimeError('MOOSE failed; no objective or gradient will be fabricated')
        obj = float(pd.read_csv(self.output/'evaluate_out.csv')['OptimizationReporter/objective_value'].iloc[-1])
        gf = pd.read_csv(self.output/'evaluate_out_OptimizationReporter_0001.csv').iloc[-1]
        gx = gf[[f'grad_perm_{i+1}' for i in range(200)]].to_numpy(dtype=float)
        data_g = np.array([gx[idx].sum() for idx in self.indices])
        if self.cfg['solve_kyy']:
            data_g = np.r_[data_g, gf[[f'grad_{n}' for n in YY_NAMES]].to_numpy(dtype=float)]
        reg, reg_g = self.regularization(theta)
        scale = self.cfg['objective_scale']
        value, gradient = (obj+reg)*scale, (data_g+reg_g)*scale
        assert np.isfinite(value) and np.isfinite(gradient).all()
        report = self.find_report()  # Same rectification, t=0 exclusion, data/objective consistency checks.
        folder = self.run/'evaluations'/f'{self.count:04d}'
        folder.mkdir(parents=True)
        shutil.copy2(report, folder/'strain.csv')
        np.savetxt(folder/'gradient_scaled.txt', gradient)
        np.savetxt(folder/'data_gradient_unscaled.txt', data_g)
        evaluation = dict(evaluation=self.count, purpose=purpose, theta=np.asarray(theta).tolist(),
                          data_objective=obj, regularization=float(reg), scaled_objective=float(value),
                          gradient_scaled=gradient.tolist(), data_gradient_unscaled=data_g.tolist(),
                          wall_seconds=time.monotonic()-started)
        save_json(folder/'evaluation.json', evaluation)
        row = {k: evaluation[k] for k in ['evaluation', 'purpose', 'data_objective', 'regularization', 'scaled_objective', 'wall_seconds']}
        row.update(gradient_norm=float(np.linalg.norm(gradient)))
        row.update({name: float(v) for name, v in zip(self.cfg['parameter_order'], theta)})
        hist = self.run/'objective_history.csv'
        pd.DataFrame([row]).to_csv(hist, index=False, mode='a', header=not hist.exists())
        self.artifacts[key] = folder
        self.cache[key] = value, gradient
        print(json.dumps(row), flush=True)
        return value, gradient

    def record(self, theta, iteration):
        source = self.artifacts[tuple(theta)]
        target = self.run/'iterations'/f'{iteration:02d}'
        shutil.copytree(source, target)
        ev = json.loads((source/'evaluation.json').read_text())
        frame = pd.read_csv(source/'strain.csv')
        active = frame.measurement_time > 0
        row = dict(iteration=iteration, evaluation=ev['evaluation'],
                   data_objective=ev['data_objective'], regularization=ev['regularization'],
                   scaled_objective=ev['scaled_objective'],
                   residual_rms_microstrain=float(np.sqrt(np.mean(frame.loc[active, 'misfit_values']**2))*1e6))
        xx, yy = components(theta, self.cfg)
        row.update({f'{direction}_{label}': float(v) for direction, vals in [('xx', xx), ('yy', yy)]
                    for label, v in zip('ABC', vals)})
        hist = self.run/'accepted_iteration_history.csv'
        pd.DataFrame([row]).to_csv(hist, index=False, mode='a', header=not hist.exists())
        np.savetxt(self.run/'checkpoint_theta.txt', theta)
        plot_qc(self.run, source/'strain.csv', theta, f'iter_{iteration:02d}')
        print('ACCEPTED_ITERATION '+json.dumps(row), flush=True)

    def validate_new_yy(self):
        initial = json.loads((self.artifacts[tuple(self.theta0)]/'evaluation.json').read_text())
        checks = []
        # Test the newly added data gradient alone: regularization cannot hide a wiring error.
        for yy in [np.array([-.7, -.9, .6]), np.array([.5, .8, -.6])]:
            direction = np.r_[np.zeros(3), yy/np.linalg.norm(yy)]
            h = .02
            trial = []
            for sign in [1, -1]:
                theta = self.theta0+sign*h*direction
                self.evaluate(theta, purpose='new_yy_gradient_check')
                ev = json.loads((self.artifacts[tuple(theta)]/'evaluation.json').read_text())
                trial.append(ev['data_objective']*self.cfg['objective_scale'])
            fd = (trial[0]-trial[1])/(2*h)
            adj = float(np.asarray(initial['data_gradient_unscaled'])@direction*self.cfg['objective_scale'])
            passed = abs(fd-adj) <= max(1e-7, .05*max(abs(fd), abs(adj)))
            checks.append(dict(direction=direction.tolist(), h=h, fd=fd, adjoint=adj,
                               relative_error=abs(fd-adj)/max(abs(fd), abs(adj), 1e-8), passed=bool(passed)))
            save_json(self.run/'new_yy_gradient_check.json', checks)
            print('NEW_YY_GRADIENT_CHECK '+json.dumps(checks[-1]), flush=True)
            if not passed:
                raise RuntimeError('New kyy gradient check failed; free-yy optimization not started')

    def run_comparison(self):
        self.evaluate(self.theta0, 'initial')
        self.record(self.theta0, 0)
        if self.cfg['solve_kyy']:
            self.validate_new_yy()
        accepted = 0

        def callback(theta):
            nonlocal accepted
            accepted += 1
            assert accepted <= 60
            self.record(theta, accepted)

        result = minimize(self.evaluate, self.theta0, jac=True, method='L-BFGS-B',
                          bounds=[tuple(self.cfg['theta_bounds'])]*len(self.theta0), callback=callback,
                          options=dict(maxiter=60, ftol=1e-7, gtol=1e-8, maxls=20))
        limited = int(result.status) == 1 and int(result.nit) == 60
        save_json(self.run/'optimizer_result.json', dict(
            success=bool(result.success), status=int(result.status), message=str(result.message),
            iterations=int(result.nit), optimizer_evaluations=int(result.nfev),
            total_evaluations=self.count, theta=result.x.tolist(), objective=float(result.fun),
            termination='user_iteration_limit' if limited else 'converged' if result.success else 'optimizer_failure'))
        folder = self.artifacts[tuple(result.x)]
        shutil.copy2(folder/'strain.csv', self.run/'strain_final.csv')
        shutil.copy2(folder/'evaluation.json', self.run/'final_evaluation.json')
        np.savetxt(self.run/'theta_final.txt', result.x)
        plot_qc(self.run, self.run/'strain_final.csv', result.x, 'final')
        xx, yy = components(result.x, self.cfg)
        rows = []
        itx, ity = iter(xx), iter(yy)
        for z in self.cfg['zones']:
            ax = next(itx) if z['optimized'] else z['alpha']
            ay = next(ity) if z['optimized'] else -21.
            rows.append(dict(lo_ft=z['lo_ft'], hi_ft=z['hi_ft'], optimized_xx=z['optimized'],
                             optimized_yy=z['optimized'] and self.cfg['solve_kyy'],
                             alpha_xx=ax, alpha_yy=ay, kxx_m2=10.**ax, kyy_m2=10.**ay, kzz_m2=1e-21))
        pd.DataFrame(rows).to_csv(self.run/'permeability_final.csv', index=False)
        if not result.success and not limited:
            raise RuntimeError(f'Optimizer failed: {result.message}; candidate retained')
        (self.run/'COMPLETE').write_text(f'{accepted} accepted iterations; {result.message}\n')
        print('COMPLETE; no automatic restart', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['fixed_yy', 'free_yy'])
    parser.add_argument('--check-input', action='store_true')
    args = parser.parse_args()
    run = ROOT/args.mode
    audit = json.loads((run/'preflight.json').read_text())
    for name, digest in audit['input_sha256'].items():
        assert hashlib.sha256((run/name).read_bytes()).hexdigest() == digest, name
    if (run/'RUNNING').exists() or (run/'objective_history.csv').exists():
        raise RuntimeError('Run already started; refusing duplicate or overwrite')
    model = Comparison(run)
    np.testing.assert_array_equal(model.expand(model.theta0), np.loadtxt(run/'initial_alpha.txt'))
    if args.check_input:
        ok, stdout, stderr = model.runner.run(str(run/'optimize.i'), str(run/'input_check'),
                            num_processors=20, additional_args=['--check-input'], log_file_name='check.log',
                            stream_output=False, clean_output_dir=False)
        (run/'input_check/stderr.log').write_text(stderr)
        if not ok:
            raise RuntimeError('Input check failed: '+stderr[-4000:])
        (run/'INPUT_CHECK_PASSED').write_text('MOOSE --check-input passed\n')
        return
    if not (run/'INPUT_CHECK_PASSED').exists():
        raise RuntimeError('Run --check-input first')
    (run/'RUNNING').write_text(f'pid={os.getpid()}\n')
    try:
        model.run_comparison()
    except Exception as exc:
        (run/'FAILED').write_text(f'{type(exc).__name__}: {exc}\n')
        raise
    finally:
        (run/'RUNNING').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
