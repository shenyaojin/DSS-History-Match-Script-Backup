"""Prepare an isolated isotropic-background comparison from the completed case."""
from pathlib import Path
import hashlib
import json
import re
import shutil

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'real_rfsdss_fixed_zones_14710_14725_400000s'
RUN = HERE / 'inversion'


def main():
    if RUN.exists():
        raise RuntimeError('Case already prepared; inspect before changing it')
    cfg = json.loads((OLD / 'inversion/config.json').read_text())
    cfg.update(background_alpha=-21.,
               background_change='Background kxx outside the five zones and global background kyy/kzz are 1e-21; zone kxx amplitudes retained',
               background_permeability_tensor_m2=[1e-21, 1e-21, 1e-21],
               transverse_permeability_m2=1e-21,
               case_stage='Initial response comparison; bounds/L1 reference unchanged pending user choice')
    alpha = np.full(200, cfg['background_alpha'])
    for z in cfg['zones']:
        alpha[z['layer_indices_zero_based']] = z['alpha']
    old_alpha = np.loadtxt(OLD / 'inversion/initial_alpha.txt')
    zone_indices = set(i for z in cfg['zones'] for i in z['layer_indices_zero_based'])
    assert np.array_equal(np.flatnonzero(alpha != old_alpha),
                          [i for i in range(200) if i not in zone_indices])
    master = (OLD / 'inversion/optimize.i').read_text()
    master, count = re.subn(r"(initial_condition\s*=\s*')[^']*(')",
                           lambda m: m[1]+'; '.join(f'{v:.16g}' for v in alpha)+m[2], master, count=1)
    assert count == 1
    forward = (OLD / 'inversion/forward_and_adjoint.i').read_text()
    # The parameter reporter defaults also reflect this case. MultiApp transfers
    # supply the same expanded vector at runtime.
    forward, count = re.subn(r"(  \[params\].*?real_vector_values = ')[^']*(')",
                            lambda m: m[1]+'; '.join(f'{v:.16g}' for v in alpha)+m[2],
                            forward, count=1, flags=re.S)
    assert count == 1
    for name in ['func_kyy', 'func_zero']:
        pattern = r"(  \["+name+r"\].*?expression = ')1e-18(')"
        forward, count = re.subn(pattern, lambda m: m[1]+'1e-21'+m[2],
                                forward, count=1, flags=re.S)
        assert count == 1, name
    source = (OLD / 'run_inversion.py').read_text()
    source = source.replace("z['hi_ft']),-18.)", "z['hi_ft']),cfg['background_alpha'])")
    source = source.replace("xlim=(-18.5,-11.5)", "xlim=(min(cfg['background_alpha'],cfg['theta_bounds'][0])-.5,cfg['theta_bounds'][1]+.5)")
    source = source.replace('deviation=np.asarray(theta)+18.;denom=np.sqrt(deviation**2+.05**2)',
                            "deviation=np.asarray(theta)-self.cfg['regularization_reference_alpha'];delta=self.cfg['regularization_delta'];denom=np.sqrt(deviation**2+delta**2)")
    source = source.replace('reg=beta*np.sum(denom-.05)', 'reg=beta*np.sum(denom-delta)')
    source = source.replace("                reused=pd.read_csv(self.run/'strain_initial_reused.csv')\n                current=pd.read_csv(report)\n                np.testing.assert_allclose(current.simulation_values,reused.simulation_values,rtol=1e-5,atol=1e-12)\n",
                            '                # Changed background: compute a new initial response.\n')
    source = source.replace('np.sqrt((np.asarray(theta)+18.)**2+.05**2)-.05',
                            "np.sqrt((np.asarray(theta)-self.cfg['regularization_reference_alpha'])**2+self.cfg['regularization_delta']**2)-self.cfg['regularization_delta']")
    source = source.replace('using saved initial objective and adjoint gradient',
                            'using the current case initial objective and adjoint gradient')
    source = source[:source.index('def main():')] + '''def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-input',action='store_true')
    args=parser.parse_args();run=HERE/'inversion'
    if (run/'RUNNING').exists() or (not args.check_input and (run/'objective_history.csv').exists()):
        raise RuntimeError('Existing run: refusing to overwrite')
    import hashlib
    audit=json.loads((run/'preflight.json').read_text())
    for name,digest in audit['input_sha256'].items():
        assert hashlib.sha256((run/name).read_bytes()).hexdigest()==digest,name
    model=Inversion(run,20)
    if args.check_input:
        ok,stdout,stderr=model.runner.run(str(run/'optimize.i'),str(run/'input_check'),num_processors=20,
                        additional_args=['--check-input'],log_file_name='check.log',stream_output=False,clean_output_dir=False)
        (run/'input_check/stderr.log').write_text(stderr)
        if not ok:raise RuntimeError(stderr[-5000:])
        return
    (run/'RUNNING').write_text(f'pid={os.getpid()}\\n')
    try:
        model.evaluate(model.theta0,'initial')
        (run/'INITIAL_COMPLETE').write_text('New-background initial forward/adjoint complete; no optimization.\\n')
    except Exception as exc:
        (run/'FAILED').write_text(f'{type(exc).__name__}: {exc}\\n')
        raise
    finally:
        (run/'RUNNING').unlink(missing_ok=True)

if __name__=='__main__':main()
'''
    compile(source, str(HERE / 'run_inversion.py'), 'exec')
    RUN.mkdir()
    (HERE / 'run_inversion.py').write_text(source)
    (RUN / 'optimize.i').write_text(master)
    (RUN / 'forward_and_adjoint.i').write_text(forward)
    (RUN / 'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    np.savetxt(RUN / 'initial_alpha.txt', alpha)
    for name in ['measurement_data.csv', 'solver_times.csv']:
        shutil.copy2(OLD / 'inversion' / name, RUN / name)
    audit = dict(background_alpha=-21., changed_layer_count=int(np.count_nonzero(alpha != old_alpha)),
                 transverse_permeability_m2=1e-21, initial_theta=cfg['initial_theta'],
                 theta_bounds=cfg['theta_bounds'], regularization_reference_alpha=cfg['regularization_reference_alpha'],
                 beta_l1=cfg['beta_l1'], n_solver_steps=129, max_solver_steps=240,
                 scope='Initial response only; no finite differences and no optimization launched',
                 source_case=str(OLD))
    audit['input_sha256'] = {name: hashlib.sha256((RUN/name).read_bytes()).hexdigest()
                             for name in ['forward_and_adjoint.i', 'optimize.i', 'measurement_data.csv', 'config.json']}
    (RUN / 'preflight.json').write_text(json.dumps(audit, indent=2)+'\n')
    (HERE / 'README.md').write_text('''# Isotropic background comparison: alpha = -21

Background kxx outside the five predefined zones and global kyy/kzz are reduced
from 1e-18 to 1e-21 m², per the user's clarified instruction. Zone anomalies
remain in kxx only. All five zone kxx initial values, data, mesh, pressure source,
mechanics, 129-step schedule, 240-step cap, 0.001 tolerances, and 20 MPI ranks
are retained. The original completed case is unchanged.

This launcher computes the new initial forward/adjoint response only, so its
data gradient is available for a subsequent inversion. It does not run finite
differences or optimize. The earlier background-18 response/gradient is not
reused. Until the user chooses otherwise, bounds remain [-18,-12], the L1
reference remains -18, delta remains 0.05, and beta remains
5.049311379654719e-15. Background kxx and the L1 reference are separate settings.

The new driver reads the regularization reference and delta from configuration
instead of the old hard-coded -18/.05. This preserves the current formula while
allowing a separately requested reference change. QC uses the actual background.

The earlier kxx-only comparison was explicitly superseded and cancelled. Its
inputs and partial outputs are preserved in `cancelled_kxx_only/`.
''')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
