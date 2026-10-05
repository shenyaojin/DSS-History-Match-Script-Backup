"""Actual cached/uncached FE comparison, including load reversal and unequal steps."""
import json
from pathlib import Path
import numpy as np
from run_validation import ROOT, metrics
from fiberis.moose.numerical_ddm import NumericalConfig, Element, FractureZ


def main():
    out = ROOT / 'output/numerical_ddm/constant_jacobian_check'
    points = np.column_stack([np.linspace(-100, 100, 51), np.full(51, 17.), np.full(51, 40.)])
    times = np.array([0., 1., 2., 4.])
    factors = np.array([0., 1., -.7, .2])
    outputs = []
    for name,cached,recycle in [('uncached',False,False),('cached',True,False),('recycling',True,True)]:
        cfg = NumericalConfig(
            str(ROOT / 'moose_env/moose/modules/combined/combined-opt'),
            str(out / name),
            mesh_size=40, core_extent=80, half_extent=400,
            processors=8, mpiexec=str(Path.home() / 'miniforge/envs/moose/bin/mpiexec'),
            reuse_constant_jacobian=cached,recycle_krylov_guess=recycle)
        fracture = FractureZ().set_fracture_elements(Element(100, 60, .001, .0004, -.0006))
        outputs.append(fracture.sample_history(points, times, factors, cfg))
    result = dict(cached_vs_uncached=metrics(outputs[1], outputs[0]),
                  recycling_vs_uncached=metrics(outputs[2],outputs[0]),
                  times=times.tolist(), factors=factors.tolist(),
                  actual_nonzero_solves_per_method=3)
    np.savez(out / 'comparison.npz', uncached=outputs[0], cached=outputs[1],recycling=outputs[2],
             points=points, times=times, factors=factors)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    if result['cached_vs_uncached']['relative_l2'] > 1e-7:
        raise RuntimeError('Cached and independently assembled FE histories disagree')
    if result['recycling_vs_uncached']['relative_l2'] > 1e-7:
        raise RuntimeError('Recycled and independently assembled FE histories disagree')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
