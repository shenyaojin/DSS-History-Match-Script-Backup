"""Independent domain and thickness checks for the notebook crossing well.

Uses the same immutable notebook model, all 300 points, and original 0930
reference. Only the final static configuration is needed for these checks.
"""
import json
from dataclasses import replace
import numpy as np
from reproduce_notebook import OUT, original, NumericalConfig, from_ddmpy, metrics


def main():
    fracture, points, times, md, ref_u, ref_strain, ref_rate = original()
    baseline = OUT / 'pilot_h20_t0.2_L2400_refined'
    cfg = NumericalConfig(**json.loads((baseline / 'model.json').read_text())['config'])
    report = {}
    for name, changes in [('domain_L4800', dict(half_extent=4800)),
                          ('thickness_0p1', dict(band_thickness=.1))]:
        directory = OUT / name
        numerical = from_ddmpy(fracture.fractures[-1], replace(
            cfg, output_directory=str(directory), reuse_constant_jacobian=False, **changes))
        u = numerical.sample(points)
        well = numerical.monitor_wells[0]
        well.u1, well.u2, well.u3 = u.T
        well.calculate_strain()
        report[name] = dict(displacement=metrics(u, ref_u[-1]),
                            strain=metrics(well.strain, ref_strain[:, -1]))
        np.savez(directory / 'comparison.npz', points=points, md=md, u=u,
                 reference_u=ref_u[-1], strain=well.strain,
                 reference_strain=ref_strain[:, -1])
        (OUT / 'diagnostics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
