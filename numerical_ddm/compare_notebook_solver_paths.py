"""Compare full recycled history with independently iterated large-mesh frames."""
import json
import re
import numpy as np
from run_validation import ROOT, metrics


def main():
    out = ROOT / 'output/numerical_ddm/notebook_existing_tensile_shear'
    full = out / 'full_h20_t0.2_L2400_refined_recycled'
    new = np.load(full / 'comparison.npz')
    pilot = np.load(out / 'pilot_h20_t0.2_L2400_refined/comparison.npz')
    indices = [int(np.argmin(abs(new['times_seconds'] - t))) for t in pilot['times_seconds']]
    np.testing.assert_allclose(new['times_seconds'][indices], pilot['times_seconds'], atol=1e-9)
    result = dict(versus_independent_pilot=dict(
        displacement=metrics(new['u'][indices], pilot['u']),
        strain=metrics(new['strain'][:, indices], pilot['strain'])))
    cold = out / 'full_h20_t0.2_L2400_refined'
    frame = json.loads((full / 'global_frame.json').read_text())
    rotation = np.array(frame['global_to_local_rotation'])
    files = sorted(p for p in cold.glob('solution_receivers_*.csv')
                   if re.fullmatch(r'solution_receivers_\d+\.csv', p.name))
    values = []
    for file in files:
        csv = np.sort(np.genfromtxt(file, delimiter=',', names=True), order='id')
        values.append(np.column_stack([csv[v] for v in ['ux', 'uy', 'uz']]) @ rotation)
    if values:
        result['versus_completed_cold_frames'] = metrics(new['u'][:len(values)], np.array(values))
        result['cold_completed_frame_count_including_zero'] = len(values)
    (out / 'solver_path_comparison.json').write_text(json.dumps(result, indent=2))
    for quantity in ['displacement', 'strain']:
        if result['versus_independent_pilot'][quantity]['relative_l2'] > 1e-6:
            raise RuntimeError('Recycling changes the converged notebook response')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
