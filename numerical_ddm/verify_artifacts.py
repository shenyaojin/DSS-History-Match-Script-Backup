"""Check saved numerical arrays against actual MOOSE CSV and frozen sources."""
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/numerical_ddm'
sys.path.insert(0, str(ROOT / 'fibeRIS/src'))
from fiberis.moose.numerical_ddm import rotation as ddm_rotation


def main():
    report = {'verified_runs': [], 'skipped_runs': [], 'immutable_reference_files': 0}
    for manifest in sorted(OUT.rglob('run.json')):
        folder = manifest.parent
        if 'initial_binary_runs' in folder.parts:
            continue
        run = json.loads(manifest.read_text())
        if not run.get('success'):
            report['skipped_runs'].append(str(folder.relative_to(OUT)))
            continue
        frame_path = folder / 'global_frame.json'
        static, history = folder / 'displacement.npz', folder / 'displacement_history.npz'
        if not static.exists() and not history.exists():
            raise RuntimeError(f'Success without numerical displacement: {folder}')
        data = np.load(history if history.exists() else static)
        if frame_path.exists():
            frame = json.loads(frame_path.read_text())
            rotation = np.array(frame['global_to_local_rotation'])
        else:
            model = json.loads((folder / 'model.json').read_text())
            rotation = np.eye(3)
            origin = np.zeros(3)
            # These early runs predate frame-file logging. Recover only the
            # explicitly prescribed transforms from their saved case definitions,
            # and independently check their receiver-coordinate mapping below.
            if folder.name == 'rotated_translated':
                rotation, origin = ddm_rotation(37, 63), np.array([210, -130, 80])
            elif folder.parent.name == 'notebook_existing_tensile_shear':
                physical = json.loads((folder.parent / 'reference_provenance.json').read_text())['physical']
                rotation = ddm_rotation(physical['strike_deg'], physical['dip_deg'], legacy_vertical=True)
                origin = np.array(physical['center_m'])
            if not np.allclose(np.asarray(model['points']) @ rotation + origin, data['points'], rtol=0, atol=1e-8):
                report['skipped_runs'].append(str(folder.relative_to(OUT)) + ': old frame metadata absent')
                continue
        files = sorted(p for p in folder.glob('solution_receivers_*.csv')
                       if re.fullmatch(r'solution_receivers_\d+\.csv', p.name))
        if not history.exists():
            files = files[-1:]
        numerical = []
        for file in files:
            csv = np.atleast_1d(np.genfromtxt(file, delimiter=',', names=True))
            csv = np.sort(csv, order='id')
            numerical.append(np.column_stack([csv[v] for v in ['ux', 'uy', 'uz']]) @ rotation)
        expected = data['displacement'] if history.exists() else data['displacement'][None]
        np.testing.assert_allclose(numerical, expected, rtol=2e-12, atol=1e-16,
                                   err_msg=str(folder))
        if 'input_sha256' in run:
            assert hashlib.sha256((folder / 'model.i').read_bytes()).hexdigest() == run['input_sha256']
        report['verified_runs'].append(dict(directory=str(folder.relative_to(OUT)), frames=len(files)))
    manifest = OUT / 'notebook_existing_tensile_shear/reference_sha256.json'
    for relative, digest in json.loads(manifest.read_text()).items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
        report['immutable_reference_files'] += 1
    (OUT / 'artifact_checks.json').write_text(json.dumps(report, indent=2))
    print(f"Verified {len(report['verified_runs'])} numerical runs against MOOSE CSV; "
          f"{report['immutable_reference_files']} original reference files unchanged.")
    print('Skipped/incomplete:', report['skipped_runs'])


if __name__ == '__main__':
    main()
