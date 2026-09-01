"""D4 - which gauge series does each manuscript figure actually use?

Reviewer 2's charge is circularity: a dataset used as model input and then shown as
validation. Answering it needs the *realised* gauge identity of every archived run and
every figure script, not the identity the filename or the comment claims.

Three of the legacy scripts pick their driving gauge by position in ``os.listdir``:

    for f in os.listdir(pg_data_folder): gauge_data_all.append(load(f))
    ...
    gauge_idx = 5                       # "gauge 6"
    simulator.set_source(gauge_data_all[gauge_idx])

``os.listdir`` returns directory-hash order on ext4, so the index-to-gauge map is a
property of the filesystem at run time, not of the code. It cannot be recovered by
reading the script, and ``.git`` is empty, so it cannot be recovered from history either.

It CAN be recovered from the archived results. Those runs set a spatially constant
initial condition equal to the driving gauge's first cropped sample
(``initial_snapshot[:] = gauge_data_all[gauge_idx].data[0]``), so the first snapshot of
the packed .npz is a one-number fingerprint of the gauge that drove it. That is what this
script matches, and the match is exact rather than nearest: an exact hit identifies the
gauge, and no hit means the run cannot be attributed and must be re-run.

Read-only. Writes only into output/rev2_20260901/D4/.
"""

import datetime
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(REPO, 'output/rev2_20260901/D4')

sys.path.insert(0, os.path.join(REPO, 'fibeRIS/src'))
from fiberis.analyzer.Data1D import Data1D_Gauge  # noqa: E402


def first_sample(path, t0, t1):
    """First sample of a gauge series over the same crop window the legacy script used."""
    g = Data1D_Gauge.Data1DGauge()
    g.load_npz(path)
    g.crop(t0, t1)
    a = np.asarray(g.data, dtype=float)
    return None if a.size == 0 else float(a[0])


def canonical(npz):
    """Return the field as (n_t, n_x), refusing to guess when the axes are ambiguous."""
    t, d, data = npz['taxis'], npz['daxis'], npz['data']
    if data.shape == (len(t), len(d)) and data.shape == (len(d), len(t)):
        raise ValueError('square field: layout is ambiguous, refusing to guess')
    if data.shape == (len(t), len(d)):
        return data, '(n_t, n_x)'
    if data.shape == (len(d), len(t)):
        return data.T, '(n_x, n_t)'
    raise ValueError(f'field shape {data.shape} matches neither (len taxis {len(t)}, '
                     f'len daxis {len(d)}) nor its transpose')


def attribute(run_npz, gauge_dir, t0, t1, tol=1e-6):
    """Identify the driving gauge of an archived run from its initial condition."""
    z = np.load(run_npz, allow_pickle=True)
    field, layout = canonical(z)
    snap0 = field[0]
    constant = bool(np.allclose(snap0, snap0[0]))
    ic = float(snap0[0])
    cands = []
    for name in sorted(os.listdir(gauge_dir)):
        if not name.endswith('.npz'):
            continue
        v = first_sample(os.path.join(gauge_dir, name), t0, t1)
        cands.append({'file': name, 'first_sample_psi': v,
                      'delta': None if v is None else abs(v - ic)})
    exact = [c for c in cands if c['delta'] is not None and c['delta'] <= tol]
    return {
        'run': os.path.relpath(run_npz, REPO),
        'layout': layout,
        'n_time': int(field.shape[0]), 'n_depth': int(field.shape[1]),
        'md_range_ft': [float(z['daxis'].min()), float(z['daxis'].max())],
        'initial_condition_psi': ic,
        'initial_snapshot_is_constant': constant,
        'attributed_to': exact[0]['file'] if len(exact) == 1 else None,
        'n_exact_matches': len(exact),
        'listdir_order_today': os.listdir(gauge_dir),
        'candidates': sorted(
            [c for c in cands if c['delta'] is not None], key=lambda c: c['delta'])[:5],
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    os.chdir(REPO)
    prod_window = (datetime.datetime(2020, 4, 1), datetime.datetime(2021, 7, 1))
    report = {'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'attributions': [], 'figure_scripts': {}}

    # 102r drives from s_well gauges; 103r drives from the PRODUCER gauges (line 18).
    for run, gdir in [
            ('output/0224_forward_modeling_mix/0.1_gauge6.npz',
             'data/fiberis_format/s_well/gauges'),
            ('output/0324_forward_simulator/1.0_gauge3.npz',
             'data/fiberis_format/prod/gauges'),
            ('output/0324_forward_simulator/1e-05_gauge3.npz',
             'data/fiberis_format/prod/gauges')]:
        if os.path.exists(run):
            report['attributions'].append(attribute(run, gdir, *prod_window))

    # Fig. 5 / Fig. 6 gauge selection, re-run rather than read off the source.
    fh = 'data/legacy/s_well/geometry/frac_hit/'
    s7 = np.load(fh + 'frac_hit_stage_7_swell.npz')['data']
    s8 = np.load(fh + 'frac_hit_stage_8_swell.npz')['data']
    gmd = np.load('data/legacy/s_well/geometry/gauge_md_swell.npz')['data']
    lo, hi = float(np.min(s8) - 500), float(np.max(s7) + 500)
    ind = np.where((gmd <= hi) & (gmd >= lo))[0]
    report['figure_scripts']['gauge_selection'] = {
        'rule': 'min(frac_hit_stg8) - 500 <= MD <= max(frac_hit_stg7) + 500',
        'window_md_ft': [lo, hi],
        'selected_indices': ind.tolist(),
        'selected_gauge_numbers': (ind + 1).tolist(),
        'selected_md_ft': gmd[ind].tolist(),
        'selected_gauge_num_3_is_gauge': int((ind + 1)[3]),
        'selected_gauge_num_3_md_ft': float(gmd[ind][3]),
        'shared_by': ['DAS_history_matching_visualization/102_IMAGE25abstract.py:62',
                      'DAS_history_matching_visualization/102r_IMAGE25abstract_with_scalar.py:63',
                      'DAS_history_matching_visualization/104_full_history_matching_manuscript.py:62'],
        'note_101': ('101_fiberis_matching.py:72 uses gauge_md[4:10] = gauges 5-10, one more '
                     'than the five the figure scripts select.'),
    }

    # Fig. 7b legend: which file each hard-coded label lands on.
    d = 'output/0324_forward_simulator/'
    if os.path.isdir(d):
        names = os.listdir(d)
        labels = ['Field data',
                  'Simulated drawdown (min D = 1 x baseline)',
                  'Simulated drawdown (min D = 1e-5 x baseline)']
        # pressure_drop_down[0] is field; entry j>=1 comes from dataframe_full[:-1][j-1].
        used = [names[i - 1] if i >= 1 else 'FIELD' for i in (0, 2, 4)]
        report['figure_scripts']['fig7b_legend'] = {
            'script': 'scripts/sponsor_meeting_report_2025/production_sim/'
                      '103p_forward_modeling_viz_without_scalar.py',
            'listdir_order_today': names,
            'plotted_indices': [0, 2, 4],
            'label_to_actual_file': dict(zip(labels, used)),
            'silently_dropped_by_slice': names[-1],
        }

    with open(os.path.join(OUT, 'd4_provenance.json'), 'w') as fh_out:
        json.dump(report, fh_out, indent=2)
    print(json.dumps(report, indent=2))
    print('\nwrote', os.path.join(OUT, 'd4_provenance.json'))


if __name__ == '__main__':
    main()
