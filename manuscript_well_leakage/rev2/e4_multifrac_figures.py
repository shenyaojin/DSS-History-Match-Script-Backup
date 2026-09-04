"""E4 aggregate manifest for the figure/summary products.

`e4_multifrac.py --stage figures` writes the five figures and `e4_summary_v1.json`
but no manifest: it runs no solver, so house rule 3 does not bite. This file
pins those products anyway, the way B2 and E2 do, so that every file in
`output/rev2_20260901/E4/` is hash-recorded and `rev2_manifest.verify` has
something to check them against.

It is a SEPARATE file from the runner on purpose. `rev2_manifest.code_closure`
hashes every repo `.py` in `sys.modules`, so editing the runner after its 126
run manifests were written would make all 126 report code drift — which is
exactly what happened to the superseded first pass in `runs/` and is recorded
in the README.

Run (from the repo root, after `--stage figures`):

    python3 scripts/manuscript_well_leakage/rev2/e4_multifrac_figures.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rev2_data as rd        # noqa: E402
import rev2_manifest as rm    # noqa: E402

OUTDIR = os.path.join('output', rm.ROUND_TAG, 'E4')
CONFIG = os.path.join('configs', 'rev2', 'e4_multifrac.json')
REFERENCE_RUN = ('runs_v2/frachit_all_at_source_r1em05_pad05000_D140_'
                 'prod_g1_dt60/manifest.json')
# Versioned per house rule 9. v1 was written before two count corrections to
# README.md, so its declared hash for that file no longer matches; it is kept
# (house rule 2), superseded, and declared as an output here.
MANIFEST_NAME = 'manifest_figures_v2.json'

FIGURES = [
    ('fig01_e4_dss_picks_v1.png',
     'the DSS net-tensile metric, the four k-sigma thresholds, the picks, the '
     'catalogued frac hits and the gauge positions'),
    ('fig02_e4_drawdown_profile_v1.png',
     'THE deliverable: the adopted scenario (hypothesis B) against the two '
     'Fig. 7b counter-examples, full range and magnified'),
    ('fig03_e4_selection_spread_v1.png',
     'sensitivity to the fracture selection and to the barrier reduction ratio'),
    ('fig04_e4_padding_v1.png',
     'the domain-robustness test: level, relative change and shape vs pad'),
    ('fig05_e4_level_v1.png',
     'what sets the predicted level: driver gauge identity and annulus '
     'diffusivity'),
]
JSONS = [
    ('e4_summary_v1.json', 'headline runs with their per-gauge tables, and '
                           'every sensitivity spread'),
    ('e4_results_v2.json', 'the authoritative run table: 96 runs'),
    ('e4_results_ext_v1.json', 'the companion sweep: 30 runs (low D, '
                               'E2-converged domain)'),
    ('dss_picks_v1.json', 'every DSS-derived fracture selection, its rule, its '
                          'channels and their amplitudes'),
    ('e4_results_v1.json', 'SUPERSEDED first-pass run table (runs/); kept per '
                           'house rule 2, not quotable -- see README section 8'),
    ('e4_results_smoke.json', 'SUPERSEDED single-run smoke test of the first '
                              'pass; kept per house rule 2, not quotable'),
    ('manifest_figures.json', 'SUPERSEDED v1 of this aggregate manifest; its '
                              'README.md hash pre-dates two count corrections'),
    ('manifest_figures.json.sha256', "v1's sidecar"),
]
REPORTS = [('README.md', 'the task report')]


def main():
    ref_path = os.path.join(OUTDIR, REFERENCE_RUN)
    with open(ref_path) as fh:
        ref = json.load(fh)
    with open(CONFIG) as fh:
        cfg = json.load(fh)

    sp = dict(ref['source_protocol'])
    sp[rm._BUILDER] = 'source_protocol'
    num = dict(ref['numerics'])
    num[rm._BUILDER] = 'numerics'

    mpath = os.path.join(OUTDIR, MANIFEST_NAME)
    rm.assert_absent([mpath])
    with rm.RunRecorder(mpath, study_id='E4_multifrac_hypothesisB:figures',
                        task_id='E4', config=cfg, config_path=CONFIG,
                        run_label='e4_figures_v1',
                        require_modules=('rev2_manifest',)) as rec:
        rec.declare_inputs(
            [(rd.repo_path(OUTDIR, 'runs_v2', d, 'manifest.json'),
              'prior_run_output', d)
             for d in sorted(os.listdir(os.path.join(OUTDIR, 'runs_v2')))
             if os.path.isdir(os.path.join(OUTDIR, 'runs_v2', d))]
            + [(rd.repo_path(OUTDIR, 'runs_ext', d, 'manifest.json'),
                'prior_run_output', d)
               for d in sorted(os.listdir(os.path.join(OUTDIR, 'runs_ext')))
               if os.path.isdir(os.path.join(OUTDIR, 'runs_ext', d))])
        for name, note in FIGURES:
            rec.declare_output(os.path.join(OUTDIR, name), role='figure_png',
                               dpi=300, note=note)
        for name, note in JSONS:
            rec.declare_output(os.path.join(OUTDIR, name), role='json',
                               note=note)
        for name, note in REPORTS:
            rec.declare_output(os.path.join(OUTDIR, name), role='report_md',
                               note=note)
        rec.set_source(sp)
        rec.set_numerics(num)
        rec.set_results({
            'kind': 'aggregate figure manifest -- no solver ran here',
            'figures': [f for f, _ in FIGURES],
            'tables': [j for j, _ in JSONS],
            'n_runs_declared_as_inputs':
                len(os.listdir(os.path.join(OUTDIR, 'runs_v2')))
                + len(os.listdir(os.path.join(OUTDIR, 'runs_ext'))),
            'produced_by': ('python3 scripts/manuscript_well_leakage/rev2/'
                            'e4_multifrac.py --stage figures '
                            '--results-name e4_results_v2.json '
                            '--fig-version v1')})
        rec.note('source_protocol and numerics are COPIED VERBATIM from '
                 + REFERENCE_RUN + ' (the headline hypothesis-B run) so that '
                 'this aggregate record carries a complete, valid description '
                 'of the model the figures are drawn from. They describe that '
                 'run, not the other 125; each of those has its own manifest, '
                 'and all 126 are declared as inputs here.')
    print(f'wrote {mpath}')
    print(rm.verify(mpath)['status'])


if __name__ == '__main__':
    main()
