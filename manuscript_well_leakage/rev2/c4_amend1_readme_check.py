#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C4 amendment 1 / 1b -- hostile re-read of output/rev2_20260901/C4/README.md.

Three independent kinds of check, all run from the repo root:

1.  EXISTENCE.  Every path the README names in backticks or in its deliverables
    block must exist on disk.  This class of check is the whole reason the file
    exists: amendment 1 declared four figure products, referred to them in three
    places, and never ran the command that writes them (README section 11.5).
    A number that traces to a file which is not there traces to nothing.

2.  VALUES.  Every load-bearing number in the README is re-derived here from
    `c4_stage2_amend1_v1.json` / `c4_stage2_analysis_v1.json` and asserted to be
    present in the file, formatted exactly as the README prints it.  Nothing is
    hard-coded except the rounding.

3.  RETIRED CLAIMS and MANIFEST STATUS.  Strings the amendment retracted must be
    absent as live claims, and the nine C4 manifests must verify with the status
    section 8 says they do.

Exit 0 and one PASS line per check, or exit 1 at the first failure.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
C4 = os.path.join(ROOT, 'output', 'rev2_20260901', 'C4')
README = os.path.join(C4, 'README.md')

if HERE not in sys.path:
    sys.path.insert(0, HERE)
import rev2_manifest as rm   # noqa: E402

N_PASS = 0
FAILURES = []


def ok(msg):
    global N_PASS
    N_PASS += 1
    print('PASS  %s' % msg)


def check(cond, msg, detail=''):
    if cond:
        ok(msg)
    else:
        FAILURES.append((msg, detail))
        print('FAIL  %s%s' % (msg, ('\n        ' + detail) if detail else ''))


def contains(text, needle, msg):
    check(needle in text, msg, 'not found in README: %r' % needle)


def absent(text, needle, msg):
    check(needle not in text, msg, 'still present in README: %r' % needle)


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
R = open(README, encoding='utf-8').read()
AM = json.load(open(os.path.join(C4, 'c4_stage2_amend1_v1.json')))
S2 = json.load(open(os.path.join(C4, 'c4_stage2_analysis_v1.json')))

KEYS = [('pad5000:140', 140), ('pad20000:550', 550), ('pad20000:1150', 1150)]
PL = AM['planes']
HC = AM['h_convergence_at_w1']
MC = AM['minimum_convention']['rows']
RI = AM['resistance_invariance_converged']
AN = AM['anisotropy_converged']
DC = AM['d_curve_converged_ladder']
DROW = {r['D0']: r for r in DC['rows']}


def w1(k):
    return [q for q in PL[k]['per_w'] if abs(q['w_ft'] - 1.0) < 1e-9][0]


print('=' * 78)
print('C4 README audit  --  %s' % README)
print('=' * 78)

# ---------------------------------------------------------------------------
# 1. EXISTENCE
# ---------------------------------------------------------------------------
print('\n--- 1. every path the README names exists ---')

# Names that are generic or belong to another task's tree are resolved
# explicitly; everything else is looked for under the repo root, under the C4
# directory and under C4/figs, in that order.
GENERIC = {
    # written by rev2_manifest inside each run directory, never at a fixed path
    'manifest.json',
    'manifest.json.sha256',
    # named as a module, not as a path
    'rev2_core.py', 'rev2_data.py', 'rev2_manifest.py',
    'core1D.py', 'core2D.py', 'matbuilder.py',
}
PAT = re.compile(r'`([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:png|json|py|md|log|npz|csv|txt))`')
named = sorted({m for m in PAT.findall(R)})
missing = []
for nm in named:
    if nm in GENERIC:
        continue
    cands = [os.path.join(ROOT, nm), os.path.join(C4, nm),
             os.path.join(C4, 'figs', nm), os.path.join(HERE, nm),
             os.path.join(ROOT, 'output', 'rev2_20260901', nm),
             os.path.join(ROOT, 'scripts', 'manuscript_well_leakage',
                          'baseline_calibration', nm)]
    if not any(os.path.exists(c) for c in cands):
        missing.append(nm)
check(not missing, 'all %d backticked file names resolve on disk' % len(named),
      'missing: %s' % ', '.join(missing))

# the deliverables block lists paths without backticks; check the trio and the
# provenance records explicitly, because those are what amendment 1b was about.
DELIVERABLES = [
    'figs/fig_c4_plane_converged_v5.png',
    'figs/fig_c4_resistance_v5.png',
    'figs/fig_c4_dcurve_converged_v5.png',
    'figs/figures_provenance_stage2_v5.json',
    'figs/fig_c4_plane_converged_v6.png',
    'figs/fig_c4_resistance_v6.png',
    'figs/fig_c4_dcurve_converged_v6.png',
    'figs/figures_provenance_stage2_v6.json',
    'c4_stage2_amend1_v1.json',
    'c4_stage2_analysis_v1.json',
    'logs/amend1_finegrid.log',
    'logs/amend1b_redo140.log',
    'finegrid_pad5000_140/manifest.json',
    'finegrid_pad20000_550/manifest.json',
    'finegrid_pad20000_1150/manifest.json',
    'finegrid_dcurve_pad20000/manifest.json',
]
gone = [d for d in DELIVERABLES if not os.path.exists(os.path.join(C4, d))]
check(not gone, 'all %d declared amendment deliverables exist' % len(DELIVERABLES),
      'missing: %s' % ', '.join(gone))

# the amendment figure trio must be NEWER than the amendment analysis it draws
t_am = os.path.getmtime(os.path.join(C4, 'c4_stage2_amend1_v1.json'))
stale = [d for d in DELIVERABLES[:8]
         if os.path.getmtime(os.path.join(C4, d)) < t_am]
check(not stale, 'the v5/v6 figure products post-date c4_stage2_amend1_v1.json',
      'older than the analysis: %s' % ', '.join(stale))

# v5 and v6 differ only in the resistance figure
import hashlib


def sha(pth):
    return hashlib.sha256(open(pth, 'rb').read()).hexdigest()


for stem in ('fig_c4_plane_converged', 'fig_c4_dcurve_converged'):
    a = sha(os.path.join(C4, 'figs', stem + '_v5.png'))
    b = sha(os.path.join(C4, 'figs', stem + '_v6.png'))
    check(a == b, '%s is byte-identical between _v5 and _v6' % stem,
          '%s vs %s' % (a[:16], b[:16]))
a = sha(os.path.join(C4, 'figs', 'fig_c4_resistance_v5.png'))
b = sha(os.path.join(C4, 'figs', 'fig_c4_resistance_v6.png'))
check(a != b, 'fig_c4_resistance differs between _v5 and _v6 (the layout repair)')

# the provenance record of v6 must name the three figures it wrote
prov = json.load(open(os.path.join(C4, 'figs',
                                   'figures_provenance_stage2_v6.json')))
paths = {os.path.basename(f['path']) for f in prov['figures']}
check(paths == {'fig_c4_plane_converged_v6.png', 'fig_c4_resistance_v6.png',
                'fig_c4_dcurve_converged_v6.png'},
      'figures_provenance_stage2_v6.json declares the three v6 figures',
      str(sorted(paths)))
for f in prov['figures']:
    fp = os.path.join(ROOT, f['path'])
    check(os.path.exists(fp) and sha(fp) == f['sha256'],
          'provenance sha256 matches on disk: %s' % os.path.basename(f['path']))
check(prov['amend1_analysis_sha256'] ==
      sha(os.path.join(C4, 'c4_stage2_amend1_v1.json')),
      'v6 provenance pins the amendment analysis JSON now on disk')

# ---------------------------------------------------------------------------
# 2. VALUES
# ---------------------------------------------------------------------------
print('\n--- 2. every load-bearing number is re-derived from the JSONs ---')

# 2a  the headline resistance
conv = RI['converged_mean_resistance_s_per_ft']
coarse = RI['coarse_mean_resistance_s_per_ft']
contains(R, ' / '.join('%.0f' % v for v in conv),
         'headline resistance %s s/ft' % ' / '.join('%.0f' % v for v in conv))
contains(R, ' / '.join('%.0f' % v for v in coarse),
         'superseded resistance %s s/ft is shown as superseded'
         % ' / '.join('%.0f' % v for v in coarse))
contains(R, '%.1f %%' % RI['converged_spread_pct_of_mean'],
         'resistance spread across D0 = %.1f %%'
         % RI['converged_spread_pct_of_mean'])
pb = RI['per_barrier_s_per_ft']
contains(R, '%.0f-%.0f s/ft per barrier' % (min(pb), max(pb)),
         'per-barrier resistance %.0f-%.0f s/ft' % (min(pb), max(pb)))

# 2b  the exponent
for k, D0 in KEYS:
    f = PL[k]['exponent_fit_log10_Dbarrier_vs_log10_w_fine']
    contains(R, '%.4f ± %.4f' % (f['slope'], f['slope_stderr']),
             'exponent p at D0 = %d is %.4f ± %.4f'
             % (D0, f['slope'], f['slope_stderr']))
    check(abs(f['r2'] - 1.0) < 1e-4,
          'exponent fit R2 at D0 = %d rounds to 0.99999 (%.7f)' % (D0, f['r2']))

# 2c  the floor variation, its monotonicity, the anisotropy
fl = [PL[k]['floor_variation_psi'] for k, _ in KEYS]
contains(R, ' / '.join('%.2f' % v['converged_vertex'] for v in fl),
         'floor variation %s psi'
         % ' / '.join('%.2f' % v['converged_vertex'] for v in fl))
check(all(v['monotone_in_w'] and v['n_positive_differences'] == v['n_differences']
          for v in fl),
      'the floor rises monotonically with w on all three planes (7/7 each)')
ani = [AN[k]['anisotropy_along_over_across'] for k, _ in KEYS]
contains(R, '%.1f-%.1f' % (min(ani), max(ani)),
         'anisotropy range %.1f-%.1f' % (min(ani), max(ani)))
for k, D0 in KEYS:
    contains(R, '%.1f' % AN[k]['anisotropy_along_over_across'],
             'anisotropy at D0 = %d is %.1f'
             % (D0, AN[k]['anisotropy_along_over_across']))

# 2d  the manuscript-ratio penalty, under the one convention
pen = [MC[k]['penalty_pct_vs_converged_min'] for k, _ in KEYS]
contains(R, '**+%.1f %% / +%.1f %% / +%.1f %%**' % tuple(pen),
         'manuscript-ratio penalty +%.1f / +%.1f / +%.1f %%' % tuple(pen))
old_pen = [MC[k]['penalty_pct_vs_grid_min'] for k, _ in KEYS]
contains(R, '+%.1f / +%.1f / +%.1f %%' % tuple(old_pen),
         'the superseded grid-minimum penalty is shown as superseded')

# 2e  the tightening threshold
thr = [MC[k]['floor_variation_as_pct_of_converged_min'] for k, _ in KEYS]
contains(R, '+%.2f %% at `D0` = 140, +%.2f %% at 550, +%.2f %% at 1150'
         % tuple(thr),
         'tightening threshold +%.2f / +%.2f / +%.2f %%' % tuple(thr))

# 2f  the h-convergence ladder
for k, D0 in KEYS:
    b = HC[k]['resistance_s_per_ft_by_step']
    e_fine = 100.0 * (b['0.05'] / b['0.0125'] - 1.0)
    e_coarse = 100.0 * (b['0.25'] / b['0.0125'] - 1.0)
    contains(R, '+%.2f %%' % e_fine,
             'h-ladder: 0.05-decade error at D0 = %d is +%.2f %%' % (D0, e_fine))
    contains(R, '+%.2f %%' % e_coarse,
             'h-ladder: 0.25-decade error at D0 = %d is +%.2f %%' % (D0, e_coarse))
    contains(R, '%.2f' % HC[k]['observed_order_p'],
             'h-ladder: observed order at D0 = %d is %.2f'
             % (D0, HC[k]['observed_order_p']))

# 2g  the two independent ladders agree at the shared cells
worst = 0.0
for k, D0 in KEYS:
    a = DROW[float(D0)]['fine_resistance_s_per_ft']
    b = w1(k)['fine_resistance_s_per_ft']
    worst = max(worst, abs(a / b - 1.0) * 100.0)
    contains(R, '%.1f vs %.1f' % (a, b),
             'D curve vs plane at D0 = %d: %.1f vs %.1f s/ft' % (D0, a, b))
contains(R, '%.2f %%' % worst,
         'worst D-curve/plane disagreement is %.2f %%' % worst)

# 2h  the D-curve fits and the 4600 exclusion
f8 = DC['fit_log10_ratio_vs_log10_D0_8rows_fine']
f9 = DC['fit_log10_ratio_vs_log10_D0_9rows_coarse']
f8c = DC['fit_log10_ratio_vs_log10_D0_8rows_coarse']
contains(R, 'D0^(-%.4f ± %.4f)' % (-f8['slope'], f8['slope_stderr']),
         'the quotable D-curve slope is -%.4f ± %.4f'
         % (-f8['slope'], f8['slope_stderr']))
contains(R, '%.0fx' % DC['D0_span_factor_8rows'],
         'the quotable span is %.0fx' % DC['D0_span_factor_8rows'])
contains(R, '%.0fx' % DC['D0_span_factor_9rows'],
         'the retracted %.0fx span is named as retracted'
         % DC['D0_span_factor_9rows'])
# the file writes negative numbers with U+2212 in prose; assert that form
contains(R, '\u2212%.4f \u00b1 %.4f' % (-f9['slope'], f9['slope_stderr']),
         'the 9-row published fit -%.4f +- %.4f is shown as superseded'
         % (-f9['slope'], f9['slope_stderr']))
contains(R, '\u2212%.4f \u00b1 %.4f' % (-f8c['slope'], f8c['slope_stderr']),
         'the 8-row coarse fit -%.4f +- %.4f is shown as superseded'
         % (-f8c['slope'], f8c['slope_stderr']))
check(DC['padding_not_converged_D0'] == [4600.0]
      and DC['n_padding_converged_rows'] == 8,
      'the JSON excludes exactly D0 = 4600, leaving 8 rows')
lo, hi = DC['D_barrier_opt_range_8rows_fine']
contains(R, '%.4f-%.4f ft²/s' % (lo, hi),
         'D_barrier* range %.4f-%.4f ft2/s' % (lo, hi))
contains(R, '%.1f %%' % (100.0 * (hi - lo) / (0.5 * (lo + hi))),
         'D_barrier* spread %.1f %%' % (100.0 * (hi - lo) / (0.5 * (lo + hi))))
fR = DC['fit_log10_R_vs_log10_D0_8rows_fine']
contains(R, '+%.3f ± %.3f' % (fR['slope'], fR['slope_stderr']),
         'D-curve resistance trend +%.3f ± %.3f'
         % (fR['slope'], fR['slope_stderr']))
gR = RI['fit_log10_R_vs_log10_D0']
contains(R, '+%.3f ± %.3f' % (gR['slope'], gR['slope_stderr']),
         'plane resistance trend +%.3f ± %.3f'
         % (gR['slope'], gR['slope_stderr']))
sig8 = abs(f8['slope'] + 1.0) / f8['slope_stderr']
contains(R, '**%.1f standard errors**' % sig8,
         'converged 8-row departure is %.1f sigma' % sig8)

# 2i  the D-curve table itself (nine rows, resistance and D_barrier)
for r in DC['rows']:
    contains(R, '| %.4f | %.0f |'
             % (r['fine_D_barrier_opt_ft2_s'], r['fine_resistance_s_per_ft']),
             'D-curve row D0 = %g prints D_b* = %.4f and W/D_b* = %.0f'
             % (r['D0'], r['fine_D_barrier_opt_ft2_s'],
                r['fine_resistance_s_per_ft']))

# 2j  the per-w resistance row of the D0 = 140 plane (section 2)
row = ' | '.join('%.0f' % q['fine_resistance_s_per_ft']
                 for q in PL['pad5000:140']['per_w'])
contains(R, row, 'the D0 = 140 per-w resistance row is %s' % row)
sp = PL['pad5000:140']['resistance_s_per_ft']['converged_spread_pct_of_mean']
contains(R, '±%.1f %%' % (sp / 2.0),
         'that row is constant to +-%.1f %%' % (sp / 2.0))

# 2k  the crossing time, which is what p = 1 rules out
ct = [q['crossing_time_s'] for q in PL['pad5000:140']['per_w']]
contains(R, ' | '.join('%.0f' % v for v in ct),
         'the D0 = 140 crossing-time row is %s' % ' | '.join('%.0f' % v for v in ct))
contains(R, '%.1f' % (max(ct) / min(ct)),
         'the crossing time varies by %.1fx over the same w range'
         % (max(ct) / min(ct)))

# 2l  the plane minima
for k, D0 in KEYS:
    if D0 == 140:
        continue
    contains(R, '**%.3f**' % PL[k]['converged_minimum_psi'],
             'converged plane minimum at D0 = %d is %.3f psi'
             % (D0, PL[k]['converged_minimum_psi']))

# ---------------------------------------------------------------------------
# 3. RETIRED CLAIMS and MANIFEST STATUS
# ---------------------------------------------------------------------------
print('\n--- 3. retired claims are gone, manifests verify as section 8 says ---')

RETIRED = [
    ('The valley is 12.9-15.4x longer', 'the pre-amendment anisotropy'),
    ('tightened below +1.1 %', 'the pre-amendment tightening threshold'),
    ('the misfit varies **0.69-0.90 psi**', 'the vertex-estimator floor'),
    ('<- QUOTE v5, NOT v4', 'the pre-1b figure pointer'),
    ('Quote v4.', 'the instruction to quote the v4 trio'),
]
for needle, what in RETIRED:
    absent(R, needle, 'retired claim absent: %s' % what)

# Superseded numbers are deliberately kept beside their replacements in this
# file, so "absent" is the wrong test for them.  What must hold instead is that
# each appears only inside a sentence that retracts it.
RETRACTION_WORDS = ('as published', 'pre-amendment', 'superseded', 'SUPERSEDED',
                    'must not be quoted', 'MUST\n      NOT BE QUOTED',
                    'got wrong', 'wrong about the cure', 'high', 'amended',
                    'quotable statement is', 'do not quote',
                    'it was quoted as', 'a null either way')
IN_CONTEXT = ['quote the planes for the VALUE', '645 / 682 / 679',
              'D0^(-1.046 \u00b1 0.020)', '12.9-15.4', '0.0127-0.0177']
for needle in IN_CONTEXT:
    hits = [m for m in range(len(R)) if R.startswith(needle, m)]
    bad = [m for m in hits
           if not any(w in R[max(0, m - 500):m + 500] for w in RETRACTION_WORDS)]
    check(bool(hits) and not bad,
          'superseded claim %r appears only in a retraction context' % needle,
          '%d occurrence(s), %d with no retraction word within 500 chars'
          % (len(hits), len(bad)))

# the README must not tell anyone to quote a superseded figure
for bad in ('QUOTE v3', 'QUOTE v4', 'QUOTE v5'):
    absent(R, bad, 'no instruction to quote %s' % bad)
contains(R, '**Quote the `_v6` trio.**', 'the README points at the v6 trio')

EXPECTED_STATUS = {
    'sweep': 'drift', 'padcheck': 'drift', 'sweep_pad20000': 'drift',
    'dcurve_pad20000': 'clean', 'padprobe40000': 'clean',
    'finegrid_pad5000_140': 'drift', 'finegrid_pad20000_550': 'clean',
    'finegrid_pad20000_1150': 'clean', 'finegrid_dcurve_pad20000': 'clean',
}
for d, want in sorted(EXPECTED_STATUS.items()):
    v = rm.verify(os.path.join(C4, d, 'manifest.json'))
    bad = [g for g in ('inputs', 'outputs') if v[g]['drift'] or v[g]['missing']]
    check(v['status'] == want and not bad,
          'manifest %s verifies %s, inputs/outputs ok' % (d, want),
          'status=%s, groups with drift=%s' % (v['status'], bad))

# and the drift that IS there is confined to code
for d in ('sweep', 'padcheck', 'sweep_pad20000', 'finegrid_pad5000_140'):
    v = rm.verify(os.path.join(C4, d, 'manifest.json'))
    check(v['code']['drift'] and not v['inputs']['drift']
          and not v['outputs']['drift'] and v['manifest_self']['status'] == 'ok',
          '%s: drift is in code only, sidecar ok' % d)

print('\n' + '=' * 78)
if FAILURES:
    print('%d PASS, %d FAIL' % (N_PASS, len(FAILURES)))
    for m, d in FAILURES:
        print('  FAIL %s\n       %s' % (m, d))
    sys.exit(1)
print('%d/%d PASS -- every number in the C4 README traces to a file' % (N_PASS, N_PASS))
sys.exit(0)
