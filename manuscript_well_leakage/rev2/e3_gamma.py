"""E3 -- put every psi -> strain conversion on Gamma, and say what that moves.

The legacy conversion is

    pf_dataframe.data = pf_dataframe.data * 6894.76 / 30e9        (= P / E)

i.e. 2.2982533e-07 strain/psi, applied at four sites named in the task package.
It is replaced everywhere by

    strain_rate = Gamma * dP/dt,   Gamma = 8.94e-9 psi^-1

so the synthetic panels carry strain RATE in s^-1. Never microstrain: the
manuscript body prints Gamma as `ue*psi^-1`, which is wrong by 1e6, and the
computation side is the reference (house rules, "Symbols and units").

WHAT THIS SCRIPT IS NOT. It runs no solver. It reads the frozen 2025 archive
`output/0211_simulation_MULTIstage/` through `rev2_layout` -- the same input the
four legacy scripts read -- so that the ONLY thing changed between the legacy
figure and the regenerated one is the coefficient. Correcting the underlying
simulation (physical barrier width, dt = 1 s, the ratio the manuscript claims)
is E1's job; doing it here would confound the two changes and make the answer to
"what does Gamma alone move?" unmeasurable. The legacy scripts are not edited and
are not imported.

Sections
  A  audit the conversion sites: exact line text, file hash, and a repo-wide
     sweep for sites the task package's list of four missed
  B  rebuild the three-phase chain the legacy figures plot, in numpy, replicating
     fibeRIS `select_time(30, end)` / `right_merge` / `select_depth` semantics
  C  the conversion itself: rescale invariance, colour-limit saturation, and
     whether anything in the rendered panel moves that is not the scale factor
  D  the gradient is POST-PROCESSING, not solver output -- what that costs
  E  the two panels of the manuscript figure on one physical axis
  F  figures (>= 300 dpi, versioned, never overwriting)
  G  JSON record + manifest

Run:
    python3 scripts/manuscript_well_leakage/rev2/e3_gamma.py \
        --config configs/rev2/e3_gamma.json
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.colors import Normalize    # noqa: E402
import matplotlib.dates as mdates          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'manuscript_well_leakage',
                                'baseline_calibration'))

import rev2_layout as rl        # noqa: E402
import rev2_data as rd          # noqa: E402
import rev2_manifest as rm      # noqa: E402

TAG = '[E3]'


def log(msg):
    print(f"{TAG} {msg}", flush=True)


def jd(x):
    """JSON-safe."""
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else str(v)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [jd(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {k: jd(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jd(v) for v in x]
    if isinstance(x, datetime.datetime):
        return x.isoformat()
    return x


# ---------------------------------------------------------------------------
# A -- audit the conversion sites
# ---------------------------------------------------------------------------

def audit_sites(cfg):
    """Verify each declared line still says what the package claims.

    A line number alone is not provenance: the file could have moved under it.
    Each site therefore records the file's sha256, the exact stripped text, and
    whether it matches. A repo-wide sweep then looks for sites the package's list
    of four does not contain.
    """
    out = {'declared': [], 'sweep': {}}
    for site in cfg['audit_sites']:
        path = os.path.join(REPO, site['file'])
        with open(path, 'r') as fh:
            lines = fh.read().split('\n')
        n = int(site['line'])
        text = lines[n - 1].strip() if 0 < n <= len(lines) else None
        rec = {
            'file': site['file'], 'line': n, 'declared_by': site['declared_by'],
            'expect': site['expect'], 'actual': text,
            'matches': text == site['expect'].strip(),
            'sha256': rm.sha256_file(path),
            'n_lines': len(lines),
        }
        if site.get('gradient_lines'):
            rec['gradient_lines'] = {
                str(g): lines[g - 1].strip() for g in site['gradient_lines']
                if 0 < g <= len(lines)}
        # the coefficient this site actually applies, and how far it is from
        # Gamma. Site 5 divides psi by a modulus in Pa, so it is short by the
        # 6894.76 psi->Pa factor on top of using E instead of Gamma.
        gamma = float(cfg['coefficients']['Gamma_psi_inv'])
        psi_pa = float(cfg['coefficients']['legacy_psi_to_pa'])
        E = float(cfg['coefficients']['legacy_youngs_modulus_pa'])
        act = rec['actual'] or ''
        if '6894.76' in act and '30e9' in act:
            k, form = psi_pa / E, 'P[psi] * 6894.76 / E  =  P/E'
        elif act.endswith('/ E'):
            k, form = 1.0 / E, ('P[psi] / E[Pa]  -- the psi->Pa factor is '
                                'MISSING, so this is not P/E either')
        else:
            k, form = None, 'no conversion on this line (constant definition)'
        rec['implied_coefficient'] = {
            'strain_per_psi': k, 'form': form,
            'ratio_to_Gamma': None if k is None else k / gamma,
            'ratio_to_the_other_four_sites': None if k is None
            else k / (psi_pa / E)}
        out['declared'].append(rec)

    # ---- orientation audit: can these scripts still be run at all? --------
    out['orientation'] = []
    oa = cfg.get('orientation_audit', {})
    for sc in oa.get('scripts', []):
        with open(os.path.join(REPO, sc['file'])) as fh:
            lines = fh.read().split('\n')

        def active(n):
            txt = lines[int(n) - 1].strip()
            return {'line': int(n), 'text': txt,
                    'active': bool(txt) and not txt.startswith('#')}

        p3 = rl.load_panel(os.path.join(REPO, sc['phase3_file']),
                           compute_sha256=False)
        t3 = active(sc['phase3_T_line'])
        # what orientation reaches select_time / right_merge / the gradient loop
        reached = p3.detected_layout
        if t3['active'] and '.T' in t3['text']:
            reached = (rl.LAYOUT_DEPTH_MAJOR
                       if reached == rl.LAYOUT_TIME_MAJOR
                       else rl.LAYOUT_TIME_MAJOR)
        st = active(sc['select_time_line'])
        out['orientation'].append({
            'file': sc['file'],
            'phase3_file': sc['phase3_file'],
            'phase3_on_disk_layout': p3.detected_layout,
            'phase1_transpose': active(sc['phase1_T_line']),
            'phase2_transpose': active(sc['phase2_T_line']),
            'phase3_transpose': t3,
            'phase3_orientation_reaching_fiberis': reached,
            'phase3_orientation_correct': reached == rl.LAYOUT_DEPTH_MAJOR,
            'select_time_call': st,
            'select_time_raises_typeerror_today': bool(
                'get_end_time()' in st['text'] and st['active']),
        })

    # repo-wide sweep. Two independent needles so a renamed constant is still
    # caught: the psi->Pa literal, and Young's modulus.
    needles = ('6894.76', '30e9')
    hits = {k: [] for k in needles}
    self_path = os.path.abspath(__file__)
    for dirpath, dirnames, filenames in os.walk(os.path.join(REPO, 'scripts')):
        dirnames[:] = [d for d in dirnames if d not in ('__pycache__', '.git')]
        for fn in filenames:
            if not fn.endswith('.py'):
                continue
            p = os.path.join(dirpath, fn)
            if os.path.abspath(p) == self_path:
                continue   # this file quotes both literals to audit them
            try:
                with open(p, 'r', errors='replace') as fh:
                    txt = fh.read()
            except OSError:
                continue
            for i, line in enumerate(txt.split('\n'), start=1):
                for nd in needles:
                    if nd in line:
                        hits[nd].append({'file': os.path.relpath(p, REPO),
                                         'line': i, 'text': line.strip()[:160]})
    # the psi->strain sites are the lines carrying BOTH literals, plus the
    # `/ E` sites where E was bound to 30e9 earlier in the same file.
    both = [h for h in hits['6894.76']
            if any(g['file'] == h['file'] and g['line'] == h['line']
                   for g in hits['30e9'])]
    declared_keys = {(s['file'], s['line']) for s in cfg['audit_sites']}
    undeclared = [h for h in both if (h['file'], h['line']) not in declared_keys]
    # This round's own tooling under scripts/manuscript_well_leakage/rev2/ quotes
    # the legacy expression in prose (E1 and E3 both do). Those are not
    # conversion sites; keep them separate so the headline count stays a count of
    # legacy CODE.
    rev2_dir = os.path.join('scripts', 'manuscript_well_leakage', 'rev2') + os.sep
    out['sweep'] = {
        'needles': list(needles),
        'n_hits_6894_76': len(hits['6894.76']),
        'n_hits_30e9': len(hits['30e9']),
        'lines_carrying_both': both,
        'undeclared_legacy_lines_carrying_both': [
            h for h in undeclared if not h['file'].startswith(rev2_dir)],
        'undeclared_in_rev2_tooling_prose_only': [
            h for h in undeclared if h['file'].startswith(rev2_dir)],
        'files_binding_E_to_30e9': sorted(
            {h['file'] for h in hits['30e9']
             if h['text'].replace(' ', '').startswith('E=30e9')}),
    }
    return out


# ---------------------------------------------------------------------------
# B -- rebuild the chain the legacy figures plot
# ---------------------------------------------------------------------------

def _panel_dict(p):
    return {'data': p.data, 'taxis': p.taxis.copy(), 'daxis': p.daxis.copy(),
            'start': p.start_time, 'src': p.source_path,
            'layout': p.detected_layout, 'sha256': p.sha256}


def select_time_from(pd_, t_start_s):
    """fibeRIS Data2D.select_time(t_start, end) in canonical (n_t, n_x) form.

    core2D.py:475-499. The taxis is rebased to the FIRST IN-WINDOW SAMPLE, not
    to the requested start, and start_time absorbs that offset. Reimplemented
    rather than called because `select_time(30, obj.get_end_time())` -- the idiom
    all four legacy scripts use -- now raises TypeError (core2D.py:473, one int
    and one datetime), so the legacy call cannot be executed today at all.
    """
    m = pd_['taxis'] >= float(t_start_s)
    if not m.any():
        raise ValueError('select_time removed every sample')
    off = float(pd_['taxis'][m][0])
    return {'data': pd_['data'][m, :], 'taxis': pd_['taxis'][m] - off,
            'daxis': pd_['daxis'], 'start': pd_['start'] +
            datetime.timedelta(seconds=off), 'src': pd_['src'],
            'layout': pd_['layout'], 'sha256': pd_['sha256'],
            'select_time_offset_s': off, 'n_dropped': int((~m).sum())}


def right_merge(a, b):
    """fibeRIS Data2D.right_merge (core2D.py:632-651) in canonical form."""
    if not np.array_equal(a['daxis'], b['daxis']):
        raise ValueError('daxis mismatch; right_merge would be refused')
    off = (b['start'] - a['start']).total_seconds()
    end_a = a['start'] + datetime.timedelta(seconds=float(a['taxis'][-1]))
    if b['start'] < end_a:
        raise ValueError('overlap; right_merge would be refused')
    return {'data': np.concatenate([a['data'], b['data']], axis=0),
            'taxis': np.concatenate([a['taxis'], b['taxis'] + off]),
            'daxis': a['daxis'], 'start': a['start'],
            'merge_offsets_s': a.get('merge_offsets_s', []) + [off],
            'phase_lengths': a.get('phase_lengths',
                                   [int(a['data'].shape[0])])
                             + [int(b['data'].shape[0])]}


def build_chain(cfg, variant_key):
    """phase1 |> phase2(select_time 30) |> phase3(select_time 30), depth-cropped."""
    arch = cfg['archive']
    p1 = _panel_dict(rl.load_panel(os.path.join(REPO, arch['phase1'])))
    p2 = select_time_from(
        _panel_dict(rl.load_panel(os.path.join(REPO, arch['phase2']))),
        cfg['legacy_figure_recipe']['phase2_select_time_start_s'])
    p3raw = _panel_dict(rl.load_panel(
        os.path.join(REPO, arch['phase3_variants'][variant_key])))
    p3 = select_time_from(
        p3raw, cfg['legacy_figure_recipe']['phase3_select_time_start_s'])
    merged = right_merge(right_merge(p1, p2), p3)

    fh7 = rd.load_frac_hits(7)
    fh8 = rd.load_frac_hits(8)
    lo, hi = float(fh8.min()) - 500.0, float(fh7.max()) + 500.0
    dm = (merged['daxis'] >= lo) & (merged['daxis'] <= hi)

    rec = {
        'variant': variant_key,
        'phase_files': {'phase1': arch['phase1'], 'phase2': arch['phase2'],
                        'phase3': arch['phase3_variants'][variant_key]},
        'on_disk_layouts': {'phase1': p1['layout'], 'phase2': p2['layout'],
                            'phase3': p3['layout']},
        'phase_lengths_after_crop': merged['phase_lengths'],
        'merge_offsets_s': merged['merge_offsets_s'],
        'phase2_select_time_offset_s': p2['select_time_offset_s'],
        'phase2_samples_dropped': p2['n_dropped'],
        'phase3_select_time_offset_s': p3['select_time_offset_s'],
        'phase3_samples_dropped': p3['n_dropped'],
        'depth_window_ft': [lo, hi],
        'n_depth_kept': int(dm.sum()),
        'daxis_kept_ft': [float(merged['daxis'][dm][0]),
                          float(merged['daxis'][dm][-1])],
        'n_time': int(merged['data'].shape[0]),
        'start_time': merged['start'],
        't_span_s': [float(merged['taxis'][0]), float(merged['taxis'][-1])],
    }
    return {
        'P': np.ascontiguousarray(merged['data'][:, dm]),   # (n_t, n_x) psi
        't': merged['taxis'], 'd': merged['daxis'][dm],
        'daxis_full': merged['daxis'], 'start': merged['start'],
        'seam_rows': [merged['phase_lengths'][0],
                      merged['phase_lengths'][0] + merged['phase_lengths'][1]],
        'record': rec,
        'panels': {'phase1': p1, 'phase2': p2, 'phase3': p3},
    }


# ---------------------------------------------------------------------------
# D -- the gradient is post-processing
# ---------------------------------------------------------------------------

def gradient_forms(P, t):
    """Both differencing forms the codebase contains, on the same field.

    legacy  np.gradient(P, axis=0) / np.gradient(t)   -- unit-spacing gradient
            divided by the gradient of the axis. EXACT on a locally uniform grid,
            first-order-wrong where dt changes.
    correct np.gradient(P, t, axis=0)                 -- the second-order
            non-uniform coefficients.
    """
    g_leg = np.gradient(P, axis=0) / np.gradient(t)[:, None]
    g_cor = np.gradient(P, t, axis=0)
    return g_leg, g_cor


def gradient_report(P, t, seam_rows, gamma, k_legacy):
    g_leg, g_cor = gradient_forms(P, t)
    err = np.abs(g_leg - g_cor)
    dts = np.diff(t)
    nonuni = np.zeros(t.size, dtype=bool)
    nonuni[1:-1] = np.abs(dts[1:] - dts[:-1]) > 1e-9

    rowmax_leg = np.abs(g_leg).max(axis=1)
    rowmax_cor = np.abs(g_cor).max(axis=1)
    med_rowmax = float(np.median(rowmax_cor))

    # ordering: convert-then-difference vs difference-then-convert.
    # np.gradient is linear, so these must agree to round-off. Measured, not
    # assumed, because the task package flags the ordering as load-bearing.
    conv_first = np.gradient(gamma * P, t, axis=0)
    diff_first = gamma * g_cor
    order_absmax = float(np.abs(conv_first - diff_first).max())
    order_rel = order_absmax / float(np.abs(diff_first).max())

    seams = {}
    for lbl, i in (('phase1_to_phase2', seam_rows[0]),
                   ('phase2_to_phase3', seam_rows[1])):
        i = int(i)
        seams[lbl] = {
            'first_row_of_next_phase': i,
            'dt_across_seam_s': float(t[i] - t[i - 1]),
            'rows': {str(j): {'t_s': float(t[j]),
                              'max_abs_dPdt_legacy_psi_s': float(rowmax_leg[j]),
                              'max_abs_dPdt_correct_psi_s': float(rowmax_cor[j])}
                     for j in range(i - 3, i + 3) if 0 <= j < t.size},
            # the restart JUMP: last row untouched by the seam stencil -> first
            # row that is. The central difference smears it over two rows.
            'jump_factor_rowmax_correct':
                float(rowmax_cor[i - 1] / max(rowmax_cor[i - 2], 1e-300)),
            'jump_factor_rowmax_legacy':
                float(rowmax_leg[i - 1] / max(rowmax_leg[i - 2], 1e-300)),
            'seam_rowmax_over_median_correct':
                float(max(rowmax_cor[i - 1], rowmax_cor[i]) / med_rowmax),
            'seam_rowmax_over_median_legacy':
                float(max(rowmax_leg[i - 1], rowmax_leg[i]) / med_rowmax),
            'legacy_over_correct_at_seam_rows': [
                float(rowmax_leg[i - 1] / rowmax_cor[i - 1]),
                float(rowmax_leg[i] / rowmax_cor[i])],
        }

    frac = err.max(axis=1) / np.maximum(rowmax_cor, 1e-300)
    iworst = int(np.argmax(frac))
    dt_med = float(np.median(dts))
    return {
        'legacy_vs_correct_gradient_form': {
            'n_rows_total': int(t.size),
            'n_rows_locally_nonuniform_dt': int(nonuni.sum()),
            'nonuniform_row_indices': [int(i) for i in np.flatnonzero(nonuni)],
            'max_abs_diff_on_uniform_rows_psi_s':
                float(err[~nonuni].max()) if (~nonuni).any() else None,
            'max_abs_diff_on_nonuniform_rows_psi_s': float(err[nonuni].max()),
            'max_abs_dPdt_correct_psi_s': float(np.abs(g_cor).max()),
            'worst_relative_row': {
                'row': iworst, 't_s': float(t[iworst]),
                'ratio_err_over_rowmax': float(frac[iworst]),
                'rowmax_correct_psi_s': float(rowmax_cor[iworst]),
                'note': ('a ratio > 1 means the row is quiet: the absolute error '
                         'there is small. Read it with rowmax_correct_psi_s.'),
            },
            'rms_diff_over_rms_dPdt': float(
                np.sqrt((err ** 2).mean()) / np.sqrt((g_cor ** 2).mean())),
        },
        'ordering_convert_then_difference': {
            'max_abs_diff_per_s': order_absmax,
            'relative_to_max_signal': order_rel,
            'verdict': ('commutes to floating-point round-off: multiplying by a '
                        'constant and np.gradient are both linear, so the order '
                        'of the two operations changes nothing'),
        },
        'time_axis': {
            'dt_min_s': float(dts.min()), 'dt_median_s': dt_med,
            'dt_max_s': float(dts.max()),
            'n_dt_values_gt_29_9': int((dts > 29.9).sum()),
            'central_difference_full_span_at_median_dt_s': 2.0 * dt_med,
            'first_transfer_null_hz': 1.0 / (2.0 * dt_med),
            'nyquist_of_sample_grid_hz': 1.0 / (2.0 * dt_med),
        },
        'phase_seams': seams,
        'median_row_max_abs_dPdt_psi_s': med_rowmax,
        'k_legacy_strain_per_psi': k_legacy,
    }, g_leg, g_cor


# ---------------------------------------------------------------------------
# C -- the conversion
# ---------------------------------------------------------------------------

def conversion_report(dPdt, gamma, k_legacy, clim_legacy):
    """Is the change a pure rescale of the rendered panel, or is it more?"""
    s_leg = k_legacy * dPdt
    s_gam = gamma * dPdt
    ratio = k_legacy / gamma
    clim_gamma = clim_legacy * gamma / k_legacy

    # 1. the arrays are exactly proportional
    resid = np.abs(s_gam - (gamma / k_legacy) * s_leg)
    pure_rescale = {
        'max_abs_residual_per_s': float(resid.max()),
        'relative_to_max_signal': float(resid.max() /
                                        np.abs(s_gam).max()),
        'verdict': 'pure rescale to floating-point round-off',
    }

    # 2. saturation against the hard-coded colour limit
    def sat(a, c):
        return {'frac_above_clim': float(np.mean(np.abs(a) > c)),
                'frac_at_or_above_half_clim': float(np.mean(np.abs(a) > 0.5 * c)),
                'max_abs': float(np.abs(a).max()),
                'max_abs_over_clim': float(np.abs(a).max() / c),
                'p99_9_abs': float(np.percentile(np.abs(a), 99.9))}

    # 3. rendered-pixel identity. The panel is Normalize(-c, +c) -> cmap; if the
    #    clim is rescaled with the coefficient the RGBA buffers must be equal.
    cmap = matplotlib.colormaps['bwr']
    rgba_leg = cmap(Normalize(-clim_legacy, clim_legacy)(s_leg), bytes=True)
    rgba_gam_same = cmap(Normalize(-clim_legacy, clim_legacy)(s_gam), bytes=True)
    rgba_gam_resc = cmap(Normalize(-clim_gamma, clim_gamma)(s_gam), bytes=True)
    n_px = int(rgba_leg.shape[0] * rgba_leg.shape[1])
    n_diff_same = int(np.any(rgba_leg != rgba_gam_same, axis=-1).sum())
    n_diff_resc = int(np.any(rgba_leg != rgba_gam_resc, axis=-1).sum())

    return {
        'coefficients': {
            'legacy_P_over_E_strain_per_psi': k_legacy,
            'Gamma_psi_inv': gamma,
            'ratio_legacy_over_Gamma': ratio,
        },
        'pure_rescale_check': pure_rescale,
        'strain_rate_range_per_s': {
            'legacy': [float(s_leg.min()), float(s_leg.max())],
            'gamma': [float(s_gam.min()), float(s_gam.max())],
        },
        'dPdt_range_psi_per_s': [float(dPdt.min()), float(dPdt.max())],
        'colour_limits': {
            'legacy_hard_coded_per_s': clim_legacy,
            'gamma_equivalent_per_s': clim_gamma,
            'clim_in_dPdt_psi_per_s': clim_legacy / k_legacy,
            'note': ('the colour limit is a hard-coded absolute number '
                     '(104:202 img3.set_clim(cx * 1e-7)); it does not follow the '
                     'coefficient, so it must be rescaled by hand'),
        },
        'saturation': {
            'legacy_at_legacy_clim': sat(s_leg, clim_legacy),
            'gamma_at_legacy_clim_UNRESCALED': sat(s_gam, clim_legacy),
            'gamma_at_rescaled_clim': sat(s_gam, clim_gamma),
        },
        'rendered_pixel_identity': {
            'n_pixels': n_px,
            'n_pixels_differing_gamma_at_legacy_clim': n_diff_same,
            'frac_pixels_differing_gamma_at_legacy_clim': n_diff_same / n_px,
            'n_pixels_differing_gamma_at_rescaled_clim': n_diff_resc,
            'colormap': 'bwr',
        },
    }, s_leg, s_gam, clim_gamma


# ---------------------------------------------------------------------------
# E -- the two panels of the manuscript figure on one physical axis
# ---------------------------------------------------------------------------

def load_das_window(cfg, lo, hi):
    """The three LF-DAS panels 104 merges, concatenated on absolute time."""
    recs = []
    for stage, kind in cfg['das']['panels']:
        recs.append(rd.load_das_stage(int(stage), kind=kind, md_range=(lo, hi)))
    base = recs[0]
    for r in recs[1:]:
        if not np.array_equal(r.daxis_ft, base.daxis_ft):
            raise ValueError('DAS daxis mismatch across panels')
    abs_t = np.concatenate([r.abs_times() for r in recs])
    data = np.concatenate([r.data for r in recs], axis=1)
    order = np.argsort(abs_t)
    return {'daxis': base.daxis_ft, 'abs_t': abs_t[order],
            'data': data[:, order],
            'panels': [{'stage': r.stage, 'kind': r.kind,
                        'shape': list(r.data.shape),
                        't0_abs': r.t0_abs,
                        'dt_s': float(np.median(np.diff(r.taxis_s))),
                        'source_path': os.path.relpath(r.source_path, REPO)}
                       for r in recs]}


def cross_scale_report(cfg, das, s_gam, clim_legacy, clim_gamma, dt_med):
    """The manuscript figure draws two panels on two independent colour scales.

    F1 (docs/lfdas_processing.md section 4) already established that the LF-DAS
    is deliberately kept in raw counts and that no Gamma-based amplitude match
    may be claimed from it. This block does NOT claim one. It converts the two
    hard-coded colour limits onto one axis so the caption can state how far apart
    the two display scales are -- which is a property of the figure, not of the
    rock.
    """
    counts = np.abs(das['data'])
    c_das = float(cfg['legacy_figure_recipe']['clim_das_counts'])

    # band-match: the synthetic panel's differencing stencil averages over
    # ~2*dt_med seconds, so a boxcar of that length is the fairest comparison
    # the DAS can be given without leaving raw units.
    win = max(1, int(round(2.0 * dt_med)))
    ntrim = (das['data'].shape[1] // win) * win
    boxed = das['data'][:, :ntrim].reshape(das['data'].shape[0], -1, win).mean(2)

    out = {'das_panels': das['panels'],
           'clim_das_counts': c_das,
           'frac_das_pixels_saturated_at_clim': float(np.mean(counts > c_das)),
           'das_counts_max_abs': float(counts.max()),
           'das_counts_p99_9': float(np.percentile(counts, 99.9)),
           'das_boxcar_window_samples': win,
           'das_boxcar_max_abs_counts': float(np.abs(boxed).max()),
           'das_boxcar_p99_9_counts': float(np.percentile(np.abs(boxed), 99.9)),
           'synthetic_max_abs_strain_rate_gamma_per_s': float(np.abs(s_gam).max()),
           'chains': {}}
    for key, s in cfg['coefficients']['das_counts_to_strain_rate'].items():
        if not isinstance(s, (int, float)):
            continue
        das_clim_sr = c_das * float(s)
        out['chains'][key] = {
            'counts_to_strain_rate_per_count': float(s),
            'das_clim_as_strain_rate_per_s': das_clim_sr,
            'scale_gap_legacy_synthetic_over_das': clim_legacy / das_clim_sr,
            'scale_gap_gamma_synthetic_over_das': clim_gamma / das_clim_sr,
            'das_max_abs_strain_rate_per_s': float(counts.max() * s),
            'das_boxcar_max_abs_strain_rate_per_s': float(np.abs(boxed).max() * s),
            'synthetic_over_das_boxcar_max':
                float(np.abs(s_gam).max() / (np.abs(boxed).max() * s)),
        }
    lp = cfg['legacy_figure_recipe']['das_lpfilter']
    out['bandwidth'] = {
        'das_sample_dt_s': lp['dt_s'],
        'das_lowpass_cut_hz': lp['freqcut_hz'],
        'das_lowpass': f"butterworth order {lp['order']}, zero-phase (filtfilt)",
        'synthetic_sample_dt_s': dt_med,
        'synthetic_first_null_hz': 1.0 / (2.0 * dt_med),
        'octaves_of_das_passband_the_synthetic_cannot_carry':
            float(np.log2(lp['freqcut_hz'] * 2.0 * dt_med)),
        'note': ('the ax4 trace of 102/102r low-passes the LF-DAS at 0.05 Hz on '
                 'a 1 s grid, while the synthetic panel is a central difference '
                 'on a ~30 s grid whose transfer function first nulls at '
                 '1/(2*dt). The DAS panel therefore carries content the '
                 'synthetic panel structurally cannot contain, in both figures, '
                 'independently of any coefficient.'),
    }
    return out


# ---------------------------------------------------------------------------
# F -- figures
# ---------------------------------------------------------------------------

def _wf(ax, t_abs, d, arr, clim, *, cmap='bwr', invert=True, scale=1.0):
    """Waterfall. `scale` divides the data so the colourbar carries no offset
    text -- the exponent goes into the colourbar LABEL instead, where it cannot
    collide with the panel title."""
    m = ax.pcolormesh(t_abs, d, arr / scale, cmap=cmap, shading='auto',
                      vmin=-clim / scale, vmax=clim / scale, rasterized=True)
    if invert:
        ax.invert_yaxis()
    return m


def _sr_label(scale):
    e = int(round(np.log10(scale)))
    return f'strain rate / $10^{{{e}}}$ s$^{{-1}}$'


def _timeaxis(ax, hours=3):
    ax.set_xlabel('time (UTC)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=hours))


# gauges the figure window contains (D4: Fig. 5/6 show gauges 5-9; gauge 8 is
# the sole held-out validation point -- it drives nothing).
FIG_GAUGES = {5: 15599.0, 6: 15344.0, 7: 15075.0, 8: 14821.0, 9: 14552.0}
HELD_OUT_GAUGE = 8


def _gauge_lines(ax, label_side=None):
    for g, md in FIG_GAUGES.items():
        held = (g == HELD_OUT_GAUGE)
        ax.axhline(md, color='k', ls='--', lw=1.3 if held else 0.7,
                   alpha=0.85 if held else 0.45)
        if label_side is not None:
            ax.annotate(f'g{g}' + (' (held out)' if held else ''),
                        xy=(label_side, md), xycoords=('axes fraction', 'data'),
                        xytext=(3, 2), textcoords='offset points', fontsize=7,
                        color='k', alpha=0.9 if held else 0.6)


def _abs_times(start, taxis):
    base = np.datetime64(start, 'us')
    return base + (taxis * 1e6).astype('timedelta64[us]')


def fig_rescale(path, ch, s_leg, s_gam, clim_legacy, clim_gamma, conv, dpi):
    """Three renderings of ONE field: the whole E3 argument in one picture."""
    t_abs = _abs_times(ch['start'], ch['t'])
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), sharey=True,
                             layout='constrained')
    panels = [
        (axes[0], s_leg.T, clim_legacy, 1e-7,
         'a   legacy  $P/E$ = 2.298$\\times10^{-7}$ strain psi$^{-1}$\n'
         f'clim $\\pm$1.0$\\times10^{{-7}}$ s$^{{-1}}$ (hard-coded, 104:202)'),
        (axes[1], s_gam.T, clim_legacy, 1e-7,
         'b   $\\Gamma$ = 8.94$\\times10^{-9}$ psi$^{-1}$, clim NOT rescaled\n'
         'same $\\pm$1.0$\\times10^{-7}$ s$^{-1}$ -- the panel goes blank'),
        (axes[2], s_gam.T, clim_gamma, 1e-9,
         'c   $\\Gamma$, clim rescaled by the same 25.71\n'
         f'clim $\\pm${clim_gamma/1e-9:.2f}$\\times10^{{-9}}$ s$^{{-1}}$ '
         '-- pixel-identical to a'),
    ]
    for ax, arr, cl, sc, title in panels:
        m = _wf(ax, t_abs, ch['d'], arr, cl, invert=(ax is axes[0]), scale=sc)
        cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(_sr_label(sc))
        ax.set_title(title, fontsize=10, loc='left')
        _timeaxis(ax)
    axes[0].set_ylabel('measured depth / ft')

    n = conv['rendered_pixel_identity']
    sat = conv['saturation']
    fig.suptitle(
        'E3   psi$\\rightarrow$strain on $\\Gamma$ is a pure rescale of the data '
        '-- but the colour limit is a hard-coded absolute number and does not '
        'follow it\n'
        f"pixels differing a vs c: {n['n_pixels_differing_gamma_at_rescaled_clim']} "
        f"of {n['n_pixels']:,}      "
        f"pixels differing a vs b: {n['n_pixels_differing_gamma_at_legacy_clim']:,}"
        f" ({100*n['frac_pixels_differing_gamma_at_legacy_clim']:.1f} %)      "
        f"saturated fraction "
        f"{100*sat['legacy_at_legacy_clim']['frac_above_clim']:.2f} % "
        f"$\\rightarrow$ "
        f"{100*sat['gamma_at_legacy_clim_UNRESCALED']['frac_above_clim']:.2f} % "
        f"$\\rightarrow$ "
        f"{100*sat['gamma_at_rescaled_clim']['frac_above_clim']:.2f} %",
        fontsize=10.5)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    log(f'wrote {path}')


def fig_panel(path, ch, s_gam, clim_gamma, das, cfg, cross, dpi):
    """The manuscript-style side-by-side, with the synthetic panel on Gamma."""
    t_abs = _abs_times(ch['start'], ch['t'])
    c_das = float(cfg['legacy_figure_recipe']['clim_das_counts'])
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2), sharey=True,
                             layout='constrained')

    m0 = _wf(axes[0], das['abs_t'], das['daxis'], das['data'], c_das,
             invert=True)
    cb0 = fig.colorbar(m0, ax=axes[0], fraction=0.046, pad=0.02)
    cb0.set_label('LF-DAS / raw counts')
    chain = cross['chains']['matching_104']
    axes[0].set_title(
        'a   measured LF-DAS, stage 7 + interval + stage 8 (raw counts)\n'
        f"clim $\\pm${c_das:.0f} counts $\\equiv$ $\\pm$"
        f"{chain['das_clim_as_strain_rate_per_s']/1e-9:.2f}$\\times10^{{-9}}$ "
        f"s$^{{-1}}$; {100*cross['frac_das_pixels_saturated_at_clim']:.1f} % of "
        'pixels saturate', fontsize=10, loc='left')

    m1 = _wf(axes[1], t_abs, ch['d'], s_gam.T, clim_gamma, invert=False,
             scale=1e-9)
    cb1 = fig.colorbar(m1, ax=axes[1], fraction=0.046, pad=0.02)
    cb1.set_label(_sr_label(1e-9))
    axes[1].set_title(
        'b   synthetic  $\\dot\\varepsilon=\\Gamma\\,\\partial P/\\partial t$,   '
        '$\\Gamma$ = 8.94$\\times10^{-9}$ psi$^{-1}$\n'
        f"clim $\\pm${clim_gamma/1e-9:.2f}$\\times10^{{-9}}$ s$^{{-1}}$;  archive "
        f"{os.path.basename(ch['record']['phase_files']['phase3'])} -- A5 "
        'measures ratio 1.0, i.e. NO barrier', fontsize=10, loc='left')

    for ax in axes:
        _timeaxis(ax)
    _gauge_lines(axes[0], label_side=0.005)
    _gauge_lines(axes[1], label_side=0.005)
    axes[0].set_ylabel('measured depth / ft')

    fig.suptitle(
        'E3   regenerated synthetic strain-rate panel; axis units s$^{-1}$, '
        'never microstrain\n'
        'the two panels are on INDEPENDENT colour scales. Placed on one physical '
        f"axis their limits sit a factor "
        f"{chain['scale_gap_gamma_synthetic_over_das']:.1f} apart with $\\Gamma$, "
        f"against {chain['scale_gap_legacy_synthetic_over_das']:.0f} with $P/E$",
        fontsize=10.5)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    log(f'wrote {path}')


def fig_variants(path, variants, gamma, dpi):
    """The same Gamma panel for the three phase-3 files in the archive."""
    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.4), sharey=True,
                             layout='constrained')
    axes = np.atleast_1d(axes)
    for j, (ax, (key, v)) in enumerate(zip(axes, variants.items())):
        t_abs = _abs_times(v['ch']['start'], v['ch']['t'])
        m = _wf(ax, t_abs, v['ch']['d'], v['s_gam'].T, v['clim_gamma'],
                invert=(j == 0), scale=1e-9)
        cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(_sr_label(1e-9))
        ax.set_title(f"{'abc'[j]}   {key}\n{v['label']}\n"
                     f"max $|\\dot\\varepsilon|$ = "
                     f"{np.abs(v['s_gam']).max()/1e-9:.2f}"
                     f"$\\times10^{{-9}}$ s$^{{-1}}$", fontsize=9.5, loc='left')
        _timeaxis(ax, hours=4)
        _gauge_lines(ax)
    axes[0].set_ylabel('measured depth / ft')
    fig.suptitle('E3   the $\\Gamma$ rescale (25.708x) is the same constant for '
                 'every archived phase-3 barrier strength; only the pressure '
                 'field differs.\nOnly the 1e-05 panel shows a barrier: '
                 'phase3_test is indistinguishable from the 0.1 run at this '
                 'display scale', fontsize=10.5)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    log(f'wrote {path}')


def fig_gradient(path, ch, g_leg, g_cor, grad, cfg, dpi):
    fig = plt.figure(figsize=(13.0, 11.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.0], hspace=0.62,
                          wspace=0.24)
    t = ch['t']
    rl, rc = np.abs(g_leg).max(axis=1), np.abs(g_cor).max(axis=1)
    gf = grad['legacy_vs_correct_gradient_form']

    ax = fig.add_subplot(gs[0, :])
    ax.semilogy(t / 3600.0, rc, color='#1f77b4', lw=1.0,
                label='np.gradient(P, t)   -- correct non-uniform coefficients')
    ax.semilogy(t / 3600.0, rl, color='#d62728', lw=1.0, ls='--',
                label='np.gradient(P) / np.gradient(t)   -- what all four '
                      'scripts do')
    ax.axhline(grad['median_row_max_abs_dPdt_psi_s'], color='grey', lw=0.9,
               label='median over rows of row max')
    ax.set_ylim(1e-3, 20)
    for lbl, i in (('phase 1 | 2', ch['seam_rows'][0]),
                   ('phase 2 | 3', ch['seam_rows'][1])):
        ax.axvline(t[int(i)] / 3600.0, color='k', ls=':', lw=1.1)
        ax.annotate(lbl, (t[int(i)] / 3600.0, 12), fontsize=8,
                    xytext=(3, 0), textcoords='offset points')
    ax.set_xlabel('time since phase-1 start / h')
    ax.set_ylabel('row max $|\\partial P/\\partial t|$ / psi s$^{-1}$')
    ax.legend(fontsize=8, loc='lower right', ncol=1)
    ax.set_title(
        'a   the synthetic strain rate is a POST-PROCESSED central difference on '
        'the simulation time axis, not solver output.\n'
        f"     The legacy form is exact wherever dt is locally uniform "
        f"({gf['max_abs_diff_on_uniform_rows_psi_s']:.1e} psi/s on "
        f"{gf['n_rows_total'] - gf['n_rows_locally_nonuniform_dt']} of "
        f"{gf['n_rows_total']} rows) and wrong on the "
        f"{gf['n_rows_locally_nonuniform_dt']} rows where dt changes.",
        fontsize=9.5, loc='left')

    for j, (lbl, key) in enumerate((('phase 1 | 2', 'phase1_to_phase2'),
                                    ('phase 2 | 3', 'phase2_to_phase3'))):
        s = grad['phase_seams'][key]
        i = int(s['first_row_of_next_phase'])
        sl = slice(max(0, i - 8), min(t.size, i + 8))
        ax = fig.add_subplot(gs[1, j])
        ax.plot(np.arange(sl.start, sl.stop), rc[sl], 'o-', ms=3.5,
                color='#1f77b4', label='correct')
        ax.plot(np.arange(sl.start, sl.stop), rl[sl], 's--', ms=3.5,
                color='#d62728', label='legacy form')
        ax.axvline(i - 0.5, color='k', ls=':', lw=1.2)
        ax.set_xlabel('row index in the merged panel')
        ax.set_ylabel('row max $|\\partial P/\\partial t|$ / psi s$^{-1}$')
        ax.legend(fontsize=8)
        ax.set_title(
            f"{'bc'[j]}   {lbl} restart, dt jumps to "
            f"{s['dt_across_seam_s']:.2f} s\n"
            f"     row max x{s['jump_factor_rowmax_correct']:.2f} in one step, "
            f"{s['seam_rowmax_over_median_correct']:.1f}x the median row\n"
            f"     legacy form reads "
            f"{100*(s['legacy_over_correct_at_seam_rows'][0]-1):+.1f} % there",
            fontsize=9, loc='left')

    ax = fig.add_subplot(gs[2, :])
    dt = grad['time_axis']['dt_median_s']
    f = np.logspace(-5, -0.7, 900)
    H = np.abs(np.sinc(2.0 * f * dt))       # |sin(2 pi f dt) / (2 pi f dt)|
    ax.semilogx(f, H, color='#1f77b4', lw=1.7,
                label=f'synthetic: central difference on dt = {dt:.0f} s')
    lp = cfg['legacy_figure_recipe']['das_lpfilter']
    ax.axvline(1.0 / (2.0 * dt), color='#1f77b4', ls=':', lw=1.2)
    ax.axvline(lp['freqcut_hz'], color='#d62728', ls='--', lw=1.5,
               label=f"LF-DAS trace low-pass cut {lp['freqcut_hz']} Hz "
                     f"(dt = {lp['dt_s']:.0f} s)")
    ax.axvspan(1.0 / (2.0 * dt), lp['freqcut_hz'], color='#d62728', alpha=0.13)
    ax.set_xlabel('frequency / Hz')
    ax.set_ylabel('$|H(f)|$ of the differencing stencil')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5, loc='lower left')
    ax.set_title(
        'd   the shaded band is LF-DAS passband the synthetic panel cannot '
        f"carry: {1.0/(2.0*dt):.4f} Hz to {lp['freqcut_hz']} Hz, "
        f"{np.log2(lp['freqcut_hz'] * 2.0 * dt):.2f} octaves.\n"
        '     Neither coefficient changes this; it is a property of the '
        'simulation time axis (A3: the manuscript run is effectively dt = 30 s).',
        fontsize=9.5, loc='left')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    log(f'wrote {path}')


# ---------------------------------------------------------------------------
# G -- manifest
# ---------------------------------------------------------------------------

def write_e3_manifest(cfg, cfg_path, ch, outputs, results, notes, started):
    """E3 ran no solve. The declared protocol describes the ARCHIVED runs whose
    fields are post-processed, transcribed from A5's reproduction manifest, so a
    reader of this manifest can see what produced the pressure that Gamma is
    applied to. Every such field is flagged in `notes`."""
    from fiberis.utils import mesh_utils
    an = cfg['archive_numerics_transcribed_from_A5']
    mesh = ch['daxis_full']
    fh7, fh8 = rd.load_frac_hits(7), rd.load_frac_hits(8)

    def srcs(fhs, gauge, phase_label):
        drv = rm.driver_record(
            kind='gauge_series', baseline_removal='none_absolute_psi',
            value_units='psi',
            series_path=f'data/fiberis_format/s_well/gauges/gauge{gauge}_data_swell.npz',
            gauge_number=gauge)
        out = []
        for j, md in enumerate(fhs):
            idx = int(mesh_utils.locate(mesh, float(md))[0])
            out.append(rm.source_record(mesh, md_requested_ft=float(md),
                                        mesh_idx=idx, driver=drv,
                                        label=f'{phase_label}_fh{j}',
                                        index_in_source_list=j))
        return out

    src = rm.source_protocol(
        application='dirichlet_node',
        solver_class=('ARCHIVE ONLY -- fiberis PDS1D_MultiSource, as run by '
                      '101_fiberis_matching.py in Feb 2025. E3 ran no solver.'),
        placement_rule=('frac-hit MDs snapped with fiberis.utils.mesh_utils.'
                        'locate on the archived 5656-node mesh'),
        sources=[srcs(fh7, 6, 'p1'), srcs(fh7, 6, 'p2'), srcs(fh8, 7, 'p3')],
        phase_labels=['phase1', 'phase2', 'phase3'],
        targets=[], time_level=an['source_time_level'],
        phase_chaining='phase n+1 initial field = phase n final field',
        boundary_conditions={'lbc': 'Neumann', 'rbc': 'Neumann'})

    times = []
    for lbl, key in (('phase1', 'phase1'), ('phase2', 'phase2'),
                     ('phase3', 'phase3')):
        p = ch['panels'][key]
        t = p['taxis']
        times.append(rm.time_record(
            t, mode='adaptive', theta=an['theta'],
            t_total_requested_s=float(t[-1]), dt_init_s=an['dt_init_s'],
            tol=an['tol'], controller_tol=an['controller_tol'],
            max_dt_s=an['max_dt_s'], min_dt_s=an['min_dt_s'],
            safety_factor=an['safety_factor'], order_p=an['order_p'],
            n_steps_rejected=an['n_steps_rejected'],
            source_time_level=an['source_time_level'],
            zero_field_policy=an['zero_field_policy'], label=lbl))

    lo, hi = ch['record']['depth_window_ft']
    num = rm.numerics(
        time=times,
        mesh=rm.mesh_record(mesh, dx_requested_ft=an['dx_requested_ft'],
                            window_md_ft=(lo, hi),
                            pad_low_ft=float(lo - mesh[0]),
                            pad_high_ft=float(mesh[-1] - hi),
                            refinement=('legacy refine_mesh x5 over +-1 ft of '
                                        'each stage-7 and stage-8 frac hit; A1 '
                                        'measures the realised single-node '
                                        'barrier width at 0.13333 ft')),
        interface_avg=an['interface_avg'],
        boundary={'lbc': 'Neumann', 'rbc': 'Neumann', 'pml_thickness': 0.0,
                  'sigma_max': 0.0},
        diffusivity={'profile_family': 'uniform_with_single_node_barriers',
                     'D_baseline_ft2_s': an['diffusivity_ft2_s'],
                     'note': ('archive value, 101_fiberis_matching.py:93 '
                              '(d = 140, hard-coded, never fitted)')},
        barriers=rm.NONE_DECLARED,
        leakage=rm.NONE_DECLARED,
        kernel={'name': 'none -- E3 is post-processing only',
                'post_processing': ('np.gradient along time per depth, then '
                                    'multiply by Gamma = 8.94e-9 psi^-1'),
                'equivalence_reference':
                    'output/rev2_20260901/A5/legacy_repro/manifest.json'},
        rng=rm.NONE_DECLARED,
        parallel={'processes': 1, 'backend': 'none'},
        amplification={'Gamma_psi_inv': cfg['coefficients']['Gamma_psi_inv'],
                       'legacy_P_over_E_strain_per_psi':
                           cfg['coefficients']['legacy_psi_to_pa'] /
                           cfg['coefficients']['legacy_youngs_modulus_pa'],
                       'strain_rate_units': 's^-1'})

    inputs = [(cfg['archive']['phase1'], 'prior_run_output', 'phase1'),
              (cfg['archive']['phase2'], 'prior_run_output', 'phase2')]
    for k, v in cfg['archive']['phase3_variants'].items():
        inputs.append((v, 'prior_run_output', k))
    inputs += [(cfg['geometry']['frac_hit_stage7_npz'], 'geometry', 'frac_hit_stg7'),
               (cfg['geometry']['frac_hit_stage8_npz'], 'geometry', 'frac_hit_stg8'),
               (cfg['geometry']['gauge_md_npz'], 'geometry', 'gauge_md'),
               ('output/rev2_20260901/A5/legacy_repro/manifest.json',
                'prior_run_output', 'A5_legacy_repro_manifest')]
    for s in cfg['audit_sites']:
        inputs.append((s['file'], 'other',
                       'legacy_site:' + os.path.basename(s['file'])))
    for stage, kind in cfg['das']['panels']:
        suffix = '' if kind == 'stage' else '_interval'
        inputs.append((f'data/fiberis_format/s_well/DAS/'
                       f'LFDASdata_stg{stage}{suffix}_swell.npz',
                       'das', f'das_stg{stage}{suffix}'))

    return rm.write_manifest(
        os.path.join(REPO, cfg['output']['manifest']),
        study_id=cfg['study_id'], task_id=cfg['task_id'], config=cfg,
        config_path=cfg_path, inputs=inputs, source=src, numerics=num,
        outputs=outputs, results=results, notes=notes, started_utc=started,
        run_label='e3_gamma_v1',
        require_modules=('rev2_layout', 'rev2_data', 'rev2_manifest'),
        allow_undeclared_outputs=False)


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/rev2/e3_gamma.json')
    args = ap.parse_args(argv)
    started = datetime.datetime.now(datetime.timezone.utc)
    t0 = time.time()

    cfg_path = os.path.join(REPO, args.config)
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    outdir = os.path.join(REPO, cfg['output']['dir'])
    os.makedirs(outdir, exist_ok=True)

    figpaths = {k: os.path.join(REPO, v)
                for k, v in cfg['output']['figures'].items()}
    jpath = os.path.join(REPO, cfg['output']['json'])
    apath = os.path.join(REPO, cfg['output']['arrays'])
    mpath = os.path.join(REPO, cfg['output']['manifest'])
    rm.assert_absent(list(figpaths.values()) + [jpath, apath, mpath])

    gamma = float(cfg['coefficients']['Gamma_psi_inv'])
    k_legacy = (float(cfg['coefficients']['legacy_psi_to_pa']) /
                float(cfg['coefficients']['legacy_youngs_modulus_pa']))
    clim_legacy = float(
        cfg['legacy_figure_recipe']['clim_synthetic_legacy_strain_rate'])
    dpi = int(cfg['output']['figure_dpi'])

    doc = {'task_id': cfg['task_id'], 'study_id': cfg['study_id'],
           'round_tag': cfg['round_tag'],
           'run_utc': started.isoformat(),
           'coefficients': {'Gamma_psi_inv': gamma,
                            'legacy_P_over_E_strain_per_psi': k_legacy,
                            'ratio_legacy_over_Gamma': k_legacy / gamma,
                            'axis_units': 's^-1 (strain rate); NEVER microstrain'}}

    log('A  auditing the psi->strain conversion sites')
    doc['A_audit'] = audit_sites(cfg)
    for r in doc['A_audit']['declared']:
        log(f"   {'OK ' if r['matches'] else 'MISMATCH'} {r['file']}:{r['line']}"
            f"  {r['actual']!r}")
    for o in doc['A_audit']['orientation']:
        log(f"   orientation {os.path.basename(o['file'])}: phase3 on disk "
            f"{o['phase3_on_disk_layout']}, transpose "
            f"{'ON' if o['phase3_transpose']['active'] else 'OFF'} -> reaches "
            f"fibeRIS {o['phase3_orientation_reaching_fiberis']} "
            f"({'OK' if o['phase3_orientation_correct'] else 'WRONG'})")
    und = doc['A_audit']['sweep']['undeclared_legacy_lines_carrying_both']
    log(f"   repo sweep of scripts/: {len(und)} undeclared LEGACY line(s) "
        f"carrying both literals; "
        f"{len(doc['A_audit']['sweep']['undeclared_in_rev2_tooling_prose_only'])}"
        f" prose mention(s) in this round's own tooling")

    log('B  rebuilding the legacy three-phase chain (numpy, no fibeRIS objects)')
    variants = {}
    labels = {'phase3_test': 'plotted by 102 / 102r / 104 (Fig. 6). '
                             'A5: ratio 1.0 = no barrier',
              'phase3_0p1': 'plotted by 101p',
              'phase3_1e-05': 'the ratio the manuscript claims'}
    for key in cfg['archive']['phase3_variants']:
        ch = build_chain(cfg, key)
        log(f"   {key}: {ch['P'].shape} (n_t, n_x)  seams at rows "
            f"{ch['seam_rows']}")
        g_leg, g_cor = gradient_forms(ch['P'], ch['t'])
        conv, s_leg, s_gam, clim_gamma = conversion_report(
            g_leg, gamma, k_legacy, clim_legacy)
        variants[key] = {'ch': ch, 's_leg': s_leg, 's_gam': s_gam,
                         'clim_gamma': clim_gamma, 'conv': conv,
                         'g_leg': g_leg, 'g_cor': g_cor,
                         'label': labels.get(key, '')}

    primary = cfg['archive']['primary_variant']
    v = variants[primary]
    ch = v['ch']

    log('C  conversion / rescale invariance / colour-limit saturation')
    doc['C_conversion'] = {k: variants[k]['conv'] for k in variants}
    doc['C_conversion']['primary_variant'] = primary

    log('D  gradient post-processing')
    grad, _, _ = gradient_report(ch['P'], ch['t'], ch['seam_rows'], gamma,
                                 k_legacy)
    doc['D_gradient'] = grad

    log('E  cross-panel scale against LF-DAS')
    lo, hi = ch['record']['depth_window_ft']
    das = load_das_window(cfg, lo, hi)
    doc['E_cross_scale'] = cross_scale_report(
        cfg, das, v['s_gam'], clim_legacy, v['clim_gamma'],
        grad['time_axis']['dt_median_s'])

    doc['B_chains'] = {k: jd(variants[k]['ch']['record']) for k in variants}

    log('F  figures')
    fig_rescale(figpaths['rescale'], ch, v['s_leg'], v['s_gam'], clim_legacy,
                v['clim_gamma'], v['conv'], dpi)
    fig_panel(figpaths['panel'], ch, v['s_gam'], v['clim_gamma'], das, cfg,
              doc['E_cross_scale'], dpi)
    fig_variants(figpaths['variants'], variants, gamma, dpi)
    fig_gradient(figpaths['gradient'], ch, v['g_leg'], v['g_cor'], grad, cfg, dpi)

    np.savez_compressed(
        apath,
        strain_rate_gamma_per_s=v['s_gam'].astype(np.float32),
        dPdt_psi_per_s_legacy_form=v['g_leg'].astype(np.float32),
        dPdt_psi_per_s_correct_form=v['g_cor'].astype(np.float32),
        taxis_s=ch['t'], daxis_ft=ch['d'],
        start_time=np.array(str(ch['start'])),
        Gamma_psi_inv=np.array(gamma),
        legacy_P_over_E_strain_per_psi=np.array(k_legacy),
        clim_gamma_per_s=np.array(v['clim_gamma']),
        clim_legacy_per_s=np.array(clim_legacy),
        seam_rows=np.array(ch['seam_rows']),
        variant=np.array(primary),
        units=np.array('strain rate in s^-1; dP/dt in psi s^-1; never microstrain'))
    log(f'wrote {apath}')

    doc['wall_seconds'] = time.time() - t0
    with open(jpath, 'w') as fh:
        json.dump(jd(doc), fh, indent=1, sort_keys=True)
    log(f'wrote {jpath}')

    # House rule 2 forbids deleting or overwriting, so the superseded first pass
    # stays on disk. It is declared here rather than swept under
    # allow_undeclared_outputs, so an auditor can see exactly which files are the
    # deliverable and which are the earlier draft.
    keep = {os.path.realpath(x) for x in
            list(figpaths.values()) + [jpath, apath, mpath, mpath + '.sha256',
                                       os.path.join(outdir, 'README.md')]}
    superseded = []
    for fn in sorted(os.listdir(outdir)):
        p = os.path.join(outdir, fn)
        if not os.path.isfile(p) or os.path.realpath(p) in keep:
            continue
        role = ('figure_png' if fn.endswith('.png') else
                'arrays_npz' if fn.endswith('.npz') else
                'manifest_json' if fn.startswith('manifest') and
                fn.endswith('.json') else
                'json' if fn.endswith('.json') else 'other')
        superseded.append(rm.output_decl(
            p, role=role, dpi=dpi if role == 'figure_png' else None,
            note='SUPERSEDED earlier pass; kept because house rule 2 forbids '
                 'deleting an existing output. The newest version named in '
                 'configs/rev2/e3_gamma.json is the deliverable.'))

    outputs = [rm.output_decl(jpath, role='json',
                              note='every number quoted in the E3 README'),
               rm.output_decl(apath, role='arrays_npz',
                              note='Gamma strain-rate field + both gradient forms'),
               rm.output_decl(figpaths['rescale'], role='figure_png', dpi=dpi),
               rm.output_decl(figpaths['panel'], role='figure_png', dpi=dpi),
               rm.output_decl(figpaths['variants'], role='figure_png', dpi=dpi),
               rm.output_decl(figpaths['gradient'], role='figure_png', dpi=dpi)]
    outputs += superseded
    results = {
        'ratio_legacy_over_Gamma': k_legacy / gamma,
        'pure_rescale': doc['C_conversion'][primary]['pure_rescale_check'],
        'saturation': doc['C_conversion'][primary]['saturation'],
        'rendered_pixel_identity':
            doc['C_conversion'][primary]['rendered_pixel_identity'],
        'gradient_form': grad['legacy_vs_correct_gradient_form'],
        'ordering': grad['ordering_convert_then_difference'],
        'cross_panel_scale_gap': {
            k: {'legacy': cvals['scale_gap_legacy_synthetic_over_das'],
                'gamma': cvals['scale_gap_gamma_synthetic_over_das']}
            for k, cvals in doc['E_cross_scale']['chains'].items()},
        'audit_all_four_declared_sites_match': all(
            r['matches'] for r in doc['A_audit']['declared'][:4]),
        'implied_coefficients': {r['file']: r['implied_coefficient']
                                 for r in doc['A_audit']['declared']},
        'legacy_scripts_runnable_today': {
            o['file']: {'phase3_orientation_correct':
                        o['phase3_orientation_correct'],
                        'select_time_raises_typeerror_today':
                        o['select_time_raises_typeerror_today']}
            for o in doc['A_audit']['orientation']},
    }
    notes = [
        'E3 ran NO solver. It post-processes the frozen 2025 archive so that the '
        'ONLY difference from the legacy figure is the psi->strain coefficient.',
        'source_protocol and the three time_records describe the ARCHIVED runs, '
        'transcribed from output/rev2_20260901/A5/legacy_repro/manifest.json; '
        'they are provenance for the input field, not a record of an E3 solve.',
        'Axis units are s^-1. The manuscript body prints Gamma as ue*psi^-1, '
        'which is wrong by 1e6; the computation side is the reference.',
        'The mesh_record low-end padding warning is expected and correct: the '
        'archive has 1940 ft of low-end pad against the 5000 ft house rule, '
        'which is exactly the contamination B2 measured for Fig. 6. E3 does not '
        'fix it -- E1 does.',
        'phase3_test.npz carries no barrier (A5). The regenerated panel is '
        'labelled with that fact rather than with the caption the manuscript '
        'currently carries.',
        'A fifth psi->strain site exists that the task package does not list: '
        '104_initial_diffusivity_derive.py:82 divides gauge pressure in PSI by '
        'a Young\'s modulus in PA, so its coefficient is 1/E = 3.333e-11 '
        'strain/psi -- short of P/E by the whole 6894.76 psi->Pa factor and 268x '
        'below Gamma. It must go on Gamma too if that script is ever revived.',
        'The v1, v2 and v3 files in this directory are superseded earlier passes, '
        'kept because house rule 2 forbids deleting an output. v3 is the '
        'deliverable (v4 adds the implied-coefficient table and the '
        'orientation audit). v2 fixed a double y-axis inversion on the shared-axis '
        'LF-DAS panel, moved the colourbar exponent out of the title band, added '
        'the gauge overlay and the two seam zooms, and split the repo sweep into '
        'legacy code versus this round\'s own prose. v3 is layout only, plus '
        'wording of the seam jump factor. The NUMBERS are identical in all '
        'three: diff the JSON files to confirm.',
    ]
    write_e3_manifest(cfg, cfg_path, ch, outputs, results, notes, started)
    log(f'wrote {mpath}')
    log(f'TOTAL wall {time.time() - t0:.1f} s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
