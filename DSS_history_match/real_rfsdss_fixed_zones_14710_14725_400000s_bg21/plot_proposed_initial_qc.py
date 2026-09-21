"""Plot a review-only permeability proposal against the existing DSS caches."""
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / 'real_rfsdss_fixed_zones_14710_14725_400000s'
OUT = HERE / 'proposed_initial_qc'
PROPOSED = [-18., -17.5, -19.]
COLORS = ['#2563a6', '#d97706', '#16806a']


def load_data(name):
    path = BASELINE / 'data' / f'half_wave_positive_{name}.npz'
    with np.load(path) as a:
        return path, a['daxis'].copy(), a['taxis'].copy(), a['data'].copy()


def plot(md, times, data, zones, label, filename):
    fig, (ax, perm) = plt.subplots(
        1, 2, figsize=(14, 8), sharey=True,
        gridspec_kw={'width_ratios': [3.4, 1.6]}, layout='constrained')
    cmap = LinearSegmentedColormap.from_list('positive', ['white', '#ffaaaa', '#ff3333', '#a50000'])
    im = ax.pcolormesh(times, md, data*1e6, shading='nearest',
                       cmap=cmap, vmin=0, vmax=3.5, rasterized=True)
    ax.axvspan(times[-1], 400000, facecolor='.9', hatch='///', edgecolor='.6')
    ax.text((times[-1]+400000)/2, 14719, 'No data', rotation=90,
            ha='center', va='center', fontsize=8, color='.35')
    ax.set(xlim=(0, 400000), ylim=(14725, 14710), xlabel='Time since DSS start [s]',
           ylabel='Measured depth [ft]', title=f'Observed positive DSS | {label}')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.set_yticks(np.arange(14710, 14726, 1))
    fig.colorbar(im, ax=ax, orientation='horizontal', pad=.07, fraction=.055,
                 label='Positive strain change [microstrain]')
    # Full zone widths are retained; only the viewing window is cropped.
    edges = sorted(set([14710., 14725.] + [max(14710., min(14725., z[k]))
                   for z in zones for k in ['lo_ft', 'hi_ft']]))
    xx, yy = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid = (lo+hi)/2
        alpha = next((z['proposed_alpha'] for z in zones if z['lo_ft'] <= mid < z['hi_ft']), -21.)
        xx.extend([alpha, alpha])
        yy.extend([lo, hi])
    perm.plot(xx, yy, color='.2', lw=2.2)
    perm.axvline(-21, color='.5', ls=':', label='Background: 1e-21 m²')
    for z, color in zip([z for z in zones if z['optimized']], COLORS):
        lo, hi = max(14710, z['lo_ft']), min(14725, z['hi_ft'])
        for axis in (ax, perm):
            for edge in [z['lo_ft'], z['hi_ft']]:
                if 14710 < edge < 14725:
                    axis.axhline(edge, color=color, lw=1, ls='--', alpha=.7)
        perm.axhspan(lo, hi, color=color, alpha=.07)
        perm.plot([z['proposed_alpha']]*2, [lo, hi], color=color, lw=4)
        perm.text(-16.6, (lo+hi)/2,
                  f"{z['label']}: alpha = {z['proposed_alpha']:g}\n"
                  f"kxx = {z['proposed_kxx_m2']:.2g} m²\n"
                  f"{z['lo_ft']:g}–{z['hi_ft']:g} ft",
                  va='center', fontsize=9, color=color)
    ax.text(9000, 14713.65, 'Signal outside current zones', fontsize=9,
            bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
    perm.set(xlim=(-21.5, -12.5), xticks=[-21, -20, -19, -18, -17, -16, -15, -14, -13],
             xlabel='alpha = log10(kxx / m²)', title='Proposed initial permeability')
    perm.grid(axis='x', alpha=.15)
    perm.legend(loc='lower right', fontsize=8)
    fig.suptitle('S-well RFSDSS | 14710–14725 ft | Initial-model QC\n'
                 'Background kxx = kyy = kzz = 1e-21 m²; fixed zone boundaries', fontsize=15)
    fig.savefig(OUT / f'{filename}.png', dpi=180)
    fig.savefig(OUT / f'{filename}.pdf')
    plt.close(fig)


def main():
    cfg = json.loads((HERE / 'pilot_5iter/config.json').read_text())
    zones = sorted(cfg['zones'], key=lambda z: z['lo_ft'])
    active = [z for z in zones if z['optimized']]
    assert len(active) == len(PROPOSED)
    for z in zones:
        z['previous_initial_alpha'] = z['alpha']
        z['proposed_alpha'] = z['alpha']
    for z, alpha, label in zip(active, PROPOSED, 'ABC'):
        z.update(proposed_alpha=alpha, label=label)
    for z in zones:
        z['proposed_kxx_m2'] = 10.**z['proposed_alpha']
    native_path, md, times, native = load_data('native')
    solver_path, solver_md, solver_times, solver = load_data('solver')
    assert native.shape == (45, 3318) and solver.shape == (45, 130)
    assert np.array_equal(md, solver_md) and np.all(native >= 0)
    assert np.allclose(np.array([np.interp(solver_times, times, row) for row in native]), solver)
    for z in active:
        values = solver[(md >= z['lo_ft']) & (md < z['hi_ft'])][:, solver_times > 0]*1e6
        z['solver_sample_mean_positive_microstrain'] = float(values.mean())
        z['solver_sample_peak_positive_microstrain'] = float(values.max())
    proposal = dict(
        status='QC proposal only; no forward solve or inversion launched',
        background_alpha=-21., background_tensor_m2=[1e-21]*3,
        transverse_permeability_m2=1e-21, initial_theta_proposed=PROPOSED,
        zones=zones, fixed_positions=True, depth_crop_ft=[14710, 14725],
        nominal_end_s=400000., actual_end_s=float(times[-1]),
        n_channels=45, native_n_time=3318, n_solver_times=130, n_solver_steps=129,
        transform='Existing median-corrected positive-half-wave observations; no new preprocessing',
        rationale='Heuristic amplitudes: B has strongest sustained signal, A is elevated, C is weaker. '
                  'These guesses are not a quantitative strain-to-permeability conversion.',
        original_source=json.loads((BASELINE / 'PROPOSED_PARAMETERS.json').read_text())['original_source'],
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [native_path, solver_path]})
    OUT.mkdir(exist_ok=True)
    (OUT / 'PROPOSED_INITIAL.json').write_text(json.dumps(proposal, indent=2)+'\n')
    plot(md, times, native, zones, '45 channels × 3318 native times', 'dss_waterfall_perm_native')
    plot(md, solver_times, solver, zones, '45 channels × 130 solver times', 'dss_waterfall_perm_solver')
    lines = ['# 新初值 QC：背景 −21', '',
             '只生成初值建议和共深度轴对照图；未修改现有求解配置，未启动正演或反演。', '',
             '| 区域 | 完整深度范围 [ft] | 建议 α | 建议 kxx [m²] |',
             '|---|---|---:|---:|']
    for z in zones:
        lines.append(f"| {z.get('label', '固定外部区')} | {z['lo_ft']:g}–{z['hi_ft']:g} | {z['proposed_alpha']:g} | {z['proposed_kxx_m2']:.4e} |")
    lines += ['', '- 背景三个方向为 1e-21 m²；区内只提高 kxx，kyy/kzz 仍为 1e-21 m²。',
              '- A/B/C 分别为背景的 1000、3162、100 倍。这是供审阅的启发式初值，不是从应变幅度定量推算的渗透率。',
              '- B 的持续应变最强，A 有明显响应，C 较弱；因此采用上述相对排序。',
              '- 约 −21 作为本轮背景假设；上轮近零预测解不能独立验证真实背景渗透率。',
              '- 14712–14715 ft 也有明显信号，但不在当前高渗区内；本图保留原边界并标出此处，供 QC。',
              '- A 从 14709.5 ft 开始、C 延至 14727 ft；图只显示与 14710–14725 ft 的交集。',
              '- 使用既有漂移校正和正半波数据；负值保留为零。原生瀑布图保留 3318 时刻，求解采样图为 130 时刻。',
              '- 384301.087267–400000 s 无记录，以阴影表示；不外推数据。',
              '- 原求解时序、物性、压力源、正则项以及历史结果均未改动。', '',
              '[原生 DSS 与渗透率](dss_waterfall_perm_native.png) · [130 时刻对照](dss_waterfall_perm_solver.png)', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    print(OUT / 'dss_waterfall_perm_native.png')
    print(json.dumps({'initial_theta_proposed': PROPOSED, 'status': proposal['status']}))


if __name__ == '__main__':
    main()
