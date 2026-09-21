"""Summarize completed or partial comparison runs without starting MOOSE."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'comparison_xx_yy_60'


def main():
    summaries = {}
    frames = {}
    for mode in ['fixed_yy', 'free_yy']:
        run = ROOT/mode
        hist_path = run/'accepted_iteration_history.csv'
        if not hist_path.exists():
            summaries[mode] = {'status': 'not_yet_evaluated'}
            continue
        hist = pd.read_csv(hist_path)
        iteration = int(hist.iloc[-1].iteration)
        df = pd.read_csv(run/'iterations'/f'{iteration:02d}'/'strain.csv')
        active = df.measurement_time > 0
        zero = float(.5*np.sum(df.loc[active, 'measurement_values']**2))
        data = float(hist.iloc[-1].data_objective)
        summary = dict(status='complete' if (run/'COMPLETE').exists() else 'failed' if (run/'FAILED').exists() else 'running',
                       accepted_iterations=iteration, last_accepted=hist.iloc[-1].to_dict(),
                       data_objective=data, zero_prediction_objective=zero,
                       improvement_over_zero_fraction=1-data/zero,
                       rms_microstrain=float(hist.iloc[-1].residual_rms_microstrain),
                       observation_peak_microstrain=float(df.loc[active, 'measurement_values'].max()*1e6),
                       simulation_peak_microstrain=float(df.loc[active, 'simulation_values'].max()*1e6),
                       total_evaluations=len(pd.read_csv(run/'objective_history.csv')))
        for name in ['optimizer_result.json', 'new_yy_gradient_check.json']:
            if (run/name).exists():
                summary[name.removesuffix('.json')] = json.loads((run/name).read_text())
        if (run/'FAILED').exists():
            summary['error'] = (run/'FAILED').read_text()
        summaries[mode] = summary
        frames[mode] = df
    # Verify the two arms actually evaluated the same initial tensor/data.
    if len(frames) == 2:
        initial = [pd.read_csv(ROOT/m/'iterations/00/strain.csv') for m in frames]
        np.testing.assert_allclose(initial[0].simulation_values, initial[1].simulation_values, rtol=1e-7, atol=1e-13)
        summaries['matched_initial_response'] = True
        summaries['free_vs_fixed_data_reduction_fraction'] = 1-summaries['free_yy']['data_objective']/summaries['fixed_yy']['data_objective']
    (ROOT/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    if frames:
        fig, axes = plt.subplots(len(frames), 3, figsize=(15, 6*len(frames)), squeeze=False,
                                 sharex=True, sharey=True, layout='constrained')
        vmax = max(float(max(df.measurement_values.max(), df.simulation_values.max()))*1e6 for df in frames.values())
        for row, (mode, df) in zip(axes, frames.items()):
            t = np.sort(df.measurement_time.unique())
            y = np.sort(df.measurement_ycoord.unique())
            obs, sim = [df.pivot(index='measurement_ycoord', columns='measurement_time', values=c)
                         .reindex(index=y, columns=t).to_numpy()*1e6 for c in ['measurement_values', 'simulation_values']]
            for ax, values, name in zip(row, [obs, sim, sim-obs], ['Observation', 'Simulation', 'Residual']):
                im = ax.pcolormesh(t, 14725+(y-50)/.3048, values, shading='nearest', cmap='bwr', vmin=-vmax, vmax=vmax)
                ax.axvspan(t[-1], 400000, facecolor='.9', hatch='///', edgecolor='.6')
                ax.set(title=f'{mode}: {name}', xlabel='Time [s]', ylabel='Depth [ft]',
                       xlim=(0, 400000), ylim=(14725, 14710))
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Strain [microstrain]', shrink=.6)
        fig.savefig(ROOT/'comparison_qc.png', dpi=160)
        plt.close(fig)
    lines = ['# kyy 固定 / 独立反演对照', '',
             '两组使用相同的初始 kxx、kyy 分布；只改变 kyy 是否进入优化向量。最多 60 次接受迭代，提前满足原停止条件则结束。', '',
             '| 方案 | 状态 | 接受迭代 | 数据误差 | RMS [με] | 相对全零预测改善 |',
             '|---|---|---:|---:|---:|---:|']
    for mode in ['fixed_yy', 'free_yy']:
        s = summaries[mode]
        if 'data_objective' in s:
            lines.append(f"| {mode} | {s['status']} | {s['accepted_iterations']} | {s['data_objective']:.7e} | {s['rms_microstrain']:.5f} | {s['improvement_over_zero_fraction']:.2%} |")
    lines += ['', '比较数据误差和拟合形态；两组自由度及正则项数量不同，不以总目标值直接判断优劣。', '',
              '[并排拟合图](comparison_qc.png) · [固定 kyy 结果](fixed_yy/qc_final.png) · [独立 kyy 结果](free_yy/qc_final.png)', '']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
