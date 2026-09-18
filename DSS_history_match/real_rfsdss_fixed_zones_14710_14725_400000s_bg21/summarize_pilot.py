"""Produce review artifacts from the stopped pilot; never starts a solver."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = HERE / 'pilot_5iter'


def main():
    if not (RUN / 'PILOT_COMPLETE').exists():
        raise RuntimeError('Pilot has not finished')
    result = json.loads((RUN / 'optimizer_result.json').read_text())
    cfg = json.loads((RUN / 'config.json').read_text())
    history = pd.read_csv(RUN / 'accepted_iteration_history.csv')
    final = pd.read_csv(RUN / 'strain_after_pilot.csv')
    active = final.measurement_time > 0
    theta = np.asarray(result['theta'])
    gradient = np.loadtxt(RUN / 'gradient_after_pilot.txt')
    projected = gradient.copy()
    lo, hi = cfg['theta_bounds']
    projected[(theta <= lo+1e-10) & (gradient > 0)] = 0.
    projected[(theta >= hi-1e-10) & (gradient < 0)] = 0.
    zero_objective = float(.5*np.sum(final.loc[active,'measurement_values']**2))
    final_objective = float(.5*np.sum(final.loc[active,'misfit_values']**2))
    summary = dict(
        result=result, at_lower_bound=np.isclose(theta,lo,rtol=0,atol=1e-8).tolist(),
        distance_from_lower_bound=(theta-lo).tolist(),
        projected_gradient_inf=float(np.max(abs(projected))),
        initial_data_objective=float(history.iloc[0].data_objective),
        after_pilot_data_objective=final_objective,
        zero_prediction_data_objective=zero_objective,
        improvement_over_zero_prediction_fraction=1-final_objective/zero_objective,
        data_objective_reduction_fraction=float(1-history.iloc[-1].data_objective/history.iloc[0].data_objective),
        initial_residual_rms_microstrain=float(history.iloc[0].residual_rms_microstrain),
        after_pilot_residual_rms_microstrain=float(np.sqrt(np.mean(final.loc[active,'misfit_values']**2))*1e6),
        observation_peak_microstrain=float(final.loc[active,'measurement_values'].max()*1e6),
        simulation_peak_microstrain=float(final.loc[active,'simulation_values'].max()*1e6),
        evaluations_total=len(pd.read_csv(RUN/'objective_history.csv')),
        stopped_for_user_review=True)
    (RUN / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig, axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axes[0].semilogy(history.iteration, history.data_objective, 'o-')
    axes[0].axhline(zero_objective,color='.5',ls=':',label='Zero-strain prediction')
    axes[0].legend()
    axes[0].set(xlabel='Accepted iteration (0 = initial)',ylabel='Data objective',title='Pilot misfit history')
    for j,name in enumerate(['A','B','C']):
        axes[1].plot(history.iteration,history[f'theta_{j+1}'],'o-',label=name)
    axes[1].axhline(lo,color='.5',ls='--',label='Lower bound / background')
    axes[1].set(xlabel='Accepted iteration',ylabel='alpha = log10(kxx / m²)',title='Three fixed-zone amplitudes')
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(history.iteration)
        ax.grid(alpha=.2)
    fig.savefig(RUN / 'pilot_history.png',dpi=160)
    fig.savefig(RUN / 'pilot_history.pdf')
    plt.close(fig)
    lines = ['# 背景 −21：最多 5 次迭代试跑', '',
             f"实际接受 **{result['iterations']} 次迭代**；{result['message']}。",
             '已停止，等待用户决定后续工作。未自动进入下一轮优化。', '',
             '## 迭代记录', '',
             '| 接受迭代 | α A | α B | α C | 数据误差 | 残差 RMS [με] |',
             '|---:|---:|---:|---:|---:|---:|']
    for row in history.itertuples():
        lines.append(f'| {row.iteration} | {row.theta_1:.10f} | {row.theta_2:.10f} | {row.theta_3:.10f} | {row.data_objective:.7e} | {row.residual_rms_microstrain:.5f} |')
    lines += ['', '## 当前拟合', '',
              f"- 相对新背景下的初始模型，数据误差下降 {summary['data_objective_reduction_fraction']:.2%}。",
              f"- 全零应变预测的数据误差为 {zero_objective:.7e}；当前结果相对它的误差改善为 {summary['improvement_over_zero_prediction_fraction']:.2%}（仅由观测计算的比较基准，不是额外物理求解）。",
              f"- 130 个求解时刻上的观测峰值：{summary['observation_peak_microstrain']:.4f} με；模拟峰值：{summary['simulation_peak_microstrain']:.4f} με。",
              f"- 最终投影梯度无穷范数：{summary['projected_gradient_inf']:.6e}。",
              f"- 活动参数触及下界：{summary['at_lower_bound']}（A、B、C 顺序）。", '',
              '优化器的停止状态和数据拟合质量需分开判断。若触及下界，应将结果解释为当前模型和约束下的边界解。', '',
              '## 保留设置', '',
              '- 背景 kxx=kyy=kzz=1e-21 m²；五个区的异常仅在 kxx 上。',
              '- 活动区范围 [-21,-12]；L1 参考值 -21，beta=5.049311379654719e-15，delta=0.05。',
              '- 另外两个区的 kxx 固定在 alpha=-15.5；位置、宽度、压力源和力学参数不变。',
              '- 同一 45 通道正值化观测、130 时刻、129 正演步、240 步上限、20 MPI ranks 和 0.001 容差。',
              '- 复用新背景下已完成的初始数据目标和数据梯度，只重算改变参考值后的正则项；未运行有限差分验证。',
              '- 限制的是接受迭代数，线搜索评估单独记录；提前收敛则少于 5 次。', '',
              '[当前拟合图](qc_after_pilot.png) · [迭代历史图](pilot_history.png) · [参数表](permeability_after_pilot.csv) · [接受迭代 CSV](accepted_iteration_history.csv)', '',
              '各次接受迭代图为 `qc_iter_00.png` 至最后一次迭代。原始观测报告和梯度保存在 `iterations/`，各试探评估保存在 `evaluations/`。', '']
    (RUN / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
