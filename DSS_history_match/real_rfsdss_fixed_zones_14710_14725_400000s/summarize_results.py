"""Summarize completed inversion artifacts without launching another solve."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = HERE / 'inversion'


def metrics(path):
    frame = pd.read_csv(path)
    active = frame[frame.measurement_time > 0]
    residual = active.simulation_values - active.measurement_values
    np.testing.assert_allclose(residual, active.misfit_values, rtol=1e-8, atol=1e-18)
    return dict(data_objective=float(.5*np.sum(residual**2)),
                residual_rms_microstrain=float(np.sqrt(np.mean(residual**2))*1e6),
                observation_rms_microstrain=float(np.sqrt(np.mean(active.measurement_values**2))*1e6),
                observation_peak_microstrain=float(active.measurement_values.max()*1e6),
                simulation_peak_microstrain=float(active.simulation_values.max()*1e6))


def main():
    if not (RUN / 'COMPLETE').exists():
        raise RuntimeError('Inversion and final QC are not complete')
    result = json.loads((RUN / 'optimizer_result.json').read_text())
    cfg = json.loads((RUN / 'config.json').read_text())
    initial = metrics(RUN / 'strain_initial.csv')
    final = metrics(RUN / 'strain_final.csv')
    history = pd.read_csv(RUN / 'objective_history.csv')
    initial_reg_ratio = float(history.iloc[0].regularization/history.iloc[0].data_objective)
    theta = np.asarray(result['theta'])
    lo, hi = cfg['theta_bounds']
    gradient200 = pd.read_csv(RUN / 'solver_output/evaluate_out_OptimizationReporter_0001.csv')
    raw = gradient200[[f'grad_perm_{i+1}' for i in range(200)]].iloc[-1].to_numpy()
    gradient = np.array([raw[idx].sum() for idx in cfg['zone_layer_indices']])
    deviation = theta + 18
    gradient += cfg['beta_l1']*deviation/np.sqrt(deviation**2+.05**2)
    gradient *= cfg['objective_scale']
    projected = gradient.copy()
    projected[(theta <= lo+1e-10) & (gradient > 0)] = 0
    projected[(theta >= hi-1e-10) & (gradient < 0)] = 0
    summary = dict(optimizer=result, initial=initial, final=final,
                   data_objective_reduction_fraction=1-final['data_objective']/initial['data_objective'],
                   gradient=gradient.tolist(), projected_gradient_inf=float(np.max(abs(projected))),
                   initial_regularization_to_data_ratio=initial_reg_ratio,
                   at_lower_bound=np.isclose(theta,lo,rtol=0,atol=1e-8).tolist(),
                   at_upper_bound=np.isclose(theta,hi,rtol=0,atol=1e-8).tolist(),
                   finite_difference_checks='Skipped at explicit user request; prior validation reported by user')
    (RUN / 'result_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    table = pd.read_csv(RUN / 'permeability_results.csv')
    lines = ['# RFSDSS 三参数反演结果', '',
             f"优化器状态：{result['success']}；{result['message']}。",
             f"接受迭代数：{result['iterations']}；优化器函数评估数：{result['evaluations']}。", '',
             '## 固定深度区间的渗透率', '',
             '| 深度范围 [ft] | 是否优化 | 初始 α | 最终 α | 最终 kxx [m²] |',
             '|---|---|---:|---:|---:|']
    for row in table.itertuples():
        lines.append(f'| {row.lo_ft:g}–{row.hi_ft:g} | {"是" if row.optimized else "否"} | {row.initial_alpha:.6g} | {row.final_alpha:.6g} | {row.permeability_m2:.6e} |')
    lines += ['', '## 拟合程度', '',
              f"- 数据目标函数：{initial['data_objective']:.8e} → {final['data_objective']:.8e}，下降 {summary['data_objective_reduction_fraction']:.2%}。",
              f"- 残差 RMS：{initial['residual_rms_microstrain']:.4f} → {final['residual_rms_microstrain']:.4f} με。",
              f"- 观测峰值：{final['observation_peak_microstrain']:.4f} με；最终模拟峰值：{final['simulation_peak_microstrain']:.4f} με。",
              f"- 最终投影梯度无穷范数：{summary['projected_gradient_inf']:.6e}。", '',
              f"初始 L1 正则项仅为数据项的 {initial_reg_ratio:.4e} 倍；当前正则权重远小于数据误差。", '',
              '优化器收敛描述的是当前固定模型及参数边界下的停止条件；拟合质量应结合上述残差和图判断。',
              '若参数停在边界，应按受限优化结果解释，不能据此认为三个区的真实渗透率已被准确确定。', '',
              '## 数据和设置', '',
              '- 原始数据：S 井 `Mariner 14x-36-POW-S - RFS strain change.npz`。',
              '- 使用既有中位数漂移校正；14710–14725 ft；最后观测 384301.087267 s。',
              '- 原生数据 45 × 3318；负应变置零后插值到 45 × 130，保留零观测，初始帧不计误差。',
              '- 129 个正演时间步，最大 240；20 MPI ranks；LU/MUMPS；三个求解容差均为 0.001。',
              '- 只优化三个区的 kxx 幅值；位置、宽度、另外两个区、背景、压力源和物理参数保持不变。',
              '- 按用户更新指示跳过重复有限差分验证，复用完成的初始伴随梯度。',
              f"- L1：每个活动区一项；beta={cfg['beta_l1']:.12e}，参考 alpha=-18，delta=0.05。", '',
              '[最终拟合图](inversion/qc_final.png) · [参数表](inversion/permeability_results.csv) · [逐次评估](inversion/objective_history.csv) · [接受迭代](inversion/accepted_iteration_history.csv)', '']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
