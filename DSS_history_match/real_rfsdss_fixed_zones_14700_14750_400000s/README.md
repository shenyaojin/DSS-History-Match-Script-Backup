# 当前任务：只看初始 α 正演

用户已明确改为先检查真实数据 vs 初始猜测的模拟。400000s 两组反演已停止，未进入参数优化。现在仅运行 run_initial_case.sh unrectified / run_initial_forward_only.py，用固定原始初值直接做一次 Transient 正演；不进行伴随求解、梯度检查或优化。两种处理的物理模型完全相同，半波整流图直接由同一正演结果取max(strain,0)得到。

结果目录：initial_forward_only/unrectified 和 initial_forward_only/half_wave_positive。每组产出 observed_vs_initial_simulation.png/.pdf；两组完成后汇总 initial_forward_only/observed_vs_initial_simulation_both.png。

数据仍是 0≤t≤400000s、150道×3318个原生时刻；实际末时刻384301.087267s。初始参数为[-15.5,-15.5,-15.5,-16.5,-16.5]，背景-18。L1不参与这次固定参数正演。

当前求解设置（2026-09-14 修正）：采用 V6 模板的 130 个时刻 / 129 步，只把最后时刻从 384288 调至真实数据终点 384301.087267 s。硬上限 num_steps=240，失败即停止，不自动切碎时间步。nl_abs_tol=nl_rel_tol=l_tol=0.001，l_max_its=2000、nl_max_its=200，沿用 LU/MUMPS。初始 α、几何、网格、材料和真实压力源未改变。原 3317 步任务已停止，文件保存在 initial_forward_only/stopped_3317_steps。

OptimizationData 仅在精确匹配的时间采样，因此独立生成 130 时刻的 measurement_data.csv（原始带符号观测线性插值，半波版本再取正）。左侧观测图仍显示完整 3318 个原生样本，右侧显示 130 个求解时刻。两者时间轴分别保存在 comparison_arrays.npz 的 observed_time 和 time。原始数据文件未改。运行时实时输出时间步及收敛日志。

---
以下是被用户取消的反演准备记录，仅供追溯。

# RFSDSS 时间窗更正：0–400000 s

用户确认前一轮 40000s 少写一个0。本目录是独立重跑；原 real_rfsdss_fixed_zones_14700_14750 的短窗口结果保留不覆盖。

## 数据及不变设置

- 直接从只读 POW-S RFS strain change.npz 重截取 14700–14750ft、0≤t≤400000s。
- 保留全部150道×3318个原生时刻，无时间插值或下采样。文件在384301.087267s之后下一记录为403137.828611s，故本轮实际计算终点为384301.087267s，不向400000s外推。
- 减去7500<MD<15000ft逐时刻中位数，με×1e-6，无额外校准倍数。前40000s数据逐值与前一轮一致。
- 不整流与半波整流两组；整流组观测及正演均采用max(strain,0)，梯度带有相应门控。
- 同样的五个固定区间、初值[-15.5,-15.5,-15.5,-16.5,-16.5]、背景-18、范围[-18,-12]；位置和宽度不优化。
- 生产井pressure_g1.npz按DSS start_time对齐，减t=0压力作为模型边界。本轮包含约70000s以后主要压力上升，以及后期下降。
- 同样100m×100m模型、200×200单元及材料参数；井间距仍取旧实测模板默认80ft，未另作轨迹校准。

## 优化与输出

算法、目标及L1设置规则沿用上一轮：delta=.05，alpha_ref=-18，beta=2e-11×实测/干净合成数据能量比，两组共用不整流数据算得的beta和目标缩放。更长时间窗的beta数值自动变为2.687055045421127e-13；完整配置见各组config.json。

每组先初始正演及两次方向梯度检查（共4次差分探针），通过才进入L-BFGS-B优化。两组各20 MPI核，后台运行，所有原生时间步保留，因此耗时显著长于上一轮。

此次明确将前向CSV输出设为FINAL，避免每个时刻重复输出整个观测向量。正演结果选择和目标值校核沿用已修复的find_report，不按伴随倒放的修改时间选文件。每次accepted callback保存accepted_iteration_history.csv，可画真实的已接受优化迭代曲线。

每组自动输出初始/最终QC、梯度验证、参数历史、optimized_theta.txt和permeability_results.csv。两组成功后自动汇总fig_deliverable内的观测vs模拟、观测+渗透率和misfit曲线。
