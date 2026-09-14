# RFSDSS 固定区间渗透率试验反演

仅在本目录新建运行；没有修改旧合成反演目录或数据源。

- 两组：unrectified（去漂移原样）、half_wave_positive（观测和正演均做 max(ε,0)）。
- 数据：POW-S，14700–14750 ft，0–40000 s，150 道×346 个原生时刻。首帧保留，负值组以外无额外变换，με×1e-6 转应变，无×6 校准。
- 五区独立参数，位置按此前候选固定；初值 [-15.5,-15.5,-15.5,-16.5,-16.5]，α=log10(k/m²)，背景固定 -18。优化界限 [-18,-12]。详细对应见各组 config.json。
- 沿用 v6 的 2D 耦合前向及伴随模型、材料参数和 100m×100m 域。200×200 矩形单元；将相邻层边界移动至候选区间的精确位置，区间本身不取整、不参与优化。每条观测的 strain_yy 为所在单元常数；伴随力使用该单元上下边的中点与实际层厚，而非旧模板固定 0.5m 差分。
- 模型纵坐标 y=50+(MD−14725)×0.3048 m。井间距取旧实测 baseline_model_generator.py 的默认 80ft（24.384m），这不是本区间的轨迹实测值；注入线 x=45m，观测 x=69.384m。
- 压力使用 prod/gauges/pressure_g1.npz，按 start_time 对齐 DSS，减去 DSS t=0 压力作增量边界。保留窗口内原生压力采样，求解时刻与全部 DSS 观测时刻完全一致。该窗口生产井压力约 1838–1844psi，尚未包含后期主要压力变化。
- L-BFGS-B + 轻微 smoothed L1。L1 系数按实测/合成观测平方和比例缩放，保持其相对量级；两组使用同一系数和目标缩放。
- 每组开始先做初始正演及两组方向中心差分检查，检验实际目标函数和伴随梯度。检查失败直接中止，不启动后续优化。模拟失败同样中止，不返回伪零梯度。
- 两组各 10 MPI 核后台运行。当前主机 40 核，另有旧 bg3_w3_s0_v2 占用 20 核；不终止旧任务。

每组产出 initial/final QC、gradient_check.json、objective_history.csv、optimized_theta.txt、optimized_alpha.txt、permeability_results.csv 和 optimizer_result.json。只有优化成功且最终 QC 完成才写 COMPLETE。两组均成功后生成 COMPARE_rectification.md。

这是给定几何、校准和早期时间窗下的试验性估计，绝对渗透率不能脱离这些假设解释。无真实渗透率真值，不报告“真值误差”。


## 2026-09-14 QC 校核修复

两组均已完成。旧的 qc_initial.png/strain_initial.csv 曾因伴随倒序输出选择而误取零时刻，不应使用；初始完整输出已被覆盖。最终 strain_final.csv 与 qc_final.png 已从完整正演记录恢复并按日志目标值校验。用户交付图及完整 L1 设置说明见 fig_deliverable/RESULTS_AND_L1.md。
