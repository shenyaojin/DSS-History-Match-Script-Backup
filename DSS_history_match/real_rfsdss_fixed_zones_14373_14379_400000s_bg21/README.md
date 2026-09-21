# 14376 ft 固定分区双参数反演

用户已批准 PROPOSED_PARAMETERS.md 所示初值，并于 2026-09-20 要求直接反演。提案文件保留为批准时的快照；当前运行状态以 inversion/status.json、driver.log 和完成/失败标记为准。

## 固定配置

- 观测：14373–14379 ft，18 通道，3318 原生时刻；先正半波再插值到原有 130 个求解时刻，t=0 不计入误差。
- 固定区 A：14375.442383–14376.780273 ft，宽 0.407789 m。
- 参数：alpha_xx_A、alpha_yy_A，独立求解，同初值 -17.5，界限 [-21,-12]。
- 背景三个方向及全域 kzz 固定 1e-21 m²；不优化位置或宽度。
- 200×216 网格，观测区每个原生通道各对应一个单元；区 A 四个单元、两个材料块共享同一组参数。
- 原压力曲线、100×100 m 域、80 ft 井距、材料、质量储集与力学边界保留。
- 129 步、240 步硬上限、隐式 Euler、0.001 容差、LU/MUMPS、20 MPI。
- 正则参考 -21、delta=0.05、beta=5.049311379654719e-15；每个独立参数一个正则项。objective_scale=308267582.56113124。
- L-BFGS-B：maxiter=60、ftol=1e-7、gtol=1e-8、maxls=20。

## 复用与必要适配

prepare_inversion.py 从已验证的 147xx-ft xx/yy 对照输入生成独立输入：保持 200 个材料分块及原 xx 梯度计算方式；将 yy 分区从三个缩减为一个，MOOSE reporter 槽位从 203 变为 201。SciPy 只优化两个数：第一个广播至 layer_100/101 的 xx 参数，并把两块 xx 梯度相加；第二个为覆盖相同两块的单个 yy reporter。

更新新网格、材料块边界、观测数据和精确单元边缘的伴随源。静态检查验证 Kernels、Materials、BCs、Executioner、Preconditioning、FluidProperties 与参考输入逐字一致；确认 18 个独立观测单元和高渗区四单元总宽度。输入哈希记录于 inversion/preflight.json。

run_inversion.py 按原 field/V6 驱动的评估、梯度提取、正则和缓存方式适配两个参数及 18 通道；没有修改旧驱动。额外保留有符号 strain_yy 输出，核验正半波导出与目标一致。

## 自动执行顺序

1. MOOSE --check-input 通过后才能启动。
2. 初始正演/伴随，保存 iter_00 结果和 QC。
3. 两个独立方向 [1,0]、[0,1]，h=0.02，中心差分与数据项伴随梯度比较。沿用绝对 1e-7 或相对 5% 的通过判据；正则项不参与该检查，避免掩盖映射错误。
4. 两方向通过后自动进入 L-BFGS-B，最多 60 次接受迭代。任一求解、目标一致性或梯度检查失败即停止。
5. 完成后保存最终参数、应变与 QC；不自动重启，不变更模型或放松检查。

新目录不受旧网格试验的“同参数误差必须下降”触发条件限制；用户已经独立批准本新案例。

## 查看结果

- inversion/status.json：当前阶段及正在评估的参数。
- inversion/driver.log：完整驱动进度。
- inversion/gradient_check.json：每个方向的检查结果。
- inversion/objective_history.csv：包括有限差分试算在内的每次求解。
- inversion/accepted_iteration_history.csv：仅初始点和接受迭代。
- inversion/qc_iter_XX.png：左侧观测/模拟 waterfall，中间渗透率，右侧逐通道观测和模拟曲线；有符号模拟用虚线显示。
- inversion/iterations/XX/：每次接受迭代的正半波/有符号应变与梯度。
- inversion/optimizer_result.json、theta_final.txt、permeability_final.csv、qc_final.png：最终结果。
- inversion/FAILED 或 COMPLETE：结束状态。

启动脚本 launch_inversion.sh 及驱动均拒绝覆盖已开始的运行。launch_manifest.json 保存实际启动时的驱动与输入哈希。
