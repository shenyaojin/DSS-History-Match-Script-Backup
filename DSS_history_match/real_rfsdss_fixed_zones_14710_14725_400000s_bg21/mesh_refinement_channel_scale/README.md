# DSS 通道尺度的局部网格加密

本试验按 2026-09-20 用户要求加密观测区深度方向网格。参考为
`../comparison_xx_yy_60/fixed_yy` 的最终模型，先计算一次固定参数响应。

- x 保留 200 个单元；y 从 200 增至 242 个单元，总单元数 40,000 → 48,400。
- 原材料块 layer_92–layer_100 与 14710–14725 ft 裁剪区相交，仅细分这些块。
- 使用实际最小通道间距 0.101203125 m 作为单元高度上限，块内均匀细分后为 0.087086–0.100000 m。
- 45 个观测通道分别对应 45 个不同单元，映射见 channel_element_mapping.csv。
- 原材料块边界、五个高渗区的位置和宽度不变；仍为 200 个材料块，细网格单元继承所在块的参数。
- 固定 alpha_xx = [-17.762985193845356, -17.677889678148734, -18.407556915150288]；alpha_yy = [-18, -17.5, -19]。
- 两个外围固定高渗区、所有背景分量和 kzz 与参考模型相同。
- 力学边界、流动储集、压力源、材料性质、x 网格、观测数据、目标函数均保留。
- 130 个输出时刻 / 129 步，上限 240，隐式 Euler、0.001 容差、LU/MUMPS、20 MPI 均保留。

为保持现有评估框架，运行一次既有正演/伴随评估；不调用 L-BFGS-B，不运行有限差分。
伴随的 y_up/y_down/inv_dy 已按新单元更新；仍沿用原 x_center。梯度仅作为计算产物保存，未经新网格有限差分验证，不用于新一轮反演。

增加 signed_data 观测输出，只采样原始有符号 strain_yy；该 reporter 不参与优化目标或伴随源。
生成 strain_refined.csv（正半波）及 strain_signed.csv（有符号），完成后生成
mesh_comparison.png、zone_curves_signed.png、comparison_summary.json 和 COMPLETE 标记。

本试验首先检查加密对响应的影响；即使每通道有独立单元，也不保证固定分区和当前物理模型能够产生观测中的所有细节。

准备、检查与运行分别使用 `../run_mesh_refinement.py prepare/check/run`；脚本拒绝覆盖已经存在的试验和重复求解。
preflight.json 记录输入哈希及静态一致性检查。config.json 的 mesh_edges_m 和 layer_indices 仍指材料块，mesh_element_edges_m 才是加密单元边界。
