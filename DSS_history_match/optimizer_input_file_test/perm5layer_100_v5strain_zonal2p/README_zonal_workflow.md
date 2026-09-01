# v5 · 降维分区反演（2 参数 zonal L1）· 噪声不敏感性研究

## 一句话

**正演/伴随 MOOSE 完全不改**；反演时**假定裂缝/SRV 的区域几何已知**（可从 DSS 读出），
把待反演量从"窗口内 100 个自由层"砍到**只有 2 个标量** `θ = [θ_frac, θ_srv]`——
裂缝区所有层共用一个 log10 渗透率、SRV 区所有层共用一个、其余层钉死在背景 −18。
本研究在**全部 14 个噪声档**上、用**精确/扰动两套区域几何**测试这个策略是否对噪声不敏感。
**预期：几乎完全不敏感**（65000 个观测约束 2 个未知量，严重超定）。

## 为什么这样能对噪声不敏感

层状反演有 ~100 个自由度，高噪声下会把噪声"雕"进各层，出现虚假活跃层（见 v3/v4 研究：
rfsdss 高档 `max_alpha_err` 冲到 3.5）。本策略把自由度压到 2，且强先验（区域位置已知）
让问题严重超定——噪声在 65000 个点上对 2 个标量的平均效应趋近于零。

## 核心：怎么在"正演不动"的前提下只反演 2 个参数（已对着源码核验）

MOOSE 仍暴露 200 个逐层参数 `perm_1..perm_200`。桥接全在 Python 外层驱动
`107_optimization_runner_zonal_L1.py` 里：

```
expand(θ) -> 200 维 alpha        # frac 层=θ_frac, srv 层=θ_srv, 其余=-18
                                 # 写进 optimize.i 的 initial_condition
MOOSE 一次前向+伴随 -> 200 个 grad_perm_i = ∂J_data/∂alpha_i
reduce_grad(grad200) -> 2 维      # 区内求和：∂J/∂θ_zone = Σ_{i∈zone} grad_perm_i
```

**链式法则恒等**（本仓 workflow 对抗性核验 CONFIRMED）：因为区内所有 `alpha_i ≡ θ_zone`，
故 `d(alpha_i)/d(θ_zone) = 1`，于是 `dJ/dθ_zone = Σ_{i∈zone} grad_perm_i`。
`10^alpha·ln10` 的链式因子在 MOOSE 侧（`ParsedOptimizationFunction('10^alpha')` +
`OptimizationFunctionInnerProductHelper`）已经算好，`grad_perm_i` 本身就在 **alpha 空间**，
所以 Python 侧直接求和、不再乘任何因子。数值上已用"区内求和 vs 对 θ 的有限差分"对拍，
误差 ~1e-7。

> L1 正则仍保留（`BETA_L1=2e-11`），在展开的 200 维上算平滑 L1 再按区约简；但 2 参数下
> 它相对数据项 ~1e-5，**近乎不起作用**，仅作与层状 L1 研究的一致性延续。

## 精确 vs 扰动 两套

- **exact**：裂缝 y∈[14,20]（层 129–140，12 层）、SRV y∈[−20,−16]（层 61–68，8 层），
  与合成真值完全一致。→ 纯粹测"对噪声是否敏感"。
- **pert**：把两区 y 窗口**同向整数层平移**（默认 +2 层 = +1.0 m）→ 裂缝层 131–142、SRV 层 63–70。
  被绑定的层与真值活跃层错位，恢复的 θ 会被稀释/偏置。→ 顺带测"从 DSS 读区域有误差"的鲁棒性。
  平移量由环境变量 `ZONAL_PERT_SHIFT_LAYERS`（默认 2）控制。

## 真值目标（读表基准）

| 参数 | 真值 | 区域 |
|---|---|---|
| θ_frac | log10(3e-15) ≈ **−14.5229** | 裂缝 y∈[14,20] |
| θ_srv | **−15.0** | 低 SRV y∈[−20,−16] |
| 背景 | −18.0（钉死，不反演） | 其余全部 |

## 目录结构

```
perm5layer_100_v5strain_zonal2p/
├── README_zonal_workflow.md
├── data/
│   └── obs_strain_yy.csv(.meta)              干净真值观测（130×500）
├── noise_data/                               14 份带噪观测（复用 v3+v4，不重新加噪）
│   ├── measurement_data_clean.csv
│   ├── measurement_data_peak_{0p5,1,2,5}pct.csv       (v3 peak)
│   ├── measurement_data_median_{1,2,5,10}pct.csv      (v3 median)
│   └── measurement_data_rfsdss_{0p5,1,2,5,10}pct.csv  (v4 rfsdss)
└── inv/
    ├── _template/                            被 setup 拷入各 run 文件夹的模板
    │   ├── 107_optimization_runner_zonal_L1.py   ★ 2 参数外层驱动（核心新件）
    │   ├── optimize.i / forward_and_adjoint.i    MOOSE，与 v4 逐字节相同（不改）
    │   ├── plot_inversion_qc.py / run_parameter_history_qc.py
    ├── setup_zonal_runs.sh                   暂存 28 个 run 文件夹
    ├── run_all_zonal.sh                      批量跑 2 参数反演
    ├── run_qc_zonal.sh                       批量前向 QC
    ├── compare_zonal_results.py              汇总对比 + 出图
    ├── exact_<case>/   (14 个)               ZONAL_MODE=exact 的 run 文件夹
    └── pert_<case>/    (14 个)               ZONAL_MODE=pert 的 run 文件夹
        <case> ∈ {clean, median_{1,2,5,10}pct, peak_{0p5,1,2,5}pct, rfsdss_{0p5,1,2,5,10}pct}
```

## 运行（★ run_all / run_qc 需在沙箱外，MOOSE 要 MPI）

```bash
cd scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100_v5strain_zonal2p

# 1. 暂存 28 个 run 文件夹（纯拷贝，沙箱内可跑；已执行过）
bash inv/setup_zonal_runs.sh

# 2. 跑全部 28 个 2 参数反演（每个只 2 参数，收敛快；exact/pert 由文件夹名自动传 ZONAL_MODE）
bash inv/run_all_zonal.sh
#    子集： bash inv/run_all_zonal.sh exact_*        # 只跑精确那 14 个
#           bash inv/run_all_zonal.sh exact_rfsdss_10pct pert_rfsdss_10pct

# 3. 前向 QC（可选，出 qc_strain_zonal_final.png）
bash inv/run_qc_zonal.sh

# 4. 汇总对比（纯 numpy，沙箱内可跑）
python inv/compare_zonal_results.py
```

## 输出文件

每个 run 文件夹产出**两套**（自描述 θ + 兼容 200 维 alpha）：

| 类别 | 文件 | 说明 |
|---|---|---|
| 自描述 θ | `optimized_theta_zonal.txt` | 最终解 [θ_frac, θ_srv] |
| 自描述 θ | `best_data_theta_zonal.txt` / `best_total_theta_zonal.txt` | 最优 θ |
| 自描述 θ | `theta_history_zonal.csv` / `gradient_history_zonal.csv` / `objective_history_zonal.csv` | 迭代史 |
| 兼容 200 | `optimized_alphas_L1.txt` / `best_*_alpha_L1.txt` | 展开 200 维，供 compare/QC |
| 兼容 200 | `parameter_history_L1.csv` | 每行 200 列展开 alpha，供 run_parameter_history_qc.py |

汇总（`inv/` 顶层）：
- `zonal_comparison_summary.csv` —— 每行 (variant, level, which) 的 θ、误差、区均值、rel-L2。
- `zonal_alpha_overlay.png` —— 各 run alpha 剖面 vs 真值（实线=exact，点划线=pert）。
- **`zonal_theta_vs_noise.png` —— 关键图**：恢复的 θ_frac/θ_srv 随噪声档变化。
  **若策略真不敏感，这两条线应基本水平贴在真值上**。

## 关键参数（`107_optimization_runner_zonal_L1.py` 顶部，均可环境变量覆盖）

| 变量 | 默认 | 含义 |
|---|---|---|
| `ZONAL_MODE` | exact | exact / pert（run_all 按文件夹名自动传） |
| `ZONAL_PERT_SHIFT_LAYERS` | 2 | pert 模式区边界平移层数（1 层=0.5 m） |
| `ZONAL_FRAC_INIT` / `ZONAL_SRV_INIT` | −16 / −16 | θ 初值 |
| θ bounds | [−25, −10] | 必须落在 MOOSE 每层界内，否则 taobqnls 静默裁剪破坏链式法则 |
| `BETA_L1` / `DELTA_L1` | 2e-11 / 0.05 | 平滑 L1（2 参数下近乎可忽略） |
| `ZONAL_MAXITER` | 60 | L-BFGS-B 最大迭代（2 自由度收敛极快） |
| `ZONAL_NP` | 20 | MOOSE MPI 进程数 |
| `ZONAL_USE_STEP_STOP` | 0（关） | 步长停机；2 维下 L2 判据冗余，开启则只用 Linf |

## 与 v3/v4 的关系

同一台正演/伴随、同一把标尺（合成真值、评价指标、clean 参照）。v3 问"均匀白噪声下**层状**反演坏多快"，
v4 问"真实 RFS-DSS 噪声下**层状**反演坏多快"，**v5 问"若把自由度降到 2、区域已知，是否就不怕噪声了"**。
噪声数据直接复用 v3/v4 已生成的 14 份，未重新加噪。

---

*正演零改动；梯度区内求和的链式法则已对着 MOOSE 源码对抗性核验（CONFIRMED）并用有限差分数值对拍通过。*
