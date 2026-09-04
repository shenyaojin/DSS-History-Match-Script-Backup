# v6 · 2 参数分区反演 · 三因子敏感度网格

## 一句话

在 v5 的**降维分区反演**（正演不变、区域几何已知、只反演 2 个标量 `θ=[θ_frac, θ_srv]`）基础上，
做一个 **3 因子 × 3 档 = 27 格的全因子敏感度网格**，量化三种现实扰动各自与交互的影响：
**真实 DSS 背景噪声**、**白噪声（仪器）**、**区域位置偏离**。

## 三个因子

| 因子 | 含义 | 3 档 | 作用位置 |
|---|---|---|---|
| **A** `bg1/2/3` | **真实 DSS 背景噪声**（真实纹理） | 3% / 6% / 12% × REF = 679 / 1357 / 2715 nε | 数据 |
| **B** `w1/2/3` | **白噪声**（仪器，纯 i.i.d.） | 3% / 6% / 12% × REF = 679 / 1357 / 2715 nε | 数据 |
| **C** `s0/s2/s4` | **区域位置偏离**（绑定层错位） | 0 / 2 / 4 层 = 0 / 1 / 2 m | 反演 |

`REF = 2.262e-05`（各道时间峰值的道间中位数，与 v3 median / v4 rfsdss / v5 同一锚点，可横向比较）。

**关键结构简化**：A、B 在**数据**里（9 个带噪数据集 = 3×3），C 在**反演**里（绑哪些层）。
所以只需生成 **9 个数据集**，每个再跑 C 的 3 档 → **27 个反演**，且正演模型一字不改。

## 因子 A 为什么是"真实纹理 + 缩放"（重要设计说明）

背景噪声取自真实观测应变瀑布 `output/das_observed/das_strain_waterfall_T1T3.npz`
（45 道 × 4620 时间，单位 millistrain，dt=60 s，道间距 2.047 m），
取**注入前头 ~40 分钟**（时间索引 1–41）× 全 45 道作为背景窗口（信号≈0）。

- 单位换算：millistrain × 1e-3 → 无量纲 strain，与合成 `strain_yy`（峰值 3.6e-5）同尺度。
- **实测背景幅值**：逐道去线性趋势后 σ ≈ **4.4e-8 strain**（含窗内慢漂移的原始 σ ≈ 2e-7）。

**问题**：真实背景原生幅值只有信号的 ~0.1%，而 v5 已证明该反演在 ≤5% 噪声下几乎不敏感 —— 
原生幅值的真实背景对反演**零影响**，做成 3 档毫无意义。

**做法**：保留真实噪声的**性质/纹理**（低频漂移、时间相关性——这正是它区别于白噪声之处），
但把幅值缩放到有意义的 3 档，并**与白噪声档位对齐**，使网格真正隔离出
"噪声**性质**（真实结构化 vs 白）× 幅值 × 位置误差"三者的影响。

实现（`noise_adding/add_noise_grid.py`）：每个输出道随机取一条真实道的去趋势残差，
线性重采样到 130 个合成时间步 → 保留真实的逐道时间相关结构 → 归一化后按目标 σ 缩放。

**验证**：`noise_data/grid_noise_qc.png` 面板 (e) 的功率谱显示
**A 是"红噪声"（低频主导、衰减 2 个数量级），B 完全平坦** —— 两者性质分明，设计成立。

## 数据集与噪声量级

9 个数据集 `noise_data/measurement_data_bg<i>_w<j>.csv`，总噪声 std：

| | w1 (679nε) | w2 (1357nε) | w3 (2715nε) |
|---|---|---|---|
| **bg1** (679nε) | 956 nε | 1512 nε | 2785 nε |
| **bg2** (1357nε) | 1505 nε | 1903 nε | 3025 nε |
| **bg3** (2715nε) | 2767 nε | 3005 nε | 3814 nε |

跨度 ~4% → ~17% × REF，正好横跨 v5 测出的敏感阈值（≤5% 不敏感、10% 明显偏置）。

## 真值（对比基准）

与 v3/v4/v5 完全一致：背景 −18；低 SRV 区 y∈[−20,−16]（层 61–68）= **−15.0**；
裂缝区 y∈[14,20]（层 129–140）= **log10(3e-15) ≈ −14.5229**。
即 **θ_frac 目标 −14.5229，θ_srv 目标 −15.0**。

## 目录结构

```
perm5layer_100_v6strain_zonalgrid/
├── README_grid_workflow.md
├── data/obs_strain_yy.csv(.meta)          干净合成观测（130 时刻 × 500 道）
├── noise_adding/add_noise_grid.py         生成 9 个数据集 + QC + summary
├── noise_data/
│   ├── measurement_data_bg<i>_w<j>.csv(.meta)   9 个带噪数据集
│   ├── grid_noise_summary.csv                   各档 σ 与总噪声统计
│   └── grid_noise_qc.png                        噪声性质 QC（含功率谱对比）
└── inv/
    ├── _template/                          被 setup 拷入各格的模板
    │   ├── 107_optimization_runner_zonal_L1.py   2 参数 runner（含 strip_exodus 提速）
    │   ├── optimize.i / forward_and_adjoint.i    MOOSE，与 v4/v5 逐字节相同（不改）
    │   └── plot_inversion_qc.py / run_parameter_history_qc.py
    ├── setup_grid_runs.sh                  暂存 27 个格子
    ├── launch_grid.sh                      ★ 断点续跑启动器（只跑未完成的格子）
    ├── run_all_grid.sh                     有界并行调度（PAR 路 × ZONAL_NP 进程）
    ├── compare_grid_results.py             汇总表 + 敏感度热图 + 主效应图
    └── bg<i>_w<j>_s<k>/                    27 个 run 文件夹
```

## 运行（★ 需在沙箱外，MOOSE 要 MPI）

```bash
cd scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100_v6strain_zonalgrid

# 1. 生成 9 个带噪数据集 + QC（纯 python，沙箱内可跑）
python noise_adding/add_noise_grid.py

# 2. 暂存 27 个格子（纯拷贝）
bash inv/setup_grid_runs.sh

# 3. 跑网格 —— 用断点续跑启动器，完全脱离会话
cd inv && setsid nohup bash launch_grid.sh > grid_master.log 2>&1 < /dev/null &
#    默认 PAR=2 ZONAL_NP=10（2 路 × 10 进程 = 20 物理核）——实测最优，见下方"坑"第 3 条
#    中断后直接重跑同一条命令即可续跑，已完成的格子不会重做

# 4. 汇总出图（纯 numpy，沙箱内可跑；未跑完的格子自动跳过）
python inv/compare_grid_results.py
```

### ⚠️ 运行注意（踩过的坑）

1. **不要用 tmux 起长任务**：父进程（Claude/终端会话）退出会把 tmux 会话一起带走，
   曾因此白白空转 ~12.5 小时。**用 `setsid nohup ... < /dev/null &`** 让任务成为独立
   session leader。可用 `ps -o sid= -p <pid>` 确认其 sid 与父 shell 不同。
2. **不要把文件夹列表通过命令行变量传进 setsid/nohup 链**：该环境下未加引号的变量展开
   不做词分割，25 个名字会被当成 1 个参数。`launch_grid.sh` 在脚本内部用 **bash 数组**
   构造列表并以 `"${TODO[@]}"` 传递，已规避。
3. **并行度 —— 用 `PAR=2 ZONAL_NP=10`（已实测，别乱调）**：机器是 2×10 核 Xeon（20 物理核 / 40 超线程）。
   `PAR × ZONAL_NP` 不要超过 20。更重要的是**每格进程数有甜点**，实测单次 MOOSE 求解耗时：

   | 每格进程数 | 单次求解 | 2 路/4 路并行吞吐 | 25 格 ETA |
   |---|---|---|---|
   | 5 | ~72 min | 4×5 → 0.28 格/h | 3.7 天 |
   | **10** | **~27 min** | **2×10 → 0.40 格/h** | **2.6 天** |
   | 20 | ~35–43 min | 1×20 → 更差 | — |

   10→5 进程是 **2.7× 减速**（远超线性，MUMPS 在低进程数下缩放很差）；10→20 又因通信开销变慢。
   所以**"多开几路小任务"是错的**——曾按此把配置改成 4×5，实测反而慢 1.1 天，已回退。
4. **exodus 已关闭**：runner 的 `strip_exodus()` 会从 quiet forward 里删掉 `[exodus]` 块
   （保留 `[csv]`）。反演从不读 exodus，但它每次求解要写 ~3 GB，关掉是纯收益。

## 输出与解读

每个格子 `inv/bg<i>_w<j>_s<k>/`：
- `optimized_theta_zonal.txt` —— 最终 [θ_frac, θ_srv]（**主结果**）
- `optimized_alphas_L1.txt` —— 展开的 200 层 alpha（供 compare / QC 复用）
- `theta_history_zonal.csv` / `objective_history_zonal.csv` —— 收敛轨迹

汇总（`inv/` 顶层，由 `compare_grid_results.py` 产出）：
- `grid_comparison_summary.csv` —— 每格的 θ、误差、combined_err
- **`grid_sensitivity_heatmaps.png`** —— **2×3 面板**：行 = {|θ_frac 误差|, |θ_srv 误差|}，
  列 = 位置偏移档（s0/s2/s4），每行**独立色标**。**这是网格的主图**。
  之所以拆成两行而不是画一个合并误差：实测两个参数响应**截然不同**——
  区域错位时 θ_frac 会崩（可达 2.3 dex）而 θ_srv 一直很小（≤0.3），
  合并成一个范数会把这个核心结构抹掉。
- `grid_main_effects.png` —— 每个因子的主效应，**两个参数各一条曲线**（对其余两因子求平均）

**误差度量**：`abs_frac_err`、`abs_srv_err`（dex，log10 渗透率单位）；
CSV 里同时保留 `combined_err = sqrt(frac_err² + srv_err²)` 供参考。

## 与 v3 / v4 / v5 的关系

同一台正演 + 同一把标尺（合成真值、评价指标）。

| 研究 | 问的问题 | 反演参数 |
|---|---|---|
| v3 | 均匀白噪声下**层状**反演坏多快 | ~100 自由层 |
| v4 | 真实感 RFS-DSS 噪声下**层状**反演坏多快 | ~100 自由层 |
| v5 | 降维到 2 参数、区域已知后，还怕噪声吗 | **2 标量** |
| **v6** | **真实背景噪声 × 白噪声 × 位置误差 三者各自与交互的敏感度** | **2 标量** |

**v5 已知结论**（本网格的先验）：exact 几何下，噪声 ≤5% 时 θ 偏差 ≤0.06 dex（基本不敏感），
10% 时跳到 0.45 dex；而**几何错位 +2 层就能让 θ_frac 偏 0.48 dex**——
提示**位置误差可能是比噪声更强的因子**，这正是 v6 网格要定量确认的。

---

*正演零改动；2 参数梯度由 MOOSE 逐层伴随梯度在区内求和得到（链式法则已对着源码核验 + 有限差分对拍）。*
