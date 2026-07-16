# DSS 噪声敏感性研究 · 图集与原理说明

本文件夹 `fig/` 是把 **两套并列的 DSS（分布式应变传感）噪声敏感性研究** 的所有关键图件和汇总表整理到一处的"成果图集"，并配这份尽量详尽的说明。读完这份文档，你应当能回答三件事：

1. **每一张图是什么**——它画的是什么量、怎么读、说明什么结论；
2. **公用脚本怎么跑**——从干净观测到最终对比图的 5 步流水线、三个 shell 驱动、以及它们的用法与环境要求；
3. **背后的原理是什么**——加噪模型、L1 正则反演、MOOSE 伴随法梯度、评价指标的数学定义。

> 说明对象是下面两套研究（"同一台反演器、只换观测噪声"的姊妹研究）：
> - **v3** `perm5layer_100_v3strain_noise` —— 均匀 i.i.d. 高斯**白噪声**研究（peak / median 两族）
> - **v4** `perm5layer_100_v4strain_rfsdss` —— **真实感 RFS-DSS 结构化噪声**研究（rfsdss 族，本文件夹所属）

---

## 目录

- [0. 这个文件夹的结构](#0-这个文件夹的结构)
- [1. 研究背景：为什么要做两套](#1-研究背景为什么要做两套)
- [2. 共同的物理模型与"真值"](#2-共同的物理模型与真值)
- [3. 端到端工作流与公用脚本](#3-端到端工作流与公用脚本)
  - [3.1 五步数据流](#31-五步数据流一张纵贯全程的链路)
  - [3.2 加噪脚本的原理](#32-加噪脚本的原理第-1-步)
  - [3.3 反演器 106_optimization_runner_L1.py](#33-反演器-106_optimization_runner_l1py第-3-步核心)
  - [3.4 MOOSE：optimize.i 与 forward_and_adjoint.i](#34-moose-求解器optimizei--forward_and_adjointi)
  - [3.5 前向 QC：run_parameter_history_qc.py + plot_inversion_qc.py](#35-前向-qcrun_parameter_history_qcpy--plot_inversion_qcpy第-4-步)
  - [3.6 结果对比：compare_*.py](#36-结果对比compare_py第-5-步)
  - [3.7 三个 shell 驱动脚本](#37-三个-shell-驱动脚本setup--run_all--run_qc)
- [4. 图集逐张说明](#4-图集逐张说明)
  - [4.1 01_noise_generation/ 噪声生成 QC](#41-01_noise_generation噪声生成-qc)
  - [4.2 02_inversion_comparison/ 反演结果对比](#42-02_inversion_comparison反演结果对比)
  - [4.3 03_forward_qc_strain/ 各 run 前向 QC](#43-03_forward_qc_strain各-run-前向-qc)
- [5. 关键数值表](#5-关键数值表)
- [6. 当前运行状态与待补内容](#6-当前运行状态与待补内容)
- [7. 如何复现](#7-如何复现)
- [8. 参数与文件名契约速查](#8-参数与文件名契约速查)

---

## 0. 这个文件夹的结构

```
fig/
├── README.md                         ← 本文件
├── 01_noise_generation/              ← 加噪原理 / 噪声形态 QC
│   ├── v4_rfsdss_qc_0p5pct.png       各等级 8 子图 QC（噪声形态体检）
│   ├── v4_rfsdss_qc_1pct.png
│   ├── v4_rfsdss_qc_2pct.png
│   ├── v4_rfsdss_qc_5pct.png
│   ├── v4_rfsdss_qc_10pct.png
│   ├── v4_rfsdss_levels_summary.png  五档噪声横向对比
│   ├── v4_rfsdss_noise_summary.csv   v4 各成分目标 vs 实现统计
│   └── v3_peakmedian_noise_summary.csv  v3 peak/median 加噪统计
├── 02_inversion_comparison/          ← 反演结果与真值对比
│   ├── v3_alpha_overlay.png          v3 各噪声级 alpha 剖面 vs 真值
│   ├── v3_qc_metrics.png             v3 前向 QC 指标图
│   ├── v3_comparison_summary.csv     v3 对比指标表
│   ├── v3_forward_qc_summary.csv     v3 前向 QC 详细指标表
│   ├── v3_alpha_summary_all_profiles.csv  v3 逐剖面分区统计
│   ├── v4_alpha_overlay.png          v4 各噪声级 alpha 剖面 vs 真值
│   └── v4_comparison_summary.csv     v4 对比指标表
└── 03_forward_qc_strain/             ← 每个 run 的"观测 vs 模拟应变"证据图
    ├── v3_noise_0p5pct__qc_strain.png     (peak 族 4 个)
    ├── v3_noise_1pct__qc_strain.png
    ├── v3_noise_2pct__qc_strain.png
    ├── v3_noise_5pct__qc_strain.png
    ├── v3_noise_0p5pct__alpha_profiles.png (peak 族 alpha 剖面 4 个)
    ├── v3_noise_1pct__alpha_profiles.png
    ├── v3_noise_2pct__alpha_profiles.png
    ├── v3_noise_5pct__alpha_profiles.png
    ├── v3_mednoise_1pct__qc_strain.png    (median 族 4 个)
    ├── v3_mednoise_2pct__qc_strain.png
    ├── v3_mednoise_5pct__qc_strain.png
    └── v3_mednoise_10pct__qc_strain.png
```

> **命名约定**：所有文件都加了 `v3_` / `v4_` 前缀标明出处；`__qc_strain` 表示"观测/模拟/残差/alpha 四联 QC 图"，`__alpha_profiles` 表示反演过程中的 alpha 剖面演化图。**v4（rfsdss）的前向 QC 图（`v4_rfsdss_*__qc_strain.png`）尚未生成**，原因见 [第 6 节](#6-当前运行状态与待补内容)。

---

## 1. 研究背景：为什么要做两套

两套研究回答的是同一个科学问题的两个层次：**测量噪声会把渗透率反演结果破坏到什么程度？**

- **v3** 用最简单的噪声——对每个测点加一个**绝对标准差相同、逐点独立**的高斯白噪声（`d_noisy = d + N(0, std²)`）。它是理想化的"教科书噪声"，用来建立敏感性基线。
- **v4** 用**更贴近真实 RFS-DSS（Rayleigh 频移分布式应变传感）现场数据**的结构化噪声：逐道不同的噪声底、逐道低频漂移、全道共模漂移、以及**跟着光纤应变走的稀疏尖峰**。它回答"当噪声长得像真数据时，反演坏得更快还是更慢"。

两套研究**刻意共用同一台反演器和同一把标尺**：反演代码、初始模型、分区（zone）掩膜、L1 正则设置全部逐字复制自 `perm5layer_100_v2strain/inv`，**唯一的自变量就是"注入什么噪声"**。v4 的噪声幅度还特意锚定到与 v3 median 族相同的参考幅值 `REF = 2.262e-05`，因此两张对比表可以直接横向比较。

---

## 2. 共同的物理模型与"真值"

理解所有图之前，先建立统一的物理语义。

### 2.1 被反演量 alpha

- 未知量是一个 200 维向量 `alpha ∈ ℝ²⁰⁰`，每个分量是**某一水平层的以 10 为底的对数渗透率**：`alpha_i = log₁₀(k_i)`，`k_i` 单位 m²。在对数空间反演使数值条件良好、且渗透率恒正。
- 层几何：`TOTAL_LAYERS = 200`，层高 `LAYER_HEIGHT = 0.5 m`，模型 y 方向覆盖 **y ∈ [−50, +50] m**（共 100 m）。第 i 层 `y_bottom = −50 + i·0.5`。
- **alpha → 渗透率的映射在 MOOSE 侧完成**：每层材料 `k_i = 10^{alpha_i}`。例如 alpha=−18 ⟺ k=1e-18 m²（致密基质），alpha=−15 ⟺ k=1e-15 m²。

### 2.2 合成"真值"模型（所有对比的标尺）

真值 alpha 剖面是三段式分区常数（在 `plot_inversion_qc.py` 与两个 `compare_*.py` 里用 `build_truth_alpha()` / `build_synthetic_truth_alpha()` 硬编码，必须与生成观测时一致）：

| 区域 | y 范围 | 1-based 层号 | alpha 真值 | 渗透率 |
|---|---|---|---|---|
| 背景 / 基质 | 全域默认 | — | **−18.0** | 1e-18 m² |
| 低 SRV 改造区 (`low_srv`) | y ∈ [−20, −16] | 61–68（8 层） | **−15.0** | 1e-15 m² |
| 裂缝区 (`fracture`) | y ∈ [14, 20] | 129–140（12 层） | **log₁₀(3e-15) ≈ −14.5229** | 3e-15 m² |

### 2.3 分区（zone）与自由窗口

反演并不让 200 层都自由变化：

- **自由窗口 `free_window`**：y ∈ [−25, +25]（第 51–150 层，共 100 层）——真正被数据约束、允许自由变化的中心带。
- **窗口外的 100 层**被 L-BFGS-B 的上下界**钉死在 −18**（`make_bounds()` 令 `lb=ub=−18`），不参与反演。
- 窗口内的层，盒式约束为 `alpha ∈ [−25, −10]`。

### 2.4 观测量 strain_yy 与几何

- 观测量是**沿光纤方向的应变** `strain_yy = ∂(disp_y)/∂y`，对应现场 DSS 测到的纵向应变。
- 物理场景：x=45 一条竖直**注水井**（`FunctionDirichletBC` 施加随时间变化的注入压力，峰值 ~2.2e6 Pa），x=60 一条竖直**光纤**测 `strain_yy`。
- 数据规模：**130 个时刻 × 500 个道（接收点）= 65000 个测量点**；y 从 −25 到 +25。干净观测存在 `data/obs_strain_yy.csv`。

### 2.5 沿道信号形态（尖峰模型的关键）

干净应变沿道（y 方向）的剖面有两个"应变瓣"：在 y≈+18（裂缝区）与 y≈−20（SRV 区）附近 `|strain|` 达峰（约 3.5e-5），在中心 y≈0 附近最小（约 1.0e-5）。**v4 的尖峰模型正是利用这个形态**——尖峰跟着 `|strain|` 走，所以中心谷底少、应变瓣多、早期低应变时也少（见 [3.2](#32-加噪脚本的原理第-1-步)）。

---

## 3. 端到端工作流与公用脚本

### 3.1 五步数据流（一张纵贯全程的链路）

以 v4 为例（v3 完全平行，只是族名/文件名不同）：

```
data/obs_strain_yy.csv   (干净真值, 130×500)
      │  [第1步] python noise_adding/add_noise_rfsdss.py
      │          d_noisy = d_clean + floor·N(0,1) + drift(t) + common(t) + spike(c,t)
      ▼
noise_adding/measurement_data_rfsdss_<tag>.csv   (+.meta, +QC 图, +summary)
      │  [第2步] bash inv/setup_rfsdss_runs.sh
      │          复制 v2 模板 5 件 + 把带噪 CSV 改名为 measurement_data.csv
      ▼
inv/rfsdss_<tag>/     (自包含 run 目录)
      │  [第3步] bash inv/run_all_rfsdss.sh        (需沙箱外, MOOSE 要 MPI)
      │          cd 进各目录 → python 106_optimization_runner_L1.py
      │          L-BFGS-B 外层 + MOOSE(optimize.i→forward_and_adjoint.i) 目标/梯度 oracle
      ▼
inv/rfsdss_<tag>/  optimized_alphas_L1.txt, best_data_alpha_L1.txt,
                   best_total_alpha_L1.txt, parameter_history_L1.csv, ...
      │  [第4步] bash inv/run_qc_rfsdss.sh         (需沙箱外, MOOSE 要 MPI)
      │          取 parameter_history 末行 alpha → 跑一次前向 → 画图
      ▼
inv/rfsdss_<tag>/  qc_strain_L1_final.png   (观测/模拟/残差/alpha 四联图)
      │  [第5步] python inv/compare_rfsdss_results.py   (纯 numpy, 可沙箱内)
      │          读各等级三种 alpha txt, 与合成真值对比
      ▼
inv/rfsdss_comparison_summary.csv  +  inv/rfsdss_alpha_overlay.png
   (以 clean(v2) 无噪反演为信噪比无穷大参照)
```

**每一跳靠什么衔接（传递契约）：**
1. **clean → 带噪 CSV**：加噪脚本只改写 `measurement_values` 列（叠加噪声），并把 `misfit_values`、`simulation_values` 两列清零（它们是反演产物，观测输入里必须为 0）。
2. **带噪 CSV → run 目录的 measurement_data.csv**：`optimize.i` 里写死了 `measurement_file='../measurement_data.csv'`，所以每档带噪 CSV 必须被 setup 脚本**改名**成这个固定名字。
3. **run 目录 → alpha 结果**：反演器把 MOOSE 当"给定 alpha → 返回 (目标值, 200 层梯度)"的一次性 oracle，真正的迭代在 Python 侧。
4. **alpha → QC 图 / 对比图**：见 [3.5](#35-前向-qcrun_parameter_history_qcpy--plot_inversion_qcpy第-4-步) 与 [3.6](#36-结果对比compare_py第-5-步)。

### 3.2 加噪脚本的原理（第 1 步）

#### v3 `add_noise.py`——均匀白噪声（peak / median 两族）

核心公式（对全部 65000 个点、所有时刻一视同仁）：

```
d_noisy_i = d_i + N(0, std²),   std = pct × REF
```

两族只差在**参考幅值 REF** 的取法：

- **peak 族**：`REF = max_i |d_i|`（全局峰值）= **3.589e-05**。档位 0.5 / 1 / 2 / 5%。
- **median 族**：`REF =` 各道时间峰值的**道间中位数** = **2.262e-05**。档位 1 / 2 / 5 / 10%。
  - 为什么不用 `median|d|`？因为 130×500 里有大量 t≈0 的极小早期值，直接取全体中位数会落在这些近零值上（≈3.6e-7），噪声会近乎为零、研究失去意义。所以先"每道时间维取峰值"滤掉早期近零，再"道间取中位数"稳健平均。

每档用独立 seed（peak: 20260608–11；median: 20260701–04）保证可复现。输出 `measurement_data_noise_*` / `measurement_data_mednoise_*` + `noise_summary.csv`。

#### v4 `add_noise_rfsdss.py`——真实感四成分噪声（rfsdss 族）

在干净观测上叠加四个物理上可区分的成分：

```
d_noisy(c,t) = d_clean(c,t)
             + floor_c · N(0,1)      # (1) 逐道白噪声底（绝对 nε 量级, 道间不同, 不随信号缩放）
             + drift_c(t)            # (2) 逐道低频漂移（3 模态余弦, 时间相关）
             + common(t)             # (3) 全道共模低频漂移（2 模态, 仪器/激光共模）
             + spike(c,t)            # (4) 应变驱动的稀疏重尾尖峰
```

**关键设计（都可在脚本顶部旋钮调节）：**

- **单旋钮控总强度**：噪声等级由 `floor_frac = floor_mean / REF` 一个量决定，其余三成分按**固定比例**跟着缩放：`drift = 0.75·floor`，`common = 0.5·floor`，`spike 幅值 = 7.5·floor`。所以调级只改幅值、不改噪声"形态"。
- **REF 锚定** = 与 v3 median 同一个 2.262e-05，5 档 = **0.5 / 1 / 2 / 5 / 10%**。
- **逐道噪声底**：`floor_c = floor_mean · exp(N(0, 0.4))`（lognormal，道间不同），是**绝对应变量级、不随信号峰值缩放**——这是它区别于 v3 白噪声的核心。
- **应变驱动尖峰（最关键的物理）**：尖峰概率
  ```
  p(c,t) = clip( scale · |d_clean(c,t)|/max|d_clean| · 空间权重(c), 0, cap )
  空间权重(c) = w_min + (1−w_min)·(|y_c − y0|/d_max)^γ,   y0=0, γ=2, w_min=0.3
  ```
  `scale` 自动缩放使全场平均尖峰概率命中 `SPIKE_TARGET_FRAC=1.2%`。含义：**光纤被拉得越厉害（|strain| 越大）尖峰越多**——早期低应变几乎无尖峰、应变瓣（y≈±18）多、中心谷底（y≈0）少。尖峰的时空图案只依赖干净信号+几何，故**跨等级恒定，只有幅值随等级变**。

每档独立 seed（20260709–13）。输出 `measurement_data_rfsdss_*` + 逐档 8 子图 QC `rfsdss_noise_qc_<tag>.png` + `rfsdss_noise_levels_summary.png` + `rfsdss_noise_summary.csv`。

### 3.3 反演器 `106_optimization_runner_L1.py`（第 3 步，核心）

这是**整套反演的核心公用外层驱动**（约 419 行），5 个 run 目录里逐字相同。它在 Python 侧跑一个 **L-BFGS-B 梯度优化循环**，把每步的"目标值+梯度"外包给 MOOSE 一次前向+伴随求解，并在数据失配之外叠加**平滑 L1 正则**。

#### 目标函数

```
J_total(alpha) = J_data(alpha) + J_L1(alpha)
```

- **数据失配项（MOOSE 侧算）**：`J_data = 0.5 · Σ (strain_yy_sim − meas)²`，在所有测点/时刻求和。梯度 `∂J_data/∂alpha_i` 由 MOOSE **伴随法**给出。
- **平滑 L1 正则项（Python 侧算）**：`J_L1 = BETA_L1 · Σ_i |alpha_i − (−18)|`。L1 绝对值不可微，用双曲平滑 `|z| ≈ √(z² + DELTA_L1²) − DELTA_L1`。
  - `BETA_L1 = 2e-11`（**刻意取很小**，正则约为数据项的 0.25%），`DELTA_L1 = 0.05`（平滑半径），参考值 `ALPHA_REFERENCE = −18`。
  - 作用：把各层往背景 −18 拉，鼓励"稀疏偏离"——除非数据强烈要求，各层应保持背景，从而**抑制噪声导致的虚假活跃层**。这正是研究噪声敏感性时用 L1 的意义。
- **数值缩放**：喂给优化器的是 `J_total × 1e6` 与 `grad × 1e6`（`SCALE_FACTOR=1e6`），把 ~3e-7 的目标抬到 ~0.3 量级，使 `ftol/gtol` 判据工作在合适范围。

#### 初始模型与分区

`build_initial_alpha()` 给出起点 x0：背景全 −18，SRV(61–68) 与裂缝(129–140) 抬到 −16 作为温和先验。反演目标就是从 −16 起点把这两带恢复到真值（−15 与 −14.52）。

#### 迭代与收敛

- 优化器 `scipy.optimize.minimize(method="L-BFGS-B", jac=True, bounds=...)`，`maxiter=300`、`ftol=1e-7`、`gtol=1e-10`。
- 每步 `objective_and_gradient(x)`：把当前 x 写进 `optimize.i` 的 `initial_condition` → 跑 MOOSE（20 个 MPI 进程）→ 读目标与梯度 → 加 L1 → 追踪最优 → 返回缩放量。
- **额外的步长停机**：回调 `_checkpoint` 里，若连续 3 次被接受的步都足够小（`||step||₂ ≤ 1.2e-2` 且 `max|step| ≤ 6e-3`）且已接受 ≥5 步，就 `raise StopIteration` 提前收敛。

#### 产出三种 alpha（后续对比用）

- `optimized_alphas_L1.txt`（**final**）：优化结束时最后一个迭代点。
- `best_data_alpha_L1.txt`（**best_data**）：历史中**数据失配最小**的 alpha（最贴合数据）。
- `best_total_alpha_L1.txt`（**best_total**）：历史中**总目标（数据+正则）最小**的 alpha。
- 低噪时三者往往相同；**高噪时明显分叉**——因为"最后一步"可能已开始过拟合噪声，best_data 会挑出更早、更干净的迭代点。这正是对比表里 rfsdss 1%/2%/10% 三行差异巨大的原因。

此外还写 `parameter_history_L1.csv`（每步 200 列 alpha）、`objective_history_L1.csv`、`gradient_history_L1.csv`、`step_history_L1.csv`、`initial_zones_L1.csv` 等，以及 MOOSE 工作目录 `inv_output/`（含可达数 GB 的 Exodus 场文件）。

### 3.4 MOOSE 求解器：`optimize.i` 与 `forward_and_adjoint.i`

这是一套 **MultiApp 双层结构**（PorousFlow + TensorMechanics 模块），Python 在最外层。

- **主控 `optimize.i`（父 App，873 行）**：不解 PDE，只管理 200 个参数 `perm_1..perm_200`、读观测、把 alpha 下发给子 App、把回传的目标值与梯度交出去。
  - `[OptimizationReporter] type=GeneralOptimization`：声明 200 个参数、初值 −18、界 [−25, −10]。
  - `[Transfers]`：`to_forward` 把当前 alpha（→`params/perm_i`）和观测数据传给子 App；`from_forward` 把 `objective_value` 与 200 个 `grad_perm_layer_i/inner_product` 收回。
  - `[Executioner] type=Optimize, tao_solver='taobqnls', -tao_gatol 1e50`：**梯度容差设成 1e50，使内层 TAO"一步即收敛"**——即把 MOOSE 优化模块短路成"给定 alpha → 一次前向+伴随 → 返回 (目标值, 梯度)"的**一次性预言机**。真正的最小化循环完全在 Python 的 L-BFGS-B 里。
- **子控 `forward_and_adjoint.i`（子 App，10882 行）**：一次执行同时求正演与伴随。
  - **物理**：全饱和单相**孔弹性**——达西流扩散（孔压 `pp`）+ 线弹性平衡（位移 `disp_x/y`），通过 Biot 系数 0.8 双向耦合。岩石 E=35 GPa、ν=0.25、φ=0.01、K_f=2.2 GPa。
  - **网格**：100×100 m，200×200 单元，沿 y 切成 200 条 0.5 m 水平层（`layer_1..layer_200`）。x=45 竖直注水井 nodeset。
  - **alpha→渗透率**：每层 `ParsedOptimizationFunction` 表达式 `10^alpha` → `FunctionAux` 写入 `perm_xx_layer_i` → `PorousFlowPermeabilityConstFromVar` 设定每层水平渗透率。**只反演 perm_xx**（指向注水井方向），perm_yy/zz 固定 1e-18。
  - **观测量**：`strain_yy = RankTwoAux(total_strain, i=1, j=1) = ∂disp_y/∂y`。
  - **伴随源（偶极子，关键技巧）**：因为观测是位移的空间导数 `strain_yy`，其伴随源不是单点 δ 而是 **δ 的空间导数**，用上下偏移 y±0.25、幅值 ±m/0.5 的两个点源（`ReporterTimePointSource`）近似。
  - **梯度算子**：`PorousFlowOptimizationAnisotropicDiffusionInnerProduct`，用正演场 `pp` 与伴随场 `pp_adjoint` 在每层子域上的加权内积给出 `∂J/∂alpha_i`。
  - **执行器 `TransientAndAdjoint`**：一次执行先正演时间推进（`end_time=384288 s ≈ 4.45 天`，130 个显式时刻）、再回放求伴随，直接产出目标值与梯度。直接法 `lu`+`mumps`。

一次迭代的完整回路：**外层设 alpha → optimize.i 下发 → 子 App 算 10^alpha 材料 → 正演(注水→孔压扩散→Biot→位移→strain_yy) → 组装 J 与 misfit → 偶极子加载伴随 → 内积算梯度 → 回传 → 外层加 L1、更新 alpha → 重复**。

### 3.5 前向 QC：`run_parameter_history_qc.py` + `plot_inversion_qc.py`（第 4 步）

**目的**：验证"反演出来的 alpha 到底能不能重现观测应变"。

- `run_parameter_history_qc.py`：从 `parameter_history_L1.csv` 取某一行 alpha（`--row -1` = 收敛末行），注入 `optimize.i` 的 `initial_condition`，把前向输入的 `[data]` reporter 打开 CSV 输出、**删掉 Exodus 输出块**（正常前向会写一个 ~9 GB 的 Exodus，QC 不需要），用 `--np`（默认 20）个 MPI 进程跑**一次前向**拿到 `simulation_values`，再挑出含真正模拟数据的主 reporter CSV，最后子进程调 `plot_inversion_qc.py` 出图。产物 `qc_strain_L1_final.png`。
- `plot_inversion_qc.py`：把长表 reporter CSV `pivot` 成 (y 坐标 × 时间) 网格，画 **2×2 四联图**：
  1. **观测应变**（含噪声）热图；
  2. **模拟应变**（反演 alpha 前向重建）热图；
  3. **残差 = 模拟 − 观测**热图（标题给出 `max|residual|`）；
  4. **alpha 剖面**：反演 alpha vs 合成真值（横轴 log10 渗透率，纵轴层中心 y）。
  - 热图用红蓝发散色标 `bwr`，横轴时间换算成小时，并在 y=−20/−16/14/20 画参考线标出两个活跃带边界。

### 3.6 结果对比：`compare_*.py`（第 5 步）

纯下游消费者（只 `np.loadtxt` 读 alpha 文本、不碰 MOOSE，可沙箱内跑）。对每个 run 读三种 alpha（final/best_data/best_total），与合成真值算 **6 个指标**：

| 指标 | 定义 | 理想值 | 含义 |
|---|---|---|---|
| `low_srv_mean` | `mean(alpha[low_srv])` | −15.0 | SRV 区反演准不准 |
| `fracture_mean` | `mean(alpha[fracture])` | −14.523 | 裂缝区反演准不准 |
| `matrix_free_mean` | `mean(alpha[free_window & ~low_srv & ~fracture])` | −18.0 | 中心基质本底有没有被噪声污染出假异常 |
| `max_alpha_err` | `max|alpha − truth|` | 0 | 全 200 层单层最大绝对误差（对局部失控最敏感） |
| `rel_l2_all` | `‖alpha−truth‖₂ / ‖truth‖₂`（全域） | 0 | 全域相对 L2 误差 |
| `rel_l2_free` | 同上但只在 `free_window` 内 | 0 | 有用区域反演质量的单一数字 |

输出 `*_comparison_summary.csv`（每 run 三行 + 表首一行真值参照）与 `*_alpha_overlay.png`（各噪声级 final alpha 剖面叠加，真值黑粗线，viridis 按噪声级配色；v3 里 peak 实线、median 虚线，v4 全实线）。第一行 LEVELS 都是 **clean(v2)**（无噪声的 v2 反演）作为信噪比无穷大参照。

### 3.7 三个 shell 驱动脚本（setup / run_all / run_qc）

三个脚本从 `${BASH_SOURCE[0]}` 反推路径，共同环境：`REPO_ROOT` 从 `inv/` 上溯 5 级到仓库根（校验 `$REPO_ROOT/fibeRIS/src/fiberis` 存在）、`PYBIN=$HOME/miniforge/envs/moose/bin/python`、`PYTHONPATH` 加 fibeRIS、`MPLBACKEND=Agg`。

> ⚠️ **`run_all_*` 与 `run_qc_*` 必须在 Codex 沙箱外运行**（MOOSE 求解要用 MPI/mpiexec 起多进程）。`setup_*` 与 `compare_*` 不涉及 MPI，可沙箱内跑。

- **`setup_rfsdss_runs.sh`**（无参数）：校验 v2 的 5 个模板文件在，遍历硬编码 `RUNS`，为每档 `mkdir` run 目录、`cp` 模板、把 `noise_adding/<stem>.csv` 复制改名为 `measurement_data.csv`。幂等安全（重复跑只刷新模板和数据，不动已有结果）。
- **`run_all_rfsdss.sh`**：无参数时 glob `inv/rfsdss_*` 发现所有含 `106_optimization_runner_L1.py` 的目录，串行 `cd` 进去跑反演，stdout `tee` 到 `inversion_L1.stdout`。也可传子集：`bash run_all_rfsdss.sh rfsdss_2pct rfsdss_5pct`。
- **`run_qc_rfsdss.sh`**：对每个 run（缺 `parameter_history_L1.csv` 则 SKIP）跑 `run_parameter_history_qc.py --row -1 --label L1_final --np $NP`（`NP` 默认 20），得到 `qc_strain_L1_final.png`。

---

## 4. 图集逐张说明

### 4.1 `01_noise_generation/` 噪声生成 QC

这些图体检"加进去的噪声长什么样"，**看的是噪声本身，不是反演结果**。

#### `v4_rfsdss_qc_<tag>.png`（五张，8 子图）

每档一张，8 个子图（a–h）：

![v4 rfsdss 2% 噪声 QC](01_noise_generation/v4_rfsdss_qc_2pct.png)

- **(a) 加入的噪声热图（道 × 时间）**：整体噪声纹理，能看到尖峰（暗色条纹）集中在应变瓣、晚期。
- **(b) 尖峰概率热图 p(t,c)**：这是 v4 的灵魂图——早期（左侧）整体接近 0、中心 y≈0 一条暗带，只有在**应变瓣 y≈±18 且被拉起来之后（右侧晚期）**才变亮。证明"尖峰跟着光纤应变走"。
- **(c) 逐道尖峰数（空间）+ 信号峰值叠加**：红色柱状（尖峰数）跟随绿色信号峰值曲线——应变瓣多、中心谷底少。
- **(d) 尖峰数 vs 时间（时间）+ mean|strain| 叠加**：红线（尖峰数）严格跟随绿线（平均应变）——早期低应变几乎无尖峰、随光纤被拉逐步增多。
- **(e) 逐道噪声底 floor_c + 空间权重**：蓝线是逐道 lognormal 噪声底（道间起伏），紫线是尖峰的额外空间权重。
- **(f) 干净 vs 加噪时间序列（中心 vs 应变瓣道）**：中心道信号小几乎干净，应变瓣道信号大且带尖峰。
- **(g) 应变瓣道的噪声成分分解**：白噪底 / 逐道漂移 / 共模 / 尖峰四成分叠加。
- **(h) 噪声底 vs 信号（关键）**：蓝线 floor_c 全道平在几百 nε，绿线 signal peak 随应变瓣大幅变化——**噪声底不随信号峰值缩放**，这就是它区别于 v3 白噪声的核心证据。

> 对比不同档位（`v4_rfsdss_qc_0p5pct.png` → `..._10pct.png`）可见噪声形态一致、只是幅值逐级放大。

#### `v4_rfsdss_levels_summary.png`

![v4 五档噪声横向对比](01_noise_generation/v4_rfsdss_levels_summary.png)

三联图：(a) 五档总噪声 std 柱状（256→...→5279 nε 单调放大）；(b) 应变瓣道 clean 与五档加噪散点叠加（0.5% 贴合、10% 明显发散）；(c) total std 对 floor mean 严格线性（一个旋钮控全局）。

#### 汇总表

- `v4_rfsdss_noise_summary.csv`：各档 floor/drift/common/spike 的目标值 vs 实现 std、`n_spikes`、`spike_fraction`、`total_noise_std`、信噪比 `snr_medchanpeak`。
- `v3_peakmedian_noise_summary.csv`：v3 peak/median 两族各档的 `target_std`、`realized_std`、`snr_globalpeak`。

### 4.2 `02_inversion_comparison/` 反演结果对比

这些图看的是**反演出来的 alpha 与真值差多少**。

#### `v3_alpha_overlay.png` / `v4_alpha_overlay.png`

![v4 反演 alpha 剖面 vs 真值](02_inversion_comparison/v4_alpha_overlay.png)

竖着的剖面图：横轴 log10 渗透率 alpha，纵轴层中心 y。黑粗线是真值（背景 −18、SRV −15、裂缝 −14.52），彩色线是各噪声级的 final alpha。**噪声越大，彩色线越偏离黑线**（尤其在两个活跃带和中心基质带）。

- **v3**：9 条（clean + peak 4 + median 4），peak 实线、median 虚线。
- **v4**：clean + rfsdss 各档，全实线。**注意 v4 当前 overlay 仅含已跑完的档位**（见 [第 6 节](#6-当前运行状态与待补内容)），全部跑完后需重跑 compare 刷新。

#### `v3_qc_metrics.png`、`v3_forward_qc_summary.csv`、`v3_alpha_summary_all_profiles.csv`

v3 研究的更细 QC 产物（历史脚本生成）：`v3_qc_metrics.png` 汇总各噪声级的前向 QC 指标；`v3_forward_qc_summary.csv` 含每档的 `qc_objective`、`rms_residual`、`noise_realized_std`、`rel_l2_all/free`、`active_layers` 等 22 列细指标；`v3_alpha_summary_all_profiles.csv` 是逐剖面（truth/final/best_data/...）的分区统计。

#### `v3_comparison_summary.csv` / `v4_comparison_summary.csv`

[第 3.6 节](#36-结果对比compare_py第-5-步) 定义的 6 指标表。实测要点见 [第 5 节](#5-关键数值表)。

### 4.3 `03_forward_qc_strain/` 各 run 前向 QC

每个 run 一张 `*__qc_strain.png`，就是 [3.5 节](#35-前向-qcrun_parameter_history_qcpy--plot_inversion_qcpy第-4-步) 的 2×2 四联图（观测应变 / 模拟应变 / 残差 / alpha 剖面）。这是"反演 alpha 能否复现观测应变"的直接视觉证据——**残差越淡、alpha 剖面越贴合真值，说明反演越成功**。

![v3 peak 2% 前向 QC](03_forward_qc_strain/v3_noise_2pct__qc_strain.png)

- `v3_noise_{0p5,1,2,5}pct__qc_strain.png`：peak 族 4 个。
- `v3_mednoise_{1,2,5,10}pct__qc_strain.png`：median 族 4 个。
- `v3_noise_{0p5,1,2,5}pct__alpha_profiles.png`：peak 族反演过程中的 alpha 剖面演化。
- **v4 rfsdss 的前向 QC 图尚未生成**（`run_qc_rfsdss.sh` 需要 MOOSE，还没跑）。

---

## 5. 关键数值表

### 5.1 v4 rfsdss 五档噪声（REF = 2.262e-05 = 22622 nε）

| 档位 | floor_mean | total noise std | SNR (REF/std) | seed |
|---|---|---|---|---|
| 0.5% | 113 nε | 277 nε | ~82 | 20260709 |
| 1% | 226 nε | 555 nε | ~41 | 20260710 |
| 2% | 452 nε | 1117 nε | ~20 | 20260711 |
| 5% | 1131 nε | 2684 nε | ~8 | 20260712 |
| 10% | 2262 nε | 5621 nε | ~4 | 20260713 |

> 成分比例：`floor : drift : common : spike幅值 = 1 : 0.75 : 0.5 : 7.5`（相对 floor_mean）。尖峰全场平均命中率 1.2%、峰值概率约 6.2%（在应变瓣晚期）。

### 5.2 反演对比指标（节选，越小越好）

| level | which | fracture_mean | max_alpha_err | rel_l2_all | rel_l2_free |
|---|---|---|---|---|---|
| **truth** | — | −14.523 | 0 | 0 | 0 |
| **clean (v2)** | final | −14.567 | 0.152 | 0.00153 | 0.00220 |
| rfsdss 0.5% | final | −14.524 | 0.123 | 0.00124 | 0.00179 |
| rfsdss 1% | final | −14.705 | 0.958 | 0.00685 | 0.00986 |
| rfsdss 1% | best_data | −14.644 | 0.508 | 0.00338 | 0.00486 |
| rfsdss 2% | best_data | −14.525 | 0.475 | 0.00410 | 0.00590 |
| rfsdss 2% | best_total | −16.335 | 3.512 | 0.04434 | 0.06381 |
| rfsdss 10% | final | −16.376 | 3.497 | 0.04969 | 0.07151 |

**读表结论：**
- **0.5% 档反演几乎无损**（rel_l2 与 clean 基线相当，甚至略优）。
- **1% 起 final 与 best_data 明显分叉**——说明高噪时"最后一步"开始过拟合噪声，best_data 更可靠。
- **2%（best_total）与 10% 彻底崩坏**：`fracture_mean` 从 −14.5 掉到 −16.3、`max_alpha_err` 飙到 3.5，说明某些层反演已失控。
- 这印证了 v4 的价值：**结构化的真实噪声（尖峰+漂移）在高档位对反演的破坏，比 v3 同幅值均匀白噪声更严重**。

---

## 6. 当前运行状态与待补内容

> 截至本文档生成时（v4 反演仍在进行中）：

| run | 状态 | 备注 |
|---|---|---|
| rfsdss_0p5pct | ✅ 完成 | 34 迭代收敛 |
| rfsdss_1pct | ✅ 完成 | 45 迭代 |
| rfsdss_2pct | ⏳ 运行中 | 已 3 天 / 131 迭代，收敛较慢；对比表里暂只有 best_data/best_total |
| rfsdss_5pct | ⛔ 排队 | 在 2pct 之后（glob 顺序 0p5→10→1→2→5，最难的 10pct 排第二堵过队） |
| rfsdss_10pct | ✅ 完成 | 98 迭代 |

**待补：**
1. **v4 前向 QC 图**（`v4_rfsdss_*__qc_strain.png`）：需在沙箱外跑 `bash inv/run_qc_rfsdss.sh`，再把 `inv/rfsdss_<tag>/qc_strain_L1_final.png` 拷进 `03_forward_qc_strain/`。
2. **v4 overlay/对比表刷新**：2pct、5pct 跑完后重跑 `python inv/compare_rfsdss_results.py`，再刷新 `02_inversion_comparison/v4_alpha_overlay.png` 与 `v4_comparison_summary.csv`。
3. 进程说明：当前是**单条 driver 链**（外层 bash → 子 shell → runner），不是两个在抢。若想加速 5pct，可另开 `bash inv/run_all_rfsdss.sh rfsdss_5pct` 并行（run 目录相互独立不冲突）。

---

## 7. 如何复现

```bash
cd scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100_v4strain_rfsdss

# 1. 生成 5 档 rfsdss 噪声 + 逐档 QC 图（纯 python，可沙箱内）
python noise_adding/add_noise_rfsdss.py

# 2. 暂存 5 个自包含 run 目录（纯拷贝，可沙箱内）
bash inv/setup_rfsdss_runs.sh

# 3. 跑 L1 反演（★需沙箱外，MOOSE 要 MPI；很慢）
bash inv/run_all_rfsdss.sh
#    或指定/并行子集： bash inv/run_all_rfsdss.sh rfsdss_5pct

# 4. 前向 QC（★需沙箱外）
bash inv/run_qc_rfsdss.sh

# 5. 对比真值（纯 numpy，可沙箱内）
python inv/compare_rfsdss_results.py

# 6. 重新整理图集到本 fig/（可选，按 fig/README 的命名手动或脚本化拷贝）
```

v3 完全平行，脚本换成 `add_noise.py` / `setup_noise_runs.sh` / `run_all_noise.sh` / `run_qc_noise.sh` / `compare_noise_results.py`。

---

## 8. 参数与文件名契约速查

### 反演关键常量

| 变量 | 值 | 含义 |
|---|---|---|
| `TOTAL_LAYERS` | 200 | 层数 = 参数维数 |
| `LAYER_HEIGHT` | 0.5 m | 层高，y∈[−50,50] |
| 背景 / L1 参考 | −18.0 | 基质 & L1 拉向值 |
| SRV 真值 | −15.0 | y∈[−20,−16]，层 61–68 |
| 裂缝真值 | log10(3e-15)≈−14.523 | y∈[14,20]，层 129–140 |
| 自由窗口 | y∈[−25,25]，层 51–150 | 可反演区，界 [−25,−10] |
| `BETA_L1` | 2e-11 | L1 正则权重 |
| `DELTA_L1` | 0.05 | L1 平滑半径 |
| `SCALE_FACTOR` | 1e6 | 目标/梯度缩放 |
| `MAXITER / FTOL / GTOL` | 300 / 1e-7 / 1e-10 | L-BFGS-B 停机 |
| MPI 进程 | 20 (`L1_NUM_PROCESSORS`) | MOOSE 并行 |
| MOOSE end_time | 384288 s | ≈4.45 天注水，130 时刻 |

### 文件名契约（跨脚本硬约束）

- 观测输入必须叫 `measurement_data.csv`（`optimize.i` 写死 `measurement_file='../measurement_data.csv'`）。
- 前向模板必须声明 `variable = 'strain_yy'`（反演器启动硬校验）。
- 反演输出三种 alpha：`optimized_alphas_L1.txt` / `best_data_alpha_L1.txt` / `best_total_alpha_L1.txt`（单列 200 行 log10 渗透率），是 compare 脚本的唯一契约。
- QC 图统一叫 `qc_strain_L1_final.png`（`--label L1_final`）。

### 源码位置

- v3 加噪：`perm5layer_100_v3strain_noise/noise_adding/add_noise.py`
- v4 加噪：`perm5layer_100_v4strain_rfsdss/noise_adding/add_noise_rfsdss.py`
- 反演器（每档一份，逐字相同）：`inv/<run>/106_optimization_runner_L1.py`
- MOOSE：`inv/<run>/optimize.i`、`inv/<run>/forward_and_adjoint.i`
- QC：`inv/<run>/run_parameter_history_qc.py`、`inv/<run>/plot_inversion_qc.py`
- 对比：`inv/compare_noise_results.py`、`inv/compare_rfsdss_results.py`
- 工作流 README：各研究根目录的 `README_noise_workflow.md` / `README_rfsdss_workflow.md`

---

*本图集与说明由整理脚本 + 多 agent 深读两套研究的全部 python / MOOSE 源码后综合生成；如源码更新，请同步刷新本文件。*
