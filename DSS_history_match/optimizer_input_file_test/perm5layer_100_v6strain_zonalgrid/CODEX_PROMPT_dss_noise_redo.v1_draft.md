# 任务：重做 v6 敏感度网格中的 "真实 DSS 背景噪声"（factor A）部分

你只负责**这一件事**：修正 factor A（真实 DSS 背景噪声）的提取/生成方式，重新生成依赖它的噪声数据集，并把受影响的反演网格重跑、图重出。**不要动**正演模型、反演器、白噪声（factor B）、位置偏移（factor C）、真值、评价指标。

仓库根目录：`/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner`
本研究目录（下文简称 V6）：
`/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100_v6strain_zonalgrid`

---

## 0. 背景：现在的 factor A 错在哪（先读懂再动手）

研究设计：2 参数分区渗透率反演，三因子各 3 档的 3×3×3=27 格敏感度网格。
- A = 真实 DSS 背景噪声（bg1/2/3）
- B = 白噪声（w1/2/3）= `σ_B × N(0,1)`，逐点独立
- C = 区域位置偏移（s0/s2/s4 层）——只在反演端，不改数据

A 和 B 存在于**数据**里：9 个数据集 `noise_data/measurement_data_bg<i>_w<j>.csv` = 干净观测 + A(i) + B(j)。C 在反演时由文件夹名后缀 `_s<k>` 传入。

**当前 factor A 的问题（已用数据证实）**：
生成器 `V6/noise_adding/add_noise_grid.py` 里的 `load_real_bg_residual()` 从真实 DSS 的**累积应变**（cumulative strain）里取注入前 41 个样本、逐道去线性趋势，再用 `real_texture()` 把**一条真实道的残差曲线整条线性拉伸**铺满合成的 130 个时间步。结果：
- 该残差滞后-1 自相关 = **+0.84**（强时间相关，像一条慢漂移信号），而真正的逐样本测量噪声（相邻样本一阶差分）自相关 = **+0.10**（≈时间独立）。
- 根因：DSS `data` 是由应变率**积分**得到的累积应变；白的率噪声一积分就成了随机游走（低频、强相关）。从累积值里去趋势提取到的是"积分漂移/慢变微应变"，**不是**时间独立的测量噪声。
- 再把单条曲线拉伸盖满全时间轴，就人为造出了一条"某段为正、某段为负"的相干带，看起来像信号、还和井况假性对应。

**结论**：真正的 DSS 测量噪声是**时间独立**的，藏在累积应变的**一阶差分**里。factor A 必须改成时间独立。

---

## 1. 要改的文件（只改这一个生成器）

`V6/noise_adding/add_noise_grid.py`

只重写 **factor A 那两个函数** `load_real_bg_residual()` 和 `real_texture()`（以及 `main()` 里调用它们的那几行）。**其它一律保持**：`LEVELS_A=[0.03,0.06,0.12]`、`LEVELS_B=[0.03,0.06,0.12]`、`BASE_SEED=20260901`、`median_channel_peak()`（REF 定义）、factor B 的生成 `sigma_b * rng.standard_normal((n_time,n_chan))`、`noise[0,:]=0`（t=0 参考帧置零）、输出文件名与 meta 格式。

真实 DSS 源文件（只读）：
`/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/output/das_observed/das_strain_waterfall_T1T3.npz`
- key `data`：shape (45, 4620)，**道 × 时间**，单位 **millistrain（毫应变）**，dt=60 s，是累积应变。
- 注入前背景窗口：时间列索引 **1..41**（`data[:, 1:42]`），第 0 列恒为 0 要跳过。
- 换算成无量纲 strain 要 **×1e-3**。

干净合成观测（只读）：`V6/data/obs_strain_yy.csv`
- 130 个时刻 × 500 个道 = 65000 行；列 `measurement_time, measurement_values, measurement_ycoord, ...`。
- REF = 各道随时间峰值 |strain| 的道间中位数 = **2.262e-5**（生成器里 `median_channel_peak()` 已实现，别改）。

## 2. 新的 factor A 算法（严格按此实现）

目标：**时间独立**的真实 DSS 测量噪声，保留真实的经验幅值分布（可能非高斯/逐道不同），但**不带任何时间相关结构**。

```
1. raw   = np.load(NPZ)["data"].astype(float)      # (45,4620) 毫应变
   block = raw[:, 1:42] * 1e-3                      # 注入前 45×41，转 strain
2. diff  = np.diff(block, axis=1)                   # 45×40：相邻样本增量 = 真正的逐样本测量噪声
   # 自检并打印：diff 逐道滞后-1自相关的均值应 ≈ 0（预期约 +0.1）；原生 std 应 ≈ 2.15e-8
3. pool  = diff - diff.mean(axis=1, keepdims=True)  # 逐道去均值，得到每道一个经验噪声池（各 40 个样本）
4. 造 130×500 场：对每个输出道 c（500 个），指定一个真实道 r（随机或轮询，用 rng 决定）；
   然后对该道的 130 个时间步，**每一步独立地从 pool[r] 里有放回抽 1 个样本**。
   → 每个 (t,c) 都是独立抽样，时间独立由构造保证。
   **禁止**沿时间对一条曲线做插值/拉伸（那正是旧 bug 的来源）。
5. 场整体去均值、归一化到单位 std；再缩放：noise_A = σ_A × 场，σ_A = LEVELS_A[i] × REF。
6. noise_A[0,:] = 0（保持 t=0 参考帧为零）。零均值。
```

可选（默认**关闭**，加一个常量开关 `INCLUDE_COMMON_DRIFT = False`）：若日后需要现场风格的低频**共模漂移**，另加一项 `σ_drift × 随机低频共模序列`（例如平滑随机游走，零均值，所有道共用同一条，每次实现用 rng 随机生成）。**绝不能**用某条固定的真实曲线。本次任务保持关闭，先做纯时间独立版本。

factor B（白噪声）**完全不动**。这样 A 与 B 都是时间独立的，区别只在：A 用真实经验分布（可能带重尾/逐道差异），B 用理想高斯。

## 3. 验证要求（写进生成器输出，并保存）

重跑生成器后必须打印并保存到 `V6/noise_data/grid_noise_summary.csv` 与新的 `V6/noise_data/grid_noise_qc.png`：
- 新 A 场的逐道滞后-1 自相关均值：**必须 ≈ 0**（|值| < 0.15）。若 > 0.3 说明还有时间相关，实现有误。
- 各档 σ_A、σ_B 的实现值（应约等于 679/1357/2715 nε）。
- QC 图里保留原有 6 个面板，其中功率谱面板 (e) 现在 **A 和 B 都应是平的**（不再是 A 红、B 平）；面板 (a) 里 A 不应再是平滑漂移曲线，而应是高频抖动。
- 在 QC 图或 stdout 里明确写出：`new A lag1 autocorr = ...`。

## 4. 重新生成数据并重跑网格（受影响的是全部 27 格）

因为 9 个数据集**每一个**都含 A，所以 A 一改，**27 个格子都要重跑**（C 只是反演端参数，复用同一套数据）。

### 4.1 先归档旧结果（必须，别覆盖丢数据）
```
cd V6
mv noise_data          noise_data_v1_FLAWED_cumulative_residual
mkdir -p inv_archive_v1_FLAWED && mv inv/bg*_w*_s* inv_archive_v1_FLAWED/
mv inv/grid_comparison_summary.csv   inv_archive_v1_FLAWED/
mv inv/grid_sensitivity_heatmaps.png inv_archive_v1_FLAWED/
mv inv/grid_main_effects.png         inv_archive_v1_FLAWED/
mv inv/GRID_FINALIZED.txt            inv_archive_v1_FLAWED/ 2>/dev/null
mv fig_deliverable                   fig_deliverable_v1_FLAWED
```
（`inv/_template/`、`inv/setup_grid_runs.sh`、`inv/run_all_grid.sh`、`inv/compare_grid_results.py`、`inv/finalize_grid.py` 保留。）

### 4.2 重新生成 9 个数据集
```
cd V6/noise_adding
/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python add_noise_grid.py
```
产出 `V6/noise_data/measurement_data_bg{1,2,3}_w{1,2,3}.csv`（各含 .meta）+ `grid_noise_summary.csv` + `grid_noise_qc.png`。先按第 3 节核对自相关，**不达标不要往下跑**。

### 4.3 重新暂存 27 个 run 文件夹
```
cd V6/inv && bash setup_grid_runs.sh
```
它会从 `inv/_template/` 拷 5 个模板文件、并把对应数据集拷成每格的 `measurement_data.csv`。应得到 27 个 `bg<i>_w<j>_s<k>/`。

### 4.4 跑 27 个反演（MOOSE，需要 MPI，必须脱离沙箱/终端）
```
cd V6/inv
setsid nohup bash -c 'PAR=2 ZONAL_NP=10 bash run_all_grid.sh > grid_master.log 2>&1; echo GRID_ALL_DONE_EXIT=$? >> grid_master.log' > /dev/null 2>&1 &
```
- 机器：20 物理核 / 92 GB。`PAR=2 ZONAL_NP=10`（2 路并行×10 进程）或 `PAR=4 ZONAL_NP=5` 都行，别超 20 进程。
- 每格约 4–5 小时，27 格总计 **2–3 天**。它是脱离终端的常驻进程，关掉会话也会继续跑。
- 环境由 `run_all_grid.sh` 自己设置：`PYTHONPATH=/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/fibeRIS/src`、`MPLBACKEND=Agg`、Python=`/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python`、MOOSE 可执行=`/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/combined/combined-opt`。
- 进度：`tail -f V6/inv/grid_master.log`；每格完成会在自己文件夹写 `optimized_theta_zonal.txt`。

### 4.5 自动收尾
`V6/inv/finalize_grid.py` 会等 27/27 完成后自动跑 compare 出图并写 `GRID_FINALIZED.txt`。**注意它里面 `SCHEDULER_PID = 140782` 是上一次的旧 PID，已失效**——启动前把它改成本次 `run_all_grid.sh` 的真实 PID，或者改成按 `optimized_theta_zonal.txt` 计数为准、不依赖 PID。然后同样脱离终端启动：
```
cd V6/inv
setsid nohup /rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python finalize_grid.py > finalize_grid.log 2>&1 &
```

### 4.6 全部完成后
1. `cd V6/inv && python compare_grid_results.py` → `grid_comparison_summary.csv`、`grid_sensitivity_heatmaps.png`、`grid_main_effects.png`。
2. 重跑 5 张代表性正演 QC（各一次 MOOSE 前向，可并行，每张 ~15–25 min）：格子 `bg2_w2_s0`、`bg2_w2_s2`、`bg2_w2_s4`、`bg1_w1_s0`、`bg3_w1_s4`，在各自文件夹里执行
   `python run_parameter_history_qc.py --history-file parameter_history_L1.csv --row -1 --label zonal_final --np 4`
   得到每格的 `qc_strain_zonal_final.png`。
3. 重建交付文件夹 `V6/fig_deliverable/`，命名与上一版一致：
   `01_噪声设计_QC.png`(=noise_data/grid_noise_qc.png)、`02_敏感度热图.png`、`03_主效应.png`、`04_正演QC_最好_bg1w1_对齐.png`、`05_正演QC_中噪_对齐s0.png`、`06_正演QC_中噪_错位2层.png`、`07_正演QC_中噪_错位4层.png`、`08_正演QC_最差_bg3_错位4层.png`、`grid_comparison_summary.csv`、`README.md`。
   README 参考旧版 `V6/fig_deliverable_v1_FLAWED/README.md` 的结构重写，但要把 factor A 的描述改成"时间独立的真实 DSS 测量噪声（由一阶差分经验分布自助采样）"，并注明本版修正了旧版的时间相关问题。
4. 写一份 `V6/CHANGELOG_dss_noise_redo.md`：说明旧 A 的问题、新 A 的算法、自相关前后对比、哪些结果因此变化。

---

## 5. 一定要避开的坑（都是实际踩过的）

- **`pkill -f` / `pgrep -f` 会匹配到你自己的命令行**（因为命令里含同样的字符串），会把自己的 shell 杀掉（退出码 144）。要杀 MOOSE 用按进程名精确匹配：`pkill -x combined-opt`、`pkill -x mpiexec`。
- 交互 shell 是 **zsh**：无引号的 `$VAR` **不做分词**，`for c in $LIST` 只会循环一次。脚本一律用 `bash` 跑，或在 for 里写字面列表。
- `compare_grid_results.py` 里必须用 `df["shift"]`，**不能**用 `df.shift`（那是 DataFrame 的方法，会静默匹配不到、热图全空白）。这个已修好，别改回去。
- MOOSE 前向每次会写 ~3 GB exodus；反演器 `107_optimization_runner_zonal_L1.py` 已用 `strip_exodus()` 去掉，别动它。
- 一次 MOOSE 求解 ~20–45 分钟，别在前台等；所有长任务都 `setsid nohup ... &` 脱离终端。
- 不要重新加噪 v3/v4/v5 那些旧研究，不要碰 `perm5layer_100_v3strain_noise`、`..._v4strain_rfsdss`、`..._v5strain_zonal2p`。

## 6. 完成标准

- [ ] `add_noise_grid.py` 的 factor A 改为一阶差分经验池的独立自助采样；B 不变。
- [ ] 新 A 场滞后-1 自相关 |值| < 0.15，并有打印/记录；QC 图 (e) 里 A 谱线为平。
- [ ] 旧数据与旧结果已归档到 `*_v1_FLAWED*`，未被覆盖。
- [ ] 27/27 格重跑完成，`GRID_FINALIZED.txt` 存在，`grid_comparison_summary.csv` 有 27 行。
- [ ] 热图、主效应、5 张正演 QC、`fig_deliverable/`（含中文 README）全部重生成。
- [ ] `CHANGELOG_dss_noise_redo.md` 写清改动与前后对比。

预期科学结果：位置偏移（C）仍应压倒性主导；A 轴的影响可能比旧版更小、更接近 B（因为两者现在都是时间独立的）——这本身就是要如实报告的结论，不要为了"让 A 显得有影响"去调参。
