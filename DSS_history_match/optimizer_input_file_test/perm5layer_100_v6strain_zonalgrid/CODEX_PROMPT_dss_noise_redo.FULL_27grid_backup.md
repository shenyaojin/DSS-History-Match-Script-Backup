# 任务：重做 v6 敏感度网格中的 "真实 DSS 背景噪声"（factor A）部分

你只负责**这一件事**：修正 factor A（真实 DSS 背景噪声）的生成方式，重新生成依赖它的噪声数据集，把受影响的反演网格重跑、图重出。**不要动**正演模型、反演器、白噪声（factor B）的公式与随机实现值、位置偏移（factor C）、真值、评价指标、噪声档位。

仓库根目录：`/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner`
本研究目录（下文简称 V6，**它是一个你要自己定义的 shell 变量**，见第 4 节开头）：
`/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100_v6strain_zonalgrid`

先读一遍 `V6/README_grid_workflow.md`（本网格的运行说明与踩坑记录），再动手。

---

## 0. 背景：现在的 factor A 错在哪（先读懂再动手）

研究设计：2 参数分区渗透率反演，三因子各 3 档的 3×3×3 = 27 格敏感度网格。
- A = 真实 DSS 背景噪声（bg1/2/3）
- B = 白噪声（w1/2/3）= `σ_B × N(0,1)`，逐点独立
- C = 区域位置偏移（s0/s2/s4 层）——只在反演端，不改数据

A 和 B 存在于**数据**里：9 个数据集 `noise_data/measurement_data_bg<i>_w<j>.csv` = 干净观测 + A(i) + B(j)。C 在反演时由 run 文件夹名后缀 `_s<k>` 传入。

**当前 factor A 的问题（已用数据证实）**：
生成器 `V6/noise_adding/add_noise_grid.py` 里的 `load_real_bg_residual()` 从真实 DSS 的**累积应变**里取 41 个样本、逐道去线性趋势，再用 `real_texture()` 把**一条真实道的残差曲线整条线性拉伸**铺满合成的 130 个时间步。结果：
- 该残差（41 样本）滞后-1 自相关 pooled ≈ **+0.84**（逐道均值 0.79 / 中位 0.83）；再线性拉伸到 130 步后，**真正进入数据的 A 场逐道自相关高达 +0.97**——是一条慢漂移曲线，不是噪声。相邻样本一阶差分的自相关仅 +0.10。
- 根因：DSS `data` 是由应变率**积分**得到的累积应变；白的率噪声一积分就成随机游走（低频、强相关）。从累积值里去趋势提取到的是"积分漂移/慢变微应变"，不是时间独立的测量噪声。把单条曲线拉伸盖满全时间轴，还人为造出了"某段为正、某段为负"的相干带，看起来像信号、还和井况假性对应。

**结论与模型定位（必须按此理解）**：本研究的合成观测 `obs_strain_yy.csv` 模拟的是**直读式 DSS（RFS-DSS）**在每个采集时刻的应变读数，对这类仪器合理的测量噪声模型是**逐读数独立**（每个 (t,c) 一个独立随机数，与采样间隔 Δt 无关；合成观测 Δt 为 725–11025 s，非均匀）。旧 A 把 LF-DAS 积分应变的低频漂移当噪声、又用固定曲线拉伸，既非独立噪声也非可重复随机过程，必须废除。**新 A 仍是逐点独立噪声，只从真实 LF-DAS 数据借用 60 s 增量（一阶差分）的经验幅值分布形状**——原生幅值 ~2e-8 与合成步长无对应，只借标准化后的形状，幅值由 σ_A = LEVELS_A × REF 统一给定。

要如实写进 CHANGELOG / README / meta：真实一阶差分**并不是**严格独立噪声——它有弱正自相关（lag1 +0.10±0.02）、跨道相关（平均 r≈0.19，共模≈0.47σ）和纤维级脉冲事件（|z|>3 的样本集中在少数时间索引）；新 A **有意丢弃**这些时间/空间相关结构，只保留边缘分布（excess 峰度≈6、偏度≈−0.3）与逐道幅值差异（逐道 std p90/p10≈2.0）。因此本版 factor A 的科学含义是"**经验分布（非高斯、逐道异方差）的 i.i.d. 噪声**"，不再是"含现场漂移的真实背景"。若日后要模拟 LF-DAS 型积分应变观测，正确做法是另建因子（每道每次实现新生成随机游走再整体缩放）——**本次不做**，CHANGELOG 记一句即可。

---

## 1. 要改的文件（只改这一个生成器）

`V6/noise_adding/add_noise_grid.py`

**允许改动范围**：
- `load_real_bg_residual()` → 重写并改名 `load_real_diff_pool()`；`real_texture()` → 重写并改名 `bootstrap_texture()`；`main()` 里的调用/打印行；
- `build_qc()` 中的标题/图例文字（面板数量与布局不变）；
- 文件头注释里对 factor A 的描述（第 8–30 行）；
- `.meta` 的 `noise_model` 改为 `grid_realdiffboot_plus_white`，并新增 `noise_version=2`、`common_drift=none`、`rng_scheme=v1_compatible`（其余 meta 键 `sigma_realbg`、`realbg_frac_ref`、`seed` 等不变）。

**不得改**：`LEVELS_A=[0.03,0.06,0.12]`、`LEVELS_B=[0.03,0.06,0.12]`、`BASE_SEED=20260901`、`DAS_BG_TCOLS=(1,42)`、`median_channel_peak()`（REF 定义）、factor B 公式 `sigma_b * rng.standard_normal((n_time, n_chan))`、`noise[0,:]=0`（t=0 参考帧置零）、输出文件名、summary 既有列名。

真实 DSS 源文件（只读）：
`/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/output/das_observed/das_strain_waterfall_T1T3.npz`
- key `data`：shape (45, 4620)，**道 × 时间**，float32，单位 **millistrain（毫应变）**，dt=60 s，是累积应变。
- 背景窗口：沿用 `DAS_BG_TCOLS=(1,42)`（`data[:, 1:42]`），第 0 列恒为 0 要跳过。"注入前"只是 `README_grid_workflow.md:26` 与生成器注释的断言，仓库内无独立佐证（npz 起点 2025-02-28 17:00:26），本次**不改窗口**以保证可比性，只在 CHANGELOG 注明此假设。
- 列 1 本身 ≈1e-6 mstrain、列 2 已 ≈7e-5，是积分**启动样本**；它与列 2 的差分不能进经验池（见第 2 节步骤 2）。
- 换算成无量纲 strain 要 **×1e-3**。

干净合成观测（只读）：`V6/data/obs_strain_yy.csv`
- 130 个时刻 × 500 个道 = 65000 行；列 `measurement_time, measurement_values, measurement_ycoord, ...`。
- REF = 各道随时间峰值 |strain| 的道间中位数 = **2.262e-5**（`median_channel_peak()` 已实现，别改）。σ 档 = 3%/6%/12% × REF = 679 / 1357 / 2715 nε。

## 2. 新的 factor A 算法（严格按此实现）

目标：**逐点独立**的噪声场，边缘分布借自真实 DSS 一阶差分的经验分布（重尾、偏度≈−0.3、逐道异方差），**不带任何时间/空间相关结构**。

```
1. raw   = np.load(NPZ)["data"].astype(float)      # (45,4620) 毫应变
   block = raw[:, 1:42] * 1e-3                      # 45×41，转 strain（窗口不变）
2. diff  = np.diff(block, axis=1)[:, 1:]            # 45×39：丢掉第一个增量——列 1 是积分启动样本，
                                                    #   d2-d1 是沿光纤相干的启动图案（8/45 道 |z|>3），不能进池
   # 自检并打印（只是核对读对了数据，不是验收）：diff 逐道 lag1 均值预期 +0.10±0.02（弱正相关、非严格白；
   #   自助采样会丢掉它，可接受）；原生 std 预期 ≈2.0–2.2e-8 strain（丢首增量后 ≈2.04e-8）
3. 逐道标准化后跨道合并成一个大池（不要每道各建 40 样本的小池——那样每道只有 ≤40 个离散值、逐道峰度 1.3–11 乱跳）：
   z      = (diff - diff.mean(1, keepdims=True)) / diff.std(1, keepdims=True)
   pool   = z.ravel()                               # 45×39 = 1755 个单位方差样本
   s_real = diff.std(1)                             # 45 个逐道原生 std，作为逐道幅值分布（p90/p10≈2.0）
   # 自检并打印：pool 的 excess 峰度（预期 ≈6）、偏度（≈−0.3）、s_real 的 p90/p10（≈2.0）
4. 造 130×500 场（严格按步骤 7 的 rng 顺序）：
   先为每个输出道 c 随机指定一个真实道 r[c]（沿用 v1 的随机分配，不做顺序/轮询映射）；
   然后 T[t,c] = pool[k[t,c]] * s_real[r[c]] / median(s_real)，k 为独立自助抽样索引。
   → 每个 (t,c) 都是独立抽样，时间独立由构造保证。**禁止**沿时间对任何曲线做插值/拉伸/滤波（那正是旧 bug 的来源）。
5. **整体（全场）**去均值、除以全场 std，再缩放：noise_A = σ_A × T，σ_A = LEVELS_A[i] × REF。
   **不要逐道归一化**——逐道幅值差异是新 A 相对 B 仅剩的两个特征之一（另一个是边缘分布形状），必须保留。
6. noise_A[0,:] = 0（保持 t=0 参考帧为零）。零均值。
7. **rng 消耗顺序必须与 v1 位级兼容，保证 9 个数据集的 B 与 v1 完全相同**（这样 v1↔v2 对比才能只归因于 A）：
   (a) 道分配沿用旧循环 `for c in range(n_chan): r[c] = int(rng.integers(n_rc))`（n_rc=45，共 500 次；
       `rng.integers(45, size=500)` 消耗相同随机流也可）；
   (b) **然后**先生成 `noise_b = sigma_b * rng.standard_normal((n_time, n_chan))`；
   (c) 之后再用同一 rng 抽 bootstrap 索引 `k = rng.integers(pool.size, size=(n_time, n_chan))`。
   任何在 (b) 之前多抽的随机数都会改变 B。
   自检：把旧 `load_real_bg_residual/real_texture` 改名保留为 `_v1_*`，用 seed 20261103（即 bg2_w2）按旧流程与新流程
   各生成一次 noise_b，`np.testing.assert_array_equal` 通过后才算完成。
```

**本次不实现任何共模/低频漂移项**，不加开关、不写分支代码。只在生成器 docstring 留一句 TODO：`future: optional zero-mean common-mode drift, generated fresh per realisation from rng, never a fixed real trace`。meta 写 `common_drift=none`。

factor B（白噪声）**公式、σ_B 与随机实现值都不变**（由步骤 7 保证）。A 与 B 都是时间独立、空间独立、同 σ 的噪声，**唯一**区别是边缘分布（A：经验分布，excess 峰度≈6、偏度≈−0.3、逐道 std p90/p10≈2；B：同方差高斯）。反演目标是 L1，对重尾稳健，**预期 A 轴主效应接近 B 轴——这是要如实报告的结果，不要为了让 A 显得有影响去调参**。

## 3. 验证要求（写进生成器输出，并保存）

- **自相关指标统一定义**：生成的 130×500 A 场（缩放前、去掉 t=0 行）逐道滞后-1 自相关 `sum(x[:-1]*x[1:]) / sum(x*x)`（x 已逐道去均值）在 500 道上的均值。参考值（同口径）：旧 v1 A 场 ≈ **0.97**；41 点残差本身 pooled 0.84 / 逐道 0.79；一阶差分池 ≈ 0.10。
- **新 A 场验收**：道间均值必须落在 **−0.008 ± 0.02** 内（即 `|mean + 1/129| < 0.02`，理论 sd≈0.004）；逐道值的道间标准差 ≈ 0.09（=1/√130）；任一单道 |值| > 0.35 视为异常。对 B 场做同样统计作对照，两者应无可分辨差异。**不满足则实现有误，不要往下跑。**
- 各档 σ_A、σ_B 的实现值（应约等于 679/1357/2715 nε）。
- QC 图保留原有 6 个面板，但**改写标题/图例**为新叙事：去掉 "real DSS texture"、"(e) A is red/low-freq, B flat"、"temporal wander" 等字样，改为 (a) "A: empirical i.i.d. bootstrap"、(b) "A field (i.i.d. empirical)"、(e) "temporal power spectrum (A and B both flat)"，suptitle 改为 "real DSS i.i.d. bootstrap (A) + white (B)"，并在 (e) 标题或 suptitle 里写出 `new A lag1 autocorr = <值>`。(e) 里 A、B 都应是平的；(a) 里 A 应是高频抖动，不再是平滑漂移。
- `grid_noise_summary.csv` **新增列**：`a_lag1_mean, b_lag1_mean, a_kurtosis, a_skew, a_chan_std_p90p10`（每个数据集各一行），既有列不动。

---

## 4. 重新生成数据并重跑网格（受影响的是全部 27 格）

因为 9 个数据集**每一个**都含 A，所以 A 一改，**27 个格子都要重跑**（C 只是反演端参数，复用同一套数据）。

所有代码块开头先执行（**每开一个新 shell 都要重新定义**）：
```bash
V6=/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100_v6strain_zonalgrid
```

### 4.1 先归档旧结果（必须，别覆盖丢数据）
```bash
cd "$V6"
mv noise_data          noise_data_v1_FLAWED_cumulative_residual
mkdir -p inv_archive_v1_FLAWED && mv inv/bg*_w*_s* inv_archive_v1_FLAWED/
mv inv/grid_comparison_summary.csv inv/grid_sensitivity_heatmaps.png inv/grid_main_effects.png \
   inv/GRID_FINALIZED.txt inv_archive_v1_FLAWED/ 2>/dev/null
mv inv/grid_master.log inv/finalize_grid.log inv/grid_runner.pid inv/.grid_reported \
   inv/grid_sensitivity_heatmaps_fixed.png inv/_render_test.png inv/zonal_L1.stdout \
   inv_archive_v1_FLAWED/ 2>/dev/null
mv fig_deliverable     fig_deliverable_v1_FLAWED
ls inv/                            # 应只剩 _template/ __pycache__/ setup_grid_runs.sh run_all_grid.sh launch_grid.sh compare_grid_results.py finalize_grid.py
ls -d inv_archive_v1_FLAWED/bg* | wc -l   # 27
```
（`inv/_template/`、`inv/setup_grid_runs.sh`、`inv/run_all_grid.sh`、`inv/launch_grid.sh`、`inv/compare_grid_results.py`、`inv/finalize_grid.py` 保留。）

### 4.2 重新生成 9 个数据集
```bash
cd "$V6/noise_adding"
/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python add_noise_grid.py
```
产出 `V6/noise_data/measurement_data_bg{1,2,3}_w{1,2,3}.csv`（各含 .meta）+ `grid_noise_summary.csv` + `grid_noise_qc.png`。先按第 3 节核对自相关与 B 一致性，**不达标不要往下跑**。

### 4.3 重新暂存 27 个 run 文件夹
```bash
cd "$V6/inv" && bash setup_grid_runs.sh
ls -d bg*_w*_s* | wc -l   # 27
```
它会从 `inv/_template/` 拷 5 个模板文件、并把对应数据集拷成每格的 `measurement_data.csv`。

### 4.4 跑 27 个反演（MOOSE，需要 MPI，必须脱离沙箱/终端）

**预检（全部通过才能启动网格；`run_all_grid.sh` 头注释要求在 Codex 沙箱外跑）：**
```bash
cd "$V6/inv"
timeout 60 /rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec -n 2 hostname   # 必须打印 2 行主机名，rc=0
[[ -z "$(pgrep -x combined-opt)" ]] && echo 'no MOOSE running: ok'
ls /rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/fibeRIS/src/fiberis >/dev/null && echo fiberis ok
ls -d bg*_w*_s* | wc -l   # 27
```
若 mpiexec 预检失败（沙箱拦截 socket/进程），**不要**在沙箱内硬跑或改并行参数：改用无沙箱/全权限模式执行本节，或把下面这条启动命令原样交给用户在主机终端执行，然后继续 4.5。

**启动（用可续跑启动器，必须 `< /dev/null`）：**
```bash
cd "$V6/inv"
setsid nohup bash -c 'PAR=2 ZONAL_NP=10 bash launch_grid.sh; echo GRID_ALL_DONE_EXIT=$?' > grid_master.log 2>&1 < /dev/null &
```
- 机器：20 物理核 / 92 GB。**固定 `PAR=2 ZONAL_NP=10`**（README 实测最优；NP=5 单次求解慢 2.7×，4×5 总体反而慢，**别改**）。
- `launch_grid.sh` 只跑缺 `optimized_theta_zonal.txt` 的格子：中断后用**同一条命令**续跑（输出改到 `grid_master_resume.log` 以免覆盖），已完成格子不会重做；**不要**直接重跑 `run_all_grid.sh` 全量。`< /dev/null` 必须加（README 坑 1，曾因此空转 12.5 h）。
- 实测（PAR=2, NP=10）：每次 MOOSE 前向+伴随 ≈ 20–30 min；每格 13–46 次调用，单格墙钟 **4–18 h**（中位 ~5.5 h，bg1_w2_s0 曾 17.4 h）；27 格总计约 **4–5 天**（上轮 25 格 = 95 h）。它是脱离终端的常驻进程，关掉会话也会继续跑。
- 环境由 `run_all_grid.sh` 设置：`PYTHONPATH=…/fibeRIS/src`、`MPLBACKEND=Agg`、`MPLCONFIGDIR=/tmp/mplconfig`、`PYBIN=$HOME/miniforge/envs/moose/bin/python`（HOME=/rcp/rcp42/home/shenyaojin）；MOOSE 可执行 `…/moose_env/moose/modules/combined/combined-opt` 与 mpiexec 路径硬编码在 `_template/107_optimization_runner_zonal_L1.py:185-186`，不要改。
- **判断卡住 vs 正常慢**（只读命令）：`pgrep -x combined-opt | wc -l` 应为 20；为 0 且 grid_master.log 无 `GRID_ALL_DONE_EXIT` 行 → 异常。`wc -l "$V6"/inv/bg*_w*_s*/objective_history_zonal.csv`：在跑的格子每 20–30 min 增 1 行；>90 min 无增长再看 `stat -c %y <cell>/inv_output/simulation_opt_zonal.log` 的 mtime——仍在刷新 = 正常慢；mtime 也停 >90 min 才算卡住。**不要按"4–5 h/格"去杀一个正常的 17 h 格子。**
- 卡住时：`pkill -x combined-opt; pkill -x mpiexec`（**不要** `-f`），再用上面同一条命令（输出到 `grid_master_resume.log`）续跑。
- 进度：`tail -f "$V6/inv/grid_master.log"`；每格完成会在自己文件夹写 `optimized_theta_zonal.txt`。

### 4.5 自动收尾
`V6/inv/finalize_grid.py` 会等 27/27 完成后自动跑 compare 出图并写 `GRID_FINALIZED.txt`。**它里面 `SCHEDULER_PID = 140782` 是旧 PID，`/proc/140782` 已不存在——不改的话它第一次轮询就会在 0/27 时写出 GRID_FINALIZED.txt。** 也**不要**用 `$!` 去取新 PID（交互 zsh 里 `setsid … &` 会 fork，`$!` 是立刻退出的父进程）。启动前按下面改掉 PID 依赖：
```python
# 删除 SCHEDULER_PID 与旧 scheduler_alive()，替换为：
MAX_WAIT_HOURS = 168             # 7 天硬上限（实测 25 格 = 95 h）
GRID_MASTER_LOG = os.path.join(HERE, "grid_master.log")

def scheduler_alive():
    moose_running = subprocess.run(["pgrep", "-x", "combined-opt"], capture_output=True).returncode == 0
    master_done = os.path.exists(GRID_MASTER_LOG) and "GRID_ALL_DONE_EXIT=" in open(GRID_MASTER_LOG).read()
    return moose_running or not master_done

# main() 循环改为：
    t0 = time.time()
    while True:
        done = n_done()
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  done={done}/{TOTAL_CELLS} alive={scheduler_alive()}", flush=True)
        if done >= TOTAL_CELLS:
            break
        if not scheduler_alive():
            print(f"master done & no MOOSE at {done}/{TOTAL_CELLS} -> finalizing partial", flush=True); break
        if time.time() - t0 > MAX_WAIT_HOURS * 3600:
            print("timeout -> finalizing partial", flush=True); break
        time.sleep(POLL_SECONDS)
```
（`GRID_ALL_DONE_EXIT=` 字符串必须与 4.4 启动命令写入 grid_master.log 的标记一致。）只有 `GRID_FINALIZED.txt` 里 `done=27/27 compare_rc=0` 才算完成；done<27 视为未完成，查 grid_master.log 后用 4.4 的同一条命令续跑。然后脱离终端启动：
```bash
cd "$V6/inv"
setsid nohup /rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python finalize_grid.py > finalize_grid.log 2>&1 < /dev/null &
```

### 4.6 全部完成后
1. `cd "$V6/inv" && /rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python compare_grid_results.py` → `grid_comparison_summary.csv`、`grid_sensitivity_heatmaps.png`、`grid_main_effects.png`（finalize 已跑过一次；若 `GRID_FINALIZED.txt` 显示 done=27/27 compare_rc=0 可跳过）。
2. 重跑 5 张代表性正演 QC（各一次 MOOSE 前向，5 张并行、各 `--np 4` = 20 进程，实测每张 ~45–50 min）。
   **不要用裸 `python`**（当前 shell 的 `python` 是 miniforge base，没有 fiberis；moose 环境也必须显式给 PYTHONPATH）。
   先把下面内容写成 `$V6/inv/run_qc5.sh`，再 `bash "$V6/inv/run_qc5.sh"`：
   ```bash
   #!/usr/bin/env bash
   set -u
   INV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   export PYTHONPATH=/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/fibeRIS/src
   export MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig
   PY=/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python
   [[ -z "$(pgrep -x combined-opt)" ]] || { echo 'combined-opt still running, abort'; exit 1; }
   for c in bg2_w2_s0 bg2_w2_s2 bg2_w2_s4 bg1_w1_s0 bg3_w1_s4; do
     ( cd "$INV/$c" && setsid nohup bash -c "$PY run_parameter_history_qc.py --history-file parameter_history_L1.csv --row -1 --label zonal_final --np 4 > qc_zonal.stdout 2>&1; echo QC_DONE_EXIT=\$? >> qc_zonal.stdout" > /dev/null 2>&1 < /dev/null & )
   done
   echo 'launched 5 QC runs (~45-50 min each); check: grep QC_DONE_EXIT inv/*/qc_zonal.stdout'
   ```
   全部 5 个 `QC_DONE_EXIT=0` 后，每格得到 `qc_strain_zonal_final.png`。
3. 重建交付文件夹 `V6/fig_deliverable/`，命名与上一版一致，映射（全部用 `cp`，不要 mv）：
   `01_噪声设计_QC.png ← noise_data/grid_noise_qc.png`、`02_敏感度热图.png ← inv/grid_sensitivity_heatmaps.png`、`03_主效应.png ← inv/grid_main_effects.png`、`04_正演QC_最好_bg1w1_对齐.png ← inv/bg1_w1_s0/qc_strain_zonal_final.png`、`05_正演QC_中噪_对齐s0.png ← inv/bg2_w2_s0/…`、`06_正演QC_中噪_错位2层.png ← inv/bg2_w2_s2/…`、`07_正演QC_中噪_错位4层.png ← inv/bg2_w2_s4/…`、`08_正演QC_最差_bg3_错位4层.png ← inv/bg3_w1_s4/…`、`grid_comparison_summary.csv ← inv/`、`README.md`。
   README 参考旧版 `V6/fig_deliverable_v1_FLAWED/README.md` 的**结构**重写，但把 factor A 的描述改成"时间独立的真实 DSS 测量噪声（一阶差分经验分布自助采样，i.i.d.）"，并注明本版修正了旧版的时间相关问题。旧 README 中"A 是平滑低频漂移/红噪声、(e) A 谱往高频掉 2 个量级、(b) 横条纹=时间相关"等段落必须重写，**不能照抄**。
4. 写一份 `V6/CHANGELOG_dss_noise_redo.md`：说明旧 A 的问题、新 A 的算法、自相关前后对比（同一口径：场逐道均值 0.97 → ≈−0.01）、B 已验证位级不变、哪些结果因此变化、以及第 0/1 节提到的假设（"注入前"窗口无独立佐证；真实差分并非严格独立、本版有意丢弃其相关结构）。

---

## 5. 一定要避开的坑（都是实际踩过的）

- **`pkill -f` / `pgrep -f` 会匹配到你自己的命令行**（因为命令里含同样的字符串），会把自己的 shell 杀掉（退出码 144）。要杀 MOOSE 用按进程名精确匹配：`pkill -x combined-opt`、`pkill -x mpiexec`。
- 交互 shell 是 **zsh**：无引号的 `$VAR` **不做分词**，`for c in $LIST` 只会循环一次。脚本一律用 `bash` 跑，或在 for 里写字面列表（`launch_grid.sh` 内部就是用 bash 数组规避这个）。
- 长任务一律 `setsid nohup … > log 2>&1 < /dev/null &` 脱离终端；漏掉 `< /dev/null` 曾导致任务不独立、空转 12.5 h。
- `compare_grid_results.py` 里必须用 `df["shift"]`，**不能**用 `df.shift`（那是 DataFrame 的方法，会静默匹配不到、热图全空白）。这个已修好，别改回去。
- MOOSE 前向每次会写 ~3 GB exodus；反演器 `107_optimization_runner_zonal_L1.py` 已用 `strip_exodus()` 去掉，别动它。
- 不要用裸 `python`——当前 shell 的 `python` 是 miniforge base；一律用 `/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python` 并显式 `PYTHONPATH=…/fibeRIS/src`。
- 不要重新加噪 v3/v4/v5 那些旧研究，不要碰 `perm5layer_100_v3strain_noise`、`..._v4strain_rfsdss`、`..._v5strain_zonal2p`。

## 6. 完成标准

- [ ] `add_noise_grid.py` 的 factor A 改为一阶差分（丢首增量、逐道标准化、跨道合并 1755 样本池）的 i.i.d. 自助采样，逐道幅值按 `s_real` 缩放、全场归一化；B 公式与随机实现值不变。
- [ ] `_v1_*` 旧函数保留，seed 20261103 下新旧流程生成的 noise_b `assert_array_equal` 通过。
- [ ] 新 A 场逐道滞后-1 自相关道间均值满足 `|mean + 1/129| < 0.02`，A/B 均已打印并写入 `grid_noise_summary.csv` 新列；QC 图 (e) 里 A、B 谱线都平，标题/图例已改为新叙事。
- [ ] 旧数据、旧 run 文件夹、旧日志与旧图已归档到 `*_v1_FLAWED*`，未被覆盖。
- [ ] `finalize_grid.py` 已去掉 PID 依赖；27/27 格重跑完成，`GRID_FINALIZED.txt` 显示 `done=27/27 compare_rc=0`，`grid_comparison_summary.csv` 有 27 个数据行（`tail -n +2 | wc -l` = 27）。
- [ ] 热图、主效应、5 张正演 QC、`fig_deliverable/`（含重写的中文 README）全部重生成。
- [ ] `CHANGELOG_dss_noise_redo.md` 写清改动、前后对比与假设。

预期科学结果：位置偏移（C）仍应压倒性主导；A 轴的影响预计比旧版更小、接近 B（两者现在都是 i.i.d.，仅边缘分布不同，L1 目标对重尾稳健）——**这本身就是要如实报告的结论**。
