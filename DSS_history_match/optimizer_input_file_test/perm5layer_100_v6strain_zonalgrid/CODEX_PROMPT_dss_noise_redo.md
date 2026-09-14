# 任务：修正 DSS 噪声（factor A）的取法，画清楚它长啥样，只补跑一个格子

只做三件事：**① 改 factor A 的提取方式；② 出一张图让人看懂 DSS 噪声是从哪截的、长什么样；③ 只重跑 `bg3_w3_s0` 这一个格子**（最大 DSS 噪声 + 最大白噪声 + 0 偏移），和旧结果对比。不要重跑整个 27 格网格，不要动其它任何东西。

先定义（每开一个新 shell 都要）：
```bash
V6=/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100_v6strain_zonalgrid
PY=/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python
export PYTHONPATH=/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/fibeRIS/src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig
```
**不要用裸 `python`**（当前 shell 的 python 是 miniforge base，没有 fiberis），一律用 `$PY`。

---

## 1. 现在错在哪（一句话）

生成器 `$V6/noise_adding/add_noise_grid.py` 里 factor A 从真实 DSS 的**累积应变**里取残差，再把一条曲线**拉伸**铺满时间轴——得到的是一条慢漂移（场自相关 **0.97**），不是噪声。真正逐样本的测量噪声在累积应变的**相邻样本一阶差分**里（自相关 0.10，≈时间独立）。

## 2. 数据源（只读）

- 真实 DSS：`/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/output/das_observed/das_strain_waterfall_T1T3.npz`
  key `data`，shape (45, 4620) = **道 × 时间**，单位 millistrain（×1e-3 转 strain），dt=60 s，累积应变。
- 干净合成观测：`$V6/data/obs_strain_yy.csv`（130 时刻 × 500 道）。REF = 2.262e-5（`median_channel_peak()` 已实现）；bg3 = 12%·REF = 2715 nε，w3 同。

## 3. 新的 factor A（改 `add_noise_grid.py`，其它不动）

只重写 `load_real_bg_residual()` 和 `real_texture()` 及 `main()` 里调用它们的几行；`LEVELS_A/B`、`BASE_SEED`、`median_channel_peak()`、factor B 的 `sigma_b * rng.standard_normal(...)`、`noise[0,:]=0` 都不要动。

```
1. block = data[:, 1:42] * 1e-3          # 45×41 背景窗口（跳过恒为0的第0列）
2. diff  = np.diff(block, axis=1)[:, 1:]  # 45×39，丢掉第一个增量（列1是积分启动样本）——这就是逐样本测量噪声
3. z     = (diff - diff.mean(1,keepdims=True)) / diff.std(1,keepdims=True)
   pool  = z.ravel()                      # 45×39=1755 个标准化样本，跨道合并成一个池
   s_real= diff.std(1)                    # 45 个逐道原生 std（保留"逐道幅值不同"）
4. 造 130×500 场：每个输出道 c 随机指定一个真实道 r[c]；
   T[t,c] = pool[k[t,c]] * s_real[r[c]] / median(s_real)，k 是独立自助抽样索引
   → 每个 (t,c) 独立抽，时间独立。禁止沿时间插值/拉伸任何曲线。
5. 全场去均值、除全场 std，再 ×σ_A（=档位×REF）；noise_A[0,:]=0。
```
验收：新 A 场逐道滞后-1 自相关的道间均值应 ≈ 0（`|mean| < 0.03`）；打印出来。

## 4. ★ 关键交付：一张"DSS 噪声怎么来的"图

在生成器里新加函数，输出 `$V6/noise_data/dss_noise_extraction.png`，2×3 六个面板，让人一眼看懂：
- **(a) 原始 DSS 瀑布图**：`data` 全部 45 道 × 4620 时刻（热图，单位 mstrain），用竖条**标出截取窗口**（时间索引 1–41）。→ 从哪截的。
- **(b) 截出来的累积应变**：窗口内随便 5 道的时间序列。→ 看到它是平滑漂移，不是噪声。
- **(c) 一阶差分**：同 5 道的相邻样本差。→ 这才是噪声：抖动、零均值。标出 std≈2e-8。
- **(d) 差分的直方图**：1755 个标准化样本 vs 同 std 的高斯曲线。→ 看经验分布（重尾）。
- **(e) 生成的噪声场**：bg3 档的 130×500 A 场热图。→ 独立散斑，没有条带。
- **(f) 一条道：新 A vs 白噪声 B**：时间序列叠画，标题写 `A lag1 autocorr = <值>`（旧版 0.97 → 新 ≈0）。

标题用中文也行。这张图是本任务最重要的产物。

## 5. 重生成数据 + 只跑 bg3_w3_s0

```bash
# 备份旧数据，重生成 9 个数据集（纯 python，几秒）
cd "$V6" && mv noise_data noise_data_v1_old
cd "$V6/noise_adding" && $PY add_noise_grid.py      # 产出 noise_data/measurement_data_bg*_w*.csv + dss_noise_extraction.png
```
先确认 `dss_noise_extraction.png` 出来了、自相关 ≈0，再往下。

```bash
# 新建一个独立 run 文件夹（不覆盖旧的 inv/bg3_w3_s0，留着对比）
cd "$V6/inv" && mkdir -p bg3_w3_s0_v2
cp _template/107_optimization_runner_zonal_L1.py _template/optimize.i _template/forward_and_adjoint.i \
   _template/plot_inversion_qc.py _template/run_parameter_history_qc.py bg3_w3_s0_v2/
cp ../noise_data/measurement_data_bg3_w3.csv bg3_w3_s0_v2/measurement_data.csv
cp ../noise_data/measurement_data_bg3_w3.meta bg3_w3_s0_v2/measurement_data.meta

# 只跑这一个格子（MOOSE 需要 MPI，必须脱离终端；单格用满 20 核）
cd "$V6/inv/bg3_w3_s0_v2"
setsid nohup bash -c "ZONAL_MODE=exact ZONAL_NP=20 $PY 107_optimization_runner_zonal_L1.py > zonal_L1.stdout 2>&1; echo DONE_EXIT=\$? >> zonal_L1.stdout" > /dev/null 2>&1 < /dev/null &
```
- 先用 `timeout 60 /rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec -n 2 hostname` 确认能起 MPI；起不来就把上面那条启动命令交给用户在主机终端跑。
- 一格约 **5–18 小时**，别在前台等。看进度：`wc -l zonal_L1.stdout`、`wc -l objective_history_zonal.csv`（每 20–30 min 涨 1 行算正常）。
- 跑完标志：`optimized_theta_zonal.txt` 出现，且 `zonal_L1.stdout` 末尾 `DONE_EXIT=0`。

## 6. 跑完后：QC + 对比

```bash
cd "$V6/inv/bg3_w3_s0_v2"
$PY run_parameter_history_qc.py --history-file parameter_history_L1.csv --row -1 --label zonal_final --np 20
```
出 `qc_strain_zonal_final.png`（观测/模拟/残差/alpha 剖面四联图）。

然后写一个小对比 `$V6/inv/bg3_w3_s0_v2/COMPARE_v1_vs_v2.md`，列出：
- 旧 `inv/bg3_w3_s0/optimized_theta_zonal.txt` vs 新 `bg3_w3_s0_v2/optimized_theta_zonal.txt` 的 θ_frac、θ_srv，以及各自与真值的偏差（真值 θ_frac = log10(3e-15) = −14.5229，θ_srv = −15.0）
- 两版 A 场的自相关（0.97 vs 新值）
- 一句话：换成正确的时间独立 DSS 噪声后，结果变了多少

把 `dss_noise_extraction.png`、`qc_strain_zonal_final.png`、`COMPARE_v1_vs_v2.md` 三样放到 `$V6/fig_deliverable/redo_bg3_w3_s0/` 里。

## 7. 坑

- 杀 MOOSE 用 `pkill -x combined-opt; pkill -x mpiexec`，**不要 `-f`**（会把自己 shell 杀掉）。
- 交互 shell 是 zsh，无引号 `$VAR` 不分词；脚本用 bash 跑。
- 不要碰 `inv/bg*_w*_s*` 那 27 个旧文件夹和现有的敏感度图，它们是对照基线。

## 8. 完成标准
- [ ] `dss_noise_extraction.png` 六个面板齐全，能看懂噪声从哪截、长啥样
- [ ] 新 A 场自相关 ≈0 且已打印
- [ ] `bg3_w3_s0_v2` 跑完，有 `optimized_theta_zonal.txt` 和 `qc_strain_zonal_final.png`
- [ ] `COMPARE_v1_vs_v2.md` 写清新旧 θ 与偏差
