# V1 断层:MOOSE 张性 + DDM 剪切 反演井内压力 — 方法说明

面向 V1 断层的 DAS 应变历史匹配。把观测应变分解为 **MOOSE 模拟的张性(tensile)** 与
**DDM 模拟的剪切(shear)** 两部分,叠加后与观测比对,反演出**每个 4 小时的井内(裂缝)压力**。

$$\varepsilon_{\text{obs}}(z,t)\;\approx\;\varepsilon^{\text{MOOSE}}_{\text{tensile}}(z,t;\,P)\;+\;\varepsilon^{\text{DDM}}_{\text{shear}}(z,t)$$

---

## 1. 阶段划分与时间约定

- **MOOSE 张性**:贯穿 **T1–T3** 全程都活跃。
- **DDM 剪切**:只在 **T2–T3** 活跃(T2 之前 = 0),叠加在张性之上。
  - 即 T1–T2:model = 张性;T2–T3:model = 张性 + 剪切。**张性不在 T2 停,剪切是加上去的。**
- **T1 = 2025-02-24 15:00**(张性起裂时刻;观测应变约此时才开始增长,注入压力也从此裁取)、
  **T2 = 02-28 00:00**、**T3 = 03-03 22:00**(UTC-7)。
  - 注:所有 notebook 里标注的 T1 是 11:00,但 11:00–15:00 应变≈0,改用 15:00 对结果无影响(已验证)。

## 2. 二维 MOOSE 几何(光纤⊥断层的那个截面)

- 建模平面 = **光纤与裂缝相交的面**。**Y 轴 = 光纤方向 = 测深 MD**;**X 轴 = 断层面内(走向)**,
  竖直(断层高度 4000 ft,很高)方向按**平面应变**处理。
- 裂缝在 MOOSE 里 = 一组 block:**hf(裂缝核)+ 嵌套 SRV zone**。
- **MD↔Y 映射**:`MD(ft) = 10373.4 + Y(m)/0.3048`,缝心 Y=0 ↔ **MD 10373.4 ft**(DAS 穿缝深度)。
- **SRV 沿 MD(Y)的宽度 ≈ 90 ft** —— 取自**观测**特征宽度(DDM 零厚度平面给不了宽度,这正是要用 MOOSE SRV 区的原因)。
- 光纤用 `LineValueSampler` 沿 Y 采 **`strain_yy`**;因为光纤⊥裂缝,DAS 测的轴向应变就是 strain_yy,**无需张量旋转**。

**精修几何(当前版本)**:SRV 用**渐变 4 层**(减少方盒子效应),裂缝加长到 400 ft、光纤采样点移到离缝心 40 ft
(避开裂缝端部畸变)。参数:

| 层 | 沿 MD 高度 | 渗透率 (m²) | 孔隙度 |
|---|---|---|---|
| srv_l1 | 100 ft | 3e-16 | 0.08 |
| srv_l2 | 75 ft | 1e-15 | 0.10 |
| srv_l3 | 50 ft | 3e-15 | 0.12 |
| srv_l4 | 25 ft | 1e-14 | 0.14 |
| hf(核) | 0.2 ft | 1e-13 | 0.16 |
| matrix | — | 1e-18 | 0.01 |

弹性:E=50 GPa、ν=0.2、Biot=0.7;流体:水(体积模量 2.2 GPa)。域 200 m × 200 m(米制)。

## 3. 三路输入

1. **观测**:DAS 4 小时平均**应变**剖面(MD 10200–10500,T1 参考),来自 LFDAS 处理链。
2. **DDM 剪切**:从 07152026 的**双断层总应变**里提取。方法:单个矩形 DDM 元张性应变**可分离**
   `ε_tensile(z,t)=width(t)·g(z)`(空间形状 g 不随时间变);T1–T2 剪切=0,那段总应变即纯张性,
   用它拟合 g(z)(残差 1.1%),再外推到 T2–T3,**剪切 = 总 − 张性**。得到的就是 Pengchao 原模型的
   fault2 剪切、在其原尺度上(T2 前≈0,T2 后打开,呈反对称滑移特征)。
3. **压力**:**DAS 反推的平滑压力**(`das_injection_pressure_HISTORYMATCH_C1p63e7`,单调 4056→6077 psi)。
   **为什么用平滑压力**:井口注入压力一跳一跳,但井与光纤之间隔着地层,地层是**低通滤波**,
   到裂缝处压力应当平滑 —— 用井口原始脉动压力拟合很差(方差只降 20%),换平滑压力后大幅改善。

## 4. 物理与反演原理

- MOOSE 用 **porous_flow(孔弹性)**:裂缝内加压 → 孔隙压扩散进 SRV → 岩石变形 → 沿光纤取 strain_yy。
- **关键:孔弹性应变对压力扰动 Δp 线性**(渗透率/模量都是常数)。因此:
  $$\varepsilon^{\text{MOOSE}}_{\text{tensile}}(z,t;\,s)=s\cdot\varepsilon^{\text{MOOSE}}_{\text{tensile}}(z,t;\,s{=}1)$$
  **跑一次基准压力就够**,任意 scale 只是乘个数 —— 反演变成解析最小二乘,不用逐个压力重跑 MOOSE。
- **反演**:找 scale `s` 使
  $$\min_s\;\big\|\;\varepsilon_{\text{obs}}-\big(s\cdot\varepsilon^{\text{MOOSE}}_{\text{tensile}}+\varepsilon^{\text{DDM}}_{\text{shear}}\big)\;\big\|$$
  `s` = **裂缝实际受到的压力,占 DAS 单深度标定压力(C1p63e7)的比例**。井内压力 = IC + s·(DAS Δp)。

## 5. 操作步骤(可复现)

1. **fiberis** 建 2D SRV 几何 → 生成 MOOSE `.i`(`run_v1_srv_t1_1500.py`)。
2. 喂平滑 DAS 压力(前补 T1=15:00 @ 初始压力,使 Δp 从 0 起),跑 `porous_flow-opt`(20 进程,~3 分钟,收敛)。
3. 抽光纤 `strain_yy`(z,t),`Y→MD` 映射,T1 参考,转 millistrain(`analyze_v1_srv_t1_1500.py`)。
4. 4h 平均,和观测、DDM 剪切对齐到 15:00 网格(时间插值),全部参考到 T1=15:00。
5. 解析扫 `s`,输出最佳 scale + **井内压力历史** + 对比图。

## 6. 结果(T1=15:00)

- **最佳压力 scale s ≈ 0.48**,拟合 **RMS 0.0245 → 0.0125 mε,方差降低 49%**。
- **反推井内压力:4056 → ~5033 psi(Δp 峰 ~977 psi),单调**。（即裂缝实际受压约为单深度 C1p63e7 标定值的一半。）
- **稳健性**:逐快照 s 稳定在 ~0.5;两种 SRV 几何(2 层/渐变 4 层)、两种 T1(11:00/15:00)都给 **s≈0.5、压力峰 ~5000 psi** —— 压力结果对几何/参考约定不敏感。
- 模型同时复现了**张性红瓣 + T2 之后的剪切特征**,整体形态与观测一致。

## 7. 局限(需向师兄说明)

- 残差(方差还剩 ~50%)主要是**形状**,不是压力:
  观测的张性瓣是一个**集中的圆尖峰**(变形集中在裂缝面上,DAS gauge 平滑后成峰);
  MOOSE 现在建的是**被加压的多孔 SRV 区**,变形**分散**在整个区里,呈**平顶台**。
- 加渐变 perm、移光纤都改不了这个**"孔弹性分散 vs 张开集中"的本质差异**。
- **结论**:井内压力的**量级与历史(单调 4056→~5000 psi)稳健可信**;剖面**形状**的精细匹配受当前
  "多孔区加压"建模方式限制,要治本需把裂缝建成真正的**张开**(cohesive/aperture 单元或薄高柔度核)。

## 8. 交付物

- **图**(`figs/tensile_fault_qc/v1_srv_t1_1500/`):
  `deliverable_obs_vs_model_waterfall.png`(观测 vs 模型 应变 waterfall,含 4h 剖面叠加)、
  `match_and_pressure.png`(misfit-vs-scale、剖面对比、逐快照 scale、井内压力历史)。
- **CSV**(`output/v1_srv_t1_1500/inferred_fracture_pressure_history.csv`):
  每时刻的 DAS 压力、反推裂缝压力、Δp。**主交付物**。
- **脚本**(`scripts/tensile_fault/data_transfer/from_pc/`):
  `run_v1_srv_t1_1500.py`(建几何+跑 MOOSE)、`analyze_v1_srv_t1_1500.py`(反演+出图)、
  `extract_ddm_shear_from_reference.py`(DDM 剪切提取)。
- **附带修复**:fiberis `model_builder.py` 的 `StitchedMeshGenerator`(deprecation 已于 2026-07-01 过期、报错)
  改为 `StitchMeshGenerator`(纯改名),否则任何 fiberis MOOSE 生成都跑不了。
