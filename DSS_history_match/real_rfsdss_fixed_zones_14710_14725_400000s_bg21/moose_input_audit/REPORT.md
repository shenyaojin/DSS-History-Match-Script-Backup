# MOOSE 输入审查：现有证据与最小候选修改

审查日期：2026-09-19。只读取已经完成的 fixed_yy / free_yy 对照及本地 MOOSE 源码；未修改求解输入、未启动正演或反演。新增诊断脚本 `../audit_moose_inputs.py`，计算结果见 [metrics.json](metrics.json)。

主要输入：[fixed_yy/forward_and_adjoint.i](../comparison_xx_yy_60/fixed_yy/forward_and_adjoint.i)。下述行号对应本次审查版本，SHA256 已记录在 metrics.json。

## 1. 确认的空间分辨率限制：值得做局部网格收敛试验

**当前实现：** 第 3587 行 `strain_yy` 是 CONSTANT/MONOMIAL 元素变量；第 5075 行 `RankTwoAux` 输出 total_strain 的 yy 分量。45 个 DSS 通道仅落在 9 层网格上。A 和 B 各只有一层，B 的 6 个观测通道对应完全相同的模拟值；已核实导出结果同层差值为零。

**量化：** 对每层、每个时刻取该层观测均值，这是任意非负分片常数预测能够达到的最小平方误差；这里允许时间曲线完全自由，未施加流动/力学方程，所以是乐观下界。按现有目标排除初始帧：

| 指标 | 当前最终模拟 | 最佳分片常数近似 |
|---|---:|---:|
| J_data | 3.28083645e-10 | 1.71953447e-10 |
| RMS，微应变 | 0.336207 | 0.243399 |

后者为当前 J_data 的 **52.41%**。在当前观测算子和网格下，这部分误差无法通过调整 xx、yy 消除。它包含层内观测差异及噪声，并不表示加密必然消除同等误差，也不证明所有细节都是可信物理信号。

[观测 / 最佳分片常数近似 / 当前模拟](spatial_resolution.png)

**最小候选修改：** 保持高渗区物理边界及三个区的参数数量，局部细分观测区 y 网格，在相同最终参数下先比较正演。原来的 200×200 网格保留作基线；这是有量化依据的独立收敛试验。同步重建观测位置对应的元素映射、伴随采样权重及参数到子层的映射；不能只改网格尺寸或平滑最终图。若需要仪器测量长度上的应变平均，应先查明 DSS 测量长度，不能拿通道间距当测量长度。

**限制：** 分片常数最佳近似仍能重现较早的响应，而当前模型明显较晚；时间形状问题不全是空间分辨率造成的。

## 2. 流动储集与力学反馈：应明确目前采用了怎样的近似

**当前实现：** 第 1478 行时间核为 `PorousFlowMassTimeDerivative`，各层使用 `PorousFlowPorosityConst = 0.01`；第 1464 行 Dictator 只注册 pp。力学侧通过 `PorousFlowEffectiveStressCoupling` 接受压力作用，Biot 系数 0.8。虽然定义了 `PorousFlowConstantBiotModulus` 和 `PorousFlowVolumetricStrain`，当前时间核没有使用 Biot 模量，也没有标准的 `b * d(epsilon_v)/dt` 储集项。

本地 `PorousFlowMassTimeDerivative.C` 的残差含 `(1 + trace(total_strain_old))` 乘子，因此不能说变形完全没有反馈；但这不等价于完整的线性 Biot 水力–力学耦合。固定孔隙率下，主要压力储集系数近似为 phi/Kf = 4.54545e-12 Pa^-1。

**候选修改的理由：** 如果目标是完整的饱和孔弹性，这里应采用与该方程一致的储集和位移反馈。若原模型有意采用弱耦合近似，则属于待检验的模型假设，不能仅凭拟合一般就断言程序错误。

本地版本已有 `PorousFlowFullySaturatedMassTimeDerivative` 的 HydroMechanical 模式。候选改动为：

```text
dictator: porous_flow_vars = 'pp disp_x disp_y'
dt: type = PorousFlowFullySaturatedMassTimeDerivative
    coupling_type = HydroMechanical
    biot_coefficient = 0.8
    multiply_by_density = true
```

替换时删除旧时间核专用的 `fluid_component` 参数。该时间核已包含体应变项，不要再叠加另一份体积膨胀质量项。现有本地 `PorousFlowVolumetricStrain` 可从 total_strain 与旧值计算应变率，无需为此擅自替换小应变力学模型。密度加权应与现有 Darcy 核保持一致。以上仅为候选配置，尚未通过新输入检查、正演或梯度检查。

**同时存在的参数一致性问题：** 输入 `solid_bulk_compliance = 1e-11 Pa^-1`。本地源码明确此项是排水体积模量的倒数；由 E=35 GPa、nu=0.25 得 Kd=23.3333 GPa，故应为 4.285714e-11 Pa^-1。相应 1/M 从 6.12545e-12 变为 1.131688e-11 Pa^-1。当前 M 未被时间核使用，所以单改 compliance 不应改善当前拟合。

**限制：** 增加储集不保证响应提前；在其它条件相同时甚至可能使扩散更慢。应以方程一致性为理由，固定参数做单独对照。更换方程后，原梯度验证不能直接沿用。

依据：[MOOSE 储集核](https://mooseframework.inl.gov/source/kernels/PorousFlowFullySaturatedMassTimeDerivative.html)、[Biot 模量定义](https://mooseframework.inl.gov/source/materials/PorousFlowConstantBiotModulus.html)。具体可用参数以本地源码为准。

## 3. 名称叫 confine，实际为零牵引；需区分数学定位与物理约束

**当前实现：** 第 10820、10826 行 confine_x / confine_y 都是 `NeumannBC value=0`，不是零位移。没有找到位移固定点或显式刚体零空间处理。边界条件名称不能作为围压或位移约束的证据。

**可疑之处：** 自由力学问题保留刚体平移和转动自由度，位移解不唯一；但应变不受纯刚体运动影响。当前求解成功，也未发现明确的奇异矩阵报错，不能据此把拟合偏差归因于刚体自由度。

**最小候选修改：** 在独立正演副本中仅消除刚体自由度，例如选择合适的最少点约束，或经本地版本确认的零空间处理。检查应变是否基本不变，先解决数学定义问题。

**物理边界另议：** 深部岩体中截取的 100 m 方块采用全自由边界是否合理，值得做边界/域尺寸敏感性试验。但把所有边直接固定会改变物理响应，不能作为无条件修复，也不能只为压低误差而选约束。

依据：[MOOSE NeumannBC](https://mooseframework.inl.gov/source/bcs/NeumannBC.html)。

## 4. 压力时间转换暂未发现错误；空间加载仍是简化假设

从原始 prod/gauges/pressure_g1.npz 按现有时间戳、psi→Pa、减去 DSS 起始压力的方法重建，与输入 PiecewiseLinear 曲线最大差值仅 3.73e-9 Pa。这个结果验证转换实现，并不独立证明现场时钟或井底代表性。

压力约 7 万秒开始显著上升，B 区实测也在相近时段响应；模拟正应变的主要增长明显更晚。[压力与 B 区响应](pressure_and_zone_b.png)。模拟图使用正半波输出，不能把零值直接解释成原始有符号应变为零或压力尚未到达；下一次诊断正演应额外导出有符号 strain_yy，不改变目标函数。

第 1411 行 injection_well_nodes 覆盖 x=45 m、y=0–100 m，整条线使用同一个压力历史。80 ft 井距在 config 中注明来自模板默认值，并非测量确认值。两者都可能影响物理解读，但修改前需要井轨迹、连通区间或完井资料；目前没有理由任意移位压力曲线、缩短井距或局部化压力源。

## 建议顺序

1. 保留已经完成的两组结果为基线。先在固定最终参数下确认刚体自由度处理不改变应变，并额外输出有符号应变；此步不新增反演参数。
2. 局部网格收敛试验：这是目前与拟合表现直接相关、证据最明确的数值限制。高渗区位置、宽度和参数数量保持原定义。
3. 单独测试完整 HM 储集及与 E、nu 一致的 compliance，避免和网格/边界物理修改混在一轮里。
4. 根据正演证据决定保留哪些候选，再重新验证相应梯度并反演。压力源几何及物理远场边界留待现场依据支持。

所有候选均保留 130 个输出时刻 / 129 步、上限 240、隐式 Euler、原 .001 求解容差和 LU/MUMPS。没有证据支持通过提高时间步数量、收紧容差、改变 beta 或增加更多自由参数来处理上述问题。
