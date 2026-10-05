# 代表性正演模型与控制方程

依据：[representative_forward.i](representative_forward.i)，原路径为 `scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100_v2strain/fwd/output_gt/casing_model_test.i`。本说明还核对了工作区当前 MOOSE 源码；源码核对不能单独证明历史可执行文件的编译版本。

## 几何、材料与观测

二维区域为 $[0,100]\times[-50,50]\ \mathrm{m}^2$，网格为 $200\times200$，单元宽高均为 $0.5\ \mathrm m$。未知量为孔隙压力 $p$ 和位移 $\mathbf u=(u_x,u_y)$。

| 参数 | 值 |
| --- | --- |
| Young's modulus $E$ | $3.5\times10^{10}\ \mathrm{Pa}$ |
| Poisson ratio $\nu$ | 0.25 |
| Biot coefficient $b$（有效应力项） | 0.8 |
| Constant porosity $\phi$ | 0.01 |
| Fluid bulk modulus $K_f$ | $2.2\times10^9\ \mathrm{Pa}$ |
| Viscosity $\mu$ | $10^{-3}\ \mathrm{Pa\,s}$ |
| Reference density $\rho_0$ | $1000\ \mathrm{kg\,m^{-3}}$ |
| Gravity | 0 |
| $K_{yy},K_{zz}$ | $10^{-18}\ \mathrm{m^2}$ |

真值横向渗透率为

$$
K_{xx}(y)=\begin{cases}
10^{-15}\ \mathrm{m^2},&-20\le y\le-16,\\
3\times10^{-15}\ \mathrm{m^2},&14\le y\le20,\\
10^{-18}\ \mathrm{m^2},&\text{其余区域}.
\end{cases}
$$

各区间共享网格界面；端点写法不改变区域材料定义。渗透率张量为对角张量。正演真值采用五个材料区，反演输入把同一高度范围细分成 200 个水平层。

## 力学方程

`ComputeSmallStrain` 与 `ComputeLinearElasticStress` 给出小应变各向同性弹性关系：

$$
\boldsymbol\varepsilon=\tfrac12(\nabla\mathbf u+\nabla\mathbf u^T),\qquad
\boldsymbol\sigma'=\mathbb C(E,\nu):\boldsymbol\varepsilon.
$$

`StressDivergenceTensors` 与 `PorousFlowEffectiveStressCoupling` 组合为

$$
-\nabla\cdot(\boldsymbol\sigma'-bp\mathbf I)=\mathbf0,
$$

相应弱形式的孔压贡献是 $-\int_\Omega bp\,\nabla\cdot\mathbf v\,d\Omega$。该输入没有惯性项，也没有损伤、裂缝开度演化或非线性弹性本构。

## 渗流方程：以实际 residual 为准

使用单相、单组分饱和流。Darcy 质量通量为

$$
\mathbf j=-\frac{\rho}{\mu}\mathbf K\nabla p.
$$

`SimpleFluidProperties` 使用 $\rho(p,T)=\rho_0\exp(p/K_f-\alpha_T T)$；输入提供固定温度材料，没有温度演化方程。压力依赖密度的导数为 $\partial\rho/\partial p=\rho/K_f$。

当前本地 `PorousFlowMassTimeDerivative.C` 的 `computeQpResidual()` 使用节点质量集总，且在发现 `total_strain` 材料属性时乘以上一时间步的体积应变因子。对本案例单相饱和、常孔隙度配置，其单元弱残差可记为

$$
R_{p,i}^{n}=\int_{\Omega_e}N_i(1+\varepsilon_v^{n-1})
\frac{\phi\rho_i^{n}-\phi\rho_i^{n-1}}{\Delta t_n}\,d\Omega
+\int_{\Omega_e}\nabla N_i\cdot\frac{\rho_q^n}{\mu}\mathbf K\nabla p^n\,d\Omega.
$$

这里 $i$ 表示质量集总所用节点，$q$ 表示流动项的积分点，$\varepsilon_v=\mathrm{tr}\boldsymbol\varepsilon$。本输入有 `ComputeSmallStrain`，且没有通过 `base_name` 分离质量核与应变材料，因此按当前源码存在该滞后应变因子。

若忽略这一小应变因子、线性化密度，并在流动系数中取参考密度，可用于理论介绍的主阶近似为

$$
\frac{\phi}{K_f}\frac{\partial p}{\partial t}
-\nabla\cdot\left(\frac{\mathbf K}{\mu}\nabla p\right)=0.
$$

这是一种近似表达，不能替代上述实现说明。尤其不能仅因输入中声明了 `PorousFlowConstantBiotModulus`，就把该案例写成
$M^{-1}\dot p+b\dot\varepsilon_v-\nabla\cdot(\mathbf K\nabla p/\mu)=0$：当前质量核读取的是 $\phi\rho$，没有直接读取该 Biot 模量，输入也没有独立的体积膨胀质量核。因此，不宜将已有结果称为标准完整 Biot 储存方程的验证，也不宜把实际实现称为严格单向耦合。

## 初值、边界与输出

- 初始 $p=u_x=u_y=0$。
- 内部竖线 $x=45\ \mathrm m$ 施加 `FunctionDirichletBC`：$p=p_{\rm inj}(t)$，压力序列以 `PiecewiseLinear` 嵌在输入文件中；它是规定压力，不是规定注入流量。
- 外侧压力边界没有另设条件，对应弱形式下自然零法向流量。
- 名为 `confine_x`、`confine_y` 的对象实际是 `NeumannBC(value=0)`，分别施加在左右边界的 `disp_x` 和上下边界的 `disp_y`。它们不是零位移约束；该文件也没有额外的位移 Dirichlet 锚定。论文应按实际声明描述，不能根据对象名称写成固定边界。
- 观测线为 $x=60\ \mathrm m$，$y\in[-25,25]\ \mathrm m$，500 个采样点。反演使用 `strain_yy`，由 `RankTwoAux(total_strain, index_i=1, index_j=1)` 得到；正演同时输出 `disp_y`。
- 时间由 `TimeSequenceStepper` 指定，共 130 个时间点（包括 $t=0$），终止于 $384288\ \mathrm s$；步长非均匀。

理论上的 DSS 观测映射可写作 $d_{rn}=H_r\mathbf u(t_n)=\varepsilon_{yy}(60,y_r,t_n)$。本案例拟合的是无量纲应变，绘图换算为微应变；目标函数内没有乘 $10^6$ 的观测单位转换。代码中的目标整体缩放是另外一回事。

## 对应源码

以下路径均相对项目根目录，可用于进一步核查具体实现：

| 内容 | 本地源码 |
| --- | --- |
| 集总质量与滞后应变因子 | `moose_env/moose/modules/porous_flow/src/kernels/PorousFlowMassTimeDerivative.C` |
| Darcy 质量通量 | `moose_env/moose/modules/porous_flow/src/kernels/PorousFlowFullySaturatedDarcyBase.C` |
| 有效应力中的孔压项 | `moose_env/moose/modules/porous_flow/src/kernels/PorousFlowEffectiveStressCoupling.C` |
| 压力依赖密度 | `moose_env/moose/modules/fluid_properties/src/fluidproperties/SimpleFluidProperties.C` |

本次只读取输入和源码，未修改上述模型设置，也未重新计算数值结果。
