# Objective、regularization、gradient update 与 optimizer

完整脚本：[106_optimization_runner_L1.py](original/106_optimization_runner_L1.py)。带原文件行号的逐字摘录：[key_code_excerpts.md](key_code_excerpts.md)。本说明中的数值为脚本默认设置；环境变量可覆盖部分参数，具体历史运行还应结合日志核对。

## 参数化与初始模型

$$
\alpha_i=\log_{10}\left(K_{xx,i}/\mathrm{m^2}\right),\qquad
K_{xx,i}=10^{\alpha_i}\ \mathrm{m^2},\qquad i=1,\ldots,200.
$$

`forward_and_adjoint.i` 的 `ParsedOptimizationFunction(expression='10^alpha')` 完成此映射。层厚为 0.5 m。`make_bounds()` 允许中心位于 $[-25,25]$ m 的 100 层在 $[-25,-10]$ 中变化；另外 100 层固定为 $\alpha=-18$。这才是外层实际约束，master 文件中的统一上下界不是全部外层约束的描述。

默认初值背景为 $-18$，但在 $y\in[-20,-16]$ 和 $[14,20]$ m 两个真值异常区设置 $-16$。因此这是带异常位置初始信息的恢复试验，不能写成从完全均匀初值出发的盲定位试验。

## Data objective

$$
J_d(\boldsymbol\alpha)=\frac12\sum_{(r,n)\in\mathcal D}
\left[\varepsilon_{yy}^{\rm sim}(x_r,y_r,t_n;\boldsymbol\alpha)-d_{rn}\right]^2.
$$

MOOSE 的 `OptimizationData` 在观测点取 `strain_yy`，形成 `simulation_values - measurement_values`，再计算半平方和。Python 不重新计算残差，而是读回 `OptimizationReporter/objective_value`，必要时从日志解析 `Objective value`。默认没有观测噪声协方差、样本数归一化或时间步长权重，因此不能直接把它写成一个已经做了 $\Delta t$ 求积的连续时间积分。

已有该案例的目标值核对使用 $t>0$ 的 129 个时刻、每时刻 500 个观测点（64500 项）；原测量表还包含初始时刻。这里的有效时间集合来自 MOOSE 执行时机，Python 中没有显式的 `time > 0` 过滤代码。

源实现：项目根目录下 `moose_env/moose/modules/optimization/src/reporters/OptimizationData.C` 的 `computeMisfit()` 和 `computeMisfitValue()`。

## Regularization

脚本 `l1_obj_and_grad(x)` 实现相对背景的平滑 L1 惩罚：

$$
R(\boldsymbol\alpha)=\beta\sum_{i=1}^{200}
\left[\sqrt{(\alpha_i-\alpha_{\rm ref})^2+\delta^2}-\delta\right],
$$

$$
\frac{\partial R}{\partial\alpha_i}
=\beta\frac{\alpha_i-\alpha_{\rm ref}}
{\sqrt{(\alpha_i-\alpha_{\rm ref})^2+\delta^2}},
\qquad \alpha_{\rm ref}=-18,\quad\beta=2\times10^{-11},\quad\delta=0.05.
$$

固定外部层等于背景，所以它们的惩罚及导数均为零。此正则项鼓励相对于背景的稀疏偏离；没有相邻层差分，不是 TV 或空间平滑惩罚。

传给优化器的是

$$
\widetilde J=10^6(J_d+R),\qquad
\nabla\widetilde J=10^6(\mathbf g_{\rm MOOSE}+\nabla R).
$$

其中 `BASELINE_OBJ=0`，`GRAD_CORRECTION=1`。目标整体缩放不改变数学上的最优点，但会影响数值停止阈值的意义。

## Gradient：伴随计算与 Python 合并

从理论上，将全部时间步的离散状态残差写为 $F(\mathbf q,\boldsymbol\alpha)=0$，采用拉格朗日量 $\mathcal L=J_d-\boldsymbol\lambda^T F$，有

$$
F_{\mathbf q}^T\boldsymbol\lambda=(J_d)_{\mathbf q},\qquad
\frac{dJ_d}{d\boldsymbol\alpha}=(J_d)_{\boldsymbol\alpha}-F_{\boldsymbol\alpha}^T\boldsymbol\lambda.
$$

这是统一的离散伴随表述；符号取决于拉格朗日量和源项约定。它不能单独证明本案例离散实现的每一项都与精确离散导数一致，精度应由已有梯度检验支持。

现有输入的具体流程是：

1. `TransientAndAdjoint` 运行 `nl0` 正演系统和 `adjoint` 伴随系统。
2. `OptimizationData(variable='strain_yy')` 计算应变残差 $m$。
3. 在 `disp_y_adjoint` 上，用 $y\pm0.25$ m 两个点源以及 $\pm m/0.5$ 的幅值近似应变观测的伴随源。这是间距 $h=0.5$ m 的偶极实现；不能未经核对就把它等同于离散 `RankTwoAux` 观测算子的精确转置。
4. 每层的 `PorousFlowOptimizationAnisotropicDiffusionInnerProduct` 使用 `forward_variable=pp`、`variable=pp_adjoint`、`component=0`、对应 `perm_layer_i` 函数，在 `ADJOINT_TIMESTEP_END` 计算梯度贡献。
5. master 把 `inner_product` 传回 `grad_perm_1` 至 `grad_perm_200`。Python 读取末行，将其按对 $\alpha$ 的梯度使用，加上解析正则化梯度，并与目标同步乘 $10^6$。Python 不再乘一次 $\ln(10)10^{\alpha_i}$。

若从 $K$ 空间推导，则链式法则为 $\partial K_{xx,i}/\partial\alpha_i=\ln(10)10^{\alpha_i}$。输入通过优化函数关联这一参数化，但本次在本地 MOOSE 源文件检索中没有找到上述专用 inner-product 对象的实现，因此这里保留其原始配置，不猜写它内部的符号、密度／黏度因子和时间累积细节。

## Optimizer 与参数更新

实际外层调用是：

```python
res = minimize(
    objective_and_gradient,
    x0,
    method="L-BFGS-B",
    jac=True,
    bounds=bounds,
    callback=_checkpoint,
    options={"maxiter": MAXITER, "ftol": FTOL, "gtol": GTOL},
)
```

`objective_and_gradient(x)` 每次对候选参数计算目标和梯度。方向、有限记忆拟牛顿更新、边界处理和线搜索均在 SciPy L-BFGS-B 内部完成，脚本没有自行写 `x -= learning_rate * gradient`。

`optimize.i` 中虽然使用 `Optimize`、`taobqnls`，但 `-tao_gatol=1e50` 将它用于提供当前候选点的目标与梯度；控制完整反演过程的是 Python 外层 L-BFGS-B。

默认 `maxiter=300`、`ftol=1e-7`、`gtol=1e-10`。此外 `_checkpoint(xk)` 对接受步计算 $\Delta\alpha$：当 $\|\Delta\alpha\|_2\le0.012$ 且 $\|\Delta\alpha\|_\infty\le0.006$ 连续出现 3 次，并且接受步数量至少为 5 时，通过 `StopIteration` 停止。它是步长停止条件，不等同于梯度收敛。

`objective_history_L1.csv` 的 `iter` 实际计数函数评估，可能包含线搜索试探点；`step_history_L1.csv` 才计数接受步。绘制收敛历史时要标清横轴。

原脚本还保留了失败分支：MOOSE 失败或输出解析异常时返回 `1e10` 和零梯度。这是原有异常处理约定，不能作为正常目标函数的数学导数来描述。本次原样复制代码，没有变更其行为。

## 在输入文件中快速定位

| 文件 | 行号 | 内容 |
| --- | --- | --- |
| `original/forward_and_adjoint.i` | 1480 | 正演 Kernels |
| 同上 | 1524 | 第一层 `10^alpha` 参数函数 |
| 同上 | 5075 | 各层梯度 inner-product 对象 |
| 同上 | 10761 | 应变 objective 与偶极幅值 reporter |
| 同上 | 10817 | 偶极 `ReporterTimePointSource` |
| 同上 | 10841 | `TransientAndAdjoint` |
| `original/optimize.i` | 16 | 观测 CSV 与列名 |
| 同上 | 30 | forward MultiApp |
| 同上 | 40 | 参数、目标值及梯度传递 |
| 同上 | 863 | 内层 TAO 配置 |

逐层重复块保留在完整输入文件中，没有用省略版替换它们。上述行号与本目录复制版本一致。
