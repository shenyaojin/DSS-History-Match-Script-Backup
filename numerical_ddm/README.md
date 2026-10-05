# 三维数值 DDMPY / fibeRIS

实现位于 `fibeRIS/src/fiberis/moose/numerical_ddm.py`。这里的数值解全部由 MOOSE 三维有限元求得；DDMPY 仅在验证脚本中独立产生参考场。最终实测表见 `output/numerical_ddm/RESULTS.md`，每一项结论以该表及原始日志为准。

直接查看已执行的 [DDMPY_0930_vs_MOOSE.ipynb](DDMPY_0930_vs_MOOSE.ipynb)：包含原版建模与重新计算、真实MOOSE结果、位移/应变/应变率图、误差和成本表、收敛与限制。默认读取已完成的30帧数值运行，设置 `RUN_MOOSE=True` 可在notebook中重算。`build_review_notebook.py` 使用本机base Python和IPython顺序执行各代码单元并保存输出，不需要启动Jupyter内核。

侧重方程与 API 的 [DDMPY_MOOSE_usage_comparison.ipynb](DDMPY_MOOSE_usage_comparison.ipynb) 并列展示原版与数值后端的原生建模、最小迁移、实际求解和结果读取。明确区分相同的围岩方程与有限域薄层近似；运行全部单元会重新求解 100 m × 60 m 的三维纯张开算例。也可在根目录执行 `~/miniforge/bin/python scripts/numerical_ddm/build_usage_notebook.py`，重新生成全部实际输出。

## 运行

在工作区根目录：

```bash
source ~/miniforge/etc/profile.d/conda.sh
conda activate moose
export MPLCONFIGDIR=/tmp/mplconfig
python scripts/numerical_ddm/run_validation.py --np 8 --h 7.5 --name acceptance
```

得到 `output/numerical_ddm/acceptance/{model.i,model.json,command.txt,progress.log,log.txt,run.json,comparison.npz,metrics.json,comparison.png}` 及 MOOSE 的 `solution_receivers_0001.csv`。图中包含三个独立位移分量和井轴向应变。接收点 x=-100:2:100、y=17、z=40 m；所有 101 点预先固定并全部进入误差指标。

已运行的同参数7.5 m网格：位移L2误差0.0540%，共同观测算子下应变0.3614%；额外对照细采样解析导数时，包含端点处理的总应变误差0.7984%。10 m网格相应值为0.0487%、0.9590%、1.2299%，因此推荐7.5 m作为基准验收命令。位移误差轻微非单调反映有限边界、薄层和网格误差的组合，不宣称各项误差均单调下降。

完整收敛与扩展验证（已完成基础算例会保留，扩展算例重新运行）：

```bash
python scripts/numerical_ddm/audit_reference.py
python scripts/numerical_ddm/run_validation.py --np 8 --suite
python scripts/numerical_ddm/summarize.py
```

输入文件是独立的，不依赖 Python、DDMPY 或外部加载文件。切换到该算例目录，直接运行 `command.txt` 中的命令即可。其 `file_base=solution` 相对于运行目录。MPI 在本环境的受限沙箱内无法创建 UCX socket，需要正常终端或已授权的沙箱外执行。

编译：`bash scripts/numerical_ddm/build.sh`。该脚本激活原有 `moose` conda 环境（包含 libMesh、PETSc、WASP、MPI 与编译器），然后 `make -C moose_env/moose/modules/combined -j8`。静态快照使用现有 framework 和 SolidMechanics 对象；固定几何的时序加速可选用此次新增的 `ConstantJacobianProblem`（combined 应用）。完整构建成功日志为 `output/numerical_ddm/build_resumed_configured.log`，新增扩展编译成功日志为 `build_constant_jacobian.log`。早期失败和中断日志均保留，不作为成功证据。

## 数学问题与离散化

SI 单位：长度/位移 m，E Pa，时间 s，应变无量纲，应变率 s⁻¹。DDMPY 不主动转换单位。`Element.mu` 是泊松比，不是剪切模量。基准 E=30e9 Pa，ν=0.3，长度100 m、高度60 m、输入开度0.001 m。x1 是矩形长度方向，x2 是面内高度方向，x3 是法向。输入 b=(S1,S2,width)=u(0+)−u(0−)；DDMPY 为解析公式内部储存 −b，新接口保留这一储存约定。

目标是无限均匀各向同性全空间中无体力的静态平衡：

```
div σ = 0,  σ = λ tr(ε−ε*) I + 2G(ε−ε*)
ε = sym(grad u),  G=E/[2(1+ν)],  λ=Eν/[(1+ν)(1−2ν)]
u -> 0 at infinity
ε* = sym(b ⊗ n) δ_S       (理想矩形位移间断)
```

围岩两侧面力平衡，位移跳跃给定，不反求开度。数值正则化将 δ_S 换为矩形柱内 `1/thickness`。局部 n=(0,0,1)，薄层本征应变为

```
ε*33=width/thickness
ε*13=ε*31=S1/(2 thickness)
ε*23=ε*32=S2/(2 thickness)
```

因此积分的法向塑性畸变给出输入跳跃；剪切张量分量有必要的 1/2。有限厚度不是数学上的零厚界面：它的位移连续，也不保证任意两点的实测位移差恰等于 b；离开边缘和薄层后，随厚度趋零逼近位移间断。模型保留相同均匀弹性模量；没有压力、流动、孔弹性、惯性或人为软化层。FEM 通过弱平衡传递面力，残量收敛保存在日志中；尚未额外验证逐点面力跳跃收敛。

`ComputeSmallStrain`、`ComputeIsotropicElasticityTensor`、`ComputeEigenstrain`、`ComputeLinearElasticStress` 和三个 `StressDivergenceTensors` 联立求解 ux、uy、uz。每个矩形的六个薄层边界严格落在网格面上，材料跳变不穿过积分单元。核心区使用可控最大尺寸，外部几何分级。默认 HEX27 二次位移插值与5阶 Gauss积分；3点/轴精确积分轴对齐仿射 HEX27 的弹性刚度与分片常数本征应变载荷。另以7阶积分与不同层内单元数实际比较。中心厚度为数值参数，与 height、width 完全不同。

所有六个远端边界 u=0，独立改变域半宽来测量截断效应；任何主算例均未使用 DDMPY 解析边界。`summarize.py` 中额外的解析薄层平均仅量化参考侧正则化误差，绝不参与 FE 输入。有限域误差、空间离散误差和薄层误差不会被相互抵消后宣称单独达标。

选择理由：分裂网格/给定跳跃界面可消除厚度参数，但需可靠处理有限矩形边缘、三分量约束与旋转；罚法还增加量纲和条件数验证。当前 XFEM 实现有下述已确认缺陷。对第一阶段，已有弹性材料的严格对齐三维薄层能直接运行且误差项可分别控制，故采用它。该方法不是对所有尺度和观测距离的最优选择。

## 兼容接口

[兼容范围表](COMPATIBILITY.md) 列出承诺的构造方式和方法。可直接用新的类：

```python
from fiberis.moose import numerical_ddm as ddm
import numpy as np

config = ddm.NumericalConfig(
    executable='/absolute/path/to/moose/modules/combined/combined-opt',
    output_directory='/absolute/path/to/case',
    youngs_modulus=30e9, poissons_ratio=.3,
    half_extent=800, core_extent=120, mesh_size=10,
    band_thickness=2, band_cells=2, processors=8,
    mpiexec='/absolute/path/to/moose-conda/bin/mpiexec')
e = ddm.Element(100, 60, .001, S1=0, S2=0)
e.set_global_coors(0, 0, 0)
w = ddm.Well().set_well_path(np.linspace(-100,100,101), 17., 40.)
f = ddm.FractureZ().set_fracture_elements(e).set_monitor_wells(w)
f.configure(config).calculate()
w = f.monitor_wells[0]  # 该接口与 DDMPY 一样复制监测井
print(w.u1, w.u2, w.u3, w.strain)
```

对于已有 DDMPY 对象，只需 `numerical = ddm.from_ddmpy(original_fracture, config); numerical.calculate()`。这个适配器仅读取几何与跳跃历史，不导入或调用解析计算。也支持 DynamicFracture 的快照列表。新接口的类直接导出，不模拟原始包 `Element.Element` 的模块嵌套。

`FractureGlobal` 沿用源码的 strike/dip 旋转矩阵约定（不要自行假设某种地学方位轴定义），默认准确使用给定角度。`legacy_vertical=True` 或从旧对象导入时显式保留90°时的0.001 rad偏移。数值解在裂缝坐标系中进行，接收点逆变换、位移正变换；均匀各向同性下这是完整三维方程的坐标协变变换。`Model` 可将多组不同方向裂缝的独立实际 FE 解叠加；有限域随各组旋转，截断误差分别存在。单个 Fracture 支持多个不同中心的平行矩形，在同一 FEM 解中叠加本征应变。

输出坐标约定有一处明确差异：本接口的监测井及u1/u2/u3始终保留用户输入的全局坐标；原版FractureGlobal.monitor_wells保存局部井，global_monitor_wells仅接收应变。需要原版局部分量时使用 `U_local = U_global @ R.T`，其中 `R, origin = numerical.frame()`。笔记本验证已把原版局部位移旋转到全局后再比较，未将不同坐标系的分量直接相减。

`DynamicFracture` 对每一时刻真正独立运行静态 FE，可改变几何和加载。`gather_strain_data()` 返回现有 fibeRIS `Data2D`（channel × time）；默认绝对应变，可显式 `reference_initial=True` 扣首帧。`gather_strain_rate_data()` 使用非均匀三点二次插值导数，端点用二阶单边（仅2帧时一阶）。未做动力学或时间积分；变化几何产生的尖锐事件只能由给定时间采样分辨。

## 观测与误差

所有结果先取井上 FE 插值位移，再计算中心弦方向与位移差在该方向的投影，除以中心间测量深度；方向向量一次归一化。曲井的切向量不放到微分号内。端点复制相邻内部应变。`gauge_length` 以沿弧长的分片线性应变积分求平均，端点窗口裁剪；不同于原始 DDMPY 的按点数平均或有限变形弦长法。只支持小应变 projection，其他方法明确报错。

位移相对L2为 `||U_FE−U_ref||F / ||U_ref||F`，同时保留三个分量分别的指标。应变同理。最大绝对误差除以参考全信号峰值，零参考时相对指标标记 undefined 并报告绝对值。主要目标是预先固定基准井上的位移和应变相对L2均小于1%；不要求每个近零分量的逐点相对误差小于1%，也不自动推广到其他几何/接收距离。

空间观测误差使用更密的独立 DDMPY 采样导数诊断；端点重复误差单列。时间误差使用非均匀二次/三次信号，以及真实 FE 二次开度快照验证。全部接收点的位移 CSV、原始参考和对比 NPZ 都保留，图可重新生成。默认不写整网格Exodus以控制三维文件大小；需要查看整个有限元场时可在独立输入的Outputs块加入 `exodus=true` 再运行。

## 参考源码审计

原始目录未作任何修改；冻结副本与SHA256在 `output/numerical_ddm/reference_audit`。未知上游版本号，内容哈希是本次参考版本的确切标识。

实际最小复现确认：

* Well逐分量归一化：均匀1e-4应变，在方向(2,3,4)的井得到1.2027945e-4；共同修正算子得到1e-4。
* asymmetric grow：请求width=.001、up=30，原始版本实际给width=30；数值兼容层使用明确输入width。
* t=[0,1,3,6]、f=t²：原始导数=[1,3,7,9]，正确点导数=[0,2,6,12]。
* dip=90°：原始矩阵使用90°−0.001 rad。默认数值接口不微调，旧版兼容模式另列。

这些修正没有改动 DDMPY 的 Element 解析场。初始直井只沿x方向，原始Well与共同修正算子在该例等价；倾斜/曲井不等价，必须报告修正后的共同算子指标，不能把原始错误当作物理目标。冻结 DynamicFracture 审计只跳过其 notebook 数据容器导入，所有数值 AST 原样运行；细节记在最小复现 JSON。

上述四项针对旧的 `DDMpy_log`。另行实际审计的新 `DDMpy_0930`（提交 `a65d4e3ac97d722aa546046fcac888a45887ea05`）已经修复方向向量归一化和width赋值；90°微调仍保留，且不含旧的StrainRateUtil模块。结果与冻结副本在 `output/numerical_ddm/reference_0930/`。其示例使用JIN_pylib的后向区间差分，这在非均匀时间下给出各区间斜率，不能当作各采样时刻的二阶点导数；本次同时输出原始兼容结果与明确定义的点导数。

当前 `XFEMSingleVariableConstraint.C` 中 `_jump.value(_t,_u[_qp])` 和邻侧调用把位移当作空间点，且 `jump_flux` 同样如此；没有为未使用的路线顺带修改这些对象。`GeometricCut3DUserObject.C` 明确写有 time evolving cuts not yet supported，RectangleCutUserObject 从构造时12个坐标建立固定切面。不能据参数名称推断其支持空间/时间变化。

旧的 `ModelBuilder.add_prescribed_dd_band()` 属于二维实现：x长度、y薄层、yy张开/xy剪切，无有限x2高度约束，只支持单次线性时间ramp。旧 `verify_fiberis_dd.py` 对比高度20000m的DDMPY近似长裂缝，检查某个位移跳跃，不能支持有限三维矩形误差<1%的广泛声明。此次复用其 `MooseBlock` 与 `MooseRunner` 基础；现有通用Exodus reader标注不可靠且采用二维最近点，所以使用MOOSE原生PointValueSampler三维有限元插值+按id核对坐标读取CSV。数据容器复用 fibeRIS Data2D。

## 限制与研究使用

所有精度/耗时结论仅指实际报告中的算例。尖端、裂缝面、薄层内部及远端边界附近不属于已验证观测范围。常值跳跃在矩形边缘产生奇异场；这里不报告尖端应力收敛，也不由给定开度反推压力或应力强度因子。网格张量积与多个矩形边界的组合可能显著增加成本，薄层高纵横比也影响条件数。强各向异性、非均匀材料、非线性/接触/裂缝相互作用定律、连续增长切面、完整DDMPY绘图接口、平滑真实椭圆均不在该实现中。离散椭圆始终匹配相同条带与开度分布。

固定几何且所有分量共用幅度历史时，另提供 `FractureZ.sample_history(points,times,factors)`：MOOSE 时间执行器在每个指定时间求解静态弹性平衡，不加入任何时间导数核。初始幅度必须为零。在线弹性、固定网格下切线矩阵恒定。初次测试 PETSc 的[跨次求解保留选项](https://petsc.org/release/manualpages/SNES/SNESSetLagJacobianPersists/)时，MOOSE日志仍显示每步组装，因此新增 `ConstantJacobianProblem`，调用框架已有的 `setConstJacobian(true)` 并显式启用预条件器复用。此扩展仅缓存刚度矩阵，每步残量和解向量仍由 MOOSE 求得；只允许系数、几何、边界类型均不改变的线性问题。默认静态路径不依赖它。

## 自包含旧笔记本的完整复现（0930 原包）

用户确认新上传的拟合笔记本缺少 `uniform_fracture_fit.py` 等依赖后，改选原包附带的 `Examples/Horizontal well existing tensile and shear.ipynb`。几何为300×300 m矩形，strike=0°、dip=90°、中心(150,130,-30) m；直井从(0,0,-30)到(300,0,-30)，300个点、原版31点平滑、10 m标距。S1从0到1 mm，width=S1/10，全部30帧保持原值。

```bash
# 用 base Python 运行原版 JIN_pylib 依赖；MOOSE仍由moose环境编译的可执行文件求解。
env MPLCONFIGDIR=/tmp/mplconfig ~/miniforge/bin/python scripts/numerical_ddm/reproduce_notebook.py --refine --recycle
```

原始代码单元的数值语句逐条原样执行，只有Jupyter magic、安装命令和绘图部分由运行器接管；DDMpy_0930源码未改。原版原始数组保存在 `output/numerical_ddm/notebook_existing_tensile_shear/original_0930.npz`。`--reference-only` 可单独重跑完整解析参考；`--pilot` 只运行3帧MOOSE预检，不等同完整复现。默认MOOSE运行全部30帧，其中初始零载荷零位移严格满足平衡，其余29帧均实际求解。结果与图在该目录的 `full_*` 子目录；`--analyze-only` 从保存的有限元位移重画图。

必须保留上述 `--refine` 才会使用验证过的穿面井局部加密窗口。未加密预检的应变误差为32.52%，已保留供比较；加密预检为0.2604%，位移为0.0411%。完整30帧结果另列在RESULTS.md。此局部网格精度不能推广为尖端应力精度。

`check_constant_jacobian.py` 实际比较了重新组装、仅缓存刚度、缓存并复用Krylov初始猜测三种MOOSE路径，采用非等间隔时间和幅度 `[0,1,-0.7,0.2]`；最近一次运行的位移历史相对差分别为6.73e-14和1.09e-9。三种方法均通过三个非零载荷的平衡检查，结果与原始日志保存在 `constant_jacobian_check/`。

`--recycle` 调用PETSc的[Fischer初始猜测生成器](https://petsc.org/release/manualpages/KSP/KSPSetUseFischerGuess/)，子空间完全来自此前有限元线性求解。原始右端项、KSP容差和MOOSE非线性残量检查保持不变；某些后续载荷的初始猜测已经满足线性容差，Krylov迭代数可以为0，每个时刻仍执行平衡验收。该选项单独写入配置，默认通用接口关闭。未启用此选项的旧完整时序尝试在12个非零帧后被主动停止，原输入、已完成CSV和停止原因均保留在 `full_h20_t0.2_L2400_refined/`，不算作完整复现；完整加速结果使用独立的 `_recycled` 目录。

时间单位按原笔记本图轴标签“分钟”解释，将原始0..60数值乘60送入MOOSE；原版差分输出保留为每分钟，另除60得到每秒。原版首帧应变率为0，这一兼容结果和非均匀点导数结果分别输出。此井穿过裂缝内部，默认数值薄层0.2 m，小于穿面附近1 m采样间距；其误差必须以该算例实测为准。

`reproduce_0930_shear.py` 是新 NonTouching 笔记本矩形 F2 分量的辅助复现尝试，恢复了其源函数，但缺少原辅助模块的精确几何来源；其中明确标注恢复信息与限制，**不作为完整新笔记本复现的验收结果**。

版本注记：当前源码MOOSE提交a13a1f8073e，日志显示源版本日期2026-04-14。环境的moose-dev元包为2026.02.05，预构建脚本建议2026.03.19并给出警告；本次在原环境中完整编译、链接和实际求解成功，没有擅自升级环境。具体依赖版本和二进制哈希记录在environment.json。
