# Permeability Inversion from DSS — 模型与反演材料

代表案例：`scripts/DSS_history_match/optimizer_input_file_test/perm5layer_100_v2strain` 的无噪声 `strain_yy`、逐层渗透率、平滑 L1 反演。

本目录对应你列出的第 2、3 项；第 1 项数学证明由你已有的材料提供。所有原始输入、脚本与记录均从已有案例复制；本次没有运行正演或反演。

| 材料 | 入口 | 用途 |
| --- | --- | --- |
| 2. 代表性 MOOSE forward input | [representative_forward.i](02_forward_model/representative_forward.i) | 完整五层真值模型，含压力加载与观测位置 |
| 2. Governing equations | [governing_equations.md](02_forward_model/governing_equations.md) | 参数、方程、边界条件及实现与理想化理论的差异 |
| 3. 反演方法说明 | [inversion_method.md](03_inversion/inversion_method.md) | Objective、regularization、gradient 和 optimizer 的对应关系 |
| 3. 关键代码 | [key_code_excerpts.md](03_inversion/key_code_excerpts.md) | 原样摘录，并标注原脚本行号 |
| 3. 完整原脚本 | [106_optimization_runner_L1.py](03_inversion/original/106_optimization_runner_L1.py) | 原始外层 L-BFGS-B 实现 |
| 3. MOOSE 正演与伴随 | [forward_and_adjoint.i](03_inversion/original/forward_and_adjoint.i) | 参数化、应变目标、偶极伴随源、梯度 reporter |
| 3. MOOSE master | [optimize.i](03_inversion/original/optimize.i) | 数据传递及单次目标／梯度计算 |
| 已有反演记录 | [existing_records](03_inversion/existing_records) | 初值、终值、目标值与接受步历史 |
| 来源校验 | [source_manifest.json](source_manifest.json) | 每个原样复制文件的原始路径、大小及 SHA-256 |

建议先读两份方法说明，再按代码摘录回查完整源文件。该目录是论文写作与代码审阅材料包：保留了原脚本的路径和环境依赖，没有改造成独立运行工程，也没有复制大型正演输出。原始测量文件位于案例的 `inv/measurement_data.csv`。

原脚本在顶层清理历史文件，不能用 `import` 方式安全地阅读它。`forward_and_adjoint.i` 的参数占位值需要由 master 传入，不能将它直接当作真值正演输入；真值正演请查看第 2 项的文件。

论文写作时需要区分：自由参数为 100 层（完整参数向量为 200 层）；初值在真值异常区已经设置扰动；目标函数为离散残差平方和；正则化是相对背景的平滑 L1；实际模型的储存项和机械边界以文件声明为准。详见两份说明。
