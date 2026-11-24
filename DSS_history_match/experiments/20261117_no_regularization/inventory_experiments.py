"""Read-only audit of original DSS experiments; build a navigable local index."""
from pathlib import Path
import csv
import json
import re
import hashlib
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT/'scripts/DSS_history_match'
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    groups = {
        '01_forward_and_gradient_verification': ['validation','paper_theoretical_framework'],
        '02_three_layer_prototypes': ['optimizer_input_file_test/perm_3layer','optimizer_input_file_test/perm3layer_fiberis','optimizer_input_file_test/perm3layer_v2','optimizer_input_file_test/perm3layerv3'],
        '03_legacy_layerwise_recovery': ['optimizer_input_file_test/perm5layer_100layer','optimizer_input_file_test/perm5layer_100_v2strain'],
        '04_legacy_layerwise_noise': ['optimizer_input_file_test/perm5layer_100_v3strain_noise','optimizer_input_file_test/perm5layer_100_v4strain_rfsdss'],
        '05_legacy_two_zone_noise_and_geometry': ['optimizer_input_file_test/perm5layer_100_v5strain_zonal2p','optimizer_input_file_test/perm5layer_100_v6strain_zonalgrid'],
        '06_field_inversions': [p.name for p in PROJECT.glob('real_rfsdss_*') if p.is_dir()],
        '07_field_screening': ['swell_pressure_channel_scan'],
        '08_exploratory_scripts_and_visualization': ['viz','workshop2025',*[p.name for p in PROJECT.glob('*.py')]],
    }
    index=PROJECT/'experiment_index'
    for category, relatives in groups.items():
        folder=index/category;folder.mkdir(parents=True,exist_ok=True)
        for relative in relatives:
            src=PROJECT/relative;link=folder/src.name
            if src.exists() and not link.exists():link.symlink_to(src, target_is_directory=True)
    rows=[]
    for family in ['perm5layer_100_v2strain','perm5layer_100_v3strain_noise','perm5layer_100_v4strain_rfsdss','perm5layer_100_v5strain_zonal2p','perm5layer_100_v6strain_zonalgrid']:
        base=PROJECT/'optimizer_input_file_test'/family/'inv'
        finals=sorted(base.rglob('optimized_theta_zonal.txt')) if ('zonal' in family) else sorted(base.rglob('optimized_alphas_L1.txt'))
        for final in finals:
            run=final.parent
            histories=list(run.glob('objective_history*.csv'))
            hist=next((p for p in histories if 'zonal' in p.name),histories[0] if histories else None)
            data=reg=None
            if hist:
                with hist.open() as f: records=list(csv.DictReader(f))
                if records:
                    record=records[-1]
                    for key in ['obj_raw','data_objective','objective_raw','data_misfit']:
                        if key in record:data=float(record[key]);break
                    for key in ['reg_l1','regularization','reg']:
                        if key in record:reg=float(record[key]);break
            n=2 if 'zonal' in family else 100
            rows.append(dict(family=family,case=run.name,category='historical_two_zone' if n==2 else 'historical_layerwise',
                n_optimized=n,regularization='smoothed_L1' if reg is not None and reg>0 else 'requires_history_check',
                beta=2e-11 if n==2 else '',last_data_objective=data,last_regularization=reg,
                penalty_over_data=(reg/data if data and reg is not None else ''),
                result=str(final.relative_to(ROOT)),evidence=str(hist.relative_to(ROOT)) if hist else '',
                paper_role='historical_reference_only; replace with three_Kxx experiments',
                no_reg_action='reproduce_existing_low_dimensional_control' if n==2 else 'do_not_rerun_layerwise; user excludes this parameterization'))
    for cfg in sorted(PROJECT.glob('real_rfsdss_*/**/config.json')):
        if any(s in cfg.parts for s in ['evaluations','permeability_evaluations','solver_output']):continue
        try:d=json.loads(cfg.read_text())
        except (ValueError,OSError):continue
        if 'beta_l1' not in d:continue
        result=cfg.parent/'optimizer_result.json'
        if not result.exists():continue
        record=json.loads(result.read_text())
        rows.append(dict(family=cfg.relative_to(PROJECT).parts[0],case=str(cfg.parent.relative_to(PROJECT)),category='field',
            n_optimized=len(d.get('optimized_parameters',d.get('initial_theta',[]))) or d.get('n_parameters',''),
            regularization='smoothed_L1' if d['beta_l1'] else 'none',beta=d['beta_l1'],
            last_data_objective='',last_regularization='',penalty_over_data='',result=str(result.relative_to(ROOT)),evidence=str(cfg.relative_to(ROOT)),
            paper_role='final Phase7 case' if cfg.parent.name=='fixed_geometry_inversion' else 'earlier field version',
            no_reg_action='same_data_same_geometry_beta_zero_control'))
    fields=list(rows[0])
    with (OUT/'original_experiment_audit.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    source_files=[p for p in [ROOT/'second_comp_figures/README.md',ROOT/'second_comp_figures/figure_records.json'] if p.exists()]
    audit=dict(created_utc=datetime.now(timezone.utc).isoformat(),originals_modified=False,classification='symlink index; no source files moved',
        original_run_count=len(rows),group_counts={k:sum(r['category']==k for r in rows) for k in {r['category'] for r in rows}},
        regularization_caution='Low parameter count does not prove regularization is negligible. Compare archived penalty and data terms separately; clean data can be penalty-dominated.',
        clarified_parameterization='Exactly three independent Kxx values: lower host, central hydraulic-fracture band, upper host. Kyy and Kzz fixed. No C in new synthetic inversion.',
        figure_sources={str(p.relative_to(ROOT)):digest(p) for p in source_files},groups=groups)
    (OUT/'inventory.json').write_text(json.dumps(audit,indent=2)+'\n')
    text='''# DSS History Match 实验分类

本目录是原实验的导航入口；原文件、观测、结果和图片均未移动或修改。

| 分类 | 内容 | 新论文中的定位 |
|---|---|---|
|01|正演历史比较、梯度验证、理论与方法|方法验证|
|02|早期三层原型|只作实现参考；其中部分拟合位移而不是应变，不能直接当作新的 DSS 恢复实验|
|03–04|约100个自由参数的逐层恢复与噪声测试|历史参考，不进入用户本次指定的论文实验主线|
|05|两区域渗透率、背景固定；噪声、1/2m错位及27格网格|历史参考；不是三层各一个Kxx|
|06|真实RFS-DSS现场反演及历史版本|保留；最终Phase7结果另补beta=0对照|
|07|现场通道筛选与压力相关性扫描|数据筛选，不是最终反演|
|08|早期参数扫描、模型构建、拟合与可视化脚本|工具与探索记录，不能仅凭脚本存在认定实验完成|

本轮新增实验、输入、实际运行状态与结果见 `../experiments/20261117_no_regularization/`。
逐层实验仅保留引用；新synthetic和几何敏感性实验严格只有三层三个Kxx自由参数。

并非所有历史代码都带正则：早期三层原型中有纯数据项；v2 也存在无正则逐层试算的36条评估记录，但缺少对应最终完成产物，不能算作本次三层无正则实验。主要proposal图使用的L1结果则确实包含正则。
'''
    (index/'README.md').write_text(text)
    (OUT/'README.md').write_text('''# 2026-11-17：无正则化对照与三层 Kxx 实验

用户确认的项目为 `scripts/DSS_history_match`；三个变量为三层各一个 Kxx。
原实验通过 `../../experiment_index/README.md` 分类导航；本轮不修改论文图片样式。

- `original_experiment_audit.csv`：已保存结果对应的正则化证据、目标值、自由度与处理建议。
- `inventory.json`：分类、论文图来源、范围和溯源。
- `suite.py`：新三层实验与旧低维实验的 beta=0 对照；真实 MOOSE 正演，非拟合代理。
- `run_field_no_reg.py`：最终现场几何与观测固定，仅关闭正则化的独立反演。
- `runs/`：独立运行目录；是否完成以各自 status/result/COMPLETE 为准，目录存在不表示已运行。
- `controller_status.json`：实际批处理状态；`run_summary.csv`：逐例完成状态与结果。`python summarize.py` 可随时只读刷新汇总。

本轮执行顺序：三层干净恢复与1/2m几何错位（beta=0）；最终Phase7现场beta=0；三层L1成对控制与四档噪声beta=0；proposal直接引用的六组旧低维beta=0控制。其余v6全因子网格配置只准备、未自动启动。早期现场版本作为历史版本保留，不与最终现场版本重复充当论文结果。

新三层模型沿用历史合成正演的100×100m区域、200×200网格、130原生时间、500观测点、压力历史和材料/边界/求解设置，仅改成下部基质—中央高渗带—上部基质三个材料区域。真值Kxx=[1e-18,3e-15,1e-18]m²，真值中央带y=[-3,3]m；Kyy=Kzz=1e-18m²，只有三个Kxx参与优化，没有C。干净synthetic为同网格一致性恢复；不是独立网格验证。噪声沿用旧观测中的固定噪声实现，按新/旧干净响应的通道峰值中位数比例缩放，不重新抽随机数。几何敏感性各例使用同一份真值观测，只改变反演假设位置。

当前可执行文件不包含旧的自定义各向异性伴随梯度对象；本轮低维优化使用经过h/h/2检查的有限差分。旧两区控制会先核对同初值数据误差与历史记录的差异不超过1%，不通过则停止并标注，不能冒充只改变正则项的严格同方法对照。

新三层的L1配对按恰好三个参数计算惩罚；它与旧200维展开后的L1具有不同参数化，不能直接比较惩罚大小。三层beta=0恢复是用户指定的论文主线。

必须区分：旧synthetic为两区、两参数，旧现场为Kxx/Kyy及解析幅值因子；都不同于本轮三层Kxx。
不能仅凭参数少就认定正则化无影响。例如旧 exact_clean 最后数据项约1.52e-12、正则约1.29e-9，必须实际作beta=0对照。
''')
    print(json.dumps(audit['group_counts']));print('INDEX',index)


if __name__=='__main__':main()
