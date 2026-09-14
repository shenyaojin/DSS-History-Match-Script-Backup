from pathlib import Path
import pandas as pd
import json
HERE=Path(__file__).resolve().parent
cases=['unrectified','half_wave_positive']
if not all((HERE/c/'COMPLETE').exists() for c in cases):
    print('Comparison pending: both optimizations must finish successfully.')
    raise SystemExit(0)
a=pd.read_csv(HERE/cases[0]/'permeability_results.csv')
b=pd.read_csv(HERE/cases[1]/'permeability_results.csv')
lines=['# RFSDSS 固定区间反演对比','',
       '14700–14750 ft；0–40000 s；5 个区间独立反演渗透率，位置及背景固定。两组初值和物理模型相同。','',
       '| 区间(ft) | 初始 α | 不整流 α | 半波整流 α | 不整流 k(m²) | 半波整流 k(m²) |',
       '|---|---:|---:|---:|---:|---:|']
for i,row in a.iterrows():
    r=b.iloc[i]
    lines.append(f'| {row.lo_ft:g}–{row.hi_ft:g} | {row.initial_alpha:.3f} | {row.final_alpha:.6f} | {r.final_alpha:.6f} | {row.permeability_m2:.5e} | {r.permeability_m2:.5e} |')
lines+=['','本轮采用旧实测模板默认的 80 ft 井间距，未经测井轨迹验证；直接用 με×1e-6，无额外校准倍率。0–40000 s 为主要压力变化之前的早期窗口，结果应结合敏感度与边界命中情况判断。',
        '','整流组对观测与模拟都取 max(ε,0)；两组目标函数的原始大小不可直接当作同一拟合指标比较。']
for c in cases:
    result=json.loads((HERE/c/'optimizer_result.json').read_text())
    lines.extend(['',f"{c}: {result['message']}; iterations={result['iterations']}; scaled objective={result['objective']:.6g}"])
tmp=HERE/'COMPARE_rectification.md.tmp'
tmp.write_text('\n'.join(lines)+'\n');tmp.replace(HERE/'COMPARE_rectification.md')
print('Comparison written.')
