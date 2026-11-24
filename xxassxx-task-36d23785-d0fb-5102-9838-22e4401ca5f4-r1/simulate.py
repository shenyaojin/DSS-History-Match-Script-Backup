#!/usr/bin/env python3.12
"""Standard-library FTCS heat diffusion simulation and SVG export."""
import csv
import math
import platform
from pathlib import Path

L, ALPHA, N, DT, END = 1.0, 1e-6, 101, 20, 10000
TIMES = (0, 2500, 5000, 7500, 10000)
DX = L / (N - 1)
R = ALPHA * DT / DX**2

def exact(x, t):
    return 100 * math.exp(-ALPHA * math.pi**2 * t / L**2) * math.sin(math.pi * x / L)

def main():
    assert platform.python_version_tuple()[:2] == ('3', '12')
    assert 0 < R <= 0.5
    xs = [i * DX for i in range(N)]
    u = [exact(x, 0) for x in xs]
    u[0] = u[-1] = 0.0
    saved = {0: u[:]}
    all_error = 0.0
    for step in range(1, END // DT + 1):
        u = [0.0] + [u[i] + R*(u[i-1]-2*u[i]+u[i+1]) for i in range(1, N-1)] + [0.0]
        t = step * DT
        all_error = max(all_error, max(abs(v-exact(x,t)) for x,v in zip(xs,u)))
        assert all(math.isfinite(v) and 0 <= v <= 100 for v in u)
        if t in TIMES:
            saved[t] = u[:]
    with open('temperature.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s', 'x_m', 'temperature_C'])
        for t in TIMES:
            writer.writerows((t, format(x,'.12g'), format(v,'.15g')) for x,v in zip(xs,saved[t]))
    colors = ['#202020','#0072b2','#d55e00','#009e73','#cc79a7']
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="560" viewBox="0 0 900 560">', '<rect width="900" height="560" fill="white"/>', '<g font-family="sans-serif" font-size="14">', '<text x="450" y="28" text-anchor="middle">1D heat diffusion: explicit finite differences</text>']
    for val in range(0,101,20):
        y = 470-4*val
        svg += [f'<path d="M80 {y} H680" stroke="#ddd"/>', f'<text x="68" y="{y+5}" text-anchor="end">{val}</text>']
    for i in range(6):
        x = 80+120*i
        svg += [f'<text x="{x}" y="495" text-anchor="middle">{i/5:.1f}</text>']
    svg += ['<path d="M80 70 V470 H680" fill="none" stroke="black"/>', '<text x="380" y="535" text-anchor="middle">Position x (m)</text>', '<text transform="translate(22,270) rotate(-90)" text-anchor="middle">Temperature (°C)</text>']
    for j,t in enumerate(TIMES):
        points = ' '.join(f'{80+600*x:.4f},{470-4*v:.4f}' for x,v in zip(xs,saved[t]))
        svg += [f'<polyline points="{points}" fill="none" stroke="{colors[j]}" stroke-width="2"/>', f'<path d="M710 {95+32*j} h28" stroke="{colors[j]}" stroke-width="3"/>', f'<text x="748" y="{100+32*j}">t = {t} s</text>']
    svg += ['</g></svg>']
    Path('temperature.svg').write_text('\n'.join(svg)+'\n', encoding='utf-8')
    final_error = max(abs(v-exact(x,END)) for x,v in zip(xs,u))
    print(f'Python {platform.python_version()}; command: python3.12 simulate.py')
    print(f'N={N}; dx={DX} m; dt={DT} s; alpha={ALPHA} m^2/s; steps={END//DT}')
    print(f'Stability ratio r={R:.12g} <= 0.5: PASS')
    print(f'Final center numerical={u[50]:.12f} C; analytical={exact(0.5,END):.12f} C')
    print(f'Final max absolute error={final_error:.12g} C; all-step max={all_error:.12g} C')
    print('Saved times:', TIMES)
    print('Finite values, bounds and fixed zero boundaries: PASS')
    report = f'''# 一维热扩散实际运行报告

这是用于验证模拟与结果交付流程的合成算例，不代表真实材料实验。

## 参数与方法
L=1 m，alpha=1e-6 m²/s；101 个均匀节点，dx=0.01 m。
初始温度为 100*sin(pi*x/L) °C，两端始终固定 0 °C。
显式中心空间差分、前向时间差分（FTCS），dt=20 s，共 500 步至 10000 s。
保存时刻为 0、2500、5000、7500、10000 s，均对应整数时间步。
新建 simulate.py 实现上述参数，未修改原项目文件；无外部数据与依赖。

## 实际运行
命令：`python3.12 simulate.py > run.log 2>&1`。
解释器：Python {platform.python_version()}，仅标准库，无联网或依赖安装。
退出状态由执行工具确认并在运行后补记。真实标准输出见 run.log。

## 稳定性与误差
稳定性比值 r=alpha*dt/dx²={R:.12g}，满足显式格式 r≤0.5 的稳定条件。
解析解为 T=100*exp(-alpha*pi²*t/L²)*sin(pi*x/L)。
最终中心温度：{u[50]:.12f} °C；解析值：{exact(0.5,END):.12f} °C。
最终全部节点的最大绝对误差：{final_error:.12g} °C。
全部 500 个时间步、全部节点的最大绝对误差：{all_error:.12g} °C。
运行中检查有限数值、0–100 °C 范围；边界每步显式置零。
误差来自空间与时间离散；仅验证给定网格和时间步，未做收敛性研究。

## 文件清单
- temperature.csv：指定五个时刻的 505 条数据，列 time_s,x_m,temperature_C。
- temperature.svg：五条温度曲线，含坐标轴、单位和图例。
- simulate.py：实际运行的完整标准库脚本。
- report.md：参数、命令与校核报告。
- run.log：真实运行日志。

计算与文件生成不等于附件送达；需附件传输及接收确认后才可宣告交付完成。
'''
    Path('report.md').write_text(report, encoding='utf-8')

if __name__ == '__main__':
    main()
