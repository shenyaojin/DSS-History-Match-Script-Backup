from pathlib import Path
import shutil
import numpy as np
r = Path(__file__).resolve().parent
v = r.parent.parent
assert "DONE_EXIT=0" in (r/"zonal_L1.stdout").read_text().splitlines()
assert (r/"qc_strain_zonal_final.png").is_file()
old = np.loadtxt(r.parent/"bg3_w3_s0/optimized_theta_zonal.txt").ravel()
new = np.loadtxt(r/"optimized_theta_zonal.txt").ravel()
truth = np.array([np.log10(3e-15), -15.0])
lines = ["# bg3_w3_s0：v1 vs v2", "", "偏差定义：估计值 − 真值；θ 为 log10(渗透率)。", "", "| 参数 | 真值 | v1 | v1 偏差 | v2 | v2 偏差 | v2 − v1 |", "|---|---:|---:|---:|---:|---:|---:|"]
for i,name in enumerate(["θ_frac", "θ_srv"]):
    lines.append(f"| {name} | {truth[i]:.6f} | {old[i]:.6f} | {old[i]-truth[i]:+.6f} | {new[i]:.6f} | {new[i]-truth[i]:+.6f} | {new[i]-old[i]:+.6f} |")
lines += ["", "A 场逐道滞后-1 Pearson 自相关的道间均值：v1 ≈ 0.97（原基线值），v2 = -0.007380（包含归零的首帧）。", "", f"换成时间独立 DSS 噪声后，θ_frac 改变 {new[0]-old[0]:+.6f}，θ_srv 改变 {new[1]-old[1]:+.6f}；绝对真值偏差分别从 {abs(old[0]-truth[0]):.6f}、{abs(old[1]-truth[1]):.6f} 变为 {abs(new[0]-truth[0]):.6f}、{abs(new[1]-truth[1]):.6f}。", "", "A 的抽样算法改变了随机数消耗顺序；B 的公式、幅值及 BASE_SEED 保持不变，但 B 的具体随机实现随之变化，本次是按指定生成流程的新旧比较。"]
(r/"COMPARE_v1_vs_v2.md").write_text("\n".join(lines)+"\n")
dest=v/"fig_deliverable/redo_bg3_w3_s0"
for source in [v/"noise_data/dss_noise_extraction.png", r/"qc_strain_zonal_final.png", r/"COMPARE_v1_vs_v2.md"]:
    shutil.copy2(source, dest/source.name)
print("DELIVERABLES_COMPLETE", flush=True)
