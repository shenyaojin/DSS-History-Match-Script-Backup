"""Build a self-contained HTML report (figures embedded) for the crossing_cutout MOOSE-tensile study (scripts 126/127)."""

import base64
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG = REPO_ROOT / "figs" / "tensile_fault_qc" / "crossing_cutout_tensile"
OUT = FIG / "crossing_cutout_tensile_report.html"


def img(path, caption):
    b64 = base64.b64encode((FIG / path).read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}">'
            f"<figcaption>{caption}</figcaption></figure>")


s = pd.read_csv(FIG / "combined_summary.csv").set_index("case")
rows = "".join(
    f"<tr><td>{c}</td><td>{r.moose_peak_strain:.4f}</td><td>{r.ddm_peak_strain:.4f}</td>"
    f"<td>{r.shear_peak_strain:.4f}</td><td>{100*r.rel_rmse_strain:.1f}%</td><td>{100*r.rel_rmse_rate:.1f}%</td></tr>"
    for c, r in s.iterrows()
)

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crossing-Cutout Tensile Report</title>
<style>
:root{{--bg:#fff;--fg:#1d2330;--mute:#5b6477;--line:#d9dde6;--acc:#1f5fbf;--card:#f6f8fb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12151c;--fg:#e6e9f0;--mute:#9aa3b5;--line:#2c3342;--acc:#7fb0ff;--card:#1a1f29}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:32px 16px 64px}}
h1{{font-size:1.9rem;margin:0 0 4px}} h2{{margin-top:2.4rem;border-bottom:1px solid var(--line);padding-bottom:6px}}
.sub{{color:var(--mute)}} figure{{margin:20px 0}} img{{max-width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:6px}}
figcaption{{color:var(--mute);font-size:.9rem;margin-top:6px}}
table{{border-collapse:collapse;width:100%;font-size:.95rem}} th,td{{border-bottom:1px solid var(--line);padding:6px 10px;text-align:right}}
th:first-child,td:first-child{{text-align:left}} .note{{background:var(--card);border-left:3px solid var(--acc);padding:10px 14px;border-radius:4px}}
code{{background:var(--card);padding:1px 5px;border-radius:3px}}
</style></head><body><main>
<h1>Crossing-Cutout Case: MOOSE Tensile Replacing DDM Tensile</h1>
<p class="sub">Fervo Gold 4-PB, 2-hour windows, 2025-02-24 13:00 to 2025-03-05 23:00 (UTC-7). Peak channel 10346.5 ft.</p>

<h2>1. Summary</h2>
<ul>
<li>The tensile strain profiles from notebooks 03 (baseline) and 04 (inverted) were converted to injection-pressure histories and run through the MOOSE perpendicular-monitor tensile-fault model.</li>
<li>Strain at the peak channel is reproduced to about 1.5% (relative RMSE) in both cases. This is partly by construction: the coefficient C was calibrated against this same MOOSE model.</li>
<li>Strain rate is reproduced to 9% (baseline) and 25% (inverted). MOOSE smooths the sharp rate spikes of the inverted profile and relaxes smoothly after 2/28, where the baseline DDM rate drops instantly.</li>
<li>Adding the DDM shear component changes the total by only a few percent relative to the all-DDM decomposition.</li>
</ul>

<h2>2. Method</h2>
<ol>
<li>Read <code>baseline_tensile_profiles_2h.csv</code> and <code>inverted_tensile_profiles_2h.csv</code> (114 windows x 44 channels each).</li>
<li>Take the channel with the largest tensile strain (10346.5 ft). Convert millistrain to strain and subtract the first window.</li>
<li>Pressure: <code>P(t) = P0 + C&middot;S(t)</code> with P0 = 4056 psi and C = 1.63e7 psi/strain (the history-match value from the DAS calibration, scripts 117-120). Baseline strain gives the baseline pressure; inverted strain gives the inverted pressure.</li>
<li>Run each pressure through the MOOSE model of script 111 (250 ft fracture, monitor line at 5/6 of the fracture length, matrix permeability 1e-18 m&sup2;). About 6 minutes per case on 20 processors.</li>
<li>Compare modeled monitor strain and strain rate with the DDM tensile profile at the peak channel, then add the DDM shear profiles (<code>baseline_shear_profiles_2h.csv</code>, <code>inverted_shear_profiles_2h.csv</code>).</li>
</ol>
<p class="note">Not included: the DDM shear solve itself (the supplied shear CSVs were used as given), a check of the corrected DDMpy version, and any comparison with the observed DAS data. The notebooks' input files and DDMpy were not available on this machine.</p>

<h2>3. Geometry setup</h2>
{img("model_geometry.png", "MOOSE model geometry: fracture (yellow) and perpendicular monitor fiber (dashed) crossing at 5/6 of the fracture length.")}

<h2>4. Baseline and inverted pressure curves</h2>
{img("baseline_vs_inverted_pressure.png", "Pressure derived from the tensile strain at 10347 ft. Both start at 4056 psi; by 3/6 the baseline reaches about 5720 psi and the inverted about 5560 psi.")}

<h2>5. MOOSE tensile vs DDM tensile</h2>
<h3>Baseline</h3>
{img("baseline_moose_vs_ddm_tensile.png", "Baseline: MOOSE monitor (blue) vs DDM tensile (black). Strain agrees closely; MOOSE rate relaxes smoothly after 2/28.")}
<h3>Inverted</h3>
{img("inverted_moose_vs_ddm_tensile.png", "Inverted: MOOSE monitor (red) vs DDM tensile (black).")}

<h2>6. Combined with DDM shear</h2>
{img("combined_moose_tensile_ddm_shear.png", "Left: baseline. Right: inverted. Black = DDM tensile + DDM shear; colored = MOOSE tensile + DDM shear; green dashed = DDM shear.")}
<table><tr><th>Case</th><th>MOOSE peak strain (milli)</th><th>DDM peak strain (milli)</th><th>Shear peak (milli)</th><th>Rel. RMSE strain</th><th>Rel. RMSE rate</th></tr>{rows}</table>

<h2>7. MOOSE outputs along the monitor line</h2>
<h3>Baseline</h3>
{img("baseline/pressure_map.png", "Baseline: pore pressure along the monitor line.")}
{img("baseline/monitor_normal_strain_no_rotation.png", "Baseline: monitor-normal strain.")}
{img("baseline/monitor_normal_strain_rate_no_rotation.png", "Baseline: monitor-normal strain rate.")}
<h3>Inverted</h3>
{img("inverted/pressure_map.png", "Inverted: pore pressure along the monitor line.")}
{img("inverted/monitor_normal_strain_no_rotation.png", "Inverted: monitor-normal strain.")}
{img("inverted/monitor_normal_strain_rate_no_rotation.png", "Inverted: monitor-normal strain rate.")}

<h2>8. Caveats and next steps</h2>
<ul>
<li>The comparison is at the single peak channel. The MOOSE monitor line spans only about 50 ft either side of the fracture, while the DDM channels span about 290 ft. A full-depth waterfall needs a rule mapping monitor position to measured depth.</li>
<li>C = 1.63e7 psi/strain is an assumption. The pressure-record fit gives 3.13e7, which would roughly double the pressure excursion (about +3000 psi instead of +1600 psi).</li>
<li>The per-case <code>perpendicular_monitor_summary.txt</code> files list the old synthetic pressure path under "pressure source". That line is stale; the pressure actually used is the "cropped pressure" file listed beneath it.</li>
<li>Scripts: <code>126_run_crossing_cutout_tensile_pressure.py</code>, <code>127_combine_moose_tensile_ddm_shear.py</code>, <code>128_build_html_report.py</code> in <code>scripts/tensile_fault/</code>.</li>
</ul>
</main></body></html>
"""
OUT.write_text(html, encoding="utf-8")
print(OUT, f"{OUT.stat().st_size/1e6:.1f} MB")
