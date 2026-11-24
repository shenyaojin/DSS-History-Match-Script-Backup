"""Self-contained HTML report for moose_ddm_workflow.py (notebooks 07/08 with MOOSE tensile)."""
import base64
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIG = REPO / "figs" / "tensile_fault_qc" / "moose_tensile_ddm_workflow"
OUT = FIG / "moose_tensile_ddm_workflow_report.html"
S = json.loads((REPO / "output" / "v11_pinv" / "ddm_workflow" / "summary.json").read_text())
GEOM = REPO / "figs" / "tensile_fault_qc" / "v11_pressure_inversion" / "01_v11_geometry.png"
MPA = 0.00689476
B, I = S["baseline"], S["inverted"]


def img(path, caption):
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOOSE Tensile in DDM Workflow</title>
<style>
:root{{--bg:#fff;--fg:#1d2330;--mute:#5b6477;--line:#d9dde6;--acc:#1f5fbf;--card:#f6f8fb;--warn:#b36b00}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12151c;--fg:#e6e9f0;--mute:#9aa3b5;--line:#2c3342;--acc:#7fb0ff;--card:#1a1f29;--warn:#e0a040}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:32px 16px 64px}}
h1{{font-size:1.9rem;margin:0 0 4px}} h2{{margin-top:2.4rem;border-bottom:1px solid var(--line);padding-bottom:6px}}
.sub{{color:var(--mute)}} figure{{margin:20px 0}} img{{max-width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:6px}}
figcaption{{color:var(--mute);font-size:.9rem;margin-top:6px}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:.95rem;margin:12px 0}}
th,td{{border-bottom:1px solid var(--line);padding:6px 10px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
.note{{background:var(--card);border-left:3px solid var(--acc);padding:10px 14px;border-radius:4px}}
.warn{{background:var(--card);border-left:3px solid var(--warn);padding:10px 14px;border-radius:4px}}
code{{background:var(--card);padding:1px 5px;border-radius:3px;font-size:.9em}}
</style></head><body><main>
<h1>Crossing-Cutout Strain Fit with a MOOSE Tensile Source</h1>
<p class="sub">Notebooks 07/08 (crossing_cutout, 2-hour strain fit), with the DDM tensile source replaced by the MOOSE V11 poroelastic model. Shear stays DDM. T1 = 2025-02-24 12:00, T2 = 2025-02-28 00:00, T3 = 2025-03-06 00:00 (fixed UTC−7).</p>

<h2>1. Result</h2>
<ul>
<li><b>Baseline</b> (07 time law): the MOOSE boundary pressure rises <b>{S['dP_base_T2_psi']:.0f} psi ({S['dP_base_T2_psi']*MPA:.2f} MPa) from T1 to T2</b> and {S['dP_base_T3_psi']:.0f} psi ({S['dP_base_T3_psi']*MPA:.2f} MPa) by T3.</li>
<li><b>Inverted</b> (08 per-window scaling): the pressure rises <b>{I['dP_last_window_before_T2_psi']:.0f} psi ({I['dP_last_window_before_T2_psi']*MPA:.2f} MPa) by the last window before T2</b> and {I['dP_last_window_psi']:.0f} psi ({I['dP_last_window_psi']*MPA:.2f} MPa) by the last window.</li>
<li><b>The pressure is now smooth</b>, with no spike at T1. It rises steadily and flattens after about 3/4 (dipping by about 25 psi near 3/5). It follows the same pattern as the DDM tensile opening (Section 5).</li>
<li><b>Fit quality:</b> strain RMSE against the observed profiles is {B['rmse_strain']:.4f} (baseline) and {I['rmse_strain']:.4f} (inverted) millistrain. The all-DDM fits in 07/08 reach {B['ddm_rmse_strain']:.4f} and {I['ddm_rmse_strain']:.4f}.</li>
<li><b>Why MOOSE fits worse:</b> spatial shape. The V11 tensile profile is narrower than the DDM ellipse and turns slightly compressive beyond about ±80 ft from the fracture, while the observed strain stays positive there. Least squares balances these flanks against the peak, so the peak after 3/1 is under-predicted (about 0.086 vs 0.13 millistrain at the fracture channel).</li>
</ul>
<p class="note">P(T1) = 4056 psi is the V11 initial pressure and is assumed. Strain only constrains pressure <i>changes</i>. The pressure is the Dirichlet boundary pressure at the monitoring-well node of the V11 model.</p>

<h2>2. What was done</h2>
<p>Pengchao's workflow, with only the tensile source swapped:</p>
<div class="tw"><table>
<tr><th>Step</th><th>DDM notebooks 07/08</th><th>This study</th></tr>
<tr><td>Tensile source</td><td>DDM ellipse, opening (mm)</td><td>MOOSE V11 SRV model, boundary pressure (psi)</td></tr>
<tr><td>Shear source</td><td>DDM small rectangle with cutout</td><td>Same: the 07 baseline and 08 inverted shear profiles, used as supplied</td></tr>
<tr><td>Baseline time law</td><td>opening = A₀·x² (T1–T2) + A₁·y (T2–T3)</td><td>ΔP = c₀·x² + c₁·y (same law). c ≥ 0 fitted to observed − DDM shear(07): c₀ = {S['c_psi']['T1_T2_x2']:.0f} psi, c₁ = {S['c_psi']['T2_T3_linear']:.0f} psi</td></tr>
<tr><td>Inversion</td><td>each 2-h window: independent non-negative scale of the baseline profile</td><td>same: α<sub>j</sub> ≥ 0 scales the MOOSE baseline profile to fit observed − DDM shear(08). Pressure = α<sub>j</sub> × baseline pressure</td></tr>
<tr><td>Objective</td><td>2-h strain profiles, first window = reference</td><td>same, strain only (no rate)</td></tr>
</table></div>
<p><b>Computation.</b> V11 is linear in pressure, so all of this comes from one existing MOOSE ramp run plus superposition. No new MOOSE runs were made, and the script runs in about 20 s on 2 cores.</p>
<p><b>Check.</b> The 08-style scaling assumes the profile scales with the pressure at that time. MOOSE has diffusion memory, so the inverted pressure was also run exactly through the superposition. The exact and scaled profiles differ by {100*I['exact_vs_scaled_rel_diff']:.1f}% overall, and the exact forward RMSE is {I['rmse_strain_exact_forward']:.4f} millistrain. The scaling is therefore a good approximation (Section 6).</p>
{img(GEOM, "V11 geometry, unchanged: SRV 5234 ft along strike, tensile fracture crossing the monitoring well at MD 10365.1 ft, DDM shear plane at MD 10317.8 ft.")}

<h2>3. Baseline: pressure, shear and waterfalls</h2>
{img(FIG / "baseline_summary.png", "Notebook layout. Top: recorded injection pressure with stage numbers. Second: MOOSE tensile pressure (blue) and DDM shear slip (red). Third: observed strain waterfall. Bottom: simulated (MOOSE tensile + DDM shear) waterfall. Black lines are observed profiles; green dashed lines are simulated profiles.")}

<h2>4. Inverted: pressure, shear and waterfalls</h2>
{img(FIG / "inverted_summary.png", "Same layout for the inverted (per-window) fit. The simulated flanks above about 10290 ft and below about 10440 ft are slightly negative (light blue), where the observed strain is slightly positive. This is the V11 shape limit.")}

<h2>5. MOOSE pressure vs DDM tensile opening</h2>
{img(FIG / "pressure_vs_ddm_opening.png", "Top: baseline and inverted MOOSE pressure. Bottom: each normalized to its last window, against the DDM tensile opening from 07/08.")}
<p>Normalized, the MOOSE pressure tracks the DDM opening closely in both cases. The inverted pressure runs slightly ahead of the opening before T2 and around 3/1–3/3, because MOOSE needs pressure to lead strain while pressure diffuses into the SRV. The inverted pressure flattens after about 3/4, as the opening does.</p>

<h2>6. Fitted vs observed profiles</h2>
{img(FIG / "profiles_fitted_vs_observed.png", "Six windows. Black: observed. Green dashed: MOOSE tensile + DDM shear. Blue: MOOSE tensile. Blue dotted: DDM tensile from 07/08, for comparison. Red: DDM shear.")}
{img(FIG / "scaling_check.png", "Left: at the fracture channel, the per-window scaled profile (blue) and the exact MOOSE response to the inverted pressure (orange dashed) agree. Both fall below the target after 3/1 (spatial shape limit). Right: the per-window scale α. It is large only in the first windows, where the baseline pressure is nearly zero, so pressure there remains small.")}

<h2>7. Comparison with the all-DDM fits</h2>
<div class="tw"><table><tr><th>Case</th><th>MOOSE tensile + DDM shear, strain RMSE</th><th>All-DDM (07/08), strain RMSE</th><th>MOOSE tensile vs DDM tensile, RMSE</th></tr>
<tr><td>Baseline</td><td>{B['rmse_strain']:.4f}</td><td>{B['ddm_rmse_strain']:.4f}</td><td>{B['rmse_vs_ddm_tensile']:.4f}</td></tr>
<tr><td>Inverted</td><td>{I['rmse_strain']:.4f}</td><td>{I['ddm_rmse_strain']:.4f}</td><td>{I['rmse_vs_ddm_tensile']:.4f}</td></tr></table></div>
<p class="warn">The DDM and MOOSE tensile sources are only approximately equivalent. Both explain the core of the signal. The DDM opening ellipse produces broader, all-positive flanks. The V11 pressurized SRV produces a narrower peak with mild compression outside it. Closing this gap would need a change in the MOOSE geometry (for example a taller or wider pressurized zone), which was kept fixed here.</p>

<h2>8. Earlier attempt: why it had a spike</h2>
<p>The previous report solved for 114 free pressure knots with only a smoothness penalty. That freedom let the inversion add an early pressure pulse to imitate the positive flanks the V11 shape cannot produce. Following the DDM workflow (a fixed time law, then a non-negative scale per window) removes that freedom, and the pressure becomes smooth. The tensile, shear and observed profiles in <code>crossing_cutout/strain_fit_2h</code> are identical (to 1e-16) to the files used before. Only the method changed.</p>

<h2>9. Files</h2>
<ul>
<li><b>Pressure per 2-h window</b> (baseline, inverted, α, psi and MPa, DDM opening alongside): <code>output/v11_pinv/ddm_workflow/moose_pressure_2h.csv</code></li>
<li><b>Profiles</b> (observed, MOOSE tensile, DDM shear, residual): <code>output/v11_pinv/ddm_workflow/{{baseline,inverted}}_profiles_2h.csv</code></li>
<li><b>Metrics:</b> <code>output/v11_pinv/ddm_workflow/summary.json</code></li>
<li><b>Figures:</b> <code>figs/tensile_fault_qc/moose_tensile_ddm_workflow/</code></li>
<li><b>Scripts:</b> <code>scripts/tensile_fault/v11_pressure_inversion/moose_ddm_workflow.py</code> (analysis and figures) and <code>build_ddm_workflow_report.py</code> (this page)</li>
<li><b>MOOSE response:</b> <code>output/v11_pinv/base/</code> (V11 deck, unit ramp, LINEAR solve). The solver fix is described in the previous report.</li>
<li><b>Inputs:</b> <code>Pengchao/Fervo_Small_Shear_Test/output/crossing_cutout/strain_fit_2h/</code> (07/08 outputs)</li>
</ul>
</main></body></html>
"""
OUT.write_text(html, encoding="utf-8")
print(OUT, f"{OUT.stat().st_size/1e6:.1f} MB")
