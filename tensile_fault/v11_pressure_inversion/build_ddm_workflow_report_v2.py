"""v2 HTML report: MOOSE tensile + DDM shear with a narrowed SRV (all lengths in ft)."""
import base64
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
TAG = "v2_h060"
FIG = REPO / "figs" / "tensile_fault_qc" / f"moose_tensile_ddm_workflow_{TAG}"
CMP = REPO / "figs" / "tensile_fault_qc" / "moose_tensile_ddm_workflow_v2"
OUT = CMP / "moose_tensile_ddm_workflow_v2_report.html"
S = json.loads((REPO / "output" / "v11_pinv" / f"ddm_workflow_{TAG}" / "summary.json").read_text())
S1 = json.loads((REPO / "output" / "v11_pinv" / "ddm_workflow_v1" / "summary.json").read_text())
W = pd.read_csv(CMP / "srv_width_comparison.csv")
MPA = 0.00689476
B, I = S["baseline"], S["inverted"]
PIECES = [("1_tensile_only", "T1–T2: purely tensile (02/24 12:00 – 02/28 00:00)"),
          ("2_shear_first_half", "T2–mid: shear, first half (02/28 00:00 – 03/03 00:00)"),
          ("3_shear_second_half", "mid–T3: shear, second half (03/03 00:00 – 03/06 00:00)")]


def img(path, caption):
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


def srv_table(srv):
    r = "".join(f"<tr><td>{z['zone']}</td><td>±{z['half_height_ft']:.1f}</td><td>{2*z['half_height_ft']:.1f}</td>"
                f"<td>{z['md_top_ft']:.1f}</td><td>{z['md_bottom_ft']:.1f}</td><td>{z['length_ft']:.0f}</td>"
                f"<td>{z['perm_m2']:.0e}</td><td>{z['perm_mD']:.3g}</td><td>{z['porosity']:.2f}</td></tr>" for z in srv)
    return ('<div class="tw"><table><tr><th>Zone</th><th>Half-height (ft)</th><th>Total height (ft)</th><th>MD top (ft)</th>'
            '<th>MD bottom (ft)</th><th>Length along strike (ft)</th><th>Perm (m²)</th><th>Perm (mD)</th><th>Porosity</th></tr>'
            + r + '<tr><td>matrix</td><td colspan="5">everything else in the 7200 ft domain</td><td>1e-18</td><td>0.00101</td><td>0.01</td></tr></table></div>')


def wrow(r):
    name = {"v1": "×1.0 (v1)", "v2_h060": "<b>×0.6 (v2)</b>", "v2_h040": "×0.4"}[r.variant]
    return (f"<tr><td>{name}</td><td>{r.case}</td><td>{r.srv_md_top_ft:.1f}–{r.srv_md_bottom_ft:.1f}</td>"
            f"<td>{r.zero_top_sim:.1f}</td><td>{r.zero_bottom_sim:.1f}</td><td>{r.halfmax_top_sim:.1f}–{r.halfmax_bottom_sim:.1f}</td>"
            f"<td>{r.peak_ratio:.2f}</td><td>{r['rmse_T1-T2']:.4f}</td><td>{r['rmse_T2-mid']:.4f}</td><td>{r['rmse_mid-T3']:.4f}</td>"
            f"<td>{r.rmse:.4f}</td></tr>")


obs = W.iloc[0]
html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOOSE Tensile DDM v2</title>
<style>
:root{{--bg:#fff;--fg:#1d2330;--mute:#5b6477;--line:#d9dde6;--acc:#1f5fbf;--card:#f6f8fb;--warn:#b36b00}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12151c;--fg:#e6e9f0;--mute:#9aa3b5;--line:#2c3342;--acc:#7fb0ff;--card:#1a1f29;--warn:#e0a040}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:32px 16px 64px}}
h1{{font-size:1.9rem;margin:0 0 4px}} h2{{margin-top:2.4rem;border-bottom:1px solid var(--line);padding-bottom:6px}}
.sub{{color:var(--mute)}} figure{{margin:20px 0}} img{{max-width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:6px}}
figcaption{{color:var(--mute);font-size:.9rem;margin-top:6px}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:.92rem;margin:12px 0}}
th,td{{border-bottom:1px solid var(--line);padding:6px 9px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
.note{{background:var(--card);border-left:3px solid var(--acc);padding:10px 14px;border-radius:4px}}
.warn{{background:var(--card);border-left:3px solid var(--warn);padding:10px 14px;border-radius:4px}}
code{{background:var(--card);padding:1px 5px;border-radius:3px;font-size:.9em}}
</style></head><body><main>
<h1>MOOSE Tensile + DDM Shear, v2: Narrowed SRV</h1>
<p class="sub">Same workflow as v1 (notebooks 07/08, 2-hour strain fit, MOOSE V11 tensile pressure, DDM shear), with the SRV height scaled by 0.6 about the tensile fracture. All lengths in ft. T1 = 2025-02-24 12:00, T2 = 2025-02-28 00:00, T3 = 2025-03-06 00:00 (fixed UTC−7).</p>

<h2>1. Result</h2>
<ul>
<li><b>Shear plane vs SRV top.</b> Pengchao's <code>fixed_plane_geometry.csv</code> puts the DDM shear plane (F2) at <b>MD 10317.8 ft</b>, the same MD as the v1 (V11) SRV top, 10317.8 ft. F1 tensile is at 10365.1 ft in both. So in v1 the SRV upper boundary sat on the shear plane, and the SRV had to be narrowed.</li>
<li><b>v2 SRV:</b> all zone heights × 0.6. The outer SRV now spans <b>MD {S['srv'][0]['md_top_ft']:.1f}–{S['srv'][0]['md_bottom_ft']:.1f} ft</b> (±{S['srv'][0]['half_height_ft']:.1f} ft), compared with 10317.8–10412.4 ft (±47.3 ft) in v1. Permeability, porosity, length and everything else are unchanged.</li>
<li><b>Red/blue boundary above the fracture</b> (zero-strain crossing, mean over windows after 2/26 12:00): observed <b>{obs.zero_top_obs:.1f} ft</b>. Inverted v2 gives <b>{I['red_blue_boundary_md_ft']['simulated_top']:.1f} ft</b>, compared with {S1['inverted']['red_blue_boundary_md_ft']['simulated_top']:.1f} ft in v1. The half-max edges of the red zone also move closer: observed {obs.halfmax_top_obs:.1f}–{obs.halfmax_bottom_obs:.1f} ft; v2 gives {I['red_zone_halfmax_md_ft']['simulated_top']:.1f}–{I['red_zone_halfmax_md_ft']['simulated_bottom']:.1f} ft.</li>
<li><b>Fit:</b> strain RMSE improves from {S1['baseline']['rmse_strain']:.4f} to <b>{B['rmse_strain']:.4f}</b> millistrain (baseline) and from {S1['inverted']['rmse_strain']:.4f} to <b>{I['rmse_strain']:.4f}</b> (inverted). The all-DDM fits in 07/08 reach {B['ddm_rmse_strain']:.4f} and {I['ddm_rmse_strain']:.4f}. Most of the gain is in the second half of the shear period, where RMSE drops from {S1['inverted']['rmse_by_piece']['mid-T3']:.4f} to {I['rmse_by_piece']['mid-T3']:.4f}. The peak at the last window rises from {S1['inverted']['peak_ratio_sim_obs_last_window']:.2f} to {I['peak_ratio_sim_obs_last_window']:.2f} of observed.</li>
<li><b>Pressure</b> stays smooth. Baseline ΔP = {S['dP_base_T2_psi']:.0f} psi by T2 and {S['dP_base_T3_psi']:.0f} psi by T3. Inverted ΔP = {I['dP_last_window_before_T2_psi']:.0f} psi by T2 and {I['dP_last_window_psi']:.0f} psi at the last window, starting from an assumed 4056 psi.</li>
<li><b>Still unresolved:</b> below the fracture, the simulated strain turns negative (blue) at about {I['red_blue_boundary_md_ft']['simulated_bottom']:.0f} ft, while the observed strain stays slightly positive down to 10494 ft (Section 6).</li>
</ul>

<h2>2. SRV setup</h2>
<p>MOOSE V11 2-D plane-strain poroelastic model in the fault-normal plane. E = 50 GPa (7.25×10⁶ psi), ν = 0.2. Domain 7200 ft along strike. The SRV is a stack of nested rectangles, symmetric about the tensile fracture (F1, MD 10365.1 ft), all 5234 ft long along strike. Pressure is applied as a Dirichlet boundary condition on the fracture node at the monitoring well. Fiber MD = 10365.1 ft + fault-normal offset (ft). The mesh is refined toward the fracture.</p>
<h3>v2 (SRV height × 0.6), used for all results below</h3>
{srv_table(S['srv'])}
<h3>v1 (V11 original, SRV top on the shear plane)</h3>
{srv_table(S1['srv'])}
{img(FIG / "srv_setup_ft.png", "v2 SRV in plan view (ft). Left: whole domain. Right: zoom at the monitoring well, with MD on the right axis. The teal dashed line is the DDM shear plane at 10317.8 ft, now about 19 ft above the SRV top.")}

<h2>3. SRV height choice</h2>
<p>Two narrower SRVs were run (×0.6 and ×0.4 of the V11 heights; 2 MPI ranks each, 4 cores total). Because the model is linear in pressure, each variant needs only one ramp run.</p>
<div class="tw"><table><tr><th>SRV height</th><th>Case</th><th>Outer SRV MD (ft)</th><th>Red/blue top (ft)</th><th>Red/blue bottom (ft)</th><th>Half-max zone (ft)</th><th>Peak sim/obs</th><th>RMSE T1–T2</th><th>RMSE T2–mid</th><th>RMSE mid–T3</th><th>RMSE all</th></tr>
<tr><td><i>Observed</i></td><td></td><td></td><td>{obs.zero_top_obs:.1f}</td><td>none (stays positive)</td><td>{obs.halfmax_top_obs:.1f}–{obs.halfmax_bottom_obs:.1f}</td><td>1</td><td></td><td></td><td></td><td></td></tr>
{''.join(wrow(r) for _, r in W.iterrows())}
</table></div>
<p><b>×0.6 is the chosen v2.</b> It places the upper red/blue boundary within 0.2 ft of the observed and keeps the half-max zone close to the observed width. ×0.4 lowers RMSE a little further ({W.query("variant=='v2_h040' and case=='inverted'").rmse.iloc[0]:.4f}) because its peak is higher. However, its red/blue boundary falls 3 ft below the observed, and its red zone is about 15 ft narrower than observed at half-max, so it over-narrows the pattern you asked to match. RMSE units are millistrain. Metrics use windows after T1 + 2 days.</p>
{img(CMP / "srv_width_profiles.png", "Inverted profiles at T2, mid-shear and the last window for the three SRV heights. Narrowing moves the upper zero crossing onto the observed one and raises the peak. The observed peak sits about 10 ft below F1, which the symmetric SRV does not reproduce.")}

<h2>4. Baseline: full view and three pieces</h2>
{img(FIG / "baseline_summary.png", "Baseline (07 time law). Top: recorded injection pressure with stage numbers. Second: MOOSE tensile pressure (blue) and DDM |shear| slip (red). Third: observed waterfall. Bottom: simulated (MOOSE tensile + DDM shear). Black lines: observed profiles. Green dashed lines: simulated profiles. Dotted black lines: v2 SRV top/bottom.")}
{''.join(img(FIG / f"baseline_zoom_{k}.png", f"Baseline zoom, {t}. Colour scale set per piece (see title).") for k, t in PIECES)}

<h2>5. Inverted: full view and three pieces</h2>
{img(FIG / "inverted_summary.png", "Inverted (08 per-window scaling). Same layout as Section 4.")}
{''.join(img(FIG / f"inverted_zoom_{k}.png", f"Inverted zoom, {t}. Colour scale set per piece (see title).") for k, t in PIECES)}

<h2>6. What still does not match</h2>
<ul>
<li><b>Lower flank.</b> Below about 10420 ft, the simulated strain is about −0.02 to −0.03 millistrain at the last window, while the observed is about +0.01. At 10474 ft, the MOOSE tensile part contributes −0.018 and the DDM shear (08) contributes −0.011, so both sources contribute. Narrowing the SRV only moved this crossing from 10428 ft to 10421 ft.</li>
<li><b>Tensile-only period (T1–T2).</b> With the zoom's colour scale (±0.05), the observed strain is weakly positive across the whole fiber (10205–10494 ft) by 2/27. The simulation is positive only inside about 10310–10415 ft and slightly negative outside. Before any shear, the far-field extension cannot come from the shear source, and the pressurized SRV produces compression outside it. This is the main structural limit of the V11 tensile geometry.</li>
<li><b>Peak position.</b> The observed peak is near MD 10374 ft, about 9 ft below F1. The symmetric SRV peaks at F1, which leaves a dipole-shaped residual about the peak.</li>
<li><b>Peak height.</b> The inverted peak is {I['peak_ratio_sim_obs_last_window']:.2f} of observed at the last window. The per-window scale trades peak against flanks.</li>
</ul>
<p class="note">Possible next steps, not done here: an asymmetric SRV (top fixed near 10336 ft, bottom extended deeper, or the SRV centre shifted about 9 ft down), or refitting the DDM shear jointly with the MOOSE tensile instead of taking the 07/08 shear as fixed. Each SRV geometry change needs one MOOSE run (about 1.5 h on 2 cores).</p>

<h2>7. Pressure vs DDM tensile opening</h2>
{img(FIG / "pressure_vs_ddm_opening.png", "Top: baseline and inverted MOOSE pressure (v2 SRV). Bottom: each normalized to its last window, against the DDM tensile opening from 07/08. Baseline pressure is compared with baseline tensile, and inverted pressure with inverted tensile.")}
{img(FIG / "profiles_fitted_vs_observed.png", "Six windows. Black: observed. Green dashed: MOOSE tensile + DDM shear. Blue: MOOSE tensile. Blue dotted: DDM tensile. Red: DDM shear.")}
{img(FIG / "scaling_check.png", f"Per-window scaling vs exact superposition of the inverted pressure: {100*I['exact_vs_scaled_rel_diff']:.1f}% relative difference, exact-forward RMSE {I['rmse_strain_exact_forward']:.4f} millistrain.")}

<h2>8. Files</h2>
<ul>
<li><b>v2 outputs:</b> <code>output/v11_pinv/ddm_workflow_v2_h060/</code> (moose_pressure_2h.csv, {{baseline,inverted}}_profiles_2h.csv, srv_setup.csv, summary.json)</li>
<li><b>v2 figures:</b> <code>figs/tensile_fault_qc/moose_tensile_ddm_workflow_v2_h060/</code></li>
<li><b>×0.4 sensitivity:</b> <code>output/v11_pinv/ddm_workflow_v2_h040/</code> and <code>figs/tensile_fault_qc/moose_tensile_ddm_workflow_v2_h040/</code></li>
<li><b>Width comparison:</b> <code>figs/tensile_fault_qc/moose_tensile_ddm_workflow_v2/srv_width_comparison.csv</code> and <code>srv_width_profiles.png</code></li>
<li><b>v1 (kept):</b> <code>output/v11_pinv/ddm_workflow_v1/</code> and <code>figs/tensile_fault_qc/moose_tensile_ddm_workflow_v1/</code>. The v1 report is <code>figs/tensile_fault_qc/moose_tensile_ddm_workflow/moose_tensile_ddm_workflow_report.html</code></li>
<li><b>MOOSE runs:</b> <code>output/v11_pinv/h060/</code>, <code>h040/</code> and <code>base/</code>. Created with <code>run_step_response.py --srv-height-scale 0.6|0.4 --np 2 --run</code></li>
<li><b>Scripts</b> (<code>scripts/tensile_fault/v11_pressure_inversion/</code>): <code>moose_ddm_workflow.py --run h060 --tag v2_h060</code>, <code>compare_srv_widths.py</code> and <code>build_ddm_workflow_report_v2.py</code></li>
</ul>
</main></body></html>
"""
OUT.write_text(html, encoding="utf-8")
print(OUT, f"{OUT.stat().st_size/1e6:.1f} MB")
