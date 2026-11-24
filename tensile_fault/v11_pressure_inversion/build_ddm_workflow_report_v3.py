"""v3 HTML report: lobe diagnosis, BC-node fix, central-channel pressure baseline, BC at Bearskin 4-PB (lengths in ft)."""
import base64
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUTD = REPO / "output" / "v11_pinv"
FIGS = REPO / "figs" / "tensile_fault_qc"
V3 = FIGS / "moose_tensile_ddm_workflow_v3"
OUT = V3 / "moose_tensile_ddm_workflow_v3_report.html"
SEL = "ctr_h060_w300_well"                               # selected setup
FIG = FIGS / f"moose_tensile_ddm_workflow_{SEL}"
CASES = [("v2_h060", "v2 reference: SRV ×0.6, walls ±328 ft, 07 time law"),
         ("v3_h060_w300", "SRV ×0.6, walls ±984 ft, 07 time law"),
         ("ctr_v11_w300_well", "SRV ×1.0, BC on well, central channel"),
         ("ctr_h060_w300_well", "<b>SRV ×0.6, BC on well, central channel</b>"),
         ("ctr_v11_w300_4pb", "SRV ×1.0, BC at 4-PB, central channel"),
         ("ctr_h060_w300_4pb", "SRV ×0.6, BC at 4-PB, central channel")]
S = {k: json.loads((OUTD / f"ddm_workflow_{k}" / "summary.json").read_text()) for k, _ in CASES}
SS = S[SEL]
B, I = SS["baseline"], SS["inverted"]
LOBE = pd.read_csv(V3 / "lobe_diagnosis.csv")
BCC = pd.read_csv(V3 / "bc_location_check.csv").set_index("run")
PIECES = [("1_tensile_only", "T1–T2: purely tensile (02/24 12:00 – 02/28 00:00)"),
          ("2_shear_first_half", "T2–mid: shear, first half (02/28 00:00 – 03/03 00:00)"),
          ("3_shear_second_half", "mid–T3: shear, second half (03/03 00:00 – 03/06 00:00)")]


def img(path, caption):
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


def f1(v, fmt="{:.1f}"):
    return "—" if v is None or v != v else fmt.format(v)


def srv_table(srv):
    r = "".join(f"<tr><td>{z['zone']}</td><td>±{z['half_height_ft']:.1f}</td><td>{2*z['half_height_ft']:.1f}</td>"
                f"<td>{z['md_top_ft']:.1f}</td><td>{z['md_bottom_ft']:.1f}</td><td>{z['length_ft']:.0f}</td>"
                f"<td>{z['perm_m2']:.0e}</td><td>{z['perm_mD']:.3g}</td><td>{z['porosity']:.2f}</td></tr>" for z in srv)
    return ('<div class="tw"><table><tr><th>Zone</th><th>Half-height (ft)</th><th>Total height (ft)</th><th>MD top (ft)</th>'
            '<th>MD bottom (ft)</th><th>Length along strike (ft)</th><th>Perm (m²)</th><th>Perm (mD)</th><th>Porosity</th></tr>'
            + r + '<tr><td>matrix</td><td colspan="5">rest of the 7174.9 ft × 1969 ft domain</td><td>1e-18</td><td>0.00101</td><td>0.01</td></tr></table></div>')


def case_rows(kind):
    out = []
    for k, name in CASES:
        m = S[k][kind]
        rb, hm, p = m["red_blue_boundary_md_ft"], m["red_zone_halfmax_md_ft"], m["rmse_by_piece"]
        out.append(f"<tr><td>{name}</td><td>{f1(rb['simulated_top'])}</td><td>{f1(rb['simulated_bottom'])}</td>"
                   f"<td>{f1(hm['simulated_top'])}–{f1(hm['simulated_bottom'])}</td><td>{m['peak_ratio_sim_obs_last_window']:.2f}</td>"
                   f"<td>{m['dP_last_window_before_T2_psi']:,.0f}</td><td>{m['dP_last_window_psi']:,.0f}</td>"
                   f"<td>{p['T1-T2']:.4f}</td><td>{p['T2-mid']:.4f}</td><td>{p['mid-T3']:.4f}</td><td>{m['rmse_strain']:.4f}</td></tr>")
    return "".join(out)


ob = B["red_blue_boundary_md_ft"]["observed_top"]
oh = B["red_zone_halfmax_md_ft"]
lobe = LOBE.set_index("model").min_over_peak
CT = ('<div class="tw"><table><tr><th>Case</th><th>Red/blue top (ft)</th><th>Red/blue bottom (ft)</th><th>Half-max zone (ft)</th>'
      '<th>Peak sim/obs</th><th>ΔP by T2 (psi)</th><th>ΔP last window (psi)</th><th>RMSE T1–T2</th><th>RMSE T2–mid</th>'
      '<th>RMSE mid–T3</th><th>RMSE all</th></tr>'
      f'<tr><td><i>Observed</i></td><td>{ob:.1f}</td><td>none (stays positive)</td><td>{oh["observed_top"]:.1f}–{oh["observed_bottom"]:.1f}</td>'
      '<td>1</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>')

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOOSE Tensile DDM v3</title>
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
<h1>MOOSE Tensile + DDM Shear, v3</h1>
<p class="sub">Lobe diagnosis, pressure-node fix, a pressure baseline fitted to the central channel, and a BC at Bearskin 4-PB. Same strain workflow as v1/v2 (notebooks 07/08, 2-hour windows, MOOSE V11 tensile pressure + DDM shear). All lengths in ft. T1 = 2025-02-24 12:00, T2 = 2025-02-28 00:00, T3 = 2025-03-06 00:00 (fixed UTC−7). Injection colours: Bearskin 3-PA red, 1-IA earth-yellow, 4-PB purple.</p>

<h2>1. Result</h2>
<ul>
<li><b>Selected setup:</b> V11 SRV with permeabilities unchanged, heights ×0.6, walls at ±984 ft, pressure BC exactly on the monitoring well, baseline pressure fitted to the central channel (MD {SS['baseline_method']['channel_md_ft']:.1f} ft). Strain RMSE is <b>{B['rmse_strain']:.4f}</b> millistrain (baseline) and <b>{I['rmse_strain']:.4f}</b> (inverted). The all-DDM fits in 07/08 reach {B['ddm_rmse_strain']:.4f} and {I['ddm_rmse_strain']:.4f}.</li>
<li><b>Original SRV height is too broad.</b> With the SRV top on the shear plane (10317.8 ft), the central-channel baseline puts the red/blue boundary at {S['ctr_v11_w300_well']['baseline']['red_blue_boundary_md_ft']['simulated_top']:.1f} ft, compared with {ob:.1f} ft observed. ×0.6 gives {B['red_blue_boundary_md_ft']['simulated_top']:.1f} ft (baseline) and {I['red_blue_boundary_md_ft']['simulated_top']:.1f} ft (inverted). Following your rule, ×0.6 is used.</li>
<li><b>Central-channel pressure</b> is smooth and non-decreasing by construction. It rises to ΔP ≈ {SS['dP_base_T2_psi']:.0f} psi by T2 and {SS['dP_base_T3_psi']:.0f} psi by T3, and flattens once injection stops on 03/04. This fits the central channel to {SS['baseline_method']['center_rmse_millistrain']:.5f} millistrain, but over-predicts the peak elsewhere (sim/obs {B['peak_ratio_sim_obs_last_window']:.2f} at the last window). Over the whole fiber it is slightly worse than the 07 time law fitted to all channels (RMSE {S['v3_h060_w300']['baseline']['rmse_strain']:.4f}).</li>
<li><b>BC at Bearskin 4-PB does not work with the V11 hydraulics.</b> A pressure step 1973 ft along strike delivers only about {BCC.loc['h060_w300_4pb','fiber_dP_fraction_at_end']:.0e} of itself to the fiber after 228 h. The central-channel fit then needs ΔP of about {S['ctr_h060_w300_4pb']['dP_base_T3_psi']/1e6:.1f} million psi, which is not physical (Section 6).</li>
<li><b>Negative lobes</b> are mostly mechanical. The fixed walls at ±328 ft in V11 pushed the lobe to {lobe.iloc[1]:+.2f} of the peak; walls at ±984 ft reduce it to {lobe.iloc[2]:+.2f}. The rest comes from the fiber crossing the pressurized zone, 1500 ft inside its tip. The DDM tensile ellipse ends 21 ft before the fiber, which is why its flanks stay positive ({lobe.iloc[0]:+.3f}). SRV permeability ×3 or ÷3 changes the lobe by only about 0.03 (Section 3).</li>
</ul>

<h2>2. Experiment priority</h2>
<div class="tw"><table><tr><th>#</th><th>Experiment</th><th>Status</th></tr>
<tr><td>1</td><td>Original V11 SRV, BC on the intersection channel vs at 4-PB, central-channel baseline</td><td>Done. Original height too broad; 4-PB BC not viable with V11 hydraulics</td></tr>
<tr><td>2</td><td>Same with SRV ×0.6 (fallback rule)</td><td>Done. <b>Selected</b></td></tr>
<tr><td>3</td><td>Fracture-core conductivity sweep with the BC at 4-PB (what conductivity carries 4-PB pressure 1973 ft in hours)</td><td>Not run. Needed only if the injector-side BC is to be kept. About 1.5 h per value on 2 cores</td></tr>
<tr><td>4</td><td>Asymmetric / shifted SRV (observed peak about 9 ft below F1; lower flank)</td><td>Not run</td></tr>
<tr><td>5</td><td>P0 = 3000 psi instead of 4056 psi</td><td>Not run. The model is linear in ΔP, apart from about 0.3% from fluid density</td></tr>
<tr><td>—</td><td>Transition layer / higher SRV permeability</td><td>Dropped. Permeability barely moves the lobes</td></tr>
</table></div>

<h2>3. Why the MOOSE lobes are more negative than the DDM</h2>
{img(V3 / "lobe_diagnosis.png", "(a) Tensile profile shape on the 44 DAS channels, last window, each scaled to its peak: DDM tensile (07) vs MOOSE for three wall distances. (b) MOOSE strain along the whole fiber per psi of BC pressure: the walls (disp_y = 0) force zero net extension, which shows up as compression outside the SRV. (c) Analytic 2-D pressurized crack with no walls: a fiber just beyond the crack tip (DDM geometry) sees positive flanks; a fiber crossing the crack 1500 ft inside the tip (MOOSE geometry) sees a small negative lobe.")}
<div class="tw"><table><tr><th>Model</th><th>Lobe min / peak (DAS window)</th><th>Strain at 295 ft / peak</th></tr>
{''.join(f"<tr><td>{r.model}</td><td>{r.min_over_peak:+.3f}</td><td>{f1(r.far_field_over_peak_295ft, '{:+.3f}')}</td></tr>" for r in LOBE.itertuples())}
</table></div>
<p>Walls moved from ±328 ft to ±984 ft are used from here on. Moving them further, to ±3281 ft, changes little, so the remaining lobe belongs to the geometry, not to the boundary.</p>

<h2>4. Pressure BC node: fix</h2>
<p>MOOSE's <code>ExtraNodesetGenerator</code> needs the BC point on a mesh node. In V11, the 7200 ft domain with 880 cells has no node at the monitoring well, so the BC was snapped to the nearest node, <b>3.9 ft from the fiber</b>. This was not intended. v3 sets the domain to 7174.9 ft so that a node falls exactly on the well (snap error 0.00 ft). The 4-PB BC snaps to within 0.7 ft of its target.</p>

<h2>5. Pressure baseline from the central channel</h2>
<p>The baseline pressure is no longer the 07 DDM time law. ΔP(t) is fitted (non-negative increments, so pressure never falls) to the observed minus DDM shear (07) strain at the channel nearest the crossing, MD {SS['baseline_method']['channel_md_ft']:.1f} ft. A second-difference penalty keeps the curve smooth. Its weight (λ = {SS['baseline_method']['smoothing_lambda']:.3g}) is the largest that keeps the central-channel RMSE within 10% of its minimum. Every other channel is then a forward prediction. The inverted case is unchanged: the 08 per-window scaling, applied on top of this baseline.</p>
<h3>Case comparison</h3>
<p>Metrics use windows after T1 + 2 days. RMSE in millistrain.</p>
<h4>Baseline</h4>{CT}{case_rows('baseline')}</table></div>
<h4>Inverted (08 per-window scaling)</h4>{CT}{case_rows('inverted')}</table></div>
{img(FIGS / "moose_tensile_ddm_workflow_ctr_v11_w300_well" / "baseline_summary.png", "Original SRV height (×1.0, top on the shear plane 10317.8 ft), central-channel baseline: the red zone is broader than observed and the blue starts about 11 ft too high.")}

<h2>6. BC at the Bearskin 4-PB injection point</h2>
<p><b>Geometry.</b> The 4-PB trajectory crosses the F1 plane at 4-PB MD 10143.3 ft, 1973.8 ft along strike from the Gold 4-PB fiber crossing, and 39.5 ft beyond the DDM tensile centre (1934.3 ft). The SRV extends 3734 ft toward the injector, so the BC node (+1973.1 ft) sits inside the SRV and on the fracture core.</p>
{img(FIGS / "moose_tensile_ddm_workflow_ctr_h060_w300_4pb" / "srv_setup_ft.png", "SRV ×0.6 with the BC node at Bearskin 4-PB. The fiber is at 0 ft along strike.")}
{img(V3 / "bc_location_check.png", "(a) Pressure at the fiber crossing as a fraction of the BC step. With the BC on the well it is 96% after 24 h; with the BC at 4-PB it stays at a numerical floor of about 2×10⁻⁶ and starts to rise only after about 130 h. (b) Strain at the central channel per psi of BC pressure: about 1700× smaller for the 4-PB BC at the end of the run, and that small strain is far-field poroelastic deformation, not pressure arriving at the fiber.")}
<p class="warn">With the V11 fracture core (1e-13 m², 0.2 ft thick), pressure needs about L²/D ≈ 4 days to diffuse 1974 ft even without leak-off. Leak-off into the SRV removes almost all of it. A real stage fracture is much more conductive. To use an injector-side BC, the fracture-core conductivity has to be calibrated first (priority 3). The 4-PB central-channel fits return ΔP of millions of psi and are kept only as evidence. Their figures are in the folders listed below.</p>

<h2>7. Selected setup: SRV</h2>
{srv_table(SS['srv'])}
{img(FIG / "srv_setup_ft.png", "SRV ×0.6 with the BC node exactly on the monitoring well. The teal dashed line is the DDM shear plane at 10317.8 ft.")}

<h2>8. Selected setup, baseline: full view and three pieces</h2>
{img(FIG / "baseline_summary.png", "Baseline (central-channel pressure). Top: recorded injection pressure. Second: MOOSE tensile pressure (blue) and DDM |shear| slip (red). Third: observed waterfall. Bottom: simulated. Black lines: observed profiles. Green dashed lines: simulated profiles. Dotted lines: SRV top/bottom.")}
{''.join(img(FIG / f"baseline_zoom_{k}.png", f"Baseline zoom, {t}. Colour scale set per piece.") for k, t in PIECES)}

<h2>9. Selected setup, inverted: full view and three pieces</h2>
{img(FIG / "inverted_summary.png", "Inverted (08 per-window scaling on the central-channel baseline). Same layout as Section 8.")}
{''.join(img(FIG / f"inverted_zoom_{k}.png", f"Inverted zoom, {t}. Colour scale set per piece.") for k, t in PIECES)}

<h2>10. Pressure vs DDM tensile opening</h2>
{img(FIG / "pressure_vs_ddm_opening.png", "Top: baseline (central-channel) and inverted MOOSE pressure. Bottom: each normalized to its last window, against the DDM tensile opening from 07/08.")}
{img(FIG / "profiles_fitted_vs_observed.png", "Six windows. Black: observed. Green dashed: MOOSE tensile + DDM shear. Blue: MOOSE tensile. Blue dotted: DDM tensile. Red: DDM shear.")}
{img(FIG / "scaling_check.png", f"Per-window scaling vs exact superposition of the inverted pressure: {100*I['exact_vs_scaled_rel_diff']:.1f}% relative difference, exact-forward RMSE {I['rmse_strain_exact_forward']:.4f} millistrain.")}

<h2>11. What still does not match</h2>
<ul>
<li><b>Flanks:</b> lobes are weaker with the ±984 ft walls but still negative outside about {B['red_blue_boundary_md_ft']['simulated_top']:.0f}–{B['red_blue_boundary_md_ft']['simulated_bottom']:.0f} ft. The observed strain stays weakly positive across the window, already in T1–T2.</li>
<li><b>Peak position:</b> the observed peak is about 9 ft below F1; the symmetric SRV peaks at F1.</li>
<li><b>Peak height vs central channel:</b> fitting one channel exactly makes the simulated peak {B['peak_ratio_sim_obs_last_window']:.2f}× observed at the last window. The per-window scaling brings it to {I['peak_ratio_sim_obs_last_window']:.2f}.</li>
</ul>

<h2>12. Files</h2>
<ul>
<li><b>MOOSE runs</b> (<code>output/v11_pinv/</code>): <code>v11_w300_well</code>, <code>v11_w300_4pb</code>, <code>h060_w300_well</code>, <code>h060_w300_4pb</code>, plus <code>h060_w300</code> and <code>h060_w1000</code> (wall study). Built with <code>run_step_response.py --model-width 300 --align-well-node [--srv-height-scale 0.6] [--bc-along-strike-ft 1973.8] --np 2|3 --run</code></li>
<li><b>Fits:</b> <code>output/v11_pinv/ddm_workflow_ctr_*/</code>. Figures: <code>figs/tensile_fault_qc/moose_tensile_ddm_workflow_ctr_*/</code>. Built with <code>moose_ddm_workflow.py --run &lt;run&gt; --tag ctr_&lt;run&gt; --baseline center</code></li>
<li><b>Diagnostics:</b> <code>figs/tensile_fault_qc/moose_tensile_ddm_workflow_v3/</code> (lobe_diagnosis.png/csv, bc_location_check.png/csv). Scripts: <code>lobe_diagnosis.py</code>, <code>crack_lobe_check.py</code>, <code>bc_location_check.py</code></li>
<li><b>Dropped transition-layer runs:</b> <code>output/v11_pinv/failed_runs/</code></li>
<li><b>Earlier versions kept:</b> v1 and v2 outputs, figures and reports are unchanged</li>
<li><b>This report:</b> <code>build_ddm_workflow_report_v3.py</code></li>
</ul>
</main></body></html>
"""
OUT.write_text(html, encoding="utf-8")
print(OUT, f"{OUT.stat().st_size/1e6:.1f} MB")
