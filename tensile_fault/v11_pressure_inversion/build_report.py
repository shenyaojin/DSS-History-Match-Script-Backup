"""Self-contained HTML report (figures embedded) for the V11 pressure inversion (approaches A and B)."""
import base64
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIG = REPO / "figs" / "tensile_fault_qc" / "v11_pressure_inversion"
RES = REPO / "output" / "v11_pinv" / "results"
OUT = FIG / "v11_pressure_inversion_report.html"
S = json.loads((RES / "summary.json").read_text())
PV = json.loads((RES / "provenance.json").read_text())
MPA = 0.00689476


def img(name, caption):
    b64 = base64.b64encode((FIG / name).read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}">'
            f"<figcaption>{caption}</figcaption></figure>")


A = S["A/base/inverted"]
V = A["vs_opening"]
P0 = S["P0_psi"]
dT2, dT3 = A["dP_T2_psi"], A["dP_T3_psi"]

LABEL = {"base": "V11 as is", "fine": "time steps halved", "srvx3": "SRV permeability ×3",
         "srvd3": "SRV permeability ÷3", "hfcore16": "fracture core 1e-16 m² (no fast core)"}
sens = "".join(
    f"<tr><td>{r}</td><td>{LABEL[r]}</td><td>{S[f'A/{r}/inverted']['dP_T2_psi']:.0f}</td>"
    f"<td>{S[f'A/{r}/inverted']['dP_T2_psi']*MPA:.2f}</td><td>{S[f'A/{r}/inverted']['dP_T3_psi']:.0f}</td>"
    f"<td>{S[f'A/{r}/inverted']['dP_T2_fiber_crossing_psi']:.0f}</td><td>{S[f'A/{r}/inverted']['dP_T2_srv_mean_psi']:.0f}</td>"
    f"<td>{100*S[f'A/{r}/inverted']['rel_rmse_strain']:.1f}%</td></tr>" for r in S["runs"])
targets = "".join(
    f"<tr><td>{lbl}</td><td>{S[f'A/base/{k}']['dP_T2_psi']:.0f} ± {S[f'A/base/{k}']['dP_T2_sigma_psi']:.0f}</td>"
    f"<td>{S[f'A/base/{k}']['dP_T2_psi']*MPA:.2f}</td><td>{S[f'A/base/{k}']['dP_T3_psi']:.0f}</td>"
    f"<td>{100*S[f'A/base/{k}']['rel_rmse_strain']:.1f}%</td><td>{100*S[f'A/base/{k}']['rel_rmse_rate']:.0f}%</td></tr>"
    for k, lbl in [("inverted", "DDM inverted tensile, strain fit (primary)"), ("handoff_ratefit", "DDM inverted tensile, handoff rate fit"),
                   ("baseline", "DDM baseline tensile")])
brows = "".join(
    f"<tr><td>{r}</td><td>{S[f'B/{r}/strain']['a']:.3f}</td><td>{S[f'B/{r}/strain']['b']:.3f}</td><td>{S[f'B/{r}/strain']['rmse_strain']:.4f}</td>"
    f"<td>{S[f'B/{r}/rate']['a']:.3f}</td><td>{S[f'B/{r}/rate']['b']:.3f}</td><td>{S[f'B/{r}/rate']['rmse_rate']:.3f}</td></tr>"
    for r in S["runs"])
ref = S["reference_fits"]
refrows = "".join(f"<tr><td>{k}</td><td>{v['rmse_strain']:.4f}</td><td>{v['rmse_rate']:.3f}</td></tr>" for k, v in ref.items())
b = PV["runs"]["base"]

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V11 Pressure Inversion</title>
<style>
:root{{--bg:#fff;--fg:#1d2330;--mute:#5b6477;--line:#d9dde6;--acc:#1f5fbf;--card:#f6f8fb;--warn:#b36b00}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12151c;--fg:#e6e9f0;--mute:#9aa3b5;--line:#2c3342;--acc:#7fb0ff;--card:#1a1f29;--warn:#e0a040}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:32px 16px 64px}}
h1{{font-size:1.9rem;margin:0 0 4px}} h2{{margin-top:2.4rem;border-bottom:1px solid var(--line);padding-bottom:6px}}
.sub{{color:var(--mute)}} figure{{margin:20px 0}} img{{max-width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:6px}}
figcaption{{color:var(--mute);font-size:.9rem;margin-top:6px}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:.93rem;margin:12px 0}}
th,td{{border-bottom:1px solid var(--line);padding:6px 10px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
.note{{background:var(--card);border-left:3px solid var(--acc);padding:10px 14px;border-radius:4px}}
.warn{{background:var(--card);border-left:3px solid var(--warn);padding:10px 14px;border-radius:4px}}
.big{{font-size:1.15rem}} code{{background:var(--card);padding:1px 5px;border-radius:3px;font-size:.9em}}
</style></head><body><main>
<h1>Pressure History from the V11 SRV Model</h1>
<p class="sub">Fervo Gold 4-PB, crossing-cutout case. Two-hour windows, T1 = 2025-02-24 12:00, T2 = 2025-02-28 00:00, T3 = 2025-03-06 00:00 (fixed UTC−7). Responds to the handoff request of 2026-10-05.</p>

<h2>1. Answer</h2>
<p class="note big">Fitting the DDM inverted tensile strain with the unchanged V11 SRV model gives a pressure increase from T1 to T2 of
<b>ΔP = {dT2:.0f} psi ({dT2*MPA:.2f} MPa)</b> at the injection boundary on the monitoring well, and {dT3:.0f} psi ({dT3*MPA:.2f} MPa) by T3.
With the assumed P(T1) = {P0:.0f} psi ({P0*MPA:.2f} MPa), P(T2) = {P0+dT2:.0f} psi ({(P0+dT2)*MPA:.2f} MPa).</p>
<ul>
<li><b>Most of the uncertainty comes from the model's permeability structure, not from noise.</b> The formal ±1σ is only {A['dP_T2_sigma_psi']:.0f} psi. Changing SRV permeability by a factor of 3 moves ΔP(T2) between {S['A/srvx3/inverted']['dP_T2_psi']:.0f} and {S['A/srvd3/inverted']['dP_T2_psi']:.0f} psi. Removing the high-permeability fracture core doubles it, to {S['A/hfcore16/inverted']['dP_T2_psi']:.0f} psi.</li>
<li><b>Pressure and tensile opening follow similar patterns over T1–T3</b> (linear fit {V['alpha_psi_per_mm']:.0f} psi/mm, R² = {V['r2']:.2f}). They differ in T1–T2: pressure rises earlier than opening (normalized shape RMSE {V['shape_rmse_normalized_T1T2']:.2f}).</li>
<li><b>V11 cannot reproduce the spatial shape of the DDM tensile profile.</b> It puts the flanks into compression, while the DDM profile stays positive there. This sets a floor of about 33% relative misfit whatever the pressure history (Section 6).</li>
<li><b>Approach B</b> (scaled baseline pressure plus scaled baseline shear, fitted to the observed strain) returns scale factors of about 1 (a = {S['B/base/strain']['a']:.2f} for pressure, b = {S['B/base/strain']['b']:.2f} for shear). In other words, the baseline already is the best fit of that form.</li>
<li>This fit does not establish a shear-failure threshold. It gives the boundary pressure needed for V11 to reproduce the tensile signal. Whether shear starts at T2 because of that pressure is a separate question.</li>
</ul>

<h2>2. Model setup (unchanged V11)</h2>
{img("01_v11_geometry.png", "V11 geometry: SRV 5234 ft along strike, domain 7200 ft, monitoring well crossing the tensile fracture at MD 10365.1 ft, DDM shear plane at MD 10317.8 ft, Bearskin 3-PA stage MD 9726 ft at 1934 ft.")}
<ul>
<li>The input deck is the V11 deck (<code>output/v11_srv_extended</code>). The mesh block is byte-identical (sha256 <code>{b['mesh_block_sha256'][:16]}…</code>). Materials, the 3-zone SRV permeability stack, E = 50 GPa, ν = 0.2, Biot 0.7, and the BC node are all unchanged.</li>
<li><b>Pressure definition:</b> a Dirichlet pore pressure at one node on the monitoring well (x = 755.627 m). It is a boundary forcing, not a uniform SRV pressure. Pressure diffuses outward from that node through the SRV. Section 4 therefore also reports the resulting pore pressure on the fiber at the fracture crossing and averaged over the SRV outer zone.</li>
<li><b>Observation operator</b> (as in the handoff): MD = 10365.107 + y/0.3048, axial strain = strain_yy, 10-m gauge, means over the valid native samples of each 2-hour window, first-window strain reference, 44 channels from 10,205.4 to 10,494.3 ft.</li>
<li><b>Changed vs the V11 deck:</b>
  <ol>
  <li>The pressure curve, now a unit ramp.</li>
  <li>The time-step sequence and end time.</li>
  <li><code>solve_type = LINEAR</code> instead of Newton.</li>
  </ol>
  The solver change is a correctness fix, explained in Section 8.</li>
</ul>

<h2>3. Method</h2>
<ol>
<li><b>One MOOSE run per model variant.</b> V11 is linear in pressure change, so one run with a unit pressure ramp (+1000 psi over 2 h, then held) gives the response to any pressure history built from 2-hour piecewise-linear knots. These responses are just shifted and summed copies of the ramp response. Each run takes about 20 min on 20 cores.</li>
<li><b>Approach A:</b> solve for the 114 pressure knots (T1 to T3) that best reproduce the DDM tensile strain profiles (44 channels × 114 windows). Least squares, all channels weighted equally, no bounds. Smoothness regularization (second differences), with its weight chosen by generalized cross-validation (GCV). P(T1) is fixed at the V11 initial pressure. Strain rate is a diagnostic only and is not fitted.</li>
<li><b>Approach B:</b> take the baseline pressure (approach A applied to the baseline tensile) and fit two global scalars to the observed strain: <code>observed ≈ a·MOOSE(baseline pressure) + b·DDM baseline shear</code>. The scaled pressure is P0 + a·ΔP_baseline(t). It is fitted separately to observed strain and to observed rate.</li>
</ol>

<h2>4. Approach A: pressure history</h2>
{img("04_pressure_approachA.png", "Inverted boundary pressure for the three tensile targets. The shaded band is the formal ±2σ for the primary target.")}
<div class="tw"><table><tr><th>Target</th><th>ΔP(T1→T2) psi</th><th>MPa</th><th>ΔP(T1→T3) psi</th><th>Strain misfit</th><th>Rate misfit (diagnostic)</th></tr>{targets}</table></div>
<p>The two fits to the inverted tensile agree to within 3 psi, as expected: their strain targets differ by at most 0.0005 millistrain.</p>
<p><b>Imposed vs local pressure at T2:</b> boundary {dT2:.0f} psi, fiber at the fracture crossing {A['dP_T2_fiber_crossing_psi']:.0f} psi, SRV-outer-zone mean {A['dP_T2_srv_mean_psi']:.0f} psi. The local pressures lag the boundary because pressure takes time to diffuse into the SRV.</p>
<p class="warn"><b>Early spike (first ~8 h after T1):</b> this is not a real pressure. A short pulse that then decays is the only way V11 can produce the positive late-time flank strain the DDM target has. Replacing knots 1–5 with a straight line raises the misfit only from {100*A['rel_rmse_strain']:.1f}% to {100*A['rel_rmse_strain_nospike']:.1f}%, and ΔP(T2) is unaffected.</p>
{img("03_gcv_lcurve.png", "Choice of the smoothing weight: GCV minimum and L-curve. ΔP(T2) changes by under 3% for 10× or 100× the weight, and the unregularized solution gives a similar value.")}

<h2>5. Pressure vs tensile opening</h2>
{img("07_pressure_vs_opening.png", "Raw curves, curves normalized to T3, T1–T2 zoom normalized to T2, and ΔP plotted against opening.")}
<ul>
<li><b>Over T1–T3:</b> ΔP ≈ {V['alpha_psi_per_mm']:.0f} psi/mm × opening + {V['beta_psi']:.0f} psi (R² = {V['r2']:.3f}). Normalized shape RMSE is {V['shape_rmse_normalized']:.2f}. The offset of about 300 psi mostly reflects the early rise.</li>
<li><b>In T1–T2:</b> pressure rises faster than opening at first and the two converge by T2 (shape RMSE {V['shape_rmse_normalized_T1T2']:.2f}). Pressure was not forced to follow opening. The similar overall trend is consistent with Pengchao's hypothesis, but the T1–T2 mismatch is real.</li>
<li><b>After about 3/3:</b> opening flattens and pressure keeps rising.</li>
<li><b>Lag:</b> cross-correlating the 2-hour increments gives a best lag of {V['best_lag_hours']} h, but the correlation is weak, so no meaningful lag can be resolved.</li>
</ul>

<h2>6. Fit quality and the spatial-shape limit</h2>
{img("05_profiles_inverted.png", "Target (black) vs model (red) profiles at six times. Top: strain (fitted). Bottom: rate (diagnostic).")}
{img("06_waterfall_inverted.png", "Waterfalls of target, model and residual for strain (left) and rate (right).")}
{img("13_spatial_shape.png", "Profile shape normalized to its peak, before T2 and at T3. In every model variant V11 compresses the flanks beyond about ±80 ft; the DDM target stays positive.")}
<p>The core (±100 ft) is reproduced well. The 38% overall misfit comes from the flanks. Even with a free amplitude in every window, the V11 profile shape cannot get below about 33%. Changing SRV permeability does not change this. Matching the flanks needs a different geometry or source, for example a pressurized zone with more vertical extent, or an opening (displacement-discontinuity) source rather than pore pressure alone. That is outside the fixed-geometry request.</p>
<p>The rate misfit is large (about 130%) because the target rate has sharp 2-hour features. Pressure diffusion smooths them out, so V11 cannot follow them.</p>

<h2>7. Approach B: scaled baseline pressure + scaled baseline shear vs observed</h2>
{img("08_pressure_approachB.png", "Approach B pressures: P0 + a·ΔP_baseline. The strain fit leaves the baseline essentially unscaled. The rate fit wants a much smaller pressure.")}
<div class="tw"><table><tr><th>Run</th><th>a (strain fit)</th><th>b (strain fit)</th><th>Strain RMSE (millistrain)</th><th>a (rate fit)</th><th>b (rate fit)</th><th>Rate RMSE (nanostrain/s)</th></tr>{brows}</table></div>
<p>Reference: how well the DDM-only decompositions and the approach-A combination match the observed data.</p>
<div class="tw"><table><tr><th>Combination</th><th>Strain RMSE</th><th>Rate RMSE</th></tr>{refrows}</table></div>
{img("09_waterfall_approachB.png", "Observed vs approach-B model, with residuals.")}
{img("10_channel_timeseries.png", "Three channels: shear plane (10319.6 ft), fracture (10366.7 ft) and lower flank (10420.4 ft).")}
<ul>
<li><b>Fitted to strain:</b> a ≈ 1 and b ≈ 1. The baseline tensile pressure and baseline shear are already the best combination of this form. The fit is worse than the all-DDM decomposition (strain RMSE {S['B/base/strain']['rmse_strain']:.4f} vs {ref['DDM baseline total (nb 03)']['rmse_strain']:.4f}), because V11's tensile shape differs from the DDM's (Section 6).</li>
<li><b>Main misfits:</b>
  <ul>
  <li>At the fracture channel the model under-predicts observed strain after 3/1 (about 0.095 vs 0.133 millistrain at T3).</li>
  <li>At the shear-plane channel the observed strain flattens after 3/1, while every model keeps rising.</li>
  </ul></li>
<li><b>Fitted to rate,</b> the scalars fall (a = {S['B/base/rate']['a']:.2f}, b = {S['B/base/rate']['b']:.2f}). Strain and rate disagree about the amplitude, so two global scalars are not a consistent description.</li>
</ul>

<h2>8. Checks, sensitivity and caveats</h2>
{img("02_ramp_response_convergence.png", "Left: ramp response along the fiber. Middle: response at the fracture channel for each variant. Right: base vs halved time steps. The difference is under 0.6% after the first hours.")}
{img("11_sensitivity.png", "ΔP history for each model variant (left), and normalized to T3 together with opening (right).")}
<div class="tw"><table><tr><th>Run</th><th>Variant</th><th>ΔP(T2) BC psi</th><th>MPa</th><th>ΔP(T3) psi</th><th>ΔP(T2) fiber crossing</th><th>ΔP(T2) SRV mean</th><th>Strain misfit</th></tr>{sens}</table></div>
<ul>
<li><b>Time-step convergence:</b> halving the time steps changes ΔP(T2) by {abs(S['A/fine/inverted']['dP_T2_psi']-dT2):.0f} psi.</li>
<li><b>Permeability:</b> SRV permeability ×3 or ÷3 moves ΔP(T2) by −13% / +27%. Removing the fast fracture core roughly doubles it, because then pressure must diffuse through the 1e-16 m² narrow zone.</li>
<li><b>Initial pressure:</b> the model is linear in pressure change, so ΔP does not depend on the assumed P(T1) = {P0:.0f} psi apart from second-order fluid-density effects. The absolute P(T2) moves one-for-one with whatever P(T1) is assumed. P(T1) is not identifiable from strain.</li>
<li><b>Stiffness:</b> poroelastic strain scales with ΔP/E, so ΔP scales roughly in proportion to the assumed E = 50 GPa. This was not run.</li>
</ul>
<div class="warn"><b>Solver fix, which affects earlier runs.</b>
<ul>
<li><b>The flaw:</b> the template's Newton settings (<code>nl_abs_tol = 1e-3</code>) accept a time step with zero iterations whenever the initial residual is already small. Once the boundary pressure stops changing, pressure diffusion silently freezes.</li>
<li><b>First attempted fix:</b> forcing one iteration ran into a round-off floor and stalled the time stepping.</li>
<li><b>Fix used here:</b> <code>solve_type = LINEAR</code>, one direct (MUMPS) linear solve per step. That is exact implicit Euler for this nearly linear model.</li>
<li><b>Check:</b> it matches the forced-iteration run exactly for the first 4 h. After that it grows as √t, as diffusion should, while the old settings stall.</li>
<li><b>Earlier runs:</b> the same tolerance flaw may also affect earlier V11 runs and the crossing-cutout runs (scripts 126/127) during flat stretches of their pressure curves. Those runs have not been re-checked.</li>
<li><b>Bulk compliance:</b> <code>solid_bulk_compliance</code> is pinned to the V11 value of 2e-11, because the template default has since changed.</li>
</ul></div>
<p class="note">Not done: a direct forward MOOSE run with the inverted pressure (the superposition is exact for a linear model, and the time-step check bounds its error), and an initial-pressure run (see the argument above). DDMpy was not checked against the corrected GitHub version, because the repository was not accessible from here. The DDM tensile, shear and observed profiles were used as supplied.</p>

<h2>9. Files</h2>
<ul>
<li><b>Pressure CSVs</b> (fixed UTC−7, elapsed s from T1, psi and MPa, ΔP, formal σ, fiber-crossing and SRV-mean pressure, T1/T2/T3 marked): <code>output/v11_pinv/results/A_inverted_pressure_knots.csv</code> (primary), <code>A_handoff_ratefit_…</code>, <code>A_baseline_…</code>, <code>B_fitstrain_…</code>, <code>B_fitrate_…</code></li>
<li><b>Predicted profiles and residuals</b> on the 44-channel MD grid, strain and rate: <code>output/v11_pinv/results/A_*_profiles_2h.csv</code></li>
<li><b>All metrics:</b> <code>results/summary.json</code>. Arrays: <code>results/inversion_results.npz</code>. Versions and hashes: <code>results/provenance.json</code></li>
<li><b>MOOSE inputs:</b> <code>output/v11_pinv/&lt;run&gt;/v11_pinv_&lt;run&gt;_input.i</code>. Abandoned solver attempts are in <code>output/v11_pinv/failed_runs/</code></li>
<li><b>Software:</b> MOOSE {b['MOOSE Version']}, PETSc {b['PETSc Version']}, fibeRIS {PV['fibeRIS_commit'][:10]}, 20 MPI ranks (base)</li>
<li><b>Scripts</b> in <code>scripts/tensile_fault/v11_pressure_inversion/</code>:
  <ul>
  <li><code>run_step_response.py</code>: MOOSE ramp runs</li>
  <li><code>invert_pressure.py</code>: approaches A and B</li>
  <li><code>make_figures.py</code>: figures</li>
  <li><code>provenance.py</code>: versions and hashes</li>
  <li><code>build_report.py</code>: this page</li>
  </ul></li>
</ul>
</main></body></html>
"""
OUT.write_text(html, encoding="utf-8")
print(OUT, f"{OUT.stat().st_size/1e6:.1f} MB")
