const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
const OUT = "/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/figs/tensile_fault_qc/moose_tensile_ddm_workflow_v4/";
const INK = "222831", MUTED = "5A6270", RED = "B2182B", BLUE = "2166AC", TINT = "F3F5F8";
const H = "Cambria", B = "Calibri";

// ---------- slide 1: the result ----------
let s = pres.addSlide(); s.background = { color: "FFFFFF" };
s.addText("No crossing SRV setup removes the two-sided blue", {
  x: 0.5, y: 0.3, w: 12.3, h: 0.9, fontFace: H, fontSize: 28, bold: true, color: INK, margin: 0 });
s.addText("MOOSE V11 tensile (pressure BC at the fiber/fracture node) + Pengchao's DDM shear; Gold 4-PB LF-DAS, T1–T3, 2-h windows", {
  x: 0.5, y: 1.15, w: 12.3, h: 0.35, fontFace: B, fontSize: 13, color: MUTED, italic: true, margin: 0 });
s.addImage({ path: OUT + "slide_lobe_profiles.png", x: 0.5, y: 1.65, w: 8.2, h: 4.62 });
const card = (y, h, head, body, col) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.95, y, w: 3.9, h, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
  s.addText([{ text: head, options: { bold: true, color: col, fontSize: 13, breakLine: true } },
             { text: body, options: { color: INK, fontSize: 11 } }],
    { x: 9.1, y: y + 0.08, w: 3.6, h: h - 0.16, fontFace: B, valign: "top", margin: 0, paraSpaceAfter: 4 });
};
card(1.65, 1.45, "All crossing SRVs: blue on both sides",
  "Size, shift, asymmetry, graded transition: lobes stay at −0.11 to −0.15 of peak. A pressurized zone the fiber crosses always compresses the rock around it.", BLUE);
card(3.22, 1.45, "Softer SRV only shrinks them",
  "E_srv 25 GPa → −0.09, 10 GPa → −0.06 of peak (matrix 50 GPa). Best crossing fit: 25 GPa, RMSE 0.0121 vs all-DDM 0.0126.", RED);
card(4.79, 1.48, "Lower blue is mostly DDM shear",
  "DDM shear gives −0.011 below the red. In the all-DDM fit the DDM tensile crack (fiber beyond its tip) cancels it with +0.015; MOOSE tensile gives ≤ +0.008.", INK);
s.addText("Next: refit the shear slip jointly with the MOOSE pressure (fit only, no new MOOSE runs).", {
  x: 0.5, y: 6.55, w: 12.3, h: 0.45, fontFace: B, fontSize: 15, bold: true, color: RED, margin: 0 });
s.addNotes("Panel (a): tensile-only fiber strain at the end of a +1000 psi ramp, 10-m gauge, normalised to peak. Panel (b): mean strain >= 10420 ft over the last 24 windows split into tensile and DDM shear parts. Excluded: fracture tip short of the fiber (violates crossing constraint) and free top/bottom walls (adds a uniform +3.5e-4 strain).");


// ---------- slide 2: common model setup ----------
const IMG = "/tmp/claude-439599/-rcp-rcp42-home-shenyaojin-Documents-bakken-mariner/61d67328-9082-41f9-9b14-92452866e333/scratchpad/deck/img/";
s = pres.addSlide(); s.background = { color: "FFFFFF" };
s.addText("Common setup: SRV crosses the monitoring well", { x: 0.5, y: 0.3, w: 12.3, h: 0.7, fontFace: H, fontSize: 28, bold: true, color: INK, margin: 0 });
s.addImage({ path: IMG + "plan_h060.png", x: 0.5, y: 1.2, w: 7.6, h: 4.9 });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.4, y: 1.2, w: 4.45, h: 4.9, fill: { color: TINT }, line: { color: TINT }, rectRadius: 0.08 });
s.addText([
  { text: "Same in every run", options: { bold: true, fontSize: 15, color: INK, breakLine: true } },
  { text: "Fracture + SRV 5226 ft long, 1500 ft past the fiber (crosses it)", options: { bullet: true, breakLine: true } },
  { text: "Tensile plane F1 at MD 10365.1 ft; DDM shear plane F2 at 10317.8 ft", options: { bullet: true, breakLine: true } },
  { text: "Pressure BC on one node: fracture core x monitoring well; pressure spreads by diffusion along the 1e-13 m² core", options: { bullet: true, breakLine: true } },
  { text: "Walls ±984 ft, disp_y = 0; matrix E 50 GPa, ν 0.2", options: { bullet: true, breakLine: true } },
  { text: "Pressure history fitted to the observed strain at channel 10366.7 ft (minus DDM shear); shear = Pengchao's DDM 08", options: { bullet: true } }],
  { x: 8.6, y: 1.35, w: 4.05, h: 4.6, fontFace: B, fontSize: 12.5, color: INK, valign: "top", margin: 0, paraSpaceAfter: 6 });
s.addText("Each following slide: SRV layering at the fiber (left), observed vs simulated waterfall (right, inverted fit).", {
  x: 0.5, y: 6.4, w: 12.3, h: 0.5, fontFace: B, fontSize: 14, italic: true, color: MUTED, margin: 0 });

const SETUPS = [
  ["v11_w300_well", "1. V11 SRV, full height", "SRV ±47 ft (V11 layers), centred on F1, E = matrix", "−0.16 / −0.16", "10323–10403", "−0.011", "−0.010", "0.0148", "Red too broad; blue above and below.", BLUE],
  ["h060_w300_well", "2. SRV height x0.6", "SRV ±28 ft, centred on F1", "−0.13 / −0.13", "10328–10398", "−0.019", "−0.015", "0.0132", "Red width close; blue on both sides remains.", BLUE],
  ["g060_w300_well", "3. x0.6 + graded SRV→matrix transition", "Four outer shells to ±51 ft, perm 6e-18 → 2e-18 m²", "−0.15 / −0.15", "10328–10397", "−0.015", "−0.013", "0.0136", "Smoother transition makes the lobes larger, not smaller.", BLUE],
  ["s060p15_w300_well", "4. x0.6, SRV 15 ft deeper", "SRV stack shifted +15 ft; fracture core stays at F1", "−0.14 / −0.11", "10335–10407", "−0.026", "−0.005", "0.0125", "Best position and upper blue; lower flank still slightly negative.", MUTED],
  ["a080p9_w300_well", "5. x0.8 asymmetric (top fixed, bottom extended)", "Outer SRV 10336.7–10412.4 ft", "−0.15 / −0.13", "10329–10407", "−0.022", "−0.007", "0.0132", "No gain over the symmetric shift.", BLUE],
  ["s060p9f_w300_well", "6. Free top/bottom walls (x0.6, 9 ft deeper)", "disp_y held on the strike ends instead of the walls", "+0.83 (uniform offset)", "10316–10392", "+0.011", "+0.020", "0.0249", "Adds a uniform +3.5e-4 strain along the fiber: unphysical.", BLUE],
  ["e25s15_h060_w300_well", "7. Softer SRV: E_srv 25 GPa (x0.6, 15 ft deeper)", "SRV + fracture core 25 GPa, matrix 50 GPa", "−0.09 / −0.08", "10341–10404", "−0.027", "−0.010", "0.0121", "Lobes about a third smaller; lower blue now mostly the DDM shear.", RED],
  ["e10s15_h060_w300_well", "8. Softer SRV: E_srv 10 GPa (x0.6, 15 ft deeper)", "SRV + fracture core 10 GPa, matrix 50 GPa", "−0.06 / −0.06", "10345–10401", "−0.027", "−0.013", "0.0130", "Lobes halved but red too narrow; lower flank more negative.", RED]];
for (const [run, title, geo, lobes, hm, up, low, rmse, verdict, col] of SETUPS) {
  s = pres.addSlide(); s.background = { color: "FFFFFF" };
  s.addText(title, { x: 0.4, y: 0.25, w: 12.5, h: 0.6, fontFace: H, fontSize: 24, bold: true, color: INK, margin: 0 });
  s.addText(geo + "   |   run " + run, { x: 0.4, y: 0.85, w: 12.5, h: 0.35, fontFace: B, fontSize: 12, italic: true, color: MUTED, margin: 0 });
  s.addImage({ path: IMG + "geo_" + run + ".png", x: 0.4, y: 1.35, w: 3.6, h: 3.0 });
  s.addImage({ path: IMG + "wf_" + run + ".png", x: 4.2, y: 1.35, w: 8.75, h: 4.01 });
  const m = [["Tensile lobes / peak", lobes], ["Red half-max (obs 10327–10403)", hm], ["Upper flank (obs −0.024)", up],
             ["Lower flank (obs +0.010)", low], ["RMSE (all-DDM 0.0126)", rmse]];
  s.addTable(m.map(([a, b]) => [{ text: a, options: { color: MUTED } }, { text: b, options: { bold: true, color: INK } }]),
    { x: 0.4, y: 4.6, w: 3.6, colW: [2.25, 1.35], rowH: 0.36, fontFace: B, fontSize: 10.5, fill: { color: TINT },
      border: { type: "solid", pt: 0.5, color: "FFFFFF" } });
  s.addText(verdict, { x: 4.2, y: 5.6, w: 8.75, h: 0.6, fontFace: B, fontSize: 16, bold: true, color: col, margin: 0, valign: "top" });
  s.addText("Flanks: mean simulated strain, millistrain, last 24 two-hour windows (upper 10270–10310 ft, lower >= 10420 ft).", {
    x: 4.2, y: 6.35, w: 8.75, h: 0.4, fontFace: B, fontSize: 10.5, color: MUTED, margin: 0 });
}
// ---------- last slide: test log ----------
s = pres.addSlide(); s.background = { color: "FFFFFF" };
s.addText("What was tested (Nov 8–9, 2026)", { x: 0.5, y: 0.3, w: 12.3, h: 0.8, fontFace: H, fontSize: 28, bold: true, color: INK, margin: 0 });
const hdr = ["SRV setup", "Tensile lobes / peak (up / low)", "Red half-max (ft)", "Below-red flank", "RMSE", "Verdict"]
  .map(t => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: INK } } }));
const R = (a, verdictCol) => a.map((t, i) => ({ text: t, options: i === 5 ? { color: verdictCol, bold: true } : {} }));
const rows = [hdr,
  R(["V11 SRV, full height", "−0.16 / −0.16", "10323–10403", "−0.010", "0.0148", "red too broad"], BLUE),
  R(["V11 x0.6, at F1 (reference)", "−0.13 / −0.13", "10328–10398", "−0.015", "0.0132", "two-sided blue"], BLUE),
  R(["x0.6 + graded SRV→matrix transition", "−0.15 / −0.15", "10328–10397", "−0.013", "0.0136", "worse"], BLUE),
  R(["x0.6, 9 ft deeper", "−0.14 / −0.12", "10332–10405", "−0.010", "0.0122", "position better"], MUTED),
  R(["x0.6, 15 ft deeper", "−0.14 / −0.11", "10335–10407", "−0.005", "0.0125", "best geometry"], MUTED),
  R(["x0.8 asymmetric (top fixed)", "−0.15 / −0.13", "10329–10407", "−0.007", "0.0132", "worse"], BLUE),
  R(["Free top/bottom walls", "uniform +3.5e-4 offset", "—", "+0.020", "0.025", "unphysical"], BLUE),
  R(["15 ft deeper, E_srv 25 GPa", "−0.09 / −0.08", "10341–10404", "−0.010", "0.0121", "lobes smaller"], RED),
  R(["15 ft deeper, E_srv 10 GPa", "−0.06 / −0.06", "10345–10401", "−0.013", "0.0130", "red too narrow"], RED),
  R(["Tip 20–120 ft short of fiber", "−0.12 → +0.17", "—", "−0.011 → +0.024", "0.0109+", "not allowed (must cross)"], MUTED)];
s.addTable(rows, { x: 0.5, y: 1.2, w: 12.3, colW: [3.3, 2.3, 1.7, 1.6, 1.0, 2.4], fontFace: B, fontSize: 12, color: INK,
  border: { type: "solid", pt: 0.5, color: "C9CED6" }, rowH: 0.39, fill: { color: "FFFFFF" } });
s.addText([
  { text: "Observed: ", options: { bold: true } },
  { text: "half-max 10327–10403 ft, blue above −0.024, below-red flank +0.010 millistrain. All-DDM RMSE 0.0126. Flanks: last 24 windows; RMSE: inverted fit, millistrain. Pressure baseline choice (observed vs DDM-tensile, channel, monotonic or not) changes RMSE by ≤0.0002 and not the shape.", options: {} }],
  { x: 0.5, y: 5.6, w: 12.3, h: 0.9, fontFace: B, fontSize: 12, color: MUTED, margin: 0, valign: "top" });
pres.writeFile({ fileName: OUT + "srv_lobe_tests_2026-11-09.pptx" }).then(f => console.log(f));
