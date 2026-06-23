// pyMack validation deck — one slide per successful case (overlay + run conditions + agreement).
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const DATA = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf-8"));

// ---- palette ----
const NAVY = "102A43", TEAL = "1C7293", ORANGE = "C2410C";
const SLATE = "334155", BODY = "475569", MUTED = "7C8DA0";
const ICE = "CADCFC", CARD = "F4F7FA";
const GREEN = "2C7A4B", AMBER = "B45309";
const GREEN_T = "EAF4EE", AMBER_T = "FBF1E0";

const HDR = "Cambria", SANS = "Calibri";
const sh = () => ({ type: "outer", color: "000000", blur: 7, offset: 3, angle: 90, opacity: 0.13 });

let pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";          // 13.3 x 7.5 in
pres.author = "pyMack";
pres.title = "pyMack LST Verification";
const PW = 13.3, PH = 7.5;

function pngSize(p) {
  const b = fs.readFileSync(p).slice(0, 24);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

// ---------- title slide ----------
(() => {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("pyMack", { x: 0.9, y: 1.55, w: 11.5, h: 1.3, fontSize: 66, bold: true, color: "FFFFFF", fontFace: HDR, margin: 0 });
  s.addText("Compressible Linear Stability Theory", { x: 0.95, y: 2.95, w: 11.5, h: 0.6, fontSize: 27, color: ICE, fontFace: HDR, margin: 0 });
  s.addText("Verification against published benchmarks", { x: 0.97, y: 3.62, w: 11.5, h: 0.5, fontSize: 18, italic: true, color: "9FB3D9", fontFace: SANS, margin: 0 });
  s.addText([
    { text: "20", options: { bold: true, color: "FFFFFF" } },
    { text: " validation cases   ·   ", options: { color: "9FB3D9" } },
    { text: "8", options: { bold: true, color: "FFFFFF" } },
    { text: " independent sources", options: { color: "9FB3D9" } },
  ], { x: 0.97, y: 4.55, w: 11.5, h: 0.5, fontSize: 18, fontFace: SANS, margin: 0 });
  s.addText("Mack (1984)   ·   Malik (1990)   ·   Balakumar & Malik (1992)   ·   Ma & Zhong (2003)   ·   Egorov et al. (2006)   ·   Sivasubramanian & Fasel (2015)   ·   Özgen & Kırcalı (2008)",
    { x: 0.97, y: 6.35, w: 11.4, h: 0.5, fontSize: 12, color: TEAL, fontFace: SANS, margin: 0 });
  s.addText("June 2026", { x: 0.97, y: 6.85, w: 6, h: 0.35, fontSize: 12, color: MUTED, fontFace: SANS, margin: 0 });
})();

// ---------- overview slide ----------
(() => {
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  s.addText("Validation overview", { x: 0.55, y: 0.45, w: 12, h: 0.6, fontSize: 32, bold: true, color: NAVY, fontFace: HDR, margin: 0 });
  s.addText("pyMack run at each benchmark's exact conditions, overlaid on the published / independent reference",
    { x: 0.57, y: 1.12, w: 12, h: 0.4, fontSize: 15, color: TEAL, fontFace: SANS, margin: 0 });
  // stat callouts
  const stats = [["20", "successful cases"], ["11", "second-mode"], ["9", "first-mode"], ["8", "sources"]];
  const sw = 2.9, gap = 0.25, x0 = 0.55, y0 = 1.95;
  stats.forEach((st, i) => {
    const x = x0 + i * (sw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y0, w: sw, h: 1.5, fill: { color: CARD }, rectRadius: 0.08, shadow: sh() });
    s.addText(st[0], { x, y: y0 + 0.12, w: sw, h: 0.85, fontSize: 48, bold: true, color: TEAL, fontFace: HDR, align: "center", margin: 0 });
    s.addText(st[1], { x, y: y0 + 0.98, w: sw, h: 0.4, fontSize: 14, color: SLATE, fontFace: SANS, align: "center", margin: 0 });
  });
  // method + quantities
  s.addText("What each slide shows", { x: 0.55, y: 3.85, w: 12, h: 0.4, fontSize: 18, bold: true, color: NAVY, fontFace: HDR, margin: 0 });
  s.addText([
    { text: "pyMack result overlaid on the benchmark", options: { bold: true, color: SLATE, breakLine: true } },
    { text: "— for neutral curves, the pyMack cᵢ = 0 contour vs. the digitized paper curve; for growth rates / eigenvalues, the pyMack curve vs. the reference points.", options: { color: BODY, breakLine: true } },
    { text: "Full run conditions", options: { bold: true, color: SLATE, breakLine: true } },
    { text: "— Mach, gas, wall BC (with adiabatic recovery Tw/Te), edge temperature, Prandtl, transport / viscosity law, Reynolds scaling, and formulation.", options: { color: BODY, breakLine: true } },
    { text: "An honest agreement metric", options: { bold: true, color: SLATE, breakLine: true } },
    { text: "— median relative error or eigenvalue match, at the digitization noise floor where applicable.", options: { color: BODY } },
  ], { x: 0.55, y: 4.3, w: 12.2, h: 2.6, fontSize: 14.5, fontFace: SANS, margin: 0, valign: "top", paraSpaceAfter: 7 });
})();

// ---------- section divider ----------
function divider(title, sub) {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText(title, { x: 0.95, y: 2.85, w: 11.5, h: 1.0, fontSize: 44, bold: true, color: "FFFFFF", fontFace: HDR, margin: 0 });
  s.addText(sub, { x: 0.97, y: 3.95, w: 11.5, h: 0.5, fontSize: 19, color: ICE, fontFace: SANS, margin: 0 });
}

// ---------- case slide ----------
function caseSlide(c) {
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  s.addText(c.title, { x: 0.5, y: 0.32, w: 12.3, h: 0.5, fontSize: 25, bold: true, color: NAVY, fontFace: HDR, margin: 0, valign: "middle" });
  s.addText(c.subtitle, { x: 0.52, y: 0.84, w: 12.3, h: 0.36, fontSize: 14.5, color: TEAL, fontFace: SANS, margin: 0, valign: "middle" });

  // figure: frame hugs the image, centered (H & V) in the left region that matches
  // the right-hand cards' vertical span, so every figure reads as a tight framed unit
  const regX = 0.5, regW = 7.7, regY = 1.42, regH = 5.5, pad = 0.16;
  const dim = pngSize(c.overlay);
  const r = Math.min((regW - 2 * pad) / dim.w, (regH - 2 * pad) / dim.h);
  const iw = dim.w * r, ih = dim.h * r;
  const fw = iw + 2 * pad, fh = ih + 2 * pad;
  const fx = regX + (regW - fw) / 2, fy = regY + (regH - fh) / 2;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: fx, y: fy, w: fw, h: fh, fill: { color: "FFFFFF" }, line: { color: "DBE3EC", width: 1 }, rectRadius: 0.05, shadow: sh() });
  s.addImage({ path: c.overlay, x: fx + pad, y: fy + pad, w: iw, h: ih });

  // conditions card (right)
  const cx = 8.45, cw = 4.35, cy = 1.42, ch = 3.85;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: cy, w: cw, h: ch, fill: { color: CARD }, rectRadius: 0.06, shadow: sh() });
  s.addText("RUN CONDITIONS", { x: cx + 0.26, y: cy + 0.16, w: cw - 0.5, h: 0.3, fontSize: 11.5, bold: true, color: TEAL, charSpacing: 2, fontFace: SANS, margin: 0 });
  const rt = [];
  c.conditions.forEach(([lab, val], i) => {
    rt.push({ text: lab + ":  ", options: { bold: true, color: SLATE, breakLine: false } });
    rt.push({ text: String(val), options: { color: BODY, breakLine: true } });
  });
  s.addText(rt, { x: cx + 0.26, y: cy + 0.5, w: cw - 0.5, h: ch - 0.66, fontSize: 9.5, fontFace: SANS, margin: 0, valign: "top", paraSpaceAfter: 3, fit: "shrink" });

  // agreement card (right, below)
  const ay = cy + ch + 0.1, ah = 6.92 - ay;
  const agree = c.verdict === "agrees";
  const tint = agree ? GREEN_T : AMBER_T, accent = agree ? GREEN : AMBER;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: ay, w: cw, h: ah, fill: { color: tint }, rectRadius: 0.06, shadow: sh() });
  s.addText("AGREEMENT", { x: cx + 0.26, y: ay + 0.14, w: 2.2, h: 0.3, fontSize: 11.5, bold: true, color: accent, charSpacing: 2, fontFace: SANS, margin: 0 });
  s.addText((agree ? "AGREES" : "ACCEPTABLE"), { x: cx + cw - 1.85, y: ay + 0.12, w: 1.6, h: 0.32, fontSize: 12, bold: true, color: "FFFFFF", fontFace: SANS, align: "center", valign: "middle", fill: { color: accent }, rectRadius: 0.04, margin: 0 });
  s.addText(String(c.agreement), { x: cx + 0.26, y: ay + 0.5, w: cw - 0.5, h: ah - 0.62, fontSize: 9.5, color: SLATE, fontFace: SANS, margin: 0, valign: "top", fit: "shrink" });

  // footer
  s.addText([
    { text: c.case_id, options: { bold: true, color: SLATE } },
    { text: "    ·    pyMack vs. benchmark — verification audit", options: { color: MUTED } },
  ], { x: 0.5, y: 7.04, w: 12.3, h: 0.3, fontSize: 9, fontFace: SANS, margin: 0 });
}

// ---------- closing ----------
function closing() {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("Summary", { x: 0.95, y: 0.6, w: 11.5, h: 0.7, fontSize: 36, bold: true, color: "FFFFFF", fontFace: HDR, margin: 0 });
  s.addText([
    { text: "Second (Mack) mode — validated.", options: { bold: true, color: "FFFFFF", breakLine: true } },
    { text: "Agrees with Mack (1984), Malik (1990), Balakumar & Malik (1992), Ma & Zhong (2003), Egorov (2006), an independent collaborator solver, and a cone (Sivasubramanian & Fasel 2015), across Mach 4.5–10.", options: { color: ICE, breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 8 } },
    { text: "First mode — agrees where the discrete mode is resolvable.", options: { bold: true, color: "FFFFFF", breakLine: true } },
    { text: "Özgen & Kırcalı (2008) neutral curves M = 2–10: second-mode branches and the first-mode cutoff agree once a discrete-mode (eigenfunction-decay) extractor replaces the phase-speed-band classifier.", options: { color: ICE, breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 8 } },
    { text: "Honest open items.", options: { bold: true, color: "FFFFFF", breakLine: true } },
    { text: "The low-α first-mode onset at low Mach is continuous-spectrum-limited; at high Mach pyMack marginally over-predicts the inter-mode region. Mack Fig 10.1/10.4 low-Mach first modes await the same discrete-mode scrutiny.", options: { color: ICE } },
  ], { x: 0.97, y: 1.6, w: 11.4, h: 5.2, fontSize: 16, fontFace: SANS, margin: 0, valign: "top", paraSpaceAfter: 6 });
}

// ---------- assemble ----------
const second = DATA.filter(c => c.mode_dir === "second_mode");
const first = DATA.filter(c => c.mode_dir !== "second_mode");
divider("Second (Mack) mode", `${second.length} validation cases — pyMack's design target`);
second.forEach(caseSlide);
divider("First mode", `${first.length} validation cases — Özgen Fig 3 + Mack oblique first mode`);
first.forEach(caseSlide);
closing();

pres.writeFile({ fileName: path.join(__dirname, "pyMack_validation.pptx") }).then(f => console.log("wrote", f));
