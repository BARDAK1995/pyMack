"""Assemble per-case data for the pyMack validation slide deck.

Reads every successful (agrees/acceptable) verdict.json, builds a clean conditions
spec sheet per case (computing the adiabatic recovery Tw/Te where the wall is
adiabatic and not otherwise given), reuses build_success_matrix._headline for the
agreement line, and writes deck_data.json for the pptxgenjs generator.
"""
from __future__ import annotations
import json
import os
import sys
from math import sqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent
VER = HERE.parent
REPO = VER.parent
sys.path.insert(0, str(VER))
from _compare_lib import VERDICT_BADGE  # noqa: E402
from build_success_matrix import _headline  # noqa: E402

GOOD = ("agrees", "acceptable")
MODE_ORDER = {"second_mode": 0, "first_mode": 1, "other": 2}

# Human title + citation per case_id (and a short quantity tag).
TITLES = {
    "sean_m5p35": ("Independent collaborator LST", "M = 5.35 · N₂ · 2nd-mode neutral curve"),
    "mazhong2003_m4p5": ("Ma & Zhong (2003), JFM", "M = 4.5 · 2nd-mode neutral curve"),
    "egorov2006_m6": ("Egorov, Fedorov & Soudakov (2006)", "M = 6 · 2nd-mode growth band"),
    "cone_sivasubramanian_fasel_2015": ("Sivasubramanian & Fasel (2015)", "7° cone, M_e = 5.36 · 2nd-mode growth / N"),
    "balakumar_malik1992_branches": ("Balakumar & Malik (1992)", "M = 4.5 · discrete vs continuous spectrum"),
    "balakumar_malik1992_via_xirenfu": ("Balakumar & Malik (1992)", "M = 4.5 · spatial eigenvalue α"),
    "malik_case6": ("Malik (1990) Case 6", "M = 4.5 · spatial eigenvalue α"),
    "mack_fig10_6_M45": ("Mack (1984) Fig 10.6", "M = 4.5 · max 2nd-mode temporal growth"),
    "mack_fig10_6_M58": ("Mack (1984) Fig 10.6", "M = 5.8 · max 2nd-mode temporal growth"),
    "mack_fig10_6_M70": ("Mack (1984) Fig 10.6", "M = 7.0 · max 2nd-mode temporal growth"),
    "mack_fig10_6_M100": ("Mack (1984) Fig 10.6", "M = 10 · max 2nd-mode temporal growth"),
    "mack_fig10_3_m1p3": ("Mack (1984) Fig 10.3", "M = 1.3, ψ = 45° · max 1st-mode growth"),
    "mack_fig10_4_M45": ("Mack (1984) Fig 10.4", "M = 4.5 · max 1st-mode growth (oblique)"),
    "ozgen_m2": ("Özgen & Kırcalı (2008) Fig 3", "M = 2 · neutral curve"),
    "ozgen_m3": ("Özgen & Kırcalı (2008) Fig 3", "M = 3 · neutral curve"),
    "ozgen_m4": ("Özgen & Kırcalı (2008) Fig 3", "M = 4 · neutral curve (two lobes)"),
    "ozgen_m6": ("Özgen & Kırcalı (2008) Fig 3", "M = 6 · neutral curve (two lobes)"),
    "ozgen_m7": ("Özgen & Kırcalı (2008) Fig 3", "M = 7 · neutral curve"),
    "ozgen_m8": ("Özgen & Kırcalı (2008) Fig 3", "M = 8 · neutral curve"),
    "ozgen_m10": ("Özgen & Kırcalı (2008) Fig 3", "M = 10 · neutral curve"),
}


def fmt_re(c):
    if "unit_Re_per_m" in c:
        return f"unit Re = {c['unit_Re_per_m']/1e6:g}×10⁶ /m"
    if "Re_L_plate" in c:
        return f"Re_x = {c['Re_L_plate']/1e6:g}×10⁶ (R = √Re_x)"
    if "Re_l" in c:
        return f"R = {c['Re_l']:g}  (= √Re_x)"
    return None


def mach_of(c):
    for k in ("Ma", "Ma_edge", "Ma_freestream"):
        if k in c:
            return float(c[k])
    return None


def tw_over_te(c, ma):
    """Adiabatic recovery Tw/Te = 1 + r (gamma-1)/2 M^2, r = sqrt(Pr)."""
    wall = str(c.get("wall", "")).lower()
    if "adiabatic" not in wall or ma is None:
        return None
    pr = float(c.get("Pr", 0.72)); g = float(c.get("gamma", 1.4))
    return 1.0 + sqrt(pr) * 0.5 * (g - 1.0) * ma * ma


def conditions_rows(case_id, c):
    """Ordered (label, value) rows for the spec sheet."""
    ma = mach_of(c)
    rows = []
    rows.append(("Mach", f"{ma:g}" + (f"  (edge; M∞ = {c['Ma_freestream']:g})" if "Ma_edge" in c and "Ma_freestream" in c else "")))
    is_ozgen = case_id.startswith("ozgen")
    # gas / gamma
    gas = c.get("gas", "air")
    g = c.get("gamma", 1.4)
    rows.append(("Gas / γ", f"{gas} / {g:g}"))
    # wall + Tw/Te
    wall = c.get("wall", "adiabatic")
    twte = tw_over_te(c, ma)
    if "Tw/Te" in str(wall) or "Tw=" in str(wall):
        rows.append(("Wall BC", str(wall)))
    elif twte is not None:
        rows.append(("Wall BC", f"adiabatic (recovery Tw/Te ≈ {twte:.2f})"))
    else:
        rows.append(("Wall BC", str(wall)))
    # edge / stagnation temp
    if is_ozgen:
        rows.append(("Edge temp Te", "288 K"))
    elif "T_edge_K" in c:
        s = f"{c['T_edge_K']:.1f} K"
        if "T0_K" in c:
            s += f"  (T0 = {c['T0_K']:.0f} K)"
        rows.append(("Edge temp Te", s))
    elif "T0_K" in c:
        rows.append(("Stagn. temp T0", f"{c['T0_K']:.0f} K"))
    # Prandtl
    rows.append(("Prandtl Pr", f"{c.get('Pr', 0.72):g}"))
    # transport / viscosity
    if is_ozgen:
        rows.append(("Transport", "Özgen T-dependent μ, κ (Sutherland-type)"))
    else:
        t = c.get("transport") or c.get("viscosity")
        if t:
            rows.append(("Transport", str(t)))
    # Reynolds
    re = fmt_re(c)
    if re:
        rows.append(("Reynolds", re))
    # frequency / wavenumber
    if "F" in c:
        rows.append(("Reduced freq F", f"{c['F']:g}  (ω = R·F)"))
    elif "omega" in c:
        rows.append(("Frequency ω", f"{c['omega']:g}"))
    # length scale (Ozgen + L*-scale cases)
    ls = c.get("length_scale")
    if is_ozgen:
        rows.append(("Length scale", "L* = √(νₑ x / Uₑ),  R = √Re_x"))
    elif ls == "L_star":
        rows.append(("Length scale", "L* = √(νₑ x / Uₑ)"))
    # formulation + wave angle
    form = c.get("formulation", "temporal 2D" if is_ozgen else "")
    psi = c.get("psi_deg")
    if psi not in (None, 0, "0", 0.0) and "ψ" not in form and "psi" not in str(form).lower():
        form = f"{form}, ψ = {psi}°" if form else f"ψ = {psi}°"
    if form:
        rows.append(("Formulation", str(form)))
    return rows


def main():
    out = []
    for mode_dir in ("second_mode", "first_mode", "other"):
        d = VER / mode_dir
        if not d.is_dir():
            continue
        for vf in sorted(d.glob("*/verdict.json")):
            v = json.loads(vf.read_text(encoding="utf-8"))
            if v.get("verdict") not in GOOD:
                continue
            cid = v["case_id"]
            c = v.get("conditions", {})
            ov = (v.get("artifacts") or {}).get("overlay")
            ov_abs = str((REPO / ov)) if ov else None
            if not ov_abs or not os.path.exists(ov_abs):
                print(f"  WARN {cid}: overlay missing ({ov})")
                continue
            title, qtag = TITLES.get(cid, (v.get("source", cid), v.get("quantity", "")))
            out.append({
                "case_id": cid,
                "mode_dir": mode_dir,
                "category": v.get("category"),
                "title": title,
                "subtitle": qtag,
                "source": v.get("source"),
                "verdict": v.get("verdict"),
                "badge": VERDICT_BADGE.get(v.get("verdict"), ""),
                "agreement": _headline(v),
                "conditions": conditions_rows(cid, c),
                "overlay": ov_abs.replace("\\", "/"),
            })
    out.sort(key=lambda r: (MODE_ORDER.get(r["mode_dir"], 9), r["case_id"]))
    (HERE / "deck_data.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote deck_data.json: {len(out)} cases")
    for r in out:
        print(f"  [{r['mode_dir']:11s}] {r['case_id']:30s} {r['verdict']:10s} | {r['title']}")


if __name__ == "__main__":
    main()
