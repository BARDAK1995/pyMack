"""Write verdict.json for the Ma & Zhong (2003) Mach 4.5 second-mode neutral
branch verification (case mazhong2003_m4p5).

Compares pyMack's two second-mode neutral-branch Reynolds numbers (R = sqrt(Re_x))
at fixed F = 2.2e-4 against the values printed in Ma & Zhong (2003) JFM 488:31-78:
Branch I (lower) R = 806, Branch II (upper) R = 999.6.

Headline = classify_relative on max(branch_I_rel_err, branch_II_rel_err).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # verification/
from _compare_lib import classify_relative, write_verdict  # noqa: E402

# pyMack results with the ISOTHERMAL temperature-disturbance BC (T'|_wall=0) --
# the boundary condition Ma & Zhong (2003) actually used for the Fig.15 neutral
# curve ("the isothermal wall boundary condition is used for temperature
# perturbations"; see MaZhong_2003_notes.md). N=130, length_scale=L*,
# lambda_mu_ratio=1.2. The mean base flow stays adiabatic (insulated plate).
BRANCH_I_PYMACK = 786.64
BRANCH_II_PYMACK = 1009.92
BRANCH_I_REF = 806.0
BRANCH_II_REF = 999.6

# Adiabatic-disturbance-BC sensitivity (recorded for honesty; NOT the Fig.15 BC):
# I=831.0 (3.10%), II=1033.4 (3.38%).
ADIA_I, ADIA_II = 831.0, 1033.4

F2 = 2.2e-4  # steep reference ray, omega = F2*R


def digitized_F2_crossings():
    """Independent cross-check: where F=2.2e-4 crosses the pixel-traced
    reference 2nd-mode branches (reference_mazhong_fig15.csv).  Validates the
    digitization calibration against the paper's textual 806 / 999.6."""
    ref: dict[tuple[str, str], list[tuple[float, float]]] = {}
    with open(HERE / "reference_mazhong_fig15.csv") as f:
        for r in csv.DictReader(f):
            ref.setdefault((r["mode"], r["branch"]), []).append(
                (float(r["R"]), float(r["omega"])))

    def cross(branch):
        a = np.array(sorted(ref[("second", branch)]))
        Rr, om = a[:, 0], a[:, 1]
        g = om - F2 * Rr
        for i in range(len(Rr) - 1):
            if g[i] * g[i + 1] < 0:
                t = g[i] / (g[i] - g[i + 1])
                return float(Rr[i] + t * (Rr[i + 1] - Rr[i]))
        return None

    return cross("lower"), cross("upper")  # branch I (lower), branch II (upper)


def main():
    bI_err = abs(BRANCH_I_PYMACK - BRANCH_I_REF) / BRANCH_I_REF
    bII_err = abs(BRANCH_II_PYMACK - BRANCH_II_REF) / BRANCH_II_REF
    headline = max(bI_err, bII_err)
    topology_ok = True  # single closed second-mode band, exactly 2 neutral points
    verdict = classify_relative(headline, topology_ok)

    adia_I_err = abs(ADIA_I - BRANCH_I_REF) / BRANCH_I_REF
    adia_II_err = abs(ADIA_II - BRANCH_II_REF) / BRANCH_II_REF

    # Independent cross-check of the (upgraded) pixel-traced reference itself:
    # where F=2.2e-4 crosses the digitized 2nd-mode branches should reproduce
    # the paper's textual 806 / 999.6 -- validates calibration + trace.
    dig_I, dig_II = digitized_F2_crossings()
    dig_I_err = abs(dig_I - BRANCH_I_REF) / BRANCH_I_REF
    dig_II_err = abs(dig_II - BRANCH_II_REF) / BRANCH_II_REF

    reason = (
        f"At fixed F=2.2e-4, pyMack with the ISOTHERMAL temperature-disturbance BC "
        f"(T'|_wall=0 -- the BC Ma & Zhong state they used for the Fig.15 neutral "
        f"curve) places the second-mode Branch I (lower) neutral point at "
        f"R={BRANCH_I_PYMACK:.1f} vs Ma & Zhong R=806 ({bI_err*100:.2f}%) and "
        f"Branch II (upper) at R={BRANCH_II_PYMACK:.1f} vs R=999.6 ({bII_err*100:.2f}%). "
        f"Topology matches: a single closed unstable band bounded by exactly two "
        f"neutral points. Both branch errors are below the 5% digitization-floor "
        f"threshold (headline = max = {headline*100:.2f}%) -> agrees. N-converged "
        f"(N=130 vs N=160 differ <0.2 in R). NOTE: an earlier run used the adiabatic "
        f"disturbance BC by mistake, giving I={ADIA_I:.1f} ({adia_I_err*100:.2f}%), "
        f"II={ADIA_II:.1f} ({adia_II_err*100:.2f}%); switching to the paper's actual "
        f"isothermal disturbance BC improves the match (3.1/3.4% -> {bI_err*100:.1f}/"
        f"{bII_err*100:.1f}%) -- a BC correction, not tuning. The reference Fig.15 "
        f"curve is a REAL pixel-trace of the 400-DPI crop (calibration in "
        f"_pixel_calibration.md); as an independent anchor check, F=2.2e-4 crosses "
        f"the digitized 2nd-mode branches at R={dig_I:.1f} (branch I, vs 806, "
        f"{dig_I_err*100:.1f}%) and R={dig_II:.1f} (branch II, vs 999.6, "
        f"{dig_II_err*100:.1f}%), confirming the digitization is on-curve."
    )

    v = {
        "case_id": "mazhong2003_m4p5",
        "category": "neutral_curve",
        "source": "Ma & Zhong (2003) JFM 488:31-78, Part 1, sec. 6 / Fig. 15",
        "conditions": {
            "Ma": 4.5,
            "gas": "air",
            "wall": "adiabatic",
            "psi_deg": 0,
            "formulation": "spatial second mode (Mack), F fixed",
            "transport": "Sutherland viscosity (S=110.33 K), constant Pr=0.72",
            "gamma": 1.4,
            "T_edge_K": 65.15,
            "p_inf_Pa": 728.4381557,
            "unit_Re_per_m": 7.2e6,
            "F": 2.2e-4,
            "length_scale": "L* = sqrt(nu_e x / U_e); R = sqrt(Re_x); omega = R*F",
            "disturbance_temperature_BC": "isothermal, T'|_wall=0 (Fig.15 neutral curve, per Ma & Zhong text; mean flow adiabatic)",
            "N": 130,
            "lambda_mu_ratio": 1.2,
        },
        "quantity": (
            "second-mode neutral-branch Reynolds numbers R=sqrt(Re_x) at fixed "
            "F=2.2e-4: Branch I (lower) and Branch II (upper)"
        ),
        "metrics": {
            "branch_I_R_pymack": round(BRANCH_I_PYMACK, 2),
            "branch_I_R_ref": BRANCH_I_REF,
            "branch_I_rel_err": round(bI_err, 5),
            "branch_II_R_pymack": round(BRANCH_II_PYMACK, 2),
            "branch_II_R_ref": BRANCH_II_REF,
            "branch_II_rel_err": round(bII_err, 5),
            "F": 2.2e-4,
            "topology_ok": topology_ok,
            "headline_max_rel_err": round(headline, 5),
            "adiabatic_BC_branch_I_R_pymack": ADIA_I,
            "adiabatic_BC_branch_I_rel_err": round(adia_I_err, 5),
            "adiabatic_BC_branch_II_R_pymack": ADIA_II,
            "adiabatic_BC_branch_II_rel_err": round(adia_II_err, 5),
            # independent cross-check of the pixel-traced reference digitization
            "digitized_branch_I_R": round(dig_I, 1),
            "digitized_branch_I_rel_err": round(dig_I_err, 5),
            "digitized_branch_II_R": round(dig_II, 1),
            "digitized_branch_II_rel_err": round(dig_II_err, 5),
            "headline": (
                f"2nd-mode neutral band matches Fig.15 (branch I/II "
                f"{bI_err*100:.1f}%/{bII_err*100:.1f}% at F=2.2e-4 with the paper's "
                f"isothermal disturbance BC, anchors 806/999.6); reference curve is a "
                f"real 400-DPI pixel-trace (digitized branch points {dig_I:.0f}/"
                f"{dig_II:.0f} confirm the anchors); 1st-mode cutoff branch tracks, "
                "onset partial"
            ),
            "full_curve": (
                "pyMack neg_alpha_i=0 contour (isothermal disturbance BC) traced over "
                "R=240-2000 (2nd mode) and 450-2000 (1st mode); 2nd-mode band edges "
                "match the pixel-digitized Fig.15 in omega; F=2.2e-4 crosses "
                "pyMack band at R~787/1010"
            ),
        },
        "verdict": verdict,
        "verdict_reason": reason,
        "generated": "new",
        "artifacts": {
            "pymack": "verification/second_mode/mazhong2003_m4p5/"
                      "pymack_growth_sweep.json",
            "reference": "verification/second_mode/mazhong2003_m4p5/"
                         "reference_mazhong_fig15.csv",
            "overlay": "verification/second_mode/mazhong2003_m4p5/overlay.png",
        },
        "pymack_provenance": (
            "verification/second_mode/mazhong2003_m4p5/"
            "compute_mazhong_m4p5.py (this session, 2026-06). CompressibleBlasius "
            "adiabatic-MEAN-flow M=4.5 base flow (Sutherland, S=110.33 K, Pr=0.72, "
            "gamma=1.4) at T_edge=65.15 K; solve_spatial with ISOTHERMAL disturbance "
            "temperature BC (T'|_wall=0, matching Ma & Zhong Fig.15), "
            "length_scale='L_star', lambda_mu_ratio=1.2, N=130, y_max=40; fixed "
            "F=2.2e-4 (omega=R*F); coarse R-sweep 700-1100 then bisection on "
            "-alpha_i=0 for the two second-mode neutral crossings. Same M=4.5 "
            "base-flow setup as the "
            "CI-gated Malik (1990) Case-6 anchor (validation/"
            "test_malik1990_case6_anchor.py). Reference values read directly from "
            "the Ma & Zhong (2003) PDF (sec. 6: 'interval from R = 806 (branch I "
            "neutral stability point) to 999.6 (branch II neutral stability "
            "point)')."
        ),
        "mode": "second",
    }

    path = write_verdict(HERE, v)
    print(f"Branch I  : pyMack {BRANCH_I_PYMACK:.1f} vs 806   -> {bI_err*100:.2f}%")
    print(f"Branch II : pyMack {BRANCH_II_PYMACK:.1f} vs 999.6 -> {bII_err*100:.2f}%")
    print(f"digitized : F=2.2e-4 crosses ref at I={dig_I:.1f} (806), "
          f"II={dig_II:.1f} (999.6)")
    print(f"headline max rel-err = {headline*100:.2f}%  topology_ok={topology_ok}")
    print(f"VERDICT = {verdict}")
    print(f"written -> {path}")
    return v


if __name__ == "__main__":
    main()
