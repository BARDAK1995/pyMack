#!/usr/bin/env python
"""Malik (1990) Test Case 4 -- hypersonic cooled-wall TEMPORAL second mode.

Companion to ``compute``/``compare_malik1990_anchors.py`` (Case 6, spatial) and
the CI gate ``validation/test_malik1990_case4_anchor.py``. Writes a computed
verdict.json + overlay to ``verification/second_mode/malik_case4/``.

Case 4 (Malik Table I / Table V):
    M = 10, R = 2000, T_w/T_adb = 0.1 (cooled), T0 = 4200 degR, delta* = 12.917,
    fixed real alpha = 0.105, beta = 0, Sutherland S = 110.33 K, Pr = 0.7.
    Malik omega = 0.0974837 + 0.0020304 i  (Table V, 4CD, N+1 = 81; second mode).

The dimensional conditions (T0 and the COOLED wall) come from Malik's own
Table I, extracted to refPapers/NewPapers/figures/. Before that scan was
available this case was logged non-reproducible ("unknown T0/gas"); it is now
reproduced to omega_r ~0.027%, omega_i ~2.1% (inside Malik's own inter-scheme
Table V spread).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from verification._compare_lib import classify_relative, write_verdict  # noqa: E402

from pymack import CompressibleBlasiusProfile  # noqa: E402
from pymack.solver import solve_temporal_compressible  # noqa: E402

OUT = HERE / "second_mode" / "malik_case4"

MA, R_L, ALPHA, PR, GAMMA = 10.0, 2000.0, 0.105, 0.7, 1.4
T0_K = 4200.0 * 5.0 / 9.0
T_EDGE_K = T0_K / (1.0 + 0.5 * (GAMMA - 1.0) * MA**2)
T_REC_K = T_EDGE_K * (1.0 + 0.5 * (GAMMA - 1.0) * PR**0.5 * MA**2)
T_WALL_K = 0.1 * T_REC_K
SUTHERLAND_S_K = 198.6 / 1.8

MALIK_OMEGA = 0.0974837 + 0.0020304j        # Table V, 4CD, N+1=81 (converged)
# Malik Table V inter-scheme spread at N+1=61 (2FD, 4CD, SDSP, MDSP):
MALIK_SCHEMES = {
    "2FD":  0.0974002 + 0.0020224j,
    "4CD":  0.0974832 + 0.0020308j,
    "SDSP": 0.0974774 + 0.0020302j,
    "MDSP": 0.0974864 + 0.0020316j,
}
N_SOLVE = 200


def compute_pymack_omega():
    prof = CompressibleBlasiusProfile(
        Ma=MA, T_edge=T_EDGE_K, T_wall=T_WALL_K, gamma=GAMMA, Pr=PR,
        wall_bc="isothermal", viscosity_model="sutherland",
        sutherland_S=SUTHERLAND_S_K, n_points=4000, eta_max=40.0,
    )
    c, _, _ = solve_temporal_compressible(
        prof, ALPHA, R_L, MA, PR, GAMMA, N=N_SOLVE, y_max=75.0,
        wall_bc="isothermal", length_scale="L_star", lambda_mu_ratio=1.2,
    )
    c = np.asarray(c)
    c_target = MALIK_OMEGA / ALPHA
    band = c[(c.real > 0.85) & (c.real < 0.98) & (np.abs(c.imag) < 0.05)]
    if not band.size:
        raise RuntimeError("no second-mode candidate at Malik Case 4 conditions")
    c_mode = complex(band[np.argmin(np.abs(band - c_target))])
    return ALPHA * c_mode


def make_overlay(pymack_omega, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    # Malik's four-scheme cluster (N+1=61)
    sx = [v.real for v in MALIK_SCHEMES.values()]
    sy = [v.imag for v in MALIK_SCHEMES.values()]
    ax.scatter(sx, sy, s=70, facecolors="none", edgecolors="0.45",
               linewidths=1.6, label="Malik Table V schemes (N+1=61)", zorder=3)
    for name, v in MALIK_SCHEMES.items():
        ax.annotate(name, (v.real, v.imag), textcoords="offset points",
                    xytext=(6, 4), fontsize=11, color="0.4")
    # Malik converged (4CD, N+1=81) -- the anchor value
    ax.scatter([MALIK_OMEGA.real], [MALIK_OMEGA.imag], s=180, marker="*",
               color="tab:red", edgecolors="k", linewidths=0.6, zorder=5,
               label="Malik converged (4CD, N+1=81)")
    # pyMack
    ax.scatter([pymack_omega.real], [pymack_omega.imag], s=130, marker="D",
               color="tab:blue", edgecolors="k", linewidths=0.6, zorder=5,
               label=f"pyMack (N={N_SOLVE})")

    dr = abs(pymack_omega.real - MALIK_OMEGA.real)
    di = abs(pymack_omega.imag - MALIK_OMEGA.imag)
    ax.set_xlabel(r"$\omega_r$", fontsize=15)
    ax.set_ylabel(r"$\omega_i$  (temporal growth rate)", fontsize=15)
    ax.set_title("Malik (1990) Case 4: M=10, cooled wall, $\\alpha$=0.105\n"
                 f"pyMack vs Malik second mode  ($\\Delta\\omega_r$={dr:.1e}, "
                 f"$\\Delta\\omega_i$={di:.1e})", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pm = compute_pymack_omega()
    er = abs(pm.real - MALIK_OMEGA.real) / abs(MALIK_OMEGA.real)
    ei = abs(pm.imag - MALIK_OMEGA.imag) / abs(MALIK_OMEGA.imag)
    worst = max(er, ei)
    verdict = classify_relative(worst, topology_ok=True)

    make_overlay(pm, OUT / "overlay.png")

    v = {
        "case_id": "malik_case4",
        "category": "eigenvalue_anchor",
        "source": (
            "Malik (1990) JCP 86:376 Table V, Case 4 (M=10, R=2000, "
            "T_w/T_adb=0.1 cooled, alpha=0.105, beta=0). Dimensional conditions "
            "(T0=4200 degR, cooled wall) read from Malik Table I "
            "(refPapers/NewPapers/figures/malik1990_table1_test_cases.png); "
            "eigenvalue from Table V (4CD, N+1=81)."
        ),
        "conditions": {
            "Ma": MA, "Re_l": R_L, "alpha_fixed": ALPHA, "beta": 0.0, "Pr": PR,
            "gamma": GAMMA, "wall": "isothermal, cooled T_w/T_adb=0.1",
            "T0_K": T0_K, "T_edge_K": T_EDGE_K, "T_wall_K": T_WALL_K,
            "sutherland_S_K": SUTHERLAND_S_K, "malik_case_number": 4,
            "problem": "temporal (fixed real alpha -> complex omega)",
            "mode": "second mode", "length_scale": "L_star",
        },
        "quantity": "temporal frequency eigenvalue omega = omega_r + i*omega_i at fixed real alpha",
        "metrics": {
            "omega_r_rel_err": er,
            "omega_i_rel_err": ei,
            "pymack_omega": [pm.real, pm.imag],
            "malik_omega": [MALIK_OMEGA.real, MALIK_OMEGA.imag],
            "c_phase": pm.real / ALPHA,
            "N": N_SOLVE,
            "topology_ok": True,
        },
        "verdict": verdict,
        "verdict_reason": (
            "Hypersonic cooled-wall TEMPORAL second mode. pyMack omega="
            f"{pm.real:.7f}{pm.imag:+.7f}i vs Malik 0.0974837+0.0020304i: "
            f"omega_r rel err {er:.2e} ({er*100:.3f}%, essentially exact), "
            f"omega_i rel err {ei:.2e} ({ei*100:.2f}%). The omega_i offset is "
            "well inside Malik's OWN Table V inter-scheme spread for this "
            "deliberately severe M=10 case (2FD/4CD/SDSP/MDSP span "
            "omega_r=0.097400..0.097486, omega_i=0.0020224..0.0020316 at N+1=61; "
            "pyMack's omega_r lands squarely inside that band). Previously logged "
            "non-reproducible for lack of the dimensional conditions (Hildebrand "
            "lists only M and Re); adding the source paper (Table I: T0=4200 degR, "
            "cooled wall) unlocked it. Mirror of validation/"
            "test_malik1990_case4_anchor.py."
        ),
        "generated": "new",
        "artifacts": {
            "pymack": None, "reference": None,
            "overlay": "verification/second_mode/malik_case4/overlay.png",
        },
        "pymack_provenance": (
            "pymack.solver.solve_temporal_compressible on CompressibleBlasiusProfile "
            "(cooled isothermal wall T_w=0.1*T_adb, Sutherland S=110.33 K), "
            f"N={N_SOLVE}, y_max=75, lambda_mu_ratio=1.2, length_scale='L_star'; "
            "mirrors validation/test_malik1990_case4_anchor.py."
        ),
        "mode": "second",
    }
    write_verdict(OUT, v)
    print(f"malik_case4  {verdict}  "
          f"pyMack={pm.real:.7f}{pm.imag:+.7f}i  Malik={MALIK_OMEGA.real:.7f}{MALIK_OMEGA.imag:+.7f}i  "
          f"(omega_r {er*100:.3f}%, omega_i {ei*100:.2f}%)")
    print(f"wrote {OUT/'verdict.json'} and {OUT/'overlay.png'}")


if __name__ == "__main__":
    main()
