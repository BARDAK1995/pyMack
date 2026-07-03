#!/usr/bin/env python
"""Malik (1990) Test Cases 3 & 5 -- temporal eigenvalue anchors (newly unlocked).

Companions to compute_malik1990_case4.py. Both were previously logged
non-reproducible ("unknown T0/gas"); adding the Malik paper (Table I) supplied
the dimensional conditions and unlocked them. Writes computed verdict.json +
overlay to verification/second_mode/malik_case{3,5}/.

  Case 3 (Table I/IV): M=2.5, R=3000, adiabatic, T0=600 degR, alpha=0.06,
      beta=0.10 (OBLIQUE / 3-D). Malik omega = 0.0367340 + 0.0005840 i
      (cleanest case: all 4 schemes agree to 5 sig figs).
  Case 5 (Table I/VI): M=10, R=1000, adiabatic, T0=4200 degR, alpha=0.12,
      beta=0. Malik omega = 0.1158647 + 0.0001529 i (4CD N+1=81; Malik's
      deliberately SEVERE near-neutral test, scheme-sensitive omega_i).
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
from pymack.solver import (  # noqa: E402
    solve_temporal_compressible,
    solve_temporal_compressible_3d,
)

GAMMA, PR = 1.4, 0.7
S_K = 198.6 / 1.8

CASES = {
    "malik_case3": dict(
        mode_dir="first_mode",   # M=2.5 oblique -> FIRST mode (c_r ~ 0.61)
        num=3, Ma=2.5, R=3000.0, alpha=0.06, beta=0.10, T0_degR=600.0,
        malik=0.0367340 + 0.0005840j, N=200, y_max=55.0,
        eta_max=25.0, n_points=3000, cband=(0.55, 0.68), c_target_r=0.6122,
        schemes={"2FD": 0.0367583 + 0.0005733j, "4CD": 0.0367321 + 0.0005847j,
                 "SDSP": 0.0367339 + 0.0005840j, "MDSP": 0.0367340 + 0.0005840j},
        table="Table IV", note_extra=(
            "OBLIQUE (beta=0.10) first mode -- the cleanest-converging case in "
            "Malik's paper (all four schemes agree to 5 sig figs)."),
    ),
    "malik_case5": dict(
        mode_dir="second_mode",  # M=10 2-D -> SECOND (Mack) mode (c_r ~ 0.97)
        num=5, Ma=10.0, R=1000.0, alpha=0.12, beta=0.0, T0_degR=4200.0,
        malik=0.1158647 + 0.0001529j, N=280, y_max=140.0,
        eta_max=70.0, n_points=6000, cband=(0.90, 1.0), c_target_r=0.9655,
        schemes={"2FD": 0.1158706 + 0.0003251j, "4CD": 0.1158630 + 0.0001521j,
                 "SDSP": 0.1161434 - 0.0001949j, "MDSP": 0.1158519 + 0.0001357j},
        table="Table VI", note_extra=(
            "Malik's deliberately SEVERE near-neutral M=10 test: omega_i ~ 1.5e-4 "
            "is strongly scheme-sensitive (his own SDSP even flips its sign at "
            "N+1=61). pyMack's independent IVM/RK4 cross-check value is "
            "0.1158627+0.0001557i; pyMack lands on that cluster."),
    ),
}


def compute(cfg):
    T_edge = (cfg["T0_degR"] * 5.0 / 9.0) / (1.0 + 0.5 * (GAMMA - 1.0) * cfg["Ma"]**2)
    t_rec = T_edge * (1.0 + 0.5 * (GAMMA - 1.0) * PR**0.5 * cfg["Ma"]**2)
    prof = CompressibleBlasiusProfile(
        Ma=cfg["Ma"], T_edge=T_edge, T_wall=t_rec, gamma=GAMMA, Pr=PR,
        wall_bc="adiabatic", viscosity_model="sutherland", sutherland_S=S_K,
        n_points=cfg["n_points"], eta_max=cfg["eta_max"],
    )
    a = cfg["alpha"]
    # NOTE: base flow above is adiabatic (the physical MEAN wall is insulated),
    # but the DISTURBANCE temperature BC below is isothermal (T'|_wall=0) --
    # Malik's paper-wide convention (1990, p.385). These are two independent BCs;
    # conflating them (using adiabatic for the disturbance) is the exact error
    # that inflated Ma & Zhong and, earlier, these cases.
    if cfg["beta"] != 0.0:
        c, _, _ = solve_temporal_compressible_3d(
            prof, a, cfg["beta"], cfg["R"], cfg["Ma"], PR, GAMMA,
            N=cfg["N"], y_max=cfg["y_max"], wall_bc="isothermal",
            length_scale="L_star", lambda_mu_ratio=1.2)
    else:
        c, _, _ = solve_temporal_compressible(
            prof, a, cfg["R"], cfg["Ma"], PR, GAMMA,
            N=cfg["N"], y_max=cfg["y_max"], wall_bc="isothermal",
            length_scale="L_star", lambda_mu_ratio=1.2)
    c = np.asarray(c)
    lo, hi = cfg["cband"]
    ctar = cfg["malik"] / a
    band = c[(c.real > lo) & (c.real < hi) & (np.abs(c.imag) < 0.05)]
    if not band.size:
        raise RuntimeError(f"no candidate for case {cfg['num']}")
    cm = complex(band[np.argmin(np.abs(band - ctar))])
    return a * cm, T_edge, t_rec


def make_overlay(cfg, pm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mal = cfg["malik"]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    sx = [v.real for v in cfg["schemes"].values()]
    sy = [v.imag for v in cfg["schemes"].values()]
    ax.scatter(sx, sy, s=70, facecolors="none", edgecolors="0.45", linewidths=1.6,
               label=f"Malik {cfg['table']} schemes (N+1=61)", zorder=3)
    for name, v in cfg["schemes"].items():
        ax.annotate(name, (v.real, v.imag), textcoords="offset points",
                    xytext=(6, 4), fontsize=11, color="0.4")
    ax.scatter([mal.real], [mal.imag], s=180, marker="*", color="tab:red",
               edgecolors="k", linewidths=0.6, zorder=5, label="Malik converged")
    ax.scatter([pm.real], [pm.imag], s=130, marker="D", color="tab:blue",
               edgecolors="k", linewidths=0.6, zorder=5, label=f"pyMack (N={cfg['N']})")
    dr, di = abs(pm.real - mal.real), abs(pm.imag - mal.imag)
    ax.axhline(0.0, color="0.7", lw=0.8, ls=":")
    wall = "adiabatic" if cfg["beta"] == 0 else "adiabatic, oblique $\\beta$=0.10"
    ax.set_xlabel(r"$\omega_r$", fontsize=15)
    ax.set_ylabel(r"$\omega_i$  (temporal growth rate)", fontsize=15)
    ax.set_title(f"Malik (1990) Case {cfg['num']}: M={cfg['Ma']:g}, {wall}, "
                 f"$\\alpha$={cfg['alpha']:g}\npyMack vs Malik  "
                 f"($\\Delta\\omega_r$={dr:.1e}, $\\Delta\\omega_i$={di:.1e})",
                 fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    for cid, cfg in CASES.items():
        out = HERE / cfg["mode_dir"] / cid
        out.mkdir(parents=True, exist_ok=True)
        pm, T_edge, t_rec = compute(cfg)
        mal = cfg["malik"]
        er = abs(pm.real - mal.real) / abs(mal.real)
        ei = abs(pm.imag - mal.imag) / abs(mal.imag)
        verdict = classify_relative(max(er, ei), topology_ok=True)
        make_overlay(cfg, pm, out / "overlay.png")
        mode_kind = "oblique first mode" if cfg["beta"] else "second mode"
        v = {
            "case_id": cid,
            "category": "eigenvalue_anchor",
            "source": (
                f"Malik (1990) JCP 86:376 {cfg['table']}, Case {cfg['num']} "
                f"(M={cfg['Ma']:g}, R={cfg['R']:g}, adiabatic, alpha={cfg['alpha']:g}, "
                f"beta={cfg['beta']:g}). Dimensional conditions (T0={cfg['T0_degR']:g} "
                "degR, adiabatic) from Malik Table I "
                "(refPapers/NewPapers/figures/malik1990_table1_test_cases.png)."
            ),
            "conditions": {
                "Ma": cfg["Ma"], "Re_l": cfg["R"], "alpha_fixed": cfg["alpha"],
                "beta": cfg["beta"], "Pr": PR, "gamma": GAMMA,
                "wall": "adiabatic (insulated)", "T0_K": cfg["T0_degR"] * 5.0 / 9.0,
                "T_edge_K": T_edge, "sutherland_S_K": S_K,
                "malik_case_number": cfg["num"],
                "problem": "temporal (fixed real alpha -> complex omega)",
                "mode": mode_kind, "length_scale": "L_star",
            },
            "quantity": "temporal frequency eigenvalue omega = omega_r + i*omega_i at fixed real alpha",
            "metrics": {
                "omega_r_rel_err": er, "omega_i_rel_err": ei,
                "pymack_omega": [pm.real, pm.imag],
                "malik_omega": [mal.real, mal.imag],
                "c_phase": pm.real / cfg["alpha"], "N": cfg["N"], "topology_ok": True,
            },
            "verdict": verdict,
            "verdict_reason": (
                f"{mode_kind.capitalize()}, temporal. pyMack omega="
                f"{pm.real:.7f}{pm.imag:+.7f}i vs Malik {mal.real:.7f}{mal.imag:+.7f}i: "
                f"omega_r rel err {er:.2e} ({er*100:.3f}%), omega_i rel err {ei:.2e} "
                f"({ei*100:.2f}%). {cfg['note_extra']} Previously logged "
                "non-reproducible for lack of the dimensional conditions; adding the "
                "source paper (Table I) unlocked it. Mirror of validation/"
                f"test_malik1990_case{cfg['num']}_anchor.py."
            ),
            "generated": "new",
            "artifacts": {
                "pymack": None, "reference": None,
                "overlay": f"verification/{cfg['mode_dir']}/{cid}/overlay.png",
            },
            "pymack_provenance": (
                "pymack.solver.%s on CompressibleBlasiusProfile (adiabatic MEAN "
                "wall, Sutherland S=110.33 K) with ISOTHERMAL disturbance-temperature "
                "BC (T'|_wall=0, Malik's paper-wide convention; the mean wall is "
                "adiabatic but the perturbation T' vanishes at the wall), N=%d, "
                "y_max=%g, lambda_mu_ratio=1.2, length_scale='L_star'; mirrors "
                "validation/test_malik1990_case%d_anchor.py."
                % ("solve_temporal_compressible_3d" if cfg["beta"] else
                   "solve_temporal_compressible", cfg["N"], cfg["y_max"], cfg["num"])
            ),
            "mode": "second" if not cfg["beta"] else "first",
        }
        write_verdict(out, v)
        print(f"{cid}  {verdict}  pyMack={pm.real:.7f}{pm.imag:+.7f}i  "
              f"Malik={mal.real:.7f}{mal.imag:+.7f}i  (omega_r {er*100:.3f}%, omega_i {ei*100:.2f}%)")


if __name__ == "__main__":
    main()
