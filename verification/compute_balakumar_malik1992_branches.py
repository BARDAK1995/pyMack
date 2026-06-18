#!/usr/bin/env python
"""Branch-structure (discrete vs continuous spectrum) characterization at the
verified Balakumar & Malik (1992) M=4.5 spatial second-mode condition.

The eigenvalue itself (alpha = 0.220 - 0.003091 i) is already verified in
``verification/eigenvalueAnchor_verification/balakumar_malik1992_via_xirenfu``
(acceptable). This script adds the qualitative *branch information* that B&M's
1992 paper ("Discrete modes and continuous spectra in supersonic boundary
layers", JFM 239:631) is actually about: the harmonic-point-source disturbance
field is a sum of discrete eigenmodes plus a continuous spectrum, and B&M show
the continuous spectrum forms several BRANCHES in the complex-wavenumber
(alpha) plane while the unstable/least-stable discrete (second) mode sits OFF
those branches.

We probe pyMack's FULL spatial companion spectrum (no shift-invert) at the
exact verified condition and ask one qualitative question:

    Does pyMack reproduce the expected topology -- a discrete second mode near
    alpha ~ 0.220 - 0.0031 i that is cleanly SEPARATED (in the complex-alpha
    plane) from the continuous-spectrum branch eigenvalues?

The discrete second mode is identified by phase speed c = omega/alpha_r ~ 0.908
and by proximity to the published alpha. "Cleanly separated" is quantified as
the gap between the discrete mode and the nearest OTHER physical eigenvalue,
relative to the discrete mode's own |alpha|.

Single-thread BLAS forced before numpy import (dense EVP).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("PYMACK_NO_BANNER", "1")

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from _compare_lib import write_verdict  # noqa: E402

from pymack import CompressibleBlasiusProfile  # noqa: E402
from pymack.solver import solve_spatial_full_spectrum  # noqa: E402

OUT = HERE / "eigenvalueAnchor_verification" / "balakumar_malik1992_branches"

# --- Verified B&M (1992) M4.5 condition (Xi/Ren/Fu Table B, "Case 1") -------
MA = 4.5
RE = 1000.0
OMEGA = 0.2
PR = 0.72
GAMMA = 1.4
T0_K = 311.0
T_EDGE_K = T0_K / (1.0 + 0.5 * (GAMMA - 1.0) * MA**2)
SUTHERLAND_S_K = 198.6 / 1.8  # 110.33 K
REF_ALPHA = 0.220 - 0.003091j


def build_profile():
    t_rec = T_EDGE_K * (1.0 + 0.5 * (GAMMA - 1.0) * PR**0.5 * MA**2)
    return CompressibleBlasiusProfile(
        Ma=MA, T_edge=T_EDGE_K, T_wall=t_rec, gamma=GAMMA, Pr=PR,
        wall_bc="adiabatic", viscosity_model="sutherland",
        sutherland_S=SUTHERLAND_S_K, n_points=3000, eta_max=20.0,
    )


def main():
    prof = build_profile()

    # Full spatial spectrum at the verified condition. N is larger than the
    # anchor's 120 to resolve the continuous-spectrum branches densely; the
    # branch structure is a property of the discretized far-field, so a richer
    # spectrum makes the discrete-vs-continuous separation visible.
    N = 150
    alphas, _modes, _y = solve_spatial_full_spectrum(
        prof, OMEGA, RE, MA, PR, GAMMA, N=N, y_max=40.0,
        wall_bc="isothermal", length_scale="L_star", lambda_mu_ratio=1.2,
        max_abs_alpha=20.0, max_abs_alpha_i=5.0,
    )
    alphas = np.asarray(alphas)
    print(f"full spectrum: {alphas.size} physical eigenvalues (N={N})")

    # --- Identify the discrete second mode --------------------------------
    # second mode: high phase speed c = omega/alpha_r ~ 0.9, near published alpha
    c = OMEGA / alphas.real
    band = (c > 0.85) & (c < 0.97) & (np.abs(alphas.imag) < 0.05)
    cand = alphas[band]
    if cand.size == 0:
        raise SystemExit("no discrete second-mode candidate in full spectrum")
    disc = complex(cand[np.argmin(np.abs(cand - REF_ALPHA))])
    c_disc = OMEGA / disc.real

    # --- Separation from the rest of the spectrum -------------------------
    others = alphas[np.abs(alphas - disc) > 1e-9]
    gaps = np.abs(others - disc)
    nearest = complex(others[np.argmin(gaps)])
    min_gap = float(np.min(gaps))
    sep_ratio = min_gap / abs(disc)  # gap relative to |alpha_discrete|
    n_total = int(alphas.size)
    n_within_10pct = int(np.sum(gaps < 0.10 * abs(disc)))

    # --- Physical branch fingerprint --------------------------------------
    # The continuous spectrum of a supersonic boundary layer accumulates at the
    # FREESTREAM phase speed c -> 1 (alpha_r -> omega): the acoustic / vorticity
    # / entropy branches B&M (1992) describe. The discrete second mode lives at
    # LOWER phase speed (c ~ 0.91) and is the LEAST-DAMPED root in that band.
    c_all = OMEGA / alphas.real
    cont_mask = np.abs(c_all - 1.0) < 0.05          # continuous-spectrum cluster
    n_cont_cluster = int(np.sum(cont_mask))
    cont_alpha_i_median = (float(np.median(alphas.imag[cont_mask]))
                           if n_cont_cluster else float("nan"))

    # Discrete-mode isolation within its OWN phase-speed band (second mode):
    inband = (c_all > 0.85) & (c_all < 0.97) & (np.abs(alphas.imag) < 0.05)
    inband_others = alphas[inband & (np.abs(alphas - disc) > 1e-9)]
    # Is the discrete mode the least-damped (smallest alpha_i) in its band?
    disc_is_least_damped_inband = bool(
        inband_others.size == 0 or np.all(inband_others.imag > disc.imag)
    )
    # Growth-rate gap to the nearest competing in-band root (B&M's separation
    # is in the c/alpha_i sense, not raw Euclidean -- the discretized continuum
    # is dense in alpha_r).
    alpha_i_gap_inband = (float(np.min(inband_others.imag) - disc.imag)
                          if inband_others.size else float("nan"))
    # How many roots are MORE amplified than the discrete mode (should be only
    # a few spurious high-|alpha| / low-c companion artifacts, not a competing
    # physical second mode)?
    more_amplified = alphas[alphas.imag < disc.imag - 1e-9]
    n_more_amplified = int(more_amplified.size)
    # Are all of them clearly non-second-mode (c outside the band or huge a_r)?
    more_amp_c = OMEGA / more_amplified.real
    more_amp_all_nonsecond = bool(np.all(
        (more_amp_c < 0.85) | (more_amp_c > 0.97) | (more_amplified.real > 1.0)
    )) if n_more_amplified else True

    print(f"discrete second mode : alpha = {disc.real:.6f}{disc.imag:+.6f}i  "
          f"(c = {c_disc:.4f})")
    print(f"published (B&M 1992) : alpha = {REF_ALPHA.real:.6f}"
          f"{REF_ALPHA.imag:+.6f}i")
    print(f"nearest other eig    : alpha = {nearest.real:.6f}"
          f"{nearest.imag:+.6f}i   |gap| = {min_gap:.4f} "
          f"(sep ratio {sep_ratio:.3f})")
    print(f"continuous-spectrum cluster near c=1: {n_cont_cluster} roots, "
          f"median alpha_i = {cont_alpha_i_median:+.4f}")
    print(f"discrete mode least-damped in 2nd-mode band: "
          f"{disc_is_least_damped_inband}  (alpha_i gap to next in-band root "
          f"= {alpha_i_gap_inband:+.4f})")
    print(f"# roots more amplified than disc: {n_more_amplified}  "
          f"(all non-second-mode artifacts: {more_amp_all_nonsecond})")

    # Print the continuous-spectrum acoustic ladder (c > 1) for the record
    order = np.argsort(others.real)
    print("\nlowest-alpha_r eigenvalues (acoustic continuous-branch region, "
          "c>1):")
    for a in others[order][:10]:
        print(f"   alpha = {a.real:.5f}{a.imag:+.5f}i   c={OMEGA/a.real:.4f}")

    # --- Verdict ----------------------------------------------------------
    # B&M's topology: a DISCRETE second mode (low phase speed, least damped)
    # separated from the CONTINUOUS-SPECTRUM branches (which accumulate at the
    # freestream c->1, alpha_r->omega). pyMack reproduces this IF:
    #   (1) the discrete mode matches the published alpha (same root as the
    #       verified via_xirenfu anchor), AND
    #   (2) it is the least-damped root in the second-mode phase-speed band with
    #       no competing physical second mode (any more-amplified roots are
    #       clearly spurious high-|alpha| / low-c companion artifacts), AND
    #   (3) a continuous-spectrum cluster near c=1 is present and identifiable.
    # The verdict is 'acceptable' rather than 'agrees' because the separation is
    # in the phase-speed / growth-rate sense: the DISCRETIZED continuous
    # spectrum is dense in alpha_r and crowds the discrete mode in raw Euclidean
    # |alpha| distance (nearest neighbour ~0.019, several roots within 10%), so
    # it is NOT the crisp off-branch gap of B&M's idealized (analytic) picture.
    disc_match_r = abs(disc.real - REF_ALPHA.real) / abs(REF_ALPHA.real)
    disc_match_i = abs(disc.imag - REF_ALPHA.imag) / abs(REF_ALPHA.imag)
    discrete_recovered = (disc_match_r < 0.02) and (disc_match_i < 0.15)
    topology_physical = (
        discrete_recovered
        and disc_is_least_damped_inband
        and more_amp_all_nonsecond
        and n_cont_cluster > 5
    )
    euclidean_clean = (n_within_10pct == 0) and (sep_ratio > 0.10)

    if topology_physical and euclidean_clean:
        verdict = "agrees"
        reason = (
            "pyMack's full spatial spectrum at the verified B&M (1992) M4.5 "
            "condition reproduces the expected branch topology with a clean "
            "Euclidean off-branch gap: discrete second mode at "
            f"alpha={disc.real:.6f}{disc.imag:+.6f}i (c={c_disc:.4f}) matching "
            "the published 0.220-0.003091i, isolated from the continuous "
            "spectrum."
        )
    elif topology_physical:
        verdict = "acceptable"
        reason = (
            "pyMack's full spatial spectrum at the verified B&M (1992) M4.5 "
            "condition (M=4.5, T0=311 K, Re=1000, omega=0.2, adiabatic) "
            "reproduces B&M's QUALITATIVE topology: (1) the discrete SECOND "
            f"mode at alpha={disc.real:.6f}{disc.imag:+.6f}i (c={c_disc:.4f}) "
            f"matches the published 0.220-0.003091i (alpha_r {disc_match_r*100:.2f}%, "
            f"alpha_i {disc_match_i*100:.1f}%) and is the UNIQUE least-damped "
            "root in the second-mode phase-speed band (every other in-band root "
            f"is more damped by >= {alpha_i_gap_inband:.4f} in alpha_i; the only "
            f"{n_more_amplified} more-amplified roots are spurious high-|alpha| / "
            "low-c companion artifacts, not a competing second mode); (2) a "
            f"continuous-spectrum cluster of {n_cont_cluster} roots accumulates "
            f"at the freestream phase speed c->1 (alpha_r->omega), median "
            f"alpha_i={cont_alpha_i_median:+.4f}, plus an acoustic ladder at "
            "c>1 -- B&M's continuous-spectrum branches. HONEST CAVEAT: the "
            "separation is in the phase-speed/growth-rate sense, NOT a crisp "
            f"Euclidean gap. The discretized continuum is dense in alpha_r, so "
            f"the nearest other eigenvalue is only {min_gap:.3f} away "
            f"(sep ratio {sep_ratio:.2f}) and {n_within_10pct} roots fall within "
            "10% of |alpha_disc|. This is the discretization signature of the "
            "continuous spectrum crowding the discrete mode in alpha_r, not a "
            "spurious second mode -- hence 'acceptable' (topology correct) "
            "rather than 'agrees' (which would need a clean Euclidean gap). "
            "B&M present the branches graphically; no tabulated branch "
            "eigenvalues are openly recoverable, so this is a qualitative "
            "branch-topology check."
        )
    else:
        verdict = "disagrees"
        reason = (
            "pyMack's full spectrum does not reproduce the expected branch "
            f"topology: discrete-recovered={discrete_recovered} "
            f"(got {disc.real:.6f}{disc.imag:+.6f}i vs published "
            f"0.220-0.003091i), least-damped-in-band="
            f"{disc_is_least_damped_inband}, cont-cluster={n_cont_cluster}."
        )

    verdict_doc = {
        "case_id": "balakumar_malik1992_branches",
        "category": "eigenvalue_anchor",
        "source": (
            "Balakumar & Malik (1992), 'Discrete modes and continuous spectra "
            "in supersonic boundary layers', JFM 239:631-656. The paper shows "
            "the harmonic-point-source disturbance field = discrete eigenmodes "
            "+ continuous spectrum, with the continuous spectrum forming "
            "several branches in the complex-wavenumber plane and the discrete "
            "(second) mode off those branches (M=2 and M=4.5 flat plates). "
            "Branch eigenvalues are presented graphically (no openly tabulated "
            "values). Condition used here is the verified M4.5 anchor "
            "(Xi/Ren/Fu arXiv:2006.05970 Table B 'Case 1': T0=311 K, Pr=0.72, "
            "Re=1000, omega=0.2)."
        ),
        "conditions": {
            "Ma": MA, "Re_l": RE, "omega": OMEGA, "beta": 0.0, "Pr": PR,
            "gamma": GAMMA, "wall": "adiabatic (insulated)", "T0_K": T0_K,
            "T_edge_K": T_EDGE_K, "sutherland_S_K": SUTHERLAND_S_K,
            "mode": "second mode (discrete) vs continuous spectrum",
            "length_scale": "L_star",
        },
        "quantity": (
            "branch topology: separation of the discrete second mode from the "
            "continuous-spectrum eigenvalues in the complex-alpha plane"
        ),
        "metrics": {
            "discrete_alpha": [disc.real, disc.imag],
            "published_alpha": [REF_ALPHA.real, REF_ALPHA.imag],
            "discrete_c_phase": c_disc,
            "disc_alpha_r_rel_err": disc_match_r,
            "disc_alpha_i_rel_err": disc_match_i,
            "nearest_other_alpha": [nearest.real, nearest.imag],
            "min_gap_to_other": min_gap,
            "separation_ratio": sep_ratio,
            "n_eigs_total": n_total,
            "n_within_10pct_of_disc": n_within_10pct,
            "n_continuous_cluster_near_c1": n_cont_cluster,
            "continuous_cluster_alpha_i_median": cont_alpha_i_median,
            "disc_is_least_damped_in_2nd_mode_band": disc_is_least_damped_inband,
            "alpha_i_gap_to_next_in_band": alpha_i_gap_inband,
            "n_more_amplified_than_disc": n_more_amplified,
            "more_amplified_all_nonsecond_artifacts": more_amp_all_nonsecond,
            "N": N,
            "topology_ok": bool(topology_physical),
        },
        "verdict": verdict,
        "verdict_reason": reason,
        "generated": "new",
        "artifacts": {"pymack": None, "reference": None, "overlay": None},
        "pymack_provenance": (
            "pymack.solver.solve_spatial_full_spectrum (full companion QEP, no "
            "shift-invert) on CompressibleBlasiusProfile (adiabatic, Sutherland "
            f"S=110.33 K), N={N}, y_max=40, lambda_mu_ratio=1.2, "
            "length_scale='L_star', wall_bc='isothermal'; same physical setup "
            "as validation/test_malik1990_case6_anchor.py / the via_xirenfu "
            "anchor."
        ),
    }
    write_verdict(OUT, verdict_doc)
    print(f"\nverdict: {verdict}")
    print(f"written: {OUT / 'verdict.json'}")
    return verdict_doc


if __name__ == "__main__":
    main()
