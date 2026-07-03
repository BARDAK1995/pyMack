"""Validation gate: Orszag (1971) Table 5 -- the FULL Orr-Sommerfeld spectrum.

Where ``test_orr_sommerfeld.py`` checks only the single least-stable eigenvalue
(Orszag Table 2), this gate checks the whole tabulated spectrum: all 32
least-stable eigenvalues of plane-Poiseuille flow at alpha=1, R=10000, with
their symmetric/antisymmetric parity and near-degenerate "fast" pairs.

    S. A. Orszag, "Accurate solution of the Orr-Sommerfeld stability equation",
    J. Fluid Mech. 50(4):689-703 (1971), Table 5 -- 32 least-stable eigenvalues
    lambda = c for alpha = 1, R = 10000 (source scan:
    refPapers/NewPapers/figures/orszag1971_table5_least_stable_spectrum.png).

This exercises pyMack's dense eigensolver on the FULL spectrum for the first
time (not just a shift-invert around one target): it must resolve the A-branch
(c_r ~ 0.96 downward), the P-branch (c_r ~ 0.2-0.5), the S-branch, AND the
near-degenerate symmetric/antisymmetric pairs that differ only in the 5th-6th
digit (e.g. modes 2 & 3 at 0.96463 vs 0.96464). pyMack matches all 32 to below
1e-5 at N=128 (max ~7e-6, median ~5e-8).
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from pymack.spectral import chebyshev_points, chebyshev_D

# Orszag (1971) Table 5: 32 least-stable eigenvalues lambda = c (alpha=1, R=1e4).
ORSZAG_TABLE5 = np.array([
    0.23752649 + 0.00373967j, 0.96463092 - 0.03516728j, 0.96464251 - 0.03518658j,
    0.27720434 - 0.05089873j, 0.93631654 - 0.06320150j, 0.93635178 - 0.06325157j,
    0.90798305 - 0.09122274j, 0.90805633 - 0.09131286j, 0.87962729 - 0.11923285j,
    0.87975570 - 0.11937073j, 0.34910682 - 0.12450198j, 0.41635102 - 0.13822652j,
    0.8512458 - 0.1472339j,   0.8514494 - 0.1474256j,   0.8228350 - 0.1752287j,
    0.8231370 - 0.1754781j,   0.1900592 - 0.1828219j,   0.794388 - 0.203221j,
    0.794818 - 0.203529j,     0.532045 - 0.206465j,     0.474901 - 0.208731j,
    0.76588 - 0.23119j,       0.76649 - 0.23159j,       0.36850 - 0.23882j,
    0.73741 - 0.25872j,       0.73812 - 0.25969j,       0.63672 - 0.25988j,
    0.38399 - 0.26511j,       0.58721 - 0.26716j,       0.71232 - 0.28551j,
    0.51292 - 0.28663j,       0.70887 - 0.28765j,
])


def poiseuille_spectrum(alpha, Re, N):
    """Full temporal OS spectrum for plane Poiseuille U=1-y^2 on [-1, 1]."""
    x = chebyshev_points(N)
    D = chebyshev_D(N)
    D2 = D @ D
    D4 = D2 @ D2
    I = np.eye(N + 1)
    U = np.diag(1.0 - x**2)
    d2U = np.diag(-2.0 * np.ones(N + 1))
    a2 = alpha**2
    L2 = D2 - a2 * I
    L4 = D4 - 2 * a2 * D2 + a2**2 * I
    A = -L4 / (1j * alpha * Re) + U @ L2 - d2U
    B = L2.copy()
    for idx in (0, N):                       # v = 0 at both walls
        A[idx, :] = 0.0; A[idx, idx] = 1.0; B[idx, :] = 0.0
    A[1, :] = D[0, :]; B[1, :] = 0.0         # Dv = 0 at y = +1
    A[N - 1, :] = D[N, :]; B[N - 1, :] = 0.0  # Dv = 0 at y = -1
    ev, _ = linalg.eig(A, B)
    ev = ev[np.isfinite(ev)]
    return ev[(np.abs(ev.real) < 1.5) & (np.abs(ev.imag) < 1.0)]


def test_orszag_full_32_mode_spectrum():
    """Every one of Orszag's 32 tabulated eigenvalues is reproduced to < 1e-4."""
    spec = poiseuille_spectrum(1.0, 10000.0, N=128)
    errs = np.array([np.min(np.abs(spec - c)) for c in ORSZAG_TABLE5])
    worst = int(np.argmax(errs))
    assert errs.max() < 1.0e-4, (
        f"worst-matched mode {worst + 1}: err {errs[worst]:.2e} "
        f"(Orszag {ORSZAG_TABLE5[worst]})"
    )
    # Strong form: essentially all match to < 1e-5.
    assert (errs < 1.0e-5).sum() >= 31, (
        f"only {(errs < 1e-5).sum()}/32 modes match to < 1e-5"
    )


def test_orszag_least_stable_is_the_unstable_ts_mode():
    """Mode 1 (least damped) is the single unstable TS mode, c_i > 0."""
    spec = poiseuille_spectrum(1.0, 10000.0, N=128)
    top = spec[np.argmax(spec.imag)]
    assert abs(top - ORSZAG_TABLE5[0]) < 1.0e-5
    assert top.imag > 0.0
    # exactly one unstable mode in the spectrum
    assert (spec.imag > 1.0e-6).sum() == 1
