"""Capture bitwise fixtures for the 2-D temporal assembly extraction (slice 04).

Run this script from the repository root BEFORE refactoring the inline
assemblies.  It monkeypatches each solver module's ``scipy.linalg.eig`` entry
point, copies the exact (A, B) matrix pair handed to QZ, then delegates to the
real eigensolver so the public function still executes normally.  Nothing in
``pymack/`` is modified; the interception lives only in this script.

For every parameter case the fixture records, in one compressed npz per
assembler:

- the full parameter tuple (``cases_json``),
- the discretized inputs the assembler consumed: grid ``y``, derivative
  matrices ``D1``/``D2``, and every sampled base-flow field (``bf_{i}_{key}``),
- the captured operator pair ``A_{i}``/``B_{i}`` exactly as passed to
  ``scipy.linalg.eig``.

Storing the discretized inputs lets the post-refactor test feed the extracted
assemblers bit-identical inputs without re-running any transcendental math, so
the bitwise comparison certifies the code motion and nothing else.

Targets:

- ``pymack.solver.solve_temporal_compressible`` (Mack enthalpy form)
  -> ``temporal_2d_mack_assembly_fixtures.npz``
- ``pymack.temporal_solver.solve_temporal_2d`` (Ozgen form)
  -> ``temporal_2d_ozgen_assembly_fixtures.npz``
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FIXTURE_VERSION = 2
MACK_FIXTURE = Path(__file__).with_name("temporal_2d_mack_assembly_fixtures.npz")
OZGEN_FIXTURE = Path(__file__).with_name("temporal_2d_ozgen_assembly_fixtures.npz")


def _profile(u_decay, t_amp, t_decay, omega, pr_local,
             provide_kappa=True, provide_pr_local=True):
    return {
        "u_decay": float(u_decay),
        "t_amp": float(t_amp),
        "t_decay": float(t_decay),
        "omega": float(omega),
        "pr_local": float(pr_local),
        "provide_kappa": bool(provide_kappa),
        "provide_pr_local": bool(provide_pr_local),
    }


# Cases span alpha, Re, Ma, Pr, wall_bc, N in {31, 64}, y_max, L, and (for the
# Mack form) lambda_mu_ratio.  The last case of each list drops optional
# base-flow keys to exercise the fallback branch inside the moved assembly
# body (needs_pr_prefactor for Mack, the Pr_local default for Ozgen).
MACK_CASES = [
    {
        "alpha": 0.04,
        "Re": 450.0,
        "Ma": 1.8,
        "Pr": 0.72,
        "gamma": 1.4,
        "N": 31,
        "y_max": None,
        "L": None,
        "wall_bc": "isothermal",
        "length_scale": "delta_star",
        "lambda_mu_ratio": 1.2,
        "profile": _profile(0.82, 0.25, 0.42, 0.70, 0.72),
    },
    {
        "alpha": 0.08,
        "Re": 800.0,
        "Ma": 2.2,
        "Pr": 0.72,
        "gamma": 1.4,
        "N": 31,
        "y_max": None,
        "L": None,
        "wall_bc": "adiabatic",
        "length_scale": "delta_star",
        "lambda_mu_ratio": 1.2,
        "profile": _profile(0.70, -0.18, 0.35, 0.74, 0.72),
    },
    {
        "alpha": 0.12,
        "Re": 1500.0,
        "Ma": 4.5,
        "Pr": 0.70,
        "gamma": 1.4,
        "N": 64,
        "y_max": None,
        "L": None,
        "wall_bc": "isothermal",
        "length_scale": "delta_star",
        "lambda_mu_ratio": 1.2,
        "profile": _profile(0.95, 0.42, 0.48, 0.76, 0.70),
    },
    {
        "alpha": 0.18,
        "Re": 2600.0,
        "Ma": 6.0,
        "Pr": 0.72,
        "gamma": 1.4,
        "N": 31,
        "y_max": None,
        "L": None,
        "wall_bc": "adiabatic",
        "length_scale": "delta_star",
        "lambda_mu_ratio": 1.2,
        "profile": _profile(1.10, 0.15, 0.55, 0.80, 0.72),
    },
    {
        "alpha": 0.22,
        "Re": 5500.0,
        "Ma": 8.0,
        "Pr": 0.75,
        "gamma": 1.4,
        "N": 64,
        "y_max": None,
        "L": None,
        "wall_bc": "isothermal",
        "length_scale": "delta_star",
        "lambda_mu_ratio": 1.2,
        "profile": _profile(0.88, -0.22, 0.40, 0.68, 0.75),
    },
    {
        "alpha": 0.10,
        "Re": 1200.0,
        "Ma": 3.0,
        "Pr": 0.72,
        "gamma": 1.4,
        "N": 31,
        "y_max": 7.5,
        "L": 2.0,
        "wall_bc": "adiabatic",
        "length_scale": "delta_star",
        "lambda_mu_ratio": 1.2,
        "profile": _profile(0.78, 0.32, 0.30, 0.72, 0.72),
    },
    {
        # No kappa data in the base flow: transport_conductivity_data returns
        # needs_pr_prefactor=True, flipping the moved `cond` conditional.
        # Also a non-default lambda_mu_ratio.
        "alpha": 0.15,
        "Re": 2000.0,
        "Ma": 5.0,
        "Pr": 0.72,
        "gamma": 1.4,
        "N": 31,
        "y_max": None,
        "L": None,
        "wall_bc": "isothermal",
        "length_scale": "delta_star",
        "lambda_mu_ratio": 1.5,
        "profile": _profile(0.90, 0.30, 0.44, 0.71, 0.72, provide_kappa=False),
    },
]


OZGEN_CASES = [
    {
        "alpha": 0.05,
        "Re": 500.0,
        "Ma": 2.0,
        "Pr": 0.72,
        "gamma": 1.4,
        "N": 31,
        "y_max": None,
        "L": None,
        "wall_bc": "isothermal",
        "length_scale": "delta_star",
        "profile": _profile(0.76, 0.22, 0.38, 0.72, 0.72),
    },
    {
        "alpha": 0.09,
        "Re": 900.0,
        "Ma": 3.0,
        "Pr": 0.72,
        "gamma": 1.4,
        "N": 31,
        "y_max": None,
        "L": None,
        "wall_bc": "adiabatic",
        "length_scale": "delta_star",
        "profile": _profile(0.91, -0.16, 0.45, 0.74, 0.72),
    },
    {
        "alpha": 0.14,
        "Re": 1800.0,
        "Ma": 4.5,
        "Pr": 0.70,
        "gamma": 1.4,
        "N": 64,
        "y_max": None,
        "L": None,
        "wall_bc": "isothermal",
        "length_scale": "delta_star",
        "profile": _profile(1.04, 0.36, 0.50, 0.78, 0.70),
    },
    {
        "alpha": 0.19,
        "Re": 3200.0,
        "Ma": 6.0,
        "Pr": 0.72,
        "gamma": 1.4,
        "N": 31,
        "y_max": None,
        "L": None,
        "wall_bc": "adiabatic",
        "length_scale": "delta_star",
        "profile": _profile(0.84, 0.18, 0.33, 0.69, 0.72),
    },
    {
        "alpha": 0.24,
        "Re": 6500.0,
        "Ma": 7.5,
        "Pr": 0.75,
        "gamma": 1.4,
        "N": 64,
        "y_max": None,
        "L": None,
        "wall_bc": "isothermal",
        "length_scale": "delta_star",
        "profile": _profile(1.18, -0.24, 0.58, 0.76, 0.75),
    },
    {
        "alpha": 0.11,
        "Re": 1300.0,
        "Ma": 2.8,
        "Pr": 0.72,
        "gamma": 1.4,
        "N": 31,
        "y_max": 8.0,
        "L": 2.0,
        "wall_bc": "adiabatic",
        "length_scale": "delta_star",
        "profile": _profile(0.66, 0.28, 0.28, 0.73, 0.72),
    },
    {
        # No Pr_local in the base flow: exercises the moved
        # `bf.get('Pr_local', np.full_like(T, Pr))` fallback.
        "alpha": 0.16,
        "Re": 2200.0,
        "Ma": 5.0,
        "Pr": 0.71,
        "gamma": 1.4,
        "N": 31,
        "y_max": None,
        "L": None,
        "wall_bc": "isothermal",
        "length_scale": "delta_star",
        "profile": _profile(0.87, 0.26, 0.41, 0.70, 0.72, provide_pr_local=False),
    },
]


class AssemblyFixtureProfile:
    """Small deterministic analytic mean flow for assembly-level fixtures."""

    def __init__(self, *, u_decay, t_amp, t_decay, omega, pr_local,
                 provide_kappa=True, provide_pr_local=True):
        self.u_decay = float(u_decay)
        self.t_amp = float(t_amp)
        self.t_decay = float(t_decay)
        self.omega = float(omega)
        self.pr_local = float(pr_local)
        self.provide_kappa = bool(provide_kappa)
        self.provide_pr_local = bool(provide_pr_local)

    def __call__(self, y):
        y = np.asarray(y, dtype=float)
        exp_u = np.exp(-self.u_decay * y)
        exp_t = np.exp(-self.t_decay * y)

        U = 1.0 - exp_u
        dU = self.u_decay * exp_u
        d2U = -(self.u_decay**2) * exp_u

        T = 1.0 + self.t_amp * exp_t
        dT = -self.t_amp * self.t_decay * exp_t
        d2T = self.t_amp * (self.t_decay**2) * exp_t

        rho = 1.0 / T
        mu = T**self.omega
        dmu_dT = self.omega * T**(self.omega - 1.0)
        d2mu_dT2 = self.omega * (self.omega - 1.0) * T**(self.omega - 2.0)
        dmu = dmu_dT * dT

        bf = {
            "U": U,
            "dU": dU,
            "d2U": d2U,
            "T": T,
            "dT": dT,
            "d2T": d2T,
            "rho": rho,
            "mu": mu,
            "dmu": dmu,
            "dmu_dT": dmu_dT,
            "d2mu_dT2": d2mu_dT2,
        }
        if self.provide_kappa:
            bf["kappa"] = mu
            bf["dkappa"] = dmu
            bf["dkappa_dT"] = dmu_dT
            bf["d2kappa_dT2"] = d2mu_dT2
        if self.provide_pr_local:
            bf["Pr_local"] = np.full_like(T, self.pr_local)
        return bf


def profile_from_case(case):
    return AssemblyFixtureProfile(**case["profile"])


def mack_discretization(case):
    """Reproduce the discretization solve_temporal_compressible performs."""
    from pymack.solver import (
        _scaled_compressible_problem,
        chebyshev_D,
        physical_derivatives,
    )

    y_max = case["y_max"]
    if y_max is None:
        y_max = 6.0 if case["Ma"] > 2.0 else 12.0
    D_eta = chebyshev_D(case["N"])
    y, D1, D2 = physical_derivatives(D_eta, y_max, case["N"], case["L"])
    bf, D1, D2 = _scaled_compressible_problem(
        profile_from_case(case), y, D1, D2, case["length_scale"]
    )
    return y, D1, D2, bf


def ozgen_discretization(case):
    """Reproduce the discretization solve_temporal_2d performs."""
    from pymack.spectral import chebyshev_D, physical_derivatives
    from pymack.temporal_solver import _scaled_problem

    y_max = case["y_max"]
    if y_max is None:
        y_max = 6.0 if case["Ma"] > 2.0 else 12.0
    D_eta = chebyshev_D(case["N"])
    y, D1, D2 = physical_derivatives(D_eta, y_max, case["N"], case["L"])
    bf, D1, D2 = _scaled_problem(
        profile_from_case(case), y, D1, D2, case["length_scale"]
    )
    return y, D1, D2, bf


@contextmanager
def capture_module_eig(module):
    """Intercept the (A, B) pair a solver module passes to scipy.linalg.eig.

    Both solver modules alias the shared ``scipy.linalg`` module, so this
    patches the single global entry point and restores it afterwards; capture
    runs are sequential so there is no cross-talk.
    """
    original_eig = module.linalg.eig
    captured = []

    def capturing_eig(A, B, *args, **kwargs):
        captured.append((np.array(A, copy=True), np.array(B, copy=True)))
        return original_eig(A, B, *args, **kwargs)

    module.linalg.eig = capturing_eig
    try:
        yield captured
    finally:
        module.linalg.eig = original_eig


def _check_pair(A, B, n):
    nn = 4 * n
    if A.shape != (nn, nn) or B.shape != (nn, nn):
        raise RuntimeError(f"unexpected shapes A{A.shape} B{B.shape} for n={n}")
    if A.dtype != np.complex128 or B.dtype != np.complex128:
        raise RuntimeError(f"unexpected dtypes A={A.dtype} B={B.dtype}")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
        raise RuntimeError("captured matrices contain non-finite entries")


def _capture_mack(case):
    import pymack.solver as solver

    with capture_module_eig(solver) as captured:
        solver.solve_temporal_compressible(
            profile_from_case(case),
            case["alpha"],
            case["Re"],
            case["Ma"],
            case["Pr"],
            case["gamma"],
            N=case["N"],
            y_max=case["y_max"],
            L=case["L"],
            wall_bc=case["wall_bc"],
            length_scale=case["length_scale"],
            lambda_mu_ratio=case["lambda_mu_ratio"],
        )
    if len(captured) != 1:
        raise RuntimeError(f"expected one eig call, captured {len(captured)}")
    return captured[0]


def _capture_ozgen(case):
    import pymack.temporal_solver as temporal_solver

    with capture_module_eig(temporal_solver) as captured:
        temporal_solver.solve_temporal_2d(
            profile_from_case(case),
            case["alpha"],
            case["Re"],
            case["Ma"],
            Pr=case["Pr"],
            gamma=case["gamma"],
            N=case["N"],
            y_max=case["y_max"],
            L=case["L"],
            wall_bc=case["wall_bc"],
            length_scale=case["length_scale"],
        )
    if len(captured) != 1:
        raise RuntimeError(f"expected one eig call, captured {len(captured)}")
    return captured[0]


def _write_fixture(path, cases, records):
    payload = {
        "fixture_version": np.array(FIXTURE_VERSION, dtype=np.int64),
        "numpy_version": np.array(np.__version__),
        "n_cases": np.array(len(cases), dtype=np.int64),
        "cases_json": np.array(json.dumps(cases, sort_keys=True)),
    }
    for i, rec in enumerate(records):
        payload[f"A_{i}"] = rec["A"]
        payload[f"B_{i}"] = rec["B"]
        payload[f"y_{i}"] = rec["y"]
        payload[f"D1_{i}"] = rec["D1"]
        payload[f"D2_{i}"] = rec["D2"]
        for key, value in rec["bf"].items():
            payload[f"bf_{i}_{key}"] = np.asarray(value)
    np.savez_compressed(path, **payload)


def main():
    mack_records = []
    for case in MACK_CASES:
        A, B = _capture_mack(case)
        y, D1, D2, bf = mack_discretization(case)
        _check_pair(A, B, len(y))
        mack_records.append({"A": A, "B": B, "y": y, "D1": D1, "D2": D2, "bf": bf})

    ozgen_records = []
    for case in OZGEN_CASES:
        A, B = _capture_ozgen(case)
        y, D1, D2, bf = ozgen_discretization(case)
        _check_pair(A, B, len(y))
        ozgen_records.append({"A": A, "B": B, "y": y, "D1": D1, "D2": D2, "bf": bf})

    _write_fixture(MACK_FIXTURE, MACK_CASES, mack_records)
    _write_fixture(OZGEN_FIXTURE, OZGEN_CASES, ozgen_records)

    for path in (MACK_FIXTURE, OZGEN_FIXTURE):
        print(f"{path}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
