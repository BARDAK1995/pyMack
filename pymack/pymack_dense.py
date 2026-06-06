"""Dense pyMack-style spatial backend for 2-D Mack/S branch checks.

This module is intentionally narrow.  It implements the compact dense
spatial QEP path used by the external ``CodexFinal`` reference solver:

    q'(x,y,t) = qhat(y) exp(i alpha x - i omega t)
    omega = F * R,  R = sqrt(Re_x),  y is scaled by L*
    spatial growth = -Im(alpha)

The backend is useful as an independent reference for Mach-6 Mack/S branch
work because it solves the full dense companion spectrum and tracks the curve
from a global most-amplified seed.  It is not a replacement for the broader
``lst.solver`` API until it has been validated across other regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import PchipInterpolator
from scipy.linalg import eig


@dataclass(frozen=True)
class DenseGasModel:
    gamma: float = 1.4
    prandtl: float = 0.72
    viscosity_law: str = "sutherland"
    mu_power: float = 0.7
    sutherland_S_K: float = 110.4
    T_edge_K: float = 300.0


@dataclass(frozen=True)
class DenseBaseFlowConfig:
    mach_edge: float = 6.0
    Tw_Te: float = 5.88
    eta_max: float = 16.0
    eta_nodes: int = 80
    bvp_tol: float = 1.0e-4
    adiabatic: bool = False


@dataclass(frozen=True)
class DenseLSTConfig:
    ny: int = 31
    y_max: float = 30.0
    c_min: float = 0.86
    c_max: float = 0.97
    max_abs_alpha: float = 8.0
    max_abs_ai: float = 0.4
    max_ai_over_ar: float = 1.0


@dataclass
class DenseBaseFlow:
    y: np.ndarray
    U: np.ndarray
    T: np.ndarray
    rho: np.ndarray
    mu: np.ndarray
    dmu_dT: np.ndarray
    eta: np.ndarray
    f: np.ndarray
    notes: Dict[str, float]


def nondim_mu(T: np.ndarray, gas: DenseGasModel) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if gas.viscosity_law.lower() == "power":
        return np.maximum(T, 1.0e-12) ** gas.mu_power
    if gas.viscosity_law.lower() == "sutherland":
        S_over_Te = gas.sutherland_S_K / gas.T_edge_K
        return (
            np.maximum(T, 1.0e-12) ** 1.5
            * (1.0 + S_over_Te)
            / (T + S_over_Te)
        )
    raise ValueError(f"unknown viscosity law: {gas.viscosity_law}")


def nondim_dmu_dT(T: np.ndarray, gas: DenseGasModel) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if gas.viscosity_law.lower() == "power":
        return gas.mu_power * np.maximum(T, 1.0e-12) ** (gas.mu_power - 1.0)
    if gas.viscosity_law.lower() == "sutherland":
        S_over_Te = gas.sutherland_S_K / gas.T_edge_K
        A = 1.0 + S_over_Te
        T_safe = np.maximum(T, 1.0e-12)
        return A * (
            1.5 * np.sqrt(T_safe) * (T_safe + S_over_Te) - T_safe**1.5
        ) / (T_safe + S_over_Te) ** 2
    raise ValueError(f"unknown viscosity law: {gas.viscosity_law}")


def solve_base_flow(
    base_cfg: DenseBaseFlowConfig,
    gas: DenseGasModel,
) -> DenseBaseFlow:
    """Solve the Lees-Dorodnitsyn compressible similarity profile."""
    Me = base_cfg.mach_edge
    gam = gas.gamma
    Pr = gas.prandtl
    Tw = base_cfg.Tw_Te

    eta = np.linspace(0.0, base_cfg.eta_max, base_cfg.eta_nodes)
    U_guess = np.tanh(eta / 3.0)
    f_guess = np.zeros_like(eta)
    f_guess[1:] = np.cumsum(0.5 * (U_guess[:-1] + U_guess[1:]) * np.diff(eta))
    Upp_guess = (1.0 / 3.0) / np.cosh(eta / 3.0) ** 2
    if base_cfg.adiabatic:
        T_wall_guess = 1.0 + np.sqrt(Pr) * (gam - 1.0) * 0.5 * Me**2
        T_guess = 1.0 + (T_wall_guess - 1.0) * np.exp(-eta / 4.0)
        Tp_guess = -(T_wall_guess - 1.0) * np.exp(-eta / 4.0) / 4.0
    else:
        T_guess = 1.0 + (Tw - 1.0) * np.exp(-eta / 4.0)
        Tp_guess = -(Tw - 1.0) * np.exp(-eta / 4.0) / 4.0
    Y0 = np.vstack([f_guess, U_guess, Upp_guess, T_guess, Tp_guess])

    def rhs(_eta: np.ndarray, Y: np.ndarray) -> np.ndarray:
        f, U, Upp, T, Tp = Y
        mu = nondim_mu(T, gas)
        dmu = nondim_dmu_dT(T, gas)
        C = mu / T
        dC_dT = (dmu * T - mu) / T**2
        Cp = dC_dT * Tp
        f3 = -(Cp * Upp + f * Upp) / C
        Tpp = -(
            Cp * Tp
            + Pr * f * Tp
            + Pr * (gam - 1.0) * Me**2 * C * Upp**2
        ) / C
        return np.vstack([U, Upp, f3, Tp, Tpp])

    def bc(Ya: np.ndarray, Yb: np.ndarray) -> np.ndarray:
        if base_cfg.adiabatic:
            return np.array([Ya[0], Ya[1], Ya[4], Yb[1] - 1.0, Yb[3] - 1.0])
        return np.array([Ya[0], Ya[1], Ya[3] - Tw, Yb[1] - 1.0, Yb[3] - 1.0])

    sol = solve_bvp(
        rhs,
        bc,
        eta,
        Y0,
        tol=base_cfg.bvp_tol,
        max_nodes=40000,
        verbose=0,
    )
    if sol.status != 0:
        raise RuntimeError(f"base-flow BVP failed: {sol.message}")

    eta_dense = np.linspace(0.0, base_cfg.eta_max, max(base_cfg.eta_nodes, 3000))
    f, U, Upp, T, _Tp = sol.sol(eta_dense)
    y = np.zeros_like(eta_dense)
    y[1:] = np.sqrt(2.0) * np.cumsum(
        0.5 * (T[:-1] + T[1:]) * np.diff(eta_dense)
    )

    rho = 1.0 / T
    mu = nondim_mu(T, gas)
    dmu_dT = nondim_dmu_dT(T, gas)
    notes = {
        "eta_max": float(base_cfg.eta_max),
        "y_max_base": float(y[-1]),
        "U_edge_error": float(abs(U[-1] - 1.0)),
        "T_edge_error": float(abs(T[-1] - 1.0)),
        "T_wall": float(T[0]),
        "T_max": float(np.max(T)),
        "viscosity_wall": float(mu[0]),
        "skin_shear_fpp0_eta": float(Upp[0]),
    }
    return DenseBaseFlow(
        y=y,
        U=U,
        T=T,
        rho=rho,
        mu=mu,
        dmu_dT=dmu_dT,
        eta=eta_dense,
        f=f,
        notes=notes,
    )


def cheb_y(ny: int, y_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chebyshev points y in [0, y_max] and derivative matrices."""
    if ny < 8:
        raise ValueError("ny must be at least 8")
    N = ny - 1
    k = np.arange(ny)
    x = np.cos(np.pi * k / N)
    c = np.ones(ny)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * (-1.0) ** k
    X = np.tile(x[:, None], (1, ny))
    dX = X - X.T
    D_x = (np.outer(c, 1.0 / c)) / (dX + np.eye(ny))
    D_x = D_x - np.diag(np.sum(D_x, axis=1))
    y = 0.5 * y_max * (1.0 - x)
    D_y = -(2.0 / y_max) * D_x
    return y, D_y, D_y @ D_y


def interpolate_base_to_grid(
    base: DenseBaseFlow,
    y: np.ndarray,
    gas: DenseGasModel,
) -> Dict[str, np.ndarray]:
    if y[-1] > base.y[-1]:
        raise ValueError(
            f"LST y_max={y[-1]:.3g} exceeds base-flow y_max={base.y[-1]:.3g}"
        )
    U = PchipInterpolator(base.y, base.U, extrapolate=False)(y)
    T = PchipInterpolator(base.y, base.T, extrapolate=False)(y)
    U[-1] = min(U[-1], 1.0)
    T[-1] = max(T[-1], 1.0)
    rho = 1.0 / T
    mu = nondim_mu(T, gas)
    dmu = nondim_dmu_dT(T, gas)
    return {"U": U, "T": T, "rho": rho, "mu": mu, "dmu_dT": dmu}


def _block_insert(M: np.ndarray, row0: int, col0: int, block: np.ndarray) -> None:
    nr, nc = block.shape
    M[row0: row0 + nr, col0: col0 + nc] += block


def assemble_operator(
    alpha: complex,
    omega: float,
    R: float,
    y: np.ndarray,
    D: np.ndarray,
    base_grid: Dict[str, np.ndarray],
    gas: DenseGasModel,
    lst_cfg: DenseLSTConfig,
) -> np.ndarray:
    """Assemble L(alpha) q = 0 for q=[rho,u,v,T]."""
    del lst_cfg
    n = len(y)
    I = np.eye(n, dtype=complex)
    U = base_grid["U"].astype(float)
    T = base_grid["T"].astype(float)
    rho = base_grid["rho"].astype(float)
    mu = base_grid["mu"].astype(float)
    dmu = base_grid["dmu_dT"].astype(float)
    Uy = D @ U
    Ty = D @ T

    def diag(a):
        return np.diag(np.asarray(a, dtype=complex))

    i = 1j
    s = i * alpha * U - i * omega
    Me = float(base_grid["mach_edge"])
    pscale = 1.0 / (gas.gamma * Me**2)
    P_r = pscale * diag(T)
    P_t = pscale * diag(rho)
    Mu = diag(mu)
    Rho = diag(rho)
    RhoT = diag(rho * T)
    Sdiag = diag(s)

    ir, iu, iv, it = 0, n, 2 * n, 3 * n
    L = np.zeros((4 * n, 4 * n), dtype=complex)

    row = ir
    _block_insert(L, row, ir, Sdiag)
    _block_insert(L, row, iu, diag(rho) * (i * alpha))
    _block_insert(L, row, iv, diag(D @ rho) + diag(rho) @ D)

    tauxx_u = Mu * ((4.0 / 3.0) * i * alpha)
    tauxx_v = Mu @ (-(2.0 / 3.0) * D)
    tauxy_u = Mu @ D
    tauxy_v = Mu * (i * alpha)
    tauxy_t = diag(dmu * Uy)
    tauyy_u = Mu * (-(2.0 / 3.0) * i * alpha)
    tauyy_v = Mu @ ((4.0 / 3.0) * D)

    visc_x_u = i * alpha * tauxx_u + D @ tauxy_u
    visc_x_v = i * alpha * tauxx_v + D @ tauxy_v
    visc_x_t = D @ tauxy_t
    visc_y_u = i * alpha * tauxy_u + D @ tauyy_u
    visc_y_v = i * alpha * tauxy_v + D @ tauyy_v
    visc_y_t = i * alpha * tauxy_t

    row = iu
    _block_insert(L, row, ir, i * alpha * P_r)
    _block_insert(L, row, iu, Rho @ Sdiag - visc_x_u / R)
    _block_insert(L, row, iv, diag(rho * Uy) - visc_x_v / R)
    _block_insert(L, row, it, i * alpha * P_t - visc_x_t / R)

    row = iv
    _block_insert(L, row, ir, D @ P_r)
    _block_insert(L, row, iu, -visc_y_u / R)
    _block_insert(L, row, iv, Rho @ Sdiag - visc_y_v / R)
    _block_insert(L, row, it, D @ P_t - visc_y_t / R)

    conduction_t = (-alpha**2) * Mu + D @ (Mu @ D + diag(dmu * Ty))
    diss_u = diag(2.0 * mu * Uy) @ D
    diss_v = diag(2.0 * mu * Uy) * (i * alpha)
    diss_t = diag(dmu * Uy**2)
    gam = gas.gamma
    Pr = gas.prandtl
    diss_coeff = gam * (gam - 1.0) * Me**2 / R
    cond_coeff = gam / (Pr * R)

    row = it
    _block_insert(
        L,
        row,
        iu,
        (gam - 1.0) * RhoT * (i * alpha) - diss_coeff * diss_u,
    )
    _block_insert(
        L,
        row,
        iv,
        diag(rho * Ty) + (gam - 1.0) * RhoT @ D - diss_coeff * diss_v,
    )
    _block_insert(
        L,
        row,
        it,
        Rho @ Sdiag - cond_coeff * conduction_t - diss_coeff * diss_t,
    )

    def set_bc(row_index: int, col_index: int) -> None:
        L[row_index, :] = 0.0
        L[row_index, col_index] = 1.0

    wall = 0
    far = n - 1
    set_bc(iu + wall, iu + wall)
    set_bc(iv + wall, iv + wall)
    set_bc(it + wall, it + wall)
    set_bc(ir + far, ir + far)
    set_bc(iu + far, iu + far)
    set_bc(iv + far, iv + far)
    set_bc(it + far, it + far)
    return L


def quadratic_matrices(
    omega: float,
    R: float,
    y: np.ndarray,
    D: np.ndarray,
    base_grid: Dict[str, np.ndarray],
    gas: DenseGasModel,
    lst_cfg: DenseLSTConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    L0 = assemble_operator(0.0 + 0.0j, omega, R, y, D, base_grid, gas, lst_cfg)
    Lp = assemble_operator(1.0 + 0.0j, omega, R, y, D, base_grid, gas, lst_cfg)
    Lm = assemble_operator(-1.0 + 0.0j, omega, R, y, D, base_grid, gas, lst_cfg)
    A0 = L0
    A1 = 0.5 * (Lp - Lm)
    A2 = 0.5 * (Lp + Lm) - L0
    return A0, A1, A2


def solve_spatial_evp(
    omega: float,
    R: float,
    y: np.ndarray,
    D: np.ndarray,
    base_grid: Dict[str, np.ndarray],
    gas: DenseGasModel,
    lst_cfg: DenseLSTConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    A0, A1, A2 = quadratic_matrices(omega, R, y, D, base_grid, gas, lst_cfg)
    n4 = A0.shape[0]
    O = np.zeros_like(A0)
    I4 = np.eye(n4, dtype=complex)
    Llin = np.block([[O, I4], [-A0, -A1]])
    Mlin = np.block([[I4, O], [O, A2]])
    vals, vecs = eig(Llin, Mlin, overwrite_a=True, overwrite_b=True, check_finite=False)
    return vals, vecs[:n4, :]


def candidate_indices(vals: np.ndarray, omega: float, cfg: DenseLSTConfig) -> np.ndarray:
    vals = np.asarray(vals)
    finite = np.isfinite(vals.real) & np.isfinite(vals.imag)
    ar = vals.real
    ai = vals.imag
    with np.errstate(divide="ignore", invalid="ignore"):
        c = omega / ar
    mask = (
        finite
        & (ar > 1.0e-8)
        & (np.abs(vals) < cfg.max_abs_alpha)
        & (np.abs(ai) < cfg.max_abs_ai)
        & (np.abs(ai) / np.maximum(ar, 1.0e-12) < cfg.max_ai_over_ar)
        & (c > cfg.c_min)
        & (c < cfg.c_max)
    )
    return np.flatnonzero(mask)


def _select_seed(vals: np.ndarray, omega: float, cfg: DenseLSTConfig) -> Optional[int]:
    idx = candidate_indices(vals, omega, cfg)
    if idx.size == 0:
        return None
    growth = -vals[idx].imag
    phase = omega / vals[idx].real
    score = growth - 0.03 * np.abs(phase - 0.92)
    return int(idx[np.argmax(score)])


def _select_nearest(
    vals: np.ndarray,
    omega: float,
    previous_alpha: complex,
    cfg: DenseLSTConfig,
) -> Optional[int]:
    idx = candidate_indices(vals, omega, cfg)
    if idx.size == 0:
        return None
    scale = max(abs(previous_alpha), 1.0e-3)
    dist = np.abs(vals[idx] - previous_alpha) / scale
    growth = -vals[idx].imag
    score = dist - 0.02 * growth
    return int(idx[np.argmin(score)])


def omega_from_frequency(value: float, R: float, convention: str = "mack") -> float:
    if convention == "mack":
        return float(value * R)
    if convention == "mack_x1e6":
        return float(value * 1.0e-6 * R)
    if convention == "omega":
        return float(value)
    raise ValueError(f"unknown frequency convention {convention}")


def prepare_dense_case(
    gas: DenseGasModel,
    base_cfg: DenseBaseFlowConfig,
    lst_cfg: DenseLSTConfig,
):
    """Return reusable base-flow/grid data for repeated frequency sweeps."""
    base = solve_base_flow(base_cfg, gas)
    y, D, _D2 = cheb_y(lst_cfg.ny, lst_cfg.y_max)
    base_grid = interpolate_base_to_grid(base, y, gas)
    base_grid["mach_edge"] = np.array(base_cfg.mach_edge)
    return base, y, D, base_grid


def solve_mack_branch(
    F: float,
    R_values: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
    base_grid: Dict[str, np.ndarray],
    gas: DenseGasModel,
    lst_cfg: DenseLSTConfig,
    convention: str = "mack",
) -> List[Dict[str, float]]:
    """Track one Mack/S spatial branch over a Reynolds grid."""
    spectra = []
    seed_candidates = []
    R_values = np.asarray(R_values, dtype=float)

    for pos, R in enumerate(R_values):
        omega = omega_from_frequency(float(F), float(R), convention)
        vals, _vecs = solve_spatial_evp(omega, float(R), y, D, base_grid, gas, lst_cfg)
        spectra.append((float(R), omega, vals))
        seed = _select_seed(vals, omega, lst_cfg)
        if seed is not None:
            seed_candidates.append((pos, vals[seed], -vals[seed].imag))

    selected: Dict[int, complex] = {}
    if seed_candidates:
        seed_pos, seed_alpha, _growth = max(seed_candidates, key=lambda item: item[2])
        selected[seed_pos] = seed_alpha

        prev = seed_alpha
        for pos in range(seed_pos - 1, -1, -1):
            _R, omega, vals = spectra[pos]
            idx = _select_nearest(vals, omega, prev, lst_cfg)
            if idx is None:
                selected[pos] = np.nan + 1j * np.nan
                continue
            prev = vals[idx]
            selected[pos] = prev

        prev = seed_alpha
        for pos in range(seed_pos + 1, len(spectra)):
            _R, omega, vals = spectra[pos]
            idx = _select_nearest(vals, omega, prev, lst_cfg)
            if idx is None:
                selected[pos] = np.nan + 1j * np.nan
                continue
            prev = vals[idx]
            selected[pos] = prev

    rows = []
    for pos, (R, omega, vals) in enumerate(spectra):
        alpha = selected.get(pos, np.nan + 1j * np.nan)
        if np.isfinite(alpha.real) and alpha.real > 0.0:
            phase_speed = omega / alpha.real
            selected_flag = 1.0
            note = 0.0
        else:
            alpha = np.nan + 1j * np.nan
            phase_speed = np.nan
            selected_flag = 0.0
            note = 1.0
        rows.append(
            {
                "F": float(F),
                "R": float(R),
                "omega": float(omega),
                "alpha_real": float(alpha.real),
                "alpha_imag": float(alpha.imag),
                "growth": float(-alpha.imag),
                "phase_speed": float(phase_speed),
                "selected": float(selected_flag),
                "tracking_failed": float(note),
                "n_candidates": float(len(candidate_indices(vals, omega, lst_cfg))),
            }
        )
    return rows


def find_neutral_crossings(rows: List[Dict[str, float]]) -> List[float]:
    """Return linear-interpolated R where growth changes sign."""
    data = sorted(rows, key=lambda item: item["R"])
    out: List[float] = []
    for left, right in zip(data[:-1], data[1:]):
        g0 = left["growth"]
        g1 = right["growth"]
        if not (np.isfinite(g0) and np.isfinite(g1)):
            continue
        if g0 == 0.0:
            out.append(float(left["R"]))
        elif g0 * g1 < 0.0:
            R0 = left["R"]
            R1 = right["R"]
            out.append(float(R0 - g0 * (R1 - R0) / (g1 - g0)))
    return out
