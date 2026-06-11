"""
Mean-flow profile interfaces for the stability solver.

Provides Blasius (incompressible) and compressible self-similar profile classes.
All expose a uniform callable API:

    profile(y) -> dict with
        U, dU, d2U, T, dT, d2T, rho,
        mu, dmu, dmu_dT, d2mu_dT2,
        kappa, dkappa, dkappa_dT, d2kappa_dT2, Pr_local

where `y` is non-dimensionalized by the physical displacement thickness.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_bvp, solve_ivp, trapezoid
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline


OZGEN_SUTHERLAND_S1 = 110.0
OZGEN_CONDUCTIVITY_S2 = 2.646e-3
OZGEN_CONDUCTIVITY_S3 = 245.4
OZGEN_CONDUCTIVITY_S4 = 12.0
OZGEN_THETA_VIB = 3055.0
OZGEN_CP_REF = 1006.0
OZGEN_GAMMA_REF = 1.4


def ozgen_viscosity_ratio(T_ratio, T_edge, S1=OZGEN_SUTHERLAND_S1):
    """Return Ozgen Eq. 2.36 viscosity ratio ``mu/mu_e``."""
    T_ratio = np.maximum(np.asarray(T_ratio, dtype=float), 1e-8)
    return T_ratio**1.5 * (T_edge + S1) / (T_edge * T_ratio + S1)


def ozgen_conductivity_ratio(
    T_ratio,
    T_edge,
    S2=OZGEN_CONDUCTIVITY_S2,
    S3=OZGEN_CONDUCTIVITY_S3,
    S4=OZGEN_CONDUCTIVITY_S4,
):
    """Return Ozgen Eq. 2.37 conductivity ratio ``kappa/kappa_e``."""
    T_ratio = np.maximum(np.asarray(T_ratio, dtype=float), 1e-8)
    T_abs = T_edge * T_ratio
    kappa = _ozgen_conductivity_star(T_abs, S2=S2, S3=S3, S4=S4)
    kappa_edge = _ozgen_conductivity_star(T_edge, S2=S2, S3=S3, S4=S4)
    return kappa / kappa_edge


def ozgen_cp_ratio(
    T_ratio,
    T_edge,
    cp_ref=OZGEN_CP_REF,
    gamma_ref=OZGEN_GAMMA_REF,
    theta=OZGEN_THETA_VIB,
):
    """Return Ozgen Eq. 2.38 specific-heat ratio ``Cp/Cp_e``."""
    T_ratio = np.maximum(np.asarray(T_ratio, dtype=float), 1e-8)
    T_abs = T_edge * T_ratio
    cp = _ozgen_cp_star(T_abs, cp_ref=cp_ref, gamma_ref=gamma_ref, theta=theta)
    cp_edge = _ozgen_cp_star(T_edge, cp_ref=cp_ref, gamma_ref=gamma_ref, theta=theta)
    return cp / cp_edge


def ozgen_local_prandtl(T_ratio, T_edge, Pr_edge=0.72, **kwargs):
    """Return local Ozgen Prandtl number using Eqs. 2.36-2.38."""
    mu_ratio = ozgen_viscosity_ratio(
        T_ratio,
        T_edge=T_edge,
        S1=kwargs.get('S1', OZGEN_SUTHERLAND_S1),
    )
    kappa_ratio = ozgen_conductivity_ratio(
        T_ratio,
        T_edge=T_edge,
        S2=kwargs.get('S2', OZGEN_CONDUCTIVITY_S2),
        S3=kwargs.get('S3', OZGEN_CONDUCTIVITY_S3),
        S4=kwargs.get('S4', OZGEN_CONDUCTIVITY_S4),
    )
    cp_ratio = ozgen_cp_ratio(
        T_ratio,
        T_edge=T_edge,
        cp_ref=kwargs.get('cp_ref', OZGEN_CP_REF),
        gamma_ref=kwargs.get('gamma_ref', OZGEN_GAMMA_REF),
        theta=kwargs.get('theta', OZGEN_THETA_VIB),
    )
    return Pr_edge * mu_ratio * cp_ratio / kappa_ratio


def _ozgen_conductivity_star(T_abs, S2, S3, S4):
    T_abs = np.maximum(np.asarray(T_abs, dtype=float), 1e-8)
    return S2 * np.sqrt(T_abs) / (
        1.0 + (S3 / T_abs) * 10.0 ** (-S4 / T_abs)
    )


def _ozgen_conductivity_ratio_derivatives(
    T_ratio,
    T_edge,
    Pr_edge,
    S2=OZGEN_CONDUCTIVITY_S2,
    S3=OZGEN_CONDUCTIVITY_S3,
    S4=OZGEN_CONDUCTIVITY_S4,
):
    """Return code-normalized kappa and derivatives with respect to T/T_e."""
    T_ratio = np.maximum(np.asarray(T_ratio, dtype=float), 1e-8)
    T_abs = T_edge * T_ratio
    kappa_ratio = ozgen_conductivity_ratio(
        T_ratio, T_edge=T_edge, S2=S2, S3=S3, S4=S4
    )
    log1, log2 = _ozgen_conductivity_log_derivatives(T_abs, S3=S3, S4=S4)
    kappa_code = kappa_ratio / Pr_edge
    dkappa_dT = kappa_code * T_edge * log1
    d2kappa_dT2 = kappa_code * T_edge**2 * (log1**2 + log2)
    return kappa_code, dkappa_dT, d2kappa_dT2


def _ozgen_conductivity_log_derivatives(T_abs, S3, S4):
    T_abs = np.maximum(np.asarray(T_abs, dtype=float), 1e-8)
    q = np.log(10.0) * S4
    A = (S3 / T_abs) * np.exp(-q / T_abs)
    D = 1.0 + A
    a = -1.0 / T_abs + q / T_abs**2
    ap = 1.0 / T_abs**2 - 2.0 * q / T_abs**3
    dlog = 0.5 / T_abs - A * a / D
    term_prime = (A * (a**2 + ap) * D - A**2 * a**2) / D**2
    d2log = -0.5 / T_abs**2 - term_prime
    return dlog, d2log


def _ozgen_cp_star(T_abs, cp_ref, gamma_ref, theta):
    T_abs = np.maximum(np.asarray(T_abs, dtype=float), 1e-8)
    z = theta / T_abs
    z_cap = np.minimum(z, 50.0)
    ez = np.exp(z_cap)
    denom = np.expm1(z_cap)
    vibrational = np.where(
        z > 50.0,
        0.0,
        z**2 * ez / np.maximum(denom**2, 1e-300),
    )
    return cp_ref * (
        1.0 + ((gamma_ref - 1.0) / gamma_ref) * (vibrational - 1.0)
    )


def _ozgen_mean_flow_diffusivity_ratio(T_ratio, T_edge, Pr_edge):
    """Return the ``mu/sigma`` ratio in Ozgen Eq. 2.33."""
    return (
        ozgen_viscosity_ratio(T_ratio, T_edge=T_edge)
        / ozgen_local_prandtl(T_ratio, T_edge=T_edge, Pr_edge=Pr_edge)
    )


def _ozgen_d_mean_flow_diffusivity_dT(T_ratio, T_edge, Pr_edge):
    """Return d(mu/sigma)/d(T/T_e) for Ozgen Eq. 2.33."""
    T_ratio = np.maximum(np.asarray(T_ratio, dtype=float), 1e-8)
    h = 1e-5 * np.maximum(1.0, np.abs(T_ratio))
    return (
        _ozgen_mean_flow_diffusivity_ratio(T_ratio + h, T_edge, Pr_edge)
        - _ozgen_mean_flow_diffusivity_ratio(T_ratio - h, T_edge, Pr_edge)
    ) / (2.0 * h)


class BlasiusProfile:
    """Incompressible Blasius boundary layer.

    Solves f''' + 0.5*f*f'' = 0 with f(0)=f'(0)=0, f'(inf)=1.
    """

    def __init__(self, Re_delta_star=1000, n_points=2000, eta_max=20.0):
        self.Re_delta_star = Re_delta_star
        self._solve_blasius(n_points, eta_max)

    def _solve_blasius(self, n_points, eta_max):
        f_pp_0 = 0.332057336215196

        def ode(eta, w):
            return [w[1], w[2], -0.5 * w[0] * w[2]]

        sol = solve_ivp(
            ode, [0, eta_max], [0, 0, f_pp_0],
            method='RK45', dense_output=True,
            rtol=1e-12, atol=1e-14, max_step=0.01)

        eta = np.linspace(0, eta_max, n_points)
        w = sol.sol(eta)

        self._eta = eta
        self._fp = w[1]
        self._fpp = w[2]
        fppp = -0.5 * w[0] * w[2]

        self._delta_star_eta = trapezoid(1.0 - self._fp, eta)

        self._spl_fp = CubicSpline(eta, self._fp)
        self._spl_fpp = CubicSpline(eta, self._fpp)
        self._spl_fppp = CubicSpline(eta, fppp)

    def __call__(self, y):
        """Evaluate at physical y/delta* locations."""
        eta = y * self._delta_star_eta

        U = np.clip(np.asarray(self._spl_fp(eta)), 0, 1)
        dU = np.asarray(self._spl_fpp(eta)) * self._delta_star_eta
        d2U = np.asarray(self._spl_fppp(eta)) * self._delta_star_eta**2

        return {'U': U, 'dU': dU, 'd2U': d2U}


class CompressibleBlasiusProfile:
    """Compressible flat-plate self-similar boundary layer.

    This class solves the coupled compressible mean-flow equations in the
    transformed similarity variable used by the compressible stability theory,
    instead of relying on a Crocco-Busemann closure. The physical wall-normal
    coordinate is then reconstructed from the transformed variable.

    Supported wall conditions:
    - `isothermal`: fixed wall temperature
    - `adiabatic`: zero wall heat flux

    Supported transport models:
    - `power_law`: mu/mu_e = (T/T_e)^omega, constant Pr
    - `sutherland`: Sutherland-law ratio referenced to the edge state, constant Pr
    - `mack`: Appendix-A air transport model from Mack AGARD 709
    """

    def __init__(self, Ma=5.35, T_wall=370.0, T_edge=56.0,
                 gamma=1.4, Pr=0.72, omega=0.74, R_gas=296.8,
                 Re_delta_star=1000, n_points=4000, eta_max=40.0,
                 wall_bc='isothermal', viscosity_model='power_law',
                 sutherland_S=110.4, initial_guess_profile=None):
        self.Ma = Ma
        self.T_wall = T_wall
        self.T_edge = T_edge
        self.gamma = gamma
        self.Pr = Pr
        self.omega = omega
        self.R_gas = R_gas
        self.Re_delta_star = Re_delta_star
        self.wall_bc = wall_bc
        self.viscosity_model = viscosity_model
        self.sutherland_S = sutherland_S
        self.initial_guess_profile = initial_guess_profile

        if self.wall_bc not in {'isothermal', 'adiabatic'}:
            raise ValueError("wall_bc must be 'isothermal' or 'adiabatic'")
        if self.viscosity_model not in {'power_law', 'sutherland', 'mack'}:
            raise ValueError("viscosity_model must be 'power_law', 'sutherland', or 'mack'")

        self.T_ratio_wall = T_wall / T_edge if T_edge else np.nan
        self.T_recovery = T_edge * (1 + 0.5 * (gamma - 1) * Pr**0.5 * Ma**2)
        self._mack_cp = 0.24
        self._mu_edge_star = self._mu_star_mack(self.T_edge)

        self._solve(n_points, eta_max)

    def _mu_star_mack(self, T_star):
        """Dimensional air viscosity law used in Mack Appendix A."""
        T_star = np.maximum(np.asarray(T_star, dtype=float), 1e-8)
        hot = 1.458 * T_star**1.5 / (T_star + 110.4)
        cold = 0.0693873 * T_star
        return np.where(T_star >= 110.4, hot, cold)

    def _d_mu_star_mack_dT(self, T_star):
        """Temperature derivative of Mack's dimensional viscosity law."""
        T_star = np.maximum(np.asarray(T_star, dtype=float), 1e-8)
        mu_star = self._mu_star_mack(T_star)
        hot = mu_star * (1.5 / T_star - 1.0 / (T_star + 110.4))
        cold = np.full_like(T_star, 0.0693873)
        return np.where(T_star >= 110.4, hot, cold)

    def _d2_mu_star_mack_dT2(self, T_star):
        """Second temperature derivative of Mack's dimensional viscosity law."""
        T_star = np.maximum(np.asarray(T_star, dtype=float), 1e-8)
        mu_star = self._mu_star_mack(T_star)
        g = 1.5 / T_star - 1.0 / (T_star + 110.4)
        gp = -1.5 / T_star**2 + 1.0 / (T_star + 110.4)**2
        hot = mu_star * (g**2 + gp)
        cold = np.zeros_like(T_star)
        return np.where(T_star >= 110.4, hot, cold)

    def _kappa_star_mack(self, T_star):
        """Dimensional conductivity law used in Mack Appendix A."""
        T_star = np.maximum(np.asarray(T_star, dtype=float), 1e-8)
        expo = np.exp(-(12.0 * np.log(10.0)) / T_star)
        denom = 1.0 + (245.4 / T_star) * expo
        return 0.6325 * T_star**0.5 / denom

    def _d_kappa_star_mack_dT(self, T_star):
        """Temperature derivative of Mack's dimensional conductivity law."""
        T_star = np.maximum(np.asarray(T_star, dtype=float), 1e-8)
        kappa_star = self._kappa_star_mack(T_star)
        expo = np.exp(-(12.0 * np.log(10.0)) / T_star)
        b = (245.4 / T_star) * expo
        q = -1.0 / T_star + (12.0 * np.log(10.0)) / T_star**2
        b1 = b * q
        g1 = 0.5 / T_star - b1 / (1.0 + b)
        return kappa_star * g1

    def _d2_kappa_star_mack_dT2(self, T_star):
        """Second temperature derivative of Mack's dimensional conductivity law."""
        T_star = np.maximum(np.asarray(T_star, dtype=float), 1e-8)
        kappa_star = self._kappa_star_mack(T_star)
        expo = np.exp(-(12.0 * np.log(10.0)) / T_star)
        b = (245.4 / T_star) * expo
        q = -1.0 / T_star + (12.0 * np.log(10.0)) / T_star**2
        q1 = 1.0 / T_star**2 - 2.0 * (12.0 * np.log(10.0)) / T_star**3
        b1 = b * q
        b2 = b * (q**2 + q1)
        frac1 = b1 / (1.0 + b)
        frac1_p = (b2 * (1.0 + b) - b1**2) / (1.0 + b)**2
        g1 = 0.5 / T_star - frac1
        g1_p = -0.5 / T_star**2 - frac1_p
        return kappa_star * (g1**2 + g1_p)

    def _viscosity_ratio(self, T_ratio):
        """Return mu/mu_e as a function of T/T_e."""
        T_ratio = np.maximum(T_ratio, 1e-8)
        if self.viscosity_model == 'power_law':
            return T_ratio**self.omega
        if self.viscosity_model == 'mack':
            T_star = self.T_edge * T_ratio
            return self._mu_star_mack(T_star) / self._mu_edge_star

        return (
            T_ratio**1.5
            * (self.T_edge + self.sutherland_S)
            / (self.T_edge * T_ratio + self.sutherland_S)
        )

    def _d_viscosity_ratio_dT(self, T_ratio):
        """Return d(mu/mu_e)/d(T/T_e)."""
        T_ratio = np.maximum(T_ratio, 1e-8)
        mu_ratio = self._viscosity_ratio(T_ratio)

        if self.viscosity_model == 'power_law':
            return self.omega * T_ratio**(self.omega - 1.0)
        if self.viscosity_model == 'mack':
            T_star = self.T_edge * T_ratio
            return self.T_edge * self._d_mu_star_mack_dT(T_star) / self._mu_edge_star

        return mu_ratio * (
            1.5 / T_ratio
            - self.T_edge / (self.T_edge * T_ratio + self.sutherland_S)
        )

    def _d2_viscosity_ratio_dT2(self, T_ratio):
        """Return d^2(mu/mu_e)/d(T/T_e)^2."""
        T_ratio = np.maximum(T_ratio, 1e-8)

        if self.viscosity_model == 'power_law':
            return self.omega * (self.omega - 1.0) * T_ratio**(self.omega - 2.0)
        if self.viscosity_model == 'mack':
            T_star = self.T_edge * T_ratio
            return (
                self.T_edge**2
                * self._d2_mu_star_mack_dT2(T_star)
                / self._mu_edge_star
            )

        mu_ratio = self._viscosity_ratio(T_ratio)
        g = (
            1.5 / T_ratio
            - self.T_edge / (self.T_edge * T_ratio + self.sutherland_S)
        )
        gp = (
            -1.5 / T_ratio**2
            + self.T_edge**2 / (self.T_edge * T_ratio + self.sutherland_S)**2
        )
        return mu_ratio * (g**2 + gp)

    def _conductivity_ratio(self, T_ratio):
        """Return kappa/(cp * mu_e) as a function of T/T_e."""
        T_ratio = np.maximum(T_ratio, 1e-8)
        if self.viscosity_model == 'mack':
            T_star = self.T_edge * T_ratio
            return self._kappa_star_mack(T_star) / (self._mack_cp * self._mu_edge_star)
        return self._viscosity_ratio(T_ratio) / self.Pr

    def _d_conductivity_ratio_dT(self, T_ratio):
        """Return d[kappa/(cp*mu_e)]/d(T/T_e)."""
        T_ratio = np.maximum(T_ratio, 1e-8)
        if self.viscosity_model == 'mack':
            T_star = self.T_edge * T_ratio
            return (
                self.T_edge * self._d_kappa_star_mack_dT(T_star)
                / (self._mack_cp * self._mu_edge_star)
            )
        return self._d_viscosity_ratio_dT(T_ratio) / self.Pr

    def _d2_conductivity_ratio_dT2(self, T_ratio):
        """Return d^2[kappa/(cp*mu_e)]/d(T/T_e)^2."""
        T_ratio = np.maximum(T_ratio, 1e-8)
        if self.viscosity_model == 'mack':
            T_star = self.T_edge * T_ratio
            return (
                self.T_edge**2 * self._d2_kappa_star_mack_dT2(T_star)
                / (self._mack_cp * self._mu_edge_star)
            )
        return self._d2_viscosity_ratio_dT2(T_ratio) / self.Pr

    def _chapman_rubesin(self, T_ratio):
        """Return the Chapman-Rubesin parameter C = mu / T."""
        T_ratio = np.maximum(T_ratio, 1e-8)
        return self._viscosity_ratio(T_ratio) / T_ratio

    def _d_chapman_rubesin_dT(self, T_ratio):
        """Return dC/d(T/T_e) for C = mu / T."""
        T_ratio = np.maximum(T_ratio, 1e-8)
        mu_ratio = self._viscosity_ratio(T_ratio)
        dmu = self._d_viscosity_ratio_dT(T_ratio)
        return (dmu * T_ratio - mu_ratio) / T_ratio**2

    def _initial_guess(self, eta):
        """Construct a robust initial guess for the mean-flow BVP."""
        if self.initial_guess_profile is not None:
            try:
                guess = self.initial_guess_profile._solution.sol(eta)
                if np.all(np.isfinite(guess)):
                    if self.wall_bc == 'isothermal':
                        target_wall = self.T_ratio_wall
                        wall_shift = target_wall - guess[3, 0]
                        relax = np.exp(-eta)
                        guess[3, :] = guess[3, :] + wall_shift * relax
                        guess[4, :] = guess[4, :] - wall_shift * relax
                    return guess
            except Exception:
                pass

        U = 1.0 - np.exp(-eta)
        Up = np.exp(-eta)
        f = eta - 1.0 + np.exp(-eta)

        if self.wall_bc == 'isothermal':
            T0 = self.T_ratio_wall
            T = 1.0 + (T0 - 1.0) * np.exp(-eta)
            Tp = -(T0 - 1.0) * np.exp(-eta)
        else:
            Taw_guess = 1.0 + 0.5 * (self.gamma - 1.0) * self.Pr**0.5 * self.Ma**2
            shape = np.exp(-eta**2 / 4.0)
            T = 1.0 + (Taw_guess - 1.0) * shape
            Tp = -(Taw_guess - 1.0) * 0.5 * eta * shape

        return np.vstack((f, U, Up, T, Tp))

    def _solve(self, n_points, eta_max):
        gm1 = self.gamma - 1.0

        def rhs(eta, y):
            f, U, Up, T, Tp = y
            C = self._chapman_rubesin(T)
            dC_dT = self._d_chapman_rubesin_dT(T)
            K = self._conductivity_ratio(T)
            dK_dT = self._d_conductivity_ratio_dT(T)
            B = K / T
            dB_dT = (dK_dT * T - K) / T**2

            Upp = -(dC_dT * Tp + f) * Up / C
            Tpp = (
                -(dB_dT / B) * Tp**2
                - (f / B) * Tp
                - gm1 * self.Ma**2 * (C / B) * Up**2
            )
            return np.vstack((U, Up, Upp, Tp, Tpp))

        def bc(ya, yb):
            if self.wall_bc == 'isothermal':
                return np.array([
                    ya[0],
                    ya[1],
                    ya[3] - self.T_ratio_wall,
                    yb[1] - 1.0,
                    yb[3] - 1.0,
                ])

            return np.array([
                ya[0],
                ya[1],
                ya[4],
                yb[1] - 1.0,
                yb[3] - 1.0,
            ])

        eta_mesh = np.linspace(0.0, eta_max, 400)
        tolerances = [1e-5]
        if self.wall_bc == 'isothermal' and self.initial_guess_profile is not None:
            tolerances.extend([3e-5, 1e-4])

        sol = None
        for attempt, tol in enumerate(tolerances):
            guess = self._initial_guess(eta_mesh)
            sol = solve_bvp(
                rhs, bc, eta_mesh, guess,
                tol=tol, max_nodes=100000 if attempt == 0 else 200000)
            if sol.success:
                break
            if attempt < len(tolerances) - 1:
                eta_mesh = np.linspace(0.0, eta_max, min(1200, 400 * (attempt + 2)))

        if not sol.success:
            raise RuntimeError(f'Compressible mean-flow BVP failed: {sol.message}')

        eta = np.linspace(0.0, eta_max, n_points)
        f_arr, U_arr, Up_arr, T_arr, Tp_arr = sol.sol(eta)

        C_arr = self._chapman_rubesin(T_arr)
        dC_dT_arr = self._d_chapman_rubesin_dT(T_arr)
        K_arr = self._conductivity_ratio(T_arr)
        dK_dT_arr = self._d_conductivity_ratio_dT(T_arr)
        B_arr = K_arr / T_arr
        dB_dT_arr = (dK_dT_arr * T_arr - K_arr) / T_arr**2
        Upp_arr = -(dC_dT_arr * Tp_arr + f_arr) * Up_arr / C_arr
        Tpp_arr = (
            -(dB_dT_arr / B_arr) * Tp_arr**2
            - (f_arr / B_arr) * Tp_arr
            - (self.gamma - 1.0) * self.Ma**2 * (C_arr / B_arr) * Up_arr**2
        )

        # In the compressible similarity transformation used here, the
        # physical wall-normal coordinate based on Mack's L* scale is
        # y/L* = sqrt(2) * integral_0^eta (T/T_e) d_eta.
        y_L = np.sqrt(2.0) * cumulative_trapezoid(T_arr, eta, initial=0.0)

        # delta*/L* and theta/L*.  With dy/L* = sqrt(2) T d_eta and
        # rho/rho_e = 1/T:  delta*/L* = sqrt(2) int (T - U) d_eta and
        # theta/L* = int (rho U)(1 - U) dy/L* = sqrt(2) int U (1 - U) d_eta
        # (the T factors cancel in theta's integrand).
        self._delta_star = np.sqrt(2.0) * trapezoid(T_arr - U_arr, eta)
        self._theta = np.sqrt(2.0) * trapezoid(U_arr * (1.0 - U_arr), eta)

        y_nd = y_L / self._delta_star

        # Convert derivatives from the transformed variable eta to the solver's
        # physical wall-normal coordinate y/delta*.
        fac1 = self._delta_star / (np.sqrt(2.0) * T_arr)
        fac2 = self._delta_star**2 / 2.0

        dU_arr = fac1 * Up_arr
        d2U_arr = fac2 * (Upp_arr / T_arr**2 - Tp_arr * Up_arr / T_arr**3)

        dT_arr = fac1 * Tp_arr
        d2T_arr = fac2 * (Tpp_arr / T_arr**2 - Tp_arr**2 / T_arr**3)

        rho_arr = 1.0 / T_arr
        drho_arr = -self._delta_star * Tp_arr / (np.sqrt(2.0) * T_arr**2)

        mu_arr = self._viscosity_ratio(T_arr)
        dmu_dT_arr = self._d_viscosity_ratio_dT(T_arr)
        d2mu_dT2_arr = self._d2_viscosity_ratio_dT2(T_arr)
        dmu_deta = dmu_dT_arr * Tp_arr
        dmu_arr = self._delta_star * dmu_deta / (np.sqrt(2.0) * T_arr)
        kappa_arr = K_arr
        dkappa_dT_arr = dK_dT_arr
        d2kappa_dT2_arr = self._d2_conductivity_ratio_dT2(T_arr)
        dkappa_deta = dkappa_dT_arr * Tp_arr
        dkappa_arr = self._delta_star * dkappa_deta / (np.sqrt(2.0) * T_arr)
        Pr_local_arr = mu_arr / np.maximum(kappa_arr, 1e-12)

        self._eta = eta
        self._f = f_arr
        self._U = U_arr
        self._T = T_arr
        self._y_L = y_L
        self._y_nd = y_nd
        self._rho = rho_arr
        self._mu = mu_arr
        self._dU = dU_arr
        self._d2U = d2U_arr
        self._dT = dT_arr
        self._d2T = d2T_arr
        self._drho = drho_arr
        self._dmu = dmu_arr
        self._dmu_dT = dmu_dT_arr
        self._d2mu_dT2 = d2mu_dT2_arr
        self._kappa = kappa_arr
        self._dkappa = dkappa_arr
        self._dkappa_dT = dkappa_dT_arr
        self._d2kappa_dT2 = d2kappa_dT2_arr
        self._Pr_local = Pr_local_arr
        self._solution = sol

        if self.wall_bc == 'adiabatic':
            self.T_ratio_wall = T_arr[0]
            self.T_wall = self.T_ratio_wall * self.T_edge

        self._build_splines(
            y_nd, U_arr, dU_arr, d2U_arr,
            T_arr, dT_arr, d2T_arr,
            rho_arr, mu_arr, dmu_arr,
            kappa_arr, dkappa_arr)

    def _build_splines(self, y_nd, U, dU, d2U, T, dT, d2T, rho, mu, dmu,
                       kappa, dkappa):
        """Build splines parameterized by y/delta*."""
        self._spl = {}
        for name, data in [
            ('U', U), ('dU', dU), ('d2U', d2U),
            ('T', T), ('dT', dT), ('d2T', d2T),
            ('rho', rho), ('mu', mu), ('dmu', dmu),
            ('kappa', kappa), ('dkappa', dkappa),
            ('dmu_dT', self._dmu_dT),
            ('d2mu_dT2', self._d2mu_dT2),
            ('dkappa_dT', self._dkappa_dT),
            ('d2kappa_dT2', self._d2kappa_dT2),
            ('Pr_local', self._Pr_local),
        ]:
            self._spl[name] = CubicSpline(y_nd, data, extrapolate=True)

    def __call__(self, y):
        """Evaluate the compressible mean flow at y/delta* locations."""
        y = np.asarray(y, dtype=float)
        result = {}
        for name in [
            'U', 'dU', 'd2U', 'T', 'dT', 'd2T',
            'rho', 'mu', 'dmu', 'dmu_dT', 'd2mu_dT2',
            'kappa', 'dkappa', 'dkappa_dT', 'd2kappa_dT2', 'Pr_local',
        ]:
            result[name] = np.asarray(self._spl[name](y))

        result['U'] = np.clip(result['U'], 0.0, 1.0)
        result['T'] = np.clip(result['T'], 0.05, None)
        result['rho'] = np.clip(result['rho'], 0.01, None)
        result['mu'] = np.clip(result['mu'], 0.01, None)
        result['kappa'] = np.clip(result['kappa'], 0.01, None)
        result['Pr_local'] = np.clip(result['Pr_local'], 0.01, None)
        return result


class OzgenFlatPlateProfile:
    """Ozgen-style compressible self-similar flat-plate profile.

    This profile solves Ozgen & Kircali's coupled mean-flow equations,
    Eqs. 2.32-2.35, using their temperature-dependent viscosity, conductivity,
    heat capacity, and local Prandtl-number laws.
    """

    def __init__(
        self,
        Ma,
        T_wall,
        T_edge,
        gamma=1.4,
        Pr=0.72,
        S=OZGEN_SUTHERLAND_S1,
        Re_delta_star=1000,
        n_points=4000,
        eta_max=40.0,
    ):
        self.Ma = Ma
        self.T_wall = T_wall
        self.T_edge = T_edge
        self.gamma = gamma
        self.Pr = Pr
        self.S = S
        self.Re_delta_star = Re_delta_star
        self.T_ratio_wall = None if T_wall is None else T_wall / T_edge
        self._solve(n_points, eta_max)

    def _mu_ratio(self, T_ratio):
        """Return Ozgen's Sutherland viscosity ratio mu/mu_e."""
        return ozgen_viscosity_ratio(T_ratio, T_edge=self.T_edge, S1=self.S)

    def _d_mu_ratio_dT(self, T_ratio):
        """Return d(mu/mu_e)/d(T/T_e) for Ozgen's Sutherland law."""
        T_ratio = np.maximum(np.asarray(T_ratio, dtype=float), 1e-8)
        mu_ratio = self._mu_ratio(T_ratio)
        return mu_ratio * (
            1.5 / T_ratio
            - self.T_edge / (self.T_edge * T_ratio + self.S)
        )

    def _d2_mu_ratio_dT2(self, T_ratio):
        """Return d^2(mu/mu_e)/d(T/T_e)^2 for Ozgen's Sutherland law."""
        T_ratio = np.maximum(np.asarray(T_ratio, dtype=float), 1e-8)
        mu_ratio = self._mu_ratio(T_ratio)
        g = (
            1.5 / T_ratio
            - self.T_edge / (self.T_edge * T_ratio + self.S)
        )
        gp = (
            -1.5 / T_ratio**2
            + self.T_edge**2 / (self.T_edge * T_ratio + self.S)**2
        )
        return mu_ratio * (g**2 + gp)

    def _solve(self, n_points, eta_max):
        Ma = self.Ma
        gm1 = self.gamma - 1.0
        is_adiabatic = self.T_ratio_wall is None

        def diffusivity_ratio(T_ratio):
            return _ozgen_mean_flow_diffusivity_ratio(
                T_ratio,
                T_edge=self.T_edge,
                Pr_edge=self.Pr,
            )

        def d_diffusivity_dT(T_ratio):
            return _ozgen_d_mean_flow_diffusivity_dT(
                T_ratio,
                T_edge=self.T_edge,
                Pr_edge=self.Pr,
            )

        def rhs(eta, u):
            F, U, Up, T, Tp = u
            T_safe = np.maximum(T, 0.05)
            mu = self._mu_ratio(T_safe)
            dmu_dT = self._d_mu_ratio_dT(T_safe)
            B = diffusivity_ratio(T_safe)
            dB_dT = d_diffusivity_dT(T_safe)

            Fp = U / T_safe
            Upp = -(0.5 * F * Up + dmu_dT * Tp * Up) / mu
            Tpp = (
                -gm1 * Ma**2 * mu * Up**2
                - 0.5 * F * Tp
                - dB_dT * Tp**2
            ) / B
            return np.vstack((Fp, Up, Upp, Tp, Tpp))

        def bc(ya, yb):
            thermal_wall = ya[4] if is_adiabatic else ya[3] - self.T_ratio_wall
            return np.array([
                ya[0],
                ya[1],
                thermal_wall,
                yb[1] - 1.0,
                yb[3] - 1.0,
            ])

        eta_mesh = np.linspace(0.0, eta_max, 800)
        velocity_width = max(1.0, 0.8 * Ma)
        U_guess = 1.0 - np.exp(-eta_mesh / velocity_width)
        Up_guess = np.exp(-eta_mesh / velocity_width) / velocity_width
        if is_adiabatic:
            recovery = 1.0 + 0.5 * gm1 * Ma**2 * np.sqrt(self.Pr)
            thermal_width = max(3.0, 0.8 * Ma)
            T_guess = 1.0 + (recovery - 1.0) * np.exp(-(eta_mesh / thermal_width) ** 2)
            Tp_guess = (
                (recovery - 1.0)
                * np.exp(-(eta_mesh / thermal_width) ** 2)
                * (-2.0 * eta_mesh / thermal_width**2)
            )
            Tp_guess[0] = 0.0
        else:
            # Hot/cold fixed walls at high Mach need a boundary-layer-width
            # thermal guess; a unit-width exponential can make the collocation
            # Jacobian singular before Newton reaches the physical branch.
            thermal_width = max(3.0, 0.8 * Ma)
            T_guess = 1.0 + (self.T_ratio_wall - 1.0) * np.exp(
                -(eta_mesh / thermal_width) ** 2
            )
            Tp_guess = (
                (self.T_ratio_wall - 1.0)
                * np.exp(-(eta_mesh / thermal_width) ** 2)
                * (-2.0 * eta_mesh / thermal_width**2)
            )

        F_guess = cumulative_trapezoid(U_guess, eta_mesh, initial=0.0)
        guess = np.vstack((F_guess, U_guess, Up_guess, T_guess, Tp_guess))
        sol = None
        for tol, max_nodes in ((1e-5, 200000), (5e-5, 300000), (1e-4, 400000)):
            sol = solve_bvp(
                rhs,
                bc,
                eta_mesh,
                guess,
                tol=tol,
                max_nodes=max_nodes,
            )
            if sol.success:
                break
        if not sol.success:
            raise RuntimeError(f'Ozgen mean-flow BVP failed: {sol.message}')

        eta = np.linspace(0.0, eta_max, n_points)
        F_arr, fp_arr, fpp_arr, T_arr, Tp_arr = sol.sol(eta)
        T_arr = np.maximum(T_arr, 0.05)

        y_L = eta.copy()
        rho_arr = 1.0 / T_arr
        self._delta_star = trapezoid(1.0 - rho_arr * fp_arr, eta)
        self._theta = trapezoid(rho_arr * fp_arr * (1.0 - fp_arr), eta)
        y_nd = y_L / self._delta_star

        mu_arr = self._mu_ratio(T_arr)
        dmu_dT_arr = self._d_mu_ratio_dT(T_arr)
        d2mu_dT2_arr = self._d2_mu_ratio_dT2(T_arr)
        B_arr = diffusivity_ratio(T_arr)
        dB_dT_arr = d_diffusivity_dT(T_arr)
        fppp_arr = -(
            0.5 * F_arr * fpp_arr
            + dmu_dT_arr * Tp_arr * fpp_arr
        ) / mu_arr
        d2T_deta2 = (
            -gm1 * Ma**2 * mu_arr * fpp_arr**2
            - 0.5 * F_arr * Tp_arr
            - dB_dT_arr * Tp_arr**2
        ) / B_arr

        fac1 = self._delta_star
        fac2 = self._delta_star**2

        U_arr = fp_arr
        dU_arr = fac1 * fpp_arr
        d2U_arr = fac2 * (fppp_arr / T_arr**2 - Tp_arr * fpp_arr / T_arr**3)

        dT_arr = fac1 * Tp_arr
        d2T_arr = fac2 * (d2T_deta2 / T_arr**2 - Tp_arr**2 / T_arr**3)

        rho_arr = 1.0 / T_arr
        dmu_deta = dmu_dT_arr * Tp_arr
        dmu_arr = fac1 * dmu_deta

        kappa_arr, dkappa_dT_arr, d2kappa_dT2_arr = (
            _ozgen_conductivity_ratio_derivatives(
                T_arr,
                T_edge=self.T_edge,
                Pr_edge=self.Pr,
            )
        )
        dkappa_arr = dkappa_dT_arr * dT_arr
        Pr_local_arr = ozgen_local_prandtl(
            T_arr,
            T_edge=self.T_edge,
            Pr_edge=self.Pr,
            S1=self.S,
        )

        self._eta = eta
        self._F = F_arr
        self._f = F_arr
        self._fp = fp_arr
        self._fpp = fpp_arr
        self._fppp = fppp_arr
        self._T_eta = Tp_arr
        self._T_eta_eta = d2T_deta2
        self._U = U_arr
        self._T = T_arr
        self._rho = rho_arr
        self._mu = mu_arr
        self._y_L = y_L
        self._y_nd = y_nd
        self._dU = dU_arr
        self._d2U = d2U_arr
        self._dT = dT_arr
        self._d2T = d2T_arr
        self._dmu = dmu_arr
        self._dmu_dT = dmu_dT_arr
        self._d2mu_dT2 = d2mu_dT2_arr
        self._kappa = kappa_arr
        self._dkappa = dkappa_arr
        self._dkappa_dT = dkappa_dT_arr
        self._d2kappa_dT2 = d2kappa_dT2_arr
        self._Pr_local = Pr_local_arr
        self._mean_flow_diffusivity = B_arr
        self._d_mean_flow_diffusivity_dT = dB_dT_arr
        if is_adiabatic:
            self.T_ratio_wall = float(T_arr[0])
            self.T_wall = self.T_ratio_wall * self.T_edge

        self._spl = {}
        for name, data in [
            ('U', U_arr), ('dU', dU_arr), ('d2U', d2U_arr),
            ('T', T_arr), ('dT', dT_arr), ('d2T', d2T_arr),
            ('rho', rho_arr), ('mu', mu_arr), ('dmu', dmu_arr),
            ('dmu_dT', dmu_dT_arr), ('d2mu_dT2', d2mu_dT2_arr),
            ('kappa', kappa_arr), ('dkappa', dkappa_arr),
            ('dkappa_dT', dkappa_dT_arr), ('d2kappa_dT2', d2kappa_dT2_arr),
            ('Pr_local', Pr_local_arr),
        ]:
            self._spl[name] = CubicSpline(y_nd, data, extrapolate=True)

    def mean_flow_residuals(self):
        """Return residuals of Ozgen Eqs. 2.32-2.33 in the similarity scale."""
        momentum = (
            2.0 * (
                self._dmu_dT * self._T_eta * self._fpp
                + self._mu * self._fppp
            )
            + self._F * self._fpp
        )
        energy = (
            2.0 * (
                self._d_mean_flow_diffusivity_dT * self._T_eta**2
                + self._mean_flow_diffusivity * self._T_eta_eta
            )
            + self._F * self._T_eta
            + 2.0 * (self.gamma - 1.0) * self.Ma**2 * self._mu * self._fpp**2
        )
        return momentum, energy

    def __call__(self, y):
        """Evaluate the Ozgen mean flow at y/delta* locations."""
        y = np.asarray(y, dtype=float)
        result = {}
        for name in [
            'U', 'dU', 'd2U', 'T', 'dT', 'd2T',
            'rho', 'mu', 'dmu', 'dmu_dT', 'd2mu_dT2',
            'kappa', 'dkappa', 'dkappa_dT', 'd2kappa_dT2', 'Pr_local',
        ]:
            result[name] = np.asarray(self._spl[name](y))

        result['U'] = np.clip(result['U'], 0.0, 1.0)
        result['T'] = np.clip(result['T'], 0.05, None)
        result['rho'] = np.clip(result['rho'], 0.01, None)
        result['mu'] = np.clip(result['mu'], 0.01, None)
        result['kappa'] = np.clip(result['kappa'], 0.01, None)
        result['Pr_local'] = np.clip(result['Pr_local'], 0.01, None)
        return result


def ozgen_adiabatic_wall_temperature(Ma, T_edge, Pr=0.72, gamma=1.4):
    """Return the adiabatic-wall temperature used by the Ozgen reproductions."""
    return T_edge * (1.0 + Pr**0.5 * 0.5 * (gamma - 1.0) * Ma**2)


def make_ozgen_profile(
    Ma,
    T_edge=288.0,
    T_wall=None,
    gamma=1.4,
    Pr=0.72,
    S=OZGEN_SUTHERLAND_S1,
    Re_delta_star=1000,
    n_points=4000,
    eta_max=40.0,
):
    """Build the shared Ozgen-style compressible flat-plate profile."""
    return OzgenFlatPlateProfile(
        Ma=Ma,
        T_wall=T_wall,
        T_edge=T_edge,
        gamma=gamma,
        Pr=Pr,
        S=S,
        Re_delta_star=Re_delta_star,
        n_points=n_points,
        eta_max=eta_max,
    )
