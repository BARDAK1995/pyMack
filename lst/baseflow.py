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
from scipy.integrate import cumulative_trapezoid, solve_bvp, solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline


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

        self._delta_star_eta = np.trapz(1.0 - self._fp, eta)

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

        # delta*/L* and theta/L*
        self._delta_star = np.sqrt(2.0) * np.trapz(T_arr - U_arr, eta)
        self._theta = np.sqrt(2.0) * np.trapz(U_arr * (T_arr - U_arr), eta)

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

    This profile matches the Sutherland-law adiabatic-wall mean-flow path used
    by the current Ozgen chapter reproduction code. The governing ODE is the
    shoot-search form embedded in the original chapter script rather than the
    BVP form used by :class:`CompressibleBlasiusProfile`.
    """

    def __init__(
        self,
        Ma,
        T_wall,
        T_edge,
        gamma=1.4,
        Pr=0.72,
        S=110.4,
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
        self.T_ratio_wall = T_wall / T_edge
        self._solve(n_points, eta_max)

    def _mu_ratio(self, T_ratio):
        """Return Ozgen's Sutherland viscosity ratio mu/mu_e."""
        T_ratio = np.maximum(np.asarray(T_ratio, dtype=float), 1e-8)
        return (
            T_ratio**1.5
            * (self.T_edge + self.S)
            / (self.T_edge * T_ratio + self.S)
        )

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
        g_w = self.T_ratio_wall

        def g_from_fp(fp):
            return g_w + (1.0 - g_w) * fp + 0.5 * gm1 * Ma**2 * fp * (1.0 - fp)

        def dg_dfp(fp):
            return (1.0 - g_w) + 0.5 * gm1 * Ma**2 * (1.0 - 2.0 * fp)

        def ode(eta, u):
            f, fp, fpp = u
            g = max(g_from_fp(fp), 0.01)
            c_ratio = self._mu_ratio(g)
            dc_dg = self._d_mu_ratio_dT(g)
            dg = dg_dfp(fp)
            fppp = -(f * fpp + dc_dg * dg * fpp**2) / c_ratio
            return [fp, fpp, fppp]

        def shoot(fpp0):
            try:
                sol = solve_ivp(
                    ode,
                    [0.0, eta_max],
                    [0.0, 0.0, fpp0],
                    method='RK45',
                    rtol=1e-10,
                    atol=1e-12,
                    max_step=0.05,
                )
                return sol.y[1, -1] - 1.0
            except Exception:
                return 1.0

        fpp0 = brentq(shoot, 0.01, 2.0, xtol=1e-12)
        sol = solve_ivp(
            ode,
            [0.0, eta_max],
            [0.0, 0.0, fpp0],
            method='RK45',
            dense_output=True,
            rtol=1e-12,
            atol=1e-14,
            max_step=0.005,
        )

        eta = np.linspace(0.0, eta_max, n_points)
        w = sol.sol(eta)
        f_arr = w[0]
        fp_arr = w[1]
        fpp_arr = w[2]
        T_arr = np.maximum(g_from_fp(fp_arr), 0.01)

        y_phys = cumulative_trapezoid(T_arr, eta, initial=0.0)
        y_L = np.sqrt(2.0) * y_phys
        self._delta_star = np.sqrt(2.0) * np.trapz(T_arr - fp_arr, eta)
        self._theta = np.sqrt(2.0) * np.trapz(fp_arr * (T_arr - fp_arr), eta)
        y_nd = y_L / self._delta_star

        dT_deta = dg_dfp(fp_arr) * fpp_arr
        mu_arr = self._mu_ratio(T_arr)
        dmu_dT_arr = self._d_mu_ratio_dT(T_arr)
        d2mu_dT2_arr = self._d2_mu_ratio_dT2(T_arr)
        fppp_arr = -(f_arr * fpp_arr + dmu_dT_arr * dg_dfp(fp_arr) * fpp_arr**2) / mu_arr

        d2g_dfp2 = -gm1 * Ma**2
        d2T_deta2 = d2g_dfp2 * fpp_arr**2 + dg_dfp(fp_arr) * fppp_arr

        fac1 = self._delta_star / (np.sqrt(2.0) * T_arr)
        fac2 = self._delta_star**2 / 2.0

        U_arr = fp_arr
        dU_arr = fac1 * fpp_arr
        d2U_arr = fac2 * (fppp_arr / T_arr**2 - dT_deta * fpp_arr / T_arr**3)

        dT_arr = fac1 * dT_deta
        d2T_arr = fac2 * (d2T_deta2 / T_arr**2 - dT_deta**2 / T_arr**3)

        rho_arr = 1.0 / T_arr
        dmu_deta = dmu_dT_arr * dT_deta
        dmu_arr = fac1 * dmu_deta

        # Match the legacy Ozgen chapter behavior under the shared solver
        # interface: constant Pr, so kappa/(cp*mu_e) = mu/(Pr*mu_e).
        kappa_arr = mu_arr / self.Pr
        dkappa_arr = dmu_arr / self.Pr
        dkappa_dT_arr = dmu_dT_arr / self.Pr
        d2kappa_dT2_arr = d2mu_dT2_arr / self.Pr
        Pr_local_arr = np.full_like(T_arr, self.Pr)

        self._eta = eta
        self._f = f_arr
        self._fp = fp_arr
        self._fpp = fpp_arr
        self._fppp = fppp_arr
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
    S=110.4,
    Re_delta_star=1000,
    n_points=4000,
    eta_max=40.0,
):
    """Build the shared Ozgen-style compressible flat-plate profile."""
    if T_wall is None:
        T_wall = ozgen_adiabatic_wall_temperature(Ma, T_edge, Pr=Pr, gamma=gamma)
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
