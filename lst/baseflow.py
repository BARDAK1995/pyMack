"""
Mean flow profile interfaces for the stability solver.

Provides Blasius (incompressible) and compressible self-similar
profile classes. All expose a uniform callable API:
    profile(y) -> dict with U, dU, d2U, T, dT, d2T, rho, mu, dmu
where y is non-dimensionalized by physical delta*.
"""

import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq


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

        sol = solve_ivp(ode, [0, eta_max], [0, 0, f_pp_0],
                        method='RK45', dense_output=True,
                        rtol=1e-12, atol=1e-14, max_step=0.01)

        eta = np.linspace(0, eta_max, n_points)
        w = sol.sol(eta)

        self._eta = eta
        self._fp = w[1]
        self._fpp = w[2]
        fppp = -0.5 * w[0] * w[2]

        # delta* in similarity coords
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
    """Compressible self-similar boundary layer.

    Uses the Illingworth-Stewartson transformation with Crocco-Busemann
    temperature relation and power-law viscosity mu = (T/T_e)^omega.

    The key coordinate mapping: physical y relates to similarity eta by
        y_phys(eta) = integral_0^eta g(s) ds
    where g = T/T_e. This nonlinear mapping is computed exactly.

    Parameters
    ----------
    Ma : float
        Edge Mach number.
    T_wall, T_edge : float
        Wall and edge temperatures [K].
    gamma, Pr, omega : float
        Gas properties.
    """

    def __init__(self, Ma=5.35, T_wall=370.0, T_edge=56.0,
                 gamma=1.4, Pr=0.72, omega=0.74, R_gas=296.8,
                 Re_delta_star=1000, n_points=4000, eta_max=30.0):
        self.Ma = Ma
        self.T_wall = T_wall
        self.T_edge = T_edge
        self.gamma = gamma
        self.Pr = Pr
        self.omega = omega
        self.R_gas = R_gas
        self.Re_delta_star = Re_delta_star

        self.T_ratio_wall = T_wall / T_edge
        self.T_recovery = T_edge * (1 + 0.5 * (gamma - 1) * Pr**0.5 * Ma**2)

        self._solve(n_points, eta_max)

    def _solve(self, n_points, eta_max):
        Ma = self.Ma
        gm1 = self.gamma - 1
        g_w = self.T_ratio_wall

        def g_from_fp(fp):
            return g_w + (1 - g_w) * fp + 0.5 * gm1 * Ma**2 * fp * (1 - fp)

        def dg_dfp(fp):
            return (1 - g_w) + 0.5 * gm1 * Ma**2 * (1 - 2 * fp)

        def ode(eta, u):
            f, fp, fpp = u
            g = g_from_fp(fp)
            g = max(g, 0.01)
            C = g**(self.omega - 1)
            dg = dg_dfp(fp)
            dCdg = (self.omega - 1) * g**(self.omega - 2) if g > 0.01 else 0.0
            fppp = -(f * fpp + dCdg * dg * fpp**2) / C
            return [fp, fpp, fppp]

        def shoot(fpp0):
            try:
                sol = solve_ivp(ode, [0, eta_max], [0, 0, fpp0],
                                method='RK45', rtol=1e-10, atol=1e-12,
                                max_step=0.05)
                return sol.y[1, -1] - 1.0
            except Exception:
                return 1.0

        fpp0 = brentq(shoot, 0.01, 2.0, xtol=1e-12)

        sol = solve_ivp(ode, [0, eta_max], [0, 0, fpp0],
                        method='RK45', dense_output=True,
                        rtol=1e-12, atol=1e-14, max_step=0.005)

        eta = np.linspace(0, eta_max, n_points)
        w = sol.sol(eta)

        self._eta = eta
        f_arr = w[0]
        fp_arr = w[1]    # U/U_e
        fpp_arr = w[2]
        g_arr = g_from_fp(fp_arr)  # T/T_e
        g_arr = np.maximum(g_arr, 0.01)

        # ============================================================
        # PHYSICAL COORDINATE MAPPING
        # y_phys(eta) = integral_0^eta g(s) ds
        # (because dy = (rho_e/rho) deta = g deta for the Lees-Dorodnitsyn transform)
        # ============================================================
        y_phys = np.zeros(n_points)
        y_phys[1:] = cumulative_trapezoid(g_arr, eta)

        # Physical displacement thickness
        # delta* = integral_0^inf (1 - rho*U/(rho_e*U_e)) dy
        # = integral_0^inf (1 - fp/g) * g deta = integral_0^inf (g - fp) deta
        self._delta_star = np.trapz(g_arr - fp_arr, eta)

        # Normalize physical coordinate by delta*
        y_nd = y_phys / self._delta_star

        # ============================================================
        # COMPUTE PROFILES IN PHYSICAL COORDINATES
        # d/dy = (1/g) * d/deta
        # d^2/dy^2 = (1/g^2)*d^2/deta^2 - (g'/(g^3))*d/deta
        # ============================================================
        # Derivative of g w.r.t. eta
        dg_deta = dg_dfp(fp_arr) * fpp_arr

        # f''' from the ODE
        g_safe = np.maximum(g_arr, 0.01)
        C_arr = g_safe**(self.omega - 1)
        dCdg_arr = (self.omega - 1) * g_safe**(self.omega - 2)
        fppp_arr = -(f_arr * fpp_arr + dCdg_arr * dg_dfp(fp_arr) * fpp_arr**2) / C_arr

        # Second derivative of g w.r.t. eta
        d2g_dfp2 = -gm1 * Ma**2
        d2g_deta2 = d2g_dfp2 * fpp_arr**2 + dg_dfp(fp_arr) * fppp_arr

        # U = f'(eta), dU/dy = f''/g, d2U/dy2 = f'''/g^2 - f''*g'/(g^3)
        U_arr = fp_arr
        dU_arr = fpp_arr / g_arr
        d2U_arr = fppp_arr / g_arr**2 - fpp_arr * dg_deta / g_arr**3

        # T = g(eta), dT/dy = g'/g, d2T/dy2 = g''/g^2 - (g')^2/g^3
        T_arr = g_arr
        dT_arr = dg_deta / g_arr
        d2T_arr = d2g_deta2 / g_arr**2 - dg_deta**2 / g_arr**3

        # rho = 1/g, drho/dy = -(g'/g^2)/g = -g'/g^3... wait
        # rho = rho_e/T_bar_e * T_e/T = 1/g (from rho*T = const at const pressure)
        rho_arr = 1.0 / g_arr
        drho_arr = -dg_deta / g_arr**3  # drho/dy = d(1/g)/dy = -(1/g^2)(g'/g) = -g'/g^3
        # Actually: drho/dy = d(1/g)/dy = -(1/g^2)*dg/dy = -(1/g^2)*(g'/g) = -g'/(g^3)
        # And g' = dg/deta, dg/dy = g'/g
        drho_arr = -(dg_deta / g_arr) / g_arr**2  # = -dT/dy * rho / T = -(g'/g) / g^2

        # mu = g^omega, dmu/dy = omega*g^(omega-1)*g'/g = omega*mu/g * g'/g
        mu_arr = g_arr**self.omega
        dmu_arr = self.omega * g_arr**(self.omega - 1) * dg_deta / g_arr

        # Store as y_nd-parameterized splines
        self._y_nd = y_nd
        self._build_splines(y_nd, U_arr, dU_arr, d2U_arr,
                            T_arr, dT_arr, d2T_arr,
                            rho_arr, mu_arr, dmu_arr)

    def _build_splines(self, y_nd, U, dU, d2U, T, dT, d2T, rho, mu, dmu):
        """Build splines parameterized by y/delta*."""
        self._spl = {}
        for name, data in [('U', U), ('dU', dU), ('d2U', d2U),
                           ('T', T), ('dT', dT), ('d2T', d2T),
                           ('rho', rho), ('mu', mu), ('dmu', dmu)]:
            self._spl[name] = CubicSpline(y_nd, data, extrapolate=True)

    def __call__(self, y):
        """Evaluate compressible mean flow at y/delta* locations.

        Parameters
        ----------
        y : array
            Non-dimensional wall-normal coordinate y/delta*.

        Returns
        -------
        dict with U, dU, d2U, T, dT, d2T, rho, mu, dmu
        """
        y = np.asarray(y, dtype=float)
        result = {}
        for name in ['U', 'dU', 'd2U', 'T', 'dT', 'd2T', 'rho', 'mu', 'dmu']:
            result[name] = np.asarray(self._spl[name](y))

        # Enforce physical bounds
        result['U'] = np.clip(result['U'], 0, 1)
        result['T'] = np.clip(result['T'], 0.5, None)
        result['rho'] = np.clip(result['rho'], 0.01, None)
        result['mu'] = np.clip(result['mu'], 0.01, None)

        return result
