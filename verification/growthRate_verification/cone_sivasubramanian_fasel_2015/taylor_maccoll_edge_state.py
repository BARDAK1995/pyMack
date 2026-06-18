"""Taylor-Maccoll post-shock cone edge state for a sharp 7-deg cone at M_inf=6.

Perfect gas, gamma=1.4. Freestream from the BAM6QT canonical table (design
review arxiv:2509.10411, Table 2, case 3) which matches Sivasubramanian &
Fasel (2015) sharp-cone DNS to ~3.5% in unit Reynolds number:

    M_inf = 6.0, T0 = 430 K, T_inf = 52.4 K, V_inf = 871 m/s, Tw = 300 K,
    Re/m (case3) = 11.2e6 /m   (S&F 2015 used 10.82e6 /m)

We integrate the Taylor-Maccoll ODE from the oblique (conical) shock to the
cone surface to find the SHOCK angle that makes the flow tangent to a 7-deg
cone, then read the SURFACE = boundary-layer-edge state.
"""
import os
os.environ.setdefault("PYMACK_NO_BANNER", "1")
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

GAMMA = 1.4
M_INF = 6.0
THETA_C_DEG = 7.0
T_INF = 52.4          # K
V_INF = 871.0         # m/s  (freestream velocity from table)
RE_UNIT_INF = 11.2e6  # 1/m  freestream unit Reynolds
T_WALL = 300.0
R_AIR = 287.05        # J/kg/K
# Sutherland (air)
MU_REF = 1.716e-5; T_REF = 273.15; S_SUTH = 110.4

def mu_sutherland(T):
    return MU_REF * (T / T_REF) ** 1.5 * (T_REF + S_SUTH) / (T + S_SUTH)

a_inf = np.sqrt(GAMMA * R_AIR * T_INF)
# Use the consistent velocity from M_inf*a_inf; compare with table's 871.
V_inf_consistent = M_INF * a_inf
T0 = T_INF * (1 + 0.5*(GAMMA-1)*M_INF**2)

# Total (max) velocity Vmax = sqrt(2 h0) = sqrt(2 cp T0); nondim by Vmax.
cp = GAMMA * R_AIR / (GAMMA - 1)
Vmax = np.sqrt(2 * cp * T0)

def V_from_M(M):
    # V/Vmax as function of local Mach (energy: V^2/Vmax^2 = [(g-1)/2 M^2]/[1+(g-1)/2 M^2])
    k = 0.5*(GAMMA-1)*M*M
    return np.sqrt(k/(1+k))

def oblique_shock(M1, beta):
    """Return (M2, deflection theta, p2/p1, T2/T1, rho2/rho1) for shock angle beta."""
    Mn1 = M1*np.sin(beta)
    Mn2 = np.sqrt((1 + 0.5*(GAMMA-1)*Mn1**2)/(GAMMA*Mn1**2 - 0.5*(GAMMA-1)))
    p2p1 = 1 + 2*GAMMA/(GAMMA+1)*(Mn1**2 - 1)
    rho2rho1 = (GAMMA+1)*Mn1**2/((GAMMA-1)*Mn1**2 + 2)
    T2T1 = p2p1/rho2rho1
    tan_theta = 2/np.tan(beta)*(M1**2*np.sin(beta)**2 - 1)/(M1**2*(GAMMA+np.cos(2*beta))+2)
    theta = np.arctan(tan_theta)
    M2 = Mn2/np.sin(beta - theta)
    return M2, theta, p2p1, T2T1, rho2rho1

def taylor_maccoll_rhs(theta, y):
    # y = [Vr, Vtheta] nondim by Vmax. Standard TM equations.
    Vr, Vth = y
    # a^2/Vmax^2 = (g-1)/2 (1 - Vr^2 - Vth^2)
    a2 = 0.5*(GAMMA-1)*(1 - Vr**2 - Vth**2)
    dVr = Vth
    num = Vr*Vth**2 - a2*(2*Vr + Vth/np.tan(theta))
    den = a2 - Vth**2
    dVth = num/den
    return [dVr, dVth]

def surface_Vth(beta):
    """Integrate TM from shock to cone surface; return Vtheta at theta_c (=0 when tangent)."""
    M2, defl, p2p1, T2T1, rho2rho1 = oblique_shock(M_INF, beta)
    V2 = V_from_M(M2)  # /Vmax
    # flow direction behind shock is deflected by 'defl' from freestream;
    # at the shock (theta=beta) decompose V2 into radial/normal-to-ray comps.
    flow_ang = defl
    Vr = V2*np.cos(beta - flow_ang)
    Vth = -V2*np.sin(beta - flow_ang)
    sol = solve_ivp(taylor_maccoll_rhs, [beta, np.radians(THETA_C_DEG)],
                    [Vr, Vth], rtol=1e-10, atol=1e-12, dense_output=True, max_step=1e-3)
    return sol.y[1, -1], sol

# Find shock angle beta s.t. Vtheta=0 at the cone surface (flow tangent to cone).
beta_lo = np.radians(THETA_C_DEG + 0.5)
beta_hi = np.radians(40.0)
beta_sol = brentq(lambda b: surface_Vth(b)[0], beta_lo, beta_hi, xtol=1e-10)
Vth_end, sol = surface_Vth(beta_sol)

# Surface state
Vr_s = sol.y[0, -1]; Vth_s = sol.y[1, -1]
Vmag_s = np.sqrt(Vr_s**2 + Vth_s**2)  # /Vmax
a2_s = 0.5*(GAMMA-1)*(1 - Vmag_s**2)
M_e = Vmag_s/np.sqrt(a2_s)
# Temperature ratio from energy: T/T0 = a^2/a0^2 = (1 - Vmag^2) since a0^2/Vmax^2=(g-1)/2
T_e = T0 * (1 - Vmag_s**2)
U_e = Vmag_s * Vmax   # surface velocity magnitude (along cone ray) in m/s
a_e = np.sqrt(GAMMA*R_AIR*T_e)

# Edge density / unit Reynolds. Across shock + isentropic compression to surface.
M2, defl, p2p1, T2T1, rho2rho1 = oblique_shock(M_INF, beta_sol)
# p0 is conserved (adiabatic) but stagnation pressure drops across shock.
# Surface static p: use isentropic from post-shock stagnation. Simpler: use
# p_e/p_inf via T and the fact entropy constant downstream of shock (conical
# flow isentropic between shock and surface).
# Post-shock state:
T2 = T_INF*T2T1
p_inf = RE_UNIT_INF*0  # placeholder; compute rho_inf from ideal gas with p_inf
# freestream pressure from table case3: p_inf = 654.9 Pa
p_inf = 654.9
rho_inf = p_inf/(R_AIR*T_INF)
p2 = p_inf*p2p1
rho2 = rho_inf*rho2rho1
# isentropic from post-shock (state 2) to surface (state e): p/rho^g const, T/p^((g-1)/g) const
T2_K = T2
p_e = p2*(T_e/T2_K)**(GAMMA/(GAMMA-1))
rho_e = p_e/(R_AIR*T_e)
mu_e = mu_sutherland(T_e)
nu_e = mu_e/rho_e
Re_unit_e = U_e/nu_e

print(f"shock angle beta = {np.degrees(beta_sol):.4f} deg  (deflection check defl={np.degrees(defl):.4f} vs cone 7)")
print(f"a_inf={a_inf:.3f} m/s  V_inf(M*a)={V_inf_consistent:.2f} m/s (table 871)  T0={T0:.2f} K  Vmax={Vmax:.2f} m/s")
print(f"--- POST-SHOCK CONE EDGE (surface) ---")
print(f"M_e   = {M_e:.4f}")
print(f"T_e   = {T_e:.3f} K")
print(f"U_e   = {U_e:.3f} m/s")
print(f"a_e   = {a_e:.3f} m/s   (M_e*a_e={M_e*a_e:.2f})")
print(f"p_e   = {p_e:.3f} Pa   rho_e={rho_e:.5f} kg/m3")
print(f"mu_e  = {mu_e:.4e} kg/m/s   nu_e={nu_e:.6e} m2/s")
print(f"Re_unit_e = {Re_unit_e:.4e} /m   (freestream {RE_UNIT_INF:.3e}/m)")
print(f"Tw/Te = {T_WALL/T_e:.4f}")
