"""
Compressible stability equation coefficient matrices.

Assembles the linearized Navier-Stokes perturbation equations
for a parallel compressible boundary layer as a quadratic eigenvalue problem:

    (C0 + alpha * C1 + alpha^2 * C2) * phi = 0

State vector: phi = [u_hat, v_hat, T_hat, p_hat]^T
Normal mode: q' = q_hat(y) * exp[i(alpha*x - omega*t)]

Non-dimensionalization: edge values (U_e, T_e, rho_e, mu_e, delta*).
Pressure normalized by rho_e * U_e^2.

Energy equation uses the enthalpy form (Form 1):
    rho DT/Dt = (gamma-1)*Ma^2 * Dp/Dt + (1/(Pr*Re))*div(k*grad T)
                + ((gamma-1)*Ma^2/Re) * Phi
"""

import numpy as np


def assemble_compressible_matrices(D1, D2, y, baseflow, omega, Re, Ma, Pr, gamma):
    """Build C0, C1, C2 coefficient matrices for spatial stability.

    All equations written in the form L*phi = 0 with L = C0 + alpha*C1 + alpha^2*C2.
    Equations are divided by rho_bar for better conditioning.

    Parameters
    ----------
    D1, D2 : (n, n) arrays
        Physical-space derivative matrices.
    y : (n,) array
        Physical grid points.
    baseflow : dict
        Mean flow: U, dU, d2U, T, dT, d2T, rho, mu, dmu.
    omega : float
        Angular frequency (real for spatial analysis).
    Re : float
        Reynolds number.
    Ma : float
        Edge Mach number.
    Pr : float
        Prandtl number.
    gamma : float
        Specific heat ratio.

    Returns
    -------
    C0, C1, C2 : (4n, 4n) complex arrays
    """
    n = len(y)
    I = np.eye(n)

    # Mean flow diagonal matrices
    Ub = np.diag(baseflow['U'])
    dUb = np.diag(baseflow['dU'])
    d2Ub = np.diag(baseflow['d2U'])
    Tb = np.diag(baseflow['T'])
    dTb = np.diag(baseflow['dT'])
    rhob = np.diag(baseflow['rho'])
    mub = np.diag(baseflow['mu'])
    dmub = np.diag(baseflow['dmu'])

    T_v = baseflow['T']
    rho_v = baseflow['rho']
    mu_v = baseflow['mu']
    dT_v = baseflow['dT']
    dU_v = baseflow['dU']

    rhoI = np.diag(1.0 / rho_v)
    TI = np.diag(1.0 / T_v)

    # Viscosity power law: mu = T^omega_v, dmu/dT = omega_v * mu/T
    log_T = np.log(np.maximum(T_v, 1e-30))
    log_mu = np.log(np.maximum(mu_v, 1e-30))
    omega_v = np.where(
        np.abs(log_T) > 1e-10,
        log_mu / log_T,
        0.74)
    omega_v = np.clip(omega_v, 0.5, 1.0)
    dmu_dT_v = omega_v * mu_v / T_v
    d2mu_dT2_v = omega_v * (omega_v - 1) * mu_v / T_v**2
    dmu_dT = np.diag(dmu_dT_v)
    d2mu_dT2 = np.diag(d2mu_dT2_v)

    # Conductivity (same power law, kappa/kappa_e = mu/mu_e for const Pr)
    kappab = mub
    dkappab = dmub
    dkappa_dT = dmu_dT

    gm1 = gamma - 1.0
    Ma2 = Ma**2
    iw = 1j * omega

    # Prefactors (all divided by rho_bar)
    visc = rhoI / Re                          # viscous: 1/(Re*rho)
    cond = rhoI / (Pr * Re)                   # conduction: 1/(Pr*Re*rho)
    diss = gm1 * Ma2 * rhoI / Re             # dissipation: (g-1)*Ma^2/(Re*rho)

    # Initialize complex matrices
    C0 = np.zeros((4*n, 4*n), dtype=complex)
    C1 = np.zeros((4*n, 4*n), dtype=complex)
    C2 = np.zeros((4*n, 4*n), dtype=complex)

    def blk(i, j):
        return (slice(i*n, (i+1)*n), slice(j*n, (j+1)*n))

    # =================================================================
    # CONTINUITY (row 0)
    # (-iw + i*alpha*U)*(gamma*Ma^2*p - T/T_bar) + i*alpha*u + Dv - (DT/T)*v = 0
    # =================================================================
    C0[blk(0, 1)] = D1 - TI @ dTb                  # Dv - (DT/T)*v
    C0[blk(0, 2)] = iw * TI                         # iw*T_hat/T (from -iw*(-T/T))
    C0[blk(0, 3)] = -iw * gamma * Ma2 * I           # -iw*gamma*Ma^2*p

    C1[blk(0, 0)] = 1j * I                          # i*u
    C1[blk(0, 2)] = -1j * Ub @ TI                   # -i*U*T/T
    C1[blk(0, 3)] = 1j * gamma * Ma2 * Ub           # i*gamma*Ma^2*U*p

    # =================================================================
    # X-MOMENTUM (row 1) — divided by rho_bar
    # (-iw+i*alpha*U)*u + DU*v + i*alpha*p/rho
    # - visc*[mu*D^2(u) + Dmu*D(u) + dmu_dT*DT*D(u)   (C0 on u)
    #         - (4/3)*alpha^2*mu*u                       (C2 on u)
    #         + (i*alpha/3)*(mu*D(v) + Dmu*v)            (C1 on v)
    #         + dmu_dT*(D^2U*T + DU*D(T))               (C0 on T)
    #         + d2mu_dT2*DU*DT*T]                        (C0 on T)
    # = 0
    # =================================================================
    C0[blk(1, 0)] = (-iw * I
                      - visc @ (mub @ D2 + dmub @ D1 + dmu_dT @ dTb @ D1))
    C0[blk(1, 1)] = dUb
    C0[blk(1, 2)] = -visc @ (dmu_dT @ d2Ub + d2mu_dT2 @ dUb @ dTb
                              + dmu_dT @ dUb @ D1)

    C1[blk(1, 0)] = 1j * Ub
    C1[blk(1, 1)] = -(1j / 3.0) * visc @ (mub @ D1 + dmub)
    C1[blk(1, 3)] = 1j * rhoI                        # +i*p/rho (NO gamma*Ma^2!)

    C2[blk(1, 0)] = (4.0 / 3.0) * visc @ mub        # +(4/3)*visc*mu (positive!)

    # =================================================================
    # Y-MOMENTUM (row 2) — divided by rho_bar
    # (-iw+i*alpha*U)*v + Dp/rho
    # - visc*[mu*D^2(v) + (4/3)*Dmu*D(v) + dmu_dT*DT*D(v)   (C0 on v)
    #         - alpha^2*mu*v                                    (C2 on v)
    #         + (i*alpha/3)*(mu*D(u) + Dmu*u)]                  (C1 on u)
    # = 0
    # =================================================================
    C0[blk(2, 1)] = (-iw * I
                      - visc @ (mub @ D2 + (4.0/3.0) * dmub @ D1
                                + dmu_dT @ dTb @ D1))
    C0[blk(2, 3)] = rhoI @ D1                        # +Dp/rho (NO gamma*Ma^2!)

    C1[blk(2, 0)] = -(1j / 3.0) * visc @ (mub @ D1 + dmub)
    C1[blk(2, 1)] = 1j * Ub

    C2[blk(2, 1)] = visc @ mub                       # +visc*mu (positive!)

    # =================================================================
    # ENERGY (row 3) — Form 1 (enthalpy), divided by rho_bar
    # (-iw+i*alpha*U)*T + DT*v
    # - (g-1)*Ma^2*(-iw+i*alpha*U)*p/rho
    # - cond*[kappa*D^2(T) + Dkappa*D(T) + dkappa_dT*DT*D(T)
    #         - alpha^2*kappa*T]
    # - diss*[2*mu*DU*(D(u) + i*alpha*v)]
    # = 0
    # =================================================================
    C0[blk(3, 0)] = -diss @ (2.0 * mub @ dUb @ D1)   # -diss*2*mu*DU*D(u)
    C0[blk(3, 1)] = np.diag(dT_v)                     # DT*v (convection of mean T)
    C0[blk(3, 2)] = (-iw * I
                      - cond @ (kappab @ D2 + dkappab @ D1
                                + dkappa_dT @ dTb @ D1))
    C0[blk(3, 3)] = iw * gm1 * Ma2 * rhoI            # +iw*(g-1)*Ma^2*p/rho

    C1[blk(3, 1)] = -2j * diss @ (mub @ dUb)          # -i*diss*2*mu*DU*v (dissip.)
    C1[blk(3, 2)] = 1j * Ub                            # i*U*T
    C1[blk(3, 3)] = -1j * gm1 * Ma2 * Ub @ rhoI       # -i*(g-1)*Ma^2*U*p/rho

    C2[blk(3, 2)] = cond @ kappab                      # +cond*kappa (positive!)

    return C0, C1, C2


def assemble_orr_sommerfeld(D1, D2, y, baseflow, Re):
    """Build Orr-Sommerfeld matrices for temporal analysis: A*v = c*B*v.

    Standard O-S: (D^2-alpha^2)^2 v = i*alpha*Re*[(U-c)(D^2-alpha^2) - U''] v
    """
    n = len(y)
    I_mat = np.eye(n)
    U_diag = np.diag(baseflow['U'])
    d2U_diag = np.diag(baseflow['d2U'])
    D4 = D2 @ D2

    def build_evp(alpha):
        alpha2 = alpha**2
        L2 = D2 - alpha2 * I_mat
        L4 = D4 - 2 * alpha2 * D2 + alpha2**2 * I_mat

        # A = -L4/(i*alpha*Re) + U*L2 - U'' = i*L4/(alpha*Re) + U*L2 - U''
        A = -L4 / (1j * alpha * Re) + U_diag @ L2 - d2U_diag
        B = L2.copy()

        # BCs: v=0, Dv=0 at wall (y[-1]=0) and freestream (y[0]=y_max)
        A[-1, :] = 0;    A[-1, -1] = 1;    B[-1, :] = 0
        A[-2, :] = D1[-1, :];              B[-2, :] = 0
        A[0, :] = 0;     A[0, 0] = 1;      B[0, :] = 0
        A[1, :] = D1[0, :];                B[1, :] = 0

        return A, B

    return build_evp
