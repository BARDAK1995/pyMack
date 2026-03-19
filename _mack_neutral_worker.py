"""Worker: compute Mack neutral curve for one Mach number using bisection.

Usage: python -u _mack_neutral_worker.py <Mach> <figure_type>

figure_type: 'alpha' for (R, alpha) neutral curve
             'freq'  for (R, F*1e4) neutral curve

Uses Sutherland viscosity (correct TS mode behavior).
Finds exact neutral points via bisection at each Re.
Coordinates: delta*-based (R = Re_delta*, alpha = alpha_delta*).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from chapters.ozgen_kircali_2008.ozgen import SutherlandBlasiusProfile
from lst.solver import solve_temporal_compressible

Ma = float(sys.argv[1])
fig_type = sys.argv[2] if len(sys.argv) > 2 else 'alpha'

# Mack's wind-tunnel temperatures
T_EDGE = {1.3: 311, 1.6: 311, 2.2: 311, 3.0: 311,
           3.8: 311, 4.5: 311, 4.8: 311,
           5.8: 50, 7.0: 50, 8.0: 50, 10.0: 50}
nearest_T = T_EDGE[min(T_EDGE, key=lambda k: abs(k - Ma))]
r = 0.72**0.5
T_aw = nearest_T * (1 + r * 0.2 * Ma**2)

print(f'M={Ma}, T_edge={nearest_T}K, T_aw={T_aw:.1f}K', flush=True)

# Create Sutherland profile (self-similar — one per Mach)
bf = SutherlandBlasiusProfile(Ma=Ma, T_wall=T_aw, T_edge=nearest_T,
                               gamma=1.4, Pr=0.72, S=110.4,
                               n_points=4000, eta_max=40.0)


def ci_at_alpha(alpha, Re, cr_range=(0.10, 0.50), N=120):
    """Get c_i for TS-like mode (first mode) at given (alpha, Re)."""
    y_max = 15.0 if Ma < 2 else (10.0 if Ma < 4 else 6.0)
    c, _, _ = solve_temporal_compressible(bf, alpha, Re, Ma, 0.72, 1.4,
                                          N=N, y_max=y_max)
    mask = (c.real > cr_range[0]) & (c.real < cr_range[1]) & (np.abs(c.imag) < 0.3)
    cands = c[mask]
    if len(cands) == 0:
        return np.nan, np.nan
    best = cands[np.argmax(cands.imag)]
    return best.imag, best.real  # ci, cr


def ci_second_mode(alpha, Re, N=120):
    """Get c_i for second mode (Mack mode) at given (alpha, Re)."""
    y_max = 8.0 if Ma < 6 else 6.0
    c, _, _ = solve_temporal_compressible(bf, alpha, Re, Ma, 0.72, 1.4,
                                          N=N, y_max=y_max)
    # Second mode: c_r between sonic speed and ~0.98
    c_s = 1 - 1.0 / Ma
    cr_lo = max(0.85, c_s - 0.05)
    cr_hi = min(0.98, c_s + 0.15)
    mask = (c.real > cr_lo) & (c.real < cr_hi) & (np.abs(c.imag) < 0.3)
    cands = c[mask]
    if len(cands) == 0:
        return np.nan, np.nan
    # Isolation filter
    best_ci = -np.inf
    best_cr = np.nan
    for cc in cands:
        n_near = np.sum(np.abs(c - cc) < 0.015) - 1
        if n_near <= 2 and cc.imag > best_ci:
            best_ci = cc.imag
            best_cr = cc.real
    if best_ci == -np.inf:
        return np.nan, np.nan
    return best_ci, best_cr


def bisect_neutral(ci_func, a_lo, a_hi, ci_lo, ci_hi, tol=5e-4, max_iter=25):
    """Bisect to find alpha where c_i = 0."""
    for _ in range(max_iter):
        a_mid = (a_lo + a_hi) / 2
        ci_mid, _ = ci_func(a_mid)
        if np.isnan(ci_mid):
            return np.nan, np.nan
        if abs(a_hi - a_lo) < tol:
            break
        if ci_mid * ci_lo < 0:
            a_hi = a_mid
            ci_hi = ci_mid
        else:
            a_lo = a_mid
            ci_lo = ci_mid
    a_n = (a_lo + a_hi) / 2
    _, cr_n = ci_func(a_n)
    return a_n, cr_n


def find_neutrals_at_Re(Re, ci_func, alpha_range, n_coarse=30):
    """Find all neutral alpha values at one Re. Returns [(alpha, cr), ...]."""
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_coarse)
    cis = np.zeros(n_coarse)
    crs = np.zeros(n_coarse)
    for i, a in enumerate(alphas):
        cis[i], crs[i] = ci_func(a, Re)
        if np.isnan(cis[i]):
            cis[i] = -0.1  # treat as stable

    neutrals = []
    for i in range(len(cis) - 1):
        if cis[i] * cis[i + 1] < 0:
            a_n, cr_n = bisect_neutral(
                lambda a: ci_func(a, Re),
                alphas[i], alphas[i + 1], cis[i], cis[i + 1])
            if np.isfinite(a_n):
                neutrals.append((a_n, cr_n))
    return neutrals


# ============================================================
# Re sweep — find neutral curve
# ============================================================

# Alpha range depends on mode
if Ma <= 3.0:
    # First mode only
    alpha_range_1st = (0.05, 0.45)
    Re_arr = np.concatenate([
        np.arange(200, 600, 25),
        np.arange(600, 1200, 50),
        np.arange(1200, 3001, 100)])
    do_second = False
elif Ma <= 5.0:
    # Both first and second mode
    alpha_range_1st = (0.02, 0.20)
    alpha_range_2nd = (0.05, 0.30)
    Re_arr = np.concatenate([
        np.arange(100, 500, 25),
        np.arange(500, 1500, 50),
        np.arange(1500, 3001, 100)])
    do_second = True
else:
    # Primarily second mode
    alpha_range_1st = (0.02, 0.15)
    alpha_range_2nd = (0.02, 0.25)
    Re_arr = np.concatenate([
        np.arange(100, 500, 25),
        np.arange(500, 2000, 50),
        np.arange(2000, 5001, 200)])
    do_second = True

print(f'Re range: [{Re_arr[0]:.0f}, {Re_arr[-1]:.0f}], {len(Re_arr)} pts', flush=True)

# First mode neutral curve
pts_1st = []  # (Re, alpha, cr, F)
for idx, Re in enumerate(Re_arr):
    neutrals = find_neutrals_at_Re(Re, ci_at_alpha, alpha_range_1st)
    for a_n, cr_n in neutrals:
        F = a_n * cr_n / Re  # frequency parameter
        pts_1st.append((Re, a_n, cr_n, F))
    if (idx + 1) % 10 == 0:
        print(f'  1st mode: {idx+1}/{len(Re_arr)} Re, {len(pts_1st)} pts', flush=True)
print(f'First mode: {len(pts_1st)} neutral points', flush=True)

# Second mode neutral curve
pts_2nd = []
if do_second:
    for idx, Re in enumerate(Re_arr):
        neutrals = find_neutrals_at_Re(Re, ci_second_mode, alpha_range_2nd)
        for a_n, cr_n in neutrals:
            F = a_n * cr_n / Re
            pts_2nd.append((Re, a_n, cr_n, F))
        if (idx + 1) % 10 == 0:
            print(f'  2nd mode: {idx+1}/{len(Re_arr)} Re, {len(pts_2nd)} pts', flush=True)
    print(f'Second mode: {len(pts_2nd)} neutral points', flush=True)

# Save
outpath = os.path.join('chapters', 'ch10_compressible_viscous',
                        f'neutral_M{Ma:.1f}.npz')
np.savez(outpath,
         first=np.array(pts_1st) if pts_1st else np.zeros((0, 4)),
         second=np.array(pts_2nd) if pts_2nd else np.zeros((0, 4)))
print(f'Saved: {outpath}', flush=True)
