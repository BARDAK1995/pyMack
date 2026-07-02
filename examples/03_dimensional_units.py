"""From solver units to laboratory units.

pyMack's solvers are dimensionless (Mack's L* scaling). This example converts
a nondimensional result into kHz, millimetres, and 1/m for a concrete wind-
tunnel-like edge state -- no solve required.

Run:  python examples/03_dimensional_units.py
"""

import math

try:
    import pymack as pm
except ModuleNotFoundError:  # running from a repo checkout without install
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import pymack as pm

# Edge conditions: Mach 6 nitrogen-like flow (values from the collaborator
# benchmark in reference_data/collaborator_mach5p35).
edge = pm.DimensionalEdgeState(
    U_e=857.0,            # m/s
    nu_e=7.2877e-5,       # m^2/s  (kinematic viscosity)
)

R = 5500.0               # R = sqrt(Re_x)
F = 3.0e-5               # reduced frequency
alpha_L = 0.174          # nondimensional wavenumber
sigma_L = 4.3e-3         # nondimensional spatial growth

print(f'station:      R = {R:.0f}  ->  x = {pm.R_L_to_x_mm(R, edge):8.1f} mm')
print(f'frequency:    F = {F:.1e}  ->  f = {pm.F_to_frequency_khz(F, edge):8.1f} kHz')
print(f'wavenumber:   alpha_L = {alpha_L}  ->  alpha = {pm.alpha_L_to_per_m(alpha_L, R, edge):8.1f} 1/m')
wavelength_L = 2.0 * math.pi / alpha_L    # lambda in L* units
print(f'              wavelength = {pm.wavelength_L_to_mm(wavelength_L, R, edge):.2f} mm')
print(f'growth:       sigma_L = {sigma_L}  ->  sigma = {pm.sigma_L_to_per_m(sigma_L, R, edge):8.2f} 1/m')
