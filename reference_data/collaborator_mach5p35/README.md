# Mach 5.35 Collaborator LST Benchmark

This folder stores the external Mach 5.35 flat-plate LST neutral curve used in
the DNS-DSMC particulate collaboration and the AviationAbstract2 particulate
manuscript.

Files:

- `LST_neutral_curve_M5p35.dat`: raw supplied Tecplot-style curve, preserved as
  copied from the collaborator repository.
- `LST_neutral_curve_M5p35.csv`: normalized CSV for pyMack comparisons. It keeps
  the original `x0` and `x1` columns and adds `x_left` and `x_right` branch
  coordinates in meters and millimeters.
- `conditions.json`: flow conditions, units, provenance, data ranges, and file
  checksums.

The curve is dimensional. Frequency is in Hz in the raw file, while branch
locations are in meters. The normalized CSV also gives frequency in kHz and
branch locations in millimeters. At each frequency, `x_left` and `x_right` are
the two streamwise neutral-branch locations enclosing the unstable second-mode
band.

The corresponding flow condition is a Mach 5.35 nitrogen flat-plate boundary
layer with `T_inf = 64 K`, `U_inf = 857 m/s`, `P_inf = 1300 Pa`,
`rho_inf = 0.0684 kg/m^3`, and a `T_w = 370 K` adiabatic-wall approximation.
The 2025 LST conversion path for this curve uses `Re_inf/L = 11.76e6 1/m`,
while the newer DSMC/abstract condition table reports `1.1935e7 1/m` from
`Prot0.dat`; both values are recorded in `conditions.json`.

This is an external benchmark reference, not a pyMack-generated solution.
