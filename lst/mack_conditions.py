"""Reference-condition helpers for Mack AGARD 709 reproductions.

Mack's report does not use a single universal external-temperature schedule for
all figures and tables. Two distinct condition sets matter in this repository:

- ``table_11_1``:
  An inferred low-/mid-Mach total-temperature fit that reproduces the
  displacement-thickness values in Mack's Table 11.1 when combined with the
  adiabatic-wall mean-flow solver and Appendix-A transport model.
- ``wind_tunnel``:
  The figure-caption conditions used throughout Chapters 9-11, where the
  low-/mid-Mach cases use ``T_1^* = 311 K`` and the hypersonic cases use
  ``T_1^* = 50 K``.

These helpers make that distinction explicit so the chapter scripts stop mixing
benchmark and figure conditions.
"""

from __future__ import annotations

import numpy as np

from .baseflow import CompressibleBlasiusProfile


MACK_FIGURE_EDGE_TEMPERATURES = {
    1.3: 311.0,
    1.6: 311.0,
    2.2: 311.0,
    3.0: 311.0,
    3.8: 311.0,
    4.5: 311.0,
    4.8: 311.0,
    5.8: 50.0,
    7.0: 50.0,
    8.0: 50.0,
    10.0: 50.0,
}


def mack_table_11_1_edge_temperature(
    Ma,
    gamma=1.4,
    reference_total_temperature=305.0,
    hypersonic_edge_temperature=50.0,
    hypersonic_switch_mach=5.0,
):
    """Return the edge temperature schedule that reproduces Table 11.1.

    For ``M_1 < 5``, the low-/mid-Mach cases are well matched by a constant
    total-temperature fit ``T_0^* ~= 305 K``:

        T_1^* = T_0^* / (1 + 0.5 * (gamma - 1) * M_1^2)

    For the hypersonic cases in the table, Mack's report uses ``T_1^* = 50 K``.
    """
    Ma = float(Ma)
    if Ma >= hypersonic_switch_mach:
        return hypersonic_edge_temperature
    return reference_total_temperature / (1.0 + 0.5 * (gamma - 1.0) * Ma**2)


def mack_figure_edge_temperature(
    Ma,
    low_mid_edge_temperature=311.0,
    hypersonic_edge_temperature=50.0,
    hypersonic_switch_mach=5.0,
):
    """Return the figure-caption wind-tunnel edge temperature for a Mach number."""
    Ma = float(Ma)
    if Ma in MACK_FIGURE_EDGE_TEMPERATURES:
        return MACK_FIGURE_EDGE_TEMPERATURES[Ma]
    if Ma >= hypersonic_switch_mach:
        return hypersonic_edge_temperature
    return low_mid_edge_temperature


def resolve_mack_edge_temperature(Ma, condition='wind_tunnel'):
    """Resolve ``T_1^*`` for a named Mack reproduction condition set."""
    if condition in {'wind_tunnel', 'figure', 'figures'}:
        return mack_figure_edge_temperature(Ma)
    if condition in {'table_11_1', 'table'}:
        return mack_table_11_1_edge_temperature(Ma)
    raise ValueError(
        "condition must be 'wind_tunnel'/'figure' or 'table_11_1'/'table'"
    )


def mack_adiabatic_wall_temperature(Ma, T_edge, Pr=0.72, gamma=1.4):
    """Return adiabatic-wall temperature from the standard recovery relation."""
    recovery = Pr**0.5
    return T_edge * (1.0 + recovery * 0.5 * (gamma - 1.0) * Ma**2)


def make_mack_profile(
    Ma,
    T_edge=None,
    Tw_ratio=None,
    condition='wind_tunnel',
    gamma=1.4,
    Pr=0.72,
    omega=0.74,
    Re_delta_star=1000,
):
    """Build a Mack-style compressible flat-plate profile for reproductions."""
    if T_edge is None:
        T_edge = resolve_mack_edge_temperature(Ma, condition=condition)

    T_recovery = mack_adiabatic_wall_temperature(Ma, T_edge, Pr=Pr, gamma=gamma)
    if Tw_ratio is None:
        return CompressibleBlasiusProfile(
            Ma=Ma,
            T_wall=T_recovery,
            T_edge=T_edge,
            gamma=gamma,
            Pr=Pr,
            omega=omega,
            Re_delta_star=Re_delta_star,
            wall_bc='adiabatic',
            viscosity_model='mack',
        )

    target_T_wall = T_recovery * Tw_ratio
    if np.isclose(Tw_ratio, 1.0):
        return CompressibleBlasiusProfile(
            Ma=Ma,
            T_wall=target_T_wall,
            T_edge=T_edge,
            gamma=gamma,
            Pr=Pr,
            omega=omega,
            Re_delta_star=Re_delta_star,
            wall_bc='isothermal',
            viscosity_model='mack',
        )

    # Strongly cooled or heated walls converge more reliably through
    # isothermal continuation from the adiabatic solution.
    profile = CompressibleBlasiusProfile(
        Ma=Ma,
        T_wall=T_recovery,
        T_edge=T_edge,
        gamma=gamma,
        Pr=Pr,
        omega=omega,
        Re_delta_star=Re_delta_star,
        wall_bc='adiabatic',
        viscosity_model='mack',
    )
    profile = CompressibleBlasiusProfile(
        Ma=Ma,
        T_wall=T_recovery,
        T_edge=T_edge,
        gamma=gamma,
        Pr=Pr,
        omega=omega,
        Re_delta_star=Re_delta_star,
        wall_bc='isothermal',
        viscosity_model='mack',
        initial_guess_profile=profile,
    )
    n_steps = max(4, int(np.ceil(abs(Tw_ratio - 1.0) / 0.05)))
    for ratio in np.linspace(1.0, Tw_ratio, n_steps + 1)[1:]:
        profile = CompressibleBlasiusProfile(
            Ma=Ma,
            T_wall=T_recovery * ratio,
            T_edge=T_edge,
            gamma=gamma,
            Pr=Pr,
            omega=omega,
            Re_delta_star=Re_delta_star,
            wall_bc='isothermal',
            viscosity_model='mack',
            initial_guess_profile=profile,
        )
    return profile
