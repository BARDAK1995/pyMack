"""Generate a standalone compressible flat-plate boundary-layer profile.

Thin CLI over :func:`pymack.boundary_layer.generate_boundary_layer`.

Examples
--------
Adiabatic Mach 4.5 air profile, summary to stdout::

    python scripts/generate_boundary_layer.py --ma 4.5

Cold isothermal wall with CSV/JSON export::

    python scripts/generate_boundary_layer.py --ma 6 --wall isothermal \
        --tw-over-te 0.5 --viscosity mack --csv profile.csv --json profile.json

Dimensional output at a station (requires the edge-state group)::

    python scripts/generate_boundary_layer.py --ma 6 --u-edge-m-s 900 \
        --nu-edge-m2-s 1.5e-5 --r-l 2000 --rho-edge-kg-m3 0.05 --csv profile.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pymack.boundary_layer import generate_boundary_layer  # noqa: E402
from pymack.scales import DimensionalEdgeState  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Generate a compressible flat-plate self-similar boundary-layer '
            'profile (standalone or as stability-solver input).'
        ),
    )
    parser.add_argument('--ma', type=float, required=True,
                        help='Edge Mach number.')

    wall = parser.add_argument_group('wall condition')
    wall.add_argument('--wall', choices=['adiabatic', 'isothermal'],
                      default='adiabatic', help='Thermal wall condition.')
    wall_spec = wall.add_mutually_exclusive_group()
    wall_spec.add_argument('--tw-over-te', type=float, default=None,
                           help='Wall temperature as a ratio of T_edge.')
    wall_spec.add_argument('--tw-over-taw', type=float, default=None,
                           help='Wall temperature as a ratio of the formula '
                                'recovery temperature Taw.')
    wall_spec.add_argument('--t-wall-k', type=float, default=None,
                           help='Absolute wall temperature in Kelvin.')

    gas = parser.add_argument_group('gas / transport')
    gas.add_argument('--t-edge-k', type=float, default=300.0,
                     help='Edge temperature in Kelvin (default 300).')
    gas.add_argument('--viscosity',
                     choices=['sutherland', 'power_law', 'mack'],
                     default='sutherland', help='Transport model.')
    gas.add_argument('--gas', choices=['air', 'nitrogen'], default='air',
                     help='Gas preset for gamma/Pr/Sutherland defaults.')
    gas.add_argument('--gamma', type=float, default=None,
                     help='Override preset specific-heat ratio.')
    gas.add_argument('--pr', type=float, default=None,
                     help='Override preset Prandtl number.')
    gas.add_argument('--omega-exp', type=float, default=0.74,
                     help='Power-law viscosity exponent (default 0.74).')
    gas.add_argument('--sutherland-s-k', type=float, default=None,
                     help='Sutherland constant in Kelvin (preset default).')

    num = parser.add_argument_group('numerics')
    num.add_argument('--n-points', type=int, default=4000,
                     help='Tabulated profile points (default 4000).')
    num.add_argument('--eta-max', type=float, default=40.0,
                     help='Similarity-domain extent (default 40).')
    num.add_argument('--no-continuation', action='store_true',
                     help='Disable the continuation fallback for difficult '
                          'isothermal walls (fail fast instead).')

    out = parser.add_argument_group('output')
    out.add_argument('--csv', type=Path, default=None,
                     help='Write the nondimensional profile table here. When '
                          'the dimensional edge-state group is also given, a '
                          'companion SI table is written to <stem>_si.csv.')
    out.add_argument('--json', type=Path, default=None,
                     help='Write the condition/scalar record here.')
    out.add_argument('--scale', choices=['L_star', 'delta_star'],
                     default='L_star',
                     help='Derivative scale for the CSV table '
                          '(default L_star).')
    out.add_argument('--plot', type=Path, nargs='?', const=None,
                     default=argparse.SUPPRESS,
                     help='Write a U/T/rho profile figure (optional path; '
                          'defaults next to --csv or ./boundary_layer.png).')
    out.add_argument('--quiet', action='store_true',
                     help='Suppress progress messages.')

    dim = parser.add_argument_group(
        'dimensional station (all-or-nothing: edge velocity, edge viscosity, '
        'and one of --x-m/--r-l)')
    dim.add_argument('--u-edge-m-s', type=float, default=None,
                     help='Edge velocity U_e in m/s.')
    dim.add_argument('--nu-edge-m2-s', type=float, default=None,
                     help='Edge kinematic viscosity nu_e in m^2/s.')
    station = dim.add_mutually_exclusive_group()
    station.add_argument('--x-m', type=float, default=None,
                         help='Streamwise station in meters.')
    station.add_argument('--r-l', type=float, default=None,
                         help='Mack Reynolds number R_L = sqrt(Re_x).')
    dim.add_argument('--rho-edge-kg-m3', type=float, default=None,
                     help='Edge density (enables dimensional rho and mu).')

    return parser


def resolve_dimensional_request(args, parser):
    """Validate the all-or-nothing dimensional argument group."""
    dim_args = [args.u_edge_m_s, args.nu_edge_m2_s]
    station_given = (args.x_m is not None) or (args.r_l is not None)
    any_given = any(v is not None for v in dim_args) or station_given
    if not any_given:
        if args.rho_edge_kg_m3 is not None:
            parser.error('--rho-edge-kg-m3 requires the full dimensional '
                         'group (--u-edge-m-s, --nu-edge-m2-s, --x-m/--r-l)')
        return None
    if any(v is None for v in dim_args) or not station_given:
        parser.error('dimensional output requires --u-edge-m-s, '
                     '--nu-edge-m2-s, and one of --x-m/--r-l')
    return DimensionalEdgeState(
        U_e=args.u_edge_m_s, nu_e=args.nu_edge_m2_s, T_e=None, gas=args.gas)


def write_plot(result, path):
    """Write a U/T/rho versus y/L* profile figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.plot(result.U, result.y_over_Lstar, lw=2.0, label=r'$U/U_e$')
    ax.plot(result.T, result.y_over_Lstar, lw=2.0, label=r'$T/T_e$')
    ax.plot(result.rho, result.y_over_Lstar, lw=2.0, label=r'$\rho/\rho_e$')
    ax.axhline(result.delta_star_over_Lstar, color='0.5', ls='--', lw=1.2,
               label=r'$\delta^*/L^*$')
    ax.set_xlabel('edge-normalized quantity', fontsize=14)
    ax.set_ylabel(r'$y / L^*$', fontsize=14)
    ax.set_ylim(0.0, min(result.y_over_Lstar[-1],
                         3.0 * result.delta_star_over_Lstar))
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    ax.set_title(
        f'M={result.Ma:g}, {result.wall_bc}, '
        f'$T_w/T_e$={result.Tw_over_Te:.2f} ({result.viscosity_model})',
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_summary(result, dimensional):
    print('pymack boundary-layer generator')
    print(f'  Ma                = {result.Ma:g}')
    print(f'  wall condition    = {result.wall_bc} '
          f'({result.viscosity_model}, gas={result.gas})')
    print(f'  T_edge            = {result.T_edge_K:.2f} K')
    print(f'  Tw/Te             = {result.Tw_over_Te:.4f} '
          f'(T_wall = {result.T_wall_K:.2f} K'
          + (', solved adiabatic' if result.wall_bc == 'adiabatic' else '')
          + ')')
    print(f'  Taw/Te (formula)  = {result.Taw_over_Te_formula:.4f}')
    if result.recovery_factor_solved is not None:
        print(f'  recovery (solved) = {result.recovery_factor_solved:.4f}')
    print(f'  delta*/L*         = {result.delta_star_over_Lstar:.4f}')
    print(f'  theta/L*          = {result.theta_over_Lstar:.4f}')
    print(f'  H = delta*/theta  = {result.shape_factor_H:.4f}')
    print(f'  delta99/L*        = {result.delta99_over_Lstar:.4f}')
    print(f'  continuation used = {result.used_continuation} '
          f'({result.n_continuation_steps} ramp steps)')
    if dimensional is not None:
        print('  --- dimensional station ---')
        print(f'  x                 = {dimensional.x_m:.6g} m '
              f'(R_L = {dimensional.R_L:.6g}, Re_x = {dimensional.Re_x:.6g})')
        print(f'  L*                = {dimensional.L_star_m:.6g} m')
        print(f'  delta*            = {dimensional.delta_star_m:.6g} m')
        print(f'  theta             = {dimensional.theta_m:.6g} m')
        print(f'  delta99           = {dimensional.delta99_m:.6g} m')


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    edge_state = resolve_dimensional_request(args, parser)
    progress = None if args.quiet else (lambda msg: print(f'  [{msg}]'))

    try:
        result = generate_boundary_layer(
            args.ma,
            wall_bc=args.wall,
            Tw_over_Te=args.tw_over_te,
            Tw_over_Taw=args.tw_over_taw,
            T_wall_K=args.t_wall_k,
            T_edge_K=args.t_edge_k,
            viscosity_model=args.viscosity,
            gas=args.gas,
            gamma=args.gamma,
            Pr=args.pr,
            omega=args.omega_exp,
            sutherland_S_K=args.sutherland_s_k,
            n_points=args.n_points,
            eta_max=args.eta_max,
            continuation='never' if args.no_continuation else 'auto',
            progress=progress,
        )
    except (RuntimeError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1

    dimensional = None
    if edge_state is not None:
        dimensional = result.dimensionalize(
            edge_state, x_m=args.x_m, R_L=args.r_l,
            rho_e_kg_m3=args.rho_edge_kg_m3)

    if args.csv is not None:
        written = result.to_csv(args.csv, scale=args.scale)
        print(f'wrote profile table: {written}')
        if dimensional is not None:
            si_path = written.with_name(written.stem + '_si' + written.suffix)
            dimensional.to_csv(si_path)
            print(f'wrote dimensional table: {si_path}')

    if args.json is not None:
        record = result.to_dict()
        if dimensional is not None:
            record['dimensional'] = dimensional.to_dict()
        args.json.write_text(json.dumps(record, indent=2))
        print(f'wrote condition record: {args.json}')

    if hasattr(args, 'plot'):
        plot_path = args.plot
        if plot_path is None:
            if args.csv is not None:
                plot_path = args.csv.with_suffix('.png')
            else:
                plot_path = Path('boundary_layer.png')
        write_plot(result, plot_path)
        print(f'wrote profile figure: {plot_path}')

    print_summary(result, dimensional)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
