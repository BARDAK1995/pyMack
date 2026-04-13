"""
Validation: Mack mean-flow reproduction benchmarks.

Tests:
1. Table 11.1 displacement thickness is reproduced across Mach number.
2. The shared condition helpers distinguish table and figure setups.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lst.mack_conditions import (
    make_mack_profile,
    mack_figure_edge_temperature,
    mack_table_11_1_edge_temperature,
)


def test_table_11_1_displacement_thickness():
    """Table 11.1 d* values should be matched within 1 percent."""
    print('Test 1: Table 11.1 displacement thickness benchmark')

    refs = {
        1.0: 2.13,
        1.6: 2.77,
        2.0: 3.37,
        2.2: 3.72,
        3.0: 5.48,
        3.8: 7.83,
        4.5: 10.34,
        5.8: 15.73,
        7.0: 21.19,
        8.0: 26.13,
        10.0: 36.88,
    }

    worst_rel_err = 0.0
    for Ma, ref in refs.items():
        bf = make_mack_profile(Ma, condition='table_11_1', Re_delta_star=2000)
        rel_err = abs(bf._delta_star - ref) / ref
        worst_rel_err = max(worst_rel_err, rel_err)
        print(
            f'  M={Ma:4.1f}: d*={bf._delta_star:8.3f}, '
            f'ref={ref:8.3f}, rel_err={100.0 * rel_err:6.3f}%'
        )
        assert rel_err < 0.01, (
            f'Table 11.1 mismatch at M={Ma}: computed {bf._delta_star:.4f}, '
            f'reference {ref:.4f}'
        )

    print(f'  Worst relative error: {100.0 * worst_rel_err:.3f}%')
    print('  PASSED\n')


def test_temperature_condition_split():
    """Table 11.1 and figure-caption helpers should not collapse together."""
    print('Test 2: Mack condition helper split')

    t_table = mack_table_11_1_edge_temperature(4.5)
    t_figure = mack_figure_edge_temperature(4.5)
    print(f'  M=4.5: table T1*={t_table:.3f} K, figure T1*={t_figure:.3f} K')

    assert abs(t_table - t_figure) > 100.0, (
        'Table and figure temperature schedules should remain distinct at M=4.5'
    )
    assert mack_figure_edge_temperature(5.8) == 50.0
    assert mack_table_11_1_edge_temperature(5.8) == 50.0
    print('  PASSED\n')


if __name__ == '__main__':
    print('=' * 60)
    print('MACK MEAN-FLOW VALIDATION')
    print('=' * 60 + '\n')

    test_table_11_1_displacement_thickness()
    test_temperature_condition_split()

    print('=' * 60)
    print('MACK MEAN-FLOW TESTS COMPLETE')
    print('=' * 60)
