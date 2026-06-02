"""Validation: Mack Table 10.1 diagnostic wiring."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lst.mack_table_10_1 import (
    DEFAULT_TABLE_10_1_CONDITION,
    DEFAULT_TABLE_10_1_WALL_BC,
    evaluate_table_10_1_exact_shooting,
    family_case_sequence,
    load_low_mid_table_10_1_families,
)


def test_low_mid_table_10_1_diagnostic_uses_reference_cases():
    """Curated shooting diagnostics should derive cases from shared CSV data."""
    print('Test 1: Low/mid Mack Table 10.1 diagnostic reference wiring')

    families = load_low_mid_table_10_1_families([1.3])
    assert len(families) == 1
    family = families[0]
    assert family['Ma'] == 1.3
    assert family['condition'] == DEFAULT_TABLE_10_1_CONDITION
    assert len(family['cases']) == 2
    assert family['cases'][0].omega_i_8th == 8.24e-04

    sequence = family_case_sequence(family)
    assert sequence[0]['alpha'] == family['cases'][0].alpha_L
    assert sequence[0]['Re'] == family['cases'][0].Re_L
    assert sequence[0]['beta'] == family['cases'][0].beta_L
    assert DEFAULT_TABLE_10_1_WALL_BC == 'isothermal'
    print('  PASSED\n')


def test_low_mid_table_10_1_exact_shooting_representative_acceptance():
    """A representative Table 10.1 row should pass the exact-shooting tolerance."""
    rows = evaluate_table_10_1_exact_shooting(
        [1.3],
        limit=1,
        n_steps=120,
        order='both',
        skip_reduced=True,
    )
    assert len(rows) == 2
    assert {row['system_order'] for row in rows} == {'sixth', 'eighth'}
    assert all(row['condition'] == DEFAULT_TABLE_10_1_CONDITION for row in rows)
    assert all(row['wall_bc'] == DEFAULT_TABLE_10_1_WALL_BC for row in rows)
    assert max(abs(row['shooting_rel_error']) for row in rows) < 0.015


@pytest.mark.skipif(
    os.environ.get('RUN_LST_SLOW_ACCEPTANCE') != '1',
    reason='set RUN_LST_SLOW_ACCEPTANCE=1 to run the full Mack Table 10.1 sweep',
)
def test_low_mid_table_10_1_exact_shooting_full_acceptance():
    """All low-/mid-Mach Table 10.1 rows should pass the paper tolerance."""
    rows = evaluate_table_10_1_exact_shooting(
        n_steps=300,
        order='both',
        skip_reduced=True,
    )
    assert len(rows) == 14
    assert max(abs(row['shooting_rel_error']) for row in rows) < 0.015


if __name__ == '__main__':
    test_low_mid_table_10_1_diagnostic_uses_reference_cases()
    test_low_mid_table_10_1_exact_shooting_representative_acceptance()
