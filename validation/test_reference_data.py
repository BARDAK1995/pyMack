"""Validation: shared paper-target registry and numeric reference data."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lst.reference_data import (
    iter_registry_reference_paths,
    load_paper_target_registry,
    load_reference_csv,
    reference_data_root,
)


def test_registry_and_reference_files_exist():
    """All registry-declared reference CSVs should exist and load."""
    print('Test 1: Registry-declared reference data files exist')

    registry = load_paper_target_registry()
    assert registry.get('papers'), 'Registry should contain paper entries'

    root = reference_data_root()
    for relative_path in iter_registry_reference_paths():
        csv_rows = load_reference_csv(relative_path)
        print(f'  {relative_path}: {len(csv_rows)} rows')
        assert (root / relative_path).exists(), f'Missing reference file: {relative_path}'
        assert len(csv_rows) > 0, f'Empty reference file: {relative_path}'

    print('  PASSED\n')


if __name__ == '__main__':
    print('=' * 60)
    print('REFERENCE DATA VALIDATION')
    print('=' * 60 + '\n')

    test_registry_and_reference_files_exist()

    print('=' * 60)
    print('REFERENCE DATA TESTS COMPLETE')
    print('=' * 60)
