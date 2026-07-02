"""Validation: shared paper-target registry and numeric reference data."""



from pymack.reference_data import (
    iter_registry_reference_paths,
    load_collaborator_mach5p35_conditions,
    load_collaborator_mach5p35_neutral_curve,
    load_mack_table_10_1_cases,
    load_paper_target_registry,
    load_reference_csv,
    mack_table_10_1_case_key,
    select_mack_table_10_1_cases,
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


def test_registry_has_individual_mack_and_ozgen_targets():
    """Registry should enumerate scoped paper figures/tables individually."""
    print('Test 2: Registry contains individual Mack/Ozgen target IDs')

    registry = load_paper_target_registry()
    target_ids = {
        target['id']
        for paper in registry.get('papers', [])
        for target in paper.get('targets', [])
    }

    required_ids = {
        'mack_ch05_fig5_1',
        'mack_ch05_fig5_3',
        'mack_ch05_fig5_7',
        'mack_ch05_fig5_8',
        'mack_ch06_table_11_1',
        'mack_ch09_fig9_1',
        'mack_ch09_fig9_2',
        'mack_ch09_fig9_3',
        'mack_ch09_fig9_4',
        'mack_ch09_fig9_5',
        'mack_ch09_fig9_6',
        'mack_ch09_fig9_7',
        'mack_ch09_fig9_8',
        'mack_ch09_fig9_9',
        'mack_ch09_fig9_10',
        'mack_ch09_fig9_11',
        'mack_ch09_fig9_12',
        'mack_ch09_fig9_13',
        'mack_ch10_fig10_1',
        'mack_ch10_fig10_2',
        'mack_ch10_fig10_3',
        'mack_ch10_fig10_4',
        'mack_ch10_fig10_5',
        'mack_ch10_fig10_6',
        'mack_ch10_fig10_7',
        'mack_ch10_fig10_8',
        'mack_ch10_fig10_9',
        'mack_ch10_fig10_10',
        'mack_ch10_fig10_11',
        'mack_ch10_table_10_1',
        'ozgen_fig1_profiles',
        'ozgen_fig2_profile_validation',
        'ozgen_fig3_2d_stability',
        'ozgen_fig4_critical_reynolds',
        'ozgen_fig5_m8_comparison',
        'ozgen_fig6_wave_angle_effect',
        'ozgen_fig7_critical_re_vs_wave_angle',
        'ozgen_fig8_mach_independence',
        'ozgen_fig9_reference_temperature_2d',
        'ozgen_fig10_reference_temperature_3d',
    }

    missing = sorted(required_ids - target_ids)
    assert not missing, f'Missing registry target IDs: {missing}'
    print('  PASSED\n')


def test_mack_table_10_1_reference_loader_is_typed_and_filterable():
    """Mack Table 10.1 should be loaded from one canonical CSV source."""
    print('Test 3: Mack Table 10.1 reference loader')

    cases = load_mack_table_10_1_cases()
    assert len(cases) == 12
    assert cases[0].Ma == 1.3
    assert cases[0].Re_L == 500.0
    assert cases[0].alpha_L == 0.075
    assert cases[0].psi_deg == 45.0
    assert abs(cases[0].beta_L - 0.075) < 1e-12
    assert cases[0].omega_i_8th == 8.24e-04

    low_mid = select_mack_table_10_1_cases(Ma=[1.3, 1.6, 2.2])
    assert len(low_mid) == 7
    assert all(case.Ma in {1.3, 1.6, 2.2} for case in low_mid)

    selected = select_mack_table_10_1_cases(Ma=4.5, Re_L=1500.0)
    assert len(selected) == 1
    assert mack_table_10_1_case_key(selected[0]) == (4.5, 1500.0, 0.05, 60.0)
    print('  PASSED\n')


def test_collaborator_mach5p35_benchmark_is_loadable():
    """The imported Mach 5.35 LST benchmark should be self-consistent."""
    print('Test 4: Collaborator Mach 5.35 neutral-curve benchmark')

    conditions = load_collaborator_mach5p35_conditions()
    curve = load_collaborator_mach5p35_neutral_curve()

    assert conditions['benchmark_id'] == 'collaborator_mach5p35_second_mode_neutral'
    assert conditions['flow_conditions']['mach'] == 5.35
    assert conditions['flow_conditions']['gas'] == 'molecular nitrogen'
    assert conditions['flow_conditions']['unit_reynolds_number_per_m_used_in_lst_conversion'] == 1.176e7
    assert conditions['flow_conditions']['unit_reynolds_number_per_m_dsmc_prot0'] == 1.1935e7
    assert conditions['curve_data']['row_count'] == 251

    assert len(curve) == 251
    assert curve[0].frequency_khz == 100.0
    assert curve[-1].frequency_khz == 600.0
    assert min(point.x_left_mm for point in curve) == 14.95
    assert max(point.x_right_mm for point in curve) == 997.525
    assert all(point.x_left_mm <= point.x_right_mm for point in curve)
    print('  PASSED\n')


if __name__ == '__main__':
    print('=' * 60)
    print('REFERENCE DATA VALIDATION')
    print('=' * 60 + '\n')

    test_registry_and_reference_files_exist()
    test_registry_has_individual_mack_and_ozgen_targets()
    test_mack_table_10_1_reference_loader_is_typed_and_filterable()
    test_collaborator_mach5p35_benchmark_is_loadable()

    print('=' * 60)
    print('REFERENCE DATA TESTS COMPLETE')
    print('=' * 60)
