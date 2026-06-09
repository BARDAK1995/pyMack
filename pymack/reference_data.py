"""Helpers for loading shared paper target and numeric reference data."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


_REFERENCE_ROOT = Path(__file__).resolve().parent.parent / 'reference_data'


@dataclass(frozen=True)
class MackTable101Case:
    """One Mack Table 10.1 oblique temporal-growth reference point."""

    Ma: float
    Re_L: float
    alpha_L: float
    psi_deg: float
    omega_i_6th: float
    omega_i_8th: float

    @property
    def beta_L(self) -> float:
        """Return the spanwise wavenumber on Mack's L* scale."""
        return self.alpha_L * math.tan(math.radians(self.psi_deg))


@dataclass(frozen=True)
class DimensionalNeutralCurvePoint:
    """One dimensional neutral-curve row in frequency/x-branch coordinates."""

    frequency_hz: float
    frequency_khz: float
    x0_m: float
    x1_m: float
    x_left_m: float
    x_right_m: float
    x_left_mm: float
    x_right_mm: float


def reference_data_root() -> Path:
    """Return the root directory for shared paper reference data."""
    return _REFERENCE_ROOT


def load_paper_target_registry():
    """Load the shared Mack/Ozgen paper target registry JSON."""
    registry_path = _REFERENCE_ROOT / 'paper_target_registry.json'
    with registry_path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def load_reference_csv(relative_path):
    """Load one CSV reference table from the shared reference-data area."""
    csv_path = _REFERENCE_ROOT / relative_path
    with csv_path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _as_float_selector(value):
    """Normalize a scalar/list selector into a set of floats, or None."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return {float(value)}
    try:
        return {float(item) for item in value}
    except TypeError:
        return {float(value)}


def load_mack_table_10_1_cases():
    """Load Mack Table 10.1 reference points as typed case records."""
    rows = load_reference_csv('mack/table_10_1_oblique_growth.csv')
    return [
        MackTable101Case(
            Ma=float(row['Ma']),
            Re_L=float(row['R_L']),
            alpha_L=float(row['alpha_L']),
            psi_deg=float(row['psi_deg']),
            omega_i_6th=float(row['omega_i_6th']),
            omega_i_8th=float(row['omega_i_8th']),
        )
        for row in rows
    ]


def load_collaborator_mach5p35_conditions():
    """Load the Mach 5.35 collaborator benchmark condition metadata."""
    conditions_path = _REFERENCE_ROOT / 'collaborator_mach5p35' / 'conditions.json'
    with conditions_path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def load_collaborator_mach5p35_neutral_curve():
    """Load the Mach 5.35 dimensional LST neutral-curve benchmark."""
    rows = load_reference_csv('collaborator_mach5p35/LST_neutral_curve_M5p35.csv')
    return [
        DimensionalNeutralCurvePoint(
            frequency_hz=float(row['frequency_hz']),
            frequency_khz=float(row['frequency_khz']),
            x0_m=float(row['x0_m']),
            x1_m=float(row['x1_m']),
            x_left_m=float(row['x_left_m']),
            x_right_m=float(row['x_right_m']),
            x_left_mm=float(row['x_left_mm']),
            x_right_mm=float(row['x_right_mm']),
        )
        for row in rows
    ]


def mack_table_10_1_case_key(case):
    """Return a stable tuple key for one Mack Table 10.1 case."""
    return (
        float(case.Ma),
        float(case.Re_L),
        float(case.alpha_L),
        float(case.psi_deg),
    )


def select_mack_table_10_1_cases(Ma=None, Re_L=None, psi_deg=None):
    """Return Mack Table 10.1 cases filtered by Mach, Reynolds, or wave angle."""
    ma_values = _as_float_selector(Ma)
    re_values = _as_float_selector(Re_L)
    psi_values = _as_float_selector(psi_deg)

    selected = []
    for case in load_mack_table_10_1_cases():
        if ma_values is not None and float(case.Ma) not in ma_values:
            continue
        if re_values is not None and float(case.Re_L) not in re_values:
            continue
        if psi_values is not None and float(case.psi_deg) not in psi_values:
            continue
        selected.append(case)
    return selected


def iter_registry_reference_paths():
    """Yield all registry-relative reference-data file paths."""
    registry = load_paper_target_registry()
    for paper in registry.get('papers', []):
        for target in paper.get('targets', []):
            for relative_path in target.get('reference_data', []):
                yield relative_path


def find_paper_target(target_id):
    """Return one target entry from the shared registry by id."""
    registry = load_paper_target_registry()
    for paper in registry.get('papers', []):
        for target in paper.get('targets', []):
            if target.get('id') == target_id:
                return target
    raise KeyError(f'Unknown paper target id: {target_id}')
