"""Helpers for loading shared paper target and numeric reference data."""

from __future__ import annotations

import csv
import json
from pathlib import Path


_REFERENCE_ROOT = Path(__file__).resolve().parent.parent / 'reference_data'


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
