"""Shared comparison helpers for the pyMack verification audit.

Every verified case writes a ``verdict.json`` (schema below); the agreement
classification is computed here so the thresholds live in exactly one place.

verdict.json schema
-------------------
{
  "case_id":       str,   # unique, == folder name
  "category":      "neutral_curve" | "growth_rate",
  "source":        str,   # e.g. "Mack (1984) Fig. 10.3"
  "conditions":    {Ma, gas, wall, psi_deg, formulation, transport, ...},
  "quantity":      str,   # what is being compared
  "metrics":       {free-form numbers, e.g. median_rel_err, mae_mm, ...},
  "verdict":       "agrees" | "acceptable" | "disagrees" | "pending",
  "verdict_reason": str,
  "generated":     "reuse" | "new" | "pending",
  "artifacts":     {pymack, reference, overlay},   # repo-relative paths or null
  "pymack_provenance": str
}

The verdict is *recorded* in the file (so the matrix builder never recomputes
physics). ``classify_relative`` / ``classify_dimensional`` are provided so the
per-case comparison scripts assign it consistently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --- Agreement thresholds -------------------------------------------------
# Hand-digitized literature curves carry a ~2-5% reading error, so a median
# relative error at or below 5% is genuinely "at the digitization noise floor"
# = perfect agreement. 5-15% is a real but bounded offset (correct physics,
# quantifiable bias). Above 15%, or any topology/mode-count mismatch, is a
# genuine disagreement.
PERFECT_REL_ERR = 0.05
ACCEPTABLE_REL_ERR = 0.15

VERDICT_BADGE = {
    "agrees": "✅ agrees",
    "acceptable": "🟡 acceptable",
    "disagrees": "❌ disagrees",
    "pending": "⬜ pending",
}
VERDICT_ORDER = {"agrees": 0, "acceptable": 1, "disagrees": 2, "pending": 3}


def classify_relative(median_rel_err: float, topology_ok: bool) -> str:
    """Three-tier verdict from a median relative error and a topology flag."""
    if not topology_ok:
        return "disagrees"
    if median_rel_err <= PERFECT_REL_ERR:
        return "agrees"
    if median_rel_err <= ACCEPTABLE_REL_ERR:
        return "acceptable"
    return "disagrees"


def classify_dimensional(mae: float, curve_span: float, topology_ok: bool) -> str:
    """Verdict for a dimensional curve: MAE as a fraction of the curve's span."""
    if curve_span <= 0:
        raise ValueError("curve_span must be positive")
    return classify_relative(mae / curve_span, topology_ok)


# --- Curve comparison on a shared domain ----------------------------------

def overlapping_domain(ref_x, test_x):
    """Inclusive [lo, hi] overlap of two abscissa ranges (None if disjoint)."""
    lo = max(np.min(ref_x), np.min(test_x))
    hi = min(np.max(ref_x), np.max(test_x))
    return (lo, hi) if hi > lo else None


def interp_errors(ref_x, ref_y, test_x, test_y):
    """Per-sample |test - ref(interp)| and relative error over the overlap.

    Reference curve is interpolated onto the test abscissae that fall inside
    the overlapping domain. Returns (abs_err, rel_err, n) where rel_err is
    normalized by |ref| with a small floor to avoid divide-by-zero.
    """
    ref_x = np.asarray(ref_x, float)
    ref_y = np.asarray(ref_y, float)
    test_x = np.asarray(test_x, float)
    test_y = np.asarray(test_y, float)
    order = np.argsort(ref_x)
    ref_x, ref_y = ref_x[order], ref_y[order]

    dom = overlapping_domain(ref_x, test_x)
    if dom is None:
        return np.array([]), np.array([]), 0
    lo, hi = dom
    mask = (test_x >= lo) & (test_x <= hi)
    tx, ty = test_x[mask], test_y[mask]
    if tx.size == 0:
        return np.array([]), np.array([]), 0
    ry = np.interp(tx, ref_x, ref_y)
    abs_err = np.abs(ty - ry)
    floor = 0.05 * np.max(np.abs(ref_y)) if np.max(np.abs(ref_y)) > 0 else 1.0
    rel_err = abs_err / np.maximum(np.abs(ry), floor)
    return abs_err, rel_err, int(tx.size)


# --- verdict.json IO ------------------------------------------------------

def write_verdict(folder: Path, verdict: dict) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "verdict.json"
    path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return path


def read_verdict(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
