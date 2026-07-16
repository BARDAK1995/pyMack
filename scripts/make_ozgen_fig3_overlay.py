"""Qualitative overlay: pyMack 2D temporal stability map vs Ozgen & Kircali (2008) Fig. 3.

This is Layer 6 of ``docs/VALIDATION_STRATEGY.md``: a *demonstration*, not a
gate.  The figure shows pyMack's gridded temporal growth field (phase-speed
imaginary part ``c_i``) for the Ozgen flat-plate baseline at each requested
Mach number, with constant-``c_i`` contours drawn on top, and the digitized
Fig. 3 curves from the paper overlaid for qualitative comparison of shape,
mode topology, and trends.  No tolerance is asserted.

Coordinates are Mack's L* scale: ``R_L = sqrt(Re_x)`` on the x-axis and the
streamwise wavenumber ``alpha_L*`` on the y-axis, matching the paper's axes.

Implementation notes (the two known pitfalls):

1. ``solve_temporal_2d`` has a delta*-tuned default ``y_max``.  On the
   L* scale the domain must be opened up explicitly: this script passes
   ``y_max = 6 * (delta*/L*)`` (about 40-45 L* units).
2. The raw spectrum cannot be reduced with ``argmax(c_i)``: spurious
   inflectional/continuous-band eigenvalues with 0.45 < c_r < 0.88 would
   pollute the map.  Modes are classified first: among eigenvalues with
   ``|c_i| < 0.05``, only the TS family (``c_r < 0.45``) and the Mack family
   (``0.88 < c_r < 0.99``) are admitted, then the most unstable admitted
   mode is taken.  Discretized free-stream continuous-spectrum modes scatter
   around ``c_r ~ 1`` (both sides) with spurious ``c_i`` up to ~0.2 and are
   excluded by the ``c_r < 0.99`` cap.

Outputs: a PNG figure, a JSON metadata file, and a CSV of the gridded
``c_i`` field so the figure can be re-drawn without recomputing.

Two compute engines produce identical grids: ``--engine sweep`` (default)
goes through the ``pymack.sweep`` batch facade (parallel CPU backend), while
``--engine point`` keeps the original serial per-point
loop for A/B checks.  Equality of the two paths is asserted in
``validation/test_sweep_cpu_backend.py``.

Usage::

    python scripts/make_ozgen_fig3_overlay.py --quality production
    python scripts/make_ozgen_fig3_overlay.py --quality smoke \
        --output-png /tmp/smoke.png --output-json /tmp/smoke.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYMACK_NO_BANNER", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pymack import load_reference_csv, make_flatplate_profile  # noqa: E402
from pymack.sweep import CBand, temporal_sweep  # noqa: E402
from pymack.temporal_solver import solve_temporal_2d  # noqa: E402
from pymack.scales import delta_star_over_lstar  # noqa: E402


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

QUALITY_SETTINGS = {
    "smoke": {"n_re": 10, "n_alpha": 12, "N": 80},
    "production": {"n_re": 24, "n_alpha": 30, "N": 128},
}

RE_RANGE = (300.0, 4500.0)      # R_L = sqrt(Re_x), log-spaced
ALPHA_RANGE = (0.02, 0.24)      # alpha on Mack's L* scale, linear

# Domain height on the L* scale: the solver's default y_max is delta*-tuned,
# so it MUST be widened here (delta*/L* is ~7 for these profiles).
Y_MAX_FACTOR = 6.0

# Mode-family classification of the temporal spectrum (phase speed c):
CI_ABS_MAX = 0.05    # physical |c_i| in this map is < ~0.02; larger is junk
TS_CR_MAX = 0.45     # c_r below this: viscous TS family
MACK_CR_MIN = 0.88   # c_r above this: Mack (second-mode) family ...
MACK_CR_MAX = 0.99   # ... but clearly below the free-stream c = 1 cluster
# The 0.45 < c_r < 0.85-0.88 band holds inflectional/continuous-spectrum
# artifacts and is excluded on purpose.  Likewise c_r >~ 0.99: discretized
# free-stream continuous-spectrum modes scatter around c_r ~ 1 (both sides)
# with spurious c_i as large as 0.2 and must not enter the growth map.

# Digitized Ozgen & Kircali (2008) Fig. 3 curves available per Mach number:
# (csv path relative to reference_data/, c_i level, legend label)
DIGITIZED_REGISTRY = {
    2.0: [
        ("digitized/ozgen_fig3_M2_neutral.csv", 0.0, "neutral ($c_i=0$)"),
        ("digitized/ozgen_fig3_M2_004.csv", 0.004, "$c_i=0.004$"),
        ("digitized/ozgen_fig3_M2_012.csv", 0.012, "$c_i=0.012$"),
    ],
    3.0: [
        ("digitized/ozgen_fig3_M3_neutral.csv", 0.0, "neutral ($c_i=0$)"),
    ],
    4.0: [
        ("digitized/ozgen_fig3_M4_neutral.csv", 0.0, "neutral ($c_i=0$)"),
        ("digitized/ozgen_fig3_M4_004.csv", 0.004, "$c_i=0.004$"),
    ],
    6.0: [
        ("digitized/ozgen_fig3_M6_neutral.csv", 0.0, "neutral ($c_i=0$)"),
    ],
}

# Styling for the pyMack contour levels (level -> linestyle).
LEVEL_LINESTYLES = {0.0: "solid", 0.004: "dashed", 0.012: "dotted"}
# Color per c_i level, shared between pyMack contour and digitized markers.
LEVEL_COLORS = {0.0: "#1a1a1a", 0.004: "#3210a8", 0.012: "#ff7433"}
LEVEL_MARKERS = {0.0: "o", 0.004: "s", 0.012: "^"}


# --------------------------------------------------------------------------
# Computation
# --------------------------------------------------------------------------

def classify_most_unstable(eigenvalues):
    """Return (c_i, c_r, family) of the most unstable *classified* mode.

    Admits only |c_i| < CI_ABS_MAX with c_r in the TS (< TS_CR_MAX) or Mack
    (MACK_CR_MIN < c_r < MACK_CR_MAX) family bands; the inflectional band in
    between and the supersonic c_r >= 1 band are excluded.  Returns
    (nan, nan, "") when no admissible mode exists.
    """
    c = np.asarray(eigenvalues, dtype=complex)
    if c.size == 0:
        return float("nan"), float("nan"), ""
    cr = c.real
    ci = c.imag
    admissible = (np.abs(ci) < CI_ABS_MAX) & (
        (cr < TS_CR_MAX) | ((cr > MACK_CR_MIN) & (cr < MACK_CR_MAX))
    )
    idx = np.flatnonzero(admissible)
    if idx.size == 0:
        return float("nan"), float("nan"), ""
    best = idx[np.argmax(ci[idx])]
    family = "TS" if cr[best] < TS_CR_MAX else "Mack"
    return float(ci[best]), float(cr[best]), family


def compute_panel(Ma, re_values, alpha_values, N, *, verbose=True):
    """Compute the classified c_i / c_r maps for one Mach number."""
    profile = make_flatplate_profile(Ma)
    delta_over_l = delta_star_over_lstar(profile)
    y_max = Y_MAX_FACTOR * delta_over_l

    n_a, n_r = len(alpha_values), len(re_values)
    ci_grid = np.full((n_a, n_r), np.nan)
    cr_grid = np.full((n_a, n_r), np.nan)
    family_grid = [["" for _ in range(n_r)] for _ in range(n_a)]

    t0 = time.perf_counter()
    for j, Re in enumerate(re_values):
        for i, alpha in enumerate(alpha_values):
            evals, _, _ = solve_temporal_2d(
                profile,
                float(alpha),
                float(Re),
                Ma,
                N=N,
                y_max=y_max,
                length_scale="L_star",
            )
            ci, cr, family = classify_most_unstable(evals)
            ci_grid[i, j] = ci
            cr_grid[i, j] = cr
            family_grid[i][j] = family
        if verbose:
            elapsed = time.perf_counter() - t0
            print(
                f"  [M={Ma:g}] Re column {j + 1}/{n_r} (R_L={Re:.0f}) done, "
                f"{elapsed:.1f}s elapsed",
                flush=True,
            )

    return {
        "Ma": float(Ma),
        "re_values": np.asarray(re_values, dtype=float),
        "alpha_values": np.asarray(alpha_values, dtype=float),
        "ci_grid": ci_grid,
        "cr_grid": cr_grid,
        "family_grid": family_grid,
        "delta_star_over_lstar": float(delta_over_l),
        "y_max_lstar": float(y_max),
        "N": int(N),
        "wall_time_s": float(time.perf_counter() - t0),
    }


def compute_panel_sweep(Ma, re_values, alpha_values, N, *, backend="cpu",
                        workers=None, blas_threads=None,
                        eigenvalues_only=False, verbose=True):
    """Compute one panel through the ``pymack.sweep`` batch facade.

    Produces grids identical to :func:`compute_panel`: the two CBand
    families below partition exactly the admissible set of
    :func:`classify_most_unstable` (TS: ``c_r < TS_CR_MAX``; Mack:
    ``MACK_CR_MIN < c_r < MACK_CR_MAX``; both ``|c_i| < CI_ABS_MAX``), the
    per-family selection is the same most-unstable-admitted-mode rule, and
    the cross-family argmax below reproduces the global argmax over the
    union EXACTLY -- including the deployed tie-break.  On bitwise-equal
    ``c_i`` across families, ``classify_most_unstable`` keeps the eigenvalue
    earliest in solver order (``np.argmax`` returns the first maximum), so
    the combiner breaks ties by the smaller ``mode_index`` carried in each
    family record, not by family order.
    """
    profile = make_flatplate_profile(Ma)
    delta_over_l = delta_star_over_lstar(profile)
    y_max = Y_MAX_FACTOR * delta_over_l

    t0 = time.perf_counter()
    res = temporal_sweep(
        profile,
        alpha_values,
        re_values,
        Ma=Ma,
        N=N,
        y_max=y_max,
        length_scale="L_star",
        operator="ozgen_2d",
        families=(
            CBand(float("-inf"), TS_CR_MAX, ci_abs_max=CI_ABS_MAX, label="TS"),
            CBand(MACK_CR_MIN, MACK_CR_MAX, ci_abs_max=CI_ABS_MAX,
                  label="Mack"),
        ),
        backend=backend,
        cpu_workers=workers,
        cpu_blas_threads=blas_threads,
        cpu_eigenvalues_only=eigenvalues_only,
    )

    n_a, n_r = len(alpha_values), len(re_values)
    ci_grid = np.full((n_a, n_r), np.nan)
    cr_grid = np.full((n_a, n_r), np.nan)
    family_grid = [["" for _ in range(n_r)] for _ in range(n_a)]
    ci_stack = np.stack([fam.c.imag for fam in res.families])
    cr_stack = np.stack([fam.c.real for fam in res.families])
    idx_stack = np.stack([fam.mode_index for fam in res.families])
    labels = [fam.band.label for fam in res.families]
    for i in range(n_a):
        for j in range(n_r):
            best_k, best_ci, best_idx = -1, -np.inf, None
            for k in range(len(labels)):
                v = ci_stack[k, i, j]
                if not np.isfinite(v):
                    continue
                idx_k = int(idx_stack[k, i, j])
                # deployed global-argmax tie-break: on bit-equal c_i, the
                # smaller solver-order index wins (np.argmax first-maximum).
                if v > best_ci or (v == best_ci and idx_k < best_idx):
                    best_k, best_ci, best_idx = k, v, idx_k
            if best_k >= 0:
                ci_grid[i, j] = ci_stack[best_k, i, j]
                cr_grid[i, j] = cr_stack[best_k, i, j]
                family_grid[i][j] = labels[best_k]

    wall = time.perf_counter() - t0
    if verbose:
        print(
            f"  [M={Ma:g}] sweep engine (backend={res.meta['backend']}, "
            f"{res.meta['cpu_workers']} workers): {n_a * n_r} nodes in "
            f"{wall:.1f}s",
            flush=True,
        )

    return {
        "Ma": float(Ma),
        "re_values": np.asarray(re_values, dtype=float),
        "alpha_values": np.asarray(alpha_values, dtype=float),
        "ci_grid": ci_grid,
        "cr_grid": cr_grid,
        "family_grid": family_grid,
        "delta_star_over_lstar": float(delta_over_l),
        "y_max_lstar": float(y_max),
        "N": int(N),
        "wall_time_s": float(wall),
        "sweep_meta": res.meta,
        "sweep_result": res,
    }


def load_digitized_curves(Ma):
    """Load digitized (R_L, alpha_L) curves for one Mach panel, floats."""
    curves = []
    for rel_path, level, label in DIGITIZED_REGISTRY.get(float(Ma), []):
        rows = load_reference_csv(rel_path)
        x = np.array([float(row["x"]) for row in rows])
        y = np.array([float(row["y"]) for row in rows])
        order = np.argsort(x)
        curves.append(
            {"path": rel_path, "level": float(level), "label": label,
             "R_L": x[order], "alpha_L": y[order]}
        )
    return curves


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

def write_grid_csv(panels, path: Path):
    """Persist the gridded c_i field so the figure is re-drawable offline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Ma", "Re_L", "alpha_L", "c_i", "c_r", "family"])
        for panel in panels:
            for i, alpha in enumerate(panel["alpha_values"]):
                for j, Re in enumerate(panel["re_values"]):
                    writer.writerow(
                        [
                            f"{panel['Ma']:g}",
                            f"{Re:.6f}",
                            f"{alpha:.6f}",
                            f"{panel['ci_grid'][i, j]:.8e}",
                            f"{panel['cr_grid'][i, j]:.8e}",
                            panel["family_grid"][i][j],
                        ]
                    )


def plot_overlay(panels, digitized_by_ma, output_png: Path, quality: str):
    """Draw the qualitative overlay figure."""
    n_panels = len(panels)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(8.4 * n_panels, 7.2), squeeze=False
    )

    for ax, panel in zip(axes[0], panels):
        Ma = panel["Ma"]
        R = panel["re_values"]
        A = panel["alpha_values"]
        ci = np.ma.masked_invalid(panel["ci_grid"])

        max_abs = float(np.nanmax(np.abs(panel["ci_grid"])))
        if not np.isfinite(max_abs) or max_abs <= 0.0:
            max_abs = 1.0e-3
        fill_levels = np.linspace(-max_abs, max_abs, 27)
        filled = ax.contourf(
            R, A, ci, levels=fill_levels, cmap="RdBu_r", extend="both"
        )

        digitized = digitized_by_ma.get(float(Ma), [])
        contour_levels = sorted({0.0} | {c["level"] for c in digitized})

        # pyMack contours of the classified c_i field.
        for level in contour_levels:
            try:
                ax.contour(
                    R,
                    A,
                    ci,
                    levels=[level],
                    colors=[LEVEL_COLORS.get(level, "0.3")],
                    linestyles=[LEVEL_LINESTYLES.get(level, "dashdot")],
                    linewidths=2.0,
                )
            except ValueError:
                print(f"  [M={Ma:g}] contour level c_i={level} not in field range")

        # Digitized paper curves.
        for curve in digitized:
            level = curve["level"]
            ax.plot(
                curve["R_L"],
                curve["alpha_L"],
                linestyle="none",
                marker=LEVEL_MARKERS.get(level, "d"),
                markersize=7.5,
                markerfacecolor="none",
                markeredgewidth=1.8,
                markeredgecolor=LEVEL_COLORS.get(level, "0.3"),
            )

        ax.set_xlabel(r"$R_L=\sqrt{Re_x}$", fontsize=15)
        ax.set_ylabel(r"$\alpha_{L^*}$", fontsize=15)
        ax.set_title(f"$M = {Ma:g}$", fontsize=16, pad=10)
        ax.set_xlim(RE_RANGE)
        ax.set_ylim(ALPHA_RANGE)
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.3, linestyle="--")

        colorbar = fig.colorbar(filled, ax=ax, pad=0.025)
        colorbar.set_label(r"classified $c_i$ (pyMack)", fontsize=14)
        colorbar.ax.tick_params(labelsize=12)

        # Legend: pyMack contour lines vs digitized markers, per level.
        handles = []
        for level in contour_levels:
            handles.append(
                Line2D(
                    [], [],
                    color=LEVEL_COLORS.get(level, "0.3"),
                    linestyle=LEVEL_LINESTYLES.get(level, "dashdot"),
                    linewidth=2.0,
                    label=f"pyMack $c_i={level:g}$",
                )
            )
        for curve in digitized:
            level = curve["level"]
            handles.append(
                Line2D(
                    [], [],
                    linestyle="none",
                    marker=LEVEL_MARKERS.get(level, "d"),
                    markersize=7.5,
                    markerfacecolor="none",
                    markeredgewidth=1.8,
                    markeredgecolor=LEVEL_COLORS.get(level, "0.3"),
                    label=f"digitized {curve['label']}",
                )
            )
        ax.legend(handles=handles, loc="lower right", fontsize=11, frameon=True)

    suffix = " [smoke quality]" if quality == "smoke" else ""
    fig.suptitle(
        "pyMack vs. Ozgen & Kircali (2008) Fig. 3 — qualitative comparison"
        f"{suffix}",
        fontsize=17,
        y=0.99,
    )
    fig.text(
        0.5,
        0.005,
        "Demonstration only (validation Layer 6) — shapes and mode topology, "
        "not a pass/fail gate. Markers: curves digitized from the paper.",
        ha="center",
        fontsize=12,
        style="italic",
        color="0.25",
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def build_metadata(panels, digitized_by_ma, args, paths, total_wall_s):
    panel_meta = []
    for panel in panels:
        ci = panel["ci_grid"]
        finite = np.isfinite(ci)
        has_sign_change = bool(
            finite.any() and np.nanmin(ci) < 0.0 < np.nanmax(ci)
        )
        panel_meta.append(
            {
                "Ma": panel["Ma"],
                "N": panel["N"],
                "delta_star_over_lstar": panel["delta_star_over_lstar"],
                "y_max_lstar": panel["y_max_lstar"],
                "n_re": int(len(panel["re_values"])),
                "n_alpha": int(len(panel["alpha_values"])),
                "n_nodes": int(ci.size),
                "n_classified": int(finite.sum()),
                "ci_min": float(np.nanmin(ci)) if finite.any() else None,
                "ci_max": float(np.nanmax(ci)) if finite.any() else None,
                "neutral_contour_present": has_sign_change,
                "wall_time_s": panel["wall_time_s"],
                "digitized_curves": [
                    {"path": c["path"], "level": c["level"],
                     "n_points": int(len(c["R_L"]))}
                    for c in digitized_by_ma.get(panel["Ma"], [])
                ],
            }
        )
    metadata = {
        "purpose": (
            "Qualitative overlay (demonstration, not a gate) of pyMack's 2D "
            "temporal stability map against digitized Ozgen & Kircali (2008) "
            "Fig. 3 curves. Layer 6 of docs/VALIDATION_STRATEGY.md."
        ),
        "quality": args.quality,
        "engine": args.engine,
        "sweep_backend": args.sweep_backend if args.engine == "sweep" else None,
        "panels_requested": args.panels,
        "re_range": list(RE_RANGE),
        "re_spacing": "log",
        "alpha_range": list(ALPHA_RANGE),
        "alpha_spacing": "linear",
        "length_scale": "L_star",
        "wall_bc": "isothermal (Ozgen Eq. 19: T_tilde(0)=0; mean flow adiabatic)",
        "y_max_rule": f"{Y_MAX_FACTOR:g} * delta_star_over_lstar (L* units)",
        "mode_classification": {
            "ci_abs_max": CI_ABS_MAX,
            "ts_family_cr_max": TS_CR_MAX,
            "mack_family_cr_min": MACK_CR_MIN,
            "mack_family_cr_max": MACK_CR_MAX,
            "excluded_bands": (
                "0.45 < c_r < 0.88 (inflectional/continuous) and "
                "c_r >= 0.99 (free-stream c~1 continuous-spectrum cluster)"
            ),
        },
        "profile": "pymack.make_flatplate_profile(Ma) defaults "
                   "(T_edge=288 K, adiabatic wall, Ozgen transport)",
        "known_artifacts": [
            "At M=2, high-Re/high-alpha corner: weak (c_i ~ 1e-3) spurious "
            "instability from the discretized slow-acoustic continuous "
            "spectrum (c_r <= 1 - 1/Ma overlaps the TS band); it is "
            "y_max-sensitive, unlike the converged mid-lobe discrete modes.",
            "At M=4, the genuine mid-band first mode (c_r ~ 0.54-0.62, "
            "inside the excluded 0.45-0.88 band) is marginally damped in "
            "this formulation where the paper shows marginal growth — the "
            "repo's documented first-mode formulation discrepancy. The "
            "excluded band cannot be admitted: it also carries spurious "
            "weakly-unstable continuous-spectrum modes that flood the map.",
        ],
        "output_png": str(paths["png"]),
        "output_grid_csv": str(paths["grid_csv"]),
        "total_wall_time_s": float(total_wall_s),
        "panels": panel_meta,
    }
    if args.blas_threads is not None or args.eigenvalues_only:
        metadata["fast_cpu_sweep"] = {
            "blas_threads": args.blas_threads,
            "eigenvalues_only": bool(args.eigenvalues_only),
        }
    return metadata


def verify_grid_against_committed(path: Path, *, reference: Path | None = None,
                                  atol=1.0e-9):
    """Compare a production M=2 grid at the committed artifact's precision."""
    if reference is None:
        reference = (
            REPO_ROOT / "verification" / "mixed_mode" / "ozgen_fig3"
            / "_compute" / "ozgen_M2_ci_grid.csv"
        )
    actual_bytes = path.read_bytes()
    reference_bytes = reference.read_bytes()
    with path.open(newline="", encoding="utf-8") as handle:
        actual_rows = list(csv.DictReader(handle))
    with reference.open(newline="", encoding="utf-8") as handle:
        reference_rows = list(csv.DictReader(handle))
    max_abs_diff = {"c_i": 0.0, "c_r": 0.0}
    binary_comparison_slack = 1.0e-15
    mismatches = []
    for index, (actual, expected) in enumerate(zip(actual_rows, reference_rows)):
        reasons = []
        for key in ("Ma", "Re_L", "alpha_L"):
            if actual[key] != expected[key]:
                reasons.append(f"{key}: {actual[key]!r} != {expected[key]!r}")
        if actual["family"] != expected["family"]:
            reasons.append(
                f"family: {actual['family']!r} != {expected['family']!r}")
        for key in ("c_i", "c_r"):
            got = float(actual[key])
            ref = float(expected[key])
            got_nan = bool(np.isnan(got))
            ref_nan = bool(np.isnan(ref))
            if got_nan != ref_nan:
                reasons.append(f"{key}: finite/NaN decision differs")
            elif not got_nan:
                diff = abs(got - ref)
                max_abs_diff[key] = max(max_abs_diff[key], diff)
                if diff > atol + binary_comparison_slack:
                    reasons.append(f"{key}: abs diff {diff:.3e} > {atol:.3e}")
        if reasons:
            mismatches.append({"row": index, "reasons": reasons})
    row_count_ok = len(actual_rows) == len(reference_rows) == 720
    ok = row_count_ok and not mismatches
    return {
        "ok": ok,
        "comparison": (
            "exact coordinate/family/finite decisions and absolute c_r/c_i "
            "agreement at committed CSV precision"
        ),
        "atol": atol,
        "binary_comparison_slack": binary_comparison_slack,
        "rows_expected": 720,
        "rows_actual": len(actual_rows),
        "rows_matched": len(actual_rows) - len(mismatches) if row_count_ok else None,
        "max_abs_diff": max_abs_diff,
        "mismatches": mismatches[:20],
        "mismatches_truncated": len(mismatches) > 20,
        "byte_equal": actual_bytes == reference_bytes,
        "actual_bytes": len(actual_bytes),
        "reference_bytes": len(reference_bytes),
        "reference": str(reference.relative_to(REPO_ROOT)),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--quality",
        choices=sorted(QUALITY_SETTINGS),
        default="production",
        help="smoke: coarse 10x12 grid at N=80; production: 24x30 at N=128",
    )
    parser.add_argument(
        "--output-png",
        default=str(REPO_ROOT / "docs" / "figures" / "ozgen_fig3_overlay.png"),
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="metadata JSON path (default: PNG path with .json extension)",
    )
    parser.add_argument(
        "--output-grid-csv",
        default=None,
        help="c_i grid CSV path (default: PNG path with _ci_grid.csv suffix)",
    )
    parser.add_argument(
        "--panels",
        default="2,4",
        help="comma-separated Mach numbers, e.g. '2,4' (default)",
    )
    parser.add_argument(
        "--engine",
        choices=("sweep", "point"),
        default="sweep",
        help="sweep: pymack.sweep batch facade (default); "
             "point: legacy serial per-point loop (kept for A/B checks)",
    )
    parser.add_argument(
        "--sweep-backend",
        default="cpu",
        help="pymack.sweep backend for --engine sweep (default cpu; "
             "'auto' defers to PYMACK_SWEEP_BACKEND, otherwise stays CPU)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="CPU worker processes for --engine sweep "
             "(default: all cores, capped at the number of grid points)",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=None,
        help="opt-in BLAS threads per CPU sweep worker (tuned recipe: 1)",
    )
    parser.add_argument(
        "--eigenvalues-only",
        action="store_true",
        help="opt-in CPU 2-D QZ values-only path with selected-vector recompute",
    )
    parser.add_argument(
        "--verify-against-committed",
        action="store_true",
        help="require a production M=2 grid to match ozgen_M2_ci_grid.csv at its stored precision",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    settings = QUALITY_SETTINGS[args.quality]

    mach_numbers = [float(token) for token in args.panels.split(",") if token.strip()]
    if not mach_numbers:
        raise SystemExit("--panels produced no Mach numbers")
    if args.blas_threads is not None and args.blas_threads < 1:
        raise SystemExit("--blas-threads must be >= 1")
    if (args.blas_threads is not None or args.eigenvalues_only) and (
        args.engine != "sweep" or args.sweep_backend != "cpu"
    ):
        raise SystemExit(
            "--blas-threads/--eigenvalues-only require --engine sweep "
            "--sweep-backend cpu")
    if args.verify_against_committed and (
        args.quality != "production" or mach_numbers != [2.0]
    ):
        raise SystemExit(
            "--verify-against-committed requires --quality production --panels 2")

    output_png = Path(args.output_png)
    output_json = (
        Path(args.output_json)
        if args.output_json
        else output_png.with_suffix(".json")
    )
    output_grid_csv = (
        Path(args.output_grid_csv)
        if args.output_grid_csv
        else output_png.with_name(output_png.stem + "_ci_grid.csv")
    )

    re_values = np.logspace(
        np.log10(RE_RANGE[0]), np.log10(RE_RANGE[1]), settings["n_re"]
    )
    alpha_values = np.linspace(ALPHA_RANGE[0], ALPHA_RANGE[1], settings["n_alpha"])

    print(
        f"quality={args.quality}: {settings['n_re']} Re x "
        f"{settings['n_alpha']} alpha grid, N={settings['N']}, "
        f"panels M={mach_numbers}",
        flush=True,
    )

    t_start = time.perf_counter()
    panels = []
    digitized_by_ma = {}
    for Ma in mach_numbers:
        if float(Ma) not in DIGITIZED_REGISTRY:
            print(
                f"  [M={Ma:g}] no digitized Fig. 3 curves registered; "
                "pyMack contours will be drawn without overlay"
            )
        digitized_by_ma[float(Ma)] = load_digitized_curves(Ma)
        if args.engine == "sweep":
            sweep_options = {
                "backend": args.sweep_backend,
                "workers": args.workers,
            }
            if args.blas_threads is not None:
                sweep_options["blas_threads"] = args.blas_threads
            if args.eigenvalues_only:
                sweep_options["eigenvalues_only"] = True
            panels.append(
                compute_panel_sweep(
                    Ma, re_values, alpha_values, settings["N"],
                    **sweep_options,
                )
            )
        else:
            panels.append(
                compute_panel(Ma, re_values, alpha_values, settings["N"])
            )

    write_grid_csv(panels, output_grid_csv)
    plot_overlay(panels, digitized_by_ma, output_png, args.quality)

    total_wall_s = time.perf_counter() - t_start
    metadata = build_metadata(
        panels,
        digitized_by_ma,
        args,
        {"png": output_png, "grid_csv": output_grid_csv},
        total_wall_s,
    )
    identity = None
    if args.verify_against_committed:
        identity = verify_grid_against_committed(output_grid_csv)
        metadata["committed_grid_identity"] = identity
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"output_png={output_png}")
    print(f"output_json={output_json}")
    print(f"output_grid_csv={output_grid_csv}")
    if identity is not None:
        print(
            "committed_grid_identity="
            + ("PASS" if identity["ok"] else "FAIL")
        )
    for meta in metadata["panels"]:
        print(
            f"  M={meta['Ma']:g}: classified {meta['n_classified']}/"
            f"{meta['n_nodes']} nodes, c_i in "
            f"[{meta['ci_min']:.4g}, {meta['ci_max']:.4g}], "
            f"neutral contour present: {meta['neutral_contour_present']}"
        )
    print(f"total_wall_time_s={total_wall_s:.1f}")
    for meta in metadata["panels"]:
        if not meta["neutral_contour_present"]:
            print(
                f"WARNING: M={meta['Ma']:g} c_i field does not change sign — "
                "no neutral contour. Check y_max / mode classification."
            )
    if identity is not None and not identity["ok"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
