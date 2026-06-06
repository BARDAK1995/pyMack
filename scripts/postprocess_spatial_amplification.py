"""Postprocess fixed-frequency spatial growth into amplification and N factors."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SECOND_MODE_ALPHA_MIN_L = None


def _read_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out = dict(row)
            for key in (
                "freq_parameter",
                "R_L",
                "omega_L",
                "sigma_L",
                "wavelength_L",
                "alpha_r_L",
                "alpha_i_L",
            ):
                if key in out:
                    try:
                        out[key] = float(out[key])
                    except ValueError:
                        out[key] = math.nan
            rows.append(out)
    return rows


def _apply_mode_family_defaults(args):
    if (
        args.mode_family == "second_mode"
        and args.alpha_min_l is None
        and SECOND_MODE_ALPHA_MIN_L is not None
    ):
        args.alpha_min_l = SECOND_MODE_ALPHA_MIN_L
    return args


def _interpolate_row(left, right, R_cross):
    R0 = float(left["R_L"])
    R1 = float(right["R_L"])
    if R1 == R0:
        weight = 0.0
    else:
        weight = (R_cross - R0) / (R1 - R0)
    out = dict(left)
    out["R_L"] = float(R_cross)
    for key in ("omega_L", "wavelength_L", "alpha_r_L", "alpha_i_L"):
        if key in left and key in right:
            v0 = float(left[key])
            v1 = float(right[key])
            if math.isfinite(v0) and math.isfinite(v1):
                out[key] = float(v0 + weight * (v1 - v0))
    out["sigma_L"] = 0.0
    out["status"] = "interpolated_neutral"
    return out


def _crossings(rows):
    crossings = []
    previous = None
    for row in rows:
        sig = float(row["sigma_L"])
        if not math.isfinite(sig):
            previous = None
            continue
        if previous is not None:
            sig0 = float(previous["sigma_L"])
            if math.isfinite(sig0) and sig0 * sig < 0.0:
                R0 = float(previous["R_L"])
                R1 = float(row["R_L"])
                R_cross = R0 + (0.0 - sig0) * (R1 - R0) / (sig - sig0)
                crossings.append((float(R_cross), previous, row))
        previous = row
    return crossings


def _select_unstable_lobe(freq_rows, crossings):
    if len(crossings) < 2:
        return None
    finite_rows = [
        row for row in freq_rows
        if math.isfinite(float(row["R_L"])) and math.isfinite(float(row["sigma_L"]))
    ]
    if not finite_rows:
        return crossings[0], crossings[-1]
    peak_row = max(finite_rows, key=lambda row: float(row["sigma_L"]))
    peak_R = float(peak_row["R_L"])
    best_pair = None
    best_score = -math.inf
    for left, right in zip(crossings[:-1], crossings[1:]):
        lower_R = float(left[0])
        upper_R = float(right[0])
        if upper_R <= lower_R:
            continue
        interval_rows = [
            row for row in finite_rows
            if lower_R <= float(row["R_L"]) <= upper_R
        ]
        if not interval_rows:
            continue
        max_sigma = max(float(row["sigma_L"]) for row in interval_rows)
        if lower_R <= peak_R <= upper_R:
            return left, right
        if max_sigma > best_score:
            best_pair = (left, right)
            best_score = max_sigma
    if best_pair is not None:
        return best_pair
    return crossings[0], crossings[-1]


def _integrate_lobe(rows, lower, upper, *, dx_over_lstar_per_dR):
    lower_R, lower_left, lower_right = lower
    upper_R, upper_left, upper_right = upper
    lower_row = _interpolate_row(lower_left, lower_right, lower_R)
    upper_row = _interpolate_row(upper_left, upper_right, upper_R)

    path_rows = [lower_row]
    path_rows.extend([
        row for row in rows
        if lower_R < float(row["R_L"]) < upper_R and math.isfinite(float(row["sigma_L"]))
    ])
    path_rows.append(upper_row)
    path_rows.sort(key=lambda row: float(row["R_L"]))

    R = np.array([float(row["R_L"]) for row in path_rows], dtype=float)
    sigma = np.array([float(row["sigma_L"]) for row in path_rows], dtype=float)
    integrand = float(dx_over_lstar_per_dR) * sigma
    integrand_pos = np.maximum(integrand, 0.0)

    signed = np.zeros(len(R), dtype=float)
    positive = np.zeros(len(R), dtype=float)
    for i in range(1, len(R)):
        dR = R[i] - R[i - 1]
        signed[i] = signed[i - 1] + 0.5 * (integrand[i] + integrand[i - 1]) * dR
        positive[i] = positive[i - 1] + 0.5 * (integrand_pos[i] + integrand_pos[i - 1]) * dR

    envelope = np.maximum.accumulate(signed)
    return {
        "lower_R": lower_R,
        "upper_R": upper_R,
        "path_rows": path_rows,
        "R": R,
        "signed_N": signed,
        "positive_N": positive,
        "envelope_N": envelope,
    }


def _integrate_full_path(rows, *, dx_over_lstar_per_dR):
    """Integrate signed/positive growth from the first sampled station."""
    path_rows = [
        row for row in rows
        if math.isfinite(float(row["R_L"])) and math.isfinite(float(row["sigma_L"]))
    ]
    path_rows.sort(key=lambda row: float(row["R_L"]))
    R = np.array([float(row["R_L"]) for row in path_rows], dtype=float)
    sigma = np.array([float(row["sigma_L"]) for row in path_rows], dtype=float)
    integrand = float(dx_over_lstar_per_dR) * sigma
    integrand_pos = np.maximum(integrand, 0.0)

    signed = np.zeros(len(R), dtype=float)
    positive = np.zeros(len(R), dtype=float)
    for i in range(1, len(R)):
        dR = R[i] - R[i - 1]
        signed[i] = signed[i - 1] + 0.5 * (integrand[i] + integrand[i - 1]) * dR
        positive[i] = positive[i - 1] + 0.5 * (integrand_pos[i] + integrand_pos[i - 1]) * dR

    envelope = np.maximum.accumulate(signed)
    return {
        "path_rows": path_rows,
        "R": R,
        "signed_N": signed,
        "positive_N": positive,
        "envelope_N": envelope,
    }


def _integrate_from_lower_neutral(rows, lower, upper=None, *, dx_over_lstar_per_dR):
    """Integrate signed growth from the lower neutral through the full path.

    This is the physical amplitude history for a wave normalized by its lower
    neutral amplitude.  It keeps decaying after the upper neutral instead of
    freezing at the lobe peak/exit value.
    """
    lower_R, lower_left, lower_right = lower
    path_rows = [_interpolate_row(lower_left, lower_right, lower_R)]
    path_rows.extend([
        row for row in rows
        if lower_R < float(row["R_L"]) and math.isfinite(float(row["sigma_L"]))
    ])
    if upper is not None:
        upper_R, upper_left, upper_right = upper
        if upper_R > lower_R:
            path_rows.append(_interpolate_row(upper_left, upper_right, upper_R))

    deduped = {}
    for row in path_rows:
        deduped[round(float(row["R_L"]), 12)] = row
    path_rows = sorted(deduped.values(), key=lambda row: float(row["R_L"]))

    R = np.array([float(row["R_L"]) for row in path_rows], dtype=float)
    sigma = np.array([float(row["sigma_L"]) for row in path_rows], dtype=float)
    integrand = float(dx_over_lstar_per_dR) * sigma
    integrand_pos = np.maximum(integrand, 0.0)

    signed = np.zeros(len(R), dtype=float)
    positive = np.zeros(len(R), dtype=float)
    for i in range(1, len(R)):
        dR = R[i] - R[i - 1]
        signed[i] = signed[i - 1] + 0.5 * (integrand[i] + integrand[i - 1]) * dR
        positive[i] = positive[i - 1] + 0.5 * (integrand_pos[i] + integrand_pos[i - 1]) * dR

    envelope = np.maximum.accumulate(signed)
    return {
        "lower_R": lower_R,
        "upper_R": math.nan if upper is None else upper[0],
        "path_rows": path_rows,
        "R": R,
        "signed_N": signed,
        "positive_N": positive,
        "envelope_N": envelope,
    }


def _interp_series(x, xp, fp, fill_before=0.0, fill_after=None):
    if fill_after is None:
        fill_after = float(fp[-1])
    return float(np.interp(float(x), xp, fp, left=fill_before, right=fill_after))


def _mode_family_accepts(row, *, alpha_min_l, alpha_max_l):
    alpha_r = float(row.get("alpha_r_L", math.nan))
    if not math.isfinite(alpha_r):
        return False
    if alpha_min_l is not None and alpha_r < alpha_min_l:
        return False
    if alpha_max_l is not None and alpha_r > alpha_max_l:
        return False
    return True


def _process_frequency(
    rows,
    freq,
    *,
    view_min,
    view_max,
    alpha_min_l,
    alpha_max_l,
    dx_over_lstar_per_dR,
):
    freq_rows = sorted([
        row for row in rows
        if math.isclose(float(row["freq_parameter"]), float(freq), rel_tol=0.0, abs_tol=1.0e-14)
        and view_min <= float(row["R_L"]) <= view_max
        and math.isfinite(float(row["sigma_L"]))
        and _mode_family_accepts(row, alpha_min_l=alpha_min_l, alpha_max_l=alpha_max_l)
    ], key=lambda row: float(row["R_L"]))
    crosses = _crossings(freq_rows)
    if len(crosses) < 2:
        if not freq_rows:
            return [], {
                "freq_parameter": float(freq),
                "status": "no_finite_rows",
                "n_crossings": len(crosses),
            }
        full_path = _integrate_full_path(
            freq_rows,
            dx_over_lstar_per_dR=dx_over_lstar_per_dR,
        )
        R_full = full_path["R"]
        signed_full = full_path["signed_N"]
        positive_full = full_path["positive_N"]
        envelope_full = full_path["envelope_N"]
        out_rows = []
        for row in freq_rows:
            R = float(row["R_L"])
            signed_N_start = _interp_series(
                R,
                R_full,
                signed_full,
                fill_before=0.0,
                fill_after=float(signed_full[-1]),
            )
            positive_N_start = _interp_series(
                R,
                R_full,
                positive_full,
                fill_before=0.0,
                fill_after=float(positive_full[-1]),
            )
            envelope_N_start = _interp_series(
                R,
                R_full,
                envelope_full,
                fill_before=0.0,
                fill_after=float(envelope_full[-1]),
            )
            out = dict(row)
            out.update({
                "lower_neutral_R_L": math.nan,
                "upper_neutral_R_L": math.nan,
                "N_signed_from_lower": math.nan,
                "amplification_signed_from_lower": math.nan,
                "N_positive_from_lower": math.nan,
                "amplification_positive_from_lower": math.nan,
                "N_envelope_from_lower": math.nan,
                "amplification_envelope_from_lower": math.nan,
                "N_signed_from_start": float(signed_N_start),
                "amplification_signed_from_start": float(math.exp(signed_N_start)),
                "N_positive_from_start": float(positive_N_start),
                "amplification_positive_from_start": float(math.exp(positive_N_start)),
                "N_envelope_from_start": float(envelope_N_start),
                "amplification_envelope_from_start": float(math.exp(envelope_N_start)),
                "amplification_region": "no_two_crossing_lobe",
            })
            out_rows.append(out)
        peak_index_start = int(np.argmax(signed_full))
        peak_growth_row = max(freq_rows, key=lambda row: float(row["sigma_L"]))
        return out_rows, {
            "freq_parameter": float(freq),
            "status": "missing_two_crossings",
            "n_crossings": len(crosses),
            "lower_neutral_R_L": math.nan,
            "upper_neutral_R_L": math.nan,
            "peak_N_R_L": math.nan,
            "N_signed_peak": math.nan,
            "amplification_signed_peak": math.nan,
            "peak_N_R_L_from_start": float(R_full[peak_index_start]),
            "N_signed_peak_from_start": float(signed_full[peak_index_start]),
            "amplification_signed_peak_from_start": float(math.exp(float(signed_full[peak_index_start]))),
            "N_positive_at_end_from_start": float(positive_full[-1]),
            "amplification_positive_at_end_from_start": float(math.exp(float(positive_full[-1]))),
            "N_envelope_at_end_from_start": float(envelope_full[-1]),
            "amplification_envelope_at_end_from_start": float(math.exp(float(envelope_full[-1]))),
            "N_signed_at_end_from_start": float(signed_full[-1]),
            "amplification_signed_at_end_from_start": float(math.exp(float(signed_full[-1]))),
            "peak_growth_R_L": float(peak_growth_row["R_L"]),
            "peak_sigma_L": float(peak_growth_row["sigma_L"]),
            "peak_wavelength_L": float(peak_growth_row["wavelength_L"]),
            "n_samples": int(len(freq_rows)),
        }

    lower, upper = _select_unstable_lobe(freq_rows, crosses)
    lobe = _integrate_lobe(
        freq_rows,
        lower,
        upper,
        dx_over_lstar_per_dR=dx_over_lstar_per_dR,
    )
    lower_path = _integrate_from_lower_neutral(
        freq_rows,
        lower,
        upper,
        dx_over_lstar_per_dR=dx_over_lstar_per_dR,
    )
    full_path = _integrate_full_path(
        freq_rows,
        dx_over_lstar_per_dR=dx_over_lstar_per_dR,
    )
    R_path = lobe["R"]
    signed = lobe["signed_N"]
    positive = lobe["positive_N"]
    envelope = lobe["envelope_N"]
    R_lower_path = lower_path["R"]
    signed_lower_path = lower_path["signed_N"]
    positive_lower_path = lower_path["positive_N"]
    envelope_lower_path = lower_path["envelope_N"]
    R_full = full_path["R"]
    signed_full = full_path["signed_N"]
    positive_full = full_path["positive_N"]
    envelope_full = full_path["envelope_N"]

    out_rows = []
    for row in freq_rows:
        R = float(row["R_L"])
        if R < lobe["lower_R"]:
            region = "pre_lobe_stable"
            signed_N = 0.0
            positive_N = 0.0
            envelope_N = 0.0
        else:
            region = "inside_lobe" if R <= lobe["upper_R"] else "post_lobe_stable"
            signed_N = _interp_series(
                R,
                R_lower_path,
                signed_lower_path,
                fill_before=0.0,
                fill_after=float(signed_lower_path[-1]),
            )
            positive_N = _interp_series(
                R,
                R_lower_path,
                positive_lower_path,
                fill_before=0.0,
                fill_after=float(positive_lower_path[-1]),
            )
            envelope_N = _interp_series(
                R,
                R_lower_path,
                envelope_lower_path,
                fill_before=0.0,
                fill_after=float(envelope_lower_path[-1]),
            )
        signed_N_start = _interp_series(
            R,
            R_full,
            signed_full,
            fill_before=0.0,
            fill_after=float(signed_full[-1]),
        )
        positive_N_start = _interp_series(
            R,
            R_full,
            positive_full,
            fill_before=0.0,
            fill_after=float(positive_full[-1]),
        )
        envelope_N_start = _interp_series(
            R,
            R_full,
            envelope_full,
            fill_before=0.0,
            fill_after=float(envelope_full[-1]),
        )
        out = dict(row)
        out.update({
            "lower_neutral_R_L": float(lobe["lower_R"]),
            "upper_neutral_R_L": float(lobe["upper_R"]),
            "N_signed_from_lower": float(signed_N),
            "amplification_signed_from_lower": float(math.exp(signed_N)),
            "N_positive_from_lower": float(positive_N),
            "amplification_positive_from_lower": float(math.exp(positive_N)),
            "N_envelope_from_lower": float(envelope_N),
            "amplification_envelope_from_lower": float(math.exp(envelope_N)),
            "N_signed_from_start": float(signed_N_start),
            "amplification_signed_from_start": float(math.exp(signed_N_start)),
            "N_positive_from_start": float(positive_N_start),
            "amplification_positive_from_start": float(math.exp(positive_N_start)),
            "N_envelope_from_start": float(envelope_N_start),
            "amplification_envelope_from_start": float(math.exp(envelope_N_start)),
            "amplification_region": region,
        })
        out_rows.append(out)

    peak_index = int(np.argmax(signed))
    lower_peak_index = int(np.argmax(signed_lower_path))
    peak_index_start = int(np.argmax(signed_full))
    lobe_rows = [
        row for row in freq_rows
        if lobe["lower_R"] <= float(row["R_L"]) <= lobe["upper_R"]
    ]
    peak_growth_row = max(lobe_rows, key=lambda row: float(row["sigma_L"]))
    summary = {
        "freq_parameter": float(freq),
        "status": "ok",
        "lower_neutral_R_L": float(lobe["lower_R"]),
        "upper_neutral_R_L": float(lobe["upper_R"]),
        "peak_N_R_L": float(R_path[peak_index]),
        "N_signed_peak": float(signed[peak_index]),
        "amplification_signed_peak": float(math.exp(float(signed[peak_index]))),
        "peak_N_R_L_from_lower_full_path": float(R_lower_path[lower_peak_index]),
        "N_signed_peak_from_lower_full_path": float(signed_lower_path[lower_peak_index]),
        "amplification_signed_peak_from_lower_full_path": float(math.exp(float(signed_lower_path[lower_peak_index]))),
        "peak_N_R_L_from_start": float(R_full[peak_index_start]),
        "N_signed_peak_from_start": float(signed_full[peak_index_start]),
        "amplification_signed_peak_from_start": float(math.exp(float(signed_full[peak_index_start]))),
        "N_positive_at_end_from_start": float(positive_full[-1]),
        "amplification_positive_at_end_from_start": float(math.exp(float(positive_full[-1]))),
        "N_envelope_at_end_from_start": float(envelope_full[-1]),
        "amplification_envelope_at_end_from_start": float(math.exp(float(envelope_full[-1]))),
        "N_signed_at_end_from_start": float(signed_full[-1]),
        "amplification_signed_at_end_from_start": float(math.exp(float(signed_full[-1]))),
        "N_positive_at_upper": float(positive[-1]),
        "amplification_positive_at_upper": float(math.exp(float(positive[-1]))),
        "N_envelope_at_upper": float(envelope[-1]),
        "amplification_envelope_at_upper": float(math.exp(float(envelope[-1]))),
        "N_signed_at_upper": float(signed[-1]),
        "amplification_signed_at_upper": float(math.exp(float(signed[-1]))),
        "N_signed_at_end_from_lower": float(signed_lower_path[-1]),
        "amplification_signed_at_end_from_lower": float(math.exp(float(signed_lower_path[-1]))),
        "N_envelope_at_end_from_lower": float(envelope_lower_path[-1]),
        "amplification_envelope_at_end_from_lower": float(math.exp(float(envelope_lower_path[-1]))),
        "peak_growth_R_L": float(peak_growth_row["R_L"]),
        "peak_sigma_L": float(peak_growth_row["sigma_L"]),
        "peak_wavelength_L": float(peak_growth_row["wavelength_L"]),
        "n_samples": int(len(freq_rows)),
    }
    return out_rows, summary


def _write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _apply_plot_xlim(ax, plot_x_min, plot_x_max):
    if plot_x_min is not None or plot_x_max is not None:
        ax.set_xlim(left=plot_x_min, right=plot_x_max)


def _plot(path, rows, summaries, title, *, plot_x_min=None, plot_x_max=None):
    freqs = [row["freq_parameter"] for row in summaries if row.get("status") == "ok"]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(freqs)))
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 9.0), sharex=True)

    if not freqs:
        axes[0].text(
            0.5,
            0.5,
            "No rows passed the selected mode-family filter",
            transform=axes[0].transAxes,
            ha="center",
            va="center",
        )

    for color, freq in zip(colors, freqs):
        freq_rows = sorted([
            row for row in rows
            if math.isclose(float(row["freq_parameter"]), float(freq), rel_tol=0.0, abs_tol=1.0e-14)
        ], key=lambda row: float(row["R_L"]))
        R = np.array([float(row["R_L"]) for row in freq_rows], dtype=float)
        sigma = np.array([float(row["sigma_L"]) for row in freq_rows], dtype=float)
        signed = np.array([float(row["N_signed_from_lower"]) for row in freq_rows], dtype=float)
        envelope = np.array([float(row["N_envelope_from_lower"]) for row in freq_rows], dtype=float)
        amp_signed = np.array([float(row["amplification_signed_from_lower"]) for row in freq_rows], dtype=float)
        amp_env = np.array([float(row["amplification_envelope_from_lower"]) for row in freq_rows], dtype=float)

        label = rf"$F={freq:.2e}$"
        axes[0].plot(R, sigma, "o-", color=color, label=label)
        axes[1].plot(R, signed, "o-", color=color)
        axes[1].plot(R, envelope, "--", color=color, alpha=0.8)
        axes[2].semilogy(R, amp_signed, "o-", color=color)
        axes[2].semilogy(R, amp_env, "--", color=color, alpha=0.8)

    axes[0].axhline(0.0, color="0.2", lw=0.8)
    axes[0].set_ylabel(r"$\sigma_L$")
    axes[0].set_title(title)
    if freqs:
        axes[0].legend(loc="best", fontsize=8)
    axes[1].set_ylabel("N from lower neutral")
    axes[2].set_ylabel("A / A_lower")
    axes[2].set_xlabel(r"$R$ (input Reynolds-coordinate path)")
    for ax in axes:
        _apply_plot_xlim(ax, plot_x_min, plot_x_max)
        ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_linear_amplification(path, rows, summaries, title, *, plot_x_min=None, plot_x_max=None):
    freqs = [row["freq_parameter"] for row in summaries if row.get("status") == "ok"]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(freqs)))
    fig, ax = plt.subplots(figsize=(8.4, 5.4))

    if not freqs:
        ax.text(
            0.5,
            0.5,
            "No rows passed the selected mode-family filter",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    for color, freq in zip(colors, freqs):
        freq_rows = sorted([
            row for row in rows
            if math.isclose(float(row["freq_parameter"]), float(freq), rel_tol=0.0, abs_tol=1.0e-14)
        ], key=lambda row: float(row["R_L"]))
        R = np.array([float(row["R_L"]) for row in freq_rows], dtype=float)
        amp_signed = np.array([float(row["amplification_signed_from_lower"]) for row in freq_rows], dtype=float)
        amp_env = np.array([float(row["amplification_envelope_from_lower"]) for row in freq_rows], dtype=float)

        label = rf"$F={freq:.2e}$"
        ax.plot(R, amp_signed, "o-", color=color, label=label)
        ax.plot(R, amp_env, "--", color=color, alpha=0.8)

    ax.axhline(1.0, color="0.2", lw=0.8)
    ax.set_xlabel(r"$R$ (input Reynolds-coordinate path)")
    ax.set_ylabel(r"amplification ratio $A/A_{\mathrm{lower}}$")
    ax.set_title(f"{title}: linear amplification ratio")
    if freqs:
        ax.legend(loc="best", fontsize=8)
    _apply_plot_xlim(ax, plot_x_min, plot_x_max)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_linear_amplification_from_start(
    path,
    rows,
    summaries,
    title,
    *,
    plot_x_min=None,
    plot_x_max=None,
):
    freqs = [
        row["freq_parameter"] for row in summaries
        if row.get("status") in {"ok", "missing_two_crossings"}
    ]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(freqs)))
    fig, ax = plt.subplots(figsize=(8.4, 5.4))

    if not freqs:
        ax.text(
            0.5,
            0.5,
            "No rows passed the selected mode-family filter",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    for color, freq in zip(colors, freqs):
        freq_rows = sorted([
            row for row in rows
            if math.isclose(float(row["freq_parameter"]), float(freq), rel_tol=0.0, abs_tol=1.0e-14)
        ], key=lambda row: float(row["R_L"]))
        R = np.array([float(row["R_L"]) for row in freq_rows], dtype=float)
        amp_signed = np.array([float(row["amplification_signed_from_start"]) for row in freq_rows], dtype=float)
        amp_env = np.array([float(row["amplification_envelope_from_start"]) for row in freq_rows], dtype=float)

        label = rf"$F={freq:.2e}$"
        ax.plot(R, amp_signed, "o-", color=color, label=label)
        ax.plot(R, amp_env, "--", color=color, alpha=0.8)

    ax.axhline(1.0, color="0.2", lw=0.8)
    ax.set_xlabel(r"$R$ (input Reynolds-coordinate path)")
    ax.set_ylabel(r"amplification ratio $A/A_{\mathrm{start}}$")
    ax.set_title(f"{title}: linear amplification from first station")
    if freqs:
        ax.legend(loc="best", fontsize=8)
    _apply_plot_xlim(ax, plot_x_min, plot_x_max)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _neutral_envelope_rows(summaries):
    rows = []
    for summary in summaries:
        if summary.get("status") != "ok":
            continue
        lower = float(summary.get("lower_neutral_R_L", math.nan))
        upper = float(summary.get("upper_neutral_R_L", math.nan))
        freq = float(summary.get("freq_parameter", math.nan))
        if not (math.isfinite(freq) and math.isfinite(lower) and math.isfinite(upper)):
            continue
        rows.append({
            "freq_parameter": freq,
            "F_x1e4": 1.0e4 * freq,
            "lower_neutral_R_L": lower,
            "upper_neutral_R_L": upper,
            "peak_growth_R_L": float(summary.get("peak_growth_R_L", math.nan)),
            "peak_sigma_L": float(summary.get("peak_sigma_L", math.nan)),
            "peak_wavelength_L": float(summary.get("peak_wavelength_L", math.nan)),
            "N_signed_peak": float(summary.get("N_signed_peak", math.nan)),
            "amplification_signed_peak": float(summary.get("amplification_signed_peak", math.nan)),
        })
    return sorted(rows, key=lambda row: row["freq_parameter"])


def _plot_neutral_envelope(path, envelope_rows, title):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if not envelope_rows:
        ax.text(
            0.5,
            0.5,
            "No two-crossing neutral branches found",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    else:
        F = np.array([row["F_x1e4"] for row in envelope_rows], dtype=float)
        lower = np.array([row["lower_neutral_R_L"] for row in envelope_rows], dtype=float)
        upper = np.array([row["upper_neutral_R_L"] for row in envelope_rows], dtype=float)
        ax.plot(lower, F, "o-", lw=2.0, ms=4.5, label="lower neutral")
        ax.plot(upper, F, "o-", lw=2.0, ms=4.5, label="upper neutral")
        for lo, hi, f_val in zip(lower, upper, F):
            ax.plot([lo, hi], [f_val, f_val], color="0.85", lw=0.8, zorder=0)
        ax.legend(frameon=False)
    ax.set_xlabel(r"$R=\sqrt{Re_x}$")
    ax.set_ylabel(r"$F \times 10^4$")
    ax.set_title(f"{title}: neutral envelope")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_neutral_peak_growth(path, envelope_rows, title):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if not envelope_rows:
        ax.text(
            0.5,
            0.5,
            "No two-crossing neutral branches found",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    else:
        F = np.array([row["F_x1e4"] for row in envelope_rows], dtype=float)
        lower = np.array([row["lower_neutral_R_L"] for row in envelope_rows], dtype=float)
        upper = np.array([row["upper_neutral_R_L"] for row in envelope_rows], dtype=float)
        peak_R = np.array([row["peak_growth_R_L"] for row in envelope_rows], dtype=float)
        peak_sigma = np.array([row["peak_sigma_L"] for row in envelope_rows], dtype=float)
        ax.plot(lower, F, "k--", lw=1.2, label="lower neutral")
        ax.plot(upper, F, "k-", lw=1.2, label="upper neutral")
        sc = ax.scatter(peak_R, F, c=peak_sigma, cmap="viridis", s=45)
        ax.legend(frameon=False)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(r"peak $-\alpha_i L^*$")
    ax.set_xlabel(r"$R=\sqrt{Re_x}$")
    ax.set_ylabel(r"$F \times 10^4$")
    ax.set_title(f"{title}: peak-growth locations")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_growth_contour_neutral_envelope(
    path,
    rows,
    envelope_rows,
    title,
    *,
    plot_x_min=None,
    plot_x_max=None,
):
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    finite_rows = [
        row for row in rows
        if math.isfinite(float(row.get("R_L", math.nan)))
        and math.isfinite(float(row.get("freq_parameter", math.nan)))
        and math.isfinite(float(row.get("sigma_L", math.nan)))
    ]
    if not finite_rows:
        ax.text(
            0.5,
            0.5,
            "No finite growth samples found",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    else:
        R = np.array([float(row["R_L"]) for row in finite_rows], dtype=float)
        F = np.array([1.0e4 * float(row["freq_parameter"]) for row in finite_rows], dtype=float)
        sigma = np.array([float(row["sigma_L"]) for row in finite_rows], dtype=float)
        max_abs = float(np.nanmax(np.abs(sigma)))
        if not math.isfinite(max_abs) or max_abs <= 0.0:
            max_abs = 1.0
        levels = np.linspace(-max_abs, max_abs, 33)
        R_unique = np.array(sorted(set(float(value) for value in R)), dtype=float)
        F_unique = np.array(sorted(set(float(value) for value in F)), dtype=float)
        if len(R_unique) * len(F_unique) >= len(finite_rows):
            r_index = {value: i for i, value in enumerate(R_unique)}
            f_index = {value: i for i, value in enumerate(F_unique)}
            Z = np.full((len(F_unique), len(R_unique)), np.nan, dtype=float)
            for r_value, f_value, sigma_value in zip(R, F, sigma):
                Z[f_index[float(f_value)], r_index[float(r_value)]] = sigma_value
            Z_masked = np.ma.masked_invalid(Z)
            cf = ax.contourf(R_unique, F_unique, Z_masked, levels=levels, cmap="coolwarm", extend="both")
            try:
                ax.contour(R_unique, F_unique, Z_masked, levels=[0.0], colors="0.1", linewidths=1.1)
            except ValueError:
                pass
        else:
            cf = ax.tricontourf(R, F, sigma, levels=levels, cmap="coolwarm", extend="both")
            try:
                ax.tricontour(R, F, sigma, levels=[0.0], colors="0.1", linewidths=1.1)
            except ValueError:
                pass
        cb = fig.colorbar(cf, ax=ax)
        cb.set_label(r"spatial growth $\sigma_L=-\mathrm{Im}(\alpha_L)$")

    if envelope_rows:
        F_env = np.array([row["F_x1e4"] for row in envelope_rows], dtype=float)
        lower = np.array([row["lower_neutral_R_L"] for row in envelope_rows], dtype=float)
        upper = np.array([row["upper_neutral_R_L"] for row in envelope_rows], dtype=float)
        ax.plot(lower, F_env, "k--", lw=2.0, label="lower neutral")
        ax.plot(upper, F_env, "k-", lw=2.0, label="upper neutral")
        ax.legend(frameon=False)
    ax.set_xlabel(r"$R=\sqrt{Re_x}$")
    ax.set_ylabel(r"$F \times 10^4$")
    ax.set_title(f"{title}: growth contour and neutral envelope")
    _apply_plot_xlim(ax, plot_x_min, plot_x_max)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--view-min", type=float, default=-math.inf)
    parser.add_argument("--view-max", type=float, default=math.inf)
    parser.add_argument("--plot-x-min", type=float, default=None)
    parser.add_argument("--plot-x-max", type=float, default=None)
    parser.add_argument(
        "--mode-family",
        choices=["second_mode", "unconstrained"],
        default="second_mode",
        help=(
            "Rows to integrate. second_mode applies no alpha floor by default; "
            "provide --alpha-min-l only for deliberately high-alpha diagnostics."
        ),
    )
    parser.add_argument("--alpha-min-l", type=float, default=None)
    parser.add_argument("--alpha-max-l", type=float, default=None)
    parser.add_argument(
        "--r-convention",
        choices=["sqrt_2_re_x", "sqrt_re_x", "custom"],
        default="sqrt_re_x",
        help=(
            "How the supplied R_L path maps to physical x. sqrt_re_x uses "
            "dx/L*=2 dR for R_L=sqrt(Re_x), matching the pyMack/Mack L* "
            "convention. sqrt_2_re_x uses dx/L*=dR. custom requires "
            "--dx-over-lstar-per-dr."
        ),
    )
    parser.add_argument(
        "--dx-over-lstar-per-dr",
        type=float,
        default=None,
        help="Custom multiplier in N = integral multiplier*sigma_L dR.",
    )
    parser.add_argument("--title", default="Spatial amplification and N factor")
    return parser.parse_args(argv)


def _dx_over_lstar_per_dR(args):
    if args.r_convention == "sqrt_2_re_x":
        if args.dx_over_lstar_per_dr is not None:
            raise ValueError("--dx-over-lstar-per-dr is only valid with --r-convention custom")
        return 1.0
    if args.r_convention == "sqrt_re_x":
        if args.dx_over_lstar_per_dr is not None:
            raise ValueError("--dx-over-lstar-per-dr is only valid with --r-convention custom")
        return 2.0
    if args.dx_over_lstar_per_dr is None:
        raise ValueError("--r-convention custom requires --dx-over-lstar-per-dr")
    value = float(args.dx_over_lstar_per_dr)
    if value <= 0.0:
        raise ValueError("--dx-over-lstar-per-dr must be positive")
    return value


def main(argv=None):
    args = _apply_mode_family_defaults(parse_args(argv))
    dx_over_lstar_per_dR = _dx_over_lstar_per_dR(args)
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(input_csv)
    freqs = sorted({float(row["freq_parameter"]) for row in rows})
    amplified_rows = []
    summaries = []
    for freq in freqs:
        freq_rows, summary = _process_frequency(
            rows,
            freq,
            view_min=float(args.view_min),
            view_max=float(args.view_max),
            alpha_min_l=args.alpha_min_l,
            alpha_max_l=args.alpha_max_l,
            dx_over_lstar_per_dR=dx_over_lstar_per_dR,
        )
        amplified_rows.extend(freq_rows)
        summaries.append(summary)

    curves_csv = output_dir / "spatial_fixed_frequency_amplification_curves.csv"
    summary_csv = output_dir / "spatial_fixed_frequency_amplification_summary.csv"
    neutral_envelope_csv = output_dir / "spatial_fixed_frequency_neutral_envelope.csv"
    png_path = output_dir / "spatial_fixed_frequency_amplification.png"
    linear_png_path = output_dir / "spatial_fixed_frequency_amplification_linear.png"
    linear_start_png_path = output_dir / "spatial_fixed_frequency_amplification_linear_from_start.png"
    neutral_envelope_png_path = output_dir / "spatial_fixed_frequency_neutral_envelope.png"
    neutral_peak_png_path = output_dir / "spatial_fixed_frequency_neutral_envelope_peak_growth.png"
    neutral_contour_png_path = output_dir / "spatial_fixed_frequency_neutral_envelope_growth_contour.png"
    metadata_path = output_dir / "spatial_fixed_frequency_amplification_metadata.json"
    neutral_envelope_rows = _neutral_envelope_rows(summaries)
    _write_csv(curves_csv, amplified_rows)
    _write_csv(summary_csv, summaries)
    _write_csv(neutral_envelope_csv, neutral_envelope_rows)
    _plot(
        png_path,
        amplified_rows,
        summaries,
        args.title,
        plot_x_min=args.plot_x_min,
        plot_x_max=args.plot_x_max,
    )
    _plot_linear_amplification(
        linear_png_path,
        amplified_rows,
        summaries,
        args.title,
        plot_x_min=args.plot_x_min,
        plot_x_max=args.plot_x_max,
    )
    _plot_linear_amplification_from_start(
        linear_start_png_path,
        amplified_rows,
        summaries,
        args.title,
        plot_x_min=args.plot_x_min,
        plot_x_max=args.plot_x_max,
    )
    _plot_neutral_envelope(
        neutral_envelope_png_path,
        neutral_envelope_rows,
        args.title,
    )
    _plot_neutral_peak_growth(
        neutral_peak_png_path,
        neutral_envelope_rows,
        args.title,
    )
    _plot_growth_contour_neutral_envelope(
        neutral_contour_png_path,
        amplified_rows,
        neutral_envelope_rows,
        args.title,
        plot_x_min=args.plot_x_min,
        plot_x_max=args.plot_x_max,
    )
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump({
            "status": "diagnostic_not_paper_certified",
            "source_csv": str(input_csv).replace("\\", "/"),
            "view_min_R_L": float(args.view_min),
            "view_max_R_L": float(args.view_max),
            "plot_x_min": args.plot_x_min,
            "plot_x_max": args.plot_x_max,
            "mode_family": args.mode_family,
            "alpha_L_filter": [
                None if args.alpha_min_l is None else float(args.alpha_min_l),
                None if args.alpha_max_l is None else float(args.alpha_max_l),
            ],
            "r_convention": args.r_convention,
            "dx_over_lstar_per_dR": float(dx_over_lstar_per_dR),
            "neutral_envelope_source": (
                "linear interpolation of signed fixed-frequency growth "
                "crossings sigma_L=0"
            ),
            "neutral_contour_source": (
                "sampled fixed-frequency growth field sigma_L over the same "
                "R_L and frequency grid used for branch integration"
            ),
            "n_factor_formula": (
                "N = integral sigma_phys dx = integral "
                f"{dx_over_lstar_per_dR:g}*sigma_L dR_L"
            ),
            "amplification_formula": "A/A_lower = exp(N_signed_from_lower)",
            "n_factor_convention": (
                "N_signed_from_lower integrates signed growth from the lower "
                "neutral point through the full plotted path, so stable "
                "post-upper-neutral damping reduces A/A_lower. "
                "N_envelope_from_lower is the cumulative maximum of signed N; "
                "N_positive_from_lower integrates "
                f"max({dx_over_lstar_per_dR:g}*sigma_L, 0). "
                "N_signed_from_start and amplification_signed_from_start include "
                "stable damping from the first sampled station."
            ),
            "flat_plate_scaling": (
                "The streamwise multiplier is convention-dependent. "
                "For R=sqrt(Re_x), dx/L*=2 dR. For R=sqrt(2 Re_x), "
                "dx/L*=dR. This file records the multiplier actually used."
            ),
        }, handle, indent=2)

    print(f"curves_csv={curves_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"neutral_envelope_csv={neutral_envelope_csv}")
    print(f"png={png_path}")
    print(f"linear_png={linear_png_path}")
    print(f"linear_start_png={linear_start_png_path}")
    print(f"neutral_envelope_png={neutral_envelope_png_path}")
    print(f"neutral_peak_png={neutral_peak_png_path}")
    print(f"neutral_contour_png={neutral_contour_png_path}")
    print(f"metadata={metadata_path}")
    for summary in summaries:
        if summary.get("status") != "ok":
            print(f"freq={summary['freq_parameter']:.6g}: {summary['status']}")
            continue
        print(
            f"freq={summary['freq_parameter']:.6g}: "
            f"lower={summary['lower_neutral_R_L']:.1f}, "
            f"upper={summary['upper_neutral_R_L']:.1f}, "
            f"N_peak={summary['N_signed_peak']:.6e}, "
            f"A_peak={summary['amplification_signed_peak']:.6e}, "
            f"N_peak_from_start={summary['N_signed_peak_from_start']:.6e}, "
            f"A_peak_from_start={summary['amplification_signed_peak_from_start']:.6e}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
