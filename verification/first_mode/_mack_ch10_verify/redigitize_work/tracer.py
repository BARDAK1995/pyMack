"""Generic adaptive curve tracer: follows ink in a binary mask starting from a
seed point, stepping in whichever axis (row or column) currently has the
shallower local slope, always picking among nearby ink pixels (perpendicular
search window) the one closest to the predicted continuation.

Key robustness choices (learned from tracing the Mack Fig 10.1 nested-loop
neutral curves, which have a near-vertical cusp at each loop's nose):
  - The "row direction sign" used while in steep (near-vertical) mode is a
    fixed, caller-supplied invariant (row_dir), NOT re-derived from a noisy
    local slope estimate every step -- re-deriving it let the tracer's slope
    estimate flip sign from pixel noise near the flat-topped cusp and walk
    back over the nose apex onto the neighboring branch.
  - Column position must be monotonic non-decreasing (for direction_sign=+1)
    within a small backlash tolerance; this rejects any candidate that would
    require hopping backward onto a different branch.
"""
import numpy as np


def _groups_along(mask, fixed_idx, sweep_lo, sweep_hi, axis):
    """axis=0: fixed_idx is column, sweep is row range -> return row groups.
    axis=1: fixed_idx is row, sweep is col range -> return col groups."""
    if axis == 0:
        line = mask[sweep_lo:sweep_hi, fixed_idx]
    else:
        line = mask[fixed_idx, sweep_lo:sweep_hi]
    idx = np.where(line)[0]
    if len(idx) == 0:
        return []
    idx = idx + sweep_lo
    groups = []
    cur = [idx[0]]
    for v in idx[1:]:
        if v - cur[-1] <= 2:
            cur.append(v)
        else:
            groups.append((cur[0] + cur[-1]) / 2.0)
            cur = [v]
    groups.append((cur[0] + cur[-1]) / 2.0)
    return groups


def trace_adaptive(mask, direction_sign, row_dir, r_bounds, c_bounds,
                    seed_path, search_radius=4, step=1, max_steps=5000,
                    slope_window=5, col_backlash=1.5):
    """Trace a curve starting from seed_path (list of >=2 (row,col) points
    establishing the initial direction), continuing outward.

    direction_sign: overall +1 (increasing col/R) or -1 (decreasing).
    row_dir: fixed sign for row advancement while in steep (near-vertical)
      mode -- +1 if the branch's row increases (F decreases) as we move away
      from the nose, -1 if row decreases. This must be supplied by the
      caller (from the seed geometry) and is never flipped mid-trace.

    Returns list of (row, col) float pixel coordinates.
    """
    r0, r1 = r_bounds
    c0, c1 = c_bounds
    path = [tuple(p) for p in seed_path]
    row, col = path[-1]
    max_col_seen = col if direction_sign > 0 else -np.inf
    min_col_seen = col if direction_sign < 0 else np.inf

    def local_slope():
        n = min(slope_window, len(path) - 1)
        r_a, c_a = path[-1 - n]
        r_b, c_b = path[-1]
        dc = (c_b - c_a)
        dr = (r_b - r_a)
        if abs(dc) < 1e-6:
            return np.inf
        return dr / dc

    for _ in range(max_steps):
        slope = local_slope()
        steep = abs(slope) > 1.3
        if steep:
            new_row = row + row_dir * step
            if not (r0 <= new_row < r1):
                break
            lo = max(c0, int(col - search_radius))
            hi = min(c1, int(col + search_radius) + 1)
            groups = _groups_along(mask, int(round(new_row)), lo, hi, axis=1)
            if not groups:
                lo2 = max(c0, int(col - 2 * search_radius))
                hi2 = min(c1, int(col + 2 * search_radius) + 1)
                groups = _groups_along(mask, int(round(new_row)), lo2, hi2, axis=1)
                if not groups:
                    break
            g = np.array(groups)
            # enforce column monotonicity (within backlash) consistent with direction_sign
            if direction_sign > 0:
                ok = g >= (max_col_seen - col_backlash)
            else:
                ok = g <= (min_col_seen + col_backlash)
            if ok.any():
                g = g[ok]
            cand = g[np.argmin(np.abs(g - col))]
            row, col = new_row, cand
        else:
            new_col = col + direction_sign * step
            if not (c0 <= new_col < c1):
                break
            lo = max(r0, int(row - search_radius))
            hi = min(r1, int(row + search_radius) + 1)
            groups = _groups_along(mask, int(round(new_col)), lo, hi, axis=0)
            if not groups:
                lo2 = max(r0, int(row - 2 * search_radius))
                hi2 = min(r1, int(row + 2 * search_radius) + 1)
                groups = _groups_along(mask, int(round(new_col)), lo2, hi2, axis=0)
                if not groups:
                    break
            g = np.array(groups)
            cand = g[np.argmin(np.abs(g - row))]
            row, col = cand, new_col
        if direction_sign > 0:
            max_col_seen = max(max_col_seen, col)
        else:
            min_col_seen = min(min_col_seen, col)
        path.append((row, col))
    return path
