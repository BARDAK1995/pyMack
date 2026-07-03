"""Robust arc-length curve tracer: advances along the local tangent direction
by a fixed step, and at each step searches a perpendicular line segment for
the nearest ink pixel. This handles arbitrary local curve orientation
(vertical, horizontal, or anything between) uniformly, unlike a tracer that
switches between pure row-stepping and pure column-stepping based on a noisy
local-slope estimate (which proved fragile near near-vertical runs with small
pixel noise -- see tracer.py docstring).
"""
import numpy as np


def _ink_points_near(mask, r0, r1, c0, c1, center, direction, perp_radius, n_samples=25):
    """Sample points along the perpendicular to `direction` through `center`,
    within +/- perp_radius, and return those that are ink (True) in mask."""
    perp = np.array([-direction[1], direction[0]])
    perp = perp / (np.linalg.norm(perp) + 1e-12)
    ts = np.linspace(-perp_radius, perp_radius, n_samples)
    pts = center + np.outer(ts, perp)
    hits = []
    for (r, c) in pts:
        ri, ci = int(round(r)), int(round(c))
        if r0 <= ri < r1 and c0 <= ci < c1 and mask[ri, ci]:
            hits.append((r, c))
    return hits


def _cluster_1d(hits, center, perp):
    """Cluster hit points by their signed distance along perp from center,
    merging points within 2px, return cluster centroids (r,c)."""
    if not hits:
        return []
    proj = [np.dot(np.array(p) - center, perp) for p in hits]
    order = np.argsort(proj)
    hits_sorted = [hits[i] for i in order]
    proj_sorted = [proj[i] for i in order]
    clusters = [[hits_sorted[0]]]
    cluster_proj = [[proj_sorted[0]]]
    for p, pr in zip(hits_sorted[1:], proj_sorted[1:]):
        if pr - cluster_proj[-1][-1] <= 2.5:
            clusters[-1].append(p)
            cluster_proj[-1].append(pr)
        else:
            clusters.append([p])
            cluster_proj.append([pr])
    return [tuple(np.mean(cl, axis=0)) for cl in clusters]


def trace_arclength(mask, r_bounds, c_bounds, seed_path, step=1.5,
                     perp_radius=6, max_steps=3000, min_forward_cos=0.3,
                     baseline=6, col_monotonic=None, col_backlash=1.5):
    """seed_path: list of >=2 (row,col) points establishing initial direction
    and position (last point is the current tip). Advances forward (in the
    direction implied by the seed) along the curve.

    At each iteration: direction = normalized(tip - point `baseline` steps
    back) -- using a multi-step baseline (not just the immediately preceding
    point) damps pixel-level centroid noise that otherwise lets the heading
    drift and derail the trace. predicted = tip + step*direction; search
    perpendicular to `direction` through `predicted` for ink clusters; choose
    the cluster whose implied new direction deviates least from the current
    direction (cos angle above min_forward_cos), preferring the one closest
    to `predicted`.
    """
    r0, r1 = r_bounds
    c0, c1 = c_bounds
    path = [np.array(p, dtype=float) for p in seed_path]
    max_col_seen = path[-1][1] if col_monotonic == 'increasing' else None
    min_col_seen = path[-1][1] if col_monotonic == 'decreasing' else None

    for _ in range(max_steps):
        n_back = min(baseline, len(path) - 1)
        direction = path[-1] - path[-1 - n_back]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = path[-1] - path[-2]
            norm = np.linalg.norm(direction)
        direction = direction / norm
        predicted = path[-1] + step * direction
        if not (r0 <= predicted[0] < r1 and c0 <= predicted[1] < c1):
            break
        hits = _ink_points_near(mask, r0, r1, c0, c1, predicted, direction, perp_radius)
        clusters = _cluster_1d(hits, predicted, np.array([-direction[1], direction[0]]))
        if not clusters:
            # widen once
            hits = _ink_points_near(mask, r0, r1, c0, c1, predicted, direction, perp_radius * 2)
            clusters = _cluster_1d(hits, predicted, np.array([-direction[1], direction[0]]))
            if not clusters:
                break
        best = None
        best_score = -np.inf
        for cl in clusters:
            cl = np.array(cl)
            if max_col_seen is not None and cl[1] < max_col_seen - col_backlash:
                continue
            if min_col_seen is not None and cl[1] > min_col_seen + col_backlash:
                continue
            new_dir = cl - path[-1]
            nnorm = np.linalg.norm(new_dir)
            if nnorm < 1e-6:
                continue
            new_dir = new_dir / nnorm
            cosang = np.dot(new_dir, direction)
            if cosang < min_forward_cos:
                continue
            dist_penalty = np.linalg.norm(cl - predicted)
            score = cosang - 0.05 * dist_penalty
            if score > best_score:
                best_score = score
                best = cl
        if best is None:
            break
        if max_col_seen is not None:
            max_col_seen = max(max_col_seen, best[1])
        if min_col_seen is not None:
            min_col_seen = min(min_col_seen, best[1])
        path.append(best)
    return [tuple(p) for p in path]
