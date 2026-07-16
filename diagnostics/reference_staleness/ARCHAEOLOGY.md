# Reference-staleness archaeology: `mack_fig10_6_M45`

Stage: 1 of 3 (read-only archaeology)

Recorded: 2026-07-12T17:12:41-05:00

Worktree head at investigation start: `4471c8f` (`diag/reference-staleness`)

Constraint: no committed verification reference was modified or regenerated.

## Bottom line from history

The current M45 curve is not the June 18 curve. It was numerically rewritten in
commit `f0f47bfb` on 2026-07-02, under the message **“Regenerate Mack Ch.10
verdicts and overlays from the corrected references.”** The rewrite changed all
11 pyMack rows as well as the paper-comparison verdict. No Python, NumPy, SciPy,
BLAS, or platform metadata is stored in either the committed curve JSON or its
verdict JSON.

The driver has been byte-identical since 2026-06-18 (`cfa4ffcd`): for M=4.5 it
uses `N=120`, `y_max=40`, and `ALPHA_SCAN=(0.08, 0.40, 0.005)`. The only
`solve_temporal_compressible` code change after the July 2 curve rewrite is the
July 4 extraction of the inline assembly into `_assemble_temporal_2d_evp`,
committed and tested as pure code motion (`71a1b374`).

Most importantly, slice 13R's current CPU rows are numerically identical, field
for field, to the 11-row curve at `f0f47bfb^` (the pre-July-2 curve), and are not
identical to the July-2/current committed curve. This is a strong discriminator:
the July-2 curve is the historical outlier, not a gradual drift across the
driver's parameter history.

## Curve and verdict history

| Commit | Commit time | File event | Message / significance |
|---|---:|---|---|
| `ecdcbda5` | 2026-06-17T23:24:19-05:00 | added | `Expand verification audit: eigenvalue anchors + Ma-Zhong + Mack 10.6 (24 cases)` |
| `cfa4ffcd1` | 2026-06-18T00:15:07-05:00 | modified | `Second-mode benchmark batch + Mack 10.6 high-Mach domain fix (27 cases)`; introduced per-Mach M45 `N=120`, `y_max=40` |
| `fbaca209f` | 2026-06-19T05:37:26-05:00 | renamed | moved from `growthRate_verification/` to `second_mode/` |
| `e96328669` | 2026-06-19T09:28:48-05:00 | verdict only | overlay/gallery bookkeeping |
| `f0f47bfb0` | 2026-07-02T22:09:38-05:00 | curve + verdict modified | `Regenerate Mack Ch.10 verdicts and overlays from the corrected references`; this is the current committed numeric curve |

The July 2 curve rewrite changed, for example, R=300 from
`omega_i_max=0.0008802555034254461` to `0.0008529766267072418`, while leaving
`alpha_peak=0.20750000000000013`. The verdict moved from 1.0% paper-curve
median relative error (`agrees`) to 6.0% (`acceptable`), primarily because the
paper digitization was also corrected.

Current committed curve SHA-256:
`a8b49dad0ad4a3476b92b06d01b4483ad6b7962d8bbb3ad0c4131380f2d62237`.

## Metadata inside the committed JSONs

The curve JSON is a bare list of 11 objects containing only `R`,
`omega_i_max`, `alpha_peak`, `c_r`, and `c_i`. The verdict records physical and
numerical provenance (`N=120`, `y_max=40`, `L_star`,
`lambda_mu_ratio=0.0`, `condition=table_11_1`, single-thread BLAS), but no
generation timestamp, source commit, Python version, NumPy version, SciPy
version, BLAS/LAPACK vendor, or OS/platform.

The field `"generated": "new"` is not an environment record.

## Driver parameter history

`verification/compute_mack_fig10_6.py` was introduced at `ecdcbda5` with:

- `N_DEFAULT=110`, `Y_MAX_DEFAULT=30.0`;
- M45 `ALPHA_SCAN=(0.08, 0.40, 0.005)`;
- the same coarse scan and five-point quarter-step local refinement used now.

Commit `cfa4ffcd1` added only the per-Mach domain/resolution maps and routed the
parallel work units through them:

- `Y_MAX_BY_MACH[4.5] = 40.0`;
- `N_BY_MACH[4.5] = 120`.

`ALPHA_SCAN` did not change. `git diff --quiet cfa4ffcd..HEAD --
verification/compute_mack_fig10_6.py` exited 0, proving that the full driver is
unchanged from June 18 through the current head.

## Solver-path history since the M45 parameters stabilized

The driver imports `pymack.solver.solve_temporal_compressible` directly.
Relevant history after `cfa4ffcd1`:

1. `d0ddcd346` (2026-07-01): shared base-flow sampling was moved into
   `scales.sample_baseflow`, and private helper names were promoted. The old and
   new L-star sampling expressions are the same operations in the same order.
2. `71a1b374a` (2026-07-04): the inline 2-D temporal assembly body was moved to
   `_assemble_temporal_2d_evp`; the commit reports byte-motion and intercepted
   `(A,B)` identity tests. Solve/filter/sort behavior remained in the public
   function.

No physics/operator edit to the 2-D Mack temporal equations appears in this
interval. The July 4 extraction is the only `pymack/solver.py` commit after the
July 2 curve rewrite.

## Hypotheses separated by evidence

| Hypothesis | Stage-1 evidence | Archaeology disposition |
|---|---|---|
| Driver parameters changed | Driver is byte-identical from June 18 to HEAD; M45 scan, N, and y_max are stable | **Contradicted** |
| Solver behavior changed | July 1 and July 4 refactors touched the path, but are source-equivalent/pure-motion; current rows exactly recover the pre-July-2 curve | **Possible but poorly supported**; July 4 extraction remains the only post-generation code event to test against census behavior |
| Environment changed | July-2 JSONs omit all environment metadata; same stable driver/current solver class now returns the pre-July-2 values | **Not provable from committed provenance, but consistent with the observed discontinuity** |
| July-2 curve came from a non-recorded alternate invocation/artifact | Commit message says regenerated, but no command or environment is embedded | **Possible and not distinguishable from environment drift using the committed files alone** |

## Current environment (not historical provenance)

Read-only preflight on 2026-07-12:

```text
Python 3.12.7
NumPy 2.2.6
SciPy 1.14.1
Windows-10-10.0.19045-SP0
```

These versions describe the current replay environment only. They must not be
attributed to the July 2 generation because the committed artifact does not
record them.

## Exact archaeology commands

```text
git log --follow --name-status --date=iso-strict --format="COMMIT %H %ad %s" -- verification/second_mode/mack_fig10_6_M45/pymack_curve.json
git log --follow --name-status --date=iso-strict --format="COMMIT %H %ad %s" -- verification/second_mode/mack_fig10_6_M45/verdict.json
git diff ecdcbda5..cfa4ffcd -- verification/compute_mack_fig10_6.py
git diff --quiet cfa4ffcd..HEAD -- verification/compute_mack_fig10_6.py
git log --since="2026-07-02T22:09:38-05:00" --date=iso-strict --format="%H %ad %s" -- pymack/solver.py
git show 71a1b374 -- pymack/solver.py
python --version
python -m pip show numpy scipy pytest
PowerShell stdin Python comparison of f0f47bfb^ curve, current committed curve, and verification/gpu_certification/mack_m45_cpu_identity.json
```

The numeric comparison reported:

```text
identity_rows_equal_pre_july2_numeric= True
identity_rows_equal_committed_numeric= False
row_counts= 11 11 11
```
