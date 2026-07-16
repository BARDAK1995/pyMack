# Reference-staleness census

Stage: 2 of 3 (scratch-only cheap regeneration probes)

No file under `verification/` or `reference_data/` was written. Each probe ran
the current validation driver logic with `PYTHONPATH` pointed at this worktree
and wrote only to `diagnostics/reference_staleness/SCRATCH/`.

## Classification rule

For the committed numeric payload selected from each verdict JSON, the probe
uses this ordered classification:

1. `byte-identical`: canonical JSON bytes of the selected numeric arrays are
   identical;
2. `numeric-within-1e-9`: not byte-identical, but maximum absolute component
   drift is at most `1e-9`;
3. `drifted`: maximum absolute component drift is greater than `1e-9`.

This byte test is deliberately limited to the numeric payload. A scratch result
file cannot be byte-compared wholesale with a committed verdict because their
schemas and metadata differ.

## Results

| Case | Numeric payload committed | Driver / path exercised | Wall (s) | Regen verdict | Drift |
|---|---|---|---:|---|---:|
| `malik_case3` | 2026-07-02 22:10:10 -05:00, `0182004f` | N=200 3-D temporal `solve_temporal_compressible_3d` | 15.010 | numeric-within-1e-9 | max abs `8.410e-15`; max component rel `9.560e-12` |
| `malik_case4` | 2026-07-02 22:10:10 -05:00, `0182004f` | N=200 2-D temporal `solve_temporal_compressible` | 8.177 | numeric-within-1e-9 | max abs `4.389e-13`; max component rel `5.170e-11` |
| `malik_case5` | 2026-07-02 22:10:10 -05:00, `0182004f` | N=280 2-D temporal `solve_temporal_compressible` | 19.275 | numeric-within-1e-9 | max abs `1.411e-12`; max component rel `9.063e-9` (tiny growth component) |
| `malik_case6` | 2026-06-17 23:24:19 -05:00, `ecdcbda5` | N=120 spatial `solve_spatial` | 1.517 | numeric-within-1e-9 | max abs `1.106e-14`; max component rel `4.443e-12` |
| `malik_tableX` | 2026-07-02 22:10:10 -05:00, `0182004f` | N=200 spatial `solve_spatial` | 5.180 | numeric-within-1e-9 | max abs `2.887e-14`; max component rel `1.244e-12` |
| `orszag_spectrum` | 2026-07-02 22:10:10 -05:00, `0182004f` | N=128 dense Orr-Sommerfeld spectrum | 0.461 | drifted | max abs `3.235e-8` in the stored worst-error metric; 32/32 still match under `1e-5` |

No case was byte-identical. Five of six reproduce within `1e-9`. Orszag is a
real strict-numeric drift in the stored aggregate error metric, not a gate or
mode-identity failure: regenerated `max_abs_err=7.17727089329012e-6` versus
committed `7.20962323561089e-6`; all 32 modes still match under `1e-5`. Its
stored least-stable eigenvalue is additionally rounded to the paper's eight
digits, so that field is not a full-precision solver checkpoint.

## What this sizes

The census does **not** support systemic M45-scale staleness across the sampled
references. It does show that bitwise reproducibility is absent even for the
stable anchors, and one aggregate spectrum metric moves at `O(1e-8)` absolute.
The five full-precision Malik payloads remain stable to `O(1e-12)` or better,
including four numeric payloads committed in the same July-2 minute as the M45
rewrite. Against that control group, M45's `2.453e-4` relative eigenvalue drift
and `3.198%` relative growth drift are exceptional by many orders of magnitude.

This is a bounded six-case sample, not clearance for the 36 cited-not-rerun
outputs. It narrows the diagnosis: strict byte identity is generally too strong
for these dense eigensolves, but M45 is not explained by ordinary last-bit
variation.

## Walls and measurement lock

Driver inspection described the chosen cases as one dense solve each (Case 6
explicitly about 1 s; Case 4 a few seconds). Actual walls were 0.461--19.275 s,
all far below the approximately two-minute lock boundary. Therefore
`scripts/hunt/measurement_lock.py` was not entered. Each invocation had a
120-second external command timeout; none reached it. No wall was encountered.

## Exact commands and exit status

All commands exited 0:

```text
$env:PYTHONPATH=(Get-Location).Path; python -m py_compile diagnostics\reference_staleness\run_census_probe.py
$env:PYTHONPATH=(Get-Location).Path; python diagnostics\reference_staleness\run_census_probe.py orszag_spectrum
$env:PYTHONPATH=(Get-Location).Path; python diagnostics\reference_staleness\run_census_probe.py malik_case6
$env:PYTHONPATH=(Get-Location).Path; python diagnostics\reference_staleness\run_census_probe.py malik_case4
$env:PYTHONPATH=(Get-Location).Path; python diagnostics\reference_staleness\run_census_probe.py malik_case5
$env:PYTHONPATH=(Get-Location).Path; python diagnostics\reference_staleness\run_census_probe.py malik_case3
$env:PYTHONPATH=(Get-Location).Path; python diagnostics\reference_staleness\run_census_probe.py malik_tableX
```

Every scratch JSON records the exact committed and regenerated payloads, full
component drifts, wall time, current environment, BLAS thread environment, and
SHA-256 of the committed verdict it read.

## Scratch artifact set

- `SCRATCH/malik_case3.json`
- `SCRATCH/malik_case4.json`
- `SCRATCH/malik_case5.json`
- `SCRATCH/malik_case6.json`
- `SCRATCH/malik_tableX.json`
- `SCRATCH/orszag_spectrum.json`
- `run_census_probe.py` (the scratch-only driver)
