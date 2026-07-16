# Verification Provenance Census

Coverage: **37/37 cases**. Outcome: **1 DRIFTED**, **1 REPAIRED**, **5 DEFERRED**, and **30 other non-drifted records**. Census regeneration was scratch-only; the user-ratified M58 repair later replaced its mislabeled committed record while preserving the archived record and DRIFTED history.

## DRIFTED (1)

- [`orszag_spectrum`](provenance_census/orszag_spectrum.json): DRIFTED; maximum absolute drift 3.23523423208e-08.

## REPAIRED (1)

- [`mack_fig10_6_M58`](provenance_census/mack_fig10_6_M58.json): REPAIRED-REGENERATED; acceptable -> agrees; historical drift and current replay: was 0.06297953518 at `metrics.curve_median_rel_err`; replay 0.

## DEFERRED (5)

- [`mack_fig10_3_m1p3`](provenance_census/mack_fig10_3_m1p3.json): **DEFERRED_COST** - The production driver is an eight-station serial exact-shooting sweep, records 615-843 s per station, writes its comparable payload only after the complete sweep, and exposes neither station selection nor resume.
- [`mack_fig10_3_m2p2`](provenance_census/mack_fig10_3_m2p2.json): **DEFERRED_DRIVER_WALL** - The committed driver did not produce a comparable verdict; the captured stderr is a binding driver wall.
- [`mack_fig10_3_m3p0`](provenance_census/mack_fig10_3_m3p0.json): **DEFERRED_DRIVER_WALL** - The committed driver did not produce a comparable verdict; the captured stderr is a binding driver wall.
- [`malik_case1`](provenance_census/malik_case1.json): **DEFERRED** - No reproducible compute surface: the committed row explicitly records unknown dimensional total-temperature data and scopes the effectively incompressible coverage to Orszag. This is a scientific-input wall, not a machine-cost deferral.
- [`ozgen_m2`](provenance_census/ozgen_m2.json): **DEFERRED-RUNAWAY** - The regeneration exceeded its hard 1800 s wall; the complete worker process tree was killed under walls doctrine.

## Master table

| Case | Committed when | Regeneration config | Verdict | Drift | Wall | Provenance |
|---|---|---|---|---:|---:|---|
| `orszag_spectrum` | 2026-07-02T22:10:10-05:00 (`0182004`) | carried 2026-07-12; `validation/test_orszag_full_spectrum.py::poiseuille_spectrum` | DRIFTED | 3.23523423208e-08 | 0.461 s | [census JSON](provenance_census/orszag_spectrum.json) |
| `mack_fig10_6_M58` | 2026-07-15T05:09:35-05:00 (`fe8f03f`) | reference repair; M=5.8; N=150; ymax=64; 12 R x 51 alpha; cpu; BLAS=1 | REPAIRED-REGENERATED; acceptable -> agrees | was 0.06297953518 at `metrics.curve_median_rel_err`; replay 0 | 486.171 s | [census JSON](provenance_census/mack_fig10_6_M58.json) |
| `mack_fig10_3_m1p3` | 2026-07-02T22:09:38-05:00 (`f0f47bf`) | not rerun; exact 8-station production sweep; historical wall=5951.3 s | DEFERRED_COST | not measured | not run | [census JSON](provenance_census/mack_fig10_3_m1p3.json) |
| `mack_fig10_3_m2p2` | 2026-07-02T22:09:38-05:00 (`f0f47bf`) | scratch rerun; BLAS=1; exact command in JSON | DEFERRED_DRIVER_WALL | not measured | 0.583 s | [census JSON](provenance_census/mack_fig10_3_m2p2.json) |
| `mack_fig10_3_m3p0` | 2026-07-02T22:09:38-05:00 (`f0f47bf`) | scratch rerun; BLAS=1; exact command in JSON | DEFERRED_DRIVER_WALL | not measured | 0.549 s | [census JSON](provenance_census/mack_fig10_3_m3p0.json) |
| `malik_case1` | 2026-06-19T05:37:26-05:00 (`fbaca20`) | not runnable; required dimensional total temperature unavailable | DEFERRED | not measured | not run | [census JSON](provenance_census/malik_case1.json) |
| `ozgen_m2` | 2026-07-02T22:09:54-05:00 (`8888319`) | M=2; N=220; serial-eigenvalue-continuation; BLAS=1 | DEFERRED-RUNAWAY | not measured | 1800.158 s / 1800 s timeout | [census JSON](provenance_census/ozgen_m2.json) |
| `balakumar_malik1992_branches` | 2026-06-19T09:28:48-05:00 (`e963286`) | scratch rerun; BLAS=1; exact command in JSON | acceptable (numeric-within-1e-9) | 0 at `metrics.N` (within 1e-9) | 27.067 s | [census JSON](provenance_census/balakumar_malik1992_branches.json) |
| `balakumar_malik1992_via_xirenfu` | 2026-06-19T09:28:48-05:00 (`e963286`) | scratch rerun; BLAS=1; exact command in JSON | acceptable (numeric-within-1e-9) | 4.18942658342e-13 at `metrics.alpha_i_rel_err` (within 1e-9) | 1.686 s | [census JSON](provenance_census/balakumar_malik1992_via_xirenfu.json) |
| `cone_sivasubramanian_fasel_2015` | 2026-07-02T22:09:38-05:00 (`f0f47bf`) | N=90; ymax=40; 840 solves; serial frequency-station-grid; BLAS=1 | acceptable (numeric-within-1e-9) | 0 at `metrics.N_collocation` (within 1e-9) | 416.733 s | [census JSON](provenance_census/cone_sivasubramanian_fasel_2015.json) |
| `egorov2006_m6` | 2026-06-19T09:28:48-05:00 (`e963286`) | scratch rerun; BLAS=1; exact command in JSON | acceptable (numeric-within-1e-9) | 0 at `metrics.N` (within 1e-9) | 45.083 s | [census JSON](provenance_census/egorov2006_m6.json) |
| `mack_fig10_1_m1p6` | 2026-07-02T22:09:38-05:00 (`f0f47bf`) | scratch rerun; BLAS=1; exact command in JSON | acceptable (numeric-within-1e-9) | 0 at `metrics.R_crit_mack` (within 1e-9) | 111.777 s | [census JSON](provenance_census/mack_fig10_1_m1p6.json) |
| `mack_fig10_1_m2p2` | 2026-07-02T22:09:38-05:00 (`f0f47bf`) | scratch rerun; BLAS=1; exact command in JSON | disagrees (numeric-within-1e-9) | 0 at `metrics.R_crit_mack` (within 1e-9) | 129.333 s | [census JSON](provenance_census/mack_fig10_1_m2p2.json) |
| `mack_fig10_4_M100` | 2026-07-02T22:09:38-05:00 (`f0f47bf`) | M=10.0; N=200; ymax=150; 9 R x 17 alpha; point-parallel; 61 workers; BLAS=1 | acceptable (numeric-within-1e-9) | 0 at `metrics.N` (within 1e-9) | 1079.373 s | [census JSON](provenance_census/mack_fig10_4_M100.json) |
| `mack_fig10_4_M45` | 2026-07-02T22:09:38-05:00 (`f0f47bf`) | M=4.5; N=130; ymax=42; 12 R x 17 alpha; point-parallel; 61 workers; BLAS=1 | agrees (numeric-within-1e-9) | 0 at `metrics.N` (within 1e-9) | 367.364 s | [census JSON](provenance_census/mack_fig10_4_M45.json) |
| `mack_fig10_4_M58` | 2026-07-02T22:09:38-05:00 (`f0f47bf`) | M=5.8; N=150; ymax=64; 12 R x 17 alpha; point-parallel; 61 workers; BLAS=1 | agrees (numeric-within-1e-9) | 0 at `metrics.N` (within 1e-9) | 575.030 s | [census JSON](provenance_census/mack_fig10_4_M58.json) |
| `mack_fig10_4_M70` | 2026-07-02T22:09:38-05:00 (`f0f47bf`) | M=7.0; N=170; ymax=86; 11 R x 14 alpha; point-parallel; 61 workers; BLAS=1 | agrees (numeric-within-1e-9) | 0 at `metrics.N` (within 1e-9) | 668.195 s | [census JSON](provenance_census/mack_fig10_4_M70.json) |
| `mack_fig10_4_family` | 2026-06-19T05:37:26-05:00 (`fbaca20`) | pointer-only finalizer; `verification/verify_mack_fig10_4.py` | byte-identical | 0 (byte-identical) | 0.000 s | [census JSON](provenance_census/mack_fig10_4_family.json) |
| `mack_fig10_6_M100` | 2026-06-19T09:28:48-05:00 (`e963286`) | M=10.0; N=200; ymax=140; 11 R x 41 alpha; station-serial-alpha-parallel; 1 station worker; BLAS=1 | agrees (numeric-within-1e-9) | 0 at `metrics.N` (within 1e-9) | 944.190 s | [census JSON](provenance_census/mack_fig10_6_M100.json) |
| `mack_fig10_6_M45` | 2026-07-13T10:46:48-05:00 (`67b4cc2`) | fresh inherited run; M=4.5; N=120; ymax=40; 11 R x 65 alpha; cpu | byte-identical | 0 (byte-identical) | 272.912 s | [census JSON](provenance_census/mack_fig10_6_M45.json) |
| `mack_fig10_6_M70` | 2026-06-19T09:28:48-05:00 (`e963286`) | M=7.0; N=170; ymax=88; 12 R x 47 alpha; station-serial-alpha-parallel; 1 station worker; BLAS=1 | agrees (numeric-within-1e-9) | 0 at `metrics.N` (within 1e-9) | 660.356 s | [census JSON](provenance_census/mack_fig10_6_M70.json) |
| `mack_fig10_6_family` | 2026-06-19T05:37:26-05:00 (`fbaca20`) | pointer-only finalizer; `verification/verify_mack_fig10_6.py` | byte-identical | 0 (byte-identical) | 0.000 s | [census JSON](provenance_census/mack_fig10_6_family.json) |
| `malik_case3` | 2026-07-02T22:10:10-05:00 (`0182004`) | carried 2026-07-12; `validation/test_malik1990_case3_anchor.py` | numeric-within-1e-9 | 8.40993941154e-15 (within 1e-9) | 15.010 s | [census JSON](provenance_census/malik_case3.json) |
| `malik_case4` | 2026-07-02T22:10:10-05:00 (`0182004`) | carried 2026-07-12; `validation/test_malik1990_case4_anchor.py` | numeric-within-1e-9 | 4.38857283847e-13 (within 1e-9) | 8.177 s | [census JSON](provenance_census/malik_case4.json) |
| `malik_case5` | 2026-07-02T22:10:10-05:00 (`0182004`) | carried 2026-07-12; `validation/test_malik1990_case5_anchor.py` | numeric-within-1e-9 | 1.41111531097e-12 (within 1e-9) | 19.275 s | [census JSON](provenance_census/malik_case5.json) |
| `malik_case6` | 2026-06-19T09:28:48-05:00 (`e963286`) | carried 2026-07-12; `validation/test_malik1990_case6_anchor.py` | numeric-within-1e-9 | 1.10623316063e-14 (within 1e-9) | 1.517 s | [census JSON](provenance_census/malik_case6.json) |
| `malik_fig4_eigenfunction` | 2026-07-02T22:10:10-05:00 (`0182004`) | scratch rerun; BLAS=1; exact command in JSON | agrees (numeric-within-1e-9) | 7.52596207576e-10 at `metrics.T_over_u_dominance` (within 1e-9) | 9.088 s | [census JSON](provenance_census/malik_fig4_eigenfunction.json) |
| `malik_tableX` | 2026-07-02T22:10:10-05:00 (`0182004`) | carried 2026-07-12; `validation/test_malik1990_tableX_anchor.py` | numeric-within-1e-9 | 2.88657986403e-14 (within 1e-9) | 5.180 s | [census JSON](provenance_census/malik_tableX.json) |
| `mazhong2003_m4p5` | 2026-07-02T22:10:10-05:00 (`0182004`) | scratch rerun; BLAS=1; exact command in JSON | agrees (numeric-within-1e-9) | 0 at `metrics.F` (within 1e-9) | 54.897 s | [census JSON](provenance_census/mazhong2003_m4p5.json) |
| `ozgen_fig3_lobes` | 2026-07-02T22:09:54-05:00 (`8888319`) | fresh inherited run; M=2.0; N=128; 720 nodes; cpu; 24 workers; BLAS=1 | numeric-within-1e-9 | 1.00000002723e-09 (within 1e-9) | 110.951 s | [census JSON](provenance_census/ozgen_fig3_lobes.json) |
| `ozgen_m10` | 2026-07-02T22:09:54-05:00 (`8888319`) | M=10; N=200/180; 677 nodes; point-parallel-discrete-mode-grid; 61 workers; BLAS=1 | acceptable (byte-identical) | 0 (byte-identical) | 971.922 s | [census JSON](provenance_census/ozgen_m10.json) |
| `ozgen_m3` | 2026-07-02T22:09:54-05:00 (`8888319`) | M=3; N=200; 260 nodes; point-parallel-discrete-mode-grid; 61 workers; BLAS=1 | acceptable (byte-identical) | 0 (byte-identical) | 472.739 s | [census JSON](provenance_census/ozgen_m3.json) |
| `ozgen_m4` | 2026-07-02T22:09:54-05:00 (`8888319`) | M=4; N=200/180; 442 nodes; point-parallel-discrete-mode-grid; 61 workers; BLAS=1 | agrees (byte-identical) | 0 (byte-identical) | 698.626 s | [census JSON](provenance_census/ozgen_m4.json) |
| `ozgen_m6` | 2026-07-02T22:09:54-05:00 (`8888319`) | M=6; N=200/180; 481 nodes; point-parallel-discrete-mode-grid; 61 workers; BLAS=1 | acceptable (byte-identical) | 0 (byte-identical) | 766.944 s | [census JSON](provenance_census/ozgen_m6.json) |
| `ozgen_m7` | 2026-07-02T22:09:54-05:00 (`8888319`) | M=7; N=200/180; 677 nodes; point-parallel-discrete-mode-grid; 61 workers; BLAS=1 | acceptable (byte-identical) | 0 (byte-identical) | 1041.547 s | [census JSON](provenance_census/ozgen_m7.json) |
| `ozgen_m8` | 2026-07-02T22:09:54-05:00 (`8888319`) | M=8; N=200/180; 677 nodes; point-parallel-discrete-mode-grid; 61 workers; BLAS=1 | acceptable (numeric-within-1e-9) | 2.50939269364e-11 at `metrics.per_branch_nearest_rel_err.first_lower.median_rel_err` (within 1e-9) | 1006.448 s | [census JSON](provenance_census/ozgen_m8.json) |
| `sean_m5p35` | 2026-06-19T09:28:48-05:00 (`e963286`) | scratch rerun; BLAS=1 | acceptable (numeric-within-1e-9) | 0 at `metrics.lower_branch_MAE_mm_330_600kHz` (within 1e-9) | 1.481 s | [census JSON](provenance_census/sean_m5p35.json) |

Each census JSON is the authoritative provenance pointer: it records the committed reference and commit time, exact command, effective parameters, runtime environment, hashes, comparison, wall, timeout, and lock evidence applicable to that case.
