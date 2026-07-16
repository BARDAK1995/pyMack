# Amendment: Mack Fig. 10.6 M=5.8 validation record

Amendment recorded: 2026-07-15

User ratification: 2026-07-15, M58 validation-record repair following the M45
worked precedent

## What was wrong

Before the M45 repair, `verification/verify_mack_fig10_6.py::verify_mach` had
two inconsistent paths. Its single-Mach path called `compute_curve(mach)` and
therefore consumed the legacy defaults N=110/y_max=30, while the verdict
metadata independently read the per-Mach maps and claimed the M=5.8 values
N=150/y_max=64. The all-Mach parallel path already consumed those maps
correctly.

The July 2 committed M58 curve and overlay therefore represent the N110/y30
calculation and the accompanying verdict is mislabeled. The record has been
moved intact to
`verification/_archive/reference_records/mack_fig10_6_M58/2026-07-02_mislabeled_N110_ymax30/`.

## Fingerprint evidence

- Across all 12 rows, the July 2 curve differs from the original June 17
  N110/y30 curve by at most `3.746669641202516e-12` absolute over
  `omega_i_max`, `alpha_peak`, `c_r`, and `c_i`.
- Its maximum absolute `omega_i_max` difference from the corrected June 18
  N150/y64 curve is `3.177367236255812e-4`, at R=240.
- At R=240, the mislabeled curve stores
  `omega_i_max=0.0008692116915646328`; the corrected June 18 N150/y64 curve
  stores `0.001186948415190214`.
- The July 15 census regeneration at N150/y64 changed
  `metrics.curve_median_rel_err` by `0.06297953517954663` and changed the
  scientific verdict from `acceptable` to `agrees`.
- The archived curve SHA-256 is
  `80a9cccaab8631da592bf712139952d295598564887934a28e633a0fe208d1d7`.

## Correction and standing scope

Source commit `a0b331aca97a2e342807e45c6ebaaf55e38b4d2f` routes the
single-Mach compute call through the same effective N/y_max values recorded in
metadata and adds a regression gate for that equality. The replacement M58
record is regenerated at N=150/y_max=64 with embedded runtime, command, source,
effective-parameter, and SHA-256 provenance.

This is the second confirmed instance of the M45 mislabel class. This
amendment corrects M58 only: it does not alter any other committed reference,
`pymack/` code, or paper wording, and it does not erase the census history that
M58 was `DRIFTED` before repair.
