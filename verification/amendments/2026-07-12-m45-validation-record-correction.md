# Amendment: Mack Fig. 10.6 M=4.5 validation record

Amendment recorded: 2026-07-13

User ratification: 2026-07-12, option `fix + regenerate + phased census`

## What was wrong

`verification/verify_mack_fig10_6.py::verify_mach` had two inconsistent paths.
Its single-Mach path called `compute_curve(mach)` and therefore consumed the
legacy defaults N=110/y_max=30, while the verdict metadata independently read
the per-Mach maps and claimed N=120/y_max=40. The all-Mach parallel path already
consumed those maps correctly.

The July-2 committed M45 curve and overlay therefore represent the N110/y30
calculation and the accompanying verdict is mislabeled. The record has been
moved intact to
`verification/_archive/reference_records/mack_fig10_6_M45/2026-07-02_mislabeled_N110_ymax30/`.

## Fingerprint evidence

- Across all 11 rows and the fields `omega_i_max`, `alpha_peak`, `c_r`, and
  `c_i`, the July-2 curve differs from the original June-17 N110/y30 curve by
  at most `2.0883295093199195e-12` absolute.
- Its maximum absolute difference from the corrected June-18 N120/y40 curve is
  `1.8021177799842913e-4`.
- At R=300, the mislabeled curve stores
  `omega_i_max=0.0008529766267072418`; the N120/y40 curve stores
  `0.0008802555034254461`.
- A bounded live probe reproduced the former with N110/y30 to `4.651e-13`
  maximum absolute drift and reproduced the latter exactly with N120/y40.
- The archived curve SHA-256 is
  `a8b49dad0ad4a3476b92b06d01b4483ad6b7962d8bbb3ad0c4131380f2d62237`.

## Correction and standing scope

Source commit `a0b331aca97a2e342807e45c6ebaaf55e38b4d2f` routes the
single-Mach compute call through the same effective N/y_max values recorded in
metadata and adds a regression gate for that equality. The replacement M45
record is regenerated at N=120/y_max=40 with embedded runtime, command, source,
effective-parameter, and SHA-256 provenance.

This amendment corrects M45 only. The phased census of the remaining committed
references remains standing; it is not converted into a clearance claim by
this repair.
