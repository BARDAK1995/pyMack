# Archived M58 record: provenance correction

Archived: 2026-07-15

Original record commit: `f0f47bfb072e971092b1a97ccd2164901042a8ee`
(authored 2026-07-02T22:09:38-05:00)

User-ratified correction: 2026-07-15

This preserved record is the Mack Fig. 10.6 M=5.8 curve computed with the
legacy defaults **N=110 and y_max=30**, despite its original verdict claiming
N=150 and y_max=64. It must not be used as the N150/y64 reference.

The July 15 provenance census recorded this case as `DRIFTED`: regeneration at
N=150/y_max=64 changed `metrics.curve_median_rel_err` by
`0.06297953517954663` and changed the scientific verdict from `acceptable` to
`agrees`. The archived curve also fingerprints the June 17 legacy N110/y30
curve across all 12 rows: maximum absolute differences are
`4.374872859813639e-13` in `omega_i_max`, exactly `0.0` in `alpha_peak`,
`3.746669641202516e-12` in `c_r`, and `3.1249100529429086e-12` in `c_i`.

The files are retained verbatim under corrected names:

- `pymack_curve.N110_ymax30.json` - SHA-256
  `80a9cccaab8631da592bf712139952d295598564887934a28e633a0fe208d1d7`;
- `verdict.mislabeled_N110_ymax30.json` - SHA-256
  `b2f6aac16e5c48e0ccd92837c20fb57cde4852871a5ac81fdbd7cf1f9330bb6b`;
- `overlay.mislabeled_N110_ymax30.png` - SHA-256
  `db78b160d4fa5d2ec8e8156764564c6e2c8c713afe6b70d4b63f477f389e6f4f`.

The original JSON is intentionally not edited: its incorrect N150/y64 claim
is part of the evidence. The corrected provenance is this note and the archive
path. The replacement record is generated only through the fixed single-Mach
verifier, which passes the per-Mach parameter-map values into the compute call.
