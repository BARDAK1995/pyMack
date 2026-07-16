# Archived M45 record: provenance correction

Archived: 2026-07-13

Original record commit: `f0f47bfb072e971092b1a97ccd2164901042a8ee`

User-ratified correction: 2026-07-12

This preserved record is the Mack Fig. 10.6 M=4.5 curve computed with the
legacy defaults **N=110 and y_max=30**, despite its original verdict claiming
N=120 and y_max=40. It must not be used as the N120/y40 reference.

The files are retained verbatim under corrected names:

- `pymack_curve.N110_ymax30.json` — SHA-256
  `a8b49dad0ad4a3476b92b06d01b4483ad6b7962d8bbb3ad0c4131380f2d62237`;
- `verdict.mislabeled_N110_ymax30.json` — SHA-256
  `15e91dd62025b5d1761d866f597e67014cde3f0e28c11b97ec26623dfe998e74`;
- `overlay.mislabeled_N110_ymax30.png` — SHA-256
  `55f9edcc6652fbdd3ec0ca30db829c9e7d9b9192b6fa1780f0c4ad1bb967de3f`.

The original JSON is intentionally not edited: its incorrect N120/y40 claim is
part of the evidence. The corrected provenance is this note and the archive
path. The replacement record is generated only after the single-Mach verifier
was fixed to pass its per-Mach parameter-map values into the compute call.
