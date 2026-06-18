# Verification Targets — exact conditions

Every case in the audit, its published conditions, the digitized/external
reference, the pyMack tool that generates it, and the compute tier. Conditions
are taken from `reference_data/paper_target_registry.json`,
`reference_data/collaborator_mach5p35/conditions.json`, and the Mack/Özgen
papers. "Tier" is compute readiness, not importance:

- **T0** already computed (reuse)
- **T1** cheap re-run with an existing parameterized tool
- **T2** moderate — needs an existing tool generalized
- **T3** stretch — needs new generation machinery and/or has known open issues

## Neutral curves

| case_id | source | M | wall | ψ | gas/transport | formulation | reference data | tool | tier |
|---|---|---|---|---|---|---|---|---|---|
| `sean_m5p35` | Collaborator independent LST | 5.35 | adiabatic≈370 K | 0 | N₂, Sutherland/power-law | spatial 2nd mode | `collaborator_mach5p35/LST_neutral_curve_M5p35.csv` | `run_mach6_spatial_neutral_case.py` | **T0** |
| `ozgen_m2` | Özgen & Kırcalı (2008) Fig 3 | 2.0 | adiabatic | 0 | air, Özgen T-dependent | temporal 2D | `digitized/ozgen_fig3_M2_neutral.csv` | `make_ozgen_fig3_overlay.py` | **T0/T1** |
| `ozgen_m3` | Özgen & Kırcalı (2008) Fig 3 | 3.0 | adiabatic | 0 | air, Özgen T-dependent | temporal 2D | `digitized/ozgen_fig3_M3_neutral.csv` | `make_ozgen_fig3_overlay.py` | **T1** |
| `ozgen_m4` | Özgen & Kırcalı (2008) Fig 3 | 4.0 | adiabatic | 0 | air, Özgen T-dependent | temporal 2D | `digitized/ozgen_fig3_M4_neutral.csv` | `make_ozgen_fig3_overlay.py` | **T0/T1** |
| `ozgen_m6` | Özgen & Kırcalı (2008) Fig 3 | 6.0 | adiabatic | 0 | air, Özgen T-dependent | temporal 2D | `digitized/ozgen_fig3_M6_neutral.csv` | `make_ozgen_fig3_overlay.py` | **T1** |
| `mack_fig10_1_m1p6` | Mack (1984) Fig 10.1 | 1.6 | adiabatic | 0 | air, Mack transport | temporal neutral frequency | `digitized/mack_ch10_fig10_1_M16_paper_*.csv` | (new temporal-neutral runner) | **T3** |
| `mack_fig10_1_m2p2` | Mack (1984) Fig 10.1 | 2.2 | adiabatic | 0 | air, Mack transport | temporal neutral frequency | `digitized/mack_ch10_fig10_1_M22_paper_*.csv` | (new temporal-neutral runner) | **T3** |

Notes — Özgen Fig 3 uses the Özgen temperature-dependent transport model and
adiabatic wall (their Eq. 19 disturbance BC). Fig 3 also carries c_iᵢ growth
contours (levels 0.002/0.004/0.012) → see growth-rate section. Mack Fig 10.1 is
the low-Mach first-mode viscous neutral-*frequency* curve; pyMack has no clean
public temporal-neutral runner for it yet (only the Fig 10.3 max-growth path),
so it is T3.

## Growth rates

| case_id | source | M | ψ | quantity | reference data | tool | tier |
|---|---|---|---|---|---|---|---|
| `mack_fig10_3_m1p3` | Mack (1984) Fig 10.3 | 1.3 | 45° | max temporal ω_i vs R | `digitized/mack_ch10_fig10_3_M13_paper_psi45.csv` | `make_mack_fig10_3_overlay.py` | **T0** |
| `mack_fig10_3_m2p2` | Mack (1984) Fig 10.3 | 2.2 | 45° | max temporal ω_i vs R | `digitized/mack_ch10_fig10_3_M22_paper_psi45.csv` | generalize Mack overlay | **T2** |
| `mack_fig10_3_m3p0` | Mack (1984) Fig 10.3 | 3.0 | 60° | max temporal ω_i vs R | `digitized/mack_ch10_fig10_3_M30_paper_psi60.csv` | generalize Mack overlay | **T2** |
| `ozgen_fig3_lobes` | Özgen & Kırcalı (2008) Fig 3 | 2–6 | 0 | c_iᵢ growth contours (0.002/0.004/0.012) | `digitized/ozgen_fig3_M*_004/012*.csv` | `make_ozgen_fig3_overlay.py` | **T1** |
| `mack_fig10_4_*` | Mack (1984) Fig 10.4 | 4.5/5.8/7.0/10 | first-mode opt | max first-mode ω_i vs R | `digitized/mack_ch10_fig10_4_M*_paper.csv` | (new high-Mach runner) | **T3** |
| `mack_fig10_6_*` | Mack (1984) Fig 10.6 | 4.5/5.8/7.0/10 | 0 (2D) | max second-mode ω_i vs R | `digitized/mack_ch10_fig10_6_M*_paper.csv` | (new high-Mach runner) | **T3** ⚠ |

⚠ **Mack Fig 10.6** was probed earlier and rejected as a clean overlay target: a
grid-converged probe showed a ~6× magnitude gap under the repo's condition
mapping. It is kept here as an explicit ❌/⬜ matrix row (the audit reports
disagreements, it does not hide them), pending a condition-mapping resolution.

## Out of current scope (recorded for completeness)

Mack Ch. 9 inviscid higher-mode figures (9.1–9.13), Mack Fig 10.2/10.5/10.7–
10.11, Özgen Fig 6/7/8 (wave-angle/spatial-3D) and Fig 9/10 (reference-
temperature) — see the registry. These need inviscid or true-3D-spatial
machinery beyond the neutral-curve / max-growth scope of this audit.
