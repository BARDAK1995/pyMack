# Mach 6 Spatial Neutral Workflow

This is the canonical workflow for the Mach 6 air, `Tw/Te=5.88`, 2D
second-mode spatial calculation used by the README figure.

Run the production case:

```bash
python scripts/run_mach6_spatial_neutral_case.py --quality production
```

Run a faster guardrail check:

```bash
python scripts/run_mach6_spatial_neutral_case.py --quality smoke
```

The runner executes three steps:

1. `scripts/compute_spatial_fixed_frequency_curves.py`
2. `scripts/postprocess_spatial_amplification.py`
3. `scripts/plot_spatial_neutral_envelope.py`

The production defaults are:

- `M_e = 6`
- air, Sutherland viscosity, `T_e = 300 K`
- `Tw/Te = 5.88`
- isothermal wall
- `F = omega_L / R_L` from `6.0e-5` to `2.30e-4`
- `R_L = sqrt(Re_x)` from `200` to `3300`
- high-phase-speed second-mode branch window `0.90 <= c <= 0.97`
- `pymack_dense` backend with `pymack_continuation`

Policy:

- one fixed-frequency sweep only
- no stitched frequency bands
- no smoothed growth field
- display color limits may be set explicitly, but the metadata records both
  the full data range and the displayed range
- generated results are under `chapters/**/results/` and are intentionally
  ignored by git because they are reproducible
- the durable entry points are this document, the runner, its tests, and the
  tracked README figure in `docs/figures/`

Expected production behavior:

- every requested frequency has finite spatial-growth samples
- every requested frequency has a lower and upper neutral crossing
- for the default case, `F=6.0e-5` should give approximately
  `R_lower = 1543`, `R_upper = 3041`
- the runner writes `mach6_spatial_neutral_case_manifest.json` with
  `single_sweep=true`, `stitching=none`, and `smoothing=none`
