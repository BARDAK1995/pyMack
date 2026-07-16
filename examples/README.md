# pyMack examples

Small, runnable introductions -- each is a single file with no arguments.

| Example | What it shows | Runtime |
|---|---|---|
| `01_first_mack_mode.py` | Base flow -> temporal & spatial Mack mode -> eigenfunctions | ~15 s |
| `02_growth_curve_and_n_factor.py` | Fixed-frequency downstream march, sigma_L(R), N-factor | ~1 min |
| `03_dimensional_units.py` | Nondimensional results -> kHz / mm / 1/m | instant |
| `04_parameter_sweep.py` | `pymack.sweep`: one call over an (alpha, Re) grid -> stability map | ~10 s |

Run from the repository root (or after `pip install -e .`):

```bash
python examples/01_first_mack_mode.py
```

For complete workflows (neutral-curve tracing, production Mach-6 case,
cone N-factors) see `scripts/` and `docs/LST_API_CHEATSHEET.md`. For the
batch-sweep facade (`pymack.sweep`, example 04) see
[`docs/SWEEP_API.md`](../docs/SWEEP_API.md); every example here runs on the
public CPU implementation.
