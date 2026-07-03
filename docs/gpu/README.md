# pyMack-GPU planning record (`gpu` branch)

This directory records the approved plan for **pyMack-GPU** — a GPU-native batched LST engine:
full stability diagrams as single batched GPU computations instead of thousands of serial
full-spectrum QZ solves. Planned 2026-07-02; execution deferred (start at milestone M0).

| Document | What it is |
|---|---|
| [PLAN.md](PLAN.md) | The approved plan: goals, the 4-pillar scheme, architecture, milestones M0–M6 with wall-clock gates, risks, verification |
| [design_algorithms.md](design_algorithms.md) | Numerical algorithm spec: batched two-sided RQI, bordered implicit-determinant Newton, wavefront continuation, mixed-precision scheme, per-workload mapping, performance model |
| [design_architecture.md](design_architecture.md) | Software integration design: `pymack/sweep.py` + `pymack/gpu/` layout, AffineOperatorCache probe API, batch scheduler, certification harness, roadmap protocols |
| [design_redteam.md](design_redteam.md) | Adversarial review with measured evidence: affinity census per workload, condition-number/precision measurements, mode-tracking failure analysis, baseline-fairness benchmark design |

Key headline facts (measured on the real pyMack matrices during planning):
- The assembled operators are **exactly affine** in the sweep parameters (residuals ~1e-16);
  the 3D wave-aligned operator requires the (k, cos ψ, sin ψ)×{1, 1/Re} basis.
- Two-sided equilibration collapses κ from 2.0e10 → 1.2e5 on the M10 dim-1005 flagship,
  enabling complex64 batched LU + FP64 refinement to ~1e-11 eigenvalue accuracy.
- CPU prototypes of the batched kernels already beat QZ 10–30× per point (0.29 s vs 7.3 s at
  dim 1005); the GPU batching multiplies that by the hardware factor.

Status: **planned, not yet implemented**. First step on resume: M0 CuPy spike (see PLAN.md).
