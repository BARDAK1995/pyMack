# AGENT ONBOARDING — read this first

You (the agent) have been launched to work on **pyMack-GPU**: a GPU-native, batched
linear-stability-theory (LST) engine built on top of the validated pyMack solver. This file tells
you where the plan lives and how to proceed. Read it fully before doing anything.

## 1. Make sure you are on the right branch

The entire GPU plan lives on the **`gpu`** branch only (NOT on `master` or `verification`).

```
git branch --show-current      # must print: gpu
git status                     # working tree should be clean
```

If you are not on `gpu`, run `git checkout gpu`. The plan files below only exist on this branch.

## 2. Read the plan (this IS the source of truth — self-contained)

Read these in order, under `docs/gpu/`:

1. **`docs/gpu/README.md`** — one-page index + the key measured facts.
2. **`docs/gpu/PLAN.md`** — THE approved plan: goal, the 4-pillar scheme, architecture
   (`pymack/sweep.py` + `pymack/gpu/`), milestones **M0–M6** each with a wall-clock gate, risks,
   verification strategy.
3. **`docs/gpu/design_algorithms.md`** — numerical spec (batched two-sided RQI, bordered
   implicit-determinant Newton, wavefront continuation, mixed-precision scheme, performance model).
4. **`docs/gpu/design_architecture.md`** — software integration design + public API + roadmap protocols.
5. **`docs/gpu/design_redteam.md`** — adversarial review with measurements on the real matrices
   (affinity census, condition numbers, mode-tracking failure analysis, baseline-fairness design).

## 3. Non-negotiable ground rules (from the plan)

- **Goal ordering:** build something GREAT first (first-of-its-kind, drastically faster, scales).
  Publication (JCP/CPC methods paper) is a byproduct, not the driver. Performance / scalability /
  GPU-nativeness win design ties.
- The pyMack repo is **PUBLIC**. NEVER commit copyrighted material (reference-paper PDFs/scans;
  `refPapers/` is gitignored). Production overlays plot pyMack curves over digitized POINTS only.
- **Do NOT change the behavior of the validated CPU solver.** The GPU engine consumes the existing
  CPU assembly functions as oracles (via probing) and the CPU QZ solvers as seeds/fallbacks. The
  37-case verification harness (`verification/`, `verdict.json`, `build_success_matrix.py`) is the
  acceptance gate — GPU results must be verdict-identical to the CPU path.
- **Commit/push only when the user explicitly asks.** No `Co-Authored-By: Claude` trailers.
- Every milestone is gated by a real wall-clock number on the local GPU — if a step doesn't move a
  real workload's clock, resequence it.

## 4. Environment (verified working 2026-07-03 on this Windows 10 machine)

- Python **3.12.7** (`C:\Program Files\Python312\python.exe`) — always invoke as `python` or
  `py -3.12`. (3.10 and 3.8 are also installed; do not use them.)
- GPU: **NVIDIA RTX 4070 Ti, 12 GB, compute 8.9 (Ada)**, driver 13.2, CUDA runtime 12.9.
- Stack: `cupy-cuda12x` 14.1.1 + the NVIDIA redistributable library wheels
  (`nvidia-cublas-cu12`, `cusolver`, `cusparse`, `cufft`, `curand`, `cuda-nvrtc`, `cuda-runtime`,
  `nvjitlink` — all CUDA 12.9), numpy 2.2.6, scipy 1.14.1.
- **GOTCHA:** the `cupy-cuda12x` v14 wheel does NOT bundle the CUDA libraries — if `cublasLt*.dll`
  is "not found", `pip install` the `nvidia-*-cu12` wheels above (they are what make it work).
- Smoke-tested green: batched complex64 LU solve (cuBLAS), `eigvalsh` (cuSOLVER), and RawKernel
  NVRTC compilation of a `complex<float>` kernel (the key Windows risk — it works).
- CuPy's batched `cupy.linalg.solve(A, b)` needs `b` shaped `(batch, n, K)`, not `(batch, n)`.

## 5. How to proceed

Start at **milestone M0** (the CuPy benchmark spike) as written in `docs/gpu/PLAN.md`, unless the
user directs otherwise. Before trusting any GPU timing, the user must set (one-time, manual):
**NVIDIA Control Panel → Manage 3D Settings → CUDA - Sysmem Fallback Policy → "Prefer No Sysmem
Fallback"** (driver ≥536 silently spills VRAM to host RAM and corrupts benchmarks).

Then work milestone by milestone, keeping the CPU path untouched and gating each step on the
verification harness and a measured wall-clock number.
