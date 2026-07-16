# Reproducibility commands

Run commands from the repository root. This public snapshot contains the CPU
package, its complete public test suite, the JOSS paper, the judged validation
record, and the original machine-readable CPU performance artifacts.

## Fresh virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
$env:PYTHONPATH = (Resolve-Path .).Path
python -c "import pymack; print(pymack.__file__)"
python -m pytest -q
```

POSIX shells use `source .venv/bin/activate` and
`export PYTHONPATH="$PWD"`.

The printed import path must resolve inside this checkout. The release gate is
the full public suite, not a reduced smoke subset.

## Render the paper

From `paper/`:

```powershell
pandoc paper.md --citeproc --bibliography paper.bib --standalone --output paper.html
```

The command must exit zero and the rendered HTML must resolve
`figures/ozgen_m2_4x_ci_map.png` from this tree.

## Performance evidence

The performance measurements are hardware-specific records, not portable
promises. Their exact JSON artifacts are indexed in
[`docs/benchmarks/README.md`](benchmarks/README.md). They are copied without
regeneration or content edits so recorded commands, source commits, machine
paths, and environment fields remain honest.

The key CPU paths can be exercised with the committed user-facing drivers:

```powershell
python scripts/make_ozgen_fig3_overlay.py --quality production --panels 2 --engine sweep --workers 24 --blas-threads 1 --eigenvalues-only --verify-against-committed
python verification/compute_mack_fig10_4.py --mach 10 --point-parallel --workers 61 --blas-threads 1 --verify-against-committed
```

These are expensive workload runs. Use a clean dedicated worktree and preserve
new outputs separately; do not overwrite the committed measurement records.

## Validation and provenance boundary

`verification/SUCCESS_MATRIX.md` is the judged scientific record.
`verification/PROVENANCE_CENSUS.md` records which verdicts were regenerated,
repaired, drifted slightly, or deferred. The amendments and
`verification/_archive/` preserve the reference-repair history instead of
silently replacing it.
