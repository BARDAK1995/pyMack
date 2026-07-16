# M45 deep dive and ranked causes

Stage: 3 of 3 (diagnosis only)

## Most likely cause

The committed July-2 M45 curve was generated through the verifier's
single-Mach/default-parameter path at `N=110, y_max=30`, then labeled in its
verdict as the corrected per-Mach `N=120, y_max=40` result.

This is a driver invocation/provenance defect, not evidence that the current
solver changed the intended N120/y40 result.

## Decisive evidence chain

1. The original June-17 M45 curve (`ecdcbda5`) was generated before the
   per-Mach domain maps existed, so it used `N_DEFAULT=110` and
   `Y_MAX_DEFAULT=30`.
2. Commit `cfa4ffcd` added the corrected M45 maps `N=120`, `y_max=40` and
   replaced the curve. Slice 13R's current CPU rows are exactly equal across all
   11 stations to that June-18 corrected curve.
3. The July-2/current committed curve is numerically the June-17 default curve:
   across all 11 rows and the fields `omega_i_max`, `alpha_peak`, `c_r`, and
   `c_i`, its maximum absolute difference from the June-17 curve is
   `2.0883295093199195e-12`. Its maximum absolute difference from the June-18
   corrected curve is `1.8021177799842913e-4`.
4. `verification/verify_mack_fig10_6.py::verify_mach` computes rows with
   `engine.compute_curve(mach, verbose=True)` when called without supplied
   rows. That function's defaults remain `N=110`, `y_max=30`.
5. The same verifier writes metrics and provenance with
   `engine._N_for(mach)` / `engine._ymax_for(mach)`, which report `N=120`,
   `y_max=40` for M45 regardless of which compute path produced the rows.
6. The all-Mach verifier path supplies rows from `compute_curves_parallel`,
   which correctly routes through the per-Mach maps. The single-Mach path
   (`--mach 4.5 --force`) does not. Thus the code contains a direct route to
   produce exactly the observed stale/mislabeled artifact.

The July-2 commit changed only output/reference material and path strings in
the verifier, not the numerical engine. Its M45 curve reverted to the old
default-domain numeric family while its metadata continued to claim the
corrected domain.

## Ranked hypotheses

1. **Single-Mach verifier used default N/y_max and mislabeled the result.**
   Direct code path plus 11-row numeric fingerprint; overwhelmingly strongest.
2. **Equivalent stale artifact or alternate default-parameter invocation was
   copied during the July-2 audit.** Also consistent, but less specific than the
   live code path that already explains the fingerprint.
3. **Environment/BLAS/LAPACK change.** Historical JSON records no environment,
   so this cannot be categorically excluded. It is weakly supported: four
   full-precision Malik payloads committed at 22:10:10 on July 2 reproduce in
   the current NumPy 2.2.6 / SciPy 1.14.1 stack to `O(1e-12)` or better, and
   M45 matches a parameter-specific older curve rather than showing an
   unstructured backend perturbation.
4. **Solver source behavior changed.** Poorly supported. Slice 13R exactly
   recovers the corrected June-18 curve, and the post-generation 2-D solver
   change was the July-4 assembly extraction certified as pure code motion.
5. **The intended driver parameter definitions changed after July 2.**
   Contradicted: `ALPHA_SCAN`, `N_BY_MACH`, and `Y_MAX_BY_MACH` are unchanged
   since June 18. The problem is which invocation path consumed them.

## Environment comparison limit

Current replay environment:

```text
Python 3.12.7
NumPy 2.2.6
SciPy 1.14.1
Windows 10.0.19045
```

The committed M45 curve/verdict contains no Python, NumPy, SciPy, BLAS/LAPACK,
platform, source-commit, command, or generation-time metadata. Therefore no
honest historical version-to-version environment comparison is possible.

## Bounded live-probe wall and late child completion

An optional scratch-only script ran one R=300 default invocation and one
corrected invocation in a single externally bounded command. The PowerShell
wrapper reached the wall:

```text
$env:PYTHONPATH=(Get-Location).Path; python -m py_compile diagnostics\reference_staleness\probe_m45_invocation.py; python diagnostics\reference_staleness\probe_m45_invocation.py
wrapper exit 124: command timed out after 124031 milliseconds
```

On Windows the timeout did not terminate the child Python process. Without a
retry or extension, that child later completed and wrote
`SCRATCH/m45_invocation_probe.json`. It records:

- default N110/y30: 73.978 s; exact match to the June-17 row and
  `4.651e-13` maximum absolute drift from the July-2 committed row;
- corrected N120/y40: 93.820 s; exact match to the June-18 corrected row.

The serialized total was 167.798 s, beyond the approximate two-minute combined
lock boundary. The measurement lock was not acquired because the two individual
solves were expected and observed below two minutes, but treating them as one
unlocked combined run was a process mistake. The artifact is valid numerical
diagnosis evidence; its wall times are **not contention-free measurement
evidence**. The wrapper exit remains 124, and the command was not split,
extended, or retried after that wall.

## Scope conclusion

The six-case census does not clear the other cited-not-rerun outputs. It shows
five full-precision anchors stable within `1e-9`, one Orszag aggregate metric
drifted by `3.235e-8`, and no byte-identical payloads. M45 is therefore not the
only place strict bytes move, but it is the only sampled case with material
percent-level growth drift, and it has a direct parameter-routing explanation.

No committed verification reference was modified.
