# CPU benchmark evidence

These are the original committed measurement artifacts supporting the CPU
performance statements in `paper/paper.md`. The JSON files were copied into
this public layout byte-for-byte; their recorded commands, source commits,
environment details, and machine-local paths were not rewritten.

## Paper claim map

| Paper statement | Committed authority |
|---|---|
| 720-node deployed serial wall: 1598.6827 s | `verification/mixed_mode/ozgen_fig3/_compute/ozgen_M2.json` |
| 720-node tuned values-only wall: 46.7037 s; 720/720 classification identity | `cpu_floor_sweep.json` |
| 34.23-fold reduction | Ratio of the two authorities above |
| Mach 10 serial wall: 9717.0784 s | `cpu_fig10_4_m10_serial.json` |
| Mach 10 point-parallel wall: 1110.5274 s; same nine-station verdict | `cpu_fig10_4_m10_pointparallel_61w.json` |
| 2880-node full-QZ map: 399.8026 s | `ozgen_4x_full_qz.json` |
| 2880-node values-only map: 169.4716 s | `ozgen_4x_eigenvalues_only.json` |
| 24 newly added nodes independently re-solved bitwise-identically | `ozgen_4x_deployed_path_spotcheck.json` and `ozgen_4x_demo.json` |
| Figure 1 provenance | `ozgen_m2_4x_ci_map.provenance.json` |

`cpu_point_budget.json` records the measured per-point cost decomposition used
to interpret the CPU floor. It is retained as supporting evidence even though
the paper does not quote its internal timings directly.

## Integrity hashes

| File | SHA-256 |
|---|---|
| `cpu_fig10_4_m10_pointparallel_61w.json` | `48a0b501d02b9c5cc5b569a758d2f38d429764b0447b76b309d152433e9ff8ff` |
| `cpu_fig10_4_m10_serial.json` | `8d5031ad92dfaac0f90182a4ecf28f37e3388461851c690e5d5a793c7e4f5680` |
| `cpu_floor_sweep.json` | `6b29ee0dda84218aaac717fd41d2281be31247f3c77f636afc117739aaed77ea` |
| `cpu_point_budget.json` | `90767b17d85b7be3e4c63c6d0e486359bd2ddfb0bbde6aa8d7c5b7e10fe7675a` |
| `ozgen_4x_demo.json` | `ce967494b514692c5953d23ac8a6d56de93183e9d2edde18370714aa387d8f8d` |
| `ozgen_4x_deployed_path_spotcheck.json` | `556fbd89f39df78c1b3fc53fa336f1a20d055797a586887bea7bd50dce71b23c` |
| `ozgen_4x_eigenvalues_only.json` | `5a8586a1b99e6a21468123293067988bb0a0ea2d00853f699dd8a996b131a2f6` |
| `ozgen_4x_full_qz.json` | `21c1ad4e964487beaca342dd862a18bc4a0aca8ae83e1f67b20231fc9027710d` |
| `ozgen_m2_4x_ci_map.provenance.json` | `21bc8df9dd84de4585849447073bac643bacc045c450565f136c2554b9a2da2e` |
