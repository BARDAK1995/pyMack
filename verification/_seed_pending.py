"""Create 'pending' verdict.json stubs for any audit case that lacks one.

Idempotent: never overwrites an existing verdict.json (so a computed verdict
from a compare_*.py engine is preserved). This encodes the canonical case list
for the audit; running it then build_success_matrix.py shows the full intended
matrix with pending rows before the expensive runs finish.

    python verification/_seed_pending.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (category_subdir, case_id, source, conditions, quantity, pending_reason)
CASES = [
    ("neutralCurve_verification", "sean_m5p35",
     "Collaborator independent LST (M5.35 N2)",
     {"Ma": 5.35, "gas": "nitrogen", "wall": "adiabatic~370K", "psi_deg": 0,
      "formulation": "spatial second mode"},
     "dimensional neutral branches x_left/x_right vs frequency (mm)",
     "T0 reuse — filled by compare_sean_m5p35.py"),
    ("neutralCurve_verification", "ozgen_m2",
     "Özgen & Kırcalı (2008) Fig 3",
     {"Ma": 2.0, "gas": "air", "wall": "adiabatic", "psi_deg": 0,
      "formulation": "temporal 2D", "transport": "Özgen T-dependent"},
     "neutral curve (c_i=0) in (Re, alpha)",
     "T1 — filled by compare_ozgen_fig3.py"),
    ("neutralCurve_verification", "ozgen_m3",
     "Özgen & Kırcalı (2008) Fig 3",
     {"Ma": 3.0, "gas": "air", "wall": "adiabatic", "psi_deg": 0,
      "formulation": "temporal 2D", "transport": "Özgen T-dependent"},
     "neutral curve (c_i=0) in (Re, alpha)",
     "T1 — awaiting Özgen panels-2,3,4,6 compute, then compare_ozgen_fig3.py"),
    ("neutralCurve_verification", "ozgen_m4",
     "Özgen & Kırcalı (2008) Fig 3",
     {"Ma": 4.0, "gas": "air", "wall": "adiabatic", "psi_deg": 0,
      "formulation": "temporal 2D", "transport": "Özgen T-dependent"},
     "neutral curve (c_i=0) in (Re, alpha)",
     "T1 — filled by compare_ozgen_fig3.py"),
    ("neutralCurve_verification", "ozgen_m6",
     "Özgen & Kırcalı (2008) Fig 3",
     {"Ma": 6.0, "gas": "air", "wall": "adiabatic", "psi_deg": 0,
      "formulation": "temporal 2D", "transport": "Özgen T-dependent"},
     "neutral curve (c_i=0) in (Re, alpha)",
     "T1 — awaiting Özgen panels-2,3,4,6 compute, then compare_ozgen_fig3.py"),
    ("neutralCurve_verification", "mack_fig10_1_m1p6",
     "Mack (1984) Fig 10.1",
     {"Ma": 1.6, "gas": "air", "wall": "adiabatic", "psi_deg": 0,
      "formulation": "temporal neutral frequency", "transport": "Mack"},
     "neutral-stability frequency vs R",
     "T3 — needs a public temporal-neutral runner (not yet built)"),
    ("neutralCurve_verification", "mack_fig10_1_m2p2",
     "Mack (1984) Fig 10.1",
     {"Ma": 2.2, "gas": "air", "wall": "adiabatic", "psi_deg": 0,
      "formulation": "temporal neutral frequency", "transport": "Mack"},
     "neutral-stability frequency vs R",
     "T3 — needs a public temporal-neutral runner (not yet built)"),

    ("growthRate_verification", "mack_fig10_3_m1p3",
     "Mack (1984) Fig 10.3",
     {"Ma": 1.3, "gas": "air", "wall": "adiabatic", "psi_deg": 45,
      "formulation": "temporal 3D max growth", "transport": "Mack"},
     "max temporal ω_i vs R",
     "T0 reuse — filled by compare_mack_fig10_3.py"),
    ("growthRate_verification", "mack_fig10_3_m2p2",
     "Mack (1984) Fig 10.3",
     {"Ma": 2.2, "gas": "air", "wall": "adiabatic", "psi_deg": 45,
      "formulation": "temporal 3D max growth", "transport": "Mack"},
     "max temporal ω_i vs R",
     "T2 — needs Mack overlay generalized for (M, ψ)"),
    ("growthRate_verification", "mack_fig10_3_m3p0",
     "Mack (1984) Fig 10.3",
     {"Ma": 3.0, "gas": "air", "wall": "adiabatic", "psi_deg": 60,
      "formulation": "temporal 3D max growth", "transport": "Mack"},
     "max temporal ω_i vs R",
     "T2 — needs Mack overlay generalized for (M, ψ)"),
    ("growthRate_verification", "ozgen_fig3_lobes",
     "Özgen & Kırcalı (2008) Fig 3",
     {"Ma": "2-6", "gas": "air", "wall": "adiabatic", "psi_deg": 0,
      "formulation": "temporal 2D", "transport": "Özgen T-dependent"},
     "growth contours c_i=0.002/0.004/0.012 in (Re, alpha)",
     "T1 — from the Özgen panels-2,3,4,6 c_i grid"),
    ("growthRate_verification", "mack_fig10_4_family",
     "Mack (1984) Fig 10.4",
     {"Ma": "4.5/5.8/7/10", "gas": "air", "wall": "adiabatic", "psi_deg": None,
      "formulation": "temporal 3D max first-mode growth", "transport": "Mack"},
     "max first-mode ω_i vs R",
     "T3 — needs a high-Mach max-growth runner (not yet built)"),
    ("growthRate_verification", "mack_fig10_6_family",
     "Mack (1984) Fig 10.6 ⚠",
     {"Ma": "4.5/5.8/7/10", "gas": "air", "wall": "adiabatic", "psi_deg": 0,
      "formulation": "temporal 2D max second-mode growth", "transport": "Mack"},
     "max second-mode ω_i vs R",
     "T3 ⚠ — earlier probe showed ~6x magnitude gap under repo condition mapping; open issue"),
]


def main() -> int:
    n_created = 0
    for sub, case_id, source, cond, quantity, reason in CASES:
        folder = HERE / sub / case_id
        vf = folder / "verdict.json"
        if vf.exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        verdict = {
            "case_id": case_id,
            "category": "neutral_curve" if "neutral" in sub else "growth_rate",
            "source": source,
            "conditions": cond,
            "quantity": quantity,
            "metrics": {},
            "verdict": "pending",
            "verdict_reason": reason,
            "generated": "pending",
            "artifacts": {"pymack": None, "reference": None, "overlay": None},
            "pymack_provenance": "",
        }
        vf.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        n_created += 1
    print(f"seeded {n_created} pending stub(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
