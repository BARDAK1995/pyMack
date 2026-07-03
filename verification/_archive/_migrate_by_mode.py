"""One-shot migration: reorganize verification case folders by physical MODE.

Before: verification/{neutralCurve,growthRate,eigenvalueAnchor}_verification/<case>/
After:  verification/{second_mode,first_mode,other}/<case>/   (flat; the quantity
        stays in each verdict's `category` field).

Mode is the meaningful axis: pyMack's SECOND (Mack) mode is validated across
sources/Mach; its FIRST mode is the documented weak spot. `other` holds the
incompressible / unrecoverable-condition Malik temporal cases (unmeasured).

Adds a "mode" field to each verdict.json and rewrites in-verdict artifact paths.
Idempotent-ish: skips a case already located under its target mode dir.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V = REPO / "verification"

# case_id -> (old_category_dir, target_mode)
MAP = {
    # --- neutralCurve_verification ---
    "mack_fig10_1_m1p6": ("neutralCurve_verification", "first_mode"),
    "mack_fig10_1_m2p2": ("neutralCurve_verification", "first_mode"),
    "mazhong2003_m4p5": ("neutralCurve_verification", "second_mode"),
    "ozgen_m2": ("neutralCurve_verification", "first_mode"),
    "ozgen_m3": ("neutralCurve_verification", "first_mode"),
    "ozgen_m4": ("neutralCurve_verification", "first_mode"),
    "ozgen_m6": ("neutralCurve_verification", "first_mode"),
    "sean_m5p35": ("neutralCurve_verification", "second_mode"),
    # --- growthRate_verification ---
    "cone_sivasubramanian_fasel_2015": ("growthRate_verification", "second_mode"),
    "egorov2006_m6": ("growthRate_verification", "second_mode"),
    "mack_fig10_3_m1p3": ("growthRate_verification", "first_mode"),
    "mack_fig10_3_m2p2": ("growthRate_verification", "first_mode"),
    "mack_fig10_3_m3p0": ("growthRate_verification", "first_mode"),
    "mack_fig10_4_family": ("growthRate_verification", "first_mode"),
    "mack_fig10_4_M100": ("growthRate_verification", "first_mode"),
    "mack_fig10_4_M45": ("growthRate_verification", "first_mode"),
    "mack_fig10_4_M58": ("growthRate_verification", "first_mode"),
    "mack_fig10_4_M70": ("growthRate_verification", "first_mode"),
    "mack_fig10_6_family": ("growthRate_verification", "second_mode"),
    "mack_fig10_6_M100": ("growthRate_verification", "second_mode"),
    "mack_fig10_6_M45": ("growthRate_verification", "second_mode"),
    "mack_fig10_6_M58": ("growthRate_verification", "second_mode"),
    "mack_fig10_6_M70": ("growthRate_verification", "second_mode"),
    "ozgen_fig3_lobes": ("growthRate_verification", "first_mode"),
    # --- eigenvalueAnchor_verification ---
    "balakumar_malik1992_branches": ("eigenvalueAnchor_verification", "second_mode"),
    "balakumar_malik1992_via_xirenfu": ("eigenvalueAnchor_verification", "second_mode"),
    "malik_case1": ("eigenvalueAnchor_verification", "other"),
    "malik_case3": ("eigenvalueAnchor_verification", "second_mode"),
    "malik_case4": ("eigenvalueAnchor_verification", "second_mode"),
    "malik_case5": ("eigenvalueAnchor_verification", "second_mode"),
    "malik_case6": ("eigenvalueAnchor_verification", "second_mode"),
}


def git_mv(src: Path, dst: Path):
    subprocess.run(["git", "mv", str(src.relative_to(REPO)).replace("\\", "/"),
                    str(dst.relative_to(REPO)).replace("\\", "/")],
                   cwd=REPO, check=True)


def main():
    moved = 0
    for case, (oldcat, mode) in MAP.items():
        src = V / oldcat / case
        dst = V / mode / case
        if dst.exists():
            print(f"  skip {case} (already at {mode})")
            continue
        if not src.exists():
            print(f"  WARN {case}: source {src} missing")
            continue
        (V / mode).mkdir(parents=True, exist_ok=True)
        git_mv(src, dst)
        # update verdict.json: add mode + rewrite in-verdict artifact paths
        vf = dst / "verdict.json"
        v = json.loads(vf.read_text(encoding="utf-8"))
        v["mode"] = {"second_mode": "second", "first_mode": "first",
                     "other": "other"}[mode]
        oldp = f"verification/{oldcat}/{case}/"
        newp = f"verification/{mode}/{case}/"
        arts = v.get("artifacts", {})
        for k, val in list(arts.items()):
            if isinstance(val, str) and oldp in val:
                arts[k] = val.replace(oldp, newp)
        vf.write_text(json.dumps(v, indent=2), encoding="utf-8")
        print(f"  {case}: {oldcat} -> {mode}")
        moved += 1
    print(f"moved {moved} case(s)")


if __name__ == "__main__":
    main()
