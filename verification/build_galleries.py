"""Tile per-case overlay.png into at-a-glance gallery montages, one per mode.

    python verification/build_galleries.py

Writes verification/second_mode_gallery.png and first_mode_gallery.png. Each
panel is one case's pyMack-vs-reference overlay, captioned by case_id and
colored by verdict (green=agrees, amber=acceptable, red=disagrees). Lets a
reader see the whole second-mode-agrees / first-mode-disagrees story at once.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = Path(__file__).resolve().parent
VCOLOR = {"agrees": "#2ca02c", "acceptable": "#d9920a",
          "disagrees": "#d62728", "pending": "#888888"}


def main() -> int:
    for mode, title in [("second_mode", "Second (Mack) mode — pyMack's design target"),
                        ("first_mode", "First mode — documented weak spot")]:
        cases = []
        for cd in sorted((HERE / mode).glob("*")):
            ov, vf = cd / "overlay.png", cd / "verdict.json"
            if ov.exists() and vf.exists():
                v = json.loads(vf.read_text(encoding="utf-8"))
                cases.append((cd.name, ov, v.get("verdict", "?")))
        if not cases:
            continue
        n = len(cases)
        ncol = 3
        nrow = math.ceil(n / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 5.4, nrow * 4.1))
        axes = axes.ravel()
        for ax, (name, ov, verd) in zip(axes, cases):
            ax.imshow(mpimg.imread(str(ov)))
            ax.set_title(f"{name}  [{verd}]", fontsize=12,
                         color=VCOLOR.get(verd, "k"), fontweight="bold")
            ax.axis("off")
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(f"pyMack verification — {title}  ({n} cases)", fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        out = HERE / f"{mode}_gallery.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.name}: {n} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
