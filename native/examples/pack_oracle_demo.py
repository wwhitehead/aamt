#!/usr/bin/env python3
# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""Pack the 256 Odu oracle cards into a demo .aamt substrate.

Each card's id "upper-lower" already names its two Meji indices — the two
tree levels WP-23 identifies as the Odu lattice. The TERA centroid of a card
is recovered by decoding, per axis, the (upper-bit, lower-bit) pair into one
of the four quartile centers the visualizer's OduCoverageTab already uses:
0->0.125, 1->0.375, 2->0.625, 3->0.875. This gives real, named, two-level
data to route against instead of synthetic noise — packing a card via its own
geometry and reading it back by walking the same tree that names it.

Usage: python3 pack_oracle_demo.py [out.tsv] [out.aamt]
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FOUNDATIONS = os.path.join(HERE, "..", "..", "aamt-foundations", "oracle-256-cards.json")
CLI = os.path.join(HERE, "..", "build", "aamt")

QUARTILE = [0.125, 0.375, 0.625, 0.875]


def card_tera(upper: int, lower: int) -> list[float]:
    tera = []
    for d in range(4):  # T, E, R, A bit d
        bu = (upper >> d) & 1
        bl = (lower >> d) & 1
        level = bu * 2 + bl
        tera.append(QUARTILE[level])
    return tera


def main() -> int:
    out_tsv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "oracle-demo.tsv")
    out_aamt = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "oracle-demo.aamt")

    data = json.load(open(FOUNDATIONS, encoding="utf-8"))
    cards = data["cards"]

    with open(out_tsv, "w", encoding="utf-8") as f:
        for i, card in enumerate(cards):
            upper = card["upper"]["id"]
            lower = card["lower"]["id"]
            tera = card_tera(upper, lower)
            payload = json.dumps({
                "id": card["id"],
                "upper": card["upper"]["aamt_native_name"],
                "lower": card["lower"]["aamt_native_name"],
                "title": card["title"],
                "tagline": card["tagline"],
            }, separators=(",", ":"))
            f.write(f"{i}\t{tera[0]:.6f}\t{tera[1]:.6f}\t{tera[2]:.6f}\t{tera[3]:.6f}\t{payload}\n")

    subprocess.run([CLI, "pack", out_tsv, out_aamt, "8", "4"], check=True)
    print(f"packed {len(cards)} Odu cards -> {out_aamt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
