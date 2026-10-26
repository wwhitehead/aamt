#!/usr/bin/env python3
# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""Seeded uniform records for `aamt pack`, so the routing benchmark is reproducible.

    python3 native/examples/gen_random_tsv.py 1000000 > r1m.tsv
    native/build/aamt pack r1m.tsv r1m.aamt
    native/build/aamt bench r1m.aamt 1000000
"""
import random
import sys

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
rng = random.Random(seed)
out = sys.stdout
for i in range(n):
    t, e, r, a = (rng.random() for _ in range(4))
    out.write(f"{i}\t{t:.6f}\t{e:.6f}\t{r:.6f}\t{a:.6f}\tp{i}\n")
