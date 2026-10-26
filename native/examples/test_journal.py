#!/usr/bin/env python3
# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""Round-trip test for the lineage journal Python bindings (AamtJournal,
read_journal, verify_journal) — WP-23 section 5's "deterministic replay of
internal routing state" claim, exercised as a real callable operation
against real accumulated history, not just tests/roundtrip.cpp's synthetic
1000-query C++ batch.

  1. pack a substrate, log 50 hard-routed queries + 10 soft-routing rank-0
     picks through the same journal
  2. read the journal back; must recover all 60 entries exactly
  3. verify_journal against the SAME file: every entry must match
  4. repack the same path with different data; verify_journal must report
     every entry as a hash mismatch and zero false route mismatches

Run: python3 test_journal.py
"""
import os
import random
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bindings", "python"))
import aamt_ffi  # noqa: E402


def pack(path, seed, n=1000):
    with aamt_ffi.AamtBuilder(max_depth=8, bucket_capacity=16) as b:
        rng = random.Random(seed)
        for i in range(n):
            b.add(i, [rng.random() for _ in range(4)], f"mem-{i}".encode())
        b.write(path)


def main():
    tmp = tempfile.mkdtemp(prefix="aamt_journal_test_")
    aamt_path = os.path.join(tmp, "test.aamt")
    journal_path = os.path.join(tmp, "test.aamtj")

    pack(aamt_path, seed=3)
    rng = random.Random(9)
    queries = [[rng.random() for _ in range(4)] for _ in range(50)]

    with aamt_ffi.AamtSubstrate.open(aamt_path) as sub:
        with aamt_ffi.AamtJournal(journal_path) as j:
            for q in queries:
                j.log(sub, q, sub.query(q, capacity=4))
            for q in queries[:10]:
                branches = sub.query_soft(q, k=16, per_branch_capacity=1)
                j.log(sub, q, branches[0])  # rank-0 == hard route, by construction

    entries = aamt_ffi.read_journal(journal_path)
    assert len(entries) == 60, f"expected 60 entries, got {len(entries)}"

    result = aamt_ffi.verify_journal(aamt_path, journal_path)
    assert result.ok, result
    assert result.total == 60 and result.matched == 60
    print(f"fresh journal: {result.matched}/{result.total} matched, ok={result.ok}")

    # Repack the same path with different data -> every entry must show as a
    # hash mismatch, and NONE should be misreported as a route mismatch
    # (that would mean a stale-journal case looks like a router bug).
    pack(aamt_path, seed=999)
    stale = aamt_ffi.verify_journal(aamt_path, journal_path)
    assert stale.hash_mismatches == stale.total == 60
    assert stale.route_mismatches == []
    assert not stale.ok
    print(f"stale journal (repacked file): {stale.hash_mismatches}/{stale.total} "
          f"hash mismatches, 0 false route mismatches, ok={stale.ok}")

    print("PASS test_journal: log/read/verify round-trip, staleness correctly detected")


if __name__ == "__main__":
    main()
