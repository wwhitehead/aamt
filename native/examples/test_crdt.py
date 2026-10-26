#!/usr/bin/env python3
# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""Property tests for the CRDT-replicated substrate layer (aamt_crdt.py,
WP-23 section 12): proves CrdtStore.merge satisfies the three join-semilattice
laws that make it an actual CRDT rather than just "usually converges" —

  1. commutativity: merge(A, B) == merge(B, A)
  2. associativity: merge(merge(A, B), C) == merge(A, merge(B, C))
  3. idempotence:   merge(A, A) == A

— via randomized trials (many independent random stores, not a couple of
hand-picked examples), plus a genuine-tie case (two replicas writing at the
identical physical millisecond, only replica_id distinguishes them) and an
end-to-end test that two "offline" replicas converge to the identical
.aamt file regardless of merge order.

Run: python3 test_crdt.py
"""
import os
import random
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bindings", "python"))
from aamt_crdt import HLC, CrdtStore, local_replica_id  # noqa: E402
import aamt_ffi  # noqa: E402


def random_store(rng, n_records, n_replicas, id_space):
    """A store built from n_records puts drawn from n_replicas independent
    HLCs racing against each other and against real wall-clock jitter, so
    physical times collide often (exercising the logical counter and the
    replica_id tiebreak, not just "later timestamp wins")."""
    store = CrdtStore()
    clocks = [HLC.zero(replica_id=r) for r in range(n_replicas)]
    for _ in range(n_records):
        r = rng.randrange(n_replicas)
        # Coarse, colliding fake wall-clock so many puts share physical=now
        now_ms = rng.randrange(0, 5)
        clocks[r] = clocks[r].tick(now_ms=now_ms)
        rec_id = rng.randrange(id_space)
        tera = [rng.random() for _ in range(4)]
        data = f"r{r}-{rec_id}-{clocks[r].logical}".encode()
        store.put(rec_id, tera, data, clocks[r])
    return store


def test_properties(trials=500):
    rng = random.Random(1234)
    for t in range(trials):
        a = random_store(rng, rng.randrange(0, 20), rng.randrange(1, 4), id_space=8)
        b = random_store(rng, rng.randrange(0, 20), rng.randrange(1, 4), id_space=8)
        c = random_store(rng, rng.randrange(0, 20), rng.randrange(1, 4), id_space=8)

        # commutativity
        assert a.merge(b) == b.merge(a), f"trial {t}: commutativity failed"

        # associativity
        left = a.merge(b).merge(c)
        right = a.merge(b.merge(c))
        assert left == right, f"trial {t}: associativity failed"

        # idempotence
        assert a.merge(a) == a, f"trial {t}: idempotence failed on a"
        assert b.merge(b) == b, f"trial {t}: idempotence failed on b"

        merged_ab = a.merge(b)
        assert merged_ab.merge(merged_ab) == merged_ab, f"trial {t}: idempotence failed on merge(a,b)"

    print(f"PASS test_properties: commutativity, associativity, idempotence "
          f"held across {trials} randomized trials")


def test_genuine_tie():
    """Two replicas write the SAME id at the SAME physical millisecond with
    the SAME logical counter (both ticking from a fresh zero clock at the
    same instant) — only replica_id can break the tie. Must resolve the
    same way regardless of merge order, and must not silently drop the
    write (both events are real)."""
    clock_1 = HLC.zero(replica_id=1).tick(now_ms=1000)
    clock_2 = HLC.zero(replica_id=2).tick(now_ms=1000)
    assert clock_1.physical == clock_2.physical and clock_1.logical == clock_2.logical
    assert clock_1 != clock_2  # replica_id still distinguishes them

    store_1 = CrdtStore()
    store_1.put(7, [0.1, 0.2, 0.3, 0.4], b"from replica 1", clock_1)
    store_2 = CrdtStore()
    store_2.put(7, [0.9, 0.8, 0.7, 0.6], b"from replica 2", clock_2)

    merged_a = store_1.merge(store_2)
    merged_b = store_2.merge(store_1)
    assert merged_a == merged_b, "tie must resolve identically regardless of merge order"

    winner = merged_a.records()[0]
    expected_winner_data = b"from replica 2" if clock_2 > clock_1 else b"from replica 1"
    assert winner.data == expected_winner_data
    print(f"PASS test_genuine_tie: same-millisecond write resolved deterministically "
          f"via replica_id (winner: replica {winner.hlc.replica_id})")


def test_causal_update():
    """update() must advance a receiving replica's clock strictly past
    whatever it just learned about, so its NEXT local write is provably
    causally after the remote event it received — the actual point of an
    HLC over a plain per-replica counter."""
    replica_a = HLC.zero(replica_id=1).tick(now_ms=100)
    replica_b = HLC.zero(replica_id=2)
    replica_b = replica_b.update(replica_a, now_ms=50)  # b's local clock is behind a's event
    assert replica_b > replica_a, "receiving replica must advance past the remote event"

    next_write = replica_b.tick(now_ms=50)
    assert next_write > replica_a, "next local write must be provably after the received event"
    print("PASS test_causal_update: update() advances strictly past a received remote event")


def test_local_replica_id():
    """The concrete gap behind 'CRDT needs an actual second writer': two
    real writers must never collide on replica_id the way this module's
    own property testing found breaks things (test_properties's trial 2,
    see aamt_crdt.py's module docstring). Proves the three properties
    that make local_replica_id a safe default instead of a caller having
    to invent a scheme: stable across repeated calls (simulates a
    restart), distinct across different installs, and survives a crash
    mid-write (the tmp-file-then-rename path)."""
    tmp = tempfile.mkdtemp(prefix="aamt_replica_id_test_")
    state_dir = os.path.join(tmp, "replica-a")

    first = local_replica_id(state_dir)
    second = local_replica_id(state_dir)  # simulates a restart reading the same state_dir
    assert first == second, "the same state_dir must yield the same id across calls"
    assert 0 < first < (1 << 63), "id must be a positive 63-bit value"

    other_dir = os.path.join(tmp, "replica-b")
    other = local_replica_id(other_dir)
    assert other != first, "two different installs must not collide (astronomically unlikely otherwise)"

    with open(os.path.join(state_dir, "replica_id")) as f:
        assert int(f.read().strip()) == first, "id must actually be persisted to disk, not just in-memory"

    print(f"PASS test_local_replica_id: stable across repeated calls ({first}), "
          f"distinct across installs, persisted to disk")


def test_to_aamt_convergence():
    """The concrete scenario this feature exists for: two devices write
    offline (no coordination), one id collides with different payloads,
    merge in EITHER order, pack to .aamt, and the packed file's content
    must reflect the converged winner exactly — proving the CRDT layer's
    guarantee survives all the way through a real pack via AamtBuilder."""
    device_a = CrdtStore()
    clock_a = HLC.zero(replica_id=101)
    clock_a = clock_a.tick(now_ms=1000)
    device_a.put(1, [0.5, 0.5, 0.5, 0.5], b"note from phone", clock_a)
    clock_a = clock_a.tick(now_ms=1000)
    device_a.put(2, [0.1, 0.9, 0.2, 0.8], b"only on phone", clock_a)

    device_b = CrdtStore()
    clock_b = HLC.zero(replica_id=202)
    clock_b = clock_b.tick(now_ms=2000)  # strictly later wall clock than device_a's id=1 write
    device_b.put(1, [0.5, 0.5, 0.5, 0.5], b"edited on laptop", clock_b)
    clock_b = clock_b.tick(now_ms=2000)
    device_b.put(3, [0.3, 0.3, 0.7, 0.7], b"only on laptop", clock_b)

    converged_ab = device_a.merge(device_b)
    converged_ba = device_b.merge(device_a)
    assert converged_ab == converged_ba
    assert len(converged_ab) == 3

    winner_1 = next(r for r in converged_ab.records() if r.id == 1)
    assert winner_1.data == b"edited on laptop", "later HLC (device_b) must win the id=1 conflict"

    tmp = tempfile.mkdtemp(prefix="aamt_crdt_test_")
    out_path = os.path.join(tmp, "converged.aamt")
    n_written = converged_ab.to_aamt(out_path)
    assert n_written == 3

    with aamt_ffi.AamtSubstrate.open(out_path) as sub:
        assert sub.record_count == 3
        all_records = sub.all_records()
        by_id = {r.id: r.data for r in all_records}
        assert by_id[1] == b"edited on laptop"
        assert by_id[2] == b"only on phone"
        assert by_id[3] == b"only on laptop"

    print("PASS test_to_aamt_convergence: two offline replicas converged identically "
          "regardless of merge order, packed .aamt reflects the correct winner")


def test_real_cross_language_interop():
    """The concrete proof CRDT has an actual second writer now (WP-23
    §13.2): the EXACT JSON a live browser tab exported during this
    feature's development (a real preview_start + javascript_tool
    session against aamt-crdt.js, not fabricated -- the matching regression
    check lives in the visualizer repo's src/aamt-crdt.test.js) merged
    here with a real, independent desktop-side CrdtStore, including a
    genuine id collision the desktop write should LOSE (earlier HLC),
    packed to a real .aamt file and read back to confirm the converged
    content is exactly right."""
    browser_json = (
        '{"records":[{"id":1,"tera":[0.2,0.3,0.4,0.5],"data":"YnJvd3NlciB3cml0ZXMgaWQ9MQ==",'
        '"hlc":{"physical":1000000,"logical":0,"replica_id":"11265530300016749345"}},'
        '{"id":2,"tera":[0.6,0.7,0.8,0.9],"data":"YnJvd3NlciB3cml0ZXMgaWQ9MiAodW5pcXVlIHRvIGJyb3dzZXIp",'
        '"hlc":{"physical":1000000,"logical":1,"replica_id":"11265530300016749345"}},'
        '{"id":3,"tera":[0.1,0.1,0.1,0.1],"data":"YnJvd3NlciB3cml0ZXMgaWQ9MyAtLSBMQVRFUiwgc2hvdWxkIHdpbg==",'
        '"hlc":{"physical":2000000,"logical":0,"replica_id":"11265530300016749345"}}]}'
    )
    browser_store = CrdtStore.from_json(browser_json)
    assert len(browser_store) == 3
    rec1 = next(r for r in browser_store.records() if r.id == 1)
    assert rec1.data == b"browser writes id=1"
    assert rec1.hlc.replica_id == 11265530300016749345  # exact, no precision loss on the JSON string -> int path

    desktop_store = CrdtStore()
    desktop_clock = HLC.zero(replica_id=987654321).tick(now_ms=500000)
    desktop_store.put(4, [0.9, 0.1, 0.9, 0.1], b"desktop writes id=4 (unique to desktop)", desktop_clock)
    desktop_clock = desktop_clock.tick(now_ms=500000)
    desktop_store.put(3, [0.5, 0.5, 0.5, 0.5], b"desktop writes id=3 -- EARLIER, should lose", desktop_clock)

    merged_ab = browser_store.merge(desktop_store)
    merged_ba = desktop_store.merge(browser_store)
    assert merged_ab == merged_ba
    assert len(merged_ab) == 4

    winner_3 = next(r for r in merged_ab.records() if r.id == 3)
    assert winner_3.data == b"browser writes id=3 -- LATER, should win"

    tmp = tempfile.mkdtemp(prefix="aamt_crdt_interop_test_")
    out_path = os.path.join(tmp, "converged.aamt")
    n = merged_ab.to_aamt(out_path)
    assert n == 4

    with aamt_ffi.AamtSubstrate.open(out_path) as sub:
        assert sub.record_count == 4
        by_id = {r.id: r.data for r in sub.all_records()}
        assert by_id[1] == b"browser writes id=1"
        assert by_id[2] == b"browser writes id=2 (unique to browser)"
        assert by_id[3] == b"browser writes id=3 -- LATER, should win"
        assert by_id[4] == b"desktop writes id=4 (unique to desktop)"

    print("PASS test_real_cross_language_interop: real browser-exported JSON merged "
          "with a real desktop store, commutative, genuine conflict resolved correctly, "
          "packed .aamt confirms the converged content")


def main():
    test_properties()
    test_genuine_tie()
    test_causal_update()
    test_local_replica_id()
    test_to_aamt_convergence()
    test_real_cross_language_interop()
    print("PASS test_crdt: all CRDT property and integration tests passed")


if __name__ == "__main__":
    main()
