#!/usr/bin/env python3
# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""Closes the loop between two demos that, until now, only lived side by
side: the CRDT layer's "genuine second writer" story (aamt_crdt.py,
examples/test_crdt.py) and the substrate's "genuine fault tolerance" story
(mirror pairing + RS(15,13)/GF(16) ECC, tests/roundtrip.cpp). CrdtStore has
only ever lived as an in-memory object with a JSON interchange format —
this proves its RESOLVED, merged output actually survives a real pack into
a real mirror+RS-protected .aamt file on disk, and that the protection
genuinely works on CRDT-sourced data specifically (not just on synthetic
loop-generated records the way roundtrip.cpp's own sections 5-7 test it).

Three things are proven, each with real file I/O — no mocks, no simulated
corruption:

  1. Two independent CrdtStore replicas (real, distinct, persisted
     replica_ids via local_replica_id) write concurrently, one id
     collision resolves deterministically regardless of merge order, and
     the merged/converged store is packed via CrdtStore.to_aamt(...,
     mirrored_ecc=True) -- reusing AamtBuilder.add_mirrored_ecc, the same
     packer pack_oracle_demo.py and tests/roundtrip.cpp section 7 use, not
     a reimplementation.

  2. Every converged CRDT record round-trips out of the packed file via
     AamtSubstrate.read_verified (the composed RS+mirror read path, WP-23
     §13) with exact id, tera, and payload bytes, zero RS corrections.

  3. A real corrupt-and-recover test: AamtSubstrate.record_file_offset
     locates two CRDT records' actual on-disk byte offsets, a second raw
     file handle XORs real bytes there (same methodology as
     tests/roundtrip.cpp sections 5-7), and:
       a. a single corrupted nibble in one record's RS-protected payload
          is caught by the byte-level chirality/consistency gate AND
          transparently repaired by read_verified via RS alone
          (status OK_RS_REPAIRED) -- the mirror is never touched;
       b. a scrambled id field in a second, different record (a real
          structural failure -- the primary becomes unfindable by id at
          its own leaf) is ALSO caught by the chirality gate, and
          read_verified recovers the exact original payload and tera by
          falling through to the mirror (status OK_MIRROR_RECOVERED).

Run: python3 test_crdt_mirrored_ecc.py
"""
import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bindings", "python"))
from aamt_crdt import HLC, CrdtStore, local_replica_id  # noqa: E402
import aamt_ffi  # noqa: E402


def build_converged_store(tmp: str) -> tuple[CrdtStore, dict[int, tuple]]:
    """Two real, independent replicas -- persisted replica_ids under two
    different state dirs, exactly as a real deployment would provision
    them (local_replica_id's own contract). device_a writes ids 0-19 at
    an EARLIER wall-clock; device_b writes ids 10-29 at a LATER one, so
    ids 10-19 are a genuine collision device_b's write must win. Returns
    the converged store plus an independent id -> (tera, data) reference
    dict computed WITHOUT going through the store, so the round-trip
    assertions below aren't just comparing the store against itself."""
    replica_a_dir = os.path.join(tmp, "replica-a")
    replica_b_dir = os.path.join(tmp, "replica-b")
    replica_a_id = local_replica_id(replica_a_dir)
    replica_b_id = local_replica_id(replica_b_dir)
    assert replica_a_id != replica_b_id

    expect: dict[int, tuple] = {}

    device_a = CrdtStore()
    clock_a = HLC.zero(replica_id=replica_a_id).tick(now_ms=1_000_000)
    for i in range(20):
        tera = (0.05 * i % 1.0, 0.11 * i % 1.0, 0.23 * i % 1.0, 0.37 * i % 1.0)
        data = f"device-a-record-{i}".encode()
        clock_a = clock_a.tick(now_ms=1_000_000)
        device_a.put(i, tera, data, clock_a)
        expect[i] = (tera, data)

    device_b = CrdtStore()
    clock_b = HLC.zero(replica_id=replica_b_id).tick(now_ms=2_000_000)  # strictly later
    for i in range(10, 30):
        tera = (0.61 * i % 1.0, 0.47 * i % 1.0, 0.83 * i % 1.0, 0.29 * i % 1.0)
        data = f"device-b-record-{i}".encode()
        clock_b = clock_b.tick(now_ms=2_000_000)
        device_b.put(i, tera, data, clock_b)
        expect[i] = (tera, data)  # device_b's later write wins ids 10-19 too

    converged_ab = device_a.merge(device_b)
    converged_ba = device_b.merge(device_a)
    assert converged_ab == converged_ba, "merge must be order-independent"
    assert len(converged_ab) == 30

    # Confirm the collision actually resolved to device_b, not just that
    # SOME value is present -- this is the thing that would silently mask
    # a broken merge if it weren't checked explicitly.
    winner_15 = next(r for r in converged_ab.records() if r.id == 15)
    assert winner_15.data == b"device-b-record-15"

    return converged_ab, expect


def f32(tera) -> tuple:
    """Round-trips through float32 exactly as the C ABI stores tera, so
    comparisons against on-disk values are exact, not approximate."""
    import ctypes
    return tuple(ctypes.c_float(v).value for v in tera)


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="aamt_crdt_mirrored_ecc_")
    out_path = os.path.join(tmp, "crdt-converged-mirrored-ecc.aamt")

    converged, expect = build_converged_store(tmp)

    # 1. Pack through the real pipeline: CrdtStore.to_aamt(mirrored_ecc=True)
    #    -> AamtBuilder.add_mirrored_ecc for every record, the same call
    #    tests/roundtrip.cpp section 7 and (mirror-only) pack_oracle_demo.py
    #    reuse -- no reimplemented RS/mirror math in this test or in to_aamt.
    n_written = converged.to_aamt(out_path, mirrored_ecc=True)
    assert n_written == 30
    assert os.path.exists(out_path)
    file_size = os.path.getsize(out_path)
    print(f"PASS pack: {n_written} converged CRDT records packed mirror+RS-protected "
          f"-> {out_path} ({file_size} bytes)")

    # 2. Every record round-trips exactly via the composed read path.
    with aamt_ffi.AamtSubstrate.open(out_path) as sub:
        assert sub.record_count == 60, "30 primaries + 30 mirrors"
        for rec_id, (tera, data) in expect.items():
            result = sub.read_verified(rec_id, tera)
            assert result.status == aamt_ffi.ReadStatus.OK, (rec_id, result.status)
            assert result.corrected_symbols == 0
            assert result.data == data, (rec_id, result.data, data)
            assert result.tera == f32(tera), (rec_id, result.tera, f32(tera))
        print(f"PASS round-trip: all {len(expect)} CRDT-converged records read back via "
              f"read_verified with exact id, tera, and payload bytes (status OK, 0 corrections)")

        # Baseline: clean substrate is 100% chirality-consistent before any
        # corruption -- the control this test's corruption cases are
        # contrasted against.
        clean_report = sub.verify_chirality()
        assert clean_report.primary_count == 30
        assert clean_report.mirror_count == 30
        assert clean_report.consistent == 30
        assert clean_report.inconsistent_count == 0
        print("PASS clean baseline: verify_chirality reports 30/30 primaries consistent "
              "with their mirrors, 0 inconsistent")

        # Locate two DIFFERENT CRDT records' real on-disk offsets before
        # touching any bytes (structure doesn't move; only content will).
        rs_target_id = 3       # device_a's write, untouched by the merge collision
        mirror_target_id = 25  # device_b's write
        rs_tera, rs_data = expect[rs_target_id]
        mirror_tera, mirror_data = expect[mirror_target_id]
        rs_offset = sub.record_file_offset(rs_target_id, rs_tera)
        mirror_offset = sub.record_file_offset(mirror_target_id, mirror_tera)

    # 3a. Real byte corruption #1: flip one nibble inside the RS-protected
    #     encoded payload (byte 10, high nibble -- same spot
    #     tests/roundtrip.cpp sections 6/7b corrupt: past the 4-byte size
    #     prefix, inside the first RS block).
    with open(out_path, "r+b") as raw:
        raw.seek(rs_offset + 32 + 10)
        original_byte = raw.read(1)
        raw.seek(rs_offset + 32 + 10)
        raw.write(bytes([original_byte[0] ^ 0xF0]))
    print(f"corrupted 1 nibble at file offset {rs_offset + 32 + 10} "
          f"(record id={rs_target_id}'s RS-encoded payload)")

    # 3b. Real byte corruption #2, a DIFFERENT record: scramble the
    #     primary's 8-byte external_id field -- a structural failure, not
    #     a payload one (findInLeaf can no longer find this record by id
    #     at its own leaf at all).
    with open(out_path, "r+b") as raw:
        raw.seek(mirror_offset)
        raw.write(struct.pack("<Q", 0xFFFFFFFFFFFFFFFF))
    print(f"corrupted 8-byte id field at file offset {mirror_offset} "
          f"(record id={mirror_target_id}'s external_id)")

    # Reopen fresh (matches tests/roundtrip.cpp's own methodology: never
    # trust a handle opened before the raw edit) and check both claims.
    with aamt_ffi.AamtSubstrate.open(out_path) as sub:
        # The chirality/consistency gate genuinely catches BOTH corruptions
        # -- it does a raw byte-for-byte comparison against the mirror, so
        # it doesn't care that RS could still repair corruption #1 on its
        # own; a byte-different primary IS an inconsistency, full stop.
        dirty_report = sub.verify_chirality(id_capacity=64)
        assert dirty_report.primary_count == 30
        assert dirty_report.inconsistent_count == 2, dirty_report
        assert dirty_report.consistent == 28
        assert rs_target_id in dirty_report.inconsistent_ids, dirty_report.inconsistent_ids
        assert 0xFFFFFFFFFFFFFFFF in dirty_report.inconsistent_ids, dirty_report.inconsistent_ids
        print(f"PASS chirality catches both corruptions: {dirty_report.inconsistent_count} "
              f"inconsistent (was 0), including record id={rs_target_id} (payload corruption) "
              f"and the scrambled id (structural corruption) -- 28/30 still consistent")

        # read_verified on the RS-corrupted record: repaired via RS ALONE,
        # exact original bytes, mirror never needed for this one.
        rs_result = sub.read_verified(rs_target_id, rs_tera)
        assert rs_result.status == aamt_ffi.ReadStatus.OK_RS_REPAIRED, rs_result.status
        assert rs_result.corrected_symbols == 1
        assert rs_result.data == rs_data, (rs_result.data, rs_data)
        print(f"PASS RS self-repair: record id={rs_target_id} read back exact original "
              f"bytes via read_verified (status OK_RS_REPAIRED, 1 symbol corrected, "
              f"mirror not needed)")

        # read_verified on the id-scrambled record, using the id/tera the
        # caller always knew (independent of the now-garbled on-disk
        # bytes) -- primary is unfindable by id, so this MUST recover via
        # the mirror: real mirror-based fault tolerance on CRDT-sourced
        # data, not a synthetic loop record.
        mirror_result = sub.read_verified(mirror_target_id, mirror_tera)
        assert mirror_result.status == aamt_ffi.ReadStatus.OK_MIRROR_RECOVERED, mirror_result.status
        assert mirror_result.data == mirror_data, (mirror_result.data, mirror_data)
        assert mirror_result.tera == f32(mirror_tera), (mirror_result.tera, f32(mirror_tera))
        print(f"PASS mirror recovery: record id={mirror_target_id} (id field scrambled to "
              f"an unfindable value) read back exact original bytes AND exact original tera "
              f"via read_verified's mirror fallback (status OK_MIRROR_RECOVERED)")

        # Every OTHER record (28 of the 30) is still clean and unaffected
        # by either corruption -- the damage stayed localized, exactly as
        # WP-23 section 10's mirror-pairing rationale claims.
        untouched = 0
        for rec_id, (tera, data) in expect.items():
            if rec_id in (rs_target_id, mirror_target_id):
                continue
            result = sub.read_verified(rec_id, tera)
            assert result.status == aamt_ffi.ReadStatus.OK
            assert result.corrected_symbols == 0
            assert result.data == data
            untouched += 1
        print(f"PASS locality: the other {untouched} CRDT records are unaffected by either "
              f"corruption (status OK, 0 corrections, exact bytes)")

    print("PASS test_crdt_mirrored_ecc: CRDT-merged records packed mirror+RS-protected, "
          "round-tripped exactly, and genuinely recovered from two independent real "
          "on-disk byte corruptions via the composed RS+mirror read path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
