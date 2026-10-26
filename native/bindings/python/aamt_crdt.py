# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""CRDT-replicated substrate (WP-23): instant, coordination-free writing
across multiple independent replicas, converging deterministically to one
.aamt substrate.

The .aamt FILE stays exactly what WP-23 already made it: immutable once
packed. This module doesn't touch that — it governs the IN-MEMORY
convergence process that happens BEFORE a pack, so multiple devices (or
multiple offline sessions) can each write freely, merge without
coordination, and produce the identical converged state regardless of
merge order. Once converged, CrdtStore.to_aamt() is a normal, deterministic
snapshot through the existing AamtBuilder.

The mechanism is a Hybrid Logical Clock (Kulkarni et al., "Logical Physical
Clocks and Consistent Snapshots in Globally Distributed Databases" — the
same construction CockroachDB and MongoDB use) plus a Last-Writer-Wins map:
each id's winner is whichever write carries the larger (hlc, data, tera)
key, a per-key `max` over a total order. A map of independent per-key
`max`-resolved registers is a textbook CRDT (an LWW-Map) — commutative,
associative, and idempotent by construction, not by hope. See
examples/test_crdt.py for property-based proof of exactly those three laws
across randomized trials, not just "it seemed to work in an example" — that
testing is what caught the (hlc, data, tera) key needing content in it at
all: HLC alone looked sufficient until a same-HLC, different-content
collision in the trials broke commutativity, forcing the fix below.

    from aamt_crdt import HLC, CrdtStore, local_replica_id

    # local_replica_id persists a random 63-bit id under state_dir, so
    # every call on THIS install returns the same one — call once per
    # install, not once per literal integer you make up.
    clock_a = HLC.zero(replica_id=local_replica_id("/path/to/replica-a/state"))
    store_a = CrdtStore()
    clock_a = clock_a.tick()
    store_a.put(42, [0.5, 0.5, 0.5, 0.5], b"from replica a", clock_a)

    clock_b = HLC.zero(replica_id=local_replica_id("/path/to/replica-b/state"))
    store_b = CrdtStore()
    clock_b = clock_b.tick()
    store_b.put(42, [0.5, 0.5, 0.5, 0.5], b"from replica b", clock_b)

    converged = store_a.merge(store_b)  # == store_b.merge(store_a), always
    converged.to_aamt("converged.aamt")
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import time
from dataclasses import dataclass


def local_replica_id(state_dir: str) -> int:
    """The concrete engineering gap behind "CRDT needs an actual second
    writer": nothing stops a second writer from existing, but nothing
    previously gave it a safe identity either. Two real writers each
    calling HLC.zero(replica_id=1) independently is exactly the collision
    this module's own property testing found breaks the assumption behind
    LWW's tiebreak (WP-23 §12.2 documents the fix for when it happens, but
    the fix is a safety net, not a reason to court it) — the actual
    correctness need is that distinct writers get distinct ids, without
    requiring a central coordinator (a CRDT's whole point). This provides
    one: a random 63-bit id, generated once and persisted under
    `state_dir/replica_id`, so every later call in the same state_dir —
    including across process restarts — returns the SAME id. Callers
    should call this once per install and reuse the result, the same way
    a machine has one hostname, not a fresh one per process.

    Honest limit, stated rather than hidden: this makes replica IDENTITY
    stable across restarts. It does NOT persist the HLC's logical
    counter (see HLC's own docstring) — a restarted replica still relies
    on physical time having actually advanced since its last write to
    stay ahead of itself, which is true for any real restart but is a
    real, separate limitation from what this function solves."""
    path = os.path.join(state_dir, "replica_id")
    if os.path.exists(path):
        with open(path) as f:
            return int(f.read().strip())
    os.makedirs(state_dir, exist_ok=True)
    # 63 bits fits a signed 64-bit integer cleanly across every language
    # binding this substrate has (Python int, C++ uint64_t, C# long); at
    # this scale the birthday-bound collision probability across even
    # 10^9 independently-provisioned replicas is astronomically small.
    new_id = secrets.randbits(63) or 1  # 0 excluded: keep it a non-sentinel value
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        f.write(str(new_id))
    os.replace(tmp_path, path)  # atomic on POSIX and Windows: no torn reads on crash
    return new_id


@dataclass(frozen=True, order=True)
class HLC:
    """A Hybrid Logical Clock timestamp: (physical_ms, logical, replica_id),
    compared lexicographically in that order. physical+logical give a total
    causal order across events on the SAME replica and correctly bound real
    clock skew across replicas (the HLC guarantee: HLC time never falls
    further behind wall-clock time than the max clock skew observed);
    replica_id is a final, deterministic tiebreak for two events that are
    otherwise indistinguishable (same physical ms, same logical counter —
    only possible across DIFFERENT replicas, since a single replica's own
    tick()/update() calls always strictly increase its own (physical,
    logical) pair)."""
    physical: int
    logical: int
    replica_id: int

    @staticmethod
    def zero(replica_id: int) -> "HLC":
        return HLC(0, 0, replica_id)

    def tick(self, now_ms: int | None = None) -> "HLC":
        """A new local event on this replica."""
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        if now > self.physical:
            return HLC(now, 0, self.replica_id)
        return HLC(self.physical, self.logical + 1, self.replica_id)

    def update(self, remote: "HLC", now_ms: int | None = None) -> "HLC":
        """Received an event stamped `remote` from another replica; advance
        this replica's own clock to remain causally consistent with it."""
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        new_physical = max(now, self.physical, remote.physical)
        if new_physical == self.physical and new_physical == remote.physical:
            new_logical = max(self.logical, remote.logical) + 1
        elif new_physical == self.physical:
            new_logical = self.logical + 1
        elif new_physical == remote.physical:
            new_logical = remote.logical + 1
        else:
            new_logical = 0
        return HLC(new_physical, new_logical, self.replica_id)


@dataclass(frozen=True)
class CrdtRecord:
    id: int
    tera: tuple
    data: bytes
    hlc: HLC


class CrdtStore:
    """A Last-Writer-Wins map: id -> CrdtRecord, resolved per-key by HLC.
    `merge` is commutative, associative, and idempotent — see
    test_aamt_crdt.py for exhaustive property tests, not just examples."""

    def __init__(self):
        self._records: dict[int, CrdtRecord] = {}

    def put(self, id_: int, tera, data: bytes, hlc: HLC) -> None:
        """Local or replayed-remote write. Resolution key is (hlc, data,
        tera), compared as a tuple — HLC decides almost every real case,
        but is NOT by itself a safe tiebreak: two structurally different
        records can carry an identical HLC (colliding replica_ids, a reset
        logical counter, or — measured directly during property testing of
        this module — merely a coincidence where two records happen to
        differ only in `tera`). Breaking a same-HLC tie by "whichever was
        already in the map" is order-dependent (merge(A,B) then keeps A's
        version, merge(B,A) keeps B's) and silently breaks commutativity.
        Comparing the full content makes the winner a pure function of the
        two candidate records, independent of merge order — so an
        HLC-equal write is a true no-op only when it's byte-for-byte the
        same event; anything else still resolves deterministically instead
        of nondeterministically favoring whichever side merge() started
        from."""
        tera = tuple(tera)
        data = bytes(data)
        existing = self._records.get(id_)
        if existing is not None:
            incoming_key = (hlc, data, tera)
            existing_key = (existing.hlc, existing.data, existing.tera)
            if incoming_key <= existing_key:
                return
        self._records[id_] = CrdtRecord(id=id_, tera=tera, data=data, hlc=hlc)

    def merge(self, other: "CrdtStore") -> "CrdtStore":
        """Order-independent by construction: each id resolves to
        whichever of the two stores holds the larger HLC for it — a `max`
        over a total order, applied independently per key."""
        result = CrdtStore()
        result._records = dict(self._records)
        for rec in other._records.values():
            result.put(rec.id, rec.tera, rec.data, rec.hlc)
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CrdtStore):
            return NotImplemented
        return self._records == other._records

    def __len__(self) -> int:
        return len(self._records)

    def records(self):
        return list(self._records.values())

    def to_json(self) -> str:
        """Serializes every record to the cross-language interchange format
        a real second writer uses to hand its writes to this process — see
        from_json. Binary payloads are base64 (JSON has no byte-string
        type); HLC fields are exactly the (physical, logical, replica_id)
        triple the comparison itself is defined over, so a non-Python
        writer only needs to reproduce that triple correctly, not any
        Python-specific encoding. replica_id is always a JSON STRING, not
        a bare number — found necessary by testing against a real second
        writer (aamt-crdt.js), not assumed: a 63-bit replica_id safely
        exceeds JavaScript's 2^53 safe-integer range, so a JS writer MUST
        emit it as a string; emitting it any other way from Python would
        make the two writers' own export formats disagree with each
        other, defeating the point of a shared interchange format."""
        records = [{
            "id": rec.id,
            "tera": list(rec.tera),
            "data": base64.b64encode(rec.data).decode("ascii"),
            "hlc": {"physical": rec.hlc.physical, "logical": rec.hlc.logical,
                    "replica_id": str(rec.hlc.replica_id)},
        } for rec in self._records.values()]
        return json.dumps({"records": records})

    @staticmethod
    def from_json(text: str) -> "CrdtStore":
        """Inverse of to_json — imports a real second writer's exported
        writes as a CrdtStore ready to merge() with a local one. Does not
        validate that the sender's replica_id is actually distinct from
        this process's own (see local_replica_id's own docstring for why
        that matters and how it's meant to be avoided at the source, not
        policed on import)."""
        payload = json.loads(text)
        store = CrdtStore()
        for rec in payload["records"]:
            # int() accepts either a JSON string or a JSON number for
            # replica_id -- robust to a sender emitting either, since the
            # interchange format's own writers (this module and
            # aamt-crdt.js) don't agree on which JSON type to use and
            # int() makes that disagreement harmless instead of a parse
            # error.
            hlc = HLC(rec["hlc"]["physical"], rec["hlc"]["logical"], int(rec["hlc"]["replica_id"]))
            store.put(rec["id"], rec["tera"], base64.b64decode(rec["data"]), hlc)
        return store

    def to_aamt(self, path: str, max_depth: int = 8, bucket_capacity: int = 32,
               mirrored: bool = False, mirrored_ecc: bool = False,
               lib_path: str | None = None) -> int:
        """Packs the CURRENT converged state as a normal, immutable .aamt
        file — a deterministic snapshot, not a new persistence mechanism.
        Returns the record count written.

        This is the closing half of the CRDT story (WP-23 §13.3): the
        LWW-merge above resolves WHAT the converged record set is; this
        method is where it actually becomes durable substrate bytes, via
        the exact same AamtBuilder every other .aamt producer in this repo
        uses (pack_oracle_demo.py, tests/roundtrip.cpp's ABI) — no
        CRDT-specific packing path to keep separately correct.

        `mirrored=True` applies WP-23 section 10's geometric redundancy
        (a Householder-T reflected copy per record) to every record.
        `mirrored_ecc=True` applies BOTH defenses (AamtBuilder.add_mirrored_ecc):
        mirror pairing AND RS(15,13)/GF(16) ECC on every record, feeding
        AamtSubstrate.read_verified's composed read path — the right choice
        whenever the packed file needs to survive real on-disk corruption,
        not just serve as a snapshot. The two are mutually exclusive (pick
        one level of protection, not a partial mix)."""
        if mirrored and mirrored_ecc:
            raise ValueError("to_aamt: mirrored and mirrored_ecc are mutually exclusive "
                             "-- mirrored_ecc already implies mirror protection, plus RS")
        import aamt_ffi  # local import: CRDT layer doesn't require libaamt for merge/property tests
        with aamt_ffi.AamtBuilder(max_depth, bucket_capacity, lib_path=lib_path) as b:
            for rec in self._records.values():
                if mirrored_ecc:
                    b.add_mirrored_ecc(rec.id, rec.tera, rec.data)
                elif mirrored:
                    b.add_mirrored(rec.id, rec.tera, rec.data)
                else:
                    b.add(rec.id, rec.tera, rec.data)
            b.write(path)
        return len(self._records)
