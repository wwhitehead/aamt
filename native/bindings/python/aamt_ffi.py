# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""ctypes binding for the .aamt substrate (WP-23).

Thin wrapper over the C ABI in include/aamt/aamt.h — the same boundary the
CLI, the WASM build, and (eventually) Unity's P/Invoke bindings all use.
No packaging step: ctypes loads the shared library directly by path, so any
Python process (regardless of venv) can use this as long as it can find
libaamt.dylib/.so.

    from aamt_ffi import AamtSubstrate, AamtBuilder, find_libaamt

    sub = AamtSubstrate.open("memory.aamt")
    result = sub.query([0.5, 0.5, 0.5, 0.5], capacity=16)
    for rec in result.records:
        print(rec.id, rec.tera, rec.data)

    b = AamtBuilder(max_depth=8, bucket_capacity=32)
    b.add(1, [0.2, 0.3, 0.4, 0.5], b"payload bytes")
    b.write("out.aamt")
"""
from __future__ import annotations

import ctypes
import enum
import os
import platform
import struct
from dataclasses import dataclass


class AamtError(RuntimeError):
    pass


def find_libaamt() -> str | None:
    """Resolves libaamt by, in order: $AAMT_LIB_PATH, then the build output
    next to these bindings (native/build/), for the common case of this
    script running inside a checkout of the aamt repository. Returns None
    (rather than raising) so callers can degrade gracefully — this is a
    substrate accelerator, not a required dependency."""
    env = os.environ.get("AAMT_LIB_PATH")
    if env and os.path.exists(env):
        return env

    ext = "dylib" if platform.system() == "Darwin" else "so"
    here = os.path.dirname(os.path.abspath(__file__))
    # bindings/python -> native
    candidates = [
        os.path.join(here, "..", "..", "build", f"libaamt.{ext}"),
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.exists(c):
            return c
    return None


class _Route(ctypes.Structure):
    _fields_ = [("path", ctypes.c_uint32), ("depth", ctypes.c_uint32),
               ("record_count", ctypes.c_uint32)]


class _RecordView(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64), ("tera", ctypes.c_float * 4),
               ("data", ctypes.POINTER(ctypes.c_uint8)), ("size", ctypes.c_uint32)]


class _SoftBranch(ctypes.Structure):
    _fields_ = [("weight", ctypes.c_float), ("meji", ctypes.c_uint32), ("route", _Route)]


class _ChiralityReport(ctypes.Structure):
    _fields_ = [("primary_count", ctypes.c_uint64), ("mirror_count", ctypes.c_uint64),
               ("consistent", ctypes.c_uint64), ("inconsistent_count", ctypes.c_uint64)]


class _VerifiedMeta(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64), ("tera", ctypes.c_float * 4),
               ("status", ctypes.c_uint32), ("corrected_symbols", ctypes.c_uint32)]


@dataclass
class Record:
    id: int
    tera: tuple
    data: bytes


@dataclass
class QueryResult:
    path: int
    depth: int
    bucket_size: int
    records: list


@dataclass
class VerifiedRecord:
    """One entry from AamtSubstrate.query_verified — a primary that passed
    the read_verified cascade (WP-23 §13.1)."""
    id: int
    tera: tuple
    data: bytes
    status: "ReadStatus"
    corrected_symbols: int


@dataclass
class VerifiedQueryResult:
    path: int
    depth: int
    primary_count: int  # raw primaries in the routed leaf, before any were dropped
    records: list        # only entries that passed verification (len <= primary_count)


@dataclass
class SoftBranch:
    weight: float
    meji: int
    path: int
    depth: int
    bucket_size: int
    records: list


@dataclass
class VerifiedSoftBranch:
    weight: float
    meji: int
    path: int
    depth: int
    primary_count: int  # raw primaries in this branch's leaf, before verification
    records: list         # only entries that passed verification


@dataclass
class ChiralityReport:
    """The mirror-pairing gate's verdict. `inconsistent_ids` (empty unless
    requested) lists the specific "bad reflections" — see
    AamtSubstrate.verify_chirality()."""
    primary_count: int
    mirror_count: int
    consistent: int
    inconsistent_count: int
    inconsistent_ids: list

    @property
    def ok(self) -> bool:
        return self.inconsistent_count == 0


class ReadStatus(enum.IntEnum):
    """Verdict from AamtSubstrate.read_verified() — see its docstring for
    the decided policy (WP-23 §13)."""
    OK = 0                  # primary decoded, zero RS corrections
    OK_RS_REPAIRED = 1      # primary needed a correction; trusted
    OK_MIRROR_RECOVERED = 2 # primary missing/unresolvable; mirror served it
    UNAVAILABLE = 3         # both unreadable, or in positive disagreement


@dataclass
class ReadResult:
    status: ReadStatus
    data: bytes | None
    tera: tuple | None
    corrected_symbols: int

    @property
    def ok(self) -> bool:
        return self.status != ReadStatus.UNAVAILABLE


def odu_word(path: int, depth: int) -> str:
    """Renders a routing path as a dotted Odu word, e.g. '5.6.13.1'."""
    return ".".join(str((path >> (4 * i)) & 0xF) for i in range(depth - 1, -1, -1))


def tree_cell_to_axis_major(cell: int) -> int:
    """Converts a depth-2 cell index from the tree convention to voxel.py's.

    Tree (the substrate's own truth — a routing path IS the address):
      cell = (m1 << 4) | m2, Meji nibbles; within each nibble T is bit 0.
    voxel.py / OduCoverageMap (axis-major):
      cell = (lT << 6) | (lE << 4) | (lR << 2) | lA, quartile levels, T high.
    Same 256 quartile cells, same >= boundary rule on both sides (verified:
    voxel._level's floor(t*4) and the tree's two >= splits agree everywhere,
    including boundaries) — only the bit packing differs. Per axis d
    (0=T..3=A): level l_d = 2*bit_d(m1) + bit_d(m2)."""
    m1, m2 = (cell >> 4) & 0xF, cell & 0xF
    out = 0
    for d in range(4):  # d: 0=T, 1=E, 2=R, 3=A
        level = 2 * ((m1 >> d) & 1) + ((m2 >> d) & 1)
        out |= level << (6 - 2 * d)  # T lands in bits 7-6, A in bits 1-0
    return out


_lib_cache: dict[str, ctypes.CDLL] = {}


def _load(path: str) -> ctypes.CDLL:
    if path not in _lib_cache:
        lib = ctypes.CDLL(path)
        lib.aamt_open.restype = ctypes.c_void_p
        lib.aamt_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.aamt_close.argtypes = [ctypes.c_void_p]
        lib.aamt_node_count.restype = ctypes.c_uint64
        lib.aamt_node_count.argtypes = [ctypes.c_void_p]
        lib.aamt_record_count.restype = ctypes.c_uint64
        lib.aamt_record_count.argtypes = [ctypes.c_void_p]
        lib.aamt_content_hash.restype = ctypes.c_uint64
        lib.aamt_content_hash.argtypes = [ctypes.c_void_p]
        lib.aamt_max_depth.restype = ctypes.c_uint32
        lib.aamt_max_depth.argtypes = [ctypes.c_void_p]
        lib.aamt_query_soft.restype = ctypes.c_int32
        lib.aamt_query_soft.argtypes = [ctypes.c_void_p, ctypes.c_float * 4, ctypes.c_uint32,
                                        ctypes.POINTER(_SoftBranch), ctypes.POINTER(_RecordView),
                                        ctypes.c_uint32]
        lib.aamt_odu_profile.restype = None
        lib.aamt_odu_profile.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
                                         ctypes.POINTER(ctypes.c_float)]
        lib.aamt_query.restype = ctypes.c_int32
        lib.aamt_query.argtypes = [ctypes.c_void_p, ctypes.c_float * 4,
                                   ctypes.POINTER(_RecordView), ctypes.c_int32,
                                   ctypes.POINTER(_Route)]
        lib.aamt_all_records.restype = ctypes.c_int64
        lib.aamt_all_records.argtypes = [ctypes.c_void_p, ctypes.POINTER(_RecordView),
                                         ctypes.c_int64]
        lib.aamt_payload_arena.restype = ctypes.POINTER(ctypes.c_uint8)
        lib.aamt_payload_arena.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        lib.aamt_all_records_columnar.restype = ctypes.c_int64
        lib.aamt_all_records_columnar.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),  # out_ids
            ctypes.c_void_p,                  # out_tera (nullable; float*)
            ctypes.POINTER(ctypes.c_uint8),   # out_payload
            ctypes.c_uint64,                  # payload_capacity
            ctypes.POINTER(ctypes.c_uint64),  # out_offsets
            ctypes.POINTER(ctypes.c_uint32),  # out_sizes
            ctypes.c_int64,                   # capacity
        ]
        lib.aamt_builder_new.restype = ctypes.c_void_p
        lib.aamt_builder_new.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        lib.aamt_builder_add.restype = ctypes.c_int
        lib.aamt_builder_add.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                         ctypes.c_float * 4, ctypes.c_char_p, ctypes.c_uint32]
        lib.aamt_builder_add_mirrored.restype = ctypes.c_int
        lib.aamt_builder_add_mirrored.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                                  ctypes.c_float * 4, ctypes.c_char_p,
                                                  ctypes.c_uint32]
        lib.aamt_builder_add_ecc.restype = ctypes.c_int
        lib.aamt_builder_add_ecc.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                             ctypes.c_float * 4, ctypes.c_char_p, ctypes.c_uint32]
        lib.aamt_encode_ecc_payload.restype = ctypes.c_uint32
        lib.aamt_encode_ecc_payload.argtypes = [ctypes.c_char_p, ctypes.c_uint32,
                                                ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32]
        lib.aamt_decode_ecc_payload.restype = ctypes.c_int64
        lib.aamt_decode_ecc_payload.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
                                                ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64,
                                                ctypes.POINTER(ctypes.c_uint32)]
        lib.aamt_verify_chirality.restype = None
        lib.aamt_verify_chirality.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ChiralityReport),
                                              ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64]
        lib.aamt_recover_record.restype = ctypes.c_int64
        lib.aamt_recover_record.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                            ctypes.c_float * 4, ctypes.POINTER(ctypes.c_uint8),
                                            ctypes.c_uint64, ctypes.c_float * 4]
        lib.aamt_builder_add_mirrored_ecc.restype = ctypes.c_int
        lib.aamt_builder_add_mirrored_ecc.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                                       ctypes.c_float * 4, ctypes.c_char_p,
                                                       ctypes.c_uint32]
        for fn in ('aamt_builder_add_ecc_checked', 'aamt_builder_add_mirrored_ecc_checked'):
            getattr(lib, fn).restype = ctypes.c_int
            getattr(lib, fn).argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_float * 4,
                                         ctypes.c_char_p, ctypes.c_uint32]
        lib.aamt_read_verified.restype = ctypes.c_int64
        lib.aamt_read_verified.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                           ctypes.c_float * 4, ctypes.POINTER(ctypes.c_uint8),
                                           ctypes.c_uint64, ctypes.c_float * 4,
                                           ctypes.POINTER(ctypes.c_uint32),
                                           ctypes.POINTER(ctypes.c_uint32)]
        lib.aamt_query_verified.restype = ctypes.c_int32
        lib.aamt_query_verified.argtypes = [ctypes.c_void_p, ctypes.c_float * 4, ctypes.c_int32,
                                            ctypes.POINTER(_VerifiedMeta),
                                            ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64,
                                            ctypes.POINTER(ctypes.c_uint64),
                                            ctypes.POINTER(ctypes.c_uint32),
                                            ctypes.POINTER(_Route)]
        lib.aamt_query_verified_soft.restype = ctypes.c_int32
        lib.aamt_query_verified_soft.argtypes = [
            ctypes.c_void_p, ctypes.c_float * 4, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(_SoftBranch), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(_VerifiedMeta),
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.aamt_builder_write.restype = ctypes.c_int
        lib.aamt_builder_write.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                           ctypes.c_char_p, ctypes.c_size_t]
        lib.aamt_builder_free.argtypes = [ctypes.c_void_p]
        lib.aamt_journal_open.restype = ctypes.c_void_p
        lib.aamt_journal_open.argtypes = [ctypes.c_char_p]
        lib.aamt_journal_log.restype = ctypes.c_int
        lib.aamt_journal_log.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_float * 4, ctypes.POINTER(_Route)]
        lib.aamt_journal_close.argtypes = [ctypes.c_void_p]
        _lib_cache[path] = lib
    return _lib_cache[path]


class AamtSubstrate:
    """Read-only handle on a .aamt file, mmap'd by the native reader."""

    def __init__(self, lib: ctypes.CDLL, handle: int, path: str | None = None):
        self._lib = lib
        self._handle = handle
        self._path = path  # kept for record_file_offset's own raw-file header read

    @classmethod
    def open(cls, path: str, lib_path: str | None = None) -> "AamtSubstrate":
        lib_path = lib_path or find_libaamt()
        if not lib_path:
            raise AamtError("libaamt not found; set AAMT_LIB_PATH or build packages/libaamt")
        lib = _load(lib_path)
        err = ctypes.create_string_buffer(256)
        handle = lib.aamt_open(path.encode(), err, 256)
        if not handle:
            raise AamtError(f"aamt_open({path!r}) failed: {err.value.decode(errors='replace')}")
        return cls(lib, handle, path=path)

    @property
    def node_count(self) -> int:
        return self._lib.aamt_node_count(self._handle)

    @property
    def record_count(self) -> int:
        return self._lib.aamt_record_count(self._handle)

    @property
    def max_depth(self) -> int:
        return self._lib.aamt_max_depth(self._handle)

    @property
    def content_hash(self) -> str:
        return f"{self._lib.aamt_content_hash(self._handle):016x}"

    def query(self, tera, capacity: int = 16) -> QueryResult:
        tera_arr = (ctypes.c_float * 4)(*tera)
        views = (_RecordView * capacity)()
        route = _Route()
        n = self._lib.aamt_query(self._handle, tera_arr, views, capacity, ctypes.byref(route))
        records = [Record(id=views[i].id, tera=tuple(views[i].tera),
                          data=ctypes.string_at(views[i].data, views[i].size))
                  for i in range(n)]
        return QueryResult(path=route.path, depth=route.depth,
                           bucket_size=route.record_count, records=records)

    def query_soft(self, tera, k: int = 16, per_branch_capacity: int = 8) -> list:
        """Soft routing: ranks all 16 root-level Meji orthants by
        mejiDistribution weight (descending), hard-descending the top `k`
        each to their own leaf. Empty orthants are reported too (weight,
        zero records). Purely geometric — how much `tera` resembles each
        orthant, no visit history — so this widens the candidate pool near
        real data but is NOT WP-22's occupancy-aware escape search (that's
        odu_coverage_map.py's own session tracking; see WP-23 §7 for why
        they don't merge). branches[0] always matches query()'s hard-argmax
        result exactly."""
        tera_arr = (ctypes.c_float * 4)(*tera)
        k = max(1, min(k, 16))
        branches = (_SoftBranch * k)()
        views = (_RecordView * (k * per_branch_capacity))()
        n = self._lib.aamt_query_soft(self._handle, tera_arr, k, branches, views,
                                      per_branch_capacity)

        out = []
        for i in range(n):
            b = branches[i]
            base = i * per_branch_capacity
            emitted = min(b.route.record_count, per_branch_capacity)
            records = [Record(id=views[base + j].id, tera=tuple(views[base + j].tera),
                              data=ctypes.string_at(views[base + j].data, views[base + j].size))
                      for j in range(emitted)]
            out.append(SoftBranch(weight=b.weight, meji=b.meji, path=b.route.path,
                                  depth=b.route.depth, bucket_size=b.route.record_count,
                                  records=records))
        return out

    def odu_profile(self) -> tuple:
        """Persisted per-Odu-cell profile: (counts, centroids) over the 256
        depth-2 cells, computed from every record's stored TERA coordinate.
        counts is a list of 256 ints; centroids a list of 256 (T,E,R,A)
        tuples, (0,0,0,0) for empty cells. Cell indices use the TREE
        convention (m1<<4)|m2 — NOT voxel.py's axis-major packing; use
        tree_cell_to_axis_major() to permute for OduCoverageMap."""
        counts = (ctypes.c_uint32 * 256)()
        centroids = (ctypes.c_float * (256 * 4))()
        self._lib.aamt_odu_profile(self._handle, counts, centroids)
        return (list(counts),
                [tuple(centroids[c * 4 + d] for d in range(4)) for c in range(256)])

    def verify_chirality(self, id_capacity: int = 256) -> ChiralityReport:
        """The mirror-pairing gate: for every non-mirror record, re-routes
        its Householder-T reflection for real and checks a byte-identical
        mirror exists there. Only meaningful on a substrate built with
        AamtBuilder.add_mirrored — on a plain-add substrate every record
        reports inconsistent (no mirrors exist at all). `id_capacity` caps
        how many inconsistent ids come back (the aggregate counts are always
        exact regardless of this cap)."""
        rep = _ChiralityReport()
        ids = (ctypes.c_uint64 * id_capacity)()
        self._lib.aamt_verify_chirality(self._handle, ctypes.byref(rep), ids, id_capacity)
        n = min(rep.inconsistent_count, id_capacity)
        return ChiralityReport(primary_count=rep.primary_count, mirror_count=rep.mirror_count,
                               consistent=rep.consistent,
                               inconsistent_count=rep.inconsistent_count,
                               inconsistent_ids=[ids[i] for i in range(n)])

    def record_file_offset(self, id_: int, tera, capacity: int | None = None) -> int:
        """Locates the on-disk byte offset of a record's 32-byte RecordHeader
        (format.hpp) — exposed so Python callers can write REAL corruption
        tests (open the file separately in 'r+b' and flip actual bytes) the
        same way tests/roundtrip.cpp does (sections 5-7), instead of only
        being able to simulate corruption. Two pieces of native state make
        this possible without re-deriving the packed layout by hand: the
        file's own 128-byte FileHeader stores `payload_arena_offset`
        (format.hpp), and aamt_payload_arena / aamt_all_records hand back
        real in-process pointers into the same mmap — their difference IS
        the record's offset within that arena, so
        payload_arena_offset + (record_ptr - arena_base_ptr) - 32 is its
        real file offset (32 == sizeof(RecordHeader), landing on the start
        of the header rather than the payload bytes after it).

        Matches by id AND tera (both float32-truncated exactly as the C ABI
        stores them, so equality is exact, not approximate) so a PRIMARY is
        found even on a mirrored substrate: the mirror shares the same id
        but stores the Householder-T reflection of tera, which is never
        equal to the coordinate a caller actually inserted at (see
        format.hpp's householderReflectT). Raises AamtError if no such
        record exists, or if this substrate wasn't opened from a path
        (record_file_offset needs to re-read the file's own header)."""
        if self._path is None:
            raise AamtError("record_file_offset requires a substrate opened via "
                            "AamtSubstrate.open(path) -- no path was recorded")
        with open(self._path, "rb") as f:
            header = f.read(128)
        if len(header) < 128:
            raise AamtError(f"record_file_offset: {self._path!r} has a truncated FileHeader")
        (payload_arena_offset,) = struct.unpack_from("<Q", header, 32)

        arena_size = ctypes.c_uint64(0)
        arena_base = self._lib.aamt_payload_arena(self._handle, ctypes.byref(arena_size))
        base_addr = ctypes.cast(arena_base, ctypes.c_void_p).value

        if capacity is None:
            capacity = self.record_count  # already counts mirrors; see format.hpp / core.cpp
        views = (_RecordView * capacity)()
        got = self._lib.aamt_all_records(self._handle, views, capacity)

        # Truncate through float32 exactly as the packer did, so comparison
        # against the on-disk (already-float32) tera is exact, not fuzzy.
        target_tera = tuple(ctypes.c_float(v).value for v in tera)
        for i in range(min(got, capacity)):
            if views[i].id == id_ and tuple(views[i].tera) == target_tera:
                data_addr = ctypes.cast(views[i].data, ctypes.c_void_p).value
                return payload_arena_offset + (data_addr - base_addr) - 32
        raise AamtError(f"record_file_offset: no record with id={id_} tera={target_tera} "
                        f"found in {self._path!r}")

    def recover_record(self, id_: int, tera_hint, capacity: int = 4096):
        """Recovery scoped to what two copies can actually decide: if no
        primary is found at tera_hint (a structural failure), falls back to
        the mirror, recovering the true tera via the Householder reflection
        (its own inverse). Does NOT adjudicate a payload-only disagreement
        where both copies are structurally consistent — see
        verify_chirality for that (genuinely undecidable with 2 copies and
        no independent checksum). Returns None if neither copy is found."""
        tera_arr = (ctypes.c_float * 4)(*tera_hint)
        buf = (ctypes.c_uint8 * capacity)()
        out_tera = (ctypes.c_float * 4)()
        n = self._lib.aamt_recover_record(self._handle, id_, tera_arr, buf, capacity, out_tera)
        if n < 0:
            return None
        return (bytes(buf[:n]), tuple(out_tera))

    def read_verified(self, id_: int, tera_hint, capacity: int = 4096) -> ReadResult:
        """The composed read path (WP-23 §13): RS(15,13) is the cheap first
        line of defense, the mirror is the independent second, wired
        together instead of left for the caller to sequence by hand. Most
        useful on a substrate packed with AamtBuilder.add_mirrored_ecc;
        degrades gracefully (RS-only or mirror-only trust) on one built
        with just one of the two protections, or neither (plain add() —
        every read reports UNAVAILABLE, since there is nothing to verify
        against, correctly refusing to fabricate a guarantee that isn't
        there).

        Decided policy: a mirror that positively decodes to DIFFERENT bytes
        than the primary is the one case with real counter-evidence of an
        unresolved disagreement between two independently stored,
        independently decoded copies — genuinely undecidable with what
        this format stores (no 3rd copy, no independent checksum), so it
        is refused (status UNAVAILABLE, data None) rather than guessed. An
        absent, unreadable, or merely non-confirming mirror is NOT
        counter-evidence and never blocks trusting a primary that decodes
        cleanly."""
        tera_arr = (ctypes.c_float * 4)(*tera_hint)
        buf = (ctypes.c_uint8 * capacity)()
        out_tera = (ctypes.c_float * 4)()
        status = ctypes.c_uint32()
        corrected = ctypes.c_uint32()
        n = self._lib.aamt_read_verified(self._handle, id_, tera_arr, buf, capacity, out_tera,
                                         ctypes.byref(status), ctypes.byref(corrected))
        st = ReadStatus(status.value)
        if n < 0:
            return ReadResult(status=st, data=None, tera=None, corrected_symbols=corrected.value)
        return ReadResult(status=st, data=bytes(buf[:n]), tera=tuple(out_tera),
                          corrected_symbols=corrected.value)

    def query_verified(self, tera, capacity: int = 32,
                       payload_capacity: int = 65536) -> VerifiedQueryResult:
        """The discovery-path counterpart to query()/read_verified(): routes
        tera and returns only the leaf's PRIMARY records (never mirrors —
        a mirror's own stored tera is a reflection, not "near tera" in any
        sense this call means), each independently verified through the
        same RS+mirror cascade read_verified uses. A candidate that cannot
        be vouched for (mirror positively disagrees, or both copies are
        unreadable) is DROPPED, not returned with garbage bytes — use this
        instead of query() wherever a raw candidate's payload is about to
        be trusted downstream (e.g. treated as an id to look up elsewhere).
        `primary_count` on the result is the raw count before any were
        dropped, so "empty leaf" is distinguishable from "leaf had
        candidates, all refused"."""
        tera_arr = (ctypes.c_float * 4)(*tera)
        meta = (_VerifiedMeta * capacity)()
        payload = (ctypes.c_uint8 * payload_capacity)()
        offsets = (ctypes.c_uint64 * capacity)()
        sizes = (ctypes.c_uint32 * capacity)()
        route = _Route()
        n = self._lib.aamt_query_verified(self._handle, tera_arr, capacity, meta,
                                          payload, payload_capacity, offsets, sizes,
                                          ctypes.byref(route))
        records = []
        for i in range(n):
            off, sz = offsets[i], sizes[i]
            records.append(VerifiedRecord(
                id=meta[i].id, tera=tuple(meta[i].tera),
                data=bytes(payload[off:off + sz]),
                status=ReadStatus(meta[i].status),
                corrected_symbols=meta[i].corrected_symbols))
        return VerifiedQueryResult(path=route.path, depth=route.depth,
                                   primary_count=route.record_count, records=records)

    def query_verified_soft(self, tera, k: int = 16, per_branch_capacity: int = 8,
                            payload_capacity: int = 65536) -> list:
        """The soft-routing counterpart to query_verified: ranks all 16
        root-level Meji orthants (routeSoftTopK's own ranking) and applies
        the same primary-only, RS+mirror-verified treatment to each
        branch's leaf. Exists because raw query_soft() branches have the
        exact same primary/mirror ambiguity query_verified resolves for
        the hard route — verifying raw branch records one at a time from
        Python (which cannot see record flags) would repeat the bug
        query_verified was built to fix. Returns a list of
        VerifiedSoftBranch, ranked by weight descending."""
        tera_arr = (ctypes.c_float * 4)(*tera)
        branches_native = (_SoftBranch * k)()
        branch_counts = (ctypes.c_uint32 * k)()
        meta = (_VerifiedMeta * (k * per_branch_capacity))()
        payload = (ctypes.c_uint8 * payload_capacity)()
        offsets = (ctypes.c_uint64 * (k * per_branch_capacity))()
        sizes = (ctypes.c_uint32 * (k * per_branch_capacity))()
        nb = self._lib.aamt_query_verified_soft(
            self._handle, tera_arr, k, per_branch_capacity, branches_native, branch_counts,
            meta, payload, payload_capacity, offsets, sizes)
        out = []
        for bi in range(nb):
            b = branches_native[bi]
            records = []
            for j in range(branch_counts[bi]):
                idx = bi * per_branch_capacity + j
                off, sz = offsets[idx], sizes[idx]
                records.append(VerifiedRecord(
                    id=meta[idx].id, tera=tuple(meta[idx].tera),
                    data=bytes(payload[off:off + sz]),
                    status=ReadStatus(meta[idx].status),
                    corrected_symbols=meta[idx].corrected_symbols))
            out.append(VerifiedSoftBranch(
                weight=b.weight, meji=b.meji, path=b.route.path, depth=b.route.depth,
                primary_count=b.route.record_count, records=records))
        return out

    def all_records(self, capacity: int | None = None) -> list:
        """Walks every leaf and returns every record — the zero-JSON-parse
        replacement for reading a full id/label table off disk. Pass
        `capacity` (e.g. a known record count from a small header) to skip
        the extra native call this method would otherwise make to probe the
        size first."""
        if capacity is None:
            capacity = self._lib.aamt_all_records(self._handle, None, 0)
        if capacity <= 0:
            return []
        views = (_RecordView * capacity)()
        got = self._lib.aamt_all_records(self._handle, views, capacity)
        n = min(got, capacity)
        return [Record(id=views[i].id, tera=tuple(views[i].tera),
                       data=ctypes.string_at(views[i].data, views[i].size))
               for i in range(n)]

    def all_ids_and_text(self, capacity: int | None = None) -> list:
        """Leaner bulk read for the common id + UTF-8 payload case (no tera,
        no Record allocation). ~2x faster than all_records() in practice: the
        per-record cost here is entirely the ctypes FFI-crossing tax of
        string_at() + decode(), which measurably dominates over the native
        tree walk (see WP-23 memory_index_bridge.py benchmark) — this trims
        it to the minimum unavoidable work, not zero."""
        if capacity is None:
            capacity = self._lib.aamt_all_records(self._handle, None, 0)
        if capacity <= 0:
            return []
        views = (_RecordView * capacity)()
        got = self._lib.aamt_all_records(self._handle, views, capacity)
        n = min(got, capacity)
        string_at = ctypes.string_at
        return [(views[i].id, string_at(views[i].data, views[i].size).decode("utf-8"))
               for i in range(n)]

    def all_ids_and_text_bulk(self, capacity: int | None = None) -> list:
        """Columnar bulk read: one native call fills flat primitive arrays
        (no per-record ctypes.Structure allocation) and one bulk copy pulls
        the whole payload blob into a single Python bytes object; per-record
        work after that is pure bytes-slicing + decode, no FFI crossing at
        all. This is the fix for the loss all_ids_and_text() measures against
        json.load (WP-23 corrections ledger #8) — see
        examples/bench_memory_index_bridge.py for the re-measurement."""
        if capacity is None:
            capacity = self.record_count
        if capacity <= 0:
            return []
        payload_cap = ctypes.c_uint64(0)
        self._lib.aamt_payload_arena(self._handle, ctypes.byref(payload_cap))
        payload_cap = payload_cap.value

        ids = (ctypes.c_uint64 * capacity)()
        payload = (ctypes.c_uint8 * payload_cap)()
        offsets = (ctypes.c_uint64 * capacity)()
        sizes = (ctypes.c_uint32 * capacity)()

        n = self._lib.aamt_all_records_columnar(
            self._handle, ids, None, payload, payload_cap, offsets, sizes, capacity)
        n = min(n, capacity)

        blob = bytes(payload)  # one bulk copy, not one per record
        return [(ids[i], blob[offsets[i]:offsets[i] + sizes[i]].decode("utf-8"))
               for i in range(n)]

    def close(self):
        if self._handle:
            self._lib.aamt_close(self._handle)
            self._handle = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class AamtJournal:
    """Append-only sidecar journal (.aamtj) — one 40-byte record per routing
    decision: (t_ns, file_hash, tera, path, depth). WP-23 section 5 claims
    "deterministic replay of internal routing state" as the substrate's
    falsifiable property; this is that claim made callable, not just a
    fact exercised in isolation by tests/roundtrip.cpp. Read back with
    read_journal(); check accumulated real history against the current
    substrate with verify_journal()."""

    def __init__(self, path: str, lib_path: str | None = None):
        lib_path = lib_path or find_libaamt()
        if not lib_path:
            raise AamtError("libaamt not found; set AAMT_LIB_PATH or build packages/libaamt")
        self._lib = _load(lib_path)
        self._handle = self._lib.aamt_journal_open(path.encode())
        if not self._handle:
            raise AamtError(f"aamt_journal_open({path!r}) failed (cannot open for append)")

    def log(self, substrate: "AamtSubstrate", tera, result) -> None:
        """Logs one routing decision. `result` is anything with .path/.depth
        (a QueryResult from query(), a SoftBranch from query_soft(), or a
        VerifiedSoftBranch from query_verified_soft() — soft routing's
        rank-0 branch equals the hard route by construction, tested in
        tests/roundtrip.cpp, so logging it through the same journal format
        is correct). record_count is informational only (not part of
        verify_journal's correctness check, which is path/depth replay);
        checks both attribute names since verified results report
        primary_count rather than bucket_size."""
        tera_arr = (ctypes.c_float * 4)(*tera)
        record_count = getattr(result, "bucket_size", None)
        if record_count is None:
            record_count = getattr(result, "primary_count", 0)
        route = _Route(path=result.path, depth=result.depth, record_count=record_count)
        if self._lib.aamt_journal_log(self._handle, substrate._handle, tera_arr,
                                      ctypes.byref(route)) != 0:
            raise AamtError("aamt_journal_log failed")

    def close(self):
        if self._handle:
            self._lib.aamt_journal_close(self._handle)
            self._handle = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


_JOURNAL_RECORD_FMT = "<QQ4fII"
_JOURNAL_RECORD_SIZE = struct.calcsize(_JOURNAL_RECORD_FMT)
assert _JOURNAL_RECORD_SIZE == 40


@dataclass
class JournalEntry:
    t_ns: int
    file_hash: int
    tera: tuple
    path: int
    depth: int


def read_journal(path: str) -> list:
    """Reads a .aamtj sidecar directly. The format is small, fixed-width,
    and documented (format.hpp's JournalRecord) — no C ABI round-trip
    needed to read it back, only to write it (aamt_journal_log is the
    single writer; this mirrors what tests/roundtrip.cpp does in C++)."""
    entries = []
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_JOURNAL_RECORD_SIZE)
            if not chunk:
                break
            if len(chunk) < _JOURNAL_RECORD_SIZE:
                raise AamtError(f"truncated journal record in {path!r}")
            t_ns, file_hash, t0, t1, t2, t3, path_, depth = struct.unpack(
                _JOURNAL_RECORD_FMT, chunk)
            entries.append(JournalEntry(t_ns=t_ns, file_hash=file_hash,
                                        tera=(t0, t1, t2, t3), path=path_, depth=depth))
    return entries


@dataclass
class VerifyResult:
    total: int
    matched: int
    hash_mismatches: int
    route_mismatches: list  # [(index, JournalEntry, actual_path, actual_depth), ...]

    @property
    def ok(self) -> bool:
        return self.hash_mismatches == 0 and not self.route_mismatches


def verify_journal(aamt_path: str, journal_path: str) -> VerifyResult:
    """Replays every logged query against the CURRENT substrate and checks
    the routing decision is bit-exact. A hash mismatch means the substrate
    file changed since logging (expected: any repack invalidates old
    entries, since the whole point is that a file's routing behavior is
    fixed by its content); a route mismatch on a matching hash would mean
    the router itself is non-deterministic — the same property
    tests/roundtrip.cpp checks on a synthetic 1000-query batch, checked here
    against whatever history actually accumulated."""
    entries = read_journal(journal_path)
    mismatches = []
    hash_mismatches = 0
    with AamtSubstrate.open(aamt_path) as sub:
        current_hash = int(sub.content_hash, 16)
        for i, e in enumerate(entries):
            if e.file_hash != current_hash:
                hash_mismatches += 1
                continue
            result = sub.query(e.tera, capacity=1)
            if result.path != e.path or result.depth != e.depth:
                mismatches.append((i, e, result.path, result.depth))
    matched = len(entries) - hash_mismatches - len(mismatches)
    return VerifyResult(total=len(entries), matched=matched,
                        hash_mismatches=hash_mismatches, route_mismatches=mismatches)


@dataclass
class EccDecodeResult:
    ok: bool
    corrected_symbols: int
    data: bytes


def encode_ecc_payload(data: bytes, lib_path: str | None = None) -> bytes:
    """RS(15,13)/GF(16)-encodes `data` — a stateless byte transform, not
    tied to any open substrate. ~15/13 size expansion. See
    AamtBuilder.add_ecc / decode_ecc_payload."""
    lib_path = lib_path or find_libaamt()
    if not lib_path:
        raise AamtError("libaamt not found; set AAMT_LIB_PATH or build packages/libaamt")
    lib = _load(lib_path)
    size = lib.aamt_encode_ecc_payload(data, len(data), None, 0)
    buf = (ctypes.c_uint8 * size)()
    lib.aamt_encode_ecc_payload(data, len(data), buf, size)
    return bytes(buf)


def decode_ecc_payload(encoded: bytes, capacity: int | None = None,
                       lib_path: str | None = None) -> EccDecodeResult:
    """Decodes and repairs an RS-encoded buffer — free, exact repair of one
    corrupted nibble anywhere in `encoded`. NOT a guarantee beyond that:
    measured (tests/test_rs_gf16.cpp, exhaustive), a second corrupted
    nibble in the same 13-nibble block is silently "corrected" to the
    wrong data 86.7% of the time — pair with mirror-pairing / verify_journal
    for anything this can't decide on its own. Raises AamtError if the
    syndrome could not be resolved at all (ok=False case)."""
    lib_path = lib_path or find_libaamt()
    if not lib_path:
        raise AamtError("libaamt not found; set AAMT_LIB_PATH or build packages/libaamt")
    lib = _load(lib_path)
    if capacity is None:
        capacity = max(len(encoded) * 2, 64)
    buf = (ctypes.c_uint8 * capacity)()
    enc_arr = (ctypes.c_uint8 * len(encoded))(*encoded)
    corrected = ctypes.c_uint32(0)
    n = lib.aamt_decode_ecc_payload(enc_arr, len(encoded), buf, capacity, ctypes.byref(corrected))
    if n < 0:
        raise AamtError("ECC payload unresolvable — more corruption than this code can detect")
    return EccDecodeResult(ok=True, corrected_symbols=corrected.value, data=bytes(buf[:n]))


class AamtBuilder:
    """Accumulates records in memory, then packs a .aamt file in one write()."""

    def __init__(self, max_depth: int = 8, bucket_capacity: int = 32, lib_path: str | None = None):
        lib_path = lib_path or find_libaamt()
        if not lib_path:
            raise AamtError("libaamt not found; set AAMT_LIB_PATH or build packages/libaamt")
        self._lib = _load(lib_path)
        self._handle = self._lib.aamt_builder_new(max_depth, bucket_capacity)

    def add(self, id_: int, tera, data: bytes) -> None:
        tera_arr = (ctypes.c_float * 4)(*tera)
        self._lib.aamt_builder_add(self._handle, id_, tera_arr, data, len(data))

    def add_mirrored(self, id_: int, tera, data: bytes) -> None:
        """Adds the record normally, AND a full redundant copy at the leaf
        its Householder-T reflection (T -> 1-T) routes to — a different
        subtree, a different region of the payload arena, so damage
        localized to one region doesn't take out both copies. Real 2x
        storage cost for real geometric redundancy. See
        AamtSubstrate.verify_chirality / recover_record for the read side."""
        tera_arr = (ctypes.c_float * 4)(*tera)
        self._lib.aamt_builder_add_mirrored(self._handle, id_, tera_arr, data, len(data))

    def add_ecc(self, id_: int, tera, data: bytes) -> None:
        """RS(15,13)/GF(16)-encodes the payload before storing — free,
        exact repair of one corrupted nibble anywhere in the payload on
        read (decode_ecc_payload), at ~15/13 size expansion. Complements,
        does not replace, add_mirrored — see decode_ecc_payload's
        docstring for the measured reason why."""
        tera_arr = (ctypes.c_float * 4)(*tera)
        self._lib.aamt_builder_add_ecc(self._handle, id_, tera_arr, data, len(data))

    def add_mirrored_ecc(self, id_: int, tera, data: bytes) -> None:
        """Both defenses on one record: primary AND mirror each store the
        SAME RS(15,13)-encoded payload. Feeds AamtSubstrate.read_verified,
        the composed read path this exists for (WP-23 §13)."""
        tera_arr = (ctypes.c_float * 4)(*tera)
        self._lib.aamt_builder_add_mirrored_ecc(self._handle, id_, tera_arr, data, len(data))

    def add_ecc_checked(self, id_: int, tera, data: bytes) -> None:
        """add_ecc plus a CRC-32 of the original bytes inside the RS protection,
        so heavy damage RS would mis-correct is detected instead of served (see
        aamt.h, aamt_builder_add_ecc_checked, for the measurement)."""
        tera_arr = (ctypes.c_float * 4)(*tera)
        self._lib.aamt_builder_add_ecc_checked(self._handle, id_, tera_arr, data, len(data))

    def add_mirrored_ecc_checked(self, id_: int, tera, data: bytes) -> None:
        """add_mirrored_ecc with the CRC check: a shredded copy is unreadable,
        so read_verified recovers the record from the other copy. Use this for
        new substrates; read_verified needs no change."""
        tera_arr = (ctypes.c_float * 4)(*tera)
        self._lib.aamt_builder_add_mirrored_ecc_checked(self._handle, id_, tera_arr, data, len(data))
    def write(self, path: str) -> None:
        err = ctypes.create_string_buffer(256)
        if self._lib.aamt_builder_write(self._handle, path.encode(), err, 256) != 0:
            raise AamtError(f"aamt_builder_write({path!r}) failed: {err.value.decode(errors='replace')}")

    def close(self):
        if self._handle:
            self._lib.aamt_builder_free(self._handle)
            self._handle = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
