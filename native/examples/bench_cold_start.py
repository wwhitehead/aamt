#!/usr/bin/env python3
# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""WP-23 Section 6 benchmark: cold start to first query.

Compares the platform's current persistence path (parse a TSV into Python
dicts, as the desktop bridges do) against the .aamt substrate (mmap open via
the C ABI + one routed query). Generates its own seeded synthetic data.

Usage: python3 bench_cold_start.py [n_records=100000]
"""
import ctypes
import os
import random
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "..", "build", "libaamt.dylib")
CLI = os.path.join(HERE, "..", "build", "aamt")


class Route(ctypes.Structure):
    _fields_ = [("path", ctypes.c_uint32), ("depth", ctypes.c_uint32),
                ("record_count", ctypes.c_uint32)]


class RecordView(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64), ("tera", ctypes.c_float * 4),
                ("data", ctypes.POINTER(ctypes.c_uint8)),
                ("size", ctypes.c_uint32)]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    rng = random.Random(0x5EED)

    tmp = tempfile.mkdtemp(prefix="aamt_bench_")
    tsv = os.path.join(tmp, "records.tsv")
    aamt = os.path.join(tmp, "records.aamt")

    with open(tsv, "w") as f:
        for i in range(n):
            t, e, r, a = (rng.random() for _ in range(4))
            f.write(f"{i}\t{t:.6f}\t{e:.6f}\t{r:.6f}\t{a:.6f}\t"
                    f"capsule payload for record {i}\n")

    subprocess.run([CLI, "pack", tsv, aamt], check=True, capture_output=True)
    size_mb = os.path.getsize(aamt) / 1e6

    # --- current path: parse TSV into dicts (what the bridges do today) ---
    t0 = time.perf_counter()
    store = {}
    with open(tsv) as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            store[int(cols[0])] = ((float(cols[1]), float(cols[2]),
                                    float(cols[3]), float(cols[4])), cols[5])
    probe = store[0]  # first usable lookup
    t_parse = time.perf_counter() - t0

    # --- substrate path: mmap open + one routed query ---
    lib = ctypes.CDLL(LIB)
    lib.aamt_open.restype = ctypes.c_void_p
    lib.aamt_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.aamt_query.restype = ctypes.c_int32
    lib.aamt_query.argtypes = [ctypes.c_void_p, ctypes.c_float * 4,
                               ctypes.POINTER(RecordView), ctypes.c_int32,
                               ctypes.POINTER(Route)]
    lib.aamt_close.argtypes = [ctypes.c_void_p]

    err = ctypes.create_string_buffer(256)
    t0 = time.perf_counter()
    handle = lib.aamt_open(aamt.encode(), err, 256)
    assert handle, err.value
    tera = (ctypes.c_float * 4)(0.5, 0.5, 0.5, 0.5)
    views = (RecordView * 64)()
    route = Route()
    got = lib.aamt_query(handle, tera, views, 64, ctypes.byref(route))
    t_mmap = time.perf_counter() - t0

    print(f"records:                 {n:,}  ({size_mb:.1f} MB .aamt)")
    print(f"parse TSV -> dicts:      {t_parse * 1000:8.1f} ms   (grows with N)")
    print(f"aamt_open + first query: {t_mmap * 1000:8.1f} ms   "
          f"(independent of N; routed depth={route.depth}, "
          f"bucket={route.record_count}, returned={got})")
    print(f"speedup: {t_parse / t_mmap:,.0f}x")
    _ = probe
    lib.aamt_close(handle)


if __name__ == "__main__":
    main()
