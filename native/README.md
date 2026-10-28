# @aamt/libaamt

Native core of the AAMT Micro-VM layer (**WP-23**, `docs/papers/sources/wp-23-aamt-substrate.md`).

The `.aamt` substrate is a memory-mapped binary file holding a hexadeca-tree
spatial index over the 4D TERA cube. Each node splits its box into the
**2⁴ = 16 orthants — the 16 Meji**; two levels = the 256 Odu; a query's routing
path is literally an Odu word. Nodes are true 64-byte cache lines (occupancy
bitmask + popcount child indexing, the HAMT/ESVO technique). Payloads are
opaque, zero-copy slices served straight from the mapping. A sidecar `.aamtj`
journal makes every routing decision bit-exactly replayable against the file's
content hash.

Positioning (per WP-23 §7): this is the **geometric memory substrate beside**
the LLM runtime — payloads carry EXP-01 trit anchors, WP-21 evicted capsules,
WP-22 coverage state, embeddings/HNSW shards. It does not replace model weights.

## Self-repair

Records can be written with two independent defences, and read back through one verified path:

- **Mirror** (`aamt_builder_add_mirrored`): a second copy at the record's Householder-T reflection — the opposite half
  of the tree, a different region of the file. `aamt_verify_chirality` audits every pair.
- **RS(15,13)/GF(16)** (`aamt_builder_add_ecc`): exact repair of one corrupted nibble per 13-nibble block.
- **Checked** (`aamt_builder_add_mirrored_ecc_checked`): both, plus a CRC-32 of the original bytes inside the RS
  protection. RS alone cannot detect heavy damage — it repairs garbage into a valid-looking different payload — so
  without the check a damaged copy can be served wrong. With it, a damaged copy is unreadable and
  `aamt_read_verified` serves the other one, or refuses.

`tests/test_selfheal.cpp`, 20 random nibble flips across both copies of a record, 400 trials: unchecked, 162 reads
returned wrong bytes with a success status; checked, 0 — and shredding either copy of a checked record is recovered
from the other. One flipped nibble anywhere is always repaired. Use the checked builders for new files; files written
with the unchecked ones read exactly as before.

## Build & test

```sh
./build.sh test        # cmake + ctest (round-trip + journal replay, GF(16) RS, self-repair)
```

## CLI

```sh
build/aamt pack records.tsv out.aamt   # TSV: id\tT\tE\tR\tA\tpayload
build/aamt inspect out.aamt
build/aamt query out.aamt 0.5 0.5 0.5 0.5
build/aamt bench out.aamt 1000000      # ns/query routing latency
```

## Benchmarks (WP-23 §6)

```sh
python3 examples/bench_cold_start.py 100000
```

## Bindings

One C ABI in `include/aamt/aamt.h` serves every surface:

- **Python** — `ctypes` against `build/libaamt.dylib` (see `examples/bench_cold_start.py`)
- **C# / Unity** — P/Invoke the same symbols
- **Web** — compile `src/*.cpp` with Emscripten; same header, WASM linear memory
