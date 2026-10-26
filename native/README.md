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

## Build & test

```sh
./build.sh test        # cmake + ctest (round-trip + journal replay)
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
