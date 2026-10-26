# Claims

Every number in the README, what produced it, and what it does not mean. The
convention follows the AAMT working papers: **measured** (a run printed it),
**derived** (arithmetic on measured numbers), **recorded elsewhere** (measured,
but not reproducible from this repository), and **not claimed**.

Machine for every measurement below unless stated: Apple M1 Max, 32 GB,
macOS 26.5.2, Node 22.21.1, Dawn via `webgpu` 0.4, Apple clang. Captured
2026-10-25. Numbers drift from run to run; where it matters, the median and
range of five runs are given.

## Measured

These are the receipts captured for the launch film (`capture-receipts.mts`,
section `aamt`, in the MaiiaM video pipeline). That script re-runs this
repository at commit `8ddeac0`. Before each benchmark it waits for the 1-minute
load average to fall to 3 or below, and it reports the median of five runs.
No other benchmark, long-running job or video call was active.

**Between sessions, the numbers move more than within one.** The same
measurements, taken in four sessions on the same day and machine, each a
median of five runs:

| measurement | session 1 | session 2 | session 3 | session 4 (this capture) |
|---|---|---|---|---|
| int4 product, 8192² | 334 µs | 320 µs | 272 µs | **295 µs** |
| ternary product, 8192² | 240 µs | 206 µs | 185 µs | **172 µs** |
| 1-bit product, 8192² | 261 µs | 196 µs | 259 µs | **261 µs** |
| Qwen3 `lm_head` product | 799 µs | 630 µs | 692 µs | **641 µs** |
| one route, 1M records | 374 ns | 228 ns | 328 ns | **317 ns** |

Sessions 1 and 2 ran alongside two unrelated processes using ~40% of a core
each; sessions 3 and 4 ran after they were stopped. That did not explain the spread:
routing was fastest in session 2. Within any one session the five runs agree
closely (routing in session 3: 322–341 ns). The cause between sessions
(thermal state, clock behaviour, other foreground apps) has not been isolated.
Read every number below as one session's median, about ±25% from another
session's.

### GPU: weights multiplied straight from `.aamt` bytes

`AAMT_M=8192 AAMT_K=8192 AAMT_ITERS=100 pnpm gpu:check`, five runs. One product
is `y = W·x` for an 8192 × 8192 matrix (67.1M weights), timed over 100
back-to-back products after 10 warm-up products.

| kind | file | bits / weight | time per product, median (range) | weights / s, median (range) |
|---|---|---|---|---|
| int4 | 41.94 MB | 4.50 | 295 µs (263–352) | 228 G (191–255) |
| ternary | 25.17 MB | 2.50 | 172 µs (167–298) | 391 G (225–403) |
| 1bit | 16.78 MB | 1.50 | 261 µs (248–276) | 257 G (243–271) |

Largest relative error against the CPU across every kind and run: **1.4e-7**.
Bits per weight include the f16 scales the GPU reads (one per 32 weights). The
relative error is `max |gpu − cpu| / max |cpu|`, with the CPU reference in f64
using the same f16-rounded scales. The check fails above 1e-4.

Ternary was the fastest kind in every session. 1-bit moves the fewest bytes,
but it is not reliably faster than ternary or int4, because at these widths the
kernel's inner loop is latency-bound, not bandwidth-bound (see the comments in
`aamt_gemv_mr.wgsl`).

### GPU: a real model tensor, read by offset

`AAMT_FILE=qwen3-0.6b.aamt AAMT_TENSOR=lm_head.weight AAMT_ITERS=50 pnpm gpu:check`,
five runs. The tensor is Qwen3-0.6B's output projection, 151,936 × 1024 (155.6M
weights), int4, 97.24 MB. Only the footer, the manifest and that tensor's bytes
are read from the file.

- **641 µs** per product, median (range 626–670 µs); **243 G weights/s**.
- Relative error against the CPU: **1.1e-7**.

### Baking a real checkpoint

`pnpm bake <Qwen3-0.6B model.safetensors> qwen3-0.6b.aamt`

- 311 tensors: 198 quantised to int4, 113 kept f32 (norms and other 1-D tensors).
- 1.50 GB (bf16) → 470 MB, in 10.4 s (10.6–11.0 s in earlier runs).
- Peak resident memory 1.6 GB (measured with `/usr/bin/time -l` on the earlier
  run). The baker holds one tensor at a time, so peak memory follows the
  largest tensor (here the 151,936 × 1024 embedding as f32, about 622 MB, plus
  its source bytes and output), not the file. For a model this small that is
  about the size of the file; it doesn't show baking in small memory.

### Quality: what baking to int4 costs

`python python/examples/quality.py qwen3-0.6b.aamt Qwen/Qwen3-0.6B`, one run,
Apple M1 Max GPU (MPS), float32. Both models run in Hugging Face transformers
5.17: the original from its own checkpoint, the other with every weight from the
`.aamt` file. The text is the first 64 items of the GSM8K test set (question and
worked answer), at most 256 tokens each: 11,317 tokens. Nothing is sampled.
Full result: `python/examples/quality-qwen3-0.6b-int4.json`.

| | original (bf16) | `.aamt` int4 |
|---|---|---|
| perplexity | 5.089 | 6.051 (**+18.9%**) |
| same next-token prediction | — | **80.0%** of positions |
| mean KL, original ‖ `.aamt` | — | 0.252 nats per token |
| greedy output, 8 prompts × 64 tokens | — | 0 of 8 identical; first difference after 1.1 tokens on average |

- **The harness is exact.** An all-f32 `.aamt` of the same weights reports 0.0%
  perplexity change and 100% agreement, so the loader adds no error.
- **The loss is the quantiser.** Restoring the (tied) 151,936 × 1024 embedding
  to full precision recovers only 2.4 points (+16.5% remains). Most of the cost
  is plain round-to-nearest int4 in the transformer layers, with no
  calibration, on a small model that is sensitive to it.
- The baked model still answers the questions coherently, but it phrases them
  differently and is measurably worse. This is a weakness of the current baker,
  not of the file layout, and it is not compared against any other quantiser.

### Memory substrate

`native/examples/gen_random_tsv.py 1000000 7`, then `aamt pack` and
`aamt bench … 1000000`, five bench runs.

- Routing: **317 ns/query** median (range 300–357), single-threaded, through the
  full C ABI including per-call record marshalling. Other sessions: 228 to
  374 ns (see the table above).
- Open (`mmap` + header validation): **0.030 ms**.
- Pack: 1,000,000 records, 69,962 nodes × 64 B, in 268–421 ms across runs.
- Determinism: the packed file's content hash was `48c4a574eabcb9f0` on every
  pack in every session, and the bench checksum `23230724` on every run.
- Query `(0.72, 0.31, 0.88, 0.15)` routes to cell `5.6.13.1` at depth 4 with a
  20-record bucket, the same route WP-23 §6 reports.

### Tests

- `pnpm test`: 205 tests in 10 files pass, including the GPU suite (13 cases:
  three kinds × four shapes that cross the kernel's tile and row edges, plus a
  refusal case).
- `./native/build.sh test`: round-trip with journal replay, and GF(16)
  Reed–Solomon, pass.
- `pnpm typecheck`: clean under `strict` and `noUncheckedIndexedAccess`.
- `python -m unittest discover -s python/tests`: 9 tests pass. The Python
  reader reproduces the TypeScript reader's dequantised values exactly, for
  every kind, on a fixture written by the TypeScript writer.

## Derived

- Qwen3-0.6B at int4 is **31.3%** of its bf16 size (470.0 / 1503.3).
- 4.50 / 2.50 / 1.50 bits per weight: 4 / 2 / 1 bits of code plus 16 bits of f16
  scale per 32 weights.

## The runtime — reproducible with `npx @asamanthinks/aamt`, not from this repository's source

These come from the closed engine that runs complete models from `.aamt` files. It ships compiled as `npx @asamanthinks/aamt`,
so anyone can rerun them on their own hardware; its source is not in this repository.

`aamt bench`, one run per model on the M1 Max above, 2026-10-25, load average about 4 during the run (other
work on the machine). Decode is measured over 128-token greedy replies on the GEMV split the engine's own
auto-tune chose, median of 3.

| | InnerSight (1.7B ternary) | NightOwl (8B 1-bit) | GaIaN (8B ternary) |
|---|---|---|---|
| load | 2.0 s | 5.2 s | 6.1 s |
| decode, 128-token replies | 85.9 tok/s | 38.6 tok/s | 41.2 tok/s |
| first token, ~4.1k-token prompt: cold → already read | 13.5 s → 130 ms | 31.8 s → 169 ms | 32.7 s → 178 ms (184×) |
| branch sharing 2,082 tokens: cold → reused | 5.0 s → 132 ms | 14.3 s → 272 ms | 14.5 s → 252 ms |
| GEMV check, real tensor, GPU vs CPU | max error 1.2e-7 | 1.7e-7 | 2.6e-7 |

The engine's own receipts record a higher decode figure for GaIaN, **47.5 tok/s** (engine commit `4d9cb90e`,
2026-10-22): that is the decode-only rate over 16-token bursts during calibration on a quieter machine. The card
reports the sustained 128-token figure, and that is the one to quote.

Other recorded figures: GaIaN is 3.23 GB on disk as `.aamt`; a 16-bit 8B model is 16.4 GB.

## Not claimed

- **No end-to-end tokens/s from this repository's source.** It holds the format,
  the baker and two kernels. The runtime that produces the figures above is
  `npx @asamanthinks/aamt`, which is closed-source.
- **No claim that `.aamt` is faster than GGUF**, or than any other format. No
  head-to-head benchmark has been run. The comparison in the README is about
  what each format is designed to hold.
- **Only Metal has been measured.** Dawn also targets Vulkan and D3D12, and the
  kernels use no optional WebGPU features, but no other backend has been run.
- **Zero-copy GPU loading is not universal.** On Apple Silicon, wrapping mapped
  memory as a Metal buffer measured 4.18× faster than copying at 528 MB, and
  *slower* (0.56×) at 4.8 MB (WP-23 §4; `native/metal`). WebGPU has no shared
  address space with mapped files, so a WebGPU upload is always one copy.
- **The int4 baker is not lossless, nor state of the art.** It costs Qwen3-0.6B
  +18.9% perplexity (above). No comparison with other int4 schemes (GGUF
  Q4 variants, GPTQ, AWQ) has been run.
- **1-bit and ternary are not a compression mode for ordinary models.** Baking
  a full-precision model to them runs, and ruins it. They are for models
  trained or distilled for those kinds.
- **Windows native builds are unverified.** The C++ core builds and passes its
  tests on macOS (Apple clang) and Linux (GCC, `ubuntu-latest` in CI). It uses
  POSIX `mmap`, so Windows needs a port.
- **The substrate's routing is O(depth), not O(1).** Scan-free and bounded,
  growing with the logarithm of the record count (WP-23 §6).
