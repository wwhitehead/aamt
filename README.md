# .aamt

**The GPU-native file format for local AI.**

GGUF turned a model into one file you could download and run on your own CPU.
`.aamt` is built for what runs on your own **GPU**: weights bit-packed for
compute shaders, a spatial memory index, and content-addressed layers for
everything a model needs besides its weights, in one immutable, memory-mapped
file.

**Open spec, source-available:** free for individuals, research and small teams;
commercial use is licensed ([LICENSE.md](LICENSE.md), [COMMERCIAL.md](COMMERCIAL.md)).

## Try it: one command

```
npx @asamanthinks/aamt                  # chat with InnerSight on your GPU (0.72 GB download)
npx @asamanthinks/aamt run gaian        # BeyondSight GaIaN, 8B ternary (3.23 GB)
npx @asamanthinks/aamt bench gaian      # measure it on your machine; prints a shareable card
```

`aamt` is the runtime: the engine that runs complete models from `.aamt` files,
free to use under its own licence. Models download once, are checked against an
Ed25519-signed catalog, and run locally; nothing else leaves the machine. On an
M1 Max, `aamt bench gaian` measured 41.2 tok/s on 128-token replies, and the
first token of a 4,096-token prompt fell from 32.7 s to 178 ms once the prompt
had been read (184×). Run it and post your card.

## Check the format yourself

```
git clone https://github.com/wwhitehead/aamt && cd aamt && pnpm install
pnpm gpu:check
```

```
adapter: apple · metal-3 · Metal driver on macOS Version 26.5.2 (Build 25F84)
matrix:  8192 × 8192 (67.1M weights), seeded, 100 timed products

kind      file        bits/weight  rel. error   time/product   weights/s       weight GB/s
int4      41.94 MB    4.50         9.11e-8      262.1 µs       256 G           144.0
ternary   25.17 MB    2.50         1.13e-7      178.7 µs       376 G           117.4
1bit      16.78 MB    1.50         1.38e-7      222.7 µs       301 G           56.5

GPU == CPU (relative error ≤ 0.0001)
```

That run baked a matrix into a real `.aamt` file, read it back, and multiplied it
on the GPU through WebGPU, once per weight kind. It then checked every output
against a CPU reference. That was `AAMT_M=8192 AAMT_K=8192 AAMT_ITERS=100 pnpm gpu:check`
on an Apple M1 Max (Node 22, Dawn). GPU times vary by 10–20% from run to
run, and up to 25% between sessions. [CLAIMS.md](CLAIMS.md) gives medians, ranges, how each number was measured,
and what is not claimed.

---

## Why a new format

A model file used to be a container of tensors. Local AI now needs more than
that in the same place: the weights, the memory the model reads and writes, the
world it acts in, and the code it runs, each with an identity you can verify.

|                    | safetensors          | GGUF                                    | **.aamt**                                                                 |
|--------------------|----------------------|-----------------------------------------|---------------------------------------------------------------------------|
| Weights            | raw dtypes           | ggml block quants (ggml began as a CPU library) | **u32 words a shader reads once: 8 int4, 16 ternary or 32 1-bit weights** |
| Built around       | PyTorch              | ggml / llama.cpp                        | **WebGPU compute**: Dawn in Node, or a browser; Dawn targets Metal, Vulkan and D3D12 |
| Beyond weights     | string metadata      | key–value metadata                      | **5 typed layers**: model, world, mind, experience, code                   |
| Identity           | none                 | none                                    | **SHA-384** per record, per layer and per file, separated by layer         |
| Memory index       | none                 | none                                    | **hexadeca-tree** over a 4-D coordinate space: ~230–370 ns per route at 1M records (medians across sessions) |
| Replay             | none                 | none                                    | **40-byte journal** per decision, replayed bit-exactly against the file hash |
| Alignment          | 8 B                  | 32 B (configurable)                     | **16 KiB pages** for GPU-candidate arenas (Apple Silicon zero-copy)          |

GGUF is excellent at what it was built for, and `.aamt` reads none of its
files yet (an importer is on the [roadmap](#roadmap)). The difference is the
starting point. `.aamt` weights are laid out for the inner loop of a GPU kernel
(`js/packages/webgpu/src/shaders/aamt_gemv_mr.wgsl`), and the file has room for
the state a local agent accumulates, not only the weights it starts with.

## Bake a real model

```
pnpm bake ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors qwen3-0.6b.aamt
```

```
baked 311 tensors (198 int4, 113 f32) in 10.6 s
1503.3 MB -> 470.0 MB  (31.3% of the input)
```

Matrices whose rows are a multiple of 32 are quantised. Norm weights, biases and
odd shapes stay f32. Then multiply one of its real tensors on the GPU, read by
offset from the file (footer → manifest → tensor; nothing else is loaded):

```
AAMT_FILE=qwen3-0.6b.aamt AAMT_TENSOR=lm_head.weight pnpm gpu:check
```

```
tensor:  lm_head.weight — 151936 × 1024, read by offset from qwen3-0.6b.aamt
int4      97.24 MB    4.50         1.08e-7      643.3 µs       242 G           136.0
GPU == CPU (relative error ≤ 0.0001)
```

### Check it with tools you already trust

The Python package opens a `.aamt` model in Hugging Face transformers, with
every weight read from the `.aamt` file, so you can generate text from it and
measure what baking cost:

```
pip install -e "python[quality]"
python python/examples/generate.py qwen3-0.6b.aamt Qwen/Qwen3-0.6B "Why does local AI protect privacy?"
python python/examples/quality.py  qwen3-0.6b.aamt Qwen/Qwen3-0.6B
```

```
perplexity      original 5.089   .aamt 6.051   (+18.92%)
top-1 agreement 80.01% of next-token predictions
mean KL         0.2518 nats per token (original ‖ .aamt)
greedy          0/8 outputs identical over 64 tokens; first difference after 1.1 tokens on average
```

That is the honest price of the current baker. It uses plain round-to-nearest
int4 with no calibration, and a 0.6B model is sensitive to it: the baked model
still answers sensibly, but it is measurably worse. The same harness on an
all-f32 `.aamt` of the same weights reports 0.0% change and 100% agreement, so
the loss is the quantiser, not the file or the loader. Better quantisers are
the first item on the [roadmap](#roadmap).

`ternary` and `1bit` are for models trained or distilled for those kinds.
Baking a full-precision model straight to 1-bit runs, but it destroys the model.

## The memory substrate

```
./native/build.sh test
python3 native/examples/gen_random_tsv.py 1000000 > r1m.tsv
native/build/aamt pack r1m.tsv r1m.aamt
native/build/aamt bench r1m.aamt 1000000
native/build/aamt query r1m.aamt 0.72 0.31 0.88 0.15
```

```
packed 1000000 records -> r1m.aamt (318.9 ms)
open (mmap + validate): 0.052 ms
1000000 queries: 354 ns/query (2.83 Mq/s)  [checksum 23230724]
route: odu-word=5.6.13.1 depth=4 bucket=20
```

Each node splits its box in four dimensions at once, so it has exactly 2⁴ = 16
children. Two levels give 256 cells. A query's route *is* its address: the path
`5.6.13.1` above is the cell, read top-down. Nodes are single 64-byte cache
lines (occupancy bitmask + `popcount` child slots, verified by `static_assert`).
The file is opened with `mmap` and never parsed. The same C ABI
(`native/include/aamt/aamt.h`) serves the CLI, Python (`ctypes`), WebAssembly
and C# (P/Invoke).

The geometry and its measurements are in the working papers:
[WP-23](https://doi.org/10.5281/zenodo.21316192) (the substrate),
[WP-24](https://doi.org/10.5281/zenodo.21271708) (the architecture around it),
[WP-25](https://doi.org/10.5281/zenodo.21316193) (integration and wiring status).
Each paper has a corrections ledger, a list of what an earlier draft claimed
and what measurement changed.

## What is in this repository

| Path | What | Tests |
|---|---|---|
| [`SPEC.md`](SPEC.md) | The byte layouts: tensor, model (AMDL), container v1.x, substrate v0, and how to tell them apart | — |
| `native/` | C++17 substrate core, C ABI, CLI, Python bindings, WebAssembly build, Metal zero-copy bench | round-trip, journal replay, GF(16) Reed–Solomon |
| `js/packages/tera` | The coordinate math: 16 Meji orthants, 256 Odu cells, product Fisher–Rao metric, deterministic transcendentals | ALU properties, 400-case parity fixture |
| `js/packages/container` | The unified container: five layers, 384-bit ids, streaming writer, zero-copy reader, Odu index, adaptive tree | 118, including bit-flip fuzzing |
| `js/packages/model` | Tensor and model formats, the streaming safetensors baker, f16 scale packing | bake round-trip from real safetensors |
| `js/packages/webgpu` | The dequantisation and multi-row GEMV kernels, and the GPU-equals-CPU check | 13 shapes × kinds, on any machine with a GPU |
| `python/` | A numpy reader, a Hugging Face transformers loader, a text-generation demo, and the quality check | 9, against files the TypeScript writer produced |

```
pnpm test        # 205 tests (the GPU suite skips itself where there is no adapter)
pnpm typecheck
pnpm native      # build + test the C++ core
python3 -m unittest discover -s python/tests
```

**Not in this repository:** the full inference engine (forward passes, KV cache,
sampling, tuning) and the BeyondSight model weights. The engine ships compiled
inside `npx @asamanthinks/aamt`; the models are on Hugging Face under the BeyondSight Model
License. For a commercial licence or a pilot: licensing@asamanthinks.com.

## Roadmap

- **A better int4 quantiser.** Use the full [−8, 7] code range, and add
  calibration-based rounding, to cut the +18.9% perplexity cost measured on
  Qwen3-0.6B. Keep large embeddings at higher precision. `quality.py` is the
  yardstick.
- **GGUF and safetensors importers** that keep the source quantisation where a kind matches.
- **f16 scales on disk.** The GPU already reads scales as f16 pairs; storing them that way halves scale bytes on disk too.
- **Embedded models by default.** Pack the AMDL model inside the container's page-aligned model layer, so a model, its memory and its identity ship as one file.
- **Windows native build.** The C++ core builds and passes its tests on macOS and Linux in CI; Windows needs a port from POSIX `mmap`.
- **npm and PyPI packages for the format libraries.** Today, clone and `pnpm install`; the runtime is already `npx @asamanthinks/aamt`.

Issues and pull requests are welcome; contributions need the CLA in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

- **Code:** [AAMT Community License 1.0](LICENSE.md). Free for personal use,
  non-commercial research, a 60-day evaluation, and organisations under US $1M
  revenue, US $5M raised and 25 people (free registration). Everything else,
  including any product, service or competing runtime, needs a
  [commercial licence](COMMERCIAL.md).
- **Specification:** [CC BY-ND 4.0](SPEC.md), with no patent licence.
- **Patents:** pending; see [PATENTS.md](PATENTS.md).
- **Papers, books and trademarks:** see [COPYRIGHT.md](COPYRIGHT.md). The working
  papers are separate works under their own licences (WP-23, WP-24 and WP-25 are
  CC BY 4.0 — a licence to the text, not to the methods).

This is source-available software, not open source as the OSI defines it.

Built by Weslyn Cory Whitehead Jr. ·
[ORCID 0009-0005-7707-3210](https://orcid.org/0009-0005-7707-3210) ·
[asamanthinks.com](https://asamanthinks.com) · weslyn@asamanthinks.com
