# The .aamt file format — specification, version 1

> Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research. This specification is licensed under
> [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/): you may share it unchanged, with attribution.
> **It grants no patent licence.** The format is covered by pending patent applications (PATENTS.md);
> implementing it for any commercial purpose requires a licence (COMMERCIAL.md).

This document describes the bytes. The reference implementations are normative
where this text is silent: `js/packages/model` (tensor, model),
`js/packages/container` (container) and `native/src/format.hpp` (substrate).
Every integer is little-endian. Every offset is in bytes from the start of the
file unless stated otherwise.

`.aamt` names four layouts. One is current and recommended for new files (the
**container**). The others are the building blocks it carries (**tensor**,
**model**) and the memory index that preceded it (**substrate v0**). All four
can be told apart from a file's first 64 and last 32 bytes ([§5](#5-telling-them-apart)).

| Layout | Identifying bytes | Use |
|---|---|---|
| [Tensor](#1-tensor) | `AAMT` + u32 version (1 or 2) at offset 0 | One quantised tensor, as a GPU storage buffer reads it |
| [Model (AMDL)](#2-model-amdl) | `AMDL` in a 32-byte **footer** | A whole model: tensor blobs, a manifest, a footer |
| [Container v1.x](#3-container-v1x) | `\x89AAMT\r\n\x1a` at offset 0 | Five typed layers, 384-bit identities, page-aligned arenas |
| [Substrate v0](#4-substrate-v0) | `AAMT` + u16 1 + u16 0 at offset 0, and `file_size` at offset 8 | Memory-mapped hexadeca-tree over 4-D space |

---

## 1. Tensor

One tensor, quantised group-wise and bit-packed so that a compute shader reads
one `u32` and unpacks 8, 16 or 32 weights in registers.

```
offset  type                 field
0       u32                  magic        0x544D4141 ('AAMT')
4       u32                  version      2
8       u32                  kind         0 = int4, 1 = ternary, 2 = 1bit
12      u32                  count        N, number of weights
16      u32                  groupSize    G, weights per scale (32 in practice; a power of two for the GEMV kernel)
20      u32                  groupCount   ceil(N / G)
24      f32[groupCount]      scales
…       u32[ceil(N / W)]     packed       W = 8 (int4), 16 (ternary), 32 (1bit)
```

Weight `i` lives in word `i / W` at bit offset `(i mod W) · (32 / W)`:

| kind | bits | code → value |
|---|---|---|
| int4 | 4 | nibble `n` → `(n − 8) · scale`, `n − 8 ∈ [−8, 7]` |
| ternary | 2 | `00` → 0, `01` → `+scale`, `11` → `−scale` (`10` unused) |
| 1bit | 1 | `0` → `−scale`, `1` → `+scale` |

`scale` is `scales[i / G]`. Writers choose `scale = max|w| / 7` for int4 and
`max|w|` for ternary and 1bit, over the group. A **matrix** of `M` rows and `K`
columns is stored row-major; with `K` a multiple of 32, every row starts on a
word boundary and on a group boundary, which is what lets the GEMV kernel index
row `m` as word `m · K / W` and scale `m · K / G`.

Version 1 omits the `kind` word (header is five `u32`s) and is always int4.
Readers must accept both.

**On the GPU**, the reference kernels (`js/packages/webgpu/src/shaders`) take
the `packed` words as they sit in the file. The f32 `scales` are converted to
f16 pairs at upload (two per `u32`, read with `unpack2x16float`), because at
1–2 bits per weight an f32 scale per 32 weights is as many bytes as the weights
themselves.

## 2. Model (AMDL)

A model is a sequence of tensor blobs followed by a JSON manifest and a fixed
footer. The footer is at the end so a writer can stream tensors to disk
without knowing the total size first.

```
[ blob 0 ][ blob 1 ] … [ blob n−1 ][ manifest: UTF-8 JSON ][ footer: 32 B ]
```

Footer, at `fileSize − 32`:

```
0   u32  magic           0x4C444D41 ('AMDL')
4   u32  version         1
8   u32  tensorCount
12  u32  reserved        0
16  f64  manifestOffset  (exact for any offset < 2⁵³)
24  f64  manifestLength
```

The manifest is an array, one entry per tensor, in blob order:

```json
[{ "name": "model.layers.0.mlp.down_proj.weight", "shape": [1024, 3072],
   "offset": 0, "length": 1966104, "kind": "int4" }]
```

- `offset` and `length` locate the blob in the file.
- `kind` is `int4`, `ternary`, `1bit` or `f32`. Absent means `int4`. A
  quantised blob's own header must agree with its manifest `kind`.
- An `f32` blob is the raw little-endian floats, with no header.
- An entry with `length` 0 and a `metadata` object carries model-level
  metadata (for example a tokenizer family) instead of a tensor.

To read one tensor: read the footer, read the manifest, read `length` bytes at
`offset`. Nothing else in the file is touched (`readTensorFromFile`).

## 3. Container v1.x

The container is the recommended form for new files. It holds up to five
layers, each a sorted index of fixed-size record entries plus an arena of
payloads, and it gives every record, layer and file a 384-bit identity.

```
┌──────────────┬───────────────────────┬──────────────────────────────────────────┐
│ header 128 B │ layer table 128 B × L │ per layer: index (128 B × N), arena,     │
│              │                       │ then optional Odu index (v1.2) and tree  │
│              │                       │ (v1.3) sections                          │
└──────────────┴───────────────────────┴──────────────────────────────────────────┘
  immutable after pack · mmap-able as-is · opened with a few bounded reads, no parse
```

### 3.1 Header (128 B)

```
0   u8[8]   magic             89 41 41 4D 54 0D 0A 1A   ("\x89AAMT\r\n\x1a")
8   u16     versionMajor      1
10  u16     versionMinor      2, or 3 when any layer carries a tree section
12  u32     layerCount
16  u64     fileBytes         must equal the file's real length
24  u64     layerTableOffset  128
32  u8[48]  fileId            SHA-384("MaiiaM Yare v1 · container" ‖ 0x00 ‖ whole file with this field zeroed)
80  u32     flags             0
84  u8[44]  reserved          0
```

Byte 0 is non-ASCII, so no legacy `AAMT` writer can produce this magic. The
`\r\n` and `\x1a` catch line-ending and text-mode corruption, as PNG's do.

### 3.2 Layer entry (128 B)

```
0    u32     kind          1 model, 2 world, 3 mind, 4 experience, 5 code
4    u32     flags         bit 0 read-only, 1 copy-on-write, 2 disposable, 3 journaled, 4 keyed (reserved)
8    u64     indexOffset   absolute, 64-aligned
16   u64     recordCount
24   u64     arenaOffset   absolute; aligned per layer (below)
32   u64     arenaBytes
40   u8[48]  layerId       SHA-384(layer context ‖ 0x00 ‖ index ‖ arena)
88   u64     oduOffset     v1.2: Odu index section, 0 if none
96   u64     oduBytes
104  u64     treeOffset    v1.3: adaptive tree section, 0 if none
112  u64     treeBytes
120  u8[8]   reserved
```

Layers always appear in kind order, so two packs of the same content are
byte-identical.

| Layer | Locator | Index sorted by | Arena alignment | Default flags |
|---|---|---|---|---|
| model | none | id | 16 KiB | read-only |
| world | integer lattice `(x, y, z)` | Morton key, then id | 16 KiB | journaled |
| mind | integer lattice `(x, y, z)` | Morton key, then id | 16 KiB | copy-on-write, disposable |
| experience | 4-D point + lens + tier | id | 64 B | journaled |
| code | 4-D point + lens + tier | id | 64 B | read-only |

**Alignment.** 16 KiB is the Apple Silicon VM page, which
`newBufferWithBytesNoCopy` requires to wrap mapped memory as a GPU buffer
without copying. It is also a multiple of the 4 KiB page used elsewhere.
Payloads are 16-byte aligned (one `vec4<f32>`), except in the model layer,
where each payload is page-aligned so an embedded model can be handed to the
GPU without a copy.

**Only spatial locators order storage.** A manifold (4-D) locator is a search
key; it never orders, offsets or buckets records.

### 3.3 Record entry (128 B)

```
0    u8[48]  id              SHA-384(layer context ‖ 0x00 ‖ u32 kind ‖ [locator, spatial layers only] ‖ payload)
48   u32     kind            0 opaque, 1 AMDL embedded, 2 AMDL reference, 6 model manifest; 3, 4, 5, 7 reserved for code and encoders
52   u32     flags           bit 0: mirror strand
56   u64     payloadOffset   relative to the layer's arena
64   u64     payloadBytes
72   u8[32]  locator         spatial:  i32 x, i32 y, i32 z, u32 0, u64 mortonKey, u64 0
                             manifold: f32 point[4], u32 lens, u32 tier, u64 0
                             none:     zeros
104  u8[24]  reserved
```

Readers must skip record kinds they do not know. Morton keys interleave three
21-bit fields (x at bit 0, y at bit 1, z at bit 2) after offsetting each
coordinate by 2²⁰.

### 3.4 Identity

```
id = SHA-384( "MaiiaM Yare v1 · <layer>" ‖ 0x00 ‖ bytes )
```

The context string is UTF-8 (`·` is U+00B7), contains no `0x00`, and is
followed by one, so `context ‖ 0x00 ‖ bytes` is prefix-free. `<layer>` is one of
`model`, `world`, `mind`, `experience`, `code`; the file id uses `container`.
The layer name in the context string separates layers, so identical bytes in
two layers never share an id. A record's locator is part of its identity only
in spatial layers, where position is part of what a thing is. SHA-384 is
available natively in `node:crypto`, WebCrypto, Python, .NET and C++, so every
surface computes the same id without a dependency. It gives 2¹⁹² classical
collision resistance. The id-context string is fixed for v1 files.

### 3.5 Optional sections

- **Odu index (v1.2)**, magic `ODU1`: for each of the 256 cells of a manifold
  layer, the positions of the records whose locator falls in that cell. It
  narrows a search and never decides storage. Layout in
  `js/packages/container/src/odu.ts`.
- **Adaptive tree (v1.3)**: a variable-depth version of the same partition for
  skewed data. Layout in `js/packages/container/src/tree.ts`.

A reader that predates a section ignores its fields (they were reserved and
zero) and still opens the file.

## 4. Substrate v0

A memory-mapped hexadeca-tree over the unit 4-cube, written by `native/`.

```
+----------------+---------------------------+---------------------------+
|  header 128 B  |  node arena (64 B × N)    |  payload arena (records)  |
+----------------+---------------------------+---------------------------+
```

Header:

```
0    u32   magic                 0x544D4141 ('AAMT')
4    u16   version_major         1
6    u16   version_minor         0
8    u64   file_size
16   u64   node_arena_offset
24   u64   node_count
32   u64   payload_arena_offset
40   u64   payload_arena_size
48   u64   record_count
56   u32   max_depth             ≤ 8
60   u32   bucket_capacity
64   u64   content_hash          FNV-1a 64 over node arena ‖ payload arena
72   u8[56] reserved
```

Node (exactly 64 bytes, `static_assert`ed):

```
0   u32  coordinate_id    this node's path, one 4-bit Meji per level
4   f32  chirality_spin   reserved
8   u16  child_mask       which of the 16 children exist
10  u16  flags            bit 0 leaf; bits 8–15 depth
12  u32  record_count     leaf bucket size
16  u64  children_base    node index of the first existing child
24  u64  payload_offset   leaf bucket, in the payload arena
32  u64  payload_size
40  u8[24] reserved
```

Child `m` of a node is at `children_base + popcount(child_mask & ((1 << m) − 1))`.
A point `v` in a box with midpoint `mid` goes to child
`m = b₀ + 2b₁ + 4b₂ + 8b₃`, `b_d = [v_d ≥ mid_d]`. Ties go up. Records are a
32-byte header (`u64 id, f32 point[4], u32 size, u8 flags, u8[3]`) followed by
`size` bytes, padded to 8.

The **journal** is a sidecar `.aamtj` of 40-byte records
`{ u64 t_ns, u64 file_hash, f32 point[4], u32 path, u32 depth }`, one per query.
Because the file is immutable and content-hashed, `(file_hash, point)`
determines the route, and replaying the journal must reproduce every `path`
bit-for-bit.

Note the numbering difference: the substrate puts axis 0 at bit 0 of a Meji
nibble, while the container's Odu cell packs axis-major with axis 0 in the
high bits. The two agree on every point; `js/packages/container/src/odu.ts`
converts explicitly (`libaamtCellToOdu`).

## 5. Telling them apart

Read the first 64 bytes, the last 32, and the file length, then check in this
order (`sniffAamt` in `js/packages/container/src/sniff.ts`):

1. **Container**: the 8-byte magic at offset 0 and `fileBytes` equal to the length.
2. **Model**: `AMDL` in the footer. Checked before the `AAMT` head checks,
   because a model's first blob is itself a tensor that begins with `AAMT`.
3. **Substrate v0**: `AAMT`, u16 1, and `file_size` at offset 8 equal to the
   length.
4. **Tensor**: `AAMT`, version 1 or 2, and the exact length its own header
   implies.

A version-1 tensor and a v0 substrate are identical through byte 8. They
separate on the length check, which a valid tensor cannot pass.

## 6. Versioning

The container's major version changes only if an existing field changes
meaning. Minor versions may add sections in reserved space; they must leave
older readers able to open the file and ignore what they do not know. A file is
stamped with the lowest minor version that describes it.
