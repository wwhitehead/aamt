// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * The unified `.aamt` container — byte layout.
 *
 * ## Why a new container at all
 *
 * `.aamt` named three formats, and two of them could not be told apart by their
 * first eight bytes:
 *
 * | format | first bytes | where |
 * |---|---|---|
 * | AMDL multi-tensor (models) | `AMDL` in a 32-byte **footer** | `plugin-sdk/src/aamt-format.ts` |
 * | INT4/ternary/1-bit tensor blob | `AAMT` + u32 version 1·2 | `voxel-runtime/src/aamt-format.ts` |
 * | hexadeca-tree substrate (WP-23) | `AAMT` + u16 1 + u16 0 | `libaamt/src/format.hpp` |
 *
 * A v1 blob and a v1.0 substrate are byte-identical through offset 8. The same
 * file extension was carrying model data and world data with nothing in the
 * bytes to say which. This container gives every `.aamt` written from now on
 * one header, one magic no legacy writer can produce, and a **layer table** that
 * says what each region is. {@link sniffAamt} still classifies all three legacy
 * formats exactly, so nothing already on disk stops working.
 *
 * ## File layout
 *
 * ```
 *  ┌──────────────┬───────────────────────┬────────────────────────────────────┐
 *  │ header 128 B │ layer table 128 B × L │ per layer: index (128 B × N), then │
 *  │              │                       │ arena (page-aligned payloads), then│
 *  │              │                       │ an optional Odu index (v1.2)       │
 *  └──────────────┴───────────────────────┴────────────────────────────────────┘
 *    immutable after pack · every integer little-endian · mmap-able as-is
 * ```
 *
 * Everything a reader needs is reachable from the header by bounded offsets, so
 * opening a container is a few `DataView` reads and no parse. Payloads are
 * handed out as `subarray` views into the caller's buffer: zero copies between
 * the bytes on disk and the consumer (WP-23 §3.3).
 *
 * ## Alignment
 *
 * - Header, layer table and every index entry: 64-byte multiples (cache lines).
 * - Model, world and mind arenas start on a {@link PAGE_BYTES} boundary; experience
 *   and code arenas on 64 bytes ({@link ARENA_ALIGN}, v1.1). 16 KiB is the Apple
 *   Silicon VM page, and `newBufferWithBytesNoCopy` requires page alignment
 *   (WP-23 §4 measured that zero-copy wins 2–4× on large arenas); it is also a
 *   multiple of the 4 KiB page `libaamt` and Linux use, so one rule serves both.
 * - Payloads: 16-byte aligned (a `vec4<f32>`) — except in the **model** layer,
 *   where each payload is page-aligned so an embedded AMDL file can be wrapped as
 *   a GPU buffer without a copy.
 */

/** `\x89 A A M T \r \n \x1a`. Byte 0 is non-ASCII, so no legacy `AAMT` file can match. */
export const MAGIC = Uint8Array.of(0x89, 0x41, 0x41, 0x4d, 0x54, 0x0d, 0x0a, 0x1a);
export const VERSION_MAJOR = 1;
/**
 * v1.1: arena alignment is per layer ({@link ARENA_ALIGN}). Every v1.0 file page-aligned
 * all arenas, which satisfies v1.1's rule too, so v1.0 files open unchanged.
 *
 * v1.2: a manifold layer may carry an Odu index section (`odu.ts`, ADR-018), located by
 * two layer-entry fields that were reserved and zero in v1.0/v1.1. A reader ignores
 * those fields in files older than v1.2, so v1.0 and v1.1 files open unchanged; the
 * section sits after its layer's arena, so a v1.1 reader still opens a v1.2 file.
 *
 * v1.3: a manifold layer may also carry an adaptive Meji tree section (`tree.ts`, ADR-023), located
 * by two more fields that were reserved and zero. A file is stamped 1.3 **only when it holds a
 * tree**; every other file is stamped {@link VERSION_MINOR} and is byte-for-byte what it was. A
 * v1.2 reader never looks at the new fields, so it opens a v1.3 file and uses its Odu index (or
 * scans) as before.
 */
export const VERSION_MINOR = 2;
/** The minor version of a file that carries a tree section. */
export const VERSION_MINOR_TREE = 3;

export const HEADER_BYTES = 128;
export const LAYER_ENTRY_BYTES = 128;
export const RECORD_ENTRY_BYTES = 128;
export const PAGE_BYTES = 16384;
/**
 * Where each layer's arena starts. Model, world and mind arenas are page-aligned:
 * they are the candidates for zero-copy GPU wrapping (WP-23 §4). Experience and code
 * arenas only need cache-line alignment, so a one-function code layer is not padded
 * to 16 KiB (measured in ADR-008: 16.8 KB, almost all padding).
 */
export const ARENA_ALIGN: Readonly<Record<'model' | 'world' | 'mind' | 'experience' | 'code', number>> = {
  model: PAGE_BYTES,
  world: PAGE_BYTES,
  mind: PAGE_BYTES,
  experience: 64,
  code: 64,
};
export const PAYLOAD_ALIGN = 16;
/** SHA-384 output: 48 bytes, twelve `u32` words. */
export const ID_BYTES = 48;

// ── Header (128 B) ──────────────────────────────────────────────────────────
//   0  u8[8]  magic
//   8  u16    versionMajor
//  10  u16    versionMinor
//  12  u32    layerCount
//  16  u64    fileBytes
//  24  u64    layerTableOffset            (always HEADER_BYTES in v1.0)
//  32  u8[48] fileId   = SHA-384 over the whole file with this field zeroed
//  80  u32    flags    (reserved, 0)
//  84  u8[44] reserved (0)
export const H_VERSION_MAJOR = 8;
export const H_VERSION_MINOR = 10;
export const H_LAYER_COUNT = 12;
export const H_FILE_BYTES = 16;
export const H_LAYER_TABLE = 24;
export const H_FILE_ID = 32;
export const H_FLAGS = 80;

// ── Layer entry (128 B) ─────────────────────────────────────────────────────
//   0  u32    kind                         (LayerKind id)
//   4  u32    flags                        (LAYER_FLAG_*)
//   8  u64    indexOffset                  (absolute; 64-aligned)
//  16  u64    recordCount
//  24  u64    arenaOffset                  (absolute; PAGE-aligned)
//  32  u64    arenaBytes
//  40  u8[48] layerId  = SHA-384(layer context ‖ 0 ‖ index ‖ arena)
//  88  u64    oduOffset                    (v1.2; absolute, 64-aligned; 0 = no Odu index)
//  96  u64    oduBytes                     (v1.2; 0 = no Odu index)
// 104  u64    treeOffset                   (v1.3; absolute, 64-aligned; 0 = no tree)
// 112  u64    treeBytes                    (v1.3; 0 = no tree)
// 120  u8[8]  reserved
export const L_KIND = 0;
export const L_FLAGS = 4;
export const L_INDEX = 8;
export const L_COUNT = 16;
export const L_ARENA = 24;
export const L_ARENA_BYTES = 32;
export const L_ID = 40;
export const L_ODU_OFFSET = 88;
export const L_ODU_BYTES = 96;
export const L_TREE_OFFSET = 104;
export const L_TREE_BYTES = 112;
/** The first minor version whose layer entries may locate an Odu index section. */
export const ODU_SINCE_MINOR = 2;
/** The first minor version whose layer entries may locate a tree section. */
export const TREE_SINCE_MINOR = 3;

// ── Record entry (128 B) ────────────────────────────────────────────────────
//   0  u8[48] id       = SHA-384(layer context ‖ 0 ‖ u32 kind ‖ [locator, spatial only] ‖ payload)
//  48  u32    kind                         (RECORD_KIND_*)
//  52  u32    flags                        (RECORD_FLAG_*)
//  56  u64    payloadOffset                (relative to the layer arena)
//  64  u64    payloadBytes
//  72  u8[32] locator  — interpretation depends on the layer's locator class:
//               spatial  : i32 x, i32 y, i32 z, u32 0, u64 mortonKey, u64 0
//               manifold : f32 tera[4], u32 lens, u32 tier, u64 0
//               none     : zeros
// 104  u8[24] reserved
export const R_ID = 0;
export const R_KIND = 48;
export const R_FLAGS = 52;
export const R_PAYLOAD_OFFSET = 56;
export const R_PAYLOAD_BYTES = 64;
export const R_LOCATOR = 72;
export const LOCATOR_BYTES = 32;

// ── Layers ──────────────────────────────────────────────────────────────────

/**
 * The five layers. The numeric id is the on-disk value and the canonical
 * order: a packed container always lists its layers in this order, whatever
 * order the caller supplied them in, so two packs of the same content are
 * byte-identical.
 */
export const LAYER_KINDS = ['model', 'world', 'mind', 'experience', 'code'] as const;
export type LayerKind = (typeof LAYER_KINDS)[number];

export const LAYER_ID: Readonly<Record<LayerKind, number>> = {
  model: 1,
  world: 2,
  mind: 3,
  experience: 4,
  code: 5,
};

/**
 * How a layer's records are located, and therefore how its index is sorted.
 *
 * - `spatial` — an integer lattice coordinate. The index is sorted by Morton key
 *   so neighbouring cells are neighbouring entries, exactly as the Vortex fabric
 *   lays out pages. This is the **only** class whose locator orders storage.
 * - `manifold` — a TERA point, stored as a **search key**. The index is sorted by
 *   id. The TERA value never orders, offsets or buckets storage: that is the
 *   Vortex VM's one rule ("semantic math shapes values and admission, never
 *   addresses"), and this layout keeps it by construction. The optional Odu index
 *   (v1.2) groups *references* to records by cell for search; the records
 *   themselves stay in id order and are fetched by id.
 * - `none` — no locator; sorted by id.
 */
export type LocatorClass = 'spatial' | 'manifold' | 'none';

export const LOCATOR_CLASS: Readonly<Record<LayerKind, LocatorClass>> = {
  model: 'none',
  world: 'spatial',
  mind: 'spatial',
  experience: 'manifold',
  code: 'manifold',
};

/** The layer's live copy may not be mutated by a runtime. */
export const LAYER_FLAG_READ_ONLY = 1 << 0;
/** A runtime mutates a copy-on-write overlay, never the base (WP-43 §5.4). */
export const LAYER_FLAG_COPY_ON_WRITE = 1 << 1;
/** The layer may be discarded whole (a `mind` sandbox). */
export const LAYER_FLAG_DISPOSABLE = 1 << 2;
/** Changes are appended to a sidecar journal and replayable (WP-23 §5). */
export const LAYER_FLAG_JOURNALED = 1 << 3;
/** Reserved: payloads are encrypted under a layer/scope key. Not written in v1.0. */
export const LAYER_FLAG_KEYED = 1 << 4;

/**
 * Default flags per layer. They describe how a *runtime* may treat its live
 * copy of the layer; the packed file itself is always immutable (WP-23 §3.1).
 */
export const DEFAULT_LAYER_FLAGS: Readonly<Record<LayerKind, number>> = {
  model: LAYER_FLAG_READ_ONLY,
  world: LAYER_FLAG_JOURNALED,
  mind: LAYER_FLAG_COPY_ON_WRITE | LAYER_FLAG_DISPOSABLE,
  experience: LAYER_FLAG_JOURNALED,
  code: LAYER_FLAG_READ_ONLY,
};

// ── Records ─────────────────────────────────────────────────────────────────

/** Opaque bytes; the container selects, it never interprets (WP-23 §3.3). */
export const RECORD_KIND_OPAQUE = 0;
/** A complete AMDL file (tensor blobs + manifest + footer), page-aligned in the model layer. */
export const RECORD_KIND_AMDL_EMBEDDED = 1;
/** A reference to an AMDL file kept outside the container (UTF-8 JSON). */
export const RECORD_KIND_AMDL_REF = 2;
/** Reserved for phase 1: a compiled Yare program (Vortex bytecode + pool + constants). */
export const RECORD_KIND_YARE_BYTECODE = 3;
/** A code layer's name index: UTF-8 JSON `[{ module, name, id }]`, so a function is found without parsing every signature. */
export const RECORD_KIND_YARE_INDEX = 4;
/**
 * A code-layer function's embedding (`YAEM`): the 48-byte id of the bytecode record it
 * describes, the embedding model's id, the text that was embedded, and an `f32` vector.
 * The container never interprets it; Yare's search verifies the binding (ADR-020).
 */
export const RECORD_KIND_YARE_EMBEDDING = 5;
/**
 * A model-layer manifest for weights that live **outside** the container (the wrapper guide,
 * `docs/aamt/MODEL-WRAPPER-GUIDE.md`): UTF-8 JSON naming each file, its SHA-384 and the engine that
 * serves it. Observers that do not know this kind skip the record, so adding it is backward
 * compatible; the container still never interprets it.
 */
export const RECORD_KIND_MODEL_MANIFEST = 6;
/**
 * A model-layer Ether encoder (`ETHR`): the five-section int8/Q16.16 checkpoint described in
 * `packages/vortex-vm/src/ether-checkpoint.ts`. The container transports it and never interprets
 * it; the Ether runtime verifies the inner SHA-384 before trusting a weight. Observers that do not
 * know this kind skip the record, so adding it is backward compatible.
 */
export const RECORD_KIND_ETHER_ENCODER = 7;

/** This record is the mirror strand of a duplex (WP-37; `libaamt` `kRecordMirror`). */
export const RECORD_FLAG_MIRROR = 1 << 0;

/** Lens of a manifold locator. */
export const LENS = { none: 0, wisdom: 1, technical: 2 } as const;
export type Lens = keyof typeof LENS;

/**
 * Where a manifold coordinate came from (WP-34 §5). Kept on the record so a
 * reconstructed coordinate can never be silently merged with a declared one.
 */
export const TIER = { none: 0, declared: 1, reconstructed: 2 } as const;
export type Tier = keyof typeof TIER;

// ── Morton keys for spatial layers ──────────────────────────────────────────

/** Bits per axis. Three 21-bit fields fill 63 bits of a `u64`. */
export const MORTON_AXIS_BITS = 21;
/** Same offset as the Vortex fabric (`MORTON_OFFSET`), so signed coordinates stay non-negative. */
export const MORTON_OFFSET = 1 << 20;
export const COORD_MIN = -MORTON_OFFSET;
export const COORD_MAX = MORTON_OFFSET - 1;

/** An integer lattice coordinate. Three integers, nothing else — the fabric's `SpatialCoord`. */
export interface Coord {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

function spread21(n: bigint): bigint {
  let v = n & 0x1fffffn;
  v = (v | (v << 32n)) & 0x1f00000000ffffn;
  v = (v | (v << 16n)) & 0x1f0000ff0000ffn;
  v = (v | (v << 8n)) & 0x100f00f00f00f00fn;
  v = (v | (v << 4n)) & 0x10c30c30c30c30c3n;
  v = (v | (v << 2n)) & 0x1249249249249249n;
  return v;
}

/** Throw unless `c` is three integers inside the 21-bit signed range. */
export function assertCoord(c: Coord): void {
  for (const axis of ['x', 'y', 'z'] as const) {
    const v = c[axis];
    if (!Number.isInteger(v) || v < COORD_MIN || v > COORD_MAX) {
      throw new RangeError(
        `aamt-container: coordinate ${axis}=${String(v)} must be an integer in [${COORD_MIN}, ${COORD_MAX}].`,
      );
    }
  }
}

/**
 * Z-order key of a lattice coordinate: x at bit 0, y at bit 1, z at bit 2 — the
 * fabric's `morton3` convention, widened from 10 to 21 bits per axis.
 */
export function mortonKey(c: Coord): bigint {
  assertCoord(c);
  const off = BigInt(MORTON_OFFSET);
  return (
    spread21(BigInt(c.x) + off) | (spread21(BigInt(c.y) + off) << 1n) | (spread21(BigInt(c.z) + off) << 2n)
  );
}

export const alignUp = (n: number, a: number): number => Math.ceil(n / a) * a;

/** A `u32` as four little-endian bytes, independent of host byte order. */
export function u32le(n: number): Uint8Array {
  const b = new Uint8Array(4);
  new DataView(b.buffer).setUint32(0, n >>> 0, true);
  return b;
}
