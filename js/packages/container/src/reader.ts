// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * Opening and verifying a container.
 *
 * ## Two levels of trust, on purpose
 *
 * {@link openContainer} is the hot path: it validates the header and layer table
 * (a few hundred bytes, independent of record count), and checks each record's
 * bounds and alignment **when that record is touched**. That keeps open cost
 * O(layers), which is the point of an mmap-able format — WP-23 measured native
 * open at 0.021 ms on a million-record file, and a format that walked its whole
 * index on open would throw that away.
 *
 * {@link verifyContainer} is the audit path: every record id, every layer id,
 * the file id, index order and payload layout, recomputed. Run it when a file
 * arrives from somewhere you do not control, and in tests. A file that opens
 * but fails verification is corrupt or forged; lookups on it are not trusted.
 *
 * ## Zero copy
 *
 * Every `id` and `payload` handed out is a `subarray` of the caller's bytes.
 * Nothing is copied between the file and the consumer (WP-23 §3.3); a caller who
 * needs to keep a payload past the buffer's lifetime copies it explicitly.
 */

import { CONTEXT, compareIds, idHex, identify, type Sha384 } from './identity';
import { openPayload, recordAad, type CryptoPorts } from './crypto';
import { checkOduSemantics, oduCellsWithin, parseOduSection, type OduSectionShape } from './odu';
import {
  checkTreeSemantics,
  parseTreeSection,
  treeCandidates,
  type TreeSearchStats,
  type TreeSectionShape,
} from './tree';
import type { Transcendentals } from '@aamt/tera';
import {
  HEADER_BYTES,
  H_FILE_BYTES,
  H_FILE_ID,
  H_LAYER_COUNT,
  H_LAYER_TABLE,
  H_VERSION_MAJOR,
  H_VERSION_MINOR,
  ID_BYTES,
  LAYER_ENTRY_BYTES,
  LAYER_FLAG_KEYED,
  LAYER_ID,
  LAYER_KINDS,
  LENS,
  LOCATOR_BYTES,
  LOCATOR_CLASS,
  L_ARENA,
  L_ARENA_BYTES,
  L_COUNT,
  L_FLAGS,
  L_ID,
  L_INDEX,
  L_KIND,
  L_ODU_BYTES,
  L_ODU_OFFSET,
  L_TREE_BYTES,
  L_TREE_OFFSET,
  MAGIC,
  ODU_SINCE_MINOR,
  TREE_SINCE_MINOR,
  PAGE_BYTES,
  ARENA_ALIGN,
  PAYLOAD_ALIGN,
  RECORD_ENTRY_BYTES,
  R_FLAGS,
  R_ID,
  R_KIND,
  R_LOCATOR,
  R_PAYLOAD_BYTES,
  R_PAYLOAD_OFFSET,
  TIER,
  VERSION_MAJOR,
  mortonKey,
  u32le,
  type Coord,
  type LayerKind,
  type Lens,
  type LocatorClass,
  type Tier,
} from './layout';

/** A structural defect: the bytes are not a well-formed v1 container. */
export class ContainerFormatError extends Error {
  constructor(message: string) {
    super(`aamt-container: ${message}`);
    this.name = 'ContainerFormatError';
  }
}

export interface SpatialLocator {
  readonly class: 'spatial';
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly morton: bigint;
}

export interface ManifoldLocator {
  readonly class: 'manifold';
  /** `null` when the record carries no TERA point. Stored as `f32`. */
  readonly tera: readonly [number, number, number, number] | null;
  readonly lens: Lens;
  readonly tier: Tier;
}

export interface RecordView {
  readonly index: number;
  /** 48 bytes; a view into the container. */
  readonly id: Uint8Array;
  readonly kind: number;
  readonly flags: number;
  /** A view into the container — no copy. */
  readonly payload: Uint8Array;
  readonly locator: SpatialLocator | ManifoldLocator | null;
  /** The locator's 32 stored bytes (a view) — what a keyed payload is bound to. */
  readonly locatorBytes: Uint8Array;
}

/**
 * A manifold layer's Odu index (v1.2, ADR-018): which records lie in which of the
 * 256 Odu cells. A search structure — it returns candidates that the caller still
 * checks by distance, and it never changes where or how a record is stored.
 *
 * Opening the container checks the section's structure in O(cells). Before the
 * first lookup, the index is checked in full against the layer (every position in
 * range, every record in the cell its own locator maps to, none missing); a corrupt
 * index is refused with a {@link ContainerFormatError}, never used.
 */
export interface OduIndexView {
  /** Records routed: every record in the layer that carries a TERA point. */
  readonly entryCount: number;
  readonly occupiedCells: number;
  /** The full check, run once and remembered. Defects; empty when the index is sound. */
  check(): readonly string[];
  /** Positions (in the id-sorted record table) of the records in cell `c`, ascending. */
  cell(c: number): number[];
  /**
   * Positions of every record in a cell that a Fisher–Rao ball of `radius` around
   * `v` can touch, ascending: a superset of the records within `radius` (see
   * `oduCellsWithin` for why none is lost). Pass the transcendental port the
   * distance will be measured with; the default is the VM's reference port.
   */
  candidates(v: readonly [number, number, number, number], radius: number, tr?: Transcendentals): Uint32Array;
}

/**
 * A manifold layer's adaptive Meji tree (v1.3, ADR-023): the same promise as {@link OduIndexView}
 * with a cell size that follows the data. It narrows a scan and never decides where a record lives.
 * Opening checks the section's structure in O(nodes); before the first lookup the whole tree is
 * checked against the layer, and a corrupt one is refused with a {@link ContainerFormatError}.
 */
export interface TreeIndexView {
  /** Records routed: every record in the layer that carries a TERA point. */
  readonly entryCount: number;
  readonly nodeCount: number;
  readonly leafCount: number;
  readonly leafCapacity: number;
  readonly maxDepth: number;
  /** The deepest level any node reaches. */
  readonly deepest: number;
  /** The full check, run once and remembered. Defects; empty when the tree is sound. */
  check(): readonly string[];
  /**
   * Positions (in the id-sorted record table) of every record in a leaf that a Fisher–Rao ball of
   * `radius` around `v` can touch, ascending: a superset of the records within `radius`.
   * `stats` says how much of the tree the query had to look at.
   */
  candidates(
    v: readonly [number, number, number, number],
    radius: number,
    tr?: Transcendentals,
  ): Uint32Array;
  /** As {@link TreeIndexView.candidates}, with the work done. */
  search(
    v: readonly [number, number, number, number],
    radius: number,
    tr?: Transcendentals,
  ): { readonly positions: Uint32Array; readonly stats: TreeSearchStats };
}

export interface LayerView {
  readonly kind: LayerKind;
  readonly flags: number;
  readonly locatorClass: LocatorClass;
  /** 48 bytes; a view into the container. */
  readonly id: Uint8Array;
  readonly count: number;
  record(index: number): RecordView;
  /** Find a record by id: binary search for id-sorted layers, a scan for spatial ones. */
  get(id: Uint8Array): RecordView | undefined;
  /** Every record at a lattice cell (spatial layers only), by binary search on the Morton key. */
  at(coord: Coord): RecordView[];
  records(): IterableIterator<RecordView>;
  /** The layer's Odu index, or `null` when it has none (every pre-v1.2 file; any non-manifold layer). */
  readonly odu: OduIndexView | null;
  /** The layer's adaptive tree, or `null` when it has none (every file not stamped v1.3). */
  readonly tree: TreeIndexView | null;
}

export interface Container {
  readonly bytes: Uint8Array;
  readonly versionMajor: number;
  readonly versionMinor: number;
  readonly fileBytes: number;
  /** 48 bytes; a view into the container. */
  readonly fileId: Uint8Array;
  readonly layers: readonly LayerView[];
  layer(kind: LayerKind): LayerView | undefined;
}

const KIND_BY_ID: ReadonlyMap<number, LayerKind> = new Map(LAYER_KINDS.map((k) => [LAYER_ID[k], k]));
const LENS_BY_ID: ReadonlyMap<number, Lens> = new Map(Object.entries(LENS).map(([k, v]) => [v, k as Lens]));
const TIER_BY_ID: ReadonlyMap<number, Tier> = new Map(Object.entries(TIER).map(([k, v]) => [v, k as Tier]));

function u64(dv: DataView, at: number, what: string): number {
  const v = dv.getBigUint64(at, true);
  if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new ContainerFormatError(`${what} ${v} exceeds 2^53.`);
  return Number(v);
}

/** True when `bytes` begins with the container magic. */
export function hasContainerMagic(bytes: Uint8Array): boolean {
  if (bytes.byteLength < MAGIC.byteLength) return false;
  for (let i = 0; i < MAGIC.byteLength; i++) if (bytes[i] !== MAGIC[i]) return false;
  return true;
}

/** Open a container over `bytes` (the whole file). Validates structure only; see {@link verifyContainer}. */
export function openContainer(bytes: Uint8Array): Container {
  if (bytes.byteLength < HEADER_BYTES)
    throw new ContainerFormatError(
      `file is ${bytes.byteLength} bytes, smaller than the ${HEADER_BYTES}-byte header.`,
    );
  if (!hasContainerMagic(bytes))
    throw new ContainerFormatError(
      'bad magic; not a v1 container (use sniffAamt to classify legacy .aamt files).',
    );
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const versionMajor = dv.getUint16(H_VERSION_MAJOR, true);
  const versionMinor = dv.getUint16(H_VERSION_MINOR, true);
  if (versionMajor !== VERSION_MAJOR)
    throw new ContainerFormatError(`unsupported major version ${versionMajor}.`);
  const fileBytes = u64(dv, H_FILE_BYTES, 'fileBytes');
  if (fileBytes !== bytes.byteLength)
    throw new ContainerFormatError(
      `header says ${fileBytes} bytes but ${bytes.byteLength} were supplied (truncated or padded).`,
    );
  const layerCount = dv.getUint32(H_LAYER_COUNT, true);
  const tableAt = u64(dv, H_LAYER_TABLE, 'layerTableOffset');
  if (tableAt < HEADER_BYTES || tableAt % 64 !== 0 || tableAt + layerCount * LAYER_ENTRY_BYTES > fileBytes) {
    throw new ContainerFormatError('layer table is misaligned or outside the file.');
  }

  const layers: LayerView[] = [];
  let prevKindId = 0;
  let prevEnd = tableAt + layerCount * LAYER_ENTRY_BYTES;
  for (let li = 0; li < layerCount; li++) {
    const le = tableAt + li * LAYER_ENTRY_BYTES;
    const kindId = dv.getUint32(le + L_KIND, true);
    const kind = KIND_BY_ID.get(kindId);
    if (!kind) throw new ContainerFormatError(`layer ${li} has unknown kind ${kindId}.`);
    if (kindId <= prevKindId)
      throw new ContainerFormatError(`layer ${kind} is duplicated or out of canonical order.`);
    prevKindId = kindId;
    const flags = dv.getUint32(le + L_FLAGS, true);
    const indexOffset = u64(dv, le + L_INDEX, `${kind} indexOffset`);
    const count = u64(dv, le + L_COUNT, `${kind} recordCount`);
    const arenaOffset = u64(dv, le + L_ARENA, `${kind} arenaOffset`);
    const arenaBytes = u64(dv, le + L_ARENA_BYTES, `${kind} arenaBytes`);
    const indexEnd = indexOffset + count * RECORD_ENTRY_BYTES;
    if (indexOffset % 64 !== 0 || indexOffset < prevEnd || indexEnd > arenaOffset)
      throw new ContainerFormatError(`layer ${kind} index is misaligned, overlaps, or runs into its arena.`);
    if (arenaOffset % ARENA_ALIGN[kind] !== 0 || arenaOffset + arenaBytes > fileBytes)
      throw new ContainerFormatError(
        `layer ${kind} arena is misaligned for its kind (ARENA_ALIGN) or runs past the file.`,
      );
    prevEnd = arenaOffset + arenaBytes;
    let odu: OduSectionShape | null = null;
    // Before v1.2 these fields were reserved: they are not read, so old files open unchanged.
    if (versionMinor >= ODU_SINCE_MINOR) {
      const oduOffset = u64(dv, le + L_ODU_OFFSET, `${kind} oduOffset`);
      const oduBytes = u64(dv, le + L_ODU_BYTES, `${kind} oduBytes`);
      if (oduOffset !== 0 || oduBytes !== 0) {
        if (LOCATOR_CLASS[kind] !== 'manifold')
          throw new ContainerFormatError(`layer ${kind} is not a manifold layer but claims an Odu index.`);
        if (oduOffset % 64 !== 0 || oduOffset < prevEnd || oduOffset + oduBytes > fileBytes)
          throw new ContainerFormatError(
            `layer ${kind} Odu index is misaligned, overlaps, or runs past the file.`,
          );
        const parsed = parseOduSection(bytes.subarray(oduOffset, oduOffset + oduBytes), count);
        if (!parsed.ok) throw new ContainerFormatError(`layer ${kind}: ${parsed.problem}`);
        odu = parsed.shape;
        prevEnd = oduOffset + oduBytes;
      }
    }
    let tree: TreeSectionShape | null = null;
    // Before v1.3 these fields were reserved: a v1.2 file's are not read, so it opens unchanged.
    if (versionMinor >= TREE_SINCE_MINOR) {
      const treeOffset = u64(dv, le + L_TREE_OFFSET, `${kind} treeOffset`);
      const treeBytes = u64(dv, le + L_TREE_BYTES, `${kind} treeBytes`);
      if (treeOffset !== 0 || treeBytes !== 0) {
        if (LOCATOR_CLASS[kind] !== 'manifold')
          throw new ContainerFormatError(`layer ${kind} is not a manifold layer but claims a tree index.`);
        if (treeOffset % 64 !== 0 || treeOffset < prevEnd || treeOffset + treeBytes > fileBytes)
          throw new ContainerFormatError(
            `layer ${kind} tree index is misaligned, overlaps, or runs past the file.`,
          );
        const parsed = parseTreeSection(bytes.subarray(treeOffset, treeOffset + treeBytes), count);
        if (!parsed.ok) throw new ContainerFormatError(`layer ${kind}: ${parsed.problem}`);
        tree = parsed.shape;
        prevEnd = treeOffset + treeBytes;
      }
    }
    layers.push(
      makeLayer(bytes, dv, {
        kind,
        flags,
        id: bytes.subarray(le + L_ID, le + L_ID + ID_BYTES),
        indexOffset,
        count,
        arenaOffset,
        arenaBytes,
        odu,
        tree,
      }),
    );
  }

  return {
    bytes,
    versionMajor,
    versionMinor,
    fileBytes,
    fileId: bytes.subarray(H_FILE_ID, H_FILE_ID + ID_BYTES),
    layers,
    layer: (kind) => layers.find((l) => l.kind === kind),
  };
}

interface LayerInfo {
  readonly kind: LayerKind;
  readonly flags: number;
  readonly id: Uint8Array;
  readonly indexOffset: number;
  readonly count: number;
  readonly arenaOffset: number;
  readonly arenaBytes: number;
  readonly odu: OduSectionShape | null;
  readonly tree: TreeSectionShape | null;
}

function makeLayer(bytes: Uint8Array, dv: DataView, info: LayerInfo): LayerView {
  const locatorClass = LOCATOR_CLASS[info.kind];
  const align = info.kind === 'model' ? PAGE_BYTES : PAYLOAD_ALIGN;
  const entry = (i: number): number => info.indexOffset + i * RECORD_ENTRY_BYTES;
  const idAt = (i: number): Uint8Array => bytes.subarray(entry(i) + R_ID, entry(i) + R_ID + ID_BYTES);
  const mortonAt = (i: number): bigint => dv.getBigUint64(entry(i) + R_LOCATOR + 16, true);

  const record = (i: number): RecordView => {
    if (!Number.isInteger(i) || i < 0 || i >= info.count)
      throw new RangeError(`aamt-container: record ${i} out of range for ${info.kind} (${info.count}).`);
    const e = entry(i);
    const offset = u64(dv, e + R_PAYLOAD_OFFSET, 'payloadOffset');
    const length = u64(dv, e + R_PAYLOAD_BYTES, 'payloadBytes');
    if (offset % align !== 0 || offset + length > info.arenaBytes)
      throw new ContainerFormatError(`${info.kind} record ${i} payload is misaligned or outside its arena.`);
    const start = info.arenaOffset + offset;
    return {
      index: i,
      id: idAt(i),
      kind: dv.getUint32(e + R_KIND, true),
      flags: dv.getUint32(e + R_FLAGS, true),
      payload: bytes.subarray(start, start + length),
      locator: decodeLocator(dv, e + R_LOCATOR, locatorClass, info.kind, i),
      locatorBytes: bytes.subarray(e + R_LOCATOR, e + R_LOCATOR + LOCATOR_BYTES),
    };
  };

  return {
    kind: info.kind,
    flags: info.flags,
    locatorClass,
    id: info.id,
    count: info.count,
    record,
    get(id) {
      if (locatorClass === 'spatial') {
        for (let i = 0; i < info.count; i++) if (compareIds(idAt(i), id) === 0) return record(i);
        return undefined;
      }
      let lo = 0;
      let hi = info.count - 1;
      while (lo <= hi) {
        const mid = (lo + hi) >>> 1;
        const c = compareIds(idAt(mid), id);
        if (c === 0) return record(mid);
        if (c < 0) lo = mid + 1;
        else hi = mid - 1;
      }
      return undefined;
    },
    at(coord) {
      if (locatorClass !== 'spatial')
        throw new TypeError(
          `aamt-container: the ${info.kind} layer is not spatial; at() needs a world or mind layer.`,
        );
      const key = mortonKey(coord);
      let lo = 0;
      let hi = info.count;
      while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (mortonAt(mid) < key) lo = mid + 1;
        else hi = mid;
      }
      const out: RecordView[] = [];
      for (let i = lo; i < info.count && mortonAt(i) === key; i++) out.push(record(i));
      return out;
    },
    *records() {
      for (let i = 0; i < info.count; i++) yield record(i);
    },
    odu: info.odu ? makeOduView(info, dv, entry) : null,
    tree: info.tree ? makeTreeView(info, dv, entry) : null,
  };
}

function makeTreeView(info: LayerInfo, dv: DataView, entry: (i: number) => number): TreeIndexView {
  const shape = info.tree!;
  let problems: readonly string[] | undefined;
  const check = (): readonly string[] =>
    (problems ??= checkTreeSemantics(shape, info.count, dv, (p) => entry(p) + R_LOCATOR).map(
      (m) => `${info.kind}: ${m}`,
    ));
  const sound = (): void => {
    const p = check();
    if (p.length > 0) throw new ContainerFormatError(`refusing a corrupt tree index — ${p[0]}`);
  };
  return {
    entryCount: shape.entryCount,
    nodeCount: shape.nodeCount,
    leafCount: shape.leafCount,
    leafCapacity: shape.leafCapacity,
    maxDepth: shape.maxDepth,
    deepest: shape.deepest,
    check,
    candidates(v, radius, tr) {
      sound();
      return treeCandidates(shape, v, radius, tr).positions;
    },
    search(v, radius, tr) {
      sound();
      return treeCandidates(shape, v, radius, tr);
    },
  };
}

function makeOduView(info: LayerInfo, dv: DataView, entry: (i: number) => number): OduIndexView {
  const shape = info.odu!;
  let problems: readonly string[] | undefined;
  const check = (): readonly string[] =>
    (problems ??= checkOduSemantics(shape, info.count, dv, (p) => entry(p) + R_LOCATOR).map(
      (m) => `${info.kind}: ${m}`,
    ));
  const sound = (): void => {
    const p = check();
    if (p.length > 0) throw new ContainerFormatError(`refusing a corrupt Odu index — ${p[0]}`);
  };
  const cell = (c: number): number[] => {
    if (!Number.isInteger(c) || c < 0 || c >= shape.cellStart.length - 1)
      throw new RangeError(`aamt-container: Odu cell ${c} is not in 0..255.`);
    const out: number[] = [];
    for (let i = shape.cellStart[c]!; i < shape.cellStart[c + 1]!; i++) out.push(shape.positions(i));
    return out;
  };
  return {
    entryCount: shape.entryCount,
    occupiedCells: shape.occupied,
    check,
    cell(c) {
      sound();
      return cell(c);
    },
    candidates(v, radius, tr) {
      sound();
      const cells = oduCellsWithin(v, radius, tr);
      let n = 0;
      for (const c of cells) n += shape.cellStart[c + 1]! - shape.cellStart[c]!;
      const out = new Uint32Array(n);
      let k = 0;
      for (const c of cells)
        for (let i = shape.cellStart[c]!; i < shape.cellStart[c + 1]!; i++) out[k++] = shape.positions(i);
      // Record order (a typed-array sort is numeric), so candidates are visited in the order a scan visits them.
      return out.sort();
    },
  };
}

function decodeLocator(
  dv: DataView,
  at: number,
  klass: LocatorClass,
  kind: LayerKind,
  i: number,
): SpatialLocator | ManifoldLocator | null {
  if (klass === 'spatial') {
    return {
      class: 'spatial',
      x: dv.getInt32(at, true),
      y: dv.getInt32(at + 4, true),
      z: dv.getInt32(at + 8, true),
      morton: dv.getBigUint64(at + 16, true),
    };
  }
  if (klass === 'manifold') {
    const lens = LENS_BY_ID.get(dv.getUint32(at + 16, true));
    const tier = TIER_BY_ID.get(dv.getUint32(at + 20, true));
    if (!lens || !tier) throw new ContainerFormatError(`${kind} record ${i} has an unknown lens or tier.`);
    const tera =
      tier === 'none'
        ? null
        : ([0, 1, 2, 3].map((k) => dv.getFloat32(at + k * 4, true)) as [number, number, number, number]);
    return { class: 'manifold', tera, lens, tier };
  }
  return null;
}

export interface VerifyResult {
  readonly ok: boolean;
  /** Human-readable defects, each naming the layer and record it concerns. Empty when `ok`. */
  readonly problems: readonly string[];
}

/**
 * Recompute every identity and check every invariant. Never throws on a bad
 * file: defects come back in `problems`, so a caller can report all of them.
 */
export function verifyContainer(bytes: Uint8Array, sha384: Sha384): VerifyResult {
  const problems: string[] = [];
  let c: Container;
  try {
    c = openContainer(bytes);
  } catch (e) {
    return { ok: false, problems: [(e as Error).message] };
  }

  const zeroed = new Uint8Array(ID_BYTES);
  const fileId = identify(sha384, CONTEXT.container, [
    bytes.subarray(0, H_FILE_ID),
    zeroed,
    bytes.subarray(H_FILE_ID + ID_BYTES),
  ]);
  if (compareIds(fileId, c.fileId) !== 0)
    problems.push(`fileId mismatch: stored ${idHex(c.fileId)}, computed ${idHex(fileId)}.`);

  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tableAt = Number(dv.getBigUint64(H_LAYER_TABLE, true));
  for (const [li, layer] of c.layers.entries()) {
    const le = tableAt + li * LAYER_ENTRY_BYTES;
    const indexOffset = Number(dv.getBigUint64(le + L_INDEX, true));
    const arenaOffset = Number(dv.getBigUint64(le + L_ARENA, true));
    const arenaBytes = Number(dv.getBigUint64(le + L_ARENA_BYTES, true));
    const layerId = identify(sha384, CONTEXT[layer.kind], [
      bytes.subarray(indexOffset, indexOffset + layer.count * RECORD_ENTRY_BYTES),
      bytes.subarray(arenaOffset, arenaOffset + arenaBytes),
    ]);
    if (compareIds(layerId, layer.id) !== 0) problems.push(`${layer.kind}: layerId mismatch.`);
    if (layer.odu) problems.push(...layer.odu.check());
    if (layer.tree) problems.push(...layer.tree.check());

    let prev: RecordView | undefined;
    let prevEnd = 0;
    for (let i = 0; i < layer.count; i++) {
      let r: RecordView;
      try {
        r = layer.record(i);
      } catch (e) {
        problems.push((e as Error).message);
        continue;
      }
      const parts: Uint8Array[] = [u32le(r.kind)];
      if (layer.locatorClass === 'spatial')
        parts.push(
          bytes.subarray(
            indexOffset + i * RECORD_ENTRY_BYTES + R_LOCATOR,
            indexOffset + i * RECORD_ENTRY_BYTES + R_LOCATOR + LOCATOR_BYTES,
          ),
        );
      parts.push(r.payload);
      if (compareIds(identify(sha384, CONTEXT[layer.kind], parts), r.id) !== 0)
        problems.push(
          `${layer.kind} record ${i}: id does not match its content (${idHex(r.id).slice(0, 16)}…).`,
        );

      if (r.locator?.class === 'spatial') {
        try {
          if (mortonKey(r.locator) !== r.locator.morton)
            problems.push(`${layer.kind} record ${i}: stored Morton key does not match its coordinate.`);
        } catch (e) {
          problems.push(`${layer.kind} record ${i}: ${(e as Error).message}`);
        }
      }
      if (prev) {
        const order =
          prev.locator?.class === 'spatial' &&
          r.locator?.class === 'spatial' &&
          prev.locator.morton !== r.locator.morton
            ? prev.locator.morton < r.locator.morton
              ? -1
              : 1
            : compareIds(prev.id, r.id);
        if (order >= 0) problems.push(`${layer.kind} record ${i}: index out of order or duplicated.`);
      }
      const start = r.payload.byteOffset - bytes.byteOffset - arenaOffset;
      if (start < prevEnd) problems.push(`${layer.kind} record ${i}: payload overlaps the previous one.`);
      prevEnd = start + r.payload.byteLength;
      prev = r;
    }
  }
  return { ok: problems.length === 0, problems };
}

/** Is this layer's payload sealed (LAYER_FLAG_KEYED)? */
export const isKeyed = (layer: LayerView): boolean => (layer.flags & LAYER_FLAG_KEYED) !== 0;

/**
 * The plaintext of a record in a keyed layer, or `null` if the key is wrong or
 * the record was altered or moved (its sealed payload is bound to its layer, kind
 * and locator). For an unkeyed layer, the payload as stored.
 */
export function unsealRecord(
  layer: LayerView,
  record: RecordView,
  key: Uint8Array,
  ports: Pick<CryptoPorts, 'aead'>,
): Uint8Array | null {
  if (!isKeyed(layer)) return record.payload;
  return openPayload(ports, key, record.payload, recordAad(layer.kind, record.kind, record.locatorBytes));
}
