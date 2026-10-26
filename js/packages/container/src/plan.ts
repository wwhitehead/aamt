// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * What both writers share: how a record input is interpreted, how a layer's
 * records are ordered, where everything goes, and how the fixed structures are
 * encoded. {@link packContainer} and {@link writeContainer} call the same
 * functions here, so "byte-identical" is a property of one planner fed the same
 * ids and lengths, not of two planners that happen to agree.
 *
 * Internal: not re-exported from the package entry.
 */

import { compareIds } from './identity';
import { buildOduSection } from './odu';
import { buildTreeSection, type TreeOptions } from './tree';
import {
  DEFAULT_LAYER_FLAGS,
  HEADER_BYTES,
  H_FILE_BYTES,
  H_LAYER_COUNT,
  H_LAYER_TABLE,
  H_VERSION_MAJOR,
  H_VERSION_MINOR,
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
  PAGE_BYTES,
  ARENA_ALIGN,
  PAYLOAD_ALIGN,
  RECORD_ENTRY_BYTES,
  RECORD_KIND_OPAQUE,
  R_FLAGS,
  R_ID,
  R_KIND,
  R_LOCATOR,
  R_PAYLOAD_BYTES,
  R_PAYLOAD_OFFSET,
  TIER,
  VERSION_MAJOR,
  VERSION_MINOR,
  VERSION_MINOR_TREE,
  alignUp,
  mortonKey,
  u32le,
  type Coord,
  type LayerKind,
  type Lens,
  type Tier,
} from './layout';

export function fail(message: string): never {
  throw new Error(`aamt-container: ${message}`);
}

/** The fields of a record input other than its payload. */
export interface RecordFields {
  readonly kind?: number;
  readonly flags?: number;
  readonly at?: Coord;
  readonly tera?: readonly [number, number, number, number];
  readonly lens?: Lens;
  readonly tier?: Exclude<Tier, 'none'>;
}

/** A record's metadata, validated: everything but its payload and id. */
export interface RecordMeta {
  readonly kind: number;
  readonly flags: number;
  readonly locator: Uint8Array;
  readonly morton: bigint;
}

/**
 * What a record id covers before the payload: `u32 kind ‖ [locator, spatial only]`.
 * Computed on use rather than stored, because the streaming writer keeps a
 * {@link RecordMeta} per record for the whole write.
 */
export function idPrefix(layer: LayerKind, meta: RecordMeta): Uint8Array[] {
  return LOCATOR_CLASS[layer] === 'spatial' ? [u32le(meta.kind), meta.locator] : [u32le(meta.kind)];
}

function spatialLocator(at: Coord): { locator: Uint8Array; morton: bigint } {
  const morton = mortonKey(at);
  const locator = new Uint8Array(LOCATOR_BYTES);
  const dv = new DataView(locator.buffer);
  dv.setInt32(0, at.x, true);
  dv.setInt32(4, at.y, true);
  dv.setInt32(8, at.z, true);
  dv.setBigUint64(16, morton, true);
  return { locator, morton };
}

function manifoldLocator(r: RecordFields, layer: LayerKind): Uint8Array {
  const locator = new Uint8Array(LOCATOR_BYTES);
  if (r.tera === undefined) {
    if (r.tier !== undefined || r.lens !== undefined)
      fail(`${layer} record has a lens or tier but no tera point.`);
    return locator;
  }
  if (r.tera.length !== 4 || r.tera.some((v) => !Number.isFinite(v) || v < 0 || v > 1)) {
    fail(`${layer} record tera must be four finite values in [0,1] (v coordinates).`);
  }
  if (r.tier === undefined)
    fail(
      `${layer} record has a tera point but no tier; say whether it was declared or reconstructed (WP-34 §5).`,
    );
  const dv = new DataView(locator.buffer);
  r.tera.forEach((v, i) => dv.setFloat32(i * 4, v, true));
  dv.setUint32(16, LENS[r.lens ?? 'none'], true);
  dv.setUint32(20, TIER[r.tier], true);
  return locator;
}

/** Validate a record's kind and locator for `layer`. */
export function describeRecord(layer: LayerKind, r: RecordFields): RecordMeta {
  const klass = LOCATOR_CLASS[layer];
  const kind = r.kind ?? RECORD_KIND_OPAQUE;
  if (!Number.isInteger(kind) || kind < 0 || kind > 0xffffffff)
    fail(`record kind ${String(kind)} is not a u32.`);
  let locator: Uint8Array = new Uint8Array(LOCATOR_BYTES);
  let morton = 0n;
  if (klass === 'spatial') {
    ({ locator, morton } = spatialLocator(r.at as Coord));
  } else if (klass === 'manifold') {
    locator = manifoldLocator(r, layer);
  }
  return { kind, flags: (r.flags ?? 0) >>> 0, locator, morton };
}

export const isKeyedLayer = (l: { readonly flags?: number }): boolean =>
  ((l.flags ?? 0) & LAYER_FLAG_KEYED) !== 0;

/** Validate the layer list and return it in canonical order. */
export function checkLayers<L extends { readonly kind: LayerKind; readonly flags?: number }>(
  layers: readonly L[],
  hasKey: (kind: LayerKind) => boolean,
): L[] {
  const seen = new Set<LayerKind>();
  for (const l of layers) {
    if (!LAYER_KINDS.includes(l.kind)) fail(`unknown layer kind ${String(l.kind)}.`);
    if (seen.has(l.kind))
      fail(`layer ${l.kind} appears twice; a container holds at most one layer of each kind.`);
    seen.add(l.kind);
    if (isKeyedLayer(l) && !hasKey(l.kind)) fail(`layer ${l.kind} is KEYED but no key was supplied for it.`);
  }
  return [...layers].sort((a, b) => LAYER_ID[a.kind] - LAYER_ID[b.kind]);
}

/** Sort a layer's records into index order (Morton then id, or id) and refuse duplicates. */
export function sortRecords<T extends { readonly id: Uint8Array; readonly meta: RecordMeta }>(
  records: T[],
  layer: LayerKind,
): T[] {
  const spatial = LOCATOR_CLASS[layer] === 'spatial';
  records.sort(
    spatial
      ? (a, b) =>
          a.meta.morton < b.meta.morton ? -1 : a.meta.morton > b.meta.morton ? 1 : compareIds(a.id, b.id)
      : (a, b) => compareIds(a.id, b.id),
  );
  for (let i = 1; i < records.length; i++) {
    if (compareIds(records[i - 1]!.id, records[i]!.id) === 0)
      fail(`duplicate record in the ${layer} layer (same kind, payload${spatial ? ' and cell' : ''}).`);
  }
  return records;
}

export interface LayerPlan {
  readonly indexOffset: number;
  readonly arenaOffset: number;
  readonly arenaBytes: number;
  /** Each payload's offset, relative to the arena, in index order. */
  readonly offsets: readonly number[];
  /** The layer's Odu index section (v1.2, `odu.ts`), or null; it follows the arena. */
  readonly odu: Uint8Array | null;
  readonly oduOffset: number;
  /** The layer's adaptive tree section (v1.3, `tree.ts`), or null; it follows the Odu section, if any. */
  readonly tree: Uint8Array | null;
  readonly treeOffset: number;
}

/**
 * The Odu index section for a layer whose locators are `locators` in index order:
 * manifold layers only, unless the input opts out with `oduIndex: false`. Null when
 * no record carries a TERA point. A search table over the records, never their address.
 */
export function oduSection(
  layer: { readonly kind: LayerKind; readonly oduIndex?: boolean },
  locators: readonly Uint8Array[],
): Uint8Array | null {
  return LOCATOR_CLASS[layer.kind] === 'manifold' && layer.oduIndex !== false
    ? buildOduSection(locators)
    : null;
}

/**
 * The adaptive tree section for a layer (v1.3, `tree.ts`): manifold layers only, and only when the
 * input asks with `treeIndex`. Null when not asked, or when no record carries a TERA point.
 */
export function treeSection(
  layer: { readonly kind: LayerKind; readonly treeIndex?: boolean | TreeOptions },
  locators: readonly Uint8Array[],
): Uint8Array | null {
  if (!layer.treeIndex) return null;
  if (LOCATOR_CLASS[layer.kind] !== 'manifold')
    fail(`a ${layer.kind} layer has no TERA locators; a tree index applies to manifold layers only.`);
  return buildTreeSection(locators, layer.treeIndex === true ? undefined : layer.treeIndex);
}

/** Where every structure and payload goes, given each layer's payload lengths in index order. */
export function planLayers(
  layers: readonly {
    readonly kind: LayerKind;
    readonly lengths: readonly number[];
    readonly odu?: Uint8Array | null;
    readonly tree?: Uint8Array | null;
  }[],
): {
  plans: LayerPlan[];
  fileBytes: number;
  /** 1.3 when any layer holds a tree, else the unchanged {@link VERSION_MINOR}. */
  versionMinor: number;
} {
  let cursor = HEADER_BYTES + layers.length * LAYER_ENTRY_BYTES;
  const plans = layers.map((l) => {
    const indexOffset = alignUp(cursor, 64);
    const arenaOffset = alignUp(indexOffset + l.lengths.length * RECORD_ENTRY_BYTES, ARENA_ALIGN[l.kind]);
    const align = l.kind === 'model' ? PAGE_BYTES : PAYLOAD_ALIGN;
    let at = 0;
    const offsets = l.lengths.map((n) => {
      const o = alignUp(at, align);
      at = o + n;
      return o;
    });
    const odu = l.odu ?? null;
    const oduOffset = odu ? alignUp(arenaOffset + at, 64) : 0;
    const afterOdu = odu ? oduOffset + odu.byteLength : arenaOffset + at;
    const tree = l.tree ?? null;
    const treeOffset = tree ? alignUp(afterOdu, 64) : 0;
    cursor = tree ? treeOffset + tree.byteLength : afterOdu;
    return { indexOffset, arenaOffset, arenaBytes: at, offsets, odu, oduOffset, tree, treeOffset };
  });
  if (!Number.isSafeInteger(cursor)) fail('container exceeds 2^53 bytes.');
  return { plans, fileBytes: cursor, versionMinor: plans.some((p) => p.tree) ? VERSION_MINOR_TREE : VERSION_MINOR };
}

/** The 128-byte header, with the fileId field zero. */
export function encodeHeader(layerCount: number, fileBytes: number, versionMinor = VERSION_MINOR): Uint8Array {
  const out = new Uint8Array(HEADER_BYTES);
  const dv = new DataView(out.buffer);
  out.set(MAGIC, 0);
  dv.setUint16(H_VERSION_MAJOR, VERSION_MAJOR, true);
  dv.setUint16(H_VERSION_MINOR, versionMinor, true);
  dv.setUint32(H_LAYER_COUNT, layerCount, true);
  dv.setBigUint64(H_FILE_BYTES, BigInt(fileBytes), true);
  dv.setBigUint64(H_LAYER_TABLE, BigInt(HEADER_BYTES), true);
  return out;
}

export interface IndexedRecord {
  readonly id: Uint8Array;
  readonly meta: RecordMeta;
  /** Stored payload length (sealed length in a keyed layer). */
  readonly bytes: number;
}

/** Index entries `[from, to)` of a layer, in index order. */
export function encodeIndex(
  records: readonly IndexedRecord[],
  offsets: readonly number[],
  from = 0,
  to = records.length,
): Uint8Array {
  const out = new Uint8Array((to - from) * RECORD_ENTRY_BYTES);
  const dv = new DataView(out.buffer);
  for (let ri = from; ri < to; ri++) {
    const r = records[ri]!;
    const e = (ri - from) * RECORD_ENTRY_BYTES;
    out.set(r.id, e + R_ID);
    dv.setUint32(e + R_KIND, r.meta.kind, true);
    dv.setUint32(e + R_FLAGS, r.meta.flags, true);
    dv.setBigUint64(e + R_PAYLOAD_OFFSET, BigInt(offsets[ri]!), true);
    dv.setBigUint64(e + R_PAYLOAD_BYTES, BigInt(r.bytes), true);
    out.set(r.meta.locator, e + R_LOCATOR);
  }
  return out;
}

/** One 128-byte layer-table entry. */
export function encodeLayerEntry(
  layer: { readonly kind: LayerKind; readonly flags?: number },
  plan: LayerPlan,
  layerId: Uint8Array,
): Uint8Array {
  const out = new Uint8Array(LAYER_ENTRY_BYTES);
  const dv = new DataView(out.buffer);
  dv.setUint32(L_KIND, LAYER_ID[layer.kind], true);
  dv.setUint32(L_FLAGS, (layer.flags ?? DEFAULT_LAYER_FLAGS[layer.kind]) >>> 0, true);
  dv.setBigUint64(L_INDEX, BigInt(plan.indexOffset), true);
  dv.setBigUint64(L_COUNT, BigInt(plan.offsets.length), true);
  dv.setBigUint64(L_ARENA, BigInt(plan.arenaOffset), true);
  dv.setBigUint64(L_ARENA_BYTES, BigInt(plan.arenaBytes), true);
  if (plan.odu) {
    dv.setBigUint64(L_ODU_OFFSET, BigInt(plan.oduOffset), true);
    dv.setBigUint64(L_ODU_BYTES, BigInt(plan.odu.byteLength), true);
  }
  if (plan.tree) {
    dv.setBigUint64(L_TREE_OFFSET, BigInt(plan.treeOffset), true);
    dv.setBigUint64(L_TREE_BYTES, BigInt(plan.tree.byteLength), true);
  }
  out.set(layerId, L_ID);
  return out;
}
