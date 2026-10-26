// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * Packing a container.
 *
 * ## Determinism
 *
 * The same content packs to the same bytes, whatever order the caller supplied
 * it in: layers are emitted in {@link LAYER_KINDS} order, records are sorted by
 * their layer's natural key (Morton key then id for spatial layers, id for the
 * rest), every padding byte is zero, and nothing time- or host-dependent is
 * written. That is what makes `fileId` a statement about content rather than
 * about the run that produced it — the property the journal (WP-23 §5) and a
 * world hash both depend on.
 *
 * ## What an id covers
 *
 * `SHA-384(layer context ‖ 0x00 ‖ u32 kind ‖ [locator] ‖ payload)`. The locator
 * is included only for spatial layers, where *where* a thing is is part of what
 * it is: two identical empty bricks at different cells are two records. For
 * manifold layers it is excluded, because the TERA point is a search key —
 * changing what a function is declared to *mean* must not change which function
 * it *is* (Yare design §3).
 *
 * ## In memory, or streamed
 *
 * {@link packContainer} builds the file in memory: simplest, and right for
 * anything that fits. {@link writeContainer} (`./stream`, ADR-017) writes the
 * byte-identical file to a sink with payloads read from re-openable sources, so
 * a multi-gigabyte model layer never has to be resident. Both go through the
 * same planner (`./plan`). Layers flagged {@link LAYER_FLAG_KEYED} are sealed
 * per record (ADR-010).
 */

import { CONTEXT, identify, type Sha384 } from './identity';
import { recordAad, sealPayload, type CryptoPorts } from './crypto';
import {
  checkLayers,
  describeRecord,
  encodeHeader,
  encodeIndex,
  encodeLayerEntry,
  idPrefix,
  isKeyedLayer,
  planLayers,
  oduSection,
  sortRecords,
  treeSection,
  type IndexedRecord,
} from './plan';
import type { TreeOptions } from './tree';
import {
  HEADER_BYTES,
  H_FILE_ID,
  LAYER_ENTRY_BYTES,
  type Coord,
  type LayerKind,
  type Lens,
  type Tier,
} from './layout';

interface RecordBase {
  /** `RECORD_KIND_*`; defaults to opaque. */
  readonly kind?: number;
  /** `RECORD_FLAG_*`. */
  readonly flags?: number;
  readonly payload: Uint8Array;
}

export interface SpatialRecordInput extends RecordBase {
  readonly at: Coord;
}

export interface ManifoldRecordInput extends RecordBase {
  /** A TERA point in `v` coordinates, each axis in `[0,1]`. A search key, never an address. */
  readonly tera?: readonly [number, number, number, number];
  readonly lens?: Lens;
  /** Required whenever `tera` is given: declared or reconstructed (WP-34 §5). */
  readonly tier?: Exclude<Tier, 'none'>;
}

export type PlainRecordInput = RecordBase;

export type LayerInput =
  | {
      readonly kind: 'world' | 'mind';
      readonly flags?: number;
      readonly records: readonly SpatialRecordInput[];
    }
  | {
      readonly kind: 'experience' | 'code';
      readonly flags?: number;
      readonly records: readonly ManifoldRecordInput[];
      /**
       * Write an Odu index section (v1.2, `odu.ts`) when any record carries a TERA
       * point. Default `true`; `false` packs the layer without one.
       */
      readonly oduIndex?: boolean;
      /**
       * Also write an adaptive Meji tree section (v1.3, `tree.ts`, ADR-023): deeper where the
       * data clusters. Off by default; `true` uses the default capacity and depth. A file with a
       * tree is stamped 1.3 and any other file is byte-identical to before.
       */
      readonly treeIndex?: boolean | TreeOptions;
    }
  | { readonly kind: 'model'; readonly flags?: number; readonly records: readonly PlainRecordInput[] };

export interface ContainerInput {
  readonly layers: readonly LayerInput[];
}

interface Prepared extends IndexedRecord {
  readonly payload: Uint8Array;
}

function prepare(
  layer: LayerInput,
  sha384: Sha384,
  seal?: { ports: CryptoPorts; key: Uint8Array },
): Prepared[] {
  const context = CONTEXT[layer.kind];
  const out: Prepared[] = layer.records.map((r) => {
    const meta = describeRecord(layer.kind, r);
    // A keyed layer stores sealed bytes, and its identities cover what is stored (crypto.ts).
    const payload = seal
      ? sealPayload(seal.ports, seal.key, r.payload, recordAad(layer.kind, meta.kind, meta.locator))
      : r.payload;
    return {
      id: identify(sha384, context, [...idPrefix(layer.kind, meta), payload]),
      meta,
      payload,
      bytes: payload.byteLength,
    };
  });
  return sortRecords(out, layer.kind);
}

/** Keys for layers flagged {@link LAYER_FLAG_KEYED}, and the primitives to seal them with. */
export interface SealOptions {
  readonly ports: CryptoPorts;
  readonly keys: Partial<Record<LayerKind, Uint8Array>>;
}

/** Pack `input` into one immutable container. Layers flagged KEYED are sealed with `seal`. */
export function packContainer(input: ContainerInput, sha384: Sha384, seal?: SealOptions): Uint8Array {
  const layers = checkLayers(input.layers, (k) => !!seal?.keys[k]);
  const prepared = layers.map((l) =>
    prepare(l, sha384, isKeyedLayer(l) ? { ports: seal!.ports, key: seal!.keys[l.kind]! } : undefined),
  );
  const { plans, fileBytes, versionMinor } = planLayers(
    layers.map((l, li) => ({
      kind: l.kind,
      lengths: prepared[li]!.map((r) => r.bytes),
      odu: oduSection(
        l,
        prepared[li]!.map((r) => r.meta.locator),
      ),
      tree: treeSection(
        l,
        prepared[li]!.map((r) => r.meta.locator),
      ),
    })),
  );

  const out = new Uint8Array(fileBytes);
  out.set(encodeHeader(layers.length, fileBytes, versionMinor), 0);
  layers.forEach((l, li) => {
    const plan = plans[li]!;
    const records = prepared[li]!;
    const index = encodeIndex(records, plan.offsets);
    out.set(index, plan.indexOffset);
    records.forEach((r, ri) => out.set(r.payload, plan.arenaOffset + plan.offsets[ri]!));
    // The Odu index, if any, follows the arena: a search table over the records, never their address.
    if (plan.odu) out.set(plan.odu, plan.oduOffset);
    if (plan.tree) out.set(plan.tree, plan.treeOffset);
    const layerId = identify(sha384, CONTEXT[l.kind], [
      index,
      out.subarray(plan.arenaOffset, plan.arenaOffset + plan.arenaBytes),
    ]);
    out.set(encodeLayerEntry(l, plan, layerId), HEADER_BYTES + li * LAYER_ENTRY_BYTES);
  });

  // fileId last: the hash covers every byte with the fileId field still zero.
  out.set(identify(sha384, CONTEXT.container, [out]), H_FILE_ID);
  return out;
}
