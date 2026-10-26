// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * The streaming writer (ADR-017): {@link packContainer}'s bytes, without
 * {@link packContainer}'s memory.
 *
 * ## Why one pass cannot work
 *
 * The format puts every fact about a payload *before* the payload, and every
 * one of those facts depends on payload bytes:
 *
 * ```
 *  header      fileId    = SHA-384 over the whole file      ← needs every byte
 *  layer table layerId   = SHA-384(index ‖ arena)            ← needs the arena
 *  index       sorted by id; offsets follow that order       ← needs every record id
 *  arena       payloads, in index order
 * ```
 *
 * A payload's position is its rank among the layer's ids, so nothing can be
 * placed until every record of the layer has been hashed. SHA-384 is sequential
 * and the layer table precedes the arenas it describes, so the file hash cannot
 * start until every layer id is known. The format is not changed to fix this
 * (readers and files on disk stay as they are); the writer pays in passes:
 *
 * | pass | reads sources in | computes | writes |
 * |---|---|---|---|
 * | 1 | input order | record ids, stored lengths → sort, plan | nothing |
 * | 2 | index order | layer ids (index ‖ arena) | nothing |
 * | 3 | index order | file id (fileId field zero) | the whole file, in order |
 * | — | — | — | patch the 48-byte fileId at offset 32 |
 *
 * A sink without `patch` (a pipe, an upload) gets a fourth pass instead: pass 3
 * only hashes, and pass 4 writes the file with the fileId already in place. The
 * sink then sees strictly sequential appends.
 *
 * Every SHA-384 the in-memory writer computes is computed here too — once per
 * record, once per layer, once per file — so the hashing work is the same; the
 * extra cost is re-reading sources. With `checkSources` (the default) passes 2–4
 * also recompute each record id, so a source that yields different bytes on a
 * later read is refused instead of producing a file whose index lies.
 *
 * ## Memory
 *
 * O(1) in payload bytes: one source chunk, one {@link bufferBytes} write buffer,
 * index entries encoded {@link INDEX_CHUNK_ENTRIES} at a time. O(records) in
 * metadata, because the index must be sorted before anything is written.
 *
 * ## Keyed layers
 *
 * Supported. A sealed payload is `nonce ‖ AES-GCM(plaintext) ‖ tag`, and the nonce
 * is derived from SHA-384 of the plaintext, so a keyed record is read twice in
 * pass 1 (hash, then seal-and-hash) and its nonce kept. With an
 * {@link AeadStream} port (`nodeCrypto` has one) sealing streams; without one
 * (WebCrypto has no streaming GCM) each keyed record is sealed whole, so memory
 * is bounded by the largest keyed record rather than the file.
 *
 * ## Sources and sinks
 *
 * A payload is a `Uint8Array` or a {@link ChunkSource}: a function that opens a
 * fresh read of the same bytes each time it is called. The writer has consumed
 * a chunk before it asks for the next, so a source may reuse its buffer. The
 * writer likewise reuses its buffers once a sink's `write` has returned (or its
 * promise resolved); a sink that keeps bytes must copy them.
 */

import { CONTEXT, identifying, type Sha384Incremental, type Sha384State } from './identity';
import { recordAad, sealPayload, syntheticNonce, type CryptoPorts } from './crypto';
import {
  checkLayers,
  describeRecord,
  encodeHeader,
  encodeIndex,
  encodeLayerEntry,
  fail,
  idPrefix,
  isKeyedLayer,
  planLayers,
  oduSection,
  sortRecords,
  treeSection,
  type IndexedRecord,
  type LayerPlan,
  type RecordMeta,
} from './plan';
import { H_FILE_ID, LAYER_ENTRY_BYTES, PAGE_BYTES, type LayerKind } from './layout';
import type { TreeOptions } from './tree';
import type { ManifoldRecordInput, PlainRecordInput, SealOptions, SpatialRecordInput } from './writer';

/** Opens a fresh read of a payload. Every call must yield the same bytes. */
export type ChunkSource = () => AsyncIterable<Uint8Array> | Iterable<Uint8Array>;

/** A payload held in memory, or one read on demand. */
export type PayloadSource = Uint8Array | ChunkSource;

type Streamed<T> = Omit<T, 'payload'> & { readonly payload: PayloadSource };

export type StreamLayerInput =
  | {
      readonly kind: 'world' | 'mind';
      readonly flags?: number;
      readonly records: readonly Streamed<SpatialRecordInput>[];
    }
  | {
      readonly kind: 'experience' | 'code';
      readonly flags?: number;
      readonly records: readonly Streamed<ManifoldRecordInput>[];
      /** As for {@link packContainer}: write an Odu index section unless `false`. */
      readonly oduIndex?: boolean;
      /** As for {@link packContainer}: also write an adaptive tree section (v1.3). */
      readonly treeIndex?: boolean | TreeOptions;
    }
  | {
      readonly kind: 'model';
      readonly flags?: number;
      readonly records: readonly Streamed<PlainRecordInput>[];
    };

export interface StreamContainerInput {
  readonly layers: readonly StreamLayerInput[];
}

/** Where the file goes. `write` appends; `patch` overwrites bytes already written. */
export interface ContainerSink {
  write(bytes: Uint8Array): void | Promise<void>;
  /** Optional. Without it the writer makes one more pass so it never has to go back. */
  patch?(offset: number, bytes: Uint8Array): void | Promise<void>;
}

export interface StreamWriteOptions {
  /** Keys and primitives for layers flagged KEYED, as for {@link packContainer}. */
  readonly seal?: SealOptions;
  /** Recompute every record id on each later pass and refuse a source whose bytes changed. Default true. */
  readonly checkSources?: boolean;
  /** Coalesce writes smaller than this many bytes into one sink write. Default 1 MiB; 0 passes every write through. */
  readonly bufferBytes?: number;
}

export interface StreamWriteResult {
  readonly fileBytes: number;
  readonly fileId: Uint8Array;
  /** How many times each unkeyed source was read: 3 with a patching sink, 4 without. Keyed sources once more. */
  readonly passes: number;
}

/** Index entries encoded per write: 8,192 × 128 B = 1 MiB. */
export const INDEX_CHUNK_ENTRIES = 8192;
const DEFAULT_BUFFER_BYTES = 1 << 20;
/** The largest gap the planner can leave is under one page (model payload or arena alignment). */
const ZEROS = new Uint8Array(PAGE_BYTES);

interface Planned extends IndexedRecord {
  readonly source: PayloadSource;
  /** Plaintext length; equals `bytes` in an unkeyed layer. */
  readonly plainBytes: number;
  /** Keyed records only: the synthetic nonce, kept from pass 1 so later passes need not re-hash the plaintext. */
  readonly nonce?: Uint8Array;
}

type Seal = { readonly ports: CryptoPorts; readonly key: Uint8Array };

interface LayerJob {
  readonly kind: LayerKind;
  readonly flags?: number;
  readonly seal?: Seal;
  readonly records: Planned[];
}

async function* plainChunks(source: PayloadSource): AsyncGenerator<Uint8Array> {
  if (source instanceof Uint8Array) {
    if (source.byteLength > 0) yield source;
    return;
  }
  for await (const chunk of source()) {
    if (!(chunk instanceof Uint8Array)) fail('a ChunkSource yielded something other than a Uint8Array.');
    if (chunk.byteLength > 0) yield chunk;
  }
}

async function collect(source: PayloadSource): Promise<Uint8Array> {
  if (source instanceof Uint8Array) return source;
  const parts: Uint8Array[] = [];
  let n = 0;
  for await (const c of plainChunks(source)) {
    parts.push(c.slice());
    n += c.byteLength;
  }
  const out = new Uint8Array(n);
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.byteLength;
  }
  return out;
}

/**
 * The stored bytes of a record: the plaintext, or `nonce ‖ ciphertext ‖ tag` in a
 * keyed layer. Adds the plaintext length read to `tally.plain`.
 */
async function* storedChunks(
  layer: LayerKind,
  seal: Seal | undefined,
  meta: RecordMeta,
  source: PayloadSource,
  nonce: Uint8Array | undefined,
  tally: { plain: number },
): AsyncGenerator<Uint8Array> {
  if (!seal) {
    for await (const c of plainChunks(source)) {
      tally.plain += c.byteLength;
      yield c;
    }
    return;
  }
  const aad = recordAad(layer, meta.kind, meta.locator);
  const stream = seal.ports.aeadStream;
  if (stream && nonce) {
    yield nonce;
    const s = stream.seal(seal.key, nonce, aad);
    for await (const c of plainChunks(source)) {
      tally.plain += c.byteLength;
      const out = s.update(c);
      if (out.byteLength > 0) yield out;
    }
    yield s.final();
    return;
  }
  const whole = await collect(source);
  tally.plain += whole.byteLength;
  yield sealPayload(seal.ports, seal.key, whole, aad);
}

/** Pass 1 for one layer: every record's id and stored length, then index order. */
async function planLayer(
  kind: LayerKind,
  seal: Seal | undefined,
  input: StreamLayerInput,
  sha384: Sha384Incremental,
): Promise<Planned[]> {
  const context = CONTEXT[kind];
  const out: Planned[] = [];
  for (const [ri, r] of input.records.entries()) {
    const meta = describeRecord(kind, r);
    let nonce: Uint8Array | undefined;
    let hashed: number | undefined;
    if (seal?.ports.aeadStream) {
      // The nonce is a function of the whole plaintext, so it costs a read of its own.
      const h = sha384();
      hashed = 0;
      for await (const c of plainChunks(r.payload)) {
        h.update(c);
        hashed += c.byteLength;
      }
      nonce = syntheticNonce(seal.ports.kdf, seal.key, h.digest());
    }
    const idState = identifying(sha384, context);
    for (const p of idPrefix(kind, meta)) idState.update(p);
    const tally = { plain: 0 };
    let bytes = 0;
    for await (const c of storedChunks(kind, seal, meta, r.payload, nonce, tally)) {
      idState.update(c);
      bytes += c.byteLength;
    }
    if (hashed !== undefined && hashed !== tally.plain)
      fail(`the source for ${kind} record ${ri} (input order) yielded ${tally.plain} bytes, then ${hashed}.`);
    out.push({ id: idState.digest(), meta, bytes, plainBytes: tally.plain, source: r.payload, nonce });
  }
  return sortRecords(out, kind);
}

/**
 * Feed one layer's index and arena, in file order, to `visit`. With `gaps` false
 * the alignment padding before the index and between index and arena is skipped:
 * that is exactly what the layer id covers (`index ‖ arena`; padding *inside* the
 * arena is part of the arena). With `gaps` true it is the file's bytes.
 */
async function streamLayer(
  job: LayerJob,
  plan: LayerPlan,
  sha384: Sha384Incremental,
  checkSources: boolean,
  visit: (bytes: Uint8Array) => void | Promise<void>,
  pos: { at: number },
  gaps: boolean,
): Promise<void> {
  const emit = async (bytes: Uint8Array): Promise<void> => {
    pos.at += bytes.byteLength;
    await visit(bytes);
  };
  const pad = async (to: number): Promise<void> => {
    if (to < pos.at) fail(`internal: planned offset ${to} is behind the write position ${pos.at}.`);
    while (pos.at < to) await emit(ZEROS.subarray(0, Math.min(ZEROS.byteLength, to - pos.at)));
  };

  const gap = async (to: number): Promise<void> => {
    if (gaps) await pad(to);
    else pos.at = to;
  };

  await gap(plan.indexOffset);
  for (let from = 0; from < job.records.length; from += INDEX_CHUNK_ENTRIES) {
    await emit(
      encodeIndex(job.records, plan.offsets, from, Math.min(job.records.length, from + INDEX_CHUNK_ENTRIES)),
    );
  }
  await gap(plan.arenaOffset);
  const context = CONTEXT[job.kind];
  for (const [ri, r] of job.records.entries()) {
    await pad(plan.arenaOffset + plan.offsets[ri]!);
    const start = pos.at;
    let check: Sha384State | undefined;
    if (checkSources) {
      check = identifying(sha384, context);
      for (const p of idPrefix(job.kind, r.meta)) check.update(p);
    }
    const label = `${job.kind} record ${ri} (index order)`;
    const tally = { plain: 0 };
    for await (const c of storedChunks(job.kind, job.seal, r.meta, r.source, r.nonce, tally)) {
      check?.update(c);
      await emit(c);
    }
    if (tally.plain !== r.plainBytes)
      fail(`the source for ${label} yielded ${tally.plain} bytes, but ${r.plainBytes} on its first read.`);
    if (pos.at - start !== r.bytes)
      fail(`the source for ${label} stored ${pos.at - start} bytes, but ${r.bytes} on its first read.`);
    if (check && !equalBytes(check.digest(), r.id))
      fail(
        `the source for ${label} yielded different bytes than on its first read; its id no longer matches.`,
      );
  }
  if (pos.at !== plan.arenaOffset + plan.arenaBytes)
    fail(`internal: ${job.kind} arena ended at ${pos.at}, planned ${plan.arenaOffset + plan.arenaBytes}.`);
  // The Odu index section follows the arena in the file. It is not part of the layer id, so the
  // layer-id pass (no gaps) stops here.
  if (gaps && plan.odu) {
    await pad(plan.oduOffset);
    await emit(plan.odu);
  }
  // The tree section (v1.3) follows the Odu section, if any, under the same rule.
  if (gaps && plan.tree) {
    await pad(plan.treeOffset);
    await emit(plan.tree);
  }
}

function equalBytes(a: Uint8Array, b: Uint8Array): boolean {
  if (a.byteLength !== b.byteLength) return false;
  for (let i = 0; i < a.byteLength; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** Coalesces small writes; passes large ones straight through. */
class Staging {
  private readonly buf: Uint8Array;
  private used = 0;
  constructor(
    private readonly sink: ContainerSink,
    size: number,
  ) {
    this.buf = new Uint8Array(size);
  }
  async write(bytes: Uint8Array): Promise<void> {
    if (bytes.byteLength >= this.buf.byteLength) {
      await this.flush();
      await this.sink.write(bytes);
      return;
    }
    if (this.used + bytes.byteLength > this.buf.byteLength) await this.flush();
    this.buf.set(bytes, this.used);
    this.used += bytes.byteLength;
  }
  async flush(): Promise<void> {
    if (this.used === 0) return;
    const n = this.used;
    this.used = 0;
    await this.sink.write(this.buf.subarray(0, n));
  }
}

/**
 * Write `input` to `sink` as a container byte-identical to
 * `packContainer(input', sha384Of(sha384), options.seal)`, where `input'` holds each
 * source's bytes in memory. Payload memory stays O(1); see the module comment.
 */
export async function writeContainer(
  input: StreamContainerInput,
  sha384: Sha384Incremental,
  sink: ContainerSink,
  options: StreamWriteOptions = {},
): Promise<StreamWriteResult> {
  const { seal, checkSources = true, bufferBytes = DEFAULT_BUFFER_BYTES } = options;
  const layers = checkLayers(input.layers, (k) => !!seal?.keys[k]);

  // Pass 1.
  const jobs: LayerJob[] = [];
  for (const l of layers) {
    const layerSeal = isKeyedLayer(l) ? { ports: seal!.ports, key: seal!.keys[l.kind]! } : undefined;
    jobs.push({
      kind: l.kind,
      flags: l.flags,
      seal: layerSeal,
      records: await planLayer(l.kind, layerSeal, l, sha384),
    });
  }
  const { plans, fileBytes, versionMinor } = planLayers(
    jobs.map((j, li) => ({
      kind: j.kind,
      lengths: j.records.map((r) => r.bytes),
      odu: oduSection(
        layers[li]!,
        j.records.map((r) => r.meta.locator),
      ),
      tree: treeSection(
        layers[li]!,
        j.records.map((r) => r.meta.locator),
      ),
    })),
  );

  // Pass 2: layer ids.
  const table = new Uint8Array(jobs.length * LAYER_ENTRY_BYTES);
  for (const [li, job] of jobs.entries()) {
    const plan = plans[li]!;
    const h = identifying(sha384, CONTEXT[job.kind]);
    await streamLayer(job, plan, sha384, checkSources, (b) => h.update(b), { at: 0 }, false);
    table.set(encodeLayerEntry(job, plan, h.digest()), li * LAYER_ENTRY_BYTES);
  }

  const header = encodeHeader(jobs.length, fileBytes, versionMinor);
  const emitFile = async (visit: (bytes: Uint8Array) => void | Promise<void>): Promise<void> => {
    const pos = { at: 0 };
    pos.at += header.byteLength;
    await visit(header);
    pos.at += table.byteLength;
    await visit(table);
    for (const [li, job] of jobs.entries())
      await streamLayer(job, plans[li]!, sha384, checkSources, visit, pos, true);
    if (pos.at !== fileBytes) fail(`internal: wrote ${pos.at} bytes, planned ${fileBytes}.`);
  };

  // Pass 3 (and 4): the file id, and the file.
  const staging = bufferBytes > 0 ? new Staging(sink, bufferBytes) : undefined;
  const write = (b: Uint8Array): void | Promise<void> => (staging ? staging.write(b) : sink.write(b));
  const fileHash = identifying(sha384, CONTEXT.container);
  let fileId: Uint8Array;
  if (sink.patch) {
    await emitFile(async (b) => {
      fileHash.update(b);
      await write(b);
    });
    await staging?.flush();
    fileId = fileHash.digest();
    await sink.patch(H_FILE_ID, fileId);
  } else {
    await emitFile((b) => fileHash.update(b));
    fileId = fileHash.digest();
    header.set(fileId, H_FILE_ID);
    await emitFile(write);
    await staging?.flush();
  }
  const keyed = jobs.some((j) => j.seal?.ports.aeadStream);
  return { fileBytes, fileId, passes: (sink.patch ? 3 : 4) + (keyed ? 1 : 0) };
}
