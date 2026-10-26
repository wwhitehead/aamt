// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * Multi-tensor .aamt model container (PRD §4/§5) — bakes a whole model (every
 * weight tensor) into one streamable blob so it drops straight into VRAM.
 *
 * Layout (streamable — footer at the end so a baker can write tensors as it goes
 * without knowing total size up front):
 *
 *   [ tensor blob 0 ][ tensor blob 1 ] … [ manifest JSON ][ footer (32 B) ]
 *
 * Each tensor blob is a full single-tensor .aamt (see aamt-format.ts). The
 * manifest is `[{ name, shape, offset, length }]`. The footer holds the magic,
 * version, tensor count, and the manifest's offset/length (f64, exact to 2^53
 * so it addresses multi-GB models).
 */

import { closeSync, fstatSync, openSync, readSync } from 'node:fs';
import { bake, parse, type AamtParsed } from './tensor';

export const AMDL_MAGIC = 0x4c444d41; // 'AMDL' little-endian
export const AMDL_VERSION = 1;
export const AMDL_FOOTER_BYTES = 32;

export interface TensorInput {
  name: string;
  shape: number[];
  data: Float32Array;
  /**
   * 'int4' (default) group-quantizes via aamt-format.ts. 'f32' stores the
   * data as-is (no packing) — for small precision-sensitive tensors like
   * RMSNorm weights. 'ternary' and '1bit' enable ultra-low precision modes.
   */
  kind?: 'int4' | 'f32' | 'ternary' | '1bit';
}

export interface TensorEntry {
  name: string;
  shape: number[];
  offset: number; // byte offset of this tensor's blob from file start
  length: number; // blob byte length
  kind?: 'int4' | 'f32' | 'ternary' | '1bit'; // absent = 'int4' (back-compat)
  /** Present only on zero-length sentinel entries (e.g. __tera_meta__,
   *  __hybrid_meta__ from bake_bonsai_aamt.py) that carry manifest-level
   *  metadata rather than a real tensor blob. */
  metadata?: Record<string, unknown>;
}

export interface AamtModel {
  manifest: TensorEntry[];
  /** Parse one INT4 tensor by name into its packed sections + shape. Throws if the tensor is kind 'f32' — use getRaw() instead. */
  get(name: string): { shape: number[]; parsed: AamtParsed };
  /** Read one plain f32 tensor by name (RMSNorm weights, biases, etc). Throws if the tensor is kind 'int4'. */
  getRaw(name: string): { shape: number[]; data: Float32Array };
}

/** Encode the 32-byte footer into a view. */
export function writeFooter(view: DataView, tensorCount: number, manifestOffset: number, manifestLength: number): void {
  view.setUint32(0, AMDL_MAGIC, true);
  view.setUint32(4, AMDL_VERSION, true);
  view.setUint32(8, tensorCount, true);
  view.setUint32(12, 0, true);
  view.setFloat64(16, manifestOffset, true);
  view.setFloat64(24, manifestLength, true);
}

/** Decode the footer; throws on bad magic/version. */
export function readFooter(view: DataView): { tensorCount: number; manifestOffset: number; manifestLength: number } {
  if (view.getUint32(0, true) !== AMDL_MAGIC) throw new Error('aamt-model: bad footer magic (not an .aamt model).');
  const version = view.getUint32(4, true);
  if (version !== AMDL_VERSION) throw new Error(`aamt-model: unsupported model version ${version}.`);
  return {
    tensorCount: view.getUint32(8, true),
    manifestOffset: view.getFloat64(16, true),
    manifestLength: view.getFloat64(24, true),
  };
}

/** Bake a set of tensors into an in-memory model blob (small models / tests). */
export function bakeModel(tensors: readonly TensorInput[], groupSize = 32): ArrayBuffer {
  const blobs = tensors.map((t) => {
    const kind = t.kind ?? 'int4';
    // The blob's own header must say the kind the manifest says: a reader that binds by kind refuses a mismatch.
    const blob = kind === 'f32' ? t.data.slice().buffer : bake(t.data, groupSize, kind);
    return { name: t.name, shape: t.shape, kind, blob };
  });
  const manifest: TensorEntry[] = [];
  let offset = 0;
  for (const b of blobs) {
    manifest.push({ name: b.name, shape: b.shape, offset, length: b.blob.byteLength, kind: b.kind });
    offset += b.blob.byteLength;
  }
  const manifestBytes = new TextEncoder().encode(JSON.stringify(manifest));
  const total = offset + manifestBytes.byteLength + AMDL_FOOTER_BYTES;
  const out = new Uint8Array(total);
  let p = 0;
  for (const b of blobs) { out.set(new Uint8Array(b.blob), p); p += b.blob.byteLength; }
  out.set(manifestBytes, p); p += manifestBytes.byteLength;
  writeFooter(new DataView(out.buffer, p, AMDL_FOOTER_BYTES), tensors.length, offset, manifestBytes.byteLength);
  return out.buffer;
}

/**
 * Read one tensor from a model file on disk without loading the rest: the
 * footer says where the manifest is, the manifest says where the tensor is, and
 * those bytes are the blob as the GPU will consume it. Three reads, any file size.
 */
export function readTensorFromFile(path: string, name: string): { entry: TensorEntry; blob: ArrayBuffer } {
  const fd = openSync(path, 'r');
  try {
    const size = fstatSync(fd).size;
    const footer = new Uint8Array(AMDL_FOOTER_BYTES);
    readSync(fd, footer, 0, AMDL_FOOTER_BYTES, size - AMDL_FOOTER_BYTES);
    const f = readFooter(new DataView(footer.buffer));
    const manifestBytes = new Uint8Array(f.manifestLength);
    readSync(fd, manifestBytes, 0, f.manifestLength, f.manifestOffset);
    const manifest: TensorEntry[] = JSON.parse(new TextDecoder().decode(manifestBytes));
    const entry = manifest.find((m) => m.name === name);
    if (!entry) throw new Error(`aamt-model: tensor "${name}" not found.`);
    const blob = new Uint8Array(entry.length);
    readSync(fd, blob, 0, entry.length, entry.offset);
    return { entry, blob: blob.buffer };
  } finally {
    closeSync(fd);
  }
}

/** Parse an in-memory model blob. */
export function parseModel(buffer: ArrayBuffer): AamtModel {
  const size = buffer.byteLength;
  const footer = readFooter(new DataView(buffer, size - AMDL_FOOTER_BYTES, AMDL_FOOTER_BYTES));
  const manifestBytes = new Uint8Array(buffer, footer.manifestOffset, footer.manifestLength);
  const manifest: TensorEntry[] = JSON.parse(new TextDecoder().decode(manifestBytes));
  const findEntry = (name: string): TensorEntry => {
    const e = manifest.find((m) => m.name === name);
    if (!e) throw new Error(`aamt-model: tensor "${name}" not found.`);
    return e;
  };
  return {
    manifest,
    get(name: string) {
      const e = findEntry(name);
      if ((e.kind ?? 'int4') === 'f32') throw new Error(`aamt-model: "${name}" is kind '${e.kind}', use getRaw().`);
      return { shape: e.shape, parsed: parse(buffer.slice(e.offset, e.offset + e.length)) };
    },
    getRaw(name: string) {
      const e = findEntry(name);
      if (e.kind !== 'f32') throw new Error(`aamt-model: "${name}" is kind '${e.kind ?? 'int4'}', use get().`);
      return { shape: e.shape, data: new Float32Array(buffer.slice(e.offset, e.offset + e.length)) };
    },
  };
}
