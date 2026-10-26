// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * Read-only inspection of the AMDL multi-tensor container (`.aamt`, Part 9.1.1).
 * Never loads weights: reads the 32-byte footer and the manifest JSON only.
 *
 *   [tensor blobs…][manifest JSON][footer 32 bytes, little-endian]
 *   0 u32 magic 0x4c444d41 ('AMDL') · 4 u32 version (1) · 8 u32 tensorCount · 12 u32 reserved
 *   16 f64 manifestOffset · 24 f64 manifestLength
 */
import { open, stat } from 'node:fs/promises';

export const AMDL_MAGIC = 0x4c444d41;
export const AAMT_BLOB_MAGIC = 0x544d4141;
export const AMDL_FOOTER_BYTES = 32;
const MAX_MANIFEST_BYTES = 64 * 1024 * 1024;

export interface AmdlFooter {
  magic: number;
  version: number;
  tensorCount: number;
  reserved: number;
  manifestOffset: number;
  manifestLength: number;
}

export interface TensorEntry {
  name: string;
  shape: number[];
  offset: number;
  length: number;
  kind?: 'int4' | 'f32' | 'ternary' | '1bit';
  metadata?: Record<string, unknown>;
}

export interface AmdlInspection {
  ok: boolean;
  error?: string;
  path: string;
  sizeBytes: number;
  mtimeMs: number;
  footer?: AmdlFooter;
  tensorCount?: number;
  kinds?: Record<string, number>;
  teraMeta?: Record<string, unknown>;
  hybridMeta?: Record<string, unknown>;
  family?: string;
  vocabSize?: number;
  tokenizer?: string;
}

export function parseFooter(buf: Uint8Array): AmdlFooter {
  if (buf.byteLength < AMDL_FOOTER_BYTES) throw new Error('footer too short');
  const dv = new DataView(buf.buffer, buf.byteOffset + buf.byteLength - AMDL_FOOTER_BYTES, AMDL_FOOTER_BYTES);
  return {
    magic: dv.getUint32(0, true),
    version: dv.getUint32(4, true),
    tensorCount: dv.getUint32(8, true),
    reserved: dv.getUint32(12, true),
    manifestOffset: dv.getFloat64(16, true),
    manifestLength: dv.getFloat64(24, true),
  };
}

export function encodeFooter(f: Omit<AmdlFooter, 'magic' | 'reserved'> & { magic?: number }): Uint8Array {
  const b = new Uint8Array(AMDL_FOOTER_BYTES);
  const dv = new DataView(b.buffer);
  dv.setUint32(0, f.magic ?? AMDL_MAGIC, true);
  dv.setUint32(4, f.version, true);
  dv.setUint32(8, f.tensorCount, true);
  dv.setUint32(12, 0, true);
  dv.setFloat64(16, f.manifestOffset, true);
  dv.setFloat64(24, f.manifestLength, true);
  return b;
}

/** Quick check used by the `magic:AMDL@footer` validator. */
export async function hasAmdlFooter(path: string): Promise<{ ok: boolean; detail: string }> {
  try {
    const st = await stat(path);
    if (st.size < AMDL_FOOTER_BYTES) return { ok: false, detail: 'file smaller than footer' };
    const fh = await open(path, 'r');
    try {
      const b = Buffer.alloc(AMDL_FOOTER_BYTES);
      await fh.read(b, 0, AMDL_FOOTER_BYTES, st.size - AMDL_FOOTER_BYTES);
      const f = parseFooter(b);
      if (f.magic !== AMDL_MAGIC) return { ok: false, detail: `bad magic 0x${f.magic.toString(16)}` };
      if (f.version !== 1) return { ok: false, detail: `unsupported version ${f.version}` };
      if (!(f.manifestOffset >= 0 && f.manifestOffset + f.manifestLength <= st.size - AMDL_FOOTER_BYTES))
        return { ok: false, detail: 'manifest range outside file' };
      return { ok: true, detail: `AMDL v1 · ${f.tensorCount} tensors` };
    } finally {
      await fh.close();
    }
  } catch (e) {
    return { ok: false, detail: (e as Error).message };
  }
}

export function guessFamily(names: string[], fileName: string): string {
  const lower = fileName.toLowerCase();
  if (lower.includes('flow') || lower.includes('nomad')) return 'flow';
  if (lower.includes('gaian') || lower.includes('nightowl')) return 'gaian';
  if (lower.includes('bonsai')) return 'bonsai';
  if (names.some((n) => n.includes('linear_attn') || n.includes('ssm'))) return 'hybrid';
  if (names.some((n) => n.startsWith('model.layers.'))) return 'qwen';
  return 'unknown';
}

/** Full read-only inspection for the Model Manager. */
export async function inspectAamt(path: string): Promise<AmdlInspection> {
  let st;
  try {
    st = await stat(path);
  } catch (e) {
    return { ok: false, error: (e as Error).message, path, sizeBytes: 0, mtimeMs: 0 };
  }
  const base: AmdlInspection = { ok: false, path, sizeBytes: st.size, mtimeMs: st.mtimeMs };
  const fh = await open(path, 'r');
  try {
    if (st.size < AMDL_FOOTER_BYTES) return { ...base, error: 'file smaller than footer' };
    const fb = Buffer.alloc(AMDL_FOOTER_BYTES);
    await fh.read(fb, 0, AMDL_FOOTER_BYTES, st.size - AMDL_FOOTER_BYTES);
    const footer = parseFooter(fb);
    if (footer.magic !== AMDL_MAGIC) return { ...base, footer, error: 'not an AMDL container (bad magic)' };
    if (footer.version !== 1) return { ...base, footer, error: `unsupported AMDL version ${footer.version}` };
    if (footer.manifestLength > MAX_MANIFEST_BYTES) return { ...base, footer, error: 'manifest too large' };
    if (footer.manifestOffset + footer.manifestLength > st.size - AMDL_FOOTER_BYTES) return { ...base, footer, error: 'manifest range outside file' };
    const mb = Buffer.alloc(footer.manifestLength);
    await fh.read(mb, 0, footer.manifestLength, footer.manifestOffset);
    let entries: TensorEntry[];
    try {
      entries = JSON.parse(mb.toString('utf8'));
      if (!Array.isArray(entries)) throw new Error('manifest is not an array');
    } catch (e) {
      return { ...base, footer, error: `manifest JSON invalid: ${(e as Error).message}` };
    }
    const kinds: Record<string, number> = {};
    let teraMeta: Record<string, unknown> | undefined;
    let hybridMeta: Record<string, unknown> | undefined;
    for (const t of entries) {
      if (t.length === 0 && t.metadata) {
        if (t.name === '__tera_meta__') teraMeta = t.metadata;
        if (t.name === '__hybrid_meta__') hybridMeta = t.metadata;
        continue;
      }
      const k = t.kind ?? 'int4';
      kinds[k] = (kinds[k] ?? 0) + 1;
    }
    const fileName = path.split(/[\\/]/).pop() ?? path;
    const embedEntry = entries.find((e) => e.name === 'model.embed_tokens.weight');
    const vocabSize = embedEntry?.shape?.[0] ?? (teraMeta?.vocab_size as number | undefined) ?? (teraMeta?.vocabSize as number | undefined);
    const tokenizer = vocabSize !== undefined ? (vocabSize >= 200000 ? 'bonsai-tokenizer' : 'qwen-tokenizer') : undefined;
    return {
      ...base,
      ok: true,
      footer,
      tensorCount: footer.tensorCount,
      kinds,
      teraMeta,
      hybridMeta,
      family: guessFamily(
        entries.map((e) => e.name),
        fileName,
      ),
      vocabSize,
      tokenizer,
    };
  } finally {
    await fh.close();
  }
}

/** Builds a tiny, valid AMDL file in memory (fixtures & tests only; contains no weights). */
export function buildAmdlFixture(tensors: { name: string; shape: number[]; kind?: TensorEntry['kind']; bytes?: number; metadata?: Record<string, unknown> }[]): Uint8Array {
  const blobs: Uint8Array[] = [];
  const entries: TensorEntry[] = [];
  let offset = 0;
  for (const t of tensors) {
    const len = t.metadata ? 0 : (t.bytes ?? 16);
    blobs.push(new Uint8Array(len));
    entries.push({ name: t.name, shape: t.shape, offset, length: len, ...(t.kind ? { kind: t.kind } : {}), ...(t.metadata ? { metadata: t.metadata } : {}) });
    offset += len;
  }
  const manifest = new TextEncoder().encode(JSON.stringify(entries));
  const footer = encodeFooter({ version: 1, tensorCount: entries.length, manifestOffset: offset, manifestLength: manifest.byteLength });
  const out = new Uint8Array(offset + manifest.byteLength + footer.byteLength);
  let p = 0;
  for (const b of blobs) {
    out.set(b, p);
    p += b.byteLength;
  }
  out.set(manifest, p);
  out.set(footer, p + manifest.byteLength);
  return out;
}
