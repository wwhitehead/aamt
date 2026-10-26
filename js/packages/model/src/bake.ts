// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * Streaming safetensors → `.aamt` model baker.
 *
 * The AMDL layout puts the manifest and footer at the END of the file, so a
 * baker can write each tensor as soon as it is quantised, without knowing the
 * total size up front. This uses that: one tensor in memory at a time, so peak
 * memory follows the largest tensor, not the checkpoint. (Measured on
 * Qwen3-0.6B: 1.6 GB peak RSS, dominated by its 151,936 × 1024 embedding held
 * as f32 — so for a small model that is about the file's size; the gap opens as
 * models grow and their largest tensor becomes a small share of the whole.)
 *
 * Kind policy, per tensor:
 * - a matrix whose row length is a multiple of the quant group gets the chosen
 *   kind (`int4` by default; `ternary` or `1bit` on request) — this is what the
 *   GEMV kernel reads;
 * - everything else (norm weights, biases, odd shapes) stays `f32`, because
 *   those tensors are small and precision-sensitive.
 */
import { closeSync, openSync, readSync, writeSync } from 'node:fs';
import { AMDL_FOOTER_BYTES, writeFooter, type TensorEntry } from './model';
import { isFloatDtype, parseSafetensorsHeader, safetensorsHeaderLength, toFloat32 } from './safetensors';
import { DEFAULT_GROUP, bake, type AamtKind } from './tensor';

export interface BakeOptions {
  kind?: AamtKind;
  groupSize?: number;
  /** Called once per tensor, after it is written. */
  onTensor?: (entry: TensorEntry, index: number, total: number) => void;
}

export interface BakeSummary {
  tensors: number;
  bytesIn: number;
  bytesOut: number;
  kinds: Record<string, number>;
}

/** Which kind a tensor of this shape is stored as. */
export function kindFor(shape: readonly number[], kind: AamtKind, groupSize: number): AamtKind | 'f32' {
  const cols = shape[shape.length - 1] ?? 0;
  return shape.length >= 2 && cols > 0 && cols % groupSize === 0 && cols % 32 === 0 ? kind : 'f32';
}

export function bakeSafetensorsFile(inputPath: string, outputPath: string, opts: BakeOptions = {}): BakeSummary {
  const kind = opts.kind ?? 'int4';
  const groupSize = opts.groupSize ?? DEFAULT_GROUP;
  const fin = openSync(inputPath, 'r');
  const fout = openSync(outputPath, 'w');
  try {
    const first8 = new Uint8Array(8);
    readSync(fin, first8, 0, 8, 0);
    const headBytes = new Uint8Array(safetensorsHeaderLength(first8));
    readSync(fin, headBytes, 0, headBytes.byteLength, 0);
    const header = parseSafetensorsHeader(headBytes);
    const names = Object.keys(header.tensors);

    const manifest: TensorEntry[] = [];
    const kinds: Record<string, number> = {};
    let offset = 0;
    let bytesIn = 0;
    names.forEach((name, i) => {
      const meta = header.tensors[name]!;
      if (!isFloatDtype(meta.dtype)) throw new Error(`bake: tensor "${name}" is ${meta.dtype}; only float tensors are supported`);
      const raw = new Uint8Array(meta.end - meta.start);
      readSync(fin, raw, 0, raw.byteLength, header.dataStart + meta.start);
      bytesIn += raw.byteLength;
      const data = toFloat32(meta.dtype, raw);
      const k = kindFor(meta.shape, kind, groupSize);
      // The blob's own header must say the kind the manifest says: a reader that binds by kind refuses a mismatch.
      const blob = new Uint8Array(k === 'f32' ? data.slice().buffer : bake(data, groupSize, k));
      writeSync(fout, blob, 0, blob.byteLength, offset);
      const entry: TensorEntry = { name, shape: meta.shape, offset, length: blob.byteLength, kind: k };
      manifest.push(entry);
      kinds[k] = (kinds[k] ?? 0) + 1;
      offset += blob.byteLength;
      opts.onTensor?.(entry, i, names.length);
    });

    const manifestBytes = new TextEncoder().encode(JSON.stringify(manifest));
    writeSync(fout, manifestBytes, 0, manifestBytes.byteLength, offset);
    const footer = new Uint8Array(AMDL_FOOTER_BYTES);
    writeFooter(new DataView(footer.buffer), manifest.length, offset, manifestBytes.byteLength);
    writeSync(fout, footer, 0, footer.byteLength, offset + manifestBytes.byteLength);
    return { tensors: manifest.length, bytesIn, bytesOut: offset + manifestBytes.byteLength + AMDL_FOOTER_BYTES, kinds };
  } finally {
    closeSync(fin);
    closeSync(fout);
  }
}
