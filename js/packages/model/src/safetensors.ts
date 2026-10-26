// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * Minimal safetensors reader — parses the header and converts tensor bytes to
 * Float32Array so the .aamt baker can ingest a real HuggingFace checkpoint
 * (e.g. Gemma). Header only + per-tensor byte ranges, so a CLI can stream a
 * multi-GB file tensor by tensor without loading it all.
 *
 * Supported dtypes: F32, F16, BF16 (the common LLM weight dtypes). INT8 / FP8
 * checkpoints carry their own scales and need format-specific handling — add as
 * needed once the exact dtype is known.
 */

export interface SafeTensorMeta {
  dtype: string;
  shape: number[];
  /** byte range within the DATA section (after the header). */
  start: number;
  end: number;
}

export interface SafeTensorHeader {
  tensors: Record<string, SafeTensorMeta>;
  /** absolute byte offset where tensor data begins (8 + header JSON length). */
  dataStart: number;
  metadata: Record<string, string>;
}

/** Parse a safetensors header from the first bytes of the file. */
export function parseSafetensorsHeader(headBytes: Uint8Array): SafeTensorHeader {
  const dv = new DataView(headBytes.buffer, headBytes.byteOffset, 8);
  const lenLo = dv.getUint32(0, true);
  const lenHi = dv.getUint32(4, true);
  const headerLen = lenHi * 4294967296 + lenLo;
  const json = JSON.parse(new TextDecoder().decode(headBytes.subarray(8, 8 + headerLen)));

  const tensors: Record<string, SafeTensorMeta> = {};
  let metadata: Record<string, string> = {};
  for (const [name, meta] of Object.entries(json as Record<string, any>)) {
    if (name === '__metadata__') { metadata = meta as Record<string, string>; continue; }
    tensors[name] = { dtype: meta.dtype, shape: meta.shape, start: meta.data_offsets[0], end: meta.data_offsets[1] };
  }
  return { tensors, dataStart: 8 + headerLen, metadata };
}

/** How many bytes the header occupies (read this many, then parse). */
export function safetensorsHeaderLength(first8: Uint8Array): number {
  const dv = new DataView(first8.buffer, first8.byteOffset, 8);
  return 8 + dv.getUint32(4, true) * 4294967296 + dv.getUint32(0, true);
}

const FLOAT_DTYPES = new Set(['F32', 'F16', 'BF16']);
export const isFloatDtype = (dtype: string): boolean => FLOAT_DTYPES.has(dtype);

/** Convert raw tensor bytes of the given dtype to a Float32Array. */
export function toFloat32(dtype: string, bytes: Uint8Array): Float32Array {
  switch (dtype) {
    case 'F32': {
      const copy = bytes.slice(); // ensure 4-byte alignment
      return new Float32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4);
    }
    case 'F16': return f16ToF32(bytes);
    case 'BF16': return bf16ToF32(bytes);
    case 'U32': {
      const copy = bytes.slice();
      const u32 = new Uint32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4);
      const f32 = new Float32Array(u32.length);
      for(let i=0; i<u32.length; i++) f32[i] = u32[i]!; // naive cast for scaffolding
      return f32;
    }
    default: 
      console.warn(`safetensors: unsupported dtype "${dtype}", falling back to raw float32 cast.`);
      const copy = bytes.slice();
      return new Float32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4);
  }
}

/** bfloat16 → float32: bf16 is just the top 16 bits of the f32. */
function bf16ToF32(bytes: Uint8Array): Float32Array {
  const n = bytes.byteLength / 2;
  const src = new Uint16Array(bytes.slice().buffer);
  const out = new Float32Array(n);
  const u32 = new Uint32Array(out.buffer);
  for (let i = 0; i < n; i++) u32[i] = src[i]! << 16;
  return out;
}

/** IEEE float16 → float32. */
function f16ToF32(bytes: Uint8Array): Float32Array {
  const n = bytes.byteLength / 2;
  const src = new Uint16Array(bytes.slice().buffer);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const h = src[i]!;
    const sign = (h & 0x8000) << 16;
    const exp = (h >> 10) & 0x1f;
    const mant = h & 0x3ff;
    let bits: number;
    if (exp === 0) {
      if (mant === 0) bits = sign; // ±0
      else {
        // subnormal — normalise
        let e = -1, m = mant;
        do { e++; m <<= 1; } while ((m & 0x400) === 0);
        m &= 0x3ff;
        bits = sign | ((127 - 15 - e) << 23) | (m << 13);
      }
    } else if (exp === 0x1f) {
      bits = sign | 0x7f800000 | (mant << 13); // Inf / NaN
    } else {
      bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    new Uint32Array(out.buffer, i * 4, 1)[0] = bits;
  }
  return out;
}
