// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * .aamt weight format — the shader-native baked tensor.
 *
 * Open-weight tensors are baked into a compact binary that streams straight into
 * a GPU storage buffer with no framework parse at runtime:
 *
 *   - group-wise symmetric quantization (group size G, one f32 scale per group),
 *   - weights bit-packed into u32 words — 8 int4, 16 ternary or 32 1-bit per
 *     word — so a shader reads one u32 and unpacks them in-register,
 *   - a tiny fixed header so the blob is self-describing.
 *
 * Layout, version 2 (all little-endian, 4-byte aligned):
 *   u32 magic 'AAMT' | u32 version (2) | u32 kind (0 int4, 1 ternary, 2 1bit)
 *   | u32 count N | u32 groupSize G | u32 groupCount
 *   f32 scale[groupCount]
 *   u32 packed[ceil(N / weightsPerWord)]
 *
 * Codes: int4 nibble = q + 8, q ∈ [-8, 7]; ternary 2 bits, 00 = 0, 01 = +1,
 * 11 = −1; 1bit 0 = −1, 1 = +1. Dequant multiplies by the group's scale.
 * Version 1 (no kind word, always int4) is still read. The GPU twins are
 * aamt_dequant.wgsl and aamt_gemv_mr.wgsl in @aamt/webgpu.
 */

export const AAMT_MAGIC = 0x544d4141; // 'AAMT' little-endian
export const AAMT_VERSION = 2; // V2 adds 'kind' to header
export const DEFAULT_GROUP = 32;

export type AamtKind = 'int4' | 'ternary' | '1bit';
const KIND_TO_ID: Record<AamtKind, number> = { 'int4': 0, 'ternary': 1, '1bit': 2 };
const ID_TO_KIND: Record<number, AamtKind> = { 0: 'int4', 1: 'ternary', 2: '1bit' };

export interface AamtParsed {
  kind: AamtKind;
  count: number;
  groupSize: number;
  groupCount: number;
  scales: Float32Array;
  packed: Uint32Array;
}

export function bake(weights: Float32Array, groupSize = DEFAULT_GROUP, kind: AamtKind = 'int4'): ArrayBuffer {
  const n = weights.length;
  const groupCount = Math.ceil(n / groupSize);
  const scales = new Float32Array(groupCount);
  
  const weightsPerU32 = kind === '1bit' ? 32 : kind === 'ternary' ? 16 : 8;
  const packed = new Uint32Array(Math.ceil(n / weightsPerU32));

  for (let g = 0; g < groupCount; g++) {
    const start = g * groupSize;
    const end = Math.min(start + groupSize, n);
    let maxAbs = 0;
    for (let i = start; i < end; i++) maxAbs = Math.max(maxAbs, Math.abs(weights[i]!));
    
    // For 1bit and ternary, scale maps to the magnitude of the 1 state.
    const scale = maxAbs > 0 ? (kind === 'int4' ? maxAbs / 7 : maxAbs) : 1;
    scales[g] = scale;
    
    for (let i = start; i < end; i++) {
      if (kind === 'int4') {
        let q = Math.round(weights[i]! / scale);
        if (q < -8) q = -8; else if (q > 7) q = 7;
        const nibble = (q + 8) & 0xf;
        packed[i >> 3]! |= nibble << ((i & 7) * 4);
      } else if (kind === 'ternary') {
        // Ternary: -1, 0, 1 mapped to 2 bits. (00 = 0, 01 = +1, 11 = -1)
        const v = weights[i]! / scale;
        let q = Math.abs(v) < 0.5 ? 0 : (v > 0 ? 1 : -1);
        let bits = q === 0 ? 0 : (q === 1 ? 1 : 3);
        packed[i >> 4]! |= bits << ((i & 15) * 2);
      } else if (kind === '1bit') {
        // 1bit: -1, 1 mapped to 1 bit. (0 = -1, 1 = +1)
        const v = weights[i]!;
        let q = v >= 0 ? 1 : 0;
        packed[i >> 5]! |= q << (i & 31);
      }
    }
  }

  const headerU32 = 6;
  const buf = new ArrayBuffer((headerU32 + groupCount) * 4 + packed.length * 4);
  const h = new Uint32Array(buf, 0, headerU32);
  h[0] = AAMT_MAGIC; h[1] = AAMT_VERSION; h[2] = KIND_TO_ID[kind]; h[3] = n; h[4] = groupSize; h[5] = groupCount;
  new Float32Array(buf, headerU32 * 4, groupCount).set(scales);
  new Uint32Array(buf, (headerU32 + groupCount) * 4, packed.length).set(packed);
  return buf;
}

export function parse(buffer: ArrayBuffer): AamtParsed {
  const h = new Uint32Array(buffer, 0, 6);
  if (h[0] !== AAMT_MAGIC) throw new Error('parse: not an .aamt blob (bad magic).');
  
  // Handle V1 fallback
  let kind: AamtKind = 'int4';
  let count: number, groupSize: number, groupCount: number, headerU32: number;
  if (h[1] === 1) {
     headerU32 = 5;
     count = h[2]!; groupSize = h[3]!; groupCount = h[4]!;
  } else if (h[1] === AAMT_VERSION) {
     headerU32 = 6;
     kind = ID_TO_KIND[h[2]!] || 'int4';
     count = h[3]!; groupSize = h[4]!; groupCount = h[5]!;
  } else {
     throw new Error(`parse: unsupported .aamt version ${h[1]}.`);
  }
  
  const scales = new Float32Array(buffer, headerU32 * 4, groupCount);
  const weightsPerU32 = kind === '1bit' ? 32 : kind === 'ternary' ? 16 : 8;
  const packed = new Uint32Array(buffer, (headerU32 + groupCount) * 4, Math.ceil(count / weightsPerU32));
  return { kind, count, groupSize, groupCount, scales, packed };
}

export function dequantCPU(p: AamtParsed): Float32Array {
  const out = new Float32Array(p.count);
  for (let i = 0; i < p.count; i++) {
    const scale = p.scales[(i / p.groupSize) | 0]!;
    if (p.kind === 'int4') {
      const nibble = (p.packed[i >> 3]! >>> ((i & 7) * 4)) & 0xf;
      out[i] = (nibble - 8) * scale;
    } else if (p.kind === 'ternary') {
      const bits = (p.packed[i >> 4]! >>> ((i & 15) * 2)) & 0x3;
      out[i] = bits === 3 ? -scale : (bits === 1 ? scale : 0);
    } else if (p.kind === '1bit') {
      const bit = (p.packed[i >> 5]! >>> (i & 31)) & 0x1;
      out[i] = bit === 1 ? scale : -scale;
    }
  }
  return out;
}

/** Bits per weight actually stored (4 for data + scale amortised over the group). */
export function bitsPerWeight(groupSize = DEFAULT_GROUP): number {
  return 4 + 32 / groupSize;
}
