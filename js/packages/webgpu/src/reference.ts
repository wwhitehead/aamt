// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * The CPU answer the GPU is checked against.
 *
 * The GPU reads group scales as f16 (halving their bandwidth), so the reference
 * dequantises with f16-rounded scales too — otherwise the comparison would
 * measure the f16 conversion, not the kernel. Sums are accumulated in f64.
 */
import { f16Round } from '@aamt/model/f16';
import { dequantCPU, type AamtParsed } from '@aamt/model/tensor';

export function gemvReference(t: AamtParsed, rows: number, cols: number, x: Float32Array): Float64Array {
  const w = dequantCPU({ ...t, scales: Float32Array.from(t.scales, f16Round) });
  const y = new Float64Array(rows);
  for (let r = 0; r < rows; r++) {
    let acc = 0;
    const base = r * cols;
    for (let k = 0; k < cols; k++) acc += w[base + k]! * x[k]!;
    y[r] = acc;
  }
  return y;
}

/** Largest |gpu − cpu| divided by the largest |cpu|, so the bound does not depend on the matrix's scale. */
export function relativeError(gpu: Float32Array, cpu: Float64Array): number {
  let maxDiff = 0;
  let maxRef = 0;
  for (let i = 0; i < cpu.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(gpu[i]! - cpu[i]!));
    maxRef = Math.max(maxRef, Math.abs(cpu[i]!));
  }
  return maxRef === 0 ? maxDiff : maxDiff / maxRef;
}

/** Seeded uniform floats in (−0.5, 0.5) — SplitMix32-style, so every run checks the same matrix. */
export function seeded(n: number, seed: number): Float32Array {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
    out[i] = ((z ^ (z >>> 16)) >>> 0) / 4294967296 - 0.5;
  }
  return out;
}
