// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// f16.ts
// -----------------------------------------------------------------------------
// IEEE-754 binary16 conversion, and packing two f16s per u32 for storage
// buffers.
//
// Why this exists: the `.aamt` format stores one f32 scale per group of
// `groupSize` weights. At the default group size of 32 that is 32 bits per 32
// weights = 1 bit per weight -- REGARDLESS of how small the weights themselves
// are. The BeyondSight checkpoints are `1bit` almost throughout, so the scales
// cost exactly as much memory traffic as the weights: measured from their own
// manifests, 3.362 GB of Yare's 7.211 GB per-token read is scale factors (47%),
// and 1.024 GB of NightOwl's 2.281 GB (45%).
//
// Decode is bandwidth-bound, so halving the scale traffic removes ~23% of every
// byte moved per token -- a ~1.30x speedup for a conversion pass at load time.
//
// Two deliberate design choices:
//
//   1. PACK TWO f16 INTO ONE u32 and unpack in WGSL with `unpack2x16float()`,
//      rather than declaring the buffer `array<f16>`. `unpack2x16float` is a
//      core WGSL builtin requiring NO feature flag, whereas an `f16` storage
//      type needs `shader-f16` (and, for arithmetic in subgroup ops,
//      `subgroups-f16`). So this optimization works on every adapter, including
//      ones that grant no optional features at all.
//
//   2. CONVERT AT UPLOAD, not in the baker. A storage buffer's declared element
//      type is not part of its bind group layout, so the buffer and binding are
//      unchanged -- only its contents and size. That means existing baked
//      checkpoints get the full bandwidth win with NO re-bake: Yare and
//      NightOwl are 7.2 GB and 2.3 GB on disk and re-baking needs the original
//      unquantized weights. A future format revision can store f16 scales
//      directly to halve on-disk size too; that is a separate, optional win.
//
// Precision: f16 has a 10-bit mantissa, so ~3 decimal digits and a relative
// error under 2^-11 (0.049%). Group scales are a per-group magnitude in a range
// like 1e-4..1e-1 for these checkpoints -- comfortably inside f16's normal range
// (6.1e-5 to 65504), and the resulting error is far below the error already
// accepted by quantizing the weights themselves to 1 bit.
// -----------------------------------------------------------------------------

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

/**
 * Encode an f32 as IEEE-754 binary16 bits (round-to-nearest-even), returned in
 * the low 16 bits of the result.
 *
 * Handles the three cases a naive implementation gets wrong: values too small
 * to be a normal f16 (emitted as subnormals, then as signed zero below ~6e-8),
 * values too large (saturate to Inf rather than wrapping the exponent), and
 * ties (rounded to even, matching hardware).
 */
export function f32ToF16Bits(value: number): number {
  f32Scratch[0] = value;
  const bits = u32Scratch[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exp32 = (bits >>> 23) & 0xff;
  const mant32 = bits & 0x7fffff;

  // Inf / NaN. Keep NaN a NaN (any non-zero payload) rather than letting it
  // collapse to Inf, which would silently turn a poisoned scale into a huge one.
  if (exp32 === 0xff) return sign | 0x7c00 | (mant32 !== 0 ? 0x0200 : 0);

  // Rebias: f32 bias 127 -> f16 bias 15.
  const e = exp32 - 127 + 15;

  // Overflow: saturate to Inf. Wrapping here would produce a small finite
  // number from a large one -- the worst possible failure for a scale factor.
  if (e >= 0x1f) return sign | 0x7c00;

  if (e <= 0) {
    // Subnormal range, or underflow to zero. The smallest positive f16
    // subnormal is 2^-24; below half of that, nothing survives rounding.
    if (e < -10) return sign;
    // Restore f32's implicit leading 1, then shift into subnormal position.
    const mant = mant32 | 0x800000;
    const shift = 14 - e; // e <= 0, so shift is 14..24
    let half = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (rem > halfway || (rem === halfway && (half & 1) !== 0)) half++;
    return sign | half;
  }

  // Normal range. Truncate 23 mantissa bits to 10, then round to nearest even.
  // A carry out of the mantissa propagates into the exponent field naturally
  // because the fields are contiguous -- and if that pushes the exponent to
  // 0x1f the result is Inf, which is the correct saturating behaviour.
  let half = (e << 10) | (mant32 >>> 13);
  const rem = mant32 & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1) !== 0)) half++;
  return sign | half;
}

/** Decode IEEE-754 binary16 bits to an f32. Exact: every f16 is representable. */
export function f16BitsToF32(half: number): number {
  const sign = (half & 0x8000) !== 0 ? -1 : 1;
  const exp = (half >>> 10) & 0x1f;
  const mant = half & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24; // zero or subnormal
  if (exp === 0x1f) return mant === 0 ? sign * Infinity : NaN;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

/** Round an f32 to the nearest f16-representable value, staying in f32. */
export function f16Round(value: number): number {
  return f16BitsToF32(f32ToF16Bits(value));
}

/**
 * Pack an f32 array into u32s holding two f16 each, little-endian within the
 * word: element 2i goes in the low half, 2i+1 in the high half. This is exactly
 * the layout WGSL's `unpack2x16float()` expects -- `.x` is the low half.
 *
 * An odd-length input is padded with a zero in the final word's high half.
 * Result byte length is `ceil(n/2) * 4`, i.e. half the input's (rounded up),
 * which is the whole point.
 */
export function packF16Pairs(src: Float32Array): Uint32Array {
  const n = src.length;
  const out = new Uint32Array(Math.ceil(n / 2));
  let w = 0;
  let i = 0;
  for (; i + 1 < n; i += 2, w++) {
    out[w] = f32ToF16Bits(src[i]!) | (f32ToF16Bits(src[i + 1]!) << 16);
  }
  if (i < n) out[w] = f32ToF16Bits(src[i]!);
  return out;
}

/** Inverse of packF16Pairs. `count` is the number of f16 values to extract
 *  (needed because an odd-length source was padded). */
export function unpackF16Pairs(src: Uint32Array, count: number): Float32Array {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const word = src[i >>> 1]!;
    out[i] = f16BitsToF32((i & 1) === 0 ? word & 0xffff : word >>> 16);
  }
  return out;
}

/**
 * Allocate and fill a storage buffer holding `scales` as f16 pairs.
 *
 * This is the one place the conversion happens, shared by every checkpoint
 * upload path (qwen-forward, gemma-forward, aamt-runtime). It replaces the
 * previous pattern of
 *
 *     size: Math.ceil(parsed.scales.byteLength / 4) * 4   // f32, full size
 *     queue.writeBuffer(buf, 0, parsed.scales)
 *
 * at each site. The buffer's *binding* is unchanged — a storage buffer's
 * declared element type is not part of its bind group layout, so the shaders
 * simply declare `array<u32>` and call the generated `*_at()` accessor. Nothing
 * about the `.aamt` file changes, so existing baked checkpoints get the full
 * bandwidth win with no re-bake.
 *
 * `scales.length` is passed to the shader implicitly: the accessor derives the
 * word index as `i >> 1`, so an odd count's padding half is simply never read.
 */
export function createF16ScalesBuffer(
  device: GPUDevice,
  scales: Float32Array,
  label: string,
): GPUBuffer {
  const packed = packF16Pairs(scales);
  const buffer = device.createBuffer({
    // Already a multiple of 4 (Uint32Array), but keep the guard explicit: a
    // zero-length scales array would otherwise request a 0-byte buffer, which
    // is a validation error.
    size: Math.max(4, packed.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label,
  });
  device.queue.writeBuffer(buffer, 0, packed as unknown as GPUAllowSharedBufferSource);
  return buffer;
}

/**
 * The WGSL side of `packF16Pairs`. Paste this accessor into any shader whose
 * scales buffer has been converted, and declare the binding as
 * `array<u32>` instead of `array<f32>`.
 *
 * `select` rather than a dynamic vector index (`two[i & 1u]`), because indexing
 * a vector with a runtime value forces some backends to spill it to scratch
 * memory; `select` stays in registers.
 */
export const WGSL_F16_SCALE_ACCESSOR = /* wgsl */ `
// Two f16 scales per u32; see src/f16.ts. unpack2x16float is core WGSL and
// needs no feature flag, unlike an f16 storage type.
fn scale_at(i : u32) -> f32 {
  let two = unpack2x16float(scales[i >> 1u]);
  return select(two.x, two.y, (i & 1u) == 1u);
}
`;
