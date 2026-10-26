// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// aamt_gemv_mr.wgsl
// -----------------------------------------------------------------------------
// Multi-row GEMV with shared-memory activation staging. Replaces
// aamt_matmul.wgsl's `matmul_wg` on the decode hot path.
//
// WHY, measured on BeyondSight-Ezmarelda's q_proj (M = K = 5120, `1bit` weights,
// group size 32, f16 scales):
//
//   one weight row  = K/8 bytes weights + (K/32)*2 bytes scales =   960 B
//   the x vector    = K*4 bytes                                 = 20480 B  (21x)
//
// The old kernel gave ONE workgroup to ONE output row, so every row re-read the
// whole of x. Across M rows that is 105 MB of x traffic against 4.9 MB of
// weights -- at plausible rates, ~70us of x reads versus ~12us of weight reads.
// The kernel looked weight-bound and was actually activation-bound. This is a
// direct consequence of aggressive quantization: at 1 bit per weight, an f32
// activation is 32x larger per element than the weight it multiplies.
//
// So the primary fix is REUSE OF X, not the reduction:
//
//   1. ROWS_PER_WG = 8 output rows per workgroup, sharing one staged copy of
//      each x tile -> 8x less x traffic (105 MB -> 13 MB on the shape above).
//   2. x is staged in workgroup memory in K-tiles, so the inner loop reads it
//      from shared memory rather than hoping it stays in L1.
//   3. The row reduction is over LANES_PER_ROW lanes. `gemv_mr` does it with a
//      shared-memory tree -- log2(LANES_PER_ROW) barriers covering ALL rows in
//      the workgroup at once, versus the old kernel's 7 barriers per single row.
//      `gemv_mr_sg` does it with one `subgroupAdd` and no barriers, but requires
//      LANES_PER_ROW == 32 (see the gating note below).
//   4. Scale indexing uses `>> gShift` instead of `/ groupSize`. groupSize is a
//      runtime uniform, so the old form emitted a real 32-bit integer division
//      -- not a single-cycle op -- once per weight word in the hottest loop.
//
// Thread layout: WG_SIZE threads = ROWS_PER_WG x LANES_PER_ROW.
//   rowInWg = lid.x / LANES_PER_ROW,  lane = lid.x % LANES_PER_ROW
// Dispatch: ceil(M / ROWS_PER_WG) workgroups, laid out 2D as
// (gridW, ceil(nWg/gridW)) so M can exceed the 65535 per-dimension limit
// (lm_head M = vocab_size). The host must use the SAME ROWS_PER_WG it
// substituted -- see QwenForward.gemvRowsPerWg.
// NOTE `params.gridW` counts WORKGROUPS here, not rows as in aamt_matmul.wgsl.
//
// No early `return` for out-of-range rows: threads must reach every
// workgroupBarrier() below or the workgroup deadlocks, and a partially-exited
// subgroup would make `subgroupAdd` sum the wrong lane set. Out-of-range rows
// instead compute against clamped indices and skip only the final store.
// -----------------------------------------------------------------------------

// NO `enable subgroups;` here, deliberately. An `enable` directive makes the
// WHOLE module fail to compile on an adapter lacking the feature -- including
// the shared-memory entry points below, which need nothing. So the subgroup
// variants live in aamt_gemv_mr_sg.wgsl and the host prepends the directive and
// appends those entry points to this source only when the feature is granted
// (see buildGemvMrSource() in gpu-features.ts). WGSL requires `enable` to
// precede all declarations, which is why it is prepended rather than pasted in.

// ROWS_PER_WG and LANES_PER_ROW are SUBSTITUTED by buildGemvMrSource() in
// gpu-features.ts (WGSL has no preprocessor, and workgroup array sizes must be
// `const`, not `override`). Their product must stay WG_SIZE. The values written
// here are the shipped default and keep this file valid standalone so it parses
// and can be diffed on its own.
//
// This split is the most consequential number in the decode path, because it
// divides the activation traffic -- see GEMV_DEFAULT_ROWS_PER_WG for the measured
// table. At 8 rows, x was still ~0.98x the weight traffic; at 16 it is ~0.49x.
// Fewer lanes per row also means more words per lane and a shallower reduction.
// The subgroup variant requires LANES_PER_ROW == 32 and is skipped otherwise.
const ROWS_PER_WG   : u32 = 16u;
const LANES_PER_ROW : u32 = 16u;
const WG_SIZE       : u32 = 256u;   // ROWS_PER_WG * LANES_PER_ROW
// x elements staged per tile. 2048 f32 = 8KB, plus 1KB of reduction scratch,
// well inside the 16KB maxComputeWorkgroupStorageSize floor. Must be a multiple
// of 32 so tiles align with both quant-group boundaries (groupSize is 32 on
// every real checkpoint) and 1bit word boundaries (32 weights per u32).
const X_TILE        : u32 = 2048u;

struct MRParams {
  M         : u32,  // output rows
  K         : u32,  // input columns
  groupSize : u32,  // quant group along K
  gridW     : u32,  // workgroup-grid width (in WORKGROUPS)
  kind      : u32,  // 0 = int4 (8/word), 1 = ternary (16/word), 2 = 1bit (32/word)
  gShift    : u32,  // log2(groupSize); scale index = k >> gShift
  rowOffset : u32,  // starting row offset in packed/scales/yout
  pad       : u32,
};

@group(0) @binding(0) var<storage, read>       packed     : array<u32>;
@group(0) @binding(1) var<storage, read>       scales_f16 : array<u32>;

// f16 scales, two per u32 -- see src/f16.ts. One f32 scale per 32-weight group
// costs 1 bit per weight, which on these 1bit checkpoints equals the weight
// traffic itself: 46% of every byte read per token was scale factors.
// unpack2x16float is core WGSL and needs no feature flag, unlike an f16 storage
// type. `select` rather than a dynamic vector index, which some backends spill
// to scratch memory.
fn scales_at(i : u32) -> f32 {
  let two = unpack2x16float(scales_f16[i >> 1u]);
  return select(two.x, two.y, (i & 1u) == 1u);
}

// vec4 view of x. Same buffer and binding as before -- a storage buffer's
// declared element type is not part of its bind group layout -- but loaded 4
// floats per instruction. Requires K % 4 == 0 (K is a multiple of the quant
// group, 32) and a 16B-aligned binding offset (all binds are 256B-aligned).
@group(0) @binding(2) var<storage, read>       xin    : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> yout   : array<f32>;
@group(0) @binding(4) var<uniform>             params : MRParams;
@group(0) @binding(5) var<storage, read>       bias_vec : array<f32>;

// X_TILE floats held as vec4s: 512 x vec4 = 2048 floats = 8KB. vec4-typed so the
// inner loops read 4 activations per instruction; see the dot helpers below.
var<workgroup> xTile : array<vec4<f32>, 512>;
var<workgroup> red   : array<f32, 256>;

/** Weights packed per u32, by kind. */
fn weights_per_word(kind : u32) -> u32 {
  if (kind == 2u) { return 32u; }
  if (kind == 1u) { return 16u; }
  return 8u;
}

// --- per-word dot products against the STAGED tile ---------------------------
//
// `klv` is the offset of this word's first weight within the current x tile,
// in vec4 UNITS (kl >> 2). Always exact: kl = w * weightsPerWord and every kind
// packs 8, 16 or 32 weights per word, all multiples of 4.
//
// These loops were the real cost centre and it was NOT bandwidth. The first
// version accumulated scalars: `for j in 0..32 { acc = acc + select(-x, x, bit) }`
// -- 32 FP adds all dependent on the previous one. At the 16x16 split a lane
// handles ~10 words of a 1bit K=5120 row, so the chain was ~320 dependent FP adds
// deep while fetching just 40 bytes of weights. At ~4 cycles of add latency that
// is ~1280 cycles of pure serial latency per lane, with the memory system idle.
// Latency bound, not bandwidth bound.
//
// Fixed two ways at once:
//   - vec4 math, so each op does 4 lanes of work (xTile is a vec4 array).
//   - SEVERAL INDEPENDENT ACCUMULATORS, so the adds pipeline instead of
//     serializing. 1bit goes from a 32-deep scalar chain to four 2-deep vec4
//     chains plus a short combine.
// The horizontal sum happens once per word, not once per weight.

/** vec4<bool> for 4 consecutive bits of `word` starting at `sh`. */
fn bits4(word : u32, sh : u32) -> vec4<bool> {
  let w = word >> sh;
  return vec4<bool>((w & 1u) != 0u, (w & 2u) != 0u, (w & 4u) != 0u, (w & 8u) != 0u);
}

/** Horizontal sum of a vec4. */
fn hsum(v : vec4<f32>) -> f32 {
  return (v.x + v.y) + (v.z + v.w);
}

/** 4 int4 nibbles starting at nibble `j0`, dequantized to [-8, 7]. */
fn nib4(word : u32, j0 : u32) -> vec4<f32> {
  let s = j0 * 4u;
  return vec4<f32>(
    f32(i32((word >> s) & 0xFu) - 8),
    f32(i32((word >> (s + 4u)) & 0xFu) - 8),
    f32(i32((word >> (s + 8u)) & 0xFu) - 8),
    f32(i32((word >> (s + 12u)) & 0xFu) - 8),
  );
}

/** One 2-bit ternary code -> -1, 0 or +1. (00 = 0, 01 = +1, 11 = -1) */
fn tern1(b : u32) -> f32 {
  return select(select(0.0, 1.0, b == 1u), -1.0, b == 3u);
}

/** 4 ternary codes starting at code `j0`. */
fn tern4(word : u32, j0 : u32) -> vec4<f32> {
  let f = word >> (j0 * 2u);
  return vec4<f32>(tern1(f & 3u), tern1((f >> 2u) & 3u), tern1((f >> 4u) & 3u), tern1((f >> 6u) & 3u));
}

fn dot8_int4(word : u32, klv : u32) -> f32 {
  // 8 weights = 2 vec4. Two independent dot()s, summed once.
  let d0 = dot(nib4(word, 0u), xTile[klv]);
  let d1 = dot(nib4(word, 4u), xTile[klv + 1u]);
  return d0 + d1;
}

fn dot16_ternary(word : u32, klv : u32) -> f32 {
  // 16 weights = 4 vec4. Two independent chains, then combine.
  let a = dot(tern4(word, 0u), xTile[klv]) + dot(tern4(word, 8u), xTile[klv + 2u]);
  let b = dot(tern4(word, 4u), xTile[klv + 1u]) + dot(tern4(word, 12u), xTile[klv + 3u]);
  return a + b;
}

fn dot32_1bit(word : u32, klv : u32) -> f32 {
  // 32 weights = 8 vec4, and no multiplies at all: a 1bit weight is a sign, so
  // `select(-x, x, bit)` is the whole product. Four independent vec4 accumulators
  // keep the adds pipelined; each is only 2 deep.
  let x0 = xTile[klv];      let x1 = xTile[klv + 1u];
  let x2 = xTile[klv + 2u]; let x3 = xTile[klv + 3u];
  let x4 = xTile[klv + 4u]; let x5 = xTile[klv + 5u];
  let x6 = xTile[klv + 6u]; let x7 = xTile[klv + 7u];

  let a0 = select(-x0, x0, bits4(word, 0u))  + select(-x4, x4, bits4(word, 16u));
  let a1 = select(-x1, x1, bits4(word, 4u))  + select(-x5, x5, bits4(word, 20u));
  let a2 = select(-x2, x2, bits4(word, 8u))  + select(-x6, x6, bits4(word, 24u));
  let a3 = select(-x3, x3, bits4(word, 12u)) + select(-x7, x7, bits4(word, 28u));

  return hsum((a0 + a1) + (a2 + a3));
}

/**
 * This lane's partial dot product for row `m` over the staged tile
 * [k0, k0 + tileLen). Lanes stride by LANES_PER_ROW across the tile's words, so
 * adjacent lanes read adjacent u32s -- coalesced.
 */
fn tile_partial(m : u32, lane : u32, k0 : u32, tileLen : u32) -> f32 {
  let kind = params.kind;
  let wpw = weights_per_word(kind);
  let wordsPerRow = params.K / wpw;
  let rowWordBase = m * wordsPerRow;
  let tileWordStart = k0 / wpw;
  let wordsInTile = tileLen / wpw;
  let scaleRowBase = m * (params.K >> params.gShift);
  let gShift = params.gShift;

  var acc = 0.0;
  for (var w = lane; w < wordsInTile; w = w + LANES_PER_ROW) {
    let word = packed[rowWordBase + tileWordStart + w];
    let kl = w * wpw;                       // offset inside the tile, in floats
    // vec4 units. Exact: wpw is 8, 16 or 32, so kl is always a multiple of 4.
    let klv = kl >> 2u;
    let scale = scales_at(scaleRowBase + ((k0 + kl) >> gShift));
    var d = 0.0;
    if (kind == 2u) { d = dot32_1bit(word, klv); }
    else if (kind == 1u) { d = dot16_ternary(word, klv); }
    else { d = dot8_int4(word, klv); }
    acc = acc + d * scale;
  }
  return acc;
}

/**
 * Shared core. Walks x in tiles, staging each tile cooperatively across all 256
 * threads, then having each 32-lane group accumulate its own row against it.
 *
 * Returns this lane's partial; the caller reduces it.
 *
 * A word's weights never straddle a tile edge: X_TILE and K are both multiples
 * of 32, and every kind packs 8, 16 or 32 weights per word -- all divisors of 32.
 */
fn accumulate_rows(m : u32, lane : u32, lid : u32) -> f32 {
  let K = params.K;
  var acc = 0.0;
  var k0 = 0u;
  loop {
    if (k0 >= K) { break; }
    let tileLen = min(X_TILE, K - k0);

    // Stage this tile of x, in vec4s. Strided by WG_SIZE so all threads
    // participate and consecutive threads read consecutive 16-byte chunks.
    // tileLen is always a multiple of 32 floats (K and X_TILE both are), so the
    // vec4 count divides exactly and no partial vec4 is ever needed.
    let tileVec4 = tileLen >> 2u;
    let srcVec4Base = k0 >> 2u;
    for (var i = lid; i < tileVec4; i = i + WG_SIZE) {
      xTile[i] = xin[srcVec4Base + i];
    }
    workgroupBarrier();

    acc = acc + tile_partial(m, lane, k0, tileLen);

    // Second barrier: without it a fast thread could begin overwriting xTile
    // for the next tile while a slow one is still reading this one.
    workgroupBarrier();
    k0 = k0 + X_TILE;
  }
  return acc;
}

/**
 * Reduction across each row's LANES_PER_ROW lanes, via shared memory.
 * log2(LANES_PER_ROW) barriers, and they cover EVERY row in the workgroup
 * simultaneously -- so at 16x16 that is 4 barriers for 16 rows, against the old
 * kernel's 7 barriers for 1 row.
 *
 * Barriers sit OUTSIDE the `if` so every thread reaches them; a barrier inside
 * divergent control flow deadlocks the workgroup. Generic in LANES_PER_ROW, so
 * substituting the split needs no change here.
 */
fn reduce_row_shared(lid : u32, lane : u32, value : f32) -> f32 {
  red[lid] = value;
  workgroupBarrier();
  var s = LANES_PER_ROW >> 1u;
  loop {
    if (s == 0u) { break; }
    if (lane < s) { red[lid] = red[lid] + red[lid + s]; }
    workgroupBarrier();
    s = s >> 1u;
  }
  return red[lid - lane];        // row-group base holds the total
}

/** Row index for this thread, and whether it is in range. */
fn row_of(wg : vec3<u32>, lid : u32) -> vec2<u32> {
  let rowInWg = lid / LANES_PER_ROW;
  let wgIndex = wg.y * params.gridW + wg.x;
  let m = params.rowOffset + wgIndex * ROWS_PER_WG + rowInWg;
  let endRow = params.rowOffset + params.M;
  // Clamp rather than return: out-of-range lanes must still hit every barrier.
  return vec2<u32>(select(m, 0u, m >= endRow), select(0u, 1u, m < endRow));
}

// =============================================================================
// Shared-memory reduction variants -- no feature requirements, the default path.
// =============================================================================

@compute @workgroup_size(256)
fn gemv_mr(
  @builtin(workgroup_id) wg : vec3<u32>,
  @builtin(local_invocation_id) lid3 : vec3<u32>,
) {
  let lid = lid3.x;
  let lane = lid % LANES_PER_ROW;
  let r = row_of(wg, lid);
  let m = r.x;
  let total = reduce_row_shared(lid, lane, accumulate_rows(m, lane, lid));
  if (lane == 0u && r.y == 1u) { yout[m] = total; }
}

@compute @workgroup_size(256)
fn gemv_mr_bias(
  @builtin(workgroup_id) wg : vec3<u32>,
  @builtin(local_invocation_id) lid3 : vec3<u32>,
) {
  let lid = lid3.x;
  let lane = lid % LANES_PER_ROW;
  let r = row_of(wg, lid);
  let m = r.x;
  let total = reduce_row_shared(lid, lane, accumulate_rows(m, lane, lid));
  if (lane == 0u && r.y == 1u) { yout[m] = total + bias_vec[m]; }
}
