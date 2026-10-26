// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// aamt_dequant.wgsl
// -----------------------------------------------------------------------------
// AAMT Core Engine — GPU-side .aamt dequantization (PRD §4 "read one uint from
// VRAM and unpack eight neural connections in parallel"). Reads the packed INT4
// weights + per-group scales straight from storage buffers and reconstructs the
// f32 weights. The runtime never parses a model graph; the blob is just data.
// GPU==CPU against dequantCPU() in aamt-format.ts.
// -----------------------------------------------------------------------------

struct AamtParams {
  count : u32,
  groupSize : u32,
  kind : u32,
  _p1 : u32,
};

@group(0) @binding(0) var<storage, read>       packed : array<u32>;
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
@group(0) @binding(2) var<storage, read_write> weights : array<f32>;
@group(0) @binding(3) var<uniform>             params  : AamtParams;

@compute @workgroup_size(64)
fn dequant(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  
  let scale = scales_at(i / params.groupSize);
  
  if (params.kind == 0u) {
    // INT4 (8 weights per u32)
    let word = packed[i >> 3u];
    let nib = (word >> ((i & 7u) * 4u)) & 0xFu;
    let q = i32(nib) - 8;
    weights[i] = f32(q) * scale;
  } else if (params.kind == 1u) {
    // Ternary (16 weights per u32, 2 bits each)
    let word = packed[i >> 4u];
    let bits = (word >> ((i & 15u) * 2u)) & 0x3u;
    var q = 0.0;
    if (bits == 1u) { q = 1.0; }
    else if (bits == 3u) { q = -1.0; }
    weights[i] = q * scale;
  } else if (params.kind == 2u) {
    // 1bit (32 weights per u32, 1 bit each)
    let word = packed[i >> 5u];
    let bit = (word >> (i & 31u)) & 0x1u;
    var q = -1.0;
    if (bit == 1u) { q = 1.0; }
    weights[i] = q * scale;
  }
}
