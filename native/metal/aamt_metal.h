// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/* AAMT WP-23 §3.2/§4 — Apple Silicon zero-copy payload compute.
 *
 * On unified memory, a .aamt payload arena can be wrapped directly as an
 * MTLBuffer with newBufferWithBytesNoCopy: the GPU reads the same physical
 * pages the CPU just routed to, no transfer. This header exposes exactly
 * enough surface to measure that claim against the honest alternative
 * (newBufferWithBytes:, which copies) on real payload data.
 *
 * The compute kernel (L2 norm of fixed-size float vectors) is compiled from
 * an embedded MSL source string at process startup via
 * MTLDevice.newLibraryWithSource — this requires only Metal.framework
 * (present on every Mac), not the standalone `metal` shader compiler that
 * ships with full Xcode. No .metal file, no Xcode project, no metallib.
 */
#ifndef AAMT_METAL_H
#define AAMT_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aamt_metal_ctx aamt_metal_ctx;

/* NULL if no Metal device is available (e.g. non-Apple hardware). */
aamt_metal_ctx* aamt_metal_init(void);
void aamt_metal_free(aamt_metal_ctx* ctx);
const char* aamt_metal_device_name(const aamt_metal_ctx* ctx);
int aamt_metal_has_unified_memory(const aamt_metal_ctx* ctx);

/* Computes the L2 norm of `count` fixed-size float vectors of `dim` floats,
 * each starting at byte offset (base_offset + i * stride) within `data`.
 * `zero_copy` selects newBufferWithBytesNoCopy: (true, requires page-aligned
 * `data`/`length`) vs newBufferWithBytes: (false, copies). Writes `count`
 * floats to caller-allocated `out_norms` and the GPU-side wall time (buffer
 * wrap + dispatch + readback) in milliseconds to `out_ms`. Returns 0 on
 * success. */
int aamt_metal_l2norm(aamt_metal_ctx* ctx, const void* data, uint64_t length,
                      uint64_t base_offset, uint32_t stride, uint32_t dim,
                      uint32_t count, int zero_copy, float* out_norms,
                      double* out_ms);

#ifdef __cplusplus
}
#endif

#endif /* AAMT_METAL_H */
