// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
#import "aamt_metal.h"
#import <Metal/Metal.h>
#include <chrono>
#include <cstring>
#include <vector>

namespace {

constexpr size_t kPageSize = 4096;
inline size_t alignUp(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

// Params struct mirrored in the embedded MSL source below.
struct Params {
    uint32_t base_offset;
    uint32_t stride;
    uint32_t dim;
};

// Runtime-compiled MSL source — no .metal file, no `metal` CLI compiler,
// no Xcode project. MTLDevice.newLibraryWithSource ships with
// Metal.framework itself.
NSString* const kKernelSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint32_t base_offset;
    uint32_t stride;
    uint32_t dim;
};

kernel void aamt_l2norm(const device uint8_t* data [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant Params& p [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    const uint32_t offset = p.base_offset + gid * p.stride;
    const device float* vec = (const device float*)(data + offset);
    float sum = 0.0f;
    for (uint32_t i = 0; i < p.dim; i++) {
        float v = vec[i];
        sum += v * v;
    }
    out[gid] = sqrt(sum);
}
)MSL";

} // namespace

struct aamt_metal_ctx {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
};

extern "C" {

aamt_metal_ctx* aamt_metal_init(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return nullptr;

        NSError* err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:kKernelSource options:nil error:&err];
        if (!lib) return nullptr;
        id<MTLFunction> fn = [lib newFunctionWithName:@"aamt_l2norm"];
        if (!fn) return nullptr;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pipeline) return nullptr;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return nullptr;

        auto* ctx = new aamt_metal_ctx();
        ctx->device = device;
        ctx->queue = queue;
        ctx->pipeline = pipeline;
        return ctx;
    }
}

void aamt_metal_free(aamt_metal_ctx* ctx) { delete ctx; }

const char* aamt_metal_device_name(const aamt_metal_ctx* ctx) {
    if (!ctx) return "(none)";
    return ctx->device.name.UTF8String;
}

int aamt_metal_has_unified_memory(const aamt_metal_ctx* ctx) {
    if (!ctx) return 0;
    return ctx->device.hasUnifiedMemory ? 1 : 0;
}

int aamt_metal_l2norm(aamt_metal_ctx* ctx, const void* data, uint64_t length,
                      uint64_t base_offset, uint32_t stride, uint32_t dim,
                      uint32_t count, int zero_copy, float* out_norms,
                      double* out_ms) {
    if (!ctx || !data || count == 0) return -1;

    @autoreleasepool {
        const auto t0 = std::chrono::steady_clock::now();

        id<MTLBuffer> inBuf;
        if (zero_copy) {
            // Requires page-aligned pointer; length rounded up to a whole
            // page is safe because mmap always backs a full final page
            // (OS zero-fills past EOF within that page).
            const size_t rounded = alignUp(static_cast<size_t>(length), kPageSize);
            inBuf = [ctx->device newBufferWithBytesNoCopy:const_cast<void*>(data)
                                                    length:rounded
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
        } else {
            inBuf = [ctx->device newBufferWithBytes:data
                                              length:static_cast<NSUInteger>(length)
                                             options:MTLResourceStorageModeShared];
        }
        if (!inBuf) return -2;

        id<MTLBuffer> outBuf = [ctx->device newBufferWithLength:sizeof(float) * count
                                                         options:MTLResourceStorageModeShared];
        Params params{static_cast<uint32_t>(base_offset), stride, dim};
        id<MTLBuffer> paramsBuf = [ctx->device newBufferWithBytes:&params
                                                            length:sizeof(params)
                                                           options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx->pipeline];
        [enc setBuffer:inBuf offset:0 atIndex:0];
        [enc setBuffer:outBuf offset:0 atIndex:1];
        [enc setBuffer:paramsBuf offset:0 atIndex:2];

        const NSUInteger tgSize = std::min<NSUInteger>(ctx->pipeline.maxTotalThreadsPerThreadgroup, 256);
        [enc dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        std::memcpy(out_norms, outBuf.contents, sizeof(float) * count);

        const auto t1 = std::chrono::steady_clock::now();
        if (out_ms) *out_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return 0;
    }
}

} // extern "C"
