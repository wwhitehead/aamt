// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// WP-23 §3.2/§6 — measures the zero-copy claim on Apple Silicon: wrap a
// .aamt payload arena directly as an MTLBuffer (newBufferWithBytesNoCopy)
// vs. the copying alternative (newBufferWithBytes), running the identical
// L2-norm compute kernel over the identical data, timed end-to-end
// (buffer setup + dispatch + waitUntilCompleted + readback).
#include "aamt/aamt.h"
#include "aamt_metal.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const uint32_t count = argc > 1 ? static_cast<uint32_t>(std::atoi(argv[1])) : 200000;
    const uint32_t dim = argc > 2 ? static_cast<uint32_t>(std::atoi(argv[2])) : 64;
    const std::string path = "/tmp/aamt_metal_bench.aamt";

    // 1. Build a substrate where every record carries a fixed-size float
    //    vector payload, so the whole arena is a uniform-stride array the
    //    kernel can walk in one dispatch. Records land in the arena in
    //    BFS/leaf order, not insertion order — the correctness check below
    //    reads back the actual arena bytes rather than assuming order.
    aamt_builder* b = aamt_builder_new(8, 64);
    std::mt19937 rng(0x5EED);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::vector<float> scratch(dim);
    for (uint32_t i = 0; i < count; ++i) {
        float tera[4] = {uni(rng), uni(rng), uni(rng), uni(rng)};
        for (uint32_t d = 0; d < dim; ++d) scratch[d] = nd(rng);
        aamt_builder_add(b, i, tera, scratch.data(),
                         static_cast<uint32_t>(dim * sizeof(float)));
    }
    char err[256] = {0};
    if (aamt_builder_write(b, path.c_str(), err, sizeof(err)) != 0) {
        std::fprintf(stderr, "pack failed: %s\n", err);
        return 1;
    }
    aamt_builder_free(b);

    aamt_file* f = aamt_open(path.c_str(), err, sizeof(err));
    if (!f) {
        std::fprintf(stderr, "open failed: %s\n", err);
        return 1;
    }

    uint64_t arena_size = 0;
    const uint8_t* arena = aamt_payload_arena(f, &arena_size);
    std::printf("payload arena: %llu bytes, %u records x %u floats\n",
                static_cast<unsigned long long>(arena_size), count, dim);

    aamt_metal_ctx* ctx = aamt_metal_init();
    if (!ctx) {
        std::fprintf(stderr, "Metal unavailable on this device.\n");
        aamt_close(f);
        return 1;
    }
    std::printf("device: %s (unified memory: %s)\n", aamt_metal_device_name(ctx),
                aamt_metal_has_unified_memory(ctx) ? "yes" : "no");

    const uint32_t header_size = 32; // RecordHeader
    const uint32_t stride = header_size + static_cast<uint32_t>(dim * sizeof(float));

    std::vector<float> normsZC(count), normsCopy(count);
    double msZC = 0, msCopy = 0;

    int rc1 = aamt_metal_l2norm(ctx, arena, arena_size, header_size, stride, dim,
                               count, /*zero_copy=*/1, normsZC.data(), &msZC);
    int rc2 = aamt_metal_l2norm(ctx, arena, arena_size, header_size, stride, dim,
                               count, /*zero_copy=*/0, normsCopy.data(), &msCopy);
    if (rc1 != 0 || rc2 != 0) {
        std::fprintf(stderr, "kernel dispatch failed (zero_copy=%d, copy=%d)\n", rc1, rc2);
        aamt_metal_free(ctx);
        aamt_close(f);
        return 1;
    }

    // Correctness: recompute the norm on CPU straight from the same arena
    // bytes the kernel read (arena order, not insertion order) and compare.
    double maxErr = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const auto* vec = reinterpret_cast<const float*>(arena + header_size + i * stride);
        double sum = 0;
        for (uint32_t d = 0; d < dim; ++d) sum += static_cast<double>(vec[d]) * vec[d];
        const double expect = std::sqrt(sum);
        maxErr = std::max({maxErr, std::abs(expect - static_cast<double>(normsZC[i])),
                           std::abs(expect - static_cast<double>(normsCopy[i]))});
    }

    std::printf("zero-copy (newBufferWithBytesNoCopy): %8.3f ms  (%.1f ns/record)\n",
                msZC, msZC * 1e6 / count);
    std::printf("copy      (newBufferWithBytes):       %8.3f ms  (%.1f ns/record)\n",
                msCopy, msCopy * 1e6 / count);
    std::printf("speedup: %.2fx   max |GPU - CPU ref| error: %.6g\n", msCopy / msZC, maxErr);

    aamt_metal_free(ctx);
    aamt_close(f);
    return maxErr < 1e-2 ? 0 : 1;
}
