// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// aamt — pack / inspect / query / bench for .aamt substrate files.
#include "aamt/aamt.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

int usage() {
    std::fprintf(stderr,
        "aamt — .aamt substrate tool (WP-23)\n"
        "  aamt pack <in.tsv> <out.aamt> [max_depth=8] [bucket=32]\n"
        "       TSV columns: id \\t T \\t E \\t R \\t A \\t payload-text\n"
        "  aamt inspect <file.aamt>\n"
        "  aamt query <file.aamt> <T> <E> <R> <A>\n"
        "  aamt bench <file.aamt> [queries=1000000]\n");
    return 2;
}

std::string oduWord(uint32_t path, uint32_t depth) {
    std::ostringstream os;
    for (int i = static_cast<int>(depth) - 1; i >= 0; --i) {
        os << ((path >> (4 * i)) & 0xF);
        if (i > 0) os << ".";
    }
    return os.str();
}

int cmdPack(int argc, char** argv) {
    if (argc < 4) return usage();
    const char* in = argv[2];
    const char* out = argv[3];
    const uint32_t max_depth = argc > 4 ? static_cast<uint32_t>(std::atoi(argv[4])) : 8;
    const uint32_t bucket = argc > 5 ? static_cast<uint32_t>(std::atoi(argv[5])) : 32;

    std::ifstream f(in);
    if (!f) {
        std::fprintf(stderr, "cannot read %s\n", in);
        return 1;
    }
    aamt_builder* b = aamt_builder_new(max_depth, bucket);
    std::string line;
    uint64_t rows = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string col;
        uint64_t id = 0;
        float tera[4] = {0, 0, 0, 0};
        if (!std::getline(ss, col, '\t')) continue;
        id = std::strtoull(col.c_str(), nullptr, 10);
        bool ok = true;
        for (int d = 0; d < 4 && ok; ++d) {
            ok = static_cast<bool>(std::getline(ss, col, '\t'));
            if (ok) tera[d] = std::strtof(col.c_str(), nullptr);
        }
        if (!ok) continue;
        std::string payload;
        std::getline(ss, payload); // rest of line, may contain tabs
        aamt_builder_add(b, id, tera, payload.data(),
                         static_cast<uint32_t>(payload.size()));
        ++rows;
    }
    char err[256] = {0};
    const auto t0 = std::chrono::steady_clock::now();
    if (aamt_builder_write(b, out, err, sizeof(err)) != 0) {
        std::fprintf(stderr, "pack failed: %s\n", err);
        aamt_builder_free(b);
        return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    aamt_builder_free(b);
    std::printf("packed %" PRIu64 " records -> %s (%.1f ms)\n", rows, out,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
    return 0;
}

int cmdInspect(int argc, char** argv) {
    if (argc < 3) return usage();
    char err[256] = {0};
    aamt_file* f = aamt_open(argv[2], err, sizeof(err));
    if (!f) {
        std::fprintf(stderr, "open failed: %s\n", err);
        return 1;
    }
    std::printf("file:          %s\n", argv[2]);
    std::printf("nodes:         %" PRIu64 " (64 B each)\n", aamt_node_count(f));
    std::printf("records:       %" PRIu64 "\n", aamt_record_count(f));
    std::printf("max depth:     %u\n", aamt_max_depth(f));
    std::printf("content hash:  %016" PRIx64 "\n", aamt_content_hash(f));
    aamt_close(f);
    return 0;
}

int cmdQuery(int argc, char** argv) {
    if (argc < 7) return usage();
    char err[256] = {0};
    aamt_file* f = aamt_open(argv[2], err, sizeof(err));
    if (!f) {
        std::fprintf(stderr, "open failed: %s\n", err);
        return 1;
    }
    float tera[4];
    for (int d = 0; d < 4; ++d) tera[d] = std::strtof(argv[3 + d], nullptr);
    std::vector<aamt_record_view> views(64);
    aamt_route route;
    const int32_t n = aamt_query(f, tera, views.data(),
                                 static_cast<int32_t>(views.size()), &route);
    std::printf("route: odu-word=%s depth=%u bucket=%u\n",
                oduWord(route.path, route.depth).c_str(), route.depth,
                route.record_count);
    for (int32_t i = 0; i < n; ++i) {
        std::printf("  id=%" PRIu64 " tera=(%.3f %.3f %.3f %.3f) %.*s\n",
                    views[i].id, views[i].tera[0], views[i].tera[1],
                    views[i].tera[2], views[i].tera[3],
                    static_cast<int>(views[i].size > 60 ? 60 : views[i].size),
                    reinterpret_cast<const char*>(views[i].data));
    }
    aamt_close(f);
    return 0;
}

int cmdBench(int argc, char** argv) {
    if (argc < 3) return usage();
    const uint64_t n = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 1000000ull;

    const auto o0 = std::chrono::steady_clock::now();
    char err[256] = {0};
    aamt_file* f = aamt_open(argv[2], err, sizeof(err));
    const auto o1 = std::chrono::steady_clock::now();
    if (!f) {
        std::fprintf(stderr, "open failed: %s\n", err);
        return 1;
    }
    std::printf("open (mmap + validate): %.3f ms\n",
                std::chrono::duration<double, std::milli>(o1 - o0).count());

    std::mt19937 rng(0x5EED);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::vector<float> teras(n * 4);
    for (auto& v : teras) v = uni(rng);

    std::vector<aamt_record_view> views(8);
    aamt_route route;
    uint64_t checksum = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < n; ++i) {
        const int32_t k = aamt_query(f, &teras[i * 4], views.data(), 8, &route);
        checksum += route.record_count + static_cast<uint64_t>(k);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns =
        std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(n);
    std::printf("%" PRIu64 " queries: %.0f ns/query (%.2f Mq/s)  [checksum %" PRIu64 "]\n",
                n, ns, 1000.0 / ns, checksum);
    aamt_close(f);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    if (std::strcmp(argv[1], "pack") == 0) return cmdPack(argc, argv);
    if (std::strcmp(argv[1], "inspect") == 0) return cmdInspect(argc, argv);
    if (std::strcmp(argv[1], "query") == 0) return cmdQuery(argc, argv);
    if (std::strcmp(argv[1], "bench") == 0) return cmdBench(argc, argv);
    return usage();
}
