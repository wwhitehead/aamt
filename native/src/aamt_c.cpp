// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
#include "aamt/aamt.h"

#include "core.hpp"
#include "rs_gf16.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

using aamt::Builder;
using aamt::Reader;

struct aamt_file {
    std::unique_ptr<Reader> reader;
};
struct aamt_builder {
    std::unique_ptr<Builder> builder;
};
struct aamt_journal {
    FILE* f;
};

static void setErr(char* errbuf, size_t errlen, const std::string& msg) {
    if (errbuf && errlen > 0) {
        std::snprintf(errbuf, errlen, "%s", msg.c_str());
    }
}

extern "C" {

aamt_file* aamt_open(const char* path, char* errbuf, size_t errlen) {
    std::string err;
    auto r = Reader::open(path, err);
    if (!r) {
        setErr(errbuf, errlen, err);
        return nullptr;
    }
    return new aamt_file{std::move(r)};
}

void aamt_close(aamt_file* f) { delete f; }

uint64_t aamt_node_count(const aamt_file* f) { return f->reader->header().node_count; }
uint64_t aamt_record_count(const aamt_file* f) { return f->reader->header().record_count; }
uint64_t aamt_content_hash(const aamt_file* f) { return f->reader->header().content_hash; }
uint32_t aamt_max_depth(const aamt_file* f) { return f->reader->header().max_depth; }

int32_t aamt_query(const aamt_file* f, const float tera[4],
                   aamt_record_view* out, int32_t capacity, aamt_route* route) {
    const aamt::RouteResult r = f->reader->route(tera);
    const uint32_t cap = capacity > 0 ? static_cast<uint32_t>(capacity) : 0;
    std::vector<aamt::RecordView> views(cap);
    const uint32_t total = f->reader->records(r, views.data(), cap);
    const uint32_t emitted = total < cap ? total : cap;
    for (uint32_t i = 0; i < emitted; ++i) {
        out[i].id = views[i].external_id;
        std::memcpy(out[i].tera, views[i].tera, sizeof(out[i].tera));
        out[i].data = views[i].data;
        out[i].size = views[i].size;
    }
    if (route) {
        route->path = r.path;
        route->depth = r.depth;
        route->record_count = total;
    }
    return static_cast<int32_t>(emitted);
}

int32_t aamt_query_soft(const aamt_file* f, const float tera[4], uint32_t k,
                       aamt_soft_branch* out_branches,
                       aamt_record_view* out_records, uint32_t per_branch_capacity) {
    const std::vector<aamt::SoftBranch> branches = f->reader->routeSoftTopK(tera, k);
    for (size_t i = 0; i < branches.size(); ++i) {
        const auto& b = branches[i];
        out_branches[i].weight = b.weight;
        out_branches[i].meji = b.meji;
        out_branches[i].route.path = b.route.path;
        out_branches[i].route.depth = b.route.depth;

        std::vector<aamt::RecordView> views(per_branch_capacity);
        const uint32_t total = f->reader->records(b.route, views.data(), per_branch_capacity);
        out_branches[i].route.record_count = total;

        const uint32_t emitted = total < per_branch_capacity ? total : per_branch_capacity;
        aamt_record_view* dst = out_records + i * per_branch_capacity;
        for (uint32_t j = 0; j < emitted; ++j) {
            dst[j].id = views[j].external_id;
            std::memcpy(dst[j].tera, views[j].tera, sizeof(dst[j].tera));
            dst[j].data = views[j].data;
            dst[j].size = views[j].size;
        }
    }
    return static_cast<int32_t>(branches.size());
}

const uint8_t* aamt_payload_arena(const aamt_file* f, uint64_t* out_size) {
    if (out_size) *out_size = f->reader->payloadArenaSize();
    return f->reader->payloadArena();
}

int64_t aamt_all_records(const aamt_file* f, aamt_record_view* out, int64_t capacity) {
    const uint64_t cap = capacity > 0 ? static_cast<uint64_t>(capacity) : 0;
    std::vector<aamt::RecordView> views(cap);
    const uint64_t total = f->reader->allRecords(views.data(), cap);
    const uint64_t emitted = total < cap ? total : cap;
    for (uint64_t i = 0; i < emitted; ++i) {
        out[i].id = views[i].external_id;
        std::memcpy(out[i].tera, views[i].tera, sizeof(out[i].tera));
        out[i].data = views[i].data;
        out[i].size = views[i].size;
    }
    return static_cast<int64_t>(total);
}

void aamt_odu_profile(const aamt_file* f, uint32_t* out_counts, float* out_centroids) {
    f->reader->oduProfile(out_counts, out_centroids);
}

int64_t aamt_all_records_columnar(const aamt_file* f,
                                  uint64_t* out_ids, float* out_tera,
                                  uint8_t* out_payload, uint64_t payload_capacity,
                                  uint64_t* out_offsets, uint32_t* out_sizes,
                                  int64_t capacity) {
    const uint64_t cap = capacity > 0 ? static_cast<uint64_t>(capacity) : 0;
    const uint64_t written = f->reader->allRecordsColumnar(
        out_ids, out_tera, out_payload, payload_capacity, out_offsets, out_sizes, cap);
    return static_cast<int64_t>(written);
}

void aamt_verify_chirality(const aamt_file* f, aamt_chirality_report* out_report,
                          uint64_t* out_inconsistent_ids, uint64_t capacity) {
    const aamt::Reader::ChiralityReport r = f->reader->verifyChirality();
    if (out_report) {
        out_report->primary_count = r.primary_count;
        out_report->mirror_count = r.mirror_count;
        out_report->consistent = r.consistent;
        out_report->inconsistent_count = r.inconsistent_ids.size();
    }
    if (out_inconsistent_ids && capacity > 0) {
        const uint64_t n = std::min<uint64_t>(capacity, r.inconsistent_ids.size());
        std::memcpy(out_inconsistent_ids, r.inconsistent_ids.data(), n * sizeof(uint64_t));
    }
}

int64_t aamt_recover_record(const aamt_file* f, uint64_t id, const float tera_hint[4],
                            uint8_t* out_data, uint64_t out_capacity, float out_tera[4]) {
    std::vector<uint8_t> data;
    if (!f->reader->recoverRecord(id, tera_hint, &data, out_tera)) return -1;
    const uint64_t n = std::min<uint64_t>(out_capacity, data.size());
    std::memcpy(out_data, data.data(), n);
    return static_cast<int64_t>(data.size());
}

int64_t aamt_read_verified(const aamt_file* f, uint64_t id, const float tera_hint[4],
                           uint8_t* out_data, uint64_t out_capacity, float out_tera[4],
                           uint32_t* out_status, uint32_t* out_corrected_symbols) {
    const Reader::ReadResult r = f->reader->readVerified(id, tera_hint);
    if (out_status) *out_status = static_cast<uint32_t>(r.status);
    if (out_corrected_symbols) *out_corrected_symbols = r.correctedSymbols;
    if (r.status == Reader::ReadStatus::kUnavailable) return -1;
    if (out_tera) std::memcpy(out_tera, r.tera, sizeof(r.tera));
    const uint64_t n = std::min<uint64_t>(out_capacity, r.data.size());
    std::memcpy(out_data, r.data.data(), n);
    return static_cast<int64_t>(r.data.size());
}

int32_t aamt_query_verified(const aamt_file* f, const float tera[4], int32_t capacity,
                            aamt_verified_meta* out_meta,
                            uint8_t* out_payload, uint64_t payload_capacity,
                            uint64_t* out_offsets, uint32_t* out_sizes,
                            aamt_route* route) {
    const uint32_t cap32 = capacity > 0 ? static_cast<uint32_t>(capacity) : 0;
    const Reader::VerifiedQueryResult r = f->reader->queryVerified(tera, cap32);
    if (route) {
        route->path = r.path;
        route->depth = r.depth;
        route->record_count = r.primaryCount;
    }
    uint64_t payload_used = 0;
    int32_t emitted = 0;
    for (const auto& vr : r.records) {
        if (emitted >= capacity) break;
        const auto& data = vr.read.data;
        if (payload_used + data.size() <= payload_capacity) {
            std::memcpy(out_payload + payload_used, data.data(), data.size());
        }
        out_meta[emitted].id = vr.id;
        std::memcpy(out_meta[emitted].tera, vr.read.tera, sizeof(vr.read.tera));
        out_meta[emitted].status = static_cast<uint32_t>(vr.read.status);
        out_meta[emitted].corrected_symbols = vr.read.correctedSymbols;
        out_offsets[emitted] = payload_used;
        out_sizes[emitted] = static_cast<uint32_t>(data.size());
        payload_used += data.size();
        ++emitted;
    }
    return emitted;
}

int32_t aamt_query_verified_soft(const aamt_file* f, const float tera[4], uint32_t k,
                                 uint32_t per_branch_capacity,
                                 aamt_soft_branch* out_branches,
                                 uint32_t* out_branch_counts,
                                 aamt_verified_meta* out_meta,
                                 uint8_t* out_payload, uint64_t payload_capacity,
                                 uint64_t* out_offsets, uint32_t* out_sizes) {
    const auto branches = f->reader->queryVerifiedSoft(tera, k, per_branch_capacity);
    uint64_t payload_used = 0;
    for (size_t i = 0; i < branches.size(); ++i) {
        const auto& vb = branches[i];
        out_branches[i].weight = vb.weight;
        out_branches[i].meji = vb.meji;
        out_branches[i].route.path = vb.path;
        out_branches[i].route.depth = vb.depth;
        out_branches[i].route.record_count = vb.primaryCount;

        const uint32_t stride = per_branch_capacity;
        uint32_t emitted = 0;
        for (const auto& vr : vb.records) {
            if (emitted >= stride) break;
            const auto& data = vr.read.data;
            const size_t idx = i * stride + emitted;
            if (payload_used + data.size() <= payload_capacity) {
                std::memcpy(out_payload + payload_used, data.data(), data.size());
            }
            out_meta[idx].id = vr.id;
            std::memcpy(out_meta[idx].tera, vr.read.tera, sizeof(vr.read.tera));
            out_meta[idx].status = static_cast<uint32_t>(vr.read.status);
            out_meta[idx].corrected_symbols = vr.read.correctedSymbols;
            out_offsets[idx] = payload_used;
            out_sizes[idx] = static_cast<uint32_t>(data.size());
            payload_used += data.size();
            ++emitted;
        }
        out_branch_counts[i] = emitted;
    }
    return static_cast<int32_t>(branches.size());
}

aamt_builder* aamt_builder_new(uint32_t max_depth, uint32_t bucket_capacity) {
    return new aamt_builder{std::make_unique<Builder>(max_depth, bucket_capacity)};
}

int aamt_builder_add(aamt_builder* b, uint64_t id, const float tera[4],
                     const void* data, uint32_t size) {
    b->builder->add(id, tera, data, size);
    return 0;
}

int aamt_builder_add_mirrored(aamt_builder* b, uint64_t id, const float tera[4],
                              const void* data, uint32_t size) {
    b->builder->addMirrored(id, tera, data, size);
    return 0;
}

int aamt_builder_add_ecc(aamt_builder* b, uint64_t id, const float tera[4],
                         const void* data, uint32_t size) {
    b->builder->addWithECC(id, tera, data, size);
    return 0;
}

int aamt_builder_add_mirrored_ecc(aamt_builder* b, uint64_t id, const float tera[4],
                                  const void* data, uint32_t size) {
    b->builder->addMirroredWithECC(id, tera, data, size);
    return 0;
}

uint32_t aamt_encode_ecc_payload(const uint8_t* data, uint32_t size,
                                 uint8_t* out_encoded, uint32_t out_capacity) {
    const auto encoded = aamt::rs15_13::encodePayload(data, size);
    if (out_encoded && out_capacity >= encoded.size()) {
        std::memcpy(out_encoded, encoded.data(), encoded.size());
    }
    return static_cast<uint32_t>(encoded.size());
}

int64_t aamt_decode_ecc_payload(const uint8_t* encoded, uint32_t encoded_size,
                                uint8_t* out_data, uint64_t out_capacity,
                                uint32_t* out_corrected_symbols) {
    const auto result = aamt::rs15_13::decodePayload(encoded, encoded_size);
    if (out_corrected_symbols) *out_corrected_symbols = result.correctedSymbols;
    if (!result.ok) return -1;
    const uint64_t n = std::min<uint64_t>(out_capacity, result.data.size());
    std::memcpy(out_data, result.data.data(), n);
    return static_cast<int64_t>(result.data.size());
}

int aamt_builder_write(aamt_builder* b, const char* path, char* errbuf, size_t errlen) {
    std::string err;
    if (!b->builder->write(path, err)) {
        setErr(errbuf, errlen, err);
        return -1;
    }
    return 0;
}

void aamt_builder_free(aamt_builder* b) { delete b; }

aamt_journal* aamt_journal_open(const char* path) {
    FILE* f = std::fopen(path, "ab");
    if (!f) return nullptr;
    return new aamt_journal{f};
}

int aamt_journal_log(aamt_journal* j, const aamt_file* f,
                     const float tera[4], const aamt_route* route) {
    aamt::JournalRecord rec;
    std::memset(&rec, 0, sizeof(rec));
    rec.t_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    rec.file_hash = f->reader->header().content_hash;
    std::memcpy(rec.tera, tera, sizeof(rec.tera));
    rec.path = route->path;
    rec.depth = route->depth;
    if (std::fwrite(&rec, sizeof(rec), 1, j->f) != 1) return -1;
    // Flush per entry: a lineage journal that silently loses its tail on
    // crash (or reads short mid-session, e.g. live verify_journal) defeats
    // its purpose. 40 bytes/entry at route() rates makes this cheap.
    return std::fflush(j->f) == 0 ? 0 : -1;
}

void aamt_journal_close(aamt_journal* j) {
    if (!j) return;
    std::fclose(j->f);
    delete j;
}

} // extern "C"
