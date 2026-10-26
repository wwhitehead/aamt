// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
#include "core.hpp"
#include "rs_gf16.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <deque>
#include <optional>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace aamt {

// ---------------------------------------------------------------- Builder

Builder::Builder(uint32_t max_depth, uint32_t bucket_capacity)
    : max_depth_(max_depth == 0 ? 1 : (max_depth > kMaxDepthCap ? kMaxDepthCap : max_depth)),
      bucket_capacity_(bucket_capacity == 0 ? 1 : bucket_capacity) {}

namespace {
void clampTera4(const float in[4], float out[4]) {
    for (int d = 0; d < 4; ++d) {
        float v = in[d];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        out[d] = v;
    }
}
} // namespace

void Builder::add(uint64_t id, const float tera[4], const void* data, uint32_t size) {
    OwnedRecord r;
    r.id = id;
    clampTera4(tera, r.tera);
    r.data.assign(static_cast<const uint8_t*>(data),
                  static_cast<const uint8_t*>(data) + size);
    r.flags = 0;
    records_.push_back(std::move(r));
}

void Builder::addMirrored(uint64_t id, const float tera[4], const void* data, uint32_t size) {
    float clamped[4];
    clampTera4(tera, clamped);
    add(id, clamped, data, size); // primary, flags = 0

    OwnedRecord mirror;
    mirror.id = id;
    householderReflectT(clamped, mirror.tera); // routing key AND stored tera
    mirror.data.assign(static_cast<const uint8_t*>(data),
                       static_cast<const uint8_t*>(data) + size);
    mirror.flags = kRecordMirror;
    records_.push_back(std::move(mirror));
}

void Builder::addWithECC(uint64_t id, const float tera[4], const void* data, uint32_t size) {
    const auto encoded = rs15_13::encodePayload(static_cast<const uint8_t*>(data), size);
    OwnedRecord r;
    r.id = id;
    clampTera4(tera, r.tera);
    r.data = encoded;
    r.flags = kRecordRS;
    records_.push_back(std::move(r));
}

void Builder::addMirroredWithECC(uint64_t id, const float tera[4], const void* data, uint32_t size) {
    float clamped[4];
    clampTera4(tera, clamped);
    const auto encoded = rs15_13::encodePayload(static_cast<const uint8_t*>(data), size);

    OwnedRecord primary;
    primary.id = id;
    std::memcpy(primary.tera, clamped, sizeof(clamped));
    primary.data = encoded;
    primary.flags = kRecordRS;
    records_.push_back(std::move(primary));

    OwnedRecord mirror;
    mirror.id = id;
    householderReflectT(clamped, mirror.tera); // routing key AND stored tera
    mirror.data = encoded; // same RS-encoded bytes, stored at the reflected leaf
    mirror.flags = kRecordMirror | kRecordRS;
    records_.push_back(std::move(mirror));
}

void Builder::insert(BuildNode* n, float lo[4], float hi[4], size_t recIdx) {
    while (!n->leaf) {
        const uint32_t m = mejiIndex(records_[recIdx].tera, lo, hi);
        if (!n->child[m]) {
            auto c = std::make_unique<BuildNode>();
            c->depth = n->depth + 1;
            c->path = (n->path << 4) | m;
            n->child[m] = std::move(c);
            n->mask |= static_cast<uint16_t>(1u << m);
        }
        childBox(m, lo, hi);
        n = n->child[m].get();
    }
    n->records.push_back(recIdx);
    if (n->records.size() > bucket_capacity_ && n->depth < max_depth_) split(n, lo, hi);
}

void Builder::split(BuildNode* n, const float lo[4], const float hi[4]) {
    n->leaf = false;
    std::vector<size_t> recs = std::move(n->records);
    n->records.clear();
    for (size_t idx : recs) {
        const uint32_t m = mejiIndex(records_[idx].tera, lo, hi);
        if (!n->child[m]) {
            auto c = std::make_unique<BuildNode>();
            c->depth = n->depth + 1;
            c->path = (n->path << 4) | m;
            n->child[m] = std::move(c);
            n->mask |= static_cast<uint16_t>(1u << m);
        }
        n->child[m]->records.push_back(idx);
    }
    for (uint32_t m = 0; m < 16; ++m) {
        BuildNode* c = n->child[m].get();
        if (!c) continue;
        if (c->records.size() > bucket_capacity_ && c->depth < max_depth_) {
            float clo[4], chi[4];
            std::memcpy(clo, lo, sizeof(clo));
            std::memcpy(chi, hi, sizeof(chi));
            childBox(m, clo, chi);
            split(c, clo, chi);
        }
    }
}

bool Builder::write(const std::string& path, std::string& err) {
    BuildNode root;
    for (size_t i = 0; i < records_.size(); ++i) {
        float lo[4] = {0.f, 0.f, 0.f, 0.f};
        float hi[4] = {1.f, 1.f, 1.f, 1.f};
        insert(&root, lo, hi, i);
    }

    // Count nodes.
    size_t total = 0;
    {
        std::deque<const BuildNode*> q{&root};
        while (!q.empty()) {
            const BuildNode* n = q.front();
            q.pop_front();
            ++total;
            for (const auto& c : n->child)
                if (c) q.push_back(c.get());
        }
    }

    // BFS serialize: children of each node land in contiguous slots.
    std::vector<Node> nodes(total);
    std::memset(nodes.data(), 0, nodes.size() * sizeof(Node));
    std::vector<uint8_t> payload;
    {
        struct QItem { const BuildNode* bn; uint64_t idx; };
        std::deque<QItem> q{{&root, 0}};
        uint64_t next_free = 1;
        while (!q.empty()) {
            const QItem it = q.front();
            q.pop_front();
            const BuildNode* bn = it.bn;
            Node& out = nodes[it.idx];
            out.coordinate_id = bn->path;
            out.chirality_spin = 0.0f;
            out.flags = static_cast<uint16_t>((bn->depth << 8) | (bn->leaf ? kNodeLeaf : 0));
            if (bn->leaf) {
                out.record_count = static_cast<uint32_t>(bn->records.size());
                out.payload_offset = payload.size();
                for (size_t idx : bn->records) {
                    const OwnedRecord& r = records_[idx];
                    RecordHeader rh;
                    std::memset(&rh, 0, sizeof(rh));
                    rh.external_id = r.id;
                    std::memcpy(rh.tera, r.tera, sizeof(rh.tera));
                    rh.size = static_cast<uint32_t>(r.data.size());
                    rh.flags = r.flags;
                    const uint8_t* p = reinterpret_cast<const uint8_t*>(&rh);
                    payload.insert(payload.end(), p, p + sizeof(rh));
                    payload.insert(payload.end(), r.data.begin(), r.data.end());
                    payload.resize(alignUp(payload.size(), 8), 0);
                }
                out.payload_size = payload.size() - out.payload_offset;
            } else {
                out.child_mask = bn->mask;
                out.children_base = next_free;
                for (uint32_t m = 0; m < 16; ++m)
                    if (bn->child[m]) q.push_back({bn->child[m].get(), next_free++});
            }
        }
    }

    FileHeader hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    hdr.magic = kMagic;
    hdr.version_major = kVersionMajor;
    hdr.version_minor = kVersionMinor;
    hdr.node_arena_offset = sizeof(FileHeader);
    hdr.node_count = nodes.size();
    hdr.payload_arena_offset =
        alignUp(hdr.node_arena_offset + nodes.size() * sizeof(Node), kPageSize);
    hdr.payload_arena_size = payload.size();
    hdr.record_count = records_.size();
    hdr.max_depth = max_depth_;
    hdr.bucket_capacity = bucket_capacity_;
    hdr.file_size = hdr.payload_arena_offset + hdr.payload_arena_size;

    uint64_t h = fnv1a64(reinterpret_cast<const uint8_t*>(nodes.data()),
                         nodes.size() * sizeof(Node));
    h = fnv1a64(payload.data(), payload.size(), h);
    hdr.content_hash = h;

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        err = "cannot open for write: " + path;
        return false;
    }
    bool ok = std::fwrite(&hdr, sizeof(hdr), 1, f) == 1;
    ok = ok && (nodes.empty() ||
                std::fwrite(nodes.data(), sizeof(Node), nodes.size(), f) == nodes.size());
    // zero-fill up to the page-aligned payload arena
    if (ok) {
        uint64_t pos = hdr.node_arena_offset + nodes.size() * sizeof(Node);
        static const uint8_t zeros[512] = {0};
        while (ok && pos < hdr.payload_arena_offset) {
            const uint64_t chunk = std::min<uint64_t>(sizeof(zeros), hdr.payload_arena_offset - pos);
            ok = std::fwrite(zeros, 1, chunk, f) == chunk;
            pos += chunk;
        }
    }
    ok = ok && (payload.empty() ||
                std::fwrite(payload.data(), 1, payload.size(), f) == payload.size());
    ok = std::fclose(f) == 0 && ok;
    if (!ok) err = "short write: " + path;
    return ok;
}

// ----------------------------------------------------------------- Reader

Reader::~Reader() {
    if (map_ && map_ != MAP_FAILED) ::munmap(map_, map_size_);
    if (fd_ >= 0) ::close(fd_);
}

std::unique_ptr<Reader> Reader::open(const std::string& path, std::string& err) {
    std::unique_ptr<Reader> r(new Reader());
    r->fd_ = ::open(path.c_str(), O_RDONLY);
    if (r->fd_ < 0) {
        err = "cannot open: " + path;
        return nullptr;
    }
    struct stat st;
    if (::fstat(r->fd_, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(FileHeader))) {
        err = "not an .aamt file (too small): " + path;
        return nullptr;
    }
    r->map_size_ = static_cast<size_t>(st.st_size);
    r->map_ = ::mmap(nullptr, r->map_size_, PROT_READ, MAP_SHARED, r->fd_, 0);
    if (r->map_ == MAP_FAILED) {
        err = "mmap failed: " + path;
        r->map_ = nullptr;
        return nullptr;
    }
    const auto* hdr = static_cast<const FileHeader*>(r->map_);
    if (hdr->magic != kMagic) {
        err = "bad magic";
        return nullptr;
    }
    if (hdr->version_major != kVersionMajor) {
        err = "unsupported version";
        return nullptr;
    }
    if (hdr->file_size != r->map_size_ ||
        hdr->node_arena_offset + hdr->node_count * sizeof(Node) > r->map_size_ ||
        hdr->payload_arena_offset + hdr->payload_arena_size > r->map_size_) {
        err = "corrupt header (sizes)";
        return nullptr;
    }
    r->hdr_ = hdr;
    r->nodes_ = reinterpret_cast<const Node*>(
        static_cast<const uint8_t*>(r->map_) + hdr->node_arena_offset);
    r->payload_ = static_cast<const uint8_t*>(r->map_) + hdr->payload_arena_offset;
    return r;
}

RouteResult Reader::route(const float tera[4]) const {
    RouteResult res;
    if (hdr_->node_count == 0) return res;
    const Node* n = &nodes_[0];
    float lo[4] = {0.f, 0.f, 0.f, 0.f};
    float hi[4] = {1.f, 1.f, 1.f, 1.f};
    while (!nodeIsLeaf(*n)) {
        const uint32_t m = mejiIndex(tera, lo, hi);
        res.path = (res.path << 4) | m;
        res.depth += 1;
        if (!(n->child_mask & (1u << m))) return res; // empty orthant: leaf stays null
        const uint32_t slot =
            static_cast<uint32_t>(__builtin_popcount(n->child_mask & ((1u << m) - 1u)));
        childBox(m, lo, hi);
        n = &nodes_[n->children_base + slot];
    }
    res.leaf = n;
    return res;
}

std::vector<SoftBranch> Reader::routeSoftTopK(const float tera[4], uint32_t k) const {
    std::vector<SoftBranch> out;
    if (hdr_->node_count == 0) return out;

    constexpr float kRootLo[4] = {0.f, 0.f, 0.f, 0.f};
    constexpr float kRootHi[4] = {1.f, 1.f, 1.f, 1.f};

    std::array<std::pair<float, uint32_t>, 16> ranked;
    for (uint32_t m = 0; m < 16; ++m) ranked[m] = {mejiWeight(tera, kRootLo, kRootHi, m), m};

    // mejiIndex's argmax IS the same weight ranking's rank-0 pick in every
    // generic case; they can only disagree on an exact weight tie (e.g. the
    // TERA cube's center, where all 16 orthants weigh 1/16). mejiIndex
    // resolves such ties via its ">=" rule (the Ox chiral tie-break, WP-23
    // §2); the sort below must honor the same pick, or routeSoftTopK's
    // rank-0 branch could silently disagree with route()'s hard result.
    const uint32_t hardM = mejiIndex(tera, kRootLo, kRootHi);
    std::stable_sort(ranked.begin(), ranked.end(), [hardM](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        if (a.second == hardM) return true;
        if (b.second == hardM) return false;
        return a.second < b.second;
    });

    const uint32_t take = k < 16 ? k : 16;
    const Node& root = nodes_[0];
    out.reserve(take);

    for (uint32_t i = 0; i < take; ++i) {
        SoftBranch b;
        b.weight = ranked[i].first;
        b.meji = ranked[i].second;
        b.route.path = b.meji;
        b.route.depth = 1;

        if (root.child_mask & (1u << b.meji)) {
            float lo[4], hi[4];
            std::memcpy(lo, kRootLo, sizeof(lo));
            std::memcpy(hi, kRootHi, sizeof(hi));
            childBox(b.meji, lo, hi);
            const uint32_t slot = static_cast<uint32_t>(
                __builtin_popcount(root.child_mask & ((1u << b.meji) - 1u)));
            const Node* n = &nodes_[root.children_base + slot];
            while (!nodeIsLeaf(*n)) {
                const uint32_t m = mejiIndex(tera, lo, hi);
                b.route.path = (b.route.path << 4) | m;
                b.route.depth += 1;
                if (!(n->child_mask & (1u << m))) break; // empty deeper orthant
                const uint32_t slot2 = static_cast<uint32_t>(
                    __builtin_popcount(n->child_mask & ((1u << m) - 1u)));
                childBox(m, lo, hi);
                n = &nodes_[n->children_base + slot2];
            }
            if (nodeIsLeaf(*n)) b.route.leaf = n;
        }
        // else: root has no child at this Meji — an empty top-level orthant,
        // reported with weight and no leaf (still a valid, ranked branch).

        out.push_back(b);
    }
    return out;
}

namespace {
// Appends up to (capacity - emitted) views from one leaf's bucket into out,
// starting at out[emitted]; advances and returns the new emitted count. The
// leaf's own record_count is added to *total regardless of capacity.
uint32_t extractLeaf(const uint8_t* payload_base, const Node& leaf, RecordView* out,
                     uint32_t capacity, uint32_t emitted) {
    const uint8_t* p = payload_base + leaf.payload_offset;
    const uint8_t* end = p + leaf.payload_size;
    for (uint32_t i = 0; i < leaf.record_count && p + sizeof(RecordHeader) <= end; ++i) {
        const auto* rh = reinterpret_cast<const RecordHeader*>(p);
        if (emitted < capacity) {
            out[emitted].external_id = rh->external_id;
            out[emitted].tera = rh->tera;
            out[emitted].data = p + sizeof(RecordHeader);
            out[emitted].size = rh->size;
            ++emitted;
        }
        p += recordStride(rh->size);
    }
    return emitted;
}
} // namespace

uint32_t Reader::records(const RouteResult& r, RecordView* out, uint32_t capacity) const {
    if (!r.leaf) return 0;
    extractLeaf(payload_, *r.leaf, out, capacity, 0);
    return r.leaf->record_count;
}

uint64_t Reader::allRecords(RecordView* out, uint64_t capacity) const {
    uint64_t total = 0;
    uint32_t emitted = 0;
    const uint32_t cap32 = capacity > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<uint32_t>(capacity);
    for (uint64_t i = 0; i < hdr_->node_count; ++i) {
        const Node& n = nodes_[i];
        if (!nodeIsLeaf(n) || n.record_count == 0) continue;
        total += n.record_count;
        emitted = extractLeaf(payload_, n, out, cap32, emitted);
    }
    return total;
}

void Reader::oduProfile(uint32_t* out_counts, float* out_centroids) const {
    std::memset(out_counts, 0, 256 * sizeof(uint32_t));
    // Accumulate sums in double to keep centroid means exact-ish for large
    // cells, then divide down into the float output.
    std::array<std::array<double, 4>, 256> sums{};

    for (uint64_t i = 0; i < hdr_->node_count; ++i) {
        const Node& n = nodes_[i];
        if (!nodeIsLeaf(n) || n.record_count == 0) continue;
        const uint8_t* p = payload_ + n.payload_offset;
        const uint8_t* end = p + n.payload_size;
        for (uint32_t j = 0; j < n.record_count && p + sizeof(RecordHeader) <= end; ++j) {
            const auto* rh = reinterpret_cast<const RecordHeader*>(p);
            // Depth-2 cell of this record, tree convention (m1<<4)|m2, with
            // the same >= boundary rule (Ox) the router applies: b1 at 0.5,
            // b2 at 0.25 or 0.75 depending on b1's half.
            uint32_t m1 = 0, m2 = 0;
            for (int d = 0; d < 4; ++d) {
                const float t = rh->tera[d];
                const bool b1 = t >= 0.5f;
                const bool b2 = t >= (b1 ? 0.75f : 0.25f);
                if (b1) m1 |= (1u << d);
                if (b2) m2 |= (1u << d);
            }
            const uint32_t cell = (m1 << 4) | m2;
            out_counts[cell] += 1;
            for (int d = 0; d < 4; ++d) sums[cell][d] += rh->tera[d];
            p += recordStride(rh->size);
        }
    }

    for (uint32_t c = 0; c < 256; ++c) {
        for (int d = 0; d < 4; ++d) {
            out_centroids[c * 4 + d] =
                out_counts[c] ? static_cast<float>(sums[c][d] / out_counts[c]) : 0.0f;
        }
    }
}

Reader::ChiralityReport Reader::verifyChirality() const {
    ChiralityReport report;

    for (uint64_t i = 0; i < hdr_->node_count; ++i) {
        const Node& n = nodes_[i];
        if (!nodeIsLeaf(n) || n.record_count == 0) continue;
        const uint8_t* p = payload_ + n.payload_offset;
        const uint8_t* end = p + n.payload_size;
        for (uint32_t j = 0; j < n.record_count && p + sizeof(RecordHeader) <= end; ++j) {
            const auto* rh = reinterpret_cast<const RecordHeader*>(p);
            const uint8_t* data = p + sizeof(RecordHeader);

            if (rh->flags & kRecordMirror) {
                ++report.mirror_count;
                p += recordStride(rh->size);
                continue;
            }
            ++report.primary_count;

            float mirror_tera[4];
            householderReflectT(rh->tera, mirror_tera);
            const RouteResult mr = route(mirror_tera); // real descent, not the XOR shortcut

            bool found_consistent = false;
            if (mr.leaf) {
                const uint8_t* mp = payload_ + mr.leaf->payload_offset;
                const uint8_t* mend = mp + mr.leaf->payload_size;
                for (uint32_t k = 0; k < mr.leaf->record_count &&
                                     mp + sizeof(RecordHeader) <= mend; ++k) {
                    const auto* mrh = reinterpret_cast<const RecordHeader*>(mp);
                    if ((mrh->flags & kRecordMirror) && mrh->external_id == rh->external_id) {
                        const uint8_t* mdata = mp + sizeof(RecordHeader);
                        if (mrh->size == rh->size &&
                            std::memcmp(mdata, data, rh->size) == 0) {
                            found_consistent = true;
                        }
                        break; // matching id found; stop regardless of payload outcome
                    }
                    mp += recordStride(mrh->size);
                }
            }

            if (found_consistent) ++report.consistent;
            else report.inconsistent_ids.push_back(rh->external_id);

            p += recordStride(rh->size);
        }
    }
    return report;
}

bool Reader::recoverRecord(uint64_t id, const float tera_hint[4],
                           std::vector<uint8_t>* out_data, float out_tera[4]) const {
    // Try the primary at tera_hint first.
    const RouteResult r = route(tera_hint);
    if (r.leaf) {
        const uint8_t* p = payload_ + r.leaf->payload_offset;
        const uint8_t* end = p + r.leaf->payload_size;
        for (uint32_t i = 0; i < r.leaf->record_count && p + sizeof(RecordHeader) <= end; ++i) {
            const auto* rh = reinterpret_cast<const RecordHeader*>(p);
            if (!(rh->flags & kRecordMirror) && rh->external_id == id) {
                out_data->assign(p + sizeof(RecordHeader), p + sizeof(RecordHeader) + rh->size);
                std::memcpy(out_tera, rh->tera, sizeof(rh->tera));
                return true;
            }
            p += recordStride(rh->size);
        }
    }

    // Primary not found at tera_hint (structural failure) — fall back to
    // the mirror at H_T(tera_hint); its stored tera IS the reflection, so
    // H_T of it recovers the true coordinate exactly.
    float mirror_tera[4];
    householderReflectT(tera_hint, mirror_tera);
    const RouteResult mr = route(mirror_tera);
    if (!mr.leaf) return false;
    const uint8_t* p = payload_ + mr.leaf->payload_offset;
    const uint8_t* end = p + mr.leaf->payload_size;
    for (uint32_t i = 0; i < mr.leaf->record_count && p + sizeof(RecordHeader) <= end; ++i) {
        const auto* rh = reinterpret_cast<const RecordHeader*>(p);
        if ((rh->flags & kRecordMirror) && rh->external_id == id) {
            out_data->assign(p + sizeof(RecordHeader), p + sizeof(RecordHeader) + rh->size);
            householderReflectT(rh->tera, out_tera); // recover true tera
            return true;
        }
        p += recordStride(rh->size);
    }
    return false;
}

namespace {
// Finds the record with the given id and mirror-flag-state in a leaf's
// bucket (nullptr leaf or no match -> nullptr).
const RecordHeader* findInLeaf(const uint8_t* payload_base, const Node* leaf,
                               uint64_t id, bool wantMirror) {
    if (!leaf) return nullptr;
    const uint8_t* p = payload_base + leaf->payload_offset;
    const uint8_t* end = p + leaf->payload_size;
    for (uint32_t i = 0; i < leaf->record_count && p + sizeof(RecordHeader) <= end; ++i) {
        const auto* rh = reinterpret_cast<const RecordHeader*>(p);
        if (static_cast<bool>(rh->flags & kRecordMirror) == wantMirror &&
            rh->external_id == id) {
            return rh;
        }
        p += recordStride(rh->size);
    }
    return nullptr;
}

// Decodes a record's payload if it was RS-encoded (kRecordRS); otherwise
// the "decode" is the identity — raw bytes, zero corrections. Returns
// std::nullopt only when rh itself is null (no record to decode).
std::optional<rs15_13::PayloadDecodeResult> decodeRecord(const RecordHeader* rh) {
    if (!rh) return std::nullopt;
    const uint8_t* data = reinterpret_cast<const uint8_t*>(rh) + sizeof(RecordHeader);
    if (rh->flags & kRecordRS) {
        return rs15_13::decodePayload(data, rh->size);
    }
    rs15_13::PayloadDecodeResult raw;
    raw.ok = true;
    raw.correctedSymbols = 0;
    raw.data.assign(data, data + rh->size);
    return raw;
}
} // namespace

Reader::ReadResult Reader::readVerified(uint64_t id, const float tera_hint[4]) const {
    ReadResult result;

    const RouteResult primaryRoute = route(tera_hint);
    const RecordHeader* primary = findInLeaf(payload_, primaryRoute.leaf, id, /*wantMirror=*/false);

    float mirrorTera[4];
    householderReflectT(tera_hint, mirrorTera);
    const RouteResult mirrorRoute = route(mirrorTera); // real descent, not the XOR shortcut
    const RecordHeader* mirror = findInLeaf(payload_, mirrorRoute.leaf, id, /*wantMirror=*/true);

    const auto primaryDecoded = decodeRecord(primary);
    const auto mirrorDecoded = decodeRecord(mirror);
    const bool primaryOk = primaryDecoded && primaryDecoded->ok;
    const bool mirrorOk = mirrorDecoded && mirrorDecoded->ok;

    if (primaryOk) {
        // A mirror that positively decodes to DIFFERENT bytes is the one
        // case with real, positive evidence of an unresolved disagreement
        // between two independently stored, independently decoded copies
        // — genuinely undecidable with what this format stores (WP-23
        // §10.2). Everything else (mirror absent, mirror itself
        // unreadable, or mirror agrees) is not counter-evidence, so a
        // clean or RS-repaired primary is trusted.
        if (mirrorOk && mirrorDecoded->data != primaryDecoded->data) {
            result.status = ReadStatus::kUnavailable;
            return result;
        }
        result.status = primaryDecoded->correctedSymbols == 0
            ? ReadStatus::kOk : ReadStatus::kOkRsRepaired;
        result.correctedSymbols = primaryDecoded->correctedSymbols;
        result.data = primaryDecoded->data;
        std::memcpy(result.tera, primary->tera, sizeof(result.tera));
        return result;
    }

    if (mirrorOk) {
        // Primary missing or its RS syndrome was unresolvable outright —
        // the same structural/decode fallback recoverRecord already does,
        // generalized to also cover "primary present but RS gave up."
        result.status = ReadStatus::kOkMirrorRecovered;
        result.correctedSymbols = mirrorDecoded->correctedSymbols;
        result.data = mirrorDecoded->data;
        householderReflectT(mirror->tera, result.tera); // recover true tera
        return result;
    }

    result.status = ReadStatus::kUnavailable; // both unreadable — nothing to guess from
    return result;
}

namespace {
// Shared by queryVerified and queryVerifiedSoft: walks a leaf's PRIMARY
// records only, verifying each via readVerified (Reader method pointer
// passed in since this is a free function). Returns the raw primary count
// (before capacity or drops) and appends passing records to *out.
uint32_t verifyLeafPrimaries(const Reader& reader, const uint8_t* payload_base, const Node* leaf,
                             uint32_t capacity, std::vector<Reader::VerifiedRecord>* out) {
    if (!leaf) return 0;
    uint32_t primaryCount = 0;
    const uint8_t* p = payload_base + leaf->payload_offset;
    const uint8_t* end = p + leaf->payload_size;
    for (uint32_t i = 0; i < leaf->record_count && p + sizeof(RecordHeader) <= end; ++i) {
        const auto* rh = reinterpret_cast<const RecordHeader*>(p);
        if (!(rh->flags & kRecordMirror)) {
            ++primaryCount;
            if (out->size() < capacity) {
                Reader::VerifiedRecord vr;
                vr.id = rh->external_id;
                vr.read = reader.readVerified(rh->external_id, rh->tera);
                if (vr.read.status != Reader::ReadStatus::kUnavailable) {
                    out->push_back(std::move(vr));
                }
            }
        }
        p += recordStride(rh->size);
    }
    return primaryCount;
}
} // namespace

Reader::VerifiedQueryResult Reader::queryVerified(const float tera[4], uint32_t capacity) const {
    VerifiedQueryResult out;
    const RouteResult r = route(tera);
    out.path = r.path;
    out.depth = r.depth;
    out.primaryCount = verifyLeafPrimaries(*this, payload_, r.leaf, capacity, &out.records);
    return out;
}

std::vector<Reader::VerifiedSoftBranch> Reader::queryVerifiedSoft(
        const float tera[4], uint32_t k, uint32_t perBranchCapacity) const {
    const std::vector<SoftBranch> branches = routeSoftTopK(tera, k);
    std::vector<VerifiedSoftBranch> out;
    out.reserve(branches.size());
    for (const auto& b : branches) {
        VerifiedSoftBranch vb;
        vb.weight = b.weight;
        vb.meji = b.meji;
        vb.path = b.route.path;
        vb.depth = b.route.depth;
        vb.primaryCount = verifyLeafPrimaries(*this, payload_, b.route.leaf,
                                              perBranchCapacity, &vb.records);
        out.push_back(std::move(vb));
    }
    return out;
}

uint64_t Reader::allRecordsColumnar(uint64_t* out_ids, float* out_tera,
                                    uint8_t* out_payload, uint64_t payload_capacity,
                                    uint64_t* out_offsets, uint32_t* out_sizes,
                                    uint64_t capacity) const {
    uint64_t emitted = 0;
    uint64_t payload_used = 0;
    for (uint64_t i = 0; i < hdr_->node_count && emitted < capacity; ++i) {
        const Node& n = nodes_[i];
        if (!nodeIsLeaf(n) || n.record_count == 0) continue;
        const uint8_t* p = payload_ + n.payload_offset;
        const uint8_t* end = p + n.payload_size;
        for (uint32_t j = 0; j < n.record_count && emitted < capacity &&
                             p + sizeof(RecordHeader) <= end; ++j) {
            const auto* rh = reinterpret_cast<const RecordHeader*>(p);
            const uint8_t* data = p + sizeof(RecordHeader);
            if (payload_used + rh->size <= payload_capacity) {
                std::memcpy(out_payload + payload_used, data, rh->size);
            }
            out_ids[emitted] = rh->external_id;
            if (out_tera) std::memcpy(out_tera + emitted * 4, rh->tera, sizeof(rh->tera));
            out_offsets[emitted] = payload_used;
            out_sizes[emitted] = rh->size;
            payload_used += rh->size;
            ++emitted;
            p += recordStride(rh->size);
        }
    }
    return emitted;
}

} // namespace aamt
