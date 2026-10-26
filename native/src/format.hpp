// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// AAMT WP-23 — on-disk format of the .aamt substrate.
// The file is immutable after pack and always mapped PROT_READ.
// Layout: [FileHeader 128B][node arena, 64B nodes][payload arena, page-aligned]
#pragma once

#include <cstddef>
#include <cstdint>

namespace aamt {

constexpr uint32_t kMagic        = 0x544D4141; // "AAMT" little-endian
constexpr uint16_t kVersionMajor = 1;
constexpr uint16_t kVersionMinor = 0;
constexpr uint64_t kPageSize     = 4096;
// coordinate_id packs the node's Meji path as nibbles -> depth is capped at 8
// (two levels = an Odu; eight levels = a four-Odu word, 16^8 cells).
constexpr uint32_t kMaxDepthCap  = 8;

struct alignas(64) FileHeader {
    uint32_t magic;
    uint16_t version_major;
    uint16_t version_minor;
    uint64_t file_size;
    uint64_t node_arena_offset;
    uint64_t node_count;
    uint64_t payload_arena_offset;
    uint64_t payload_arena_size;
    uint64_t record_count;
    uint32_t max_depth;
    uint32_t bucket_capacity;
    uint64_t content_hash; // FNV-1a 64 over node arena + payload arena
    uint8_t  reserved[56];
};
static_assert(sizeof(FileHeader) == 128, "header must be 128 bytes");

// One true cache line. Children are packed contiguously in the node arena;
// the slot of Meji child m is popcount(child_mask & ((1 << m) - 1)).
struct alignas(64) Node {
    uint32_t coordinate_id;  // this node's own Meji path, packed nibbles
    float    chirality_spin; // reserved for field-state annotation (EXP-01)
    uint16_t child_mask;     // occupancy over the 16 Meji orthants
    uint16_t flags;          // bit 0: leaf; bits 8..15: depth
    uint32_t record_count;   // leaf bucket size (0 on internal nodes)
    uint64_t children_base;  // node-arena index of first child
    uint64_t payload_offset; // byte offset into payload arena
    uint64_t payload_size;   // byte length of this leaf's bucket
    uint8_t  reserved[24];
};
static_assert(sizeof(Node) == 64, "node must be one cache line");

constexpr uint16_t kNodeLeaf = 1u << 0;

inline uint32_t nodeDepth(const Node& n) { return n.flags >> 8; }
inline bool     nodeIsLeaf(const Node& n) { return (n.flags & kNodeLeaf) != 0; }

// Payload record: fixed 32-byte header, opaque data, stride padded to 8.
struct RecordHeader {
    uint64_t external_id;
    float    tera[4];
    uint32_t size;
    uint8_t  flags;      // bit 0: kRecordMirror (see householderReflectT)
    uint8_t  reserved[3];
};
static_assert(sizeof(RecordHeader) == 32, "record header must be 32 bytes");

constexpr uint8_t kRecordMirror = 1u << 0;
// This record's `data` holds an rs15_13::encodePayload()-encoded payload
// (see rs_gf16.hpp), not the raw bytes — Builder::addWithECC / the
// aamt_encode_ecc_payload / aamt_decode_ecc_payload C ABI functions are
// the intended way to write and read it. Complements, not replaces,
// kRecordMirror: RS gives free, exact repair of the common single-nibble
// corruption case; mirror pairing is the independent check for anything
// beyond that a 2-parity-symbol code cannot reliably detect on its own
// (see rs_gf16.hpp's file comment for the measured 86.7% figure).
constexpr uint8_t kRecordRS = 1u << 1;

inline uint64_t alignUp(uint64_t x, uint64_t a) { return (x + a - 1) & ~(a - 1); }
inline uint64_t recordStride(uint32_t size) { return sizeof(RecordHeader) + alignUp(size, 8); }

// Meji orthant of a coordinate within box [lo, hi): one bit per TERA axis.
// The comparison v >= mid resolves an exact hit on the splitting hyperplane
// upward — the right-continuous (0+) choice; this is the router's Ox operator.
inline uint32_t mejiIndex(const float tera[4], const float lo[4], const float hi[4]) {
    uint32_t m = 0;
    for (int d = 0; d < 4; ++d) {
        const float mid = 0.5f * (lo[d] + hi[d]);
        if (tera[d] >= mid) m |= (1u << d);
    }
    return m;
}

inline void childBox(uint32_t m, float lo[4], float hi[4]) {
    for (int d = 0; d < 4; ++d) {
        const float mid = 0.5f * (lo[d] + hi[d]);
        if (m & (1u << d)) lo[d] = mid; else hi[d] = mid;
    }
}

// Product-Bernoulli weight of orthant m within box [lo, hi) — the same
// distribution @aamt/vortex-runtime's mejiDistribution computes at the
// root box [0,1)^4: p(M_m) = prod_d [ t_d if bit_d(m) else (1 - t_d) ],
// where t_d is tera[d] renormalized to its position within [lo_d, hi_d).
// mejiIndex (hard argmax) is exactly argmax_m of this weight; soft routing
// (Reader::routeSoftTopK) ranks all 16 orthants by it instead of picking one.
inline float mejiWeight(const float tera[4], const float lo[4], const float hi[4], uint32_t m) {
    float w = 1.0f;
    for (int d = 0; d < 4; ++d) {
        const float span = hi[d] - lo[d];
        float t = span > 0.0f ? (tera[d] - lo[d]) / span : 0.5f;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        w *= (m & (1u << d)) ? t : (1.0f - t);
    }
    return w;
}

// The substrate's geometric mirror: reflect the T axis about the cube's own
// center (T -> 1-T). This is a genuine Householder reflection — flipping a
// single axis is orientation-reversing (det = -1) — not the antipodal map
// (flip all 4 axes), which in this *even* dimension is orientation-
// PRESERVING (det = (-1)^4 = +1) and so is a rotation, not a mirror, despite
// reading like one. H is an exact involution (H(H(v)) = v, no drift, no
// stored inverse needed) and always maps [0,1)^4 to itself.
//
// Structural fact (tested in tests/roundtrip.cpp, with the one honest
// exception noted): for a generic (non-dyadic) tera, route(H(v)).path
// equals route(v).path with the T bit (bit 0) of every nibble flipped —
// H flips exactly the T-bit of the first split, and since childBox leaves
// the other three axes' bounds untouched, the same flip propagates through
// every subsequent level by induction. The exception is real, not
// papered over: if some tera[0] lands exactly on a splitting boundary at
// some depth (a dyadic rational, measure zero but not empty — 0.5, 0.25,
// 0.75, ...), the router's own Ox tie-break (">=") can leave that level's
// bit unchanged for both v and H(v) rather than flipped. The verification
// path (Reader::verifyChirality) therefore never relies on this shortcut —
// it re-routes H(v) for real and checks what it finds, so the shortcut is
// a documented, testable insight, not a load-bearing assumption.
inline void householderReflectT(const float tera[4], float out[4]) {
    out[0] = 1.0f - tera[0];
    out[1] = tera[1];
    out[2] = tera[2];
    out[3] = tera[3];
}

inline uint64_t fnv1a64(const uint8_t* data, size_t len, uint64_t h = 0xcbf29ce484222325ull) {
    for (size_t i = 0; i < len; ++i) {
        h ^= data[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

// Sidecar journal record (.aamtj): 40 bytes per routing decision.
struct JournalRecord {
    uint64_t t_ns;
    uint64_t file_hash;
    float    tera[4];
    uint32_t path;  // packed Meji nibbles — the query's Odu word
    uint32_t depth;
};
static_assert(sizeof(JournalRecord) == 40, "journal record must be 40 bytes");

} // namespace aamt
