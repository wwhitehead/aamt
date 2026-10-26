// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// Internal C++ API: Builder (pack) and Reader (mmap + route).
#pragma once

#include "format.hpp"

#include <memory>
#include <string>
#include <vector>

namespace aamt {

struct RouteResult {
    const Node* leaf = nullptr; // null if the query fell into an empty orthant
    uint32_t    path = 0;       // packed Meji nibbles, first split in highest nibble used
    uint32_t    depth = 0;
};

// One ranked branch from routeSoftTopK: a top-level Meji orthant (occupied
// or not) plus the weight its mejiDistribution mass carries and the leaf
// that lies beneath it (null if that orthant is empty).
struct SoftBranch {
    float       weight = 0.0f;
    uint32_t    meji = 0; // 0..15, the root-level Meji this branch descended from
    RouteResult route;
};

struct RecordView {
    uint64_t       external_id;
    const float*   tera;
    const uint8_t* data;
    uint32_t       size;
};

class Builder {
public:
    Builder(uint32_t max_depth, uint32_t bucket_capacity);

    void add(uint64_t id, const float tera[4], const void* data, uint32_t size);
    // Adds the record normally, AND a full redundant copy at the leaf its
    // Householder-T reflection routes to (format.hpp's householderReflectT
    // — see Reader::verifyChirality for the read-side gate). Real 2x
    // storage for real geometric redundancy: the mirror lands under the
    // opposite T-orthant at the root, a different subtree and a different
    // region of the payload arena, so damage localized to one region
    // doesn't take out both copies of a record.
    void addMirrored(uint64_t id, const float tera[4], const void* data, uint32_t size);
    // RS(15,13)/GF(16)-encodes the payload (rs_gf16.hpp) before storing —
    // free, exact repair of a single corrupted nibble anywhere in the
    // payload on read (decode with rs15_13::decodePayload directly, or via
    // the aamt_decode_ecc_payload C ABI function — the encoding is a
    // stateless byte transform, not a substrate-specific operation), at
    // the cost of ~15/13 size expansion. Not a substitute for
    // addMirrored's independent check; see rs_gf16.hpp for the measured
    // reason why.
    void addWithECC(uint64_t id, const float tera[4], const void* data, uint32_t size);
    // Both defenses on one record: primary AND mirror each store the SAME
    // RS(15,13)-encoded payload, so either copy can self-repair a single
    // corrupted nibble for free, and the mirror also backstops whatever a
    // lone RS block cannot decide about itself. Feeds Reader::readVerified,
    // the read-side function this exists for.
    void addMirroredWithECC(uint64_t id, const float tera[4], const void* data, uint32_t size);
    // Packs the tree and writes the file. Returns false with err set on failure.
    bool write(const std::string& path, std::string& err);

    uint64_t recordCount() const { return records_.size(); }

private:
    struct OwnedRecord {
        uint64_t id;
        float    tera[4];
        std::vector<uint8_t> data;
        uint8_t  flags = 0;
    };
    struct BuildNode {
        std::unique_ptr<BuildNode> child[16];
        std::vector<size_t> records; // indices into records_
        uint32_t path = 0;
        uint32_t depth = 0;
        uint16_t mask = 0;
        bool leaf = true;
    };

    void insert(BuildNode* n, float lo[4], float hi[4], size_t recIdx);
    void split(BuildNode* n, const float lo[4], const float hi[4]);

    uint32_t max_depth_;
    uint32_t bucket_capacity_;
    std::vector<OwnedRecord> records_;
};

class Reader {
public:
    ~Reader();
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    static std::unique_ptr<Reader> open(const std::string& path, std::string& err);

    RouteResult route(const float tera[4]) const;
    // Soft routing: ranks all 16 root-level Meji orthants by mejiWeight
    // (descending) — the same product-Bernoulli distribution
    // @aamt/vortex-runtime's mejiDistribution computes — and hard-descends
    // the top `k` each to their own leaf. Unlike route()'s single argmax
    // path, this reports empty orthants too (weight, no leaf). Purely
    // geometric ranking of `tera` against each orthant, no visit history —
    // widens the candidate pool near real data, but is NOT an
    // occupancy-aware "find the under-visited cell" search (that's WP-22's
    // own session tracking in odu_coverage_map.py, unrelated to this).
    // Returns up to min(k, 16) branches, sorted by weight descending.
    std::vector<SoftBranch> routeSoftTopK(const float tera[4], uint32_t k) const;
    // Fills up to `capacity` views from the routed leaf; returns total records
    // in the bucket (which may exceed capacity).
    uint32_t records(const RouteResult& r, RecordView* out, uint32_t capacity) const;
    // Fills up to `capacity` views with every record in the file, walking
    // all leaves in on-disk (arena) order. Returns the total record count
    // (may exceed capacity). For reconstructing a full label/id table at
    // load time without a per-record JSON parse.
    uint64_t allRecords(RecordView* out, uint64_t capacity) const;

    // Bulk columnar export: flat parallel arrays instead of one RecordView
    // per record, so a language binding pays for one FFI crossing (or one
    // memcpy) instead of N. Concatenates every record's payload bytes into
    // out_payload with running offsets in out_offsets. out_tera may be
    // nullptr to skip (id + payload only). `payload_capacity` should be
    // >= payloadArenaSize() (a safe over-estimate: real payload bytes are
    // always less, due to per-record header/padding overhead in the arena).
    // Returns the number of records written (<= capacity).
    uint64_t allRecordsColumnar(uint64_t* out_ids, float* out_tera,
                                uint8_t* out_payload, uint64_t payload_capacity,
                                uint64_t* out_offsets, uint32_t* out_sizes,
                                uint64_t capacity) const;

    // Persisted per-Odu-cell profile: record counts and mean TERA centroids
    // over the 256 depth-2 cells, computed from every record's stored
    // coordinate. This is the substrate's cross-session answer to "what
    // actually lives in cell j" — WP-22's escape search consumes it as the
    // fallback target for cells not yet visited in the current session
    // (out-of-session cells otherwise default to abstract geometric cell
    // centers). Cell indices use the TREE convention: (m1 << 4) | m2, first
    // Meji nibble high — NOT voxel.py's axis-major packing; consumers on
    // that convention must permute (see memory_index_bridge.py's profile
    // command for the tested mapping). out_counts: 256 entries;
    // out_centroids: 256*4 floats, zero for empty cells.
    void oduProfile(uint32_t* out_counts, float* out_centroids) const;

    // The chirality gate: for every non-mirror record, re-routes its
    // Householder-T reflection FOR REAL (full tree descent — never the
    // path-XOR shortcut, so this stays correct even at the documented
    // boundary exception) and checks that a mirror-flagged record with
    // matching id exists there, AND that its payload bytes are identical
    // to the primary's. Either failure — mirror missing (structural: the
    // primary's own tera no longer reflects to where its mirror should
    // be) or payload mismatch (content: the two independently-stored
    // copies disagree) — is a "bad reflection": counted here rather than
    // silently served. Note what a 2-copy mirror honestly cannot do: on a
    // pure payload mismatch where both copies are structurally consistent,
    // this detects disagreement but cannot decide which copy is right —
    // that needs a 3rd copy or an independent checksum, neither of which
    // this stores. Populated only for substrates built with
    // Builder::addMirrored; on one built with plain add(), every primary
    // reports inconsistent (no mirrors exist at all).
    struct ChiralityReport {
        uint64_t primary_count = 0;
        uint64_t mirror_count = 0;
        uint64_t consistent = 0;
        std::vector<uint64_t> inconsistent_ids;
    };
    ChiralityReport verifyChirality() const;

    // Recovery scoped to what's actually decidable with 2 copies: if the
    // primary is missing or unreadable at tera_hint (structural failure —
    // routing tera_hint finds no matching-id record there), falls back to
    // the mirror at H_T(tera_hint), recovering the true tera via H_T (its
    // own inverse) since the mirror's stored tera IS the reflection. Does
    // NOT adjudicate a case where primary and mirror both exist,
    // structurally check out, but disagree on payload bytes — that is
    // reported as an inconsistency (verifyChirality), not resolved here,
    // for the honest reason above. Returns false if neither copy is found.
    bool recoverRecord(uint64_t id, const float tera_hint[4],
                       std::vector<uint8_t>* out_data, float out_tera[4]) const;

    // The composed read path: RS(15,13) is the cheap first line of defense,
    // the mirror is the independent second — wired together for real,
    // rather than left as two mechanisms a caller has to sequence by hand.
    // The decided policy (WP-23 §13): a positive, decoded DISAGREEMENT
    // between two independently-decoded copies is the only case this
    // refuses to serve — an absent, unreadable, or merely-unconfirming
    // mirror never blocks trusting a primary that decodes cleanly, because
    // there is no actual counter-evidence in those cases, only missing
    // corroboration.
    enum class ReadStatus : uint32_t {
        kOk = 0,                // primary decoded with zero RS corrections
        kOkRsRepaired = 1,      // primary needed a correction; trusted (mirror agreed, absent, or itself unreadable)
        kOkMirrorRecovered = 2, // primary missing/unresolvable; recovered from the mirror
        kUnavailable = 3,       // both unreadable, or both readable and in positive disagreement — not served
    };
    struct ReadResult {
        ReadStatus status = ReadStatus::kUnavailable;
        uint32_t correctedSymbols = 0; // informational: RS blocks repaired in whichever copy was trusted
        std::vector<uint8_t> data;
        float tera[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    };
    ReadResult readVerified(uint64_t id, const float tera_hint[4]) const;

    // The discovery-path counterpart to readVerified: routes tera and
    // walks the leaf's PRIMARY records only (a mirror is a backup copy,
    // not an independently-discoverable candidate — its own stored tera
    // is a reflection, not "near tera" in any sense a caller asked about),
    // applying readVerified's cascade to each one before it comes back.
    // Entries whose read is refused (ReadStatus::kUnavailable) are DROPPED
    // rather than returned empty — a caller doing discovery (find whatever
    // lives near this coordinate) gets fewer, trustworthy results instead
    // of one it has to separately check. `primaryCount` is the RAW count
    // before any were dropped, so a caller can tell "empty leaf" from
    // "leaf had candidates, all refused."
    struct VerifiedRecord {
        uint64_t id = 0;
        ReadResult read;
    };
    struct VerifiedQueryResult {
        uint32_t path = 0;
        uint32_t depth = 0;
        uint32_t primaryCount = 0; // raw primary records in the routed leaf, before verification
        std::vector<VerifiedRecord> records; // only the ones that passed
    };
    VerifiedQueryResult queryVerified(const float tera[4], uint32_t capacity) const;

    // The soft-routing counterpart to queryVerified, built on
    // routeSoftTopK's own branch ranking rather than duplicating it — each
    // of the top-k ranked Meji orthants gets the same primary-only,
    // verified-candidate treatment queryVerified gives the single hard
    // route. Exists because routeSoftTopK's raw branches have the exact
    // same primary/mirror ambiguity queryVerified was built to resolve for
    // the hard-route case; verifying raw soft branches one-record-at-a-time
    // from a caller with no visibility into record flags would repeat the
    // same bug.
    struct VerifiedSoftBranch {
        float weight = 0.0f;
        uint32_t meji = 0;
        uint32_t path = 0;
        uint32_t depth = 0;
        uint32_t primaryCount = 0;
        std::vector<VerifiedRecord> records;
    };
    std::vector<VerifiedSoftBranch> queryVerifiedSoft(const float tera[4], uint32_t k,
                                                       uint32_t perBranchCapacity) const;

    const FileHeader& header() const { return *hdr_; }

    // Raw page-aligned payload arena, for zero-copy GPU buffer wrapping
    // (Apple Silicon unified memory; see packages/libaamt/metal). The
    // returned pointer aliases the mmap'd file and is read-only.
    const uint8_t* payloadArena() const { return payload_; }
    uint64_t payloadArenaSize() const { return hdr_->payload_arena_size; }

private:
    Reader() = default;
    int         fd_ = -1;
    void*       map_ = nullptr;
    size_t      map_size_ = 0;
    const FileHeader* hdr_ = nullptr;
    const Node*       nodes_ = nullptr;
    const uint8_t*    payload_ = nullptr;
};

} // namespace aamt
