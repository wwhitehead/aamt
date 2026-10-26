// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/* AAMT WP-23 — C ABI for the .aamt substrate.
 *
 * This is the single boundary every expression layer calls:
 *   Python  -> ctypes          (desktop Alchemist bridges)
 *   C#      -> P/Invoke        (Unity / DreamOS)
 *   JS/TS   -> Emscripten/WASM (Chiral Lattice visualizer)
 *   C++     -> directly
 */
#ifndef AAMT_H
#define AAMT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aamt_file aamt_file;
typedef struct aamt_builder aamt_builder;
typedef struct aamt_journal aamt_journal;

typedef struct {
    uint64_t id;
    float tera[4];
    const uint8_t* data; /* zero-copy pointer into the mapping */
    uint32_t size;
} aamt_record_view;

typedef struct {
    uint32_t path;         /* packed Meji nibbles — the query's Odu word     */
    uint32_t depth;        /* number of nibbles in path                       */
    uint32_t record_count; /* total records in the routed leaf (0 if empty)   */
} aamt_route;

typedef struct {
    float weight;          /* mejiDistribution mass for this branch's root Meji */
    uint32_t meji;          /* 0..15, the root-level Meji this branch descended from */
    aamt_route route;       /* record_count is 0 for an empty (unoccupied) branch */
} aamt_soft_branch;

/* ---- reading ---------------------------------------------------------- */
aamt_file* aamt_open(const char* path, char* errbuf, size_t errlen);
void aamt_close(aamt_file* f);

uint64_t aamt_node_count(const aamt_file* f);
uint64_t aamt_record_count(const aamt_file* f);
uint64_t aamt_content_hash(const aamt_file* f);
uint32_t aamt_max_depth(const aamt_file* f);

/* Routes tera through the hexadeca-tree and fills up to `capacity` record
 * views from the leaf bucket. Returns the number of views written (>= 0).
 * route (optional) receives the full path/depth/bucket size. Returned data
 * pointers alias the mapping and stay valid until aamt_close. */
int32_t aamt_query(const aamt_file* f, const float tera[4],
                   aamt_record_view* out, int32_t capacity, aamt_route* route);

/* Soft routing (WP-23 §7's designated extension point): ranks all 16
 * root-level Meji orthants by mejiWeight — @aamt/vortex-runtime's
 * mejiDistribution, descending — and hard-descends the top `k` each to
 * their own leaf. Unlike aamt_query's single argmax path, empty orthants
 * are reported too (weight, record_count 0). This is a purely geometric
 * ranking of `tera` against each orthant, with no notion of visit
 * history — it widens the candidate pool near real data (a thin hard
 * bucket's neighbor branches often hold more) but is NOT an
 * occupancy-aware "find the cold/under-visited cell" search; WP-22's own
 * session occupancy tracking (odu_coverage_map.py) answers that question
 * and does not delegate to this. Fills up to min(k, 16) branches into
 * `out_branches` (already sorted by weight, descending) and, for branch i,
 * up to `per_branch_capacity` record views into
 * out_records[i*per_branch_capacity .. +per_branch_capacity). Returns the
 * number of branches written. */
int32_t aamt_query_soft(const aamt_file* f, const float tera[4], uint32_t k,
                       aamt_soft_branch* out_branches,
                       aamt_record_view* out_records, uint32_t per_branch_capacity);

/* Raw page-aligned payload arena — the region GPU backends wrap zero-copy
 * (see packages/libaamt/metal on Apple Silicon). Read-only, aliases the
 * mapping; valid until aamt_close. */
const uint8_t* aamt_payload_arena(const aamt_file* f, uint64_t* out_size);

/* Fills up to `capacity` views with every record in the file (arena order).
 * Returns the total record count, which may exceed capacity. For
 * reconstructing a full id/label table at load time in one native walk
 * instead of a per-record parse (see WP-23 memory_index_bridge.py). */
int64_t aamt_all_records(const aamt_file* f, aamt_record_view* out, int64_t capacity);

/* Persisted per-Odu-cell profile: record counts (256 entries) and mean TERA
 * centroids (256*4 floats, zero for empty cells) over the depth-2 cells,
 * computed from every record's stored coordinate. The substrate's
 * cross-session answer to "what actually lives in cell j" — WP-22's escape
 * search consumes it as the fallback target for cells not visited in the
 * current session. Cell indices use the TREE convention (m1<<4)|m2, first
 * Meji nibble high — NOT voxel.py's axis-major packing; see
 * memory_index_bridge.py's profile command for the tested permutation. */
void aamt_odu_profile(const aamt_file* f, uint32_t* out_counts, float* out_centroids);

/* Bulk columnar export: flat parallel arrays instead of one aamt_record_view
 * struct per record. Where aamt_all_records costs one FFI-crossing (and one
 * Python object allocation) per record in a ctypes marshalling loop —
 * measured to lose to json.load by 15-40x at scale, see WP-23 corrections
 * ledger #8 — this costs one native call and lets the caller pull ids as a
 * single primitive array and payloads as a single contiguous blob, doing
 * only pure-language bytes-slicing per record instead of crossing the FFI
 * boundary per record.
 *
 * `capacity` should be >= aamt_record_count(). `payload_capacity` should be
 * >= the size aamt_payload_arena reports (always sufficient: real payload
 * bytes are strictly less, due to per-record header/padding overhead in the
 * arena). out_tera may be NULL to skip (id + payload only).
 * Returns the number of records written. */
int64_t aamt_all_records_columnar(const aamt_file* f,
                                  uint64_t* out_ids, float* out_tera,
                                  uint8_t* out_payload, uint64_t payload_capacity,
                                  uint64_t* out_offsets, uint32_t* out_sizes,
                                  int64_t capacity);

/* ---- packing ---------------------------------------------------------- */
aamt_builder* aamt_builder_new(uint32_t max_depth, uint32_t bucket_capacity);
int aamt_builder_add(aamt_builder* b, uint64_t id, const float tera[4],
                     const void* data, uint32_t size);
/* Adds the record normally, AND a full redundant copy at the leaf its
 * Householder-T reflection (T -> 1-T) routes to. Real 2x storage cost for
 * real geometric redundancy: the mirror lands under the opposite T-orthant
 * at the root — a different subtree, a different region of the payload
 * arena — so damage localized to one region doesn't take out both copies
 * of a record. See aamt_verify_chirality / aamt_recover_record. */
int aamt_builder_add_mirrored(aamt_builder* b, uint64_t id, const float tera[4],
                              const void* data, uint32_t size);
/* RS(15,13)/GF(16)-encodes the payload before storing (see
 * aamt_decode_ecc_payload) — free, exact repair of a single corrupted
 * nibble anywhere in the payload, at ~15/13 size expansion. This is NOT a
 * substitute for aamt_builder_add_mirrored's independent check: measured
 * (packages/libaamt/tests/test_rs_gf16.cpp, exhaustive over all two-error
 * patterns), a SECOND corrupted nibble in the same 13-nibble block is
 * silently "corrected" to the WRONG data 86.7% of the time — an exact
 * property of using the minimum 2 parity symbols for 1-error correction,
 * not a bug. Use RS for the common single-bitrot case; pair with mirroring
 * for anything beyond it. */
int aamt_builder_add_ecc(aamt_builder* b, uint64_t id, const float tera[4],
                         const void* data, uint32_t size);
/* Both defenses on one record: primary AND mirror each store the SAME
 * RS(15,13)-encoded payload. Feeds aamt_read_verified, the read-side
 * function this exists for — see its doc for the decided policy. */
int aamt_builder_add_mirrored_ecc(aamt_builder* b, uint64_t id, const float tera[4],
                                  const void* data, uint32_t size);
int aamt_builder_write(aamt_builder* b, const char* path,
                       char* errbuf, size_t errlen);
void aamt_builder_free(aamt_builder* b);

/* ---- RS(15,13)/GF(16) payload codec (stateless — no aamt_file needed) -- */

/* Encodes `size` bytes into an RS-protected buffer. Returns the encoded
 * byte count (always > size: a 4-byte header plus ~15/13 expansion), or
 * writes nothing and returns 0 if out_capacity is insufficient — call once
 * with out_capacity=0 to learn the required size (out_encoded may be
 * NULL in that case). */
uint32_t aamt_encode_ecc_payload(const uint8_t* data, uint32_t size,
                                 uint8_t* out_encoded, uint32_t out_capacity);

/* Decodes and repairs an RS-encoded buffer. Returns the decoded size
 * (<= out_capacity, truncated if insufficient), or -1 if a block's
 * syndrome could not be resolved (see aamt_builder_add_ecc for what this
 * can and cannot guarantee). out_corrected_symbols (optional) receives the
 * count of 13-nibble blocks where a single-nibble repair fired. */
int64_t aamt_decode_ecc_payload(const uint8_t* encoded, uint32_t encoded_size,
                                uint8_t* out_data, uint64_t out_capacity,
                                uint32_t* out_corrected_symbols);

/* ---- mirror pairing / chirality gate ----------------------------------- */

typedef struct {
    uint64_t primary_count;
    uint64_t mirror_count;
    uint64_t consistent;
    uint64_t inconsistent_count;
} aamt_chirality_report;

/* The chirality gate: for every primary record, re-routes its Householder-T
 * reflection for real (full tree descent, never a shortcut) and checks a
 * mirror with matching id exists there with byte-identical payload. A
 * mismatch — mirror missing, or payload disagreement — is a "bad
 * reflection": counted here, not silently served. Only meaningful on a
 * substrate built with aamt_builder_add_mirrored; a plain-add substrate
 * reports every primary as inconsistent (no mirrors exist).
 * `out_inconsistent_ids` (optional) receives up to `capacity` ids of
 * inconsistent records; pass NULL/0 to skip. */
void aamt_verify_chirality(const aamt_file* f, aamt_chirality_report* out_report,
                           uint64_t* out_inconsistent_ids, uint64_t capacity);

/* Recovery scoped to what two copies can actually decide: if no primary is
 * found at tera_hint (a structural failure — routing tera_hint finds no
 * matching-id record there), falls back to the mirror at H(tera_hint),
 * recovering the true tera via H (its own inverse). Does not adjudicate a
 * pure payload disagreement where both copies are structurally consistent
 * — that's reported by aamt_verify_chirality, not resolved here, because
 * two copies with no independent checksum cannot decide which is right.
 * out_data must have capacity >= aamt_payload_arena's reported size (a safe
 * over-estimate for any single record). Returns the recovered size (>= 0),
 * or -1 if neither copy was found. */
int64_t aamt_recover_record(const aamt_file* f, uint64_t id, const float tera_hint[4],
                            uint8_t* out_data, uint64_t out_capacity, float out_tera[4]);

/* ---- composed verified read (WP-23 §13) -------------------------------- */

enum {
    AAMT_READ_OK = 0,                 /* primary decoded, zero RS corrections            */
    AAMT_READ_OK_RS_REPAIRED = 1,     /* primary needed a correction; trusted             */
    AAMT_READ_OK_MIRROR_RECOVERED = 2,/* primary missing/unresolvable; mirror served it   */
    AAMT_READ_UNAVAILABLE = 3,        /* both unreadable, or in positive disagreement     */
};

/* Wires RS(15,13) and mirror pairing together instead of leaving a caller
 * to sequence two mechanisms by hand. Decided policy: a mirror that
 * positively decodes to DIFFERENT bytes than the primary is the only case
 * with real counter-evidence of an unresolved disagreement (undecidable
 * with what this format stores, no 3rd copy or independent checksum) —
 * that case, and only that case (plus both copies being unreadable), is
 * refused rather than guessed. An absent, unreadable, or merely
 * non-confirming mirror never blocks trusting a primary that decodes
 * cleanly. Most useful on a substrate built with
 * aamt_builder_add_mirrored_ecc; degrades gracefully (RS-only or
 * mirror-only trust) on one built with just one of the two.
 * out_data must have capacity >= aamt_payload_arena's reported size.
 * Returns the decoded size (>= 0) written to out_data and out_status, or
 * -1 with *out_status == AAMT_READ_UNAVAILABLE and nothing written to
 * out_data. out_corrected_symbols (optional) is informational. */
int64_t aamt_read_verified(const aamt_file* f, uint64_t id, const float tera_hint[4],
                           uint8_t* out_data, uint64_t out_capacity, float out_tera[4],
                           uint32_t* out_status, uint32_t* out_corrected_symbols);

typedef struct {
    uint64_t id;
    float tera[4];
    uint32_t status;             /* AAMT_READ_* */
    uint32_t corrected_symbols;
} aamt_verified_meta;

/* The discovery-path counterpart to aamt_query: routes tera and walks only
 * the leaf's PRIMARY records (a mirror is a backup copy, not an
 * independently-discoverable candidate — its own stored tera is a
 * reflection, not "near tera" in any sense a caller asked about),
 * verifying each one exactly as aamt_read_verified does. A refused
 * candidate is DROPPED, not returned empty — this exists so a discovery
 * path (aamt_query's job) never hands back a payload it cannot vouch for,
 * without the caller needing to know which raw candidates were primaries
 * versus mirrors (record flags are not otherwise exposed across this ABI).
 *
 * Columnar output, same convention as aamt_all_records_columnar: up to
 * `capacity` entries in out_meta, concatenated decoded payload bytes in
 * out_payload with running offsets/sizes in out_offsets/out_sizes.
 * route (optional) receives path/depth and record_count == the RAW
 * primary count in the leaf, before any were dropped — a caller can tell
 * "empty leaf" from "leaf had candidates, all refused."
 * Returns the number of entries written (<= capacity). */
int32_t aamt_query_verified(const aamt_file* f, const float tera[4], int32_t capacity,
                            aamt_verified_meta* out_meta,
                            uint8_t* out_payload, uint64_t payload_capacity,
                            uint64_t* out_offsets, uint32_t* out_sizes,
                            aamt_route* route);

/* The soft-routing counterpart, mirroring aamt_query_soft's branch
 * structure rather than duplicating aamt_query_verified's primary/mirror
 * resolution per caller: each of the top-k ranked Meji orthants gets the
 * same primary-only, verified-candidate treatment. out_branches gets up
 * to k branches (weight/meji/route — route.record_count is the RAW
 * primary count in that branch's leaf, before verification, matching
 * aamt_query_verified's route semantics). out_meta and the payload/offset/
 * size arrays use a fixed per_branch_capacity stride: branch i's entries
 * live at index i*per_branch_capacity + j; out_branch_counts[i] gives how
 * many of that stride are actually filled (<= per_branch_capacity).
 * Returns the number of branches written (<= min(k, 16)). */
int32_t aamt_query_verified_soft(const aamt_file* f, const float tera[4], uint32_t k,
                                 uint32_t per_branch_capacity,
                                 aamt_soft_branch* out_branches,
                                 uint32_t* out_branch_counts,
                                 aamt_verified_meta* out_meta,
                                 uint8_t* out_payload, uint64_t payload_capacity,
                                 uint64_t* out_offsets, uint32_t* out_sizes);

/* ---- lineage journal (.aamtj sidecar) --------------------------------- */
aamt_journal* aamt_journal_open(const char* path); /* append mode */
int aamt_journal_log(aamt_journal* j, const aamt_file* f,
                     const float tera[4], const aamt_route* route);
void aamt_journal_close(aamt_journal* j);

#ifdef __cplusplus
}
#endif

#endif /* AAMT_H */
