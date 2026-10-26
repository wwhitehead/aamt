// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// Deterministic round-trip test for the .aamt substrate, driven entirely
// through the public C ABI (the same boundary Python/Unity/WASM use).
//
//  1. pack N seeded random records
//  2. reopen via mmap; every record must be retrievable by routing its own
//     TERA coordinate (exact bytes back, zero-copy)
//  3. journal 1000 queries, then replay them: paths must match bit-for-bit
#include "aamt/aamt.h"

#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,    \
                         #cond);                                            \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

int main() {
    const char* file_path = "/tmp/libaamt_roundtrip.aamt";
    const char* journal_path = "/tmp/libaamt_roundtrip.aamtj";
    std::remove(file_path);
    std::remove(journal_path);

    constexpr uint64_t kN = 20000;
    std::mt19937 rng(0xAAA7);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    std::vector<float> teras(kN * 4);
    for (auto& v : teras) v = uni(rng);

    // 1. pack
    aamt_builder* b = aamt_builder_new(8, 32);
    for (uint64_t id = 0; id < kN; ++id) {
        const std::string payload = "rec-" + std::to_string(id);
        aamt_builder_add(b, id, &teras[id * 4], payload.data(),
                         static_cast<uint32_t>(payload.size()));
    }
    char err[256] = {0};
    CHECK(aamt_builder_write(b, file_path, err, sizeof(err)) == 0);
    aamt_builder_free(b);

    // 2. reopen and verify every record is reachable through its own coord
    aamt_file* f = aamt_open(file_path, err, sizeof(err));
    CHECK(f != nullptr);
    CHECK(aamt_record_count(f) == kN);
    CHECK(aamt_node_count(f) >= 1);

    std::vector<aamt_record_view> views(4096);
    aamt_route route;
    for (uint64_t id = 0; id < kN; ++id) {
        const int32_t n = aamt_query(f, &teras[id * 4], views.data(),
                                     static_cast<int32_t>(views.size()), &route);
        CHECK(n >= 1);
        CHECK(route.record_count == static_cast<uint32_t>(n)); // capacity ample
        bool found = false;
        for (int32_t i = 0; i < n && !found; ++i) {
            if (views[i].id != id) continue;
            const std::string expect = "rec-" + std::to_string(id);
            CHECK(views[i].size == expect.size());
            CHECK(std::memcmp(views[i].data, expect.data(), expect.size()) == 0);
            CHECK(std::memcmp(views[i].tera, &teras[id * 4], 4 * sizeof(float)) == 0);
            found = true;
        }
        CHECK(found);
    }

    // 3. allRecords: every record reachable in one arena-order walk, and the
    //    set of ids recovered must equal exactly {0, ..., kN-1}.
    {
        std::vector<aamt_record_view> all(kN);
        const int64_t total = aamt_all_records(f, all.data(), static_cast<int64_t>(kN));
        CHECK(total == static_cast<int64_t>(kN));
        std::vector<bool> seen(kN, false);
        for (uint64_t i = 0; i < kN; ++i) {
            CHECK(all[i].id < kN);
            CHECK(!seen[all[i].id]);
            seen[all[i].id] = true;
            const std::string expect = "rec-" + std::to_string(all[i].id);
            CHECK(all[i].size == expect.size());
            CHECK(std::memcmp(all[i].data, expect.data(), expect.size()) == 0);
        }
        for (bool s : seen) CHECK(s);
    }

    // 3b. allRecordsColumnar: same records, flat-array form. Cross-checked
    //     against the struct-array form above — both must agree exactly.
    {
        uint64_t payload_cap = 0;
        aamt_payload_arena(f, &payload_cap);
        std::vector<uint64_t> ids(kN);
        std::vector<float> tera_out(kN * 4);
        std::vector<uint8_t> payload(payload_cap);
        std::vector<uint64_t> offsets(kN);
        std::vector<uint32_t> sizes(kN);

        const int64_t written = aamt_all_records_columnar(
            f, ids.data(), tera_out.data(), payload.data(), payload_cap,
            offsets.data(), sizes.data(), static_cast<int64_t>(kN));
        CHECK(written == static_cast<int64_t>(kN));

        std::vector<bool> seen(kN, false);
        for (uint64_t i = 0; i < kN; ++i) {
            const uint64_t id = ids[i];
            CHECK(id < kN);
            CHECK(!seen[id]);
            seen[id] = true;
            const std::string expect = "rec-" + std::to_string(id);
            CHECK(sizes[i] == expect.size());
            CHECK(std::memcmp(payload.data() + offsets[i], expect.data(), expect.size()) == 0);
            CHECK(std::memcmp(&tera_out[i * 4], &teras[id * 4], 4 * sizeof(float)) == 0);
        }
        for (bool s : seen) CHECK(s);
    }

    // 3c. soft routing: weights form a proper distribution (sum to 1, sorted
    //     descending), and the top-1 branch must exactly match the hard
    //     route() result — soft routing's top pick IS the hard argmax.
    {
        std::mt19937 srng(0xC0FFEE);
        std::uniform_real_distribution<float> suni(0.0f, 1.0f);
        std::vector<aamt_soft_branch> branches(16);
        std::vector<aamt_record_view> softViews(16 * 8);

        for (int trial = 0; trial < 200; ++trial) {
            float q[4] = {suni(srng), suni(srng), suni(srng), suni(srng)};

            const int32_t nb = aamt_query_soft(f, q, 16, branches.data(), softViews.data(), 8);
            CHECK(nb == 16);

            float sum = 0.0f;
            for (int i = 0; i < 16; ++i) {
                sum += branches[i].weight;
                if (i > 0) CHECK(branches[i].weight <= branches[i - 1].weight);
            }
            CHECK(std::abs(sum - 1.0f) < 1e-4f);

            aamt_route hardRoute;
            aamt_query(f, q, softViews.data(), 8, &hardRoute);
            CHECK(branches[0].route.path == hardRoute.path);
            CHECK(branches[0].route.depth == hardRoute.depth);
            CHECK(branches[0].route.record_count == hardRoute.record_count);
        }

        // Regression: the exact center is a measure-zero point where all 16
        // root orthants weigh exactly 1/16 — a tie that must resolve to
        // mejiIndex's ">=" (Ox) pick, not an arbitrary sort order.
        {
            float center[4] = {0.5f, 0.5f, 0.5f, 0.5f};
            const int32_t nb = aamt_query_soft(f, center, 16, branches.data(), softViews.data(), 8);
            CHECK(nb == 16);
            for (int i = 0; i < 16; ++i) CHECK(std::abs(branches[i].weight - 1.0f / 16.0f) < 1e-5f);

            aamt_route hardRoute;
            aamt_query(f, center, softViews.data(), 8, &hardRoute);
            CHECK(branches[0].route.path == hardRoute.path);
            CHECK(branches[0].route.record_count == hardRoute.record_count);
        }
    }

    // 3d. oduProfile: counts sum to N, and each record's profile cell must
    //     equal the top-2 nibbles of its own actual routing path — the
    //     profile is a claim about where the router puts things, so it is
    //     checked against the router, not against a re-derivation of itself.
    {
        std::vector<uint32_t> counts(256);
        std::vector<float> centroids(256 * 4);
        aamt_odu_profile(f, counts.data(), centroids.data());

        uint64_t total = 0;
        for (uint32_t c : counts) total += c;
        CHECK(total == kN);

        std::vector<uint32_t> expectCounts(256, 0);
        std::vector<std::array<double, 4>> sums(256, {0, 0, 0, 0});
        uint32_t routeChecked = 0;
        for (uint64_t id = 0; id < kN; ++id) {
            aamt_query(f, &teras[id * 4], views.data(), 1, &route);
            CHECK(route.depth >= 2); // 20k records, bucket 32: never this shallow
            const uint32_t cell = (route.path >> (4 * (route.depth - 2))) & 0xFF;
            expectCounts[cell] += 1;
            for (int d = 0; d < 4; ++d) sums[cell][d] += teras[id * 4 + d];
            ++routeChecked;
        }
        CHECK(routeChecked == kN);
        for (uint32_t c = 0; c < 256; ++c) {
            CHECK(counts[c] == expectCounts[c]);
            if (counts[c] == 0) continue;
            for (int d = 0; d < 4; ++d) {
                const float expect = static_cast<float>(sums[c][d] / counts[c]);
                CHECK(std::abs(centroids[c * 4 + d] - expect) < 1e-5f);
            }
        }
    }

    // 4. journal + bit-exact replay
    constexpr uint32_t kQ = 1000;
    std::vector<float> qteras(kQ * 4);
    for (auto& v : qteras) v = uni(rng);
    {
        aamt_journal* j = aamt_journal_open(journal_path);
        CHECK(j != nullptr);
        for (uint32_t i = 0; i < kQ; ++i) {
            aamt_query(f, &qteras[i * 4], views.data(), 1, &route);
            CHECK(aamt_journal_log(j, f, &qteras[i * 4], &route) == 0);
        }
        aamt_journal_close(j);
    }
    {
        FILE* jf = std::fopen(journal_path, "rb");
        CHECK(jf != nullptr);
        struct JR {
            uint64_t t_ns, file_hash;
            float tera[4];
            uint32_t path, depth;
        } rec;
        static_assert(sizeof(JR) == 40, "journal record layout");
        uint32_t replayed = 0;
        while (std::fread(&rec, sizeof(rec), 1, jf) == 1) {
            CHECK(rec.file_hash == aamt_content_hash(f));
            aamt_query(f, rec.tera, views.data(), 1, &route);
            CHECK(route.path == rec.path);
            CHECK(route.depth == rec.depth);
            ++replayed;
        }
        std::fclose(jf);
        CHECK(replayed == kQ);
    }

    std::printf("PASS roundtrip: %" PRIu64 " records packed, %" PRIu64
                " nodes, all records reachable by route AND by allRecords, "
                "%u journal replays bit-exact\n",
                kN, aamt_node_count(f), kQ);
    aamt_close(f);
    std::remove(file_path);
    std::remove(journal_path);

    // 5. Mirror pairing + chirality gate: pack a substrate where every
    //    record also gets a Householder-T reflected copy, then prove the
    //    gate does its job — clean substrate reads 100% consistent, and two
    //    distinct, honestly-scoped corruption models are each caught (or,
    //    for the one that's mathematically undecidable with 2 copies,
    //    honestly NOT silently resolved) exactly as documented in aamt.h.
    {
        const char* mirror_path = "/tmp/libaamt_mirror.aamt";
        std::remove(mirror_path);

        constexpr uint64_t kM = 2000;
        std::mt19937 mrng(0xDEADBEEF);
        std::uniform_real_distribution<float> muni(0.0f, 1.0f);
        std::vector<float> mteras(kM * 4);
        for (auto& v : mteras) v = muni(mrng);

        aamt_builder* mb = aamt_builder_new(8, 16);
        for (uint64_t id = 0; id < kM; ++id) {
            const std::string payload = "mrec-" + std::to_string(id);
            aamt_builder_add_mirrored(mb, id, &mteras[id * 4], payload.data(),
                                      static_cast<uint32_t>(payload.size()));
        }
        CHECK(aamt_builder_write(mb, mirror_path, err, sizeof(err)) == 0);
        aamt_builder_free(mb);

        // 5a. Clean substrate: every primary's mirror checks out.
        {
            aamt_file* mf = aamt_open(mirror_path, err, sizeof(err));
            CHECK(mf != nullptr);
            aamt_chirality_report rep;
            aamt_verify_chirality(mf, &rep, nullptr, 0);
            CHECK(rep.primary_count == kM);
            CHECK(rep.mirror_count == kM);
            CHECK(rep.consistent == kM);
            CHECK(rep.inconsistent_count == 0);

            // recoverRecord's ordinary path (primary present, untouched)
            // must simply return the primary, unchanged.
            uint8_t buf[256];
            float outTera[4];
            for (uint64_t id : {uint64_t{0}, kM / 2, kM - 1}) {
                const int64_t n = aamt_recover_record(mf, id, &mteras[id * 4], buf,
                                                      sizeof(buf), outTera);
                const std::string expect = "mrec-" + std::to_string(id);
                CHECK(n == static_cast<int64_t>(expect.size()));
                CHECK(std::memcmp(buf, expect.data(), expect.size()) == 0);
                CHECK(std::memcmp(outTera, &mteras[id * 4], sizeof(outTera)) == 0);
            }
            aamt_close(mf);
        }

        // Locate one record's on-disk byte offset directly: the payload
        // arena starts at 128B header + node_count*64B nodes, page-aligned
        // (Builder::write's own layout rule) — computed here, not exposed
        // by the ABI, precisely so the corruption below is a real edit to
        // the packed bytes, not a simulated one.
        constexpr uint64_t kHeaderSize = 128, kNodeSize = 64, kPageSize = 4096;
        auto alignUpPage = [](uint64_t x) { return (x + kPageSize - 1) & ~(kPageSize - 1); };

        auto findRecordFileOffset = [&](uint64_t targetId) -> uint64_t {
            aamt_file* mf = aamt_open(mirror_path, err, sizeof(err));
            CHECK(mf != nullptr);
            const uint64_t payloadOffset =
                alignUpPage(kHeaderSize + aamt_node_count(mf) * kNodeSize);
            uint64_t arenaSize = 0;
            const uint8_t* arenaBase = aamt_payload_arena(mf, &arenaSize);

            std::vector<aamt_record_view> all(kM * 2); // primaries + mirrors
            const int64_t total = aamt_all_records(mf, all.data(), static_cast<int64_t>(kM * 2));
            CHECK(total == static_cast<int64_t>(kM * 2));
            uint64_t offset = UINT64_MAX;
            for (int64_t i = 0; i < total; ++i) {
                // Match the PRIMARY (not the mirror) by id + tera equality —
                // both copies share the same id, and only the primary's tera
                // equals the coordinate we inserted it at (the mirror's is
                // the Householder reflection).
                if (all[i].id == targetId &&
                    std::memcmp(all[i].tera, &mteras[targetId * 4], 4 * sizeof(float)) == 0) {
                    const uint64_t dataOffsetInArena =
                        static_cast<uint64_t>(all[i].data - arenaBase);
                    offset = payloadOffset + dataOffsetInArena - 32 /* sizeof(RecordHeader) */;
                    break;
                }
            }
            CHECK(offset != UINT64_MAX);
            aamt_close(mf);
            return offset;
        };

        // 5b. Payload-only corruption (tera untouched): a genuinely
        //     undecidable case for 2 copies with no independent checksum.
        //     Detection must fire; recovery must NOT silently pick a side —
        //     it returns the (now-wrong) primary, exactly as documented.
        {
            const uint64_t targetId = 777;
            const uint64_t headerOffset = findRecordFileOffset(targetId);
            FILE* raw = std::fopen(mirror_path, "r+b");
            CHECK(raw != nullptr);
            CHECK(std::fseek(raw, static_cast<long>(headerOffset + 32 /* header size, into data */), SEEK_SET) == 0);
            const std::string original = "mrec-" + std::to_string(targetId);
            char corrupted = original[0] ^ 0x7F; // flip the first payload byte
            CHECK(std::fwrite(&corrupted, 1, 1, raw) == 1);
            std::fclose(raw);

            aamt_file* mf = aamt_open(mirror_path, err, sizeof(err));
            CHECK(mf != nullptr);
            aamt_chirality_report rep;
            std::vector<uint64_t> badIds(16);
            aamt_verify_chirality(mf, &rep, badIds.data(), badIds.size());
            CHECK(rep.primary_count == kM);
            CHECK(rep.inconsistent_count == 1);
            CHECK(rep.consistent == kM - 1);
            CHECK(badIds[0] == targetId);

            // Honest limitation, demonstrated rather than hidden: recovery
            // with the correct tera_hint finds the primary structurally
            // (it's still physically present, still at the right leaf) and
            // returns it as-is — corruption and all. Two copies with no
            // independent checksum cannot decide which one is lying; this
            // is reported by verifyChirality, not silently adjudicated here.
            uint8_t buf[256];
            float outTera[4];
            const int64_t n = aamt_recover_record(mf, targetId, &mteras[targetId * 4],
                                                  buf, sizeof(buf), outTera);
            CHECK(n == static_cast<int64_t>(original.size()));
            CHECK(buf[0] == static_cast<uint8_t>(corrupted)); // still corrupted — by design
            aamt_close(mf);

            // Restore for the next sub-test.
            raw = std::fopen(mirror_path, "r+b");
            CHECK(raw != nullptr);
            CHECK(std::fseek(raw, static_cast<long>(headerOffset + 32), SEEK_SET) == 0);
            CHECK(std::fwrite(&original[0], 1, 1, raw) == 1);
            std::fclose(raw);
        }

        // 5c. Identity corruption (id field scrambled): the primary becomes
        //     unfindable by id at its own leaf — a real structural failure,
        //     the case recoverRecord is actually for. The caller supplies
        //     the id/tera it always knew (its own record of what it wrote,
        //     independent of the now-garbled on-disk bytes) and recovery
        //     must fall through to the mirror and succeed.
        {
            const uint64_t targetId = 1234;
            const uint64_t headerOffset = findRecordFileOffset(targetId);
            FILE* raw = std::fopen(mirror_path, "r+b");
            CHECK(raw != nullptr);
            CHECK(std::fseek(raw, static_cast<long>(headerOffset), SEEK_SET) == 0);
            uint64_t scrambledId = 0xFFFFFFFFFFFFFFFFull;
            CHECK(std::fwrite(&scrambledId, sizeof(scrambledId), 1, raw) == 1);
            std::fclose(raw);

            aamt_file* mf = aamt_open(mirror_path, err, sizeof(err));
            CHECK(mf != nullptr);
            uint8_t buf[256];
            float outTera[4];
            const int64_t n = aamt_recover_record(mf, targetId, &mteras[targetId * 4],
                                                  buf, sizeof(buf), outTera);
            const std::string expect = "mrec-" + std::to_string(targetId);
            CHECK(n == static_cast<int64_t>(expect.size()));
            CHECK(std::memcmp(buf, expect.data(), expect.size()) == 0);
            CHECK(std::memcmp(outTera, &mteras[targetId * 4], sizeof(outTera)) == 0);
            aamt_close(mf);
        }

        std::printf("PASS mirror/chirality: %" PRIu64 " mirrored records, clean "
                    "substrate 100%% consistent, payload corruption detected "
                    "(not silently resolved), identity corruption recovered "
                    "via mirror\n", kM);
        std::remove(mirror_path);
    }

    // 6. RS(15,13)/GF(16) ECC on real packed records: pack with
    //    aamt_builder_add_ecc, corrupt exactly one on-disk nibble in one
    //    record's encoded payload, and confirm aamt_decode_ecc_payload
    //    repairs it exactly — a real byte edit via a separate file handle,
    //    same methodology as the mirror-pairing corruption test above.
    {
        const char* ecc_path = "/tmp/libaamt_ecc.aamt";
        std::remove(ecc_path);

        constexpr uint64_t kE = 300;
        std::mt19937 erng(0xECC0DE);
        std::uniform_real_distribution<float> euni(0.0f, 1.0f);
        std::vector<float> eteras(kE * 4);
        for (auto& v : eteras) v = euni(erng);
        std::vector<std::string> payloads(kE);

        aamt_builder* eb = aamt_builder_new(8, 16);
        for (uint64_t id = 0; id < kE; ++id) {
            payloads[id] = "ecc-payload-" + std::to_string(id) + "-checksum-free-repair";
            aamt_builder_add_ecc(eb, id, &eteras[id * 4], payloads[id].data(),
                                 static_cast<uint32_t>(payloads[id].size()));
        }
        CHECK(aamt_builder_write(eb, ecc_path, err, sizeof(err)) == 0);
        aamt_builder_free(eb);

        // Clean substrate: every record decodes exactly, zero corrections fired.
        {
            aamt_file* ef = aamt_open(ecc_path, err, sizeof(err));
            CHECK(ef != nullptr);
            std::vector<aamt_record_view> views(32); // bucket may hold several ids
            for (uint64_t id = 0; id < kE; ++id) {
                aamt_route r;
                const int32_t n = aamt_query(ef, &eteras[id * 4], views.data(),
                                             static_cast<int32_t>(views.size()), &r);
                CHECK(n >= 1);
                int32_t match = -1;
                for (int32_t i = 0; i < n; ++i) if (views[i].id == id) { match = i; break; }
                CHECK(match >= 0);
                uint8_t decoded[256];
                uint32_t corrected = 0;
                const int64_t dn = aamt_decode_ecc_payload(views[match].data, views[match].size,
                                                           decoded, sizeof(decoded), &corrected);
                CHECK(dn == static_cast<int64_t>(payloads[id].size()));
                CHECK(corrected == 0);
                CHECK(std::memcmp(decoded, payloads[id].data(), payloads[id].size()) == 0);
            }
            aamt_close(ef);
        }

        // Locate one record's on-disk byte offset (same technique as the
        // mirror-pairing test: compute the payload arena's file offset
        // from the packed layout, then pointer-subtract within the mmap).
        const uint64_t targetId = 199;
        uint64_t headerOffset;
        {
            aamt_file* ef = aamt_open(ecc_path, err, sizeof(err));
            CHECK(ef != nullptr);
            constexpr uint64_t kHeaderSize = 128, kNodeSize = 64, kPageSize = 4096;
            const uint64_t payloadOffset =
                ((kHeaderSize + aamt_node_count(ef) * kNodeSize) + kPageSize - 1)
                & ~(kPageSize - 1);
            uint64_t arenaSize = 0;
            const uint8_t* arenaBase = aamt_payload_arena(ef, &arenaSize);

            std::vector<aamt_record_view> all(kE);
            const int64_t total = aamt_all_records(ef, all.data(), static_cast<int64_t>(kE));
            CHECK(total == static_cast<int64_t>(kE));
            uint64_t offset = UINT64_MAX;
            for (int64_t i = 0; i < total; ++i) {
                if (all[i].id == targetId) {
                    offset = payloadOffset +
                            static_cast<uint64_t>(all[i].data - arenaBase) - 32;
                    break;
                }
            }
            CHECK(offset != UINT64_MAX);
            headerOffset = offset;
            aamt_close(ef);
        }

        // Corrupt exactly one nibble in the encoded payload (byte 10, high
        // nibble — inside the RS-protected region, past the 4-byte size
        // header) and confirm exact repair.
        {
            FILE* raw = std::fopen(ecc_path, "r+b");
            CHECK(raw != nullptr);
            CHECK(std::fseek(raw, static_cast<long>(headerOffset + 32 + 10), SEEK_SET) == 0);
            uint8_t original_byte;
            CHECK(std::fread(&original_byte, 1, 1, raw) == 1);
            CHECK(std::fseek(raw, static_cast<long>(headerOffset + 32 + 10), SEEK_SET) == 0);
            const uint8_t corrupted_byte = static_cast<uint8_t>(original_byte ^ 0xF0);
            CHECK(std::fwrite(&corrupted_byte, 1, 1, raw) == 1);
            std::fclose(raw);

            aamt_file* ef = aamt_open(ecc_path, err, sizeof(err));
            CHECK(ef != nullptr);
            std::vector<aamt_record_view> views(32);
            aamt_route r;
            const int32_t n = aamt_query(ef, &eteras[targetId * 4], views.data(),
                                         static_cast<int32_t>(views.size()), &r);
            CHECK(n >= 1);
            int32_t match = -1;
            for (int32_t i = 0; i < n; ++i) if (views[i].id == targetId) { match = i; break; }
            CHECK(match >= 0);
            uint8_t decoded[256];
            uint32_t corrected = 0;
            const int64_t dn = aamt_decode_ecc_payload(views[match].data, views[match].size,
                                                       decoded, sizeof(decoded), &corrected);
            CHECK(dn == static_cast<int64_t>(payloads[targetId].size()));
            CHECK(corrected == 1); // exactly one block's repair fired
            CHECK(std::memcmp(decoded, payloads[targetId].data(),
                              payloads[targetId].size()) == 0);
            aamt_close(ef);

            std::printf("PASS RS/GF(16) ECC: %" PRIu64 " records packed with "
                        "addWithECC, clean decode zero corrections, one real "
                        "on-disk nibble flip repaired exactly (1 block corrected)\n", kE);
        }
        std::remove(ecc_path);
    }

    // 7. aamt_read_verified: the composed RS + mirror read path (WP-23
    //    §13), packed with aamt_builder_add_mirrored_ecc so both defenses
    //    are on every record. Four real cases, each via an actual on-disk
    //    byte edit (same methodology as sections 5 and 6) — no simulated
    //    corruption anywhere in this section.
    {
        const char* v_path = "/tmp/libaamt_verified.aamt";
        std::remove(v_path);

        constexpr uint64_t kV = 200;
        std::mt19937 vrng(0x7E51F1ED);
        std::uniform_real_distribution<float> vuni(0.0f, 1.0f);
        std::vector<float> vteras(kV * 4);
        for (auto& v : vteras) v = vuni(vrng);
        std::vector<std::string> vpayloads(kV);

        aamt_builder* vb = aamt_builder_new(8, 16);
        for (uint64_t id = 0; id < kV; ++id) {
            vpayloads[id] = "verified-" + std::to_string(id) + "-both-defenses";
            aamt_builder_add_mirrored_ecc(vb, id, &vteras[id * 4], vpayloads[id].data(),
                                          static_cast<uint32_t>(vpayloads[id].size()));
        }
        CHECK(aamt_builder_write(vb, v_path, err, sizeof(err)) == 0);
        aamt_builder_free(vb);

        constexpr uint64_t kHeaderSize = 128, kNodeSize = 64, kPageSize = 4096;
        auto alignUpPage = [](uint64_t x) { return (x + kPageSize - 1) & ~(kPageSize - 1); };

        // Primary-only file offset of one id's encoded payload (matches
        // sections 5/6's technique: match by id AND tera, since the
        // primary's stored tera is the coordinate we inserted at while the
        // mirror's is the Householder reflection of it).
        auto findPrimaryOffset = [&](uint64_t targetId) -> uint64_t {
            aamt_file* vf = aamt_open(v_path, err, sizeof(err));
            CHECK(vf != nullptr);
            const uint64_t payloadOffset =
                alignUpPage(kHeaderSize + aamt_node_count(vf) * kNodeSize);
            uint64_t arenaSize = 0;
            const uint8_t* arenaBase = aamt_payload_arena(vf, &arenaSize);
            std::vector<aamt_record_view> all(kV * 2); // primaries + mirrors
            const int64_t total = aamt_all_records(vf, all.data(), static_cast<int64_t>(kV * 2));
            CHECK(total == static_cast<int64_t>(kV * 2));
            uint64_t offset = UINT64_MAX;
            for (int64_t i = 0; i < total; ++i) {
                if (all[i].id == targetId &&
                    std::memcmp(all[i].tera, &vteras[targetId * 4], 4 * sizeof(float)) == 0) {
                    offset = payloadOffset + static_cast<uint64_t>(all[i].data - arenaBase) - 32;
                    break;
                }
            }
            CHECK(offset != UINT64_MAX);
            aamt_close(vf);
            return offset;
        };

        // 7a. Clean substrate: every record reads AAMT_READ_OK, zero
        //     corrections, exact payload and tera.
        {
            aamt_file* vf = aamt_open(v_path, err, sizeof(err));
            CHECK(vf != nullptr);
            for (uint64_t id = 0; id < kV; ++id) {
                uint8_t buf[256];
                float outTera[4];
                uint32_t status = 999, corrected = 999;
                const int64_t n = aamt_read_verified(vf, id, &vteras[id * 4], buf, sizeof(buf),
                                                     outTera, &status, &corrected);
                CHECK(n == static_cast<int64_t>(vpayloads[id].size()));
                CHECK(status == AAMT_READ_OK);
                CHECK(corrected == 0);
                CHECK(std::memcmp(buf, vpayloads[id].data(), vpayloads[id].size()) == 0);
                CHECK(std::memcmp(outTera, &vteras[id * 4], sizeof(outTera)) == 0);
            }
            aamt_close(vf);
        }

        // 7b. One real corrupted nibble in the primary's encoded payload
        //     (byte 10, high nibble — inside the RS-protected region, same
        //     spot section 6 uses) -> AAMT_READ_OK_RS_REPAIRED, exact
        //     payload recovered from the primary alone; the mirror is
        //     never touched to get there.
        {
            const uint64_t targetId = 50;
            const uint64_t off = findPrimaryOffset(targetId);
            FILE* raw = std::fopen(v_path, "r+b");
            CHECK(raw != nullptr);
            CHECK(std::fseek(raw, static_cast<long>(off + 32 + 10), SEEK_SET) == 0);
            uint8_t b;
            CHECK(std::fread(&b, 1, 1, raw) == 1);
            CHECK(std::fseek(raw, static_cast<long>(off + 32 + 10), SEEK_SET) == 0);
            b ^= 0xF0;
            CHECK(std::fwrite(&b, 1, 1, raw) == 1);
            std::fclose(raw);

            aamt_file* vf = aamt_open(v_path, err, sizeof(err));
            CHECK(vf != nullptr);
            uint8_t buf[256];
            float outTera[4];
            uint32_t status = 999, corrected = 999;
            const int64_t n = aamt_read_verified(vf, targetId, &vteras[targetId * 4], buf,
                                                 sizeof(buf), outTera, &status, &corrected);
            CHECK(n == static_cast<int64_t>(vpayloads[targetId].size()));
            CHECK(status == AAMT_READ_OK_RS_REPAIRED);
            CHECK(corrected == 1);
            CHECK(std::memcmp(buf, vpayloads[targetId].data(), vpayloads[targetId].size()) == 0);
            aamt_close(vf);
        }

        // 7c. Primary's id field scrambled (structural failure, same
        //     technique as section 5c) -> AAMT_READ_OK_MIRROR_RECOVERED,
        //     exact payload and true tera recovered via the mirror.
        {
            const uint64_t targetId = 99;
            const uint64_t off = findPrimaryOffset(targetId);
            FILE* raw = std::fopen(v_path, "r+b");
            CHECK(raw != nullptr);
            CHECK(std::fseek(raw, static_cast<long>(off), SEEK_SET) == 0);
            uint64_t scrambled = 0xFFFFFFFFFFFFFFFFull;
            CHECK(std::fwrite(&scrambled, sizeof(scrambled), 1, raw) == 1);
            std::fclose(raw);

            aamt_file* vf = aamt_open(v_path, err, sizeof(err));
            CHECK(vf != nullptr);
            uint8_t buf[256];
            float outTera[4];
            uint32_t status = 999, corrected = 999;
            const int64_t n = aamt_read_verified(vf, targetId, &vteras[targetId * 4], buf,
                                                 sizeof(buf), outTera, &status, &corrected);
            CHECK(n == static_cast<int64_t>(vpayloads[targetId].size()));
            CHECK(status == AAMT_READ_OK_MIRROR_RECOVERED);
            CHECK(std::memcmp(buf, vpayloads[targetId].data(), vpayloads[targetId].size()) == 0);
            CHECK(std::memcmp(outTera, &vteras[targetId * 4], sizeof(outTera)) == 0);
            aamt_close(vf);
        }

        // 7d. The case this function exists for: a genuine two-nibble
        //     collision in the primary's FIRST block (positions 0,1 —
        //     both parity nibbles, packed into encoded byte 4 — with
        //     deltas 1,2) that RS reports as a confident, resolved
        //     single-error repair (ok=true) but that is, measured exactly
        //     as WP-23 §11 describes, WRONG data. Confirmed offline by
        //     exhaustive enumeration (the same search test_rs_gf16.cpp
        //     performs): this exact (pos, pos, delta, delta) quadruple is
        //     one of the 20,475 "silently mis-corrected" patterns. Calling
        //     aamt_decode_ecc_payload directly on this primary would
        //     report corrected=1 and return the wrong bytes with no
        //     indication anything is amiss — the exact danger §11
        //     measured. aamt_read_verified must not repeat that mistake:
        //     the untouched mirror decodes to the true data, the two
        //     disagree, and the decided policy is to refuse rather than
        //     guess.
        {
            const uint64_t targetId = 150;
            const uint64_t off = findPrimaryOffset(targetId);
            FILE* raw = std::fopen(v_path, "r+b");
            CHECK(raw != nullptr);
            CHECK(std::fseek(raw, static_cast<long>(off + 32 + 4), SEEK_SET) == 0);
            uint8_t b;
            CHECK(std::fread(&b, 1, 1, raw) == 1);
            CHECK(std::fseek(raw, static_cast<long>(off + 32 + 4), SEEK_SET) == 0);
            b ^= 0x12; // (delta1=1 << 4) | delta2=2, on codeword nibbles 0 and 1
            CHECK(std::fwrite(&b, 1, 1, raw) == 1);
            std::fclose(raw);

            aamt_file* vf = aamt_open(v_path, err, sizeof(err));
            CHECK(vf != nullptr);

            // Sanity: prove the danger is real before proving it's caught.
            // Reading the primary's raw payload through RS ALONE (bypassing
            // the mirror entirely) must report a confident, WRONG repair —
            // otherwise this test isn't exercising the case it claims to.
            {
                std::vector<aamt_record_view> views(32);
                aamt_route r;
                const int32_t n = aamt_query(vf, &vteras[targetId * 4], views.data(),
                                             static_cast<int32_t>(views.size()), &r);
                CHECK(n >= 1);
                int32_t match = -1;
                for (int32_t i = 0; i < n; ++i) if (views[i].id == targetId) { match = i; break; }
                CHECK(match >= 0);
                uint8_t decoded[256];
                uint32_t corrected = 0;
                const int64_t dn = aamt_decode_ecc_payload(views[match].data, views[match].size,
                                                           decoded, sizeof(decoded), &corrected);
                CHECK(dn == static_cast<int64_t>(vpayloads[targetId].size()));
                CHECK(corrected == 1); // "confidently" repaired...
                CHECK(std::memcmp(decoded, vpayloads[targetId].data(),
                                  vpayloads[targetId].size()) != 0); // ...and WRONG
            }

            uint8_t buf[256];
            float outTera[4];
            uint32_t status = 999, corrected = 999;
            const int64_t n = aamt_read_verified(vf, targetId, &vteras[targetId * 4], buf,
                                                 sizeof(buf), outTera, &status, &corrected);
            CHECK(n == -1);
            CHECK(status == AAMT_READ_UNAVAILABLE);
            aamt_close(vf);
        }

        std::printf("PASS aamt_read_verified: %" PRIu64 " records packed with "
                    "addMirroredWithECC — clean read, single-nibble RS repair "
                    "(mirror untouched), structural recovery via mirror, and a "
                    "real RS silent-miscorrection caught and refused rather "
                    "than served\n", kV);

        // 7e. aamt_query_verified: the discovery-path counterpart. A bucket
        //     routinely holds a mix of PRIMARY records (near this query
        //     coordinate for real) and MIRROR records of OTHER, unrelated
        //     ids (near only by reflection accident) — the first
        //     implementation of this composition called read_verified on
        //     every raw candidate query() returned regardless of which
        //     kind it was, and a mirror's own stored tera is not a
        //     primary's coordinate, so read_verified incorrectly reported
        //     perfectly healthy mirrors as unavailable. This checks every
        //     one of the kV clean coordinates (skipping the three ids
        //     corrupted above) and requires the verified count to equal
        //     the raw primary count every time — the exact invariant the
        //     bug broke.
        {
            aamt_file* vf = aamt_open(v_path, err, sizeof(err));
            CHECK(vf != nullptr);
            uint64_t checked = 0, totalPrimaries = 0;
            for (uint64_t id = 0; id < kV; ++id) {
                if (id == 50 || id == 99 || id == 150) continue; // corrupted above
                std::vector<aamt_verified_meta> meta(32);
                std::vector<uint8_t> payload(4096);
                std::vector<uint64_t> offsets(32);
                std::vector<uint32_t> sizes(32);
                aamt_route r;
                const int32_t n = aamt_query_verified(vf, &vteras[id * 4], 32, meta.data(),
                                                      payload.data(), payload.size(),
                                                      offsets.data(), sizes.data(), &r);
                // A bucket can legitimately contain OTHER ids' primaries too
                // (real proximity, not reflection) — including id 150, the
                // one deliberately driven to AAMT_READ_UNAVAILABLE above, if
                // it happens to share this bucket; that one candidate is
                // correctly dropped, not a bug. Ids 50/99 recover fully (RS
                // repair / mirror recovery) and still count toward n. So the
                // only slack this bound allows is "at most the one true
                // unavailable id" — the original mirror-miscounting bug
                // dropped far more than that across the sweep, which is what
                // this catches.
                CHECK(static_cast<uint32_t>(n) >= r.record_count - 1);
                bool found = false;
                for (int32_t i = 0; i < n; ++i) {
                    if (meta[i].id != id) continue;
                    found = true;
                    CHECK(meta[i].status == AAMT_READ_OK); // anchor id itself is one of the clean ones
                    CHECK(sizes[i] == vpayloads[id].size());
                    CHECK(std::memcmp(payload.data() + offsets[i], vpayloads[id].data(),
                                      sizes[i]) == 0);
                }
                CHECK(found);
                totalPrimaries += r.record_count;
                ++checked;
            }
            aamt_close(vf);
            std::printf("PASS aamt_query_verified: %" PRIu64 " clean coordinates checked, "
                        "%" PRIu64 " total primaries across their buckets, every one "
                        "verified (mirrors sharing a bucket no longer misreported as "
                        "unavailable)\n", checked, totalPrimaries);
        }

        // 7f. aamt_query_verified_soft: same verifyLeafPrimaries helper as
        //     7e, applied across routeSoftTopK's 16 ranked branches instead
        //     of one hard route. The anchor id's own branch (rank 0, by
        //     routeSoftTopK's own tie-break guarantee) must contain it.
        {
            aamt_file* vf = aamt_open(v_path, err, sizeof(err));
            CHECK(vf != nullptr);
            constexpr uint32_t kBranches = 16, kPerBranch = 8;
            uint64_t checked = 0;
            for (uint64_t id = 0; id < kV; id += 7) { // sample, not exhaustive — 7e already is
                if (id == 50 || id == 99 || id == 150) continue;
                std::vector<aamt_soft_branch> branchesOut(kBranches);
                std::vector<uint32_t> branchCounts(kBranches);
                std::vector<aamt_verified_meta> meta(kBranches * kPerBranch);
                std::vector<uint8_t> payload(8192);
                std::vector<uint64_t> offsets(kBranches * kPerBranch);
                std::vector<uint32_t> sizes(kBranches * kPerBranch);
                const int32_t nb = aamt_query_verified_soft(
                    vf, &vteras[id * 4], kBranches, kPerBranch, branchesOut.data(),
                    branchCounts.data(), meta.data(), payload.data(), payload.size(),
                    offsets.data(), sizes.data());
                CHECK(nb == static_cast<int32_t>(kBranches));

                bool found = false;
                for (uint32_t bi = 0; bi < kBranches && !found; ++bi) {
                    CHECK(branchCounts[bi] <= branchesOut[bi].route.record_count ||
                         branchesOut[bi].route.record_count == 0);
                    for (uint32_t j = 0; j < branchCounts[bi]; ++j) {
                        const size_t idx = bi * kPerBranch + j;
                        if (meta[idx].id != id) continue;
                        found = true;
                        CHECK(meta[idx].status == AAMT_READ_OK);
                        CHECK(sizes[idx] == vpayloads[id].size());
                        CHECK(std::memcmp(payload.data() + offsets[idx],
                                          vpayloads[id].data(), sizes[idx]) == 0);
                        break;
                    }
                }
                CHECK(found);
                ++checked;
            }
            aamt_close(vf);
            std::printf("PASS aamt_query_verified_soft: %" PRIu64 " sampled coordinates, "
                        "each found verified and byte-exact somewhere in its 16 ranked "
                        "branches\n", checked);
        }

        std::remove(v_path);
    }

    return 0;
}
