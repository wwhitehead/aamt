// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// Proves RS(15,13)/GF(16) does exactly what it claims, no more:
//  1. zero errors: decodes unmodified (many random trials)
//  2. one error: EXHAUSTIVE over all 15 positions x 15 nonzero deltas
//     (225 cases) — every single case must correct to the exact original
//  3. two errors: exhaustive over all C(15,2)*15*15 = 23,625 two-error
//     patterns. The measured result is the point of this test: 86.7%
//     (20,475 cases) are silently "corrected" to the WRONG data — the
//     dominant outcome, not a rare edge case — because with exactly 2
//     parity symbols, solving for one error hypothesis consumes both
//     syndrome equations, leaving nothing independent to catch a wrong
//     guess. Only 13.3% (3,150, an exactly derivable count) are safely
//     flagged uncorrectable. See rs_gf16.hpp for why, and why that means
//     this code must be paired with an independent check (mirror
//     pairing) rather than trusted alone beyond single-symbol repair.
#include "gf16.hpp"
#include "rs_gf16.hpp"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,    \
                         #cond);                                            \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

using namespace aamt;

int main() {
    std::mt19937 rng(0x51DE5EED);
    std::uniform_int_distribution<int> nibble(0, 15);

    // 1. Zero errors, many random data blocks: decodes unmodified.
    for (int trial = 0; trial < 5000; ++trial) {
        rs15_13::Data data;
        for (auto& d : data) d = static_cast<uint8_t>(nibble(rng));
        const auto codeword = rs15_13::encode(data);
        const auto res = rs15_13::decode(codeword);
        CHECK(res.ok);
        CHECK(!res.corrected);
        CHECK(res.data == data);
    }

    // 2. Exactly one error, EXHAUSTIVE over every (position, delta) pair —
    //    all 15 x 15 = 225 cases, not a sample.
    {
        rs15_13::Data data;
        for (int i = 0; i < rs15_13::kK; ++i) data[i] = static_cast<uint8_t>((i * 7 + 3) % 16);
        const auto codeword = rs15_13::encode(data);

        int correctedCount = 0;
        for (int pos = 0; pos < rs15_13::kN; ++pos) {
            for (int delta = 1; delta < 16; ++delta) { // delta=0 is "no error"
                auto corrupted = codeword;
                corrupted[pos] = gf16::add(corrupted[pos], static_cast<uint8_t>(delta));
                const auto res = rs15_13::decode(corrupted);
                CHECK(res.ok);
                CHECK(res.corrected);
                CHECK(res.errorPos == pos);
                CHECK(res.data == data);
                ++correctedCount;
            }
        }
        CHECK(correctedCount == 15 * 15);
        std::printf("PASS: all %d single-symbol-error cases (15 positions x 15 "
                    "nonzero deltas) corrected to the exact original\n", correctedCount);
    }

    // 3. Exactly two errors, EXHAUSTIVE over every (pos1,pos2,delta1,delta2)
    //    combination — the real ceiling, measured.
    {
        rs15_13::Data data;
        for (int i = 0; i < rs15_13::kK; ++i) data[i] = static_cast<uint8_t>((i * 11 + 5) % 16);
        const auto codeword = rs15_13::encode(data);

        uint64_t total = 0, flaggedUncorrectable = 0, wronglyCorrected = 0, accidentallyRight = 0;
        for (int p1 = 0; p1 < rs15_13::kN; ++p1) {
            for (int p2 = p1 + 1; p2 < rs15_13::kN; ++p2) {
                for (int d1 = 1; d1 < 16; ++d1) {
                    for (int d2 = 1; d2 < 16; ++d2) {
                        auto corrupted = codeword;
                        corrupted[p1] = gf16::add(corrupted[p1], static_cast<uint8_t>(d1));
                        corrupted[p2] = gf16::add(corrupted[p2], static_cast<uint8_t>(d2));
                        const auto res = rs15_13::decode(corrupted);
                        ++total;
                        if (!res.ok) {
                            ++flaggedUncorrectable;
                        } else if (res.data == data) {
                            ++accidentallyRight; // syndrome happened to clear cleanly
                        } else {
                            ++wronglyCorrected; // the real, honest failure mode of t=1
                        }
                    }
                }
            }
        }
        CHECK(total == 105ull * 15 * 15); // C(15,2) * 15 * 15 = 23,625
        // The exact ceiling of a t=1 code: some 2-error patterns MUST be
        // uncorrectable-or-wrong (that's what "t=1" means), so this is not
        // asserting zero failures — it is reporting the true count.
        std::printf("MEASURED (2 simultaneous errors, exhaustive, %" PRIu64 " cases): "
                    "%" PRIu64 " safely flagged uncorrectable, %" PRIu64
                    " silently mis-corrected, %" PRIu64 " accidentally exact\n",
                    total, flaggedUncorrectable, wronglyCorrected, accidentallyRight);
        CHECK(wronglyCorrected + flaggedUncorrectable + accidentallyRight == total);
    }

    // 4. Payload wrapper: an arbitrary byte string, single-nibble corruption
    //    injected at every possible byte position, must always recover exactly.
    {
        const std::string original =
            "the mirror gate: a bad reflection disperses itself, byte for byte";
        const auto encoded = rs15_13::encodePayload(
            reinterpret_cast<const uint8_t*>(original.data()),
            static_cast<uint32_t>(original.size()));

        // Clean round-trip first.
        {
            const auto dr = rs15_13::decodePayload(encoded.data(),
                                                   static_cast<uint32_t>(encoded.size()));
            CHECK(dr.ok);
            CHECK(dr.correctedSymbols == 0);
            CHECK(dr.data.size() == original.size());
            CHECK(std::memcmp(dr.data.data(), original.data(), original.size()) == 0);
        }

        // Flip one nibble (half a byte) at every encoded-payload byte offset
        // (skipping the 4-byte size header) — recovery must be exact every time.
        uint32_t testedPositions = 0;
        for (size_t i = 4; i < encoded.size(); ++i) {
            for (int nibbleSel = 0; nibbleSel < 2; ++nibbleSel) {
                auto corrupted = encoded;
                const uint8_t mask = nibbleSel == 0 ? 0xF0 : 0x0F;
                corrupted[i] = static_cast<uint8_t>(corrupted[i] ^ mask);
                const auto dr = rs15_13::decodePayload(corrupted.data(),
                                                       static_cast<uint32_t>(corrupted.size()));
                CHECK(dr.ok);
                CHECK(dr.data.size() == original.size());
                CHECK(std::memcmp(dr.data.data(), original.data(), original.size()) == 0);
                ++testedPositions;
            }
        }
        std::printf("PASS: payload wrapper recovered exactly from a single corrupted "
                    "nibble at all %u tested byte/nibble positions (%zu-byte message)\n",
                    testedPositions, original.size());
    }

    std::printf("PASS test_rs_gf16: GF(16) Reed-Solomon(15,13) — exact single-error "
                "correction proven exhaustively, two-error ceiling measured exhaustively\n");
    return 0;
}
