// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// RS(15,13) over GF(16) — a single-symbol-error-correcting Reed-Solomon
// code, chosen because GF(16) is the substrate's OWN alphabet (a Meji
// index is one symbol; an Odu is two), not a generic byte-oriented code
// bolted on from outside. n=15 is GF(16)*'s full order — the largest RS
// block this field supports without an extended/shortened construction.
//
// Narrow-sense, two-root generator: g(x) = (x - alpha^0)(x - alpha^1).
// Minimum distance d = n-k+1 = 3: standard coding theory reads this as
// "detects up to d-1=2 errors, corrects up to floor((d-1)/2)=1" — but that
// textbook line describes the CODE's distance, not what THIS decoder (only
// 2 parity symbols, i.e. minimum redundancy for t=1) can independently
// verify. Measured exhaustively (tests/test_rs_gf16.cpp, all 23,625
// two-error patterns): when a SECOND error is present, decode() reports
// "corrected" and returns the WRONG data 86.7% of the time (20,475/23,625)
// — not a rare edge case, the dominant outcome. The reason is algebraic,
// not a bug: with exactly 2 syndrome equations (S0, S1) and 2 unknowns
// (one hypothesized error's position and magnitude), solving for that
// single hypothesis CONSUMES both equations, so the post-correction
// re-check is guaranteed to pass — self-cancellation, not verification —
// whenever the initial solve finds any candidate at all. Only 13.3%
// (3,150 cases, an exact, derivable count: the pairs where the two real
// errors coincidentally sum or ratio to zero at alpha^0 or alpha^1)
// happen to be flagged uncorrectable, and never once does a 2-error
// pattern land on the exactly-correct answer by chance.
//
// The conclusion this measurement demands: use this code for what it
// verifiably does — free, exact repair of the common single-nibble-flip
// case — and never trust its "ok=true" as a broad integrity guarantee.
// Pair it with an INDEPENDENT check (WP-23's mirror pairing: a full,
// separately-stored payload comparison) for anything beyond that; do not
// try to fix this by adding more parity symbols to the same 2-equation
// system, since that trades payload capacity for a guarantee the mirror
// already provides for free.
#pragma once

#include "gf16.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace aamt::rs15_13 {

constexpr int kN = 15;
constexpr int kK = 13;
constexpr int kParity = kN - kK; // 2

using Codeword = std::array<uint8_t, kN>;
using Data = std::array<uint8_t, kK>;

// Codeword[0..1] = parity, Codeword[2..14] = data[0..12] (data[i] is the
// coefficient of x^(i+kParity); parity[j] the coefficient of x^j).
inline Codeword encode(const Data& data) {
    Codeword shifted{};
    for (int i = 0; i < kK; ++i) shifted[i + kParity] = data[i];

    // g(x) = x^2 + g1*x + g0, roots alpha^0 (=1) and alpha^1.
    const uint8_t g0 = gf16::alpha(1);
    const uint8_t g1 = gf16::add(1, gf16::alpha(1));

    Codeword rem = shifted;
    for (int i = kN - 1; i >= kParity; --i) {
        const uint8_t coef = rem[i];
        if (coef == 0) continue;
        rem[i] = 0;
        rem[i - 1] = gf16::add(rem[i - 1], gf16::mul(coef, g1));
        rem[i - 2] = gf16::add(rem[i - 2], gf16::mul(coef, g0));
    }

    Codeword codeword = shifted;
    codeword[0] = rem[0];
    codeword[1] = rem[1];
    return codeword;
}

inline uint8_t evalPoly(const Codeword& poly, uint8_t x0) {
    uint8_t result = 0;
    uint8_t xp = 1;
    for (int i = 0; i < kN; ++i) {
        result = gf16::add(result, gf16::mul(poly[i], xp));
        xp = gf16::mul(xp, x0);
    }
    return result;
}

struct DecodeResult {
    bool ok = false;        // false: syndrome inconsistent with 0 or 1 errors
    bool corrected = false; // true: one symbol was flipped to reach a zero syndrome
    int errorPos = -1;      // position corrected, if corrected
    Data data{};
};

inline DecodeResult decode(Codeword received) {
    const uint8_t S0 = evalPoly(received, gf16::alpha(0));
    const uint8_t S1 = evalPoly(received, gf16::alpha(1));

    DecodeResult out;
    if (S0 == 0 && S1 == 0) {
        out.ok = true;
        for (int i = 0; i < kK; ++i) out.data[i] = received[i + kParity];
        return out;
    }
    if (S0 == 0) return out; // S1 != 0 here is inconsistent with a single error

    const uint8_t Y = S0;
    const uint8_t alphaI = gf16::div(S1, S0);
    if (alphaI == 0) return out; // inconsistent (would need alpha^i = 0, impossible)
    const int pos = gf16::tables().log_[alphaI];

    received[pos] = gf16::add(received[pos], Y);
    if (evalPoly(received, gf16::alpha(0)) != 0 || evalPoly(received, gf16::alpha(1)) != 0) {
        return out; // the single-error hypothesis didn't clear the syndrome — not fixable at t=1
    }

    out.ok = true;
    out.corrected = true;
    out.errorPos = pos;
    for (int i = 0; i < kK; ++i) out.data[i] = received[i + kParity];
    return out;
}

// ---- byte-oriented payload wrapper ---------------------------------------
// Arbitrary payloads are 2 nibbles/byte, chunked into kK=13-nibble blocks
// (the final block zero-padded — reversible, since the true byte count is
// stored in a 4-byte header), each independently RS-encoded to 15 nibbles.
// One corrupted nibble ANYWHERE in the payload is repaired for free,
// independent of every other block. See the file-level comment for what
// this does NOT do (reliably handle 2 corrupted nibbles in the same
// 13-nibble/6.5-byte block).

inline std::vector<uint8_t> encodePayload(const uint8_t* data, uint32_t size) {
    std::vector<uint8_t> out = {
        static_cast<uint8_t>(size & 0xFF), static_cast<uint8_t>((size >> 8) & 0xFF),
        static_cast<uint8_t>((size >> 16) & 0xFF), static_cast<uint8_t>((size >> 24) & 0xFF),
    };

    std::vector<uint8_t> nibbles;
    nibbles.reserve(static_cast<size_t>(size) * 2);
    for (uint32_t i = 0; i < size; ++i) {
        nibbles.push_back(static_cast<uint8_t>(data[i] >> 4));
        nibbles.push_back(static_cast<uint8_t>(data[i] & 0xF));
    }

    std::vector<uint8_t> encodedNibbles;
    for (size_t off = 0; off < nibbles.size(); off += kK) {
        Data block{};
        const size_t n = std::min<size_t>(kK, nibbles.size() - off);
        for (size_t i = 0; i < n; ++i) block[i] = nibbles[off + i];
        for (uint8_t v : encode(block)) encodedNibbles.push_back(v);
    }
    if (encodedNibbles.size() % 2 != 0) encodedNibbles.push_back(0); // byte-align

    for (size_t i = 0; i < encodedNibbles.size(); i += 2) {
        out.push_back(static_cast<uint8_t>((encodedNibbles[i] << 4) | encodedNibbles[i + 1]));
    }
    return out;
}

struct PayloadDecodeResult {
    bool ok = true;              // false only if a block's syndrome was unresolvable
    uint32_t correctedSymbols = 0; // count of blocks where a single-nibble repair fired
    std::vector<uint8_t> data;
};

inline PayloadDecodeResult decodePayload(const uint8_t* encoded, uint32_t encodedSize) {
    PayloadDecodeResult result;
    if (encodedSize < 4) { result.ok = false; return result; }
    const uint32_t originalSize = static_cast<uint32_t>(encoded[0]) |
                                  (static_cast<uint32_t>(encoded[1]) << 8) |
                                  (static_cast<uint32_t>(encoded[2]) << 16) |
                                  (static_cast<uint32_t>(encoded[3]) << 24);

    std::vector<uint8_t> nibbles;
    nibbles.reserve((static_cast<size_t>(encodedSize) - 4) * 2);
    for (uint32_t i = 4; i < encodedSize; ++i) {
        nibbles.push_back(static_cast<uint8_t>(encoded[i] >> 4));
        nibbles.push_back(static_cast<uint8_t>(encoded[i] & 0xF));
    }

    std::vector<uint8_t> decodedNibbles;
    for (size_t off = 0; off + kN <= nibbles.size(); off += kN) {
        Codeword cw{};
        for (int i = 0; i < kN; ++i) cw[i] = nibbles[off + i];
        const auto dr = decode(cw);
        if (!dr.ok) { result.ok = false; return result; }
        if (dr.corrected) ++result.correctedSymbols;
        for (uint8_t v : dr.data) decodedNibbles.push_back(v);
    }

    if (static_cast<uint64_t>(originalSize) * 2 > decodedNibbles.size()) {
        result.ok = false;
        return result;
    }
    result.data.reserve(originalSize);
    for (uint32_t i = 0; i < originalSize; ++i) {
        result.data.push_back(static_cast<uint8_t>(
            (decodedNibbles[i * 2] << 4) | decodedNibbles[i * 2 + 1]));
    }
    return result;
}

} // namespace aamt::rs15_13
