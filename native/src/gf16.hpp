// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// GF(16) = GF(2^4) field arithmetic, built from the primitive polynomial
// x^4 + x + 1 (0b10011) — the standard textbook choice, the same
// construction QR codes/CDs use at a larger field size. GF(16)* is cyclic
// of order 15; log/antilog tables make multiply/divide O(1) table lookups.
//
// This exists for one reason: the substrate's native alphabet is the
// 4-bit nibble (a Meji index is one GF(16) symbol; an Odu is two). An
// error-correcting code over bytes would be a generic, borrowed tool;
// one over GF(16) speaks the format's own alphabet. See rs_gf16.hpp for
// the code built on top of this field.
#pragma once

#include <array>
#include <cstdint>

namespace aamt::gf16 {

constexpr int kOrder = 15; // |GF(16)*|

// log_[0] is unused (log(0) is undefined); antilog_ is indexed mod 15.
struct Tables {
    std::array<uint8_t, 16> log_{};
    std::array<uint8_t, 15> antilog_{};
};

constexpr Tables buildTables() {
    Tables t{};
    // Primitive polynomial x^4 + x + 1: reduce by XORing 0b10011 (0x13)
    // whenever the running value's bit 4 is set, matching standard
    // GF(2^4) generation via repeated "multiply by x, reduce mod p(x)".
    uint32_t x = 1;
    for (int i = 0; i < 15; ++i) {
        t.antilog_[i] = static_cast<uint8_t>(x);
        t.log_[x] = static_cast<uint8_t>(i);
        x <<= 1;
        if (x & 0x10) x ^= 0x13;
    }
    return t;
}

inline const Tables& tables() {
    static const Tables t = buildTables();
    return t;
}

inline uint8_t add(uint8_t a, uint8_t b) { return a ^ b; } // GF(2^m): addition is XOR

inline uint8_t mul(uint8_t a, uint8_t b) {
    if (a == 0 || b == 0) return 0;
    const auto& t = tables();
    return t.antilog_[(t.log_[a] + t.log_[b]) % kOrder];
}

inline uint8_t div(uint8_t a, uint8_t b) {
    if (a == 0) return 0;
    const auto& t = tables();
    const int d = (static_cast<int>(t.log_[a]) - static_cast<int>(t.log_[b]) + kOrder) % kOrder;
    return t.antilog_[d];
}

inline uint8_t pow(uint8_t base, int exp) {
    const auto& t = tables();
    if (base == 0) return exp == 0 ? 1 : 0;
    const int e = ((t.log_[base] * exp) % kOrder + kOrder) % kOrder;
    return t.antilog_[e];
}

// The field's primitive element, alpha = 2 (x itself) under this
// construction — alpha^0..alpha^14 enumerate all 15 nonzero elements.
inline uint8_t alpha(int exp) { return pow(2, exp); }

} // namespace aamt::gf16
