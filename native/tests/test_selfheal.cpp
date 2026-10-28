// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// Self-repair, end to end through the C ABI: damage a packed file's bytes and check what aamt_read_verified does.
//
// The rule these pin: a CHECKED record (aamt_builder_add_mirrored_ecc_checked) is either read correctly — clean,
// RS-repaired, or recovered from its mirror — or refused. It is never served wrong with a success status. And the
// reason the check exists is pinned too: the unchecked encoding, under heavy damage, does serve wrong bytes as OK,
// because RS(15,13) repairs garbage into a valid-looking different payload.
#include "aamt/aamt.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

namespace {
constexpr uint64_t kRecords = 24;
constexpr uint64_t kTarget = 5;
constexpr int kTrials = 400;

std::string payloadOf(uint64_t id) { return "Memory " + std::to_string(id) + ": Ada saw the lantern by the well and thought of her brother."; }
void teraOf(uint64_t id, float t[4]) {
    t[0] = static_cast<float>((id * 37) % 97) / 97.f;
    t[1] = static_cast<float>((id * 53) % 89) / 89.f;
    t[2] = static_cast<float>((id * 71) % 83) / 83.f;
    t[3] = static_cast<float>((id * 29) % 79) / 79.f;
}

std::vector<uint8_t> pack(const char* path, bool checked) {
    aamt_builder* b = aamt_builder_new(6, 4);
    for (uint64_t id = 1; id <= kRecords; ++id) {
        float t[4];
        teraOf(id, t);
        const std::string p = payloadOf(id);
        if (checked) aamt_builder_add_mirrored_ecc_checked(b, id, t, p.data(), static_cast<uint32_t>(p.size()));
        else aamt_builder_add_mirrored_ecc(b, id, t, p.data(), static_cast<uint32_t>(p.size()));
    }
    char err[256];
    CHECK(aamt_builder_write(b, path, err, sizeof(err)) == 0);
    aamt_builder_free(b);
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

/** Where a record's two stored copies sit in the file: both copies hold the same encoded bytes, so search for them. */
std::vector<size_t> copiesOf(const std::vector<uint8_t>& file, const std::vector<uint8_t>& encoded) {
    std::vector<size_t> at;
    for (size_t i = 0; i + encoded.size() <= file.size(); ++i)
        if (std::memcmp(file.data() + i, encoded.data(), encoded.size()) == 0) at.push_back(i);
    return at;
}

struct Read { uint32_t status; bool correct; };
Read readTarget(const char* path, const std::vector<uint8_t>& bytes) {
    { std::ofstream out(path, std::ios::binary); out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())); }
    char err[256];
    aamt_file* f = aamt_open(path, err, sizeof(err));
    CHECK(f != nullptr);
    if (!f) return { AAMT_READ_UNAVAILABLE, false };
    float t[4], outT[4];
    teraOf(kTarget, t);
    std::vector<uint8_t> out(1 << 16);
    uint32_t status = 0, corrected = 0;
    const int64_t n = aamt_read_verified(f, kTarget, t, out.data(), out.size(), outT, &status, &corrected);
    const std::string want = payloadOf(kTarget);
    const bool correct = n == static_cast<int64_t>(want.size()) && std::memcmp(out.data(), want.data(), want.size()) == 0;
    aamt_close(f);
    return { status, correct };
}

/** The encoded bytes a record of this payload carries, so its copies can be found in the file. */
std::vector<uint8_t> encodedFor(bool checked, const std::string& p, const char* probePath) {
    if (!checked) {
        std::vector<uint8_t> e(aamt_encode_ecc_payload(reinterpret_cast<const uint8_t*>(p.data()), static_cast<uint32_t>(p.size()), nullptr, 0));
        aamt_encode_ecc_payload(reinterpret_cast<const uint8_t*>(p.data()), static_cast<uint32_t>(p.size()), e.data(), static_cast<uint32_t>(e.size()));
        return e;
    }
    // The checked encoding has no stateless encoder in the ABI; read a record's raw stored bytes back instead.
    aamt_builder* b = aamt_builder_new(6, 4);
    const float t[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
    aamt_builder_add_ecc_checked(b, 1, t, p.data(), static_cast<uint32_t>(p.size()));
    char err[256];
    aamt_builder_write(b, probePath, err, sizeof(err));
    aamt_builder_free(b);
    aamt_file* f = aamt_open(probePath, err, sizeof(err));
    aamt_record_view v{};
    aamt_route r{};
    aamt_query(f, t, &v, 1, &r);
    std::vector<uint8_t> e(v.data, v.data + v.size);
    aamt_close(f);
    return e;
}

uint32_t rng = 12345u;
double rnd() { rng = rng * 1664525u + 1013904223u; return rng / 4294967296.0; }

struct Tally { int correct = 0, refused = 0, wrongButOk = 0; };
Tally trials(const std::vector<uint8_t>& clean, const std::vector<size_t>& at, size_t encodedSize, int flips, const char* path) {
    Tally t;
    for (int k = 0; k < kTrials; ++k) {
        std::vector<uint8_t> b = clean;
        for (int j = 0; j < flips; ++j) {
            const size_t copy = at[rnd() < 0.5 ? 0 : 1];
            const size_t off = copy + 4 + static_cast<size_t>(rnd() * static_cast<double>(encodedSize - 4));
            const uint8_t nib = static_cast<uint8_t>(1 + static_cast<int>(rnd() * 15));
            b[off] ^= rnd() < 0.5 ? nib : static_cast<uint8_t>(nib << 4);
        }
        const Read r = readTarget(path, b);
        if (r.status == AAMT_READ_UNAVAILABLE) ++t.refused;
        else if (r.correct) ++t.correct;
        else ++t.wrongButOk;
    }
    return t;
}
} // namespace

int main() {
    const char* path = "/tmp/aamt_selfheal_test.aamt";
    const char* probe = "/tmp/aamt_selfheal_probe.aamt";

    for (bool checked : { false, true }) {
        const std::vector<uint8_t> clean = pack(path, checked);
        const std::vector<uint8_t> enc = encodedFor(checked, payloadOf(kTarget), probe);
        const std::vector<size_t> at = copiesOf(clean, enc);
        CHECK(at.size() == 2); // primary and mirror, the same encoded bytes in two places
        if (at.size() != 2) continue;

        const Read cleanRead = readTarget(path, clean);
        CHECK(cleanRead.status == AAMT_READ_OK && cleanRead.correct);

        const Tally light = trials(clean, at, enc.size(), 1, path);
        CHECK(light.correct == kTrials); // one flipped nibble anywhere: always repaired or unaffected

        const Tally heavy = trials(clean, at, enc.size(), 20, path);
        std::printf("%s: 20 flips over both copies, %d trials: %d correct, %d refused, %d wrong-but-OK\n",
                    checked ? "checked  " : "unchecked", kTrials, heavy.correct, heavy.refused, heavy.wrongButOk);
        if (checked) {
            CHECK(heavy.wrongButOk == 0); // the rule: never serve wrong bytes as OK

            // Shred each copy in turn: the other copy serves, correctly. This is the repair the CRC makes possible.
            for (int which = 0; which < 2; ++which) {
                std::vector<uint8_t> b = clean;
                for (size_t i = 4; i < enc.size(); ++i) b[at[which] + i] ^= 0x5a;
                const Read r = readTarget(path, b);
                CHECK(r.correct);
                CHECK(r.status == AAMT_READ_OK || r.status == AAMT_READ_OK_MIRROR_RECOVERED);
            }
        } else {
            CHECK(heavy.wrongButOk > 0); // why the check exists: unchecked RS serves garbage as OK under heavy damage
        }
    }

    // The checked raw decoder refuses a mis-correction the unchecked one would hand back.
    {
        const std::string p = payloadOf(kTarget);
        std::vector<uint8_t> enc = encodedFor(true, p, probe);
        for (size_t i = 4; i < enc.size(); ++i) enc[i] ^= 0x5a;
        std::vector<uint8_t> out(1024);
        uint32_t corrected = 0;
        const int64_t n = aamt_decode_ecc_checked_payload(enc.data(), static_cast<uint32_t>(enc.size()), out.data(), out.size(), &corrected);
        CHECK(n == -1);
    }

    std::remove(path);
    std::remove(probe);
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("PASS self-heal: checked records are read correctly or refused, never served wrong; one shredded copy is recovered from the other\n");
    return 0;
}
