# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""The Python reader against the TypeScript writer and reader, weight by weight.

    python -m unittest discover -s python/tests

Needs only numpy. `fixtures/mixed.aamt` and its expected values are written by
`js/packages/model/scripts/make-python-fixture.ts`.
"""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from aamt.model import AamtModel, dequantize, parse_blob  # noqa: E402

FIXTURE = HERE / "fixtures" / "mixed.aamt"
EXPECTED = json.loads((HERE / "fixtures" / "mixed.expected.json").read_text())


class ReadsWhatTypeScriptWrote(unittest.TestCase):
    def test_every_tensor_matches_exactly(self):
        with AamtModel(FIXTURE) as m:
            self.assertEqual(sorted(m.names()), sorted(EXPECTED))
            for name, exp in EXPECTED.items():
                with self.subTest(name=name):
                    got = m.tensor(name)
                    self.assertEqual(list(got.shape), exp["shape"])
                    self.assertEqual(m.entries[name].kind, exp["kind"])
                    np.testing.assert_array_equal(got.reshape(-1), np.asarray(exp["values"], dtype=np.float32))

    def test_packed_words_are_the_file_bytes(self):
        with AamtModel(FIXTURE) as m:
            p = m.packed("1bit")
            self.assertEqual((p.kind, p.count, p.group_size), ("1bit", 320, 32))
            self.assertEqual(p.packed.size, 10)  # 320 weights, 32 per word


class RefusesBadFiles(unittest.TestCase):
    def write(self, data: bytes) -> Path:
        f = tempfile.NamedTemporaryFile(suffix=".aamt", delete=False)
        f.write(data)
        f.close()
        self.addCleanup(Path(f.name).unlink)
        return Path(f.name)

    def test_no_footer(self):
        with self.assertRaisesRegex(ValueError, "AMDL"):
            AamtModel(self.write(b"\0" * 64))

    def test_manifest_outside_file(self):
        footer = struct.pack("<IIIIdd", 0x4C444D41, 1, 0, 0, 0.0, 1e6)
        with self.assertRaisesRegex(ValueError, "outside"):
            AamtModel(self.write(b"\0" * 16 + footer))

    def test_kind_mismatch_between_manifest_and_blob(self):
        data = bytearray(FIXTURE.read_bytes())
        size = len(data)
        _, _, count, _, m_off, m_len = struct.unpack_from("<IIIIdd", data, size - 32)
        manifest = json.loads(data[int(m_off) : int(m_off + m_len)])
        for e in manifest:
            if e["name"] == "int4":
                e["kind"] = "ternary"
        new = json.dumps(manifest).encode()
        out = bytes(data[: int(m_off)]) + new + struct.pack("<IIIIdd", 0x4C444D41, 1, count, 0, m_off, float(len(new)))
        with AamtModel(self.write(out)) as m, self.assertRaisesRegex(ValueError, "manifest says ternary"):
            m.tensor("int4")


class Dequantisation(unittest.TestCase):
    def blob(self, version_kind: tuple[int, int | None], count: int, group: int, scales, words) -> bytes:
        version, kind = version_kind
        head = struct.pack("<II", 0x544D4141, version) + (struct.pack("<I", kind) if kind is not None else b"")
        head += struct.pack("<III", count, group, len(scales))
        return head + np.asarray(scales, "<f4").tobytes() + np.asarray(words, "<u4").tobytes()

    def test_int4_codes(self):
        # nibbles 0..7 in one word: values (n - 8) * scale
        word = sum(n << (4 * n) for n in range(8))
        p = parse_blob(self.blob((2, 0), 8, 8, [0.5], [word]))
        np.testing.assert_array_equal(dequantize(p), (np.arange(8) - 8) * 0.5)

    def test_version_1_is_int4(self):
        p = parse_blob(self.blob((1, None), 8, 8, [1.0], [0x88888888]))
        self.assertEqual(p.kind, "int4")
        np.testing.assert_array_equal(dequantize(p), np.zeros(8))

    def test_ternary_codes(self):
        # codes 00, 01, 11, 10 -> 0, +s, -s, 0
        word = 0b00 | (0b01 << 2) | (0b11 << 4) | (0b10 << 6)
        p = parse_blob(self.blob((2, 1), 4, 32, [2.0], [word]))
        np.testing.assert_array_equal(dequantize(p), [0, 2, -2, 0])

    def test_1bit_codes(self):
        p = parse_blob(self.blob((2, 2), 4, 32, [3.0], [0b0101]))
        np.testing.assert_array_equal(dequantize(p), [3, -3, 3, -3])


if __name__ == "__main__":
    unittest.main()
