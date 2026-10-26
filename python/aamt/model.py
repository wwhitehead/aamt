# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""Read `.aamt` model files (AMDL) into numpy arrays.

Pure numpy, no torch: footer → manifest → tensor blobs, exactly as SPEC.md §1–2
describe, with each quantised tensor dequantised by the same arithmetic as the
TypeScript reference (`js/packages/model/src/tensor.ts`, `dequantCPU`).

    from aamt.model import AamtModel
    with AamtModel("qwen3-0.6b.aamt") as m:
        print(m.names()[:3])
        w = m.tensor("model.layers.0.mlp.down_proj.weight")   # float32, shape (1024, 3072)

The file is memory-mapped, so opening it reads only the footer and the manifest,
and `tensor()` touches only that tensor's bytes. Every array returned is its own
copy: closing the file never invalidates one.
"""
from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AMDL_MAGIC = 0x4C444D41  # 'AMDL'
AAMT_MAGIC = 0x544D4141  # 'AAMT'
FOOTER_BYTES = 32
KINDS = {0: "int4", 1: "ternary", 2: "1bit"}
WEIGHTS_PER_WORD = {"int4": 8, "ternary": 16, "1bit": 32}


@dataclass(frozen=True)
class TensorEntry:
    name: str
    shape: tuple[int, ...]
    offset: int
    length: int
    kind: str  # int4 | ternary | 1bit | f32
    metadata: dict | None = None


@dataclass(frozen=True)
class Packed:
    """A quantised tensor as stored: the words the GPU reads, plus its group scales."""

    kind: str
    count: int
    group_size: int
    scales: np.ndarray  # float32, one per group
    packed: np.ndarray  # uint32


def parse_blob(blob: bytes) -> Packed:
    """Parse one quantised tensor blob (version 1 or 2)."""
    magic, version = struct.unpack_from("<II", blob, 0)
    if magic != AAMT_MAGIC:
        raise ValueError("not an .aamt tensor blob (bad magic)")
    if version == 1:
        kind, (count, group_size, group_count), header = "int4", struct.unpack_from("<III", blob, 8), 5
    elif version == 2:
        kind_id, count, group_size, group_count = struct.unpack_from("<IIII", blob, 8)
        if kind_id not in KINDS:
            raise ValueError(f"unknown tensor kind {kind_id}")
        kind, header = KINDS[kind_id], 6
    else:
        raise ValueError(f"unsupported tensor blob version {version}")
    scales = np.frombuffer(blob, dtype="<f4", count=group_count, offset=header * 4)
    words = -(-count // WEIGHTS_PER_WORD[kind])
    packed = np.frombuffer(blob, dtype="<u4", count=words, offset=(header + group_count) * 4)
    return Packed(kind, count, group_size, scales, packed)


def dequantize(p: Packed) -> np.ndarray:
    """Packed → float32, weight by weight identical to the TypeScript `dequantCPU`."""
    b = p.packed.view(np.uint8)  # little-endian: byte j of word w holds weights from the low bits up
    if p.kind == "int4":
        codes = np.stack([b & 0xF, b >> 4], axis=-1).reshape(-1)[: p.count]
        q = codes.astype(np.float32) - 8.0
    elif p.kind == "ternary":
        codes = np.stack([(b >> s) & 0x3 for s in (0, 2, 4, 6)], axis=-1).reshape(-1)[: p.count]
        q = np.select([codes == 1, codes == 3], [1.0, -1.0], 0.0).astype(np.float32)
    elif p.kind == "1bit":
        bits = np.unpackbits(b, bitorder="little")[: p.count]
        q = bits.astype(np.float32) * 2.0 - 1.0
    else:
        raise ValueError(f"unknown kind {p.kind}")
    scale = np.repeat(p.scales, p.group_size)[: p.count]
    return q * scale


class AamtModel:
    """A `.aamt` model file, memory-mapped. Use as a context manager, or call `close()`."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = open(self.path, "rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        size = len(self._map)
        if size < FOOTER_BYTES:
            raise ValueError("file too short to be an .aamt model")
        magic, version, count, _reserved, m_off, m_len = struct.unpack_from("<IIIIdd", self._map, size - FOOTER_BYTES)
        if magic != AMDL_MAGIC:
            raise ValueError("not an .aamt model (no AMDL footer)")
        if version != 1:
            raise ValueError(f"unsupported AMDL version {version}")
        m_off, m_len = int(m_off), int(m_len)
        if m_off + m_len > size - FOOTER_BYTES:
            raise ValueError("manifest lies outside the file")
        raw = json.loads(self._map[m_off : m_off + m_len].decode("utf-8"))
        self.entries: dict[str, TensorEntry] = {}
        for e in raw:
            entry = TensorEntry(e["name"], tuple(e["shape"]), int(e["offset"]), int(e["length"]), e.get("kind", "int4"), e.get("metadata"))
            if entry.offset + entry.length > m_off:
                raise ValueError(f"tensor {entry.name} lies outside the blob region")
            self.entries[entry.name] = entry
        if len(self.entries) != count:
            raise ValueError(f"footer says {count} tensors, manifest has {len(self.entries)}")

    def names(self) -> list[str]:
        """Real tensors, in file order (metadata-only entries excluded)."""
        return [n for n, e in self.entries.items() if e.metadata is None and e.length > 0]

    def raw(self, name: str) -> bytes:
        """The tensor's blob, copied out of the mapping, so nothing returned pins the file open."""
        e = self.entries[name]
        return self._map[e.offset : e.offset + e.length]

    def packed(self, name: str) -> Packed:
        e = self.entries[name]
        if e.kind == "f32":
            raise ValueError(f"{name} is f32, not quantised")
        p = parse_blob(self.raw(name))
        if p.kind != e.kind:
            raise ValueError(f"{name}: manifest says {e.kind}, blob says {p.kind}")
        return p

    def tensor(self, name: str) -> np.ndarray:
        """The tensor as float32 in its manifest shape (a copy; the file stays read-only)."""
        e = self.entries[name]
        if e.kind == "f32":
            data = np.frombuffer(self.raw(name), dtype="<f4").copy()
        else:
            data = dequantize(self.packed(name))
        return data.reshape(e.shape)

    def close(self) -> None:
        self._map.close()
        self._file.close()

    def __enter__(self) -> "AamtModel":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
