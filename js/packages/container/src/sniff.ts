// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * Classifying any `.aamt` file — the new container and the three legacy formats —
 * from its first {@link SNIFF_HEAD_BYTES} and last {@link SNIFF_TAIL_BYTES} bytes.
 *
 * Head and tail rather than the whole file, because a model is 7 GB and its
 * format marker is in a footer. The checks run in an order that matters:
 *
 * 1. **container** — the 8-byte magic at offset 0, plus `fileBytes` matching.
 * 2. **AMDL** — the footer magic. Checked *before* the `AAMT` head checks
 *    because an AMDL file's first tensor blob may itself begin with `AAMT`.
 * 3. **substrate v0** (`libaamt`) — `AAMT`, u16 major 1, and the header's own
 *    `file_size` at offset 8 equal to the real length.
 * 4. **tensor blob** (`voxel-runtime`) — `AAMT`, version 1 or 2, and the exact
 *    size the blob's own header implies.
 *
 * 3 and 4 are the collision ADR-005 exists for: a v1 blob and a v1.0 substrate
 * are identical through byte 8. They separate cleanly anyway, because the
 * substrate stores its own length at offset 8 and a valid blob cannot — there
 * the two u32s are `count` and `groupSize`, and `count + groupSize·2³²` equals
 * the file length only for a blob of `groupSize = 0`, which is not a valid blob.
 */

import { HEADER_BYTES, H_FILE_BYTES, H_VERSION_MAJOR, H_VERSION_MINOR, MAGIC } from './layout';

export const SNIFF_HEAD_BYTES = 64;
export const SNIFF_TAIL_BYTES = 32;

/** `AAMT` as a little-endian u32: shared by the tensor blob and the WP-23 substrate. */
export const LEGACY_AAMT_MAGIC = 0x544d4141;
/** `AMDL` as a little-endian u32, at offset 0 of the 32-byte footer. Same value as `plugin-sdk` `AMDL_MAGIC`. */
export const LEGACY_AMDL_MAGIC = 0x4c444d41;

export type AamtFormat =
  | { readonly format: 'container'; readonly versionMajor: number; readonly versionMinor: number }
  | { readonly format: 'amdl'; readonly version: number; readonly tensorCount: number }
  | { readonly format: 'substrate-v0'; readonly versionMajor: number; readonly versionMinor: number }
  | { readonly format: 'tensor-blob'; readonly version: 1 | 2; readonly kind: 'int4' | 'ternary' | '1bit' }
  | { readonly format: 'unknown'; readonly reason: string };

const BLOB_KIND = ['int4', 'ternary', '1bit'] as const;
const WEIGHTS_PER_U32 = [8, 16, 32] as const;

const view = (b: Uint8Array): DataView => new DataView(b.buffer, b.byteOffset, b.byteLength);

/**
 * Classify a file. `head` is its first bytes (at least 64 unless the file is
 * shorter), `tail` its last 32, `fileBytes` its full length.
 */
export function sniffAamt(head: Uint8Array, tail: Uint8Array, fileBytes: number): AamtFormat {
  const h = view(head);

  if (head.byteLength >= HEADER_BYTES / 4 && MAGIC.every((b, i) => head[i] === b)) {
    const declared = h.getBigUint64(H_FILE_BYTES, true);
    if (declared === BigInt(fileBytes)) {
      return {
        format: 'container',
        versionMajor: h.getUint16(H_VERSION_MAJOR, true),
        versionMinor: h.getUint16(H_VERSION_MINOR, true),
      };
    }
    return {
      format: 'unknown',
      reason: `container magic, but the header declares ${declared} bytes and the file has ${fileBytes} (truncated?)`,
    };
  }

  if (tail.byteLength === SNIFF_TAIL_BYTES && fileBytes >= SNIFF_TAIL_BYTES) {
    const t = view(tail);
    if (t.getUint32(0, true) === LEGACY_AMDL_MAGIC) {
      const version = t.getUint32(4, true);
      const manifestOffset = t.getFloat64(16, true);
      const manifestLength = t.getFloat64(24, true);
      if (
        version === 1 &&
        manifestOffset >= 0 &&
        manifestLength >= 0 &&
        manifestOffset + manifestLength <= fileBytes - SNIFF_TAIL_BYTES
      ) {
        return { format: 'amdl', version, tensorCount: t.getUint32(8, true) };
      }
      return { format: 'unknown', reason: 'AMDL footer magic, but the version or manifest range is invalid' };
    }
  }

  if (head.byteLength < 16 || h.getUint32(0, true) !== LEGACY_AAMT_MAGIC) {
    return { format: 'unknown', reason: 'no container magic, no AMDL footer, and no AAMT head' };
  }

  if (h.getUint16(4, true) === 1 && h.getBigUint64(8, true) === BigInt(fileBytes)) {
    return { format: 'substrate-v0', versionMajor: 1, versionMinor: h.getUint16(6, true) };
  }

  const version = h.getUint32(4, true);
  if ((version === 1 && head.byteLength >= 20) || (version === 2 && head.byteLength >= 24)) {
    const base = version === 1 ? 2 : 3;
    const kindId = version === 1 ? 0 : h.getUint32(8, true);
    const count = h.getUint32(base * 4, true);
    const groupSize = h.getUint32((base + 1) * 4, true);
    const groupCount = h.getUint32((base + 2) * 4, true);
    const wpu = WEIGHTS_PER_U32[kindId as 0 | 1 | 2];
    if (wpu !== undefined && groupSize > 0 && groupCount === Math.ceil(count / groupSize)) {
      const headerU32 = version === 1 ? 5 : 6;
      const expected = (headerU32 + groupCount) * 4 + Math.ceil(count / wpu) * 4;
      if (expected === fileBytes)
        return { format: 'tensor-blob', version, kind: BLOB_KIND[kindId as 0 | 1 | 2] };
    }
  }
  return {
    format: 'unknown',
    reason: 'AAMT head, but neither the substrate nor the tensor-blob size invariant holds',
  };
}

/** {@link sniffAamt} over a whole in-memory file. */
export function sniffBytes(bytes: Uint8Array): AamtFormat {
  const n = bytes.byteLength;
  return sniffAamt(
    bytes.subarray(0, Math.min(n, SNIFF_HEAD_BYTES)),
    bytes.subarray(Math.max(0, n - SNIFF_TAIL_BYTES)),
    n,
  );
}
