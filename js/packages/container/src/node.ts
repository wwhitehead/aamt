// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * The `node:crypto` bindings of the {@link Sha384}, {@link Aead} and {@link Kdf} ports, and the file-handle
 * source and sink for the streaming writer. Kept out of the core so the core imports nothing at runtime;
 * Kernel, CLI and tests import this entry.
 */

import { createCipheriv, createDecipheriv, createHash, hkdfSync } from 'node:crypto';
import { open } from 'node:fs/promises';
import type { Aead, AeadStream, CryptoPorts, Kdf } from './crypto';
import type { Sha384, Sha384Incremental } from './identity';
import type { ChunkSource, ContainerSink } from './stream';

export const nodeSha384: Sha384 = (parts) => {
  const h = createHash('sha384');
  for (const p of parts) h.update(p);
  return new Uint8Array(h.digest());
};

/** Incremental SHA-384 over `node:crypto`; `update` copies into the hash state, so buffers may be reused. */
export const nodeSha384Incremental: Sha384Incremental = () => {
  const h = createHash('sha384');
  return {
    update(bytes) {
      h.update(bytes);
    },
    digest: () => new Uint8Array(h.digest()),
  };
};

/** AES-256-GCM over `node:crypto`. */
export const nodeAead: Aead = {
  seal(key, nonce, plaintext, aad) {
    const c = createCipheriv('aes-256-gcm', key, nonce);
    c.setAAD(aad);
    const body = Buffer.concat([c.update(plaintext), c.final()]);
    return new Uint8Array(Buffer.concat([body, c.getAuthTag()]));
  },
  open(key, nonce, sealed, aad) {
    if (sealed.byteLength < 16) return null;
    try {
      const d = createDecipheriv('aes-256-gcm', key, nonce);
      d.setAAD(aad);
      d.setAuthTag(sealed.subarray(sealed.byteLength - 16));
      return new Uint8Array(Buffer.concat([d.update(sealed.subarray(0, sealed.byteLength - 16)), d.final()]));
    } catch {
      return null;
    }
  },
};

/** Incremental AES-256-GCM sealing over `node:crypto`. */
export const nodeAeadStream: AeadStream = {
  seal(key, nonce, aad) {
    const c = createCipheriv('aes-256-gcm', key, nonce);
    c.setAAD(aad);
    return {
      update: (chunk) => c.update(chunk),
      final: () => Buffer.concat([c.final(), c.getAuthTag()]),
    };
  },
};

/** HKDF-SHA-384 over `node:crypto`. */
export const nodeKdf: Kdf = (ikm, salt, info, length) =>
  new Uint8Array(hkdfSync('sha384', ikm, salt, info, length));

/** All the primitives, bound to `node:crypto`. */
export const nodeCrypto: CryptoPorts = {
  aead: nodeAead,
  kdf: nodeKdf,
  sha384: nodeSha384,
  aeadStream: nodeAeadStream,
};

/** A {@link ContainerSink} over a file, created or truncated. Positional writes, so `patch` is exact. */
export async function openFileSink(path: string): Promise<ContainerSink & { close(): Promise<void> }> {
  const handle = await open(path, 'w');
  let position = 0;
  const writeAll = async (bytes: Uint8Array, at: number): Promise<void> => {
    let done = 0;
    while (done < bytes.byteLength) {
      const { bytesWritten } = await handle.write(bytes, done, bytes.byteLength - done, at + done);
      done += bytesWritten;
    }
  };
  return {
    async write(bytes) {
      await writeAll(bytes, position);
      position += bytes.byteLength;
    },
    async patch(offset, bytes) {
      if (offset + bytes.byteLength > position)
        throw new RangeError('aamt-container: patch past the written end.');
      await writeAll(bytes, offset);
    },
    close: () => handle.close(),
  };
}

/**
 * A {@link ChunkSource} over bytes `[start, end)` of a file (default: all of it). Each call opens the file
 * afresh; one buffer of `chunkBytes` is reused across reads, which the streaming writer permits.
 */
export function fileSource(
  path: string,
  options: { readonly start?: number; readonly end?: number; readonly chunkBytes?: number } = {},
): ChunkSource {
  const { start = 0, end, chunkBytes = 1 << 20 } = options;
  return async function* () {
    const handle = await open(path, 'r');
    try {
      const stop = end ?? (await handle.stat()).size;
      const buf = new Uint8Array(chunkBytes);
      for (let at = start; at < stop;) {
        const { bytesRead } = await handle.read(buf, 0, Math.min(chunkBytes, stop - at), at);
        if (bytesRead === 0) throw new Error(`aamt-container: ${path} ended at ${at}, before ${stop}.`);
        at += bytesRead;
        yield buf.subarray(0, bytesRead);
      }
    } finally {
      await handle.close();
    }
  };
}
