// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * Keyed layers: payload encryption under a key tree (ADR-005 §"Not in v1.0",
 * Yare design §7.1). Standard primitives only — no custom cipher.
 *
 * ## Primitives, as ports
 *
 * - {@link Aead}: AES-256-GCM (96-bit nonce, 128-bit tag).
 * - {@link Kdf}: HKDF-SHA-384.
 *
 * Both are native in Node (`node:crypto`) and WebCrypto, and enter as ports like
 * {@link Sha384}, so this module imports nothing at runtime. `./node` binds them.
 *
 * ## The key tree
 *
 * ```
 *   root ──HKDF──▶ layer key  ("MaiiaM Yare v1 · key · <layer>")
 *        ──HKDF──▶ scope key  ("MaiiaM Yare v1 · key · <layer>/<scope>")
 * ```
 *
 * {@link deriveKey} walks a path of labels. A mind sandbox seals its layer with its
 * own key, so destroying that one key erases the sandbox's thoughts — the bytes
 * remain, the meaning does not.
 *
 * ## Deterministic sealing without nonce reuse
 *
 * Containers pack to the same bytes every time, so a random nonce is out. The
 * nonce is **synthetic**: `HKDF(key, salt = SHA-384(plaintext), info = "nonce")`,
 * 96 bits. Distinct plaintexts get distinct nonces (barring a 2⁻⁹⁶-scale
 * collision); identical plaintexts under one key give identical ciphertext, which
 * reveals only equality — something content addressing already reveals.
 *
 * ## What is bound, and what is not hidden
 *
 * The ciphertext is bound by associated data to the layer kind, the record kind
 * and the record's locator, so a ciphertext moved onto another record fails to
 * open. A keyed record's identity covers the **stored** (sealed) bytes, so anyone
 * can audit the file without the key, and an id cannot be used to confirm a
 * guessed plaintext. **Locators are not encrypted**: a world or mind cell and a
 * TERA search key stay readable, because routing must work without the key. Where
 * position itself is sensitive, store a coarse or zero locator.
 */

import { LAYER_ID, LOCATOR_BYTES, type LayerKind } from './layout';
import type { Sha384 } from './identity';

/** AES-256-GCM. `seal` returns ciphertext ‖ 16-byte tag; `open` returns null on any authentication failure. */
export interface Aead {
  seal(key: Uint8Array, nonce: Uint8Array, plaintext: Uint8Array, aad: Uint8Array): Uint8Array;
  open(key: Uint8Array, nonce: Uint8Array, sealed: Uint8Array, aad: Uint8Array): Uint8Array | null;
}

/** HKDF-SHA-384: `length` bytes of output keying material. */
export type Kdf = (ikm: Uint8Array, salt: Uint8Array, info: Uint8Array, length: number) => Uint8Array;

/**
 * AES-256-GCM sealing fed chunk by chunk, for the streaming writer (ADR-017).
 * The concatenation of every `update` result and the `final` result must equal
 * {@link Aead.seal} over the concatenated plaintext: ciphertext ‖ 16-byte tag.
 */
export interface AeadSealStream {
  update(chunk: Uint8Array): Uint8Array;
  /** Any remaining ciphertext, then the tag. */
  final(): Uint8Array;
}

/** Incremental AES-256-GCM. Optional: WebCrypto has no streaming GCM, so the streaming writer can do without it. */
export interface AeadStream {
  seal(key: Uint8Array, nonce: Uint8Array, aad: Uint8Array): AeadSealStream;
}

export interface CryptoPorts {
  readonly aead: Aead;
  readonly kdf: Kdf;
  readonly sha384: Sha384;
  /** Lets the streaming writer seal a record without holding it in memory. */
  readonly aeadStream?: AeadStream;
}

export const KEY_BYTES = 32;
export const NONCE_BYTES = 12;
export const TAG_BYTES = 16;

const enc = new TextEncoder();
const KEY_PREFIX = 'MaiiaM Yare v1 · key · ';

/** Derive a key from `root` along a path of labels (no label may contain `/`). */
export function deriveKey(kdf: Kdf, root: Uint8Array, path: readonly string[]): Uint8Array {
  if (root.byteLength < KEY_BYTES)
    throw new RangeError(`aamt-container: a root key needs at least ${KEY_BYTES} bytes.`);
  if (path.length === 0 || path.some((p) => p.length === 0 || p.includes('/')))
    throw new RangeError('aamt-container: a key path is one or more non-empty labels without "/".');
  const key = kdf(root, new Uint8Array(0), enc.encode(KEY_PREFIX + path.join('/')), KEY_BYTES);
  if (key.byteLength !== KEY_BYTES)
    throw new Error('aamt-container: the Kdf port returned the wrong length.');
  return key;
}

/** The associated data binding a sealed payload to where it lives. */
export function recordAad(layer: LayerKind, recordKind: number, locator: Uint8Array): Uint8Array {
  if (locator.byteLength !== LOCATOR_BYTES) throw new RangeError('aamt-container: a locator is 32 bytes.');
  const out = new Uint8Array(8 + LOCATOR_BYTES);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, LAYER_ID[layer], true);
  dv.setUint32(4, recordKind >>> 0, true);
  out.set(locator, 8);
  return out;
}

const NONCE_INFO = enc.encode(`${KEY_PREFIX}nonce`);

/**
 * The synthetic nonce for a plaintext whose SHA-384 is `plaintextDigest` (see the
 * module comment). Exposed so a streaming sealer can hash the plaintext in one
 * pass and encrypt it in the next.
 */
export function syntheticNonce(kdf: Kdf, key: Uint8Array, plaintextDigest: Uint8Array): Uint8Array {
  if (key.byteLength !== KEY_BYTES) throw new RangeError('aamt-container: a layer key is 32 bytes.');
  return kdf(key, plaintextDigest, NONCE_INFO, NONCE_BYTES);
}

/** Seal one payload: `nonce ‖ ciphertext ‖ tag`. Deterministic for a given key and plaintext. */
export function sealPayload(
  ports: CryptoPorts,
  key: Uint8Array,
  plaintext: Uint8Array,
  aad: Uint8Array,
): Uint8Array {
  const nonce = syntheticNonce(ports.kdf, key, ports.sha384([plaintext]));
  const sealed = ports.aead.seal(key, nonce, plaintext, aad);
  const out = new Uint8Array(NONCE_BYTES + sealed.byteLength);
  out.set(nonce, 0);
  out.set(sealed, NONCE_BYTES);
  return out;
}

/** Open one payload, or return null if the key is wrong or anything was altered or moved. */
export function openPayload(
  ports: Pick<CryptoPorts, 'aead'>,
  key: Uint8Array,
  stored: Uint8Array,
  aad: Uint8Array,
): Uint8Array | null {
  if (key.byteLength !== KEY_BYTES || stored.byteLength < NONCE_BYTES + TAG_BYTES) return null;
  return ports.aead.open(key, stored.subarray(0, NONCE_BYTES), stored.subarray(NONCE_BYTES), aad);
}
