// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * Identity: 384-bit, layer-separated (Yare design §7.1, ADR-005).
 *
 * ```
 *   id = SHA-384( context ‖ 0x00 ‖ bytes )
 * ```
 *
 * The same bytes get a different id in every layer, so a world page can never
 * stand in for a code record. The context strings contain no `0x00`, which makes
 * `context ‖ 0x00 ‖ bytes` prefix-free.
 *
 * ## The hash is a port
 *
 * This module never imports a hash implementation, for the same reason the
 * Vortex VM never imports `Math.sin`: the core has to run unchanged in the
 * Kernel, the renderer, a worker and under vitest. The caller supplies a
 * {@link Sha384}; `@aamt/container/node` provides one over `node:crypto`.
 * SHA-384 is native in every runtime the platform ships (Node, WebCrypto, Python
 * `hashlib`, Unity `System.Security.Cryptography`, C++), so every surface
 * computes the same id without a dependency.
 *
 * The port takes a list of parts rather than one buffer so a multi-gigabyte
 * arena is hashed incrementally instead of being concatenated first.
 *
 * Parts still have to be in memory at once. The streaming writer (ADR-017) hashes
 * bytes that never are, so it takes {@link Sha384Incremental}: a hash that is fed
 * chunk by chunk and finished once.
 */

import { ID_BYTES, LAYER_KINDS, type LayerKind } from './layout';

/** SHA-384 over the concatenation of `parts`, returned as 48 bytes. */
export type Sha384 = (parts: readonly Uint8Array[]) => Uint8Array;

/**
 * One SHA-384 in progress. `update` must have consumed `bytes` (hashed or copied
 * them) by the time it returns: the caller may reuse the buffer. `digest` is
 * called once and returns 48 bytes.
 */
export interface Sha384State {
  update(bytes: Uint8Array): void;
  digest(): Uint8Array;
}

/** Start a fresh incremental SHA-384. */
export type Sha384Incremental = () => Sha384State;

/** The one-shot {@link Sha384} port over an incremental one. */
export const sha384Of =
  (start: Sha384Incremental): Sha384 =>
  (parts) => {
    const h = start();
    for (const p of parts) h.update(p);
    return h.digest();
  };

const PREFIX = 'MaiiaM Yare v1 · ';

/** The identity context for each layer, plus one for the container as a whole. */
export const CONTEXT: Readonly<Record<LayerKind | 'container', string>> = Object.freeze({
  container: `${PREFIX}container`,
  ...(Object.fromEntries(LAYER_KINDS.map((k) => [k, `${PREFIX}${k}`])) as Record<LayerKind, string>),
});

const encoder = new TextEncoder();
const CONTEXT_BYTES: ReadonlyMap<string, Uint8Array> = new Map(
  Object.values(CONTEXT).map((c) => {
    const bytes = encoder.encode(c);
    if (bytes.includes(0)) throw new Error(`aamt-container: identity context ${c} contains NUL.`);
    return [c, Uint8Array.of(...bytes, 0)];
  }),
);

function contextPrefix(context: string): Uint8Array {
  const prefix = CONTEXT_BYTES.get(context);
  if (!prefix) throw new Error(`aamt-container: unknown identity context ${context}.`);
  return prefix;
}

function checkedId(id: Uint8Array): Uint8Array {
  if (id.byteLength !== ID_BYTES)
    throw new Error(`aamt-container: Sha384 port returned ${id.byteLength} bytes, expected ${ID_BYTES}.`);
  return id;
}

/** `SHA-384(context ‖ 0x00 ‖ parts…)`. */
export function identify(sha384: Sha384, context: string, parts: readonly Uint8Array[]): Uint8Array {
  return checkedId(sha384([contextPrefix(context), ...parts]));
}

/** {@link identify}, fed incrementally: the state already holds `context ‖ 0x00`. */
export function identifying(start: Sha384Incremental, context: string): Sha384State {
  const prefix = contextPrefix(context);
  const h = start();
  h.update(prefix);
  return { update: (bytes) => h.update(bytes), digest: () => checkedId(h.digest()) };
}

/** Lower-case hex of an id, for logs and journals. */
export function idHex(id: Uint8Array): string {
  let s = '';
  for (const b of id) s += b.toString(16).padStart(2, '0');
  return s;
}

/** Byte-wise comparison: the index order for id-sorted layers. */
export function compareIds(a: Uint8Array, b: Uint8Array): number {
  const n = Math.min(a.byteLength, b.byteLength);
  for (let i = 0; i < n; i++) {
    const d = a[i]! - b[i]!;
    if (d !== 0) return d;
  }
  return a.byteLength - b.byteLength;
}
