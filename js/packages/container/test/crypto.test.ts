// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
import { describe, expect, it } from 'vitest';
import {
  LAYER_FLAG_KEYED,
  RECORD_KIND_OPAQUE,
  deriveKey,
  idHex,
  identify,
  CONTEXT,
  isKeyed,
  openContainer,
  packContainer,
  unsealRecord,
  verifyContainer,
  type ContainerInput,
} from '../src/index';
import { nodeAead, nodeCrypto, nodeKdf, nodeSha384 } from '../src/node';

/** Keyed layers: standard AEAD under a key tree; ciphertext auditable, plaintext not confirmable. */

const enc = new TextEncoder();
const dec = new TextDecoder();
const root = new Uint8Array(32).map((_, i) => (i * 37 + 11) & 0xff);
const mindKey = deriveKey(nodeKdf, root, ['mind', 'sandbox-1']);
const otherKey = deriveKey(nodeKdf, root, ['mind', 'sandbox-2']);

const input = (flags = LAYER_FLAG_KEYED): ContainerInput => ({
  layers: [
    {
      kind: 'mind',
      flags,
      records: [
        { payload: enc.encode('a private thought'), at: { x: 1, y: 2, z: 3 } },
        { payload: enc.encode('another, nearby'), at: { x: 1, y: 2, z: 4 } },
      ],
    },
    { kind: 'world', records: [{ payload: enc.encode('public terrain'), at: { x: 0, y: 0, z: 0 } }] },
  ],
});
const sealed = () => packContainer(input(), nodeSha384, { ports: nodeCrypto, keys: { mind: mindKey } });

describe('the key tree (HKDF-SHA-384)', () => {
  it('derives distinct, deterministic 32-byte keys per path', () => {
    expect(mindKey.byteLength).toBe(32);
    expect(idHex(mindKey)).toBe(idHex(deriveKey(nodeKdf, root, ['mind', 'sandbox-1'])));
    expect(idHex(mindKey)).not.toBe(idHex(otherKey));
    expect(idHex(deriveKey(nodeKdf, root, ['mind']))).not.toBe(idHex(mindKey));
  });

  it('agrees with WebCrypto HKDF-SHA-384', async () => {
    for (let i = 0; i < 20; i++) {
      const ikm = new Uint8Array(32).map((_, j) => (i * 31 + j * 7) & 0xff);
      const salt = new Uint8Array(i).map((_, j) => j);
      const info = enc.encode(`label-${i}`);
      const base = await crypto.subtle.importKey('raw', ikm, 'HKDF', false, ['deriveBits']);
      const web = new Uint8Array(
        await crypto.subtle.deriveBits({ name: 'HKDF', hash: 'SHA-384', salt, info }, base, 256),
      );
      expect(idHex(nodeKdf(ikm, salt, info, 32))).toBe(idHex(web));
    }
  });

  it('refuses a short root and a malformed path', () => {
    expect(() => deriveKey(nodeKdf, new Uint8Array(16), ['mind'])).toThrow(/root key/);
    expect(() => deriveKey(nodeKdf, root, ['mind/escape'])).toThrow(/without "\/"/);
    expect(() => deriveKey(nodeKdf, root, [])).toThrow(/one or more/);
  });
});

describe('a keyed layer', () => {
  it('opens with its key, and its plaintext never appears in the file', () => {
    const bytes = sealed();
    expect(dec.decode(bytes)).not.toContain('private thought');
    expect(dec.decode(bytes)).toContain('public terrain'); // unkeyed layers are untouched
    const mind = openContainer(bytes).layer('mind')!;
    expect(isKeyed(mind)).toBe(true);
    const texts = [...mind.records()]
      .map((r) => dec.decode(unsealRecord(mind, r, mindKey, nodeCrypto)!))
      .sort();
    expect(texts).toEqual(['a private thought', 'another, nearby']);
  });

  it('can be audited without the key: identities cover the sealed bytes', () => {
    expect(verifyContainer(sealed(), nodeSha384)).toEqual({ ok: true, problems: [] });
  });

  it('does not let a stored id confirm a guessed plaintext', () => {
    const mind = openContainer(sealed()).layer('mind')!;
    const r = mind.at({ x: 1, y: 2, z: 3 })[0]!;
    const locatorOfGuess = r.locatorBytes;
    const plainId = identify(nodeSha384, CONTEXT.mind, [
      new Uint8Array([0, 0, 0, 0]),
      locatorOfGuess,
      enc.encode('a private thought'),
    ]);
    expect(idHex(r.id)).not.toBe(idHex(plainId));
  });

  it("is erased by losing its key: another sandbox's key opens nothing", () => {
    const mind = openContainer(sealed()).layer('mind')!;
    for (const r of mind.records()) expect(unsealRecord(mind, r, otherKey, nodeCrypto)).toBeNull();
  });

  it('binds each ciphertext to its place: moved onto another record, it fails to open', () => {
    const mind = openContainer(sealed()).layer('mind')!;
    const [a, b] = [...mind.records()];
    // b's sealed bytes presented as a's: the locator in the associated data no longer matches.
    expect(unsealRecord(mind, { ...a!, payload: b!.payload }, mindKey, nodeCrypto)).toBeNull();
    const flipped = b!.payload.slice();
    flipped[flipped.byteLength - 1]! ^= 1;
    expect(unsealRecord(mind, { ...b!, payload: flipped }, mindKey, nodeCrypto)).toBeNull();
  });

  it('seals deterministically: same key and content, same bytes; another key, other bytes', () => {
    expect(idHex(nodeSha384([sealed()]))).toBe(idHex(nodeSha384([sealed()])));
    const other = packContainer(input(), nodeSha384, { ports: nodeCrypto, keys: { mind: otherKey } });
    expect(idHex(nodeSha384([other]))).not.toBe(idHex(nodeSha384([sealed()])));
  });

  it('refuses to write a KEYED layer without its key', () => {
    expect(() => packContainer(input(), nodeSha384)).toThrow(/KEYED but no key/);
  });

  it('uses distinct nonces for distinct plaintexts', () => {
    const many: ContainerInput = {
      layers: [
        {
          kind: 'experience',
          flags: LAYER_FLAG_KEYED,
          records: Array.from({ length: 500 }, (_, i) => ({
            kind: RECORD_KIND_OPAQUE,
            payload: enc.encode(`memory ${i}`),
          })),
        },
      ],
    };
    const exp = openContainer(
      packContainer(many, nodeSha384, { ports: nodeCrypto, keys: { experience: mindKey } }),
    ).layer('experience')!;
    const nonces = new Set([...exp.records()].map((r) => idHex(r.payload.subarray(0, 12))));
    expect(nonces.size).toBe(500);
    expect(nodeAead.open(mindKey, new Uint8Array(12), new Uint8Array(8), new Uint8Array(0))).toBeNull();
  });
});
