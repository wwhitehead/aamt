// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
import { describe, expect, it } from 'vitest';
import { buildAmdlFixture, parseFooter, AMDL_MAGIC } from '@aamt/model/amdl-inspect';
import {
  CONTEXT,
  ContainerFormatError,
  HEADER_BYTES,
  H_FILE_ID,
  LAYER_FLAG_COPY_ON_WRITE,
  LAYER_FLAG_DISPOSABLE,
  LAYER_FLAG_KEYED,
  MAGIC,
  PAGE_BYTES,
  PAYLOAD_ALIGN,
  RECORD_KIND_AMDL_EMBEDDED,
  RECORD_FLAG_MIRROR,
  compareIds,
  idHex,
  identify,
  mortonKey,
  openContainer,
  packContainer,
  verifyContainer,
  type ContainerInput,
  type Sha384,
} from '../src/index';
import { nodeSha384 } from '../src/node';

const enc = new TextEncoder();
const bytes = (s: string): Uint8Array => enc.encode(s);

/** A deterministic LCG, so the randomized cases are reproducible. */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

const sample = (): ContainerInput => ({
  layers: [
    {
      kind: 'code',
      records: [
        { payload: bytes('fn drift'), tera: [0.62, 0.4, 0.81, 0.3], lens: 'technical', tier: 'declared' },
      ],
    },
    {
      kind: 'world',
      records: [
        { payload: bytes('brick a'), at: { x: 0, y: 0, z: 0 } },
        { payload: bytes('brick b'), at: { x: 1, y: 0, z: 0 } },
        { payload: bytes('brick c'), at: { x: -5, y: 7, z: 2 } },
        { payload: bytes('tree'), at: { x: 1, y: 0, z: 0 } },
      ],
    },
    { kind: 'mind', records: [{ payload: bytes('a new idea'), at: { x: 3, y: 3, z: 3 } }] },
    {
      kind: 'experience',
      records: [
        {
          payload: bytes('forward strand'),
          tera: [0.5, 0.5, 0.5, 0.5],
          lens: 'wisdom',
          tier: 'reconstructed',
        },
        { payload: bytes('mirror strand'), flags: RECORD_FLAG_MIRROR },
      ],
    },
    {
      kind: 'model',
      records: [
        {
          kind: RECORD_KIND_AMDL_EMBEDDED,
          payload: buildAmdlFixture([
            { name: 'model.embed_tokens.weight', shape: [8, 4], kind: 'int4', bytes: 64 },
          ]),
        },
      ],
    },
  ],
});

describe('Sha384 port', () => {
  it('matches the FIPS 180-2 "abc" vector', () => {
    expect(idHex(nodeSha384([bytes('abc')]))).toBe(
      'cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7',
    );
  });

  it('agrees with WebCrypto, and hashing parts equals hashing their concatenation', async () => {
    const rnd = lcg(384);
    for (let t = 0; t < 50; t++) {
      const parts = Array.from({ length: 1 + Math.floor(rnd() * 4) }, () =>
        Uint8Array.from({ length: Math.floor(rnd() * 300) }, () => Math.floor(rnd() * 256)),
      );
      const whole = new Uint8Array(parts.reduce((n, p) => n + p.byteLength, 0));
      parts.reduce((at, p) => (whole.set(p, at), at + p.byteLength), 0);
      const web = new Uint8Array(await globalThis.crypto.subtle.digest('SHA-384', whole));
      expect(compareIds(nodeSha384(parts), web)).toBe(0);
      expect(compareIds(nodeSha384([whole]), web)).toBe(0);
    }
  });

  it('separates layers: the same bytes get a different id in every context', () => {
    const ids = Object.values(CONTEXT).map((c) => idHex(identify(nodeSha384, c, [bytes('same bytes')])));
    expect(new Set(ids).size).toBe(Object.values(CONTEXT).length);
  });

  it('rejects a port that returns the wrong length', () => {
    const short: Sha384 = () => new Uint8Array(32);
    expect(() => identify(short, CONTEXT.code, [bytes('x')])).toThrow(/expected 48/);
  });
});

describe('pack → open round trip', () => {
  const file = packContainer(sample(), nodeSha384);
  const c = openContainer(file);

  it('writes the magic, version and length', () => {
    expect([...file.subarray(0, 8)]).toEqual([...MAGIC]);
    expect(c.versionMajor).toBe(1);
    expect(c.fileBytes).toBe(file.byteLength);
  });

  it('lists layers in canonical order regardless of input order', () => {
    expect(c.layers.map((l) => l.kind)).toEqual(['model', 'world', 'mind', 'experience', 'code']);
  });

  it('returns every payload byte-exact, as a zero-copy view', () => {
    const want = sample();
    for (const layer of want.layers) {
      const got = [...c.layer(layer.kind)!.records()].map((r) => idHex(r.payload));
      expect(got.sort()).toEqual(layer.records.map((r) => idHex(r.payload)).sort());
    }
    const r = c.layer('code')!.record(0);
    expect(r.payload.buffer).toBe(file.buffer);
  });

  it('keeps the default layer flags (mind is copy-on-write and disposable)', () => {
    expect(c.layer('mind')!.flags).toBe(LAYER_FLAG_COPY_ON_WRITE | LAYER_FLAG_DISPOSABLE);
  });

  it('round-trips manifold locators, including lens and the WP-34 tier', () => {
    const code = c.layer('code')!.record(0).locator;
    expect(code).toMatchObject({ class: 'manifold', lens: 'technical', tier: 'declared' });
    if (code?.class !== 'manifold' || !code.tera) throw new Error('expected a tera point');
    code.tera.forEach((v, i) => expect(v).toBeCloseTo([0.62, 0.4, 0.81, 0.3][i]!, 6));
    const mirror = [...c.layer('experience')!.records()].find((r) => r.flags === RECORD_FLAG_MIRROR)!;
    expect(mirror.locator).toEqual({ class: 'manifold', tera: null, lens: 'none', tier: 'none' });
  });

  it('finds records by id in every layer', () => {
    for (const layer of c.layers)
      for (const r of layer.records()) expect(layer.get(r.id)?.index).toBe(r.index);
    expect(c.layer('code')!.get(new Uint8Array(48))).toBeUndefined();
  });

  it('finds every record at a world cell, and none at an empty one', () => {
    const world = c.layer('world')!;
    expect(
      world
        .at({ x: 1, y: 0, z: 0 })
        .map((r) => new TextDecoder().decode(r.payload))
        .sort(),
    ).toEqual(['brick b', 'tree']);
    expect(world.at({ x: -5, y: 7, z: 2 })).toHaveLength(1);
    expect(world.at({ x: 9, y: 9, z: 9 })).toHaveLength(0);
    expect(() => c.layer('code')!.at({ x: 0, y: 0, z: 0 })).toThrow(/not spatial/);
  });

  it('embeds an AMDL model so the existing footer parser reads the slice unchanged', () => {
    const r = c.layer('model')!.record(0);
    expect(r.kind).toBe(RECORD_KIND_AMDL_EMBEDDED);
    expect(parseFooter(r.payload).magic).toBe(AMDL_MAGIC);
  });

  it('verifies clean', () => {
    expect(verifyContainer(file, nodeSha384)).toEqual({ ok: true, problems: [] });
  });
});

describe('alignment', () => {
  const file = packContainer(sample(), nodeSha384);
  const c = openContainer(file);

  it('page-aligns every arena and every model payload; 16-byte-aligns the rest', () => {
    for (const layer of c.layers) {
      for (const r of layer.records()) {
        const at = r.payload.byteOffset - file.byteOffset;
        expect(at % (layer.kind === 'model' ? PAGE_BYTES : PAYLOAD_ALIGN)).toBe(0);
      }
    }
  });
});

describe('determinism', () => {
  it('packs the same content to the same bytes whatever the input order', () => {
    const a = packContainer(sample(), nodeSha384);
    const shuffled = sample();
    const b = packContainer(
      {
        layers: [...shuffled.layers]
          .reverse()
          .map((l) => ({ ...l, records: [...l.records].reverse() }) as (typeof shuffled.layers)[number]),
      },
      nodeSha384,
    );
    expect(idHex(b)).toBe(idHex(a));
  });

  it('gives identical code in two containers the same id (identity is content)', () => {
    const one = openContainer(
      packContainer({ layers: [{ kind: 'code', records: [{ payload: bytes('fn x') }] }] }, nodeSha384),
    );
    const two = openContainer(packContainer(sample(), nodeSha384));
    const tagged = openContainer(
      packContainer(
        {
          layers: [
            {
              kind: 'code',
              records: [{ payload: bytes('fn x'), tera: [0.1, 0.2, 0.3, 0.4], tier: 'declared' }],
            },
          ],
        },
        nodeSha384,
      ),
    );
    expect(two.layer('code')!.get(one.layer('code')!.record(0).id)).toBeUndefined();
    // Declaring what a function means does not change which function it is (design §3).
    expect(compareIds(tagged.layer('code')!.record(0).id, one.layer('code')!.record(0).id)).toBe(0);
  });

  it('includes the cell in a world id: the same brick at two cells is two records', () => {
    const c = openContainer(
      packContainer(
        {
          layers: [
            {
              kind: 'world',
              records: [
                { payload: bytes('brick'), at: { x: 0, y: 0, z: 0 } },
                { payload: bytes('brick'), at: { x: 0, y: 0, z: 1 } },
              ],
            },
          ],
        },
        nodeSha384,
      ),
    );
    const [a, b] = [...c.layer('world')!.records()];
    expect(compareIds(a!.id, b!.id)).not.toBe(0);
  });
});

describe('Morton order (spatial layers)', () => {
  it('matches a bit-by-bit interleave over random coordinates', () => {
    const rnd = lcg(21);
    const naive = (x: number, y: number, z: number): bigint => {
      let k = 0n;
      const [ux, uy, uz] = [x, y, z].map((v) => BigInt(v + 2 ** 20));
      for (let b = 0n; b < 21n; b++)
        k |=
          (((ux! >> b) & 1n) << (3n * b)) |
          (((uy! >> b) & 1n) << (3n * b + 1n)) |
          (((uz! >> b) & 1n) << (3n * b + 2n));
      return k;
    };
    for (let i = 0; i < 2000; i++) {
      const [x, y, z] = [0, 0, 0].map(() => Math.floor(rnd() * 2 ** 21) - 2 ** 20) as [
        number,
        number,
        number,
      ];
      expect(mortonKey({ x, y, z })).toBe(naive(x, y, z));
    }
  });

  it('stores world records in Morton order, so neighbours are adjacent entries', () => {
    const rnd = lcg(7);
    const records = Array.from({ length: 200 }, (_, i) => ({
      payload: bytes(`cell ${i}`),
      at: { x: Math.floor(rnd() * 64) - 32, y: Math.floor(rnd() * 64) - 32, z: Math.floor(rnd() * 64) - 32 },
    }));
    const c = openContainer(packContainer({ layers: [{ kind: 'world', records }] }, nodeSha384));
    const keys = [...c.layer('world')!.records()].map((r) =>
      r.locator?.class === 'spatial' ? r.locator.morton : -1n,
    );
    for (let i = 1; i < keys.length; i++) expect(keys[i]! >= keys[i - 1]!).toBe(true);
  });

  it('rejects a non-integer or out-of-range coordinate', () => {
    expect(() =>
      packContainer(
        { layers: [{ kind: 'world', records: [{ payload: bytes('x'), at: { x: 0.5, y: 0, z: 0 } }] }] },
        nodeSha384,
      ),
    ).toThrow(/integer/);
    expect(() =>
      packContainer(
        { layers: [{ kind: 'world', records: [{ payload: bytes('x'), at: { x: 2 ** 20, y: 0, z: 0 } }] }] },
        nodeSha384,
      ),
    ).toThrow(/integer/);
  });
});

describe('verification catches tampering', () => {
  it('names the record whose payload changed, and the layer and file ids', () => {
    const file = packContainer(sample(), nodeSha384);
    const c = openContainer(file);
    const r = c.layer('code')!.record(0);
    const at = r.payload.byteOffset - file.byteOffset;
    const bad = file.slice();
    bad[at] = bad[at]! ^ 0x01;
    const v = verifyContainer(bad, nodeSha384);
    expect(v.ok).toBe(false);
    expect(v.problems.join('\n')).toMatch(/code record 0: id does not match/);
    expect(v.problems.join('\n')).toMatch(/code: layerId mismatch/);
    expect(v.problems.join('\n')).toMatch(/fileId mismatch/);
  });

  it('catches a change to padding, which no record covers', () => {
    const file = packContainer(sample(), nodeSha384);
    const bad = file.slice();
    bad[HEADER_BYTES - 1] = 0xff; // reserved header byte
    const v = verifyContainer(bad, nodeSha384);
    expect(v.problems).toEqual([expect.stringMatching(/fileId mismatch/)]);
  });

  it('catches a forged fileId', () => {
    const bad = packContainer(sample(), nodeSha384).slice();
    bad[H_FILE_ID] = bad[H_FILE_ID]! ^ 0xff;
    expect(verifyContainer(bad, nodeSha384).problems).toEqual([expect.stringMatching(/fileId mismatch/)]);
  });
});

describe('structural rejection', () => {
  it('rejects bad magic, truncation and unsupported versions on open', () => {
    const file = packContainer(sample(), nodeSha384);
    expect(() => openContainer(new Uint8Array(64))).toThrow(ContainerFormatError);
    const wrongMagic = file.slice();
    wrongMagic[0] = 0x41;
    expect(() => openContainer(wrongMagic)).toThrow(/bad magic/);
    expect(() => openContainer(file.subarray(0, file.byteLength - 1))).toThrow(/truncated/);
    const v2 = file.slice();
    new DataView(v2.buffer).setUint16(8, 2, true);
    expect(() => openContainer(v2)).toThrow(/major version 2/);
  });

  it('refuses duplicate layers, duplicate records, keyed layers and a tera point without a tier', () => {
    expect(() =>
      packContainer(
        {
          layers: [
            { kind: 'code', records: [] },
            { kind: 'code', records: [] },
          ],
        },
        nodeSha384,
      ),
    ).toThrow(/appears twice/);
    expect(() =>
      packContainer(
        { layers: [{ kind: 'code', records: [{ payload: bytes('a') }, { payload: bytes('a') }] }] },
        nodeSha384,
      ),
    ).toThrow(/duplicate/);
    expect(() =>
      packContainer({ layers: [{ kind: 'code', flags: LAYER_FLAG_KEYED, records: [] }] }, nodeSha384),
    ).toThrow(/KEYED/);
    expect(() =>
      packContainer(
        { layers: [{ kind: 'code', records: [{ payload: bytes('a'), tera: [0.1, 0.2, 0.3, 0.4] }] }] },
        nodeSha384,
      ),
    ).toThrow(/tier/);
    expect(() =>
      packContainer(
        {
          layers: [
            {
              kind: 'code',
              records: [{ payload: bytes('a'), tera: [0.1, 0.2, 1.5, 0.4], tier: 'declared' }],
            },
          ],
        },
        nodeSha384,
      ),
    ).toThrow(/\[0,1\]/);
  });

  it('packs an empty container that opens and verifies', () => {
    const file = packContainer({ layers: [] }, nodeSha384);
    expect(openContainer(file).layers).toHaveLength(0);
    expect(verifyContainer(file, nodeSha384).ok).toBe(true);
  });
});

describe('v1.1: arena alignment per layer kind', () => {
  it('pads a small code layer to cache lines, not to a 16 KiB page', () => {
    const file = packContainer(
      { layers: [{ kind: 'code', records: [{ payload: bytes('fn tiny') }] }] },
      nodeSha384,
    );
    expect(file.byteLength).toBeLessThan(1024);
    expect(openContainer(file).versionMinor).toBe(2);
    expect(verifyContainer(file, nodeSha384).ok).toBe(true);
  });

  it('still page-aligns the arenas a GPU may wrap without copying', () => {
    const file = packContainer(sample(), nodeSha384);
    const dv = new DataView(file.buffer, file.byteOffset);
    const c = openContainer(file);
    c.layers.forEach((layer, li) => {
      const arena = Number(dv.getBigUint64(HEADER_BYTES + li * 128 + 24, true));
      expect(arena % (['model', 'world', 'mind'].includes(layer.kind) ? PAGE_BYTES : 64), layer.kind).toBe(0);
    });
  });

  it('opens a v1.0 file: page alignment satisfies the v1.1 rule', () => {
    const file = packContainer(sample(), nodeSha384).slice();
    new DataView(file.buffer).setUint16(10, 0, true); // relabel as v1.0
    expect(openContainer(file).versionMinor).toBe(0);
  });
});
