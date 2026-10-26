// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  REFERENCE_TRANSCENDENTALS,
  frDist,
  makeReg,
  mejiIndex,
  oduIndex,
  type TeraReg,
} from '@aamt/tera';
import {
  CONTEXT,
  ContainerFormatError,
  HEADER_BYTES,
  H_FILE_ID,
  H_VERSION_MINOR,
  ID_BYTES,
  LAYER_ENTRY_BYTES,
  L_ODU_BYTES,
  L_ODU_OFFSET,
  ODU_HEADER_BYTES,
  buildOduSection,
  identify,
  libaamtCellToOdu,
  libaamtMejiToVm,
  oduCell,
  oduCellsWithin,
  openContainer,
  packContainer,
  verifyContainer,
  type Container,
  type ContainerInput,
  type LayerKind,
  type ManifoldRecordInput,
} from '../src/index';
import { nodeSha384 } from '../src/node';

type V4 = [number, number, number, number];
const enc = new TextEncoder();

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

const f32 = (x: number): number => Math.fround(x);
const QUARTILES = [0, 0.25, 0.5, 0.75, 1];

/** A coordinate biased toward the places the cell function branches: quartile edges and their f32 neighbours. */
function lane(rnd: () => number): number {
  const r = rnd();
  if (r < 0.15) return QUARTILES[Math.floor(rnd() * 5)]!;
  if (r < 0.3) {
    const q = QUARTILES[1 + Math.floor(rnd() * 3)]!;
    return Math.min(1, Math.max(0, q + (rnd() < 0.5 ? -1 : 1) * 2 ** -24 * Math.ceil(rnd() * 4)));
  }
  return rnd();
}
const point = (rnd: () => number): V4 => [lane(rnd), lane(rnd), lane(rnd), lane(rnd)];

/** The stored point: what a reader decodes and what `near` measures. */
const stored = (v: V4): V4 => v.map(f32) as V4;

function manifoldLayer(kind: 'experience' | 'code', n: number, seed: number, untagged = 0): ContainerInput {
  const rnd = lcg(seed);
  const records: ManifoldRecordInput[] = Array.from({ length: n }, (_, i) => ({
    payload: enc.encode(`${kind} ${seed} ${i}`),
    tera: point(rnd),
    lens: rnd() < 0.5 ? 'technical' : 'wisdom',
    tier: 'declared',
  }));
  for (let i = 0; i < untagged; i++) records.push({ payload: enc.encode(`untagged ${i}`) });
  return { layers: [{ kind, records }] };
}

/** Each record's register (or `null` without a point), decoded once per opened container. */
const regCache = new WeakMap<Container, Map<LayerKind, (TeraReg | null)[]>>();
function regsOf(c: Container, kind: LayerKind): (TeraReg | null)[] {
  let byKind = regCache.get(c);
  if (!byKind) regCache.set(c, (byKind = new Map()));
  let regs = byKind.get(kind);
  if (!regs) {
    regs = [...c.layer(kind)!.records()].map((r) =>
      r.locator?.class === 'manifold' && r.locator.tera ? makeReg(r.locator.tera) : null,
    );
    byKind.set(kind, regs);
  }
  return regs;
}

/** Positions of records within `radius` of `v`, by scanning every record. */
function scanWithin(c: Container, kind: LayerKind, v: V4, radius: number): number[] {
  const q = makeReg(v);
  const out: number[] = [];
  regsOf(c, kind).forEach((reg, i) => {
    if (reg && !(frDist(q, reg) > radius)) out.push(i);
  });
  return out;
}

/** The same, reading only the Odu index's candidates. */
function indexWithin(c: Container, kind: LayerKind, v: V4, radius: number): number[] {
  const q = makeReg(v);
  const regs = regsOf(c, kind);
  return Array.from(c.layer(kind)!.odu!.candidates(v, radius)).filter((i) => {
    const reg = regs[i];
    return reg !== null && reg !== undefined && !(frDist(q, reg) > radius);
  });
}

/** Byte offset of a layer's entry in the layer table. */
const entryOf = (c: Container, kind: LayerKind): number =>
  HEADER_BYTES + c.layers.findIndex((l) => l.kind === kind) * LAYER_ENTRY_BYTES;

function sectionOf(file: Uint8Array, kind: LayerKind): { at: number; bytes: number; entry: number } {
  const c = openContainer(file);
  const entry = entryOf(c, kind);
  const dv = new DataView(file.buffer, file.byteOffset);
  return {
    at: Number(dv.getBigUint64(entry + L_ODU_OFFSET, true)),
    bytes: Number(dv.getBigUint64(entry + L_ODU_BYTES, true)),
    entry,
  };
}

/** Recompute the fileId, so a deliberately edited file differs from a genuine one only where we edited it. */
function reseal(file: Uint8Array): Uint8Array {
  const out = file.slice();
  out.fill(0, H_FILE_ID, H_FILE_ID + ID_BYTES);
  out.set(identify(nodeSha384, CONTEXT.container, [out]), H_FILE_ID);
  return out;
}

describe('the cell function is the VM’s oduIndex over the stored f32 point', () => {
  const file = packContainer(manifoldLayer('experience', 3000, 1, 50), nodeSha384);
  const c = openContainer(file);
  const layer = c.layer('experience')!;

  it('routes every located record, once, to oduIndex(makeReg(stored tera))', () => {
    expect(layer.odu).not.toBeNull();
    expect(layer.odu!.entryCount).toBe(3000);
    const seen = new Set<number>();
    for (let cell = 0; cell < 256; cell++) {
      for (const p of layer.odu!.cell(cell)) {
        const loc = layer.record(p).locator;
        if (loc?.class !== 'manifold' || !loc.tera) throw new Error('indexed a record without a point');
        expect(oduIndex(makeReg(loc.tera))).toBe(cell);
        expect(seen.has(p)).toBe(false);
        seen.add(p);
      }
    }
    expect(seen.size).toBe(3000);
    expect(verifyContainer(file, nodeSha384)).toEqual({ ok: true, problems: [] });
  });

  it('uses the f32 the reader decodes, not the f64 the writer was given', () => {
    const justBelow = 0.25 - 2 ** -30; // rounds to exactly 0.25 in f32: quartile 1, not 0
    expect(f32(justBelow)).toBe(0.25);
    const one = openContainer(
      packContainer(
        {
          layers: [
            {
              kind: 'code',
              records: [{ payload: enc.encode('edge'), tera: [justBelow, 0, 0, 0], tier: 'declared' }],
            },
          ],
        },
        nodeSha384,
      ),
    );
    expect(oduCell([justBelow, 0, 0, 0])).toBe(0);
    expect(one.layer('code')!.odu!.cell(1 << 6)).toEqual([0]);
  });

  it('writes no section for a layer with no TERA points, or when asked not to', () => {
    const none = openContainer(
      packContainer({ layers: [{ kind: 'code', records: [{ payload: enc.encode('x') }] }] }, nodeSha384),
    );
    expect(none.layer('code')!.odu).toBeNull();
    const input = manifoldLayer('code', 10, 2);
    const off = openContainer(
      packContainer({ layers: [{ ...input.layers[0]!, oduIndex: false } as never] }, nodeSha384),
    );
    expect(off.layer('code')!.odu).toBeNull();
    expect(openContainer(packContainer(input, nodeSha384)).layer('code')!.odu).not.toBeNull();
  });

  it('keeps the layer id: the index is a side table, not layer content', () => {
    const input = manifoldLayer('code', 50, 3);
    const a = openContainer(packContainer(input, nodeSha384)).layer('code')!;
    const b = openContainer(
      packContainer({ layers: [{ ...input.layers[0]!, oduIndex: false } as never] }, nodeSha384),
    ).layer('code')!;
    expect([...a.id]).toEqual([...b.id]);
  });
});

describe('oduCellsWithin: which cells a Fisher–Rao ball can touch', () => {
  it('relies on toTheta being non-decreasing across every quartile edge (dense check)', () => {
    const tr = REFERENCE_TRANSCENDENTALS;
    let violations = 0;
    for (const q of [0.25, 0.5, 0.75]) {
      const edge = tr.toTheta(q);
      let prev = -Infinity;
      for (let k = -20000; k <= 20000; k++) {
        const x = q + k * 2 ** -40;
        const t = tr.toTheta(x);
        if (t < prev || (x < q ? t > edge : t < edge)) violations++;
        prev = t;
      }
    }
    const rnd = lcg(5);
    const xs = Array.from({ length: 200000 }, () => rnd()).sort((a, b) => a - b);
    for (let i = 1; i < xs.length; i++) if (tr.toTheta(xs[i]!) < tr.toTheta(xs[i - 1]!)) violations++;
    expect(violations).toBe(0);
  });

  it('never omits the cell of a point within the radius (200,000 random pairs, edge-biased)', () => {
    const rnd = lcg(11);
    const radii = [0, 1e-6, 0.03, 0.1, 0.3, 0.9, Math.PI];
    for (let i = 0; i < 4000; i++) {
      const v = point(rnd);
      const r = rnd() < 0.5 ? radii[Math.floor(rnd() * radii.length)]! : rnd() * 1.2;
      const cells = new Set(oduCellsWithin(v, r));
      const q = makeReg(v);
      for (let k = 0; k < 50; k++) {
        // Half the samples near the query, so small radii are exercised too.
        const u = stored(
          rnd() < 0.5 ? (v.map((x) => Math.min(1, Math.max(0, x + (rnd() - 0.5) * 0.2))) as V4) : point(rnd),
        );
        if (!(frDist(q, makeReg(u)) > r)) expect(cells.has(oduCell(u))).toBe(true);
      }
    }
  });

  it('is tight enough to matter: a small ball inside a cell touches only that cell', () => {
    const v: V4 = [0.6, 0.6, 0.6, 0.6]; // interior of quartile 2 on every axis
    expect(oduCellsWithin(v, 0.01)).toEqual([oduCell(v)]);
    expect(oduCellsWithin(v, Math.PI * 2)).toHaveLength(256);
    const mean = (r: number): number => {
      const rnd = lcg(17);
      let n = 0;
      for (let i = 0; i < 500; i++) n += oduCellsWithin([rnd(), rnd(), rnd(), rnd()], r).length;
      return n / 500;
    };
    expect(mean(0.1)).toBeLessThan(8);
    expect(mean(0.3)).toBeLessThan(40);
  });
});

describe('index ≡ scan: no candidate lost, for many random queries and radii', () => {
  for (const kind of ['experience', 'code'] as const) {
    it(`${kind}: 3,000 records, 1,500 queries`, () => {
      const c = openContainer(packContainer(manifoldLayer(kind, 3000, 21, 30), nodeSha384));
      const rnd = lcg(23);
      let hits = 0;
      for (let i = 0; i < 1500; i++) {
        const v = point(rnd);
        const r = [0.03, 0.1, 0.3, 0.9][i % 4]! * (0.5 + rnd());
        const want = scanWithin(c, kind, v, r);
        expect(indexWithin(c, kind, v, r)).toEqual(want);
        hits += want.length;
      }
      expect(hits).toBeGreaterThan(1500); // the queries do find things
    });
  }

  it('holds for queries placed exactly on stored points and on quartile edges', () => {
    const c = openContainer(packContainer(manifoldLayer('code', 2000, 31), nodeSha384));
    const layer = c.layer('code')!;
    const rnd = lcg(37);
    for (let i = 0; i < 400; i++) {
      const loc = layer.record(Math.floor(rnd() * layer.count)).locator;
      if (loc?.class !== 'manifold' || !loc.tera) continue;
      const v = [...loc.tera] as V4;
      if (i % 2) v[i % 4] = QUARTILES[1 + (i % 3)]!;
      for (const r of [0, 1e-9, 0.05, 0.2])
        expect(indexWithin(c, 'code', v, r)).toEqual(scanWithin(c, 'code', v, r));
    }
  });
});

describe('libaamt numbers the same cells differently, and the conversion is explicit', () => {
  /** Walk a real libaamt substrate: each leaf's Meji path and the TERA points stored under it. */
  function leaves(file: Uint8Array): { path: number; depth: number; tera: V4[] }[] {
    const dv = new DataView(file.buffer, file.byteOffset, file.byteLength);
    const nodesAt = Number(dv.getBigUint64(16, true));
    const nodeCount = Number(dv.getBigUint64(24, true));
    const payloadAt = Number(dv.getBigUint64(32, true));
    const out: { path: number; depth: number; tera: V4[] }[] = [];
    for (let n = 0; n < nodeCount; n++) {
      const at = nodesAt + n * 64;
      const flags = dv.getUint16(at + 10, true);
      if ((flags & 1) === 0) continue;
      const tera: V4[] = [];
      let r = payloadAt + Number(dv.getBigUint64(at + 24, true));
      for (let k = 0; k < dv.getUint32(at + 12, true); k++) {
        tera.push([0, 1, 2, 3].map((d) => dv.getFloat32(r + 8 + d * 4, true)) as V4);
        r += 32 + Math.ceil(dv.getUint32(r + 24, true) / 8) * 8;
      }
      out.push({ path: dv.getUint32(at, true), depth: flags >> 8, tera });
    }
    return out;
  }
  const fixture = (name: string) =>
    new Uint8Array(readFileSync(new URL(`./fixtures/${name}`, import.meta.url)));

  it('Meji: libaamt puts T at bit 0, the VM at bit 3 — a 4-bit reversal (substrate-v0.aamt)', () => {
    const ls = leaves(fixture('substrate-v0.aamt'));
    expect(ls.every((l) => l.depth === 1)).toBe(true);
    let differs = 0;
    for (const l of ls)
      for (const t of l.tera) {
        expect(mejiIndex(makeReg(t))).toBe(libaamtMejiToVm(l.path));
        if (l.path !== mejiIndex(makeReg(t))) differs++;
      }
    // The conventions genuinely disagree on this file: (0.72, 0.31, 0.88, 0.15) is libaamt 5, VM 10.
    expect(differs).toBeGreaterThan(0);
    expect(libaamtMejiToVm(5)).toBe(10);
  });

  it('Odu: libaamt’s (m₁ << 4) | m₂ maps onto the VM’s oduIndex for every record of a depth-2 tree', () => {
    // Packed by the libaamt CLI: `aamt pack odu-depth2.tsv odu-depth2.aamt 2 1` (96 records, quartile edges included).
    const ls = leaves(fixture('odu-depth2.aamt'));
    let n = 0;
    for (const l of ls) {
      expect(l.depth).toBe(2);
      for (const t of l.tera) {
        expect(oduIndex(makeReg(t))).toBe(libaamtCellToOdu(l.path));
        n++;
      }
    }
    expect(n).toBe(96);
  });

  it('agrees with WP-23’s published route: (0.72, 0.31, 0.88, 0.15) is Odu word 5.6', () => {
    expect(libaamtCellToOdu((5 << 4) | 6)).toBe(oduIndex(makeReg([0.72, 0.31, 0.88, 0.15])));
  });

  it('is a bijection on 0..255 and agrees with the VM Meji on the first level', () => {
    const image = new Set(Array.from({ length: 256 }, (_, c) => libaamtCellToOdu(c)));
    expect(image.size).toBe(256);
    for (let c = 0; c < 256; c++) {
      const odu = libaamtCellToOdu(c);
      const meji = ((odu >> 7) << 3) | (((odu >> 5) & 1) << 2) | (((odu >> 3) & 1) << 1) | ((odu >> 1) & 1);
      expect(meji).toBe(libaamtMejiToVm(c >> 4));
    }
    expect(() => libaamtCellToOdu(256)).toThrow(RangeError);
  });
});

describe('older files open unchanged', () => {
  // Packed by the v1.1 writer (the commit before ADR-018) from the input rebuilt here.
  const v11 = new Uint8Array(readFileSync(new URL('./fixtures/container-v1.1.aamt', import.meta.url)));
  const v11Input = (): ContainerInput => {
    const rnd = lcg(1101);
    const tera = (): V4 => [rnd(), rnd(), rnd(), rnd()];
    return {
      layers: [
        { kind: 'world', records: [{ payload: enc.encode('brick'), at: { x: 1, y: 2, z: 3 } }] },
        {
          kind: 'experience',
          records: Array.from({ length: 40 }, (_, i) => ({
            payload: enc.encode(`memory ${i}`),
            tera: tera(),
            lens: 'wisdom' as const,
            tier: 'reconstructed' as const,
          })),
        },
        {
          kind: 'code',
          records: [
            ...Array.from({ length: 24 }, (_, i) => ({
              payload: enc.encode(`fn f${i}`),
              tera: tera(),
              lens: 'technical' as const,
              tier: 'declared' as const,
            })),
            { payload: enc.encode('fn untagged') },
          ],
        },
      ],
    };
  };

  it('opens and verifies a file written by the v1.1 writer, with no index', () => {
    const c = openContainer(v11);
    expect(c.versionMinor).toBe(1);
    expect(verifyContainer(v11, nodeSha384)).toEqual({ ok: true, problems: [] });
    expect(c.layers.map((l) => l.odu)).toEqual([null, null, null]);
    expect(c.layer('experience')!.count).toBe(40);
  });

  it('a v1.2 pack without indexes differs from the v1.1 file only in the minor version and the fileId', () => {
    const input = v11Input();
    const off = packContainer(
      { layers: input.layers.map((l) => (l.kind === 'world' ? l : ({ ...l, oduIndex: false } as never))) },
      nodeSha384,
    );
    expect(off.byteLength).toBe(v11.byteLength);
    const relabelled = off.slice();
    new DataView(relabelled.buffer).setUint16(H_VERSION_MINOR, 1, true);
    expect([...reseal(relabelled)]).toEqual([...v11]);
  });

  it('ignores the v1.2 fields in a file that says it is older (v1.0, v1.1)', () => {
    const file = packContainer(v11Input(), nodeSha384);
    expect(openContainer(file).layer('code')!.odu).not.toBeNull();
    for (const minor of [0, 1]) {
      const old = file.slice();
      new DataView(old.buffer).setUint16(H_VERSION_MINOR, minor, true);
      const c = openContainer(reseal(old));
      expect(c.versionMinor).toBe(minor);
      expect(c.layer('code')!.odu).toBeNull();
      expect(verifyContainer(reseal(old), nodeSha384).ok).toBe(true);
    }
  });
});

describe('a corrupt index is refused, never used', () => {
  const base = packContainer(manifoldLayer('code', 400, 41, 5), nodeSha384);
  const { at, bytes, entry } = sectionOf(base, 'code');
  const posAt = (file: Uint8Array): number => {
    const occupied = new DataView(file.buffer, file.byteOffset).getUint32(at + 12, true);
    return at + ODU_HEADER_BYTES + (occupied + 1) * 4;
  };
  const edit = (f: (dv: DataView, file: Uint8Array) => void): Uint8Array => {
    const file = base.slice();
    f(new DataView(file.buffer), file);
    return file;
  };
  /** The structure is checked on open. */
  const refusedOnOpen = (file: Uint8Array, why: RegExp): void => {
    expect(() => openContainer(file)).toThrow(ContainerFormatError);
    expect(() => openContainer(file)).toThrow(why);
    expect(verifyContainer(file, nodeSha384).ok).toBe(false);
  };
  /** The meaning is checked before the first lookup, and in verifyContainer. */
  const refusedOnUse = (file: Uint8Array, why: RegExp): void => {
    const odu = openContainer(file).layer('code')!.odu!;
    expect(odu.check().join('\n')).toMatch(why);
    expect(() => odu.candidates([0.5, 0.5, 0.5, 0.5], 0.3)).toThrow(ContainerFormatError);
    expect(() => odu.cell(0)).toThrow(/refusing a corrupt Odu index/);
    const v = verifyContainer(reseal(file), nodeSha384);
    expect(v.ok).toBe(false);
    expect(v.problems.join('\n')).toMatch(why);
  };

  it('structure: magic, cell function, record count, mask, slot table, padding, placement', () => {
    refusedOnOpen(
      edit((dv) => dv.setUint32(at, 0, true)),
      /bad magic/,
    );
    refusedOnOpen(
      edit((dv) => dv.setUint32(at + 4, 2, true)),
      /cell function 2/,
    );
    refusedOnOpen(
      edit((dv) => dv.setUint32(at + 16, 999, true)),
      /built for 999 records/,
    );
    refusedOnOpen(
      edit((_, f) => (f[at + 20] = 1)),
      /reserved/,
    );
    refusedOnOpen(
      edit((_, f) => (f[at + 32] = f[at + 32]! ^ 1)),
      /occupied cells|is \d+ bytes/,
    );
    refusedOnOpen(
      edit((dv) => dv.setUint32(at + ODU_HEADER_BYTES + 4, 0, true)),
      /strictly increasing/,
    );
    const used = posAt(base) + new DataView(base.buffer).getUint32(at + 8, true) * 4;
    if (used < at + bytes)
      refusedOnOpen(
        edit((_, f) => (f[at + bytes - 1] = 7)),
        /padding/,
      );
    refusedOnOpen(
      edit((dv) => dv.setBigUint64(entry + L_ODU_OFFSET, BigInt(at + 8), true)),
      /misaligned/,
    );
    refusedOnOpen(
      edit((dv) => dv.setBigUint64(entry + L_ODU_OFFSET, BigInt(at - 64), true)),
      /overlaps/,
    );
    refusedOnOpen(
      edit((dv) => dv.setBigUint64(entry + L_ODU_BYTES, BigInt(bytes + 64), true)),
      /past the file/,
    );
  });

  it('structure: a non-manifold layer may not claim an index', () => {
    const file = packContainer(
      {
        layers: [
          { kind: 'world', records: [{ payload: enc.encode('b'), at: { x: 0, y: 0, z: 0 } }] },
          ...manifoldLayer('code', 20, 43).layers,
        ],
      },
      nodeSha384,
    );
    const s = sectionOf(file, 'code');
    const dv = new DataView(file.buffer);
    dv.setBigUint64(HEADER_BYTES + L_ODU_OFFSET, BigInt(s.at), true);
    dv.setBigUint64(HEADER_BYTES + L_ODU_BYTES, BigInt(s.bytes), true);
    refusedOnOpen(file, /not a manifold layer/);
  });

  it('meaning: a record filed under the wrong cell', () => {
    // Move the first entry of the first cell to sit, structurally valid, at the end of the previous cell.
    const file = edit((dv) => {
      const odu = openContainer(base).layer('code')!.odu!;
      const occupied = [...Array(256).keys()].filter((c) => odu.cell(c).length > 0);
      const second = occupied.findIndex((c, i) => i > 0 && odu.cell(c).length > 1);
      // Shrink the second occupied cell by one at its start: that record is now claimed by the cell before it.
      dv.setUint32(
        at + ODU_HEADER_BYTES + second * 4,
        dv.getUint32(at + ODU_HEADER_BYTES + second * 4, true) + 1,
        true,
      );
    });
    refusedOnUse(file, /belongs in cell|out of range or out of order/);
  });

  it('meaning: positions out of order, duplicated, or out of range', () => {
    const p = posAt(base);
    const odu = openContainer(base).layer('code')!.odu!;
    const cell = [...Array(256).keys()].find((c) => odu.cell(c).length >= 2)!;
    let i = 0;
    for (let c = 0; c < cell; c++) i += odu.cell(c).length;
    const [a, b] = odu.cell(cell);
    refusedOnUse(
      edit((dv) => {
        dv.setUint32(p + i * 4, b!, true);
        dv.setUint32(p + (i + 1) * 4, a!, true);
      }),
      /out of order/,
    );
    refusedOnUse(
      edit((dv) => dv.setUint32(p + (i + 1) * 4, a!, true)),
      /out of order/,
    );
    refusedOnUse(
      edit((dv) => dv.setUint32(p + i * 4, 100000, true)),
      /out of range/,
    );
  });

  it('meaning: a located record missing from the index', () => {
    const c = openContainer(base);
    const layer = c.layer('code')!;
    const locators = [...layer.records()].map((r) => r.locatorBytes.slice());
    const victim = locators.findIndex((l) => new DataView(l.buffer).getUint32(20, true) !== 0);
    new DataView(locators[victim]!.buffer).setUint32(20, 0, true); // index it as if it had no point
    const short = buildOduSection(locators)!;
    expect(short.byteLength).toBeLessThanOrEqual(bytes);
    const file = base.slice();
    file.fill(0, at, at + bytes);
    file.set(short, at);
    new DataView(file.buffer).setBigUint64(entry + L_ODU_BYTES, BigInt(short.byteLength), true);
    refusedOnUse(file, /routes 399 records; the layer has 400 with a TERA point/);
  });

  it('meaning: a record whose point moved to another cell after indexing', () => {
    const c = openContainer(base);
    const r = [...c.layer('code')!.records()].find((x) => x.locator?.class === 'manifold' && x.locator.tera)!;
    const file = base.slice();
    const at0 = r.locatorBytes.byteOffset - base.byteOffset;
    const dv = new DataView(file.buffer);
    dv.setFloat32(at0, dv.getFloat32(at0, true) < 0.5 ? 0.9 : 0.1, true);
    refusedOnUse(file, /belongs in cell/);
  });

  it('fuzz: random byte flips never crash and never yield a wrong index', () => {
    const rnd = lcg(47);
    const c0 = openContainer(base);
    const layerAt = entryOf(c0, 'code');
    const records0 = c0.layer('code')!;
    const firstLocator = records0.record(0).locatorBytes.byteOffset - base.byteOffset - 72;
    let refused = 0;
    let usableAfterFlip = 0;
    for (let t = 0; t < 1500; t++) {
      const file = base.slice();
      const zone = rnd();
      const pos =
        zone < 0.5
          ? at + Math.floor(rnd() * bytes) // the section
          : zone < 0.65
            ? layerAt + 88 + Math.floor(rnd() * 16) // the layer entry's Odu fields
            : zone < 0.9
              ? firstLocator + Math.floor(rnd() * records0.count) * 128 + 72 + Math.floor(rnd() * 24) // locators
              : Math.floor(rnd() * file.byteLength);
      file[pos] = file[pos]! ^ (1 << Math.floor(rnd() * 8));
      let c: Container;
      try {
        c = openContainer(file);
      } catch (e) {
        expect(e).toBeInstanceOf(ContainerFormatError);
        refused++;
        continue;
      }
      expect(verifyContainer(file, nodeSha384).ok).toBe(false); // fileId, at the very least
      const odu = c.layer('code')?.odu;
      const inSection = pos >= at && pos < at + bytes;
      if (!odu) {
        expect(inSection).toBe(false);
        continue;
      }
      let problems: readonly string[];
      try {
        problems = odu.check();
      } catch (e) {
        // A flipped locator lens/tier byte makes the record itself undecodable; that is a refusal too.
        expect(e).toBeInstanceOf(ContainerFormatError);
        refused++;
        continue;
      }
      if (inSection) expect(problems.length).toBeGreaterThan(0); // every section byte is load-bearing
      if (problems.length > 0) {
        refused++;
        continue;
      }
      // The index passed its check: it must agree with the scan of the file as it now is.
      usableAfterFlip++;
      for (let k = 0; k < 3; k++) {
        const v = point(rnd);
        const r = [0.1, 0.3, 0.9][k]!;
        let want: number[];
        try {
          want = scanWithin(c, 'code', v, r);
        } catch (e) {
          expect(e).toBeInstanceOf(ContainerFormatError);
          continue;
        }
        expect(indexWithin(c, 'code', v, r)).toEqual(want);
      }
    }
    expect(refused).toBeGreaterThan(700);
    expect(usableAfterFlip).toBeGreaterThan(50);
  });
});
