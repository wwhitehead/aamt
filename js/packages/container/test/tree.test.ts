// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { frDist, makeReg, mejiIndex, oduIndex, type TeraReg } from '@aamt/tera';
import {
  CONTEXT,
  ContainerFormatError,
  HEADER_BYTES,
  H_FILE_ID,
  H_VERSION_MINOR,
  ID_BYTES,
  LAYER_ENTRY_BYTES,
  L_ODU_OFFSET,
  L_TREE_BYTES,
  L_TREE_OFFSET,
  TREE_HEADER_BYTES,
  TREE_NODE_BYTES,
  buildTreeSection,
  identify,
  libaamtMejiToVm,
  oduCell,
  openContainer,
  packContainer,
  parseTreeSection,
  treeChild,
  treeCoord,
  verifyContainer,
  writeContainer,
  type Container,
  type ContainerInput,
  type ContainerSink,
  type LayerKind,
  type ManifoldRecordInput,
  type TreeOptions,
} from '../src/index';
import { nodeSha384, nodeSha384Incremental } from '../src/node';

/** The adaptive Meji tree section (v1.3, ADR-023). */

type V4 = [number, number, number, number];
const enc = new TextEncoder();

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}
const f32 = Math.fround;
const stored = (v: V4): V4 => v.map(f32) as V4;

/** A coordinate biased toward the places a cell function branches: every dyadic edge to depth 8, and its f32 neighbours. */
function lane(rnd: () => number): number {
  const r = rnd();
  if (r < 0.2) return Math.floor(rnd() * 257) / 256;
  if (r < 0.35) {
    const e = Math.floor(rnd() * 257) / 256;
    return Math.min(1, Math.max(0, e + (rnd() < 0.5 ? -1 : 1) * 2 ** -24 * Math.ceil(rnd() * 4)));
  }
  return rnd();
}
const edgePoint = (rnd: () => number): V4 => [lane(rnd), lane(rnd), lane(rnd), lane(rnd)];

/** Gaussian-ish clusters: `k` centres, each point a sum of uniforms around one, clamped to the box. */
function clustered(n: number, k: number, spread: number, seed: number): V4[] {
  const rnd = lcg(seed);
  const centres = Array.from({ length: k }, () => [rnd(), rnd(), rnd(), rnd()] as V4);
  return Array.from({ length: n }, () => {
    const c = centres[Math.floor(rnd() * k)]!;
    return c.map((x) => Math.min(1, Math.max(0, x + (rnd() + rnd() + rnd() - 1.5) * spread))) as V4;
  });
}

function layerOf(
  points: V4[],
  tree: boolean | TreeOptions | undefined,
  extra: { odu?: boolean; kind?: 'experience' | 'code'; untagged?: number } = {},
): ContainerInput {
  const kind = extra.kind ?? 'code';
  const records: ManifoldRecordInput[] = points.map((p, i) => ({
    payload: enc.encode(`${kind} ${i}`),
    tera: p,
    lens: 'technical',
    tier: 'declared',
  }));
  for (let i = 0; i < (extra.untagged ?? 0); i++) records.push({ payload: enc.encode(`untagged ${i}`) });
  return { layers: [{ kind, records, ...(tree !== undefined ? { treeIndex: tree } : {}), oduIndex: extra.odu ?? true }] };
}

const regCache = new WeakMap<Container, (TeraReg | null)[]>();
function regsOf(c: Container): (TeraReg | null)[] {
  let regs = regCache.get(c);
  if (!regs) {
    regs = [...c.layer('code')!.records()].map((r) =>
      r.locator?.class === 'manifold' && r.locator.tera ? makeReg(r.locator.tera) : null,
    );
    regCache.set(c, regs);
  }
  return regs;
}
const scanWithin = (c: Container, v: V4, radius: number): number[] => {
  const q = makeReg(v);
  const out: number[] = [];
  regsOf(c).forEach((reg, i) => {
    if (reg && !(frDist(q, reg) > radius)) out.push(i);
  });
  return out;
};
const treeWithin = (c: Container, v: V4, radius: number): number[] => {
  const q = makeReg(v);
  const regs = regsOf(c);
  return Array.from(c.layer('code')!.tree!.candidates(v, radius)).filter((i) => {
    const reg = regs[i];
    return reg !== null && reg !== undefined && !(frDist(q, reg) > radius);
  });
};

const sectionOf = (file: Uint8Array, kind: LayerKind): { at: number; bytes: number; entry: number } => {
  const c = openContainer(file);
  const entry = HEADER_BYTES + c.layers.findIndex((l) => l.kind === kind) * LAYER_ENTRY_BYTES;
  const dv = new DataView(file.buffer, file.byteOffset);
  return {
    at: Number(dv.getBigUint64(entry + L_TREE_OFFSET, true)),
    bytes: Number(dv.getBigUint64(entry + L_TREE_BYTES, true)),
    entry,
  };
};
function reseal(file: Uint8Array): Uint8Array {
  const out = file.slice();
  out.fill(0, H_FILE_ID, H_FILE_ID + ID_BYTES);
  out.set(identify(nodeSha384, CONTEXT.container, [out]), H_FILE_ID);
  return out;
}

// ── nothing changes for a file without a tree ─────────────────────────────────

describe('a file without a tree is the file it always was', () => {
  it('packs byte-identically to the writer before v1.3 (pinned: size and SHA-384)', () => {
    const r = lcg(7);
    const mk = (n: number): ManifoldRecordInput[] =>
      Array.from({ length: n }, (_, i) => ({
        kind: 0,
        payload: enc.encode(`p${i}`),
        tera: [r(), r(), r(), r()] as V4,
        lens: 'technical' as const,
        tier: 'declared' as const,
      }));
    const bytes = packContainer({ layers: [{ kind: 'experience', records: mk(500) }, { kind: 'code', records: mk(40) }] }, nodeSha384);
    expect(bytes.byteLength).toBe(81536);
    expect(createHash('sha384').update(bytes).digest('hex')).toBe(
      '8de45c3cfb11425fe8fe2693d0f7fb0d9c6aa44915038bc1a117077e38cd9ebd0d7b624c7d9680983c24b1479ee09a88',
    );
    expect(openContainer(bytes).versionMinor).toBe(2);
  });

  it('is stamped 1.3 only when it holds a tree, and a v1.2 reader still opens it and uses the Odu index', () => {
    const pts = clustered(300, 4, 0.04, 1);
    const plain = packContainer(layerOf(pts, undefined), nodeSha384);
    const treed = packContainer(layerOf(pts, true), nodeSha384);
    expect(openContainer(plain).versionMinor).toBe(2);
    const c = openContainer(treed);
    expect(c.versionMinor).toBe(3);
    expect(c.layer('code')!.tree).not.toBeNull();
    expect(c.layer('code')!.odu).not.toBeNull();
    // What a v1.2 reader sees: the same bytes, stamped 1.2. It does not read the tree fields.
    const asOld = treed.slice();
    new DataView(asOld.buffer).setUint16(H_VERSION_MINOR, 2, true);
    const old = openContainer(asOld);
    expect(old.layer('code')!.tree).toBeNull();
    expect(old.layer('code')!.odu!.candidates([0.5, 0.5, 0.5, 0.5], 0.3).length).toBeGreaterThan(0);
  });

  it('a tree-only layer has no Odu section, and still opens and searches', () => {
    const pts = clustered(200, 3, 0.05, 2);
    const c = openContainer(packContainer(layerOf(pts, true, { odu: false }), nodeSha384));
    expect(c.layer('code')!.odu).toBeNull();
    expect(c.layer('code')!.tree!.entryCount).toBe(200);
  });

  it('refuses a tree on a layer that has no TERA locators, and says so', () => {
    expect(() =>
      packContainer({ layers: [{ kind: 'world', records: [{ payload: enc.encode('x'), at: { x: 0, y: 0, z: 0 } }], treeIndex: true } as never] }, nodeSha384),
    ).toThrow(/manifold layers only/);
  });
});

// ── the cell function is the Odu index's, one level at a time ─────────────────

describe('the quantiser', () => {
  it('two levels are exactly the VM’s oduIndex (300,000 points, dyadic edges and their f32 neighbours, out-of-range included)', () => {
    const rnd = lcg(11);
    for (let i = 0; i < 300_000; i++) {
      const v: V4 = i % 7 === 0 ? [rnd() * 3 - 1, rnd() * 3 - 1, rnd() * 3 - 1, rnd() * 3 - 1] : edgePoint(rnd);
      const t = stored(v);
      let cell = 0;
      for (let d = 0; d < 4; d++) {
        const q = 2 * (treeCoord(t[d]!, 1) & 1) + (treeCoord(t[d]!, 2) & 1);
        cell |= q << (2 * (3 - d));
      }
      // Compare at depth 2 directly and through the child numbers of levels 1 and 2.
      const direct = treeCoord(t[0]!, 2) << 6 | treeCoord(t[1]!, 2) << 4 | treeCoord(t[2]!, 2) << 2 | treeCoord(t[3]!, 2);
      expect(direct).toBe(oduIndex(makeReg(t)));
      expect(cell).toBe(oduCell(t));
    }
  });

  it('non-finite lanes land in cell 0, as in the VM', () => {
    for (const x of [NaN, -1, -Infinity, -0]) expect(treeCoord(x, 5)).toBe(0);
    expect(treeCoord(Infinity, 5)).toBe(31);
    expect(treeCoord(1, 8)).toBe(255);
    expect(oduIndex(makeReg([NaN, 0.1, 0.1, 0.1] as never))).toBe(oduCell([NaN, 0.1, 0.1, 0.1] as never));
  });

  it('is nested: a cell’s coordinate at depth d+1, halved, is its coordinate at depth d', () => {
    const rnd = lcg(12);
    for (let i = 0; i < 100_000; i++) {
      const x = lane(rnd);
      for (let d = 0; d < 8; d++) expect(treeCoord(x, d + 1) >> 1).toBe(treeCoord(x, d));
    }
  });

  it('its first level is the VM’s mejiIndex', () => {
    const rnd = lcg(13);
    for (let i = 0; i < 50_000; i++) {
      const t = stored(edgePoint(rnd));
      expect(treeChild(t, 1)).toBe(mejiIndex(makeReg(t)));
    }
  });
});

// ── against files the real libaamt wrote ──────────────────────────────────────

describe('libaamt’s own trees', () => {
  interface Leaf { path: number; depth: number; tera: V4[] }
  function leaves(file: Uint8Array): Leaf[] {
    const dv = new DataView(file.buffer, file.byteOffset, file.byteLength);
    const nodesAt = Number(dv.getBigUint64(16, true));
    const nodeCount = Number(dv.getBigUint64(24, true));
    const payloadAt = Number(dv.getBigUint64(32, true));
    const out: Leaf[] = [];
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
  const fixture = (name: string) => new Uint8Array(readFileSync(new URL(`./fixtures/${name}`, import.meta.url)));
  const key = (t: V4): string => t.join(',');

  /** Leaves of a real tree built from `points` by this package: path (VM-numbered nibbles) → record keys. */
  function leavesOfMine(points: V4[], opts: TreeOptions): Map<string, string[]> {
    const locators = points.map((p) => {
      const l = new Uint8Array(32);
      const dv = new DataView(l.buffer);
      p.forEach((x, i) => dv.setFloat32(i * 4, x, true));
      dv.setUint32(20, 1, true); // tier: declared
      return l;
    });
    const section = buildTreeSection(locators, opts)!;
    const parsed = parseTreeSection(section, points.length);
    if (!parsed.ok) throw new Error(parsed.problem);
    const s = parsed.shape;
    // Rebuild each node's child path from the masks, breadth-first.
    const paths: number[][] = Array.from({ length: s.nodeCount }, () => []);
    for (let n = 0; n < s.nodeCount; n++) {
      if (s.isLeaf(n)) continue;
      let slot = 0;
      for (let k = 0; k < 16; k++) if ((s.mask(n) >> k) & 1) paths[s.a(n) + slot++] = [...paths[n]!, k];
    }
    const out = new Map<string, string[]>();
    for (let n = 0; n < s.nodeCount; n++) {
      if (!s.isLeaf(n)) continue;
      const recs: string[] = [];
      for (let i = s.a(n); i < s.a(n) + s.b(n); i++) recs.push(key(points[s.positions(i)]!));
      out.set(paths[n]!.join('.'), recs.sort());
    }
    return out;
  }

  it('the quantiser agrees with a depth-2 libaamt file, record for record (96 records, quartile edges included)', () => {
    let n = 0;
    for (const l of leaves(fixture('odu-depth2.aamt')))
      for (const t of l.tera) {
        const [m1, m2] = [libaamtMejiToVm(l.path >> 4), libaamtMejiToVm(l.path & 15)];
        expect([treeChild(t, 1), treeChild(t, 2)]).toEqual([m1, m2]);
        n++;
      }
    expect(n).toBe(96);
  });

  it('building over the same 96 records with capacity 1 and depth 2 gives libaamt’s tree: the same 82 leaves with the same records', () => {
    const theirs = new Map<string, string[]>();
    const points: V4[] = [];
    for (const l of leaves(fixture('odu-depth2.aamt'))) {
      const p = `${libaamtMejiToVm(l.path >> 4)}.${libaamtMejiToVm(l.path & 15)}`;
      theirs.set(p, l.tera.map(key).sort());
      points.push(...l.tera);
    }
    expect(theirs.size).toBe(82);
    const mine = leavesOfMine(points, { leafCapacity: 1, maxDepth: 2 });
    expect([...mine.keys()].sort()).toEqual([...theirs.keys()].sort());
    for (const [p, recs] of theirs) expect(mine.get(p)).toEqual(recs);
  });

  it('and over 606 clustered records packed by the libaamt CLI at depths 3, 5 and 8: the same leaves, record for record', () => {
    // `aamt pack tree-clustered.tsv <out> <depth> <capacity>`: five tight clusters, a scatter, 80 points exactly
    // on dyadic boundaries to depth 8, and six identical points that no depth can separate.
    for (const [file, depth, capacity] of [['tree-d3-b1.aamt', 3, 1], ['tree-d5-b2.aamt', 5, 2], ['tree-d8-b4.aamt', 8, 4]] as const) {
      const theirs = new Map<string, string[]>();
      const points: V4[] = [];
      const depths = new Set<number>();
      for (const l of leaves(fixture(file))) {
        const nibbles = Array.from({ length: l.depth }, (_, i) => libaamtMejiToVm((l.path >> (4 * (l.depth - 1 - i))) & 15));
        theirs.set(nibbles.join('.'), l.tera.map(key).sort());
        points.push(...l.tera);
        depths.add(l.depth);
      }
      expect(points.length).toBe(606);
      const mine = leavesOfMine(points, { leafCapacity: capacity, maxDepth: depth });
      expect([...mine.keys()].sort()).toEqual([...theirs.keys()].sort());
      for (const [p, recs] of theirs) expect(mine.get(p)).toEqual(recs);
      // The depth-8 file really does reach deep: the comparison is not trivially shallow.
      if (depth === 8) expect(Math.max(...depths)).toBeGreaterThanOrEqual(6);
      console.log(`libaamt depth ${depth}, capacity ${capacity}: ${theirs.size} leaves, deepest ${Math.max(...depths)}: identical`);
    }
  });

  it('and over the depth-1 file (capacity 1, depth 1): the same 3 leaves', () => {
    const theirs = new Map<string, string[]>();
    const points: V4[] = [];
    for (const l of leaves(fixture('substrate-v0.aamt'))) {
      theirs.set(`${libaamtMejiToVm(l.path)}`, l.tera.map(key).sort());
      points.push(...l.tera);
    }
    const mine = leavesOfMine(points, { leafCapacity: 1, maxDepth: 1 });
    expect([...mine.keys()].sort()).toEqual([...theirs.keys()].sort());
    for (const [p, recs] of theirs) expect(mine.get(p)).toEqual(recs);
  });
});

// ── the builder ───────────────────────────────────────────────────────────────

describe('the builder', () => {
  const locatorsOf = (points: V4[]): Uint8Array[] =>
    points.map((p) => {
      const l = new Uint8Array(32);
      const dv = new DataView(l.buffer);
      p.forEach((x, i) => dv.setFloat32(i * 4, x, true));
      dv.setUint32(20, 1, true);
      return l;
    });
  const shapeOf = (points: V4[], o?: TreeOptions) => {
    const s = buildTreeSection(locatorsOf(points), o)!;
    const p = parseTreeSection(s, points.length);
    if (!p.ok) throw new Error(p.problem);
    return { bytes: s, shape: p.shape };
  };

  it('is deterministic, and a tree has exactly one canonical shape for its records', () => {
    const pts = clustered(2000, 6, 0.03, 3);
    expect(shapeOf(pts).bytes).toEqual(shapeOf(pts).bytes);
    expect(shapeOf(pts, { leafCapacity: 8 }).shape.nodeCount).toBeGreaterThan(shapeOf(pts, { leafCapacity: 64 }).shape.nodeCount);
  });

  it('puts every located record in exactly one leaf; a leaf above capacity is only ever at the depth limit', () => {
    for (const [seed, k, spread] of [[4, 5, 0.02], [5, 1, 0.1], [6, 40, 0.3]] as const) {
      const pts = clustered(3000, k, spread, seed);
      for (const o of [{ leafCapacity: 4 }, { leafCapacity: 32, maxDepth: 3 }, { leafCapacity: 1, maxDepth: 8 }]) {
        const { shape } = shapeOf(pts, o);
        const seen = new Uint8Array(pts.length);
        for (let n = 0; n < shape.nodeCount; n++) {
          if (!shape.isLeaf(n)) continue;
          if (shape.depth(n) < shape.maxDepth) expect(shape.b(n)).toBeLessThanOrEqual(shape.leafCapacity);
          for (let i = shape.a(n); i < shape.a(n) + shape.b(n); i++) seen[shape.positions(i)]!++;
        }
        expect(seen.every((x) => x === 1)).toBe(true);
      }
    }
  });

  it('identical points never split below the depth limit: one deep bucket, never an alias', () => {
    const pts: V4[] = Array.from({ length: 100 }, () => [0.3, 0.3, 0.3, 0.3]);
    const { shape } = shapeOf(pts, { leafCapacity: 4, maxDepth: 8 });
    expect(shape.deepest).toBe(8);
    expect(shape.leafCount).toBe(1);
    expect(shape.b(shape.nodeCount - 1)).toBe(100);
  });

  it('a dense cluster is cut deeper than a sparse region', () => {
    const dense = clustered(2000, 1, 0.01, 7);
    const sparse = Array.from({ length: 200 }, (_, i): V4 => [(i % 10) / 10 + 0.05, ((i / 10) | 0) / 20 + 0.02, 0.5, 0.5]);
    const { shape } = shapeOf([...dense, ...sparse], { leafCapacity: 16 });
    const depths = new Set<number>();
    for (let n = 0; n < shape.nodeCount; n++) if (shape.isLeaf(n)) depths.add(shape.depth(n));
    expect(Math.max(...depths)).toBeGreaterThan(Math.min(...depths) + 1);
  });

  it('returns null with no TERA point, and refuses nonsense options', () => {
    expect(buildTreeSection([new Uint8Array(32)])).toBeNull();
    expect(buildTreeSection([])).toBeNull();
    for (const bad of [{ leafCapacity: 0 }, { leafCapacity: 1.5 }, { maxDepth: 0 }, { maxDepth: 9 }, { maxDepth: 2.5 }])
      expect(() => buildTreeSection(locatorsOf([[0.1, 0.2, 0.3, 0.4]]), bad)).toThrow(RangeError);
  });

  it('records without a TERA point are left out and counted correctly', () => {
    const pts = clustered(100, 2, 0.05, 8);
    const c = openContainer(packContainer(layerOf(pts, true, { untagged: 17 }), nodeSha384));
    expect(c.layer('code')!.count).toBe(117);
    expect(c.layer('code')!.tree!.entryCount).toBe(100);
    expect(c.layer('code')!.tree!.check()).toEqual([]);
  });
});

// ── the index never loses a candidate ─────────────────────────────────────────

describe('index ≡ scan: no record within the radius is lost', () => {
  const corpora: [string, V4[]][] = [
    ['uniform', Array.from({ length: 2500 }, (() => { const r = lcg(21); return () => edgePoint(r); })())],
    ['clustered', clustered(2500, 8, 0.03, 22)],
    ['one tight cluster', clustered(2000, 1, 0.005, 23)],
    ['on the grid: every point a dyadic edge', Array.from({ length: 1500 }, (() => { const r = lcg(24); return (): V4 => [Math.floor(r() * 17) / 16, Math.floor(r() * 17) / 16, Math.floor(r() * 5) / 4, Math.floor(r() * 257) / 256]; })())],
  ];
  for (const [name, pts] of corpora) {
    it(`${name}: 1,500 random queries and radii, exactly the scan’s hits`, () => {
      const c = openContainer(packContainer(layerOf(pts, { leafCapacity: 8 }), nodeSha384));
      const rnd = lcg(31);
      let nonEmpty = 0;
      for (let i = 0; i < 1500; i++) {
        const v: V4 = i % 3 === 0 ? stored(pts[Math.floor(rnd() * pts.length)]!) : edgePoint(rnd);
        const r = [0, 1e-9, 0.03, 0.1, 0.3, 0.9, Math.PI][Math.floor(rnd() * 7)]!;
        const scan = scanWithin(c, v, r);
        expect(treeWithin(c, v, r)).toEqual(scan);
        if (scan.length > 0) nonEmpty++;
      }
      expect(nonEmpty).toBeGreaterThan(300);
    });
  }

  it('agrees with the Odu index on the same layer', () => {
    const pts = clustered(1500, 5, 0.04, 25);
    const c = openContainer(packContainer(layerOf(pts, { leafCapacity: 12 }), nodeSha384));
    const q = lcg(26);
    for (let i = 0; i < 400; i++) {
      const v = edgePoint(q);
      const r = [0.03, 0.1, 0.3][i % 3]!;
      const reg = regsOf(c);
      const qr = makeReg(v);
      const odu = Array.from(c.layer('code')!.odu!.candidates(v, r)).filter((p) => !(frDist(qr, reg[p]!) > r));
      expect(treeWithin(c, v, r)).toEqual(odu);
    }
  });

  it('a radius of π reaches everything; a negative-free zero radius reaches at most the identical points', () => {
    const pts = clustered(500, 3, 0.05, 27);
    const c = openContainer(packContainer(layerOf(pts, true), nodeSha384));
    expect(c.layer('code')!.tree!.candidates([0.5, 0.5, 0.5, 0.5], 4).length).toBe(500);
    const some = stored(pts[17]!);
    expect(treeWithin(c, some, 0)).toEqual(scanWithin(c, some, 0));
  });
});

describe('what the tree buys (measured on seeded data; ADR-023 reports the full tables)', () => {
  const pack = (pts: V4[], o: TreeOptions, odu = true) =>
    openContainer(packContainer(layerOf(pts, o, { odu }), nodeSha384)).layer('code')!;
  /** Mean candidates per query over 100 queries centred on stored points. */
  function perQuery(layer: ReturnType<typeof pack>, radius: number, which: 'odu' | 'tree'): number {
    const rnd = lcg(42);
    let n = 0;
    for (let i = 0; i < 100; i++) {
      const v = (layer.record(Math.floor(rnd() * layer.count)).locator as unknown as { tera: V4 }).tera;
      n += which === 'odu' ? layer.odu!.candidates(v, radius).length : layer.tree!.candidates(v, radius).length;
    }
    return n / 100;
  }

  it('when the radius is smaller than the cluster, it reads a small fraction of what the Odu index reads', () => {
    const layer = pack(clustered(20000, 6, 0.06, 41), { leafCapacity: 16 });
    const odu = perQuery(layer, 0.03, 'odu');
    const tree = perQuery(layer, 0.03, 'tree');
    console.log(`6 clusters (σ≈0.06), r=0.03: Odu ${odu} candidates/query, tree ${tree} (${(odu / tree).toFixed(0)}×)`);
    expect(tree).toBeLessThan(odu / 20);
  });

  it('when the radius spans the cluster, the gain is modest — the extra records are real hits', () => {
    const layer = pack(clustered(20000, 6, 0.012, 41), { leafCapacity: 16 });
    const odu = perQuery(layer, 0.1, 'odu');
    const tree = perQuery(layer, 0.1, 'tree');
    console.log(`6 tight clusters (σ≈0.012), r=0.1: Odu ${odu}, tree ${tree} (${(odu / tree).toFixed(2)}×)`);
    expect(tree).toBeLessThanOrEqual(odu);
    expect(tree).toBeGreaterThan(odu / 3); // not a large gain, and not claimed to be
  });

  it('adaptivity is worth something beyond raw fineness: fewer nodes than a fixed depth-5 cut, and fewer candidates', () => {
    const pts = clustered(20000, 6, 0.06, 41);
    // capacity 1 splits every shared box down to the limit: a fixed-depth cut (singleton leaves read one record).
    const fixed = pack(pts, { leafCapacity: 1, maxDepth: 5 }, false);
    const adaptive = pack(pts, { leafCapacity: 64 }, false);
    expect(adaptive.tree!.nodeCount).toBeLessThan(fixed.tree!.nodeCount);
    for (const r of [0.03, 0.1]) {
      const a = perQuery(adaptive, r, 'tree');
      const f = perQuery(fixed, r, 'tree');
      console.log(`r=${r}: adaptive (${adaptive.tree!.nodeCount} nodes) ${a} candidates/query, fixed depth 5 (${fixed.tree!.nodeCount} nodes) ${f}`);
      expect(a).toBeLessThan(f);
    }
  });

  it('on uniform data it reads no more than the Odu index — but that is mostly fineness (20,000 points in 256 cells is ~78 a cell), not adaptivity', () => {
    const rnd = lcg(43);
    const pts = Array.from({ length: 20000 }, (): V4 => [rnd(), rnd(), rnd(), rnd()]);
    const layer = pack(pts, { leafCapacity: 16 });
    const odu = perQuery(layer, 0.1, 'odu');
    const tree = perQuery(layer, 0.1, 'tree');
    console.log(`uniform 20,000, r=0.1: Odu ${odu}, tree ${tree} (${(odu / tree).toFixed(1)}×)`);
    expect(tree).toBeLessThanOrEqual(odu);
  });
});

// ── a corrupt tree is refused, never used ─────────────────────────────────────

describe('a corrupt tree is refused', () => {
  const pts = clustered(600, 4, 0.03, 51);
  const base = packContainer(layerOf(pts, { leafCapacity: 6 }, { odu: false }), nodeSha384);
  const { at, bytes, entry } = sectionOf(base, 'code');
  const nodeCount = new DataView(base.buffer, base.byteOffset).getUint32(at + 12, true);
  const nodeAt = (n: number): number => at + TREE_HEADER_BYTES + n * TREE_NODE_BYTES;
  const edit = (f: (dv: DataView, file: Uint8Array) => void): Uint8Array => {
    const file = base.slice();
    f(new DataView(file.buffer), file);
    return reseal(file);
  };
  const refusedOnOpen = (file: Uint8Array, re: RegExp) => expect(() => openContainer(file)).toThrow(re);

  it('the genuine file is sound', () => {
    expect(verifyContainer(base, nodeSha384)).toEqual({ ok: true, problems: [] });
    expect(nodeCount).toBeGreaterThan(10);
  });

  it('structure: bad magic, cell function, record count, reserved bytes, padding, size', () => {
    refusedOnOpen(edit((dv) => dv.setUint32(at, 0, true)), /bad magic/);
    refusedOnOpen(edit((dv) => dv.setUint32(at + 4, 2, true)), /cell function/);
    refusedOnOpen(edit((dv) => dv.setUint32(at + 16, 599, true)), /built for 599 records/);
    refusedOnOpen(edit((_, f) => (f[at + 40] = 1)), /reserved/);
    refusedOnOpen(edit((_, f) => (f[at + bytes - 1] = 1)), /padding/);
  });

  it('structure: the split rule, child blocks, depths and counts are all enforced', () => {
    refusedOnOpen(edit((dv) => dv.setUint32(at + 20, 1000, true)), /split with only|should have split|capacity/); // capacity raised: internal nodes now under it
    // Capacity lowered: leaves above the depth limit now hold more than it allows, so they should have split.
    refusedOnOpen(edit((dv) => dv.setUint32(at + 20, 2, true)), /should have split/);
    refusedOnOpen(edit((dv) => dv.setUint32(at + 24, 1, true)), /depth limit|beyond its maximum/);
    refusedOnOpen(edit((dv) => dv.setUint32(nodeAt(0) + 4, 5, true)), /children start at/);
    refusedOnOpen(edit((dv) => dv.setUint16(nodeAt(0), 0, true)), /no children|child mask/);
    refusedOnOpen(edit((dv) => dv.setUint32(nodeAt(0) + 8, 1, true)), /records beneath|its root holds|split with only/);
    refusedOnOpen(edit((_, f) => (f[nodeAt(1) + 2] = 5)), /depth/);
    refusedOnOpen(edit((_, f) => (f[nodeAt(1) + 3] = 9)), /unknown flags/);
    // Make a leaf claim one record fewer, so counts no longer add up.
    const leaf = Array.from({ length: nodeCount }, (_, n) => n).find((n) => base[nodeAt(n) + 3] === 1)!;
    refusedOnOpen(edit((dv) => dv.setUint32(nodeAt(leaf) + 8, dv.getUint32(nodeAt(leaf) + 8, true) - 1, true)), /./);
  });

  it('semantics: a record moved to the wrong leaf, a duplicate, an out-of-range position — opens, but is refused before any lookup', () => {
    const positionsAt = at + TREE_HEADER_BYTES + nodeCount * TREE_NODE_BYTES;
    const wrong = edit((dv) => {
      const a = dv.getUint32(positionsAt, true);
      const b = dv.getUint32(positionsAt + 4 * 300, true);
      dv.setUint32(positionsAt, b, true);
      dv.setUint32(positionsAt + 4 * 300, a, true);
    });
    const c = openContainer(wrong);
    expect(c.layer('code')!.tree!.check().length).toBeGreaterThan(0);
    expect(() => c.layer('code')!.tree!.candidates([0.5, 0.5, 0.5, 0.5], 1)).toThrow(ContainerFormatError);
    expect(verifyContainer(wrong, nodeSha384).ok).toBe(false);
    const oob = openContainer(edit((dv) => dv.setUint32(positionsAt, 99999, true)));
    expect(oob.layer('code')!.tree!.check().join()).toMatch(/out of range|out of order/);
  });

  it('claims on a layer that has no TERA locators, or beyond the file, are refused', () => {
    const world = packContainer({ layers: [{ kind: 'world', records: [{ payload: enc.encode('x'), at: { x: 0, y: 0, z: 0 } }] }] }, nodeSha384);
    const w = world.slice();
    new DataView(w.buffer).setUint16(H_VERSION_MINOR, 3, true);
    new DataView(w.buffer).setBigUint64(HEADER_BYTES + L_TREE_OFFSET, 64n, true);
    new DataView(w.buffer).setBigUint64(HEADER_BYTES + L_TREE_BYTES, 64n, true);
    expect(() => openContainer(reseal(w))).toThrow(/not a manifold layer/);
    refusedOnOpen(edit((dv) => dv.setBigUint64(entry + L_TREE_BYTES, BigInt(bytes * 1000), true)), /misaligned|past the file/);
    refusedOnOpen(edit((dv) => dv.setBigUint64(entry + L_TREE_OFFSET, BigInt(at + 1), true)), /misaligned/);
  });

  it('fuzz: 1,500 random bit flips never crash and never yield a wrong answer', () => {
    const rnd = lcg(52);
    let refused = 0;
    let sound = 0;
    for (let t = 0; t < 1500; t++) {
      const file = base.slice();
      const zone = rnd();
      const pos =
        zone < 0.6 ? at + Math.floor(rnd() * bytes) : zone < 0.75 ? entry + L_ODU_OFFSET + Math.floor(rnd() * 32) : Math.floor(rnd() * file.byteLength);
      file[pos] = file[pos]! ^ (1 << Math.floor(rnd() * 8));
      let c: Container;
      try {
        c = openContainer(file);
      } catch (e) {
        expect(e).toBeInstanceOf(ContainerFormatError);
        refused++;
        continue;
      }
      expect(verifyContainer(file, nodeSha384).ok).toBe(false);
      const tree = c.layer('code')?.tree;
      if (!tree) continue;
      let problems: readonly string[];
      try {
        problems = tree.check();
      } catch (e) {
        expect(e).toBeInstanceOf(ContainerFormatError);
        refused++;
        continue;
      }
      if (pos >= at && pos < at + bytes) expect(problems.length).toBeGreaterThan(0); // every section byte is load-bearing
      if (problems.length > 0) {
        expect(() => tree.candidates([0.4, 0.4, 0.5, 0.5], 0.5)).toThrow(ContainerFormatError);
        refused++;
      } else {
        // The flip landed outside the section and the tree still checks out against this file's own
        // records: it must still contain every hit a scan of those records finds.
        const q: V4 = [0.4, 0.4, 0.5, 0.5];
        try {
          const got = new Set(tree.candidates(q, 0.5));
          for (const p of scanWithin(c, q, 0.5)) expect(got.has(p)).toBe(true);
          sound++;
        } catch (e) {
          // A flipped lens or tier byte makes a record itself undecodable: a refusal too.
          expect(e).toBeInstanceOf(ContainerFormatError);
          refused++;
        }
      }
    }
    expect(sound).toBeGreaterThan(50);
    expect(refused).toBeGreaterThan(1000);
  });
});

// ── the two writers agree ─────────────────────────────────────────────────────

describe('writeContainer emits the same tree as packContainer', () => {
  function memorySink(): ContainerSink & { bytes(): Uint8Array } {
    let buf = new Uint8Array(1 << 16);
    let used = 0;
    return {
      write(b) {
        if (used + b.byteLength > buf.byteLength) {
          const next = new Uint8Array(Math.max(buf.byteLength * 2, used + b.byteLength));
          next.set(buf.subarray(0, used));
          buf = next;
        }
        buf.set(b, used);
        used += b.byteLength;
      },
      patch: async (offset, b) => void buf.set(b, offset),
      bytes: () => buf.slice(0, used),
    };
  }

  it('byte for byte over 60 generated layers, with Odu on and off, mixed with untagged records and other layers', async () => {
    for (let seed = 1; seed <= 60; seed++) {
      const r = lcg(seed * 977);
      const n = Math.floor(r() * 400);
      const pts = clustered(n, 1 + Math.floor(r() * 6), 0.01 + r() * 0.2, seed);
      const tree: boolean | TreeOptions = r() < 0.4 ? true : { leafCapacity: 1 + Math.floor(r() * 12), maxDepth: 1 + Math.floor(r() * 8) };
      const layer = layerOf(pts, tree, { odu: r() < 0.5, kind: r() < 0.5 ? 'code' : 'experience', untagged: Math.floor(r() * 5) }).layers[0]!;
      const world = { kind: 'world' as const, records: [{ payload: enc.encode('w'), at: { x: 1, y: 2, z: 3 } }] };
      const packed = packContainer({ layers: [world, layer] }, nodeSha384);
      const sink = memorySink();
      await writeContainer({ layers: [world, layer] } as never, nodeSha384Incremental, sink);
      expect(Buffer.compare(sink.bytes(), packed)).toBe(0);
      expect(verifyContainer(packed, nodeSha384).ok).toBe(true);
    }
  });
});
