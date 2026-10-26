// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * The adaptive Meji tree section (format v1.3, ADR-023): WP-23's hexadeca-tree as a search
 * structure over a manifold layer, deeper where the data is dense.
 *
 * ## What it adds to the Odu index
 *
 * The Odu index (`odu.ts`) cuts every axis into quartiles: 256 cells, always. Data that
 * clusters lands in a few of them, and a query that touches such a cell reads all of it. This
 * tree starts from the whole box and splits a node only while it holds more than
 * `leafCapacity` records, one bit per axis at a time — the 16 children of a node are the 16
 * Meji cells of its box — down to `maxDepth` (WP-23's limit is 8). Sparse regions stay
 * shallow; a dense cluster is cut as finely as it needs. Like the Odu index it is a **search
 * table, never an address**: records stay in id order and are found by id, and the caller still
 * checks the real distance of every candidate.
 *
 * ## The cell function: the Odu index's, one level at a time
 *
 * At depth `d` axis `a`'s coordinate is `c = min(2ᵈ − 1, ⌊v·2ᵈ⌋)` (and `0` for anything that is
 * not `≥ 0`), taken from the locator's stored `f32` — the number a reader decodes. Scaling by a
 * power of two is exact, so the cell is the half-open box `[c/2ᵈ, (c+1)/2ᵈ)` and a child's box
 * lies inside its parent's. A node's child number is `8b_T + 4b_E + 2b_R + b_A`, `b_a` the next
 * bit of axis `a`'s coordinate: the VM's `mejiIndex` packing (T at bit 3), **not** libaamt's
 * (T at bit 0; see `libaamtMejiToVm`). Two levels down this is exactly the VM's `oduIndex`:
 * `q = 2·b(level 1) + b(level 2)` per axis, packed axis-major (tested over millions of points,
 * the quartile edges, and a real libaamt file).
 *
 * ## Layout (little-endian; starts 64-aligned, length a multiple of 64)
 *
 * ```
 *   0  u32      magic        'ATR1'
 *   4  u32      cellFn       1 = the quantiser above, over the f32 locator
 *   8  u32      entryCount   = records in the layer that carry a TERA point
 *  12  u32      nodeCount    N
 *  16  u32      recordCount  = the layer's recordCount (a cross-check)
 *  20  u32      leafCapacity B ≥ 1: a node splits only while it holds more than B
 *  24  u32      maxDepth     D in 1..8: a node at depth D never splits
 *  28  u32      deepest      the deepest level any node reaches
 *  32  u32      leafCount
 *  36  u8[28]   reserved (0)
 *  64  node[N]  12 bytes each, breadth-first; a node's children are contiguous, ascending:
 *                 u16 mask   internal: which of the 16 children exist (non-zero); leaf: 0
 *                 u8  depth  root = 0
 *                 u8  flags  0 internal, 1 leaf
 *                 u32 a      internal: index of the first child · leaf: start in positions
 *                 u32 b      internal: records beneath · leaf: count (≥ 1)
 *   …  u32[E]   positions    record positions in the id-sorted record table; leaves in node
 *                            order, strictly ascending within each leaf
 *   …  zeros to a 64-byte multiple
 * ```
 *
 * Child `k` of a node is at `a + popcount(mask & ((1 << k) − 1))`: WP-23's sparse-children
 * technique, a bitmask and packed slots. The shape is **canonical**: for given records, `B` and
 * `D` there is exactly one valid tree (a node is internal iff it holds more than `B` records
 * and sits above depth `D`), so the reader checks that rule and two writers cannot disagree.
 */

import {
  AXIS_COUNT,
  REFERENCE_TRANSCENDENTALS,
  makeReg,
  type Tera,
  type Transcendentals,
} from '@aamt/tera';
import { locatorTera } from './odu';

/** `'ATR1'` as a little-endian u32. */
export const TREE_SECTION_MAGIC = 0x31525441;
/** The cell function: `min(2ᵈ − 1, ⌊v·2ᵈ⌋)` per axis over the locator's stored f32 TERA point. */
export const TREE_CELL_FN = 1;
export const TREE_HEADER_BYTES = 64;
export const TREE_NODE_BYTES = 12;
/** WP-23's deepest level. */
export const TREE_MAX_DEPTH = 8;
export const TREE_DEFAULT_LEAF_CAPACITY = 32;
const TREE_RESERVED_AT = 36;
const FLAG_LEAF = 1;

export interface TreeOptions {
  /** A node splits only while it holds more than this many records. Default {@link TREE_DEFAULT_LEAF_CAPACITY}. */
  readonly leafCapacity?: number;
  /** A node at this depth never splits, however many records it holds. 1–8, default 8. */
  readonly maxDepth?: number;
}

/** A section's size in bytes: its nodes and its entries, rounded up to 64. */
export const treeSectionBytes = (nodes: number, entries: number): number =>
  Math.ceil((TREE_HEADER_BYTES + nodes * TREE_NODE_BYTES + entries * 4) / 64) * 64;

/** An axis coordinate at `depth`: `min(2ᵈ − 1, ⌊x·2ᵈ⌋)`, and `0` for a value that is not `≥ 0`. */
export function treeCoord(x: number, depth: number): number {
  if (!(x >= 0)) return 0;
  const n = 2 ** depth;
  const c = Math.floor(x * n) || 0; // `|| 0`: ⌊−0⌋ is −0, and a coordinate is a plain 0
  return c > n - 1 ? n - 1 : c;
}

/** The VM-Meji child number of a point's cell at level `level` (1..8): one bit per axis. */
export function treeChild(tera: Tera, level: number): number {
  let k = 0;
  for (let d = 0; d < AXIS_COUNT; d++) k |= (treeCoord(tera[d]!, level) & 1) << (AXIS_COUNT - 1 - d);
  return k;
}

const popcount = (x: number): number => {
  let n = 0;
  for (let y = x; y; y &= y - 1) n++;
  return n;
};

/** Check builder options; returns them filled in. */
export function resolveTreeOptions(o: TreeOptions | undefined): { leafCapacity: number; maxDepth: number } {
  const leafCapacity = o?.leafCapacity ?? TREE_DEFAULT_LEAF_CAPACITY;
  const maxDepth = o?.maxDepth ?? TREE_MAX_DEPTH;
  if (!Number.isInteger(leafCapacity) || leafCapacity < 1 || leafCapacity > 0xffffffff)
    throw new RangeError(`aamt-container: leafCapacity must be a positive integer, got ${leafCapacity}.`);
  if (!Number.isInteger(maxDepth) || maxDepth < 1 || maxDepth > TREE_MAX_DEPTH)
    throw new RangeError(`aamt-container: maxDepth must be an integer in 1..${TREE_MAX_DEPTH}, got ${maxDepth}.`);
  return { leafCapacity, maxDepth };
}

/**
 * Build the section for a layer whose 32-byte locators are `locators`, in record (id) order.
 * `null` when no record carries a TERA point. Deterministic: same locators and options, same bytes.
 */
export function buildTreeSection(
  locators: readonly Uint8Array[],
  options?: TreeOptions,
): Uint8Array | null {
  const { leafCapacity, maxDepth } = resolveTreeOptions(options);
  // Each located record, with its path: the Meji child number at every level, high level first.
  const pos: number[] = [];
  const word: number[] = [];
  locators.forEach((l, p) => {
    const tera = locatorTera(new DataView(l.buffer, l.byteOffset, l.byteLength));
    if (!tera) return;
    let w = 0;
    for (let level = 1; level <= TREE_MAX_DEPTH; level++) w = w * 16 + treeChild(tera, level);
    pos.push(p);
    word.push(w);
  });
  const entries = pos.length;
  if (entries === 0) return null;
  // Sorted by path, then position: every node's records are one contiguous run.
  const order = Array.from({ length: entries }, (_, i) => i).sort(
    (i, j) => word[i]! - word[j]! || pos[i]! - pos[j]!,
  );
  const nibble = (w: number, level: number): number =>
    Math.floor(w / 16 ** (TREE_MAX_DEPTH - level)) % 16;

  interface Pending {
    lo: number;
    hi: number;
    depth: number;
  }
  const queue: Pending[] = [{ lo: 0, hi: entries, depth: 0 }];
  const nodes: { mask: number; depth: number; leaf: boolean; a: number; b: number }[] = [];
  const positions: number[] = [];
  let deepest = 0;
  let leaves = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const { lo, hi, depth } = queue[qi]!;
    const count = hi - lo;
    deepest = Math.max(deepest, depth);
    if (count <= leafCapacity || depth === maxDepth) {
      nodes.push({ mask: 0, depth, leaf: true, a: positions.length, b: count });
      // The run is ordered by path; a leaf lists its records in record order.
      const inLeaf: number[] = [];
      for (let i = lo; i < hi; i++) inLeaf.push(pos[order[i]!]!);
      inLeaf.sort((x, y) => x - y);
      for (const p of inLeaf) positions.push(p);
      leaves++;
      continue;
    }
    let mask = 0;
    const first = queue.length;
    let from = lo;
    while (from < hi) {
      const k = nibble(word[order[from]!]!, depth + 1);
      let to = from + 1;
      while (to < hi && nibble(word[order[to]!]!, depth + 1) === k) to++;
      mask |= 1 << k;
      queue.push({ lo: from, hi: to, depth: depth + 1 });
      from = to;
    }
    nodes.push({ mask, depth, leaf: false, a: first, b: count });
  }

  const out = new Uint8Array(treeSectionBytes(nodes.length, entries));
  const dv = new DataView(out.buffer);
  dv.setUint32(0, TREE_SECTION_MAGIC, true);
  dv.setUint32(4, TREE_CELL_FN, true);
  dv.setUint32(8, entries, true);
  dv.setUint32(12, nodes.length, true);
  dv.setUint32(16, locators.length, true);
  dv.setUint32(20, leafCapacity, true);
  dv.setUint32(24, maxDepth, true);
  dv.setUint32(28, deepest, true);
  dv.setUint32(32, leaves, true);
  nodes.forEach((n, i) => {
    const at = TREE_HEADER_BYTES + i * TREE_NODE_BYTES;
    dv.setUint16(at, n.mask, true);
    out[at + 2] = n.depth;
    out[at + 3] = n.leaf ? FLAG_LEAF : 0;
    dv.setUint32(at + 4, n.a, true);
    dv.setUint32(at + 8, n.b, true);
  });
  const posAt = TREE_HEADER_BYTES + nodes.length * TREE_NODE_BYTES;
  positions.forEach((p, i) => dv.setUint32(posAt + i * 4, p, true));
  return out;
}

/** A structurally valid section: what a reader can use without walking the layer. */
export interface TreeSectionShape {
  readonly entryCount: number;
  readonly nodeCount: number;
  readonly leafCount: number;
  readonly leafCapacity: number;
  readonly maxDepth: number;
  readonly deepest: number;
  /** Node accessors (index in breadth-first order). */
  readonly mask: (n: number) => number;
  readonly depth: (n: number) => number;
  readonly isLeaf: (n: number) => boolean;
  /** Internal: first child's index. Leaf: start offset into positions. */
  readonly a: (n: number) => number;
  /** Internal: records beneath. Leaf: count. */
  readonly b: (n: number) => number;
  /** Entry `i` of the positions array. */
  readonly positions: (i: number) => number;
}

/**
 * Check a section's structure — header, every node, the canonical split rule, the counts — in
 * O(nodes), independent of how many records it routes. Whether each position really belongs in
 * its leaf is {@link checkTreeSemantics}. Returns a problem string on failure.
 */
export function parseTreeSection(
  section: Uint8Array,
  recordCount: number,
): { ok: true; shape: TreeSectionShape } | { ok: false; problem: string } {
  const bad = (problem: string) => ({ ok: false as const, problem: `tree index ${problem}` });
  if (section.byteLength < TREE_HEADER_BYTES + TREE_NODE_BYTES || section.byteLength % 64 !== 0)
    return bad(`section is ${section.byteLength} bytes; too short or not a multiple of 64.`);
  const dv = new DataView(section.buffer, section.byteOffset, section.byteLength);
  if (dv.getUint32(0, true) !== TREE_SECTION_MAGIC) return bad('has a bad magic.');
  const cellFn = dv.getUint32(4, true);
  if (cellFn !== TREE_CELL_FN) return bad(`names cell function ${cellFn}; only ${TREE_CELL_FN} is defined.`);
  const entryCount = dv.getUint32(8, true);
  const nodeCount = dv.getUint32(12, true);
  if (dv.getUint32(16, true) !== recordCount)
    return bad(`was built for ${dv.getUint32(16, true)} records; the layer has ${recordCount}.`);
  const leafCapacity = dv.getUint32(20, true);
  const maxDepth = dv.getUint32(24, true);
  const deepest = dv.getUint32(28, true);
  const leafCount = dv.getUint32(32, true);
  for (let i = TREE_RESERVED_AT; i < TREE_HEADER_BYTES; i++) if (section[i] !== 0) return bad('has non-zero reserved bytes.');
  if (entryCount === 0 || entryCount > recordCount)
    return bad(`routes ${entryCount} records in a layer of ${recordCount}.`);
  if (leafCapacity < 1) return bad('has a leaf capacity of 0.');
  if (maxDepth < 1 || maxDepth > TREE_MAX_DEPTH) return bad(`has a maximum depth of ${maxDepth}.`);
  if (deepest > maxDepth) return bad(`reaches depth ${deepest}, beyond its maximum ${maxDepth}.`);
  if (nodeCount < 1 || nodeCount > entryCount * (TREE_MAX_DEPTH + 1))
    return bad(`claims ${nodeCount} nodes for ${entryCount} records.`);
  if (section.byteLength !== treeSectionBytes(nodeCount, entryCount))
    return bad(
      `is ${section.byteLength} bytes; ${nodeCount} nodes and ${entryCount} records need ${treeSectionBytes(nodeCount, entryCount)}.`,
    );
  const nodeAt = (n: number): number => TREE_HEADER_BYTES + n * TREE_NODE_BYTES;
  const posAt = nodeAt(nodeCount);
  for (let i = posAt + entryCount * 4; i < section.byteLength; i++)
    if (section[i] !== 0) return bad('has non-zero padding.');

  const mask = (n: number): number => dv.getUint16(nodeAt(n), true);
  const depth = (n: number): number => section[nodeAt(n) + 2]!;
  const flags = (n: number): number => section[nodeAt(n) + 3]!;
  const a = (n: number): number => dv.getUint32(nodeAt(n) + 4, true);
  const b = (n: number): number => dv.getUint32(nodeAt(n) + 8, true);

  const expectDepth = new Uint8Array(nodeCount); // set by each parent for its children
  let nextChild = 1;
  let leafCursor = 0;
  let leaves = 0;
  let seenDeepest = 0;
  for (let n = 0; n < nodeCount; n++) {
    if (depth(n) !== expectDepth[n]) return bad(`node ${n} is at depth ${depth(n)}, expected ${expectDepth[n]}.`);
    seenDeepest = Math.max(seenDeepest, depth(n));
    const f = flags(n);
    if (f === FLAG_LEAF) {
      if (mask(n) !== 0) return bad(`leaf ${n} has a child mask.`);
      if (b(n) < 1) return bad(`leaf ${n} is empty.`);
      if (a(n) !== leafCursor) return bad(`leaf ${n} starts at ${a(n)}, expected ${leafCursor}.`);
      // The split rule: only a node at the depth limit may hold more than the capacity.
      if (depth(n) < maxDepth && b(n) > leafCapacity)
        return bad(`leaf ${n} holds ${b(n)} records above the depth limit; it should have split (capacity ${leafCapacity}).`);
      leafCursor += b(n);
      leaves++;
    } else if (f === 0) {
      const pc = popcount(mask(n));
      if (pc === 0) return bad(`node ${n} is internal with no children.`);
      if (depth(n) >= maxDepth) return bad(`node ${n} splits at the depth limit ${maxDepth}.`);
      if (b(n) <= leafCapacity) return bad(`node ${n} split with only ${b(n)} records (capacity ${leafCapacity}).`);
      if (a(n) !== nextChild) return bad(`node ${n}'s children start at ${a(n)}, expected ${nextChild}.`);
      if (nextChild + pc > nodeCount) return bad(`node ${n}'s children run past the node table.`);
      for (let c = 0; c < pc; c++) expectDepth[nextChild + c] = depth(n) + 1;
      nextChild += pc;
    } else return bad(`node ${n} has unknown flags ${f}.`);
  }
  if (nextChild !== nodeCount) return bad(`has ${nodeCount} nodes but its children account for ${nextChild}.`);
  if (leafCursor !== entryCount) return bad(`its leaves hold ${leafCursor} records, not ${entryCount}.`);
  if (leaves !== leafCount) return bad(`claims ${leafCount} leaves; it has ${leaves}.`);
  if (seenDeepest !== deepest) return bad(`claims depth ${deepest}; the deepest node is at ${seenDeepest}.`);
  // Counts beneath each internal node must add up (children are after their parent: sum backwards).
  for (let n = nodeCount - 1; n >= 0; n--) {
    if (flags(n) === FLAG_LEAF) continue;
    let sum = 0;
    const pc = popcount(mask(n));
    for (let c = 0; c < pc; c++) sum += b(a(n) + c);
    if (sum !== b(n)) return bad(`node ${n} claims ${b(n)} records beneath it; its children hold ${sum}.`);
  }
  if (b(0) !== entryCount) return bad(`its root holds ${b(0)} records, not ${entryCount}.`);
  const positions = (i: number): number => dv.getUint32(posAt + i * 4, true);
  return {
    ok: true,
    shape: {
      entryCount,
      nodeCount,
      leafCount,
      leafCapacity,
      maxDepth,
      deepest,
      mask,
      depth,
      isLeaf: (n) => flags(n) === FLAG_LEAF,
      a,
      b,
      positions,
    },
  };
}

/** Per-node cell coordinates, four per node (axis-major), derived from the masks. */
function nodeCoords(shape: TreeSectionShape): Uint8Array {
  const coords = new Uint8Array(shape.nodeCount * AXIS_COUNT);
  for (let n = 0; n < shape.nodeCount; n++) {
    if (shape.isLeaf(n)) continue;
    const m = shape.mask(n);
    let slot = 0;
    for (let k = 0; k < 16; k++) {
      if (!((m >> k) & 1)) continue;
      const child = shape.a(n) + slot++;
      for (let d = 0; d < AXIS_COUNT; d++)
        coords[child * AXIS_COUNT + d] =
          (coords[n * AXIS_COUNT + d]! << 1) | ((k >> (AXIS_COUNT - 1 - d)) & 1);
    }
  }
  return coords;
}

const TIER_AT = 20;
const TIER_NONE = 0;

/**
 * Check that the section routes exactly the layer's located records, each into the leaf whose
 * box its own locator falls in: every position in range and strictly ascending within its
 * leaf, every record carrying a TERA point whose cell at the leaf's depth is the leaf's, and as
 * many entries as located records — so none is missing. O(records). `locatorAt(p)` is the
 * DataView offset of record `p`'s locator in `dv`.
 */
export function checkTreeSemantics(
  shape: TreeSectionShape,
  recordCount: number,
  dv: DataView,
  locatorAt: (position: number) => number,
): string[] {
  const problems: string[] = [];
  const coords = nodeCoords(shape);
  for (let n = 0; n < shape.nodeCount && problems.length < 8; n++) {
    if (!shape.isLeaf(n)) continue;
    const depth = shape.depth(n);
    let last = -1;
    for (let i = shape.a(n); i < shape.a(n) + shape.b(n); i++) {
      const p = shape.positions(i);
      if (p >= recordCount || p <= last) {
        problems.push(`tree leaf ${n}: position ${p} is out of range or out of order.`);
        break;
      }
      last = p;
      const tera = locatorTera(dv, locatorAt(p));
      if (!tera) {
        problems.push(`tree leaf ${n}: record ${p} has no TERA point.`);
        continue;
      }
      for (let d = 0; d < AXIS_COUNT; d++) {
        if (treeCoord(tera[d]!, depth) !== coords[n * AXIS_COUNT + d]) {
          problems.push(`tree leaf ${n}: record ${p} belongs in another cell at depth ${depth}.`);
          break;
        }
      }
    }
  }
  let located = 0;
  for (let p = 0; p < recordCount; p++)
    if (dv.getUint32(locatorAt(p) + TIER_AT, true) !== TIER_NONE) located++;
  if (located !== shape.entryCount)
    problems.push(`tree index routes ${shape.entryCount} records; the layer has ${located} with a TERA point.`);
  return problems;
}

// ── Which nodes a Fisher–Rao ball can touch ───────────────────────────────────

export interface TreeSearchStats {
  /** Nodes whose box was tested against the ball. */
  readonly visited: number;
  /** Leaves whose records were returned. */
  readonly leaves: number;
}

/**
 * Positions of every record in a leaf that a Fisher–Rao ball of `radius` around `v` can touch,
 * ascending: a superset of the records within `radius`.
 *
 * Same argument as `oduCellsWithin`, per node instead of per cell. A node at depth `d` with
 * coordinate `c` on an axis covers `v ∈ [c/2ᵈ, (c+1)/2ᵈ)`, so `θ(v)` lies in
 * `[θ(c/2ᵈ), θ((c+1)/2ᵈ)]` (the port's `toTheta` is non-decreasing); the outermost edges are
 * open, so a clamped or out-of-range lane is still covered. The distance from the query's `θ` to
 * that box — per axis how far `θ_q` lies outside the interval, summed in squares in axis order as
 * `frDist` sums — is a lower bound on `frDist` to every record beneath the node (IEEE
 * subtraction, squaring, addition and `sqrt` are monotone). A node is skipped only when that
 * bound exceeds `radius`, and then so is every record under it, so the scan would reject them too.
 */
export function treeCandidates(
  shape: TreeSectionShape,
  v: Tera,
  radius: number,
  tr: Transcendentals = REFERENCE_TRANSCENDENTALS,
): { positions: Uint32Array; stats: TreeSearchStats } {
  const theta = makeReg(v, tr).theta;
  // edge[d][i]: θ(i / 2ᵈ), computed on first use with the port the distance will be measured with.
  const edge: (Float64Array | undefined)[] = [];
  const edgeTheta = (depth: number, i: number): number => {
    let row = edge[depth];
    if (!row) {
      row = new Float64Array(2 ** depth + 1).fill(NaN);
      edge[depth] = row;
    }
    let t = row[i]!;
    if (t !== t) t = row[i] = tr.toTheta(i / 2 ** depth);
    return t;
  };
  const found: number[] = [];
  let visited = 0;
  let leaves = 0;
  // Depth-first with an explicit stack: (node, its four coordinates).
  const stack: { n: number; c: number[] }[] = [{ n: 0, c: [0, 0, 0, 0] }];
  while (stack.length > 0) {
    const { n, c: coord } = stack.pop()!;
    visited++;
    const depth = shape.depth(n);
    const top = 2 ** depth - 1;
    let sum = 0;
    for (let d = 0; d < AXIS_COUNT; d++) {
      const cd = coord[d]!;
      const lo = cd === 0 ? -Infinity : edgeTheta(depth, cd);
      const hi = cd === top ? Infinity : edgeTheta(depth, cd + 1);
      const t = theta[d]!;
      const g = t < lo ? lo - t : t > hi ? t - hi : 0;
      sum += g * g;
    }
    if (Math.sqrt(sum) > radius) continue;
    if (shape.isLeaf(n)) {
      leaves++;
      for (let i = shape.a(n); i < shape.a(n) + shape.b(n); i++) found.push(shape.positions(i));
      continue;
    }
    const m = shape.mask(n);
    let slot = 0;
    // Pushed high-to-low so the lowest child is visited first; order does not affect the result.
    const kids: { n: number; c: number[] }[] = [];
    for (let k = 0; k < 16; k++) {
      if (!((m >> k) & 1)) continue;
      kids.push({
        n: shape.a(n) + slot++,
        c: coord.map((x, d) => (x << 1) | ((k >> (AXIS_COUNT - 1 - d)) & 1)),
      });
    }
    for (let i = kids.length - 1; i >= 0; i--) stack.push(kids[i]!);
  }
  return { positions: Uint32Array.from(found).sort(), stats: { visited, leaves } };
}
