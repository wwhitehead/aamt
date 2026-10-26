// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * The Odu index section (format v1.2, ADR-018): a search structure over a manifold
 * layer, routing each record by the Odu cell of its TERA locator.
 *
 * ## What it is, and what it is not
 *
 * A manifold record's TERA point is a **search key, never an address** — the
 * Vortex VM's one rule. This section keeps the rule: records are still stored in
 * id order and fetched by id. The section is a side table that says, for each of
 * the 256 Odu cells, which records (by their position in the id-sorted record
 * table) have a locator in that cell. A query asks which cells a Fisher–Rao ball
 * can touch ({@link oduCellsWithin}), reads those cells' records, and the caller
 * still checks the real distance of every candidate. The section can only narrow
 * a scan; it never decides where anything lives.
 *
 * ## The cell function is the VM's, and only the VM's
 *
 * A record's cell is `oduIndex(makeReg(tera))` from `@aamt/tera`, applied
 * to the locator's stored `f32` values — the exact numbers a reader decodes, not
 * the `f64` the writer was handed. Two bits per axis (quartiles of `v`, the
 * `Ox` tie-break `v ≥ boundary` rounding up), packed **axis-major, MSB-first**:
 * `cell = q_T·64 + q_E·16 + q_R·4 + q_A`.
 *
 * `libaamt` (WP-23) numbers the same 256 cells differently: its cell is the
 * hexadeca-tree path `(m₁ << 4) | m₂`, where each Meji nibble `m` puts **T at bit
 * 0** (`m = b_T + 2b_E + 4b_R + 8b_A`); the VM's Meji puts T at bit 3. The two
 * quantisers agree on every point; only the packing differs. This section uses
 * the VM's packing, because the container is the VM's storage. The conversion is
 * explicit — {@link libaamtCellToOdu}, {@link libaamtMejiToVm} — and is tested
 * against files written by the real `libaamt` CLI. Nothing converts silently.
 *
 * ## Layout (all little-endian; starts 64-aligned, length a multiple of 64)
 *
 * ```
 *   0  u32      magic        'ODU1'
 *   4  u32      cellFn       1 = vortex-vm oduIndex over the f32 locator (above)
 *   8  u32      entryCount   = records in the layer that carry a TERA point
 *  12  u32      occupied     K = cells holding at least one record
 *  16  u32      recordCount  = the layer's recordCount (a cross-check)
 *  20  u8[12]   reserved (0)
 *  32  u8[32]   occupancy    bit c (byte c>>3, bit c&7) set ⇔ cell c is occupied
 *  64  u32[K+1] starts       starts[0] = 0, strictly increasing, starts[K] = entryCount
 *   …  u32[N]   positions    record positions, grouped by cell (ascending cell),
 *                            strictly ascending within each cell
 *   …  zeros to a 64-byte multiple
 * ```
 *
 * The occupancy mask is WP-23's sparse-children technique (a bitmask plus packed
 * slots) applied to one 256-way level: an empty cell costs one bit.
 */

import {
  AXIS_COUNT,
  MEJI_CELLS,
  ODU_CELLS,
  REFERENCE_TRANSCENDENTALS,
  makeReg,
  oduIndex,
  type Tera,
  type Transcendentals,
} from '@aamt/tera';

/** `'ODU1'` as a little-endian u32. */
export const ODU_SECTION_MAGIC = 0x3155444f;
/** The cell function: vortex-vm `oduIndex` over the locator's stored f32 TERA point. */
export const ODU_CELL_FN_VM = 1;
export const ODU_HEADER_BYTES = 64;
const MASK_AT = 32;

/** Where the locator's TERA lanes and tier sit (layout.ts, manifold locator). */
const TIER_AT = 20;
const TIER_NONE = 0;

/** The TERA point a manifold locator stores, decoded exactly as the reader decodes it. `null` when it has none. */
export function locatorTera(locator: DataView, at = 0): Tera | null {
  if (locator.getUint32(at + TIER_AT, true) === TIER_NONE) return null;
  return [
    locator.getFloat32(at, true),
    locator.getFloat32(at + 4, true),
    locator.getFloat32(at + 8, true),
    locator.getFloat32(at + 12, true),
  ];
}

/** The Odu cell of a TERA point: the VM's `oduIndex`, nothing else. */
export const oduCell = (tera: Tera): number => oduIndex(makeReg(tera));

/** Size of a section holding `entries` records in `occupied` cells. */
export const oduSectionBytes = (occupied: number, entries: number): number =>
  Math.ceil((ODU_HEADER_BYTES + (occupied + 1) * 4 + entries * 4) / 64) * 64;

/**
 * Build the section for a layer whose 32-byte locators are `locators`, in record
 * (id) order. `null` when no record carries a TERA point: there is nothing to route.
 */
export function buildOduSection(locators: readonly Uint8Array[]): Uint8Array | null {
  const byCell: number[][] = Array.from({ length: ODU_CELLS }, () => []);
  let entries = 0;
  locators.forEach((l, pos) => {
    const tera = locatorTera(new DataView(l.buffer, l.byteOffset, l.byteLength));
    if (!tera) return;
    byCell[oduCell(tera)]!.push(pos);
    entries++;
  });
  if (entries === 0) return null;
  const occupied = byCell.filter((c) => c.length > 0).length;
  const out = new Uint8Array(oduSectionBytes(occupied, entries));
  const dv = new DataView(out.buffer);
  dv.setUint32(0, ODU_SECTION_MAGIC, true);
  dv.setUint32(4, ODU_CELL_FN_VM, true);
  dv.setUint32(8, entries, true);
  dv.setUint32(12, occupied, true);
  dv.setUint32(16, locators.length, true);
  let slot = 0;
  let at = ODU_HEADER_BYTES + (occupied + 1) * 4;
  byCell.forEach((positions, c) => {
    if (positions.length === 0) return;
    out[MASK_AT + (c >> 3)] = out[MASK_AT + (c >> 3)]! | (1 << (c & 7));
    dv.setUint32(ODU_HEADER_BYTES + slot * 4, (at - ODU_HEADER_BYTES - (occupied + 1) * 4) / 4, true);
    slot++;
    for (const p of positions) {
      dv.setUint32(at, p, true);
      at += 4;
    }
  });
  dv.setUint32(ODU_HEADER_BYTES + occupied * 4, entries, true);
  return out;
}

/** A structurally valid section: what a reader can use without walking the layer. */
export interface OduSectionShape {
  readonly entryCount: number;
  readonly occupied: number;
  /** `cellStart[c] .. cellStart[c+1]` indexes `positions` for cell `c`. 257 entries. */
  readonly cellStart: Uint32Array;
  /** Entry `i` of the positions array: a record's position in the id-sorted record table. */
  readonly positions: (i: number) => number;
}

/**
 * Check a section's structure — header, mask, slot table, bounds — in O(cells),
 * independent of how many records it routes. Returns a problem string on failure.
 * Whether each position really belongs in its cell is {@link checkOduSemantics}.
 */
export function parseOduSection(
  section: Uint8Array,
  recordCount: number,
): { ok: true; shape: OduSectionShape } | { ok: false; problem: string } {
  const bad = (problem: string) => ({ ok: false as const, problem: `Odu index ${problem}` });
  if (section.byteLength < ODU_HEADER_BYTES + 4 || section.byteLength % 64 !== 0)
    return bad(`section is ${section.byteLength} bytes; too short or not a multiple of 64.`);
  const dv = new DataView(section.buffer, section.byteOffset, section.byteLength);
  if (dv.getUint32(0, true) !== ODU_SECTION_MAGIC) return bad('has a bad magic.');
  const cellFn = dv.getUint32(4, true);
  if (cellFn !== ODU_CELL_FN_VM)
    return bad(`names cell function ${cellFn}; only ${ODU_CELL_FN_VM} is defined.`);
  const entryCount = dv.getUint32(8, true);
  const occupied = dv.getUint32(12, true);
  if (dv.getUint32(16, true) !== recordCount)
    return bad(`was built for ${dv.getUint32(16, true)} records; the layer has ${recordCount}.`);
  for (let i = 20; i < MASK_AT; i++) if (section[i] !== 0) return bad('has non-zero reserved bytes.');
  if (entryCount === 0 || entryCount > recordCount)
    return bad(`routes ${entryCount} records in a layer of ${recordCount}.`);
  let maskBits = 0;
  for (let i = 0; i < 32; i++) maskBits += popcount8(section[MASK_AT + i]!);
  if (occupied !== maskBits || occupied === 0 || occupied > entryCount)
    return bad(`claims ${occupied} occupied cells; its mask has ${maskBits}.`);
  if (section.byteLength !== oduSectionBytes(occupied, entryCount))
    return bad(
      `is ${section.byteLength} bytes; ${occupied} cells and ${entryCount} records need ${oduSectionBytes(occupied, entryCount)}.`,
    );
  const posAt = ODU_HEADER_BYTES + (occupied + 1) * 4;
  for (let i = posAt + entryCount * 4; i < section.byteLength; i++)
    if (section[i] !== 0) return bad('has non-zero padding.');

  const cellStart = new Uint32Array(ODU_CELLS + 1);
  let slot = 0;
  let prev = dv.getUint32(ODU_HEADER_BYTES, true);
  if (prev !== 0) return bad('slot table does not start at 0.');
  for (let c = 0; c < ODU_CELLS; c++) {
    cellStart[c] = prev;
    if ((section[MASK_AT + (c >> 3)]! >> (c & 7)) & 1) {
      const next = dv.getUint32(ODU_HEADER_BYTES + (slot + 1) * 4, true);
      if (next <= prev) return bad(`slot table is not strictly increasing at cell ${c}.`);
      prev = next;
      slot++;
    }
  }
  if (prev !== entryCount) return bad(`slot table ends at ${prev}, not at ${entryCount}.`);
  cellStart[ODU_CELLS] = entryCount;
  // DataView rather than a Uint32Array view: little-endian on every host.
  const positions = (i: number): number => dv.getUint32(posAt + i * 4, true);
  return { ok: true, shape: { entryCount, occupied, cellStart, positions } };
}

function popcount8(b: number): number {
  let n = 0;
  for (let x = b; x; x &= x - 1) n++;
  return n;
}

/**
 * Check that the section routes exactly the layer's located records, each in the
 * cell its own locator maps to: every position in range and strictly ascending
 * within its cell, every indexed record carrying a TERA point whose VM Odu cell
 * is the claimed one, and as many entries as located records — so none is
 * missing. O(records). `locatorAt(p)` is the DataView offset of record `p`'s
 * locator in `dv`.
 */
export function checkOduSemantics(
  shape: OduSectionShape,
  recordCount: number,
  dv: DataView,
  locatorAt: (position: number) => number,
): string[] {
  const problems: string[] = [];
  for (let c = 0; c < ODU_CELLS && problems.length < 8; c++) {
    let last = -1;
    for (let i = shape.cellStart[c]!; i < shape.cellStart[c + 1]!; i++) {
      const p = shape.positions(i);
      if (p >= recordCount || p <= last) {
        problems.push(`Odu index cell ${c}: position ${p} is out of range or out of order.`);
        break;
      }
      last = p;
      const tera = locatorTera(dv, locatorAt(p));
      if (!tera) {
        problems.push(`Odu index cell ${c}: record ${p} has no TERA point.`);
        continue;
      }
      const actual = oduCell(tera);
      if (actual !== c) problems.push(`Odu index cell ${c}: record ${p} belongs in cell ${actual}.`);
    }
  }
  let located = 0;
  for (let p = 0; p < recordCount; p++)
    if (dv.getUint32(locatorAt(p) + TIER_AT, true) !== TIER_NONE) located++;
  if (located !== shape.entryCount)
    problems.push(
      `Odu index routes ${shape.entryCount} records; the layer has ${located} with a TERA point.`,
    );
  return problems;
}

// ── Which cells a Fisher–Rao ball can touch ───────────────────────────────────

/**
 * Every Odu cell that can hold a point within Fisher–Rao distance `radius` of
 * `v` — the VM's `frDist`, `‖θ(u) − θ(v)‖₂` with `θ = 2·arcsin√v`.
 *
 * In `θ` each axis's four quartile cells are intervals with inner edges
 * `θ(¼), θ(½), θ(¾)`, computed with the same transcendental port the distance
 * uses. A cell is skipped only when the distance from the query to its box —
 * per axis, how far `θ_q` lies outside the interval; summed in squares in axis
 * order, exactly as `frDist` sums — exceeds `radius`.
 *
 * **No point within the radius is lost**, and the argument is exact, not a
 * tolerance: a record in cell `c` has `θ_d` inside the cell's interval on every
 * axis (the port's `toTheta` is non-decreasing on `[0,1]`, tested), so each of
 * its per-axis gaps to the query is at least the box gap; IEEE subtraction,
 * squaring, addition and `sqrt` are monotone; hence `frDist ≥ boxDist`, and a
 * skipped cell holds only records the scan would also reject (`distance > r`).
 * The outermost edges are open (−∞, +∞), so a clamped or out-of-range lane is
 * still covered.
 */
export function oduCellsWithin(
  v: Tera,
  radius: number,
  tr: Transcendentals = REFERENCE_TRANSCENDENTALS,
): number[] {
  const theta = makeReg(v, tr).theta;
  const edges = [-Infinity, tr.toTheta(0.25), tr.toTheta(0.5), tr.toTheta(0.75), Infinity];
  // gap[d][q]: how far θ_d lies outside quartile q's interval on axis d.
  const gap: number[][] = [];
  for (let d = 0; d < AXIS_COUNT; d++) {
    const t = theta[d]!;
    gap.push(
      [0, 1, 2, 3].map((q) => {
        const lo = edges[q]!;
        const hi = edges[q + 1]!;
        return t < lo ? lo - t : t > hi ? t - hi : 0;
      }),
    );
  }
  const out: number[] = [];
  for (let c = 0; c < ODU_CELLS; c++) {
    let sum = 0;
    for (let d = 0; d < AXIS_COUNT; d++) {
      const g = gap[d]![(c >> (2 * (AXIS_COUNT - 1 - d))) & 3]!;
      sum += g * g;
    }
    if (!(Math.sqrt(sum) > radius)) out.push(c);
  }
  return out;
}

// ── libaamt's numbering, converted explicitly ─────────────────────────────────

/** A libaamt Meji nibble (T at bit 0) as the VM's Meji index (T at bit 3): a 4-bit reversal. */
export function libaamtMejiToVm(m: number): number {
  if (!Number.isInteger(m) || m < 0 || m >= MEJI_CELLS)
    throw new RangeError(`Meji nibble ${m} is not in 0..15.`);
  return ((m & 1) << 3) | ((m & 2) << 1) | ((m & 4) >> 1) | ((m & 8) >> 3);
}

/**
 * A libaamt Odu cell — the depth-2 tree path `(m₁ << 4) | m₂`, T at bit 0 of each
 * nibble — as the VM's axis-major `oduIndex`. Axis `d`'s quartile is
 * `2·bit_d(m₁) + bit_d(m₂)`: the first split picks the half, the second the quarter.
 */
export function libaamtCellToOdu(treeCell: number): number {
  if (!Number.isInteger(treeCell) || treeCell < 0 || treeCell >= ODU_CELLS)
    throw new RangeError(`Odu tree cell ${treeCell} is not in 0..255.`);
  const m1 = treeCell >> 4;
  const m2 = treeCell & 15;
  let cell = 0;
  for (let d = 0; d < AXIS_COUNT; d++) {
    const q = (((m1 >> d) & 1) << 1) | ((m2 >> d) & 1);
    cell |= q << (2 * (AXIS_COUNT - 1 - d));
  }
  return cell;
}
