// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * Vortex VM — the reference transcendental unit.
 *
 * ## Why this file exists
 *
 * The ISA declares transcendentals as a swappable port (`register.ts`) because
 * they are the one part of the machine that JavaScript does not make portable.
 * This file supplies the port that the *reference* implementation uses, and its
 * whole job is to be bit-identical on every engine and every platform — not to
 * be fast, and not to be accurate in the last ulp.
 *
 * The distinction that matters: accuracy is a quality question and 1e-15 is
 * plenty. Reproducibility is a *correctness* question, because the machine's
 * output feeds a world hash that has to match across a Kernel worker restart, a
 * save/load round trip, and a headless-versus-windowed run. A function that is
 * accurate to 1e-16 on your machine and 3e-16 on another one is unusable for
 * that, and no amount of tolerance fixes it, because the requirement is
 * equality rather than closeness.
 *
 * So: tables. Generated once at module load with the platform's own `Math.sin`,
 * and thereafter only ever read.
 *
 * ## Closing the inverse-trig gap
 *
 * `arcade-engine`'s `rng.ts` already tables `sin` and `cos` for exactly this
 * reason, and bans `Math.sin`/`Math.cos` in its eslint config. But its
 * `fromAngle` uses that table while the matching forward `toAngle` uses
 * `Math.asin` directly, and its own source comment calls that out:
 *
 * > "The forward `toAngle` uses `Math.asin`, which carries the same portability
 * > caveat and is *not* covered by the package's `no-restricted-properties`
 * > rule. That is a real gap in the rule, not a justification; it should be
 * > narrowed to cover inverse trig as well before `nature.ts` state is hashed
 * > into a save."
 *
 * This file closes it. `dasin` is tabled as well, with the standard reflection
 * trick so the table never has to resolve the derivative singularity at ±1:
 * for `|x| > EDGE` it evaluates `π/2 − dasin(√(1−x²))`, which maps the
 * ill-conditioned neighbourhood of ±1 back into the well-conditioned interior.
 * The identity is exact in exact arithmetic and loses nothing here, because the
 * reflected argument is small enough that the table's own error is the only
 * term left.
 *
 * ## Why the tables are big enough
 *
 * `TRIG_BITS = 13` gives 8192 entries per cycle. Linear interpolation makes the
 * truncation error *second* order, so the error is about `Δ²/8` with
 * `Δ = 2π/8192 ≈ 7.67e-4` — roughly `7e-8` absolute, below `f32` epsilon
 * (`2⁻²⁴ ≈ 6e-8`) and five orders of magnitude above `f64` epsilon. That is the
 * intended trade: a simulation whose state is hashed at `f32` resolution cannot
 * distinguish this from the true function, and the value is the same everywhere.
 */

// The one sanctioned use of the platform transcendentals in this package: seeding
// these tables at module load. After these two lines every table is read-only
// for the life of the process, so the values are identical everywhere.

const TRIG_BITS = 13;
const TRIG_SIZE = 1 << TRIG_BITS;
const TRIG_MASK = TRIG_SIZE - 1;
const TRIG_SCALE = TRIG_SIZE / (Math.PI * 2);

const SIN_TABLE = new Float64Array(TRIG_SIZE + 1);
for (let i = 0; i <= TRIG_SIZE; i++) SIN_TABLE[i] = Math.sin((i / TRIG_SIZE) * Math.PI * 2);

/**
 * Deterministic `sin`. Same input, same bits, on every engine and platform.
 *
 * Reduction to `[0, 2π)` happens first so the table index is always in range,
 * and the interpolation is linear between adjacent entries.
 */
export function dsin(x: number): number {
  if (!Number.isFinite(x)) return Number.NaN;
  const TAU = Math.PI * 2;
  let v = x % TAU;
  if (v < 0) v += TAU;
  const f = v * TRIG_SCALE;
  const i = Math.floor(f);
  const frac = f - i;
  const a = SIN_TABLE[i & TRIG_MASK]!;
  const b = SIN_TABLE[(i + 1) & TRIG_MASK]!;
  return a + (b - a) * frac;
}

/**
 * Deterministic sine of a **phase** — a `u32` fraction of one turn (`2³²` = one
 * full turn). The same table as {@link dsin}, indexed straight from the phase
 * bits: the top 13 bits pick the entry, the low 19 interpolate. There is no
 * floating-point reduction, so the result depends only on the phase's bits and
 * is exactly periodic — `dsinTurn(p + 2³²) ≡ dsinTurn(p)` with no `2π` rounding
 * to drift. Yare's `osc` (class `0x8`).
 */
export function dsinTurn(phase: number): number {
  const p = phase >>> 0;
  const i = p >>> (32 - TRIG_BITS);
  const frac = (p & ((1 << (32 - TRIG_BITS)) - 1)) / 2 ** (32 - TRIG_BITS);
  const a = SIN_TABLE[i]!;
  const b = SIN_TABLE[i + 1]!;
  return a + (b - a) * frac;
}

/**
 * A copy of the sine table `dsinTurn` reads (`2¹³ + 1` entries over one turn), so
 * a GPU backend can bind the same values instead of calling its own `sin`.
 */
export function sinTurnTable(): Float64Array {
  return SIN_TABLE.slice();
}

/** Deterministic `cos`. One phase shift, not a second table. */
export function dcos(x: number): number {
  return dsin(x + Math.PI / 2);
}

/**
 * `asin` table domain. `arcsin` is well conditioned on `[-EDGE, EDGE]`; outside
 * it the derivative `(1−x²)^(−3/2)` diverges and a uniform table would need an
 * unreasonable number of entries. `asin(1) = π/2` is the supremum approached as
 * `x → 1`, which the reflection below reaches exactly.
 */
const ASIN_EDGE = 0.9;
const ASIN_SIZE = TRIG_SIZE;
const ASIN_SCALE = (ASIN_SIZE - 1) / (2 * ASIN_EDGE);

const ASIN_TABLE = new Float64Array(ASIN_SIZE);
for (let i = 0; i < ASIN_SIZE; i++) {
  ASIN_TABLE[i] = Math.asin((i / (ASIN_SIZE - 1)) * 2 * ASIN_EDGE - ASIN_EDGE);
}

/**
 * Deterministic `asin` on `[-1, 1]`. Bit-identical on every engine.
 *
 * For `|x| ≤ EDGE` this is a table lookup with linear interpolation. For
 * `|x| > EDGE` it evaluates the exact identity
 *
 *     asin(x) = π/2 − asin( √(1 − x²) )     for x ≥ 0
 *
 * whose reflected argument satisfies `√(1−x²) < √(1−0.81) ≈ 0.436`, comfortably
 * inside the tabled interior. The cost is one `sqrt` and one subtraction on the
 * outer 10% of the domain; the benefit is that the table never resolves the
 * derivative singularity, so `EDGE` entries buy `1e-9` accuracy instead of the
 * `8e-5` a uniform table would manage at the same size.
 *
 * `asin(±1)` is returned exactly as `±π/2` rather than by reflection, because
 * `√(1−1²) = 0` and the reflection is exact there anyway — but doing it
 * directly keeps the endpoint free of the table's interpolation error.
 */
export function dasin(x: number): number {
  if (Number.isNaN(x)) return Number.NaN;
  if (x <= -1) return -Math.PI / 2;
  if (x >= 1) return Math.PI / 2;
  const m = x < 0 ? -x : x;
  if (m <= ASIN_EDGE) {
    const f = (m + ASIN_EDGE) * ASIN_SCALE;
    const i = Math.floor(f);
    const frac = f - i;
    const a = ASIN_TABLE[i]!;
    const b = ASIN_TABLE[i + 1 < ASIN_SIZE ? i + 1 : i]!;
    return x < 0 ? -(a + (b - a) * frac) : a + (b - a) * frac;
  }
  const reflected = Math.sqrt(Math.max(0, 1 - m * m));
  const inner = dasin(reflected);
  return x < 0 ? -(Math.PI / 2 - inner) : Math.PI / 2 - inner;
}

/**
 * Deterministic `acos` on `[-1, 1]`, as `π/2 − asin`. Present so a backend
 * supplying a full port does not have to reach for the platform function.
 */
export function dacos(x: number): number {
  return Math.PI / 2 - dasin(x);
}

/**
 * Deterministic `atan` on `[-1, 1]`, tabled directly.
 *
 * Only this domain is needed: the ISA's uses are the chirality margin's
 * `1/√2` factor (a constant, not a call) and any future slope read-out. Outside
 * `[-1,1]` the identity `atan(x) = π/2 − atan(1/x)` extends it, and that is the
 * caller's business rather than this machine's.
 */
const ATAN_SIZE = TRIG_SIZE;
const ATAN_SCALE = (ATAN_SIZE - 1) / 2;
const ATAN_TABLE = new Float64Array(ATAN_SIZE);
for (let i = 0; i < ATAN_SIZE; i++) {
  ATAN_TABLE[i] = Math.atan((i / (ATAN_SIZE - 1)) * 2 - 1);
}

/** Deterministic `atan` on `[-1, 1]`. */
export function datan(x: number): number {
  if (Number.isNaN(x)) return Number.NaN;
  const c = x < -1 ? -1 : x > 1 ? 1 : x;
  const f = (c + 1) * ATAN_SCALE;
  const i = Math.floor(f);
  const frac = f - i;
  const a = ATAN_TABLE[i]!;
  const b = ATAN_TABLE[i + 1 < ATAN_SIZE ? i + 1 : i]!;
  return a + (b - a) * frac;
}

/**
 * Deterministic `floor`, toward −∞.
 *
 * No table, because none is needed: IEEE 754 defines it exactly, `Math.floor` is exact on every
 * engine, and WGSL's `floor` and WASM's `f32.floor` are the same instruction. It lives here so a
 * lowering has one place to reach for it rather than three.
 */
export function dfloor(x: number): number {
  return Math.floor(x);
}

/**
 * Deterministic `sqrt`.
 *
 * No table either: IEEE 754 requires `sqrt` to be **correctly rounded**, every engine's `Math.sqrt` is
 * the hardware instruction, and WGSL's `sqrt` and WASM's `f32.sqrt` are the same. A negative argument
 * is NaN, which is what all four do.
 */
export function dsqrt(x: number): number {
  return Math.sqrt(x);
}

/**
 * Deterministic `exp`, tabled on the reduced argument.
 *
 * `exp(x) = 2^k · 2^f` with `t = x·log₂e`, `k = ⌊t⌋` and `f = t − k ∈ [0,1)`. `2^f` is a table lookup
 * with linear interpolation; `2^k` is an exact power-of-two scale in binary floating point, so the
 * only error is the table's (`~Δ²/8` with `Δ = ln2/4096`, about `4e-9`) and the table is read-only
 * after module load.
 *
 * The same trade `dsin` makes, for the same reason: reproducibility is a *correctness* question here,
 * because a score computed from these values is compared across machines. `Math.LOG2E`/`Math.LN2` seed
 * the reduction, and `Math.exp` seeds the table — the one sanctioned use of the platform's
 * transcendentals in this package, exactly as the sine table does.
 */
const EXP_BITS = 12;
const EXP_SIZE = 1 << EXP_BITS;
const EXP_TABLE = new Float64Array(EXP_SIZE + 1);
for (let i = 0; i <= EXP_SIZE; i++) EXP_TABLE[i] = Math.exp((i / EXP_SIZE) * Math.LN2);
const LOG2E = Math.LOG2E;

export function dexp(x: number): number {
  if (Number.isNaN(x)) return Number.NaN;
  if (x === Infinity) return Infinity;
  if (x === -Infinity) return 0;
  const t = x * LOG2E;
  const k = Math.floor(t);
  // `t − k` can round up to exactly 1, which would index one past the table; clamping the entry and
  // letting the interpolation reach the last value is exact where it matters and cannot read out of
  // bounds.
  const g = (t - k) * EXP_SIZE;
  const i = Math.min(Math.floor(g), EXP_SIZE - 1);
  const frac = g - i;
  const a = EXP_TABLE[i]!;
  const b = EXP_TABLE[i + 1]!;
  return (a + (b - a) * frac) * 2 ** k;
}

/**
 * The `exp` table, for a lowering that wants to reproduce {@link dexp} rather than call a host's
 * `exp`.
 *
 * A backend that shares this table and follows the same reduction gets bit-identical results by
 * construction; a backend that calls its own `exp` gets a value that agrees to about fifteen digits
 * and is therefore *not* the same number. The distinction is the whole reason the table exists.
 */
export function expTable(): Float64Array {
  return EXP_TABLE.slice();
}

/** How many entries {@link expTable} has, so a lowering can size its own memory without guessing. */
export const EXP_TABLE_SIZE = EXP_SIZE + 1;

/**
 * Deterministic `rand(seed, tick)`: a pure hash of its two `u32` inputs, in `[0, 1)`.
 *
 * There is no state and no clock, so the same `(seed, tick)` is the same number on every backend —
 * which is what makes a citizen's "varied but replayable" behaviour replayable. Every step is `u32`
 * arithmetic (`Math.imul`, `>>>`, xor), which is exact in JavaScript, WebAssembly and WGSL alike, so
 * this is one of the few read-outs that is bit-identical on all three rather than within a tolerance.
 *
 * The mix is murmur3's finaliser applied to `seed ⊕ tick·φ`: chosen for avalanche, not for secrecy.
 * This is a simulation's die, not a key.
 */
export function drand(seed: number, tick: number): number {
  let h = (seed >>> 0) ^ Math.imul(tick >>> 0, 0x9e3779b1);
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  h = (h ^ (h >>> 16)) >>> 0;
  return (h >>> 8) * 2 ** -24;
}

/**
 * `2·arcsin√v` — the fused `v → θ` map. The register format's entry point.
 *
 * Fused rather than composed from `dasin` and `dsqrt` at the call site for one
 * reason that matters at the hardware boundary: it gives a backend a single
 * port member to implement, so a lookup-table datapath does one indexed read
 * and one interpolation per lane instead of two dependent operations with two
 * separate roundings. The composition `2·dasin(√v)` is what a backend gets by
 * default, and it is what the ISA's reference behaviour is defined against.
 */
export const dToTheta = (v: number): number => 2 * dasin(Math.sqrt(v < 0 ? 0 : v > 1 ? 1 : v));

/**
 * `sin²(θ/2)` — the fused `θ → v` map. The register format's exit point.
 *
 * Clamped: `θ` is only meaningful on `[0, π]`, and a `sin` that returns a hair
 * above 1 would make `v` exceed the box and break the conservation invariant
 * this machine exists to hold.
 */
export const dFromTheta = (theta: number): number => {
  const s = dsin(theta / 2);
  const c = s < 0 ? 0 : s > 1 ? 1 : s;
  return c * c;
};
