// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * Vortex VM — the register file.
 *
 * ## Why a register and not a lookup table
 *
 * The load-bearing decision in this package is that a TERA vector is stored as
 * **four angle coordinates**, `θ_d = 2·arcsin√v_d ∈ [0, π]`, and not as the four
 * probabilities `v_d ∈ [0,1]`.
 *
 * In `v` the Fisher information metric is curved:
 *
 *     g_FR(v) = 1 / (v(1−v))
 *
 * which blows up at both ends of each axis. A geodesic under that metric is a
 * curve and generally needs an ODE integrator. In `θ` the *same* metric is
 * flat — `g` becomes the identity — so the geodesic is a straight line and the
 * distance is a Euclidean norm:
 *
 *     d_FR(u,v) = ‖ θ(u) − θ(v) ‖₂
 *
 * That is the whole reason for the register format. Two operations that are the
 * most-used reads in the entire corpus become branch-free straight-line code
 * with no transcendental in the loop:
 *
 *     FR_DIST    rd = ‖θ_a − θ_b‖₂        one v128 subtract + one horizontal add
 *     GEODESIC   rd = θ_a + t(θ_b − θ_a)  one v128 fma
 *
 * In `v` coordinates the same two operations need four `arcsin` and four `sqrt`
 * each. On a `v128` datapath that is the difference between 2 and 10
 * instructions, and — more to the point — the difference between a datapath
 * operation and a transcendental-port call, which is the one part of this
 * machine that is *not* bit-reproducible across engines.
 *
 * The two coordinate systems are used deliberately and never interchangeably,
 * which is a distinction the source corpus insists on:
 *
 * | operation                        | coordinate | why |
 * |----------------------------------|------------|-----|
 * | geodesic, Fisher–Rao distance    | `θ`        | metric is flat, closed form |
 * | Σ_κ conservation projection      | `u = log v`| the constraint is a hyperplane |
 * | RI, entropy, Meji, argmax        | `v`        | per-axis, O(1), no transform |
 *
 * ## The cost of the format, stated plainly
 *
 * `v` is not exactly recoverable from `θ`. The round trip is `θ → v = sin²(θ/2)
 * → θ = 2·arcsin√v`, and both maps are tabled to table precision rather than to
 * `f64` precision, so the trip is lossy.
 *
 * Measured over 200,000 samples spanning `[0,1]`, the worst-case error is
 * **1.47e-7 per traversal** — below f32 resolution (`2⁻²⁴ ≈ 6e-8` is the same
 * order) and identical on every engine that runs the tables. The error
 * **accumulates linearly**: 2.94e-7 after two trips, 7.35e-7 after five, at
 * 1.47e-7 per trip with no sign of convergence. That is the real behaviour of a
 * table-backed map and it is the reason the register carries `v` beside `θ`
 * rather than deriving it — a save/load cycle through a `θ`-only register would
 * walk a lane downhill by a f32 ulp every time.
 *
 * ## Names are decoration
 *
 * `T,E,R,A` is never expanded in the source corpus, and four incompatible
 * expansions exist across the papers — Thought/Emotion/Reason/Archetype in one
 * place, Thought/Energy/Resonance/Action in another, with revision 2 downgrading
 * the claim to "each dimension name is mnemonic… the algebra is purely
 * geometric." The mathematics treats **four exchangeable axes in [0,1]⁴**. This
 * module therefore treats them that way: the axis *index* is the interface and
 * the *name* is a display label that callers may override.
 */

// ── Axes ───────────────────────────────────────────────────────────────────

/** Axis index. The interface; the names below are a display convention. */
export type Axis = 0 | 1 | 2 | 3;

export const AXIS_COUNT = 4;

/**
 * Default axis labels, inherited from `@maiiam/arcade-engine`'s `AXES` so the
 * two implementations cannot silently disagree about which index is which.
 * Cosmetic — see the module header.
 */
export const AXIS_NAMES = ['thought', 'emotion', 'reason', 'action'] as const;

/** Canonical axis order for every tie-break in the ISA. (T,E,R,A) → (0,1,2,3). */
export const AXES: readonly Axis[] = [0, 1, 2, 3];

// ── Cell sizes ─────────────────────────────────────────────────────────────

/** Meji: one bit per axis. `2⁴ = 16`. */
export const MEJI_CELLS = 16;

/** Odu: two bits per axis. `2⁸ = 256`. The tensor square of Meji. */
export const ODU_CELLS = 256;

/** The 64-state consciousness system: `Hyp(4) × Hyp(2)`, one bit per axis on 6. */
export const STATE64_CELLS = 64;

/**
 * Max entropy of a 16-cell Meji distribution, in bits: `log₂16 = 4`.
 *
 * Recorded because a gate in the upstream corpus checks
 * `MAX_ODU_ENTROPY_BITS = 6.0` against this distribution. 6 > 4, so that
 * condition can never fire — it is dead code and this machine does not port it.
 */
export const MEJI_MAX_ENTROPY_BITS = 4;

// ── Scalar domains ─────────────────────────────────────────────────────────

/** A TERA vector in probability coordinates. `v_d ∈ [0,1]`. */
export type Tera = readonly [number, number, number, number];

/** A TERA vector in angle coordinates. `θ_d ∈ [0,π]`. The register format. */
export type Theta = readonly [number, number, number, number];

/**
 * Away from the boundary, for `log` and division.
 *
 * Matches `arcade-engine`'s `EPS` and the platform's own clamp. The two
 * implementations differ at the boundary on purpose — see `ARCHITECTURE.md`
 * §"Asserted divergences" — and `1e-6` is the value they share.
 */
export const EPS = 1e-6;

/** Meji cell index, `0..15`. Packed MSB-first: `k = 8b₀ + 4b₁ + 2b₂ + b₃`. */
export type MejiIndex = number;

/** Odu cell index, `0..255`. Two bits per axis, packed MSB-first. */
export type OduIndex = number;

/** Orientation character. `0` = achiral, on a tie wall of the braid arrangement. */
export type Chirality = -1 | 0 | 1;

// ── The TERA register ──────────────────────────────────────────────────────

/**
 * A TERA register. Angle lanes, probability lanes, and the conservation
 * constant.
 *
 * Deliberately a plain object of fixed-width tuples rather than a
 * `Float64Array` view: the reference implementation is `f64` because that is
 * what JavaScript numbers *are* and what WASM provides unconditionally, and
 * because a fixed shape lets the object stay monomorphic in V8. A GPU or
 * WASM-SIMD backend that wants linear memory gets {@link thetaToArray} and
 * {@link vToArray}, which write the same lanes in the same order into whatever
 * buffer it likes.
 *
 * ## Why both coordinate systems are carried, not one derived from the other
 *
 * This was **not** the original design, and the parity fixture is what changed
 * it. The first version stored `θ` alone and derived `v` on demand through
 * `sin²(θ/2)`. That is lossy — worst case 1.47e-7 — and the loss does not stay
 * in the lanes it happened in. `axisEntropy` has gradient
 * `dH/dv = log₂((1−v)/v)`, which is about **20** at `v = 1e-6`, so a 1.5e-7
 * perturbation of a boundary axis becomes a **5.8e-6** error in the entropy
 * read-out. That is six times the machine's own 1e-6 tolerance, produced by an
 * operation that is *nominally* only a coordinate change.
 *
 * The general form of the failure: any function with a steep gradient near the
 * box boundary amplifies the register format's round-trip error by its own
 * derivative, and the ISA has several such functions by construction — they are
 * all `O(d)` per-axis readouts evaluated at the edges of `[0,1]`.
 *
 * So the register carries `θ` for geometry and `v` for readouts, each computed
 * **once** at construction and each exact with respect to the input it came
 * from. Nothing in the machine depends on `θ → v` being invertible, which is
 * the only property the single-format version was relying on and did not have.
 *
 * The cost is 8 floats instead of 4 per register. On a `v128` datapath that is
 * two registers' worth of lanes for a class of operation — geometry — that is
 * otherwise two instructions and no port call. That is a good trade, and the
 * measurement that produced it is in `test/parity.test.ts`.
 */
export interface TeraReg {
  /** `θ_d = 2·arcsin√v_d`. The metric is flat here. Geometry reads this. */
  readonly theta: Theta;
  /**
   * `v_d ∈ [0,1]`, stored exactly as supplied at construction.
   *
   * Readouts read this. **Never derive it from `theta`** — see above.
   */
  readonly v: Tera;
  /**
   * `κ = Π v_d`, cached.
   *
   * Persisted on the register so the invariant survives every conversion, and
   * so the conservation test is one compare rather than a product.
   */
  readonly kappa: number;
}

/**
 * The conservation constant of the register.
 *
 * Persisted on the register so the invariant survives every conversion. A
 * register whose `kappa` disagrees with the product of its own lanes is a
 * corrupt register, and {@link isWellFormed} is how a caller finds out.
 */
export type Kappa = number;

// ── The signed-trit register ───────────────────────────────────────────────

/**
 * A signed ternary state, WP-36 §2: `z = (p, a, c)` with `p ∈ {−1,0,+1}` and
 * `a, c ∈ [0,1]`.
 *
 * `p = 0` means *unresolved, neutral, or absent* — never "false". That
 * distinction is the entire reason the type exists (WP-36 §13), and it is why
 * this is a register rather than a float: a float cannot tell `0` apart from
 * "two opposing observations cancelled", and WP-36 names that as the category
 * error the type is built to prevent.
 *
 * **Layout in a `u32`** — the point of the packing:
 *
 * ```
 *  bit 31      30      29..15        14..0
 *     ┌────────┬────────┬─────────────┬─────────────┐
 *     │ unused │  p+1   │  activation │  confidence │
 *     │  (0)   │ 2 bits │   15 bits   │   15 bits   │
 *     └────────┴────────┴─────────────┴─────────────┘
 * ```
 *
 * `p+1` maps `{−1,0,+1}` to `{0,1,2}` in two bits. `activation` and
 * `confidence` are `Q0.15` unsigned — 15 bits of fraction, range `[0, 1]`,
 * resolution `2⁻¹⁵ ≈ 3.05e-5`.
 *
 * Why 15 bits each and not something rounder: `s(z) = p·a·c` is a product of two
 * of these, so its resolution is bounded below by the *worse* of the two and
 * the packing must not throw away precision the algebra needs. `Q0.15 × Q0.15`
 * has effective resolution `2⁻³⁰`, which is below `f32` epsilon (`2⁻²⁴`) by six
 * binary orders — so the support read-out of a packed state is exactly as
 * accurate as the `f64` reference for every value the reference can distinguish
 * in practice. The *quantisation* of `a` and `c` is of course the dominant
 * error and is bounded by `2⁻¹⁵`; that is a real, stated precision trade and it
 * is why {@link TritReg} carries the `exact` twin for reference paths.
 *
 * The one thing this layout cannot do is represent `a` or `c` as exactly `1.0`
 * plus an arbitrary f64 tail. `1.0` *is* representable (`0x7FFF`). Everything
 * strictly between the last two representable values is not, and this package
 * says so rather than pretending otherwise.
 */
export interface TritReg {
  readonly packed: number;
}

// Q0.15 fixed point, unsigned. 15 bits of fraction, biased at 0x7FFF = 1.0.
//
// The field is **15** bits wide, so its maximum is 0x7FFF and its mask is 0x7FFF.
// Writing either as 0xFFFF is a one-character mistake with a nasty failure mode:
// a 16-bit mask on a field whose bit 15 belongs to the *polarity* above it reads
// polarity as the top bit of the value, so a `+1` state decodes with activation
// and confidence both slightly above 1. It does not throw, it does not saturate,
// and it is wrong by ~2e-5 — small enough to pass a loose tolerance.
const Q15_ONE = 0x7fff;
const Q15_MASK = 0x7fff;

/** Quantise a `[0,1]` value to `Q0.15`. Round-half-up, clamped. */
export const toQ15 = (x: number): number => {
  if (!Number.isFinite(x)) return 0;
  const c = x < 0 ? 0 : x > 1 ? 1 : x;
  return Math.round(c * Q15_ONE) & Q15_MASK;
};

/** Expand `Q0.15` back to `[0,1]`. Exact inverse of {@link toQ15} on the grid. */
export const fromQ15 = (q: number): number => (q < 0 ? 0 : q > Q15_MASK ? 1 : q) / Q15_ONE;

import { dFromTheta, dToTheta } from './deterministic';

// ── Transcendental port ────────────────────────────────────────────────────

/**
 * The transcendental unit, as an explicit port.
 *
 * IEEE-754 fixes `+ − × ÷ √` and the comparisons bit-for-bit, so plain `f64`
 * arithmetic is portable across V8, JavaScriptCore, Hermes, WASM and a WGSL
 * `f64` path with **no work at all**. The transcendentals are not specified to
 * any accuracy and are demonstrably not portable — this repository has already
 * been bitten by it, which is why `arcade-engine`'s `fromAngle` uses a
 * deterministic `dsin` table instead of `Math.sin`, and why its eslint config
 * bans `Math.sin`/`Math.cos` outright.
 *
 * So this ISA does not treat transcendentals as free, and does not hide them.
 * They are a declared, swappable unit. A backend that wants `f16` intrinsics,
 * a lookup table, or a polynomial approximation supplies its own port and
 * proves it against {@link TRANSCENDENTAL_FIXTURES}; the parity gate then holds
 * the whole machine to one reference.
 *
 * Two of the four members are *fused maps* rather than raw library functions,
 * and that is an architectural decision rather than an optimisation: `v → θ` is
 * `2·arcsin√v` and `θ → v` is `sin²(θ/2)`, and fusing them into single port
 * members lets a table-backed backend do one indexed read and one interpolation
 * per lane instead of an `asin` and a `sqrt` with two separate roundings.
 */
export interface Transcendentals {
  /** Natural log. */
  log(x: number): number;
  /** Base-2 log. Injected rather than `log(x)/log(2)`, which loses a ulp. */
  log2(x: number): number;
  /** `e^x`. */
  exp(x: number): number;
  /** Fused `v → θ`: `2·arcsin√v`. `v ∈ [0,1]` → `θ ∈ [0,π]`. */
  toTheta(v: number): number;
  /** Fused `θ → v`: `sin²(θ/2)`. `θ ∈ [0,π]` → `v ∈ [0,1]`. */
  fromTheta(theta: number): number;
}

/**
 * The reference port — and the **only** default this package has.
 *
 * `sin` and `asin` are both tabled (`deterministic.ts`), so this object is
 * bit-reproducible on every engine in both directions of the register format.
 * There is deliberately no second "platform default" hanging around: an earlier
 * revision of this file exported one alongside the tabled reference, the two
 * disagreed on `toTheta`/`fromTheta`, and a caller that picked up the wrong one
 * got a register format that was not the one the docs described. Two defaults
 * for one machine is one too many.
 *
 * `log`, `log2` and `exp` remain the platform functions. That is not an
 * oversight to be fixed later — it is a *scope* decision. The ISA's core
 * geometry needs the two fused maps on every `FR_DIST`, `GEODESIC` and
 * projection, and those are the ones that had to be closed. The logarithms appear
 * in `kappaProj`, `nu` and the chiral margin: colder paths, no measured
 * cross-engine divergence attached to them yet, and the `Transcendentals` seam
 * exists precisely so that closing them later is a swap rather than a refactor.
 */
export const DEFAULT_TRANSCENDENTALS: Transcendentals = {
  log: Math.log,
  log2: Math.log2,
  exp: Math.exp,
  toTheta: dToTheta,
  fromTheta: dFromTheta,
};

/**
 * Reference values every transcendental port must reproduce.
 *
 * A backend supplies its own port and this is the gate: the machine's
 * determinism guarantee — and therefore `worldHash` stability across the Kernel
 * worker, the renderer, and a headless run — is only as good as the agreement
 * on these. Generated from {@link DEFAULT_TRANSCENDENTALS}; see
 * `scripts/gen-parity.mjs`.
 */
export const TRANSCENDENTAL_FIXTURES = {
  toTheta: [0, 0.0625, 0.25, 0.5, 0.75, 0.9375, 1] as const,
  fromTheta: [0, 0.25, 0.5, 0.7853981633974483, 1.5707963267948966, 2.356194490192345, 3.141592653589793] as const,
  log: [1e-6, 0.5, 1, 2, Math.E, 1e3] as const,
  log2: [1e-6, 0.5, 1, 2, 16, 1024] as const,
  exp: [-8, -1, 0, 1, 4, 8] as const,
} as const;

/**
 * Agreement required of a substitute transcendental port.
 *
 * `1e-12` on the maps is far tighter than the ISA's own `1e-6` cross-
 * implementation tolerance, and deliberately so: a *port* is not a
 * reimplementation of the algebra, it is the same function evaluated
 * differently. Anything looser than this has changed the machine's meaning, not
 * merely its last bits.
 */
export const TRANSCENDENTAL_TOLERANCE = 1e-12;
