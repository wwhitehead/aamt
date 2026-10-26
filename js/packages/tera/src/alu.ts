// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * Vortex VM — the arithmetic unit.
 *
 * ## The admission rule for this file
 *
 * Every operation here is **closed form and parameterless**. No training, no
 * inference, no gradient, no learned table, no fitted constant. That is not
 * asceticism — it is the property that makes the machine a *datapath*:
 *
 *   - A tick of the simulation advances every register by the same handful of
 *     float operations, which is what makes a world hash stable.
 *   - A model may **propose** an impulse; it may never write the result.
 *   - Every operation is a fixed sequence of IEEE-754 basic ops plus at most a
 *     few port calls, so it maps onto a `v128` lane without a lookup table and
 *     without a divergence risk between lanes.
 *
 * Where an operation in the source corpus is *not* closed form, it is not here.
 * The notable exclusions are recorded at the bottom of this file, because
 * "absent on purpose" and "never considered" are different things.
 *
 * ## Coordinate discipline
 *
 * Each operation declares which coordinate system it lives in, and that is not
 * decoration — the corpus contains two *different* metrics and mixing them is a
 * documented, measured error:
 *
 * | system  | metric              | flat? | used for |
 * |---------|---------------------|-------|----------|
 * | `v`     | product Fisher–Rao  | no    | per-axis readouts, RI, entropy |
 * | `θ`     | Euclidean           | yes   | distance, geodesic |
 * | `u=log` | Euclidean           | yes   | Σ_κ projection |
 *
 * Per-axis the two metrics are `g_FR(v) = 1/(v(1−v))` and `g_log(v) = 1/v²`,
 * which coincide only at `v = ½`. Measured divergence between them reaches
 * **0.22 Fisher–Rao units** at `v=(0.2,0.9,0.7,0.3)`, `κ=0.02`. Distances stay
 * Fisher–Rao; the conservation projection is log-Euclidean. Two systems,
 * deliberately, never one.
 */

import {
  AXES,
  AXIS_COUNT,
  DEFAULT_TRANSCENDENTALS,
  EPS,
  MEJI_CELLS,
  ODU_CELLS,
  type Chirality,
  type MejiIndex,
  type OduIndex,
  type Tera,
  type TeraReg,
  type Theta,
  type Transcendentals,
} from './register';

const PI = Math.PI;
const SQRT2 = Math.SQRT2;

const clamp01 = (x: number): number => (Number.isFinite(x) ? (x < 0 ? 0 : x > 1 ? 1 : x) : 0);

/**
 * Index a fixed-width tuple or typed array safely. The widths here (4 axes,
 * 16 cells) are compile-time constants of the algebra, so an out-of-range index
 * is a bug rather than a runtime condition — but `noUncheckedIndexedAccess`
 * cannot know that, and the alternative is a non-null assertion across this
 * whole file.
 *
 * Accepts `ArrayLike` because the hot paths hand in `Float64Array` views (the
 * register's linear-memory form, a distribution buffer) and copying those into
 * plain arrays before reading them would put an allocation in the middle of
 * every measurement.
 */
const at = (a: ArrayLike<number>, i: number): number => {
  const x = a[i];
  return x === undefined ? 0 : x;
};

/** Coerce a 4-lane read into the `Tera` tuple type. */
const tera4 = (v: readonly number[]): Tera => [at(v, 0), at(v, 1), at(v, 2), at(v, 3)];

/**
 * The reference transcendental port, re-exported under the name the ALU layer
 * uses. **It is the same object as `DEFAULT_TRANSCENDENTALS`** — one port, one
 * behaviour, one name per layer rather than two ports that can drift.
 *
 * Every derived operation below takes a `tr` argument defaulting to this. The
 * parameter exists so a backend can substitute a hardware port (an `f16`
 * intrinsic path, a fused polynomial approximation) and prove it against
 * `TRANSCENDENTAL_FIXTURES` — the machine's determinism guarantee is only as
 * strong as this seam.
 */
export const REFERENCE_TRANSCENDENTALS: Transcendentals = DEFAULT_TRANSCENDENTALS;

// ── Register construction and projection ───────────────────────────────────

/**
 * Build a register from probability coordinates.
 *
 * `kappa` is computed once here and carried on the register rather than being
 * recomputed on demand. The reason is the round trip: recovering `v` from `θ`
 * costs `sin²(θ/2)` and is not exact, so a `kappa` derived from a round-tripped
 * `v` would not be the `kappa` of the state the caller handed in. An invariant
 * that degrades when you look at it is not an invariant.
 */
export function makeReg(v: Tera, tr: Transcendentals = REFERENCE_TRANSCENDENTALS): TeraReg {
  const w = tera4(v);
  const theta: Theta = [tr.toTheta(clamp01(at(w, 0))), tr.toTheta(clamp01(at(w, 1))), tr.toTheta(clamp01(at(w, 2))), tr.toTheta(clamp01(at(w, 3)))];
  // `v` is stored exactly as supplied, not round-tripped through `theta`. See
  // the `TeraReg` doc comment: the round trip is lossy and steep-gradient
  // readouts amplify the loss past the machine's own tolerance.
  return { theta, v: w, kappa: at(w, 0) * at(w, 1) * at(w, 2) * at(w, 3) };
}
/**
 * Build a register directly from angle coordinates, with an explicit `kappa`.
 *
 * The other direction of construction, and the one a GPU or WASM-SIMD datapath
 * uses when its native format is already `θ`. `kappa` is required rather than
 * inferred because it cannot be recovered from `θ` without the lossy round trip
 * documented on {@link makeReg}.
 */
export function regFromTheta(theta: Theta, kappa: number, tr: Transcendentals = REFERENCE_TRANSCENDENTALS): TeraReg {
  // `θ = 2·arcsin√v` maps `v ∈ [0,1]` onto `θ ∈ [0,π]` — the FULL half-turn,
  // not `[0,π/2]`. Clamping to π/2 would silently squash every register with
  // v > 0.5 down to v = 0.5, which is invisible in a single-register read and
  // catastrophic in a geodesic. It is bounded by π here rather than by `asin`'s
  // own range so the clamp is a no-op on any well-formed input.
  const clamped: Theta = [
    theta[0] < 0 ? 0 : at(theta, 0) > PI ? PI : at(theta, 0),
    theta[1] < 0 ? 0 : at(theta, 1) > PI ? PI : at(theta, 1),
    theta[2] < 0 ? 0 : at(theta, 2) > PI ? PI : at(theta, 2),
    theta[3] < 0 ? 0 : at(theta, 3) > PI ? PI : at(theta, 3),
  ];
  // The only construction path where `v` is derived rather than supplied, and
  // therefore the only one that carries the round-trip loss. `kappa` is a
  // parameter here for exactly that reason: it cannot be recovered from `theta`
  // without the same lossy trip, and a caller that knows the invariant should
  // state it rather than have it inferred.
  return { theta: clamped, v: tera4(clamped.map((t) => tr.fromTheta(t))), kappa };
}

/**
 * The register's lanes as linear memory, for a backend that wants them in a
 * `Float64Array` or a GPU storage buffer. Layout is `[θ₀ θ₁ θ₂ θ₃]`, lane-major,
 * which is also the order every operand in this file indexes.
 */
export function thetaToArray(reg: TeraReg, out?: Float64Array): Float64Array {
  const dst = out ?? new Float64Array(AXIS_COUNT);
  for (let d = 0; d < AXIS_COUNT; d++) dst[d] = at(reg.theta, d);
  return dst;
}

/**
 * The register's probability lanes as linear memory. Same layout and order as
 * {@link thetaToArray}, so a backend can hold both in one buffer at a fixed
 * stride and address either view by a constant offset.
 */
export function vToArray(reg: TeraReg, out?: Float64Array): Float64Array {
  const dst = out ?? new Float64Array(AXIS_COUNT);
  for (let d = 0; d < AXIS_COUNT; d++) dst[d] = at(reg.v, d);
  return dst;
}

/**
 * Project a register back to probability coordinates.
 *
 * This is the only lossy direction in the machine, and it is marked as such
 * because a caller that forgets will eventually compare `v` against something
 * derived from `θ` and get a mismatch at the eighth decimal. Worst case over
 * 200,000 samples is 1.47e-7 — below `f32` resolution and identical on every
 * engine, so a `f32` simulation cannot tell it apart from the true map, but it
 * is not exact and callers needing exactness should stay in `θ`.
 */
export function regToTera(reg: TeraReg): Tera {
  return reg.v;
}

/**
 * Is the register internally consistent — is its cached `kappa` the product of
 * its own lanes?
 *
 * The tolerance is relative rather than absolute because the product of four
 * numbers in `[0,1]` can legitimately underflow toward zero, and an absolute
 * epsilon would report a perfectly healthy small-`κ` register as corrupt.
 */
export function isWellFormed(reg: TeraReg): boolean {
  const v = reg.v;
  return Math.abs(at(v, 0) * at(v, 1) * at(v, 2) * at(v, 3) - reg.kappa) <= 1e-9 * Math.max(1e-300, Math.abs(reg.kappa));
}

// ── Geometry, in θ where the metric is flat ────────────────────────────────

/**
 * Fisher–Rao distance: `d_FR(u,v) = ‖θ(u) − θ(v)‖₂`.
 *
 * **In θ because the metric is flat there.** In `v` the same distance would be
 * a curved geodesic needing an ODE integrator; in `θ` it is a Euclidean norm,
 * so this is one `v128` subtract plus one horizontal add with **no port call at
 * all**. That is the single largest reason the register is stored in `θ`.
 *
 * Range `[0, 2π]` for four axes: each `θ` spans `[0,π]`, so the widest possible
 * separation is the full diagonal.
 */
export function frDist(a: TeraReg, b: TeraReg): number {
  let sum = 0;
  for (let d = 0; d < AXIS_COUNT; d++) {
    const delta = at(a.theta, d) - at(b.theta, d);
    sum += delta * delta;
  }
  return Math.sqrt(sum);
}

/**
 * Move along the geodesic from `a` toward `b`, reaching `b` exactly at `t=1`.
 *
 * **Linear interpolation in `θ`, because that is the geodesic** — constant speed
 * under Fisher–Rao, since the interpolation is straight in the coordinates where
 * the metric is Euclidean. In `v` coordinates the same operation needs four
 * `sin²` evaluations and four `arcsin`s, and the result would only be
 * *approximately* constant-speed.
 *
 * This is the primitive a register **changing over time** is built from: a
 * per-tick drift is this at small `t`.
 */
export function geodesic(a: TeraReg, b: TeraReg, t: number, tr: Transcendentals = REFERENCE_TRANSCENDENTALS): TeraReg {
  const u = t < 0 ? 0 : t > 1 ? 1 : t;
  const theta: Theta = [
    at(a.theta, 0) + (at(b.theta, 0) - at(a.theta, 0)) * u,
    at(a.theta, 1) + (at(b.theta, 1) - at(a.theta, 1)) * u,
    at(a.theta, 2) + (at(b.theta, 2) - at(a.theta, 2)) * u,
    at(a.theta, 3) + (at(b.theta, 3) - at(a.theta, 3)) * u,
  ];
  // Interpolating in a flat coordinate does not preserve the product, so κ is
  // re-derived from the projected lanes rather than carried from either input.
  // A geodesic between two points on Σ_κ leaves Σ_κ, and pretending otherwise
  // would make the conservation invariant a lie for exactly one operation.
  //
  // Deriving it (rather than writing 0 and hoping) is what keeps the register
  // *well formed* — `isWellFormed` compares the cached κ against the product of
  // the lanes, so a register built with a placeholder constant would fail its
  // own invariant check and the `WELL_FORMED` opcode would report corruption on
  // a perfectly good state.
  const projected = regFromTheta(theta, 0, tr);
  return { ...projected, kappa: at(projected.v, 0) * at(projected.v, 1) * at(projected.v, 2) * at(projected.v, 3) };
}

/**
 * Angular separation per axis — the tangent/normal split of a displacement.
 *
 * `Δν = ν(v̂) − ν(v)` holds identically for any `κ` with no anchor point, which
 * is what makes this usable on two observed states alone. The components are
 * what a gate reads to decide whether a move is admissible: the tangential part
 * slides along the conservation surface, the normal part would move `κ`.
 */
export function frComponents(a: TeraReg, b: TeraReg): { tangential: number; normal: number; total: number } {
  const total = frDist(a, b);
  if (total === 0) return { tangential: 0, normal: 0, total: 0 };
  const comps = perAxisFr(a, b);
  const normal = Math.sqrt(comps.reduce((s, x) => s + x * x, 0));
  return { tangential: Math.sqrt(Math.max(0, total * total - normal * normal)), normal, total };
}

/** Per-axis Fisher–Rao separation, `[Δθ₀, Δθ₁, Δθ₂, Δθ₃]`. */
export function perAxisFr(a: TeraReg, b: TeraReg): number[] {
  const out: number[] = [];
  for (let d = 0; d < AXIS_COUNT; d++) out.push(Math.abs(at(a.theta, d) - at(b.theta, d)));
  return out;
}

// ── Conservation, in u = log v ─────────────────────────────────────────────

/**
 * Conservation violation `ν(v) = log(Π_d v_d) − log κ`.
 *
 * The distance-to-surface scalar. Zero on `Σ_κ`; `|ν|/2` is the distance to it
 * in the log-Euclidean metric the projection uses.
 */
export function nu(v: Tera, kappa: number, tr: Transcendentals = REFERENCE_TRANSCENDENTALS): number {
  const k = kappa < EPS ? EPS : kappa > 1 ? 1 : kappa;
  return tr.log(at(v, 0) * at(v, 1) * at(v, 2) * at(v, 3)) - tr.log(k);
}

/**
 * Distance to `Σ_κ` in log-Euclidean units: `|ν(v)| / 2`.
 */
export function manifoldDistance(v: Tera, kappa: number, tr: Transcendentals = REFERENCE_TRANSCENDENTALS): number {
  return Math.abs(nu(v, kappa, tr)) / 2;
}

/**
 * Log-Euclidean projection of `v` onto `Σ_κ = {v : Π v_d = κ}`, respecting the
 * box `v ≤ 1`.
 *
 * In `u_d = log v_d` the constraint is the hyperplane `Σ u_d = log κ`, and
 * orthogonal projection onto a hyperplane is a uniform shift:
 *
 *     u'_d = u_d − ((Σ u_d − log κ)/4)      ⟺      v'_d = v_d·(κ/Π v_j)^(1/4)
 *
 * This is the **only** coordinate choice in which the constraint is flat, which
 * is why the projection is closed form here and nowhere else.
 *
 * **The box has to be handled, and handling it is not optional.** When
 * `κ > Π v` the shift pushes one or more axes above 1. Clamping them and
 * stopping silently breaks the conservation law this function exists to enforce
 * — the product lands short of `κ` and the register's invariant becomes a lie.
 * So this is an **active-set** projection: axes that would overflow are pinned
 * at the ceiling, the remaining log budget is redistributed across the free
 * axes only, and the active set is recomputed until stable. The feasible region
 * is a hyperplane ∩ box, which is convex, so it converges; the loop is
 * additionally bounded because a caller-supplied `κ` outside the reachable set
 * has no projection and must not spin.
 *
 * A naive uniform-shift-then-clamp version differs from this by up to **0.081**
 * on 34 of 200 generated parity cases — every one of them saturating. The
 * parity harness in `test/parity.test.ts` is what catches that class of bug,
 * which is why it exists rather than a set of hand-picked unit tests.
 */
export function kappaProj(v: Tera, kappa: number, tr: Transcendentals = REFERENCE_TRANSCENDENTALS): Tera {
  const target = tr.log(kappa < EPS ? EPS : kappa > 1 ? 1 : kappa);
  const u = [0, 1, 2, 3].map((d) => tr.log(Math.max(EPS, clamp01(at(v, d)))));

  const active = new Set<number>();
  for (let pass = 0; pass < AXIS_COUNT; pass++) {
    const free = AXES.filter((d) => !active.has(d));
    if (free.length === 0) break;
    let sum = 0;
    for (const d of free) sum += at(u, d);
    const shift = (target - sum) / free.length;
    const overflow = free.filter((d) => at(u, d) + shift > 0);
    if (overflow.length === 0) {
      for (const d of free) u[d] = at(u, d) + shift;
      return unlog(u, tr);
    }
    for (const d of overflow) {
      u[d] = 0;
      active.add(d);
    }
  }

  // Every axis pinned at the ceiling: κ is unreachable from inside the box.
  // Returning (1,1,1,1) would quietly break the invariant in exactly the case
  // that is hardest to notice, so rescale onto the best feasible face instead.
  const free = AXES.filter((d) => !active.has(d));
  if (free.length === 0) return [1, 1, 1, 1];
  let sum = 0;
  for (const d of free) sum += at(u, d);
  const shift = (target - sum) / free.length;
  for (const d of free) u[d] = at(u, d) + shift;
  return unlog(u, tr);
}

/** `exp` a log-coordinate vector back to `v`, clamping into the box. */
const unlog = (u: readonly number[], tr: Transcendentals): Tera => [
  clamp01(tr.exp(at(u, 0))),
  clamp01(tr.exp(at(u, 1))),
  clamp01(tr.exp(at(u, 2))),
  clamp01(tr.exp(at(u, 3))),
];

// ── Address: the discrete cells ────────────────────────────────────────────

/**
 * Meji cell index, `0..15`. One bit per axis, packed MSB-first:
 * `k = 8b₀ + 4b₁ + 2b₂ + b₃`.
 *
 * **This is an address, and that is the one thing the ISA forbids deriving from
 * geometry.** It is a *threshold* of the register's value at the midpoint — a
 * quantisation for labelling — not a hash of it. The upstream corpus is
 * unambiguous that a 4-bit semantic address alone is useless for retrieval:
 * measured `recall@10 = 0.017`, because the address function, not the geometry,
 * is the ceiling. It is a router, not a search index, and it belongs on a
 * register's tag field rather than anywhere near a buffer offset.
 *
 * The `≥ ½` is the platform's `Ox` tie-break, matching the substrate router's
 * `v ≥ mid`. Chosen deliberately so the boundary is *closed on the upper side*:
 * the neutral register `(½,½,½,½)` therefore lands on cell 15, which is a
 * stable answer for a state that is by definition undecided.
 */
export function mejiIndex(reg: TeraReg): MejiIndex {
  const v = reg.v;
  let k = 0;
  for (let d = 0; d < AXIS_COUNT; d++) if (at(v, d) >= 0.5) k |= 1 << (AXIS_COUNT - 1 - d);
  return k;
}

/**
 * Odu cell index, `0..255`. **Two** bits per axis — 256 = 2⁸ over four axes.
 * Meji's 16 cells are one bit each; getting this wrong by a factor of 16 is the
 * quiet mistake that makes a "similar meaning" scan return nonsense.
 */
export function oduIndex(reg: TeraReg): OduIndex {
  const v = reg.v;
  const q = (x: number): number => {
    const c = x < 0 ? 0 : x > 0.999999 ? 3 : Math.floor(x * 4);
    return c;
  };
  return (q(at(v, 0)) << 6) | (q(at(v, 1)) << 4) | (q(at(v, 2)) << 2) | q(at(v, 3));
}

/**
 * The product-Bernoulli distribution over the 16 Meji cells:
 * `p(k|v) = Π_d [v_d if b_d(k)=1 else 1−v_d]`.
 *
 * **Computed by successive doubling, and that is the hardware point.** Written
 * as a nested product over all 16 cells and 4 axes this is 64
 * multiplications. Written as an outer product — which is what it *is*, the
 * family being rank-1 — it is `2⁴ − 1 = 15`. The saving is not a trick; it is
 * the same fact that makes the encoding lossless in the first place, and a
 * backend gets it for free by writing the intermediate into lane registers.
 *
 * The result sums to exactly 1 by construction: each axis contributes a
 * normalised Bernoulli.
 */
export function mejiDist(reg: TeraReg): Float64Array {
  const v = reg.v;
  let dist = new Float64Array(1);
  dist[0] = 1;
  // Axes are consumed from the LEAST significant end backwards, because the
  // cell index packs axis 0 as the most significant bit: `k = 8b₀ + 4b₁ + 2b₂ + b₃`.
  // Doubling with the new axis as the HIGH half of each pair is what puts it in
  // the right weight. Getting this order wrong permutes all sixteen cells and
  // the parity fixture catches it immediately.
  for (let d = AXIS_COUNT - 1; d >= 0; d--) {
    const p = clamp01(at(v, d));
    const next = new Float64Array(dist.length * 2);
    for (let i = 0; i < dist.length; i++) {
      const base = at(dist, i);
      next[i] = base * (1 - p);
      next[i + dist.length] = base * p;
    }
    dist = next;
  }
  return dist;
}

/**
 * Recover `v` from its 16-cell distribution, exactly.
 *
 * `recover(mejiDist(v)) === v`, because the distribution is a rank-1 outer
 * product and a Bernoulli mean recovers its parameter:
 *
 *     recover(p)_i = Σ_k p(k|v)·b_i(k) = v_i
 *
 * **This is why 16 slots are a sufficient record of a register.** The 4-bit
 * address is a *lossless* encoding of a continuous 4-D state, not a lossy
 * summary of it. Two actors in the same Meji cell are different states and the
 * distribution keeps the difference — which is also why the address is a
 * labelling scheme and never a personality.
 */
export function recoverTera(dist: ArrayLike<number>): Tera {
  const out: number[] = [0, 0, 0, 0];
  for (let d = 0; d < AXIS_COUNT; d++) {
    let acc = 0;
    for (let k = 0; k < MEJI_CELLS; k++) acc += (dist[k] ?? 0) * (((k >> (AXIS_COUNT - 1 - d)) & 1) as number);
    out[d] = clamp01(acc);
  }
  return out as unknown as Tera;
}

// ── Per-axis readouts, in v ────────────────────────────────────────────────

/**
 * Resonant Integrity: `RI = Π_d max(v_d, 1−v_d) = max_k p(k|v) ∈ [1/16, 1]`.
 *
 * How *settled* a register is. `1.0` means every axis sits decisively on one
 * side; `1/16` means all four are pinned at the midpoint and no archetype wins
 * at all. This is the scalar a gate should read, because it is already on the
 * same `0..1` scale as every confidence elsewhere in the system.
 *
 * `O(d)`, not `O(2^d)` — the maximum of a rank-1 factorised family is the
 * product of the per-axis maxima. The distributional form would be 16
 * comparisons; this is 4.
 */
export function resonantIntegrity(reg: TeraReg): number {
  const v = reg.v;
  let ri = 1;
  for (let d = 0; d < AXIS_COUNT; d++) {
    const x = clamp01(at(v, d));
    ri *= Math.max(x, 1 - x);
  }
  return ri;
}

/**
 * Resonant Integrity scaled from its `1/16` floor to `0..1`, so it has the same
 * shape as a confidence and a gate can compare it against a threshold directly.
 */
export function coherence(reg: TeraReg): number {
  return (resonantIntegrity(reg) - 1 / MEJI_CELLS) / (1 - 1 / MEJI_CELLS);
}

/**
 * Bernoulli entropy of one axis, in bits. Zero at 0 or 1, one at ½.
 *
 * `log2` comes from the port rather than being `Math.log(x)/Math.log(2)`, which
 * loses an ulp at exactly the endpoints where the answer is being compared
 * against a threshold.
 */
export function axisEntropy(x: number, tr: Transcendentals = REFERENCE_TRANSCENDENTALS): number {
  const v = clamp01(x);
  if (v <= 0 || v >= 1) return 0;
  return -v * tr.log2(v) - (1 - v) * tr.log2(1 - v);
}

/** Axis entropy under the reference port, for the readouts that carry no port. */
const refAxisEntropy = (x: number): number => axisEntropy(x, REFERENCE_TRANSCENDENTALS);

/**
 * Total entropy of a register, in bits. Range `[0, 4]` — `MEJI_MAX_ENTROPY_BITS`.
 *
 * `O(d)` for the same reason `RI` is: the family is rank-1, so the joint
 * entropy over 16 cells reduces to a *sum* of four independent per-axis terms.
 * There is no cross-axis term because there is no dependence to express — which
 * is the family's central limitation and also the reason this is four float ops
 * instead of a 16-cell reduction.
 */
export function entropy(reg: TeraReg): number {
  const v = reg.v;
  let h = 0;
  for (let d = 0; d < AXIS_COUNT; d++) h += refAxisEntropy(at(v, d));
  return h;
}

/** Axes whose entropy is under `bits` have collapsed toward 0 or 1. */
export function collapsedAxes(reg: TeraReg, bits = 0.5): boolean[] {
  const v = reg.v;
  return AXES.map((_a, d) => refAxisEntropy(at(v, d)) < bits);
}

/**
 * Zero-veto read-out: has any axis fallen below `floor`?
 *
 * The gate is a **chosen threshold, not a derived one**, and this machine says
 * so in the name and in the parameter rather than baking `0.15` into a constant
 * that reads like a physical law.
 */
export function vetoed(reg: TeraReg, floor = 0.15): boolean {
  const v = reg.v;
  for (let d = 0; d < AXIS_COUNT; d++) if (at(v, d) < floor) return true;
  return false;
}

/** Index of the lowest axis — the one a veto names. Ties resolve to the lower index. */
export function argminAxis(reg: TeraReg): number {
  const v = reg.v;
  let best = 0;
  for (let d = 1; d < AXIS_COUNT; d++) if (at(v, d) < at(v, best)) best = d;
  return best;
}

// ── Orientation: the chirality half of the address ─────────────────────────

/**
 * `χ(v) = Π_{i<j} sign(v_i − v_j) = sgn(σ_v) ∈ {−1, 0, +1}` — the sign of the
 * permutation that sorts the four axes, i.e. `(-1)^(inversions)`.
 *
 * **Why this exists when a 4-bit address already does.** The address thresholds
 * at `v_d = ½`; `χ` thresholds at `v_i = v_j`. They are *transverse*
 * arrangements, and the difference is a stability property: a uniform rescale
 * onto `Σ_κ` walks coordinates across ½ and so destroys the address, but it
 * cannot reorder the axes and so leaves `χ` untouched. Measured over 2×10⁵
 * random draws projected to `Σ₀.₀₅`, the dominant Meji changed on **48.8%** of
 * them and `χ` changed on **none**.
 *
 * The two arrangements together generate the hyperoctahedral group
 * `B₄ = ℤ₂⁴ ⋊ S₄`; the `ℤ₂⁴` half is the Householder mirror `H(v) = (1−T,E,R,A)`
 * and `χ` is the `S₄` half. Same determinant character, two complementary
 * subgroups. Enumerated over all 24 × 16 chamber/cell pairs, `χ` splits *every*
 * Meji cell exactly in half, so the joint address is 32 non-empty cells
 * against Meji's 16 — the doubling claim is exact and perfectly balanced.
 *
 * Returns `0` on any exact tie, and on non-finite input: fail-closed, because
 * `0` is the correct report that the state has no handedness rather than a
 * guess about which way it was leaning.
 *
 * Note both a strictly ascending and a strictly descending register return
 * `+1`: descending is the full reversal, six transpositions, therefore even.
 * Only an *odd* reordering — a single swap — returns `−1`.
 */
export function chirality(reg: TeraReg): Chirality {
  const v = reg.v;
  let product = 1;
  for (let i = 0; i < AXIS_COUNT; i++) {
    for (let j = i + 1; j < AXIS_COUNT; j++) {
      const diff = at(v, i) - at(v, j);
      if (diff === 0) return 0;
      product *= diff > 0 ? 1 : -1;
    }
  }
  return (product > 0 ? 1 : -1) as Chirality;
}

/**
 * Descending axis order with the `Ox` tie-break: on equality the axis earlier
 * in canonical `(0,1,2,3)` order sorts first. Returned as indices.
 */
export function oxOrder(reg: TeraReg): number[] {
  const v = reg.v;
  return [0, 1, 2, 3].sort((a, b) => at(v, b) - at(v, a) || a - b);
}

/** Parity of the `Ox`-resolved order. Never 0; a fully tied register gives `+1`. */
export function oxChirality(reg: TeraReg): -1 | 1 {
  const order = oxOrder(reg);
  let inversions = 0;
  for (let i = 0; i < AXIS_COUNT; i++) {
    for (let j = i + 1; j < AXIS_COUNT; j++) if (at(order, i) > at(order, j)) inversions++;
  }
  return inversions % 2 === 0 ? 1 : -1;
}

/**
 * `m(v) = min_r log(v₍r₎/v₍r₊₁₎)/√2` — the Euclidean distance, in log
 * coordinates on `Σ_κ`, from the register to the nearest wall of its chamber.
 *
 * `√2·m` is the log-ratio between the two closest-ranked axes, so `m ≈ 0.067`
 * means they are about 10% apart. A function of ratios only, which makes it
 * `Σ_κ`-invariant by the same argument as `χ`.
 */
export function chiralMargin(reg: TeraReg): number {
  const v = reg.v;
  if (!v.every((x) => Number.isFinite(x) && x > 0)) return 0;
  const sorted = oxOrder(reg)
    .map((i) => REFERENCE_TRANSCENDENTALS.log(at(v, i)))
    .sort((a, b) => b - a);
  let smallest = Infinity;
  for (let r = 0; r < AXIS_COUNT - 1; r++) smallest = Math.min(smallest, at(sorted, r) - at(sorted, r + 1));
  return Math.max(smallest, 0) / SQRT2;
}

/**
 * `χ(v) · m(v)` — the continuous signed chirality. Crossing a wall carries it
 * through zero instead of jumping, so this is the read-out to use where a bare
 * bit would flap on near-degenerate registers.
 */
export function signedChirality(reg: TeraReg): number {
  return chirality(reg) * chiralMargin(reg);
}

/**
 * The 5-bit joint address: the 4-bit Meji index with `χ_Ox` as a low-order
 * bit (`+1 → 1`, `−1 → 0`). All 32 values are non-empty and the chirality bit
 * is exactly balanced within every Meji cell.
 *
 * This is the ISA's semantic **tag**. It goes on the register and travels with
 * it; it never becomes an offset. See {@link mejiIndex} for why a 4-bit
 * address is a router and not a search index.
 */
export function chiralAddress(reg: TeraReg): number {
  return (mejiIndex(reg) << 1) | (oxChirality(reg) > 0 ? 1 : 0);
}

// ── The 64-state system ────────────────────────────────────────────────────

/**
 * The 64-state consciousness system as a 6-bit word.
 *
 * `Hyp(6) ≅ Hyp(4) × Hyp(2)` — the same product-Bernoulli object at a third
 * size, 16 Meji × a 2-bit fibre. It is one bit per axis on six axes, which is
 * why a distance on it is a **popcount**: `d(s,s') = popcount(s ⊕ s')` is one
 * `xor` and one popcount, with no port call and no float lane.
 */
export function state64(reg: TeraReg, extra: readonly number[]): number {
  let s = mejiIndex(reg);
  s |= (clamp01(at(extra, 0)) >= 0.5 ? 1 : 0) << 4;
  s |= (clamp01(at(extra, 1)) >= 0.5 ? 1 : 0) << 5;
  return s & 0x3f;
}

/** Hamming distance between two 6-bit states. `popcount(a ⊕ b)`. */
export function stateDistance(a: number, b: number): number {
  let x = (a ^ b) & 0x3f;
  let count = 0;
  while (x !== 0) {
    x &= x - 1;
    count++;
  }
  return count;
}

/** The Odu cell as the tensor square of Meji: `Odu = Hyp(4) × Hyp(4)`. */
export const ODU_IS_MEJI_SQUARED = ODU_CELLS === MEJI_CELLS * MEJI_CELLS;

/** `Hyp(8) ≅ Hyp(4) × Hyp(4)` — asserted so a refactor cannot break the identity. */
export const HYP8_IS_HYP4_SQUARED = ODU_CELLS === 1 << 8;

// ── Deliberately absent ────────────────────────────────────────────────────

/**
 * Recorded so "not here" reads as a decision rather than an oversight.
 *
 * - **MoE / expert dispatch.** The corpus's own note: the router "picks a name,
 *   not different weights," with no adapters loaded anywhere. There is nothing
 *   to dispatch to, and an ISA op that routes to a registry of absent experts
 *   would be a fiction.
 * - **Neural inference, beyond bound read-outs.** Every operation in this file is
 *   closed form. The original rule was stricter: a learned read-out is a model,
 *   and a model only proposes, never writes a register. ADR-015 amends it: class
 *   `0x9` (weights.ts) reads frozen, content-addressed tensors at static offsets
 *   into registers, deterministically. Still absent: training or any in-run weight
 *   update, and any read-out whose tensors are not bound and identified before
 *   the run.
 * - **A weight index computed from anything but the host.** ADR-021 admits exactly
 *   one runtime index: `W_ROW4` reads four columns of a row named by a `u32`
 *   register that Yare types `token` — a host-supplied parameter or a literal,
 *   bounds-checked against the bound shape on every path. Absent: a row computed
 *   from a register value (a TERA value, a cell, a phase, an `f32`, any
 *   arithmetic), a runtime column, a second index, and scaled or offset indexing.
 * - **Simplex routing** (`w ∈ Δ^{K−1}`, `h' = Σ wᵢhᵢ`). Specified in WP-38 with
 *   zero measurements. Adopting an unmeasured router as a hardware primitive
 *   would put the ISA's correctness behind a paper that says outright "no result
 *   in §4, §9, or §12 has been measured."
 * - **Digital-root anything.** `DR(n) ∈ [1,9]` maps all of 3-D space into 729
 *   buckets and collides every point 9 apart. It is the ISA's explicit
 *   anti-feature; `ARCHITECTURE.md` carries the receipt.
 * - **Reverse-time trajectory correction.** The mechanism is built and the
 *   regime detector classifies all three synthetic regimes correctly, but
 *   there is no evidence any real session needs it, and EXP-02's own later
 *   sections found real text statistically indistinguishable from a random
 *   shuffle after fixing a null-model bug.
 * - **Anything derived from a fitted constant.** Where the corpus has one, this
 *   file takes it as a parameter with a name that says it is chosen — `floor`
 *   on {@link vetoed}, for instance — rather than as a constant that reads like
 *   a law.
 */
export type { Tera, TeraReg, Theta, Transcendentals };

export const ABSENT_BY_DESIGN = [
  'moe-expert-dispatch',
  'neural-training',
  'computed-weight-index',
  'simplex-routing',
  'digital-root-addressing',
  'reverse-time-correction',
  'fitted-constants',
] as const;


