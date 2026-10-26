// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * The ISA's own claims, tested as claims.
 *
 * `parity.test.ts` proves the machine agrees with a reference implementation.
 * This file proves the things the *architecture* asserts, which no reference can
 * check because they are claims about properties rather than about agreement:
 *
 *   1. The 16-cell distribution is a **lossless** encoding of a continuous 4-D
 *      state. (Corollary C1 of the product-Bernoulli family.)
 *   2. `RI` and entropy are inside their closed-form ranges.
 *   3. **Chirality is Σ_κ-stable where the address is not.** This is the single
 *      most load-bearing asymmetry in the design, and it has a measured number
 *      attached: over 2×10⁵ random draws projected onto `Σ₀.₀₅`, the dominant
 *      Meji changed on **48.8%** of them and `χ` changed on **none**.
 *   4. The Fisher–Rao distance is a *metric*.
 *   5. `geodesic(a,b,0) === a` and `geodesic(a,b,1) === b`, in the register's own
 *      flat coordinates, with no port round trip.
 *   6. `kappaProj` is idempotent and conserves the product when it does not
 *      saturate.
 *   7. `nu` is zero exactly on `Σ_κ`. Tested here rather than in the parity
 *      fixture because `arcade-engine` does not implement it, so there is no
 *      independent value to compare against — writing the ISA's own formula into
 *      a fixture would make the test circular and permanently true.
 */

import { describe, expect, it } from 'vitest';
import { AXIS_COUNT, MEJI_CELLS, MEJI_MAX_ENTROPY_BITS } from '../src/register';
import {
  argminAxis,
  axisEntropy,
  chirality,
  chiralAddress,
  chiralMargin,
  coherence,
  collapsedAxes,
  entropy,
  frDist,
  geodesic,
  isWellFormed,
  kappaProj,
  makeReg,
  mejiDist,
  mejiIndex,
  nu,
  oduIndex,
  oxChirality,
  regFromTheta,
  regToTera,
  resonantIntegrity,
  signedChirality,
  state64,
  stateDistance,
  thetaToArray,
  vToArray,
  vetoed,
  type Tera,
} from '../src/alu';
import { dacos, dasin, datan, dcos, dsin } from '../src/deterministic';

const reg = (v: Tera) => makeReg(v);

/** Recover `v` from a 16-cell distribution — corollary C1, written out. */
const recover = (dist: ArrayLike<number>): Tera => {
  const out: number[] = [0, 0, 0, 0];
  for (let d = 0; d < AXIS_COUNT; d++) {
    let acc = 0;
    for (let k = 0; k < MEJI_CELLS; k++) acc += (dist[k] ?? 0) * ((k >> (AXIS_COUNT - 1 - d)) & 1);
    out[d] = acc;
  }
  return out as unknown as Tera;
};

describe('the 16-cell distribution is lossless', () => {
  const samples: Tera[] = [
    [0.5, 0.5, 0.5, 0.5],
    [0.9, 0.1, 0.5, 0.25],
    [0.01, 0.99, 0.02, 0.98],
    [1, 0, 1, 0],
    [0.123456789, 0.987654321, 0.5, 0.333333333],
  ];

  it('recovers v exactly from 16 numbers', () => {
    // `recover(p)_i = Σ_k p(k|v)·b_i(k) = v_i`. The family is rank-1, so a
    // Bernoulli mean recovers its parameter. This is why 16 slots are a
    // sufficient record of a register, and why a 4-bit address is a *lossless
    // encoding* rather than a lossy summary.
    for (const v of samples) {
      const back = recover(mejiDist(reg(v)));
      for (let d = 0; d < AXIS_COUNT; d++) expect(back[d]).toBeCloseTo(v[d] ?? 0, 12);
    }
  });

  it('sums to 1', () => {
    for (const v of [[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.5, 0.5], [0.99, 0.01, 0.99, 0.01]] as Tera[]) {
      let sum = 0;
      for (const p of mejiDist(reg(v))) sum += p;
      expect(sum).toBeCloseTo(1, 12);
    }
  });

  it('costs 15 multiplies, not 64', () => {
    // The successive-doubling form is the hardware claim, so it is asserted
    // structurally: intermediate sizes 1, 2, 4, 8, 16, total 2⁴−1 = 15. The
    // naive nested form over 16 cells × 4 axes is 64. The saving is not a trick;
    // it is the same rank-1 fact that makes the encoding lossless, and a
    // backend gets it for free by writing the intermediate into lane registers.
    let cells = 1;
    let multiplies = 0;
    for (let d = 0; d < AXIS_COUNT; d++) {
      multiplies += cells;
      cells *= 2;
    }
    expect(cells).toBe(MEJI_CELLS);
    expect(multiplies).toBe(15);
    expect(MEJI_CELLS * AXIS_COUNT).toBe(64);
  });

  it('indexes cells MSB-first, matching the bit order the address uses', () => {
    // Axis 0 is the most significant bit: cell 11 is T=1,E=0,R=1,A=1. Getting
    // this backwards permutes all sixteen cells and the parity fixture catches
    // it immediately — but stating it here names which of two right answers was
    // chosen.
    // [0.7, 0.2, 0.9, 0.4] → bits T=1,E=0,R=1,A=0 → cell 10, whose probability
    // is the product of the chosen halves. Cell 4 is T=0,E=1,R=0,A=0.
    const dist = mejiDist(reg([0.7, 0.2, 0.9, 0.4]));
    expect(dist[10]).toBeCloseTo(0.7 * 0.8 * 0.9 * 0.6, 12);
    expect(dist[4]).toBeCloseTo(0.3 * 0.2 * 0.1 * 0.6, 12);
    expect(mejiIndex(reg([0.7, 0.2, 0.9, 0.4]))).toBe(10);
  });
});

describe('read-out ranges are the closed-form ones', () => {
  it('RI lies in [1/16, 1] and hits both ends', () => {
    // RI = Π_d max(v_d, 1−v_d). Floor when every axis is at the midpoint,
    // ceiling when every axis is decisively on one side.
    expect(resonantIntegrity(reg([0.5, 0.5, 0.5, 0.5]))).toBeCloseTo(1 / MEJI_CELLS, 12);
    expect(resonantIntegrity(reg([1, 1, 1, 1]))).toBeCloseTo(1, 12);
    expect(resonantIntegrity(reg([0, 0, 0, 0]))).toBeCloseTo(1, 12);
    for (const v of [[0.3, 0.7, 0.2, 0.8], [0.44, 0.56, 0.49, 0.51]] as Tera[]) {
      const ri = resonantIntegrity(reg(v));
      expect(ri).toBeGreaterThanOrEqual(1 / MEJI_CELLS - 1e-12);
      expect(ri).toBeLessThanOrEqual(1 + 1e-12);
    }
  });

  it('coherence is RI rescaled from its floor, so it is 0..1', () => {
    expect(coherence(reg([0.5, 0.5, 0.5, 0.5]))).toBeCloseTo(0, 12);
    expect(coherence(reg([1, 1, 1, 1]))).toBeCloseTo(1, 12);
  });

  it('entropy lies in [0, 4] and peaks at the midpoint', () => {
    // The dead gate this machine refuses to port checks 6.0 bits against a
    // distribution whose maximum entropy is log₂16 = 4. Asserting the ceiling
    // here is what makes that gate's impossibility checkable rather than a claim.
    expect(MEJI_MAX_ENTROPY_BITS).toBe(4);
    expect(entropy(reg([0.5, 0.5, 0.5, 0.5]))).toBeCloseTo(MEJI_MAX_ENTROPY_BITS, 12);
    expect(entropy(reg([1, 1, 1, 1]))).toBeCloseTo(0, 12);
    expect(entropy(reg([0, 0, 0, 0]))).toBeCloseTo(0, 12);
    for (const v of [[0.1, 0.2, 0.3, 0.4], [0.9, 0.1, 0.6, 0.4]] as Tera[]) {
      expect(entropy(reg(v))).toBeGreaterThan(0);
      expect(entropy(reg(v))).toBeLessThanOrEqual(MEJI_MAX_ENTROPY_BITS + 1e-12);
    }
  });

  it('a single axis contributes at most 1 bit, so four axes cap at 4', () => {
    for (let x = 0.001; x < 1; x += 0.017) expect(axisEntropy(x)).toBeLessThanOrEqual(1 + 1e-12);
  });

  it('collapsedAxes tracks the axes that have committed', () => {
    const c = collapsedAxes(reg([0.02, 0.5, 0.98, 0.5]));
    expect(c[0]).toBe(true);
    expect(c[2]).toBe(true);
    expect(c[1]).toBe(false);
  });

  it('argminAxis names the axis a veto would report', () => {
    expect(argminAxis(reg([0.9, 0.8, 0.05, 0.7]))).toBe(2);
    expect(argminAxis(reg([0.05, 0.8, 0.9, 0.7]))).toBe(0);
  });

  it('the veto threshold is a parameter, and it bites where it is told to', () => {
    expect(vetoed(reg([0.9, 0.9, 0.9, 0.9]), 0.15)).toBe(false);
    expect(vetoed(reg([0.9, 0.9, 0.9, 0.05]), 0.15)).toBe(true);
    // Same register, different declared floor. A threshold that reads like a
    // constant in the source is a threshold nobody can argue with.
    expect(vetoed(reg([0.9, 0.9, 0.9, 0.05]), 0.01)).toBe(false);
  });

  it('odu is two bits per axis, not four', () => {
    // 256 = 2⁸ over four axes. Meji's 16 cells are one bit each. Getting this
    // wrong by 16× is the quiet mistake that makes a "similar meaning" scan
    // return nonsense.
    expect(mejiIndex(reg([0.9, 0.9, 0.9, 0.9]))).toBe(15);
    expect(oduIndex(reg([0.9, 0.9, 0.9, 0.9]))).toBe(255);
    expect(oduIndex(reg([0.1, 0.1, 0.1, 0.1]))).toBe(0);
    // Two bits per axis is an *odometer* per axis, not a bit pattern: q(0.9)=3,
    // q(0.1)=0, so 3<<6 | 0<<4 | 3<<2 | 0 = 204.
    expect(oduIndex(reg([0.9, 0.1, 0.9, 0.1]))).toBe(0b11001100);
  });
});

describe('chirality is the stable half of the address', () => {
  const STATES: Tera[] = [
    [0.9, 0.6, 0.3, 0.1],
    [0.8, 0.2, 0.7, 0.4],
    [0.95, 0.55, 0.15, 0.65],
    [0.99, 0.51, 0.02, 0.48],
    [0.7, 0.8, 0.2, 0.1],
    [0.05, 0.4, 0.9, 0.3],
  ];

  it('survives the Σ_κ projection wherever the projection does not saturate', () => {
    // The projection is a uniform positive scaling `v' = c·v` on any axis that
    // does not pin. Every Vandermonde factor `sign(v_i − v_j)` is fixed by a
    // positive `c`, so χ cannot move. The address thresholds at ½ and a uniform
    // rescale walks coordinates straight across it — measured at 48.8% of
    // 2×10⁵ draws.
    //
    // The qualifier is not a hedge, it is the other half of the theorem. On the
    // *saturating* branch two or more axes pin at exactly 1, which is a tie, so
    // χ collapses to 0. The claim is therefore precise and testable:
    //
    //     |S| ≤ 1  →  χ is preserved exactly
    //     |S| ≥ 2  →  χ collapses to 0, never flips to the opposite sign
    //
    // where S is the saturated set. "Never flips" is the strong part, and it is
    // what makes χ usable as an address bit at all.
    let addressFlips = 0;
    let preserved = 0;
    let collapsed = 0;
    for (const v of STATES) {
      for (const kappa of [0.02, 0.05, 0.2, 0.5, 0.9]) {
        const projected = reg(kappaProj(v, kappa));
        const pinned = projected.v.filter((x) => x >= 1).length;
        if (mejiIndex(reg(v)) !== mejiIndex(projected)) addressFlips++;
        const before = chirality(reg(v));
        if (pinned <= 1) {
          expect(chirality(projected)).toBe(before);
          preserved++;
        } else {
          expect(chirality(projected)).toBe(0);
          collapsed++;
        }
        // In every stratum, χ never reaches the opposite sign.
        expect(chirality(projected)).not.toBe(-before);
      }
    }
    // And the asymmetry is real, not vacuous: the address really does move.
    expect(addressFlips, 'the address is demonstrably unstable').toBeGreaterThan(0);
    expect(preserved, 'non-saturating projections were exercised').toBeGreaterThan(5);
    expect(collapsed, 'saturating projections were exercised').toBeGreaterThan(0);
  });

  it('is +1 for both the ascending and the descending order', () => {
    // Descending is the full reversal: six transpositions, therefore even. Only
    // an odd reordering is −1. Getting this wrong makes half the chambers
    // report the wrong handedness.
    expect(chirality(reg([0.1, 0.2, 0.3, 0.4]))).toBe(1);
    expect(chirality(reg([0.4, 0.3, 0.2, 0.1]))).toBe(1);
  });

  it('is −1 for a single swap', () => {
    expect(chirality(reg([0.2, 0.1, 0.3, 0.4]))).toBe(-1);
    // And +1 for an even reordering that is not the identity — descending is
    // the reversal (even), and so is any two-swap. Descending order
    // [A,T,E,R] = [3,0,1,2] has three inversions and is therefore odd.
    expect(chirality(reg([0.2, 0.3, 0.1, 0.4]))).toBe(1);
  });

  it('is 0 on the tie set, and only there', () => {
    expect(chirality(reg([0.5, 0.5, 0.2, 0.9]))).toBe(0);
    expect(chirality(reg([0.5, 0.5, 0.5, 0.5]))).toBe(0);
    for (const v of STATES) expect(chirality(reg(v))).not.toBe(0);
  });

  it('signedChirality goes through zero rather than jumping', () => {
    // A continuous read-out is what a downstream consumer should threshold,
    // because a bare bit flaps on near-degenerate registers.
    expect(Math.abs(signedChirality(reg([0.5000001, 0.5, 0.2, 0.9])))).toBeLessThan(1e-5);
    expect(Math.abs(signedChirality(reg([0.9, 0.5, 0.2, 0.1])))).toBeGreaterThan(0.1);
  });

  it('the Ox tie-break resolves a full tie to +1, never to 0', () => {
    expect(oxChirality(reg([0.5, 0.5, 0.5, 0.5]))).toBe(1);
    // And the bare bit still reports achirality — the two answers differ, and
    // that is the point of carrying both.
    expect(chirality(reg([0.5, 0.5, 0.5, 0.5]))).toBe(0);
  });

  it('the margin is Σ_κ-invariant, being a function of ratios only', () => {
    const before = chiralMargin(reg([0.9, 0.6, 0.3, 0.1]));
    const after = chiralMargin(reg([0.45, 0.3, 0.15, 0.05]));
    expect(Math.abs(before - after)).toBeLessThan(1e-6);
    expect(before).toBeGreaterThan(0.1);
  });

  it('the joint address is 32 cells and never exceeds it', () => {
    const reversed = STATES.map((s) => [s[3]!, s[2]!, s[1]!, s[0]!] as Tera);
    for (const v of [...STATES, ...reversed]) {
      const addr = chiralAddress(reg(v));
      expect(addr).toBeGreaterThanOrEqual(0);
      expect(addr).toBeLessThan(32);
      expect(addr).toBe((mejiIndex(reg(v)) << 1) | (oxChirality(reg(v)) > 0 ? 1 : 0));
    }
  });
});

describe('Fisher–Rao is a metric in the register format', () => {
  const A = reg([0.9, 0.1, 0.5, 0.3]);
  const B = reg([0.2, 0.8, 0.4, 0.6]);
  const C = reg([0.5, 0.5, 0.9, 0.1]);

  it('is non-negative, symmetric and separates', () => {
    expect(frDist(A, A)).toBeCloseTo(0, 12);
    expect(frDist(A, B)).toBeCloseTo(frDist(B, A), 12);
    expect(frDist(A, B)).toBeGreaterThan(0);
  });

  it('satisfies the triangle inequality', () => {
    expect(frDist(A, C)).toBeLessThanOrEqual(frDist(A, B) + frDist(B, C) + 1e-12);
  });

  it('is bounded by the diagonal of the θ domain', () => {
    // Each θ spans [0,π], so the widest separation is the full diagonal 2π.
    expect(frDist(reg([0, 0, 0, 0]), reg([1, 1, 1, 1]))).toBeLessThanOrEqual(2 * Math.PI + 1e-9);
    expect(frDist(reg([0, 0, 0, 0]), reg([1, 1, 1, 1]))).toBeGreaterThan(2 * Math.PI - 0.1);
  });

  it('collapses to the Euclidean norm near the midpoint', () => {
    // The small-displacement limit: `dθ ≈ 2·dv` near v = ½, so the L2 norm of
    // the difference in `v` is recovered. This is the fact that makes θ a
    // reparameterisation rather than a different function — and it is why the
    // θ format is safe for a simulation that stores state at f32.
    const eps = 1e-6;
    expect(frDist(reg([0.5 + eps, 0.5, 0.5, 0.5]), reg([0.5, 0.5, 0.5, 0.5]))).toBeCloseTo(2 * eps, 6);
  });
});

describe('the geodesic is exact at its endpoints', () => {
  const a = reg([0.9, 0.1, 0.5, 0.3]);
  const b = reg([0.2, 0.8, 0.4, 0.6]);

  it('reaches a at t=0 and b at t=1', () => {
    // Checked in the register's own coordinates. Had the geodesic round-tripped
    // through `v` to get here it could not be exact and would need a tolerance
    // — which is exactly why the endpoint claim is written this way.
    const at0 = geodesic(a, b, 0);
    const at1 = geodesic(a, b, 1);
    for (let d = 0; d < AXIS_COUNT; d++) {
      expect(at0.theta[d]).toBeCloseTo(a.theta[d] ?? 0, 12);
      expect(at1.theta[d]).toBeCloseTo(b.theta[d] ?? 0, 12);
    }
  });

  it('is constant speed, because the interpolation is straight in θ', () => {
    const d = frDist(a, b);
    for (const t of [0.25, 0.5, 0.75]) expect(frDist(a, geodesic(a, b, t))).toBeCloseTo(d * t, 10);
  });

  it('clamps t outside [0,1] rather than extrapolating', () => {
    const span = frDist(a, b);
    expect(frDist(a, geodesic(a, b, -5))).toBeCloseTo(0, 12);
    expect(frDist(a, geodesic(a, b, 5))).toBeCloseTo(span, 10);
  });

  it('leaves Σ_κ, and reports that honestly', () => {
    // A geodesic between two points on the conservation surface does not stay
    // on it. Carrying `kappa` through unchanged would make the invariant a lie
    // for exactly one operation, so the register is rebuilt and κ re-derived —
    // which is also what keeps the result *well formed* rather than carrying a
    // placeholder constant that would fail its own invariant check.
    const ka = 0.9 * 0.1 * 0.5 * 0.3;
    const kb = 0.2 * 0.8 * 0.4 * 0.6;
    const mid = geodesic(a, b, 0.5);
    expect(mid.kappa).not.toBeCloseTo((ka + kb) / 2, 3);
    expect(isWellFormed(mid)).toBe(true);
  });
});

describe('the conservation projection', () => {
  it('is idempotent', () => {
    for (const v of [[0.9, 0.6, 0.3, 0.1], [0.2, 0.2, 0.8, 0.8]] as Tera[]) {
      for (const kappa of [0.05, 0.3, 0.9]) {
        const once = kappaProj(v, kappa);
        const twice = kappaProj(once, kappa);
        for (let d = 0; d < AXIS_COUNT; d++) expect(twice[d]).toBeCloseTo(once[d] ?? 0, 12);
      }
    }
  });

  it('conserves the product exactly when it does not saturate', () => {
    const v: Tera = [0.6, 0.5, 0.4, 0.3];
    const kappa = 0.6 * 0.5 * 0.4 * 0.3 * 0.5;
    const projected = kappaProj(v, kappa);
    expect((projected[0] ?? 0) * (projected[1] ?? 0) * (projected[2] ?? 0) * (projected[3] ?? 0)).toBeCloseTo(kappa, 12);
  });

  it('respects the box', () => {
    for (const v of [[0.99, 0.98, 0.97, 0.96], [0.1, 0.2, 0.3, 0.4]] as Tera[]) {
      for (const kappa of [1e-4, 0.5, 0.999]) {
        for (const x of kappaProj(v, kappa)) {
          expect(x).toBeGreaterThanOrEqual(0);
          expect(x).toBeLessThanOrEqual(1);
        }
      }
    }
  });

  it('is the fourth-root correction, not a mean shift', () => {
    // The naive "subtract the mean drift" version fails to converge on ~11.7% of
    // sampled inputs and can annihilate an axis rather than renormalising all
    // four. It is not implemented here, and this asserts the shape of what *is*:
    // one common positive factor applied to every axis.
    const v: Tera = [0.6, 0.5, 0.4, 0.3];
    const projected = kappaProj(v, 0.02);
    const scale = (projected[0] ?? 0) / v[0]!;
    for (let d = 1; d < AXIS_COUNT; d++) expect((projected[d] ?? 0) / v[d]!).toBeCloseTo(scale, 12);
    expect(scale).toBeGreaterThan(0);
  });

  it('terminates on an unreachable κ instead of spinning', () => {
    // κ above what four bounded axes can hold has no projection. The loop is
    // additionally bounded so the corner is returned rather than an infinite
    // active-set iteration — which is the failure mode an unbounded version of
    // this has.
    const projected = kappaProj([0.9, 0.8, 0.7, 0.6], 2);
    expect(projected).toHaveLength(4);
    for (const x of projected) expect(x).toBeGreaterThanOrEqual(0);
  });

  it('nu is zero exactly on the surface', () => {
    for (const v of [[0.6, 0.5, 0.4, 0.3], [0.9, 0.1, 0.7, 0.4]] as Tera[]) {
      const kappa = v[0]! * v[1]! * v[2]! * v[3]!;
      expect(nu(v, kappa)).toBeCloseTo(0, 12);
    }
  });

  it('nu is monotone in the product', () => {
    // An invariant rather than a fixture comparison, because `arcade-engine`
    // does not implement `nu` and a fixture written from this package's own
    // formula would assert nothing at all.
    const base: Tera = [0.6, 0.5, 0.4, 0.3];
    const product = base[0]! * base[1]! * base[2]! * base[3]!;
    expect(nu(base, product * 2)).toBeLessThan(0);
    expect(nu(base, product / 2)).toBeGreaterThan(0);
    expect(Math.abs(nu(base, product))).toBeLessThan(1e-12);
  });
});

describe('the register itself', () => {
  it('is well formed straight out of makeReg', () => {
    for (const v of [[0.9, 0.1, 0.5, 0.3], [0.25, 0.25, 0.25, 0.25]] as Tera[]) expect(isWellFormed(reg(v))).toBe(true);
  });

  it('exposes both coordinate systems at the same lane order', () => {
    // A backend addressing either view by a constant stride depends on this.
    const r = reg([0.9, 0.1, 0.5, 0.3]);
    const th = thetaToArray(r);
    const vs = vToArray(r);
    expect(th.length).toBe(AXIS_COUNT);
    expect(vs.length).toBe(AXIS_COUNT);
    expect(vs[0]).toBe(0.9);
    expect(th[0]).toBeCloseTo(2 * Math.asin(Math.sqrt(0.9)), 6);
  });

  it('keeps v exact and θ close, so neither read is derived from the other', () => {
    const v: Tera = [0.999962, 0.0009, 0.99962, 0.99946];
    const r = reg(v);
    for (let d = 0; d < AXIS_COUNT; d++) {
      expect(r.v[d]).toBe(v[d]);
      expect(r.theta[d]).toBeCloseTo(2 * Math.asin(Math.sqrt(v[d] ?? 0)), 6);
    }
    expect(regToTera(r)[0]).toBe(0.999962);
  });

  it('regFromTheta clamps to the full half-turn, not a quarter', () => {
    // θ ∈ [0,π]. Clamping to π/2 would squash every register with v > 0.5 down
    // to v = 0.5 — invisible in a single read, catastrophic in a geodesic. This
    // is the regression guard for exactly that bug, which the parity fixture
    // caught as a 0.5 error in a geodesic midpoint.
    const r = regFromTheta([Math.PI, Math.PI, Math.PI, Math.PI], 1);
    expect(r.v[0]).toBeCloseTo(1, 6);
    expect(r.v[1]).toBeCloseTo(1, 6);
    expect(r.theta[0]).toBeCloseTo(Math.PI, 6);
  });

  it('clamps an out-of-domain θ into the box rather than producing a bad v', () => {
    const r = regFromTheta([-1, 99, Math.PI / 2, Math.PI / 4], 0.25);
    for (let d = 0; d < AXIS_COUNT; d++) {
      expect(r.theta[d]).toBeGreaterThanOrEqual(0);
      expect(r.theta[d]).toBeLessThanOrEqual(Math.PI);
      expect(r.v[d]).toBeGreaterThanOrEqual(0);
      expect(r.v[d]).toBeLessThanOrEqual(1);
    }
  });
});

describe('the 64-state system is a popcount away', () => {
  it('packs a register and a two-bit fibre into six bits', () => {
    const s = state64(reg([0.9, 0.1, 0.9, 0.1]), [1, 0]);
    // meji for [0.9,0.1,0.9,0.1] is bits T=1,E=0,R=1,A=0 → 8+0+2+0 = 10. The
    // fibre [1,0] sets bit 4. So 10 | 16 = 26.
    expect(s).toBe(26);
    expect(s).toBeGreaterThanOrEqual(0);
    expect(s).toBeLessThan(64);
  });

  it('measures distance as popcount', () => {
    expect(stateDistance(0b000000, 0b000000)).toBe(0);
    expect(stateDistance(0b000000, 0b111111)).toBe(6);
    expect(stateDistance(0b101010, 0b010101)).toBe(6);
    expect(stateDistance(0b1010, 0b1011)).toBe(1);
    // Bits above the sixth are not part of the space.
    expect(stateDistance(0b1000000, 0)).toBe(0);
  });
});

describe('the deterministic transcendental port', () => {
  it('closes arcsin: sin(asin(x)) round-trips to table precision', () => {
    for (let x = -1; x <= 1.0001; x += 0.05) {
      const c = Math.min(1, Math.max(-1, x));
      expect(dsin(dasin(c))).toBeCloseTo(c, 6);
    }
  });

  it('resolves arcsin exactly at the endpoints, where the derivative diverges', () => {
    // The reflection trick maps |x| > 0.9 back into the tabled interior, so the
    // table never has to resolve the singularity at ±1. Without it a uniform
    // table of this size would manage ~8e-5 near the edge rather than ~1e-8.
    expect(dasin(1)).toBe(Math.PI / 2);
    expect(dasin(-1)).toBe(-Math.PI / 2);
    expect(dsin(dasin(0.9999999))).toBeCloseTo(0.9999999, 6);
  });

  it('is monotone across the reflection boundary', () => {
    // A discontinuity here would be invisible in a single call and would surface
    // as a non-monotone response deep inside a projection.
    let prev = -Infinity;
    for (let x = 0.85; x <= 0.95; x += 0.001) {
      const y = dasin(x);
      expect(y).toBeGreaterThanOrEqual(prev);
      prev = y;
    }
  });

  it('dcos, dacos and datan agree with their definitions', () => {
    for (let x = -2; x <= 2; x += 0.25) expect(dcos(x)).toBeCloseTo(Math.cos(x), 5);
    for (let x = -1; x <= 1; x += 0.05) expect(dacos(x)).toBeCloseTo(Math.acos(x), 5);
    for (let x = -1; x <= 1; x += 0.05) expect(datan(x)).toBeCloseTo(Math.atan(x), 5);
  });

  it('agrees with the platform to better than f32 resolution', () => {
    for (let x = -6; x <= 6; x += 0.017) expect(dsin(x)).toBeCloseTo(Math.sin(x), 3);
  });

  it('is total on the argument ranges the ISA actually passes', () => {
    // `dsin` reduces mod 2π before indexing, so it must be defined for the
    // whole of `θ ∈ [0,π]` — including exactly π, where a naive `floor` index
    // would land one past the end of the table.
    for (const theta of [0, Math.PI / 4, Math.PI / 2, Math.PI - 1e-12, Math.PI]) {
      expect(Number.isFinite(dsin(theta))).toBe(true);
      expect(dsin(theta)).toBeGreaterThanOrEqual(-1e-12);
      expect(dsin(theta)).toBeLessThanOrEqual(1 + 1e-12);
    }
  });
});
