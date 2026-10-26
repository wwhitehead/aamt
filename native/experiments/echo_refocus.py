#!/usr/bin/env python3
# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""Echo-refocusing: does a single pi-pulse over TERA space cancel drift,
and precisely when.

Tier 3 of the mirror-magic research pass ("phase-conjugate mirrors") was
graded research-only rather than buildable, because the physical claim
underneath it — a distorted wavefront retraces its exact path and the
distortion cancels on the way back — has a real precondition. A phase-
conjugate mirror (nonlinear optics) and a Hahn spin echo (NMR) both
refocus DISTORTION PICKED UP FROM A STATIC, RETRACED CHANNEL: the same
medium, the same field offset, acting the same way on the way back. They
do NOT refocus genuinely random, memoryless noise — that is the textbook
distinction between T2* (inhomogeneous, static, echo-recoverable dephasing)
and T2 (homogeneous, stochastic, irrecoverable dephasing) in NMR. Before
building anything, this script tests that precondition directly, in this
architecture's own coordinate space, rather than assuming the optics
metaphor carries over.

The operator under test is a point inversion about a target center,
x -> 2*center - x. This is the correct computational analog of a Hahn
echo's 180-degree pulse: an involution that preserves distance to the
center, exactly the property a spin's phase-negation has. It is
deliberately NOT called a Householder reflection: WP-23 section 10.1 is
precise that flipping a coordinate about a point rather than a hyperplane
is orientation-preserving in even dimension (det = +1, a rotation) not a
true mirror (det = -1) — the same distinction applies here, and getting it
right is the whole reason section 10 exists as a callout in the first
place. Conflating the two would repeat exactly the error that section
already corrected once.

Two regimes, both real random-walk models over the 4D TERA cube:

  coherent   -- each trial draws ONE persistent bias vector (the "static
                medium"), added to every step. Analogous to a spin's local
                field offset: the same distorting influence, retraced.
  incoherent -- zero-mean i.i.d. noise only, no persistent structure.
                Analogous to genuinely random dephasing.

And two space conditions:

  unconstrained -- walk lives in R^4, no boundary. Proves the mechanism
                   cleanly, matching the derivation.
  tera_clamped  -- walk is clamped into [0,1]^4 after every step (the
                   substrate's own clampTera4 rule), so the boundary's
                   nonlinearity is present exactly as it would be for any
                   real TERA-space application of this operator.

Prediction (falsifiable, stated before running): the pulse should
substantially reduce final distance-to-target in the coherent regime and
should NOT measurably change it in the incoherent regime, in both space
conditions (the boundary can only ever help by construction — it can clip
excursions closer to the cube, never add drift the free-space model
lacks). If the incoherent regime instead shows a large pulse effect, the
whole analogy is broken and would need to be withdrawn, the same way
WP-22's escape-steering claim was withdrawn on inspection.

Run: python3 echo_refocus.py
"""
import numpy as np

DIM = 4
STEPS = 40
PULSE_AT = STEPS // 2
N_TRIALS = 50_000
NOISE_SIGMA = 0.01
BIAS_SIGMA = 0.01
SEED = 20260708


def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def point_invert(x, center):
    """x -> 2*center - x: the pi-pulse. An involution, isometric about
    `center` (|invert(x)-center| == |x-center| exactly, unclamped)."""
    return 2.0 * center - x


def simulate(regime: str, pulse: bool, clamp: bool, rng: np.random.Generator):
    """Vectorized over N_TRIALS simultaneous walks. Returns final
    Euclidean distance-to-target per trial, shape (N_TRIALS,)."""
    start = np.full((N_TRIALS, DIM), 0.5)
    target = start.copy()  # each trial's own start is its own target
    x = start.copy()

    bias = np.zeros((N_TRIALS, DIM))
    if regime == "coherent":
        bias = rng.normal(0.0, BIAS_SIGMA, size=(N_TRIALS, DIM))
    elif regime != "incoherent":
        raise ValueError(regime)

    for t in range(STEPS):
        noise = rng.normal(0.0, NOISE_SIGMA, size=(N_TRIALS, DIM))
        x = x + bias + noise
        if clamp:
            x = clamp01(x)
        if pulse and t == PULSE_AT:
            x = point_invert(x, target)
            if clamp:
                x = clamp01(x)

    return np.linalg.norm(x - target, axis=1)


def summarize(label, distances):
    mean = distances.mean()
    stderr = distances.std(ddof=1) / np.sqrt(len(distances))
    return mean, stderr


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for space, clamp in [("unconstrained", False), ("tera_clamped", True)]:
        for regime in ["coherent", "incoherent"]:
            d_pulse = simulate(regime, True, clamp, rng)
            d_control = simulate(regime, False, clamp, rng)
            m_p, se_p = summarize("pulse", d_pulse)
            m_c, se_c = summarize("control", d_control)
            z = (m_c - m_p) / np.sqrt(se_p**2 + se_c**2)
            reduction_pct = 100.0 * (m_c - m_p) / m_c
            rows.append((space, regime, m_p, se_p, m_c, se_c, reduction_pct, z))

    header = (f"{'space':<14}{'regime':<12}{'pulse mean':>12}{'±se':>9}"
              f"{'control mean':>14}{'±se':>9}{'reduction%':>12}{'z':>9}")
    print(header)
    print("-" * len(header))
    for space, regime, m_p, se_p, m_c, se_c, red, z in rows:
        print(f"{space:<14}{regime:<12}{m_p:>12.5f}{se_p:>9.5f}"
              f"{m_c:>14.5f}{se_c:>9.5f}{red:>11.2f}%{z:>9.2f}")

    print(f"\nN_TRIALS={N_TRIALS} per cell, STEPS={STEPS}, pulse at step "
          f"{PULSE_AT}, NOISE_SIGMA={NOISE_SIGMA}, BIAS_SIGMA={BIAS_SIGMA}, "
          f"seed={SEED}")

    coherent_reductions = [r[6] for r in rows if r[1] == "coherent"]
    incoherent_reductions = [r[6] for r in rows if r[1] == "incoherent"]
    coherent_ok = all(r > 50.0 for r in coherent_reductions)
    incoherent_ok = all(abs(r) < 5.0 for r in incoherent_reductions)

    print(f"\nPrediction check: coherent regime reduction >50% in both space "
          f"conditions: {coherent_ok} ({[f'{r:.1f}%' for r in coherent_reductions]})")
    print(f"Prediction check: incoherent regime reduction within +/-5% (no "
          f"real effect) in both space conditions: {incoherent_ok} "
          f"({[f'{r:.1f}%' for r in incoherent_reductions]})")

    if coherent_ok and incoherent_ok:
        print("\nPASS: the pi-pulse refocuses coherent (structured) drift and "
              "has no measurable effect on incoherent (random) drift, in "
              "both free space and the substrate's clamped [0,1]^4 cube — "
              "the precondition the phase-conjugate-mirror analogy requires "
              "holds for THIS operator, on THIS coordinate space.")
    else:
        print("\nFAIL: prediction not met — see table above.")


if __name__ == "__main__":
    main()
