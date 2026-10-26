// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * `@aamt/tera` — the coordinate math every `.aamt` index is built on.
 *
 * A point is a TERA register: four axes in [0, 1]. Splitting all four at their
 * midpoints gives 2⁴ = 16 orthants (the Meji); two levels give 16 × 16 = 256
 * cells (the Odu). Distance is the product Fisher–Rao metric. Transcendentals
 * are deterministic, so every surface computes bit-identical cells.
 *
 * Extracted unchanged from the Vortex VM's register file, ALU and deterministic
 * transcendentals, so there is one definition of each function.
 */

export * from './register';
export * from './alu';
export * from './deterministic';
