// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * `@aamt/container` — the unified `.aamt` container (ADR-005, Yare design §7).
 *
 * One immutable, mmap-able file; a layer table that says which region is model
 * data, world space, the mind sandbox, admitted experience or compiled code; and
 * 384-bit layer-separated identities, so the same bytes can never be mistaken
 * for one another across layers.
 *
 * - {@link packContainer} — deterministic writer.
 * - {@link writeContainer} — the same bytes, streamed to a sink from re-readable sources (ADR-017).
 * - {@link openContainer} — zero-copy reader; O(layers) to open.
 * - {@link verifyContainer} — recompute every id and invariant.
 * - {@link sniffAamt} — classify the container and all three legacy `.aamt` formats.
 * - {@link oduCellsWithin} and `LayerView.odu` — the Odu index section (v1.2, ADR-018):
 *   a search structure over manifold layers, never a storage address.
 *
 * The core's one runtime import is `@aamt/tera` — itself pure and
 * dependency-free — for the Odu cell function, so there is exactly one definition
 * of it. Hashing is a port ({@link Sha384}); `@aamt/container/node`
 * supplies the `node:crypto` binding.
 */

export * from './layout';
export * from './identity';
export * from './writer';
export * from './stream';
export * from './reader';
export * from './sniff';
export * from './crypto';
export * from './odu';
export * from './tree';
