// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { buildAmdlFixture } from '@aamt/model/amdl-inspect';
import { bake } from '@aamt/model/tensor';
import { packContainer, sniffAamt, sniffBytes } from '../src/index';
import { nodeSha384 } from '../src/node';

/**
 * Every fixture here comes from the real writer of its format:
 * - substrate-v0.aamt was packed by the native CLI
 *   (`aamt pack test/fixtures/substrate-v0.tsv test/fixtures/substrate-v0.aamt 4 2`),
 * - tensor blobs come from `@aamt/model`'s `bake`,
 * - the AMDL file from `@aamt/model`'s `buildAmdlFixture`.
 */
const substrate = new Uint8Array(readFileSync(new URL('./fixtures/substrate-v0.aamt', import.meta.url)));
const weights = Float32Array.from({ length: 100 }, (_, i) => Math.sin(i) * 0.5);

describe('sniffAamt: the container and all three legacy formats', () => {
  it('classifies the new container', () => {
    expect(sniffBytes(packContainer({ layers: [{ kind: 'code', records: [] }] }, nodeSha384))).toEqual({
      format: 'container',
      versionMajor: 1,
      versionMinor: 2,
    });
  });

  it('classifies an AMDL model file by its footer', () => {
    expect(
      sniffBytes(
        buildAmdlFixture([
          { name: 'a', shape: [4], kind: 'int4' },
          { name: 'b', shape: [4], kind: 'f32' },
        ]),
      ),
    ).toEqual({ format: 'amdl', version: 1, tensorCount: 2 });
  });

  it('classifies a real libaamt substrate', () => {
    expect(sniffBytes(substrate)).toEqual({ format: 'substrate-v0', versionMajor: 1, versionMinor: 0 });
  });

  it('classifies real tensor blobs of every kind', () => {
    for (const kind of ['int4', 'ternary', '1bit'] as const) {
      expect(sniffBytes(new Uint8Array(bake(weights, 32, kind)))).toEqual({
        format: 'tensor-blob',
        version: 2,
        kind,
      });
    }
  });

  it('separates the collision ADR-005 exists for: a v1 blob and a v1.0 substrate share their first 8 bytes', () => {
    // voxel-runtime v1 layout: magic, version 1, count, groupSize, groupCount, scales…, packed…
    const v2 = new Uint8Array(bake(weights, 32, 'int4'));
    const h = new Uint32Array(v2.buffer, 0, 6);
    const v1 = new Uint8Array(v2.byteLength - 4);
    const v1h = new Uint32Array(v1.buffer);
    v1h.set([h[0]!, 1, h[3]!, h[4]!, h[5]!]);
    v1.set(v2.subarray(24), 20);

    expect([...v1.subarray(0, 8)]).toEqual([...substrate.subarray(0, 8)]);
    expect(sniffBytes(v1)).toEqual({ format: 'tensor-blob', version: 1, kind: 'int4' });
    expect(sniffBytes(substrate).format).toBe('substrate-v0');
  });

  it('needs only the head and tail, so a 7 GB model is never read whole', () => {
    const amdl = buildAmdlFixture([{ name: 'a', shape: [4], kind: 'int4', bytes: 4096 }]);
    expect(sniffAamt(amdl.subarray(0, 64), amdl.subarray(amdl.byteLength - 32), amdl.byteLength).format).toBe(
      'amdl',
    );
  });

  it('reports why a file is not recognised', () => {
    expect(sniffBytes(new Uint8Array(100))).toMatchObject({ format: 'unknown' });
    const truncated = packContainer({ layers: [] }, nodeSha384).subarray(0, 100);
    expect(sniffBytes(truncated)).toMatchObject({
      format: 'unknown',
      reason: expect.stringMatching(/truncated/),
    });
    const lying = substrate.slice(0, 4000);
    expect(sniffBytes(lying)).toMatchObject({
      format: 'unknown',
      reason: expect.stringMatching(/neither the substrate nor the tensor-blob/),
    });
  });
});
