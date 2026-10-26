// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
import { describe, expect, it } from 'vitest';
import { bake, parse, type AamtKind } from '@aamt/model/tensor';
import { nodeDevice } from '../src/device';
import { Gemv, uploadMatrix } from '../src/gemv';
import { gemvReference, relativeError, seeded } from '../src/reference';

// CI runners have no GPU; this suite runs wherever an adapter exists and is skipped elsewhere.
const gpu = await nodeDevice().catch(() => null);

describe.skipIf(!gpu)('GEMV on the GPU equals the CPU, straight from .aamt bytes', () => {
  // Shapes chosen to cross the kernel's edges: rows not a multiple of 16, K spanning several 2048-wide x tiles.
  const shapes: [number, number][] = [
    [16, 32],
    [37, 96],
    [256, 4096],
    [1000, 5120],
  ];
  for (const kind of ['int4', 'ternary', '1bit'] as AamtKind[]) {
    for (const [m, k] of shapes) {
      it(`${kind} ${m}×${k}`, async () => {
        const t = parse(bake(seeded(m * k, m + k), 32, kind));
        const x = seeded(k, 7);
        const y = await new Gemv(gpu!.device).run(uploadMatrix(gpu!.device, t, m, k), x);
        expect(relativeError(y, gemvReference(t, m, k, x))).toBeLessThan(1e-5);
      });
    }
  }

  it('refuses a K the kernel cannot tile', () => {
    const t = parse(bake(seeded(4 * 40, 1), 8, 'int4'));
    expect(() => uploadMatrix(gpu!.device, t, 4, 40)).toThrow(/multiple of 32/);
  });
});
