// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * Writes python/tests/fixtures/{mixed.aamt, mixed.expected.json}: one tensor of every kind, baked by the TypeScript
 * writer, with the TypeScript reader's dequantised values. The Python reader must reproduce them exactly.
 *
 *   pnpm --filter @aamt/model exec tsx scripts/make-python-fixture.ts
 */
import { writeFileSync } from 'node:fs';
import { bakeModel, parseModel } from '../src/model';
import { dequantCPU } from '../src/tensor';

const out = new URL('../../../../python/tests/fixtures/', import.meta.url);
const ramp = (n: number, k: number) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.37 + k) * (0.2 + (i % 7) * 0.05));
const tensors = [
  { name: 'int4', shape: [4, 64], data: ramp(256, 1), kind: 'int4' as const },
  { name: 'ternary', shape: [3, 96], data: ramp(288, 2), kind: 'ternary' as const },
  { name: '1bit', shape: [2, 160], data: ramp(320, 3), kind: '1bit' as const },
  // odd length: the last word and the last group are partial
  { name: 'int4-odd', shape: [37], data: ramp(37, 4), kind: 'int4' as const },
  { name: 'norm', shape: [64], data: ramp(64, 5), kind: 'f32' as const },
];
const buf = bakeModel(tensors, 32);
const model = parseModel(buf);
const expected = Object.fromEntries(
  tensors.map((t) => [t.name, { shape: t.shape, kind: t.kind, values: Array.from(t.kind === 'f32' ? model.getRaw(t.name).data : dequantCPU(model.get(t.name).parsed)) }]),
);
writeFileSync(new URL('mixed.aamt', out), new Uint8Array(buf));
writeFileSync(new URL('mixed.expected.json', out), JSON.stringify(expected) + '\n');
console.log(`wrote ${tensors.length} tensors, ${buf.byteLength} bytes`);
