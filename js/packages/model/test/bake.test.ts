// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { inspectAamt } from '../src/amdl-inspect';
import { bakeSafetensorsFile, kindFor } from '../src/bake';
import { parseModel } from '../src/model';
import { dequantCPU } from '../src/tensor';

const dir = mkdtempSync(join(tmpdir(), 'aamt-bake-'));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** Write a real safetensors file: 8-byte header length, JSON header, then the data section. */
function writeSafetensors(path: string, tensors: { name: string; dtype: 'F32' | 'BF16'; shape: number[]; data: Float32Array }[]) {
  const parts: Uint8Array[] = [];
  const header: Record<string, unknown> = { __metadata__: { format: 'pt' } };
  let off = 0;
  for (const t of tensors) {
    let bytes: Uint8Array;
    if (t.dtype === 'F32') bytes = new Uint8Array(t.data.slice().buffer);
    else {
      // bf16 is the top 16 bits of the f32
      const u = new Uint32Array(t.data.slice().buffer);
      const h = new Uint16Array(u.length);
      for (let i = 0; i < u.length; i++) h[i] = u[i]! >>> 16;
      bytes = new Uint8Array(h.buffer);
    }
    header[t.name] = { dtype: t.dtype, shape: t.shape, data_offsets: [off, off + bytes.byteLength] };
    parts.push(bytes);
    off += bytes.byteLength;
  }
  const json = new TextEncoder().encode(JSON.stringify(header));
  const len = new Uint8Array(8);
  new DataView(len.buffer).setUint32(0, json.byteLength, true);
  writeFileSync(path, Buffer.concat([len, json, ...parts]));
}

const ramp = (n: number) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.37) * 0.25);

describe('bakeSafetensorsFile', () => {
  const src = join(dir, 'tiny.safetensors');
  writeSafetensors(src, [
    { name: 'layers.0.mlp.weight', dtype: 'BF16', shape: [8, 64], data: ramp(512) },
    { name: 'layers.0.norm.weight', dtype: 'F32', shape: [64], data: ramp(64) },
    { name: 'layers.0.odd.weight', dtype: 'F32', shape: [4, 20], data: ramp(80) },
  ]);

  it('stores matrices in the requested kind and keeps small or odd tensors f32', () => {
    const out = join(dir, 'tiny-1bit.aamt');
    const s = bakeSafetensorsFile(src, out, { kind: '1bit' });
    expect(s.tensors).toBe(3);
    expect(s.kinds).toEqual({ '1bit': 1, f32: 2 });
    const bytes = readFileSync(out);
    const m = parseModel(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    expect(m.manifest.map((e) => [e.name, e.kind])).toEqual([
      ['layers.0.mlp.weight', '1bit'],
      ['layers.0.norm.weight', 'f32'],
      ['layers.0.odd.weight', 'f32'],
    ]);
    expect(Array.from(m.getRaw('layers.0.norm.weight').data)).toEqual(Array.from(ramp(64)));
  });

  it('round-trips int4 within one quantisation step of the bf16 source', () => {
    const out = join(dir, 'tiny-int4.aamt');
    bakeSafetensorsFile(src, out);
    const bytes = readFileSync(out);
    const { parsed } = parseModel(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)).get('layers.0.mlp.weight');
    const back = dequantCPU(parsed);
    const ref = ramp(512);
    for (let i = 0; i < 512; i++) {
      const step = parsed.scales[(i / parsed.groupSize) | 0]!;
      expect(Math.abs(back[i]! - ref[i]!)).toBeLessThanOrEqual(step * 0.5 + 2e-3); // half a step, plus bf16 rounding
    }
  });

  it('writes a file the AMDL inspector accepts', async () => {
    const out = join(dir, 'tiny-ternary.aamt');
    bakeSafetensorsFile(src, out, { kind: 'ternary' });
    const r = await inspectAamt(out);
    expect(r.ok).toBe(true);
    expect(r.tensorCount).toBe(3);
    expect(r.kinds).toEqual({ ternary: 1, f32: 2 });
  });

  it('kindFor: only matrices with group-aligned rows are quantised', () => {
    expect(kindFor([4096, 4096], 'int4', 32)).toBe('int4');
    expect(kindFor([4096], 'int4', 32)).toBe('f32');
    expect(kindFor([10, 20], 'int4', 32)).toBe('f32');
  });
});
