// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * The GPU check: multiply `.aamt` weights on the GPU, straight from the file's
 * bytes, and compare against the CPU.
 *
 *   pnpm gpu:check                                   # seeded 4096 × 4096, baked once per kind
 *   AAMT_M=8192 AAMT_K=8192 pnpm gpu:check
 *   AAMT_FILE=model.aamt AAMT_TENSOR=model.layers.0.mlp.down_proj.weight pnpm gpu:check
 *
 * The seeded mode bakes a matrix into a real `.aamt` file per weight kind and
 * reads the file back. The file mode reads one tensor of a baked model by
 * offset — footer, manifest, tensor; nothing else of the model is loaded.
 *
 * Prints: file size, the GPU's relative error against the CPU, weights
 * multiplied per second, and the weight bandwidth the kernel achieved (bytes of
 * packed weights + f16 scales read per product, divided by the time per product
 * over AAMT_ITERS products). Fewer bits per weight means fewer bytes moved, so a
 * low-bit kind can be faster per product while reporting lower GB/s.
 */
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { bakeModel, parseModel, readTensorFromFile } from '@aamt/model/model';
import { parse, type AamtKind, type AamtParsed } from '@aamt/model/tensor';
import { nodeDevice } from './device';
import { Gemv, uploadMatrix } from './gemv';
import { gemvReference, relativeError, seeded } from './reference';

const ITERS = Number(process.env.AAMT_ITERS ?? 200);
/** Float32 accumulation over K terms against an f64 reference; 1e-4 relative is far looser than what passes. */
const TOLERANCE = 1e-4;

const gpu = await nodeDevice();
if (!gpu) {
  console.error('no WebGPU adapter available');
  process.exit(1);
}
const { device, adapterName } = gpu;
const gemv = new Gemv(device);
let failed = false;

async function measure(label: string, t: AamtParsed, M: number, K: number, fileBytes: number): Promise<void> {
  const x = seeded(K, 2);
  const mat = uploadMatrix(device, t, M, K);
  const err = relativeError(await gemv.run(mat, x), gemvReference(t, M, K, x));
  if (!(err <= TOLERANCE)) failed = true;

  const { encode } = gemv.prepare(mat, x);
  const warm = device.createCommandEncoder();
  for (let i = 0; i < 10; i++) encode(warm);
  device.queue.submit([warm.finish()]);
  await device.queue.onSubmittedWorkDone();
  const enc = device.createCommandEncoder();
  for (let i = 0; i < ITERS; i++) encode(enc);
  const t0 = performance.now();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  const s = (performance.now() - t0) / 1000 / ITERS;

  const bits = (mat.weightBytes * 8) / (M * K);
  console.log(
    `${label.padEnd(9)} ${`${(fileBytes / 1e6).toFixed(2)} MB`.padEnd(11)} ${bits.toFixed(2).padEnd(12)} ` +
      `${err.toExponential(2).padEnd(12)} ${`${(s * 1e6).toFixed(1)} µs`.padEnd(14)} ` +
      `${`${((M * K) / s / 1e9).toFixed(0)} G`.padEnd(15)} ${(mat.weightBytes / s / 1e9).toFixed(1)}` +
      (err <= TOLERANCE ? '' : '   FAIL'),
  );
}

const HEAD = 'kind      file        bits/weight  rel. error   time/product   weights/s       weight GB/s';
console.log(`adapter: ${adapterName}`);

if (process.env.AAMT_FILE) {
  const name = process.env.AAMT_TENSOR;
  if (!name) throw new Error('AAMT_FILE needs AAMT_TENSOR=<tensor name>');
  // `pnpm gpu:check` runs from the package directory; INIT_CWD is where the user typed the command.
  const file = resolve(process.env.INIT_CWD ?? process.cwd(), process.env.AAMT_FILE);
  const { entry, blob } = readTensorFromFile(file, name);
  if (entry.kind === 'f32' || entry.shape.length !== 2) throw new Error(`"${name}" is ${entry.kind} ${entry.shape.join('×')}; pick a quantised matrix`);
  const [M, K] = entry.shape as [number, number];
  console.log(`tensor:  ${name} — ${M} × ${K}, read by offset from ${process.env.AAMT_FILE}\n`);
  console.log(HEAD);
  await measure(entry.kind ?? 'int4', parse(blob), M, K, blob.byteLength);
} else {
  const M = Number(process.env.AAMT_M ?? 4096);
  const K = Number(process.env.AAMT_K ?? 4096);
  console.log(`matrix:  ${M} × ${K} (${((M * K) / 1e6).toFixed(1)}M weights), seeded, ${ITERS} timed products\n`);
  console.log(HEAD);
  const dir = mkdtempSync(join(tmpdir(), 'aamt-check-'));
  const weights = seeded(M * K, 1);
  for (const kind of ['int4', 'ternary', '1bit'] as AamtKind[]) {
    const path = join(dir, `w-${kind}.aamt`);
    writeFileSync(path, new Uint8Array(bakeModel([{ name: 'w', shape: [M, K], data: weights, kind }])));
    const bytes = readFileSync(path);
    const { parsed } = parseModel(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)).get('w');
    await measure(kind, parsed, M, K, bytes.byteLength);
  }
  rmSync(dir, { recursive: true, force: true });
}

device.destroy();
console.log(failed ? `\nFAIL: GPU disagrees with CPU beyond ${TOLERANCE}` : `\nGPU == CPU (relative error ≤ ${TOLERANCE})`);
process.exit(failed ? 1 : 0);
