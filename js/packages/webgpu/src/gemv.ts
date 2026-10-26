// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * y = W·x on the GPU, straight from a `.aamt` tensor's packed words.
 *
 * The kernel (`shaders/aamt_gemv_mr.wgsl`) is the multi-row GEMV from the decode
 * path of a working inference engine: one u32 read unpacks 8 (int4), 16
 * (ternary) or 32 (1-bit) weights in-register, against an activation tile staged
 * once in workgroup memory for 16 output rows. Nothing is parsed at load: the
 * tensor's packed words are uploaded as they sit in the file, and its f32 group
 * scales are halved to f16 pairs on the way (see `@aamt/model/f16`).
 *
 * This wrapper uses the shared-memory reduction, which needs no optional GPU
 * features, so it runs on every WebGPU adapter.
 */
import { readFileSync } from 'node:fs';
import { packF16Pairs } from '@aamt/model/f16';
import type { AamtParsed } from '@aamt/model/tensor';

/** Rows per workgroup baked into the shader as shipped (16 rows × 16 lanes = 256 threads). */
export const ROWS_PER_WG = 16;
const KIND_ID = { int4: 0, ternary: 1, '1bit': 2 } as const;

export const GEMV_WGSL = readFileSync(new URL('./shaders/aamt_gemv_mr.wgsl', import.meta.url), 'utf8');
export const DEQUANT_WGSL = readFileSync(new URL('./shaders/aamt_dequant.wgsl', import.meta.url), 'utf8');

export interface GpuMatrix {
  rows: number;
  cols: number;
  kind: AamtParsed['kind'];
  groupSize: number;
  packed: GPUBuffer;
  scales: GPUBuffer;
  /** Bytes the kernel reads from weight storage for one full product. */
  weightBytes: number;
}

const storage = (device: GPUDevice, data: ArrayBufferView, usage = 0): GPUBuffer => {
  const size = Math.max(16, Math.ceil(data.byteLength / 4) * 4);
  const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | usage });
  device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
  return buf;
};

/** Upload one `.aamt` tensor as an M×K matrix. K must be a multiple of 32. */
export function uploadMatrix(device: GPUDevice, t: AamtParsed, rows: number, cols: number): GpuMatrix {
  if (rows * cols !== t.count) throw new Error(`uploadMatrix: ${rows}×${cols} does not match ${t.count} weights`);
  if (cols % 32 !== 0) throw new Error('uploadMatrix: K must be a multiple of 32');
  if ((t.groupSize & (t.groupSize - 1)) !== 0) throw new Error('uploadMatrix: groupSize must be a power of two');
  const scales16 = packF16Pairs(t.scales);
  return {
    rows,
    cols,
    kind: t.kind,
    groupSize: t.groupSize,
    packed: storage(device, t.packed),
    scales: storage(device, scales16),
    weightBytes: t.packed.byteLength + scales16.byteLength,
  };
}

export class Gemv {
  private readonly pipeline: GPUComputePipeline;

  constructor(private readonly device: GPUDevice) {
    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: GEMV_WGSL }), entryPoint: 'gemv_mr' },
    });
  }

  /** Bind W and x once; returns a function that records one product into an encoder. */
  prepare(w: GpuMatrix, x: Float32Array): { y: GPUBuffer; encode: (enc: GPUCommandEncoder) => void } {
    if (x.length !== w.cols) throw new Error(`gemv: x has ${x.length} elements, W has ${w.cols} columns`);
    const d = this.device;
    const y = d.createBuffer({ size: w.rows * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const nWg = Math.ceil(w.rows / ROWS_PER_WG);
    const gridW = Math.min(nWg, 65535);
    const params = new Uint32Array([w.rows, w.cols, w.groupSize, gridW, KIND_ID[w.kind], Math.log2(w.groupSize), 0, 0]);
    const uniform = d.createBuffer({ size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(uniform, 0, params);
    const pipeline = this.pipeline;
    const bind = d.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: w.packed } },
        { binding: 1, resource: { buffer: w.scales } },
        { binding: 2, resource: { buffer: storage(d, x) } },
        { binding: 3, resource: { buffer: y } },
        { binding: 4, resource: { buffer: uniform } },
      ],
    });
    return {
      y,
      encode: (enc) => {
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(gridW, Math.ceil(nWg / gridW));
        pass.end();
      },
    };
  }

  /** One product, read back. */
  async run(w: GpuMatrix, x: Float32Array): Promise<Float32Array> {
    const { y, encode } = this.prepare(w, x);
    const enc = this.device.createCommandEncoder();
    encode(enc);
    const read = this.device.createBuffer({ size: w.rows * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(y, 0, read, 0, w.rows * 4);
    this.device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    return out;
  }
}
