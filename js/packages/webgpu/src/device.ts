// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * A WebGPU device under plain Node, via Dawn (the `webgpu` package) — the same
 * implementation Chrome ships. In a browser, use `navigator.gpu` instead.
 */
import { create, globals } from 'webgpu';

/**
 * Held for the life of the process. Dawn processes GPU events through the
 * instance, so if it is garbage-collected after the device is made, every later
 * `mapAsync` and `onSubmittedWorkDone` silently never resolves and Node exits.
 */
let instance: GPU | undefined;

export async function nodeDevice(): Promise<{ device: GPUDevice; adapterName: string } | null> {
  Object.assign(globalThis, globals);
  const gpu = (instance ??= create([]));
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const info = adapter.info;
  const adapterName = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(' · ') || 'unknown adapter';
  return { device, adapterName };
}
