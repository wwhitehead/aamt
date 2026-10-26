// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// High-level JS wrapper over the raw Emscripten glue (dist/aamt.mjs).
// This is the module the Chiral Lattice / AAMT Math Visualizer imports —
// same C ABI (include/aamt/aamt.h) the CLI and Python bindings use, so a
// .aamt file packed anywhere opens identically here.
import createAamtModule from './dist/aamt.mjs';

const ROUTE_SIZE = 12;   // uint32 path, depth, record_count
const VIEW_SIZE = 32;    // uint64 id; float tera[4]; ptr data; uint32 size (wasm32)

let modulePromise = null;
function loadModule() {
  if (!modulePromise) modulePromise = createAamtModule();
  return modulePromise;
}

export class AamtSubstrate {
  static async open(bytes) {
    const Module = await loadModule();
    const inst = new AamtSubstrate(Module);
    inst._open(bytes);
    return inst;
  }

  constructor(Module) {
    this.M = Module;
    this.handle = 0;
  }

  _open(bytes) {
    const path = `/aamt_${(Math.random() * 1e9) | 0}.aamt`;
    this.M.FS.writeFile(path, bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes));
    const errBuf = this.M._malloc(256);
    const pathLen = this.M.lengthBytesUTF8(path) + 1;
    const pathPtr = this.M._malloc(pathLen);
    this.M.stringToUTF8(path, pathPtr, pathLen);
    const handle = this.M._aamt_open(pathPtr, errBuf, 256);
    if (!handle) {
      const err = this.M.UTF8ToString ? this.M.UTF8ToString(errBuf) : '(open failed)';
      this.M._free(errBuf);
      this.M._free(pathPtr);
      throw new Error(`aamt_open failed: ${err}`);
    }
    this.M._free(errBuf);
    this.M._free(pathPtr);
    this.handle = handle;
  }

  get nodeCount() { return Number(this.M._aamt_node_count(this.handle)); }
  get recordCount() { return Number(this.M._aamt_record_count(this.handle)); }
  get maxDepth() { return this.M._aamt_max_depth(this.handle); }
  get contentHash() { return this.M._aamt_content_hash(this.handle).toString(16).padStart(16, '0'); }

  /** query([T,E,R,A], capacity=16) -> { path, depth, bucketSize, records:[{id,tera,text,bytes}] } */
  query(tera, capacity = 16) {
    const M = this.M;
    const teraPtr = M._malloc(16);
    for (let d = 0; d < 4; d++) M.setValue(teraPtr + d * 4, tera[d], 'float');
    const routePtr = M._malloc(ROUTE_SIZE);
    const viewsPtr = M._malloc(VIEW_SIZE * capacity);

    const n = M._aamt_query(this.handle, teraPtr, viewsPtr, capacity, routePtr);
    const path = M.getValue(routePtr, 'i32') >>> 0;
    const depth = M.getValue(routePtr + 4, 'i32') >>> 0;
    const bucketSize = M.getValue(routePtr + 8, 'i32') >>> 0;

    const records = [];
    for (let i = 0; i < n; i++) {
      const base = viewsPtr + i * VIEW_SIZE;
      const idLo = M.getValue(base, 'i32') >>> 0;
      const idHi = M.getValue(base + 4, 'i32') >>> 0;
      const t = [0, 1, 2, 3].map((d) => M.getValue(base + 8 + d * 4, 'float'));
      const dataPtr = M.getValue(base + 24, 'i32');
      const size = M.getValue(base + 28, 'i32') >>> 0;
      // Copy into a fresh, plainly-allocated Uint8Array — not .slice(), which
      // on a view over growable WASM memory can inherit a resizable backing
      // ArrayBuffer via species construction, and TextDecoder.decode()
      // rejects resizable ArrayBuffers even when the view itself is a copy.
      const bytes = new Uint8Array(size);
      bytes.set(M.HEAPU8.subarray(dataPtr, dataPtr + size));
      let text = '';
      try { text = new TextDecoder().decode(bytes); } catch { /* binary payload */ }
      records.push({ id: idHi === 0 ? idLo : idHi * 2 ** 32 + idLo, tera: t, bytes, text });
    }

    M._free(teraPtr);
    M._free(routePtr);
    M._free(viewsPtr);

    return { path, depth, bucketSize, records };
  }

  /** Renders the routing path as a dotted Odu word, e.g. "5.6.13.1". */
  static oduWord(path, depth) {
    const nibbles = [];
    for (let i = depth - 1; i >= 0; i--) nibbles.push((path >>> (4 * i)) & 0xf);
    return nibbles.join('.');
  }

  close() {
    if (this.handle) this.M._aamt_close(this.handle);
    this.handle = 0;
  }
}
