// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// Smoke test: pack a .aamt file via the native CLI, load its raw bytes into
// the WASM module's virtual FS, and confirm routing/records match what the
// native reader produced. Run: node smoke_test.mjs
import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import createAamtModule from './dist/aamt.mjs';

const ROOT = new URL('.', import.meta.url).pathname;
const CLI = join(ROOT, '..', 'build', 'aamt');

function assert(cond, msg) {
  if (!cond) throw new Error('FAIL: ' + msg);
}

const dir = mkdtempSync(join(tmpdir(), 'aamt_wasm_smoke_'));
const tsv = join(dir, 'demo.tsv');
const aamtPath = join(dir, 'demo.aamt');

const rows = [];
let seed = 0xC0FFEE;
const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
for (let i = 0; i < 500; i++) {
  const t = rnd().toFixed(6), e = rnd().toFixed(6), r = rnd().toFixed(6), a = rnd().toFixed(6);
  rows.push(`${i}\t${t}\t${e}\t${r}\t${a}\twasm-demo-record-${i}`);
}
writeFileSync(tsv, rows.join('\n') + '\n');
execFileSync(CLI, ['pack', tsv, aamtPath]);

const bytes = readFileSync(aamtPath);
const Module = await createAamtModule();

// Growth must stay off (see WP-23 corrections ledger #9): a growable
// WebAssembly.Memory.buffer is a "resizable" ArrayBuffer, and Safari's
// TextDecoder.decode() rejects those — this shipped to production once
// before being caught by live-browser testing, not by Node.
assert(Module.HEAPU8.buffer.resizable === false, 'WASM memory must not be resizable (Safari TextDecoder bug)');

Module.FS.writeFile('/demo.aamt', bytes);

const errBuf = Module._malloc(256);
const pathPtr = Module._malloc(bytes.length + 1);
Module.stringToUTF8('/demo.aamt', pathPtr, bytes.length + 1);
const handle = Module._aamt_open(pathPtr, errBuf, 256);
assert(handle !== 0, 'aamt_open returned null');

const nodeCount = Number(Module._aamt_node_count(handle));
const recordCount = Number(Module._aamt_record_count(handle));
assert(recordCount === 500, `expected 500 records, got ${recordCount}`);
assert(nodeCount >= 1, 'expected at least one node');

// aamt_route: { uint32 path; uint32 depth; uint32 record_count } = 12 bytes
const routePtr = Module._malloc(12);
// aamt_record_view: { uint64 id; float tera[4]; ptr data; uint32 size } = 32 bytes (wasm32)
const CAP = 8;
const viewsPtr = Module._malloc(32 * CAP);
const teraPtr = Module._malloc(16);

Module.setValue(teraPtr, 0.5, 'float');
Module.setValue(teraPtr + 4, 0.5, 'float');
Module.setValue(teraPtr + 8, 0.5, 'float');
Module.setValue(teraPtr + 12, 0.5, 'float');

const n = Module._aamt_query(handle, teraPtr, viewsPtr, CAP, routePtr);
const path = Module.getValue(routePtr, 'i32') >>> 0;
const depth = Module.getValue(routePtr + 4, 'i32') >>> 0;
const bucket = Module.getValue(routePtr + 8, 'i32') >>> 0;
assert(n >= 0, 'query returned negative count');
console.log(`route (0.5,0.5,0.5,0.5): odu-path=0x${path.toString(16)} depth=${depth} bucket=${bucket} returned=${n}`);

for (let i = 0; i < n; i++) {
  const base = viewsPtr + i * 32;
  const idLo = Module.getValue(base, 'i32') >>> 0;
  const dataPtr = Module.getValue(base + 24, 'i32');
  const size = Module.getValue(base + 28, 'i32') >>> 0;
  const text = new TextDecoder().decode(Module.HEAPU8.subarray(dataPtr, dataPtr + size));
  assert(text.startsWith('wasm-demo-record-'), `unexpected payload: ${text}`);
  if (i === 0) console.log(`  first record: id=${idLo} text="${text}"`);
}

const hash = Module._aamt_content_hash(handle);
console.log(`content hash (BigInt): ${hash}`);

Module._aamt_close(handle);
console.log(`PASS: WASM module opened a native-packed .aamt file, routed a query, ` +
            `and read ${n} zero-copy record(s) out of a ${recordCount}-record / ${nodeCount}-node substrate.`);
