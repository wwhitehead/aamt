// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// Smoke test for the high-level AamtSubstrate wrapper (aamt-client.mjs).
import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { AamtSubstrate } from './aamt-client.mjs';

const ROOT = new URL('.', import.meta.url).pathname;
const CLI = join(ROOT, '..', 'build', 'aamt');

const dir = mkdtempSync(join(tmpdir(), 'aamt_wasm_client_'));
const tsv = join(dir, 'demo.tsv');
const aamtPath = join(dir, 'demo.aamt');

let seed = 42;
const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
const rows = [];
for (let i = 0; i < 2000; i++) {
  rows.push(`${i}\t${rnd().toFixed(6)}\t${rnd().toFixed(6)}\t${rnd().toFixed(6)}\t${rnd().toFixed(6)}\tcapsule-${i}`);
}
writeFileSync(tsv, rows.join('\n') + '\n');
execFileSync(CLI, ['pack', tsv, aamtPath, '8', '16']);

const bytes = readFileSync(aamtPath);
const sub = await AamtSubstrate.open(bytes);

console.log(`nodes=${sub.nodeCount} records=${sub.recordCount} maxDepth=${sub.maxDepth} hash=${sub.contentHash}`);
if (sub.recordCount !== 2000) throw new Error(`expected 2000 records, got ${sub.recordCount}`);

const result = sub.query([0.5, 0.5, 0.5, 0.5], 16);
const word = AamtSubstrate.oduWord(result.path, result.depth);
console.log(`query(0.5,0.5,0.5,0.5) -> odu-word=${word} bucket=${result.bucketSize} returned=${result.records.length}`);
if (!result.records.every((r) => r.text.startsWith('capsule-'))) throw new Error('payload mismatch');
console.log(`  sample: ${result.records[0].text} tera=[${result.records[0].tera.map((v) => v.toFixed(3))}]`);

sub.close();
console.log('PASS: AamtSubstrate wrapper — open/query/close round-trip clean.');
