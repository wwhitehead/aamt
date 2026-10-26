// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
/**
 * Peak memory and time of the streaming writer against packContainer, on a model
 * layer of a few hundred MiB read from files and written to a file (ADR-017).
 * Each writer runs in its own child process, so peak RSS is that writer's alone.
 * The parent then checks the files are byte-identical and that the streamed one
 * verifies.
 *
 *   pnpm --filter @aamt/container bench:stream [MiB = 384] [records = 200000]
 *
 * The second figure sizes a many-small-records run (a world layer of 64-byte
 * payloads), where the streaming writer's cost is its per-record metadata.
 */

import { spawnSync } from 'node:child_process';
import { mkdtempSync, openSync, readSync, closeSync, rmSync, statSync } from 'node:fs';
import { readFile, writeFile, open } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  LAYER_FLAG_KEYED,
  packContainer,
  verifyContainer,
  writeContainer,
  type ContainerSink,
  type StreamContainerInput,
} from '../src/index';
import { fileSource, nodeCrypto, nodeSha384, nodeSha384Incremental, openFileSink } from '../src/node';

const MiB = 2 ** 20;
const key = new Uint8Array(32).fill(7);

/** Deterministic bytes: chunk `c` of payload `p`, 1 MiB at a time. */
function chunk(p: number, c: number, n: number): Uint8Array {
  let s = (p * 7919 + c * 104729 + 1) >>> 0;
  const words = new Uint32Array(Math.ceil(n / 4));
  for (let i = 0; i < words.length; i++) words[i] = s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
  return new Uint8Array(words.buffer, 0, n);
}

function smallPayload(i: number): Uint8Array {
  return chunk(1000 + i, 0, 64);
}

let base = 0;
let peak = 0;
let retained = 0;
let samples = 0;
/** Peak includes garbage not yet collected; every 32nd sample also collects and records what is live. */
function sample(collect = false): void {
  const m = process.memoryUsage();
  peak = Math.max(peak, m.heapUsed + m.arrayBuffers - base);
  if ((collect || samples++ % 32 === 0) && globalThis.gc) {
    globalThis.gc();
    const g = process.memoryUsage();
    retained = Math.max(retained, g.heapUsed + g.arrayBuffers - base);
  }
}

function sampling(sink: ContainerSink): ContainerSink {
  return {
    async write(b) {
      sample();
      await sink.write(b);
    },
    patch: (o, b) => sink.patch!(o, b),
  };
}

async function child(mode: string, dir: string, records: number): Promise<void> {
  const payloads = ['p0.bin', 'p1.bin', 'p2.bin'].map((f) => join(dir, f));
  const keyed = mode.endsWith('-keyed');
  const flags = keyed ? LAYER_FLAG_KEYED : undefined;
  const seal = { ports: nodeCrypto, keys: { model: key } };
  const out = join(dir, `${mode}.aamt`);
  globalThis.gc?.();
  const m0 = process.memoryUsage();
  base = m0.heapUsed + m0.arrayBuffers;
  const t0 = performance.now();

  if (mode === 'pack' || mode === 'pack-keyed') {
    const bytes = await Promise.all(payloads.map(async (p) => new Uint8Array(await readFile(p))));
    sample(true);
    const file = packContainer(
      { layers: [{ kind: 'model', flags, records: bytes.map((payload) => ({ payload })) }] },
      nodeSha384,
      seal,
    );
    sample(true);
    await writeFile(out, file);
    sample(true);
  } else if (mode === 'pack-many') {
    const recs = Array.from({ length: records }, (_, i) => ({
      payload: smallPayload(i),
      at: { x: i % 1024, y: (i >> 10) % 1024, z: i >> 20 },
    }));
    sample(true);
    const file = packContainer({ layers: [{ kind: 'world', records: recs }] }, nodeSha384);
    sample(true);
    await writeFile(out, file);
    sample(true);
  } else {
    let input: StreamContainerInput;
    if (mode === 'stream-many') {
      input = {
        layers: [
          {
            kind: 'world',
            records: Array.from({ length: records }, (_, i) => ({
              payload: () => [smallPayload(i)],
              at: { x: i % 1024, y: (i >> 10) % 1024, z: i >> 20 },
            })),
          },
        ],
      };
    } else {
      input = {
        layers: [{ kind: 'model', flags, records: payloads.map((p) => ({ payload: fileSource(p) })) }],
      };
    }
    const sink = await openFileSink(out);
    try {
      await writeContainer(input, nodeSha384Incremental, sampling(sink), {
        seal,
        checkSources: mode !== 'stream-nocheck',
      });
    } finally {
      await sink.close();
    }
    sample();
  }

  const seconds = (performance.now() - t0) / 1000;
  const maxRssMiB = process.resourceUsage().maxRSS / 1024; // maxRSS is in KiB
  process.stdout.write(
    JSON.stringify({
      mode,
      seconds,
      peakHeapMiB: peak / MiB,
      retainedMiB: retained / MiB,
      maxRssMiB,
      fileMiB: statSync(out).size / MiB,
    }) + '\n',
  );
}

function sameFile(a: string, b: string): boolean {
  if (statSync(a).size !== statSync(b).size) return false;
  const fa = openSync(a, 'r');
  const fb = openSync(b, 'r');
  const ba = Buffer.alloc(4 * MiB);
  const bb = Buffer.alloc(4 * MiB);
  try {
    for (;;) {
      const na = readSync(fa, ba, 0, ba.length, null);
      const nb = readSync(fb, bb, 0, bb.length, null);
      if (na !== nb || ba.compare(bb, 0, nb, 0, na) !== 0) return false;
      if (na === 0) return true;
    }
  } finally {
    closeSync(fa);
    closeSync(fb);
  }
}

async function parent(totalMiB: number, records: number): Promise<void> {
  const dir = mkdtempSync(join(tmpdir(), 'aamt-bench-stream-'));
  try {
    // Three payload files: a third each, the last with an odd tail. Written 1 MiB at a time.
    const sizes = [0, 1, 2].map((i) => Math.floor((totalMiB * MiB) / 3) + (i === 2 ? 12345 : 0));
    for (const [p, n] of sizes.entries()) {
      const fh = await open(join(dir, `p${p}.bin`), 'w');
      for (let c = 0, at = 0; at < n; c++, at += MiB) await fh.write(chunk(p, c, Math.min(MiB, n - at)));
      await fh.close();
    }
    console.log(
      `payloads: ${sizes.map((n) => (n / MiB).toFixed(1)).join(' + ')} MiB; small records: ${records}`,
    );

    const script = fileURLToPath(import.meta.url);
    const rows: Record<string, unknown>[] = [];
    for (const mode of [
      'pack',
      'stream',
      'stream-nocheck',
      'pack-keyed',
      'stream-keyed',
      'pack-many',
      'stream-many',
    ]) {
      const r = spawnSync(
        process.execPath,
        ['--expose-gc', ...process.execArgv, script, '--child', mode, dir, String(records)],
        {
          encoding: 'utf8',
          maxBuffer: 16 * MiB,
        },
      );
      if (r.status !== 0) throw new Error(`${mode} failed:\n${r.stderr}`);
      rows.push(JSON.parse(r.stdout.trim().split('\n').pop()!) as Record<string, unknown>);
    }
    console.table(
      rows.map((r) => ({
        mode: r.mode,
        'file MiB': (r.fileMiB as number).toFixed(1),
        seconds: (r.seconds as number).toFixed(2),
        'peak heap+arrayBuffers MiB': (r.peakHeapMiB as number).toFixed(1),
        'live after GC MiB': (r.retainedMiB as number).toFixed(1),
        'max RSS MiB': (r.maxRssMiB as number).toFixed(1),
      })),
    );

    for (const [a, b] of [
      ['pack', 'stream'],
      ['pack', 'stream-nocheck'],
      ['pack-keyed', 'stream-keyed'],
      ['pack-many', 'stream-many'],
    ] as const) {
      const same = sameFile(join(dir, `${a}.aamt`), join(dir, `${b}.aamt`));
      console.log(`${a} vs ${b}: ${same ? 'byte-identical' : 'DIFFERENT'}`);
      if (!same) process.exitCode = 1;
    }
    for (const f of ['stream', 'stream-keyed', 'stream-many']) {
      const v = verifyContainer(new Uint8Array(await readFile(join(dir, `${f}.aamt`))), nodeSha384);
      console.log(`${f}.aamt verifies: ${v.ok}${v.ok ? '' : ` (${v.problems.slice(0, 3).join('; ')})`}`);
      if (!v.ok) process.exitCode = 1;
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

const argv = process.argv.slice(2);
if (argv[0] === '--child') await child(argv[1]!, argv[2]!, Number(argv[3]));
else await parent(Number(argv[0] ?? 384), Number(argv[1] ?? 200_000));
