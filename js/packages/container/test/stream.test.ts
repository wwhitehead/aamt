// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import {
  CONTEXT,
  H_FILE_ID,
  ID_BYTES,
  LAYER_FLAG_KEYED,
  LAYER_KINDS,
  PAGE_BYTES,
  deriveKey,
  identify,
  identifying,
  openContainer,
  packContainer,
  sha384Of,
  verifyContainer,
  writeContainer,
  type ChunkSource,
  type ContainerInput,
  type ContainerSink,
  type CryptoPorts,
  type LayerKind,
  type PayloadSource,
  type SealOptions,
  type StreamContainerInput,
  type StreamWriteOptions,
} from '../src/index';
import {
  fileSource,
  nodeAead,
  nodeAeadStream,
  nodeCrypto,
  nodeKdf,
  nodeSha384,
  nodeSha384Incremental,
  openFileSink,
} from '../src/node';

/** A deterministic LCG, so the randomized cases are reproducible. */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

/** `packContainer`'s ports minus the streaming AEAD: forces the seal-each-record-whole path. */
const noStreamCrypto: CryptoPorts = { aead: nodeAead, kdf: nodeKdf, sha384: nodeSha384 };

/** An in-memory sink that copies what it is given (the writer reuses buffers) and logs every call. */
function memorySink(withPatch: boolean): ContainerSink & {
  bytes(): Uint8Array;
  writes: number[];
  patches: { offset: number; bytes: number }[];
} {
  let buf = new Uint8Array(1 << 16);
  let used = 0;
  const writes: number[] = [];
  const patches: { offset: number; bytes: number }[] = [];
  const sink: ContainerSink & { bytes(): Uint8Array; writes: number[]; patches: typeof patches } = {
    writes,
    patches,
    write(b) {
      writes.push(b.byteLength);
      if (used + b.byteLength > buf.byteLength) {
        const next = new Uint8Array(Math.max(buf.byteLength * 2, used + b.byteLength));
        next.set(buf.subarray(0, used));
        buf = next;
      }
      buf.set(b, used);
      used += b.byteLength;
    },
    bytes: () => buf.slice(0, used),
  };
  if (withPatch)
    sink.patch = async (offset, b) => {
      patches.push({ offset, bytes: b.byteLength });
      await Promise.resolve();
      buf.set(b, offset);
    };
  return sink;
}

/**
 * Serve `payload` four ways: in memory, as a sync iterable, as an async iterable,
 * and as an async source that reuses one buffer (which the writer must tolerate).
 * Chunk sizes are irregular, include 1-byte and empty chunks, and are the same on
 * every read. `opens` counts reads.
 */
function sourceFor(payload: Uint8Array, variant: number, r: () => number, opens: number[]): PayloadSource {
  if (variant === 0) return payload;
  const cuts: number[] = [];
  for (let at = 0; at < payload.byteLength;) {
    const n = r() < 0.2 ? 1 : r() < 0.1 ? 0 : 1 + Math.floor(r() * 5000);
    cuts.push(Math.min(n, payload.byteLength - at));
    at += cuts[cuts.length - 1]!;
  }
  const slot = opens.push(0) - 1;
  if (variant === 1)
    return function* () {
      opens[slot]!++;
      let at = 0;
      for (const n of cuts) {
        yield payload.subarray(at, at + n);
        at += n;
      }
    };
  if (variant === 2)
    return async function* () {
      opens[slot]!++;
      let at = 0;
      for (const n of cuts) {
        await Promise.resolve();
        yield payload.slice(at, at + n);
        at += n;
      }
    };
  const shared = new Uint8Array(5000);
  return async function* () {
    opens[slot]!++;
    let at = 0;
    for (const n of cuts) {
      shared.set(payload.subarray(at, at + n));
      yield shared.subarray(0, n);
      shared.fill(0xee); // anything the writer kept a reference to is now wrong
      at += n;
    }
  };
}

interface Case {
  readonly pack: ContainerInput;
  readonly stream: StreamContainerInput;
  readonly seal: SealOptions;
  readonly streamSeal: SealOptions;
  readonly opens: number[];
}

/** Random containers: every layer kind, empty layers, empty payloads, page-sized payloads, keyed layers. */
function generate(seed: number): Case {
  const r = lcg(seed);
  const pack: any[] = [];
  const stream: any[] = [];
  const keys: Partial<Record<LayerKind, Uint8Array>> = {};
  const opens: number[] = [];
  for (const kind of LAYER_KINDS) {
    if (r() < 0.35) continue;
    const n = r() < 0.15 ? 0 : Math.floor(r() * 10);
    const pr: any[] = [];
    const sr: any[] = [];
    for (let i = 0; i < n; i++) {
      const roll = r();
      const len =
        roll < 0.08
          ? 0
          : roll < 0.16
            ? PAGE_BYTES + Math.floor(r() * 3) - 1
            : Math.floor(r() * (r() < 0.1 ? 40000 : 400));
      const payload = new Uint8Array(len);
      for (let j = 0; j < len; j++) payload[j] = (r() * 256) | 0;
      if (len > 0) payload[0] = i;
      const rec: any = { kind: Math.floor(r() * 5), flags: r() < 0.3 ? 1 : 0 };
      if (kind === 'world' || kind === 'mind')
        rec.at = { x: Math.floor(r() * 9) - 4, y: Math.floor(r() * 9) - 4, z: Math.floor(r() * 3) };
      if ((kind === 'experience' || kind === 'code') && r() < 0.6) {
        rec.tera = [r(), r(), r(), r()];
        rec.tier = r() < 0.5 ? 'declared' : 'reconstructed';
        rec.lens = r() < 0.5 ? 'wisdom' : 'technical';
      }
      pr.push({ ...rec, payload });
      sr.push({ ...rec, payload: sourceFor(payload, Math.floor(r() * 4), r, opens) });
    }
    let flags: number | undefined;
    if (r() < 0.3) {
      flags = LAYER_FLAG_KEYED | (r() < 0.5 ? 1 : 0);
      keys[kind] = deriveKey(nodeKdf, new Uint8Array(32).fill(seed & 255), [kind]);
    } else {
      const explicitZero = r() < 0.3;
      if (explicitZero) flags = 0;
    }
    pack.push({ kind, flags, records: pr });
    stream.push({ kind, flags, records: sr });
  }
  if (r() < 0.5) {
    pack.reverse();
    stream.reverse();
  }
  return {
    pack: { layers: pack },
    stream: { layers: stream },
    seal: { ports: nodeCrypto, keys },
    streamSeal: { ports: r() < 0.5 ? nodeCrypto : noStreamCrypto, keys },
    opens,
  };
}

async function streamed(
  input: StreamContainerInput,
  withPatch: boolean,
  options: StreamWriteOptions,
): Promise<{ bytes: Uint8Array; sink: ReturnType<typeof memorySink>; fileId: Uint8Array; passes: number }> {
  const sink = memorySink(withPatch);
  const res = await writeContainer(input, nodeSha384Incremental, sink, options);
  return { bytes: sink.bytes(), sink, fileId: res.fileId, passes: res.passes };
}

describe('incremental ports', () => {
  it('sha384Of(nodeSha384Incremental) and identifying agree with the one-shot port', () => {
    const r = lcg(7);
    for (let t = 0; t < 50; t++) {
      const parts = Array.from({ length: 1 + Math.floor(r() * 5) }, () =>
        Uint8Array.from({ length: Math.floor(r() * 300) }, () => (r() * 256) | 0),
      );
      expect(sha384Of(nodeSha384Incremental)(parts)).toEqual(nodeSha384(parts));
      const h = identifying(nodeSha384Incremental, CONTEXT.world);
      parts.forEach((p) => h.update(p));
      expect(h.digest()).toEqual(identify(nodeSha384, CONTEXT.world, parts));
    }
  });

  it('nodeAeadStream equals nodeAead.seal under any chunking', () => {
    const r = lcg(11);
    const key = new Uint8Array(32).fill(3);
    const nonce = new Uint8Array(12).fill(9);
    const aad = new Uint8Array(40).fill(1);
    for (let t = 0; t < 30; t++) {
      const pt = Uint8Array.from({ length: Math.floor(r() * 3000) }, () => (r() * 256) | 0);
      const s = nodeAeadStream.seal(key, nonce, aad);
      const out: number[] = [];
      for (let at = 0; at < pt.byteLength;) {
        const n = Math.min(pt.byteLength - at, 1 + Math.floor(r() * 200));
        out.push(...s.update(pt.subarray(at, at + n)));
        at += n;
      }
      out.push(...s.final());
      expect(Uint8Array.from(out)).toEqual(nodeAead.seal(key, nonce, pt, aad));
    }
  });
});

describe('writeContainer: the same bytes as packContainer', () => {
  it('matches byte for byte over 200 generated containers, with and without patch, any chunking and buffering', async () => {
    let compared = 0;
    let keyedStreamed = 0;
    let keyedWhole = 0;
    let refused = 0;
    for (let seed = 1; seed <= 200; seed++) {
      const c = generate(seed);
      let expected: Uint8Array | undefined;
      let error: string | undefined;
      try {
        expected = packContainer(c.pack, nodeSha384, c.seal);
      } catch (e) {
        error = (e as Error).message;
      }
      const bufferBytes = [0, 7, 4096, 1 << 20][seed % 4]!;
      for (const withPatch of [true, false]) {
        const run = streamed(c.stream, withPatch, { seal: c.streamSeal, bufferBytes });
        if (error !== undefined) {
          await expect(run).rejects.toThrow(error);
          refused++;
          continue;
        }
        const got = await run;
        expect(got.bytes.byteLength).toBe(expected!.byteLength);
        expect(Buffer.compare(got.bytes, expected!)).toBe(0);
        expect(got.fileId).toEqual(expected!.subarray(H_FILE_ID, H_FILE_ID + ID_BYTES));
        compared++;
      }
      const keyedLayers = c.pack.layers.filter((l) => (l.flags ?? 0) & LAYER_FLAG_KEYED).length;
      if (keyedLayers && expected) {
        if (c.streamSeal.ports.aeadStream) keyedStreamed++;
        else keyedWhole++;
      }
    }
    // The corpus exercises what it claims to.
    expect(compared).toBeGreaterThan(350);
    expect(keyedStreamed).toBeGreaterThan(10);
    expect(keyedWhole).toBeGreaterThan(10);
    expect(refused).toBe(0); // no generated case is invalid; refusals are tested below
  });

  it('streams an empty container, and empty layers, identically', async () => {
    for (const layers of [[], LAYER_KINDS.map((kind) => ({ kind, records: [] }))] as const) {
      const expected = packContainer({ layers } as ContainerInput, nodeSha384);
      const got = await streamed({ layers } as StreamContainerInput, true, {});
      expect(Buffer.compare(got.bytes, expected)).toBe(0);
      expect(verifyContainer(got.bytes, nodeSha384).ok).toBe(true);
    }
  });

  it('page-aligns large model payloads exactly as packContainer does, under odd chunk sizes', async () => {
    const sizes = [PAGE_BYTES - 1, PAGE_BYTES, PAGE_BYTES + 1, 3 * PAGE_BYTES + 17, 1];
    const payloads = sizes.map((n, i) => Uint8Array.from({ length: n }, (_, j) => (j * 31 + i) & 255));
    const odd = (p: Uint8Array): ChunkSource =>
      function* () {
        for (let at = 0; at < p.byteLength; at += 4093)
          yield p.subarray(at, Math.min(p.byteLength, at + 4093));
      };
    const expected = packContainer(
      { layers: [{ kind: 'model', records: payloads.map((payload) => ({ payload })) }] },
      nodeSha384,
    );
    const got = await streamed(
      { layers: [{ kind: 'model', records: payloads.map((p) => ({ payload: odd(p) })) }] },
      true,
      {},
    );
    expect(Buffer.compare(got.bytes, expected)).toBe(0);
    // Both writers share the planner, so byte-identity cannot see a planner bug; the reader is the
    // independent oracle. Every model payload must start on a page, and come back byte-exact.
    const model = openContainer(got.bytes).layer('model')!;
    expect(model.count).toBe(sizes.length);
    const seen = [...model.records()].map((rec) => {
      expect(rec.payload.byteOffset % PAGE_BYTES).toBe(0);
      return rec.payload.byteLength;
    });
    expect(seen.sort((a, b) => a - b)).toEqual([...sizes].sort((a, b) => a - b));
    expect(verifyContainer(got.bytes, nodeSha384).ok).toBe(true);
  });

  it('refuses exactly what packContainer refuses, with the same message', async () => {
    const p = new Uint8Array([1, 2, 3]);
    const bad: ContainerInput[] = [
      {
        layers: [
          { kind: 'code', records: [] },
          { kind: 'code', records: [] },
        ],
      },
      {
        layers: [
          {
            kind: 'world',
            records: [
              { payload: p, at: { x: 0, y: 0, z: 0 } },
              { payload: p, at: { x: 0, y: 0, z: 0 } },
            ],
          },
        ],
      },
      { layers: [{ kind: 'code', flags: LAYER_FLAG_KEYED, records: [{ payload: p }] }] },
      { layers: [{ kind: 'code', records: [{ payload: p, tera: [0.1, 0.2, 0.3, 0.4] }] }] },
      { layers: [{ kind: 'code', records: [{ payload: p, tier: 'declared' }] }] },
      { layers: [{ kind: 'world', records: [{ payload: p, at: { x: 0.5, y: 0, z: 0 } }] }] },
      { layers: [{ kind: 'model', records: [{ payload: p, kind: -1 }] }] },
      { layers: [{ kind: 'nope' as 'model', records: [] }] },
    ];
    for (const input of bad) {
      let message = '';
      try {
        packContainer(input, nodeSha384);
      } catch (e) {
        message = (e as Error).message;
      }
      expect(message).not.toBe('');
      await expect(
        writeContainer(input as StreamContainerInput, nodeSha384Incremental, memorySink(true)),
      ).rejects.toThrow(message);
    }
  });
});

describe('writeContainer: passes, writes and sources', () => {
  const input = (opens: number[], keyed = false): { stream: StreamContainerInput; pack: ContainerInput } => {
    const r = lcg(5);
    const payloads = [900, 0, 20000, 5].map((n) => Uint8Array.from({ length: n }, () => (r() * 256) | 0));
    return {
      pack: {
        layers: [
          {
            kind: 'experience',
            flags: keyed ? LAYER_FLAG_KEYED : 0,
            records: payloads.map((payload) => ({ payload })),
          },
        ],
      },
      stream: {
        layers: [
          {
            kind: 'experience',
            flags: keyed ? LAYER_FLAG_KEYED : 0,
            records: payloads.map((p) => ({ payload: sourceFor(p, 2, r, opens) })),
          },
        ],
      },
    };
  };
  const key = new Uint8Array(32).fill(4);

  it('reads each source three times with a patching sink and four without; a streamed keyed source once more', async () => {
    for (const [withPatch, keyed, ports, reads] of [
      [true, false, nodeCrypto, 3],
      [false, false, nodeCrypto, 4],
      [true, true, nodeCrypto, 4],
      [false, true, nodeCrypto, 5],
      [true, true, noStreamCrypto, 3],
    ] as const) {
      const opens: number[] = [];
      const { stream } = input(opens, keyed);
      const res = await streamed(stream, withPatch, { seal: { ports, keys: { experience: key } } });
      expect(opens).toEqual([reads, reads, reads, reads]);
      if (ports.aeadStream || !keyed) expect(res.passes).toBe(reads);
    }
  });

  it('patches only the 48-byte fileId; without patch, the sink sees only appends', async () => {
    const a = await streamed(input([]).stream, true, {});
    expect(a.sink.patches).toEqual([{ offset: H_FILE_ID, bytes: ID_BYTES }]);
    const b = await streamed(input([]).stream, false, {});
    expect(b.sink.patches).toEqual([]);
    expect(Buffer.compare(a.bytes, b.bytes)).toBe(0);
    expect(verifyContainer(a.bytes, nodeSha384).ok).toBe(true);
  });

  it('coalesces small writes into bufferBytes: every flush but the last is nearly full', async () => {
    // Source chunks are at most 5,000 bytes, so with an 8 KiB buffer every write is a flush.
    const got = await streamed(input([]).stream, true, { bufferBytes: 8192 });
    const writes = got.sink.writes;
    expect(writes.every((n) => n <= 8192)).toBe(true);
    expect(writes.slice(0, -1).every((n) => n > 8192 - 5000)).toBe(true);
    const unbuffered = await streamed(input([]).stream, true, { bufferBytes: 0 });
    expect(unbuffered.sink.writes.length).toBeGreaterThan(3 * writes.length);
    expect(Buffer.compare(got.bytes, unbuffered.bytes)).toBe(0);
  });

  /** A source that yields `payload` on its first `honest` reads, then `later`. */
  const fickle = (payload: Uint8Array, later: Uint8Array, honest: number): ChunkSource => {
    let reads = 0;
    return function* () {
      yield reads++ < honest ? payload : later;
    };
  };

  it('refuses a source whose length changes between passes, in every pass', async () => {
    for (const honest of [1, 2]) {
      const src = fickle(new Uint8Array(10), new Uint8Array(11), honest);
      await expect(
        writeContainer(
          { layers: [{ kind: 'model', records: [{ payload: src }] }] },
          nodeSha384Incremental,
          memorySink(true),
          { checkSources: false },
        ),
      ).rejects.toThrow(/yielded 11 bytes, but 10 on its first read/);
    }
  });

  it('refuses a source whose bytes change at the same length; unchecked, the file would not verify', async () => {
    const before = new Uint8Array(64).fill(1);
    const after = new Uint8Array(64).fill(2);
    for (const honest of [1, 2]) {
      await expect(
        writeContainer(
          { layers: [{ kind: 'code', records: [{ payload: fickle(before, after, honest) }] }] },
          nodeSha384Incremental,
          memorySink(true),
        ),
      ).rejects.toThrow(/id no longer matches/);
    }
    const sink = memorySink(true);
    await writeContainer(
      { layers: [{ kind: 'code', records: [{ payload: fickle(before, after, 1) }] }] },
      nodeSha384Incremental,
      sink,
      { checkSources: false },
    );
    const v = verifyContainer(sink.bytes(), nodeSha384);
    expect(v.ok).toBe(false);
    expect(v.problems.join(' ')).toMatch(/id does not match its content/);
  });
});

describe('writeContainer: files', () => {
  let dir = '';
  beforeAll(async () => {
    dir = await mkdtemp(join(tmpdir(), 'aamt-stream-'));
  });
  afterAll(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it('streams a 64 MiB model layer from files to a file: byte-identical, with heap far below the file size', async () => {
    // Three payload files of 16, 24 and 24 MiB plus an odd tail, content from a fixed LCG.
    const sizes = [16 << 20, 24 << 20, (24 << 20) + 12345];
    const paths: string[] = [];
    for (const [i, n] of sizes.entries()) {
      const r = lcg(100 + i);
      const words = new Uint32Array(Math.ceil(n / 4));
      for (let j = 0; j < words.length; j++) words[j] = (r() * 2 ** 32) >>> 0;
      const p = join(dir, `payload-${i}.bin`);
      await writeFile(p, new Uint8Array(words.buffer, 0, n));
      paths.push(p);
    }
    const out = join(dir, 'model.aamt');
    global.gc?.();
    const base = process.memoryUsage();
    let peak = 0;
    const sink = await openFileSink(out);
    const sampling: ContainerSink = {
      async write(b) {
        const m = process.memoryUsage();
        peak = Math.max(peak, m.heapUsed + m.arrayBuffers - base.heapUsed - base.arrayBuffers);
        await sink.write(b);
      },
      patch: (o, b) => sink.patch!(o, b),
    };
    try {
      await writeContainer(
        { layers: [{ kind: 'model', records: paths.map((p) => ({ payload: fileSource(p) })) }] },
        nodeSha384Incremental,
        sampling,
      );
    } finally {
      await sink.close();
    }

    const written = new Uint8Array(await readFile(out));
    const payloads = await Promise.all(paths.map(async (p) => new Uint8Array(await readFile(p))));
    const expected = packContainer(
      { layers: [{ kind: 'model', records: payloads.map((payload) => ({ payload })) }] },
      nodeSha384,
    );
    expect(written.byteLength).toBe(expected.byteLength);
    expect(Buffer.compare(written, expected)).toBe(0);
    expect(verifyContainer(written, nodeSha384).ok).toBe(true);
    // Measured on an M1 Max: a few MiB of growth for a 64 MiB file (ADR-017). The bound is loose on
    // purpose — it is GC-dependent — but an in-memory writer would need at least the file size.
    expect(peak).toBeLessThan(24 << 20);
  });
});
