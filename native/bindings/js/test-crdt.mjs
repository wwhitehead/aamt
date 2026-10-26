// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// Convergence properties, and the wire shape the Python binding reads.
//
// Run: node bindings/js/test-crdt.mjs
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { CrdtStore, HLC, localReplicaId } from './aamt-crdt.mjs';

const enc = (s) => new TextEncoder().encode(s);
const dec = (b) => new TextDecoder().decode(b);

// 1. Merge is commutative and idempotent.
const a = new CrdtStore();
const b = new CrdtStore();
a.put(42, [0.5, 0.5, 0.5, 0.5], enc('from a'), HLC.zero('1001').tick(1000));
b.put(42, [0.5, 0.5, 0.5, 0.5], enc('from b'), HLC.zero('1002').tick(1000));
const ab = a.merge(b);
const ba = b.merge(a);
assert.deepEqual(ab.toJSON(), ba.toJSON(), 'merge(A,B) must equal merge(B,A)');
assert.deepEqual(ab.merge(b).toJSON(), ab.toJSON(), 'merge must be idempotent');

// 2. Same HLC, different content: the full (hlc, data, tera) key resolves, not insertion order.
const left = new CrdtStore();
const right = new CrdtStore();
const same = new HLC(1000, 0, 7n);
left.put(1, [0.1, 0.1, 0.1, 0.1], enc('aaa'), same);
right.put(1, [0.2, 0.2, 0.2, 0.2], enc('bbb'), same);
assert.deepEqual(left.merge(right).toJSON(), right.merge(left).toJSON(), 'same-HLC records must converge');

// 3. The replica id is a JSON string, and survives a 63-bit value.
const store = new CrdtStore();
const bigId = (1n << 62n) + 12345n;
store.put(7, [0.25, 0.5, 0.75, 1], enc('payload'), HLC.zero(bigId).tick(1234));
const wire = store.toJSON();
assert.equal(typeof wire.records[0].hlc.replica_id, 'string');
assert.equal(wire.records[0].hlc.replica_id, bigId.toString(10));
const round = CrdtStore.fromJSON(JSON.stringify(wire));
assert.equal(round.get(7).hlc.replicaId, bigId);
assert.equal(dec(round.get(7).data), 'payload');

// 4. A local replica id is stable across calls.
const dir = mkdtempSync(join(tmpdir(), 'aamt-crdt-js-'));
try {
  const one = localReplicaId(dir);
  const two = localReplicaId(dir);
  assert.equal(one, two);
  assert.ok(one > 0n && one < 1n << 63n);
} finally {
  rmSync(dir, { recursive: true, force: true });
}

console.log('aamt-crdt.js: ok');
