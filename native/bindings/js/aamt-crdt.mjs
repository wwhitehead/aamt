// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// aamt-crdt.js — the JavaScript side of the libaamt CRDT interchange (WP-23 §13.2–13.3).
//
// This is a faithful port of `bindings/python/aamt_crdt.py`, not a second design. The two modules
// must produce the same JSON and resolve the same winner, or "cross-language convergence" is a
// claim rather than a property. The rules, in the order the Python module states them:
//
//   * an HLC is `(physical, logical, replica_id)`, compared lexicographically;
//   * a record's winner is the max over `(hlc, data, tera)` — HLC alone is not a total order over
//     records, because two replicas can collide on physical+logical;
//   * `replica_id` is a JSON **string**. A 63-bit id is beyond JS's 2^53 safe-integer range, and a
//     number would be truncated before the comparison ever saw it;
//   * `data` is base64 on the wire, bytes in memory;
//   * `merge` is commutative, associative and idempotent: a map of independent per-key maxima.
//
// The substrate (`to_aamt`) is not here: packing is libaamt's C ABI, reached through the Python
// binding or the native addon. This module governs convergence *before* a pack.

import { randomBytes } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const b64encode = (bytes) => Buffer.from(bytes).toString('base64');
const b64decode = (text) => new Uint8Array(Buffer.from(text, 'base64'));

const big = (value) => {
  if (typeof value === 'bigint') return value;
  if (typeof value === 'number') {
    if (!Number.isSafeInteger(value)) throw new Error('aamt-crdt: a numeric id must be a safe integer; pass a string.');
    return BigInt(value);
  }
  if (typeof value === 'string' && /^-?\d+$/.test(value)) return BigInt(value);
  throw new Error(`aamt-crdt: invalid integer ${String(value)}.`);
};

/** Hybrid logical clock: (physical ms, logical, replica id). */
export class HLC {
  constructor(physical, logical, replicaId) {
    this.physical = Number(physical);
    this.logical = Number(logical);
    this.replicaId = big(replicaId);
    for (const [name, v] of [['physical', this.physical], ['logical', this.logical]]) {
      if (!Number.isSafeInteger(v) || v < 0) throw new Error(`aamt-crdt: ${name} must be a non-negative safe integer.`);
    }
  }

  static zero(replicaId) {
    return new HLC(0, 0, replicaId);
  }

  tick(nowMs = Date.now()) {
    const now = Math.max(0, Math.trunc(nowMs));
    if (now > this.physical) return new HLC(now, 0, this.replicaId);
    return new HLC(this.physical, this.logical + 1, this.replicaId);
  }

  update(remote, nowMs = Date.now()) {
    const now = Math.max(0, Math.trunc(nowMs));
    const physical = Math.max(now, this.physical, remote.physical);
    let logical;
    if (physical === this.physical && physical === remote.physical) logical = Math.max(this.logical, remote.logical) + 1;
    else if (physical === this.physical) logical = this.logical + 1;
    else if (physical === remote.physical) logical = remote.logical + 1;
    else logical = 0;
    return new HLC(physical, logical, this.replicaId);
  }

  compare(other) {
    if (this.physical !== other.physical) return this.physical < other.physical ? -1 : 1;
    if (this.logical !== other.logical) return this.logical < other.logical ? -1 : 1;
    if (this.replicaId === other.replicaId) return 0;
    return this.replicaId < other.replicaId ? -1 : 1;
  }

  toJSON() {
    return { physical: this.physical, logical: this.logical, replica_id: this.replicaId.toString(10) };
  }

  static fromJSON(value) {
    if (!value || typeof value !== 'object') throw new Error('aamt-crdt: an HLC must be an object.');
    return new HLC(value.physical, value.logical, value.replica_id);
  }
}

const compareBytes = (a, b) => {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
  return a.length === b.length ? 0 : a.length < b.length ? -1 : 1;
};

const compareTera = (a, b) => {
  for (let i = 0; i < 4; i++) if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
  return 0;
};

const compareRecordKey = (a, b) => {
  const h = a.hlc.compare(b.hlc);
  if (h !== 0) return h;
  const d = compareBytes(a.data, b.data);
  if (d !== 0) return d;
  return compareTera(a.tera, b.tera);
};

const checkTera = (tera) => {
  if (!Array.isArray(tera) || tera.length !== 4 || tera.some((v) => typeof v !== 'number' || !Number.isFinite(v)))
    throw new Error('aamt-crdt: tera must be four finite numbers.');
  return tera.slice();
};

/** A last-writer-wins map: id -> { id, tera, data, hlc }, resolved by (hlc, data, tera). */
export class CrdtStore {
  constructor(records = new Map()) {
    this.records = records;
  }

  put(id, tera, data, hlc) {
    const key = String(id);
    if (!(hlc instanceof HLC)) throw new Error('aamt-crdt: put needs an HLC.');
    const record = { id: key, tera: checkTera(tera), data: new Uint8Array(data), hlc };
    const existing = this.records.get(key);
    if (existing && compareRecordKey(record, existing) <= 0) return false;
    this.records.set(key, record);
    return true;
  }

  merge(other) {
    const result = new CrdtStore(new Map(this.records));
    for (const record of other.records.values()) result.put(record.id, record.tera, record.data, record.hlc);
    return result;
  }

  get(id) {
    return this.records.get(String(id));
  }

  get size() {
    return this.records.size;
  }

  values() {
    return [...this.records.values()];
  }

  toJSON() {
    return {
      records: this.values()
        .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
        .map((record) => ({
          id: Number.isSafeInteger(Number(record.id)) ? Number(record.id) : record.id,
          tera: record.tera.slice(),
          data: b64encode(record.data),
          hlc: record.hlc.toJSON(),
        })),
    };
  }

  toJson() {
    return JSON.stringify(this.toJSON());
  }

  static fromJSON(value) {
    const payload = typeof value === 'string' ? JSON.parse(value) : value;
    if (!payload || !Array.isArray(payload.records)) throw new Error('aamt-crdt: expected { records: [...] }.');
    const store = new CrdtStore();
    for (const record of payload.records) {
      if (!record || record.id === undefined || !Array.isArray(record.tera)) throw new Error('aamt-crdt: malformed record.');
      store.put(record.id, record.tera, b64decode(record.data), HLC.fromJSON(record.hlc));
    }
    return store;
  }
}

/**
 * A random 63-bit replica id, persisted under `<stateDir>/replica_id` so every call on this install
 * returns the same id — the same contract as the Python helper, and the same reason: two writers
 * that both call `HLC.zero(1)` are a collision, not a replication.
 */
export function localReplicaId(stateDir) {
  mkdirSync(stateDir, { recursive: true });
  const path = join(stateDir, 'replica_id');
  if (existsSync(path)) {
    const text = readFileSync(path, 'utf8').trim();
    if (/^\d+$/.test(text)) return big(text);
  }
  const raw = randomBytes(8);
  let id = 0n;
  for (const byte of raw) id = (id << 8n) | BigInt(byte);
  id &= (1n << 63n) - 1n;
  if (id === 0n) id = 1n;
  const tmp = `${path}.tmp.${process.pid}`;
  writeFileSync(tmp, id.toString(10));
  renameSync(tmp, path);
  return id;
}

