// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// aamt-crdt-core.mjs — the JavaScript side of the libaamt CRDT interchange (WP-23 §13.2–13.3).
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


// Base64 without Buffer, so this file runs in a browser as well as Node. Standard alphabet with padding: the same
// text Buffer and Python's base64.b64encode produce, which the cross-language round trip depends on.
const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
export const b64encode = (bytes) => {
  let out = '';
  for (let i = 0; i < bytes.length; i += 3) {
    const n = (bytes[i] << 16) | ((bytes[i + 1] ?? 0) << 8) | (bytes[i + 2] ?? 0);
    out += B64[(n >> 18) & 63] + B64[(n >> 12) & 63] + (i + 1 < bytes.length ? B64[(n >> 6) & 63] : '=') + (i + 2 < bytes.length ? B64[n & 63] : '=');
  }
  return out;
};
export const b64decode = (text) => {
  const clean = String(text).replace(/[^A-Za-z0-9+/]/g, '');
  const out = new Uint8Array(Math.floor((clean.length * 3) / 4));
  let o = 0;
  for (let i = 0; i < clean.length; i += 4) {
    const n = (B64.indexOf(clean[i]) << 18) | (B64.indexOf(clean[i + 1]) << 12) | ((B64.indexOf(clean[i + 2]) & 63) << 6) | (B64.indexOf(clean[i + 3]) & 63);
    out[o++] = (n >> 16) & 255;
    if (i + 2 < clean.length) out[o++] = (n >> 8) & 255;
    if (i + 3 < clean.length) out[o++] = n & 255;
  }
  return out.subarray(0, o);
};

export const big = (value) => {
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
 * A random 63-bit replica id from Web Crypto (browsers and Node 19+). The Node entry point (aamt-crdt.mjs) persists
 * one per install; a browser tab keeps one per tab, which is what makes two tabs two replicas.
 */
export function randomReplicaId() {
  const raw = new Uint8Array(8);
  globalThis.crypto.getRandomValues(raw);
  let id = 0n;
  for (const byte of raw) id = (id << 8n) | BigInt(byte);
  id &= (1n << 63n) - 1n;
  return id === 0n ? 1n : id;
}
