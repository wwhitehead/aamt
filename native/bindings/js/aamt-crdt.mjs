// Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
// Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
// aamt-crdt.mjs — the Node entry point of the CRDT interchange. Every rule lives in aamt-crdt-core.mjs, which also
// runs in a browser; this file adds only what needs a filesystem: a replica id persisted per install.
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { big, randomReplicaId } from './aamt-crdt-core.mjs';

export { HLC, CrdtStore, b64encode, b64decode, randomReplicaId } from './aamt-crdt-core.mjs';

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
  const id = randomReplicaId();
  const tmp = `${path}.tmp.${process.pid}`;
  writeFileSync(tmp, id.toString(10));
  renameSync(tmp, path);
  return id;
}

