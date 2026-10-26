# aamt-crdt.js

The JavaScript side of the libaamt CRDT interchange. It is a port of
`bindings/python/aamt_crdt.py`, not a second design: both modules emit the same
JSON and resolve the same winner.

```js
import { CrdtStore, HLC, localReplicaId } from './aamt-crdt.mjs';

const replica = localReplicaId('/path/to/state');
let clock = HLC.zero(replica);
const store = new CrdtStore();
store.put(42, [0.5, 0.5, 0.5, 0.5], new TextEncoder().encode('from js'), (clock = clock.tick()));

const converged = store.merge(CrdtStore.fromJSON(pythonJson));
console.log(converged.toJson());
```

## The rules

- HLC is `(physical, logical, replica_id)`, compared lexicographically.
- A record wins by `(hlc, data, tera)`; HLC alone is not a total order over records.
- `replica_id` is a JSON **string** on the wire; a 63-bit value is beyond JS's
  safe-integer range.
- `data` is base64 on the wire, bytes in memory.
- `merge` is commutative, associative and idempotent.

## Verify

```sh
node bindings/js/test-crdt.mjs
```

The Python↔JS interoperability check used during development creates a store in
`aamt_crdt.py`, imports its JSON in `aamt-crdt.mjs`, re-exports it, and compares
the two objects. The two `to_json` outputs are byte-identical for the same record
set.
