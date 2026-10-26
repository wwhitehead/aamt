#!/usr/bin/env python3
# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""Live-transport tests for aamt_crdt_relay.RelayClient (WP-23 section 13,
continued): proves a Python process can be a REAL, live peer in a CRDT
room, not just a same-language property test.

  1. test_two_python_replicas_converge_live: two independent RelayClient
     instances, two independent replica ids, connected to a real running
     relay server over a real WebSocket -- one writes, the other's
     background listener receives and merges live, including a genuine
     same-id write collision resolved by the real HLC tiebreak, not a
     hand-simulated merge() call.

  2. test_cross_language_live_via_production_relay: the same Python
     client connected to the actual deployed production relay
     (wss://asamanthinks.com/crdt-relay/), converging live with a real,
     separate Node.js process speaking the identical JSON protocol --
     the same live cross-process proof this session already established
     for browser<->Node, now extended to Python<->Node through the same
     real infrastructure. Uses a unique per-run room name so this test
     never touches the shared demo room a real visitor might be using.

Run: python3 test_crdt_relay.py [--skip-production]
  (skip production if you're offline or don't want to touch the deployed
  relay; the local test still runs and still proves the transport works)
"""
import asyncio
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bindings", "python"))
from aamt_crdt import HLC, CrdtStore  # noqa: E402
from aamt_crdt_relay import RelayClient  # noqa: E402

LOCAL_RELAY_URL = "ws://127.0.0.1:8791"
PRODUCTION_RELAY_URL = "wss://asamanthinks.com/crdt-relay/"


async def test_two_python_replicas_converge_live():
    room = f"test-py-py-{int(time.time() * 1000)}"
    store_a, store_b = CrdtStore(), CrdtStore()
    clock_a, clock_b = HLC.zero(101), HLC.zero(102)

    async with RelayClient(LOCAL_RELAY_URL, room, 101, store_a, clock_a) as a, \
               RelayClient(LOCAL_RELAY_URL, room, 102, store_b, clock_b) as b:
        listen_a = asyncio.create_task(a.run_forever())
        listen_b = asyncio.create_task(b.run_forever())
        try:
            # Distinct writes on both sides.
            await a.put(1, [0.1, 0.2, 0.3, 0.4], b"from replica A, id 1")
            await b.put(2, [0.5, 0.5, 0.5, 0.5], b"from replica B, id 2")
            await asyncio.sleep(0.3)

            # A genuine collision: both write id=3 back-to-back on the SAME
            # process tick ordering isn't guaranteed across two real
            # connections, so whichever physically lands with the later HLC
            # wins -- not scripted, the real comparison decides it.
            await a.put(3, [0.9, 0.9, 0.9, 0.9], b"A's version of id 3")
            await b.put(3, [0.1, 0.1, 0.1, 0.1], b"B's version of id 3, written after A's")
            await asyncio.sleep(0.5)  # let both sides' listeners catch up
        finally:
            listen_a.cancel()
            listen_b.cancel()
            for t in (listen_a, listen_b):
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        recs_a = {r.id: bytes(r.data) for r in a.store.records()}
        recs_b = {r.id: bytes(r.data) for r in b.store.records()}
        assert recs_a == recs_b, f"replicas did not converge: A={recs_a} B={recs_b}"
        assert len(recs_a) == 3, f"expected 3 converged records, got {len(recs_a)}"
        assert recs_a[1] == b"from replica A, id 1"
        assert recs_a[2] == b"from replica B, id 2"
        # id=3's winner is whichever HLC is actually larger -- both are
        # valid real outcomes; what matters is both replicas agree.
        assert recs_a[3] in (b"A's version of id 3", b"B's version of id 3, written after A's")
        winner_hlc_a = a.store._records[3].hlc
        winner_hlc_b = b.store._records[3].hlc
        assert winner_hlc_a == winner_hlc_b, "converged winner's HLC must match on both sides"
    print(f"PASS test_two_python_replicas_converge_live: 3 records converged over a real "
          f"WebSocket relay, id=3 collision resolved to {recs_a[3]!r} by real HLC comparison "
          f"(both replicas agree)")


async def test_cross_language_live_via_production_relay():
    room = f"test-py-node-{int(time.time() * 1000)}"
    node_script = f"""
const WebSocket = require('ws');
const ws = new WebSocket('{PRODUCTION_RELAY_URL}?room={room}');
let received = null;
ws.on('open', () => {{
  console.log('NODE_CONNECTED');
  ws.send(JSON.stringify({{type: 'hello', from: 'node-test-client'}}));
}});
ws.on('message', (m) => {{
  const msg = JSON.parse(m.toString());
  if (msg.type === 'sync' && msg.storeJson) {{
    const parsed = JSON.parse(msg.storeJson);
    const rec = parsed.records.find(r => r.id === 99);
    if (rec) {{
      received = Buffer.from(rec.data, 'base64').toString();
      console.log('NODE_RECEIVED:' + received);
      ws.send(JSON.stringify({{type: 'sync', from: 'node-test-client', storeJson: JSON.stringify({{
        records: [{{id: 100, tera: [0.5,0.5,0.5,0.5],
          data: Buffer.from('from a real node process, replying').toString('base64'),
          hlc: {{physical: Date.now(), logical: 0, replica_id: '9007199254740993'}}}}]
      }})}}));
    }}
  }}
}});
setTimeout(() => process.exit(received ? 0 : 1), 8000);
"""
    relay_cwd = os.path.join(HERE, "..", "..", "..", "apps", "crdt-relay")
    node_proc = subprocess.Popen(
        ["node", "-e", node_script], cwd=relay_cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    await asyncio.sleep(1.5)  # let the Node process connect before Python joins

    store = CrdtStore()
    clock = HLC.zero(555)
    node_reply_seen = False
    async with RelayClient(PRODUCTION_RELAY_URL, room, 555, store, clock) as client:
        listen_task = asyncio.create_task(client.run_forever())
        try:
            await client.put(99, [0.5, 0.5, 0.5, 0.5], b"from a real python process, over production wss")
            for _ in range(20):
                await asyncio.sleep(0.3)
                if any(r.id == 100 for r in client.store.records()):
                    node_reply_seen = True
                    break
        finally:
            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass

    node_out, _ = node_proc.communicate(timeout=10)
    node_saw_python_write = "NODE_RECEIVED:from a real python process, over production wss" in node_out
    assert node_saw_python_write, f"Node process did not receive the Python write; output:\n{node_out}"
    assert node_reply_seen, "Python client did not receive the Node process's reply"
    node_reply_text = bytes(next(r for r in client.store.records() if r.id == 100).data)
    assert node_reply_text == b"from a real node process, replying"
    print("PASS test_cross_language_live_via_production_relay: a real Python process and a "
          "real, separate Node.js process converged live through the actual deployed "
          "production relay (wss://asamanthinks.com/crdt-relay/) -- Python's write reached "
          "Node, Node's reply reached Python, both over the public internet")


async def main():
    await test_two_python_replicas_converge_live()
    if "--skip-production" not in sys.argv:
        await test_cross_language_live_via_production_relay()
    print("PASS test_crdt_relay.py: all live-transport tests passed")


if __name__ == "__main__":
    asyncio.run(main())
