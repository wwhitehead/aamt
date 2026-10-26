# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""Live cross-device transport for CrdtStore (WP-23 section 13, continued):
the same WebSocket relay CrdtTab.jsx uses (apps/crdt-relay in this repo,
deployed at wss://asamanthinks.com/crdt-relay/) so a Python process can be
a real, LIVE peer in a CRDT room -- not just a one-shot to_json()/from_json()
exchange. Before this module, aamt_crdt.py's cross-language interop was
proven at the merge level (a Python-built store and a JS-built store
converge correctly once you hand one to the other) but only the browser
could join a room in real time; a desktop Python process had no live
transport at all. This closes that gap by speaking the identical
3-message protocol every language now shares:

    {"type": "hello", "from": "<replica_id>"}
        Announces presence and asks for anyone's current state -- sent on
        connect, and replied to by every peer already in the room so a
        late joiner converges with writes that happened before it arrived,
        not just future ones.
    {"type": "sync", "from": "<replica_id>", "storeJson": "<CrdtStore.to_json()>"}
        A full store snapshot, sent as a hello-reply or after a local put().

Requires the `websockets` package (already used elsewhere in this repo,
e.g. maiiam-alchemist/packages/harmonic-embodied's unity_ws.py). Also
requires `certifi` for wss:// connections (used to build an explicit
SSL context -- see the CERTIFICATE_VERIFY_FAILED note in __aenter__).

    import asyncio
    from aamt_crdt import HLC, CrdtStore, local_replica_id
    from aamt_crdt_relay import RelayClient

    async def main():
        replica_id = local_replica_id("/path/to/state")
        async with RelayClient(
            "wss://asamanthinks.com/crdt-relay/", "aamt_crdt_demo_v1",
            replica_id, CrdtStore(), HLC.zero(replica_id),
        ) as client:
            listen_task = asyncio.create_task(client.run_forever())
            await client.put(42, [0.5, 0.5, 0.5, 0.5], b"from a real python process")
            await asyncio.sleep(5)  # stay connected long enough to receive replies
            listen_task.cancel()

    asyncio.run(main())
"""
from __future__ import annotations

import json

from aamt_crdt import HLC, CrdtStore


class RelayClient:
    """One WebSocket connection to one room. Owns `store`/`clock` and
    mutates them in place as messages arrive, mirroring CrdtTab.jsx's
    handleRemoteMessage/writeSlot exactly: `put()` is tick-write-broadcast,
    incoming 'hello' gets an immediate full-state reply, incoming 'sync'
    gets merged and advances the clock past the remote HLC. Deliberately
    thin -- no reconnect/backoff logic, no persistence -- a caller that
    needs those wraps this, the same way the relay server itself stays
    dumb and lets CrdtStore.merge carry the correctness."""

    def __init__(self, url: str, room: str, replica_id: int, store: CrdtStore, clock: HLC,
                 on_update=None):
        self.url = url
        self.room = room
        self.replica_id = str(replica_id)
        self.store = store
        self.clock = clock
        self._ws = None
        # Called (with this client) after every incoming 'sync' actually
        # changes `self.store` -- the extension point aamt_crdt_daemon.py
        # uses to trigger a debounced re-pack. Not called for 'hello' (no
        # store change) or for a merge that changes nothing (a duplicate
        # or stale message re-delivered).
        self.on_update = on_update

    async def __aenter__(self) -> "RelayClient":
        import websockets  # local import: this module is the only thing in the
        # CRDT layer that needs it, matching aamt_crdt.to_aamt's local `import aamt_ffi`
        sep = "&" if "?" in self.url else "?"
        full_url = f"{self.url}{sep}room={self.room}"
        kwargs = {}
        if full_url.startswith("wss://"):
            # Some Python.org macOS installs ship without their OpenSSL
            # cert.pem symlinked (a machine-setup gap, not something this
            # module should assume is fixed) -- found live when the
            # production wss:// test failed with CERTIFICATE_VERIFY_FAILED
            # despite curl/openssl on the same machine trusting the same
            # cert fine. Building the SSL context from certifi's bundle
            # explicitly makes this portable across machines instead of
            # depending on each one's system trust store being wired up.
            import certifi
            import ssl
            kwargs["ssl"] = ssl.create_default_context(cafile=certifi.where())
        self._ws = await websockets.connect(full_url, **kwargs)
        await self._send({"type": "hello", "from": self.replica_id})
        return self

    async def __aexit__(self, *exc) -> None:
        if self._ws is not None:
            await self._ws.close()

    async def _send(self, msg: dict) -> None:
        await self._ws.send(json.dumps(msg))

    async def broadcast(self) -> None:
        """Send the current converged store as a full sync -- what a peer's
        'hello' gets replied with, and what a local put() triggers."""
        await self._send({"type": "sync", "from": self.replica_id, "storeJson": self.store.to_json()})

    async def put(self, id_: int, tera, data: bytes) -> None:
        """Local write: tick the clock, apply it, broadcast it -- the same
        three steps CrdtTab.jsx's writeSlot performs, so a write made here
        is indistinguishable, to every other peer, from one made in a
        browser tab or a Unity client."""
        self.clock = self.clock.tick()
        self.store.put(id_, tera, data, self.clock)
        await self.broadcast()

    async def _handle(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        if msg.get("from") == self.replica_id:
            return
        if msg.get("type") == "hello":
            await self.broadcast()
        elif msg.get("type") == "sync" and "storeJson" in msg:
            remote = CrdtStore.from_json(msg["storeJson"])
            merged = self.store.merge(remote)
            changed = merged != self.store
            self.store = merged
            for rec in remote.records():
                self.clock = self.clock.update(rec.hlc)
            if changed and self.on_update is not None:
                self.on_update(self)

    async def run_forever(self) -> None:
        """Processes incoming messages until the connection closes (or this
        task is cancelled). Run as a background asyncio task alongside
        whatever the caller actually wants to do (write, wait, exit)."""
        async for raw in self._ws:
            await self._handle(raw)
