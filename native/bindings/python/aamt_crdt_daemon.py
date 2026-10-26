#!/usr/bin/env python3
# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""Persistence daemon for the live CRDT relay (WP-23 section 13, continued):
watches a relay room and keeps a real, mirror+RS-protected .aamt file on
disk continuously up to date with the room's converged state -- the
CRDT-relay demo (RelayClient: live sync, proven this session in Python,
JS, and C#) and the CRDT-to-file demo (CrdtStore.to_aamt(mirrored_ecc=True):
one-shot pack, proven separately) become one running system instead of two
adjacent, unconnected proofs.

Pure listener, writes nothing of its own to the room: joins via
RelayClient the same way any other peer does, merges every incoming
update, and repacks the CURRENT converged store to `output_path` shortly
after each real change via RelayClient's on_update hook -- debounced (see
--debounce-seconds) so a burst of rapid writes doesn't trigger a fresh
RS/mirror pack (real file I/O, real native library calls) on every single
message. Writes atomically (temp file + os.replace) so a reader never
observes a partially-written file, and packing runs in a thread executor
so it doesn't stall the event loop's message handling mid-pack.

    python3 aamt_crdt_daemon.py \\
        --relay-url wss://asamanthinks.com/crdt-relay/ \\
        --room aamt_crdt_demo_v1 \\
        --state-dir /path/to/daemon-state \\
        --output /path/to/converged.aamt

Honest limits, stated rather than hidden:
  - Single room, single output file per process. Watching N rooms means
    running N processes -- no multiplexing built in, because the room
    boundary IS the file boundary and conflating them would need a real
    design decision (one file per room? one merged file across rooms?)
    this daemon doesn't need to make.
  - No reconnect/backoff on a dropped connection (same honest limit
    RelayClient itself states) -- a process supervisor (pm2, systemd)
    restarting the whole process on exit is the intended recovery path,
    not retry logic inside this module.
  - The very first pack only happens once at least one real update
    arrives after connecting -- an empty room produces no file, rather
    than packing a spurious empty .aamt on startup.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from aamt_crdt import HLC, CrdtStore, local_replica_id  # noqa: E402
from aamt_crdt_relay import RelayClient  # noqa: E402

logger = logging.getLogger("aamt_crdt_daemon")


class PersistenceDaemon:
    """Owns one RelayClient for one room and keeps `output_path` in sync
    with its converged store. See module docstring for the debounce and
    atomic-write rationale."""

    def __init__(self, relay_url: str, room: str, state_dir: str, output_path: str,
                 debounce_seconds: float = 1.0, lib_path: str | None = None,
                 max_depth: int = 8, bucket_capacity: int = 32):
        self.relay_url = relay_url
        self.room = room
        self.output_path = output_path
        self.debounce_seconds = debounce_seconds
        self.lib_path = lib_path
        self.max_depth = max_depth
        self.bucket_capacity = bucket_capacity
        self.replica_id = local_replica_id(state_dir)
        self.pack_count = 0  # real packs performed, exposed for tests/observability
        self._client: RelayClient | None = None
        self._pack_task: asyncio.Task | None = None

    def _on_update(self, client: RelayClient) -> None:
        """RelayClient.on_update hook -- fires on the asyncio event loop's
        own thread whenever a real merge changes the store. Cancelling and
        recreating the debounce task on every call is the whole debounce:
        only the LAST update in a rapid burst actually survives long
        enough to trigger a pack."""
        if self._pack_task is not None and not self._pack_task.done():
            self._pack_task.cancel()
        self._pack_task = asyncio.create_task(self._debounced_pack())

    async def _debounced_pack(self) -> None:
        try:
            await asyncio.sleep(self.debounce_seconds)
        except asyncio.CancelledError:
            return  # superseded by a newer update before the debounce elapsed
        await self.pack_now()

    async def pack_now(self) -> int:
        """Packs the CURRENT converged store immediately, bypassing the
        debounce -- used internally after the debounce window elapses, and
        exposed for callers (tests, a graceful-shutdown final flush) that
        need a synchronous "pack whatever we have right now"."""
        assert self._client is not None, "pack_now called before run() connected"
        store = self._client.store
        count = len(store)
        if count == 0:
            return 0  # nothing converged yet -- don't pack a spurious empty file
        tmp_path = f"{self.output_path}.tmp.{os.getpid()}"
        loop = asyncio.get_running_loop()
        # to_aamt does real file I/O + native library calls through ctypes
        # -- off the event loop so a large pack doesn't stall message
        # handling for other peers' writes arriving mid-pack.
        await loop.run_in_executor(
            None,
            lambda: store.to_aamt(tmp_path, self.max_depth, self.bucket_capacity,
                                   False, True, self.lib_path),
        )
        os.replace(tmp_path, self.output_path)
        self.pack_count += 1
        logger.info("packed %d record(s) to %s (pack #%d)", count, self.output_path, self.pack_count)
        return count

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        """Connects, listens forever (or until `stop_event` is set), and
        does one final synchronous pack on the way out so a clean shutdown
        never loses the last debounce window's worth of writes."""
        store = CrdtStore()
        clock = HLC.zero(self.replica_id)
        async with RelayClient(self.relay_url, self.room, self.replica_id, store, clock,
                                on_update=self._on_update) as client:
            self._client = client
            logger.info("connected as replica %d, room=%s, watching for writes", self.replica_id, self.room)
            listen_task = asyncio.create_task(client.run_forever())
            try:
                if stop_event is not None:
                    stop_wait = asyncio.create_task(stop_event.wait())
                    await asyncio.wait({listen_task, stop_wait}, return_when=asyncio.FIRST_COMPLETED)
                    stop_wait.cancel()
                else:
                    await listen_task
            finally:
                listen_task.cancel()
                try:
                    await listen_task
                except asyncio.CancelledError:
                    pass
                if self._pack_task is not None:
                    self._pack_task.cancel()
                await self.pack_now()  # final flush: don't lose an in-flight debounce window


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--relay-url", required=True, help="e.g. wss://asamanthinks.com/crdt-relay/")
    p.add_argument("--room", required=True, help="e.g. aamt_crdt_demo_v1")
    p.add_argument("--state-dir", required=True, help="persists this daemon's own replica_id across restarts")
    p.add_argument("--output", required=True, dest="output_path", help="path to the .aamt file to keep in sync")
    p.add_argument("--debounce-seconds", type=float, default=1.0)
    p.add_argument("--lib-path", default=None, help="override libaamt shared library path")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


async def _main_async(args) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    daemon = PersistenceDaemon(args.relay_url, args.room, args.state_dir, args.output_path,
                                debounce_seconds=args.debounce_seconds, lib_path=args.lib_path)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass  # e.g. Windows: no add_signal_handler -- Ctrl+C still raises KeyboardInterrupt
    await daemon.run(stop_event)


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    try:
        asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
