#!/usr/bin/env python3
# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""End-to-end test for aamt_crdt_daemon.py (WP-23 section 13, continued):
proves the persistence daemon is a genuinely CONTINUOUS system, not a
one-shot pack wearing a daemon's clothes -- real subprocesses throughout,
no mocks:

  1. A real relay server subprocess (apps/crdt-relay/src/server.js) and a
     real daemon subprocess (aamt_crdt_daemon.py, run as an actual CLI
     invocation, not called as a library) are started.
  2. A separate RelayClient in THIS test process joins the same room as a
     second real writer and writes a first batch of records.
  3. After the debounce window, the daemon's output .aamt file is opened
     with AamtSubstrate and every record is confirmed present, correct,
     and mirror+RS-protected via read_verified -- proving the daemon
     actually reacted to a live write it did not originate.
  4. A SECOND batch is written, including a genuine id collision with
     batch 1 that the later write must win. After another debounce
     window, the file is re-opened and confirmed to reflect the FULL,
     updated converged state -- proving this is continuous sync, not a
     first-write-only pack.
  5. A real corrupt-and-recover check on the current (twice-repacked)
     file confirms mirror+RS protection survives repeated re-packing,
     not just the initial one.
  6. A THIRD write is made and the daemon is sent SIGTERM immediately
     (before the debounce window elapses) -- confirms the daemon's
     graceful-shutdown final flush actually fires, not just the
     background debounce timer.

Run: python3 test_crdt_daemon.py
"""
import asyncio
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bindings", "python"))
from aamt_crdt import HLC, CrdtStore, local_replica_id  # noqa: E402
from aamt_crdt_relay import RelayClient  # noqa: E402
import aamt_ffi  # noqa: E402

RELAY_PORT = 8793
RELAY_URL = f"ws://127.0.0.1:{RELAY_PORT}"
DEBOUNCE_SECONDS = 0.3
SETTLE_MARGIN = 1.2  # real wall-clock margin beyond the debounce window before we check the file


def start_relay_server():
    relay_dir = os.path.join(HERE, "..", "..", "..", "apps", "crdt-relay")
    env = dict(os.environ, CRDT_RELAY_PORT=str(RELAY_PORT), CRDT_RELAY_HOST="127.0.0.1")
    proc = subprocess.Popen(
        ["node", "src/server.js"], cwd=relay_dir, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(1.0)
    assert proc.poll() is None, f"relay server exited early:\n{proc.stdout.read()}"
    return proc


def start_daemon(room: str, state_dir: str, output_path: str):
    daemon_script = os.path.join(HERE, "..", "bindings", "python", "aamt_crdt_daemon.py")
    proc = subprocess.Popen(
        [sys.executable, daemon_script,
         "--relay-url", RELAY_URL, "--room", room,
         "--state-dir", state_dir, "--output", output_path,
         "--debounce-seconds", str(DEBOUNCE_SECONDS), "-v"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(1.0)
    assert proc.poll() is None, f"daemon exited early:\n{proc.stdout.read()}"
    return proc


def f32(tera) -> tuple:
    import ctypes
    return tuple(ctypes.c_float(v).value for v in tera)


async def main() -> int:
    tmp = tempfile.mkdtemp(prefix="aamt_crdt_daemon_test_")
    room = f"test-daemon-{int(time.time() * 1000)}"
    output_path = os.path.join(tmp, "daemon-converged.aamt")
    daemon_state_dir = os.path.join(tmp, "daemon-state")

    relay_proc = start_relay_server()
    daemon_proc = start_daemon(room, daemon_state_dir, output_path)
    print(f"PASS setup: real relay server (port {RELAY_PORT}) and real daemon subprocess "
          f"(room={room}) both running")

    try:
        # ── batch 1: a real external writer's records reach the daemon live ──
        writer_store = CrdtStore()
        writer_clock = HLC.zero(9001)
        async with RelayClient(RELAY_URL, room, 9001, writer_store, writer_clock) as writer:
            listen = asyncio.create_task(writer.run_forever())
            try:
                await writer.put(1, [0.1, 0.2, 0.3, 0.4], b"batch1 record 1")
                await writer.put(2, [0.5, 0.5, 0.5, 0.5], b"batch1 record 2")
                await writer.put(3, [0.9, 0.1, 0.9, 0.1], b"batch1 record 3 -- will be overwritten in batch 2")
                await asyncio.sleep(DEBOUNCE_SECONDS + SETTLE_MARGIN)
            finally:
                listen.cancel()
                try:
                    await listen
                except asyncio.CancelledError:
                    pass

        assert os.path.exists(output_path), "daemon did not produce an output file after batch 1"
        expect = {
            1: ((0.1, 0.2, 0.3, 0.4), b"batch1 record 1"),
            2: ((0.5, 0.5, 0.5, 0.5), b"batch1 record 2"),
            3: ((0.9, 0.1, 0.9, 0.1), b"batch1 record 3 -- will be overwritten in batch 2"),
        }
        with aamt_ffi.AamtSubstrate.open(output_path) as sub:
            assert sub.record_count == 6, f"expected 3 primaries + 3 mirrors, got {sub.record_count}"
            for rec_id, (tera, data) in expect.items():
                r = sub.read_verified(rec_id, tera)
                assert r.status == aamt_ffi.ReadStatus.OK and r.data == data, (rec_id, r.status, r.data)
        print(f"PASS batch 1: daemon reacted to a live write from a real external peer it did not "
              f"originate -- {output_path} contains all 3 records, mirror+RS-protected (6 total incl. mirrors)")

        # ── batch 2: continuous sync, including a real id collision ──
        writer_store2 = CrdtStore()
        writer_clock2 = HLC.zero(9002)
        async with RelayClient(RELAY_URL, room, 9002, writer_store2, writer_clock2) as writer2:
            listen2 = asyncio.create_task(writer2.run_forever())
            try:
                await asyncio.sleep(0.2)  # let this peer's hello/sync handshake settle first
                await writer2.put(3, [0.2, 0.2, 0.2, 0.2], b"batch2 record 3 -- LATER, must win")
                await writer2.put(4, [0.7, 0.7, 0.7, 0.7], b"batch2 record 4")
                await asyncio.sleep(DEBOUNCE_SECONDS + SETTLE_MARGIN)
            finally:
                listen2.cancel()
                try:
                    await listen2
                except asyncio.CancelledError:
                    pass

        expect[3] = ((0.2, 0.2, 0.2, 0.2), b"batch2 record 3 -- LATER, must win")
        expect[4] = ((0.7, 0.7, 0.7, 0.7), b"batch2 record 4")
        with aamt_ffi.AamtSubstrate.open(output_path) as sub:
            assert sub.record_count == 8, f"expected 4 primaries + 4 mirrors, got {sub.record_count}"
            for rec_id, (tera, data) in expect.items():
                r = sub.read_verified(rec_id, tera)
                assert r.status == aamt_ffi.ReadStatus.OK and r.data == data, (rec_id, r.status, r.data)
            rs_offset = sub.record_file_offset(1, expect[1][0])
        print("PASS batch 2: file re-packed with the FULL updated converged state (4 records, "
              "id=3 collision correctly resolved to the later write) -- this is continuous sync, "
              "not a first-write-only pack")

        # ── real corrupt-and-recover on the twice-repacked file ──
        with open(output_path, "r+b") as raw:
            raw.seek(rs_offset + 32 + 10)
            original = raw.read(1)
            raw.seek(rs_offset + 32 + 10)
            raw.write(bytes([original[0] ^ 0xF0]))
        with aamt_ffi.AamtSubstrate.open(output_path) as sub:
            r = sub.read_verified(1, expect[1][0])
            assert r.status == aamt_ffi.ReadStatus.OK_RS_REPAIRED, r.status
            assert r.data == expect[1][1]
        print("PASS corrupt-and-recover survives repeated re-packing: a real byte corruption on "
              "the twice-repacked file is still caught and RS-repaired correctly, not just on a "
              "file's first-ever pack")

        # ── graceful shutdown: a write immediately followed by SIGTERM must still be flushed ──
        writer_store3 = CrdtStore()
        writer_clock3 = HLC.zero(9003)
        async with RelayClient(RELAY_URL, room, 9003, writer_store3, writer_clock3) as writer3:
            listen3 = asyncio.create_task(writer3.run_forever())
            try:
                await asyncio.sleep(0.2)
                await writer3.put(5, [0.4, 0.4, 0.4, 0.4], b"batch3 record 5 -- written right before SIGTERM")
                await asyncio.sleep(0.05)  # deliberately well under DEBOUNCE_SECONDS
                daemon_proc.send_signal(signal.SIGTERM)
                daemon_proc.wait(timeout=10)
            finally:
                listen3.cancel()
                try:
                    await listen3
                except asyncio.CancelledError:
                    pass

        with aamt_ffi.AamtSubstrate.open(output_path) as sub:
            r = sub.read_verified(5, [0.4, 0.4, 0.4, 0.4])
            assert r.status == aamt_ffi.ReadStatus.OK, r.status
            assert r.data == b"batch3 record 5 -- written right before SIGTERM"
        print("PASS graceful shutdown: a write made 50ms before SIGTERM (well under the "
              f"{DEBOUNCE_SECONDS}s debounce window) was still flushed to disk by the daemon's "
              "final-pack-on-shutdown path, not lost")

        print("PASS test_crdt_daemon: the persistence daemon is a genuinely continuous system -- "
              "real external writes, real re-packing, real corruption recovery across repacks, "
              "real graceful-shutdown flush, no mocks")
        return 0
    finally:
        for proc, name in ((daemon_proc, "daemon"), (relay_proc, "relay")):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
