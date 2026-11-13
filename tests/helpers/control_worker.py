"""Dummy worker driven by trigger files; used by tests/dist/test_control_plane.py."""

import json
import os
import time
from pathlib import Path

from resihp.control.worker_client import ReconfigInterrupted, WorkerClient


def main() -> None:
    log = open(os.environ["TEST_EVENT_LOG"], "a")
    triggers = Path(os.environ["TEST_TRIGGER_DIR"])
    client = WorkerClient(hb_interval=0.1).connect()
    me = client.worker_id
    it = 0
    waited = False
    while True:
        try:
            prep = client.wait_prepare(timeout=60)
        except TimeoutError:
            return
        gen = prep["gen"]
        if me in prep["world"]:
            client.send_ready(gen)
        try:
            client.wait_commit(gen, timeout=60)
        except ReconfigInterrupted:
            continue
        log.write(json.dumps({"worker": me, "gen": gen, "world": prep["world"], "t": time.time()}) + "\n")
        log.flush()
        if me not in prep["world"]:
            continue
        while not client.reconfig_pending():
            if (triggers / f"stall_{me}").exists():
                time.sleep(3600)
            if not waited and (triggers / "wait").exists():
                waited = True
                client.set_progress(iteration=it, in_collective=True, op_seq=10**6)
                while not client.reconfig_pending():
                    time.sleep(0.02)
                break
            client.set_progress(iteration=it, in_collective=False)
            client.set_recovery_info(committed_iteration=it)
            it += 1
            time.sleep(0.02)


if __name__ == "__main__":
    main()
