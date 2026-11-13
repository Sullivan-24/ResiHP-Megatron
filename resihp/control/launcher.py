"""``resihp-run``: start the coordinator, node agents, or a local all-in-one job.

    resihp-run coordinator --port 29400 --expected-workers 16
    resihp-run agent --node-id n0 --coordinator c0:29400,c1:29400 --nproc 8 -- python train.py ...
    resihp-run local --nnodes 2 --nproc-per-node 4 -- python examples/cpu_resilient_demo.py
"""

from __future__ import annotations

import argparse
import importlib
import signal
import threading
import time

from resihp.control.agent import NodeAgent
from resihp.control.coordinator import Coordinator
from resihp.control.health import CompositeProbe, IBPortProbe, NVMLProbe
from resihp.control.kvstore import InMemoryKV
from resihp.utils.rpc import parse_addr


def _load_planner(spec: str | None):
    if not spec:
        return None
    mod, fn = spec.split(":", 1)
    return getattr(importlib.import_module(mod), fn)


def _kv(spec: str):
    if spec == "memory":
        return InMemoryKV()
    if spec.startswith("etcd://"):
        from resihp.control.kvstore import EtcdKV

        host, port = parse_addr(spec[len("etcd://"):])
        return EtcdKV(host, port)
    raise ValueError(f"unknown kv backend {spec}")


def _wait_forever(stop: threading.Event) -> None:
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        while not stop.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="resihp-run")
    sub = ap.add_subparsers(dest="role", required=True)

    c = sub.add_parser("coordinator")
    c.add_argument("--host", default="0.0.0.0")
    c.add_argument("--port", type=int, default=29400)
    c.add_argument("--store-port", type=int, default=29401)
    c.add_argument("--expected-workers", type=int)
    c.add_argument("--min-workers", type=int, default=1)
    c.add_argument("--lease-timeout", type=float, default=3.0)
    c.add_argument("--hang-timeout", type=float, default=300.0)
    c.add_argument("--kv", default="memory")
    c.add_argument("--planner", help="module:function implementing PlannerFn")
    c.add_argument("--id", default="coord-0")
    c.add_argument("--leader-ttl", type=float, help="enable HA election through the KV store")

    a = sub.add_parser("agent")
    a.add_argument("--node-id", required=True)
    a.add_argument("--coordinator", required=True, help="comma-separated host:port list (HA replicas)")
    a.add_argument("--nproc", type=int, default=8)
    a.add_argument("--devices", default="", help="comma-separated device list, default cuda:0..nproc-1")
    a.add_argument("--host", default="127.0.0.1")
    a.add_argument("--worker-timeout", type=float, default=2.0, help="heartbeat gap that means a hang")
    a.add_argument("--startup-timeout", type=float, default=600.0,
                   help="heartbeat gap tolerated while a worker starts up or rebuilds a generation")
    a.add_argument("cmd", nargs=argparse.REMAINDER)

    loc = sub.add_parser("local")
    loc.add_argument("--nnodes", type=int, default=2)
    loc.add_argument("--nproc-per-node", type=int, default=2)
    loc.add_argument("--planner")
    loc.add_argument("--hang-timeout", type=float, default=60.0)
    loc.add_argument("--worker-timeout", type=float, default=2.0)
    loc.add_argument("--startup-timeout", type=float, default=600.0)
    loc.add_argument("--devices", default="", help="comma-separated device list, default cpu")
    loc.add_argument("cmd", nargs=argparse.REMAINDER)

    args = ap.parse_args(argv)
    stop = threading.Event()

    if args.role == "coordinator":
        coord = Coordinator(host=args.host, port=args.port, store_port=args.store_port, kv=_kv(args.kv),
                            planner=_load_planner(args.planner), expected_workers=args.expected_workers,
                            min_workers=args.min_workers, lease_timeout=args.lease_timeout,
                            hang_timeout=args.hang_timeout, coordinator_id=args.id, leader_ttl=args.leader_ttl)
        coord.start()
        _wait_forever(stop)
        coord.stop()
        return 0

    if args.role == "agent":
        cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
        addrs = [parse_addr(s) for s in args.coordinator.split(",")]
        devices = args.devices.split(",") if args.devices else [f"cuda:{i}" for i in range(args.nproc)]
        agent = NodeAgent(args.node_id, addrs, host=args.host, worker_timeout=args.worker_timeout,
                          startup_timeout=args.startup_timeout,
                          health_probe=CompositeProbe(NVMLProbe(), IBPortProbe())).start()
        if cmd:
            agent.spawn(cmd, args.nproc, devices=devices)
        _wait_forever(stop)
        agent.stop()
        return 0

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    coord = Coordinator(planner=_load_planner(args.planner), expected_workers=args.nnodes * args.nproc_per_node,
                        hang_timeout=args.hang_timeout).start()
    agents = []
    devices = args.devices.split(",") if args.devices else ["cpu"] * args.nproc_per_node
    for n in range(args.nnodes):
        agent = NodeAgent(f"n{n}", [coord.address], worker_timeout=args.worker_timeout,
                          startup_timeout=args.startup_timeout).start()
        agent.spawn(cmd, args.nproc_per_node, devices=devices)
        agents.append(agent)

    def all_exited() -> bool:
        return all(w.proc is not None and w.proc.poll() is not None for ag in agents for w in ag.workers.values())

    while not all_exited():
        time.sleep(0.5)
    rc = max((w.proc.returncode or 0) for ag in agents for w in ag.workers.values()
             if w.proc is not None and w.worker_id in coord.world)
    for ag in agents:
        ag.stop()
    coord.stop()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
