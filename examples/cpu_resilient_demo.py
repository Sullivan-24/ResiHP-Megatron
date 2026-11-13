"""CPU end-to-end demo of checkpoint-free recovery (replica-level).

Run under the ResiHP launcher, e.g. kill one worker at iteration 3:

    RESIHP_FAULTS="kill@iter=3:target=n1g1:phase=reduce" \
    resihp-run local --nnodes 2 --nproc-per-node 2 \
        --planner resihp.train.demo_planners:dp_8mb -- python examples/cpu_resilient_demo.py --out /tmp/demo
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from resihp.comm import ResilientCommManager
from resihp.control.worker_client import WorkerClient
from resihp.inject.faults import get_injector
from resihp.runtime.optim import Adam
from resihp.runtime.reference_model import RefModelConfig, RefStage
from resihp.state.checkpoint import SliceCheckpointer
from resihp.state.redundancy import HostSnapshotManager
from resihp.state.rng_data import IndexAddressableSampler
from resihp.train.dp_binding import DPBinding
from resihp.train.resilient_loop import ResilientTrainer


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--gbs", type=int, default=16)
    ap.add_argument("--mbs", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--snapshot-interval", type=int, default=0)
    ap.add_argument("--ckpt-dir", default="")
    ap.add_argument("--ckpt-interval", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    client = WorkerClient(hb_interval=0.1).connect()
    mgr = ResilientCommManager("gloo", timeout_s=args.timeout, validate_timeout_s=args.timeout)
    cfg = RefModelConfig()
    sampler = IndexAddressableSampler(args.num_samples, args.gbs, args.mbs, seed=7)
    model = RefStage(cfg, 0, cfg.num_layers)
    opt = Adam(model.named_state(), lr=args.lr)
    binding = DPBinding(cfg, sampler, model, opt, mgr, client, injector=get_injector(),
                        snapshots=HostSnapshotManager(args.snapshot_interval) if args.snapshot_interval else None,
                        checkpointer=SliceCheckpointer(args.ckpt_dir) if args.ckpt_dir else None,
                        checkpoint_interval=args.ckpt_interval, timeout=args.timeout)
    result = ResilientTrainer(client, mgr, binding).run(args.iterations)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    me = client.worker_id
    torch.save({k: v.detach() for k, v in model.named_state().items()}, out / f"{me}.pt")
    (out / f"{me}.json").write_text(json.dumps({
        "worker": me, "losses": result.losses, "generations": result.generations,
        "reconfigurations": result.reconfigurations, "step_count": opt.step_count,
        "snapshots_held": sorted(binding.snapshots._held) if binding.snapshots else [],
    }))
    mgr.destroy()
    client.close()
    os._exit(0)


if __name__ == "__main__":
    main()
