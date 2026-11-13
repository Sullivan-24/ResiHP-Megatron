"""End-to-end demo of dynamic hybrid parallelism with live state migration.

The coordinator runs the progressive TP -> PP -> DP planner; workers run the reference
model with heterogeneous TP, uneven PP and stage-level migration, and migrate parameters
plus Adam state whenever the plan changes (see tests/dist/test_resilient_hetero.py).

``--backend gloo`` runs on CPU (the test harness); ``--backend nccl`` puts the same model
and the same instruction streams on this worker's GPU. Parameters and batches are always
generated on CPU from counter-based seeds, so both backends train the identical model.
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
from resihp.runtime.reference_model import RefModelConfig
from resihp.state.rng_data import IndexAddressableSampler
from resihp.train.hetero_binding import HeteroBinding
from resihp.train.resilient_loop import ResilientTrainer


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--gbs", type=int, default=16)
    ap.add_argument("--mbs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--channels", action="store_true")
    ap.add_argument("--iter-sleep", type=float, default=0.0, help="stretch iterations (demos/tests)")
    ap.add_argument("--amax-history", type=int, default=0, help="FP8-style delayed scaling history length")
    ap.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    client = WorkerClient(hb_interval=0.1).connect()
    device = "cpu"
    if args.backend == "nccl":
        torch.cuda.set_device(int(os.environ.get("RESIHP_LOCAL_RANK", "0")))
        device = f"cuda:{torch.cuda.current_device()}"
    mgr = ResilientCommManager(args.backend, device=device, timeout_s=args.timeout,
                               validate_timeout_s=args.timeout)
    cfg = RefModelConfig(amax_history=args.amax_history, device=device)
    sampler = IndexAddressableSampler(256, args.gbs, args.mbs, seed=7)
    binding = HeteroBinding(cfg, sampler, mgr, client, lr=args.lr, use_channels=args.channels,
                            injector=get_injector(), timeout=args.timeout, chunk=4, iter_sleep=args.iter_sleep)
    result = ResilientTrainer(client, mgr, binding).run(args.iterations)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    me = client.worker_id
    info = {"worker": me, "losses": result.losses, "generations": result.generations, "standby": result.standby,
            "reconfigurations": result.reconfigurations, "last_transfer": binding.last_transfer}
    if binding.committed is not None and not result.standby:
        a = binding.committed
        info["layout"] = {k: list(v) for k, v in a.layout.items()}
        info["generation"] = a.plan.generation
        state = {k: v.detach().cpu() for k, v in a.module.named_state().items()}
        state.update({k: v.cpu() for k, v in a.module.named_buffer_state().items()})
        torch.save(state, out / f"{me}.pt")
    (out / f"{me}.json").write_text(json.dumps(info))
    mgr.destroy()
    client.close()
    os._exit(0)


if __name__ == "__main__":
    main()
