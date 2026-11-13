"""ResilientCommManager: generation-scoped communicators that can be rebuilt without hanging.

Root causes of hangs we design out (all observed in the InternEvo prototype):
* dead ranks participating in group creation -> survivors-only creation
  (``shrink_group`` on NCCL, or a fresh world on a generation-scoped store prefix);
* creation order depending on local state -> plan-derived canonical order,
  one ``split_group`` per disjoint layer, entered by every rank of the new world;
* rendezvous store living on a training rank -> store hosted by the coordinator;
* blocking forever inside a collective -> abortable waits (``guarded_wait``) and
  per-group validation with a deadline before a generation is declared READY.
"""

from __future__ import annotations

import datetime
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

from resihp.comm.group_spec import GroupSpec, canonical_order, split_layers
from resihp.utils.log import get_logger

log = get_logger(__name__)


class CommAborted(RuntimeError):
    pass


class CommTimeout(RuntimeError):
    pass


class GenerationBuildError(RuntimeError):
    def __init__(self, msg: str, suspects: tuple[int, ...] = ()):
        super().__init__(msg)
        self.suspects = suspects


@dataclass
class GenerationComms:
    generation: int
    rank: int
    world: tuple[str, ...]
    specs: dict[str, GroupSpec]
    groups: dict[str, Any]
    path: str
    build_seconds: float = 0.0
    stats: dict = field(default_factory=dict)

    @property
    def world_size(self) -> int:
        return len(self.world)

    def group(self, name: str):
        """ProcessGroup for ``name``; ``None`` when this rank is not a member or the group has one rank."""
        if name not in self.specs:
            raise KeyError(f"unknown group {name}")
        return self.groups.get(name)

    def ranks(self, name: str) -> tuple[int, ...]:
        return self.specs[name].ranks

    def is_member(self, name: str) -> bool:
        return self.rank in self.specs[name].ranks

    def rank_of(self, worker_id: str) -> int:
        return self.world.index(worker_id)


def _is_order_preserving_subset(new: list[str], old: tuple[str, ...]) -> bool:
    pos = {w: i for i, w in enumerate(old)}
    if any(w not in pos for w in new):
        return False
    idx = [pos[w] for w in new]
    return idx == sorted(idx)


class ResilientCommManager:
    def __init__(self, backend: str = "gloo", *, device: str | None = None, timeout_s: float = 60.0,
                 validate_timeout_s: float = 30.0, validate: bool = True, shrink: str = "auto",
                 subgroup_mode: str = "split", nccl_nonblocking: bool = True, poll_interval_s: float = 0.0005):
        if shrink not in ("auto", "never", "always"):
            raise ValueError(shrink)
        if subgroup_mode not in ("split", "local"):
            raise ValueError(subgroup_mode)
        self.backend = backend
        self.device = device
        self.timeout = datetime.timedelta(seconds=timeout_s)
        self.validate_timeout_s = validate_timeout_s
        self.validate = validate
        self.shrink = shrink
        self.subgroup_mode = subgroup_mode
        self.nccl_nonblocking = nccl_nonblocking
        self.poll_interval_s = poll_interval_s
        self.current: GenerationComms | None = None
        self._abort = threading.Event()
        self._lock = threading.Lock()
        self._live: list[Any] = []
        self._store_client = None
        self._store_addr: tuple[str, int] | None = None
        self._aborted_since_build = False

    # ------------------------------------------------------------------ capabilities
    @staticmethod
    def capabilities() -> dict:
        caps = {"shrink_group": hasattr(dist, "shrink_group"), "split_group": hasattr(dist, "split_group"),
                "nccl": dist.is_nccl_available(), "gloo": dist.is_gloo_available(), "nccl_version": None}
        if torch.cuda.is_available() and dist.is_nccl_available():
            try:
                caps["nccl_version"] = torch.cuda.nccl.version()
            except Exception:  # noqa: BLE001
                pass
        return caps

    def can_shrink(self, new_world: list[str]) -> bool:
        if self.shrink == "never" or self.current is None or self.backend != "nccl":
            return False
        caps = self.capabilities()
        nccl_ok = caps["nccl_version"] is not None and tuple(caps["nccl_version"]) >= (2, 27, 3)
        if not (caps["shrink_group"] and nccl_ok):
            return False
        return len(new_world) < self.current.world_size and _is_order_preserving_subset(new_world, self.current.world)

    # ------------------------------------------------------------------ building
    def store_for(self, store_addr: tuple[str, int], prefix: str):
        if self._store_client is None or self._store_addr != tuple(store_addr):
            self._store_client = dist.TCPStore(store_addr[0], int(store_addr[1]), is_master=False,
                                               timeout=self.timeout, wait_for_workers=False)
            self._store_addr = tuple(store_addr)
        return dist.PrefixStore(prefix, self._store_client)

    def build(self, generation: int, world: list[str], my_id: str, specs: list[GroupSpec], *,
              store=None, store_addr: tuple[str, int] | None = None, store_prefix: str | None = None) -> GenerationComms:
        """Create (or shrink to) the communicators of ``generation``; raises on abort/timeout/peer failure."""
        t0 = time.monotonic()
        self._abort.clear()
        self._aborted_since_build = False
        rank = world.index(my_id)
        for s in specs:
            if any(r >= len(world) for r in s.ranks):
                raise ValueError(f"group {s.name} references rank outside world of size {len(world)}")
        path = "shrink" if (self.shrink != "never" and self.can_shrink(world)) else "fresh"
        if self.shrink == "always" and path != "shrink":
            raise RuntimeError("shrink requested but not possible for this transition")
        if path == "shrink":
            self._shrink_world(world)
        else:
            if store is None:
                if store_addr is None or store_prefix is None:
                    raise ValueError("fresh initialization needs a store (or store_addr + store_prefix)")
                store = self.store_for(store_addr, store_prefix)
            self._teardown()
            self._init_world(rank, len(world), store)
        default = c10d._get_default_group()
        with self._lock:
            self._live = [default]
        groups = self._make_subgroups(rank, specs)
        if self.validate:
            self._validate(len(world), groups, specs)
        comms = GenerationComms(generation=generation, rank=rank, world=tuple(world),
                                specs={s.name: s for s in specs}, groups=groups, path=path,
                                build_seconds=time.monotonic() - t0)
        self.current = comms
        log.info("generation %d built via %s in %.3fs (%d groups)", generation, path, comms.build_seconds, len(groups))
        return comms

    def _init_world(self, rank: int, world_size: int, store) -> None:
        kwargs: dict = {"backend": self.backend, "store": store, "rank": rank, "world_size": world_size,
                        "timeout": self.timeout}
        if self.backend == "nccl":
            if self.device is not None:
                kwargs["device_id"] = torch.device(self.device)
            if self.nccl_nonblocking:
                opts = dist.ProcessGroupNCCL.Options()
                opts.config.blocking = 0
                kwargs["pg_options"] = opts
        dist.init_process_group(**kwargs)

    def _shrink_world(self, world: list[str]) -> None:
        """Survivors-only ``ncclCommShrink``; SHRINK_ABORT cancels whatever the dead ranks left in flight.

        Shrinking the default group invalidates every subgroup, matching the
        rebuild-the-whole-generation semantics.
        """
        assert self.current is not None
        keep = set(world)
        excluded = [i for i, w in enumerate(self.current.world) if w not in keep]
        dist.shrink_group(excluded, group=None, shrink_flags=c10d.SHRINK_ABORT)

    def _make_subgroups(self, rank: int, specs: list[GroupSpec]) -> dict[str, Any]:
        groups: dict[str, Any] = {}
        by_ranks: dict[tuple[int, ...], Any] = {}
        if self.subgroup_mode == "split":
            for layer in split_layers(specs):
                self._check_abort()
                split = [list(s.ranks) for s in layer]
                pg = dist.split_group(split_ranks=split, timeout=self.timeout,
                                      group_desc=f"resihp:{layer[0].kind}:{len(split)}")
                for s in layer:
                    if rank in s.ranks:
                        by_ranks[s.ranks] = pg
        else:
            # c10d salts hashed group names with a per-process creation counter, but with
            # use_local_synchronization=True a non-member returns before the counter is bumped
            # (torch 2.15). Overlapping groups then get rank-dependent names and members
            # deadlock. Every rank therefore calls new_group for every group in canonical order
            # and non-members re-synchronize the counter themselves.
            created: set[tuple[int, ...]] = set()
            for s in canonical_order(specs):
                if not s.materialized or s.ranks in created:
                    continue
                created.add(s.ranks)
                self._check_abort()
                before = c10d._world.group_count
                pg = dist.new_group(list(s.ranks), timeout=self.timeout, use_local_synchronization=True,
                                    group_desc=f"resihp:{s.name}")
                if c10d._world.group_count == before:
                    c10d._world.group_count = before + 1
                if rank in s.ranks:
                    by_ranks[s.ranks] = pg
        for s in specs:
            groups[s.name] = by_ranks.get(s.ranks) if rank in s.ranks and s.materialized else None
        with self._lock:
            self._live.extend(pg for pg in by_ranks.values() if pg not in self._live)
        return groups

    def _validate(self, world_size: int, groups: dict[str, Any], specs: list[GroupSpec]) -> None:
        dev = torch.device(self.device) if (self.backend == "nccl" and self.device) else torch.device("cpu")
        checked: set[int] = set()
        targets = [(c10d._get_default_group(), tuple(range(world_size)), "world")]
        targets += [(groups[s.name], s.ranks, s.name) for s in canonical_order(specs) if groups.get(s.name) is not None]
        for pg, ranks, name in targets:
            if id(pg) in checked:
                continue
            checked.add(id(pg))
            t = torch.ones(1, device=dev)
            try:
                work = dist.all_reduce(t, group=pg, async_op=True)
                self.guarded_wait(work, timeout=self.validate_timeout_s, what=f"validate {name}")
            except (CommTimeout, RuntimeError) as e:
                if isinstance(e, CommAborted):
                    raise
                raise GenerationBuildError(f"group {name} failed validation: {e}", suspects=tuple(ranks)) from e
            if int(t.item()) != pg.size():
                raise GenerationBuildError(f"group {name} validation sum {t.item()} != {pg.size()}", tuple(ranks))

    # ------------------------------------------------------------------ waiting / aborting
    def _check_abort(self) -> None:
        if self._abort.is_set():
            raise CommAborted("communicators aborted by the control plane")

    def guarded_wait(self, work, timeout: float | None = None, what: str = "collective") -> None:
        """Wait for ``work`` while honoring control-plane aborts.

        Gloo cannot interrupt an in-flight collective, so the CPU path polls; on NCCL
        ``work.wait()`` only orders streams and hangs are broken by ``abort()``.
        """
        if self.backend == "nccl":
            self._check_abort()
            work.wait()
            return
        if work.is_completed():
            work.wait()
            return
        # Gloo point-to-point works only report completion from inside wait(), so the
        # blocking wait runs on a helper thread while this thread stays abortable. An
        # abandoned helper unblocks when the old communicator times out or is destroyed.
        done = threading.Event()
        err: list[BaseException] = []

        def _wait() -> None:
            try:
                work.wait()
            except BaseException as e:  # noqa: BLE001 - re-raised on the training thread
                err.append(e)
            finally:
                done.set()

        threading.Thread(target=_wait, daemon=True, name="guarded-wait").start()
        deadline = None if timeout is None else time.monotonic() + timeout
        while not done.wait(self.poll_interval_s or 0.0005):
            if self._abort.is_set():
                raise CommAborted(f"{what} aborted by the control plane")
            if deadline is not None and time.monotonic() > deadline:
                raise CommTimeout(f"{what} timed out after {timeout}s")
        if err:
            raise err[0]

    def abort(self) -> None:
        """Thread-safe; called from the control-client thread on RECONFIG_NOTICE."""
        self._abort.set()
        self._aborted_since_build = True
        if self.backend != "nccl":
            return
        with self._lock:
            live = list(self._live)
        for pg in live:
            try:
                pg.abort()
            except Exception:  # noqa: BLE001 - best effort; the watchdog is the backstop
                pass

    @property
    def aborted(self) -> bool:
        return self._abort.is_set()

    def _teardown(self) -> None:
        if not dist.is_initialized():
            self.current = None
            return
        try:
            if self.backend == "nccl":
                c10d._abort_process_group(None)
            else:
                dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            log.exception("teardown of previous generation failed; continuing with a fresh world")
        with self._lock:
            self._live = []
        self.current = None

    def destroy(self) -> None:
        self._teardown()
