"""Coordinator-side fail-slow pipeline.

timing reports (one per TP group per iteration: forward-chunk seconds + workload
features) -> per-layer, TP-normalized samples -> shared gated RLS model ->
per-group residual BOCD -> normalized throughput p for the group's devices ->
ski-rental decision -> health update + replan request. The planner reads p when it
builds the next HeteroParallelPlan, so mitigation (TP reshaping, repartition,
migration, exclusion) reuses exactly the fail-stop machinery.

Once a group is confirmed slow, every subsequent report from it pays "rent" to its
policy until the accumulated excess covers the reconfiguration cost. With a
``what_if`` pricer the excess is (current plan - replanned plan) and the cost is the
estimated state-migration + rebuild time, so a slowdown the planner cannot mitigate
never triggers a reconfiguration.
"""

from __future__ import annotations

import queue
import threading

from resihp.detect.failslow import FailSlowDetector
from resihp.detect.policy import SkiRentalPolicy
from resihp.detect.predictor import RLSPredictor
from resihp.utils.log import get_logger

log = get_logger(__name__)


class DetectorService:
    def __init__(self, coordinator, *, warmup: int = 6, threshold: float = 0.9, min_increase: float = 0.25,
                 reconfig_cost_s: float = 0.0, min_gain: float = 0.05, min_dwell: int = 0,
                 token_scale: float = 1e3, what_if=None):
        self.coord = coordinator
        self.detector_kw = dict(mode="residual", warmup=warmup, threshold=threshold, min_increase=min_increase,
                                cooldown=warmup)
        self.rls = RLSPredictor(scales=(token_scale, token_scale ** 2, 1.0), gate=0.3, warmup=32)
        self.detectors: dict[tuple, FailSlowDetector] = {}
        self.policy_kw = dict(min_gain=min_gain, min_dwell=min_dwell)
        self.policies: dict[tuple, SkiRentalPolicy] = {}
        self.reconfig_cost_s = reconfig_cost_s
        self.what_if = what_if
        self.speeds: dict[str, float] = {}
        self.events: list[dict] = []
        self._suspects: dict[tuple, dict] = {}
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="detector-service")
        self._thread.start()

    def handle(self, worker_id: str, report: dict) -> None:
        """Called from the coordinator's message path; must not block."""
        if report.get("kind") == "timing":
            self._q.put((worker_id, report))

    def stop(self) -> None:
        self._stop.set()
        self._q.put(None)

    def _loop(self) -> None:
        while not self._stop.is_set():
            item = self._q.get()
            if item is None:
                return
            try:
                self._process(*item)
            except Exception:  # noqa: BLE001 - detection must never take the coordinator down
                log.exception("timing report processing failed")

    def _process(self, worker_id: str, rep: dict) -> None:
        devices = tuple(rep["devices"])
        layers = max(1, rep["layers"])
        # One rank of a TP x CP group does 1 / (tp * cp) of each layer's work on the micro-batch.
        share = max(1, rep["tp"]) * max(1, rep.get("cp", 1))
        norm = [(n, s, sec * share / layers) for n, s, sec in rep["samples"]]
        if not norm:
            return
        observed = sum(sec for _, _, sec in norm)
        predicted = None
        if self.rls.n >= self.rls.warmup:
            predicted = sum(max(self.rls.predict(n, s), 1e-9) for n, s, _ in norm)
        gen = rep.get("gen")
        det = self.detectors.setdefault((gen, devices), FailSlowDetector(**self.detector_kw))
        hit = det.observe(rep["iteration"], observed, predicted)
        for n, s, sec in norm:
            self.rls.update(n, s, sec)
        if hit is not None and predicted is not None:
            self._suspects[devices] = {"gen": gen, "ratio": hit.magnitude, "since": rep["iteration"], "price": None}
            log.warning("fail-slow on %s: %.2fx slower", devices, hit.magnitude)
        sus = self._suspects.get(devices)
        if sus is None or sus["gen"] != gen:
            return
        current, best, cost = self._price(devices, sus, observed)
        policy = self.policies.setdefault(devices, SkiRentalPolicy(**self.policy_kw))
        decision = policy.observe(rep["iteration"], current, best, cost)
        self.events.append({"iteration": rep["iteration"], "devices": list(devices), "ratio": sus["ratio"],
                            "act": decision.act, "reason": decision.reason, "current_s": current, "best_s": best,
                            "reconfig_s": cost})
        if decision.act:
            del self._suspects[devices]
            for d in devices:
                self.speeds[d] = 1.0 / sus["ratio"]
                self.coord.set_health(d, p=1.0 / sus["ratio"])
            self.coord.request_replan(f"failslow:{','.join(devices)}")

    def _price(self, devices: tuple, sus: dict, observed: float) -> tuple[float, float, float]:
        if self.what_if is None:
            return observed, observed / sus["ratio"], self.reconfig_cost_s
        if sus["price"] is None:
            speeds = dict(self.speeds)
            speeds.update({d: 1.0 / sus["ratio"] for d in devices})
            w = self.what_if(self.coord.payload, speeds)
            sus["price"] = (w.current_s, w.mitigated_s, w.reconfig_s)
        return sus["price"]
