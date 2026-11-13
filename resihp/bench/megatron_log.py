"""Parse Megatron-LM training logs into per-iteration records (baseline reproduction)."""

from __future__ import annotations

import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass

_ITER = re.compile(r"iteration\s+(\d+)/\s*(\d+)")
_FIELD = re.compile(r"\|\s*([^|:]+?):\s*([-+0-9.eE]+)")


@dataclass
class IterRecord:
    iteration: int
    iter_time_ms: float | None
    tflops_per_gpu: float | None
    lm_loss: float | None
    global_batch_size: int | None


def parse_line(line: str) -> IterRecord | None:
    m = _ITER.search(line)
    if not m or "elapsed time per iteration" not in line:
        return None
    fields = {k.strip(): v for k, v in _FIELD.findall(line)}

    def num(key: str) -> float | None:
        return float(fields[key]) if key in fields else None

    gbs = num("global batch size")
    return IterRecord(
        iteration=int(m.group(1)),
        iter_time_ms=num("elapsed time per iteration (ms)"),
        tflops_per_gpu=num("throughput per GPU (TFLOP/s/GPU)"),
        lm_loss=num("lm loss"),
        global_batch_size=int(gbs) if gbs is not None else None,
    )


def parse_log(text: str) -> list[IterRecord]:
    return [r for r in (parse_line(l) for l in text.splitlines()) if r is not None]


def summarize(records: list[IterRecord], skip_first: int = 2) -> dict:
    steady = [r for r in records[skip_first:] if r.iter_time_ms is not None] or records
    times = [r.iter_time_ms for r in steady if r.iter_time_ms is not None]
    tflops = [r.tflops_per_gpu for r in steady if r.tflops_per_gpu is not None]
    losses = [(r.iteration, r.lm_loss) for r in records if r.lm_loss is not None]
    return {
        "iterations": len(records),
        "median_iter_time_ms": statistics.median(times) if times else None,
        "median_tflops_per_gpu": statistics.median(tflops) if tflops else None,
        "final_loss": losses[-1][1] if losses else None,
        "loss_curve": losses,
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    text = open(argv[0]).read() if argv else sys.stdin.read()
    recs = parse_log(text)
    print(json.dumps({"summary": summarize(recs), "records": [asdict(r) for r in recs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
