"""ResiBench: ``resihp-bench <experiment>`` writes JSON + Markdown under ``--out``.

    resihp-bench detection   fail-slow detection accuracy (synthetic packed-data traces)
    resihp-bench planner     planner ablation (TP shrink / repartition / migration / replica drop)
    resihp-bench calibrate   fit the cluster simulator's ResiHP model to the real planner
    resihp-bench sim         trace-driven ETTR at 1K-64K GPUs vs torchft, NVRx, ReCycle, Oobleck, Greyhound
    resihp-bench chaos       real-process chaos on CPU: MTTR breakdown + exactness check
    resihp-bench all         everything above
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _write(out: Path, name: str, data, md: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(data, indent=2, default=str))
    (out / f"{name}.md").write_text(md + "\n")
    print(md)


def cmd_detection(args) -> None:
    from resihp.bench import detection

    res = detection.run(args.seeds, args.iterations, args.slow_at, [float(x) for x in args.factors.split(",")],
                        args.horizon)
    _write(Path(args.out) / "detection", "detection", res, detection.to_markdown(res))


def cmd_planner(args) -> None:
    from resihp.bench import planner_ablation

    rows = planner_ablation.run(tuple(int(x) for x in args.dps.split(",")))
    _write(Path(args.out) / "planner", "ablation", rows, planner_ablation.to_markdown(rows))


def _calibration(out: Path) -> dict:
    from resihp.sim.calibrate import calibrate

    path = out / "sim" / "calibration.json"
    if path.exists():
        return json.loads(path.read_text())
    res = calibrate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2))
    return res


def cmd_calibrate(args) -> None:
    from resihp.sim.calibrate import calibrate

    res = calibrate()
    md = (f"beta = {res['beta']}, overhead = {res['overhead']}, rmse = {res['rmse']:.4f}\n\n"
          "| DP | scenario | planner | model |\n|---|---|---|---|\n" +
          "\n".join(f"| {dp} | {k} | {v:.3f} | {res['model'][dp][k]:.3f} |"
                    for dp, m in res["measured"].items() for k, v in m.items()))
    _write(Path(args.out) / "sim", "calibration", res, md)


def cmd_sim(args) -> None:
    from resihp.sim import experiments as ex

    out = Path(args.out)
    calib = _calibration(out)
    scales = tuple(int(x) for x in args.scales.split(","))
    report, data = [], {"calibration": calib}
    for spare in (float(x) for x in args.spares.split(",")):
        for label, trace_kw in (("fail-stop only", {"slow_per_gpu_day": 0.0}), ("fail-stop + fail-slow", {})):
            cells = [ex.run_cell(g, args.days, args.seeds, calib, trace_kw=trace_kw, spare_frac=spare) for g in scales]
            key = f"{label}, spares={spare:.0%}"
            data[key] = cells
            report += [f"\n### ETTR, {key} ({args.days:.0f} days, {args.seeds} seeds)\n", ex.scale_table(cells)]
            mid = next((c for c in cells if c["gpus"] == 16384), cells[-1])
            report += [f"\n#### Breakdown at {mid['gpus']:,} GPUs\n", ex.breakdown_table(mid)]
    if args.sensitivity:
        sens = ex.sensitivity(calib, days=args.days, seeds=max(1, args.seeds // 2))
        data["sensitivity"] = sens
        report += ["\n### Sensitivity at 16,384 GPUs (fail-stop + fail-slow, no spares)\n", ex.sensitivity_table(sens)]
    _write(out / "sim", "cluster", data, "\n".join(report))


def cmd_chaos(args) -> None:
    from resihp.bench import chaos

    res = chaos.run(args.trials, tuple(args.faults.split(",")))
    _write(Path(args.out) / "chaos", "chaos", res, chaos.to_markdown(res))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="resihp-bench", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="results")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detection")
    d.add_argument("--seeds", type=int, default=20)
    d.add_argument("--iterations", type=int, default=120)
    d.add_argument("--slow-at", type=int, default=70)
    d.add_argument("--factors", default="1.3,1.6,2.0")
    d.add_argument("--horizon", type=int, default=10)
    p = sub.add_parser("planner")
    p.add_argument("--dps", default="2,4")
    sub.add_parser("calibrate")
    s = sub.add_parser("sim")
    s.add_argument("--scales", default="1024,4096,16384,32768,65536")
    s.add_argument("--days", type=float, default=30.0)
    s.add_argument("--seeds", type=int, default=5)
    s.add_argument("--spares", default="0,0.02")
    s.add_argument("--sensitivity", action="store_true")
    c = sub.add_parser("chaos")
    c.add_argument("--trials", type=int, default=3)
    c.add_argument("--faults", default="kill,hang,slow")
    a = sub.add_parser("all")
    a.add_argument("--quick", action="store_true", help="fewer seeds/trials")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    t0 = time.monotonic()
    if args.cmd == "all":
        q = args.quick
        out = args.out
        for argv2 in (["detection", "--seeds", "10" if q else "20"], ["planner"], ["calibrate"],
                      ["sim", "--seeds", "2" if q else "5"] + ([] if q else ["--sensitivity"]),
                      ["chaos", "--trials", "1" if q else "3"]):
            main(["--out", out, *argv2])
    else:
        {"detection": cmd_detection, "planner": cmd_planner, "calibrate": cmd_calibrate, "sim": cmd_sim,
         "chaos": cmd_chaos}[args.cmd](args)
    print(f"[resihp-bench {args.cmd}] {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
