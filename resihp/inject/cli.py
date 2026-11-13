"""``resihp-inject``: validate fault specs and drive host-level fault emulation."""

from __future__ import annotations

import argparse
import json
import shlex

from resihp.inject.faults import ExternalInjector, parse_faults, run_schedule


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="resihp-inject")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_parse = sub.add_parser("parse", help="validate and normalize a fault spec")
    p_parse.add_argument("spec")
    p_exec = sub.add_parser("exec", help="apply every fault in the spec now")
    p_exec.add_argument("spec")
    p_exec.add_argument("--dry-run", action="store_true")
    p_exec.add_argument("--revert", action="store_true")
    p_sched = sub.add_parser("schedule", help="run time-triggered faults (time=... in the spec)")
    p_sched.add_argument("spec")
    p_sched.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    faults = parse_faults(args.spec)
    if args.cmd == "parse":
        print(json.dumps([f.to_spec() for f in faults], indent=2))
        return 0
    inj = ExternalInjector(dry_run=args.dry_run)
    if args.cmd == "exec":
        for f in faults:
            (inj.revert if args.revert else inj.apply)(f)
    else:
        run_schedule(faults, inj)
    for argv_ in inj.executed:
        print(shlex.join(argv_))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
