"""Thin wrapper kept for existing scripts; prefer ``resihp-bench detection``."""

from resihp.bench.runner import main

if __name__ == "__main__":
    import sys

    main(["detection", *sys.argv[1:]])
