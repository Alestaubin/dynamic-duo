#!/usr/bin/env python3
"""Clone every repo in third_party/repos.yaml at its pinned commit.

    python third_party/fetch.py            # fetch what is missing, verify the rest
    python third_party/fetch.py CCC cotta  # only these
    python third_party/fetch.py --update   # print the upstream HEAD of each repo (does not change the pin)

An existing checkout is left alone; it is only verified to be at the pinned commit.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def fetch(name: str, spec: dict) -> str:
    dest = HERE / name
    if dest.exists():
        head = _git("rev-parse", "HEAD", cwd=dest)
        return "ok" if head == spec["commit"] else f"WRONG COMMIT (at {head[:7]}, pinned {spec['commit'][:7]})"
    sparse = spec.get("sparse")
    clone = ["clone", "-q", "--no-checkout", "--filter=blob:none", spec["url"], str(dest)]
    _git(*clone)
    if sparse:
        _git("sparse-checkout", "set", *sparse, cwd=dest)
    # Fetching a commit by sha works on GitHub and avoids cloning full history.
    _git("fetch", "-q", "--depth", "1", "origin", spec["commit"], cwd=dest)
    _git("checkout", "-q", "FETCH_HEAD", cwd=dest)
    return "cloned"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="repos to fetch (default: all)")
    ap.add_argument("--update", action="store_true", help="show upstream HEAD for each repo")
    args = ap.parse_args()

    repos = yaml.safe_load((HERE / "repos.yaml").read_text())["repos"]
    unknown = [n for n in args.names if n not in repos]
    if unknown:
        print(f"unknown repo(s): {unknown}; known: {sorted(repos)}", file=sys.stderr)
        return 2

    bad = 0
    for name in args.names or sorted(repos):
        spec = repos[name]
        if args.update:
            head = _git("ls-remote", spec["url"], "HEAD").split()[0]
            print(f"{name:24s} pinned {spec['commit'][:7]}  upstream {head[:7]}  {'(same)' if head == spec['commit'] else '(newer)'}")
            continue
        status = fetch(name, spec)
        bad += status not in ("ok", "cloned")
        print(f"{name:24s} {spec['commit'][:7]}  {status}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
