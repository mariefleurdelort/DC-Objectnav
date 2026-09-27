#!/usr/bin/env python
"""Sanity-check that step 1's infrastructure is fully wired: all 4 model
servers reachable, and (optionally) the render manifest is present and
non-empty
"""
import sys
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PORTS_YAML = PROJECT_ROOT / "orchestrator" / "ports.yaml"
MANIFEST_PATH = PROJECT_ROOT / "data" / "renders" / "manifest.jsonl"


def check_servers():
    with open(PORTS_YAML) as f:
        servers = yaml.safe_load(f)["servers"]

    all_ok = True
    for name, spec in servers.items():
        url = f"http://localhost:{spec['port']}/health"
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            body = resp.json()
            print(f"  [ok]   {name:14s} {url}  -> {body}")
        except Exception as e:
            print(f"  [FAIL] {name:14s} {url}  -> {e}")
            all_ok = False
    return all_ok


def check_manifest():
    if not MANIFEST_PATH.exists():
        print(f"  [FAIL] manifest not found at {MANIFEST_PATH}")
        return False
    n_lines = sum(1 for _ in open(MANIFEST_PATH))
    if n_lines == 0:
        print(f"  [FAIL] manifest at {MANIFEST_PATH} is empty")
        return False
    print(f"  [ok]   manifest has {n_lines} rendered views: {MANIFEST_PATH}")
    return True


def main():
    print("== model servers ==")
    servers_ok = check_servers()
    print("\n== render manifest ==")
    manifest_ok = check_manifest()

    print()
    if servers_ok and manifest_ok:
        print("Step 1 infrastructure looks ready.")
        sys.exit(0)
    else:
        print("Step 1 infrastructure NOT ready — see failures above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
