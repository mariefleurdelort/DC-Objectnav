#!/usr/bin/env python
"""One-off smoke test against a single running model server — useful for
validating each server works before wiring up the full step 3 inference loop.

    conda activate vps-orchestrator
    python orchestrator/detect_client.py --model yoloworld \\
        --image data/renders/replica_cad/apt_0/chair/12/elev+00_az000_rgb.png \\
        --prompt "a chair" --threshold 0.3
"""
import argparse
import json
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PORTS_YAML = PROJECT_ROOT / "orchestrator" / "ports.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["yoloworld", "groundingdino", "owlv2", "sam3"])
    parser.add_argument("--image", required=True, help="path to an image (absolute, or relative to project root)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    with open(PORTS_YAML) as f:
        servers = yaml.safe_load(f)["servers"]
    port = servers[args.model]["port"]

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    resp = requests.post(
        f"http://localhost:{port}/detect",
        json={"image_path": str(image_path), "prompt": args.prompt, "threshold": args.threshold},
        timeout=60,
    )
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    main()
