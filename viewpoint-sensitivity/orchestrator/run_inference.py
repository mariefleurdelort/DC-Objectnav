#!/usr/bin/env python
"""Step 3: for each rendered view x each of the 4 model servers, run /detect
and log the result to a results manifest.

Two distinct, non-interchangeable notions of "success" live in this file --
see the top-level README section "What 'success' means" for the full
explanation. In short:
  - Normal mode (default): the view's real category is prompted, and success
    ("correct") means the model's top detection has IoU >= threshold against
    the manifest's real ground-truth box. This measures whether the model
    correctly recognizes what's actually there.
  - --false-positive mode: a *wrong* category (config.FALSE_CATEGORY_MAP) is
    prompted instead -- there is no ground-truth box for that category, so
    there is no "correct". Instead we log "false_positive": whether the model
    returned *any* detection at all for something that isn't in the image.
    Here, the model returning a detection is the bad outcome, the opposite
    polarity from "correct" -- which is exactly why this uses a different
    field name instead of overloading "correct" with an inverted meaning.
    We also log "iou_with_true_object"/"relabeled_true_object": the manifest
    still has the TRUE object's real ground-truth box, so a false-positive
    detection can be checked against it -- did the model land on the actual
    object and just call it the wrong name (genuine visual confusion, IoU
    high), or fabricate a box somewhere unrelated to anything really in the
    frame (a different, more concerning failure mode, IoU low)?

Run inside the vps-orchestrator env, after launch_servers.sh start:
    conda activate vps-orchestrator
    python orchestrator/run_inference.py
    python orchestrator/run_inference.py --false-positive --manifest data/renders/manifest.jsonl \\
        --output data/renders/false_positive_inference_log.jsonl

Resumable: re-running skips (view_id, model) pairs already present in the
output file, so an interrupted run (server crash, network blip) can just be
restarted.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests
import yaml

from iou import box_iou

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "rendering"))  # for config.py's FALSE_CATEGORY_MAP/CATEGORY_PROMPTS
import config as cfg  # noqa: E402
PORTS_YAML = PROJECT_ROOT / "orchestrator" / "ports.yaml"
MANIFEST_PATH = PROJECT_ROOT / "data" / "renders" / "manifest.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "renders" / "inference_log.jsonl"

# Fixed across every model and every view, per the sensitivity-analysis design
# (step 2): same prompt-per-category (already stamped into each manifest row),
# same decision threshold for every model.
DEFAULT_THRESHOLD = 0.3
# A detection counts as correct if its IoU against the manifest's ground-truth
# box is at least this. Standard object-detection convention (COCO-style AP@50).
IOU_MATCH_THRESHOLD = 0.5


def load_servers():
    with open(PORTS_YAML) as f:
        return yaml.safe_load(f)["servers"]


def load_manifest(manifest_path: Path):
    rows = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_done_keys(output_path: Path):
    done = set()
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((row["view_id"], row["model"]))
    return done


def call_detect(port: int, image_path: str, prompt: str, threshold: float, timeout: float):
    resp = requests.post(
        f"http://localhost:{port}/detect",
        json={"image_path": image_path, "prompt": prompt, "threshold": threshold},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--iou-match-threshold", type=float, default=IOU_MATCH_THRESHOLD)
    parser.add_argument("--models", nargs="+", default=None,
                         help="subset of model names to run (default: all in ports.yaml)")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N manifest rows (for a quick smoke test)")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--false-positive", action="store_true",
                         help="prompt with config.FALSE_CATEGORY_MAP's wrong category instead of the view's "
                              "real one, and log 'false_positive' (any detection at all) instead of 'correct' "
                              "(IoU against a real ground-truth box) -- there is no ground truth for a category "
                              "that isn't actually in the image.")
    args = parser.parse_args()

    servers = load_servers()
    model_names = args.models or list(servers.keys())
    for m in model_names:
        if m not in servers:
            raise SystemExit(f"Unknown model '{m}' -- must be one of {list(servers.keys())}")

    manifest_rows = load_manifest(Path(args.manifest))
    if args.limit:
        manifest_rows = manifest_rows[: args.limit]
    print(f"{len(manifest_rows)} views x {len(model_names)} models "
          f"= {len(manifest_rows) * len(model_names)} calls to make")

    output_path = Path(args.output)
    done = load_done_keys(output_path)
    if done:
        print(f"resuming: {len(done)} (view, model) pairs already logged, will be skipped")

    n_written = 0
    n_errors = 0
    with open(output_path, "a") as out_fh:
        for i, view in enumerate(manifest_rows):
            image_path = str(PROJECT_ROOT / view["rgb_path"])
            for model in model_names:
                key = (view["view_id"], model)
                if key in done:
                    continue

                port = servers[model]["port"]
                true_category = view["category"]
                if args.false_positive:
                    false_category = cfg.FALSE_CATEGORY_MAP[true_category]
                    prompt = cfg.CATEGORY_PROMPTS[false_category]
                else:
                    false_category = None
                    prompt = view["prompt"]

                row = {
                    "view_id": view["view_id"],
                    "model": model,
                    "category": true_category,
                    "model_id": view.get("model_id"),
                    "prompt": prompt,
                    "threshold": args.threshold,
                }
                if args.false_positive:
                    row["true_category"] = true_category
                    row["false_category"] = false_category
                # Carry through whichever viewpoint-describing fields this manifest
                # has -- elevation_deg/azimuth_deg for the main sphere sweep,
                # offset_x_deg/offset_y_deg/base_elevation_deg/base_azimuth_deg for
                # the offset-grid sweep. Manifest-agnostic on purpose.
                for key_name in ("elevation_deg", "azimuth_deg", "offset_x_deg", "offset_y_deg",
                                  "base_elevation_deg", "base_azimuth_deg"):
                    if key_name in view:
                        row[key_name] = view[key_name]
                t0 = time.perf_counter()
                try:
                    result = call_detect(port, image_path, prompt, args.threshold, args.timeout)
                    detections = result["detections"]
                    best = max(detections, key=lambda d: d["score"], default=None)

                    if args.false_positive:
                        # No ground truth exists for the FALSE category (it isn't
                        # actually present) -- any detection at all IS the false
                        # positive. But the manifest still has the TRUE object's
                        # real gt_bbox_xyxy, so we can check whether the false-
                        # positive box actually landed on the real object (genuine
                        # mislabeling) vs. somewhere else entirely (a different,
                        # more concerning failure mode) -- see README.
                        iou_true = box_iou(best["box_xyxy"], view.get("gt_bbox_xyxy")) if best else None
                        row.update({
                            "num_detections": len(detections),
                            "best_score": best["score"] if best else None,
                            "best_box_xyxy": best["box_xyxy"] if best else None,
                            "false_positive": bool(best),
                            "iou_with_true_object": iou_true,
                            "relabeled_true_object": bool(best and iou_true >= args.iou_match_threshold),
                            "latency_ms": result.get("latency_ms"),
                            "error": None,
                        })
                    else:
                        iou = box_iou(best["box_xyxy"], view.get("gt_bbox_xyxy")) if best else 0.0
                        row.update({
                            "num_detections": len(detections),
                            "best_score": best["score"] if best else None,
                            "best_box_xyxy": best["box_xyxy"] if best else None,
                            "iou_with_gt": iou,
                            "correct": bool(best and iou >= args.iou_match_threshold),
                            "latency_ms": result.get("latency_ms"),
                            "error": None,
                        })
                except Exception as e:
                    n_errors += 1
                    row.update({
                        "num_detections": None,
                        "best_score": None,
                        "best_box_xyxy": None,
                        "latency_ms": (time.perf_counter() - t0) * 1000,
                        "error": str(e),
                    })
                    if args.false_positive:
                        row["false_positive"] = False
                        row["iou_with_true_object"] = None
                        row["relabeled_true_object"] = False
                    else:
                        row["iou_with_gt"] = 0.0
                        row["correct"] = False

                out_fh.write(json.dumps(row) + "\n")
                out_fh.flush()
                n_written += 1

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(manifest_rows)} views processed "
                      f"({n_written} results written, {n_errors} errors)")

    print(f"\ndone: {n_written} results written to {output_path} ({n_errors} errors)")


if __name__ == "__main__":
    main()
