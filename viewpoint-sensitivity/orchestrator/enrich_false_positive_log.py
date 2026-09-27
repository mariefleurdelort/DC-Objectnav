#!/usr/bin/env python
"""One-time migration: adds iou_with_true_object/relabeled_true_object to an
existing false-positive inference log that predates those fields"""

import argparse
import json
from pathlib import Path

from iou import box_iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    log_path = Path(args.log)
    rows = [json.loads(l) for l in open(log_path)]
    manifest_rows = [json.loads(l) for l in open(args.manifest)]
    gt_by_view = {r["view_id"]: r.get("gt_bbox_xyxy") for r in manifest_rows}

    n_added = 0
    for row in rows:
        if "iou_with_true_object" in row:
            continue  # already enriched
        best_box = row.get("best_box_xyxy")
        gt_box = gt_by_view.get(row["view_id"])
        iou_true = box_iou(best_box, gt_box) if best_box else None
        row["iou_with_true_object"] = iou_true
        row["relabeled_true_object"] = bool(best_box and iou_true is not None and iou_true >= 0.5)
        n_added += 1

    backup_path = log_path.with_suffix(".jsonl.bak")
    log_path.rename(backup_path)
    with open(log_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"enriched {n_added}/{len(rows)} rows, original backed up to {backup_path}")


if __name__ == "__main__":
    main()
