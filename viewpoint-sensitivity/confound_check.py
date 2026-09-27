#!/usr/bin/env python
"""Checks whether the false-positive offset-grid's corner falloff is a genuine
recognition effect or just the object physically leaving the frame (HFOV=90deg,
offsets go to +-40deg -- close to the 45deg half-FOV edge, so this is a real
possibility, not a hypothetical).

For each model, buckets views by radial distance from the seed (0,0) and
compares mean gt_pixel_count (how much of the object is actually visible)
against mean false_positive_rate and mean confidence in the same bucket.
If pixel count collapses together with false_positive_rate at the corners,
the corner falloff is confounded by frame-exit. If false_positive_rate drops
while pixel count stays high, that's a genuine viewpoint/recognition effect.
"""
import json
import sys

import numpy as np
import pandas as pd

IMAGE_PIXELS = 1024 * 1024


def check_model(model: str):
    base = f"data/renders_framing_falsepositive/{model}"
    man = pd.DataFrame([json.loads(l) for l in open(f"{base}/framing_manifest.jsonl")])
    inf = pd.DataFrame([json.loads(l) for l in open(f"{base}/framing_inference_log.jsonl")])
    inf = inf[inf["model"] == model]

    df = inf.merge(man[["view_id", "gt_pixel_count"]], on="view_id", how="left")
    df["radius_deg"] = np.sqrt(df["offset_x_deg"] ** 2 + df["offset_y_deg"] ** 2)
    df["pct_visible"] = df["gt_pixel_count"] / IMAGE_PIXELS * 100

    bins = [0, 15, 30, 60]
    labels = ["near-center (<15deg)", "mid (15-30deg)", "far/corner (>30deg)"]
    df["bucket"] = pd.cut(df["radius_deg"], bins=bins, labels=labels)

    summary = df.groupby("bucket", observed=True).agg(
        mean_pct_object_visible=("pct_visible", "mean"),
        frac_nearly_out_of_frame=("gt_pixel_count", lambda s: (s < 500).mean()),
        mean_false_positive_rate=("false_positive", "mean"),
        mean_confidence=("best_score", "mean"),
        n=("view_id", "count"),
    ).reset_index()

    print(f"\n=== {model} ===")
    print(summary.to_string(index=False))

    corr = df[["radius_deg", "gt_pixel_count", "false_positive", "best_score"]].corr()["radius_deg"]
    print("correlation with radius_deg:")
    print(corr.drop("radius_deg").to_string())


if __name__ == "__main__":
    models = sys.argv[1:] or ["owlv2", "yoloworld", "sam3", "groundingdino"]
    for m in models:
        check_model(m)
