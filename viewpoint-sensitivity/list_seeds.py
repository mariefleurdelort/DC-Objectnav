#!/usr/bin/env python
"""Readable dump of every seed viewpoint for every model's false-positive
offset-grid experiment: which object, which angle, what it was confused for,
how confident, and the path to the literal seed image (centered, no offset --
the exact frame where the model was fooled before any XY sweep)."""
import json

MODELS = ["owlv2", "yoloworld", "sam3", "groundingdino"]

for model in MODELS:
    path = f"data/renders_framing_falsepositive/{model}/best_viewpoints.json"
    try:
        entries = json.load(open(path))
    except FileNotFoundError:
        print(f"\n=== {model}: no best_viewpoints.json found ===")
        continue
    print(f"\n=== {model} ({len(entries)} seeds) ===")
    for r in sorted(entries, key=lambda r: (r["category"], r["model_id"])):
        print(f"  {r['category']}/{r['model_id']}: elev={r.get('elevation_deg')}, "
              f"azim={r.get('azimuth_deg')}, confused for '{r.get('false_category')}' "
              f"(score={r.get('best_score'):.3f}), image={r.get('rgb_path')}")
