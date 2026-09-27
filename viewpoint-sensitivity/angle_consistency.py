#!/usr/bin/env python
"""For each (model, category), how angle-locked is the false-positive seed set?
Reads directly from data/renders_framing_falsepositive/<model>/best_viewpoints.json
-- the same file the earlier bookshelf/cabinet observation (OWLv2: all 4 bookshelf
seeds at elev=30,azim=90) came from -- and generalizes it across every category
and model instead of eyeballing one example.

n_distinct_angles == n_instances means every object failed at a different angle
(scattered, no shared viewpoint). n_distinct_angles == 1 means every single
instance of that category failed at the exact same angle (fully angle-locked).
mode_frac is the fraction of instances sharing the single most common angle --
the single clearest number for "is there usually a common failing viewpoint
for this category."
"""
import json
from collections import Counter

MODELS = ["owlv2", "yoloworld", "sam3", "groundingdino"]

rows = []
for model in MODELS:
    path = f"data/renders_framing_falsepositive/{model}/best_viewpoints.json"
    try:
        entries = json.load(open(path))
    except FileNotFoundError:
        continue
    by_cat = {}
    for r in entries:
        by_cat.setdefault(r["category"], []).append((r.get("elevation_deg"), r.get("azimuth_deg")))

    for cat, angles in sorted(by_cat.items()):
        n = len(angles)
        counts = Counter(angles)
        n_distinct = len(counts)
        mode_angle, mode_count = counts.most_common(1)[0]
        rows.append({
            "model": model, "category": cat, "n_instances": n,
            "n_distinct_angles": n_distinct, "mode_angle": mode_angle,
            "mode_frac": mode_count / n,
        })

rows.sort(key=lambda r: -r["mode_frac"])
print(f"{'model':<14}{'category':<12}{'n':>4}{'distinct':>10}{'mode_angle':>16}{'mode_frac':>11}")
for r in rows:
    print(f"{r['model']:<14}{r['category']:<12}{r['n_instances']:>4}{r['n_distinct_angles']:>10}"
          f"{str(r['mode_angle']):>16}{r['mode_frac']:>11.2f}")
