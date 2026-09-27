#!/usr/bin/env python
"""Pools ALL categories/objects together per model (main sphere sweep,
data/renders/false_positive_inference_log.jsonl) to find whether there's a
consistently bad elevation/azimuth region"""

import json

import pandas as pd

df = pd.DataFrame([json.loads(l) for l in open("data/renders/false_positive_inference_log.jsonl")])

for model, df_m in df.groupby("model"):
    print(f"\n{'='*70}\n{model}  (n={len(df_m)} views, overall fp rate={df_m['false_positive'].mean():.3f})\n{'='*70}")

    print("\n-- by ELEVATION only (pooled across all azimuths/objects/categories -- physically absolute, trustworthy) --")
    elev = df_m.groupby("elevation_deg")["false_positive"].agg(["mean", "count"]).sort_values("mean")
    print(elev.to_string())

    print("\n-- by AZIMUTH only (pooled across all elevations/objects/categories -- CAVEAT: azimuth=0 is each")
    print("   object's own mesh-authored 'front', not world-absolute -- treat with more skepticism) --")
    azim = df_m.groupby("azimuth_deg")["false_positive"].agg(["mean", "count"]).sort_values("mean")
    print(azim.to_string())

    print("\n-- top 5 WORST (elevation, azimuth) cells --")
    cell = df_m.groupby(["elevation_deg", "azimuth_deg"])["false_positive"].agg(["mean", "count"])
    cell = cell[cell["count"] >= 3]  # drop tiny/noisy cells
    print(cell.sort_values("mean", ascending=False).head(5).to_string())

    print("\n-- top 5 SAFEST (elevation, azimuth) cells --")
    print(cell.sort_values("mean", ascending=True).head(5).to_string())
