"""Viewpoint selection: deciding WHICH (eye, rotation, orbit_radius) to use as
the fixed base camera pose for an offset-grid sweep."""

import json

import config as cfg


def select_best_true_viewpoints(limit: int = None):
    """For each (category, model_id) instance, the (elevation, azimuth) row
    whose view_id has the highest mean TRUE-prompt confidence across the 4
    models in inference_log.jsonl. This is "where does the model most
    confidently recognize the real object" -- the seed viewpoint for the
    true-positive offset-grid experiment.
    """
    import pandas as pd

    inf = pd.DataFrame([json.loads(l) for l in open(cfg.RENDERS_DIR / "inference_log.jsonl")])
    man = pd.DataFrame([json.loads(l) for l in open(cfg.MANIFEST_PATH)])

    mean_score = inf.groupby("view_id")["best_score"].mean().reset_index()
    merged = man.merge(mean_score, on="view_id", how="inner")
    merged["instance_key"] = merged["category"] + "/" + merged["model_id"]

    idx = merged.groupby("instance_key")["best_score"].idxmax()
    best_rows = merged.loc[idx].sort_values("instance_key")

    records = best_rows.to_dict("records")
    for r in records:
        r["selection_strategy"] = "best_true_viewpoint"
    if limit:
        records = records[:limit]
    return records


def select_false_positive_seed_viewpoints(false_prompt_inference_log, model: str, limit: int = None):
    """For ONE model, among views where THAT model's own inference actually
    returned false_positive=True (a real, confirmed miscategorization -- not
    an averaged confidence across models), pick the highest-confidence
    (elevation, azimuth) per instance."""
    import pandas as pd

    inf = pd.DataFrame([json.loads(l) for l in open(false_prompt_inference_log)])
    inf = inf[(inf["model"] == model) & (inf["false_positive"] == True)]
    inf = inf[["view_id", "best_score", "false_category"]]

    man = pd.DataFrame([json.loads(l) for l in open(cfg.MANIFEST_PATH)])
    merged = man.merge(inf, on="view_id", how="inner")
    merged["instance_key"] = merged["category"] + "/" + merged["model_id"]

    idx = merged.groupby("instance_key")["best_score"].idxmax()
    best_rows = merged.loc[idx].sort_values("instance_key")

    records = best_rows.to_dict("records")
    for r in records:
        r["selection_strategy"] = "false_positive_seed"
        r["seed_model"] = model
    if limit:
        records = records[:limit]
    return records
