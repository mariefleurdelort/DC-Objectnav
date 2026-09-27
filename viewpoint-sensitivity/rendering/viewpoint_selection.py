"""Viewpoint selection: deciding WHICH (eye, rotation, orbit_radius) to use as
the fixed base camera pose for an offset-grid sweep. Kept separate from
rendering/camera_offset.py (which only knows how to apply an xy offset to a
*given* viewpoint) so new selection strategies can be added without touching
the offset/rendering code at all.

Every selector returns a list of dicts shaped like a manifest.jsonl row --
at minimum: category, model_id, camera_position, object_center, orbit_radius_m,
plus whatever provenance fields explain *why* that viewpoint was chosen.
"""
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
    (elevation, azimuth) per instance. These are the model's own genuine
    failure points -- "it already found the alleged [wrong] object here" --
    used to seed an offset-grid sweep testing whether shifting/tilting away
    from that exact failure corrects the model (does false_positive go away,
    does correct on the real category come back) or makes it worse.

    Instances this model never false-positived on are simply absent from the
    result -- there's nothing to investigate for them. Different models will
    generally get different seeds for "the same" instance, since they fail
    (or don't) at different viewpoints -- that's why the offset-grid render
    for --strategy false is per-model, not shared across models the way
    --strategy true's is.

    `false_prompt_inference_log` is the path to the log produced by
    orchestrator/run_inference.py --false-positive over the main manifest.
    """
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
