"""Enumerate Objaverse (LVIS-annotated subset) models for rendering, as an
alternative to ShapeNet while ShapeNet's gated-license approval is pending.

Objaverse's LVIS category keys don't follow a clean, guessable naming scheme
(e.g. a compound category is one key like "sofa/couch/lounge", others are a
single word like "chair"), so rather than hardcoding guessed key strings we
fuzzy-match our own category -> target-word list (config.OBJAVERSE_CATEGORY_KEYWORDS)
against whatever keys objaverse.load_lvis_annotations() actually returns.

Unlike ShapeNet, the actual .glb bytes are NOT stored under this repo's data/
dir -- the `objaverse` pip package caches them under ~/.objaverse/hf-objaverse-v1/
(its own hardcoded location, shared across any project on this machine that
uses it). What we DO keep in data/objaverse/ is just category_uid_map.json,
a small manifest recording which UIDs we picked per category and where
objaverse.load_objects() said their glb ended up -- see download_objaverse.py.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import config as cfg

from habitat_object_loader import load_object_on_bare_stage, object_world_aabb, remove_object  # noqa: F401


@dataclass
class ObjaverseModel:
    model_id: str      # objaverse UID
    category: str       # normalized category name (matches config.CATEGORIES)
    lvis_key: str        # the raw LVIS annotation key it was matched from
    glb_path: Path


def normalize_lvis_key(key: str) -> List[str]:
    """'sofa/couch/lounge' -> ['sofa', 'couch', 'lounge']; underscores and
    hyphens within a synonym become spaces so multi-word synonyms compare
    cleanly against a plain target phrase."""
    return [syn.strip().lower().replace("_", " ").replace("-", " ") for syn in key.split("/")]


def find_matching_lvis_keys(lvis_annotations: Dict[str, list], target_words: List[str]) -> List[str]:
    """Returns every LVIS key that has at least one synonym exactly equal to
    one of `target_words` (also lowercased). Exact-token matching, not
    substring, so target "lamp" doesn't also pull in "table lamp"/"oil lamp".
    """
    targets = {w.strip().lower() for w in target_words}
    matches = []
    for key in lvis_annotations:
        if targets & set(normalize_lvis_key(key)):
            matches.append(key)
    return matches


def list_objaverse_models(max_per_category: int = None) -> List[ObjaverseModel]:
    """Reads the manifest written by scripts/download_objaverse.sh
    (data/objaverse/category_uid_map.json) and returns models whose glb file
    actually exists on disk (in the objaverse package's own cache dir).
    """
    max_per_category = max_per_category or cfg.MAX_MODELS_PER_CATEGORY
    map_path = cfg.OBJAVERSE_ROOT / "category_uid_map.json"
    if not map_path.exists():
        print(f"  [warn] no {map_path} -- run scripts/download_objaverse.sh first")
        return []

    with open(map_path) as f:
        category_map = json.load(f)

    models = []
    for category, entries in category_map.items():
        if category not in cfg.CATEGORIES:
            continue
        for entry in entries[:max_per_category]:
            glb_path = Path(entry["glb_path"])
            if not glb_path.exists():
                print(f"  [warn] missing glb for {category}/{entry['uid']} at {glb_path} -- skipping")
                continue
            models.append(ObjaverseModel(
                model_id=entry["uid"],
                category=category,
                lvis_key=entry["lvis_key"],
                glb_path=glb_path,
            ))
    return models
