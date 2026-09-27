"""Enumerate downloaded ShapeNet models for rendering.

Per https://arxiv.org/abs/2311.17938 ("Active Open-Vocabulary Recognition"),
the controlled multi-view dataset is built by orbiting a camera around
individual CAD models on a bare habitat-sim stage, not by mining object
instances out of furnished rooms. Actual habitat-sim loading is
dataset-agnostic -- see habitat_object_loader.py.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

import config as cfg

# Re-exported so existing callers (render_multiview.py) can import everything
# from one place regardless of which dataset backend is active.
from habitat_object_loader import load_object_on_bare_stage, object_world_aabb, remove_object  # noqa: F401


@dataclass
class ShapeNetModel:
    model_id: str      # ShapeNet model hash, e.g. "1a6f615e8b1b5ae4dbbc9440457e303e"
    category: str       # normalized category name (matches config.CATEGORIES)
    synset_id: str       # ShapeNetCore synset id, e.g. "03001627"
    glb_path: Path


def list_shapenet_models(max_per_category: int = None) -> List[ShapeNetModel]:
    """Scan cfg.SHAPENET_ROOT/<synset_id>/*.glb for each configured category.

    Expects the layout produced by scripts/download_shapenet.sh:
        data/shapenet/<synset_id>/<model_id>.glb
    """
    max_per_category = max_per_category or cfg.MAX_MODELS_PER_CATEGORY
    models = []
    for category, synset_id in cfg.SHAPENET_SYNSETS.items():
        synset_dir = cfg.SHAPENET_ROOT / synset_id
        if not synset_dir.exists():
            print(f"  [warn] no directory for category={category} synset={synset_id} "
                  f"at {synset_dir} -- did scripts/download_shapenet.sh run for it?")
            continue
        glbs = sorted(synset_dir.glob("*.glb"))[:max_per_category]
        for glb_path in glbs:
            models.append(ShapeNetModel(
                model_id=glb_path.stem,
                category=category,
                synset_id=synset_id,
                glb_path=glb_path,
            ))
    return models
