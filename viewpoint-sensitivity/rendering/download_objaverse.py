#!/usr/bin/env python
"""Pick Objaverse (LVIS-annotated) UIDs for each of our target categories and
download them, writing data/objaverse/category_uid_map.json for
objaverse_objects.list_objaverse_models() to read at render time.

Run inside vps-render (or anywhere with `pip install objaverse` -- it needs no
habitat-sim, only network access):
    conda activate vps-render
    python rendering/download_objaverse.py --dry-run   # show matched LVIS keys + counts, no download
    python rendering/download_objaverse.py              # actually download

Objaverse needs no HF login/license -- unlike ShapeNet's gated dataset, this
works right now. The actual .glb bytes land in objaverse's own cache dir
(~/.objaverse/hf-objaverse-v1/glbs/...), not under this repo's data/.
"""
import argparse
import json
import multiprocessing

import config as cfg
from objaverse_objects import find_matching_lvis_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="show matched LVIS keys and UID counts per category, download nothing")
    parser.add_argument("--max-per-category", type=int, default=cfg.MAX_MODELS_PER_CATEGORY)
    args = parser.parse_args()

    import objaverse  # deferred: heavy import, and dry-run shouldn't require the pip package installed yet
    print(f"objaverse package version: {getattr(objaverse, '__version__', 'unknown')}")

    print("loading LVIS annotations (downloads the annotation json once, cached after)...")
    lvis_annotations = objaverse.load_lvis_annotations()
    print(f"{len(lvis_annotations)} LVIS categories available\n")

    category_to_uids = {}
    for category, target_words in cfg.OBJAVERSE_CATEGORY_KEYWORDS.items():
        matched_keys = find_matching_lvis_keys(lvis_annotations, target_words)
        if not matched_keys:
            print(f"  [warn] category={category}: no LVIS key matched target words {target_words}")
            continue

        uid_to_key = {}
        for key in matched_keys:
            for uid in lvis_annotations[key]:
                uid_to_key.setdefault(uid, key)  # first match wins if a uid appears under >1 matched key
        uids = list(uid_to_key.keys())[: args.max_per_category]

        print(f"  category={category}: matched LVIS keys {matched_keys} "
              f"-> {len(uid_to_key)} total UIDs, taking {len(uids)}")
        category_to_uids[category] = [(uid, uid_to_key[uid]) for uid in uids]

    if args.dry_run:
        return

    cfg.OBJAVERSE_ROOT.mkdir(parents=True, exist_ok=True)
    category_map = {}
    for category, uid_key_pairs in category_to_uids.items():
        uids = [uid for uid, _ in uid_key_pairs]
        print(f"\ndownloading {len(uids)} objects for category={category} ...")
        uid_to_path = objaverse.load_objects(
            uids=uids, download_processes=multiprocessing.cpu_count()
        )
        category_map[category] = [
            {"uid": uid, "lvis_key": key, "glb_path": str(uid_to_path[uid])}
            for uid, key in uid_key_pairs
            if uid in uid_to_path
        ]

    map_path = cfg.OBJAVERSE_ROOT / "category_uid_map.json"
    with open(map_path, "w") as f:
        json.dump(category_map, f, indent=2)
    print(f"\nwritten {map_path}")


if __name__ == "__main__":
    main()
