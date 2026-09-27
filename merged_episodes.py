import json, gzip, os
import pandas as pd

FULL_DIR = "data/datasets/objectnav/gibson/v1.1/val/content"
SUBSET_DIR = "data/datasets/objectnav/gibson/dcon_subset/v1.1_sub10/val/content"
OUT_DIR = "data/datasets/objectnav/gibson/dcon_subset/v1.1_sub10/val/content_fixed"
COMBINED_VAL = "data/datasets/objectnav/gibson/v1.1/val/val.json.gz"
os.makedirs(OUT_DIR, exist_ok=True)

# Load dataset-level metadata (category maps etc.) from the combined file,
# since per-scene content files may or may not carry these keys.
with gzip.open(COMBINED_VAL, "rt") as f:
    combined_data = json.load(f)
dataset_meta = {k: v for k, v in combined_data.items() if k != "episodes"}
print(f"Dataset-level metadata keys found: {list(dataset_meta.keys())}")

scenes = ["Collierville", "Corozal", "Darden", "Markleeville", "Wiconisco"]

for scene in scenes:
    with gzip.open(f"{FULL_DIR}/{scene}_episodes.json.gz", "rt") as f:
        full_data = json.load(f)
    # normalize episode_id to string on both sides
    full_by_id = {str(ep["episode_id"]): ep for ep in full_data["episodes"]}

    with gzip.open(f"{SUBSET_DIR}/{scene}_episodes.json.gz", "rt") as f:
        subset_data = json.load(f)

    fixed_episodes = []
    missing = []
    for ep in subset_data["episodes"]:
        eid = str(ep["episode_id"])
        if eid in full_by_id:
            fixed_episodes.append(full_by_id[eid])
        else:
            missing.append(eid)

    print(f"{scene}: matched {len(fixed_episodes)}/{len(subset_data['episodes'])}"
          f"{'  MISSING: ' + str(missing) if missing else ''}")
    
    subset_ids = [str(ep["episode_id"]) for ep in subset_data["episodes"]]
    id_counts = pd.Series(subset_ids).value_counts()
    dupe_ids = id_counts[id_counts > 1]
    print(f"{scene}: {len(dupe_ids)} duplicate episode_ids in subset_data "
          f"(total dupe rows: {dupe_ids.sum() - len(dupe_ids)})")
    if len(dupe_ids):
        print(dupe_ids.head(10))
    
    out_payload = {"episodes": fixed_episodes}
    # carry over dataset-level metadata the task loader needs (category id maps etc.)
    out_payload.update(dataset_meta)

    with gzip.open(f"{OUT_DIR}/{scene}_episodes.json.gz", "wt") as f:
        json.dump(out_payload, f)
    with gzip.open(f"{OUT_DIR}/{scene}.json.gz", "wt") as f:
        json.dump(out_payload, f)
