# clean_corozal_results.py
import json, gzip
import pandas as pd

# 1. Get the ground-truth set of valid episode_ids for Corozal
with gzip.open("data/datasets/objectnav/gibson/dcon_subset/v1.1_sub10/val/content_fixed/Corozal.json.gz", "rt") as f:
    corozal_subset = json.load(f)
valid_ids = {str(ep["episode_id"]) for ep in corozal_subset["episodes"]}
print(f"Valid Corozal episode_ids in subset: {len(valid_ids)}")  # expect 59

# 2. Load raw results and normalize episode_id type
df = pd.read_csv("data/trajectories/per_episode_results.csv")
df["episode_id"] = df["episode_id"].astype(str)

# 3. Split out Corozal, filter to only valid ids, dedup
corozal = df[df.scene_id == "Corozal"]
corozal_clean = corozal[corozal.episode_id.isin(valid_ids)]
corozal_deduped = corozal_clean.drop_duplicates(subset=["episode_id"], keep="last")
print(f"Corozal rows before: {len(corozal)}, after filter+dedup: {len(corozal_deduped)}")  # expect 59

# 4. Rebuild the full results file: other scenes deduped as before, Corozal replaced cleanly
others = df[df.scene_id != "Corozal"].drop_duplicates(subset=["scene_id", "episode_id"], keep="last")
final = pd.concat([others, corozal_deduped], ignore_index=True)
final.to_csv("data/trajectories/per_episode_results_deduped_v2.csv", index=False)
print(f"Final total rows: {len(final)}")  # expect 249 (250 minus excluded ep 3)

# still in clean_corozal_results.py, or run interactively
missing_ids = valid_ids - set(corozal_deduped.episode_id)
print(f"Missing episode_ids: {sorted(missing_ids, key=int)}")


import json, gzip
with gzip.open("data/datasets/objectnav/gibson/dcon_subset/v1.1_sub10/val/content_fixed/Corozal.json.gz", "rt") as f:
    data = json.load(f)

ids_in_order = [str(ep["episode_id"]) for ep in data["episodes"]]
print(f"Total episodes: {len(ids_in_order)}")
missing = ['59', '60', '82', '97', '107', '118']
for m in missing:
    if m in ids_in_order:
        print(f"episode_id {m} is at position {ids_in_order.index(m)} of {len(ids_in_order)}")
    else:
        print(f"episode_id {m} NOT FOUND in dataset")