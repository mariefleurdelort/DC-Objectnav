"""
combine_all_scenes.py
Combines clean, verified results across all 5 Gibson val scenes into one
final results file, and reports overall + per-scene success/SPL.
"""
import pandas as pd

CSV_PATH = "data/trajectories/per_episode_results.csv"
OLD_COLS = ["scene_id", "episode_id", "object_category", "outcome",
            "success", "spl", "distance_to_goal", "num_steps"]
NEW_COLS = ["run_id"] + OLD_COLS
COROZAL_CLEAN_RUN_ID = "20260713_054400"

# --- load full file (schema-split-aware) ---
with open(CSV_PATH) as f:
    lines = f.readlines()
split_at = next(i for i, line in enumerate(lines[1:], start=1)
                if len(line.strip().split(",")) == 9)

df_old = pd.read_csv(CSV_PATH, skiprows=1, nrows=split_at - 1,
                      names=OLD_COLS, header=None)
df_old["run_id"] = None
df_new = pd.read_csv(CSV_PATH, skiprows=split_at, names=NEW_COLS, header=None)
df = pd.concat([df_old, df_new], ignore_index=True)
df["episode_id"] = df["episode_id"].astype(str)

# --- Corozal: only the verified clean run ---
corozal = df[(df.scene_id == "Corozal") & (df.run_id == COROZAL_CLEAN_RUN_ID)]

# --- other 4 scenes: verified clean already (dedup by episode_id is safe,
#     since both of their historical runs used the correct data_path) ---
others = df[df.scene_id.isin(["Collierville", "Darden", "Markleeville", "Wiconisco"])]
others_deduped = others.drop_duplicates(subset=["scene_id", "episode_id"], keep="last")

final = pd.concat([others_deduped, corozal], ignore_index=True)
final["success"] = pd.to_numeric(final["success"], errors="coerce")
final["spl"] = pd.to_numeric(final["spl"], errors="coerce")

final.to_csv("data/trajectories/per_episode_results_final.csv", index=False)

print(f"Total episodes: {len(final)}")
print(final.groupby("scene_id").size(), "\n")

print("=== Per-scene success rate & SPL ===")
print(final.groupby("scene_id")[["success", "spl"]].mean())

print(f"\n=== Overall ===")
n_valid = final["success"].notna().sum()
print(f"Success rate: {final['success'].mean()*100:.2f}%  ({int(final['success'].sum())}/{n_valid})")
print(f"Mean SPL: {final['spl'].mean():.4f}")

print(f"\nPublished VLFM Gibson benchmark (Weighted avg. config): SR=84.0%, SPL=52.2")
print(f"Published VLFM Gibson benchmark (Replacement config):   SR=76.1%, SPL=48.0")