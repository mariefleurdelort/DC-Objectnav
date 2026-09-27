import argparse
import pandas as pd
from pathlib import Path

CSV_PATH = "data/trajectories/per_episode_results.csv"

SCHEMAS = {
    8: ["scene_id", "episode_id", "object_category", "outcome",
        "success", "spl", "distance_to_goal", "num_steps"],
    9: ["run_id", "scene_id", "episode_id", "object_category", "outcome",
        "success", "spl", "distance_to_goal", "num_steps"],
    10: ["run_id", "scene_id", "episode_id", "object_category", "outcome",
         "failure_cause", "success", "spl", "distance_to_goal", "num_steps"],
}

def load_results():
    with open(CSV_PATH) as f:
        lines = f.readlines()

    rows_by_width = {8: [], 9: [], 10: []}
    for line in lines[1:]:  # skip header (whichever schema it belongs to)
        line = line.rstrip("\n")
        if not line:
            continue
        fields = line.split(",")
        n = len(fields)
        if n in rows_by_width:
            rows_by_width[n].append(fields)
        # silently drop anything with an unexpected width rather than crash

    frames = []
    for width, rows in rows_by_width.items():
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=SCHEMAS[width])
        if "run_id" not in df.columns:
            df["run_id"] = None
        if "failure_cause" not in df.columns:
            df["failure_cause"] = None
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["episode_id"] = df["episode_id"].astype(str)
    df["success"] = pd.to_numeric(df["success"], errors="coerce")
    df["spl"] = pd.to_numeric(df["spl"], errors="coerce")
    df["num_steps"] = pd.to_numeric(df["num_steps"], errors="coerce")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--failures-only", action="store_true")
    args = ap.parse_args()

    df = load_results()
    subset = df[df.scene_id == args.scene]
    if args.run_id:
        subset = subset[subset.run_id == args.run_id]
    else:
        subset = subset.drop_duplicates(subset=["episode_id"], keep="last")
    if args.failures_only:
        subset = subset[subset.success == 0]

    subset = subset.sort_values("episode_id", key=lambda s: s.astype(int))
    cols = ["episode_id", "outcome", "failure_cause", "success", "spl", "distance_to_goal", "num_steps"]
    print(subset[cols].to_string(index=False))
    print(f"\nTotal: {len(subset)}  |  Success: {int(subset['success'].sum())}/{subset['success'].notna().sum()} "
          f"({subset['success'].mean()*100:.1f}%)")

if __name__ == "__main__":
    main()
