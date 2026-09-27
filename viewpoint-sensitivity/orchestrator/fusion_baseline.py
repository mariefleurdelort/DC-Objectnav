#!/usr/bin/env python
"""Naive multi-view fusion baseline:
For each model, compares:
  - the best single fixed viewpoint's detection rate (the (elevation, azimuth)
    that performs best on its own, evaluated across all object instances), vs.
  - the fused detection rate: an instance counts as "detected" if it was
    correctly detected from AT LEAST ONE of its sampled viewpoints (naive
    OR-fusion across elevation x azimuth).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "renders" / "inference_log.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "renders" / "fusion_summary.csv"


def load_inference_log(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["instance_key"] = df["category"] + "/" + df["model_id"].astype(str)
    df["viewpoint"] = list(zip(df["elevation_deg"], df["azimuth_deg"]))
    return df


def summarize_model(df_model: pd.DataFrame) -> dict:
    # single-viewpoint rate: for each (elev, azim), mean `correct` across
    # whichever instances have a rendered view at that viewpoint
    per_viewpoint = df_model.groupby("viewpoint")["correct"].mean()
    best_viewpoint, best_rate = per_viewpoint.idxmax(), per_viewpoint.max()

    # fused rate: per instance, True if ANY of its sampled views was correct
    fused_per_instance = df_model.groupby("instance_key")["correct"].any()
    fused_rate = fused_per_instance.mean()

    n_instances = df_model["instance_key"].nunique()
    n_views = len(df_model)

    return {
        "model": df_model["model"].iloc[0],
        "n_instances": n_instances,
        "n_views": n_views,
        "best_single_viewpoint_elev": best_viewpoint[0],
        "best_single_viewpoint_azim": best_viewpoint[1],
        "best_single_viewpoint_rate": best_rate,
        "fused_rate": fused_rate,
        "absolute_gain": fused_rate - best_rate,
        "relative_gain_pct": (fused_rate - best_rate) / best_rate * 100 if best_rate > 0 else float("inf"),
    }


def summarize_by_category(df_model: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, df_cat in df_model.groupby("category"):
        per_viewpoint = df_cat.groupby("viewpoint")["correct"].mean()
        best_rate = per_viewpoint.max() if len(per_viewpoint) else 0.0
        fused_rate = df_cat.groupby("instance_key")["correct"].any().mean()
        rows.append({
            "model": df_model["model"].iloc[0],
            "category": category,
            "best_single_viewpoint_rate": best_rate,
            "fused_rate": fused_rate,
            "absolute_gain": fused_rate - best_rate,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    df = load_inference_log(Path(args.input))
    print(f"loaded {len(df)} inference results across "
          f"{df['model'].nunique()} models, {df['instance_key'].nunique()} instances\n")

    summaries = [summarize_model(df_model) for _, df_model in df.groupby("model")]
    summary_df = pd.DataFrame(summaries).sort_values("model")

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("=== per-model: best single viewpoint vs. naive multi-view fusion ===")
    print(summary_df.to_string(index=False))

    summary_df.to_csv(args.output, index=False)
    print(f"\nwritten to {args.output}")

    by_category_path = Path(args.output).with_name("fusion_summary_by_category.csv")
    by_category_dfs = [summarize_by_category(df_model) for _, df_model in df.groupby("model")]
    by_category_df = pd.concat(by_category_dfs, ignore_index=True).sort_values(["model", "category"])
    by_category_df.to_csv(by_category_path, index=False)
    print(f"per-category breakdown written to {by_category_path}")

    overall_best = summary_df["best_single_viewpoint_rate"].mean()
    overall_fused = summary_df["fused_rate"].mean()
    print(f"\n=== headline (averaged across all {len(summary_df)} models) ===")
    print(f"  mean best-single-viewpoint rate: {overall_best:.3f}")
    print(f"  mean fused (any-viewpoint) rate: {overall_fused:.3f}")
    print(f"  absolute gain from fusion:       {overall_fused - overall_best:+.3f}")


if __name__ == "__main__":
    main()
