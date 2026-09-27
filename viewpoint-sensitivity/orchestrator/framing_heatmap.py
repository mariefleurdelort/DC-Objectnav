#!/usr/bin/env python
"""Builds XY-offset confidence + correctness heatmaps from the framing-grid
inference log
- True-positive (default), success = "correct" (IoU>=0.5 against real ground truth)
- False-positive, "success" = "false_positive" (any detection at all -- the model
being fooled is the bad outcome here, opposite polarity from "correct")
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAMING_DIR = PROJECT_ROOT / "data" / "renders_framing"
DEFAULT_MANIFEST = FRAMING_DIR / "framing_manifest.jsonl"
DEFAULT_INFERENCE_LOG = FRAMING_DIR / "framing_inference_log.jsonl"


def load_joined(manifest_path: Path, inference_log_path: Path) -> pd.DataFrame:
    man = pd.DataFrame([json.loads(l) for l in open(manifest_path)])
    inf = pd.DataFrame([json.loads(l) for l in open(inference_log_path)])
    # inf already carries offset_x_deg/offset_y_deg (run_inference.py passes through
    # whatever viewpoint fields the manifest has) -- only pull gt_pixel_count from man.
    df = inf.merge(man[["view_id", "gt_pixel_count"]], on="view_id", how="left")
    df["instance_key"] = df["category"] + "/" + df["model_id"]
    return df


def plot_heatmap(pivot: pd.DataFrame, title: str, out_path: Path, value_label: str, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    # "hot" runs black->red->orange->yellow->white, and NaN (no detection at that
    # frame position) rendering as blank/white by default is indistinguishable from
    # a genuinely high-confidence white cell -- render missing data as gray instead.
    cmap = matplotlib.colormaps["hot"].copy()
    cmap.set_bad("gray")
    im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.0f}" for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.0f}" for v in pivot.index])
    ax.set_xlabel("X offset (degrees-equivalent off-center)")
    ax.set_ylabel("Y offset (degrees-equivalent off-center)")
    ax.set_title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val < (vmax or np.nanmax(pivot.values)) * 0.6 else "black", fontsize=7)

    fig.colorbar(im, ax=ax, label=value_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_pivots(df_subset: pd.DataFrame, value_field: str):
    # dropna=False: keep every (y, x) grid cell even if every instance had
    # zero detections there
    conf_pivot = df_subset.pivot_table(index="offset_y_deg", columns="offset_x_deg", values="best_score",
                                        aggfunc="mean", dropna=False)
    success_pivot = df_subset.pivot_table(index="offset_y_deg", columns="offset_x_deg", values=value_field,
                                           aggfunc="mean", dropna=False)
    return conf_pivot, success_pivot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--inference-log", default=str(DEFAULT_INFERENCE_LOG))
    parser.add_argument("--value-field", default="correct", choices=["correct", "false_positive"],
                         help="'correct' (default, true-positive data) or 'false_positive' "
                              "(false-positive data -- see this file's docstring)")
    parser.add_argument("--heatmap-dir", default=None,
                         help="default: a 'heatmaps' folder next to --inference-log")
    parser.add_argument("--per-instance", action="store_true",
                         help="also produce one heatmap per (model, instance), not just the per-model aggregate")
    args = parser.parse_args()

    heatmap_dir = Path(args.heatmap_dir) if args.heatmap_dir else Path(args.inference_log).parent / "heatmaps"
    success_label = "fraction correct" if args.value_field == "correct" else "fraction w/ a false positive"
    success_name = "correctness" if args.value_field == "correct" else "false_positive_rate"

    df = load_joined(Path(args.manifest), Path(args.inference_log))
    print(f"loaded {len(df)} framing-grid inference results across "
          f"{df['model'].nunique()} models, {df['instance_key'].nunique()} instances "
          f"(value_field={args.value_field})")

    for model, df_model in df.groupby("model"):
        conf_pivot, success_pivot = build_pivots(df_model, args.value_field)
        plot_heatmap(conf_pivot, f"{model}: mean confidence by frame position (all instances)",
                     heatmap_dir / f"{model}_confidence.png", "mean confidence score")
        plot_heatmap(success_pivot, f"{model}: {success_name} by frame position (all instances)",
                     heatmap_dir / f"{model}_{success_name}.png", success_label, vmin=0, vmax=1)
        print(f"  {model}: wrote aggregate heatmaps")

        if args.per_instance:
            for instance_key, df_inst in df_model.groupby("instance_key"):
                conf_pivot, success_pivot = build_pivots(df_inst, args.value_field)
                safe_name = instance_key.replace("/", "_")
                plot_heatmap(conf_pivot, f"{model}: confidence -- {instance_key}",
                             heatmap_dir / model / f"{safe_name}_confidence.png", "confidence score")
                plot_heatmap(success_pivot, f"{model}: {args.value_field} -- {instance_key}",
                             heatmap_dir / model / f"{safe_name}_{success_name}.png", success_label,
                             vmin=0, vmax=1)
            print(f"  {model}: wrote {df_model['instance_key'].nunique()} per-instance heatmaps")

    print(f"\nall heatmaps written under {heatmap_dir}")


if __name__ == "__main__":
    main()
