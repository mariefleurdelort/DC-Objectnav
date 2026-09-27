#!/usr/bin/env python
"""At which viewing ANGLES (elevation/azimuth -- not frame position) does each
model hallucinate a visually-confusable wrong category most. HEATMAPS"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = PROJECT_ROOT / "data" / "renders" / "false_positive_inference_log.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "renders" / "false_positive_angle_heatmap.png"

MODEL_ORDER = ["groundingdino", "owlv2", "sam3", "yoloworld"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--value-field", default="false_positive",
                         choices=["false_positive", "relabeled_true_object"],
                         help="'false_positive' (any detection at all) or 'relabeled_true_object' "
                              "(false positive AND it was IoU>=0.5 on the real object)")
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.log)]
    df = pd.DataFrame(rows)
    print(f"loaded {len(df)} rows across {df['model'].nunique()} models")

    elevations = sorted(df["elevation_deg"].unique())
    azimuths = sorted(df["azimuth_deg"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, model in zip(axes, MODEL_ORDER):
        df_model = df[df["model"] == model]
        pivot = df_model.pivot_table(index="elevation_deg", columns="azimuth_deg",
                                      values=args.value_field, aggfunc="mean", dropna=False)
        pivot = pivot.reindex(index=elevations, columns=azimuths)  # force identical grid/order for every model

        im = ax.imshow(pivot.values, cmap="hot", vmin=0, vmax=1, origin="lower", aspect="auto")
        ax.set_xticks(range(len(azimuths)))
        ax.set_xticklabels(azimuths, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(elevations)))
        ax.set_yticklabels(elevations, fontsize=8)
        ax.set_xlabel("Azimuth (deg)")
        ax.set_ylabel("Elevation (deg)")
        rate = df_model[args.value_field].mean()
        ax.set_title(f"{model} (overall: {rate:.1%})")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color="white" if val < 0.6 else "black", fontsize=6)

    fig.colorbar(im, ax=axes, label=args.value_field, shrink=0.8)
    fig.suptitle(f"False-positive rate by viewing angle -- {args.value_field} "
                 "(shared grid + color scale across all 4 models)", fontsize=13)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"written to {out_path}")


if __name__ == "__main__":
    main()
