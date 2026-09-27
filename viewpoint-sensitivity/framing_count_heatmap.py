#!/usr/bin/env python
#Raw COUNT (not rate/fraction) of false positives per XY offset-grid cell, per model.
#How many instances actually triggered a false positive at each frame position, pooling all of that model's seeded instances together.

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS = ["owlv2", "yoloworld", "sam3", "groundingdino"]


def load(model):
    base = PROJECT_ROOT / "data" / "renders_framing_falsepositive" / model
    inf = pd.DataFrame([json.loads(l) for l in open(base / "framing_inference_log.jsonl")])
    return inf[inf["model"] == model]


def plot_count_heatmap(pivot, model, total, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pivot.values, cmap="hot", vmin=0, origin="lower", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.0f}" for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.0f}" for v in pivot.index])
    ax.set_xlabel("X offset (degrees-equivalent off-center)")
    ax.set_ylabel("Y offset (degrees-equivalent off-center)")
    ax.set_title(f"{model}: false positive COUNT by frame position (total={total})")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{int(val)}", ha="center", va="center",
                    color="white" if val < pivot.values.max() * 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="count of false positives")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


summary = []
for model in MODELS:
    df = load(model)
    total = int(df["false_positive"].sum())
    n_views = len(df)
    summary.append({"model": model, "total_false_positives": total, "n_views": n_views,
                     "rate": total / n_views})

    pivot = df.pivot_table(index="offset_y_deg", columns="offset_x_deg", values="false_positive",
                            aggfunc="sum", dropna=False).fillna(0)
    out_dir = PROJECT_ROOT / "data" / "renders_framing_falsepositive" / model / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model}_false_positive_count.png"
    plot_count_heatmap(pivot, model, total, out_path)
    print(f"{model}: {total} false positives / {n_views} views (rate={total/n_views:.3f}) -> {out_path}")

print("\n=== summary ===")
print(pd.DataFrame(summary).to_string(index=False))
