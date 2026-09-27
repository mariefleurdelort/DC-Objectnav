#!/usr/bin/env python
# False-positive analysis across models, objects, frame positions and viewpoints.

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = ["groundingdino", "owlv2", "sam3", "yoloworld"]  # fixed order -> fixed colors
MODEL_COLORS = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
# single-hue sequential ramp (light = near zero, dark = many false positives)
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
INK, MUTED = "#1f1f1f", "#6b6b6b"

def load_jsonl(path):
    with open(path) as f:
        return pd.DataFrame([json.loads(l) for l in f if l.strip()])


def clean(df, name):
    """Drop rows whose inference call errored; they are neither FP nor non-FP."""
    if "error" in df.columns:
        bad = df["error"].notna()
        if bad.any():
            print(f"  [{name}] dropping {int(bad.sum())} rows with inference errors")
        df = df[~bad].copy()
    df["false_positive"] = df["false_positive"].astype(bool)
    if "relabeled_true_object" in df.columns:
        df["relabeled_true_object"] = df["relabeled_true_object"].fillna(False).astype(bool)
    return df


def load_all(root):
    main = clean(load_jsonl(root / "data/renders/false_positive_inference_log.jsonl"), "main")
    frames = []
    for m in MODELS:
        p = root / "data/renders_framing_falsepositive" / m / "framing_inference_log.jsonl"
        if not p.exists():
            print(f"  [framing] missing {p}, skipping {m}")
            continue
        df = load_jsonl(p)
        frames.append(clean(df[df["model"] == m], f"framing/{m}"))
    framing = pd.concat(frames, ignore_index=True)
    # offsets are floats like -32.72727; round so grid cells group reliably
    framing["offset_x_deg"] = framing["offset_x_deg"].round(1)
    framing["offset_y_deg"] = framing["offset_y_deg"].round(1)
    return main, framing


def models_in(df):
    return [m for m in MODELS if m in set(df["model"])]


def summarize(df, keys):
    g = df.groupby(keys, dropna=False)
    out = g.agg(n_images=("false_positive", "size"),
                fp_count=("false_positive", "sum"),
                n_instances=("model_id", "nunique")).reset_index()
    out["fp_count"] = out["fp_count"].astype(int)
    out["fp_rate"] = out["fp_count"] / out["n_images"]
    return out

#plots
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c9c9")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color="#ececec", linewidth=0.8)
    ax.set_axisbelow(True)


def _lbl(v):
    return f"{v:.0f}" if isinstance(v, (int, float, np.number)) else str(v)


def heatmap(ax, pivot, title, vmax, fmt="{:.0f}", xlabel="X offset (deg)", ylabel="Y offset (deg)"):
    vals = pivot.values.astype(float)
    im = ax.imshow(vals, cmap=SEQ, vmin=0, vmax=vmax if vmax > 0 else 1,
                   origin="lower", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([_lbl(v) for v in pivot.columns], rotation=90, fontsize=6, color=MUTED)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_lbl(v) for v in pivot.index], fontsize=6, color=MUTED)
    ax.set_xlabel(xlabel, fontsize=7, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=7, color=MUTED)
    ax.set_title(title, fontsize=9, color=INK)
    fs = 5 if vals.shape[1] > 10 else 7
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                continue
            dark = vmax > 0 and v > 0.55 * vmax
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs,
                    color="white" if dark else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


# false positives analysis: number of fp per model, per object category,...
def part1_totals(main, framing, out):
    rows = []
    for m in MODELS:
        mm, ff = main[main["model"] == m], framing[framing["model"] == m]
        rows.append({
            "model": m,
            "main_images": len(mm),
            "main_fp": int(mm["false_positive"].sum()),
            "main_fp_rate": mm["false_positive"].mean() if len(mm) else np.nan,
            "grid_seed_instances": ff["model_id"].nunique(),
            "grid_images": len(ff),
            "grid_fp": int(ff["false_positive"].sum()),
            "grid_fp_rate": ff["false_positive"].mean() if len(ff) else np.nan,
            "grid_fp_on_true_object_pct": (100 * ff.loc[ff["false_positive"], "relabeled_true_object"].mean()
                                           if ff["false_positive"].any() else np.nan),
        })
    t = pd.DataFrame(rows)
    t.to_csv(out / "1_totals_per_model.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.6 * len(MODELS) + 1.8))
    for ax, n_col, fp_col, title in [
        (axes[0], "main_images", "main_fp", "Main sweep (every elevation x azimuth)"),
        (axes[1], "grid_images", "grid_fp", "XY offset grid (around each model's own FP seeds)")]:
        style(ax)
        ax.grid(axis="y", visible=False)
        y = np.arange(len(t))[::-1]
        ax.barh(y, t[n_col], height=0.6, color="#e4e4e4", label="images run")
        ax.barh(y, t[fp_col], height=0.6, color=[MODEL_COLORS[m] for m in t["model"]],
                edgecolor="white", linewidth=2, label="false positives")
        lim = max(t[n_col].max(), 1)
        for yi, (_, r) in zip(y, t.iterrows()):
            rate = 100 * r[fp_col] / r[n_col] if r[n_col] else 0
            ax.text(r[n_col] + lim * 0.01, yi, f"{r[fp_col]} / {r[n_col]}  ({rate:.0f}%)",
                    va="center", fontsize=8, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(t["model"], fontsize=9, color=INK)
        ax.set_xlim(0, lim * 1.35)
        ax.set_xlabel("images (grey) / false positives (color)", fontsize=8, color=MUTED)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.suptitle("Total false positives vs images per model", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "1_totals_per_model.png", dpi=150)
    plt.close(fig)
    return t


def part2_cells(framing, out):
    per_obj = summarize(framing, ["model", "category", "offset_y_deg", "offset_x_deg"])
    pooled = summarize(framing, ["model", "offset_y_deg", "offset_x_deg"]).assign(category="ALL")
    cells = pd.concat([pooled, per_obj], ignore_index=True)
    cells = cells[["model", "category", "offset_x_deg", "offset_y_deg",
                   "fp_count", "n_images", "n_instances", "fp_rate"]]
    cells.to_csv(out / "2_cell_counts.csv", index=False)

    ms = models_in(framing)
    # per model: pooled + one panel per object, shared count scale within the model
    for m in ms:
        cm = cells[cells["model"] == m]
        cats = ["ALL"] + sorted(c for c in cm["category"].unique() if c != "ALL")
        vmax = cm.loc[cm["category"] == "ALL", "fp_count"].max()
        ncol = min(3, len(cats)); nrow = int(np.ceil(len(cats) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.6 * nrow), squeeze=False)
        for ax, c in zip(axes.flat, cats):
            d = cm[cm["category"] == c]
            piv = d.pivot_table(index="offset_y_deg", columns="offset_x_deg", values="fp_count", aggfunc="sum")
            n_inst = framing[(framing["model"] == m) & ((framing["category"] == c) | (c == "ALL"))]["model_id"].nunique()
            heatmap(ax, piv, f"{c}  (total {int(d['fp_count'].sum())}, {n_inst} instances)",
                    vmax=vmax if c == "ALL" else d["fp_count"].max())
        for ax in list(axes.flat)[len(cats):]:
            ax.axis("off")
        fig.suptitle(f"{m}: false-positive count per frame-position cell (max per cell = #instances)",
                     fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / f"2_cells_{m}.png", dpi=150)
        plt.close(fig)

    # all models pooled, count row + rate row (rate is the fair cross-model comparison)
    fig, axes = plt.subplots(2, len(ms), figsize=(5.2 * len(ms), 9.2), squeeze=False)
    for j, m in enumerate(ms):
        d = cells[(cells["model"] == m) & (cells["category"] == "ALL")]
        piv_c = d.pivot_table(index="offset_y_deg", columns="offset_x_deg", values="fp_count", aggfunc="sum")
        piv_r = d.pivot_table(index="offset_y_deg", columns="offset_x_deg", values="fp_rate", aggfunc="mean")
        heatmap(axes[0, j], piv_c, f"{m}: count (total {int(d['fp_count'].sum())})", vmax=piv_c.values.max())
        heatmap(axes[1, j], piv_r * 100, f"{m}: rate %", vmax=100, fmt="{:.0f}")
    fig.suptitle("False positives by frame position, all objects pooled", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "2_cells_all_models.png", dpi=150)
    plt.close(fig)
    return cells


def part3_compare(main, framing, out):
    # 3a model x object
    a = summarize(main, ["model", "category"]).add_prefix("main_").rename(
        columns={"main_model": "model", "main_category": "category"})
    b = summarize(framing, ["model", "category"]).add_prefix("grid_").rename(
        columns={"grid_model": "model", "grid_category": "category"})
    mo = a.merge(b, on=["model", "category"], how="outer")
    mo.to_csv(out / "3a_model_by_object.csv", index=False)

    ms = models_in(main)
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * len(ms) + 3))
    for ax, col, title in [(axes[0], "main_fp_rate", "Main sweep FP rate %"),
                           (axes[1], "grid_fp_rate", "Offset-grid FP rate % (seeded instances only)")]:
        piv = mo.pivot_table(index="model", columns="category", values=col).reindex(ms) * 100
        heatmap(ax, piv, title, vmax=100, xlabel="object", ylabel="model")
        ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8, color=MUTED)
        ax.set_yticklabels(piv.index, fontsize=8, color=MUTED)
        for t in ax.texts:
            t.set_fontsize(8)
    fig.tight_layout()
    fig.savefig(out / "3a_model_by_object.png", dpi=150)
    plt.close(fig)

    # 3b angles (main sweep)
    vp = summarize(main, ["model", "category", "elevation_deg", "azimuth_deg"])
    vp.to_csv(out / "3b_model_by_viewpoint.csv", index=False)
    for m in ms:
        d = summarize(main[main["model"] == m], ["elevation_deg", "azimuth_deg"])
        piv = d.pivot_table(index="elevation_deg", columns="azimuth_deg", values="fp_rate") * 100
        fig, ax = plt.subplots(figsize=(8, 4.5))
        heatmap(ax, piv, f"{m}: main-sweep FP rate % by viewpoint (all objects)", vmax=100,
                xlabel="azimuth (deg)", ylabel="elevation (deg)")
        fig.tight_layout()
        fig.savefig(out / f"3b_elev_az_{m}.png", dpi=150)
        plt.close(fig)

    el = summarize(main, ["model", "elevation_deg"])
    fig, ax = plt.subplots(figsize=(7, 4))
    style(ax)
    for m in ms:
        d = el[el["model"] == m].sort_values("elevation_deg")
        ax.plot(d["elevation_deg"], d["fp_rate"] * 100, color=MODEL_COLORS[m], linewidth=2,
                marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1.5, label=m)
        ax.annotate(m, (d["elevation_deg"].iloc[-1], d["fp_rate"].iloc[-1] * 100),
                    xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
    ax.set_xlabel("camera elevation (deg)", fontsize=8, color=MUTED)
    ax.set_ylabel("FP rate %", fontsize=8, color=MUTED)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Main sweep: false-positive rate vs elevation", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "3b_rate_vs_elevation.png", dpi=150)
    plt.close(fig)

    # 3c frame distance (offset grid)
    f = framing.copy()
    f["frame_distance_deg"] = np.sqrt(f["offset_x_deg"] ** 2 + f["offset_y_deg"] ** 2).round(0)
    f["ring"] = (f["frame_distance_deg"] // 10 * 10).astype(int)
    rd = summarize(f, ["model", "ring"])
    rd.to_csv(out / "3c_rate_vs_frame_distance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    style(ax)
    for m in models_in(framing):
        d = rd[rd["model"] == m].sort_values("ring")
        ax.plot(d["ring"], d["fp_rate"] * 100, color=MODEL_COLORS[m], linewidth=2, marker="o",
                markersize=6, markeredgecolor="white", markeredgewidth=1.5, label=m)
    ax.set_xlabel("object distance from frame center (deg, 10-deg bins)", fontsize=8, color=MUTED)
    ax.set_ylabel("FP rate %", fontsize=8, color=MUTED)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Offset grid: does the false positive fade toward the frame edge?",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "3c_rate_vs_frame_distance.png", dpi=150)
    plt.close(fig)

    # 3d seed viewpoint (offset grid)
    summarize(framing, ["model", "base_elevation_deg", "base_azimuth_deg"]).to_csv(
        out / "3d_seed_viewpoints.csv", index=False)
    return mo, el, rd


def part4_rotation(main, framing, out):
    """Place every main-sweep view at (d_azimuth, d_elevation) from each FP seed."""
    seeds = framing[["model", "model_id", "category", "base_elevation_deg", "base_azimuth_deg"]].drop_duplicates()
    rel = main.merge(seeds, on=["model", "model_id", "category"], how="inner")
    if rel.empty:
        print("  [rotation] no seeds matched the main sweep; skipping part 4")
        return None
    rel["d_azimuth_deg"] = ((rel["azimuth_deg"] - rel["base_azimuth_deg"] + 180) % 360) - 180
    rel["d_elevation_deg"] = rel["elevation_deg"] - rel["base_elevation_deg"]
    rot = summarize(rel, ["model", "category", "d_elevation_deg", "d_azimuth_deg"])
    pooled = summarize(rel, ["model", "d_elevation_deg", "d_azimuth_deg"]).assign(category="ALL")
    pd.concat([pooled, rot], ignore_index=True).to_csv(out / "4_rotation_cells.csv", index=False)

    for m in models_in(rel):
        d = pooled[pooled["model"] == m]
        pc = d.pivot_table(index="d_elevation_deg", columns="d_azimuth_deg", values="fp_count")
        pr = d.pivot_table(index="d_elevation_deg", columns="d_azimuth_deg", values="fp_rate") * 100
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        heatmap(axes[0], pc, f"{m}: FP count", vmax=np.nanmax(pc.values),
                xlabel="azimuth change from seed (deg)", ylabel="elevation change from seed (deg)")
        heatmap(axes[1], pr, f"{m}: FP rate %  (0,0 = the seed view)", vmax=100,
                xlabel="azimuth change from seed (deg)", ylabel="elevation change from seed (deg)")
        fig.suptitle(f"{m}: orbiting away from each false-positive seed (main-sweep 30-deg grid)",
                     fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / f"4_rotation_{m}.png", dpi=150)
        plt.close(fig)
    return pooled

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT, help="viewpoint-sensitivity/ directory")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default <root>/analysis_fp)")
    args = ap.parse_args()
    out = args.out or args.root / "analysis_fp"
    out.mkdir(parents=True, exist_ok=True)

    print("loading logs ...")
    main_df, framing = load_all(args.root)

    print("\n[1] totals per model")
    t = part1_totals(main_df, framing, out)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n[2] per-cell counts")
    cells = part2_cells(framing, out)
    top = (cells[cells["category"] == "ALL"].sort_values(["model", "fp_count"], ascending=[True, False])
           .groupby("model").head(3))
    print("top 3 cells per model (all objects pooled):")
    print(top.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\n[3] comparisons")
    mo, el, rd = part3_compare(main_df, framing, out)
    print("model x object FP rate (main sweep):")
    print((mo.pivot_table(index="model", columns="category", values="main_fp_rate") * 100)
          .round(0).to_string())
    print("\nFP rate % vs elevation (main sweep):")
    print((el.pivot_table(index="model", columns="elevation_deg", values="fp_rate") * 100).round(0).to_string())
    print("\nFP rate % vs distance from frame center (offset grid):")
    print((rd.pivot_table(index="model", columns="ring", values="fp_rate") * 100).round(0).to_string())

    print("\n[4] rotation around each seed (from existing main sweep)")
    rot = part4_rotation(main_df, framing, out)
    if rot is not None:
        print((rot.pivot_table(index=["model", "d_elevation_deg"], columns="d_azimuth_deg",
                               values="fp_rate") * 100).round(0).to_string())

    print(f"\nwrote everything to {out}/")


if __name__ == "__main__":
    main()
