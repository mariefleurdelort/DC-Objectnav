#!/usr/bin/env python
"""TP / FP / FN / TN, precision, recall, F1 (+ specificity, FPR, MCC) for every
model, per viewpoint, per frame position, per word and per image.

Unit of evaluation = one QUERY: (image, model, prompted word).
  target query     (word == the rendered object):  TP if a box with IoU >= --iou
                                                   against the real object, else FN
  non-target query (any other word):               FP if ANY detection, else TN
Two scorings are reported side by side:
  box      -- the definition above (a detection on the wrong spot is an FN)
  presence -- ignores the box (TP = any detection for the right word); use this
              to compare against models that don't localize (CLIP, yes/no VLMs)

Experiments read (each is skipped if its logs are missing), run from
viewpoint-sensitivity/:

  main   data/renders/inference_log.jsonl                    (true prompt  -> TP/FN)
         data/renders/false_positive_inference_log.jsonl     (wrong prompt -> FP/TN)
         = 1 target + 1 non-target query per view (positives:negatives = 1:1)

  xy     data/renders_framing_falsepositive/<grid>/framing_inference_log.jsonl       (wrong prompt)
         data/renders_framing_falsepositive/<grid>/framing_true_inference_log.jsonl  (true prompt)
         The wrong-prompt logs exist already; produce the true-prompt ones with
         run_inference.py WITHOUT --false-positive on the same manifest (see the
         command printed when they are missing). Offsets come from framing_manifest.jsonl.

  bench  data/renders_benchmark/query_manifest.jsonl + benchmark_inference_log.jsonl
         = every image x every word (from expand_manifest_queries.py), 1:(N-1)

  Every log above is also read with a _<model> suffix (e.g. inference_log_rex_omni.jsonl),
  which is what run_vlm_inference.py writes for LocateAnything / Rex-Omni / GPT-6 Astra / Gemini / GPT.

    python confusion_metrics.py                 # all experiments found -> analysis_confusion/
    python confusion_metrics.py --iou 0.5 --boot 1000
    python confusion_metrics.py --ignore-pairs desk:table bookcase:cabinet   # ambiguous word pairs

Outputs per experiment in analysis_confusion/<experiment>/:
  overall.csv        per model: counts + metrics, 95% CI by bootstrapping object instances
  by_word.csv        per model x prompted word (precision/recall per category)
  by_view.csv        per model x (elevation, azimuth) or x (offset_x, offset_y)
  per_image.csv      per model x image (meaningful mostly for 'bench', 1 target + N-1 words)
  heatmaps_<model>.png   precision / recall / F1 / FPR over the view grid
  f1_all_models.png      F1 over the view grid, models side by side
  threshold_sweep.csv/png  recall, FPR, F1, MCC per model as the score threshold is raised
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

# empty cells (e.g. a model with no target rows) make all-NaN heatmaps / quantiles: expected, not errors
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Attempting to set identical")

ROOT = Path(__file__).resolve().parent
MODEL_ORDER = ["groundingdino", "owlv2", "sam3", "yoloworld", "locate_anything", "rex_omni", "gpt_astra",
               "clip", "gemini", "gpt"]
SERVER_MODELS = ["groundingdino", "owlv2", "sam3", "yoloworld"]   # served by orchestrator/run_inference.py
MODEL_COLORS = dict(zip(MODEL_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                      "#e87ba4", "#008300", "#e34948", "#4a3aa7", "#8a8a8a", "#8a8a8a"]))
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
INK, MUTED = "#1f1f1f", "#6b6b6b"
COLS = ["experiment", "model", "image_id", "model_id", "true_category", "query_category",
        "is_target", "detected", "iou", "score", "g1", "g2"]


# --------------------------------------------------------------------------- io
def load_jsonl(path):
    with open(path) as f:
        return pd.DataFrame([json.loads(l) for l in f if l.strip()])


def load_many(folder, stem):
    """<stem>.jsonl plus every per-model <stem>_<model>.jsonl (run_vlm_inference.py output)."""
    paths = [p for p in [folder / f"{stem}.jsonl"] + sorted(folder.glob(f"{stem}_*.jsonl")) if p.exists()]
    dfs = [d for d in (load_jsonl(p) for p in paths) if not d.empty]
    return pd.concat(dfs, ignore_index=True) if dfs else None


def detected_col(df):
    """Did the model return a detection? Decided PER ROW from whichever field that row has, because
    logs from different runners are concatenated and a column missing from one log is NaN there."""
    det = pd.Series(np.nan, index=df.index, dtype="float64")
    if "num_detections" in df.columns:
        nd = pd.to_numeric(df["num_detections"], errors="coerce")
        det = det.where(nd.isna(), (nd > 0).astype(float))
    if "best_box_xyxy" in df.columns:
        has_box = df["best_box_xyxy"].notna().astype(float)
        # a row with no num_detections field: use its box (only trust "no box" if the field was logged)
        det = det.where(det.notna() | df["best_box_xyxy"].isna(), has_box)
    if "false_positive" in df.columns:
        fp = df["false_positive"].map(lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v))
                                      else float(bool(v)))
        det = det.where(det.notna(), fp)
    return det.fillna(0).astype(bool)


def drop_errors(df, name):
    if "error" in df.columns and df["error"].notna().any():
        err = df["error"].notna()
        print(f"  [{name}] dropping {int(err.sum())} of {len(df)} rows with inference errors")
        by_model = df.loc[err].groupby("model")["error"].agg(["size", "first"]) if "model" in df else None
        if by_model is not None:
            for m, r in by_model.iterrows():
                print(f"      {m}: {r['size']} errors, e.g. {str(r['first'])[:140]}")
        if err.all():
            print(f"  [{name}] WARNING: EVERY row is an error (servers not running? wrong env?). "
                  "Fix the cause, then re-run that inference pass.")
        df = df[~err]
    return df.copy()


def norm(df, experiment, is_target, query_col, iou_col, g1, g2):
    out = pd.DataFrame({
        "experiment": experiment,
        "model": df["model"].values,
        "image_id": df["view_id"].values,
        "model_id": df["model_id"].values,
        "true_category": df["category"].values,
        "query_category": df[query_col].values,
        "is_target": is_target,
        "detected": detected_col(df).values,
        "iou": df[iou_col].fillna(0).astype(float).values if iou_col in df else 0.0,
        "score": df["best_score"].values if "best_score" in df else np.nan,
        "g1": df[g1].values,
        "g2": df[g2].values,
    })
    return out


def load_main(root):
    folder = root / "data/renders"
    tl, fl = load_many(folder, "inference_log"), load_many(folder, "false_positive_inference_log")
    if tl is None or fl is None:
        return None, None
    tl, fl = drop_errors(tl, "main/true"), drop_errors(fl, "main/false")
    rows = pd.concat([
        norm(tl, "main", True, "category", "iou_with_gt", "elevation_deg", "azimuth_deg"),
        norm(fl, "main", False, "false_category", "iou_with_true_object", "elevation_deg", "azimuth_deg"),
    ], ignore_index=True)
    return rows, ("elevation (deg)", "azimuth (deg)")


def load_xy(root):
    base = root / "data/renders_framing_falsepositive"
    if not base.exists():
        return None, None
    parts, missing = [], []
    for grid in sorted(p for p in base.iterdir() if p.is_dir()):
        man_p = grid / "framing_manifest.jsonl"
        fl, tl = load_many(grid, "framing_inference_log"), load_many(grid, "framing_true_inference_log")
        if not man_p.exists() or fl is None:
            continue
        if tl is None:
            missing.append(grid.name)
            continue
        man = load_jsonl(man_p)[["view_id", "offset_x_deg", "offset_y_deg"]]
        for df, tgt in [(tl, True), (fl, False)]:
            df = drop_errors(df, f"xy/{grid.name}/{'true' if tgt else 'false'}")
            df = df.drop(columns=[c for c in ("offset_x_deg", "offset_y_deg") if c in df.columns])
            df = df.merge(man, on="view_id", how="left")
            df["offset_x_deg"] = df["offset_x_deg"].round(1)
            df["offset_y_deg"] = df["offset_y_deg"].round(1)
            df["view_id"] = grid.name + ":" + df["view_id"]      # grids reuse view_ids
            r = norm(df, "xy", tgt, "category" if tgt else "false_category",
                     "iou_with_gt" if tgt else "iou_with_true_object", "offset_y_deg", "offset_x_deg")
            r["grid"] = grid.name
            parts.append(r)
    if missing:
        try:
            rel = base.resolve().relative_to(Path.cwd().resolve())     # paths as typed from the cwd
        except ValueError:
            rel = base
        server = " ".join(SERVER_MODELS)
        print(f"  [xy] no true-prompt log for grids {missing}. Generate them (images are already rendered):")
        print("   # the 4 server models (vps-orchestrator, servers running):")
        for g in missing:
            print(f"    python orchestrator/run_inference.py --models {server} \\\n"
                  f"        --manifest {rel}/{g}/framing_manifest.jsonl \\\n"
                  f"        --output {rel}/{g}/framing_true_inference_log.jsonl")
        print("   # the large VLMs (each in its own env), right word then wrong word:")
        for g in missing:
            print(f"    python run_vlm_inference.py --model locate_anything --manifest {rel}/{g}/framing_manifest.jsonl")
            print(f"    python run_vlm_inference.py --model locate_anything --manifest {rel}/{g}/framing_manifest.jsonl "
                  "--false-positive --false-map v1")
        print("    (repeat the VLM lines with --model rex_omni / gemini / gpt; to compare the server models "
              "like-for-like, also re-run their --false-positive pass with all 4 models on each grid)")
    if not parts:
        return None, None
    return pd.concat(parts, ignore_index=True), ("Y offset (deg)", "X offset (deg)")


def load_bench(root):
    base = root / "data/renders_benchmark"
    man_p = base / "query_manifest.jsonl"
    log = load_many(base, "benchmark_inference_log")
    if not man_p.exists() or log is None:
        return None, None
    man = load_jsonl(man_p)[["view_id", "base_view_id", "true_category", "query_category", "is_target",
                             "model_id", "elevation_deg", "azimuth_deg"]]
    df = drop_errors(log, "bench")
    df = df[["view_id", "model"] + [c for c in ("num_detections", "best_box_xyxy", "iou_with_gt",
                                                 "best_score") if c in df.columns]].merge(man, on="view_id")
    out = pd.DataFrame({
        "experiment": "bench", "model": df["model"], "image_id": df["base_view_id"],
        "model_id": df["model_id"], "true_category": df["true_category"],
        "query_category": df["query_category"], "is_target": df["is_target"].astype(bool),
        "detected": detected_col(df), "iou": df.get("iou_with_gt", 0).fillna(0).astype(float),
        "score": df.get("best_score", np.nan), "g1": df["elevation_deg"], "g2": df["azimuth_deg"],
    })
    return out, ("elevation (deg)", "azimuth (deg)")


# ---------------------------------------------------------------------- metrics
def outcomes(rows, iou_t):
    r = rows
    hit = r["detected"] & (r["iou"] >= iou_t)
    r["TP"] = r["is_target"] & hit
    r["FN"] = r["is_target"] & ~hit
    r["FP"] = ~r["is_target"] & r["detected"]
    r["TN"] = ~r["is_target"] & ~r["detected"]
    r["mislocalized"] = r["is_target"] & r["detected"] & (r["iou"] < iou_t)
    r["FP_on_object"] = r["FP"] & (r["iou"] >= iou_t)      # wrong name, right object
    r["TP_presence"] = r["is_target"] & r["detected"]
    r["FN_presence"] = r["is_target"] & ~r["detected"]
    return r


def metrics(c, tp="TP", fn="FN"):
    TP, FP, FN, TN = (c[tp].astype(float), c["FP"].astype(float), c[fn].astype(float), c["TN"].astype(float))
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = TP / (TP + FP)
        rec = TP / (TP + FN)
        f1 = 2 * TP / (2 * TP + FP + FN)
        spec = TN / (TN + FP)
        mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return pd.DataFrame({"precision": prec, "recall": rec, "f1": f1, "specificity": spec,
                         "fpr": 1 - spec, "balanced_acc": (rec + spec) / 2, "mcc": mcc})


COUNT_COLS = ["TP", "FP", "FN", "TN", "mislocalized", "FP_on_object", "TP_presence", "FN_presence"]


def table(rows, keys):
    c = rows.groupby(keys)[COUNT_COLS].sum().astype(int).reset_index()
    m = metrics(c)
    mp = metrics(c, "TP_presence", "FN_presence")[["precision", "recall", "f1"]].add_suffix("_presence")
    return pd.concat([c, m, mp], axis=1)


def bootstrap(rows, n_boot, seed=0):
    """95% CI for precision/recall/F1 per model, resampling OBJECT INSTANCES (views are correlated)."""
    rng = np.random.default_rng(seed)
    out = []
    for m, d in rows.groupby("model"):
        per = d.groupby("model_id")[["TP", "FP", "FN"]].sum().to_numpy(float)
        idx = rng.integers(0, len(per), size=(n_boot, len(per)))
        s = per[idx].sum(axis=1)
        TP, FP, FN = s[:, 0], s[:, 1], s[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            vals = {"precision": TP / (TP + FP), "recall": TP / (TP + FN), "f1": 2 * TP / (2 * TP + FP + FN)}
        row = {"model": m, "n_instances": len(per)}
        for k, v in vals.items():
            lo, hi = np.nanpercentile(v, [2.5, 97.5])
            row[f"{k}_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
        out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------- plotting
def _lbl(v):
    return f"{v:.0f}" if isinstance(v, (int, float, np.number)) else str(v)


def heatmap(ax, piv, title, xlabel, ylabel, vmax=1.0, fmt="{:.2f}"):
    vals = piv.values.astype(float)
    ax.imshow(vals, cmap=SEQ, vmin=0, vmax=vmax, origin="lower", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([_lbl(v) for v in piv.columns], rotation=90, fontsize=6, color=MUTED)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([_lbl(v) for v in piv.index], fontsize=6, color=MUTED)
    ax.set_xlabel(xlabel, fontsize=7, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=7, color=MUTED)
    ax.set_title(title, fontsize=9, color=INK)
    fs = 5 if vals.shape[1] > 10 else 6
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs,
                        color="white" if v > 0.55 * vmax else INK)
    for s in ax.spines.values():
        s.set_visible(False)


def ordered(models):
    return [m for m in MODEL_ORDER if m in models] + sorted(m for m in models if m not in MODEL_ORDER)


# ----------------------------------------------------------------------- main
def drop_ambiguous(rows, pairs):
    """Remove non-target queries whose word is a synonym/hypernym of the real object
    (e.g. 'table' on a desk): a detection there is not clearly wrong, so it is neither FP nor TN."""
    if not pairs:
        return rows
    amb = {(a, b) for a, b in pairs} | {(b, a) for a, b in pairs}
    mask = (~rows["is_target"]) & pd.Series(list(zip(rows["true_category"], rows["query_category"])),
                                            index=rows.index).isin(amb)
    if mask.any():
        print(f"  ignoring {int(mask.sum())} non-target queries on ambiguous pairs {sorted(pairs)}")
    return rows[~mask].copy()


def apply_score_threshold(rows, thr):
    """Re-score detections at a stricter score threshold (per model). Only thresholds ABOVE the one
    used at inference time can be applied: weaker detections were never logged."""
    if not thr:
        return rows
    t = rows["model"].map(lambda m: thr.get(m, thr.get("*", 0.0)))
    rows = rows.copy()
    rows["detected"] = rows["detected"] & (rows["score"].fillna(0).astype(float) >= t)
    return rows


def threshold_sweep(rows, out, name, iou_t):
    """Recall / FPR / F1 / MCC per model as the score threshold is raised from the logged minimum."""
    sc = rows.loc[rows["detected"], "score"].dropna().astype(float)
    if sc.empty:
        return
    ts = np.round(np.arange(np.floor(sc.min() * 20) / 20, 0.951, 0.05), 2)
    parts = []
    for t in ts:
        r = outcomes(apply_score_threshold(rows, {"*": t}), iou_t)
        tb = table(r, ["model"])
        tb.insert(1, "score_threshold", t)
        parts.append(tb)
    sw = pd.concat(parts, ignore_index=True)
    sw.to_csv(out / "threshold_sweep.csv", index=False)
    models = ordered(sw["model"].unique())
    colors = MODEL_COLORS
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for m in models:
        d = sw[sw["model"] == m]
        c = colors.get(m, "#888888")
        axes[0].plot(d["fpr"], d["recall"], color=c, linewidth=2, marker="o", markersize=4, label=m)
        axes[1].plot(d["score_threshold"], d["f1"], color=c, linewidth=2, marker="o", markersize=4, label=m)
    axes[0].set_xlabel("false positive rate", fontsize=8, color=MUTED)
    axes[0].set_ylabel("recall", fontsize=8, color=MUTED)
    axes[0].set_title("recall vs FPR as the threshold rises (up-right = lower threshold)",
                      fontsize=9, color=INK, loc="left")
    axes[1].set_xlabel("score threshold", fontsize=8, color=MUTED)
    axes[1].set_ylabel("F1", fontsize=8, color=MUTED)
    axes[1].set_title("F1 vs score threshold", fontsize=9, color=INK, loc="left")
    for ax in axes:
        ax.set_xlim(left=0); ax.set_ylim(0, 1)
        ax.grid(color="#ececec"); ax.legend(frameon=False, fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"[{name}] threshold sweep (starts at the inference threshold; lower needs a re-run)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "threshold_sweep.png", dpi=140)
    plt.close(fig)
    best = sw.loc[sw.groupby("model")["f1"].idxmax(), ["model", "score_threshold", "precision", "recall",
                                                          "f1", "fpr", "mcc"]]
    print("  best-F1 threshold per model (within the logged range):")
    print(best.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def run_experiment(name, rows, labels, out_root, iou_t, n_boot, ignore_pairs=(), score_thr=None):
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    rows = drop_ambiguous(rows, ignore_pairs)
    raw_rows = rows
    rows = outcomes(apply_score_threshold(rows, score_thr), iou_t)
    ylab, xlab = labels
    models = ordered(rows["model"].unique())
    pos, neg = int(rows["is_target"].sum()), int((~rows["is_target"]).sum())
    no_pos = sorted(set(rows["model"]) - set(rows.loc[rows["is_target"], "model"]))
    no_neg = sorted(set(rows["model"]) - set(rows.loc[~rows["is_target"], "model"]))
    if no_pos or no_neg:
        print(f"\n  [{name}] WARNING: incomplete logs, so recall/F1/MCC are NOT meaningful for:")
        if no_pos:
            print(f"     no right-word (target) results: {no_pos}  -> TP/FN missing, recall & F1 undefined")
        if no_neg:
            print(f"     no wrong-word (non-target) results: {no_neg}  -> FP/TN missing, FPR & precision undefined")

    overall = table(rows, ["model"]).merge(bootstrap(rows, n_boot), on="model")
    overall["model"] = pd.Categorical(overall["model"], models, ordered=True)
    overall = overall.sort_values("model")
    overall.to_csv(out / "overall.csv", index=False)
    table(rows, ["model", "query_category"]).to_csv(out / "by_word.csv", index=False)
    by_view = table(rows, ["model", "g1", "g2"]).rename(columns={"g1": ylab, "g2": xlab})
    by_view.to_csv(out / "by_view.csv", index=False)
    per_img = table(rows, ["model", "image_id", "model_id", "true_category", "g1", "g2"]).rename(
        columns={"g1": ylab, "g2": xlab})
    per_img.to_csv(out / "per_image.csv", index=False)

    for m in models:
        d = by_view[by_view["model"] == m]
        fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
        for ax, k in zip(axes, ["precision", "recall", "f1", "fpr"]):
            heatmap(ax, d.pivot_table(index=ylab, columns=xlab, values=k), f"{m}: {k}", xlab, ylab)
        fig.suptitle(f"[{name}] {m}: per-view metrics (box scoring, IoU >= {iou_t})",
                     fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / f"heatmaps_{m}.png", dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(1, len(models), figsize=(5.4 * len(models), 4.6), squeeze=False)
    for ax, m in zip(axes[0], models):
        d = by_view[by_view["model"] == m]
        heatmap(ax, d.pivot_table(index=ylab, columns=xlab, values="f1"), f"{m}: F1", xlab, ylab)
    fig.suptitle(f"[{name}] F1 per view", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "f1_all_models.png", dpi=140)
    plt.close(fig)

    show = ["model", "TP", "FP", "FN", "TN", "precision", "recall", "f1", "fpr", "mcc",
            "f1_presence", "f1_ci95", "n_instances"]
    print(f"\n=== {name}: {pos} target / {neg} non-target queries (1:{neg / max(pos, 1):.1f}) ===")
    print(overall[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"  mislocalized target hits (counted as FN): "
          + ", ".join(f"{m}={int(v)}" for m, v in overall.set_index('model')["mislocalized"].items()))
    print(f"  FPs whose box sits on the real object: "
          + ", ".join(f"{m}={int(v)}" for m, v in overall.set_index('model')["FP_on_object"].items()))
    f1v = by_view.pivot_table(index="model", columns=ylab, values="f1").reindex(models)
    print(f"  mean F1 by {ylab}:")
    print(f1v.round(2).to_string())
    threshold_sweep(raw_rows, out, name, iou_t)
    return overall


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--iou", type=float, default=0.5, help="IoU needed for a target detection to be a TP")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples for the CIs")
    ap.add_argument("--score-threshold", nargs="*", default=[], metavar="[MODEL=]T",
                    help="re-score at a stricter threshold, globally (0.5) or per model (owlv2=0.4); "
                         "only values above the inference threshold have any effect")
    ap.add_argument("--ignore-pairs", nargs="*", default=["desk:table"], metavar="A:B",
                    help="word pairs too ambiguous to call a false positive, either direction "
                         "(default desk:table). Pass --ignore-pairs with nothing to count every pair")
    args = ap.parse_args()
    out = args.out or args.root / "analysis_confusion"

    found = False
    for name, loader in [("main", load_main), ("xy", load_xy), ("bench", load_bench)]:
        rows, labels = loader(args.root)
        if rows is None:
            print(f"\n[{name}] logs not found, skipped")
            continue
        found = True
        pairs = [tuple(p.split(":", 1)) for p in args.ignore_pairs]
        thr = {}
        for item in args.score_threshold:
            k, _, v = item.rpartition("=")
            thr[k or "*"] = float(v)
        run_experiment(name, rows, labels, out, args.iou, args.boot, pairs, thr)
    if found:
        print(f"\nwrote {out}/")
        print("note: precision depends on the target:non-target ratio printed per experiment, so compare "
              "precision only within an experiment; recall, FPR and MCC are ratio-independent.")


if __name__ == "__main__":
    main()
