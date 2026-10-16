#!/usr/bin/env python
"""Paper-ready results table: precision, recall, F1, false-positive rate (FPR) and
false-negative rate (FNR) per model on the main viewpoint sweep, plus how much each
metric moves across camera elevation (viewpoint sensitivity).

    python results_table.py --root v1                    # default 6 models, main viewpoint sweep
    python results_table.py --root . --experiment bench  # the 12-word benchmark
    python results_table.py --root v1 --models sam3 clip yoloworld owlv2 locate_anything rex_omni groundingdino

Writes to <root>/results/<experiment>/:
  results_table.csv / .md / .tex     one row per model (the table for the paper)
  per_elevation.csv / .md            F1, FPR, FNR at every elevation, per model
  viewpoint_sensitivity.png          F1 / FPR / FNR vs elevation, one line per model
  (bench only) word_confusion_<model>.csv, word_confusion_top5.csv, word_confusion.png
                                     which wrong words each model accepts for which objects

Definitions (one query = one image + one word; see confusion_metrics.py):
  right word -> TP (found it) or FN (missed it);  wrong word -> FP (fooled) or TN
  precision = TP/(TP+FP)   recall = TP/(TP+FN)   F1 = 2TP/(2TP+FP+FN)
  F0.5 = 1.25TP/(1.25TP+FP+0.25FN)  (like F1 but a false positive costs 4x a false negative)
  FPR = FP/(FP+TN)  (how often a wrong word is accepted)
  FNR = FN/(TP+FN) = 1 - recall  (how often the real object is missed)
  Scoring = PRESENCE (did the model say the object is there, box ignored), so CLIP, which has
  no boxes, is comparable. Box-based F1 (IoU >= 0.5) is reported next to it for the detectors.
  Viewpoint sensitivity = spread of the metric across elevations (max - min), with the
  elevation where the model does worst.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import confusion_metrics as cm

DEFAULT_MODELS = ["sam3", "clip", "yoloworld", "owlv2", "locate_anything", "rex_omni"]
NAMES = {"groundingdino": "GroundingDINO", "owlv2": "OWLv2", "sam3": "SAM3", "yoloworld": "YOLO-World",
         "locate_anything": "LocateAnything", "rex_omni": "Rex-Omni", "clip": "CLIP",
         "gpt_astra": "GPT-6 Astra", "gemini": "Gemini", "gpt": "GPT"}
NO_BOXES = {"clip"}                     # classifiers: box-based scores are not defined


def rates(c):
    """precision / recall / F1 / FPR / FNR / MCC from TP FP FN TN columns."""
    TP, FP, FN, TN = (c[k].astype(float) for k in ("TP", "FP", "FN", "TN"))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame({
            "precision": TP / (TP + FP), "recall": TP / (TP + FN), "f1": 2 * TP / (2 * TP + FP + FN),
            "f05": 1.25 * TP / (1.25 * TP + FP + 0.25 * FN),   # F0.5: false positives weigh 4x false negatives
            "fpr": FP / (FP + TN), "fnr": FN / (TP + FN),
            "mcc": (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))})
    return out


def presence(rows):
    r = rows.copy()
    r["TP"], r["FN"] = r["TP_presence"], r["FN_presence"]
    return r


def ci_f1(rows, n_boot=1000, seed=0):
    """95% CI for F1, resampling object instances (views of one object are correlated)."""
    rng = np.random.default_rng(seed)
    per = rows.groupby("model_id")[["TP", "FP", "FN"]].sum().to_numpy(float)
    if len(per) < 2:
        return np.nan, np.nan
    s = per[rng.integers(0, len(per), size=(n_boot, len(per)))].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * s[:, 0] / (2 * s[:, 0] + s[:, 1] + s[:, 2])
    return tuple(np.nanpercentile(f1, [2.5, 97.5]))


def fmt(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{100 * v:.1f}%" if pct else f"{v:.3f}"


def write_md(df, path, title):
    cols = list(df.columns)
    lines = [f"**{title}**", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in df.itertuples(index=False)]
    path.write_text("\n".join(lines) + "\n")


def write_tex(df, path, caption):
    cols = list(df.columns)
    esc = lambda x: str(x).replace("%", r"\%").replace("_", r"\_")
    body = [r"\begin{table}[t]", r"\centering", r"\small",
            r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\toprule",
            " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    body += [" & ".join(esc(x) for x in row) + r" \\" for row in df.itertuples(index=False)]
    body += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{esc(caption)}}}", r"\end{table}"]
    path.write_text("\n".join(body) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("v1"))
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--experiment", choices=["main", "bench"], default="main",
                    help="main = viewpoint sweep (1 right + 1 wrong word per image); "
                         "bench = every image asked all words (1 right + N-1 wrong)")
    ap.add_argument("--ignore-pairs", nargs="*", default=["desk:table"], metavar="A:B",
                    help="word pairs too ambiguous to count as false positives (either direction)")
    args = ap.parse_args()
    out = args.root / "results" / args.experiment
    out.mkdir(parents=True, exist_ok=True)

    loader = cm.load_main if args.experiment == "main" else cm.load_bench
    rows, _ = loader(args.root)
    if rows is None:
        raise SystemExit(f"no {args.experiment} logs under {args.root}/data/")
    rows = cm.drop_ambiguous(rows, [tuple(p.split(":", 1)) for p in args.ignore_pairs])
    rows = cm.outcomes(rows, args.iou)
    if args.experiment == "bench":
        print("bench: each image gets 1 right word and N-1 wrong words, so precision is lower than in the "
              "main sweep by construction; compare FPR / FNR / MCC across experiments, not precision")
    have = set(rows["model"])
    missing = [m for m in args.models if m not in have]
    models = [m for m in args.models if m in have]
    if missing:
        print(f"not in the logs yet (run their inference first): {missing}")
    incomplete = [m for m in models if rows.loc[rows["model"] == m, "is_target"].nunique() < 2]
    if incomplete:
        print(f"only ONE of the two passes is logged for {incomplete}: their F1/FPR/FNR are not valid yet")
    rows = rows[rows["model"].isin(models)]
    pres = presence(rows)

    # ---- main table -------------------------------------------------------------
    recs = []
    for m in models:
        d, db = pres[pres["model"] == m], rows[rows["model"] == m]
        c = d[["TP", "FP", "FN", "TN"]].sum().to_frame().T
        r = rates(c).iloc[0]
        rb = rates(db[["TP", "FP", "FN", "TN"]].sum().to_frame().T).iloc[0]
        lo, hi = ci_f1(d, args.boot)
        el = rates(d.groupby("g1")[["TP", "FP", "FN", "TN"]].sum()).set_index(
            d.groupby("g1").size().index)
        recs.append({
            "model": NAMES.get(m, m), "_key": m,
            "images": int(c["TP"].iloc[0] + c["FN"].iloc[0]),
            "precision": r["precision"], "recall": r["recall"], "f1": r["f1"], "f1_lo": lo, "f1_hi": hi,
            "fpr": r["fpr"], "fnr": r["fnr"], "mcc": r["mcc"], "f05": r["f05"],
            "f1_box": np.nan if m in NO_BOXES else rb["f1"],
            "f1_range": el["f1"].max() - el["f1"].min(),
            "worst_elev": el["f1"].idxmin() if el["f1"].notna().any() else np.nan,
            "fpr_range": el["fpr"].max() - el["fpr"].min(),
            "worst_fpr_elev": el["fpr"].idxmax() if el["fpr"].notna().any() else np.nan,
            "fnr_range": el["fnr"].max() - el["fnr"].min(),
        })
    T = pd.DataFrame(recs)
    T.drop(columns="_key").to_csv(out / "results_table.csv", index=False)
    show = pd.DataFrame({
        "Model": T["model"], "Images": T["images"],
        "Precision": T["precision"].map(fmt), "Recall": T["recall"].map(fmt),
        "F1 [95% CI]": [f"{fmt(f)} [{fmt(l)}, {fmt(h)}]" for f, l, h in zip(T["f1"], T["f1_lo"], T["f1_hi"])],
        "FPR": T["fpr"].map(lambda v: fmt(v, True)), "FNR": T["fnr"].map(lambda v: fmt(v, True)),
        "F0.5 (FP-weighted)": T["f05"].map(fmt),
        "MCC": T["mcc"].map(fmt), "F1 (box)": T["f1_box"].map(fmt),
        "F1 range over elev.": T["f1_range"].map(fmt),
        "Worst elev.": T["worst_elev"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.0f}°"),
        "FPR range": T["fpr_range"].map(lambda v: fmt(v, True)),
        "Worst FPR elev.": T["worst_fpr_elev"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.0f}°"),
    })
    title = ("Presence-scored detection on the "
             + ("viewpoint sweep (1 right + 1 wrong word per image)" if args.experiment == "main"
                else "12-word benchmark (every image asked every word)")
             + "; right word: TP/FN, wrong word: FP/TN")
    write_md(show, out / "results_table.md", title)
    write_tex(show, out / "results_table.tex", title + ". F1 CI: bootstrap over object instances. "
              "Ranges: max minus min across camera elevations.")
    print("\n" + title)
    print(show.to_string(index=False))

    # ---- per elevation ----------------------------------------------------------
    pe = []
    for m in models:
        d = pres[pres["model"] == m]
        g = d.groupby("g1")[["TP", "FP", "FN", "TN"]].sum()
        r = rates(g)
        r.index = g.index
        for elev, rr in r.iterrows():
            pe.append({"model": NAMES.get(m, m), "_key": m, "elevation": elev, "f1": rr["f1"],
                       "precision": rr["precision"], "fpr": rr["fpr"], "fnr": rr["fnr"]})
    PE = pd.DataFrame(pe)
    PE.drop(columns="_key").to_csv(out / "per_elevation.csv", index=False)
    piv = PE.pivot_table(index="model", columns="elevation", values="f1").reindex(
        [NAMES.get(m, m) for m in models])
    write_md(piv.map(fmt).reset_index().rename(columns=lambda c: f"{c}°" if c != "model" else "Model"),
             out / "per_elevation.md", "F1 by camera elevation")
    print("\nF1 by camera elevation:")
    print(piv.round(3).to_string())
    pf = (PE.pivot_table(index="model", columns="elevation", values="fpr") * 100).reindex(
        [NAMES.get(m, m) for m in models])
    write_md(pf.map(lambda v: "n/a" if pd.isna(v) else f"{v:.1f}%").reset_index().rename(
        columns=lambda c: f"{c}°" if c != "model" else "Model"), out / "per_elevation_fpr.md",
        "False-positive rate (%) by camera elevation")
    print("\nFalse-positive rate (%) by camera elevation:")
    print(pf.round(1).to_string())

    # ---- bench only: which wrong words each model accepts for which objects ---------
    if args.experiment == "bench":
        neg = pres[~pres["is_target"]]
        mats = {}
        for m in models:
            d = neg[neg["model"] == m]
            mat = (d.groupby(["true_category", "query_category"])["FP"].mean().unstack() * 100).round(1)
            mats[m] = mat
            mat.to_csv(out / f"word_confusion_{m}.csv")
        long = pd.concat([v.stack().dropna().rename("fpr_pct").reset_index().assign(model=NAMES.get(k, k))
                          for k, v in mats.items()], ignore_index=True)
        top = long.sort_values("fpr_pct", ascending=False).groupby("model").head(5)
        top.to_csv(out / "word_confusion_top5.csv", index=False)
        print("\nTop wrong-word confusions per model (% of images of that object where the wrong word was accepted):")
        for name, g in top.groupby("model", sort=False):
            print(f"  {name:15s} " + ", ".join(f"{r.true_category}->'{r.query_category}' {r.fpr_pct:.0f}%"
                                               for r in g.itertuples()))
        n = len(mats)
        fig, axes = plt.subplots(1, n, figsize=(max(7.5, 5.2 * n), 4.4), squeeze=False)
        for ax, (m, mat) in zip(axes[0], mats.items()):
            cm.heatmap(ax, mat / 100.0, f"{NAMES.get(m, m)}: wrong word accepted", "word asked",
                       "object in the image", vmax=1.0, fmt="{:.2f}")
            ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=7, color=cm.MUTED)
            ax.set_yticklabels(mat.index, fontsize=7, color=cm.MUTED)
        fig.suptitle("Wrong words accepted, per object (fraction of images)", fontsize=11,
                     color=cm.INK, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / "word_confusion.png", dpi=140)
        plt.close(fig)

    # ---- figure: F1 / FPR / FNR vs elevation -------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, k, lab in zip(axes, ["f1", "fpr", "fnr"],
                          ["F1 (higher is better)", "False-positive rate (lower is better)",
                           "False-negative rate (lower is better)"]):
        for m in models:
            d = PE[PE["_key"] == m].sort_values("elevation")
            col = cm.MODEL_COLORS.get(m, "#8a8a8a")
            ax.plot(d["elevation"], d[k], color=col, linewidth=2, marker="o", markersize=6,
                    markeredgecolor="white", markeredgewidth=1.5, label=NAMES.get(m, m))
        ax.set_title(lab, fontsize=10, color=cm.INK, loc="left")
        ax.set_xlabel("camera elevation (deg)", fontsize=8, color=cm.MUTED)
        ax.set_ylim(0, 1)
        ax.grid(color="#ececec")
        ax.tick_params(colors=cm.MUTED, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Viewpoint sensitivity: how each model's errors change with camera elevation",
                 fontsize=11, color=cm.INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "viewpoint_sensitivity.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}/ (results_table.csv/.md/.tex, per_elevation.csv/.md, viewpoint_sensitivity.png)")


if __name__ == "__main__":
    main()
