#!/usr/bin/env python
"""Pull a random sample of a model's mistakes and put the images on one contact sheet, so you
can LOOK at what the numbers mean (is it a real confusion, a bad render, a scoring problem?).

    python sample_errors.py --root v1                          # main sweep, false positives
    python sample_errors.py --root . --experiment bench        # 12-word benchmark
    python sample_errors.py --root v1 --kind fn                # missed objects instead
    python sample_errors.py --root . --experiment bench --model owlv2 --word stool -n 24
    python sample_errors.py --root v1 --experiment bench --model rex_omni --only persistent
    python sample_errors.py --root v1 --experiment bench --model locate_anything --only fixable

--only persistent  false positives where that object is fooled by that word from >= 70% of its views
                   (the ones no viewpoint fixes, same definition as fp_resolution.py)
--only fixable     the other false positives (fooled from only some viewpoints)

Writes <root>/results/<experiment>/samples/<kind>[_filters]_<model>.png (+ .csv listing every sampled
image); the filters are in the file name, so different runs do not overwrite each other.
Captions: object -> word asked, elevation/azimuth, and the model's score.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import confusion_metrics as cm


def image_paths(root, experiment):
    man = root / ("data/renders/manifest.jsonl" if experiment == "main"
                  else "data/renders_benchmark/query_manifest.jsonl")
    paths = {}
    for line in open(man):
        if line.strip():
            r = json.loads(line)
            paths.setdefault(r.get("base_view_id", r["view_id"]), r["rgb_path"])
    return paths


def resolve(p, root):
    p = Path(p)
    for cand in (p, root / p, root.parent / p, Path.cwd() / p, root / Path(*p.parts[1:]) if len(p.parts) > 1 else p):
        if cand.exists():
            return cand
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("v1"))
    ap.add_argument("--experiment", choices=["main", "bench"], default="main")
    ap.add_argument("--kind", choices=["fp", "fn", "tp"], default="fp",
                    help="fp = wrong word accepted, fn = object missed, tp = correct finds (for contrast)")
    ap.add_argument("--model", nargs="+", default=None, help="default: every model in the logs")
    ap.add_argument("--word", default=None, help="only queries for this word")
    ap.add_argument("--object", default=None, help="only images of this object category")
    ap.add_argument("--elevation", type=int, default=None)
    ap.add_argument("--only", choices=["persistent", "fixable"], default=None,
                    help="false positives only: persistent (no viewpoint fixes) or fixable ones")
    ap.add_argument("--persistent", type=float, default=0.7, help="share of views that makes an FP persistent")
    ap.add_argument("-n", type=int, default=16, help="images per model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ignore-pairs", nargs="*", default=["desk:table"], metavar="A:B")
    args = ap.parse_args()

    rows, _ = (cm.load_main if args.experiment == "main" else cm.load_bench)(args.root)
    if rows is None:
        raise SystemExit(f"no {args.experiment} logs under {args.root}")
    rows = cm.drop_ambiguous(rows, [tuple(p.split(":", 1)) for p in args.ignore_pairs])
    pick = {"fp": ~rows["is_target"] & rows["detected"],
            "fn": rows["is_target"] & ~rows["detected"],
            "tp": rows["is_target"] & rows["detected"]}[args.kind]
    sel = rows[pick]
    if args.only:
        if args.kind != "fp":
            raise SystemExit("--only works with --kind fp")
        neg = rows[~rows["is_target"]].drop_duplicates(["model", "image_id", "query_category"])
        share = neg.groupby(["model", "model_id", "query_category"])["detected"].mean().rename("fp_share")
        sel = sel.join(share, on=["model", "model_id", "query_category"])
        keep = sel["fp_share"] >= args.persistent
        sel = sel[keep if args.only == "persistent" else ~keep]
    if args.word:
        sel = sel[sel["query_category"] == args.word]
    if args.object:
        sel = sel[sel["true_category"] == args.object]
    if args.elevation is not None:
        sel = sel[sel["g1"].round() == args.elevation]
    paths = image_paths(args.root, args.experiment)
    out = args.root / "results" / args.experiment / "samples"
    out.mkdir(parents=True, exist_ok=True)
    tag = args.kind + "".join(f"_{v}" for v in (args.only, args.object, args.word,
                                                 None if args.elevation is None else f"el{args.elevation}") if v)

    for m in args.model or cm.ordered(rows["model"].unique()):
        d = sel[sel["model"] == m]
        total = int(((rows["model"] == m) & pick).sum())
        if d.empty:
            print(f"{m}: no {args.kind.upper()} matching the filters")
            continue
        s = d.sample(min(args.n, len(d)), random_state=args.seed)
        s = s.assign(rgb_path=[paths.get(i) for i in s["image_id"]])
        s.to_csv(out / f"{tag}_{m}.csv", index=False)
        cols = 4
        nrow = -(-len(s) // cols)
        fig, axes = plt.subplots(nrow, cols, figsize=(cols * 3.2, nrow * 3.5), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        missing = 0
        for ax, r in zip(axes.flat, s.itertuples()):
            p = resolve(r.rgb_path, args.root) if r.rgb_path else None
            if p is None:
                missing += 1
                ax.text(0.5, 0.5, "image not found", ha="center", va="center", fontsize=8, color=cm.MUTED)
            else:
                ax.imshow(plt.imread(p))
            score = "" if pd.isna(r.score) else f"  score {float(r.score):.2f}"
            share = f"  fooled {100 * r.fp_share:.0f}% of views" if "fp_share" in s else ""
            ax.set_title(f"{r.true_category} → '{r.query_category}'\nel {r.g1:.0f}° az {r.g2:.0f}°{score}{share}",
                         fontsize=8, color=cm.INK)
        fig.suptitle(f"{m}: {len(s)} random {tag.upper()} of {total} {args.kind.upper()} (filtered pool {len(d)})",
                     fontsize=10, color=cm.INK, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / f"{tag}_{m}.png", dpi=110)
        plt.close(fig)
        note = f" ({missing} images not found: check the rgb_path column in the csv)" if missing else ""
        print(f"{m}: {len(s)} of {len(d)} -> {out}/{tag}_{m}.png{note}")


if __name__ == "__main__":
    main()
