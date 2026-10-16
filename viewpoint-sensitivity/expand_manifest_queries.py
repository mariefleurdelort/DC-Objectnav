#!/usr/bin/env python
"""Expand a render manifest into one row per (view x prompted word) for the
full benchmark: every image is queried with EVERY category word, so each image
yields 1 target query (the word matches the object -> TP or FN) and N-1
non-target queries (any detection -> FP, none -> TN).

The output is an ordinary manifest, so the existing inference runner scores it
unchanged (run it in TRUE-prompt mode, i.e. without --false-positive):

    python expand_manifest_queries.py \
        --manifest data/renders/manifest.jsonl \
        --out data/renders_benchmark/query_manifest.jsonl
    python orchestrator/run_inference.py \
        --manifest data/renders_benchmark/query_manifest.jsonl \
        --output data/renders_benchmark/benchmark_inference_log.jsonl
    python confusion_metrics.py

Each expanded row keeps rgb_path / gt_bbox_xyxy of the real object, so the
runner's IoU is always measured against the object that is actually in the
image. `prompt` AND `category` are set to the queried word (so the prompt is
right whichever field the runner reads); the real object is kept in
`true_category`. view_id gets a "|q=<word>" suffix so resumable runners key
each query separately.

CHECK BEFORE A BIG RUN: run the runner on the first ~20 rows (--limit-views 3)
and confirm the logged "prompt" equals each row's queried word.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def category_prompts(manifest_rows, cli_categories):
    """Prefer rendering/config.py's fixed prompts; fall back to 'a <word>'."""
    try:
        sys.path.insert(0, str(ROOT / "rendering"))
        from config import CATEGORIES, CATEGORY_PROMPTS  # noqa: E402
        cats = cli_categories or CATEGORIES
        return {c: CATEGORY_PROMPTS.get(c, f"a {c}") for c in cats}
    except Exception as e:  # config not importable from here
        if not cli_categories:
            sys.exit(f"could not import rendering/config.py ({type(e).__name__}: {e}), so the word list is "
                     "unknown. Either install the missing package in this env, or pass the words explicitly:\n"
                     "  --categories bed bench bookcase cabinet chair desk lamp sofa stool table toilet vase")
        print(f"  (rendering/config.py not importable: {e}; using --categories with 'a <word>' prompts)")
        return {c: f"a {c}" for c in cli_categories}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/renders/manifest.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data/renders_benchmark/query_manifest.jsonl")
    ap.add_argument("--alias", nargs="*", default=["bookshelf=bookcase"], metavar="OBJECT=WORD",
                    help="rendered category -> word that counts as its right answer when the names differ "
                         "(default bookshelf=bookcase, for the old renders vs the 12-word list)")
    ap.add_argument("--categories", nargs="+", default=None,
                    help="word list to query (default: config.CATEGORIES). Avoid synonyms/hypernyms of "
                         "each other (e.g. sofa+couch), or every image gets a 'false positive' for free")
    ap.add_argument("--limit-views", type=int, default=0, help="only the first N views (for a smoke test)")
    ap.add_argument("--instances-per-category", type=int, default=0,
                    help="subsample N object instances per category (0 = all); for expensive VLMs")
    ap.add_argument("--azimuth-step", type=int, default=0,
                    help="keep only views whose azimuth is a multiple of this (e.g. 60 keeps 0,60,...,300)")
    ap.add_argument("--elevations", type=int, nargs="+", default=None,
                    help="keep only these elevations (default: all rendered)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.instances_per_category:
        keep = {}
        for r in rows:
            ids = keep.setdefault(r["category"], [])
            if r["model_id"] not in ids and len(ids) < args.instances_per_category:
                ids.append(r["model_id"])
        allowed = {i for ids in keep.values() for i in ids}
        rows = [r for r in rows if r["model_id"] in allowed]
    if args.azimuth_step:
        rows = [r for r in rows if round(r.get("azimuth_deg", 0)) % args.azimuth_step == 0]
    if args.elevations:
        rows = [r for r in rows if round(r.get("elevation_deg", 0)) in set(args.elevations)]
    if args.limit_views:
        rows = rows[:args.limit_views]

    prompts = category_prompts(rows, args.categories)
    alias = dict(a.split("=", 1) for a in args.alias)
    own_word = lambda cat: alias.get(cat, cat)
    used = {c: own_word(c) for c in sorted({r["category"] for r in rows})}
    for c, w in used.items():
        if w != c and w in prompts:
            print(f"  {c} images: right word is '{w}'")
    missing = sorted(c for c, w in used.items() if w not in prompts)
    if missing:
        print(f"  NOTE: {missing} images have no matching word in the list: every word is a wrong word for them "
              "(FP/TN only, no recall)")
    absent = sorted(set(prompts) - set(used.values()))
    if absent:
        print(f"  NOTE: no images of {absent} in these renders: those words are only asked as wrong words")

    # make image paths absolute (renders may have been moved, e.g. into v1/)
    data_root = args.manifest.resolve().parents[2] if len(args.manifest.resolve().parents) > 2 else ROOT
    for r in rows:
        p = Path(r["rgb_path"])
        if not p.is_absolute():
            for cand in (ROOT / p, data_root / p):
                if cand.exists():
                    r["rgb_path"] = str(cand.resolve())
                    break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for r in rows:
            for word, prompt in prompts.items():
                q = dict(r)
                q["base_view_id"] = r["view_id"]
                q["view_id"] = f"{r['view_id']}|q={word}"
                q["true_category"] = r["category"]
                q["query_category"] = word
                q["is_target"] = word == own_word(r["category"])
                q["category"] = word
                q["prompt"] = prompt
                f.write(json.dumps(q) + "\n")
                n += 1
    n_obj = len({r["model_id"] for r in rows})
    print(f"{n_obj} objects, {len(rows)} views x {len(prompts)} words = {n} queries -> {args.out}")
    print(f"at ~1 query/s (LocateAnything) that is ~{n / 3600:.1f} h per model")
    print(f"per image: 1 right word (if its object is in the list), {len(prompts) - 1} wrong words "
          f"(positives:negatives = 1:{len(prompts) - 1}; precision depends on this ratio)")


if __name__ == "__main__":
    main()
