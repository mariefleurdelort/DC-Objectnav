#!/usr/bin/env python
"""How many false positives does a different viewpoint fix?

For every false positive (an image + a wrong word that the model accepted), look at the
OTHER rendered views of the same object asked the same wrong word, and ask: is there a
viewpoint within k degrees of camera movement where the model gets it right?

  resolved (loose)   the wrong word is rejected at that viewpoint (the false positive is gone)
  resolved (strict)  ... AND the model still finds the real object there (right word = TP),
                     i.e. it did not just go blind. This is the main number.

Reported per model, for k = 15, 30, 45, 60, 90, 120, 180 degrees (great-circle angle between
the two camera directions on the viewing sphere):
  best_*    some view within k fixes it (an oracle picking the right move: upper bound)
  random_*  a random move of about k degrees fixes it (views in the ring k-15..k, averaged)
plus the share of PERSISTENT false positives (fooled at >= 70% of that object's views:
a category confusion no viewpoint fixes) and results split by the elevation of the failure.

    python fp_resolution.py --root v1                  # main viewpoint sweep (existing logs)
    python fp_resolution.py --root . --experiment bench   # the 12-word benchmark, once run
    python fp_resolution.py --root v1 --scoring box    # strict = TP with box IoU >= 0.5

Writes <root>/results/fp_resolution_<experiment>.csv / _summary.md / .png.
Caveats to state: the main sweep's grid is 30 deg in azimuth/elevation, so the smallest move is
~15-30 deg; there is no re-render noise floor yet (a jittered re-render of each failure view would
give it).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import confusion_metrics as cm

NAMES = {"groundingdino": "GroundingDINO", "owlv2": "OWLv2", "sam3": "SAM3", "yoloworld": "YOLO-World",
         "locate_anything": "LocateAnything", "rex_omni": "Rex-Omni", "clip": "CLIP",
         "gpt_astra": "GPT-6 Astra", "gemini": "Gemini", "gpt": "GPT"}
KS = [15, 30, 45, 60, 90, 120, 180]


def unit(el, az):
    e, a = np.radians(el), np.radians(az)
    return np.stack([np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)], axis=-1)


def analyse_model(rows, scoring, persistent_frac):
    """rows: one model's rows (both passes). Returns one record per false-positive event."""
    tgt = rows[rows["is_target"]]
    ok_true = tgt["detected"] & ((tgt["iou"] >= 0.5) if scoring == "box" else True)
    true_ok = dict(zip(tgt["image_id"], ok_true))                     # view -> found the object?
    neg = rows[~rows["is_target"]]
    events = []
    for (inst, word), d in neg.groupby(["model_id", "query_category"]):
        d = d.drop_duplicates("image_id")
        if not d["detected"].any():
            continue
        u = unit(d["g1"].to_numpy(float), d["g2"].to_numpy(float))
        ang = np.degrees(np.arccos(np.clip(u @ u.T, -1, 1)))
        fp = d["detected"].to_numpy(bool)
        found = np.array([true_ok.get(v, False) for v in d["image_id"]])
        fixed_loose, fixed_strict = ~fp, ~fp & found
        persistent = fp.mean() >= persistent_frac
        for i in np.where(fp)[0]:
            other = ang[i] > 1.0                  # exclude the same camera direction (e.g. pole views)
            rec = {"model_id": inst, "word": word, "true_category": d["true_category"].iloc[i],
                   "elevation": d["g1"].iloc[i], "azimuth": d["g2"].iloc[i], "persistent": persistent,
                   "fp_share_of_views": fp.mean()}
            for k in KS:
                near = other & (ang[i] <= k)
                ring = other & (ang[i] > k - 15) & (ang[i] <= k)
                rec[f"best_loose_{k}"] = bool((near & fixed_loose).any())
                rec[f"best_strict_{k}"] = bool((near & fixed_strict).any())
                rec[f"random_loose_{k}"] = fixed_loose[ring].mean() if ring.any() else np.nan
                rec[f"random_strict_{k}"] = fixed_strict[ring].mean() if ring.any() else np.nan
            events.append(rec)
    return pd.DataFrame(events)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("v1"))
    ap.add_argument("--experiment", choices=["main", "bench"], default="main")
    ap.add_argument("--models", nargs="+", default=None, help="default: every model with both passes")
    ap.add_argument("--scoring", choices=["presence", "box"], default="presence",
                    help="what 'still finds the object' means for the strict criterion")
    ap.add_argument("--persistent", type=float, default=0.7)
    ap.add_argument("--ignore-pairs", nargs="*", default=["desk:table"], metavar="A:B")
    args = ap.parse_args()
    out = args.root / "results"
    out.mkdir(parents=True, exist_ok=True)

    loader = cm.load_main if args.experiment == "main" else cm.load_bench
    rows, _ = loader(args.root)
    if rows is None:
        raise SystemExit(f"no {args.experiment} logs under {args.root}")
    rows = cm.drop_ambiguous(rows, [tuple(p.split(":", 1)) for p in args.ignore_pairs])
    complete = [m for m, d in rows.groupby("model") if d["is_target"].nunique() == 2]
    models = [m for m in (args.models or cm.ordered(complete)) if m in complete]
    skipped = sorted(set(rows["model"]) - set(models))
    if skipped:
        print(f"skipped (need both passes): {skipped}")

    all_ev, summary = [], []
    for m in models:
        ev = analyse_model(rows[rows["model"] == m], args.scoring, args.persistent)
        if ev.empty:
            print(f"{m}: no false positives")
            continue
        ev.insert(0, "model", m)
        all_ev.append(ev)
        s = {"Model": NAMES.get(m, m), "_key": m, "False positives": len(ev),
             "Persistent (no view fixes)": f"{100 * ev['persistent'].mean():.0f}%"}
        for k in (30, 60, 90):
            s[f"Fixed ≤{k}° (strict)"] = f"{100 * ev[f'best_strict_{k}'].mean():.0f}%"
        s["Random 30° move fixes"] = f"{100 * ev['random_strict_30'].mean():.0f}%"
        s["Fixed ≤90° (loose)"] = f"{100 * ev['best_loose_90'].mean():.0f}%"
        summary.append(s)
    if not all_ev:
        raise SystemExit("no false-positive events found")
    E = pd.concat(all_ev, ignore_index=True)
    E.to_csv(out / f"fp_resolution_{args.experiment}.csv", index=False)
    S = pd.DataFrame(summary)
    cols = [c for c in S.columns if c != "_key"]
    md = [f"**How many false positives a different viewpoint fixes ({args.experiment}, strict = FP gone "
          f"AND object still found [{args.scoring}])**", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in S.iterrows()]
    (out / f"fp_resolution_{args.experiment}_summary.md").write_text("\n".join(md) + "\n")
    print("\n" + md[0].strip("*"))
    print(S[cols].to_string(index=False))

    # by elevation of the failure: share fixed within 60 deg (strict)
    byel = (E.groupby(["model", "elevation"])["best_strict_60"].mean().unstack() * 100).round(0)
    byel.index = [NAMES.get(m, m) for m in byel.index]
    byel.to_csv(out / f"fp_resolution_{args.experiment}_by_elevation.csv")
    print("\n% of false positives fixed within 60° (strict), by elevation where the failure happened:")
    print(byel.to_string())

    # curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, kind, title in [(axes[0], "best", "Best view within k° (upper bound)"),
                            (axes[1], "random", "Random camera move of ~k°")]:
        for m in models:
            d = E[E["model"] == m]
            if d.empty:
                continue
            col = cm.MODEL_COLORS.get(m, "#8a8a8a")
            ys = [100 * d[f"{kind}_strict_{k}"].mean() for k in KS]
            yl = [100 * d[f"{kind}_loose_{k}"].mean() for k in KS]
            ax.plot(KS, ys, color=col, linewidth=2, marker="o", markersize=5, markeredgecolor="white",
                    markeredgewidth=1.2, label=NAMES.get(m, m))
            ax.plot(KS, yl, color=col, linewidth=1, linestyle=":", alpha=0.8)
        ax.set_title(title, fontsize=10, color=cm.INK, loc="left")
        ax.set_xlabel("camera movement from the failure view (deg)", fontsize=8, color=cm.MUTED)
        ax.set_ylabel("% of false positives fixed", fontsize=8, color=cm.MUTED)
        ax.set_ylim(0, 100)
        ax.set_xticks(KS)
        ax.grid(color="#ececec")
        ax.tick_params(colors=cm.MUTED, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].text(0.99, 0.02, "solid: FP gone and object still found   dotted: FP gone",
                 transform=axes[1].transAxes, ha="right", fontsize=7, color=cm.MUTED)
    fig.suptitle("How many false positives does a different viewpoint fix?", fontsize=11, color=cm.INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / f"fp_resolution_{args.experiment}.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}/fp_resolution_{args.experiment}*.csv/.md/.png")


if __name__ == "__main__":
    main()
