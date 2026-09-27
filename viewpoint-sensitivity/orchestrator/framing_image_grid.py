#!/usr/bin/env python
"""Builds a 12x12 grid of the actual rendered RGB thumbnails per instance,
laid out in the same XY positions as framing_heatmap.py's heatmaps -- so each
heatmap has a matching "what did the object actually look like" image, the
same spirit as the reference paper's left-panel thumbnail grid.

Works for both true-positive and false-positive data -- see
framing_heatmap.py's docstring for the "correct" vs "false_positive" field
meanings (opposite polarity: for false-positive data, green means the model
WAS fooled, not that it did well).

Run inside vps-orchestrator (only needs PIL, already a dep), after
render_framing_grid.py + run_inference.py have produced images + a log:
    python orchestrator/framing_image_grid.py --limit 3              # quick preview, no border
    python orchestrator/framing_image_grid.py --limit 3 --model groundingdino  # + green/red border
    python orchestrator/framing_image_grid.py                        # all instances, no border
    python orchestrator/framing_image_grid.py --model groundingdino  # all instances, bordered

For false-positive data:
    python orchestrator/framing_image_grid.py --model groundingdino \\
        --manifest data/renders_framing_falsepositive/framing_manifest.jsonl \\
        --inference-log data/renders_framing_falsepositive/framing_inference_log.jsonl \\
        --value-field false_positive
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAMING_DIR = PROJECT_ROOT / "data" / "renders_framing"
DEFAULT_MANIFEST = FRAMING_DIR / "framing_manifest.jsonl"
DEFAULT_INFERENCE_LOG = FRAMING_DIR / "framing_inference_log.jsonl"

TILE_SIZE = 100
BORDER = 4


def load_manifest_rows(manifest_path: Path):
    return [json.loads(l) for l in open(manifest_path)]


def load_success(inference_log_path: Path, model: str, value_field: str):
    """view_id -> bool, for one model, for whichever value_field was requested."""
    rows = [json.loads(l) for l in open(inference_log_path)]
    return {r["view_id"]: r[value_field] for r in rows if r["model"] == model}


def build_grid_for_instance(rows, success=None):
    xs = sorted(set(r["offset_x_deg"] for r in rows))
    ys = sorted(set(r["offset_y_deg"] for r in rows), reverse=True)  # top row = highest y, matches imshow origin="lower"

    cell = TILE_SIZE + BORDER * 2
    canvas = Image.new("RGB", (len(xs) * cell, len(ys) * cell), "white")
    draw = ImageDraw.Draw(canvas)

    by_pos = {(r["offset_x_deg"], r["offset_y_deg"]): r for r in rows}

    for ti, y in enumerate(ys):
        for pi, x in enumerate(xs):
            row = by_pos.get((x, y))
            x0, y0 = pi * cell, ti * cell
            if row is None:
                continue
            img_path = PROJECT_ROOT / row["rgb_path"]
            if not img_path.exists():
                continue
            thumb = Image.open(img_path).convert("RGB").resize((TILE_SIZE, TILE_SIZE))

            border_color = "white"
            if success is not None:
                c = success.get(row["view_id"])
                border_color = "gray" if c is None else ("green" if c else "red")

            canvas.paste(thumb, (x0 + BORDER, y0 + BORDER))
            draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], outline=border_color, width=BORDER)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--inference-log", default=str(DEFAULT_INFERENCE_LOG))
    parser.add_argument("--value-field", default="correct", choices=["correct", "false_positive"],
                         help="which boolean field to color the border by -- green means this field "
                              "was True (for false_positive, green means the model WAS fooled)")
    parser.add_argument("--output-dir", default=None,
                         help="default: an 'image_grids' folder next to --inference-log")
    parser.add_argument("--model", default=None,
                         help="if given, borders each tile green/red/gray by that model's --value-field result")
    parser.add_argument("--limit", type=int, default=None, help="only the first N instances (alphabetical)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    inference_log_path = Path(args.inference_log)
    output_dir = Path(args.output_dir) if args.output_dir else inference_log_path.parent / "image_grids"

    rows = load_manifest_rows(manifest_path)
    by_instance = {}
    for r in rows:
        key = f"{r['category']}/{r['model_id']}"
        by_instance.setdefault(key, []).append(r)

    instance_keys = sorted(by_instance.keys())
    if args.limit:
        instance_keys = instance_keys[:args.limit]

    success = load_success(inference_log_path, args.model, args.value_field) if args.model else None
    out_dir = output_dir / (args.model or "raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, key in enumerate(instance_keys):
        canvas = build_grid_for_instance(by_instance[key], success)
        canvas.save(out_dir / f"{key.replace('/', '_')}.png")
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(instance_keys)} instance grids written")

    print(f"\ndone. {len(instance_keys)} image grid(s) written to {out_dir}")


if __name__ == "__main__":
    main()
