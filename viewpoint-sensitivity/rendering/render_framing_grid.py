#!/usr/bin/env python
"""Renders an offset grid for a list of selected viewpoints (see
rendering/viewpoint_selection.py for how those are chosen -- this script
doesn't care whether they came from the true-positive or false-positive
selector, it just needs eye/rotation/orbit_radius per entry).

For each entry, the camera is translated across a fixed 12x12 XY grid
(rendering/camera_offset.py) -- position moves, orientation never changes --
so the object drifts from center toward each frame edge/corner via pure
parallax, not by the camera turning to track it.

Two strategies, writing to separate output directories so they never collide:
    --strategy true   (default) seed viewpoint = each instance's best REAL
        detection confidence, from data/renders/inference_log.jsonl. Output:
        data/renders_framing/
    --strategy false --seed-model <model>   seed viewpoint = a genuine,
        confirmed false-positive from THAT SPECIFIC model (not an average
        across models -- different models fail on different instances at
        different viewpoints, so this is per-model) -- see
        viewpoint_selection.py:select_false_positive_seed_viewpoints().
        Output: data/renders_framing_falsepositive/<model>/  (one instance
        may get a different render per model, since the seed differs)

Run inside vps-render, after the main render_multiview.py + orchestrator/
run_inference.py (and, for --strategy false, a --false-positive inference
pass over the main manifest) have already produced their logs:
    conda activate vps-render
    python rendering/render_framing_grid.py --limit 3                      # pilot, true-positive seed
    python rendering/render_framing_grid.py --strategy false --seed-model groundingdino --limit 3 \\
        --false-inference-log data/renders/false_positive_inference_log.jsonl  # pilot, false-positive seed
    python rendering/render_framing_grid.py                                 # full, true-positive seed

Same subprocess-per-instance crash isolation as render_multiview.py.
"""
import argparse
import json
import subprocess
import sys

import numpy as np

import config as cfg
from viewpoint_selection import select_best_true_viewpoints, select_false_positive_seed_viewpoints

if cfg.DATASET_SOURCE == "shapenet":
    from shapenet_objects import list_shapenet_models as list_models
elif cfg.DATASET_SOURCE == "objaverse":
    from objaverse_objects import list_objaverse_models as list_models
else:
    raise ValueError(f"Unknown config.DATASET_SOURCE: {cfg.DATASET_SOURCE!r}")


def renders_dir_for(strategy: str, seed_model: str = None):
    if strategy == "true":
        return cfg.PROJECT_ROOT / "data" / "renders_framing"
    return cfg.PROJECT_ROOT / "data" / "renders_framing_falsepositive" / seed_model


def manifest_path_for(strategy: str, seed_model: str = None):
    return renders_dir_for(strategy, seed_model) / "framing_manifest.jsonl"


def best_viewpoints_path_for(strategy: str, seed_model: str = None):
    return renders_dir_for(strategy, seed_model) / "best_viewpoints.json"


def render_instance(entry: dict, dry_run: bool, manifest_fh, renders_dir):
    """Renders the XY offset grid for one instance, given its selected base
    viewpoint (has camera_position/camera_rotation_wxyz/object_center/
    orbit_radius_m already -- no need to reload/recompute the object's AABB,
    loading is deterministic).
    """
    models = [m for m in list_models(max_per_category=None)
              if m.category == entry["category"] and m.model_id == entry["model_id"]]
    if not models:
        raise SystemExit(f"model {entry['category']}/{entry['model_id']} not found by list_models()")
    model = models[0]

    rigid_obj = load_object_on_bare_stage(sim, model, semantic_id=1)
    try:
        eye = np.array(entry["camera_position"])
        # the base rotation is held FIXED for every grid point -- only eye position moves
        rot = np.quaternion(*entry["camera_rotation_wxyz"])
        orbit_radius = entry["orbit_radius_m"]
        agent = sim.get_agent(0)

        kept = 0
        for x_deg in OFFSET_X_STEPS:
            for y_deg in OFFSET_Y_STEPS:
                offset_eye = offset_eye_xy(eye, rot, orbit_radius, x_deg, y_deg)

                state = habitat_sim.AgentState()
                state.position = offset_eye
                state.rotation = rot  # unchanged -- pure translation, no re-aiming
                agent.set_state(state)

                if dry_run:
                    kept += 1
                    continue

                obs = sim.get_sensor_observations()
                semantic_mask = (obs["semantic"] == 1)
                visible_px = int(np.count_nonzero(semantic_mask))
                ys, xs = np.nonzero(semantic_mask)
                gt_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None

                view_id = f"{model.category}/{model.model_id}/x{x_deg:+.1f}_y{y_deg:+.1f}"
                out_dir = renders_dir / model.category / model.model_id
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = f"x{x_deg:+.1f}_y{y_deg:+.1f}"

                rgb_path = out_dir / f"{stem}_rgb.png"
                Image.fromarray(obs["rgb"][:, :, :3]).save(rgb_path)

                row = {
                    "view_id": view_id,
                    "dataset": cfg.DATASET_SOURCE,
                    "category": model.category,
                    "model_id": model.model_id,
                    "prompt": cfg.CATEGORY_PROMPTS[model.category],
                    "base_elevation_deg": entry.get("elevation_deg"),
                    "base_azimuth_deg": entry.get("azimuth_deg"),
                    "selection_strategy": entry.get("selection_strategy"),
                    "seed_model": entry.get("seed_model"),              # set only for --strategy false
                    "seed_false_category": entry.get("false_category"),  # set only for --strategy false
                    "offset_x_deg": x_deg,
                    "offset_y_deg": y_deg,
                    "rgb_path": str(rgb_path.relative_to(cfg.PROJECT_ROOT)),
                    "gt_bbox_xyxy": gt_bbox,
                    "gt_pixel_count": visible_px,
                    "camera_position": offset_eye.tolist(),
                    "camera_rotation_wxyz": [rot.w, rot.x, rot.y, rot.z],
                    "object_center": entry["object_center"],
                    "orbit_radius_m": orbit_radius,
                }
                manifest_fh.write(json.dumps(row) + "\n")
                manifest_fh.flush()
                kept += 1

        print(f"  {model.category}/{model.model_id}: rendered {kept}/{len(OFFSET_X_STEPS) * len(OFFSET_Y_STEPS)} "
              f"offset views (base elev={entry.get('elevation_deg')}, az={entry.get('azimuth_deg')})")
    finally:
        remove_object(sim, rigid_obj, model)


def make_sim_config():
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = "NONE"
    sim_cfg.enable_physics = True

    sensor_specs = []
    for uuid, sensor_type in [
        ("rgb", habitat_sim.SensorType.COLOR),
        ("semantic", habitat_sim.SensorType.SEMANTIC),
    ]:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH]
        spec.hfov = cfg.HFOV_DEG
        sensor_specs.append(spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def run_worker(strategy: str, seed_model: str, category: str, model_id: str, dry_run: bool):
    with open(best_viewpoints_path_for(strategy, seed_model)) as f:
        all_entries = json.load(f)
    entry = next((e for e in all_entries if e["category"] == category and e["model_id"] == model_id), None)
    if entry is None:
        raise SystemExit(f"no selected viewpoint for {category}/{model_id}")

    renders_dir = renders_dir_for(strategy, seed_model)
    renders_dir.mkdir(parents=True, exist_ok=True)
    global sim
    sim = habitat_sim.Simulator(make_sim_config())
    try:
        if dry_run:
            render_instance(entry, dry_run=True, manifest_fh=None, renders_dir=renders_dir)
        else:
            with open(manifest_path_for(strategy, seed_model), "a") as manifest_fh:
                render_instance(entry, dry_run=False, manifest_fh=manifest_fh, renders_dir=renders_dir)
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N instances (pilot runs)")
    parser.add_argument("--strategy", choices=["true", "false"], default="true",
                         help="'true' seeds from each instance's best real-detection viewpoint; "
                              "'false' seeds from --seed-model's own confirmed false positives "
                              "(requires --seed-model and --false-inference-log)")
    parser.add_argument("--seed-model", default=None,
                         help="which model's own false positives to seed from (required when --strategy false)")
    parser.add_argument("--false-inference-log", default=None,
                         help="path to a --false-positive run_inference.py log over the main manifest "
                              "(required when --strategy false)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--category", help=argparse.SUPPRESS)
    parser.add_argument("--model-id", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.strategy, args.seed_model, args.category, args.model_id, args.dry_run)
        return

    if args.strategy == "false" and not (args.false_inference_log and args.seed_model):
        raise SystemExit("--strategy false requires --seed-model and --false-inference-log")

    renders_dir = renders_dir_for(args.strategy, args.seed_model)
    renders_dir.mkdir(parents=True, exist_ok=True)

    if args.strategy == "true":
        entries = select_best_true_viewpoints(limit=args.limit)
    else:
        entries = select_false_positive_seed_viewpoints(args.false_inference_log, args.seed_model, limit=args.limit)

    print(f"selected {len(entries)} instances' viewpoints (strategy={args.strategy}"
          f"{', seed_model=' + args.seed_model if args.seed_model else ''}) "
          f"({len(OFFSET_X_STEPS)}x{len(OFFSET_Y_STEPS)}={len(OFFSET_X_STEPS)*len(OFFSET_Y_STEPS)} "
          f"offset views each = {len(entries) * len(OFFSET_X_STEPS) * len(OFFSET_Y_STEPS)} total views)")

    if not entries:
        print("nothing to render (no confirmed false positives for this model) -- exiting")
        return

    with open(best_viewpoints_path_for(args.strategy, args.seed_model), "w") as f:
        json.dump(entries, f, indent=2)

    n_crashed = 0
    for entry in entries:
        worker_cmd = [
            sys.executable, __file__, "--worker", "--strategy", args.strategy,
            "--category", entry["category"], "--model-id", entry["model_id"],
        ]
        if args.seed_model:
            worker_cmd += ["--seed-model", args.seed_model]
        if args.dry_run:
            worker_cmd.append("--dry-run")

        result = subprocess.run(worker_cmd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            n_crashed += 1
            last_line = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "(none)"
            print(f"  [CRASHED] {entry['category']}/{entry['model_id']} (exit {result.returncode}) "
                  f"-- skipped. Last stderr line: {last_line}")

    print(f"\ndone. {n_crashed} instance(s) crashed and were skipped.")
    if not args.dry_run:
        print(f"manifest written to {manifest_path_for(args.strategy, args.seed_model)}")


if __name__ == "__main__":
    import habitat_sim
    import quaternion
    from PIL import Image
    from habitat_object_loader import load_object_on_bare_stage, remove_object
    from camera_offset import offset_eye_xy, OFFSET_X_STEPS, OFFSET_Y_STEPS

    main()
