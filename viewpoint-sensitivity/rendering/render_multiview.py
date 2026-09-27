#!/usr/bin/env python
"""Render the controlled multi-view dataset with habitat-sim: each object
alone on a bare stage, viewed from a fixed elevation x azimuth grid."""

import argparse
import json
import math
import subprocess
import sys

import numpy as np

import config as cfg

if cfg.DATASET_SOURCE == "shapenet":
    from shapenet_objects import list_shapenet_models as list_models
elif cfg.DATASET_SOURCE == "objaverse":
    from objaverse_objects import list_objaverse_models as list_models
else:
    raise ValueError(f"Unknown config.DATASET_SOURCE: {cfg.DATASET_SOURCE!r}")


def spherical_offset(radius: float, elevation_deg: float, azimuth_deg: float) -> np.ndarray:
    """Offset from object center to camera eye, in world coords (Y-up)."""
    elev = math.radians(elevation_deg)
    azim = math.radians(azimuth_deg)
    horiz = radius * math.cos(elev)
    y = radius * math.sin(elev)
    x = horiz * math.cos(azim)
    z = horiz * math.sin(azim)
    return np.array([x, y, z], dtype=np.float64)


def make_sim_config():
    """A bare, mesh-less stage ("NONE") with just RGB/depth/semantic sensors --
    the one object this worker handles is added to it once.
    """
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = "NONE"
    sim_cfg.enable_physics = True  # needed to add rigid objects via the object managers

    sensor_specs = []
    for uuid, sensor_type in [
        ("rgb", habitat_sim.SensorType.COLOR),
        ("depth", habitat_sim.SensorType.DEPTH),
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


def render_model(sim, model, dry_run: bool, manifest_fh):
    rigid_obj = load_object_on_bare_stage(sim, model, semantic_id=1)
    try:
        center, aabb_diag = object_world_aabb(rigid_obj)
        radius = min(
            cfg.MAX_ORBIT_RADIUS_M,
            max(cfg.MIN_ORBIT_RADIUS_M, aabb_diag * cfg.ORBIT_RADIUS_SCALE),
        )

        agent = sim.get_agent(0)
        candidates = []  # (elev, azim, obs, visible_px, eye, rot)

        for elev in cfg.ELEVATIONS_DEG:
            for azim in cfg.AZIMUTHS_DEG:
                eye = center + spherical_offset(radius, elev, azim)
                rot = look_at_quaternion(eye, center)

                state = habitat_sim.AgentState()
                state.position = eye
                state.rotation = rot
                agent.set_state(state)

                if dry_run:
                    candidates.append((elev, azim, None, None, eye, rot))
                    continue

                obs = sim.get_sensor_observations()
                semantic = obs["semantic"]
                visible_px = int(np.count_nonzero(semantic == 1))
                candidates.append((elev, azim, obs, visible_px, eye, rot))

        if dry_run:
            print(f"  {model.category}/{model.model_id}: {len(candidates)} candidate views, "
                  f"radius={radius:.2f}m")
            return

        max_px = max((c[3] for c in candidates), default=0)
        if max_px == 0:
            print(f"  skipping {model.category}/{model.model_id}: never visible from any "
                  f"sampled view (check orbit radius / model scale)")
            return

        kept = 0
        for elev, azim, obs, visible_px, eye, rot in candidates:
            if visible_px < cfg.MIN_VISIBLE_FRACTION * max_px:
                continue

            view_id = f"{model.category}/{model.model_id}/elev{elev:+d}_az{azim:03d}"
            out_dir = cfg.RENDERS_DIR / model.category / model.model_id
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = f"elev{elev:+d}_az{azim:03d}"

            rgb_path = out_dir / f"{stem}_rgb.png"
            Image.fromarray(obs["rgb"][:, :, :3]).save(rgb_path)

            depth_path = out_dir / f"{stem}_depth.npy"
            np.save(depth_path, obs["depth"])

            semantic_mask = (obs["semantic"] == 1)
            ys, xs = np.nonzero(semantic_mask)
            gt_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None

            row = {
                "view_id": view_id,
                "dataset": cfg.DATASET_SOURCE,
                "category": model.category,
                # synset_id for ShapeNet, lvis_key for Objaverse -- whichever this model has
                "source_id": getattr(model, "synset_id", None) or getattr(model, "lvis_key", None),
                "model_id": model.model_id,
                "prompt": cfg.CATEGORY_PROMPTS[model.category],
                "elevation_deg": elev,
                "azimuth_deg": azim,
                "rgb_path": str(rgb_path.relative_to(cfg.PROJECT_ROOT)),
                "depth_path": str(depth_path.relative_to(cfg.PROJECT_ROOT)),
                "gt_bbox_xyxy": gt_bbox,
                "gt_pixel_count": int(visible_px),
                "gt_visible_fraction_of_best_view": visible_px / max_px,
                "camera_position": eye.tolist(),
                "camera_rotation_wxyz": [rot.w, rot.x, rot.y, rot.z],
                "object_center": center.tolist(),
                "orbit_radius_m": radius,
            }
            manifest_fh.write(json.dumps(row) + "\n")
            manifest_fh.flush()
            kept += 1

        print(f"  {model.category}/{model.model_id}: kept {kept}/{len(candidates)} views")
    finally:
        remove_object(sim, rigid_obj, model)


def run_worker(category: str, model_id: str, dry_run: bool):
    """Handles exactly one model in its own fresh Simulator, in this process
    (invoked as a subprocess by the dispatcher below). A native crash here
    only takes down this one worker.
    """
    model = next((m for m in list_models() if m.category == category and m.model_id == model_id), None)
    if model is None:
        raise SystemExit(f"model {category}/{model_id} not found by list_models()")

    cfg.RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    sim = habitat_sim.Simulator(make_sim_config())
    try:
        if dry_run:
            render_model(sim, model, dry_run=True, manifest_fh=None)
        else:
            with open(cfg.MANIFEST_PATH, "a") as manifest_fh:
                render_model(sim, model, dry_run=False, manifest_fh=manifest_fh)
    finally:
        sim.close()


def already_rendered_model_ids() -> set:
    if not cfg.MANIFEST_PATH.exists():
        return set()
    done = set()
    with open(cfg.MANIFEST_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["model_id"])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="list models/candidate views without rendering or writing anything")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)  # internal, see run_worker
    parser.add_argument("--category", help=argparse.SUPPRESS)
    parser.add_argument("--model-id", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.category, args.model_id, args.dry_run)
        return

    models = list_models()
    if not models:
        download_script = "download_shapenet.sh" if cfg.DATASET_SOURCE == "shapenet" else "download_objaverse.sh"
        raise SystemExit(
            f"No {cfg.DATASET_SOURCE} models found -- run scripts/{download_script} first "
            f"(or check config.DATASET_SOURCE, currently {cfg.DATASET_SOURCE!r})."
        )
    print(f"found {len(models)} {cfg.DATASET_SOURCE} models across {len(cfg.CATEGORIES)} categories")

    skip_ids = already_rendered_model_ids() if not args.dry_run else set()
    if skip_ids:
        print(f"resuming: {len(skip_ids)} models already in the manifest will be skipped")

    n_crashed = 0
    for model in models:
        if model.model_id in skip_ids:
            continue

        worker_cmd = [
            sys.executable, __file__, "--worker",
            "--category", model.category, "--model-id", model.model_id,
        ]
        if args.dry_run:
            worker_cmd.append("--dry-run")

        result = subprocess.run(worker_cmd, capture_output=True, text=True)
        # forward the worker's own prints (it does the "kept N/M views" logging)
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            n_crashed += 1
            print(f"  [CRASHED] {model.category}/{model.model_id} (exit {result.returncode}) -- skipped. "
                  f"Last stderr line: {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else '(none)'}")

    print(f"\ndone. {n_crashed} model(s) crashed and were skipped.")
    if not args.dry_run:
        print(f"manifest written to {cfg.MANIFEST_PATH}")


if __name__ == "__main__":
    # Deferred so `--worker` subprocesses (and only they) pay the import cost;
    # the dispatcher process itself never touches habitat_sim.
    import habitat_sim
    from PIL import Image
    from habitat_object_loader import (
        load_object_on_bare_stage, object_world_aabb, remove_object, look_at_quaternion,
    )

    main()
