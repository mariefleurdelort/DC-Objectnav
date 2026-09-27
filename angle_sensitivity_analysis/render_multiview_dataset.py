"""
Controlled multi-view dataset generator for viewpoint sensitivity analysis.

Given a Gibson/3DSceneGraph scene loaded in Habitat-Sim, this script:
  1. Iterates over target semantic object instances in the scene
  2. Samples camera positions on a sphere/hemisphere around each object
  3. Renders RGB + semantic observations from each camera pose
  4. Extracts per-view GT (bbox, visible pixel count, visibility flag) from
     the semantic sensor
  5. Writes images + a manifest.csv that downstream model-inference scripts
     (YOLOWorld / GroundingDINO / SAM3 / OWLv2) consume

Run this inside a Habitat-Sim-only environment (NOT your VLFM production
image) — see Dockerfile.render.

Usage:
    python render_multiview_dataset.py \
        --scene-glb /data/scene_datasets/gibson_semantic/Collierville.glb \
        --scene-name Collierville \
        --out-dir /data/viewpoint_sensitivity/renders \
        --categories chair bed "potted plant" toilet tv couch \
        --azimuth-step 30 \
        --elevations -15 0 15 30 45 60
"""

import argparse
import csv
import os
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import quaternion as qt
import habitat_sim


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

IMG_WIDTH = 640
IMG_HEIGHT = 480
HFOV = 90

MIN_VISIBLE_PIXELS = 200  # below this, mark the object "not_visible" for this view


@dataclass
class ViewRecord:
    scene: str
    object_id: str
    category: str
    azimuth_deg: float
    elevation_deg: float
    radius: float
    cam_x: float
    cam_y: float
    cam_z: float
    visible: bool
    visible_px: int
    bbox_x0: Optional[int]
    bbox_y0: Optional[int]
    bbox_x1: Optional[int]
    bbox_y1: Optional[int]
    rgb_path: str
    sem_path: str


# --------------------------------------------------------------------------
# Sim setup
# --------------------------------------------------------------------------

def make_sim(scene_glb_path: str) -> habitat_sim.Simulator:
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_glb_path
    backend_cfg.enable_physics = False

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [IMG_HEIGHT, IMG_WIDTH]
    rgb_spec.hfov = HFOV
    rgb_spec.position = [0.0, 0.0, 0.0]

    sem_spec = habitat_sim.CameraSensorSpec()
    sem_spec.uuid = "semantic"
    sem_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    sem_spec.resolution = [IMG_HEIGHT, IMG_WIDTH]
    sem_spec.hfov = HFOV
    sem_spec.position = [0.0, 0.0, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, sem_spec]

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def sphere_points(center: np.ndarray, radius: float,
                   azimuths_deg: List[float], elevations_deg: List[float]):
    """Yield (az, el, cam_pos) for every combination on a sphere around center.

    Habitat convention: Y is up. Azimuth sweeps the X-Z plane, elevation
    tilts toward +Y.
    """
    for el in elevations_deg:
        el_r = np.radians(el)
        for az in azimuths_deg:
            az_r = np.radians(az)
            x = radius * np.cos(el_r) * np.cos(az_r)
            y = radius * np.sin(el_r)
            z = radius * np.cos(el_r) * np.sin(az_r)
            cam_pos = center + np.array([x, y, z], dtype=np.float32)
            yield az, el, cam_pos


def look_at_quat(cam_pos: np.ndarray, target: np.ndarray,
                  world_up=np.array([0.0, 1.0, 0.0])) -> qt.quaternion:
    forward = target - cam_pos
    norm = np.linalg.norm(forward)
    forward = forward / norm if norm > 1e-6 else np.array([0.0, 0.0, -1.0])

    # guard against forward ~ parallel to world_up (near-vertical views)
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([1.0, 0.0, 0.0])

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    # habitat cameras look down -Z in their own local frame
    rot_mat = np.array([right, true_up, -forward]).T
    return qt.from_rotation_matrix(rot_mat)


# --------------------------------------------------------------------------
# Visibility / raycast sanity check
# --------------------------------------------------------------------------

def is_position_valid(sim: habitat_sim.Simulator, cam_pos: np.ndarray,
                       target: np.ndarray) -> bool:
    """Rejects camera positions embedded in geometry (walls/furniture) and
    positions where the line of sight to the object is blocked well short
    of the object itself."""
    ray = habitat_sim.geo.Ray()
    ray.origin = cam_pos.astype(np.float32)
    direction = target - cam_pos
    dist_to_target = float(np.linalg.norm(direction))
    ray.direction = (direction / dist_to_target).astype(np.float32)

    hit = sim.cast_ray(ray)
    if not hit.has_hits():
        return False  # escapes into open space -> likely outside the scene mesh

    first_hit_dist = hit.hits[0].ray_distance
    # allow the ray to terminate at (not before) the object itself
    return first_hit_dist >= dist_to_target - 0.15


# --------------------------------------------------------------------------
# Semantic extraction
# --------------------------------------------------------------------------

def extract_gt(sem_obs: np.ndarray, semantic_id: int):
    mask = sem_obs == semantic_id
    visible_px = int(mask.sum())
    if visible_px < MIN_VISIBLE_PIXELS:
        return False, visible_px, None
    ys, xs = np.where(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return True, visible_px, bbox


# --------------------------------------------------------------------------
# Main generation loop
# --------------------------------------------------------------------------

def generate_for_scene(sim: habitat_sim.Simulator, scene_name: str,
                        target_categories: List[str],
                        azimuths_deg: List[float], elevations_deg: List[float],
                        radius_multiplier: float, out_dir: str) -> List[ViewRecord]:
    records: List[ViewRecord] = []
    agent = sim.get_agent(0)
    scene_objs = sim.semantic_scene.objects

    for obj in scene_objs:
        if obj is None or obj.category is None:
            continue
        cat_name = obj.category.name()
        if cat_name not in target_categories:
            continue

        center = np.array(obj.aabb.center, dtype=np.float32)
        extent = np.array(obj.aabb.sizes, dtype=np.float32)
        obj_radius = float(np.linalg.norm(extent) / 2.0)
        radius = max(obj_radius * radius_multiplier, 0.75)

        obj_dir = os.path.join(out_dir, scene_name, str(obj.id))
        os.makedirs(obj_dir, exist_ok=True)

        for az, el, cam_pos in sphere_points(center, radius, azimuths_deg, elevations_deg):
            valid = is_position_valid(sim, cam_pos, center)

            if not valid:
                records.append(ViewRecord(
                    scene=scene_name, object_id=str(obj.id), category=cat_name,
                    azimuth_deg=az, elevation_deg=el, radius=radius,
                    cam_x=float(cam_pos[0]), cam_y=float(cam_pos[1]), cam_z=float(cam_pos[2]),
                    visible=False, visible_px=0,
                    bbox_x0=None, bbox_y0=None, bbox_x1=None, bbox_y1=None,
                    rgb_path="", sem_path="",
                ))
                continue

            state = habitat_sim.AgentState()
            state.position = cam_pos
            state.rotation = look_at_quat(cam_pos, center)
            agent.set_state(state)

            obs = sim.get_sensor_observations()
            rgb = obs["rgb"][:, :, :3]
            sem = obs["semantic"]

            visible, visible_px, bbox = extract_gt(sem, obj.semantic_id)

            rgb_path = os.path.join(obj_dir, f"az{int(az):03d}_el{int(el):03d}_rgb.png")
            sem_path = os.path.join(obj_dir, f"az{int(az):03d}_el{int(el):03d}_sem.npy")

            from PIL import Image
            Image.fromarray(rgb).save(rgb_path)
            np.save(sem_path, sem)

            records.append(ViewRecord(
                scene=scene_name, object_id=str(obj.id), category=cat_name,
                azimuth_deg=az, elevation_deg=el, radius=radius,
                cam_x=float(cam_pos[0]), cam_y=float(cam_pos[1]), cam_z=float(cam_pos[2]),
                visible=visible, visible_px=visible_px,
                bbox_x0=bbox[0] if bbox else None,
                bbox_y0=bbox[1] if bbox else None,
                bbox_x1=bbox[2] if bbox else None,
                bbox_y1=bbox[3] if bbox else None,
                rgb_path=rgb_path, sem_path=sem_path,
            ))

    return records


def write_manifest(records: List[ViewRecord], manifest_path: str):
    if not records:
        return
    fieldnames = list(asdict(records[0]).keys())
    write_header = not os.path.exists(manifest_path)
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render controlled multi-view dataset for viewpoint sensitivity analysis")
    parser.add_argument("--scene-glb", required=True, help="Path to scene .glb")
    parser.add_argument("--scene-name", required=True, help="Scene identifier, e.g. Collierville")
    parser.add_argument("--out-dir", required=True, help="Output root directory")
    parser.add_argument("--categories", nargs="+",
                         default=["chair", "bed", "potted plant", "toilet", "tv", "couch"])
    parser.add_argument("--azimuth-step", type=float, default=30.0)
    parser.add_argument("--elevations", nargs="+", type=float, default=[-15, 0, 15, 30, 45, 60])
    parser.add_argument("--radius-multiplier", type=float, default=2.0,
                         help="Camera distance as a multiple of the object's bounding radius")
    args = parser.parse_args()

    azimuths = list(np.arange(0, 360, args.azimuth_step))

    sim = make_sim(args.scene_glb)
    try:
        records = generate_for_scene(
            sim, args.scene_name, args.categories,
            azimuths, args.elevations, args.radius_multiplier, args.out_dir,
        )
    finally:
        sim.close()

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    write_manifest(records, manifest_path)

    n_visible = sum(1 for r in records if r.visible)
    print(f"[{args.scene_name}] wrote {len(records)} view records "
          f"({n_visible} visible, {len(records) - n_visible} invalid/occluded) "
          f"-> {manifest_path}")


if __name__ == "__main__":
    main()