import json, gzip, sys, os
import habitat_sim
import numpy as np

episodes_path = sys.argv[1] if len(sys.argv) > 1 else \
    "data/datasets/objectnav/gibson/v1.1/val/content/Corozal_episodes.json.gz"

with gzip.open(episodes_path, "rt") as f:
    data = json.load(f)

scene_name = data["episodes"][0]["scene_id"].split("/")[-1]  # e.g. "Corozal.glb"
scene_stem = os.path.splitext(scene_name)[0]
scene_path = f"data/scene_datasets/gibson_semantic/{scene_name}"
navmesh_path = f"data/scene_datasets/gibson_semantic/{scene_stem}.navmesh"

print(f"Scene: {scene_path}")
print(f"Navmesh: {navmesh_path}")

backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = scene_path
backend_cfg.enable_physics = False
agent_cfg = habitat_sim.agent.AgentConfiguration()
sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
sim = habitat_sim.Simulator(sim_cfg)

if os.path.exists(navmesh_path):
    sim.pathfinder.load_nav_mesh(navmesh_path)
if not sim.pathfinder.is_loaded:
    print("[FATAL] navmesh did not load — geodesic checks will be invalid.")
    sim.close()
    sys.exit(1)

zero_dist = 0
inf_dist = 0
all_zero_iou = 0
checked = 0

for ep in data["episodes"]:
    start = np.array(ep["start_position"], dtype=np.float32)
    goals = ep.get("goals", [])
    checked += 1

    min_geo_dist = float("inf")
    ious = []
    for g in goals:
        for vp in g.get("view_points", []):
            vp_pos = np.array(vp["agent_state"]["position"], dtype=np.float32)
            ious.append(vp.get("iou", None))

            path = habitat_sim.ShortestPath()
            path.requested_start = start
            path.requested_end = vp_pos
            found = sim.pathfinder.find_path(path)
            d = path.geodesic_distance if found else float("inf")

            if d < min_geo_dist:
                min_geo_dist = d

    if min_geo_dist == 0.0:
        zero_dist += 1
        print(f"[ZERO] episode {ep['episode_id']}: geodesic dist to nearest viewpoint = 0.0")
    elif min_geo_dist == float("inf"):
        inf_dist += 1
        print(f"[UNREACHABLE] episode {ep['episode_id']}: no reachable viewpoint (inf geodesic dist)")

    non_null_ious = [i for i in ious if i is not None]
    if non_null_ious and all(i == 0.0 for i in non_null_ious):
        all_zero_iou += 1

sim.close()

print(f"\nChecked {checked} episodes")
print(f"  zero geodesic distance: {zero_dist}")
print(f"  unreachable (inf):      {inf_dist}")
print(f"  all viewpoints iou=0:   {all_zero_iou}")