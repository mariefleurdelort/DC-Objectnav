import sys, os, json, runpy, csv
import numpy as np
import cv2
import habitat.core.vector_env as _vec_module
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy

import time

OUTPUT_DIR = "data/trajectories"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(OUTPUT_DIR, "per_episode_results_v2.csv")
_csv_header_written = os.path.exists(RESULTS_CSV)

# Unique per-invocation identifier so multiple runs never get silently mixed
RUN_ID = os.environ.get("HYDRA_RUN_DIR") or time.strftime("%Y%m%d_%H%M%S")
print(f"[traj] RUN_ID = {RUN_ID}")

def _log_episode_csv(row):
    global _csv_header_written
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not _csv_header_written:
            writer.writeheader()
            _csv_header_written = True
        writer.writerow(row)

# Force _visualize=True so ITMPolicy.act always calls _update_value_map
_orig_policy_init = BaseObjectNavPolicy.__init__
def _policy_init(self, *args, **kwargs):
    _orig_policy_init(self, *args, **kwargs)
    self._visualize = True
BaseObjectNavPolicy.__init__ = _policy_init

# Track map instances
_obs_instances = []
_val_instances = []

_orig_obs_init = ObstacleMap.__init__
def _obs_init(self, *args, **kwargs):
    _orig_obs_init(self, *args, **kwargs)
    _obs_instances.append(self)
ObstacleMap.__init__ = _obs_init

_orig_val_init = ValueMap.__init__
def _val_init(self, *args, **kwargs):
    _orig_val_init(self, *args, **kwargs)
    _val_instances.append(self)
ValueMap.__init__ = _val_init

# import heapq
# from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
# from vlfm.policy.habitat_policies import TorchActionIDs
# from vlfm.utils.geometry_utils import rho_theta

# TURN_THRESH_RAD = 0.3  # ~17 degrees; tune if the agent oscillates or undershoots

# def _astar(occ_grid, start_px, goal_px):
#     """8-connected A* over a boolean occupancy grid. True = obstacle."""
#     h, w = occ_grid.shape
#     start_px, goal_px = tuple(start_px), tuple(goal_px)

#     if not (0 <= goal_px[0] < h and 0 <= goal_px[1] < w):
#         return None
#     if occ_grid[goal_px[0], goal_px[1]]:
#         return None  # goal itself is inside an obstacle — caller should handle

#     def heuristic(a, b):
#         return np.hypot(a[0] - b[0], a[1] - b[1])

#     neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
#     open_heap = [(0.0, start_px)]
#     came_from = {}
#     g_score = {start_px: 0.0}
#     visited = set()

#     while open_heap:
#         _, current = heapq.heappop(open_heap)
#         if current in visited:
#             continue
#         visited.add(current)
#         if current == goal_px:
#             path = [current]
#             while current in came_from:
#                 current = came_from[current]
#                 path.append(current)
#             return path[::-1]
#         for dx, dy in neighbors:
#             nx, ny = current[0] + dx, current[1] +baseobject dy
#             if 0 <= nx < h and 0 <= ny < w and not occ_grid[nx, ny]:
#                 step_cost = np.hypot(dx, dy)
#                 tentative = g_score[current] + step_cost
#                 if tentative < g_score.get((nx, ny), float("inf")):
#                     g_score[(nx, ny)] = tentative
#                     came_from[(nx, ny)] = current
#                     heapq.heappush(open_heap, (tentative + heuristic((nx, ny), goal_px), (nx, ny)))
#     return None

# _recent_positions = []
# STUCK_WINDOW = 8
# STUCK_DIST_THRESH = 0.15
# _recovery_turns_remaining = [0]  # how many forced-turn steps left

# def _astar_pointnav(self, goal, stop=False):
#     robot_xy = self._observations_cache["robot_xy"]
#     heading = self._observations_cache["robot_heading"]
#     rho, theta = rho_theta(robot_xy, heading, goal)
#     self._policy_info["rho_theta"] = np.array([rho, theta])

#     if rho < self._pointnav_stop_radius and stop:
#         self._called_stop = True
#         return self._stop_action

#     # If we're in the middle of a forced recovery turn, keep turning
#     # regardless of what A* would otherwise suggest
#     if _recovery_turns_remaining[0] > 0:
#         _recovery_turns_remaining[0] -= 1
#         return TorchActionIDs.TURN_LEFT

#     _recent_positions.append(robot_xy.copy())
#     if len(_recent_positions) > STUCK_WINDOW:
#         _recent_positions.pop(0)
#     if len(_recent_positions) == STUCK_WINDOW:
#         moved = np.linalg.norm(_recent_positions[-1] - _recent_positions[0])
#         if moved < STUCK_DIST_THRESH:
#             print(f"[astar] stuck detected (moved {moved:.3f}m in {STUCK_WINDOW} steps) — forcing recovery turn sequence")
#             _recovery_turns_remaining[0] = 4  # commit to several turns in a row, not just one
#             _recent_positions.clear()
#             return TorchActionIDs.TURN_LEFT

#     occ_grid = self._obstacle_map._map
#     pixels_per_meter = self._obstacle_map.pixels_per_meter
#     start_px = self._obstacle_map._xy_to_px(robot_xy[None, :])[0].astype(int)
#     goal_px = self._obstacle_map._xy_to_px(goal[None, :])[0].astype(int)

#     path = _astar(occ_grid, start_px, goal_px)
#     if not path or len(path) < 2:
#         print(f"[astar] no path found from {start_px} to {goal_px}")
#         return TorchActionIDs.MOVE_FORWARD

#     lookahead_px = max(2, int(0.5 * pixels_per_meter))
#     waypoint_idx = min(lookahead_px, len(path) - 1)
#     next_px = np.array(path[waypoint_idx])
#     next_xy = self._obstacle_map._px_to_xy(next_px[None, :])[0]
#     _, waypoint_theta = rho_theta(robot_xy, heading, next_xy)

#     if waypoint_theta > TURN_THRESH_RAD:
#         return TorchActionIDs.TURN_LEFT
#     elif waypoint_theta < -TURN_THRESH_RAD:
#         return TorchActionIDs.TURN_RIGHT
#     else:
#         return TorchActionIDs.MOVE_FORWARD

# BaseObjectNavPolicy._pointnav = _astar_pointnav
# print("[traj] A* navigation patch applied — RL PointNav policy bypassed")

def _best_instance(instances, attr):
    candidates = [m for m in instances if np.count_nonzero(getattr(m, attr, np.array([]))) > 0]
    return candidates[0] if candidates else (instances[-1] if instances else None)

_state = {"episode_id": None, "scene_id": None, "category": None, "positions": []}

from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy

_episode_detector_calls = {"YOLO": 0, "GroundingDINO": 0}

_orig_get_object_detections = BaseObjectNavPolicy._get_object_detections
def _patched_get_object_detections(self, img):
    target_classes = self._target_object.split("|")
    from vlfm.vlm.coco_classes import COCO_CLASSES  # <-- corrected path
    has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
    detector_used = "YOLO" if has_coco else "GroundingDINO"
    _episode_detector_calls[detector_used] += 1
    return _orig_get_object_detections(self, img)

BaseObjectNavPolicy._get_object_detections = _patched_get_object_detections

def _update_episode(env):
    try:
        ep = env.current_episodes()[0]
        category = "unknown"
        goals = getattr(ep, "goals", None)
        if goals:
            g0 = goals[0]
            if not _state.get("_debug_printed"):
                print(f"[DEBUG] type(goals[0])={type(g0)}")
                print(f"[DEBUG] dir(goals[0])={[a for a in dir(g0) if not a.startswith('_')]}")
                if isinstance(g0, dict):
                    print(f"[DEBUG] goals[0] keys={list(g0.keys())}")
                _state["_debug_printed"] = True
            if isinstance(g0, dict):
                category = str(g0.get("object_category", "unknown"))
            else:
                category = str(getattr(g0, "object_category", "unknown"))
        _state.update({
            "episode_id": str(ep.episode_id),
            "scene_id":   str(ep.scene_id),
            "category":   category,
            "positions":  [],
        })
        print(f"[traj] episode {ep.episode_id} — {_state['category']}")
    except Exception as e:
        print(f"[traj] episode update error: {e}")

from vlfm.utils.episode_stats_logger import determine_failure_cause

# add near your other monkey-patches, replacing the current determine_failure_cause import/usage:
from vlfm.utils import episode_stats_logger as _stats_logger

_last_failure_cause = {"value": None}
_orig_log_episode_stats = _stats_logger.log_episode_stats

_pending_episode = {"outcome": None, "info": None}

def _patched_log_episode_stats(episode_id, scene_id, infos):
    result = _orig_log_episode_stats(episode_id, scene_id, infos)
    _last_failure_cause["value"] = result
    print(f"[traj] captured failure_cause: {result}")
    if _pending_episode["outcome"] is not None:
        _save_episode(
            _pending_episode["outcome"],
            _pending_episode["info"],
            _pending_episode["state_snapshot"],
            _pending_episode["detector_calls"],
        )
        _pending_episode["outcome"] = None
        _pending_episode["info"] = None
        _pending_episode["state_snapshot"] = None
        _pending_episode["detector_calls"] = None
    return result

_stats_logger.log_episode_stats = _patched_log_episode_stats

def _save_episode(outcome, info, state_snapshot):
    ep_id = state_snapshot["episode_id"]
    if ep_id is None:
        return

    scene_short = state_snapshot["scene_id"].split("/")[-1].split(".")[0]
    tag = f"{scene_short}_{ep_id}"

    with open(os.path.join(OUTPUT_DIR, f"episode_{tag}.json"), "w") as f:
        json.dump({**state_snapshot, "outcome": outcome}, f, indent=2)

    failure_cause = "did_not_fail" if info.get("success") == 1 else _last_failure_cause["value"]

    _log_episode_csv({
        "run_id": RUN_ID,
        "scene_id": scene_short,
        "episode_id": ep_id,
        "object_category": state_snapshot["category"],
        "outcome": outcome,
        "failure_cause": failure_cause,
        "success": info.get("success"),
        "spl": info.get("spl"),
        "distance_to_goal": info.get("distance_to_goal"),
        "num_steps": len(state_snapshot["positions"]),
        "yolo_calls": detector_calls["YOLO"],
        "grounding_dino_calls": detector_calls["GroundingDINO"],
    })
    # ... rest of the image-saving logic stays the same, just use `tag` as before
# def _save_episode(outcome, info):
#     ep_id = _state["episode_id"]
#     if ep_id is None:
#         return

#     scene_short = _state["scene_id"].split("/")[-1].split(".")[0]
#     with open(os.path.join(OUTPUT_DIR, f"episode_{ep_id}.json"), "w") as f:        json.dump({**_state, "outcome": outcome}, f, indent=2)

#     _log_episode_csv({
#         "run_id": RUN_ID,
#         "scene_id": _state["scene_id"].split("/")[-1].split(".")[0],
#         "episode_id": ep_id,
#         "object_category": _state["category"],
#         "outcome": outcome,
#         "success": info.get("success"),
#         "spl": info.get("spl"),
#         "distance_to_goal": info.get("distance_to_goal"),
#         "num_steps": len(_state["positions"]),
#     })

    obs_map = _best_instance(_obs_instances, "_map")
    val_map = _best_instance(_val_instances, "_value_map")

    if val_map is not None:
        print(f"[traj] value map: max={val_map._value_map.max():.4f}, nonzero={np.count_nonzero(val_map._value_map)}")

    if obs_map is not None:
        try:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"episode_{tag}_obstacle_map.png"),
                        obs_map.visualize())
        except Exception as e:
            print(f"[traj] obstacle map error: {e}")

    if val_map is not None:
        try:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"episode_{tag}_value_map.png"),
                        val_map.visualize(obstacle_map=obs_map))
        except Exception as e:
            print(f"[traj] value map error: {e}")

    print(f"[traj] saved episode {tag} ({len(state_snapshot['positions'])} steps, {outcome})")
    # if obs_map is not None:
    #     try:
    #         cv2.imwrite(os.path.join(OUTPUT_DIR, f"episode_{ep_id}_obstacle_map.png"),
    #                     obs_map.visualize())
    #     except Exception as e:
    #         print(f"[traj] obstacle map error: {e}")

    # if is not None:
    #     try:
    #         cv2.imwrite(os.path.join(OUTPUT_DIR, f"episode_{ep_id}_value_map.png"),
    #                     val_map.visualize(obstacle_map=obs_map))
    #     except Exception as e:
    #         print(f"[traj] value map error: {e}")

    # print(f"[traj] saved episode {ep_id} ({len(_state['positions'])} steps, {outcome})")

_orig_reset = _vec_module.VectorEnv.reset
_orig_step  = _vec_module.VectorEnv.step

_reset_count = [0]

def _patched_reset(self):
    result = _orig_reset(self)
    _update_episode(self)
    _reset_count[0] += 1
    print(f"[traj] reset #{_reset_count[0]} -> episode {_state['episode_id']}")
    return result

def _patched_step(self, data):
    result = _orig_step(self, data)
    try:
        observations, _, dones, infos = zip(*result)
        obs = observations[0] if isinstance(observations, (list, tuple)) else observations
        if isinstance(obs, dict) and "gps" in obs:
            raw = obs["gps"].tolist() if hasattr(obs["gps"], "tolist") else list(obs["gps"])
            _state["positions"].append(
                [float(raw[0]), 0.0, float(raw[1])] if len(raw) == 2 else [float(v) for v in raw]
            )
        if dones[0]:
            outcome = "success" if isinstance(infos[0], dict) and infos[0].get("success", 0) == 1 else "failed"
            _pending_episode["outcome"] = outcome
            _pending_episode["info"] = infos[0] if isinstance(infos[0], dict) else {}
            _pending_episode["state_snapshot"] = dict(_state)  # snapshot BEFORE _update_episode overwrites it
            _update_episode(self)
    except Exception as e:
        print(f"[traj] step error: {e}")
    return result

_vec_module.VectorEnv.reset = _patched_reset
_vec_module.VectorEnv.step  = _patched_step

runpy.run_module("vlfm.run", run_name="__main__", alter_sys=True)