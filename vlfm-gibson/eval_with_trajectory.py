import sys, os, json, runpy
import numpy as np
import cv2
import habitat.core.vector_env as _vec_module
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy

OUTPUT_DIR = "data/trajectories"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def _best_instance(instances, attr):
    candidates = [m for m in instances if np.count_nonzero(getattr(m, attr, np.array([]))) > 0]
    return candidates[0] if candidates else (instances[-1] if instances else None)

_state = {"episode_id": None, "scene_id": None, "category": None, "positions": []}

def _update_episode(env):
    try:
        ep = env.current_episodes()[0]
        _state.update({
            "episode_id": str(ep.episode_id),
            "scene_id":   str(ep.scene_id),
            "category":   str(getattr(ep, "object_category", "unknown")),
            "positions":  [],
        })
        print(f"[traj] episode {ep.episode_id} — {_state['category']}")
    except Exception as e:
        print(f"[traj] episode update error: {e}")

def _save_episode(outcome):
    ep_id = _state["episode_id"]
    if ep_id is None:
        return

    with open(os.path.join(OUTPUT_DIR, f"episode_{ep_id}.json"), "w") as f:
        json.dump({**_state, "outcome": outcome}, f, indent=2)

    obs_map = _best_instance(_obs_instances, "_map")
    val_map = _best_instance(_val_instances, "_value_map")

    if val_map is not None:
        print(f"[traj] value map: max={val_map._value_map.max():.4f}, nonzero={np.count_nonzero(val_map._value_map)}")

    if obs_map is not None:
        try:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"episode_{ep_id}_obstacle_map.png"),
                        obs_map.visualize())
        except Exception as e:
            print(f"[traj] obstacle map error: {e}")

    if val_map is not None:
        try:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"episode_{ep_id}_value_map.png"),
                        val_map.visualize(obstacle_map=obs_map))
        except Exception as e:
            print(f"[traj] value map error: {e}")

    print(f"[traj] saved episode {ep_id} ({len(_state['positions'])} steps, {outcome})")

_orig_reset = _vec_module.VectorEnv.reset
_orig_step  = _vec_module.VectorEnv.step

def _patched_reset(self):
    result = _orig_reset(self)
    _update_episode(self)
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
            _save_episode(outcome)
            _update_episode(self)
    except Exception as e:
        print(f"[traj] step error: {e}")
    return result

_vec_module.VectorEnv.reset = _patched_reset
_vec_module.VectorEnv.step  = _patched_step

runpy.run_module("vlfm.run", run_name="__main__", alter_sys=True)
