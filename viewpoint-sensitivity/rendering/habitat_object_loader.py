"""Dataset-agnostic habitat-sim helpers: load a single object (from any glb
path) as the only rigid body on a bare stage, read its world-space AABB, and
remove it again"""
import numpy as np
import quaternion  # numpy-quaternion, pulled in transitively by habitat-sim
from habitat_sim.physics import MotionType


def look_at_quaternion(eye: np.ndarray, target: np.ndarray, up=(0.0, 1.0, 0.0)):
    """Habitat/OpenGL camera convention: local -Z is forward, +Y is up, +X is right."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up = np.array(up, dtype=np.float64)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        # forward nearly parallel to up; perturb up to avoid a degenerate basis
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    # columns = world-space images of local (x=right, y=up, z=back=-forward)
    rot_matrix = np.stack([right, true_up, -forward], axis=1)
    return quaternion.from_rotation_matrix(rot_matrix)


def load_object_on_bare_stage(sim, model, semantic_id: int = 1):
    """Registers `model` (needs .model_id, .glb_path) as an object template and
    instantiates it at the world origin on an already-empty ("NONE" scene)
    simulator. Returns the ManagedRigidObject handle.

    Caller is responsible for removing the object before loading the next
    model -- see remove_object below -- so only one object is ever resident
    at a time.
    """
    obj_templates_mgr = sim.get_object_template_manager()
    template = obj_templates_mgr.create_new_template(model.model_id)
    template.render_asset_handle = str(model.glb_path)
    template.collision_asset_handle = str(model.glb_path)
    template_id = obj_templates_mgr.register_template(template, model.model_id)

    rigid_obj_mgr = sim.get_rigid_object_manager()
    rigid_obj = rigid_obj_mgr.add_object_by_template_id(template_id)
    rigid_obj.translation = np.array([0.0, 0.0, 0.0])
    rigid_obj.semantic_id = semantic_id
    rigid_obj.motion_type = MotionType.KINEMATIC  # avoid it falling under gravity

    return rigid_obj


def object_world_aabb(rigid_obj):
    """Returns (center, aabb_diag) in world coordinates for the object as
    currently posed. NOTE: `cumulative_bb` / Range3D API has moved around
    across habitat-sim releases -- verify this against `import habitat_sim;
    help(habitat_sim.scene.SceneNode.cumulative_bb)` for your installed version
    if this raises an AttributeError.
    """
    bb = rigid_obj.root_scene_node.cumulative_bb
    local_center = np.array(bb.center())
    size = np.array(bb.size())
    world_center = np.array(rigid_obj.translation) + local_center
    diag = float(np.linalg.norm(size))
    return world_center, diag


def remove_object(sim, rigid_obj, model):
    # Only the rigid object instance is removed. Its template is left registered
    # (keyed by the model_id, which is unique per model)
    rigid_obj_mgr = sim.get_rigid_object_manager()
    rigid_obj_mgr.remove_object_by_handle(rigid_obj.handle)
