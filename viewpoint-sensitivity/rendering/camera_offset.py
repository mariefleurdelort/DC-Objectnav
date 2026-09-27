"""Offset-grid application: given a FIXED viewpoint (eye position + rotation
+ orbit_radius), produce a grid of camera poses that shift the object across
the frame via pure XY translation -- the camera's orientation never changes,
only its position moves sideways/up-down in the plane perpendicular to its
(unchanged) viewing direction. This isolates "where in the frame is the
object" from "what angle are we viewing it at" -- panning/tilting the LOOK
DIRECTION (the earlier approach) conflates the two, since rotating the view
also changes the angle at which the object's surfaces are seen.

Grid values are expressed as nominal degrees for continuity with the earlier
rotation-based design (same range, same resolution), but converted to a real
translation distance via orbit_radius * tan(angle) -- the linear sideways
shift that would produce roughly that much apparent angular displacement if
the camera *had* rotated to track it, which it deliberately does not here.
"""
import math

import numpy as np
import quaternion  # numpy-quaternion; a standalone pip package, doesn't require habitat_sim itself

# Same range/resolution as the original pan/tilt design: 12x12, +/-40 degrees-equivalent.
OFFSET_X_STEPS = np.linspace(-40, 40, 12).tolist()
OFFSET_Y_STEPS = np.linspace(-40, 40, 12).tolist()


def offset_eye_xy(eye: np.ndarray, rotation, orbit_radius: float, x_deg: float, y_deg: float):
    """Returns a new eye position, translated sideways/up in the camera's own
    (fixed) right/up plane. `rotation` is returned unchanged -- the caller
    should reuse the exact same rotation object for every grid point, never
    recompute a look-at.
    """
    right_world = quaternion.rotate_vectors(rotation, np.array([1.0, 0.0, 0.0]))
    up_world = quaternion.rotate_vectors(rotation, np.array([0.0, 1.0, 0.0]))

    dx = orbit_radius * math.tan(math.radians(x_deg))
    dy = orbit_radius * math.tan(math.radians(y_deg))

    return eye + right_world * dx + up_world * dy
