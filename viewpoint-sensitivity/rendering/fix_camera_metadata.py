#!/usr/bin/env python
"""Recomputes the correct camera_position and camera_rotation_wxyz for every row from its already-correct object_center,
orbit_radius_m, elevation_deg, and azimuth_deg (a pure deterministic
function"""
import json

import numpy as np

import config as cfg
from habitat_object_loader import look_at_quaternion
from render_multiview import spherical_offset


def main():
    rows = [json.loads(l) for l in open(cfg.MANIFEST_PATH)]
    n_fixed = 0
    for row in rows:
        center = np.array(row["object_center"])
        radius = row["orbit_radius_m"]
        eye = center + spherical_offset(radius, row["elevation_deg"], row["azimuth_deg"])
        rot = look_at_quaternion(eye, center)

        old_pos = row["camera_position"]
        row["camera_position"] = eye.tolist()
        row["camera_rotation_wxyz"] = [rot.w, rot.x, rot.y, rot.z]
        if not np.allclose(old_pos, eye, atol=1e-6):
            n_fixed += 1

    backup_path = cfg.MANIFEST_PATH.with_suffix(".jsonl.bak")
    cfg.MANIFEST_PATH.rename(backup_path)
    with open(cfg.MANIFEST_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"patched {n_fixed}/{len(rows)} rows (camera_position/camera_rotation_wxyz were stale)")
    print(f"original backed up to {backup_path}")


if __name__ == "__main__":
    main()
