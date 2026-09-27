import json
import numpy as np

with open("data/trajectories/episode_Darden_39.json") as f:
    ep = json.load(f)

positions = ep["positions"]
print(f"Total positions logged: {len(positions)}")
print(f"Position at step 50: {positions[50]}")
print(f"Position at step 250: {positions[250]}")
print(f"Position at step 499: {positions[-1]}")

p1 = np.array(positions[90])
p2 = np.array(positions[-1])
print(f"Net displacement, step 90 -> step 499: {np.linalg.norm(p2 - p1):.3f}m")
