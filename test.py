import json, numpy as np
with open("data/trajectories/episode_Darden_39.json") as f:
    ep = json.load(f)
positions = ep["positions"]
p1 = np.array(positions[len(positions)//4])
p2 = np.array(positions[-1])
print(f"Net displacement, quarter -> end: {np.linalg.norm(p2 - p1):.3f}m")
print(f"Positions at 0, 125, 250, 375, 499: {[positions[i] for i in [0,125,250,375,499]]}")

with open("data/trajectories/episode_Darden_39.json") as f:
    ep = json.load(f)
positions = ep["positions"]
for i in range(0, len(positions), 20):
    print(i, positions[i])