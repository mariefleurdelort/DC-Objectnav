import pandas as pd

df = pd.read_csv("data/trajectories/per_episode_results.csv")
RUN_ID = "<paste from step 3>"

darden = df[(df.scene_id == "Darden") & (df.run_id == RUN_ID)]
print(f"Total: {len(darden)}")
print(f"Raw success rate: {darden.success.mean()*100:.1f}% ({int(darden.success.sum())}/{len(darden)})")

feasible = darden[~darden.failure_cause.str.contains("infeasible", na=False)]
print(f"Feasible-only: {feasible.success.mean()*100:.1f}% ({int(feasible.success.sum())}/{len(feasible)})")
print(f"Excluded as infeasible: {len(darden) - len(feasible)}")

print(darden.failure_cause.value_counts())