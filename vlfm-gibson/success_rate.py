import pandas as pd
import sys
sys.path.insert(0, '.')
from check_results import load_results

run_ids = {
    "Collierville": "20260723_141556",
    "Corozal": "20260723_141556",
    "Darden": "20260723_141947",
    "Markleeville": "20260723_155828",
    "Wiconisco": "20260723_184749",
}

df = load_results()
frames = []
for scene, rid in run_ids.items():
    subset = df[(df.scene_id == scene) & (df.run_id == rid)]
    print(f"{scene}: {len(subset)} episodes, {int(subset.success.sum())} successes ({subset.success.mean()*100:.1f}%)")
    frames.append(subset)

combined = pd.concat(frames, ignore_index=True)
print(f"\n=== Overall ({len(combined)} episodes) ===")
print(f"Success rate: {combined.success.mean()*100:.2f}% ({int(combined.success.sum())}/{len(combined)})")
print(f"\nFailure cause breakdown:")
print(combined.failure_cause.value_counts())

feasible = combined[~combined.failure_cause.str.contains("infeasible", na=False)]
print(f"\nFeasible-only success rate: {feasible.success.mean()*100:.2f}% ({int(feasible.success.sum())}/{len(feasible)})")

