import pandas as pd
df = pd.read_csv("data/trajectories/per_episode_results_v2.csv")
print(df["failure_cause"].value_counts())
print(f"\nFalse positives: {(df['failure_cause'] == 'false_positive').sum()} / {len(df)}")