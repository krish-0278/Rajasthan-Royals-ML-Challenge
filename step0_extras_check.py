import os
import pandas as pd

data_dir = "Data Pack"
df_bbb = pd.read_csv(os.path.join(data_dir, "ball_by_ball.csv"))

print("=== EXTRAS BREAKDOWN ===")
for et in df_bbb['extras_type'].dropna().unique():
    subset = df_bbb[df_bbb['extras_type'] == et]
    print(f"\nextras_type: '{et}' (count: {len(subset)})")
    print(f"  runs: min={subset['runs'].min()}, max={subset['runs'].max()}, mean={subset['runs'].mean():.2f}")
    print(f"  extras1_runs: min={subset['extras1_runs'].min()}, max={subset['extras1_runs'].max()}, value_counts:\n{subset['extras1_runs'].value_counts().head(3).to_dict()}")
    print(f"  extras_2_runs: min={subset['extras_2_runs'].min()}, max={subset['extras_2_runs'].max()}, value_counts:\n{subset['extras_2_runs'].value_counts().head(3).to_dict()}")
    print(f"  sample row: runs={subset['runs'].iloc[0]}, e1={subset['extras1_runs'].iloc[0]}, e2={subset['extras_2_runs'].iloc[0]}, four={subset['is_four'].iloc[0]}, six={subset['is_six'].iloc[0]}")

print("\n=== REGULAR BALLS (extras_type == NaN) ===")
reg = df_bbb[df_bbb['extras_type'].isnull()]
print(f"Count: {len(reg)}")
print(f"runs distribution:\n{reg['runs'].value_counts().head(10).to_dict()}")
print(f"extras1_runs: {reg['extras1_runs'].value_counts().to_dict()}")
print(f"extras_2_runs: {reg['extras_2_runs'].value_counts().to_dict()}")

# Total delivery runs calculation
# For any delivery: total score increment = batter runs + extras?
# Let's verify by checking innings totals against match scores!

