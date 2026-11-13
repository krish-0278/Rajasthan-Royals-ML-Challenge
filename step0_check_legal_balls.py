import os
import pandas as pd

data_dir = "Data Pack"
df_bbb = pd.read_csv(os.path.join(data_dir, "ball_by_ball.csv"))

# Inspect a specific over with wide or overstep
wides_matches = df_bbb[df_bbb['extras_type'] == 'wide']['match_id'].iloc[0]
subset = df_bbb[(df_bbb['match_id'] == wides_matches) & (df_bbb['innings'] == 1) & (df_bbb['overid'] == 1)]
print("Over with wide or normal balls:")
print(subset[['innings_ball_num', 'overid', 'over_ball_num', 'striker_name', 'bowler_name', 'runs', 'extras_type', 'extras1_runs']])

# Check over_ball_num max in overs
legal_overs = df_bbb[~df_bbb['extras_type'].isin(['wide', 'Overstep', 'other', 'Penalty'])]
print("\nLegal balls per over (overid, match_id, innings):")
legal_counts = legal_overs.groupby(['match_id', 'innings', 'overid']).size()
print(f"Max legal balls in an over: {legal_counts.max()}, min: {legal_counts.min()}, distribution:\n{legal_counts.value_counts().head(5)}")

# Check total balls faced by batter
print(f"\nTotal rows: {len(df_bbb)}")
print(f"Legal balls total: {len(legal_overs)}")

