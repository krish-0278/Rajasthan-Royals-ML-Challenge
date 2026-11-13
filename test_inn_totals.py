import pandas as pd
import numpy as np

df = pd.read_parquet("data_processed/cleaned_ball_by_ball.parquet")

# Check innings 1 totals vs innings 2 targets
inn1 = df[df['innings'] == 1].groupby('match_id')['total_ball_runs'].sum().reset_index()
inn1.columns = ['match_id', 'inn1_total_runs']

# Sample match
sample_m = df[df['match_id'] == 13511]
print("Match 13511 details:")
print("Innings 1 total runs:", sample_m[sample_m['innings'] == 1]['total_ball_runs'].sum())
print("Innings 2 total runs:", sample_m[sample_m['innings'] == 2]['total_ball_runs'].sum())
print("Winner:", sample_m['win_team_id'].iloc[0], "Batting team inn 1:", sample_m[sample_m['innings'] == 1]['batting_team_id'].iloc[0], "Batting team inn 2:", sample_m[sample_m['innings'] == 2]['batting_team_id'].iloc[0])

# Check revised target in matches
dls_matches = df[df['revised_target'].notnull()][['match_id', 'revised_target', 'first_inn_revised_overs', 'second_inn_revised_overs']].drop_duplicates()
print("\nSample DLS matches:")
print(dls_matches.head(5))

