"""
01_clean_and_audit.py
Step 0 & 1: Data Audit, Cleansing, Harmonization, and Referential Validation.
"""
import os
import json
import pandas as pd
import numpy as np

DATA_DIR = "Data Pack"
PROCESSED_DIR = "data_processed"
OUTPUTS_DIR = "outputs"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print("--- [Step 0] Loading Datasets ---")
df_comp = pd.read_csv(os.path.join(DATA_DIR, "competitions.csv"))
df_matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
df_teams = pd.read_csv(os.path.join(DATA_DIR, "teams.csv"))
df_map = pd.read_csv(os.path.join(DATA_DIR, "competition_team_player_mapping.csv"))
df_xi = pd.read_csv(os.path.join(DATA_DIR, "playing_eleven.csv"))
df_bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"))

audit_log = []

# Audit Competitions
audit_log.append(f"Competitions: {len(df_comp)} total rows across formats: {df_comp['match_format'].value_counts().to_dict()}")

# Standardize Competition format: 'T20' is IPL, 'T20I' is International T20
df_comp['competition_category'] = df_comp['match_format'].apply(lambda x: 'IPL' if x == 'T20' else 'T20I')

# Audit Matches
audit_log.append(f"Matches: {len(df_matches)} total matches in matches.csv")
abandoned_or_no_result = df_matches[df_matches['win_team_id'].isnull() | df_matches['result'].str.contains('No Result|Abandoned', case=False, na=False)]
audit_log.append(f"Matches abandoned/no-result: {len(abandoned_or_no_result)}")

# Merge Match metadata with Competition metadata
df_matches_clean = df_matches.merge(
    df_comp[['comp_id', 'match_format', 'season', 'competition_category']],
    on='comp_id',
    how='left'
)

# Canonical Player Mapping: create player_id -> canonical player_name
# Use the mapping table to ensure 100% dropdown compatibility
player_canonical = df_map[['player_id', 'player_name']].drop_duplicates().set_index('player_id')['player_name'].to_dict()

# Audit Ball-by-Ball
audit_log.append(f"Ball-by-ball deliveries: {len(df_bbb)} across {df_bbb['match_id'].nunique()} matches")

# 1. Clean deliveries
# Legal ball flag: wides, no-balls (Overstep, other) and penalty are NOT legal balls
df_bbb['is_legal_ball'] = ~df_bbb['extras_type'].isin(['wide', 'Overstep', 'other', 'Penalty'])
legal_count = df_bbb['is_legal_ball'].sum()
audit_log.append(f"Legal deliveries: {legal_count} ({legal_count/len(df_bbb)*100:.2f}%), Non-legal: {len(df_bbb)-legal_count}")

# 2. Runs breakdown
# Batter runs: exactly 'runs'
# Extra runs: 'extras1_runs' + 'extras_2_runs'
# Total runs scored on delivery: runs + extras1_runs + extras_2_runs
df_bbb['batter_runs'] = df_bbb['runs'].fillna(0).astype(int)
df_bbb['extra_runs'] = (df_bbb['extras1_runs'].fillna(0) + df_bbb['extras_2_runs'].fillna(0)).astype(int)
df_bbb['total_ball_runs'] = df_bbb['batter_runs'] + df_bbb['extra_runs']

# Bowler runs conceded: excludes byes, legbyes, and penalty runs
# Bowler concedes batter runs + wides + no-balls
bowler_extra_mask = df_bbb['extras_type'].isin(['wide', 'Overstep', 'other'])
df_bbb['bowler_runs_conceded'] = np.where(
    bowler_extra_mask,
    df_bbb['batter_runs'] + df_bbb['extras1_runs'].fillna(0) + df_bbb['extras_2_runs'].fillna(0),
    np.where(df_bbb['extras_type'].isin(['byes', 'legbyes', 'Penalty']), 0, df_bbb['batter_runs'])
)

# 3. Dismissal Attribution
# Bowler wickets: Caught, Bowled, LBW, Stump, Hit Wicket
# Non-bowler wickets: RunOut, Retire Out, Obstructing the field
BOWLER_WICKET_TYPES = ['Caught', 'Bowled', 'LBW', 'Stump', 'Hit Wicket']
df_bbb['is_bowler_wicket'] = (df_bbb['is_wicket'] == True) & (df_bbb['dismissal_type'].isin(BOWLER_WICKET_TYPES))
df_bbb['is_batter_out'] = (df_bbb['is_wicket'] == True) & (~df_bbb['dismissal_type'].isin(['Retire Out'])) # retire out also counts as wicket in match context
df_bbb['is_run_out'] = df_bbb['dismissal_type'] == 'RunOut'

audit_log.append(f"Wickets: Total={df_bbb['is_wicket'].sum()}, Bowler Wickets={df_bbb['is_bowler_wicket'].sum()}, Non-bowler dismissals={df_bbb['is_wicket'].sum() - df_bbb['is_bowler_wicket'].sum()}")

# 4. Canonical Player Names
df_bbb['striker_name_canonical'] = df_bbb['striker_id'].map(player_canonical).fillna(df_bbb['striker_name'])
df_bbb['bowler_name_canonical'] = df_bbb['bowler_id'].map(player_canonical).fillna(df_bbb['bowler_name'])
df_bbb['non_striker_name_canonical'] = df_bbb['non_striker_id'].map(player_canonical).fillna(df_bbb['non_striker_name'])

# 5. Merge match metadata into ball_by_ball
match_cols = [
    'match_id', 'comp_id', 'match_date', 'season', 'competition_category',
    'match_format', 'stage', 'venue_id', 'venue_country', 'home_team_id',
    'away_team_id', 'toss_win_team_id', 'toss_opted', 'win_team_id',
    'result', 'first_inn_revised_overs', 'second_inn_revised_overs', 'revised_target'
]

df_merged = df_bbb.merge(df_matches_clean[match_cols], on='match_id', how='left')

# Exclude super overs (innings > 2) from standard win probability / rate modelling,
# but keep them tracked for tie records. (Innings 1 and 2 represent regulation T20 play)
audit_log.append(f"Regulation innings deliveries (Innings 1 & 2): {len(df_merged[df_merged['innings'].isin([1, 2])])}, Super over balls: {len(df_merged[df_merged['innings'] > 2])}")

# Save clean tables
df_merged.to_parquet(os.path.join(PROCESSED_DIR, "cleaned_ball_by_ball.parquet"), index=False)
df_matches_clean.to_parquet(os.path.join(PROCESSED_DIR, "cleaned_matches.parquet"), index=False)

# Save player canonical dictionary
with open(os.path.join(PROCESSED_DIR, "player_canonical.json"), "w") as f:
    json.dump({str(k): v for k, v in player_canonical.items()}, f, indent=2)

# Save audit log
with open(os.path.join(OUTPUTS_DIR, "data_cleaning_audit_log.txt"), "w") as f:
    f.write("\n".join(audit_log))

print("\n--- Step 0 Audit Log ---")
for entry in audit_log:
    print(f"  * {entry}")
print("\nCleaned datasets successfully generated in data_processed/ and outputs/")

