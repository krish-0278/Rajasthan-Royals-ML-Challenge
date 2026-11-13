import os
import pandas as pd
import numpy as np

data_dir = "Data Pack"

print("="*60)
print("STEP 0: COMPREHENSIVE DATA AUDIT & VALIDATION")
print("="*60)

# 1. Load competitions & matches
df_comp = pd.read_csv(os.path.join(data_dir, "competitions.csv"))
df_matches = pd.read_csv(os.path.join(data_dir, "matches.csv"))
df_teams = pd.read_csv(os.path.join(data_dir, "teams.csv"))
df_players_map = pd.read_csv(os.path.join(data_dir, "competition_team_player_mapping.csv"))
df_xi = pd.read_csv(os.path.join(data_dir, "playing_eleven.csv"))

print(f"Competitions: {len(df_comp)} rows")
print(f"Competitions format value counts:\n{df_comp['match_format'].value_counts(dropna=False)}")
print(f"Competitions season value counts:\n{df_comp['season'].value_counts(dropna=False).sort_index()}")
print()

print(f"Matches: {len(df_matches)} rows")
print(f"Matches null count:\n{df_matches.isnull().sum()}")
print(f"Matches win_team_id null count: {df_matches['win_team_id'].isnull().sum()}")
print(f"Matches with revised target (DLS): {(~df_matches['revised_target'].isnull()).sum()}")
print(f"Matches result types sample:\n{df_matches['result'].value_counts().head(10)}")

# Check ties / no results
no_results = df_matches[df_matches['win_team_id'].isnull()]
print(f"Matches with win_team_id == NaN: {len(no_results)}")
if len(no_results) > 0:
    print(f"Results of matches with no win_team_id:\n{no_results['result'].value_counts()}")

# Check super over matches or ties
tied_matches = df_matches[df_matches['result'].str.contains('Tie|Super Over|tied', case=False, na=False)]
print(f"Tied / Super Over matches count: {len(tied_matches)}")
if len(tied_matches) > 0:
    print(tied_matches[['match_id', 'match_date', 'result', 'win_team_id']].head(5))

# 2. Load ball_by_ball in chunks or full (41MB is small enough to load into memory)
print("\nLoading ball_by_ball.csv...")
df_bbb = pd.read_csv(os.path.join(data_dir, "ball_by_ball.csv"))
print(f"ball_by_ball shape: {df_bbb.shape}")
print(f"ball_by_ball unique matches: {df_bbb['match_id'].nunique()}")
print(f"Matches in matches.csv not in ball_by_ball: {set(df_matches['match_id']) - set(df_bbb['match_id'])}")
print(f"Matches in ball_by_ball not in matches.csv: {set(df_bbb['match_id']) - set(df_matches['match_id'])}")

print("\nInnings distribution in ball_by_ball:")
print(df_bbb['innings'].value_counts().sort_index())

print("\nExtras type distribution in ball_by_ball:")
print(df_bbb['extras_type'].value_counts(dropna=False))

print("\nDismissal type distribution in ball_by_ball:")
print(df_bbb['dismissal_type'].value_counts(dropna=False))

print("\nRuns & Extras analysis:")
print("Sample of rows with extras_type != NaN:")
sample_extras = df_bbb[df_bbb['extras_type'].notnull()][['runs', 'extras_type', 'extras1_runs', 'extras_2_runs', 'is_four', 'is_six', 'is_wicket']].head(10)
print(sample_extras)

print("\nCheck if runs is batter runs or total runs:")
# Check when extras_type is 'wide'
wides = df_bbb[df_bbb['extras_type'] == 'wide']
print(f"When extras_type == 'wide':")
print(f"  runs value counts:\n{wides['runs'].value_counts().head(5)}")
print(f"  extras1_runs value counts:\n{wides['extras1_runs'].value_counts().head(5)}")
print(f"  extras_2_runs value counts:\n{wides['extras_2_runs'].value_counts().head(5)}")

noballs = df_bbb[df_bbb['extras_type'] == 'no ball']
print(f"When extras_type == 'no ball':")
print(f"  runs value counts:\n{noballs['runs'].value_counts().head(5)}")
print(f"  extras1_runs value counts:\n{noballs['extras1_runs'].value_counts().head(5)}")
print(f"  extras_2_runs value counts:\n{noballs['extras_2_runs'].value_counts().head(5)}")

byes = df_bbb[df_bbb['extras_type'] == 'bye']
print(f"When extras_type == 'bye':")
print(f"  runs value counts:\n{byes['runs'].value_counts().head(5)}")
print(f"  extras1_runs value counts:\n{byes['extras1_runs'].value_counts().head(5)}")

legbyes = df_bbb[df_bbb['extras_type'] == 'leg bye']
print(f"When extras_type == 'leg bye':")
print(f"  runs value counts:\n{legbyes['runs'].value_counts().head(5)}")
print(f"  extras1_runs value counts:\n{legbyes['extras1_runs'].value_counts().head(5)}")

penalty = df_bbb[df_bbb['extras_type'] == 'penalty']
print(f"When extras_type == 'penalty': {len(penalty)} rows")

# Total runs scored on a delivery calculation
print("\nCheck overid and over_ball_num:")
print(f"overid min: {df_bbb['overid'].min()}, max: {df_bbb['overid'].max()}")
print(f"over_ball_num min: {df_bbb['over_ball_num'].min()}, max: {df_bbb['over_ball_num'].max()}")

# Legal delivery definition:
# Is wide a legal ball? No.
# Is no ball a legal ball? No.
print(f"Wides count: {(df_bbb['extras_type'] == 'wide').sum()}")
print(f"No balls count: {(df_bbb['extras_type'] == 'no ball').sum()}")

# 3. Check Player mappings & names
print("\nPlayer ID & Name consistency check:")
striker_pairs = df_bbb[['striker_id', 'striker_name']].drop_duplicates()
bowler_pairs = df_bbb[['bowler_id', 'bowler_name']].drop_duplicates()
print(f"Unique striker_id: {df_bbb['striker_id'].nunique()}, unique striker_name: {df_bbb['striker_name'].nunique()}")
print(f"Unique bowler_id: {df_bbb['bowler_id'].nunique()}, unique bowler_name: {df_bbb['bowler_name'].nunique()}")

# Are there any player_ids with multiple names?
id_to_names = df_bbb.groupby('striker_id')['striker_name'].nunique()
multi_name_ids = id_to_names[id_to_names > 1]
print(f"striker_ids with >1 name in ball_by_ball: {len(multi_name_ids)}")
if len(multi_name_ids) > 0:
    print(multi_name_ids.head(5))

# Check competition_team_player_mapping
print(f"\nUnique players in player mapping: {df_players_map['player_id'].nunique()}")
print(f"Unique names in player mapping: {df_players_map['player_name'].nunique()}")

# Check joins
print("\nJoin checks:")
m_comp = df_matches.merge(df_comp, on='comp_id', how='left')
print(f"Matches merged with Competitions: {len(m_comp)} (original {len(df_matches)}) - Any null comp_name? {m_comp['comp_name'].isnull().sum()}")

bbb_m = df_bbb.merge(m_comp, on='match_id', how='left')
print(f"BBB merged with Matches: {len(bbb_m)} (original {len(df_bbb)}) - Any null match_date? {bbb_m['match_date'].isnull().sum()}")

print("\nCompetitions representation in BBB:")
print(bbb_m['match_format'].value_counts())
print(bbb_m.groupby(['match_format', 'season'])['match_id'].nunique())

