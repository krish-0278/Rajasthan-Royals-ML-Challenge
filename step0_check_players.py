import os
import pandas as pd

data_dir = "Data Pack"
df_map = pd.read_csv(os.path.join(data_dir, "competition_team_player_mapping.csv"))
df_bbb = pd.read_csv(os.path.join(data_dir, "ball_by_ball.csv"))

print("Player mapping table shape:", df_map.shape)
print("Unique player_ids in mapping:", df_map['player_id'].nunique())
print("Unique player_names in mapping:", df_map['player_name'].nunique())

# Check if any player_id has multiple player_names in mapping
multi_names = df_map.groupby('player_id')['player_name'].nunique()
print("Player IDs with >1 name in mapping:", (multi_names > 1).sum())

# Check strikers vs mapping
strikers_not_in_map = set(df_bbb['striker_id']) - set(df_map['player_id'])
print("Strikers not in mapping:", len(strikers_not_in_map))

# Check bowlers vs mapping
bowlers_not_in_map = set(df_bbb['bowler_id']) - set(df_map['player_id'])
print("Bowlers not in mapping:", len(bowlers_not_in_map))

# Compare striker_name in BBB vs player_name in mapping
bbb_strikers = df_bbb[['striker_id', 'striker_name']].drop_duplicates()
m_striker = bbb_strikers.merge(df_map[['player_id', 'player_name']].drop_duplicates(), left_on='striker_id', right_on='player_id', how='left')
diff_names = m_striker[m_striker['striker_name'] != m_striker['player_name']]
print("Strikers where BBB name != mapping name:", len(diff_names))
if len(diff_names) > 0:
    print(diff_names.head(10))

