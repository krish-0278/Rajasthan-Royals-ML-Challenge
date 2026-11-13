import pandas as pd

df_map = pd.read_csv('Data Pack/competition_team_player_mapping.csv')
p_dict = df_map[['player_id', 'player_name']].drop_duplicates().set_index('player_id')['player_name'].to_dict()

top_b = pd.read_csv('outputs/top10_batters.csv')
top_bw = pd.read_csv('outputs/top10_bowlers.csv')

print('--- BATTERS VERIFICATION ---')
for idx, r in top_b.iterrows():
    canonical = p_dict.get(r['player_id'])
    match = (r['player_name'] == canonical)
    print(f"Rank {r['rank']}. ID {r['player_id']}: '{r['player_name']}' == '{canonical}' -> {match}")

print('\n--- BOWLERS VERIFICATION ---')
for idx, r in top_bw.iterrows():
    canonical = p_dict.get(r['player_id'])
    match = (r['player_name'] == canonical)
    print(f"Rank {r['rank']}. ID {r['player_id']}: '{r['player_name']}' == '{canonical}' -> {match}")

