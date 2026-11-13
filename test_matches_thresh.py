import pandas as pd
from test_tuning import rank_players, b_all, bw_all

for min_m in [10, 12, 14]:
    b_res, bw_res = rank_players(b_all, bw_all, min_balls=100, min_matches=min_m, n0=80.0)
    print(f"\n================ Min Matches = {min_m} (min_balls=100) ================")
    print("TOP 10 BATTERS:")
    print(b_res[['player_name', 'balls', 'matches', 'strike_rate', 'score']].head(10).to_string())
    print("\nTOP 10 BOWLERS:")
    print(bw_res[['player_name', 'balls', 'matches', 'economy', 'score']].head(10).to_string())

