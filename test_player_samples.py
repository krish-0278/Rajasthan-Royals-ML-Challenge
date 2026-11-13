import pandas as pd
import numpy as np

df = pd.read_parquet("data_processed/delivery_pressure_complete.parquet")

hp = df[df['is_high_pressure']]
print(f"Total High Pressure deliveries: {len(hp)}")

# Batters
batter_hp = hp[hp['is_legal_ball']].groupby(['striker_id', 'striker_name_canonical']).agg(
    hp_balls=('total_ball_runs', 'count'),
    hp_matches=('match_id', 'nunique'),
    ipl_balls=('competition_category', lambda x: (x == 'IPL').sum()),
    t20i_balls=('competition_category', lambda x: (x == 'T20I').sum())
).reset_index()

print("\nTop 15 Batters by High Pressure balls faced:")
print(batter_hp.sort_values('hp_balls', ascending=False).head(15))

print(f"\nBatters with >= 50 hp balls: {(batter_hp['hp_balls'] >= 50).sum()}")
print(f"Batters with >= 75 hp balls: {(batter_hp['hp_balls'] >= 75).sum()}")
print(f"Batters with >= 100 hp balls: {(batter_hp['hp_balls'] >= 100).sum()}")
print(f"Batters with >= 75 hp balls & >= 10 matches: {((batter_hp['hp_balls'] >= 75) & (batter_hp['hp_matches'] >= 10)).sum()}")

# Bowlers
bowler_hp = hp[hp['is_legal_ball']].groupby(['bowler_id', 'bowler_name_canonical']).agg(
    hp_balls=('bowler_runs_conceded', 'count'),
    hp_matches=('match_id', 'nunique'),
    ipl_balls=('competition_category', lambda x: (x == 'IPL').sum()),
    t20i_balls=('competition_category', lambda x: (x == 'T20I').sum())
).reset_index()

print("\nTop 15 Bowlers by High Pressure balls bowled:")
print(bowler_hp.sort_values('hp_balls', ascending=False).head(15))

print(f"\nBowlers with >= 50 hp balls: {(bowler_hp['hp_balls'] >= 50).sum()}")
print(f"Bowlers with >= 75 hp balls: {(bowler_hp['hp_balls'] >= 75).sum()}")
print(f"Bowlers with >= 100 hp balls: {(bowler_hp['hp_balls'] >= 100).sum()}")
print(f"Bowlers with >= 75 hp balls & >= 10 matches: {((bowler_hp['hp_balls'] >= 75) & (bowler_hp['hp_matches'] >= 10)).sum()}")

