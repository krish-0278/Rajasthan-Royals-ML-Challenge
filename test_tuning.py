import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_parquet("data_processed/delivery_pressure_complete.parquet")
df_map = pd.read_csv("Data Pack/competition_team_player_mapping.csv")
p_map = df_map[['player_id', 'player_name']].drop_duplicates().set_index('player_id')['player_name'].to_dict()

# Compute RAE and RSE expectations
legal_mask = df['is_legal_ball'] == True
df['balls_rem_bin'] = pd.cut(df['balls_remaining'], bins=[-1, 10, 20, 40, 70, 121], labels=['0-10', '11-20', '21-40', '41-70', '71-120'])
exp_cols = ['phase', 'wickets_in_hand', 'balls_rem_bin', 'competition_category', 'innings']

# Batter exp
b_exp = df[legal_mask].groupby(exp_cols, observed=False)['batter_runs'].mean().reset_index().rename(columns={'batter_runs': 'exp_b_runs'})
df = df.merge(b_exp, on=exp_cols, how='left')
df['exp_b_runs'] = df['exp_b_runs'].fillna(df[legal_mask]['batter_runs'].mean())
df['rae'] = np.where(df['is_legal_ball'], df['batter_runs'] - df['exp_b_runs'], 0.0)

# Bowler exp
bw_exp = df.groupby(exp_cols, observed=False)['bowler_runs_conceded'].mean().reset_index().rename(columns={'bowler_runs_conceded': 'exp_bw_runs'})
df = df.merge(bw_exp, on=exp_cols, how='left')
df['exp_bw_runs'] = df['exp_bw_runs'].fillna(df['bowler_runs_conceded'].mean())
df['rse'] = df['exp_bw_runs'] - df['bowler_runs_conceded']

# Aggregations
hp_legal = df[df['is_high_pressure'] & df['is_legal_ball']]
base_legal = df[(~df['is_high_pressure']) & df['is_legal_ball']]

b_hp = hp_legal.groupby('striker_id').agg(
    balls=('batter_runs', 'count'),
    matches=('match_id', 'nunique'),
    runs=('batter_runs', 'sum'),
    rae=('rae', 'sum'),
    wpa=('wpa_batting', 'sum'),
    dots=('batter_runs', lambda x: (x == 0).sum()),
    fours=('is_four', 'sum'),
    sixes=('is_six', 'sum')
).reset_index()

b_base = base_legal.groupby('striker_id').agg(
    base_balls=('batter_runs', 'count'),
    base_rae=('rae', 'sum'),
    base_wpa=('wpa_batting', 'sum')
).reset_index()

b_all = b_hp.merge(b_base, on='striker_id', how='left').fillna(0)
b_all['hp_wpa_per_ball'] = b_all['wpa'] / b_all['balls']
b_all['hp_rae_per_ball'] = b_all['rae'] / b_all['balls']
b_all['base_wpa_per_ball'] = b_all['base_wpa'] / np.maximum(1, b_all['base_balls'])
b_all['base_rae_per_ball'] = b_all['base_rae'] / np.maximum(1, b_all['base_balls'])
b_all['delta_wpa'] = b_all['hp_wpa_per_ball'] - b_all['base_wpa_per_ball']
b_all['delta_rae'] = b_all['hp_rae_per_ball'] - b_all['base_rae_per_ball']
b_all['strike_rate'] = b_all['runs'] / b_all['balls'] * 100

# Bowler
hp_all = df[df['is_high_pressure']]
base_all = df[~df['is_high_pressure']]

bw_hp = hp_all.groupby('bowler_id').agg(
    balls=('bowler_runs_conceded', 'count'),
    legal_balls=('is_legal_ball', 'sum'),
    matches=('match_id', 'nunique'),
    runs=('bowler_runs_conceded', 'sum'),
    wickets=('is_bowler_wicket', 'sum'),
    rse=('rse', 'sum'),
    wpa=('wpa_bowling', 'sum')
).reset_index()

bw_base = base_all.groupby('bowler_id').agg(
    base_balls=('bowler_runs_conceded', 'count'),
    base_rse=('rse', 'sum'),
    base_wpa=('wpa_bowling', 'sum')
).reset_index()

bw_all = bw_hp.merge(bw_base, on='bowler_id', how='left').fillna(0)
bw_all['hp_wpa_per_ball'] = bw_all['wpa'] / bw_all['balls']
bw_all['hp_rse_per_ball'] = bw_all['rse'] / bw_all['balls']
bw_all['base_wpa_per_ball'] = bw_all['base_wpa'] / np.maximum(1, bw_all['base_balls'])
bw_all['base_rse_per_ball'] = bw_all['base_rse'] / np.maximum(1, bw_all['base_balls'])
bw_all['delta_wpa'] = bw_all['hp_wpa_per_ball'] - bw_all['base_wpa_per_ball']
bw_all['delta_rse'] = bw_all['hp_rse_per_ball'] - bw_all['base_rse_per_ball']
bw_all['economy'] = bw_all['runs'] / (bw_all['legal_balls'] / 6.0)

# Match consistency
b_match = hp_legal.groupby(['striker_id', 'match_id'])['wpa_batting'].sum().reset_index()
b_cons = b_match.groupby('striker_id')['wpa_batting'].apply(lambda x: (x > 0).mean()).reset_index()
b_cons.columns = ['striker_id', 'pos_match_pct']
b_all = b_all.merge(b_cons, on='striker_id', how='left').fillna(0.5)

bw_match = hp_all.groupby(['bowler_id', 'match_id'])['wpa_bowling'].sum().reset_index()
bw_cons = bw_match.groupby('bowler_id')['wpa_bowling'].apply(lambda x: (x > 0).mean()).reset_index()
bw_cons.columns = ['bowler_id', 'pos_match_pct']
bw_all = bw_all.merge(bw_cons, on='bowler_id', how='left').fillna(0.5)

def rank_players(b_data, bw_data, min_balls=80, min_matches=10, n0=80.0):
    b = b_data[(b_data['balls'] >= min_balls) & (b_data['matches'] >= min_matches)].copy()
    bw = bw_data[(bw_data['balls'] >= min_balls) & (bw_data['matches'] >= min_matches)].copy()
    
    # EB Shrinkage
    b_mean_wpa = np.average(b['hp_wpa_per_ball'], weights=b['balls'])
    b_w = b['balls'] / (b['balls'] + n0)
    b['shrunk_wpa'] = b_w * b['hp_wpa_per_ball'] + (1 - b_w) * b_mean_wpa
    b['shrunk_delta_wpa'] = b_w * b['delta_wpa'] + (1 - b_w) * 0.0
    
    z_hp = 0.6 * stats.zscore(b['shrunk_wpa']) + 0.4 * stats.zscore(b['hp_rae_per_ball'])
    z_delta = stats.zscore(b['shrunk_delta_wpa'])
    z_cons = stats.zscore(b['pos_match_pct'])
    b['score'] = 50 + 10 * (0.45 * z_hp + 0.30 * z_delta + 0.25 * z_cons)
    b['player_name'] = b['striker_id'].map(p_map)
    b = b.sort_values('score', ascending=False).reset_index(drop=True)
    
    bw_mean_wpa = np.average(bw['hp_wpa_per_ball'], weights=bw['balls'])
    bw_w = bw['balls'] / (bw['balls'] + n0)
    bw['shrunk_wpa'] = bw_w * bw['hp_wpa_per_ball'] + (1 - bw_w) * bw_mean_wpa
    bw['shrunk_delta_wpa'] = bw_w * bw['delta_wpa'] + (1 - bw_w) * 0.0
    
    z_bw_hp = 0.6 * stats.zscore(bw['shrunk_wpa']) + 0.4 * stats.zscore(bw['hp_rse_per_ball'])
    z_bw_delta = stats.zscore(bw['shrunk_delta_wpa'])
    z_bw_cons = stats.zscore(bw['pos_match_pct'])
    bw['score'] = 50 + 10 * (0.45 * z_bw_hp + 0.30 * z_bw_delta + 0.25 * z_bw_cons)
    bw['player_name'] = bw['bowler_id'].map(p_map)
    bw = bw.sort_values('score', ascending=False).reset_index(drop=True)
    
    return b, bw

for n0 in [60, 80, 100]:
    b_res, bw_res = rank_players(b_all, bw_all, min_balls=80, min_matches=10, n0=n0)
    print(f"\n=== Top 10 Batters (n0={n0}) ===")
    print(b_res[['player_name', 'balls', 'matches', 'strike_rate', 'score']].head(10).to_string())
    print(f"\n=== Top 10 Bowlers (n0={n0}) ===")
    print(bw_res[['player_name', 'balls', 'matches', 'economy', 'score']].head(10).to_string())

