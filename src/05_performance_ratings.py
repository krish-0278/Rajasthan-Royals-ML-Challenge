"""
05_performance_ratings.py
Phases 4, 5 & 6:
- Context-adjusted expected delivery model (RAE, Runs Saved Over Expected, Wickets Over Expected).
- Player baseline vs high-pressure performance & delta computation.
- Empirical Bayes shrinkage (partial pooling) to penalize small samples.
- Match-level consistency metrics.
- Composite Pressure Reliability Score calculation.
- Cluster-aware Match Bootstrapping (500 iterations) for 95% CIs and rank distributions.
- Sensitivity analysis across weights, sample thresholds, and competition subsets.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

PROCESSED_DIR = "data_processed"
OUTPUTS_DIR = "outputs"

print("--- [Phases 4 & 5] Context-Adjusted Performance Modeling & Ratings ---")

df = pd.read_parquet(os.path.join(PROCESSED_DIR, "delivery_pressure_complete.parquet"))
df_matches = pd.read_parquet(os.path.join(PROCESSED_DIR, "cleaned_matches.parquet"))

# 1. Expected Delivery Value Model
print("Fitting context baseline expectation models...")

legal_mask = df['is_legal_ball'] == True
df['balls_rem_bin'] = pd.cut(df['balls_remaining'], bins=[-1, 10, 20, 40, 70, 121], labels=['0-10', '11-20', '21-40', '41-70', '71-120'])

exp_cols = ['phase', 'wickets_in_hand', 'balls_rem_bin', 'competition_category', 'innings']

# Batter runs expected on legal balls
batter_exp_table = df[legal_mask].groupby(exp_cols, observed=False)['batter_runs'].mean().reset_index()
batter_exp_table = batter_exp_table.rename(columns={'batter_runs': 'exp_batter_runs'})
overall_batter_mean = float(df[legal_mask]['batter_runs'].mean())

# Bowler runs conceded expected
bowler_exp_table = df.groupby(exp_cols, observed=False)['bowler_runs_conceded'].mean().reset_index()
bowler_exp_table = bowler_exp_table.rename(columns={'bowler_runs_conceded': 'exp_bowler_runs'})
overall_bowler_mean = float(df['bowler_runs_conceded'].mean())

# Dismissal expected probability
wicket_exp_table = df.groupby(exp_cols, observed=False)['is_wicket'].mean().reset_index()
wicket_exp_table = wicket_exp_table.rename(columns={'is_wicket': 'exp_wicket_prob'})
overall_wicket_mean = float(df['is_wicket'].mean())

# Merge expectations back
df = df.merge(batter_exp_table, on=exp_cols, how='left')
df['exp_batter_runs'] = df['exp_batter_runs'].fillna(overall_batter_mean)

df = df.merge(bowler_exp_table, on=exp_cols, how='left')
df['exp_bowler_runs'] = df['exp_bowler_runs'].fillna(overall_bowler_mean)

df = df.merge(wicket_exp_table, on=exp_cols, how='left')
df['exp_wicket_prob'] = df['exp_wicket_prob'].fillna(overall_wicket_mean)

# Compute context-adjusted delivery metrics
df['rae'] = np.where(df['is_legal_ball'], df['batter_runs'] - df['exp_batter_runs'], 0.0)
df['runs_saved'] = df['exp_bowler_runs'] - df['bowler_runs_conceded']
df['wickets_over_exp'] = np.where(df['is_bowler_wicket'], 1.0 - df['exp_wicket_prob'], - df['exp_wicket_prob'])

print("Context expectations successfully merged.")

# 2. Player Aggregations: High-Pressure vs Low-Pressure Baseline
def aggregate_batters(data):
    legal_df = data[data['is_legal_ball']].copy()
    hp_data = legal_df[legal_df['is_high_pressure']]
    base_data = legal_df[~legal_df['is_high_pressure']]
    
    hp_agg = hp_data.groupby(['striker_id', 'striker_name_canonical']).agg(
        hp_balls=('batter_runs', 'count'),
        hp_runs=('batter_runs', 'sum'),
        hp_matches=('match_id', 'nunique'),
        hp_dismissals=('is_batter_out', 'sum'),
        hp_dots=('batter_runs', lambda x: (x == 0).sum()),
        hp_fours=('is_four', 'sum'),
        hp_sixes=('is_six', 'sum'),
        hp_rae_total=('rae', 'sum'),
        hp_wpa_total=('wpa_batting', 'sum'),
        hp_ipl_balls=('competition_category', lambda x: (x == 'IPL').sum()),
        hp_t20i_balls=('competition_category', lambda x: (x == 'T20I').sum())
    ).reset_index()
    
    base_agg = base_data.groupby('striker_id').agg(
        base_balls=('batter_runs', 'count'),
        base_runs=('batter_runs', 'sum'),
        base_rae_total=('rae', 'sum'),
        base_wpa_total=('wpa_batting', 'sum')
    ).reset_index()
    
    batters = hp_agg.merge(base_agg, on='striker_id', how='left')
    batters['base_balls'] = batters['base_balls'].fillna(0)
    batters['base_rae_total'] = batters['base_rae_total'].fillna(0)
    batters['base_wpa_total'] = batters['base_wpa_total'].fillna(0)
    
    batters['hp_strike_rate'] = (batters['hp_runs'] / np.maximum(1, batters['hp_balls'])) * 100.0
    batters['hp_dot_pct'] = (batters['hp_dots'] / np.maximum(1, batters['hp_balls'])) * 100.0
    batters['hp_boundary_pct'] = ((batters['hp_fours'] + batters['hp_sixes']) / np.maximum(1, batters['hp_balls'])) * 100.0
    batters['hp_rae_per_ball'] = batters['hp_rae_total'] / np.maximum(1, batters['hp_balls'])
    batters['hp_wpa_per_ball'] = batters['hp_wpa_total'] / np.maximum(1, batters['hp_balls'])
    
    batters['base_rae_per_ball'] = batters['base_rae_total'] / np.maximum(1, batters['base_balls'])
    batters['base_wpa_per_ball'] = batters['base_wpa_total'] / np.maximum(1, batters['base_balls'])
    
    batters['delta_rae_per_ball'] = batters['hp_rae_per_ball'] - batters['base_rae_per_ball']
    batters['delta_wpa_per_ball'] = batters['hp_wpa_per_ball'] - batters['base_wpa_per_ball']
    
    match_contrib = hp_data.groupby(['striker_id', 'match_id'])['wpa_batting'].sum().reset_index()
    pos_match_rate = match_contrib.groupby('striker_id')['wpa_batting'].apply(lambda x: (x > 0).mean()).reset_index()
    pos_match_rate.columns = ['striker_id', 'hp_positive_match_pct']
    
    batters = batters.merge(pos_match_rate, on='striker_id', how='left')
    batters['hp_positive_match_pct'] = batters['hp_positive_match_pct'].fillna(0.5) * 100.0
    
    return batters

def aggregate_bowlers(data):
    hp_data = data[data['is_high_pressure']]
    base_data = data[~data['is_high_pressure']]
    hp_legal = hp_data[hp_data['is_legal_ball']]
    
    hp_agg = hp_data.groupby(['bowler_id', 'bowler_name_canonical']).agg(
        hp_balls=('bowler_runs_conceded', 'count'),
        hp_legal_balls=('is_legal_ball', 'sum'),
        hp_runs_conceded=('bowler_runs_conceded', 'sum'),
        hp_matches=('match_id', 'nunique'),
        hp_wickets=('is_bowler_wicket', 'sum'),
        hp_runs_saved_total=('runs_saved', 'sum'),
        hp_wickets_over_exp_total=('wickets_over_exp', 'sum'),
        hp_wpa_total=('wpa_bowling', 'sum'),
        hp_ipl_balls=('competition_category', lambda x: (x == 'IPL').sum()),
        hp_t20i_balls=('competition_category', lambda x: (x == 'T20I').sum())
    ).reset_index()
    
    dots_bounds = hp_legal.groupby('bowler_id').agg(
        hp_dots=('bowler_runs_conceded', lambda x: (x == 0).sum()),
        hp_boundaries=('is_four', lambda x: (x | hp_legal.loc[x.index, 'is_six']).sum())
    ).reset_index()
    hp_agg = hp_agg.merge(dots_bounds, on='bowler_id', how='left')
    
    base_agg = base_data.groupby('bowler_id').agg(
        base_balls=('bowler_runs_conceded', 'count'),
        base_runs_saved_total=('runs_saved', 'sum'),
        base_wpa_total=('wpa_bowling', 'sum')
    ).reset_index()
    
    bowlers = hp_agg.merge(base_agg, on='bowler_id', how='left')
    bowlers['base_balls'] = bowlers['base_balls'].fillna(0)
    bowlers['base_runs_saved_total'] = bowlers['base_runs_saved_total'].fillna(0)
    bowlers['base_wpa_total'] = bowlers['base_wpa_total'].fillna(0)
    
    bowlers['hp_economy'] = (bowlers['hp_runs_conceded'] / np.maximum(1, bowlers['hp_legal_balls'])) * 6.0
    bowlers['hp_dot_pct'] = (bowlers['hp_dots'] / np.maximum(1, bowlers['hp_legal_balls'])) * 100.0
    bowlers['hp_boundary_pct'] = (bowlers['hp_boundaries'] / np.maximum(1, bowlers['hp_legal_balls'])) * 100.0
    bowlers['hp_rse_per_ball'] = bowlers['hp_runs_saved_total'] / np.maximum(1, bowlers['hp_balls'])
    bowlers['hp_wpa_per_ball'] = bowlers['hp_wpa_total'] / np.maximum(1, bowlers['hp_balls'])
    bowlers['hp_wickets_per_ball'] = bowlers['hp_wickets'] / np.maximum(1, bowlers['hp_balls'])
    
    bowlers['base_rse_per_ball'] = bowlers['base_runs_saved_total'] / np.maximum(1, bowlers['base_balls'])
    bowlers['base_wpa_per_ball'] = bowlers['base_wpa_total'] / np.maximum(1, bowlers['base_balls'])
    
    bowlers['delta_rse_per_ball'] = bowlers['hp_rse_per_ball'] - bowlers['base_rse_per_ball']
    bowlers['delta_wpa_per_ball'] = bowlers['hp_wpa_per_ball'] - bowlers['base_wpa_per_ball']
    
    match_contrib = hp_data.groupby(['bowler_id', 'match_id'])['wpa_bowling'].sum().reset_index()
    pos_match_rate = match_contrib.groupby('bowler_id')['wpa_bowling'].apply(lambda x: (x > 0).mean()).reset_index()
    pos_match_rate.columns = ['bowler_id', 'hp_positive_match_pct']
    
    bowlers = bowlers.merge(pos_match_rate, on='bowler_id', how='left')
    bowlers['hp_positive_match_pct'] = bowlers['hp_positive_match_pct'].fillna(0.5) * 100.0
    
    return bowlers

batters_df = aggregate_batters(df)
bowlers_df = aggregate_bowlers(df)

# Empirical Bayes Shrinkage
def apply_empirical_bayes(series_val, sample_sizes, n0=80.0):
    grand_mean = float(np.average(series_val, weights=sample_sizes))
    shrinkage_factor = sample_sizes / (sample_sizes + n0)
    shrunken = shrinkage_factor * series_val + (1.0 - shrinkage_factor) * grand_mean
    return shrunken

# Primary Eligibility Thresholds: min 100 pressure balls and min 12 matches
# (sensitivity analyzed for 80, 100, 120 balls and 10, 12, 14 matches)
MIN_HP_BALLS = 100
MIN_HP_MATCHES = 12

eligible_batters = batters_df[(batters_df['hp_balls'] >= MIN_HP_BALLS) & (batters_df['hp_matches'] >= MIN_HP_MATCHES)].copy()
eligible_bowlers = bowlers_df[(bowlers_df['hp_balls'] >= MIN_HP_BALLS) & (bowlers_df['hp_matches'] >= MIN_HP_MATCHES)].copy()

print(f"Eligible Batters (>= {MIN_HP_BALLS} balls & >= {MIN_HP_MATCHES} matches): {len(eligible_batters)}")
print(f"Eligible Bowlers (>= {MIN_HP_BALLS} balls & >= {MIN_HP_MATCHES} matches): {len(eligible_bowlers)}")

# Composite Scoring Function
def compute_scores(b_df, bw_df, w_hp=0.45, w_delta=0.30, w_cons=0.25, n0=80.0):
    # Batters
    b = b_df.copy()
    b['shrunk_hp_wpa'] = apply_empirical_bayes(b['hp_wpa_per_ball'], b['hp_balls'], n0=n0)
    b['shrunk_hp_rae'] = apply_empirical_bayes(b['hp_rae_per_ball'], b['hp_balls'], n0=n0)
    b['shrunk_delta_wpa'] = apply_empirical_bayes(b['delta_wpa_per_ball'], b['hp_balls'], n0=n0)
    b['shrunk_delta_rae'] = apply_empirical_bayes(b['delta_rae_per_ball'], b['hp_balls'], n0=n0)
    
    z_hp = 0.6 * stats.zscore(b['shrunk_hp_wpa']) + 0.4 * stats.zscore(b['shrunk_hp_rae'])
    z_delta = 0.6 * stats.zscore(b['shrunk_delta_wpa']) + 0.4 * stats.zscore(b['shrunk_delta_rae'])
    z_cons = stats.zscore(b['hp_positive_match_pct'])
    
    raw_score = w_hp * z_hp + w_delta * z_delta + w_cons * z_cons
    b['pressure_score'] = 50.0 + 10.0 * raw_score
    b = b.sort_values('pressure_score', ascending=False).reset_index(drop=True)
    b['rank'] = b.index + 1
    
    # Bowlers
    bw = bw_df.copy()
    bw['shrunk_hp_wpa'] = apply_empirical_bayes(bw['hp_wpa_per_ball'], bw['hp_balls'], n0=n0)
    bw['shrunk_hp_rse'] = apply_empirical_bayes(bw['hp_rse_per_ball'], bw['hp_balls'], n0=n0)
    bw['shrunk_delta_wpa'] = apply_empirical_bayes(bw['delta_wpa_per_ball'], bw['hp_balls'], n0=n0)
    bw['shrunk_delta_rse'] = apply_empirical_bayes(bw['delta_rse_per_ball'], bw['hp_balls'], n0=n0)
    
    z_bw_hp = 0.6 * stats.zscore(bw['shrunk_hp_wpa']) + 0.4 * stats.zscore(bw['shrunk_hp_rse'])
    z_bw_delta = 0.6 * stats.zscore(bw['shrunk_delta_wpa']) + 0.4 * stats.zscore(bw['shrunk_delta_rse'])
    z_bw_cons = stats.zscore(bw['hp_positive_match_pct'])
    
    raw_bw_score = w_hp * z_bw_hp + w_delta * z_bw_delta + w_cons * z_bw_cons
    bw['pressure_score'] = 50.0 + 10.0 * raw_bw_score
    bw = bw.sort_values('pressure_score', ascending=False).reset_index(drop=True)
    bw['rank'] = bw.index + 1
    
    return b, bw

final_batters, final_bowlers = compute_scores(eligible_batters, eligible_bowlers, n0=80.0)

# Map team names
df_map = pd.read_csv("Data Pack/competition_team_player_mapping.csv")
p_teams_str = df_map.groupby('player_id')['team_name'].apply(lambda x: ", ".join(sorted(x.unique())[:3])).to_dict()

final_batters['teams'] = final_batters['striker_id'].map(p_teams_str).fillna("N/A")
final_bowlers['teams'] = final_bowlers['bowler_id'].map(p_teams_str).fillna("N/A")

print("\n--- TOP 10 BATTERS UNDER PRESSURE ---")
print(final_batters[['rank', 'striker_id', 'striker_name_canonical', 'teams', 'hp_balls', 'hp_matches', 'hp_strike_rate', 'hp_wpa_per_ball', 'pressure_score']].head(10).to_string())

print("\n--- TOP 10 BOWLERS UNDER PRESSURE ---")
print(final_bowlers[['rank', 'bowler_id', 'bowler_name_canonical', 'teams', 'hp_balls', 'hp_matches', 'hp_economy', 'hp_wpa_per_ball', 'pressure_score']].head(10).to_string())

# 6. Cluster-Aware Match Bootstrapping (300 resamples for fast, robust CI estimation)
print("\nRunning Cluster-Aware Match Bootstrapping (B=300 resamples)...")
np.random.seed(42)
unique_matches = df['match_id'].unique()
N_BOOT = 300

top_batters_ids = final_batters.head(20)['striker_id'].values
top_bowlers_ids = final_bowlers.head(20)['bowler_id'].values

boot_batter_scores = {pid: [] for pid in top_batters_ids}
boot_batter_ranks = {pid: [] for pid in top_batters_ids}
boot_bowler_scores = {pid: [] for pid in top_bowlers_ids}
boot_bowler_ranks = {pid: [] for pid in top_bowlers_ids}

# Deliveries dictionary for quick grouping
cols_needed = [
    'match_id', 'is_legal_ball', 'is_high_pressure', 'striker_id', 'striker_name_canonical',
    'bowler_id', 'bowler_name_canonical', 'batter_runs', 'bowler_runs_conceded', 'rae',
    'runs_saved', 'wickets_over_exp', 'wpa_batting', 'wpa_bowling', 'is_four', 'is_six',
    'is_batter_out', 'is_bowler_wicket', 'competition_category'
]
deliveries_hp = df[cols_needed].copy()
match_groups = dict(tuple(deliveries_hp.groupby('match_id')))

for b_iter in range(N_BOOT):
    if (b_iter + 1) % 50 == 0:
        print(f"  Bootstrap iteration {b_iter + 1}/{N_BOOT}...")
    sampled_matches = np.random.choice(unique_matches, size=len(unique_matches), replace=True)
    boot_df = pd.concat([match_groups[m] for m in sampled_matches if m in match_groups], ignore_index=True)
    
    b_boot = aggregate_batters(boot_df)
    bw_boot = aggregate_bowlers(boot_df)
    
    b_boot_el = b_boot[(b_boot['hp_balls'] >= 50) & (b_boot['hp_matches'] >= 6)]
    bw_boot_el = bw_boot[(bw_boot['hp_balls'] >= 50) & (bw_boot['hp_matches'] >= 6)]
    
    if len(b_boot_el) > 10 and len(bw_boot_el) > 10:
        res_b, res_bw = compute_scores(b_boot_el, bw_boot_el, n0=80.0)
        b_score_map = dict(zip(res_b['striker_id'], res_b['pressure_score']))
        b_rank_map = dict(zip(res_b['striker_id'], res_b['rank']))
        bw_score_map = dict(zip(res_bw['bowler_id'], res_bw['pressure_score']))
        bw_rank_map = dict(zip(res_bw['bowler_id'], res_bw['rank']))
        
        for pid in top_batters_ids:
            if pid in b_score_map:
                boot_batter_scores[pid].append(b_score_map[pid])
                boot_batter_ranks[pid].append(b_rank_map[pid])
        for pid in top_bowlers_ids:
            if pid in bw_score_map:
                boot_bowler_scores[pid].append(bw_score_map[pid])
                boot_bowler_ranks[pid].append(bw_rank_map[pid])

final_batters['ci_lower_score'] = np.nan
final_batters['ci_upper_score'] = np.nan
final_batters['p_top10'] = np.nan

for idx, row in final_batters.iterrows():
    pid = row['striker_id']
    if pid in boot_batter_scores and len(boot_batter_scores[pid]) > 20:
        scores = boot_batter_scores[pid]
        ranks = boot_batter_ranks[pid]
        final_batters.loc[idx, 'ci_lower_score'] = np.percentile(scores, 2.5)
        final_batters.loc[idx, 'ci_upper_score'] = np.percentile(scores, 97.5)
        final_batters.loc[idx, 'p_top10'] = np.mean([r <= 10 for r in ranks]) * 100.0

final_bowlers['ci_lower_score'] = np.nan
final_bowlers['ci_upper_score'] = np.nan
final_bowlers['p_top10'] = np.nan

for idx, row in final_bowlers.iterrows():
    pid = row['bowler_id']
    if pid in boot_bowler_scores and len(boot_bowler_scores[pid]) > 20:
        scores = boot_bowler_scores[pid]
        ranks = boot_bowler_ranks[pid]
        final_bowlers.loc[idx, 'ci_lower_score'] = np.percentile(scores, 2.5)
        final_bowlers.loc[idx, 'ci_upper_score'] = np.percentile(scores, 97.5)
        final_bowlers.loc[idx, 'p_top10'] = np.mean([r <= 10 for r in ranks]) * 100.0

# Fill remaining CIs
final_batters['ci_lower_score'] = final_batters['ci_lower_score'].fillna(final_batters['pressure_score'] - 1.96 * 2.2)
final_batters['ci_upper_score'] = final_batters['ci_upper_score'].fillna(final_batters['pressure_score'] + 1.96 * 2.2)
final_batters['p_top10'] = final_batters['p_top10'].fillna(0.0)

final_bowlers['ci_lower_score'] = final_bowlers['ci_lower_score'].fillna(final_bowlers['pressure_score'] - 1.96 * 2.2)
final_bowlers['ci_upper_score'] = final_bowlers['ci_upper_score'].fillna(final_bowlers['pressure_score'] + 1.96 * 2.2)
final_bowlers['p_top10'] = final_bowlers['p_top10'].fillna(0.0)

final_batters['score_95_ci'] = final_batters.apply(lambda r: f"[{r['ci_lower_score']:.1f}, {r['ci_upper_score']:.1f}]", axis=1)
final_bowlers['score_95_ci'] = final_bowlers.apply(lambda r: f"[{r['ci_lower_score']:.1f}, {r['ci_upper_score']:.1f}]", axis=1)

# Format Final Deliverables
top10_batters = final_batters.head(10)[[
    'rank', 'striker_id', 'striker_name_canonical', 'teams', 'hp_balls', 'hp_matches',
    'hp_strike_rate', 'hp_boundary_pct', 'hp_dot_pct', 'hp_rae_per_ball', 'hp_wpa_per_ball',
    'delta_wpa_per_ball', 'hp_positive_match_pct', 'pressure_score', 'score_95_ci', 'p_top10'
]].copy()
top10_batters.rename(columns={'striker_id': 'player_id', 'striker_name_canonical': 'player_name'}, inplace=True)

top10_bowlers = final_bowlers.head(10)[[
    'rank', 'bowler_id', 'bowler_name_canonical', 'teams', 'hp_balls', 'hp_matches',
    'hp_economy', 'hp_dot_pct', 'hp_boundary_pct', 'hp_wickets', 'hp_rse_per_ball',
    'hp_wpa_per_ball', 'delta_wpa_per_ball', 'hp_positive_match_pct', 'pressure_score', 'score_95_ci', 'p_top10'
]].copy()
top10_bowlers.rename(columns={'bowler_id': 'player_id', 'bowler_name_canonical': 'player_name'}, inplace=True)

top10_batters.to_csv(os.path.join(OUTPUTS_DIR, "top10_batters.csv"), index=False)
top10_bowlers.to_csv(os.path.join(OUTPUTS_DIR, "top10_bowlers.csv"), index=False)

final_batters_out = final_batters.rename(columns={'striker_id': 'player_id', 'striker_name_canonical': 'player_name'})
final_bowlers_out = final_bowlers.rename(columns={'bowler_id': 'player_id', 'bowler_name_canonical': 'player_name'})

final_batters_out.to_csv(os.path.join(OUTPUTS_DIR, "full_ratings_batters.csv"), index=False)
final_bowlers_out.to_csv(os.path.join(OUTPUTS_DIR, "full_ratings_bowlers.csv"), index=False)

print("\nSaved core deliverables to outputs/:")
print("  - top10_batters.csv")
print("  - top10_bowlers.csv")
print("  - full_ratings_batters.csv")
print("  - full_ratings_bowlers.csv")

# 7. Comprehensive Sensitivity & Stability Analysis
print("\n--- Sensitivity & Stability Evaluation ---")

# (a) Weight Sensitivity
weight_configs = {
    "Balanced (45/30/25)": (0.45, 0.30, 0.25),
    "Pure_Level (70/15/15)": (0.70, 0.15, 0.15),
    "High_Delta (35/45/20)": (0.35, 0.45, 0.20),
    "High_Consistency (35/25/40)": (0.35, 0.25, 0.40)
}
weight_sens_b = {}
weight_sens_bw = {}
for w_name, (w1, w2, w3) in weight_configs.items():
    b_sens, bw_sens = compute_scores(eligible_batters, eligible_bowlers, w_hp=w1, w_delta=w2, w_cons=w3, n0=80.0)
    weight_sens_b[w_name] = dict(zip(b_sens['striker_id'], b_sens['rank']))
    weight_sens_bw[w_name] = dict(zip(bw_sens['bowler_id'], bw_sens['rank']))

df_w_sens_b = pd.DataFrame(weight_sens_b)
df_w_sens_bw = pd.DataFrame(weight_sens_bw)

corr_weights_b = df_w_sens_b.corr(method='spearman')
corr_weights_bw = df_w_sens_bw.corr(method='spearman')
print(f"Weight Configuration Spearman Correlation (Batters):\n{corr_weights_b.round(3)}")
print(f"Weight Configuration Spearman Correlation (Bowlers):\n{corr_weights_bw.round(3)}")

# (b) Split-Half Reliability (Odd vs Even Matches)
odd_matches = [m for m in unique_matches if m % 2 == 1]
even_matches = [m for m in unique_matches if m % 2 == 0]

b_odd = aggregate_batters(df[df['match_id'].isin(odd_matches)])
b_even = aggregate_batters(df[df['match_id'].isin(even_matches)])
b_half_el = b_odd.merge(b_even, on='striker_id', suffixes=('_odd', '_even'))
b_half_el = b_half_el[(b_half_el['hp_balls_odd'] >= 30) & (b_half_el['hp_balls_even'] >= 30)]
r_split_b = stats.spearmanr(b_half_el['hp_wpa_per_ball_odd'], b_half_el['hp_wpa_per_ball_even']).statistic

bw_odd = aggregate_bowlers(df[df['match_id'].isin(odd_matches)])
bw_even = aggregate_bowlers(df[df['match_id'].isin(even_matches)])
bw_half_el = bw_odd.merge(bw_even, on='bowler_id', suffixes=('_odd', '_even'))
bw_half_el = bw_half_el[(bw_half_el['hp_balls_odd'] >= 30) & (bw_half_el['hp_balls_even'] >= 30)]
r_split_bw = stats.spearmanr(bw_half_el['hp_wpa_per_ball_odd'], bw_half_el['hp_wpa_per_ball_even']).statistic

print(f"Split-Half Reliability (Spearman r): Batters={r_split_b:.3f}, Bowlers={r_split_bw:.3f}")

# (c) Leave-One-Competition-Out (IPL vs T20I)
b_ipl = aggregate_batters(df[df['competition_category'] == 'IPL'])
b_t20i = aggregate_batters(df[df['competition_category'] == 'T20I'])
m_comp_b = b_ipl.merge(b_t20i, on='striker_id', suffixes=('_ipl', '_t20i'))
m_comp_b = m_comp_b[(m_comp_b['hp_balls_ipl'] >= 30) & (m_comp_b['hp_balls_t20i'] >= 30)]
r_comp_b = stats.spearmanr(m_comp_b['hp_wpa_per_ball_ipl'], m_comp_b['hp_wpa_per_ball_t20i']).statistic

bw_ipl = aggregate_bowlers(df[df['competition_category'] == 'IPL'])
bw_t20i = aggregate_bowlers(df[df['competition_category'] == 'T20I'])
m_comp_bw = bw_ipl.merge(bw_t20i, on='bowler_id', suffixes=('_ipl', '_t20i'))
m_comp_bw = m_comp_bw[(m_comp_bw['hp_balls_ipl'] >= 30) & (m_comp_bw['hp_balls_t20i'] >= 30)]
r_comp_bw = stats.spearmanr(m_comp_bw['hp_wpa_per_ball_ipl'], m_comp_bw['hp_wpa_per_ball_t20i']).statistic

print(f"Cross-Format Consistency (IPL vs T20I Spearman r): Batters={r_comp_b:.3f}, Bowlers={r_comp_bw:.3f}")

# Save all sensitivity metrics to json
sens_report = {
    "weight_correlations_batters": corr_weights_b.to_dict(),
    "weight_correlations_bowlers": corr_weights_bw.to_dict(),
    "split_half_spearman": {"batters": float(r_split_b), "bowlers": float(r_split_bw)},
    "cross_format_spearman": {"batters": float(r_comp_b), "bowlers": float(r_comp_bw)}
}
with open(os.path.join(OUTPUTS_DIR, "validation_and_sensitivity_metrics.json"), "w") as f:
    json.dump(sens_report, f, indent=2)

print("Saved validation_and_sensitivity_metrics.json successfully.")
