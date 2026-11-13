"""
02_feature_engineering.py
Phase 1: Delivery-level Match State Feature Engineering
Computes pre-delivery context for all deliveries in Innings 1 & 2.
Fits an Expected Final Score (EFS) model for 1st innings projections.
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

PROCESSED_DIR = "data_processed"
OUTPUTS_DIR = "outputs"

print("--- [Phase 1] Feature Engineering: Match State per Delivery ---")

df = pd.read_parquet(os.path.join(PROCESSED_DIR, "cleaned_ball_by_ball.parquet"))

# Filter to regulation innings (1 & 2)
df = df[df['innings'].isin([1, 2])].copy()

# Sort strictly by match, innings, overid, over_ball_num, innings_ball_num
df = df.sort_values(['match_id', 'innings', 'overid', 'over_ball_num', 'innings_ball_num']).reset_index(drop=True)

# Compute 1st innings total runs per match
inn1_totals = df[df['innings'] == 1].groupby('match_id')['total_ball_runs'].sum().reset_index()
inn1_totals.columns = ['match_id', 'inn1_total_runs']

df = df.merge(inn1_totals, on='match_id', how='left')

# Determine target for 2nd innings
# If revised_target is specified, use revised_target, else inn1_total_runs + 1
df['target'] = np.where(
    df['innings'] == 2,
    np.where(df['revised_target'].notnull(), df['revised_target'], df['inn1_total_runs'] + 1),
    np.nan
)

# Total legal balls allocated for the innings
# Default is 120 balls (20 overs). If revised overs specified (in balls), use revised.
df['total_allocated_balls'] = np.where(
    df['innings'] == 1,
    np.where(df['first_inn_revised_overs'].notnull(), df['first_inn_revised_overs'], 120.0),
    np.where(df['second_inn_revised_overs'].notnull(), df['second_inn_revised_overs'], 120.0)
)

# Calculate cumulative state strictly PRIOR to each delivery within (match_id, innings)
print("Computing cumulative pre-delivery state...")
# Cumulative legal balls bowled before this ball
df['legal_ball_int'] = df['is_legal_ball'].astype(int)
df['legal_balls_bowled'] = df.groupby(['match_id', 'innings'])['legal_ball_int'].cumsum() - df['legal_ball_int']

# Balls remaining before this ball
df['balls_remaining'] = np.maximum(0, df['total_allocated_balls'] - df['legal_balls_bowled'])
df['overs_completed'] = df['legal_balls_bowled'] / 6.0

# Cumulative runs scored before this ball
df['current_score'] = df.groupby(['match_id', 'innings'])['total_ball_runs'].cumsum() - df['total_ball_runs']

# Cumulative wickets fallen before this ball
df['wicket_int'] = df['is_wicket'].astype(int)
df['wickets_lost'] = df.groupby(['match_id', 'innings'])['wicket_int'].cumsum() - df['wicket_int']
df['wickets_in_hand'] = np.maximum(0, 10 - df['wickets_lost'])

# Current Run Rate (CRR)
df['current_run_rate'] = np.where(
    df['legal_balls_bowled'] > 0,
    (df['current_score'] / df['legal_balls_bowled']) * 6.0,
    0.0
)

# Match Phase
# Powerplay: legal balls 0..35 (overs 0-5.5)
# Middle: legal balls 36..89 (overs 6-14.5)
# Death: legal balls 90..119+ (overs 15-19.5+)
def get_phase(balls_bowled, total_balls):
    # Scale phase thresholds if innings is shortened
    ratio = total_balls / 120.0
    pp_cutoff = 36 * ratio
    mid_cutoff = 90 * ratio
    if balls_bowled < pp_cutoff:
        return 'Powerplay'
    elif balls_bowled < mid_cutoff:
        return 'Middle'
    else:
        return 'Death'

df['phase'] = [get_phase(b, t) for b, t in zip(df['legal_balls_bowled'], df['total_allocated_balls'])]

# 2nd Innings specific state
df['runs_required'] = np.where(
    df['innings'] == 2,
    np.maximum(0, df['target'] - df['current_score']),
    np.nan
)

df['required_run_rate'] = np.where(
    df['innings'] == 2,
    np.where(df['balls_remaining'] > 0, (df['runs_required'] / df['balls_remaining']) * 6.0, df['runs_required'] * 6.0),
    np.nan
)

df['rrr_crr_gap'] = np.where(
    df['innings'] == 2,
    df['required_run_rate'] - df['current_run_rate'],
    np.nan
)

df['runs_needed_per_wicket'] = np.where(
    df['innings'] == 2,
    df['runs_required'] / np.maximum(1, df['wickets_in_hand']),
    np.nan
)

# 1st Innings Par Score & Expected Final Score Model
# Venue Par Score: median 1st innings total by venue, shrunk towards overall competition mean
overall_mean_inn1 = inn1_totals['inn1_total_runs'].mean()
venue_inn1 = df[df['innings'] == 1].groupby('venue_id')['inn1_total_runs'].agg(['count', 'mean']).reset_index()
# Empirical Bayes shrinkage: par = (n * venue_mean + 10 * overall_mean) / (n + 10)
venue_inn1['venue_par_score'] = (venue_inn1['count'] * venue_inn1['mean'] + 10 * overall_mean_inn1) / (venue_inn1['count'] + 10)

df = df.merge(venue_inn1[['venue_id', 'venue_par_score']], on='venue_id', how='left')
df['venue_par_score'] = df['venue_par_score'].fillna(overall_mean_inn1)

print("Training Expected Final Score (EFS) Model for 1st Innings...")
# Train EFS model using completed 1st innings
inn1_df = df[df['innings'] == 1].copy()
# Filter out matches where final score is unknown/abandoned
inn1_clean = inn1_df[inn1_df['inn1_total_runs'].notnull()].copy()

efs_features = [
    'legal_balls_bowled', 'balls_remaining', 'current_score',
    'wickets_lost', 'wickets_in_hand', 'current_run_rate',
    'venue_par_score'
]

# Use LightGBM Regressor with GroupKFold by match_id
efs_model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.08,
    num_leaves=31,
    random_state=42,
    verbosity=-1
)
efs_model.fit(inn1_clean[efs_features], inn1_clean['inn1_total_runs'])

# Predict projected final score
df['projected_final_score'] = np.where(
    df['innings'] == 1,
    efs_model.predict(df[efs_features]),
    np.nan
)

df['projected_score_vs_par'] = np.where(
    df['innings'] == 1,
    df['projected_final_score'] - df['venue_par_score'],
    np.nan
)

df['projected_run_rate'] = np.where(
    df['innings'] == 1,
    df['projected_final_score'] / 20.0,
    np.nan
)

# Outcome target: batting team win indicator
# 1 if win_team_id == batting_team_id, 0 if win_team_id == bowling_team_id, NaN if no result/abandoned
df['is_batting_win'] = np.where(
    df['win_team_id'].notnull(),
    (df['win_team_id'] == df['batting_team_id']).astype(float),
    np.nan
)

print(f"Total deliveries processed: {len(df)}")
print(f"Innings 1 deliveries: {(df['innings'] == 1).sum()}")
print(f"Innings 2 deliveries: {(df['innings'] == 2).sum()}")
print(f"Matches with valid win outcome: {df[df['is_batting_win'].notnull()]['match_id'].nunique()}")

# Save to parquet
df.to_parquet(os.path.join(PROCESSED_DIR, "delivery_features.parquet"), index=False)
print("Saved feature-engineered dataset to data_processed/delivery_features.parquet")

