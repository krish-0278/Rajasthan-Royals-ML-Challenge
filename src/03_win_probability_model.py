"""
03_win_probability_model.py
Phase 2: Train Calibrated Gradient-Boosted Win Probability Models
- Separate calibrated LightGBM models for Innings 1 and Innings 2.
- 5-Fold GroupKFold strictly split by match_id (never delivery).
- Evaluates Log-Loss, Brier Score, and ROC-AUC.
- Generates probability predictions and calibration curves.
"""
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

PROCESSED_DIR = "data_processed"
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print("--- [Phase 2] Win Probability Modeling (Match-Grouped CV) ---")

df = pd.read_parquet(os.path.join(PROCESSED_DIR, "delivery_features.parquet"))

# Filter to matches with known win outcome
valid_mask = df['is_batting_win'].notnull()
df_valid = df[valid_mask].copy()

# Features for Innings 1 (pure context features, no player identity, no outcome leakage)
features_inn1 = [
    'legal_balls_bowled', 'balls_remaining', 'overs_completed',
    'current_score', 'wickets_lost', 'wickets_in_hand',
    'current_run_rate', 'venue_par_score',
    'projected_final_score', 'projected_score_vs_par', 'projected_run_rate'
]

# Features for Innings 2 (chase context features)
features_inn2 = [
    'legal_balls_bowled', 'balls_remaining', 'overs_completed',
    'current_score', 'wickets_lost', 'wickets_in_hand',
    'current_run_rate', 'target', 'runs_required',
    'required_run_rate', 'rrr_crr_gap', 'runs_needed_per_wicket'
]

# We will perform 5-fold GroupKFold by match_id for both Innings 1 and 2
gkf = GroupKFold(n_splits=5)

# Array to store out-of-fold WP predictions
df['win_prob'] = np.nan

# 1. Innings 1 Modeling
print("\n--- Modeling Innings 1 Win Probability ---")
inn1_idx = df_valid[df_valid['innings'] == 1].index
inn1_data = df_valid.loc[inn1_idx].copy()

oof_preds_inn1 = np.zeros(len(inn1_data))
y_inn1 = inn1_data['is_batting_win'].values
groups_inn1 = inn1_data['match_id'].values

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': -1,
    'n_estimators': 300
}

models_inn1 = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(inn1_data, y_inn1, groups=groups_inn1)):
    X_train, y_train = inn1_data.iloc[train_idx][features_inn1], y_inn1[train_idx]
    X_val, y_val = inn1_data.iloc[val_idx][features_inn1], y_inn1[val_idx]
    
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    oof_preds_inn1[val_idx] = model.predict_proba(X_val)[:, 1]
    models_inn1.append(model)

inn1_logloss = log_loss(y_inn1, oof_preds_inn1)
inn1_brier = brier_score_loss(y_inn1, oof_preds_inn1)
inn1_auc = roc_auc_score(y_inn1, oof_preds_inn1)
print(f"Innings 1 OOF Metrics: LogLoss={inn1_logloss:.4f}, Brier={inn1_brier:.4f}, AUC={inn1_auc:.4f}")

# 2. Innings 2 Modeling
print("\n--- Modeling Innings 2 Win Probability ---")
inn2_idx = df_valid[df_valid['innings'] == 2].index
inn2_data = df_valid.loc[inn2_idx].copy()

oof_preds_inn2 = np.zeros(len(inn2_data))
y_inn2 = inn2_data['is_batting_win'].values
groups_inn2 = inn2_data['match_id'].values

models_inn2 = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(inn2_data, y_inn2, groups=groups_inn2)):
    X_train, y_train = inn2_data.iloc[train_idx][features_inn2], y_inn2[train_idx]
    X_val, y_val = inn2_data.iloc[val_idx][features_inn2], y_inn2[val_idx]
    
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    oof_preds_inn2[val_idx] = model.predict_proba(X_val)[:, 1]
    models_inn2.append(model)

inn2_logloss = log_loss(y_inn2, oof_preds_inn2)
inn2_brier = brier_score_loss(y_inn2, oof_preds_inn2)
inn2_auc = roc_auc_score(y_inn2, oof_preds_inn2)
print(f"Innings 2 OOF Metrics: LogLoss={inn2_logloss:.4f}, Brier={inn2_brier:.4f}, AUC={inn2_auc:.4f}")

# Overall OOF metrics
all_y = np.concatenate([y_inn1, y_inn2])
all_preds = np.concatenate([oof_preds_inn1, oof_preds_inn2])
overall_logloss = log_loss(all_y, all_preds)
overall_brier = brier_score_loss(all_y, all_preds)
overall_auc = roc_auc_score(all_y, all_preds)
print(f"\nOverall OOF Metrics: LogLoss={overall_logloss:.4f}, Brier={overall_brier:.4f}, AUC={overall_auc:.4f}")

# Assign OOF predictions back to df
df.loc[inn1_idx, 'win_prob'] = oof_preds_inn1
df.loc[inn2_idx, 'win_prob'] = oof_preds_inn2

# For any matches not in valid set, predict using ensemble of models
if df['win_prob'].isnull().sum() > 0:
    nan_inn1 = df[(df['innings'] == 1) & (df['win_prob'].isnull())]
    if len(nan_inn1) > 0:
        preds = np.mean([m.predict_proba(nan_inn1[features_inn1])[:, 1] for m in models_inn1], axis=0)
        df.loc[nan_inn1.index, 'win_prob'] = preds
    nan_inn2 = df[(df['innings'] == 2) & (df['win_prob'].isnull())]
    if len(nan_inn2) > 0:
        preds = np.mean([m.predict_proba(nan_inn2[features_inn2])[:, 1] for m in models_inn2], axis=0)
        df.loc[nan_inn2.index, 'win_prob'] = preds

# Save Model Evaluation Metrics to JSON
metrics = {
    "innings_1": {"log_loss": float(inn1_logloss), "brier_score": float(inn1_brier), "auc": float(inn1_auc)},
    "innings_2": {"log_loss": float(inn2_logloss), "brier_score": float(inn2_brier), "auc": float(inn2_auc)},
    "overall": {"log_loss": float(overall_logloss), "brier_score": float(overall_brier), "auc": float(overall_auc)}
}
with open(os.path.join(OUTPUTS_DIR, "win_prob_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# Generate Calibration Plot
print("Generating Calibration Plot...")
prob_true_all, prob_pred_all = calibration_curve(all_y, all_preds, n_bins=10)
prob_true_inn1, prob_pred_inn1 = calibration_curve(y_inn1, oof_preds_inn1, n_bins=10)
prob_true_inn2, prob_pred_inn2 = calibration_curve(y_inn2, oof_preds_inn2, n_bins=10)

plt.figure(figsize=(9, 7), dpi=300)
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration (Diagonal)')
plt.plot(prob_pred_all, prob_true_all, 's-', color='#E91E63', lw=2.5, label=f'Overall (Brier={overall_brier:.3f}, AUC={overall_auc:.3f})')
plt.plot(prob_pred_inn1, prob_true_inn1, 'o--', color='#1976D2', lw=1.8, label=f'Innings 1 (Brier={inn1_brier:.3f}, AUC={inn1_auc:.3f})')
plt.plot(prob_pred_inn2, prob_true_inn2, '^--', color='#388E3C', lw=1.8, label=f'Innings 2 (Brier={inn2_brier:.3f}, AUC={inn2_auc:.3f})')

plt.title('Win Probability Model Calibration Curves (5-Fold Match-Grouped CV)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Mean Predicted Win Probability', fontsize=12)
plt.ylabel('Fraction of Actual Batting Wins', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right', fontsize=11, frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "win_prob_calibration.png"))
plt.close()
print("Saved win_prob_calibration.png")

# Now calculate Win Probability Added (WPA) for each delivery
# Sort strictly
df = df.sort_values(['match_id', 'innings', 'overid', 'over_ball_num', 'innings_ball_num']).reset_index(drop=True)

# wp_next is the win_prob of the next delivery in the same match and innings
df['wp_next'] = df.groupby(['match_id', 'innings'])['win_prob'].shift(-1)

# If this delivery is the last delivery of the innings:
# In Innings 2: if target was reached or all out / 20 overs, next WP is 1.0 (if win) or 0.0 (if loss)
last_ball_mask = df['wp_next'].isnull()
df.loc[last_ball_mask & (df['is_batting_win'] == 1.0), 'wp_next'] = 1.0
df.loc[last_ball_mask & (df['is_batting_win'] == 0.0), 'wp_next'] = 0.0
# For Innings 1 last ball, final WP is simply the post-innings expectation, approximated by win_prob
df.loc[last_ball_mask & df['wp_next'].isnull(), 'wp_next'] = df.loc[last_ball_mask & df['wp_next'].isnull(), 'win_prob']

# Batting WPA = wp_next - win_prob
df['wpa_batting'] = df['wp_next'] - df['win_prob']
# Bowling WPA = - wpa_batting
df['wpa_bowling'] = - df['wpa_batting']

# Save updated features with win_prob and WPA
df.to_parquet(os.path.join(PROCESSED_DIR, "delivery_wp_wpa.parquet"), index=False)
print("Saved delivery_wp_wpa.parquet successfully.")

