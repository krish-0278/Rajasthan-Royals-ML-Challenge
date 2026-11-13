"""
generate_charts.py
Generates publication-quality charts for deliverables:
1. Top-10 Batters & Bowlers Bar Charts with 95% Bootstrap Confidence Intervals
2. Baseline vs Pressure Performance Quadrant Plots (Elevation vs Absolute Level)
3. Bootstrap Rank Distribution & Stability Plot
4. Model Weight & Eligibility Sensitivity Heatmap
"""
import os
import json
import numpy as np
import pandas as pd
os.environ['MPLCONFIGDIR'] = os.path.abspath('data_processed/mpl_config')
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUTS_DIR = "outputs"

print("--- Generating Publication Quality Charts ---")

top10_batters = pd.read_csv(os.path.join(OUTPUTS_DIR, "top10_batters.csv"))
top10_bowlers = pd.read_csv(os.path.join(OUTPUTS_DIR, "top10_bowlers.csv"))
full_batters = pd.read_csv(os.path.join(OUTPUTS_DIR, "full_ratings_batters.csv"))
full_bowlers = pd.read_csv(os.path.join(OUTPUTS_DIR, "full_ratings_bowlers.csv"))

# Parse CIs from strings e.g. "[51.6, 77.1]"
def parse_ci(df_col):
    lows = []
    highs = []
    for val in df_col:
        parts = val.strip("[]").split(",")
        lows.append(float(parts[0]))
        highs.append(float(parts[1]))
    return np.array(lows), np.array(highs)

# 1. Top 10 Batters & Bowlers Bar Charts with 95% CIs
fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# Batters Bar Chart
b_low, b_high = parse_ci(top10_batters['score_95_ci'])
b_err = [top10_batters['pressure_score'] - b_low, b_high - top10_batters['pressure_score']]

y_pos_b = np.arange(len(top10_batters))
axes[0].barh(y_pos_b, top10_batters['pressure_score'], xerr=b_err, color='#D81B60', capsize=4, alpha=0.85, edgecolor='black', lw=0.8)
axes[0].set_yticks(y_pos_b)
axes[0].set_yticklabels([f"#{r['rank']} {r['player_name']} ({r['hp_balls']}b)" for _, r in top10_batters.iterrows()], fontsize=10, fontweight='bold')
axes[0].invert_yaxis()
axes[0].set_xlabel('Pressure Reliability Score (with 95% Match Bootstrap CI)', fontsize=11, fontweight='bold')
axes[0].set_title('Top 10 Batters Under Pressure (2022-2026)', fontsize=13, fontweight='bold', pad=12)
axes[0].axvline(50, color='gray', linestyle=':', label='Population Mean (50.0)')
axes[0].set_xlim(40, 85)
axes[0].grid(True, axis='x', linestyle=':', alpha=0.6)
axes[0].legend(loc='lower right', fontsize=10)

for i, score in enumerate(top10_batters['pressure_score']):
    axes[0].text(score + 1.2, i, f"{score:.1f}", va='center', fontsize=9, fontweight='bold', color='#880E4F')

# Bowlers Bar Chart
bw_low, bw_high = parse_ci(top10_bowlers['score_95_ci'])
bw_err = [top10_bowlers['pressure_score'] - bw_low, bw_high - top10_bowlers['pressure_score']]

y_pos_bw = np.arange(len(top10_bowlers))
axes[1].barh(y_pos_bw, top10_bowlers['pressure_score'], xerr=bw_err, color='#1E88E5', capsize=4, alpha=0.85, edgecolor='black', lw=0.8)
axes[1].set_yticks(y_pos_bw)
axes[1].set_yticklabels([f"#{r['rank']} {r['player_name']} ({r['hp_balls']}b)" for _, r in top10_bowlers.iterrows()], fontsize=10, fontweight='bold')
axes[1].invert_yaxis()
axes[1].set_xlabel('Pressure Reliability Score (with 95% Match Bootstrap CI)', fontsize=11, fontweight='bold')
axes[1].set_title('Top 10 Bowlers Under Pressure (2022-2026)', fontsize=13, fontweight='bold', pad=12)
axes[1].axvline(50, color='gray', linestyle=':', label='Population Mean (50.0)')
axes[1].set_xlim(40, 85)
axes[1].grid(True, axis='x', linestyle=':', alpha=0.6)
axes[1].legend(loc='lower right', fontsize=10)

for i, score in enumerate(top10_bowlers['pressure_score']):
    axes[1].text(score + 1.2, i, f"{score:.1f}", va='center', fontsize=9, fontweight='bold', color='#0D47A1')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "top10_players_bar_ci.png"))
plt.close()
print("Saved top10_players_bar_ci.png")

# 2. Baseline vs Pressure Scatter Quadrant Plot (Batting & Bowling)
fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# Batters Quadrant: Base WPA/ball vs HP WPA/ball
axes[0].scatter(full_batters['base_wpa_per_ball'], full_batters['hp_wpa_per_ball'], c='#9E9E9E', alpha=0.5, s=40, label='Other Eligible Batters')
top_b_sub = full_batters[full_batters['rank'] <= 10]
axes[0].scatter(top_b_sub['base_wpa_per_ball'], top_b_sub['hp_wpa_per_ball'], c='#D81B60', s=100, edgecolors='black', label='Top 10 Batters', zorder=5)

for _, r in top_b_sub.iterrows():
    axes[0].annotate(r['player_name'].split()[-1], (r['base_wpa_per_ball'], r['hp_wpa_per_ball']), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

axes[0].axline((0, 0), slope=1, color='blue', linestyle='--', alpha=0.7, label='Equal Performance (Delta=0)')
axes[0].axvline(0, color='gray', linestyle=':', alpha=0.5)
axes[0].axhline(0, color='gray', linestyle=':', alpha=0.5)
axes[0].set_xlabel('Baseline WPA / Ball (Bottom 75% Leverage)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('High-Pressure WPA / Ball (Top 25% Leverage)', fontsize=11, fontweight='bold')
axes[0].set_title('Batters: Baseline vs High-Pressure Performance', fontsize=13, fontweight='bold', pad=12)
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(True, linestyle=':', alpha=0.5)

# Bowlers Quadrant: Base Bowling WPA/ball vs HP Bowling WPA/ball
axes[1].scatter(full_bowlers['base_wpa_per_ball'], full_bowlers['hp_wpa_per_ball'], c='#9E9E9E', alpha=0.5, s=40, label='Other Eligible Bowlers')
top_bw_sub = full_bowlers[full_bowlers['rank'] <= 10]
axes[1].scatter(top_bw_sub['base_wpa_per_ball'], top_bw_sub['hp_wpa_per_ball'], c='#1E88E5', s=100, edgecolors='black', label='Top 10 Bowlers', zorder=5)

for _, r in top_bw_sub.iterrows():
    axes[1].annotate(r['player_name'].split()[-1], (r['base_wpa_per_ball'], r['hp_wpa_per_ball']), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

axes[1].axline((0, 0), slope=1, color='blue', linestyle='--', alpha=0.7, label='Equal Performance (Delta=0)')
axes[1].axvline(0, color='gray', linestyle=':', alpha=0.5)
axes[1].axhline(0, color='gray', linestyle=':', alpha=0.5)
axes[1].set_xlabel('Baseline Bowling WPA / Ball (Bottom 75% Leverage)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('High-Pressure Bowling WPA / Ball (Top 25% Leverage)', fontsize=11, fontweight='bold')
axes[1].set_title('Bowlers: Baseline vs High-Pressure Performance', fontsize=13, fontweight='bold', pad=12)
axes[1].legend(loc='lower right', fontsize=10)
axes[1].grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "baseline_vs_pressure_quadrant.png"))
plt.close()
print("Saved baseline_vs_pressure_quadrant.png")

# 3. Bootstrap Rank Probability & Stability Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

axes[0].barh(np.arange(len(top10_batters)), top10_batters['p_top10'], color='#AD1457', alpha=0.85, edgecolor='black', lw=0.8)
axes[0].set_yticks(np.arange(len(top10_batters)))
axes[0].set_yticklabels([f"#{r['rank']} {r['player_name']}" for _, r in top10_batters.iterrows()], fontsize=10, fontweight='bold')
axes[0].invert_yaxis()
axes[0].set_xlabel('Probability of Being Ranked in Top 10 (%)', fontsize=11, fontweight='bold')
axes[0].set_title('Batters: Bootstrap Probability of Top 10 Rank', fontsize=13, fontweight='bold', pad=12)
axes[0].set_xlim(0, 100)
axes[0].grid(True, axis='x', linestyle=':', alpha=0.6)
for i, p in enumerate(top10_batters['p_top10']):
    axes[0].text(p + 1.5, i, f"{p:.1f}%", va='center', fontsize=9, fontweight='bold')

axes[1].barh(np.arange(len(top10_bowlers)), top10_bowlers['p_top10'], color='#1565C0', alpha=0.85, edgecolor='black', lw=0.8)
axes[1].set_yticks(np.arange(len(top10_bowlers)))
axes[1].set_yticklabels([f"#{r['rank']} {r['player_name']}" for _, r in top10_bowlers.iterrows()], fontsize=10, fontweight='bold')
axes[1].invert_yaxis()
axes[1].set_xlabel('Probability of Being Ranked in Top 10 (%)', fontsize=11, fontweight='bold')
axes[1].set_title('Bowlers: Bootstrap Probability of Top 10 Rank', fontsize=13, fontweight='bold', pad=12)
axes[1].set_xlim(0, 100)
axes[1].grid(True, axis='x', linestyle=':', alpha=0.6)
for i, p in enumerate(top10_bowlers['p_top10']):
    axes[1].text(p + 1.5, i, f"{p:.1f}%", va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "bootstrap_rank_stability.png"))
plt.close()
print("Saved bootstrap_rank_stability.png")

# 4. Weight Sensitivity Heatmap
with open(os.path.join(OUTPUTS_DIR, "validation_and_sensitivity_metrics.json"), "r") as f:
    sens_data = json.load(f)

df_corr_b = pd.DataFrame(sens_data['weight_correlations_batters'])
df_corr_bw = pd.DataFrame(sens_data['weight_correlations_bowlers'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
sns.heatmap(df_corr_b, annot=True, fmt=".3f", cmap="magma", vmin=0.95, vmax=1.0, ax=axes[0], cbar_kws={'label': 'Spearman Rank Correlation'})
axes[0].set_title('Batters: Rank Correlation Across Weight Profiles', fontsize=12, fontweight='bold', pad=12)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20, ha='right', fontsize=9)
axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=9)

sns.heatmap(df_corr_bw, annot=True, fmt=".3f", cmap="viridis", vmin=0.95, vmax=1.0, ax=axes[1], cbar_kws={'label': 'Spearman Rank Correlation'})
axes[1].set_title('Bowlers: Rank Correlation Across Weight Profiles', fontsize=12, fontweight='bold', pad=12)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=20, ha='right', fontsize=9)
axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "weight_sensitivity_heatmap.png"))
plt.close()
print("Saved weight_sensitivity_heatmap.png")

print("All publication charts generated successfully.")

