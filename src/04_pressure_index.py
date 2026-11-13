"""
04_pressure_index.py
Phase 3: Pressure Index Construction, Validation, Sensitivity, and Separation Analysis
"""
import os
import json
import numpy as np
import pandas as pd

# Set matplotlib config dir to writable local path
os.environ['MPLCONFIGDIR'] = os.path.abspath('data_processed/mpl_config')
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DIR = "data_processed"
OUTPUTS_DIR = "outputs"

print("--- [Phase 3] Pressure Index Construction and Validation ---")

df = pd.read_parquet(os.path.join(PROCESSED_DIR, "delivery_wp_wpa.parquet"))
df_matches = pd.read_parquet(os.path.join(PROCESSED_DIR, "cleaned_matches.parquet"))

# 1. Mathematical Leverage Index Formulation
# Closeness factor: maximized at WP = 0.5, 0 at WP = 0 or 1
closeness = 4.0 * df['win_prob'] * (1.0 - df['win_prob']) # ranges [0, 1], 1 at 0.5

# Resource scarcity factor: as balls run out and wickets fall, individual delivery variance has outsized impact
balls_factor = np.sqrt(120.0 / np.maximum(1.0, df['balls_remaining']))
wickets_factor = 10.0 / np.maximum(1.0, df['wickets_in_hand'])

# In 2nd innings, RRR strain adds compounding pressure
rrr_strain = np.where(
    df['innings'] == 2,
    np.clip(df['required_run_rate'] / 8.5, 0.5, 3.0),
    np.clip(df['projected_run_rate'] / 8.0, 0.5, 2.0)
)

raw_leverage = closeness * balls_factor * (0.6 + 0.4 * (wickets_factor / 1.5)) * rrr_strain

# Normalize Leverage Index so average across all deliveries is exactly 1.0
mean_raw_li = raw_leverage.mean()
df['leverage_index'] = raw_leverage / mean_raw_li

# 2. Transparent Composite Pressure Index
# Composite components:
c1 = closeness
c2 = (120.0 - df['balls_remaining']) / 120.0
c3 = df['wickets_lost'] / 10.0
c4 = np.where(
    df['innings'] == 2,
    np.clip((df['required_run_rate'] - 7.5) / 6.0, 0, 1.5),
    np.clip((df['current_run_rate'] - 7.0) / 6.0, 0, 1.5)
)

composite_score = 0.35 * c1 + 0.25 * c2 + 0.20 * c3 + 0.20 * c4
df['composite_pressure'] = composite_score

# Check correlation between Leverage Index and Composite Score
corr_pearson = float(df['leverage_index'].corr(df['composite_pressure']))
corr_spearman = float(df['leverage_index'].corr(df['composite_pressure'], method='spearman'))
print(f"Leverage Index vs Composite Pressure Correlation: Pearson={corr_pearson:.3f}, Spearman={corr_spearman:.3f}")

# 3. Standardized Pressure Quantiles (Percentile Rank within Innings)
df['pressure_percentile'] = df.groupby('innings')['leverage_index'].rank(pct=True) * 100.0

# Define High Pressure at various thresholds: 75th (top 25%), 80th (top 20%), 70th (top 30%), 90th (top 10%)
df['is_high_pressure_25'] = df['pressure_percentile'] >= 75.0
df['is_high_pressure_20'] = df['pressure_percentile'] >= 80.0
df['is_high_pressure_30'] = df['pressure_percentile'] >= 70.0
df['is_high_pressure_10'] = df['pressure_percentile'] >= 90.0

# Default operational high pressure flag: Top 25% (P >= 75)
df['is_high_pressure'] = df['is_high_pressure_25']

# 4. Evidence of Separation: High vs Low Pressure
print("\n--- Empirical Separation Analysis (High vs Low Pressure) ---")

def compute_separation_stats(data, flag_col):
    stats = {}
    for flag_val, label in [(True, 'High Pressure'), (False, 'Low/Medium Pressure')]:
        sub = data[data[flag_col] == flag_val]
        legal_sub = sub[sub['is_legal_ball']]
        
        runs_per_ball = sub['total_ball_runs'].mean()
        run_rate = runs_per_ball * 6.0
        dot_pct = (legal_sub['batter_runs'] == 0).mean() * 100.0
        boundary_pct = (legal_sub['is_four'] | legal_sub['is_six']).mean() * 100.0
        four_pct = legal_sub['is_four'].mean() * 100.0
        six_pct = legal_sub['is_six'].mean() * 100.0
        wicket_pct = sub['is_wicket'].mean() * 100.0
        bowler_wicket_pct = sub['is_bowler_wicket'].mean() * 100.0
        abs_wpa_mean = sub['wpa_batting'].abs().mean()
        
        stats[label] = {
            "Deliveries": int(len(sub)),
            "Legal Deliveries": int(len(legal_sub)),
            "Run Rate": round(float(run_rate), 2),
            "Dot Ball %": round(float(dot_pct), 2),
            "Boundary %": round(float(boundary_pct), 2),
            "Four %": round(float(four_pct), 2),
            "Six %": round(float(six_pct), 2),
            "Wicket %": round(float(wicket_pct), 2),
            "Bowler Wicket %": round(float(bowler_wicket_pct), 2),
            "Mean |WPA|": round(float(abs_wpa_mean), 4)
        }
    return stats

sep_stats_25 = compute_separation_stats(df, 'is_high_pressure_25')
print("Separation at Top 25% Threshold:")
for k, v in sep_stats_25.items():
    print(f"  {k}: {v}")

# Sensitivity across thresholds
thresh_results = {}
for thresh, col in [(10, 'is_high_pressure_10'), (20, 'is_high_pressure_20'), (25, 'is_high_pressure_25'), (30, 'is_high_pressure_30')]:
    thresh_results[f"Top_{thresh}%"] = compute_separation_stats(df, col)

with open(os.path.join(OUTPUTS_DIR, "pressure_separation_stats.json"), "w") as f:
    json.dump({
        "correlation": {"pearson": corr_pearson, "spearman": corr_spearman},
        "threshold_sensitivity": thresh_results
    }, f, indent=2)

# 5. Charts
print("\nGenerating Pressure Visualizations...")

# Chart A: Pressure Index Distribution & High-vs-Low Pressure Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# Distribution plot
sns.histplot(df['leverage_index'], bins=60, kde=True, color='#1976D2', ax=axes[0])
axes[0].axvline(df[df['is_high_pressure']]['leverage_index'].min(), color='#D32F2F', linestyle='--', lw=2, label='Top 25% High Pressure Cutoff')
axes[0].set_title('Distribution of Delivery Leverage Index', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Leverage Index (LI)', fontsize=11)
axes[0].set_ylabel('Delivery Count', fontsize=11)
axes[0].set_xlim(0, 5)
axes[0].legend(fontsize=10)
axes[0].grid(True, linestyle=':', alpha=0.6)

# Bar chart of key cricket outcomes: High vs Low Pressure
categories = ['Dot Ball %', 'Boundary %', 'Wicket %', 'Bowler Wicket %']
low_vals = [sep_stats_25['Low/Medium Pressure']['Dot Ball %'], sep_stats_25['Low/Medium Pressure']['Boundary %'], sep_stats_25['Low/Medium Pressure']['Wicket %'], sep_stats_25['Low/Medium Pressure']['Bowler Wicket %']]
high_vals = [sep_stats_25['High Pressure']['Dot Ball %'], sep_stats_25['High Pressure']['Boundary %'], sep_stats_25['High Pressure']['Wicket %'], sep_stats_25['High Pressure']['Bowler Wicket %']]

x = np.arange(len(categories))
width = 0.35
axes[1].bar(x - width/2, low_vals, width, label='Low/Med Pressure (Bottom 75%)', color='#90CAF9')
axes[1].bar(x + width/2, high_vals, width, label='High Pressure (Top 25%)', color='#E91E63')

for i in range(len(categories)):
    axes[1].text(x[i] - width/2, low_vals[i] + 0.5, f"{low_vals[i]}%", ha='center', fontsize=9, fontweight='bold')
    axes[1].text(x[i] + width/2, high_vals[i] + 0.5, f"{high_vals[i]}%", ha='center', fontsize=9, fontweight='bold')

axes[1].set_title('Outcome Dynamics: High Pressure vs Standard Play', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(categories, fontsize=10, fontweight='bold')
axes[1].set_ylabel('Rate (%)', fontsize=11)
axes[1].set_ylim(0, max(low_vals + high_vals) + 6)
axes[1].legend(fontsize=10)
axes[1].grid(True, axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "pressure_distribution_and_comparison.png"))
plt.close()
print("Saved pressure_distribution_and_comparison.png")

# Chart B: Illustrative Match Timelines
# Select a famous close match: e.g. a match with dramatic win probability shifts in Innings 2
close_matches = df[df['innings'] == 2].groupby('match_id').agg(
    max_li=('leverage_index', 'max'),
    balls=('legal_balls_bowled', 'max'),
    final_score=('current_score', 'max')
).reset_index()

close_candidates = close_matches[(close_matches['balls'] >= 110) & (close_matches['max_li'] > 3.0)]
sample_match_id = close_candidates.iloc[0]['match_id']

m_data = df[(df['match_id'] == sample_match_id) & (df['innings'] == 2)].sort_values(['overid', 'over_ball_num'])
m_meta = df_matches[df_matches['match_id'] == sample_match_id].iloc[0]

fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)
color = '#1565C0'
ax1.set_xlabel('Legal Delivery Number (Innings 2 Chase)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Win Probability (Batting Team)', color=color, fontsize=11, fontweight='bold')
ax1.plot(m_data['legal_balls_bowled'], m_data['win_prob'], color=color, lw=2.5, label='Batting Win Prob (WP)')
ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(-0.05, 1.05)

ax2 = ax1.twinx()
color = '#C2185B'
ax2.set_ylabel('Leverage / Pressure Index', color=color, fontsize=11, fontweight='bold')
ax2.plot(m_data['legal_balls_bowled'], m_data['leverage_index'], color=color, lw=2, linestyle='--', label='Leverage Index (LI)')
ax2.axhline(df[df['is_high_pressure']]['leverage_index'].min(), color='#D32F2F', linestyle=':', label='High Pressure Cutoff')
ax2.tick_params(axis='y', labelcolor=color)

# Mark wickets
wickets_m = m_data[m_data['is_wicket'] == True]
ax1.scatter(wickets_m['legal_balls_bowled'], wickets_m['win_prob'], color='red', s=70, zorder=5, label='Wicket Fallen')

plt.title(f"Illustrative Match Timeline: {m_meta['home_team']} vs {m_meta['away_team']} ({m_meta['match_date']})\nMatch Result: {m_meta['result']}", fontsize=13, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "illustrative_match_timeline.png"))
plt.close()
print("Saved illustrative_match_timeline.png")

# Save dataset with pressure metrics
df.to_parquet(os.path.join(PROCESSED_DIR, "delivery_pressure_complete.parquet"), index=False)
print("Saved delivery_pressure_complete.parquet successfully.")
