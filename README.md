# Rajasthan Royals: SupeRR Selector Season 2 - Round 1
## "Identify the Most Reliable Players Under Pressure"

This repository provides an autonomous, fully reproducible, context-adjusted analytics pipeline for quantifying in-game pressure at the ball-by-ball level and ranking the most reliable batters and bowlers under high-leverage situations across IPL and T20 International cricket (2022–2026).

---

### Pipeline Architecture & Execution Flow

The end-to-end codebase is modularized under `src/` and executes deterministically with fixed random seeds (`random_state=42`):

```bash
# Run entire pipeline end-to-end:
python src/run_pipeline.py
```

Alternatively, scripts can be executed sequentially in numbered order:

1. **`src/01_clean_and_audit.py`**:
   - Ingests raw data from `Data Pack/`: `ball_by_ball.csv`, `matches.csv`, `competitions.csv`, `teams.csv`, `competition_team_player_mapping.csv`, `playing_eleven.csv`.
   - Cleans illegal balls (wides, overstep/front-foot no-balls, height no-balls, penalty runs) vs. legal balls faced.
   - Accurately splits batter runs vs. extras, separates bowler dismissals from run-outs/retire outs, and enforces canonical player IDs.
   - Outputs `data_processed/cleaned_ball_by_ball.parquet` and `data_processed/cleaned_matches.parquet`.

2. **`src/02_feature_engineering.py`**:
   - Constructs cumulative pre-delivery match state (balls remaining, wickets in hand, score, CRR, phase).
   - In Innings 2: calculates chase metrics (target, runs required, RRR, RRR−CRR gap).
   - In Innings 1: trains an Expected Final Score (EFS) regression model conditioned on phase, wickets, run rate, and venue par score to model context before a target exists.
   - Outputs `data_processed/delivery_features.parquet`.

3. **`src/03_win_probability_model.py`**:
   - Trains 5-fold match-grouped (GroupKFold by `match_id`) calibrated gradient-boosted models (LightGBM) for 1st and 2nd innings without player identities or outcome leakage.
   - Computes delivery-level Win Probability ($WP$) and Win Probability Added ($WPA$).
   - Generates `outputs/win_prob_calibration.png` and `outputs/win_prob_metrics.json`.

4. **`src/04_pressure_index.py`**:
   - Formulates the mathematical Leverage Index ($LI$) combining win-probability swing volatility, contest closeness ($4WP(1-WP)$), and resource scarcity.
   - Correlates $LI$ with a transparent composite pressure index ($r_s = 0.876$).
   - Establishes the Top 25% threshold for High Pressure and demonstrates empirical separation across scoring rate, dot-ball%, wicket%, and boundary%.
   - Generates `outputs/pressure_distribution_and_comparison.png` and `outputs/illustrative_match_timeline.png`.

5. **`src/05_performance_ratings.py`**:
   - Builds expected delivery models for Runs Above Expected (RAE), Runs Saved Over Expected (RSE), and Wickets Over Expected.
   - Aggregates high-pressure vs. baseline performance and computes elevation deltas ($\Delta$).
   - Applies Empirical Bayes shrinkage ($n_0 = 80$ prior balls) to prevent small-sample distortion.
   - Computes the Composite Pressure Reliability Score (45% Shrunken Performance, 30% Pressure Delta, 25% Match Consistency).
   - Performs 300 match-cluster bootstrap resamples to establish 95% Confidence Intervals and rank probability.
   - Exports `outputs/top10_batters.csv`, `outputs/top10_bowlers.csv`, and full ratings files.

6. **`src/06_validation.py`**:
   - Executes validation, split-half reliability tests, cross-format comparisons, and verification checks.

7. **`src/generate_charts.py`**:
   - Renders publication-grade charts in `outputs/`.

---

### Key Deliverables in `outputs/`

- `top10_batters.csv`: Top 10 batters ranked under pressure with exact canonical names, teams, balls, matches, SR, boundary%, dot%, RAE, WPA, 95% CI, and Top-10 probability.
- `top10_bowlers.csv`: Top 10 bowlers ranked under pressure with exact canonical names, teams, balls, matches, economy, dot%, boundary%, wickets, RSE, WPA, 95% CI, and Top-10 probability.
- `full_ratings_batters.csv`: Complete ratings and metrics for all eligible batters.
- `full_ratings_bowlers.csv`: Complete ratings and metrics for all eligible bowlers.
- `Methodology_Report.docx` and `Methodology_Report.pdf`: Comprehensive 8–12 page technical methodology report.
- High-resolution publication figures (PNG):
  - `win_prob_calibration.png`
  - `pressure_distribution_and_comparison.png`
  - `illustrative_match_timeline.png`
  - `top10_players_bar_ci.png`
  - `baseline_vs_pressure_quadrant.png`
  - `bootstrap_rank_stability.png`
  - `weight_sensitivity_heatmap.png`

---

### Setup & Requirements

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

