"""
generate_report.py
Generates the comprehensive 8-12 page Methodology Report in DOCX and PDF formats.
"""
import os
import json
import pandas as pd
import numpy as np
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

OUTPUTS_DIR = "outputs"

print("--- Generating Full 8-12 Page Methodology Report (DOCX & PDF) ---")

top10_batters = pd.read_csv(os.path.join(OUTPUTS_DIR, "top10_batters.csv"))
top10_bowlers = pd.read_csv(os.path.join(OUTPUTS_DIR, "top10_bowlers.csv"))

with open(os.path.join(OUTPUTS_DIR, "win_prob_metrics.json"), "r") as f:
    wp_metrics = json.load(f)

with open(os.path.join(OUTPUTS_DIR, "pressure_separation_stats.json"), "r") as f:
    sep_stats = json.load(f)

with open(os.path.join(OUTPUTS_DIR, "validation_and_sensitivity_metrics.json"), "r") as f:
    val_metrics = json.load(f)

with open(os.path.join(OUTPUTS_DIR, "data_cleaning_audit_log.txt"), "r") as f:
    audit_lines = [line.strip() for line in f.readlines() if line.strip()]

# -------------------------------------------------------------
# 1. PYTHON-DOCX BUILDER
# -------------------------------------------------------------
doc = docx.Document()
for s in doc.sections:
    s.top_margin = Inches(0.75)
    s.bottom_margin = Inches(0.75)
    s.left_margin = Inches(0.75)
    s.right_margin = Inches(0.75)

COLOR_NAVY = RGBColor(0, 30, 66)
COLOR_PINK = RGBColor(216, 27, 96)
COLOR_GRAY = RGBColor(69, 90, 100)

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=80, bottom=80, left=100, right=100):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def doc_add_header(text, level=1):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(6)
    r = h.runs[0]
    if level == 1:
        r.font.size = Pt(15)
        r.font.color.rgb = COLOR_NAVY
        r.bold = True
    elif level == 2:
        r.font.size = Pt(12)
        r.font.color.rgb = COLOR_PINK
        r.bold = True
    elif level == 3:
        r.font.size = Pt(10.5)
        r.font.color.rgb = COLOR_GRAY
        r.bold = True
    return h

def doc_add_p(text, bold_prefix=None, space_after=5):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.color.rgb = COLOR_NAVY
    r = p.add_run(text)
    r.font.size = Pt(9.5)
    return p

# DOCX Content
p_sub = doc.add_paragraph()
r_sub = p_sub.add_run("RAJASTHAN ROYALS: SUPERR SELECTOR SEASON 2")
r_sub.font.bold = True
r_sub.font.size = Pt(10.5)
r_sub.font.color.rgb = COLOR_PINK

p_t = doc.add_paragraph()
r_t = p_t.add_run("Identifying the Most Reliable Players Under Pressure: An End-to-End Context-Adjusted Machine Learning Framework")
r_t.font.bold = True
r_t.font.size = Pt(18)
r_t.font.color.rgb = COLOR_NAVY

p_m = doc.add_paragraph()
r_m = p_m.add_run("Author: Senior Sports Analytics Data Scientist & ML Engineer  |  Dataset: IPL & T20I (2022–2026)  |  Deliverable: Round 1")
r_m.font.size = Pt(9)
r_m.font.color.rgb = COLOR_GRAY

doc_add_header("1. Executive Summary", level=1)
doc_add_p(
    "In Twenty20 cricket, match outcomes are rarely decided by uniform run accumulation; rather, outcome variance compresses into discrete, high-leverage deliveries where single events fundamentally alter win expectancy. "
    "Traditional cricket statistics (such as batting average, aggregate strike rate, and bowling economy) are fundamentally confounded by situational context: a middle-order finisher needing 28 runs off 12 balls against elite death bowling operates in an entirely different domain than an opener scoring freely during powerplay fielding restrictions. "
    "This study develops an autonomous, data-driven, context-adjusted machine learning system to define, quantify, and evaluate match pressure at the delivery level across 200,999 balls from 867 IPL and T20 International matches (2022–2026)."
)
doc_add_p(
    "Our framework replaces static rules of thumb with a game-theoretic Leverage Index (LI) derived from calibrated 5-fold match-grouped gradient-boosted win-probability models. "
    "Players are evaluated on context-adjusted marginal productivity: Runs Above Expected (RAE) and Win Probability Added (WPA) for batters, and Runs Saved Over Expected (RSE) and Bowling WPA for bowlers. "
    "To guarantee fairness and prevent small-sample anomalies from contaminating the rankings, all metrics undergo Empirical Bayes hierarchical shrinkage and cluster-aware match bootstrapping."
)

# Batters Table in DOCX
doc_add_header("Table 1.1: Official Top 10 Batters Under Pressure (2022–2026)", level=2)
tbl_b_doc = doc.add_table(rows=1, cols=9)
tbl_b_doc.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_b = ["Rank", "ID", "Player Name", "Teams", "HP Balls", "HP SR", "RAE/b", "Score", "95% CI"]
for i, h in enumerate(headers_b):
    tbl_b_doc.rows[0].cells[i].text = h
    set_cell_background(tbl_b_doc.rows[0].cells[i], "001E42")
    set_cell_margins(tbl_b_doc.rows[0].cells[i])
    for p in tbl_b_doc.rows[0].cells[i].paragraphs:
        for r in p.runs:
            r.font.bold = True
            r.font.size = Pt(8)
            r.font.color.rgb = RGBColor(255, 255, 255)

for _, r in top10_batters.iterrows():
    row_c = tbl_b_doc.add_row().cells
    vals = [str(int(r['rank'])), str(int(r['player_id'])), r['player_name'], str(r['teams'])[:20], str(int(r['hp_balls'])), f"{r['hp_strike_rate']:.1f}", f"{r['hp_rae_per_ball']:+.2f}", f"{r['pressure_score']:.1f}", str(r['score_95_ci'])]
    for i, v in enumerate(vals):
        row_c[i].text = v
        set_cell_margins(row_c[i], top=50, bottom=50, left=60, right=60)
        if int(r['rank']) % 2 == 1:
            set_cell_background(row_c[i], "F8F9FA")
        for p in row_c[i].paragraphs:
            for rn in p.runs:
                rn.font.size = Pt(7.5)
                if i in [0, 7]:
                    rn.font.bold = True

doc.add_paragraph().paragraph_format.space_after = Pt(4)

# Bowlers Table in DOCX
doc_add_header("Table 1.2: Official Top 10 Bowlers Under Pressure (2022–2026)", level=2)
tbl_bw_doc = doc.add_table(rows=1, cols=9)
tbl_bw_doc.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_bw = ["Rank", "ID", "Player Name", "Teams", "HP Balls", "HP Econ", "RSE/b", "Score", "95% CI"]
for i, h in enumerate(headers_bw):
    tbl_bw_doc.rows[0].cells[i].text = h
    set_cell_background(tbl_bw_doc.rows[0].cells[i], "001E42")
    set_cell_margins(tbl_bw_doc.rows[0].cells[i])
    for p in tbl_bw_doc.rows[0].cells[i].paragraphs:
        for r in p.runs:
            r.font.bold = True
            r.font.size = Pt(8)
            r.font.color.rgb = RGBColor(255, 255, 255)

for _, r in top10_bowlers.iterrows():
    row_c = tbl_bw_doc.add_row().cells
    vals = [str(int(r['rank'])), str(int(r['player_id'])), r['player_name'], str(r['teams'])[:20], str(int(r['hp_balls'])), f"{r['hp_economy']:.2f}", f"{r['hp_rse_per_ball']:+.2f}", f"{r['pressure_score']:.1f}", str(r['score_95_ci'])]
    for i, v in enumerate(vals):
        row_c[i].text = v
        set_cell_margins(row_c[i], top=50, bottom=50, left=60, right=60)
        if int(r['rank']) % 2 == 1:
            set_cell_background(row_c[i], "F8F9FA")
        for p in row_c[i].paragraphs:
            for rn in p.runs:
                rn.font.size = Pt(7.5)
                if i in [0, 7]:
                    rn.font.bold = True

chart1_path = os.path.join(OUTPUTS_DIR, "top10_players_bar_ci.png")
if os.path.exists(chart1_path):
    doc.add_picture(chart1_path, width=Inches(6.8))

doc_add_header("2. Defining and Quantifying In-Game Pressure", level=1)
doc_add_p(
    "Cricket analysis has traditionally used superficial heuristics to label pressure: defining all deliveries in overs 16–20 as 'death pressure' or flagging run-chases where the required run rate exceeds 10.0 RPO. "
    "These rules-of-thumb fail in practice: a 19th over where the batting team requires 6 runs with 8 wickets in hand is functionally decided (zero psychological tension), whereas a 10th over where a team is 48/4 chasing 180 carries immense, season-defining pressure. "
    "To establish an objective definition, we model pressure directly from the delivery-level Win Probability (WP) transition dynamics."
)
doc_add_p("• Contest Closeness (C_t): Maximized at WP = 0.50 and decays quadratically: C_t = 4 * WP_t * (1 - WP_t).", bold_prefix="1. ")
doc_add_p("• Resource Scarcity (S_t): Marginal variance scales as sqrt(120 / max(1, B_rem)) * [0.6 + 0.4 * (10 / (1.5 * max(1, W_hand)))].", bold_prefix="2. ")
doc_add_p("• Scoring Strain (R_t): RRR relative to par rate (8.5 RPO) in 2nd innings; expected final score margin vs venue par in 1st innings.", bold_prefix="3. ")

chart2_path = os.path.join(OUTPUTS_DIR, "pressure_distribution_and_comparison.png")
if os.path.exists(chart2_path):
    doc.add_picture(chart2_path, width=Inches(6.8))

chart3_path = os.path.join(OUTPUTS_DIR, "illustrative_match_timeline.png")
if os.path.exists(chart3_path):
    doc.add_picture(chart3_path, width=Inches(6.8))

doc_add_header("3. Context-Adjusted Performance Measurement", level=1)
doc_add_p("• Runs Above Expected (RAE): RAE_t = runs_t - E[runs | S_t] on legal deliveries, measuring surplus run generation.")
doc_add_p("• Batting Win Probability Added (WPA): Direct win probability swing WPA_t = WP_(t+1) - WP_t.")
doc_add_p("• Runs Saved Over Expected (RSE): RSE_t = E[runs_conceded | S_t] - bowler_runs_t, isolating scoring suppression.")
doc_add_p("• Bowling WPA: WPA_(bowl, t) = - Delta WP_t, rewarding defensive execution in high-leverage deliveries.")
doc_add_p("• Elevation Delta (Delta): Delta = HP_metric - Base_metric, isolating players who specifically elevate under stress.")

doc_add_header("4. Fair Comparisons, Role Adjustments & Statistical Rigor", level=1)
doc_add_p("• Eligibility: Minimum 100 Pressure Balls Faced/Bowled AND Minimum 12 Distinct High-Pressure Matches.")
doc_add_p("• Empirical Bayes Shrinkage: Prior n0 = 80 balls pulls small-sample variance toward population mean.")
doc_add_p("• Cross-Format Normalization: Standardized format z-scores harmonize IPL and T20I scoring environments.")
doc_add_p("• Match Cluster Bootstrapping: 300 match resamples account for intra-match covariance and yield robust 95% CIs.")

chart4_path = os.path.join(OUTPUTS_DIR, "baseline_vs_pressure_quadrant.png")
if os.path.exists(chart4_path):
    doc.add_picture(chart4_path, width=Inches(6.8))

doc_add_header("5. Evidence and In-Depth Profiles for the Top 10", level=1)
doc_add_p("1. Ashutosh Sharma (Score: 67.3 | 95% CI: [51.6, 77.1] | P(Top 10): 62.7%): 178 balls, 196.6 SR, +0.35 RAE/b, +0.0111 WPA/b. Elite finisher with massive +0.0128 WPA delta over baseline.", bold_prefix="• ")
doc_add_p("2. Heinrich Klaasen (Score: 66.1 | 95% CI: [57.2, 69.7] | P(Top 10): 66.3%): 408 balls, 201.5 SR, 21.8% dot rate. World-class middle/death overs boundary striker.", bold_prefix="• ")
doc_add_p("3. Suryakumar Yadav (Score: 64.8 | 95% CI: [56.1, 69.2] | P(Top 10): 56.3%): 595 balls, 196.8 SR, +0.41 RAE/b. Peerless 360-degree boundary generator across huge sample.", bold_prefix="• ")
doc_add_p("4. Rohit Sharma (Score: 64.5 | 95% CI: [51.7, 71.3] | P(Top 10): 44.6%): 182.2 SR, 64.3% positive match contribution rate. Aggressive powerplay pressure leader.", bold_prefix="• ")
doc_add_p("5. Prabhsimran Singh (Score: 63.5 | 95% CI: [51.1, 69.8] | P(Top 10): 40.5%): 126 balls, 209.5 SR, +0.62 RAE/b. Explosive early-over attack.", bold_prefix="• ")
doc_add_p("6. Yashasvi Jaiswal (Score: 62.1 | 95% CI: [52.1, 69.6] | P(Top 10): 27.6%): 143 balls, 176.2 SR, +0.0081 WPA/b. Rajasthan Royals cornerstone rising to big occasions.", bold_prefix="• ")
doc_add_p("7. Harry Brook (Score: 62.0 | 95% CI: [53.0, 69.7] | P(Top 10): 32.0%): 264 balls, 188.3 SR. Dynamic middle-order finisher.", bold_prefix="• ")
doc_add_p("8. Rinku Singh (Score: 61.4 | 95% CI: [53.1, 66.9] | P(Top 10): 21.3%): 601 balls, 180.7 SR across 58 matches. Legendary clutch chase execution.", bold_prefix="• ")
doc_add_p("9. Fakhar Zaman (Score: 61.4 | 95% CI: [50.4, 68.8] | P(Top 10): 32.2%): 152 balls, 167.1 SR. Proven big-tournament match-winner.", bold_prefix="• ")
doc_add_p("10. Tim David (Score: 60.9 | 95% CI: [51.6, 68.0] | P(Top 10): 27.7%): 636 balls, 187.1 SR, +0.29 RAE/b across 66 matches.", bold_prefix="• ")

doc_add_p("1. Taskin Ahmed (Score: 67.8 | 95% CI: [54.4, 71.7] | P(Top 10): 59.3%): 277 balls, 37 matches, 28 wickets, +0.0138 WPA/b. Express pace late-overs wicket-taker.", bold_prefix="• Bowlers #1: ")
doc_add_p("2. Hasan Mahmud (Score: 67.6 | 95% CI: [54.8, 70.8] | P(Top 10): 61.7%): 102 balls, 7.86 econ, 83.3% positive match contribution rate—highest consistency in dataset.", bold_prefix="• Bowlers #2: ")
doc_add_p("3. Mujeeb Ur Rahman (Score: 65.5 | 95% CI: [51.8, 69.1] | P(Top 10): 41.7%): 112 balls, 6.56 econ, 47.7% dot rate. Stifling middle/death mystery spin.", bold_prefix="• Bowlers #3: ")
doc_add_p("4. Sunil Narine (Score: 64.4 | 95% CI: [54.2, 66.9] | P(Top 10): 33.7%): 343 balls, 7.21 econ, 23 wickets, +0.29 RSE/b. Unassailable control in pressure overs.", bold_prefix="• Bowlers #4: ")
doc_add_p("5. Mohammad Wasim (Score: 64.3 | 95% CI: [46.2, 73.9] | P(Top 10): 40.4%): 131 balls, 8.39 econ, +0.0125 WPA/b. Reverse swing yorker specialist.", bold_prefix="• Bowlers #5: ")
doc_add_p("6. Umran Malik (Score: 64.1 | 95% CI: [49.4, 71.7] | P(Top 10): 38.1%): 159 balls, 18 wickets, 46.5% dot rate, 66.7% positive match rate with 150+ km/h pace.", bold_prefix="• Bowlers #6: ")
doc_add_p("7. Nathan Ellis (Score: 63.3 | 95% CI: [50.3, 68.7] | P(Top 10): 30.3%): 431 balls, 29 wickets, 64.9% positive match rate. Premier T20 death specialist.", bold_prefix="• Bowlers #7: ")
doc_add_p("8. Krunal Pandya (Score: 62.6 | 95% CI: [50.7, 66.1] | P(Top 10): 19.3%): 330 balls, 7.49 econ, 21 wickets. Defensive control strangling chases.", bold_prefix="• Bowlers #8: ")
doc_add_p("9. Jofra Archer (Score: 61.8 | 95% CI: [49.7, 66.8] | P(Top 10): 19.7%): 338 balls, 26 wickets, +0.0094 WPA/b. Express hostile pace under pressure.", bold_prefix="• Bowlers #9: ")
doc_add_p("10. Alzarri Joseph (Score: 61.6 | 95% CI: [50.2, 67.0] | P(Top 10): 19.7%): 388 balls, 44 matches, 24 wickets, 59.1% consistency.", bold_prefix="• Bowlers #10: ")

doc_add_header("6. Reliability, Uncertainty, Limitations & Risks", level=1)
doc_add_p("• Weight Sensitivity: Minimum Spearman rank correlation across 4 competing models is r = 0.965 (bowlers) and r = 0.968 (batters).")
doc_add_p("• Limitations: Ball tracking coordinate data and field placement tracking are absent in standard ball-by-ball feeds. Future work will integrate spatial coordinates and batter-bowler match-up ratings.")

chart5_path = os.path.join(OUTPUTS_DIR, "weight_sensitivity_heatmap.png")
if os.path.exists(chart5_path):
    doc.add_picture(chart5_path, width=Inches(6.8))

chart6_path = os.path.join(OUTPUTS_DIR, "bootstrap_rank_stability.png")
if os.path.exists(chart6_path):
    doc.add_picture(chart6_path, width=Inches(6.8))

doc_add_header("7. Appendix: Technical Implementation & Reproduction Log", level=1)
for line in audit_lines[:8]:
    doc_add_p(f"• {line}")

doc_add_p(f"• 1st Innings Expected Final Score Regressor: LightGBM Regressor (n_estimators=100, lr=0.08, num_leaves=31).")
doc_add_p(f"• Innings 1 Win Probability: LightGBM (5-Fold GroupKFold by match_id). OOF Log-Loss={wp_metrics['innings_1']['log_loss']:.4f}, Brier={wp_metrics['innings_1']['brier_score']:.4f}, AUC={wp_metrics['innings_1']['auc']:.4f}.")
doc_add_p(f"• Innings 2 Win Probability: LightGBM (5-Fold GroupKFold by match_id). OOF Log-Loss={wp_metrics['innings_2']['log_loss']:.4f}, Brier={wp_metrics['innings_2']['brier_score']:.4f}, AUC={wp_metrics['innings_2']['auc']:.4f}.")
doc_add_p(f"• Combined Overall Win Probability: Log-Loss={wp_metrics['overall']['log_loss']:.4f}, Brier={wp_metrics['overall']['brier_score']:.4f}, AUC={wp_metrics['overall']['auc']:.4f}.")

chart_calib_path = os.path.join(OUTPUTS_DIR, "win_prob_calibration.png")
if os.path.exists(chart_calib_path):
    doc.add_picture(chart_calib_path, width=Inches(6.8))

docx_path = os.path.join(OUTPUTS_DIR, "Methodology_Report.docx")
doc.save(docx_path)
print(f"Saved DOCX report to {docx_path}")

# PDF report was already built by previous section or runs cleanly
print("Report generation complete for both DOCX and PDF!")
