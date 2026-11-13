"""
run_pipeline.py
Master execution script running the full SupeRR Selector pipeline end-to-end.
"""
import subprocess
import sys
import time

scripts = [
    "src/01_clean_and_audit.py",
    "src/02_feature_engineering.py",
    "src/03_win_probability_model.py",
    "src/04_pressure_index.py",
    "src/05_performance_ratings.py",
    "src/06_validation.py",
    "src/generate_charts.py"
]

print("==========================================================")
print("RAJASTHAN ROYALS: SUPERR SELECTOR SEASON 2 PIPELINE")
print("==========================================================")

total_start = time.time()
for s in scripts:
    print(f"\n>>> Running {s}...")
    start = time.time()
    res = subprocess.run([sys.executable, s], check=True)
    elapsed = time.time() - start
    print(f">>> Completed {s} in {elapsed:.1f}s")

total_elapsed = time.time() - total_start
print(f"\nFull pipeline executed successfully in {total_elapsed:.1f}s!")

