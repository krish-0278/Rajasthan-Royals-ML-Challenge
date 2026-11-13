import os
import pandas as pd
import numpy as np

data_dir = "Data Pack"
files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
print(f"Found CSV files: {files}\n")

for f in sorted(files):
    path = os.path.join(data_dir, f)
    df = pd.read_csv(path, nrows=5)
    print(f"=== {f} ===")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample row 0:\n{df.iloc[0].to_dict()}\n")

