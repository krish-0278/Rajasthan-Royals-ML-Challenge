import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_parquet("data_processed/delivery_pressure_complete.parquet")
df_map = pd.read_csv("Data Pack/competition_team_player_mapping.csv")
p_teams_str = df_map.groupby('player_id')['team_name'].apply(lambda x: ", ".join(sorted(x.unique())[:2])).to_dict()

# Let's inspect aggregate functions
from src import performance_ratings_test as prt # or run inline

