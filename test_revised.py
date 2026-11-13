import pandas as pd
df = pd.read_parquet("data_processed/cleaned_ball_by_ball.parquet")
m = df[['match_id', 'first_inn_revised_overs', 'second_inn_revised_overs', 'revised_target']].drop_duplicates()
rev = m[m['revised_target'].notnull()]
print("Revised matches sample:\n", rev.head(10))

