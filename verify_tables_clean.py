import pandas as pd

b = pd.read_csv('outputs/top10_batters.csv')
bw = pd.read_csv('outputs/top10_bowlers.csv')

print("Batters null count:\n", b.isnull().sum())
print("\nBowlers null count:\n", bw.isnull().sum())

print("\nBatters strict rank order:", list(b['rank']) == list(range(1, 11)))
print("Bowlers strict rank order:", list(bw['rank']) == list(range(1, 11)))

print("\nBatters scores monotonically non-increasing:", (b['pressure_score'].diff().dropna() <= 0).all())
print("Bowlers scores monotonically non-increasing:", (bw['pressure_score'].diff().dropna() <= 0).all())

