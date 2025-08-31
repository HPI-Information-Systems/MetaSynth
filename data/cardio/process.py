import pandas as pd

df = pd.read_csv("original-.csv")
df = df.drop('id', axis=1)
df.to_csv("cardio.csv", index=False)