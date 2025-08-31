import pandas as pd

df = pd.read_csv("original.csv")
df = df.drop('PlayerID', axis=1)
df.to_csv("gaming.csv", index=False)