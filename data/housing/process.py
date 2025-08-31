import pandas as pd

df = pd.read_csv("original.csv")
df.to_csv("housing.csv", index=False)