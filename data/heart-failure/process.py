import pandas as pd

df = pd.read_csv("original.csv")
df.to_csv("heart-failure.csv", index=False)