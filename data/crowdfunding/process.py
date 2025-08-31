import pandas as pd

df = pd.read_csv("original-.csv")
df = df.drop('CampaignID', axis=1)
df.to_csv("original.csv", index=False)