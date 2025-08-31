import openml
import pandas as pd

# Load the dataset
dataset = openml.datasets.get_dataset(183)
df, _, _, _ = dataset.get_data()

for column in df.columns:
    unique_values = df[column].nunique()
    print(f"Number of unique values in {column}: {unique_values}")
    
df.to_csv('abalone.csv', index=False)