import openml
import pandas as pd

# Load the dataset
dataset = openml.datasets.get_dataset(183)
df, _, _, _ = dataset.get_data()
    
df.to_csv('abalone.csv', index=False)