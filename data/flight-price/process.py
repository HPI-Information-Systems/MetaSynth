import pandas as pd

# Specify the file path
file_path = '/hpi/fs00/home/philipp.hildebrandt/MA/data/flight-price/archive/Clean_Dataset.csv'

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("original.csv")

# Create a new column flight_prefix using values from column flight
df['flight_prefix'] = df['flight'].str.split('-').str[0]

# Create a new column flight_prefix using values from column flight
df['flight_number'] = df['flight'].str.split('-').str[1]

# Drop the index column
df = df.drop('Unnamed: 0', axis=1)
df = df.drop('flight', axis=1)

# Move the column 'price' to the right
df = df[[col for col in df.columns if col not in ['price']]+['price']]

# Save the DataFrame as original.csv without the index
df.to_csv('flight-price.csv', index=False)