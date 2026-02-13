# This is file is to check the data integrity and consistency in the processed data. 
# The Data contains thousands of records that will take way too long to check manually so this script automates the process. 

import pandas as pd 
import numpy as np 
#Load and read csv file using panda; 
file_path  = 'raw/processed/creditcard.csv'
df = pd.read_csv(file_path)



NullValues = df.isnull().sum()

print(NullValues)

# Python returns 0 so there are no null values in the dataset. 

# Next step is to check for duplicted values in the dataset. 
duplicated_values = df.duplicated().sum()
print(f"Number of duplicated values in the dataset: {duplicated_values} ")
data_cleaned = df.drop_duplicates() # Drop duplicated values if any exist 
data_cleaned = data_cleaned.reset_index(drop=True) # Reset index after dropping duplicates
data_cleaned.to_csv('data/processed/creditcard_cleaned.csv', index=False)

