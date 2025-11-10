import pandas as pd

#load the dataset
file_path = "../data/Sleep_health_and_lifestyle_dataset.csv"
df = pd.read_csv(file_path)

#show few row of data from the dataset
print("First five rows of the dataset:")
print(df.head())

#basic info about the dataset
print("\nBasic information about the dataset:")
print(df.info())

#check for missing values
print("\ncheck for missing values in each column:")
print(df.isnull().sum())

#quick summary of numeric values
print("\nStatistical summary:")
print(df.describe())

