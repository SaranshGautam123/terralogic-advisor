import pandas as pd

df = pd.read_csv("Crop_recommendation.csv")

print(df.head())

print(df.shape)

print(df.info())

print(df.isnull().sum())

print(df.describe())

print(df["label"].unique())

print(df["label"].nunique())