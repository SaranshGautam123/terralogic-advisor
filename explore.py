import pandas as pd

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# --- Normalize ALL column names (strips leading/trailing whitespace from every column) ---
df.columns = df.columns.str.strip()

# --- Normalize the target column values ---
if "label" in df.columns:
    df["label"] = df["label"].astype(str).str.strip()

# ---------------------------------------------------------------
print("--- Raw column names after normalization ---")
print(df.columns.tolist())

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Shape (Rows, Columns) ---")
print(df.shape)

print("\n--- Column Info ---")
print(df.info())

print("\n--- Missing Values per Column ---")
print(df.isnull().sum())

print("\n--- Summary Statistics (numeric columns) ---")
print(df.describe())

print("\n--- Unique Crop Classes (sorted) ---")
print(sorted(df["label"].unique()))

print("\n--- Number of Unique Classes ---")
print(df["label"].nunique())

print("\n--- Samples per Class ---")
print(df["label"].value_counts().sort_index())