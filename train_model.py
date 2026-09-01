import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = "Crop_recommendation.csv"
MODEL_PATH = "crop_model.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2

EXPECTED_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"

# =============================================================================
# STEP 1 — LOAD AND NORMALIZE DATASET
# =============================================================================
print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

# Normalize ALL column names: strip leading/trailing whitespace
df.columns = df.columns.str.strip()

# Normalize target label values
df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns found: {df.columns.tolist()}")

# =============================================================================
# STEP 2 — VALIDATE COLUMNS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Validating columns")
print("=" * 60)

all_expected = EXPECTED_FEATURES + [TARGET_COLUMN]
missing_cols = [c for c in all_expected if c not in df.columns]
if missing_cols:
    raise ValueError(
        f"Required columns missing from dataset after normalization: {missing_cols}\n"
        f"Columns present: {df.columns.tolist()}"
    )

# Report any extra columns that were not expected (do not silently drop them)
extra_cols = [c for c in df.columns if c not in all_expected]
if extra_cols:
    print(f"WARNING: Unexpected extra columns found (not used for training): {extra_cols}")
else:
    print("All expected columns present. No unexpected extra columns.")

print(f"Missing values:\n{df[all_expected].isnull().sum().to_string()}")
if df[all_expected].isnull().any().any():
    raise ValueError("Dataset contains null values. Please clean the data before training.")

# =============================================================================
# STEP 3 — PREPARE FEATURES AND TARGET
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Preparing features and target")
print("=" * 60)

X = df[EXPECTED_FEATURES]
y = df[TARGET_COLUMN]

print(f"Feature matrix shape : {X.shape}")
print(f"Target vector shape  : {y.shape}")
print(f"Number of classes    : {y.nunique()}")
print(f"Classes              : {sorted(y.unique())}")
print(f"\nSamples per class:\n{y.value_counts().sort_index().to_string()}")

# =============================================================================
# STEP 4 — TRAIN / TEST SPLIT
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Train/test split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y  # ensures each class is proportionally represented in both splits
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"Split ratio      : {100 * (1 - TEST_SIZE):.0f}% train / {100 * TEST_SIZE:.0f}% test")
print(f"Stratified split : Yes (each crop class is equally represented)")

# =============================================================================
# STEP 5 — TRAIN MODEL
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Training RandomForestClassifier")
print("=" * 60)

model = RandomForestClassifier(random_state=RANDOM_STATE)
model.fit(X_train, y_train)

print(f"Model type         : {type(model).__name__}")
print(f"Number of trees    : {model.n_estimators}")
print(f"random_state       : {model.random_state}")
print(f"Feature names used : {list(model.feature_names_in_)}")

# =============================================================================
# STEP 6 — EVALUATE MODEL
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Evaluating model on test set")
print("=" * 60)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Set Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(
    "\nNote: This accuracy is measured on the held-out test split of the same "
    "dataset the model was trained on. It does not guarantee real-world "
    "performance on unseen soil/climate conditions."
)

print("\n--- Classification Report (Precision / Recall / F1 per crop) ---")
print(classification_report(y_test, y_pred, zero_division=0))

print("--- Confusion Matrix (rows = actual, columns = predicted) ---")
classes = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=classes)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
print(cm_df.to_string())

print("\n--- Misclassified Samples Summary ---")
misclassified = (y_test.values != y_pred).sum()
print(f"Total misclassified: {misclassified} out of {len(y_test)}")

# =============================================================================
# STEP 7 — SAVE MODEL
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: Saving model")
print("=" * 60)

joblib.dump(model, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")

# Quick load-back verification
loaded = joblib.load(MODEL_PATH)
assert type(loaded).__name__ == "RandomForestClassifier", "Reload type check failed"
assert list(loaded.feature_names_in_) == EXPECTED_FEATURES, "Reload feature check failed"
print("Verification: model reloaded and feature names confirmed — OK")

print("\n" + "=" * 60)
print("Training pipeline complete.")
print("=" * 60)
