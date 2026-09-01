import sys
import joblib
import pandas as pd

MODEL_PATH = "crop_model.pkl"
DATA_PATH = "Crop_recommendation.csv"

EXPECTED_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
EXPECTED_CLASS_COUNT = 22
TEST_CROPS = ["rice", "cotton", "coffee", "apple", "banana"]

print("=" * 60)
print("TEST 1: Loading model artifact")
print("=" * 60)

try:
    model = joblib.load(MODEL_PATH)
    print(f"Loaded: {MODEL_PATH}")
except Exception as e:
    print(f"FAIL: Could not load model: {e}")
    sys.exit(1)

model_type = type(model).__name__
print(f"Model Type: {model_type}")
if model_type != "RandomForestClassifier":
    print(f"FAIL: Expected RandomForestClassifier, got {model_type}")
    sys.exit(1)
print("Type Check: PASS")

print("\n" + "=" * 60)
print("TEST 2: Verifying features and classes")
print("=" * 60)

actual_features = list(model.feature_names_in_)
print(f"Model Features: {actual_features}")
if actual_features != EXPECTED_FEATURES:
    print(f"FAIL: Expected {EXPECTED_FEATURES}, got {actual_features}")
    sys.exit(1)
print("Feature Names Check: PASS")

actual_classes = sorted(model.classes_)
print(f"Class Count: {len(actual_classes)}")
if len(actual_classes) != EXPECTED_CLASS_COUNT:
    print(f"FAIL: Expected {EXPECTED_CLASS_COUNT} classes, got {len(actual_classes)}")
    sys.exit(1)
print(f"Classes: {actual_classes}")
print("Class Count Check: PASS")

print("\n" + "=" * 60)
print("TEST 3: Loading dataset for ground-truth test samples")
print("=" * 60)

try:
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    print(f"FAIL: Could not load and normalize dataset: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("TEST 4: Representative crop sample predictions")
print("=" * 60)

all_passed = True
for crop in TEST_CROPS:
    crop_rows = df[df["label"] == crop]
    if crop_rows.empty:
        print(f"FAIL: Crop '{crop}' not found in dataset")
        all_passed = False
        continue

    sample_row = crop_rows.iloc[0]
    sample_features = sample_row[EXPECTED_FEATURES]
    input_df = pd.DataFrame([sample_features.values], columns=EXPECTED_FEATURES)

    predicted_crop = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    confidence = probabilities.max() * 100

    is_match = (predicted_crop == crop)
    status = "PASS" if is_match else "FAIL"
    if not is_match:
        all_passed = False

    print(f"Crop: {crop:<12} | Predicted: {predicted_crop:<12} | Confidence: {confidence:5.1f}% | {status}")

print("\n" + "=" * 60)
if all_passed:
    print("ALL TESTS PASSED: Model is verified and ready.")
    print("=" * 60)
    sys.exit(0)
else:
    print("TEST SUITE FAILED: One or more predictions did not match expected labels.")
    print("=" * 60)
    sys.exit(1)

