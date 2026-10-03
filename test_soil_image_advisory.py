"""
test_soil_image_advisory.py
===========================
TerraLogic Advisor — Soil Image Advisory API Verification Suite (Stage 4.3)

Verifies the recovered soil_image_advisory module:

  1. Module imports cleanly.
  2. SOIL_CLASSES holds exactly the four soil types.
  3. The MobileNetV2 model loads from disk.
  4. Inference works from a real image path.
  5. The returned prediction structure is complete.
  6. Confidence threshold behaviour (confident / low-confidence).
  7. All four soil advisory profiles are accessible.
  8. Invalid input handling is rejected cleanly.
  9. predict_and_advise() combines prediction and advisory.

Uses real images from data_soil/test/. Nothing is retrained.

Run:  .venv\\Scripts\\python.exe test_soil_image_advisory.py
"""

import sys
import pathlib

import soil_image_advisory as sia


# -------------------------------------------------------------
# Paths / helpers
# -------------------------------------------------------------

TEST_DIR = pathlib.Path("data_soil/test")

passed_count = 0
failed_count = 0


def run_test(test_name, test_func):
    """Run one check, print PASS/FAIL, and track totals."""
    global passed_count, failed_count
    try:
        test_func()
        print(f"  [PASS] {test_name}")
        passed_count += 1
    except AssertionError as e:
        print(f"  [FAIL] {test_name}: {e}")
        failed_count += 1
    except Exception as e:
        print(f"  [ERROR] {test_name}: Unexpected exception: {type(e).__name__}: {e}")
        failed_count += 1


def sample_image(soil_class):
    """Return one real test image path for a soil class, or None."""
    folder = TEST_DIR / soil_class
    if not folder.exists():
        return None
    images = sorted(folder.glob("*.jpg"))
    return str(images[0]) if images else None


# -------------------------------------------------------------
# Test 1: Module import
# -------------------------------------------------------------
def test_module_import():
    assert sia.SOIL_CLASSES is not None, "SOIL_CLASSES missing"
    assert hasattr(sia, "predict_soil_image"), "predict_soil_image missing"
    assert hasattr(sia, "get_soil_type_advisory"), "get_soil_type_advisory missing"
    assert hasattr(sia, "predict_and_advise"), "predict_and_advise missing"


# -------------------------------------------------------------
# Test 2: Class mapping
# -------------------------------------------------------------
def test_soil_classes():
    expected = ["Alluvial", "Black", "Clay", "Red"]
    actual = list(sia.SOIL_CLASSES)
    assert actual == expected, f"Expected {expected}, got {actual}"


# -------------------------------------------------------------
# Test 3: Model loading
# -------------------------------------------------------------
def test_model_loading():
    model = sia.load_soil_image_model()

    assert model is not None, "Model is None"
    assert hasattr(model, "predict"), "Loaded object has no predict() method"

    # Second call must return the cached model (same object).
    assert sia.load_soil_image_model() is model, "Model was not cached"


# -------------------------------------------------------------
# Test 4: Inference from a valid image path
# -------------------------------------------------------------
def test_valid_path_inference():
    image_path = sample_image("Alluvial")
    assert image_path is not None, f"No test images found in {TEST_DIR / 'Alluvial'}"

    result = sia.predict_soil_image(image_path)

    assert result["predicted_class"] in sia.SOIL_CLASSES, \
        f"Predicted class '{result['predicted_class']}' is not a valid soil type"


# -------------------------------------------------------------
# Test 5: Returned prediction structure
# -------------------------------------------------------------
def test_prediction_structure():
    image_path = sample_image("Black")
    assert image_path is not None, "No Black test image available"

    result = sia.predict_soil_image(image_path)

    for key in ("predicted_class", "confidence", "all_probabilities",
                "is_confident", "warning"):
        assert key in result, f"Missing key '{key}' in prediction result"

    # Confidence must be a probability.
    assert isinstance(result["confidence"], float), "confidence is not a float"
    assert 0.0 <= result["confidence"] <= 1.0, \
        f"confidence {result['confidence']} outside 0-1 range"

    # All four classes must carry a probability.
    probs = result["all_probabilities"]
    assert isinstance(probs, dict), "all_probabilities is not a dict"
    assert set(probs.keys()) == set(sia.SOIL_CLASSES), \
        f"Probability keys {list(probs.keys())} do not match soil classes"
    assert all(0.0 <= v <= 1.0 for v in probs.values()), \
        "A probability falls outside the 0-1 range"

    # Probabilities should sum to approximately 1.
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-3, f"Probabilities sum to {total}, expected ~1.0"

    # The predicted class must be the highest probability.
    best = max(probs, key=probs.get)
    assert best == result["predicted_class"], \
        f"Predicted '{result['predicted_class']}' but highest probability was '{best}'"


# -------------------------------------------------------------
# Test 6: Confidence threshold behaviour
# -------------------------------------------------------------
def test_confidence_threshold():
    image_path = sample_image("Clay")
    assert image_path is not None, "No Clay test image available"

    # Threshold of 0.0 -> any prediction counts as confident.
    lenient = sia.predict_soil_image(image_path, confidence_threshold=0.0)
    assert lenient["is_confident"] is True, \
        "Threshold 0.0 should always be confident"
    assert lenient["warning"] is None, \
        "A confident result must not carry a warning"

    # Threshold of 1.01 -> unreachable, forces the low-confidence branch.
    strict = sia.predict_soil_image(image_path, confidence_threshold=1.01)
    assert strict["is_confident"] is False, \
        "Threshold above 1.0 should never be confident"
    assert strict["warning"] is not None, \
        "A low-confidence result must carry a warning"
    assert "Low confidence" in strict["warning"], \
        f"Unexpected warning text: {strict['warning']}"

    # The prediction itself must not change with the threshold.
    assert lenient["predicted_class"] == strict["predicted_class"], \
        "Threshold must not alter the predicted class"
    assert abs(lenient["confidence"] - strict["confidence"]) < 1e-9, \
        "Threshold must not alter the confidence value"


# -------------------------------------------------------------
# Test 7: All four soil advisory profiles
# -------------------------------------------------------------
def test_advisory_profiles():
    for soil_type in sia.SOIL_CLASSES:
        advisory = sia.get_soil_type_advisory(soil_type)

        assert isinstance(advisory, dict), f"{soil_type}: advisory is not a dict"

        for field in ("description", "characteristics", "suitable_crops",
                      "management_tips", "caution"):
            assert field in advisory, f"{soil_type}: missing field '{field}'"

        assert len(advisory["description"]) > 0, f"{soil_type}: empty description"
        assert len(advisory["characteristics"]) > 0, f"{soil_type}: no characteristics"
        assert len(advisory["suitable_crops"]) > 0, f"{soil_type}: no suitable crops"
        assert len(advisory["management_tips"]) > 0, f"{soil_type}: no management tips"

        # The scientific boundary must be stated for every soil type.
        caution = advisory["caution"]
        assert "Do not estimate" in caution, \
            f"{soil_type}: caution does not state the scientific boundary"


# -------------------------------------------------------------
# Test 8: Invalid input handling
# -------------------------------------------------------------
def test_invalid_inputs():
    # Unsupported input type.
    try:
        sia._preprocess_image(12345)
        assert False, "Should reject an integer input"
    except TypeError as e:
        assert "Unsupported image_input type" in str(e), \
            f"Unexpected error text: {e}"

    # Missing model file.
    # load_soil_image_model() returns the cached model before it checks the
    # path, so the cold-cache behaviour is only observable with an empty
    # cache. Clear it for this check, then put it back.
    cached_model = sia._CACHED_MODEL
    sia._CACHED_MODEL = None
    try:
        try:
            sia.load_soil_image_model("no_such_model.keras")
            assert False, "Should raise FileNotFoundError for a missing model"
        except FileNotFoundError as e:
            assert "not found" in str(e), f"Unexpected error text: {e}"
    finally:
        sia._CACHED_MODEL = cached_model

    assert sia.load_soil_image_model() is cached_model, \
        "Model cache was not restored"

    # Unknown soil type.
    try:
        sia.get_soil_type_advisory("Loam")
        assert False, "Should reject an unrecognised soil type"
    except ValueError as e:
        assert "Unknown soil type" in str(e), f"Unexpected error text: {e}"


# -------------------------------------------------------------
# Test 9: predict_and_advise
# -------------------------------------------------------------
def test_predict_and_advise():
    image_path = sample_image("Red")
    assert image_path is not None, "No Red test image available"

    result = sia.predict_and_advise(image_path)

    assert "prediction" in result, "Missing 'prediction' key"
    assert "advisory" in result, "Missing 'advisory' key"

    prediction = result["prediction"]
    advisory = result["advisory"]

    assert "predicted_class" in prediction, "Prediction missing 'predicted_class'"
    assert "confidence" in prediction, "Prediction missing 'confidence'"

    # The advisory must match the predicted class.
    assert prediction["predicted_class"] in sia.SOIL_CLASSES, \
        "Predicted class is not a valid soil type"
    assert advisory == sia.SOIL_ADVISORY_DATABASE[prediction["predicted_class"]], \
        "Advisory does not match the predicted soil type"


# =============================================================
# Runner
# =============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING SOIL IMAGE ADVISORY TEST SUITE")
    print("=" * 65)

    print("\n--- Module and constants ---")
    run_test("Module import", test_module_import)
    run_test("SOIL_CLASSES mapping", test_soil_classes)

    print("\n--- Model and inference ---")
    run_test("Model loading and caching", test_model_loading)
    run_test("Inference from valid image path", test_valid_path_inference)
    run_test("Prediction structure and probabilities", test_prediction_structure)

    print("\n--- Confidence handling ---")
    run_test("Confidence threshold behaviour", test_confidence_threshold)

    print("\n--- Advisory data ---")
    run_test("All four soil advisory profiles", test_advisory_profiles)

    print("\n--- Error handling ---")
    run_test("Invalid input handling", test_invalid_inputs)

    print("\n--- Combined helper ---")
    run_test("predict_and_advise()", test_predict_and_advise)

    print("\n" + "=" * 65)
    print(f"RESULTS: {passed_count} Passed, {failed_count} Failed")
    print("=" * 65)

    if failed_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)
