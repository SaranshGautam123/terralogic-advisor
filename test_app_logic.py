"""
test_app_logic.py
=================
TerraLogic Advisor — Unified Application Logic Verification Suite (Stage 4.4)

Verifies the recovered app_logic module:

  1. Module imports cleanly.
  2. Numeric-only workflow returns a crop prediction and soil advisory.
  3. Image-only workflow returns a soil type prediction.
  4. Combined workflow runs both pipelines and produces guidance.
  5. Invalid numeric input is reported, not raised.
  6. A missing numeric key is reported clearly.
  7. Invalid image input is reported, not raised.
  8. Combined guidance is meaningful and states the scientific boundary.
  9. run_terralogic_advisory dispatches to the correct workflow.
 10. Pipeline separation: the image path never yields nutrient values.

Uses the existing trained models. Nothing is retrained.

Run:  .venv\\Scripts\\python.exe test_app_logic.py
"""

import sys
import pathlib

import app_logic as al


# -------------------------------------------------------------
# Paths / helpers
# -------------------------------------------------------------

TEST_DIR = pathlib.Path("data_soil/test")

# A known-good soil reading (also used in test_advisory.py).
GOOD_INPUTS = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.8,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.0,
}

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
    for name in ("run_terralogic_advisory", "analyze_numeric_inputs",
                 "analyze_soil_image", "synthesize_combined_guidance",
                 "load_crop_model"):
        assert hasattr(al, name), f"app_logic is missing '{name}'"


# -------------------------------------------------------------
# Test 2: Numeric-only workflow
# -------------------------------------------------------------
def test_numeric_workflow():
    result = al.analyze_numeric_inputs(**GOOD_INPUTS)

    assert result["status"] == "success", f"Unexpected status: {result['status']}"
    assert result["error"] is None, f"Unexpected error: {result['error']}"

    # A crop must be recommended.
    assert result["recommended_crop"], "No crop was recommended"
    assert result["recommended_crop"] in result["crop_probabilities"], \
        "Recommended crop is missing from the probability table"

    # Top crops list.
    assert isinstance(result["top_crops"], list), "top_crops is not a list"
    assert len(result["top_crops"]) == 5, \
        f"Expected 5 top crops, got {len(result['top_crops'])}"

    # The rule-based advisory report must be present and complete.
    advisory = result["soil_advisory"]
    assert isinstance(advisory, dict), "soil_advisory is not a dict"
    for key in ("soil_score", "soil_quality", "nutrient_status", "ph_status",
                "fertilizer_recommendations", "explanation"):
        assert key in advisory, f"Advisory missing '{key}'"

    # This vector is optimal soil, so the score should be at the maximum.
    assert advisory["soil_score"] == 100, \
        f"Expected a perfect soil score, got {advisory['soil_score']}"


# -------------------------------------------------------------
# Test 3: Image-only workflow
# -------------------------------------------------------------
def test_image_workflow():
    image_path = sample_image("Alluvial")
    assert image_path is not None, "No Alluvial test image available"

    result = al.analyze_soil_image(image_path)

    assert result["status"] == "success", f"Unexpected status: {result['status']}"
    assert result["error"] is None, f"Unexpected error: {result['error']}"

    assert result["predicted_class"] in ("Alluvial", "Black", "Clay", "Red"), \
        f"Invalid predicted class: {result['predicted_class']}"
    assert isinstance(result["confidence"], float), "confidence is not a float"
    assert 0.0 <= result["confidence"] <= 1.0, "confidence outside 0-1 range"
    assert isinstance(result["is_confident"], bool), "is_confident is not a bool"

    # The soil-type advisory must be attached.
    advisory = result["soil_advisory"]
    assert isinstance(advisory, dict), "soil_advisory is not a dict"
    assert "description" in advisory, "Advisory missing 'description'"
    assert "caution" in advisory, "Advisory missing 'caution'"


# -------------------------------------------------------------
# Test 4: Combined workflow
# -------------------------------------------------------------
def test_combined_workflow():
    image_path = sample_image("Black")
    assert image_path is not None, "No Black test image available"

    result = al.run_terralogic_advisory(
        numeric_inputs=GOOD_INPUTS,
        image_input=image_path,
    )

    assert result["workflow"] == "combined", \
        f"Expected workflow 'combined', got '{result['workflow']}'"

    assert result["error"] is None, f"Unexpected top-level error: {result['error']}"

    # Both pipelines must have produced a result.
    assert result["numeric_result"] is not None, "numeric_result missing"
    assert result["numeric_result"]["status"] == "success", "Numeric pipeline failed"
    assert result["image_result"] is not None, "image_result missing"
    assert result["image_result"]["status"] == "success", "Image pipeline failed"

    assert result["combined_guidance"] is not None, "combined_guidance missing"


# -------------------------------------------------------------
# Test 5: Invalid numeric input handling
# -------------------------------------------------------------
def test_invalid_numeric_inputs():
    # Negative nutrient value.
    bad = dict(GOOD_INPUTS)
    bad["N"] = -10
    result = al.analyze_numeric_inputs(**bad)
    assert result["status"] == "error", "Negative N should be rejected"
    assert "negative" in result["error"].lower(), \
        f"Unexpected error text: {result['error']}"
    assert result["recommended_crop"] is None, "A crop was recommended for bad input"

    # pH out of range.
    bad = dict(GOOD_INPUTS)
    bad["ph"] = 20.0
    result = al.analyze_numeric_inputs(**bad)
    assert result["status"] == "error", "pH of 20 should be rejected"
    assert "pH" in result["error"], f"Unexpected error text: {result['error']}"

    # Non-numeric value.
    bad = dict(GOOD_INPUTS)
    bad["humidity"] = "very humid"
    result = al.analyze_numeric_inputs(**bad)
    assert result["status"] == "error", "A string humidity should be rejected"
    assert "numeric" in result["error"].lower(), \
        f"Unexpected error text: {result['error']}"


# -------------------------------------------------------------
# Test 6: Missing numeric key handling
# -------------------------------------------------------------
def test_missing_numeric_key():
    incomplete = {"N": 90, "P": 42}  # K and the rest are missing

    result = al.run_terralogic_advisory(numeric_inputs=incomplete)

    assert result["workflow"] == "numeric_only", \
        f"Expected workflow 'numeric_only', got '{result['workflow']}'"
    assert result["numeric_result"]["status"] == "error", \
        "Missing keys should produce an error status"
    assert "Missing numeric input key" in result["numeric_result"]["error"], \
        f"Unexpected error text: {result['numeric_result']['error']}"


# -------------------------------------------------------------
# Test 7: Invalid image handling
# -------------------------------------------------------------
def test_invalid_image_input():
    # No image at all.
    result = al.analyze_soil_image(None)
    assert result["status"] == "error", "A missing image should be reported"
    assert result["error"] == "No image provided.", \
        f"Unexpected error text: {result['error']}"

    # Unsupported image input type.
    result = al.analyze_soil_image(12345)
    assert result["status"] == "error", "An integer should be rejected"
    assert "Invalid image input" in result["error"], \
        f"Unexpected error text: {result['error']}"

    # A file path that does not exist.
    result = al.analyze_soil_image("no_such_image.jpg")
    assert result["status"] == "error", "A missing file should be reported"
    assert result["predicted_class"] is None, "A class was predicted for a bad file"


# -------------------------------------------------------------
# Test 8: Combined guidance quality
# -------------------------------------------------------------
def test_combined_guidance():
    image_path = sample_image("Red")
    assert image_path is not None, "No Red test image available"

    combined = al.run_terralogic_advisory(
        numeric_inputs=GOOD_INPUTS,
        image_input=image_path,
    )
    guidance = combined["combined_guidance"]

    for key in ("soil_type_from_image", "crop_from_numeric",
                "soil_crop_compatibility", "affinity_crops",
                "guidance_notes", "boundary_note"):
        assert key in guidance, f"Combined guidance missing '{key}'"

    # Both sides of the comparison must be populated.
    assert guidance["soil_type_from_image"], "No soil type in combined guidance"
    assert guidance["crop_from_numeric"], "No crop in combined guidance"

    # Guidance notes must be a non-empty list of strings.
    notes = guidance["guidance_notes"]
    assert isinstance(notes, list), "guidance_notes is not a list"
    assert len(notes) > 0, "guidance_notes is empty"
    assert all(isinstance(n, str) for n in notes), "A guidance note is not a string"

    # The scientific boundary must be stated explicitly.
    boundary = guidance["boundary_note"]
    assert "does not measure" in boundary, \
        f"Boundary note does not state the scientific limit: {boundary}"
    for parameter in ("N", "P", "K", "pH"):
        assert parameter in boundary, \
            f"Boundary note does not mention {parameter}"


# -------------------------------------------------------------
# Test 9: Workflow dispatch
# -------------------------------------------------------------
def test_workflow_dispatch():
    # No inputs at all.
    result = al.run_terralogic_advisory()
    assert result["workflow"] == "error", "No input should yield workflow 'error'"
    assert result["error"], "No top-level error message was set"
    assert result["numeric_result"] is None, "numeric_result should be None"
    assert result["image_result"] is None, "image_result should be None"

    # Numeric only.
    result = al.run_terralogic_advisory(numeric_inputs=GOOD_INPUTS)
    assert result["workflow"] == "numeric_only", \
        f"Expected 'numeric_only', got '{result['workflow']}'"
    assert result["numeric_result"] is not None, "numeric_result missing"
    assert result["image_result"] is None, "image_result should be None"
    assert result["combined_guidance"] is None, "combined_guidance should be None"

    # Image only.
    image_path = sample_image("Clay")
    assert image_path is not None, "No Clay test image available"
    result = al.run_terralogic_advisory(image_input=image_path)
    assert result["workflow"] == "image_only", \
        f"Expected 'image_only', got '{result['workflow']}'"
    assert result["image_result"] is not None, "image_result missing"
    assert result["numeric_result"] is None, "numeric_result should be None"
    assert result["combined_guidance"] is None, "combined_guidance should be None"


# -------------------------------------------------------------
# Test 10: Pipeline separation (scientific boundary)
# -------------------------------------------------------------
def test_pipeline_separation():
    image_path = sample_image("Alluvial")
    assert image_path is not None, "No Alluvial test image available"

    image_result = al.analyze_soil_image(image_path)
    assert image_result["status"] == "success", "Image pipeline failed"

    # The image pipeline must report a soil class and nothing else.
    forbidden = ("N", "P", "K", "ph", "nutrient_status", "temperature",
                 "humidity", "rainfall", "moisture")
    for key in image_result:
        assert key not in forbidden, \
            f"Image pipeline must not produce nutrient field '{key}'"

    # It must not report any numeric soil-test values at all.
    numeric_keys = [k for k, v in image_result.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                    and k != "confidence"]
    assert not numeric_keys, \
        f"Image pipeline returned unexpected numeric fields: {numeric_keys}"

    # The caution text must forbid estimating nutrients from a photograph.
    caution = image_result["soil_advisory"]["caution"]
    assert "Do not estimate" in caution, \
        "Soil advisory caution does not forbid nutrient estimation from a photo"


# =============================================================
# Runner
# =============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING APPLICATION LOGIC TEST SUITE")
    print("=" * 65)

    print("\n--- Module ---")
    run_test("Module import", test_module_import)

    print("\n--- Individual pipelines ---")
    run_test("Numeric-only workflow", test_numeric_workflow)
    run_test("Image-only workflow", test_image_workflow)
    run_test("Combined workflow", test_combined_workflow)

    print("\n--- Error handling ---")
    run_test("Invalid numeric input handling", test_invalid_numeric_inputs)
    run_test("Missing numeric key handling", test_missing_numeric_key)
    run_test("Invalid image handling", test_invalid_image_input)

    print("\n--- Combined guidance ---")
    run_test("Combined guidance quality", test_combined_guidance)
    run_test("Workflow dispatch", test_workflow_dispatch)

    print("\n--- Scientific boundary ---")
    run_test("Pipeline separation", test_pipeline_separation)

    print("\n" + "=" * 65)
    print(f"RESULTS: {passed_count} Passed, {failed_count} Failed")
    print("=" * 65)

    if failed_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)
