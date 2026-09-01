"""
test_advisory.py
================
Comprehensive unit tests for the TerraLogic Advisor rule-based advisory engine.

Verifies:
  1. Input validation (types and physical range bounds)
  2. Soil health score calculation
  3. Soil quality classification
  4. NPK nutrient status categorization
  5. pH analysis classification
  6. Fertilizer recommendation generation
  7. Narrative explanation generation
  8. Unified get_soil_advisory() structured dictionary output
"""

import sys
from advisory import (
    calculate_soil_score,
    soil_quality,
    nutrient_status,
    ph_analysis,
    fertilizer_recommendation,
    generate_explanation,
    validate_soil_inputs,
    get_soil_advisory,
)

passed_count = 0
failed_count = 0


def run_test(test_name, test_func):
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


# -------------------------------------------------------------
# Test 1: Input Validation
# -------------------------------------------------------------
def test_input_validation():
    # Valid inputs should pass without exception
    validate_soil_inputs(90, 42, 43, 20.8, 82.0, 6.5, 202.0)

    # Non-numeric type checks
    try:
        validate_soil_inputs("invalid", 42, 43, 20.8, 82.0, 6.5, 202.0)
        assert False, "Should fail on string input"
    except TypeError:
        pass

    try:
        validate_soil_inputs(90, True, 43, 20.8, 82.0, 6.5, 202.0)
        assert False, "Should fail on boolean input"
    except TypeError:
        pass

    # Negative nutrient checks
    try:
        validate_soil_inputs(-10, 42, 43, 20.8, 82.0, 6.5, 202.0)
        assert False, "Should fail on negative N"
    except ValueError:
        pass

    # Invalid pH checks
    try:
        validate_soil_inputs(90, 42, 43, 20.8, 82.0, -1.0, 202.0)
        assert False, "Should fail on pH < 0"
    except ValueError:
        pass

    try:
        validate_soil_inputs(90, 42, 43, 20.8, 82.0, 15.0, 202.0)
        assert False, "Should fail on pH > 14"
    except ValueError:
        pass

    # Invalid humidity checks
    try:
        validate_soil_inputs(90, 42, 43, 20.8, 120.0, 6.5, 202.0)
        assert False, "Should fail on humidity > 100%"
    except ValueError:
        pass

    # Negative rainfall check
    try:
        validate_soil_inputs(90, 42, 43, 20.8, 82.0, 6.5, -5.0)
        assert False, "Should fail on negative rainfall"
    except ValueError:
        pass

    # Extreme temperature check
    try:
        validate_soil_inputs(90, 42, 43, 85.0, 82.0, 6.5, 202.0)
        assert False, "Should fail on unrealistic temperature > 60°C"
    except ValueError:
        pass


# -------------------------------------------------------------
# Test 2: Soil Health Score Calculation
# -------------------------------------------------------------
def test_soil_score_calculation():
    # Optimal values -> Max Score = 100
    optimal_score = calculate_soil_score(
        N=70, P=50, K=50, temperature=25.0, humidity=70.0, ph=6.8, rainfall=150.0
    )
    assert optimal_score == 100, f"Expected 100 for optimal values, got {optimal_score}"

    # All sub-optimal/out-of-range -> Score = 0
    zero_score = calculate_soil_score(
        N=5, P=5, K=5, temperature=5.0, humidity=20.0, ph=4.0, rainfall=10.0
    )
    assert zero_score == 0, f"Expected 0 for poor values, got {zero_score}"

    # Partial credit values:
    # N=30 (+10), P=15 (+10), K=15 (+10), ph=5.8 (+10), temp=25 (+10), hum=70 (+5), rain=100 (+5) = 60
    partial_score = calculate_soil_score(
        N=30, P=15, K=15, temperature=25.0, humidity=70.0, ph=5.8, rainfall=100.0
    )
    assert partial_score == 60, f"Expected 60 for partial credit values, got {partial_score}"


# -------------------------------------------------------------
# Test 3: Soil Quality Classification
# -------------------------------------------------------------
def test_soil_quality_mapping():
    assert soil_quality(100) == "Excellent"
    assert soil_quality(90) == "Excellent"
    assert soil_quality(89) == "Good"
    assert soil_quality(75) == "Good"
    assert soil_quality(74) == "Average"
    assert soil_quality(50) == "Average"
    assert soil_quality(49) == "Poor"
    assert soil_quality(0) == "Poor"


# -------------------------------------------------------------
# Test 4: NPK Nutrient Status
# -------------------------------------------------------------
def test_nutrient_status():
    # Low values
    low_status = nutrient_status(N=20, P=10, K=10)
    assert low_status["Nitrogen"] == "Low"
    assert low_status["Phosphorus"] == "Low"
    assert low_status["Potassium"] == "Low"

    # Optimal values
    opt_status = nutrient_status(N=70, P=50, K=50)
    assert opt_status["Nitrogen"] == "Optimal"
    assert opt_status["Phosphorus"] == "Optimal"
    assert opt_status["Potassium"] == "Optimal"

    # High values
    high_status = nutrient_status(N=130, P=110, K=120)
    assert high_status["Nitrogen"] == "High"
    assert high_status["Phosphorus"] == "High"
    assert high_status["Potassium"] == "High"


# -------------------------------------------------------------
# Test 5: pH Analysis
# -------------------------------------------------------------
def test_ph_analysis():
    assert ph_analysis(4.5) == "Strongly Acidic"
    assert ph_analysis(5.4) == "Strongly Acidic"
    assert ph_analysis(5.5) == "Slightly Acidic"
    assert ph_analysis(6.4) == "Slightly Acidic"
    assert ph_analysis(6.5) == "Neutral (Ideal)"
    assert ph_analysis(7.0) == "Neutral (Ideal)"
    assert ph_analysis(7.5) == "Neutral (Ideal)"
    assert ph_analysis(7.6) == "Alkaline"
    assert ph_analysis(9.0) == "Alkaline"


# -------------------------------------------------------------
# Test 6: Fertilizer Recommendations
# -------------------------------------------------------------
def test_fertilizer_recommendations():
    # Balanced soil
    balanced_advice = fertilizer_recommendation(N=70, P=50, K=50, ph=7.0)
    assert len(balanced_advice) == 1
    assert "balanced" in balanced_advice[0].lower()

    # Low N, Low P, Low K, Acidic pH
    deficient_advice = fertilizer_recommendation(N=20, P=10, K=10, ph=5.0)
    assert len(deficient_advice) == 4
    assert any("nitrogen" in a.lower() for a in deficient_advice)
    assert any("phosphorus" in a.lower() for a in deficient_advice)
    assert any("potassium" in a.lower() for a in deficient_advice)
    assert any("lime" in a.lower() for a in deficient_advice)

    # Alkaline pH advice
    alkaline_advice = fertilizer_recommendation(N=70, P=50, K=50, ph=8.2)
    assert any("alkalinity" in a.lower() or "organic matter" in a.lower() for a in alkaline_advice)


# -------------------------------------------------------------
# Test 7: Unified get_soil_advisory() Entrypoint
# -------------------------------------------------------------
def test_unified_advisory():
    # Test with crop specified
    report = get_soil_advisory(
        N=90, P=42, K=43, temperature=20.8, humidity=82.0, ph=6.5, rainfall=202.0, crop="Rice"
    )

    assert isinstance(report, dict), "Report must be a dictionary"
    assert "soil_score" in report
    assert "soil_quality" in report
    assert "nutrient_status" in report
    assert "ph_status" in report
    assert "fertilizer_recommendations" in report
    assert "explanation" in report

    assert report["soil_score"] == 100
    assert report["soil_quality"] == "Excellent"
    assert report["ph_status"] == "Neutral (Ideal)"
    assert "Rice" in report["explanation"]
    assert isinstance(report["fertilizer_recommendations"], list)
    assert len(report["fertilizer_recommendations"]) > 0

    # Test with crop=None (general advisory)
    general_report = get_soil_advisory(
        N=30, P=15, K=15, temperature=22.0, humidity=60.0, ph=5.2, rainfall=80.0
    )
    assert general_report["soil_quality"] == "Average"
    assert general_report["ph_status"] == "Strongly Acidic"
    assert "recommended crop" not in general_report["explanation"].lower()


# =============================================================
# Runner
# =============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING ADVISORY ENGINE TEST SUITE")
    print("=" * 65)

    run_test("Input Validation Checks", test_input_validation)
    run_test("Soil Score Calculation", test_soil_score_calculation)
    run_test("Soil Quality Mapping", test_soil_quality_mapping)
    run_test("Nutrient Status Categorization", test_nutrient_status)
    run_test("pH Analysis Categorization", test_ph_analysis)
    run_test("Fertilizer Recommendation Rules", test_fertilizer_recommendations)
    run_test("Unified get_soil_advisory() Interface", test_unified_advisory)

    print("\n" + "=" * 65)
    print(f"RESULTS: {passed_count} Passed, {failed_count} Failed")
    print("=" * 65)

    if failed_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)