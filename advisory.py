# advisory.py
# -------------------------------------------------------------
# TerraLogic Advisor — Rule-Based Soil Health & Advisory Engine
# -------------------------------------------------------------


# -----------------------------
# Input Validation
# -----------------------------

def validate_soil_inputs(N, P, K, temperature, humidity, ph, rainfall):
    """
    Validates numeric soil and environmental inputs against physically plausible bounds.
    Raises TypeError if non-numeric and ValueError if out of bounds.
    """
    inputs = {
        "Nitrogen (N)": N,
        "Phosphorus (P)": P,
        "Potassium (K)": K,
        "Temperature": temperature,
        "Humidity": humidity,
        "pH": ph,
        "Rainfall": rainfall,
    }

    for name, val in inputs.items():
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise TypeError(f"{name} must be a numeric value, got {type(val).__name__}: {val}")

    if N < 0 or P < 0 or K < 0:
        raise ValueError("Nutrient values (N, P, K) cannot be negative.")

    if not (0 <= ph <= 14):
        raise ValueError(f"Soil pH must be between 0 and 14, got {ph}.")

    if not (0 <= humidity <= 100):
        raise ValueError(f"Relative humidity must be between 0% and 100%, got {humidity}.")

    if rainfall < 0:
        raise ValueError(f"Rainfall cannot be negative, got {rainfall}.")

    if not (-50 <= temperature <= 60):
        raise ValueError(f"Temperature {temperature}°C is outside the plausible range (-50°C to 60°C).")


# -----------------------------
# Soil Health Score
# -----------------------------

def calculate_soil_score(N, P, K, temperature, humidity, ph, rainfall):
    score = 0

    # Nitrogen (max 20)
    if 40 <= N <= 100:
        score += 20
    elif 20 <= N < 40 or 100 < N <= 120:
        score += 10

    # Phosphorus (max 20)
    if 20 <= P <= 80:
        score += 20
    elif 10 <= P < 20 or 80 < P <= 100:
        score += 10

    # Potassium (max 20)
    if 20 <= K <= 80:
        score += 20
    elif 10 <= K < 20 or 80 < K <= 100:
        score += 10

    # pH (max 20)
    if 6.0 <= ph <= 7.5:
        score += 20
    elif 5.5 <= ph < 6.0 or 7.5 < ph <= 8.0:
        score += 10

    # Temperature (max 10)
    if 20 <= temperature <= 35:
        score += 10

    # Humidity (max 5)
    if 50 <= humidity <= 90:
        score += 5

    # Rainfall (max 5)
    if 50 <= rainfall <= 250:
        score += 5

    return score


# -----------------------------
# Soil Quality
# -----------------------------

def soil_quality(score):

    if score >= 90:
        return "Excellent"

    elif score >= 75:
        return "Good"

    elif score >= 50:
        return "Average"

    else:
        return "Poor"


# -----------------------------
# Nutrient Status
# -----------------------------

def nutrient_status(N, P, K):

    status = {}

    # Nitrogen
    if N < 40:
        status["Nitrogen"] = "Low"
    elif N <= 100:
        status["Nitrogen"] = "Optimal"
    else:
        status["Nitrogen"] = "High"

    # Phosphorus
    if P < 20:
        status["Phosphorus"] = "Low"
    elif P <= 80:
        status["Phosphorus"] = "Optimal"
    else:
        status["Phosphorus"] = "High"

    # Potassium
    if K < 20:
        status["Potassium"] = "Low"
    elif K <= 80:
        status["Potassium"] = "Optimal"
    else:
        status["Potassium"] = "High"

    return status


# -----------------------------
# pH Analysis
# -----------------------------

def ph_analysis(ph):

    if ph < 5.5:
        return "Strongly Acidic"

    elif ph < 6.5:
        return "Slightly Acidic"

    elif ph <= 7.5:
        return "Neutral (Ideal)"

    else:
        return "Alkaline"


# -----------------------------
# Fertilizer Suggestions
# -----------------------------

def fertilizer_recommendation(N, P, K, ph):

    advice = []

    if N < 40:
        advice.append(
            "Increase nitrogen using compost or a nitrogen-rich fertilizer."
        )

    if P < 20:
        advice.append(
            "Add a phosphorus-rich fertilizer to improve root development."
        )

    if K < 20:
        advice.append(
            "Apply a potassium-rich fertilizer to improve crop strength."
        )

    if ph < 5.5:
        advice.append(
            "Apply agricultural lime to reduce soil acidity."
        )

    elif ph > 7.5:
        advice.append(
            "Increase organic matter and monitor soil alkalinity."
        )

    if len(advice) == 0:
        advice.append(
            "Soil nutrients appear balanced. Maintain current soil management practices."
        )

    return advice


# -----------------------------
# Final Explanation
# -----------------------------

def generate_explanation(crop, quality, ph_status):
    if crop:
        return (
            f"The recommended crop is {crop}. "
            f"Your soil quality is rated as {quality}. "
            f"The soil pH is {ph_status}. "
            "Review the nutrient recommendations below to improve long-term soil health."
        )
    return (
        f"Your soil quality is rated as {quality}. "
        f"The soil pH is {ph_status}. "
        "Review the nutrient recommendations below to improve long-term soil health."
    )


# -----------------------------
# Unified Application Interface
# -----------------------------

def get_soil_advisory(N, P, K, temperature, humidity, ph, rainfall, crop=None):
    """
    Application-facing interface for the soil advisory engine.
    Validates input parameters and returns a complete, structured advisory report dictionary.

    Parameters:
        N (float/int): Nitrogen value
        P (float/int): Phosphorus value
        K (float/int): Potassium value
        temperature (float/int): Temperature in °C
        humidity (float/int): Relative humidity in %
        ph (float/int): Soil pH (0 to 14)
        rainfall (float/int): Rainfall depth in mm
        crop (str, optional): Name of recommended crop

    Returns:
        dict: Structured advisory report containing score, quality, nutrient status,
              pH status, fertilizer recommendations, and summary explanation.
    """
    validate_soil_inputs(N, P, K, temperature, humidity, ph, rainfall)

    score = calculate_soil_score(N, P, K, temperature, humidity, ph, rainfall)
    quality = soil_quality(score)
    nutrients = nutrient_status(N, P, K)
    ph_stat = ph_analysis(ph)
    advice = fertilizer_recommendation(N, P, K, ph)
    explanation = generate_explanation(crop, quality, ph_stat)

    return {
        "soil_score": score,
        "soil_quality": quality,
        "nutrient_status": nutrients,
        "ph_status": ph_stat,
        "fertilizer_recommendations": advice,
        "explanation": explanation,
    }