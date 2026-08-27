# advisory.py

# -----------------------------
# Soil Health Score
# -----------------------------

def calculate_soil_score(N, P, K, temperature, humidity, ph, rainfall):
    score = 0

    # Nitrogen
    if 40 <= N <= 100:
        score += 20
    elif 20 <= N < 40 or 100 < N <= 120:
        score += 10

    # Phosphorus
    if 20 <= P <= 80:
        score += 20
    elif 10 <= P < 20 or 80 < P <= 100:
        score += 10

    # Potassium
    if 20 <= K <= 80:
        score += 20
    elif 10 <= K < 20 or 80 < K <= 100:
        score += 10

    # pH
    if 6.0 <= ph <= 7.5:
        score += 20
    elif 5.5 <= ph < 6.0 or 7.5 < ph <= 8.0:
        score += 10

    # Temperature
    if 20 <= temperature <= 35:
        score += 10

    # Humidity
    if 50 <= humidity <= 90:
        score += 5

    # Rainfall
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

    return (
        f"The recommended crop is {crop}. "
        f"Your soil quality is rated as {quality}. "
        f"The soil pH is {ph_status}. "
        "Review the nutrient recommendations below to improve long-term soil health."
    )