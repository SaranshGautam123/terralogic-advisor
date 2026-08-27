from advisory import *

N = 90
P = 42
K = 43
temperature = 20.8
humidity = 82
ph = 6.5
rainfall = 202

score = calculate_soil_score(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
)

quality = soil_quality(score)

status = nutrient_status(N, P, K)

ph_status = ph_analysis(ph)

advice = fertilizer_recommendation(N, P, K, ph)

print("Score:", score)
print("Quality:", quality)
print("Nutrient Status:", status)
print("pH:", ph_status)

print("\nSuggestions")

for item in advice:
    print("-", item)

print("\nExplanation")
print(generate_explanation("Rice", quality, ph_status))