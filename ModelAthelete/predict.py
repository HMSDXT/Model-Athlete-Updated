import joblib
import pandas as pd


MODEL_PATH = "models/sports_performance_model.pkl"

model = joblib.load(MODEL_PATH)


# ------------------------------------------------------------
# Athlete psychological scores
# ------------------------------------------------------------

athlete = pd.DataFrame([{
    "Motivation": 82,
    "Anxiety": 30,
    "Confidence": 85,
    "Stress": 25,
    "Resilience": 80,
    "Mental_Toughness": 84
}])


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

prediction = model.predict(athlete)[0]

prediction = max(0, min(100, prediction))


# ------------------------------------------------------------
# Performance category
# ------------------------------------------------------------

if prediction < 50:
    category = "Low"

elif prediction < 75:
    category = "Moderate"

else:
    category = "High"


print("\n================================")
print("ATHLETE AI ASSESSMENT")
print("================================")

print(
    f"Predicted Performance: {prediction:.2f}/100"
)

print(
    f"Performance Category: {category}"
)