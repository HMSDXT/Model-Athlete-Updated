import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os


# ============================================================
# Load model and data
# ============================================================

model = joblib.load(
    "models/sports_performance_model.pkl"
)

df = pd.read_csv(
    "data/athelete_data.csv"
)


FEATURES = [
    "Motivation",
    "Anxiety",
    "Confidence",
    "Stress",
    "Resilience",
    "Mental_Toughness"
]

X = df[FEATURES]


# ============================================================
# SHAP
# ============================================================

explainer = shap.Explainer(
    model.predict,
    X
)

shap_values = explainer(X)


# ============================================================
# Summary plot
# ============================================================

shap.summary_plot(
    shap_values,
    X,
    show=False
)

os.makedirs("models", exist_ok=True)
plt.savefig("models/shap_summary.png", bbox_inches="tight")
plt.close()

print("SHAP summary saved to models/shap_summary.png")