import shap
import joblib
import pandas as pd

model = joblib.load("models/best_model.pkl")

def explain_model(X_sample):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)

    shap.summary_plot(shap_values, X_sample)