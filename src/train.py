import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from data_preprocessing import load_data, preprocess_data

df = load_data("data/discovery_set.csv")
X, y, scaler = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "LightGBM": LGBMClassifier()
}

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    print(f"{name} AUC: {auc:.4f}")

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)

        results.append((name, auc))

    if auc > best_score:
        best_score = auc
        best_model = model
        best_model_name = name
        best_model_score = auc

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, "models/features.pkl")

# Save model + scaler
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Best model saved!")
print(f"Best Model: {best_model_name} with AUC: {best_model_score:.4f}")
print("\n📊 Model Performance: On validations set")
print(f"Accuracy : {best_model_score:.4f}")
print(results)