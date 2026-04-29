import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay
)

from data_preprocessing import load_data

# -----------------------------
# Load saved artifacts
# -----------------------------
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/features.pkl")

# -----------------------------
# Load validation data
# -----------------------------
df = load_data("data/validation_set.csv")

# Clean columns
df = df.copy()
df.columns = df.columns.str.replace(".", "_", regex=False)

# Drop unwanted columns
drop_cols = ["FBID", "SampleID", "Fibrosis"]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Convert target
df["group"] = df["group"].apply(lambda x: 0 if x == "S0_2" else 1)

# -----------------------------
# Feature alignment (CRITICAL)
# -----------------------------
X = df[feature_names]
y = df["group"]

# Scale + keep DataFrame
X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
auc = roc_auc_score(y, y_prob)

print("\n📊 Model Performance:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

# -----------------------------
# FIB-4 baseline
# -----------------------------
fib4 = (df["Age"] * df["AST"]) / (df["PLT"] * (df["ALT"] ** 0.5 + 1e-6))

# Normalize (optional but better for ROC)
from sklearn.preprocessing import MinMaxScaler
fib4_scaled = MinMaxScaler().fit_transform(fib4.values.reshape(-1,1))

from sklearn.metrics import roc_auc_score
fib4_auc = roc_auc_score(y, fib4_scaled)

print(f"FIB-4 ROC-AUC: {fib4_auc:.4f}")

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# ROC Curve
# -----------------------------
RocCurveDisplay.from_predictions(y, y_prob)
plt.title("ROC Curve")
plt.show()