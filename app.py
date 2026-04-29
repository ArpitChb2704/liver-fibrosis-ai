import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, RocCurveDisplay
import plotly.graph_objects as go

# -----------------------------
# Load artifacts
# -----------------------------
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/features.pkl")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Liver Fibrosis AI", layout="wide")

st.title("🧠 Liver Fibrosis Prediction System")
st.markdown("AI-powered prediction with explainability & medical threshold tuning")

# -----------------------------
# Sidebar (threshold tuning)
# -----------------------------
st.sidebar.header("⚙️ Model Settings")


threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

st.sidebar.info("Lower threshold → higher recall (safer for medical use)")

# -----------------------------
# Input fields
# -----------------------------
st.header("📥 Patient Inputs")

inputs = {}

cols = st.columns(3)

exclude_features = ["AST_ALT", "AST_PLT"]

defaults = {
    "Age": 40,
    "ALB": 4.2,
    "ALT": 30,
    "AST": 25,
    "BMI": 25,
    "DM_IFG": 0,
    "FBG": 5.5,
    "GGT": 40,
    "HbA1c": 5.5,
    "HDL": 1.2,
    "LDL": 2.5,
    "PLT": 200,
    "Sex": 1,
    "TBIL": 12,
    "TC": 4.5,
    "TG": 1.5,
    "AST_ALT": 25/30,
    "AST_PLT": 25/200,
    "FIB4": 1.2,
    "NFS": -1.5
}

for i, feature in enumerate(feature_names):
    if feature in exclude_features:
        continue
    with cols[i % 3]:
        inputs[feature] = st.number_input(feature, value=defaults.get(feature, 0.0))

# Scale
input_df = pd.DataFrame([inputs])


# Ensure correct column order
input_df["AST_ALT"] = input_df["AST"] / (input_df["ALT"] + 1e-6)
input_df["AST_PLT"] = input_df["AST"] / (input_df["PLT"] + 1e-6)
input_df = input_df[feature_names]
# Scale
input_scaled = scaler.transform(input_df)
input_scaled = pd.DataFrame(input_scaled, columns=feature_names)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict"):

    prob = model.predict_proba(input_scaled)[0][1]
    pred = 1 if prob >= threshold else 0

    st.subheader("📊 Prediction Result")

    # -----------------------------
    # Gauge Chart (custom)
    # -----------------------------
    st.markdown("### 🎯 Risk Score")

    fig, ax = plt.subplots(width_ratios=[1], figsize=(6, 1.5))

    ax.barh(["Risk"], [prob])
    ax.set_xlim(0, 1)
    ax.set_title("Fibrosis Risk Probability")

    st.pyplot(fig, use_container_width=True)

    # Text result
    if prob < 0.3:
        st.success(f"Low Risk ({prob:.4f}). Biomarkers are largely within healthy range.")
    elif prob < 0.6:
        st.warning(f"Moderate Risk ({prob:.4f}). Further evaluation recommended.")
    else:
        st.error(f"High Risk ({prob:.4f}). High risk of advanced fibrosis. Clinical attention required.")

    # -----------------------------
    # SHAP Explainability
    # -----------------------------
    st.subheader("🔍 SHAP Explanation")

    explainer = shap.Explainer(model)
    shap_values = explainer(input_scaled)

    fig2, ax2 = plt.subplots(width_ratios=[1], figsize=(6, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig2, use_container_width=True)


# -----------------------------
    #Gauge Chart (Plotly)
    # -----------------------------
    st.subheader("🎯 Fibrosis Risk Gauge")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,  # convert to %
        title={'text': "Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},

            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"},
            ],

            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100,
            }
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Baseline Clinical Score (FIB-4)
# -----------------------------
    fib4 = (inputs["Age"] * inputs["AST"]) / (
    inputs["PLT"] * (inputs["ALT"] ** 0.5 + 1e-6)
    )

    st.subheader("🧪 Clinical Score (FIB-4)")
    st.write(f"FIB-4 Score: {fib4:.2f}")

# -----------------------------
# Model Evaluation Section
# ----------------------------
st.header("📈 Model Evaluation")

if st.checkbox("Show Evaluation Metrics"):

    # Load validation data
    df = pd.read_csv("data/validation_set.csv")
    df.columns = df.columns.str.replace(".", "_", regex=False)

    drop_cols = ["FBID", "SampleID", "Fibrosis"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df["group"] = df["group"].apply(lambda x: 0 if x == "S0_2" else 1)

    X = df[feature_names]
    y = df["group"]

    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    st.subheader("🧾 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["Low", "High"],
                yticklabels=["Low", "High"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig3)

    # -----------------------------
    # ROC Curve
    # -----------------------------
    st.subheader("📉 ROC Curve")

    fig4, ax4 = plt.subplots()
    RocCurveDisplay.from_predictions(y, y_prob, ax=ax4)
    st.pyplot(fig4)


    # -----------------------------
    # Metrics
    # -----------------------------
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    st.subheader("📊 Metrics")

    st.write(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    st.write(f"Precision: {precision_score(y, y_pred):.4f}")
    st.write(f"Recall: {recall_score(y, y_pred):.4f}")
    st.write(f"F1 Score: {f1_score(y, y_pred):.4f}")
    st.write(f"ROC-AUC: {roc_auc_score(y, y_prob):.4f}")