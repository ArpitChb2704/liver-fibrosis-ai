# AI-Powered Liver Fibrosis Risk Prediction System with Explainable ML &amp; MLOps

An end-to-end machine learning system for predicting liver fibrosis risk using clinical biomarkers.  
This project integrates model comparison, explainable AI (SHAP), and an interactive Streamlit dashboard for real-time prediction and analysis.

---

## 🚀 Features

- 🔬 Multi-model comparison (Logistic Regression, Random Forest, XGBoost, LightGBM)
- 📊 Model evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
- 🧠 SHAP explainability (global + local interpretation)
- 🎯 Threshold tuning for medical decision-making
- 📈 Confusion Matrix & ROC Curve visualization
- 🧾 Clinical interpretation layer
- 📊 Interactive Streamlit dashboard
- ⚙️ Feature engineering (AST/ALT, AST/PLT ratios)

---

## 🏥 Problem Statement

Liver fibrosis is a progressive condition that can lead to cirrhosis if not detected early.  
This project aims to assist in **early risk assessment** using non-invasive clinical data.

---

## 🧠 Machine Learning Pipeline

1. Data preprocessing & cleaning  
2. Feature engineering  
3. Model training & comparison  
4. Best model selection (based on ROC-AUC)  
5. Model evaluation on validation dataset  
6. Deployment via Streamlit  

---

## 📊 Models Used

- Logistic Regression (baseline)
- Random Forest
- XGBoost
- LightGBM ✅ (Best performing)

---

## 🔍 Explainability (SHAP)

- Feature importance analysis  
- Individual prediction explanations  
- Waterfall plots for interpretability  

---

## 🎯 Decision Threshold Tuning

Custom threshold allows:
- Higher recall → safer medical predictions  
- Reduced false negatives  

---

## 📦 Project Structure

```bash
livefbr-ml/
│
├── data/
│   ├── discovery_set.csv
│   └── validation_set.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── evaluate.py
|   ├── results.txt
│   └── explain.py
|   ├── feature_selection.py
│
├── app.py
│  
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── features.pkl
│
└── requirements.txt
```


---

## 📈 Sample Output

- Risk prediction (Low / Moderate / High)
- Probability score
- SHAP explanation
- Confusion matrix & ROC curve

---

## 💡 Key Highlights

- Built an **end-to-end ML system**, not just a model  
- Focus on **clinical interpretability**  
- Integrated **real-world decision threshold tuning**  
- Designed **interactive UI for usability**  

---

## ⚠️ Disclaimer

This project is for educational purposes only and should not be used as a substitute for professional medical advice.

---

<img width="640" height="480" alt="Figure_6" src="https://github.com/user-attachments/assets/03589fe3-f10d-4c2c-bcf3-ebdbd459d2db" />
<img width="640" height="480" alt="Figure_7" src="https://github.com/user-attachments/assets/8e330c4a-e4b6-4a26-878b-0440edc759bc" />
<img width="2878" height="1520" alt="image" src="https://github.com/user-attachments/assets/f7388e20-5cd3-4662-a4d6-342cdd757df8" />
<img width="2166" height="1132" alt="image" src="https://github.com/user-attachments/assets/46e5b0b7-d664-4a00-b22e-b0c0052ede1b" />
<img width="2122" height="1638" alt="image" src="https://github.com/user-attachments/assets/47f2b4ac-a13d-4c3e-9d98-0a1ed5d1060d" />
<img width="2154" height="1278" alt="image" src="https://github.com/user-attachments/assets/da3991ed-4315-4132-9566-31c9c7b343db" />






