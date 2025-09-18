# app.py
# Hospital Readmission Prediction Data Product - Streamlit Web App
# Author: Karuna Kathet

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

# ---------------------------
# Load and preprocess dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("hospital_readmissions.csv")
    df = df.dropna()

    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df

df = load_data()

# ---------------------------
# Train Models
# ---------------------------
X = df.drop("readmitted", axis=1)
y = df["readmitted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# ---------------------------
# Evaluation Function
# ---------------------------
def evaluate_model(name, y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return acc, prec, rec, f1

lr_results = evaluate_model("Logistic Regression", y_test, lr_preds)
rf_results = evaluate_model("Random Forest", y_test, rf_preds)

# ---------------------------
# Streamlit Interface
# ---------------------------
st.title("🏥 Hospital Readmission Prediction")
st.write("This web app predicts hospital readmissions using Logistic Regression and Random Forest.")

# Dataset preview
if st.checkbox("Show dataset preview"):
    st.write(df.head())

# Model Results
st.subheader("📊 Model Performance")
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [lr_results[0], rf_results[0]],
    "Precision": [lr_results[1], rf_results[1]],
    "Recall": [lr_results[2], rf_results[2]],
    "F1 Score": [lr_results[3], rf_results[3]]
})
st.dataframe(results)

# Confusion Matrix - Random Forest
st.subheader("📌 Confusion Matrix (Random Forest)")
cm = confusion_matrix(y_test, rf_preds)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Random Forest - Confusion Matrix")
st.pyplot(fig)

# ROC Curve - Random Forest
st.subheader("📈 ROC Curve (Random Forest)")
y_prob = rf_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0,1], [0,1], "r--")
ax.set_title("ROC Curve - Random Forest")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
st.pyplot(fig)

st.header("Reports (Placeholder)")

# Placeholder message
st.info("This section will generate reports in the final version. For now, here are placeholders.")

# PDF placeholder button
if st.button("Generate PDF Report"):
    st.warning("PDF report generation coming soon!")

# CSV placeholder (with real results if available, else sample)
try:
    df_for_report = results.copy()   # use your real results DataFrame
except NameError:
    # if results is not defined yet, create a sample
    df_for_report = pd.DataFrame({
        "Model": ["Random Forest"],
        "Accuracy": [0.87],
        "Precision": [0.80],
        "Recall": [0.85],
        "F1 Score": [0.82]
    })

csv_bytes = df_for_report.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Report (CSV)",
    data=csv_bytes,
    file_name="report.csv",
    mime="text/csv",
    help="Placeholder download with sample metrics."
)


# ---------------------------
# Interactive Prediction
# ---------------------------
st.subheader("🔮 Predict Readmission for a New Patient")

sample = X.iloc[0].tolist()  # default
user_input = []
for i, col in enumerate(X.columns):
    val = st.number_input(f"{col}", value=float(sample[i]))
    user_input.append(val)

if st.button("Predict"):
    data = np.array(user_input).reshape(1, -1)
    data = scaler.transform(data)
    prediction = rf_model.predict(data)
    result = "Readmitted" if prediction[0] == 1 else "Not Readmitted"
    st.success(f"Prediction: **{result}**")

# ---------------------------
# Help Section
# ---------------------------
st.sidebar.title("ℹ️ Help")
st.sidebar.write("""
- **Dataset:** Hospital readmissions data  
- **Models:** Logistic Regression & Random Forest  
- **Features:** Preprocessing, training, evaluation, prediction  
- **Outputs:** Confusion Matrix, ROC Curve, metrics table  
- Use the input fields to simulate a new patient record.  
""")

