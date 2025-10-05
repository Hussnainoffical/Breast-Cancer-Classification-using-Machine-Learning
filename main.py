# cancer_app.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------------
# Load & Train Model
# ------------------------
@st.cache_data
def train_model():
    data = pd.read_csv("data_cancer.csv")
    data = data.drop(["id", "Unnamed: 32"], axis=1, errors="ignore")

    X = data.drop("diagnosis", axis=1)
    y = data["diagnosis"].map({"M": 1, "B": 0})  # Encode: Malignant=1, Benign=0

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate model
    acc = accuracy_score(y_test, pipeline.predict(X_test))

    return pipeline, list(X.columns), acc

pipeline, feature_names, acc = train_model()

# ------------------------
# Streamlit UI
# ------------------------
st.title("🩺 Breast Cancer Classification App")
st.write("Enter patient tumor measurements to predict if the tumor is **Benign (0)** or **Malignant (1)**.")

st.sidebar.header("📊 Model Info")
st.sidebar.write(f"Model: Random Forest Classifier")
st.sidebar.write(f"Accuracy on test set: **{acc:.2%}**")

# ------------------------
# User Input Form
# ------------------------
st.subheader("Enter Tumor Features")

user_input = {}
for feature in feature_names[:10]:  # show only first 10 for simplicity
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Add missing columns with NaN
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = np.nan

# Reorder
input_df = input_df[feature_names]

# ------------------------
# Prediction
# ------------------------
if st.button("🔮 Predict Diagnosis"):
    prediction = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ The model predicts: **Malignant** (Probability: {prob:.2%})")
    else:
        st.success(f"✅ The model predicts: **Benign** (Probability: {prob:.2%})")
