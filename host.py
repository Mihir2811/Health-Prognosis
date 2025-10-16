import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- Load Model & Scaler ----------------
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("Models/best_model.pkl")
    scaler = joblib.load("Models/scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# ---------------- Streamlit UI ----------------
st.title("Heart Disease Prediction System")
st.write("This app predicts the likelihood of heart disease based on patient data.")

st.subheader("Enter Patient Information")

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.selectbox("Constrictive Pericarditis Type (0–4)", [0, 1, 2, 3, 4])
    bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=130)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=400, value=240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    
with col2:
    restecg = st.selectbox("Resting Electrocardiographic Result (0–2)", [0, 1, 2])
    maxhr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=6.0, value=1.0)
    slope = st.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

# ---------------- Prediction ----------------
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, bp, chol, fbs, restecg, maxhr, exang, oldpeak, slope, ca, thal]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("The model predicts that the patient **has heart disease**.")
    else:
        st.success("The model predicts that the patient **does not have heart disease**.")

st.caption("Model trained using multiple ML algorithms — automatically selected the best performer.")
