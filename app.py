import streamlit as st
import joblib
import pandas as pd

# Load files
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.title("🧬 Cancer Prediction App")

# Inputs
age = st.number_input("Age", min_value=1, max_value=120)
weight = st.number_input("Weight", min_value=1)
height = st.number_input("Height", min_value=1)

gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "Age": age,
        "Weight": weight,
        "Height": height,
        "Gender": gender
    }])

    input_data = pd.get_dummies(input_data)

    input_data = input_data.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk")
    else:
        st.success("✅ Low Risk")