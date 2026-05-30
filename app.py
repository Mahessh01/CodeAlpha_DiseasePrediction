import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Breast Cancer Prediction App")
st.write("Enter patient medical details below:")

col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input("Radius Mean", min_value=0.0)
    texture_mean = st.number_input("Texture Mean", min_value=0.0)
    perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0)
    area_mean = st.number_input("Area Mean", min_value=0.0)
    compactness_mean = st.number_input("Compactness Mean", min_value=0.0)

with col2:
    concavity_mean = st.number_input("Concavity Mean", min_value=0.0)
    concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0)
    radius_worst = st.number_input("Radius Worst", min_value=0.0)
    perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0)
    area_worst = st.number_input("Area Worst", min_value=0.0)


if st.button("Predict Diagnosis"):

    input_data = np.array([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        radius_worst,
        perimeter_worst,
        area_worst
    ]])

    # scale input
    input_scaled = scaler.transform(input_data)

    # prediction
    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)

    st.markdown("---")

    if prediction[0] == 1:
        st.error("Malignant (Cancer Detected)")
    else:
        st.success("Benign (No Cancer Detected)")

    st.write("### Prediction Confidence")
    st.write(f"Benign: {proba[0][0]*100:.2f}%")
    st.write(f"Malignant: {proba[0][1]*100:.2f}%")