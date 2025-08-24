
import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer

# Load the trained model and the scaler
try:
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('model.joblib')
except FileNotFoundError:
    st.error("Model or scaler not found. Please run train_model.py first.")
    st.stop()

# Load feature names for labels
cancer_data = load_breast_cancer()
feature_names = cancer_data.feature_names
target_names = cancer_data.target_names


# --- Streamlit App Interface ---
st.title("🩺 Breast Cancer Prediction App")

# --- Sidebar for User Input ---
st.sidebar.header("Tumor Features")
input_data = {}
for feature_name in feature_names:
    input_data[feature_name] = st.sidebar.number_input(
        f"Enter {feature_name}",
        value=float(cancer_data.data.mean(axis=0)[list(feature_names).index(feature_name)]),
        key=feature_name
    )

if st.sidebar.button("Predict"):
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    prediction_label = target_names[prediction[0]]
    st.subheader("Prediction Result")
    if prediction_label == 'malignant':
        st.error(f"The model predicts the tumor is: **Malignant**")
    else:
        st.success(f"The model predicts the tumor is: **Benign**")
