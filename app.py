import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
@st.cache_resource
def load_model():
    with open("exoplanet_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Load the scaler
@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    return scaler

model = load_model()
scaler = load_scaler()

st.title("🔭 Exoplanet Classification App")
st.write("This app predicts whether a celestial body is an exoplanet based on given features.")

st.sidebar.header("Enter Features")
feature_values = []
for i in range(41):  # Ensure correct number of features
    feature = st.sidebar.number_input(f"Feature {i+1}", min_value=0.0, format="%.4f")
    feature_values.append(feature)

input_data = np.array([feature_values])
scaled_input = scaler.transform(input_data)

if st.sidebar.button("Predict"):
    prediction = model.predict(scaled_input)
    prediction_prob = model.predict_proba(scaled_input)
    
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("🪐 The model predicts this is an **Exoplanet!**")
    else:
        st.error("🚀 The model predicts this is **NOT an Exoplanet.**")
    
    st.subheader("Prediction Confidence")
    st.write(f"Probability of being an Exoplanet: {prediction_prob[0][1]:.2%}")
    st.write(f"Probability of NOT being an Exoplanet: {prediction_prob[0][0]:.2%}")
