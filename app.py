import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import base64

# Load the trained model
@st.cache_resource
def load_model():
    with open("exoplanet_model.pkl", "rb") as file:
        return pickle.load(file)

# Load the scaler
@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()
scaler = load_scaler()

# Function to load local image as base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Convert local image to base64 for background
image_path = "nasa.jpeg"  # Ensure this file exists in the same directory
base64_image = get_base64_image(image_path)

# Apply a space-themed background using base64 image
space_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:nasa/jpeg;base64,{base64_image}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        font-family: 'Arial', sans-serif;
    }}
    .stTitle, .stMarkdown, .stSubheader {{
        color: #FFFFFF;
        text-align: center;
    }}
    .stButton>button {{
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        color: white;
        border-radius: 10px;
        padding: 12px;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        background: linear-gradient(90deg, #2575fc, #6a11cb);
    }}
    .sidebar .sidebar-content {{
        background-color: rgba(0, 0, 0, 0.7);
        padding: 20px;
        border-radius: 10px;
    }}
    .sidebar .stMarkdown, .sidebar .stHeader {{
        color: #FFFFFF;
    }}
    </style>
"""
st.markdown(space_bg, unsafe_allow_html=True)

# App Title
st.title("✨ Exoplanet Classificatio App")
st.write("### Predict whether a celestial body is an exoplanet based on given features.")

# Sidebar for user input
st.sidebar.header("🌌 Enter Features")
feature_names = [
    "Orbital Period", "Planet Radius", "Planet Mass", "Stellar Flux", "Equilibrium Temperature",
    "Insolation Flux", "Orbital Eccentricity", "Orbital Inclination", "Semi-Major Axis", "Stellar Radius",
    "Stellar Mass", "Stellar Age", "Surface Gravity", "Metallicity", "Luminosity",
    "Effective Temperature", "Spectral Type", "Rotation Period", "Axial Tilt", "Mean Anomaly",
    "Periastron Argument", "Longitude of Ascending Node", "Radial Velocity", "Transit Depth",
    "Transit Duration", "Transit Timing Variation", "Impact Parameter", "Eccentricity Variance",
    "Inclination Variance", "Orbital Velocity", "Geometric Albedo", "Phase Curve Amplitude",
    "Secondary Eclipse Depth", "Polarization Fraction", "Planet Density", "Escape Velocity",
    "Bond Albedo", "Cloud Fraction", "Atmospheric Composition", "Tidal Locking", "Magnetic Field Strength"
]

feature_values = [st.sidebar.number_input(feature, min_value=0.0) for feature in feature_names]
input_data = np.array([feature_values])
scaled_input = scaler.transform(input_data)

if st.sidebar.button("🚀 Predict", use_container_width=True):
    prediction = model.predict(scaled_input)
    prediction_prob = model.predict_proba(scaled_input)
    
    st.subheader("🌠 Prediction Result")
    if prediction[0] == 1:
        st.success("🪐 The model predicts this is an **Exoplanet!**")
    else:
        st.error("🚀 The model predicts this is **NOT an Exoplanet.**")
    
    st.subheader("🔭 Prediction Confidence")
    st.write(f"🔹 Probability of being an Exoplanet: **{prediction_prob[0][1]:.2%}**")
    st.write(f"🔹 Probability of NOT being an Exoplanet: **{prediction_prob[0][0]:.2%}**")
