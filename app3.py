# app.py
import streamlit as st
import pandas as pd
import joblib

# Load trained model and columns
model = joblib.load("model2.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🌦 Tanzania Drought Prediction App")

# User Inputs
year = st.number_input("Year", 1995, 2030, value=2025)
rainfall = st.number_input("Rainfall (mm)", value=100)
temperature = st.number_input("Temperature (°C)", value=25)
region = st.selectbox("Region", ["Dar es Salaam","Dodoma","Mbeya","Mwanza","Kinondoni","Makongo","Makumbusho","Bunju","Kawe"])
season = st.selectbox("Season", ["DJF","MAM","JJA","SON"])

if st.button("Predict"):

    # Create input dataframe
    input_dict = {
        "Year": year,
        "Rainfall_mm": rainfall,
        "Temperature_C": temperature,
        "Region": region,
        "Season": season
    }
    input_df = pd.DataFrame([input_dict])

    # One-hot encode categorical variables
    input_encoded = pd.get_dummies(input_df)

    # Add missing columns and reorder
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_columns]

    # Make prediction
    prediction = model.predict(input_encoded)

    # Show result
    if prediction[0] == 1:
        st.error("⚠ Drought Likely")
    else:
        st.success("✅ No Drought")
