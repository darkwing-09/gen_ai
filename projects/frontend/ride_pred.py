import streamlit as st
import pandas as pd
# from models import bike_demand_pipeline
import joblib

# Load the saved pipeline once
pipeline = joblib.load('/home/darkwing/Desktop/ML/projects/models/bike_demand_pipeline.pkl')

st.title("Bike Rental Demand Predictor")
st.write("Enter the conditions below to predict how many bikes will be rented this hour.")

# --- Collect real-world inputs ---
season = st.selectbox("Season", [1, 2, 3, 4], format_func=lambda x: {1:"Winter", 2:"Spring", 3:"Summer", 4:"Fall"}[x])
yr = st.selectbox("Year", [0, 1], format_func=lambda x: {0:"2011", 1:"2012"}[x])
mnth = st.slider("Month", 1, 12, 6)
hr = st.slider("Hour of day", 0, 23, 8)
holiday = st.selectbox("Is it a holiday?", [0, 1], format_func=lambda x: {0:"No", 1:"Yes"}[x])
weekday = st.slider("Weekday (0=Sunday)", 0, 6, 3)
workingday = st.selectbox("Is it a working day?", [0, 1], format_func=lambda x: {0:"No", 1:"Yes"}[x])
weathersit = st.selectbox("Weather situation", [1, 2, 3, 4], 
                           format_func=lambda x: {1:"Clear", 2:"Mist/Cloudy", 3:"Light Rain/Snow", 4:"Heavy Rain/Storm"}[x])
temp = st.slider("Temperature (normalized 0-1)", 0.0, 1.0, 0.5)
hum = st.slider("Humidity (normalized 0-1)", 0.0, 1.0, 0.5)
windspeed = st.slider("Windspeed (normalized 0-1)", 0.0, 1.0, 0.2)

# --- Build input row matching training columns ---
input_df = pd.DataFrame([{
    'season': season, 'yr': yr, 'mnth': mnth, 'hr': hr, 'holiday': holiday,
    'weekday': weekday, 'workingday': workingday, 'weathersit': weathersit,
    'temp': temp, 'hum': hum, 'windspeed': windspeed
}])

if st.button("Predict Ride Count"):
    prediction = pipeline.predict(input_df)
    st.success(f"Predicted bike rentals this hour: **{round(prediction[0])}**")