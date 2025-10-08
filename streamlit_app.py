# streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# =========================
# Load Trained Best Model
# =========================
model = joblib.load("gradient_boosting_model.pkl")  # <-- your trained model

st.title("🚕 Taxi Fare Prediction App")
st.write("Enter trip details to predict the estimated fare amount.")

# =========================
# Input Features
# =========================
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)
pickup_longitude = st.number_input("Pickup Longitude", value=-73.985428)
pickup_latitude = st.number_input("Pickup Latitude", value=40.748817)
dropoff_longitude = st.number_input("Dropoff Longitude", value=-73.985428)
dropoff_latitude = st.number_input("Dropoff Latitude", value=40.748817)
payment_type = st.selectbox("Payment Type", [1, 2, 3, 4, 5, 6])

# ✅ Date & time input
pickup_date = st.date_input("Pickup Date")
pickup_time = st.time_input("Pickup Time")#, value=datetime.now().time())
pickup_datetime = datetime.combine(pickup_date, pickup_time)

# =========================
# Feature Engineering
# =========================
pickup_day = pickup_datetime.weekday()
is_weekend = 1 if pickup_day >= 5 else 0
hour = pickup_datetime.hour
am_pm = 1 if hour >= 12 else 0
is_night = 1 if hour >= 20 or hour <= 5 else 0
pickup_hour = hour

# Distance (rough calculation in miles)
trip_distance = np.sqrt((dropoff_longitude - pickup_longitude) ** 2 +
                        (dropoff_latitude - pickup_latitude) ** 2) * 69

# Dummy placeholders (you can improve later)
trip_duration = np.random.randint(5, 30)  # in minutes
fare_per_mile = 0
fare_per_minute = 0

# =========================
# Prediction
# =========================
if st.button("Predict Fare"):
    # Build dataframe with ALL possible features
    input_data = pd.DataFrame([[
        passenger_count, pickup_longitude, pickup_latitude,
        dropoff_longitude, dropoff_latitude, payment_type,
        trip_distance, pickup_day, is_weekend, am_pm, hour,
        is_night, pickup_hour, trip_duration, fare_per_mile, fare_per_minute
    ]], columns=[
        'passenger_count', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude', 'payment_type',
        'trip_distance', 'pickup_day', 'is_weekend', 'am_pm', 'hour',
        'is_night', 'pickup_hour', 'trip_duration', 'fare_per_mile', 'fare_per_minute'
    ])

    # Align input with model’s training features
    expected_cols = model.feature_names_in_
    input_data = input_data.reindex(columns=expected_cols, fill_value=0)

    # Debug info (shows in Streamlit)
    # st.write("✅ Model expects:", list(expected_cols))
    # st.write("✅ Input provided:", list(input_data.columns))

    # Predict
    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Fare Amount: ${prediction[0]:.2f}")
