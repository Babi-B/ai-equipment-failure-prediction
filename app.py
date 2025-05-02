import streamlit as st
import tensorflow as tf
import pandas as pd
from utils.preprocess import load_nasa_data, create_sequences
import joblib

# Load pre-trained model
model = tf.keras.models.load_model('models/lstm_baseline.h5')
scaler = joblib.load("models/scaler.pkl")

st.title('AI Equipment Health Monitor')

# File upload; Allow .txt and .csv uploads
uploaded_file = st.file_uploader("Upload Sensor Data (TXT/CSV)", type=["txt", "csv"])

if uploaded_file:
    # Load data (assume test data has no RUL)
    df = load_nasa_data(uploaded_file, is_test=True)

    # Normalize sensor data using the training scaler
    sensor_cols = [col for col in df.columns if "sensor" in col]
    df[sensor_cols] = scaler.transform(df[sensor_cols])

    # Create sequences (no targets for test data)
    sequences = create_sequences(df, window_size=30, is_test=True)

    if len(sequences) > 0:
        # Predict RUL for the latest window
        prediction = model.predict(sequences[-1:])[0][0]
        st.metric("Predicted RUL", f"{prediction:.1f} cycles")

        if prediction < 30:
            st.error("Urgent: Schedule maintenance immediately!")
        elif prediction < 100:
            st.warning("Warning: Monitor closely")
        else:
            st.success("Healthy")
    else:
        st.error("Insufficient data to create sequences")
