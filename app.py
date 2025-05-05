import streamlit as st
import tensorflow as tf
import joblib
import plotly.express as px
from utils.preprocess import load_nasa_data, handle_missing_data, create_sequences, create_tabular_features
import os

# Set page config must be FIRST command
st.set_page_config(page_title="AI Equipment Health Monitor", layout="wide")


# Load models - must come after set_page_config
@st.cache_resource
def load_models():
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
    return (
        tf.keras.models.load_model(os.path.join(MODEL_DIR, 'hybrid_model.keras')),
        joblib.load(os.path.join(MODEL_DIR, "scaler_dict.pkl")),
        joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl')),
        joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    )


st.title("🛠️ AI-Powered Equipment Failure Prediction System")
st.markdown("""
    **Real-time predictive maintenance** using hybrid deep learning (LSTM + XGBoost) 
    - Upload sensor data to predict remaining useful life (RUL)
    - Compare model predictions
    - Visualize degradation trends
""")

with st.expander("📤 Data Upload"):
    uploaded_file = st.file_uploader("Upload engine sensor data (TXT)", type=["txt"])
    demo_option = st.checkbox("Use demo data (FD001)")

# Now load models after the initial UI elements
hybrid_model, scaler_dict, xgb_model, le = load_models()

if demo_option and not uploaded_file:
    demo_path = os.path.join('data', 'train_FD001.txt')
    if os.path.exists(demo_path):
        uploaded_file = demo_path
    else:
        st.warning("Demo data file not found at expected path")

if uploaded_file:
    try:
        with st.spinner("Processing data..."):
            df = load_nasa_data([uploaded_file], is_test=True)
            raw_df = df.copy()
            df = handle_missing_data(df)

            # Normalization
            sensor_cols = [col for col in df.columns if 'sensor' in col]
            for condition in df['operating_condition'].unique():
                condition_mask = df['operating_condition'] == condition
                df.loc[condition_mask, sensor_cols] = scaler_dict[condition].transform(
                    df.loc[condition_mask, sensor_cols])

            # Create features
            sequences = create_sequences(df, is_test=True)
            tab_features = create_tabular_features(df)

            if 'fault_mode' in tab_features.columns:
                tab_features['fault_mode'] = le.transform(tab_features['fault_mode'].astype(str))

            tab_features = tab_features.astype('float32')

        if len(sequences) > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🔮 Model Predictions")
                hybrid_rul = hybrid_model.predict([sequences[-1:], tab_features.iloc[-1:].values])[0][0]
                xgb_rul = xgb_model.predict(tab_features.iloc[-1:].values.reshape(1, -1))[0]

                st.metric("Hybrid AI Prediction",
                          f"{hybrid_rul:.1f} cycles",
                          delta=f"{(hybrid_rul - xgb_rul):.1f} vs XGBoost")

                st.metric("XGBoost Prediction",
                          f"{xgb_rul:.1f} cycles",
                          delta=f"{(xgb_rul - hybrid_rul):.1f} vs Hybrid")

                if hybrid_rul < 30 or xgb_rul < 30:
                    st.error("🚨 Critical Failure Imminent! Schedule maintenance immediately!")
                elif hybrid_rul < 100:
                    st.warning("⚠️ Moderate Degradation Detected! Monitor closely.")
                else:
                    st.success("✅ Equipment Healthy - No action required")

            with col2:
                st.subheader("📉 Degradation Timeline")
                fig = px.line(raw_df, x='cycle', y='sensor_7',
                              title='Sensor 7 Trend Analysis',
                              labels={'cycle': 'Operational Cycles', 'sensor_7': 'Vibration (Normalized)'})
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("❌ Insufficient data for prediction - Need at least 30 operational cycles")

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.stop()
