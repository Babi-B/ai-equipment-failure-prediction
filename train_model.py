import tensorflow as tf
import joblib
from utils.preprocess import load_data, calculate_rul, normalize_sensors
import numpy as np

# Define model architecture
def build_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(32)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs)


# Training pipeline
def train():
    # Load and prep data
    df = load_data("data/train_FD001.txt")
    df = calculate_rul(df)
    df, scaler = normalize_sensors(df)

    # Prepare sequences
    sequence_length = 30
    sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
    X, y = [], []

    for engine_id in df['engine_id'].unique():
        engine_data = df[df['engine_id'] == engine_id]
        for i in range(len(engine_data) - sequence_length):
            X.append(engine_data.iloc[i:i + sequence_length][sensor_cols].values)
            y.append(engine_data.iloc[i + sequence_length]['RUL'])

    # Train model
    model = build_model((sequence_length, len(sensor_cols)))
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.array(X), np.array(y), epochs=10, batch_size=32)

    # Save artifacts
    model.save("models/lstm_model.keras")
    joblib.dump(scaler, "models/scaler.pkl")


if __name__ == "__main__":
    train()