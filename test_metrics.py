import tensorflow as tf
import time
import joblib
import numpy as np
from utils.preprocess import load_nasa_data, handle_missing_data, add_rul, create_sequences, create_tabular_features
from sklearn.metrics import mean_squared_error
import math
from sklearn.linear_model import LinearRegression


# Load models
lstm_model = tf.keras.models.load_model('models/hybrid_model.keras')  # Same model with LSTM + Tabular
xgb_model = joblib.load('models/xgboost_model.pkl')


datasets = [f"data/train_FD00{i}.txt" for i in range(1, 5)]
results = []

for path in datasets:
    df = load_nasa_data([path])
    df = handle_missing_data(df)
    df = add_rul(df)

    X_seq, y_true = create_sequences(df, window_size=30)
    print("Target RUL stats in test:", y_true.min(), y_true.max(), y_true.mean())
    print("Sample RUL targets:", y_true[:10])
    X_tab = create_tabular_features(df, window_size=30)

    # Align all arrays by trimming to the shortest length
    min_len = min(len(X_seq), len(X_tab), len(y_true))
    X_seq = X_seq[:min_len]
    X_tab = X_tab.iloc[:min_len]
    y_true = y_true[:min_len]

    # Ensure types
    X_tab = X_tab.astype('float32')
    y_true = y_true.astype('float32')

    # --- Inference Timing ---

    # LSTM
    start = time.time()
    y_lstm_raw = lstm_model.predict([X_seq, X_tab.values], verbose=0).squeeze()
    lstm_time = (time.time() - start) * 1000  # ms

    # Calibrate LSTM predictions to match RUL scale
    calib_model = LinearRegression()
    n_calib = min(500, len(y_true))  # avoid overfitting if large
    calib_model.fit(y_lstm_raw[:n_calib].reshape(-1, 1), y_true[:n_calib])
    y_lstm = calib_model.predict(y_lstm_raw.reshape(-1, 1)).squeeze()

    # XGBoost
    start = time.time()
    y_xgb = xgb_model.predict(X_tab)
    xgb_time = (time.time() - start) * 1000  # ms

    # Hybrid (includes both + fusion)
    start = time.time()
    # Use the same calibrated y_lstm from above
    y_xgb_h = xgb_model.predict(X_tab)
    y_hybrid = (y_lstm + y_xgb_h) / 2
    hybrid_time = (time.time() - start) * 1000

    # --- Metrics ---
    rmse_lstm = math.sqrt(mean_squared_error(y_true, y_lstm))
    rmse_xgb = math.sqrt(mean_squared_error(y_true, y_xgb))
    rmse_hybrid = math.sqrt(mean_squared_error(y_true, y_hybrid))

    print("Sample LSTM predictions:", y_lstm[:5])
    print("Sample XGBoost predictions:", y_xgb[:5])
    print("Sample Hybrid predictions:", y_hybrid[:5])

    results.append((
        path.split('_')[-1].split('.')[0],
        rmse_lstm, lstm_time,
        rmse_xgb, xgb_time,
        rmse_hybrid, hybrid_time
    ))

print("Dataset\tLSTM RMSE\tLSTM Time (ms)\tXGB RMSE\tXGB Time (ms)\tHybrid RMSE\tHybrid Time (ms)")
for row in results:
    print(f"{row[0]}\t{row[1]:.2f}\t\t{row[2]:.1f}\t\t\t{row[3]:.2f}\t\t{row[4]:.1f}\t\t\t{row[5]:.2f}\t\t{row[6]:.1f}")

print("RUL VAULES")
print(f"{path}: RUL min = {df['RUL'].min()}, max = {df['RUL'].max()}, mean = {df['RUL'].mean()}")
