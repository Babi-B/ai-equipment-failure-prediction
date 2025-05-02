import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def load_nasa_data(file_path, is_test=False):
    """Load NASA CMAPSS data with proper column names"""
    cols = ["engine_id", "cycle"] + [f"op_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=cols, engine="python")
    df = df.dropna(axis=1, how="all")  # Remove empty columns

    if not is_test:
        df = add_rul(df)  # Add RUL only for training data
    return df


def add_rul(df):
    """" Calculate Remaining Useful Life (RUL) """
    df["RUL"] = df.groupby("engine_id")["cycle"].transform(lambda x: x.max() - x)
    return df


def create_sequences(data, window_size=50, is_test=False):
    """Create time-series sequences for LSTM"""
    sequences = []
    targets = [] if not is_test else None

    # Columns to exclude
    cols_to_drop = ["engine_id"]
    if not is_test:
        cols_to_drop.append("RUL")

    # Define feature columns (exclude 'engine_id' and 'RUL' for training)
    feature_columns = data.columns.drop(cols_to_drop).tolist()

    for engine_id in data["engine_id"].unique():
        engine_data = data[data["engine_id"] == engine_id]
        if len(engine_data) < window_size:
            continue
        for i in range(len(engine_data) - window_size):
            seq = engine_data.iloc[i:i + window_size]
            seq_features = seq[feature_columns].values
            sequences.append(seq_features)

            # Only get target if training data
            if not is_test:
                target = seq["RUL"].values[-1]
                targets.append(target)

    return (np.array(sequences), np.array(targets)) if not is_test else np.array(sequences)

def create_tabular_features(data, window_size=30):
    """Create tabular features FOR EACH SEQUENCE WINDOW (not per engine)"""
    tab_features = []
    for engine_id in data["engine_id"].unique():
        engine_data = data[data["engine_id"] == engine_id]
        if len(engine_data) < window_size:
            continue
        for i in range(len(engine_data) - window_size):
            window = engine_data.iloc[i:i + window_size]
            # Example features: mean, variance of sensors in the window
            features = {
                's7_mean': window['sensor_7'].mean(),
                's12_var': window['sensor_12'].var(),
                'op3_max': window['op_3'].max(),
            }
            tab_features.append(features)
    return pd.DataFrame(tab_features)
