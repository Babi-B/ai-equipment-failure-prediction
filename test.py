import numpy as np
import time
import psutil
import os
from sklearn.model_selection import train_test_split
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def generate_test_data():
    """Create synthetic data for testing"""
    np.random.seed(42)
    X_seq = np.random.rand(100, 30, 14).astype('float32')  # Reduced size for testing
    X_tab = np.random.rand(100, 8).astype('float32')
    y = np.random.rand(100).astype('float32')
    return X_seq, X_tab, y


def main():
    # Initialize TF after all imports
    import tensorflow as tf
    from xgboost import XGBRegressor

    # Disable problematic behaviors
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Generate data
    X_seq, X_tab, y = generate_test_data()
    X_train_seq, X_val_seq = train_test_split(X_seq, test_size=0.2)
    X_train_tab, X_val_tab = train_test_split(X_tab, test_size=0.2)
    y_train, y_val = train_test_split(y, test_size=0.2)

    # Simple model definitions
    lstm = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(30, 14)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    lstm.compile(optimizer='adam', loss='mse')

    xgb = XGBRegressor()
    hybrid = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(1)
    ])
    hybrid.compile(optimizer='adam', loss='mse')

    # Training measurement
    def train_model(model, X, name):
        start = time.time()
        model.fit(X, y_train, verbose=0)
        return (time.time() - start) / 3600  # hours

    # Inference measurement
    def test_model(model, X, name):
        times = []
        for _ in range(10):
            start = time.perf_counter()
            if name == "XGBoost":
                model.predict(X[:1])  # ❌ no verbose
            else:
                model.predict(X[:1], verbose=0)
            times.append((time.perf_counter() - start) * 1000)
        return np.mean(times), np.std(times)

    # Memory measurement
    def get_memory():
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)  # GB

    # Run evaluations
    results = {}
    for name, model, data in [
        ('LSTM', lstm, X_train_seq),
        ('XGBoost', xgb, X_train_tab),
        ('Hybrid', hybrid, X_train_tab)
    ]:
        try:
            train_time = train_model(model, data, name)
            inf_time, inf_std = test_model(model, data, name)
            memory = get_memory()
            results[name] = (inf_time, inf_std, memory)
        except Exception as e:
            print(f"Error with {name}: {str(e)}")

    # # Print results
    # print("\nPerformance Metrics:")
    # print(f"{'Model':<10} {'Train (h)':<10} {'Inference (ms)':<15} {'Memory (GB)':<10}")
    # for name, (train, inf, std, mem) in results.items():
    #     print(f"{name:<10} {train:<10.3f} {inf:.2f} ± {std:.2f}  {mem:<10.2f}")

    # Terminal display header
    print("\n=== Computational Performance Summary ===\n")
    df = pd.DataFrame(results).T.reset_index()
    df.columns = ["Model", "Inference Time (ms)", "Std Dev (ms)", "Memory Usage (GB)"]
    print(df.to_string(index=False, float_format="%.2f"))

    print("\n*Inference time averaged over 10 runs on single samples.")
    print("*Standard deviation indicates consistency. Lower is better.")


if __name__ == "__main__":
    main()
    input("Press Enter to exit...")

