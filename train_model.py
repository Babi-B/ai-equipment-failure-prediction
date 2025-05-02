import os
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from utils.preprocess import load_nasa_data, add_rul, create_sequences, create_tabular_features
import joblib

# Load and preprocess data
train_df = load_nasa_data('data/train_FD001.txt')
train_df = add_rul(train_df)

# Normalize sensor data
scaler = MinMaxScaler()
sensor_cols = [col for col in train_df.columns if 'sensor' in col]
train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])

# Create sequences for LSTM
window_size = 30
X_sequences, y = create_sequences(train_df, window_size=window_size)

# Create tabular features FOR EACH SEQUENCE WINDOW (not per engine)
tabular_features = create_tabular_features(train_df, window_size=window_size)

# Split data into train/val using sklearn to preserve alignment
X_train_seq, X_val_seq, X_train_tab, X_val_tab, y_train, y_val = train_test_split(
    X_sequences, tabular_features, y, test_size=0.2, random_state=42
)

# Build LSTM-only baseline
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mae')
lstm_model.fit(X_train_seq, y_train, epochs=50, validation_data=(X_val_seq, y_val))

# Build hybrid model
lstm_input = tf.keras.layers.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
tabular_input = tf.keras.layers.Input(shape=(X_train_tab.shape[1],))

# LSTM branch
x = tf.keras.layers.LSTM(64)(lstm_input)

# Tabular branch (processed features)
y = tf.keras.layers.Dense(32)(tabular_input)

# Combine branches
combined = tf.keras.layers.Concatenate()([x, y])
output = tf.keras.layers.Dense(1)(combined)

hybrid_model = tf.keras.models.Model(inputs=[lstm_input, tabular_input], outputs=output)
hybrid_model.compile(optimizer='adam', loss='mae')

# Train hybrid model
hybrid_model.fit(
    [X_train_seq, X_train_tab],
    y_train,
    epochs=50,
    validation_data=([X_val_seq, X_val_tab], y_val)
)

# Build LSTM-only baseline
lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
# lstm_model.compile(optimizer='adam', loss='mae')
# Build and compile the model
lstm_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanAbsoluteError(),  # Explicit class or 'mean_absolute_error'
    metrics=['mae']  # Metrics can still use shorthand
)
lstm_model.fit(X_train_seq, y_train, epochs=50, validation_data=(X_val_seq, y_val))

# ADD THIS LINE TO CREATE THE DIRECTORY
os.makedirs('models', exist_ok=True)

# Save model
lstm_model.save('models/lstm_baseline.h5')  # <-- Now the directory exists!
joblib.dump(scaler, "models/scaler.pkl")
