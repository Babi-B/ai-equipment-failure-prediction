import os
import tensorflow as tf
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from utils.preprocess import load_nasa_data, handle_missing_data, add_rul, create_sequences, create_tabular_features
import joblib


# Load and preprocess data
train_files = [f'data/train_FD00{i}.txt' for i in range(1, 5)]
train_df = load_nasa_data(train_files)
train_df = handle_missing_data(train_df)
train_df = add_rul(train_df)

# Data validation checks
print("\n=== Data Validation ===")
print(f"Null values in engine_id: {train_df['engine_id'].isnull().sum()}")
print(f"Duplicate engine-cycle pairs: {train_df.duplicated(['engine_id', 'cycle']).sum()}")

print("\n=== Data Sanity Check ===")
print(f"Total engines: {train_df['engine_id'].nunique()}")
print(f"Columns present: {train_df.columns.tolist()}")
print(f"Example sensor_7 values:\n{train_df['sensor_7'].head()}")
print(f"Fault modes: {train_df['fault_mode'].unique()}")

# Verify metadata exists
assert {'dataset_id', 'operating_condition', 'fault_mode'}.issubset(train_df.columns), \
    "Metadata missing in raw data!"

# Normalize sensors
sensor_cols = [col for col in train_df.columns if 'sensor' in col]
train_df[sensor_cols] = train_df[sensor_cols].astype('float32')

print("\nSensor stats:")
print(train_df[sensor_cols].describe().loc[['mean', 'std', 'min', 'max']].T)

# Normalize per operating condition
scaler_dict = {}
for condition in train_df['operating_condition'].unique():
    condition_mask = train_df['operating_condition'] == condition
    scaler = MinMaxScaler()
    train_df.loc[condition_mask, sensor_cols] = scaler.fit_transform(train_df.loc[condition_mask, sensor_cols])
    scaler_dict[condition] = scaler

# Create features - ensure we only use numeric columns
feature_columns = train_df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns.tolist()
feature_columns = [col for col in feature_columns if col not in ['engine_id', 'dataset_id', 'RUL']]

X_seq, y = create_sequences(train_df[train_df.columns.intersection(feature_columns + ['engine_id', 'RUL'])], window_size=30)
X_tab = create_tabular_features(train_df, window_size=30)

print("\n=== TABULAR FEATURES SAMPLE ===")
print(X_tab.head(3))
print("Columns:", X_tab.columns.tolist())
print("Data types:", X_tab.dtypes)

# Validate features
assert 'fault_mode' in X_tab.columns, f"fault_mode missing! Columns: {X_tab.columns.tolist()}"

# Encode fault_mode
le = LabelEncoder()
X_tab['fault_mode'] = le.fit_transform(X_tab['fault_mode'])

# Ensure all data is numeric and convert to float32
X_tab = X_tab.astype('float32')
y = y.astype('float32')

# Verify X_seq doesn't contain strings
print("\n=== Sequence Data Validation ===")
print("X_seq shape:", X_seq.shape)
print("X_seq sample:", X_seq[0][0])  # Print first sample of first sequence

# Temporal-aware train-test split
engine_groups = train_df.groupby('engine_id').ngroup().values[:len(X_seq)]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X_seq, groups=engine_groups))

# Hybrid Model Architecture
lstm_input = tf.keras.Input(shape=(X_seq.shape[1], X_seq.shape[2]), name='lstm_input')
x = tf.keras.layers.LSTM(128, return_sequences=True)(lstm_input)
x = tf.keras.layers.Attention()([x, x])
x = tf.keras.layers.GlobalAveragePooling1D()(x)

tabular_input = tf.keras.Input(shape=(X_tab.shape[1],), name='tabular_input')
y_tab = tf.keras.layers.Dense(64, activation='relu')(tabular_input)

combined = tf.keras.layers.Concatenate()([x, y_tab])
output = tf.keras.layers.Dense(1, activation='linear')(combined)

model = tf.keras.Model(inputs=[lstm_input, tabular_input], outputs=output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mae',
    metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
)

# Final data validation before training
print("\n=== Final Data Validation ===")
print(f"X_seq shape: {X_seq.shape}, dtype: {X_seq.dtype}")
print(f"X_tab shape: {X_tab.shape}, dtypes:\n{X_tab.dtypes}")
print(f"y shape: {y.shape}, dtype: {y.dtype}")

# Train model
print("\nStarting model training...")
history = model.fit(
    [X_seq[train_idx], X_tab.iloc[train_idx].values],
    y[train_idx],
    validation_data=(
        [X_seq[val_idx], X_tab.iloc[val_idx].values],
        y[val_idx]
    ),
    epochs=50,
    batch_size=256,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
)

# Train XGBoost model
print("\nTraining XGBoost model...")
xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.01,
    subsample=0.8,
    tree_method='hist'
)
xgb_model.fit(X_tab.iloc[train_idx], y[train_idx])

# Save artifacts
os.makedirs('models', exist_ok=True)
model.save('models/hybrid_model.keras')
joblib.dump(scaler_dict, "models/scaler_dict.pkl")
joblib.dump(xgb_model, 'models/xgboost_model.pkl')
joblib.dump(le, 'models/label_encoder.pkl')

print("\nTraining completed successfully! Models saved in 'models' directory.")
