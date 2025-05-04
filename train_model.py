import os
import tensorflow as tf
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from utils.preprocess import load_nasa_data, handle_missing_data, add_rul, create_sequences, create_tabular_features
import joblib

# Load and preprocess data
train_files = [f'data/train_FD00{i}.txt' for i in range(1,5)]
train_df = load_nasa_data(train_files)
train_df = handle_missing_data(train_df)
train_df = add_rul(train_df)

# Advanced conditional normalization
sensor_cols = [col for col in train_df.columns if 'sensor' in col]
scaler_dict = {}
for condition in train_df['operating_condition'].unique():
    condition_mask = train_df['operating_condition'] == condition
    scaler = MinMaxScaler()
    train_df.loc[condition_mask, sensor_cols] = scaler.fit_transform(train_df.loc[condition_mask, sensor_cols])
    scaler_dict[condition] = scaler

# Feature engineering
window_size = 30
X_seq, y = create_sequences(train_df, window_size)
X_tab = create_tabular_features(train_df, window_size)

# Encode categorical features
le = LabelEncoder()
X_tab['fault_mode'] = le.fit_transform(X_tab['fault_mode'])

# Temporal-aware train-test split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X_seq, groups=train_df['engine_id'].iloc[:len(X_seq)]))

# Hybrid Model Architecture (LSTM + XGBoost enhanced)
lstm_input = tf.keras.Input(shape=(window_size, X_seq.shape[2]))
x = tf.keras.layers.LSTM(128, return_sequences=True)(lstm_input)
x = tf.keras.layers.Attention()([x, x])
x = tf.keras.layers.GlobalAvgPool1D()(x)

tabular_input = tf.keras.Input(shape=(X_tab.shape[1],))
y = tf.keras.layers.Dense(64, activation='relu')(tabular_input)

combined = tf.keras.layers.Concatenate()([x, y])
output = tf.keras.layers.Dense(1, activation='relu')(combined)

model = tf.keras.Model(inputs=[lstm_input, tabular_input], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mae',
              metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])

# Train hybrid model
history = model.fit(
    [X_seq[train_idx], X_tab.iloc[train_idx]],
    y[train_idx],
    validation_data=([X_seq[val_idx], X_tab.iloc[val_idx]], y[val_idx]),
    epochs=50,
    batch_size=256,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
)

# Train XGBoost ensemble
xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.9,
    tree_method='hist',
    enable_categorical=True
)
xgb_model.fit(X_tab.iloc[train_idx], y[train_idx])

# Save models
os.makedirs('models', exist_ok=True)
model.save('models/hybrid_model.keras')
joblib.dump(xgb_model, 'models/xgboost_model.pkl')
joblib.dump(scaler_dict, 'models/scaler_dict.pkl')
joblib.dump(le, 'models/label_encoder.pkl')