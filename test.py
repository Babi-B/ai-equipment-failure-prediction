import tensorflow as tf

# Simple test model
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='mse')
print(f"Model test successful! Here is x: {x}")

from keras.models import Model