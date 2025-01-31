import os
import tensorflow as tf
import absl.logging

# Suppress TensorFlow and Keras warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel('ERROR')
absl.logging.set_verbosity(absl.logging.ERROR)

# Rest of the imports
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from tensorflow.keras.layers import Input

from data_loader import load_data

# Load data
df, prices_scaled, scaler = load_data()

# Function to create time-series sequences
def create_sequences(data, time_step=50):
    X, y = [], []
    if len(data) <= time_step:  # Prevents empty sequence creation
        print(f"⚠️ Not enough data! Need at least {time_step+1} data points, but got {len(data)}")
        return np.array([]), np.array([])

    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])

    return np.array(X), np.array(y)

# Prepare training data
time_step = 50
X, y = create_sequences(prices_scaled, time_step)

print("prices_scaled shape:", prices_scaled.shape)
print("prices_scaled sample:", prices_scaled[:5])  # First 5 values

# Reshape input for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build LSTM model
model = Sequential([
    Input(shape=(time_step, 1)),  # Explicitly define input layer
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Save the trained model
model.save("models/solana_lstm_model.keras")
print("Model saved successfully!")