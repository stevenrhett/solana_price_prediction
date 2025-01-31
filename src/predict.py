import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_data

# Load trained model
model = tf.keras.models.load_model("models/solana_lstm_model.keras")

# Load dataset
df, prices_scaled, scaler = load_data()

# Prepare test data
time_step = 50
X_test, y_test = [], []
for i in range(len(prices_scaled) - time_step - 1):
    X_test.append(prices_scaled[i:(i + time_step), 0])
    y_test.append(prices_scaled[i + time_step, 0])

X_test = np.array(X_test)
y_test = np.array(y_test)  # Convert to NumPy array
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Make predictions
predictions = model.predict(X_test).squeeze(-1)  # Ensure correct shape
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

# Plot actual vs predicted prices
plt.figure(figsize=(12,6))
plt.plot(df.index[time_step:time_step+len(y_test)], scaler.inverse_transform(y_test.reshape(-1, 1)), label="Actual Prices")
plt.plot(df.index[time_step:time_step+len(predictions)], predictions, label="Predicted Prices", linestyle='dashed')
plt.legend()
plt.xlabel("Date")
plt.ylabel("Solana Price (USD)")
plt.title("Solana (SOL) Price Prediction using LSTM")
plt.show()