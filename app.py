import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow debug logs

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import matplotlib.dates as mdates
from src.data_loader import load_data

# Load trained model
model = tf.keras.models.load_model("models/solana_lstm_model.keras")

# 🌟 Streamlit App Title with Emoji
st.set_page_config(page_title="Solana Price Prediction", page_icon="📈")
st.title("📈 Solana Price Prediction App")
st.write("💡 Predict future Solana prices using LSTM.")

# 🎨 Sidebar: Model Information
st.sidebar.header("🔧 About This Model")
st.sidebar.info(
    "This app predicts potential future prices of Solana (SOL) using a Long Short-Term Memory (LSTM) neural network trained on historical market data. "
    "Price data is sourced from CoinGecko and updated automatically every **12 hours**. "
    "Predictions are based solely on past trends and statistical modeling and should not be considered financial advice."
)

# Load Data
df, prices_scaled, scaler = load_data()

# Ensure 'timestamp' is properly converted
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # Convert from string/UNIX timestamp if needed
else:
    st.error("🚨 ERROR: 'timestamp' column missing from dataset!")

# Debugging: Print last 5 rows
print("🛠 DEBUG: Last 5 rows of the dataset:")
print(df.tail())

# Show raw data with option to expand
with st.expander("📋 View Raw Data", expanded=False):
    st.dataframe(df.tail(10))

# Select last `time_step` data points as input
time_step = 50
last_data = prices_scaled[-time_step:].reshape(1, time_step, 1)

# Predict next price
predicted_price = model.predict(last_data)
predicted_price = scaler.inverse_transform(predicted_price)[0][0]

# 🎨 Market Overview
st.markdown("## 📉 Market Overview")
col1, col2 = st.columns(2)

with col1:
    # Inverse transform last known price
    last_known_price = scaler.inverse_transform([[df['close'].iloc[-1]]])[0][0]
    st.metric(label="📊 **Last Known Price**", value=f"${last_known_price:.2f}", delta_color="normal")

with col2:
    # Calculate price difference & percentage change
    predicted_change = predicted_price - last_known_price
    predicted_change_percent = (predicted_change / last_known_price) * 100

    st.metric(
        label="🔮 **Predicted Next Price**",
        value=f"${predicted_price:.2f}",
        delta=f"{predicted_change:.2f} ({predicted_change_percent:.2f}%)",
    )

# 📉 24h Price Change
if len(df) > 1:
    last_price_actual = scaler.inverse_transform([[df['close'].iloc[-1]]])[0][0]
    prev_price_actual = scaler.inverse_transform([[df['close'].iloc[-2]]])[0][0]  # 24h ago

    price_change_24h = last_price_actual - prev_price_actual
    percent_change_24h = (price_change_24h / prev_price_actual) * 100

    st.metric(
        label="📉 **24h Price Change**",
        value=f"${price_change_24h:.2f}",
        delta=f"{price_change_24h:.2f} ({percent_change_24h:.2f}%)",
    )

# ✅ 7-Day High & Low
last_7_days_scaled = df[['close']].tail(7).values  # Ensure it's a 2D array
last_7_days_actual = scaler.inverse_transform(last_7_days_scaled)

high_7d = last_7_days_actual.max()
low_7d = last_7_days_actual.min()

col3, col4 = st.columns(2)
with col3:
    st.metric(label="📈 **7-Day High**", value=f"${high_7d:.2f}")
with col4:
    st.metric(label="📉 **7-Day Low**", value=f"${low_7d:.2f}")

# 📊 Historical Solana Prices Chart

# Set timestamp as index (AFTER ensuring it's in datetime format)
df.set_index("timestamp", inplace=True)

# Plot price trends with formatted x-axis labels
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df["close"], label="Solana Price")

# Format x-axis to show both months and year
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)  # Rotate dates for better visibility

plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("📊 Solana Price Trends")
plt.legend()

st.pyplot(fig)

# 📅 Last Update Timestamp
last_update = df.index[-1]  # Using index since it's already set
st.text(f"📅 Data last updated on: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")

# 📡 Data Source Information
st.markdown("### 📡 Data Source")
st.write("🔗 Live Solana price data fetched from [CoinGecko API](https://www.coingecko.com/) and [Binance API](https://www.binance.com/).")

# 📌 Footer
st.markdown(
    """
    ---
    💡 *Built with Streamlit, TensorFlow, and 💻 Python*
    """, unsafe_allow_html=True
)