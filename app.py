import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress info/debug logs

import streamlit as st
import tensorflow as tf
from src.data_loader import load_data

# Load trained model
model = tf.keras.models.load_model("models/solana_lstm_model.keras")

# 🌟 Streamlit App Title with Emoji
st.set_page_config(page_title="Solana Price Prediction", page_icon="📈")
st.title("📈 Solana Price Prediction App")
st.write("💡 Predict future Solana prices using LSTM.")

# 🎨 Add Sidebar
st.sidebar.header("🔧 Settings")
st.sidebar.info("This app predicts the future price of Solana (SOL) using an LSTM model trained on historical data.")

# Load Data
df, prices_scaled, scaler = load_data()

df, prices_scaled, scaler = load_data()
print("🛠 DEBUG: Last 5 rows of the dataset:")
print(df.tail())  # Debugging: Show last 5 rows

# Show raw data with option to expand
with st.expander("📋 View Raw Data", expanded=False):
    st.dataframe(df.tail(10))

# Select last `time_step` data points as input
time_step = 50
last_data = prices_scaled[-time_step:].reshape(1, time_step, 1)

# Predict next price
predicted_price = model.predict(last_data)
predicted_price = scaler.inverse_transform(predicted_price)[0][0]

# 🎨 Display Results with Colors
st.markdown("## 📉 Market Overview")
col1, col2 = st.columns(2)

with col1:
    # Inverse transform the last known price
    last_known_price = scaler.inverse_transform([[df['close'].iloc[-1]]])[0][0]

    # Display the corrected last known price
    st.metric(label="📊 **Last Known Price**", value=f"${last_known_price:.2f}", delta_color="normal")

with col2:
    st.metric(label="🔮 **Predicted Next Price**", value=f"${predicted_price:.2f}", delta=predicted_price - last_known_price)

# 📉 24h Price Change
df['price_change_24h'] = df['close'].diff()
last_change = scaler.inverse_transform([[df['price_change_24h'].iloc[-1]]])[0][0]
st.metric(label="📉 **24h Price Change**", value=f"${last_change:.2f}", delta=last_change)

# ✅ Fix 7-Day High & Low
last_7_days_scaled = df['close'].tail(7).values.reshape(-1, 1)
last_7_days_actual = scaler.inverse_transform(last_7_days_scaled)

high_7d = last_7_days_actual.max()
low_7d = last_7_days_actual.min()

col3, col4 = st.columns(2)
with col3:
    st.metric(label="📈 **7-Day High**", value=f"${high_7d:.2f}")
with col4:
    st.metric(label="📉 **7-Day Low**", value=f"${low_7d:.2f}")

# 📊 Historical Solana Prices Chart
st.subheader("📊 Solana Price Trends")
st.line_chart(df.set_index("timestamp")["close"])

# 📅 Last Update Timestamp
last_update = df['timestamp'].iloc[-1]
st.text(f"📅 Data last updated on: {last_update}")

# 📡 Data Source Information
st.markdown("### 📡 Data Source")
st.write("🔗 Live Solana price data fetched from [CoinGecko API](https://www.coingecko.com/) and [Binance API](https://www.binance.com/).")

# 📌 Add a Footer
st.markdown(
    """
    ---
    💡 *Built with Streamlit, TensorFlow, and 💻 Python.*
    """, unsafe_allow_html=True
)