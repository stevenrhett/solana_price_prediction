import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler

# Fetch live Solana data from CoinGecko API
def fetch_live_data():
    url = "https://api.coingecko.com/api/v3/coins/solana/market_chart"
    params = {"vs_currency": "usd", "days": "365", "interval": "daily"}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()["prices"]  # Extract only price data
        df = pd.DataFrame(data, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")  # Convert timestamp
        return df
    else:
        print("⚠️ Failed to fetch live data! Using cached data.")
        return pd.read_csv("data/sol_price.csv")  # Fallback to cached CSV

# Load Data with Scaling
def load_data():
    df = fetch_live_data()

    # Scale prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["close"] = scaler.fit_transform(df[["close"]])

    return df, df[["close"]].values, scaler