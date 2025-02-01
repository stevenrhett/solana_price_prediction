import os
from celery import Celery
import pandas as pd
from src.data_loader import fetch_live_data

app = Celery("tasks", broker="redis://localhost:6379/0")


@app.task
def update_data():
    """Fetches live Solana price data, removes duplicate timestamps, and updates CSV."""

    new_data = fetch_live_data()
    new_data["timestamp"] = pd.to_datetime(new_data["timestamp"])  # Ensure datetime format

    # Load existing data if file exists
    if os.path.exists("data/sol_price.csv"):
        df = pd.read_csv("data/sol_price.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])  # Convert to datetime
    else:
        df = pd.DataFrame(columns=["timestamp", "close"])  # Create empty DataFrame

    # 🔥 **Fix: Keep only one entry per day (latest available price)**
    df["date"] = df["timestamp"].dt.date  # Extract just the date (YYYY-MM-DD)
    new_data["date"] = new_data["timestamp"].dt.date  # Extract date from new data

    # Remove any existing entry for the same day in new_data
    df = df[~df["date"].isin(new_data["date"])]

    # Append new data and **keep only the last entry per day**
    df = pd.concat([df, new_data], ignore_index=True).drop_duplicates(subset="date", keep="last")

    # Drop the 'date' column (since we only needed it for filtering)
    df.drop(columns=["date"], inplace=True)

    # Save cleaned data
    df.to_csv("data/sol_price.csv", index=False)

    print("✅ Data updated successfully! 🚀")