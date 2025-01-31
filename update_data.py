from celery import Celery
from src.data_loader import fetch_live_data

app = Celery("tasks", broker="redis://localhost:6379/0")

@app.task
def update_data():
    df = fetch_live_data()
    df.to_csv("data/sol_price.csv", index=False)
    print("✅ Data updated successfully!")