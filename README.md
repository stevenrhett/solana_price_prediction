# 📈 Solana Price Prediction App

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/Machine%20Learning-TensorFlow-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-active-success)

> Predict future prices of **Solana (SOL)** using an LSTM neural network trained on historical data.

## 🚀 Features

- 🔮 Real-time price prediction using LSTM model
- 📉 24h and 7-day price analytics
- 📊 Interactive visualizations of historical data
- 🔄 Auto-updates price data every 12 hours
- 🧪 Clean UI powered by **Streamlit**
- 📡 Pulls live data from [CoinGecko](https://www.coingecko.com/) & [Binance](https://www.binance.com/)

## 🧠 Model Overview

- Deep learning model: **LSTM (Long Short-Term Memory)**
- Trained on Solana’s historical price data
- Preprocessed and scaled using `MinMaxScaler`
- Model saved as `models/solana_lstm_model.keras`

### 🛠️ Local Setup

#### 1. **Clone the repository**
```sh
   git clone https://github.com/yourusername/solana_price_prediction.git
   cd solana_price_prediction
 ```

#### 2.	Install dependencies
```sh
  pip install -r requirements.txt
```

#### 3.	Run the app
```sh
    streamlit run app.py
```

#### 4.	Open in browser
```
    http://localhost:8501
```


### ☁️ Deploy to Streamlit Cloud
1.	Push your code to a public GitHub repository. 
2. Go to Streamlit Cloud and connect your GitHub repo. 
3. Set your app’s entry point to app.py. 
4. Add environment variables or secrets as needed in Settings → Secrets.

## 🐳 Optional: Docker Support

### Build Docker image
```sh
  docker build -t solana-price-predictor .
```

## Run Docker container
```sh
  docker run -p 8501:8501 solana-price-predictor
````

### 📁 Project Structure

```
├── app.py                  # Streamlit app logic
├── models/
│   └── solana_lstm_model.keras
├── src/
│   └── data_loader.py      # Loads and preprocesses data
├── requirements.txt
└── README.md
```


## ⚠️ Disclaimer
This app is for educational and informational purposes only.
The price predictions are based on statistical modeling and do not constitute financial advice. Always do your own research before making investment decisions.

## 🙌 Credits & Tech Stack

Built with:
- 🧠 TensorFlow
- 📊 Streamlit
- 🐍 Python
- 📈 Matplotlib
- 🌐 CoinGecko API
- 💹 Binance API

## 📝 License
This project is licensed under the [MIT License](LICENSE).


## 📬 Contact
Feel free to open an issue, fork this repo, or drop me a message!
