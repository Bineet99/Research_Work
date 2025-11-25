import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception as e:
    ARIMA = None
    print("Warning: statsmodels not available. ARIMA will not run.", e)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
except Exception:
    tf = None
    print("Info: TensorFlow not available. LSTM step will be skipped.")

DATA_PATH = None
UPLOAD_PATH = "/mnt/data/Research.pdf"

OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PRICES_CSV = os.path.join(OUT_DIR, "market_prices.csv")
OUT_REPORT_CSV = os.path.join(OUT_DIR, "market_forecast_report.csv")

TEST_DAYS = 90
ARIMA_ORDER = (5, 1, 0)
SEQ_LEN = 30
LSTM_EPOCHS = 15
LSTM_BATCH = 16

def generate_synthetic_prices(days=3*365, start_date="2020-01-01", seed=42):
    np.random.seed(seed)
    start = datetime.fromisoformat(start_date)
    dates = [start + timedelta(days=i) for i in range(days)]
    t = np.arange(days)
    yearly = 5 * np.sin(2 * np.pi * t / 365.0)
    weekly = 1.5 * np.sin(2 * np.pi * t / 7.0)
    trend = 0.005 * t
    base = 30
    noise = np.random.normal(scale=1.2, size=days)
    spikes = np.zeros(days)
    for s in [200, 450, 700, 900]:
        if s < days:
            spikes[s:s+3] += np.linspace(0, 8, min(3, days-s))
    prices = base + yearly + weekly + trend + noise + spikes
    prices = np.round(prices, 2)
    df = pd.DataFrame({"date": dates, "price": prices})
    df.set_index("date", inplace=True)
    return df

def load_prices(path):
    df = pd.read_csv(path)
    if 'date' in df.columns and 'price' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df[['price']].sort_index()
    if 'price' in df.columns:
        try:
            df.index = pd.to_datetime(df.index)
            return df[['price']].sort_index()
        except Exception:
            raise ValueError("CSV not in expected format. Need 'date' and 'price' columns.")
    raise ValueError("CSV must have a 'price' column (and preferably a 'date' column).")

if DATA_PATH:
    prices_df = load_prices(DATA_PATH)
else:
    prices_df = generate_synthetic_prices()

prices_df.to_csv(OUT_PRICES_CSV)
print("Saved price data to:", OUT_PRICES_CSV)
print("Data sample:\n", prices_df.head())

if len(prices_df) <= TEST_DAYS + 30:
    raise ValueError("Not enough data for test split; need more days. Use synthetic or larger CSV.")

train = prices_df[:-TEST_DAYS].copy()
test = prices_df[-TEST_DAYS:].copy()
print(f"Data lengths -> total: {len(prices_df)}, train: {len(train)}, test: {len(test)}")

arima_forecast = None
arima_metrics = None
if ARIMA is not None:
    try:
        arima_model = ARIMA(train['price'], order=ARIMA_ORDER)
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=TEST_DAYS)
        arima_forecast.index = test.index
        rmse = sqrt(mean_squared_error(test['price'], arima_forecast))
        mae = mean_absolute_error(test['price'], arima_forecast)
        arima_metrics = {'rmse': rmse, 'mae': mae, 'order': ARIMA_ORDER}
        print(f"ARIMA{ARIMA_ORDER} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    except Exception as e:
        print("ARIMA training/forecast failed:", e)
else:
    print("Skipping ARIMA: statsmodels not installed.")

if arima_forecast is not None:
    plt.figure(figsize=(10,4))
    plt.plot(train.index[-200:], train['price'][-200:], label="Recent train")
    plt.plot(test.index, test['price'], label="Actual (test)")
    plt.plot(arima_forecast.index, arima_forecast.values, label="ARIMA forecast")
    plt.legend()
    plt.title("ARIMA Forecast vs Actual (Test Period)")
    plt.tight_layout()
    plt.show()

lstm_forecast = None
lstm_metrics = None
if tf is not None:
    try:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(prices_df['price'].values.reshape(-1,1))

        def create_sequences(data, seq_length=SEQ_LEN):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled, seq_length=SEQ_LEN)
        seq_dates = prices_df.index[SEQ_LEN:]

        train_size = len(train) - SEQ_LEN
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:train_size + TEST_DAYS], y[train_size:train_size + TEST_DAYS]

        X_train = X_train.reshape((-1, SEQ_LEN, 1))
        X_test = X_test.reshape((-1, SEQ_LEN, 1))

        model = Sequential()
        model.add(LSTM(64, input_shape=(SEQ_LEN, 1)))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        print("Training LSTM (this can take time)...")
        model.fit(X_train, y_train, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, validation_data=(X_test, y_test), verbose=1)

        y_pred_scaled = model.predict(X_test).flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
        y_true = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

        rmse_l = sqrt(mean_squared_error(y_true, y_pred))
        mae_l = mean_absolute_error(y_true, y_pred)
        lstm_metrics = {'rmse': rmse_l, 'mae': mae_l, 'epochs': LSTM_EPOCHS}

        lstm_forecast = pd.Series(y_pred, index=test.index)
        print(f"LSTM -> RMSE: {rmse_l:.4f}, MAE: {mae_l:.4f}")

        plt.figure(figsize=(10,4))
        plt.plot(test.index, test['price'], label="Actual (test)")
        plt.plot(lstm_forecast.index, lstm_forecast.values, label="LSTM forecast")
        plt.legend()
        plt.title("LSTM Forecast vs Actual (Test Period)")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("LSTM modelling failed:", e)
else:
    print("Skipping LSTM: TensorFlow not installed.")

report = pd.DataFrame(index=test.index)
report['actual'] = test['price']
if arima_forecast is not None:
    report['arima_forecast'] = arima_forecast
if lstm_forecast is not None:
    report['lstm_forecast'] = lstm_forecast

report.to_csv(OUT_REPORT_CSV)
print("Saved forecast report to:", OUT_REPORT_CSV)
print("Report sample:\n", report.head())

print("\nSummary metrics:")
if arima_metrics:
    print("ARIMA:", arima_metrics)
if lstm_metrics:
    print("LSTM:", lstm_metrics)
print("\nIf you want to use a real CSV, set DATA_PATH variable at top to its path.")
print("Uploaded file path available in this session (not used by default):", UPLOAD_PATH)
