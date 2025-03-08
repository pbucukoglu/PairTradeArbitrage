import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib

def train_lstm_model():
    data = pd.read_csv("processed_data/exchange_rates.csv", index_col="Date", parse_dates=True)
    data.dropna(inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['TRYJPY', 'EURTRY']])

    X, y = [], []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i, :])  # Now using both TRYJPY and EURTRY
        y.append(data_scaled[i, :])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=2)  # Now predicting both TRYJPY and EURTRY
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=20, batch_size=32)

    model.save("processed_data/lstm_model.keras")
    joblib.dump(scaler, "processed_data/scaler.pkl")

    print("✅ LSTM modeli eğitildi ve kaydedildi!")

if __name__ == "__main__":
    train_lstm_model()
