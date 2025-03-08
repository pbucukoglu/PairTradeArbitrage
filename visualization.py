import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import joblib

def plot_results():
    data = pd.read_csv("processed_data/exchange_rates.csv", parse_dates=["Date"])

    model = tf.keras.models.load_model("processed_data/lstm_model.keras")
    scaler = joblib.load("processed_data/scaler.pkl")

    data_scaled = scaler.transform(data[['TRYJPY', 'EURTRY']])
    X_test = []
    for i in range(60, len(data_scaled)):
        X_test.append(data_scaled[i-60:i, :])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Split into train and test sets
    train_size = int(0.8 * len(predicted_prices))
    train_actual = data.iloc[60:60+train_size]
    test_actual = data.iloc[60+train_size:]

    train_pred = predicted_prices[:train_size]
    test_pred = predicted_prices[train_size:]

    plt.figure(figsize=(12, 6))
    plt.plot(train_actual["Date"], train_actual["TRYJPY"], label="Actual TRY/JPY", color="blue")
    plt.plot(train_actual["Date"], train_pred[:, 0], label="Predicted TRY/JPY", color="red")
    plt.plot(train_actual["Date"], train_actual["EURTRY"], label="Actual EUR/TRY", color="green")
    plt.plot(train_actual["Date"], train_pred[:, 1], label="Predicted EUR/TRY", color="orange")
    plt.legend()
    plt.title("Positions on Train Set")
    plt.xlabel("Date")
    plt.ylabel("Exchange Rate")
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(test_actual["Date"], test_actual["TRYJPY"], label="Actual TRY/JPY", color="blue")
    plt.plot(test_actual["Date"], test_pred[:, 0], label="Predicted TRY/JPY", color="red")
    plt.plot(test_actual["Date"], test_actual["EURTRY"], label="Actual EUR/TRY", color="green")
    plt.plot(test_actual["Date"], test_pred[:, 1], label="Predicted EUR/TRY", color="orange")
    plt.legend()
    plt.title("Positions on Test Set")
    plt.xlabel("Date")
    plt.ylabel("Exchange Rate")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    plot_results()
