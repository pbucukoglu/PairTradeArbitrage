from data_fetcher import fetch_currency_data
from data_processing import process_currency_data
from machine_learning import train_lstm_model
from visualization import plot_results

if __name__ == "__main__":
    fetch_currency_data(["USDTRY=X", "USDJPY=X", "EURUSD=X"])
    process_currency_data()
    train_lstm_model()
    plot_results()
