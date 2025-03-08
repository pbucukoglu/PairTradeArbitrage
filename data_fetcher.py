import yfinance as yf
import pandas as pd
import os

def fetch_currency_data(currency_pairs, start_date="2023-03-08", end_date="2024-03-08"):
    os.makedirs("data", exist_ok=True)

    for pair in currency_pairs:
        print(f"📥 {pair} verisi indiriliyor...")
        data = yf.download(pair, start=start_date, end=end_date)
        data[['Close']].to_csv(f"data/{pair}.csv")
        print(f"✅ {pair} verisi kaydedildi!")

if __name__ == "__main__":
    fetch_currency_data(["USDTRY=X", "USDJPY=X", "EURUSD=X"])
