import pandas as pd
import os

def process_currency_data():
    usd_try = pd.read_csv("data/USDTRY=X.csv", skiprows=2, names=["Date", "USDTRY"], parse_dates=["Date"])
    usd_jpy = pd.read_csv("data/USDJPY=X.csv", skiprows=2, names=["Date", "USDJPY"], parse_dates=["Date"])
    eur_usd = pd.read_csv("data/EURUSD=X.csv", skiprows=2, names=["Date", "EURUSD"], parse_dates=["Date"])

    data = usd_try.merge(usd_jpy, on="Date", how="inner").merge(eur_usd, on="Date", how="inner")

    data["TRYJPY"] = data["USDJPY"] / data["USDTRY"]
    data["EURTRY"] = data["EURUSD"] * data["USDTRY"]

    os.makedirs("processed_data", exist_ok=True)
    data.to_csv("processed_data/exchange_rates.csv", index=False)

    print("✅ Veri işleme tamamlandı, TRY/JPY ve EUR/TRY hesaplandı!")

if __name__ == "__main__":
    process_currency_data()
